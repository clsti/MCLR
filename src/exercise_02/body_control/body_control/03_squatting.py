import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from body_control.tsid_wrapper import TSIDWrapper
import body_control.config as conf

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=True):
        '''
        Initializes the Talos robot in simulation and sets up ROS2 publishers.

        Parameters:
        - simulator: Simulation interface (PybulletWrapper)
        - urdf: Path to URDF file
        - model: Pinocchio model
        - node: ROS2 node instance
        - q: Initial joint configuration
        - verbose: Print debug info if True
        - useFixedBase: If True, base is fixed in simulation
        '''

        z_init = 1.15

        super().__init__(
            simulator,
            urdf,
            model,
            basePosition=[0, 0, z_init],
            baseQuationerion=[0, 0, 0, 1],
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose)

        self.node = node

        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        self.br = tf2_ros.TransformBroadcaster(self.node)

    def update(self):
        '''
        Updates internal simulation state from the base Robot class.
        '''
        super().update()

    def publish(self, T_b_w, tau):
        '''
        Publishes joint states and base transform to ROS2.

        Parameters:
        - T_b_w: Base-to-world transform (pinocchio SE3)
        - tau: Actuated joint torques (numpy array)
        '''

        now = self.node.get_clock().now().to_msg()

        # Publish joint states
        self.joint_msg.header.stamp = now
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # Broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.baseName()

        tf_msg.transform.translation.x = T_b_w.translation[0]
        tf_msg.transform.translation.y = T_b_w.translation[1]
        tf_msg.transform.translation.z = T_b_w.translation[2]

        q = pin.Quaternion(T_b_w.rotation)
        q.normalize()
        tf_msg.transform.rotation.x = q.x
        tf_msg.transform.rotation.y = q.y
        tf_msg.transform.rotation.z = q.z
        tf_msg.transform.rotation.w = q.w

        self.br.sendTransform(tf_msg)

################################################################################
# Application
################################################################################


class Environment(Node):
    '''
    Main ROS2 node managing simulation and control of the Talos robot.
    It interfaces the TSID controller, PyBullet simulation, and ROS2 communication.
    '''

    def __init__(self):
        '''
        Initializes the simulation environment, robot, and controller.
        '''
        super().__init__('tutorial_4_standing_node')

        self.tsid_wrapper = TSIDWrapper(conf)
        self.simulator = PybulletWrapper(sim_rate=conf.f_cntr)

        # Use q_init for robot initialization, as conf.q_home results in error
        q_init = np.hstack([
            np.array([0, 0, 1.15, 0, 0, 0, 1]),
            np.zeros_like(conf.q_actuated_home)
        ])

        self.robot = Talos(
            self.simulator,
            conf.urdf,
            self.tsid_wrapper.model,
            self,
            q=q_init,
            verbose=True,
            useFixedBase=False)

        self.t_publish = 0.0
        self.t_plot = 0.0

        # Shift center of mass to balance on one foot
        com_curr = self.tsid_wrapper.comState().value()
        foot_placement_RF = self.tsid_wrapper.get_placement_RF().translation
        com_new = np.array(
            [foot_placement_RF[0], foot_placement_RF[1], com_curr[2]])
        self.tsid_wrapper.setComRefState(com_new)

        self.duration_shift_com = 2.0
        self.foot_lf_is_lifted = False

        # Robot squatting parameters
        self.t_start_squat = 4.0  # s
        self.is_squatting = False
        self.wave_amp = 0.05  # m
        self.wave_freq = 0.5  # Hz
        self.omega_squat = 2.0 * np.pi * self.wave_freq

        # Robot arm motion parameters
        self.t_start_arm_motion = 8.0  # s
        self.arm_motion = False
        self.center_of_circle = np.array([0.4, -0.2, 1.1])
        self.arm_radius = 0.2  # m
        self.arm_freq = 0.1  # Hz
        self.arm_omega = 2.0 * np.pi * self.arm_freq

        # Visualize arm trajectory
        traj, _, _ = self.arm_circle(np.linspace(0, 1./self.arm_freq, num=100))
        self.simulator.addGlobalDebugTrajectory(
            traj[0, :], traj[1, :], traj[2, :])

        # Plotting
        if DO_PLOT:
            self.plot_time = []
            self.plot_tsid_pos = []
            self.plot_tsid_vel = []
            self.plot_tsid_acc = []
            self.plot_gt_pos = []
            self.plot_gt_vel = []
            self.plot_gt_acc = []
            self.plot_ref_pos = []
            self.plot_ref_vel = []
            self.plot_ref_acc = []

            plt.ion()
            self.fig_pos, self.axs_pos = plt.subplots(3, 1, figsize=(10, 8))
            self.fig_vel, self.axs_vel = plt.subplots(3, 1, figsize=(10, 8))
            self.fig_acc, self.axs_acc = plt.subplots(3, 1, figsize=(10, 8))

    def sine_wave_squat(self, t):
        '''
        Generates a sine wave pattern for squatting motion in the vertical (Z) direction.

        Parameters:
        - t: Time since squatting started [float]

        Returns:
        - z: Position offset in Z [float]
        - z_dot: Velocity in Z [float]
        - z_ddot: Acceleration in Z [float]
        '''

        z = self.wave_amp * np.sin(t * self.omega_squat)
        z_dot = self.wave_amp * self.omega_squat * np.cos(t * self.omega_squat)
        z_ddot = - self.wave_amp * self.omega_squat**2 * \
            np.sin(t * self.omega_squat)
        return z, z_dot, z_ddot

    def arm_circle(self, t):
        '''
        Generates a circular 3D trajectory for the right hand/arm.

        Parameters:
        - t: Time values (array-like)

        Returns:
        - pos: Position trajectory [3 x len(t) ndarray]
        - vel: Velocity trajectory [3 x len(t) ndarray]
        - acc: Acceleration trajectory [3 x len(t) ndarray]
        '''

        dx = 0.0 * t
        dy = self.arm_radius * np.cos(self.arm_omega * t)
        dz = self.arm_radius * np.sin(self.arm_omega * t)
        pos = self.center_of_circle[:, np.newaxis] + np.vstack([dx, dy, dz])
        vel = self.arm_radius * self.arm_omega * np.vstack([0.0 * t,
                                                           -np.sin(self.arm_omega * t),
                                                           np.cos(self.arm_omega * t)])
        acc = self.arm_radius * self.arm_omega**2 * np.vstack([0.0 * t,
                                                              -np.cos(self.arm_omega * t),
                                                              -np.sin(self.arm_omega * t)])
        return pos, vel, acc

    def update(self):
        '''
        Simulation and Contol loop
        '''
        # Elapsed time
        t = self.simulator.simTime()

        # Update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # Update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(
            self.robot.q(), self.robot.v(), t)

        # Command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        if t > self.duration_shift_com and not self.foot_lf_is_lifted:
            # Remove left foot contact after 2 seconds
            self.tsid_wrapper.remove_contact_LF()

            # Change left foor reference 0.3m above ground
            foot_placement_LF = self.tsid_wrapper.get_placement_LF()
            foot_placement_LF.translation[2] += 0.3
            self.tsid_wrapper.set_LF_pose_ref(foot_placement_LF)

            self.foot_lf_is_lifted = True

        if t > self.t_start_squat:
            if not self.is_squatting:
                # Get initial com position
                self.com_squat_init = self.tsid_wrapper.comState()
                self.tsid_wrapper.setComRefState(
                    self.com_squat_init.value(),
                    self.com_squat_init.derivative(),
                    self.com_squat_init.second_derivative())

            # Start squatting
            t_squat = t - self.t_start_squat
            pos_z, vel_z, acc_z = self.sine_wave_squat(t_squat)
            pos_squat = self.com_squat_init.value() + np.array([0, 0, pos_z])
            vel_squat = self.com_squat_init.derivative() + \
                np.array([0, 0, vel_z])
            acc_squat = self.com_squat_init.second_derivative() + \
                np.array([0, 0, acc_z])
            self.tsid_wrapper.setComRefState(pos_squat, vel_squat, acc_squat)

            self.is_squatting = True

        if t > self.t_start_arm_motion:
            if not self.arm_motion:
                # Set orientation gains to zero
                self.tsid_wrapper.rightHandTask.setKp(
                    100*np.array([1, 1, 1, 0, 0, 0]))
                self.tsid_wrapper.rightHandTask.setKd(
                    2.0*np.sqrt(100) * np.array([1, 1, 1, 0, 0, 0]))
                # Add right hand task
                self.tsid_wrapper.add_motion_RH()

                self.arm_motion = True

            # Start arm motion
            t_arm = t - self.t_start_arm_motion
            pos_arm, vel_arm, acc_arm = self.arm_circle(t_arm)
            self.tsid_wrapper.set_RH_pos_ref(
                pos_arm.reshape(-1), vel_arm.reshape(-1), acc_arm.reshape(-1))

        # Publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # Get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)

        # Logging and plotting
        if DO_PLOT:
            self.logging(t)
            if t - self.t_plot > 5.0:
                self.t_plot = t
                self.plot()

    def update_figure(self, axs, data_tsid, data_ref, data_sim, title=None, crop=None):
        '''
        Updates a matplotlib figure with new trajectory plots for TSID, reference, and simulated data.

        Parameters:
        - axs: List of matplotlib axes [X, Y, Z]
        - data_tsid: TSID output data [list of arrays]
        - data_ref: Reference trajectory data [list of arrays]
        - data_sim: Simulated robot data [list of arrays]
        - title: Title for the plot [str]
        - crop: Y-axis limits as (min, max) [tuple (optional)]
        '''

        labels = ['X', 'Y', 'Z']
        t = np.array(self.plot_time)
        ref = np.array(data_ref)
        tsid = np.array(data_tsid)
        sim = np.array(data_sim)

        for i in range(3):
            axs[i].cla()
            axs[i].plot(t, tsid[:, i], 'r--', label='tsid')
            axs[i].plot(t, ref[:, i], 'b--', label='ref')
            axs[i].plot(t, sim[:, i], 'g--', label='sim')
            axs[i].set_ylabel(f'{labels[i]}')
            axs[i].set_title(f'{title} - {labels[i]}')
            if crop is not None:
                axs[i].set_ylim(crop[0], crop[1])
            axs[i].legend()
            axs[i].grid()

    def logging(self, t):
        '''
        Logs TSID, reference, and simulation data at time t for plotting.

        Parameters:
        - t: Current time in simulation [float]
        '''

        self.plot_time.append(t)
        com_tsid = self.tsid_wrapper.comState()
        self.plot_tsid_pos.append(com_tsid.value())
        self.plot_tsid_vel.append(com_tsid.derivative())
        self.plot_tsid_acc.append(
            self.tsid_wrapper.comTask.getDesiredAcceleration)
        gt_pos = self.robot.baseWorldPosition()
        gt_vel = self.robot.baseWorldLinearVeloctiy()
        # Central difference method to get acceleration
        if len(self.plot_gt_vel) >= 3:
            gt_acc = (self.plot_gt_vel[-1] - self.plot_gt_vel[-3]
                      ) / (self.plot_time[-1] - self.plot_time[-3])
        else:
            gt_acc = np.zeros_like(gt_vel)
        self.plot_gt_pos.append(gt_pos)
        self.plot_gt_vel.append(gt_vel)
        self.plot_gt_acc.append(gt_acc)
        com_ref = self.tsid_wrapper.comReference()
        self.plot_ref_pos.append(com_ref.value())
        self.plot_ref_vel.append(com_ref.derivative())
        self.plot_ref_acc.append(com_ref.second_derivative())

    def plot(self):
        '''
        Renders the position, velocity, and acceleration plots in real-time using logged data.
        '''

        self.update_figure(self.axs_pos, self.plot_tsid_pos,
                           self.plot_ref_pos, self.plot_gt_pos, "Position")
        self.update_figure(self.axs_vel, self.plot_tsid_vel,
                           self.plot_ref_vel, self.plot_gt_vel, "Velocity")
        self.update_figure(self.axs_acc, self.plot_tsid_acc,
                           self.plot_ref_acc, self.plot_gt_acc, "Acceleration",
                           crop=(-3, 3))

        self.fig_pos.tight_layout()
        self.fig_vel.tight_layout()
        self.fig_acc.tight_layout()

        self.fig_pos.canvas.draw()
        self.fig_vel.canvas.draw()
        self.fig_acc.canvas.draw()

        self.fig_pos.canvas.flush_events()
        self.fig_vel.canvas.flush_events()
        self.fig_acc.canvas.flush_events()

################################################################################
# main
################################################################################


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            env.update()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
