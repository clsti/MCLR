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
        # call base class constructor

        # Initial condition for the simulator an model
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

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

    def update(self):
        # update base class
        super().update()

    def publish(self, T_b_w, tau):
        # publish jointstate
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"

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
    def __init__(self):
        super().__init__('tutorial_4_standing_node')

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        self.simulator = PybulletWrapper(sim_rate=conf.f_cntr)

        # init ROBOT
        self.robot = Talos(
            self.simulator,
            conf.urdf,
            self.tsid_wrapper.model,
            self,
            q=conf.q_home,
            verbose=True,
            useFixedBase=False)

        self.t_publish = 0.0
        self.t_plot = 0.0

        # balance on one foot
        com_curr = self.tsid_wrapper.comState().value()
        foot_placement_RF = self.tsid_wrapper.get_placement_RF().translation
        com_new = np.array(
            [foot_placement_RF[0], foot_placement_RF[1], com_curr[2]])
        self.tsid_wrapper.setComRefState(com_new)

        self.duration_shift_com = 2.0
        self.foot_lf_is_lifted = False

        # squatting
        self.t_start_squat = 4.0  # s
        self.is_squatting = False
        self.wave_amp = 0.05  # m
        self.wave_freq = 0.5  # Hz
        self.omega_squat = 2.0 * np.pi * self.wave_freq

        # Arm motions
        self.t_start_arm_motion = 8.0  # s
        self.arm_motion = False
        self.center_of_circle = np.array([0.4, -0.2, 1.1])
        self.arm_radius = 0.2  # m
        self.arm_freq = 0.1  # Hz
        self.omega_arm = 2.0 * np.pi * self.arm_freq

        # visualize arm trajectory
        traj, _, _ = self.arm_circle(np.linspace(0, 1./self.arm_freq, num=100))
        self.simulator.addGlobalDebugTrajectory(
            traj[0, :], traj[1, :], traj[2, :])

        # Plotting
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
        z = self.wave_amp * np.sin(t * self.omega_squat)
        z_dot = self.wave_amp * self.omega_squat * np.cos(t * self.omega_squat)
        z_ddot = - self.wave_amp * self.omega_squat**2 * \
            np.sin(t * self.omega_squat)
        return z, z_dot, z_ddot

    def arm_circle(self, t):
        dx = 0.0 * t
        dy = self.arm_radius * np.cos(self.omega_arm * t)
        dz = self.arm_radius * np.sin(self.omega_arm * t)
        pos = self.center_of_circle[:, np.newaxis] + np.vstack([dx, dy, dz])
        vel = self.arm_radius * self.omega_arm * np.vstack([0.0 * t,
                                                           -np.sin(self.omega_arm * t),
                                                           np.cos(self.omega_arm * t)])
        acc = self.arm_radius * self.omega_arm**2 * np.vstack([0.0 * t,
                                                              -np.cos(self.omega_arm * t),
                                                              -np.sin(self.omega_arm * t)])
        return pos, vel, acc

    def update(self):
        # elaped time
        t = self.simulator.simTime()

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(
            self.robot.q(), self.robot.v(), t)

        # command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        if t > self.duration_shift_com and not self.foot_lf_is_lifted:
            # remove left foot contact after 2 seconds
            self.tsid_wrapper.remove_contact_LF()

            # change left foor reference 0.3m above ground
            foot_placement_LF = self.tsid_wrapper.get_placement_LF()
            foot_placement_LF.translation[2] += 0.3
            self.tsid_wrapper.set_LF_pose_ref(foot_placement_LF)

            self.foot_lf_is_lifted = True

        if t > self.t_start_squat:
            if not self.is_squatting:
                # get initial position
                self.com_squat_init = self.tsid_wrapper.comState()
                self.tsid_wrapper.setComRefState(
                    self.com_squat_init.value(),
                    self.com_squat_init.derivative(),
                    self.com_squat_init.second_derivative())

            # start squatting
            t_squat = t - self.t_start_squat
            pos_z, vel_z, acc_z = self.sine_wave_squat(t_squat)
            pos = self.com_squat_init.value() + np.array([0, 0, pos_z])
            vel = self.com_squat_init.derivative() + np.array([0, 0, vel_z])
            acc = self.com_squat_init.second_derivative() + \
                np.array([0, 0, acc_z])
            self.tsid_wrapper.setComRefState(pos, vel, acc)

            self.is_squatting = True

        if t > self.t_start_arm_motion:
            if not self.arm_motion:
                # add right hand task
                self.tsid_wrapper.formulation.addMotionTask(
                    self.tsid_wrapper.rightFootTask, conf.w_foot, 1, 0.0)
                # set orientation gains to zero
                self.tsid_wrapper.rightHandTask.setKp(
                    100*np.array([1, 1, 1, 0, 0, 0]))
                self.tsid_wrapper.rightHandTask.setKd(
                    2.0*np.sqrt(100) * np.array([1, 1, 1, 0, 0, 0]))

                self.arm_motion = True

            # start arm motion
            t_arm = t - self.t_start_arm_motion
            pos, vel, acc = self.arm_circle(t_arm)
            self.tsid_wrapper.set_RH_pos_ref(
                pos.reshape(-1), vel.reshape(-1), acc.reshape(-1))

            # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)

        if DO_PLOT:
            self.logging(t)
            if t - self.t_plot > 5.0:
                self.t_plot = t
                self.plot()

    def update_figure(self, axs, data_tsid, data_ref, data_sim=None, title=None):
        labels = ['X', 'Y', 'Z']
        t = np.array(self.plot_time)
        ref = np.array(data_ref)
        tsid = np.array(data_tsid)
        sim = np.array(data_sim) if data_sim else None

        for i in range(3):
            axs[i].cla()
            axs[i].plot(t, tsid[:, i], 'r--', label='tsid')
            axs[i].plot(t, ref[:, i], 'b--', label='ref')
            if sim is not None:
                axs[i].plot(t, sim[:, i],
                            'g--', label='sim')
            axs[i].set_ylabel(f'{labels[i]}')
            axs[i].set_title(f'{title} - {labels[i]}')
            axs[i].legend()
            axs[i].grid()

    def logging(self, t):
        # get data
        self.plot_time.append(t)
        com_tsid = self.tsid_wrapper.comState()
        self.plot_tsid_pos.append(com_tsid.value())
        self.plot_tsid_vel.append(com_tsid.derivative())
        self.plot_tsid_acc.append(com_tsid.second_derivative())
        gt_pos = self.robot.baseWorldPosition()
        gt_vel = self.robot.baseWorldLinearVeloctiy()
        # gt_acc = self.robot.baseWorldAcceleration()
        self.plot_gt_pos.append(gt_pos)
        self.plot_gt_vel.append(gt_vel)
        # self.plot_gt_acc.append(gt_acc)
        com_ref = self.tsid_wrapper.comReference()
        self.plot_ref_pos.append(com_ref.value())
        self.plot_ref_vel.append(com_ref.derivative())
        self.plot_ref_acc.append(com_ref.second_derivative())

    def plot(self):
        self.update_figure(self.axs_pos, self.plot_tsid_pos,
                           self.plot_ref_pos, self.plot_gt_pos, "Position")
        self.update_figure(self.axs_vel, self.plot_tsid_vel,
                           self.plot_ref_vel, self.plot_gt_vel, "Velocity")
        self.update_figure(self.axs_acc, self.plot_tsid_acc,
                           self.plot_ref_acc, None, "Acceleration")

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
