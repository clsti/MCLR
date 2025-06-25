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
        - model: Robot model
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
        - T_b_w: World-to-base transform (pinocchio SE3)
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

        # Publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # Get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)


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
