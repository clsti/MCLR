import numpy as np

# simulator
from simulator.robot import Robot

# config parameters
import walking.conf_go2 as conf

# Ros
from rclpy.node import Node
from sensor_msgs.msg import JointState

import tsid
import pinocchio as pin


class Go2(Node):

    def __init__(self, sim):
        super().__init__('go2_node')
        self.sim = sim
        self.conf = conf

        self.tsid_robot = tsid.RobotWrapper(
            self.conf.urdf,
            [self.conf.path],
            pin.JointModelFreeFlyer(), False)
        self.model = self.tsid_robot.model()

        # Spawn robot in simulation
        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.model,
            q=conf.q_stable,
            basePosition=np.array([0, 0, 0.36]),
            baseQuationerion=np.array([0, 0, 0, 1]),
            useFixedBase=False
        )

        # Joint state publisher
        self.pub_joint = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_msg = JointState()
        self.joint_msg.name = self.robot.actuatedJointNames()

    def publish(self):
        # Publish the jointstate
        now = self.get_clock().now().to_msg()

        # Publish joint states
        self.joint_msg.header.stamp = now
        self.joint_msg.position = self.robot.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.robot.actuatedJointVelocity().tolist()
        # self.joint_msg.effort = self.tau.tolist()

        self.pub_joint.publish(self.joint_msg)
