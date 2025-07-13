from simulator.robot import Robot

from rclpy.node import Node
import walking.conf_go2 as conf
import numpy as np
import tsid
import pinocchio as pin


class Go2(Node):

    def __init__(self, sim):
        super().__init__('go2_node')
        self.sim = sim
        self.conf = conf

        self.robot = tsid.RobotWrapper(
            self.conf.urdf,
            [self.conf.path],
            pin.JointModelFreeFlyer(), False)
        self.model = self.robot.model()

        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.model,
            q=conf.q_stable,
            basePosition=np.array([0, 0, 0.47]),
            baseQuationerion=np.array([0, 0, 0, 1]),
            useFixedBase=False
        )
