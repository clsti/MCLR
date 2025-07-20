import sys
sys.path.insert(0, "..")  # noqa
sys.path.insert(0, "..")  # noqa
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
import crocoddyl
import pinocchio as pin
import numpy as np
import walking.conf_go2 as conf
from simulator.robot import Robot
import example_robot_data
package_dirs = [".."]


class Go2():

    def __init__(self, sim):
        self.sim = sim
        self.conf = conf

        self.go2 = example_robot_data.load("go2")
        self.model = self.go2.model

        q_init = self.model.referenceConfigurations["standing"].copy()
        self.q_init = q_init[7:]

        # Spawn robot in simulation
        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.model,
            q=q_init,
            basePosition=q_init[:3],
            baseQuationerion=q_init[3:7],
            useFixedBase=False
        )

    def update(self):
        self.robot.update()

    def set_torque(self):
        Kp = 100.0  # proportional gain
        Kd = 1.0   # derivative gain

        q_current = self.robot.actuatedJointPosition()
        dq_current = self.robot.actuatedJointVelocity()

        torque = Kp * (self.q_init - q_current) - Kd * dq_current
        self.robot.setActuatedJointTorques(torque)

    def publish(self):
        # Removed ROS2 publish logic
        # Instead, just print or log the joint states for debugging
        joint_positions = self.robot.actuatedJointPosition()
        joint_velocities = self.robot.actuatedJointVelocity()
        print("Joint positions:", joint_positions)
        print("Joint velocities:", joint_velocities)
