import sys
sys.path.insert(0, "..")  # noqa
import pinocchio as pin
import numpy as np
import walking.conf_go2 as conf
from simulator.robot import Robot
import example_robot_data


class Go2():

    def __init__(self, sim):
        self.sim = sim
        self.conf = conf

        self.go2 = example_robot_data.load("go2")
        self.model = self.go2.model
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.nv - 6

        self.x_0 = self.model.referenceConfigurations["standing"].copy()

        # Spawn robot in simulation
        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.model,
            q=self.x_0,
            basePosition=self.x_0[:3],
            baseQuationerion=self.x_0[3:7],
            useFixedBase=False
        )

    def update(self):
        self.robot.update()

    def get_state(self, nqx=False):
        # Get base position and orientation
        base_pos = self.robot.baseWorldPosition()
        base_orn = self.robot.baseWorldOrientation()  # quaternion [x,y,z,w]
        if nqx:
            R = pin.Quaternion(base_orn).toRotationMatrix()
            base_orn = pin.log3(R)

        # Get joint positions and velocities
        joint_pos = self.robot.actuatedJointPosition()
        joint_vel = self.robot.actuatedJointVelocity()

        # Get base velocity
        base_vel_linear = self.robot.baseWorldLinearVeloctiy()
        base_vel_angular = self.robot.baseWorldAngulaVeloctiy()

        # Construct state vector [q, v] for crocoddyl
        q = np.concatenate([base_pos, base_orn, joint_pos])
        v = np.concatenate([base_vel_linear, base_vel_angular, joint_vel])

        return np.concatenate([q, v])

    def set_torque(self, tau):
        self.robot.setActuatedJointTorques(tau)

    def set_position(self, pos, vel=None):
        self.robot.setActuatedJointPositions(pos, v=vel)
