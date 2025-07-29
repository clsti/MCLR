import sys
sys.path.insert(0, "..")  # noqa
import pinocchio as pin
import numpy as np
import walking.conf_go2 as conf
from simulator.robot import Robot
import example_robot_data


class Go2():

    def __init__(self, sim):
        """
        Initialize the Go2 robot in simulation.

        Args:
            sim: Simulator.
        """
        self.sim = sim
        self.conf = conf

        self.go2 = example_robot_data.load("go2")
        self.model = self.go2.model
        self.data = self.model.createData()
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

        # PD parameters
        self.Kp = conf.Kp
        self.Kd = conf.Kd

        # Fallback torque
        self.tau_ff_fallback = np.zeros(12)

    def update(self):
        """
        Update the robot's state.
        """
        self.robot.update()

    def get_state(self, nqx=False):
        """
        Get the robot state vector.

        Args:
            nqx (bool): If True, return base orientation in exponential coordinates.

        Returns:
            np.ndarray: Concatenated state vector [q, v].
        """
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

    def get_com(self):
        """
        Get the current position of the center of mass (CoM).

        Returns:
            np.ndarray: CoM position.
        """
        return self.robot.baseCoMPosition()

    def set_torque(self, tau_ff, q_d=None, q=None, v_d=None, v=None):
        """
        Apply torque commands to the robot's actuated joints, with optional PD control.

        Args:
            tau_ff (np.ndarray): Feedforward torques from controller.
            q_d (np.ndarray, optional): Desired joint positions.
            q (np.ndarray, optional): Current joint positions.
            v_d (np.ndarray, optional): Desired joint velocities.
            v (np.ndarray, optional): Current joint velocities.
        """
        # Torque from crocoddyl sometimes empty -> use fallback torque from previous run if possible
        if tau_ff.shape == (0,):
            tau_ff = self.tau_ff_fallback
        else:
            # store fallback torque
            self.tau_ff_fallback = tau_ff

        if all(x is not None for x in [q_d, q, v_d, v]):
            q_dif = q_d - q
            v_dif = v_d - v

            tau = self.Kp * q_dif + self.Kd * v_dif + tau_ff
        else:
            tau = tau_ff

        self.robot.setActuatedJointTorques(tau)

    def set_position(self, pos, vel=None):
        """
        Set desired joint positions (and velocities) for actuated joints.

        Args:
            pos (np.ndarray): Desired joint positions.
            vel (np.ndarray, optional): Desired joint velocities.
        """
        self.robot.setActuatedJointPositions(pos, v=vel)
