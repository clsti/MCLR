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
import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
import example_robot_data


class Go2(Node):

    def __init__(self, sim):
        super().__init__('go2_node')
        self.sim = sim
        self.conf = conf

        # Initialize robot wrapper for tsid
        self.robot_wrapper = tsid.RobotWrapper(
            self.conf.urdf,
            [self.conf.path],
            pin.JointModelFreeFlyer(), False)
        self.model = self.robot_wrapper.model()

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

        # Initialize crocoddyl walking controller
        self._init_walking_controller()

        # Control loop parameters
        self.control_step = 0
        self.replan_frequency = 25  # Replan every N steps
        self.trajectory_index = 0

    def _init_walking_controller(self):
        """Initialize crocoddyl walking problem"""
        # Load Go2 model for crocoddyl
        go2_data = example_robot_data.load("go2")
        self.croc_model = go2_data.model

        # Reduce torque limits
        lims = self.croc_model.effortLimit
        lims *= 0.5
        self.croc_model.effortLimit = lims

        # Define foot frame names
        lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        self.gait = SimpleQuadrupedalGaitProblem(
            self.croc_model, lfFoot, rfFoot, lhFoot, rhFoot)

        # Walking gait parameters
        self.walking_gait = {
            "stepLength": 0.25,
            "stepHeight": 0.15,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }

        # Initialize with standing configuration
        q0 = self.croc_model.referenceConfigurations["standing"].copy()
        v0 = pin.utils.zero(self.croc_model.nv)
        self.current_state = np.concatenate([q0, v0])

        # Initialize solver
        self.solver = None
        self.planned_trajectory = None

    def get_current_state_from_pybullet(self):
        """Step 1: Get current robot state from PyBullet"""
        # Get base position and orientation
        base_pos = self.robot.basePosition()
        base_orn = self.robot.baseOrientation()  # quaternion [x,y,z,w]

        # Get joint positions and velocities
        joint_pos = self.robot.jointPositions()
        joint_vel = self.robot.jointVelocities()

        # Get base velocity
        base_vel_linear = self.robot.baseVelocityLinear()
        base_vel_angular = self.robot.baseVelocityAngular()

        # Construct state vector [q, v] for crocoddyl
        # q = [base_pos, base_orn, joint_pos]
        # v = [base_vel_linear, base_vel_angular, joint_vel]
        q = np.concatenate([base_pos, base_orn, joint_pos])
        v = np.concatenate([base_vel_linear, base_vel_angular, joint_vel])

        return np.concatenate([q, v])

    def solve_walking_problem(self, current_state):
        """Step 2: Solve walking problem with crocoddyl"""
        try:
            # Create walking problem from current state
            self.solver = crocoddyl.SolverBoxDDP(
                self.gait.createWalkingProblem(
                    current_state,
                    self.walking_gait["stepLength"],
                    self.walking_gait["stepHeight"],
                    self.walking_gait["timeStep"],
                    self.walking_gait["stepKnots"],
                    self.walking_gait["supportKnots"],
                )
            )

            # Initial guess
            xs = [current_state] * (self.solver.problem.T + 1)
            us = self.solver.problem.quasiStatic(
                [current_state] * self.solver.problem.T)

            # Solve
            self.solver.solve(xs, us, 100, False, 0.1)

            # Store the planned trajectory
            self.planned_trajectory = {
                'states': self.solver.xs,
                'controls': self.solver.us
            }

            return True

        except Exception as e:
            print(f"Walking problem solving failed: {e}")
            return False

    def extract_control_inputs(self):
        """Step 3: Extract next control inputs from solution"""
        if self.planned_trajectory is None or self.trajectory_index >= len(self.planned_trajectory['controls']):
            return None

        # Get current control input
        u = self.planned_trajectory['controls'][self.trajectory_index]

        # Extract joint torques (skip base coordinates)
        # Assuming the control vector is joint torques
        joint_torques = u  # This might need adjustment based on crocoddyl's control structure

        return joint_torques

    def apply_joint_commands(self, joint_torques):
        """Step 4: Apply joint commands to PyBullet robot"""
        if joint_torques is not None:
            # Apply torques to robot joints
            self.robot.setJointTorques(joint_torques)

    def step_walking_controller(self):
        """Main control loop - combines all steps"""
        # Step 1: Get current state
        current_state = self.get_current_state_from_pybullet()

        # Step 2: Solve walking problem (every N steps)
        if self.control_step % self.replan_frequency == 0:
            print(f"Replanning at step {self.control_step}")
            if self.solve_walking_problem(current_state):
                self.trajectory_index = 0  # Reset trajectory index

        # Step 3: Extract control inputs
        joint_torques = self.extract_control_inputs()

        # Step 4: Apply joint commands
        self.apply_joint_commands(joint_torques)

        # Update indices
        self.trajectory_index += 1
        self.control_step += 1

        # Step 5: This happens in main loop (sim.step())
