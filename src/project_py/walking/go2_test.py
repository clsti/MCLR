import sys
sys.path.insert(0, "..")  # noqa
sys.path.insert(0, "..")  # noqa
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution
import crocoddyl
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import walking.conf_go2 as conf
from simulator.robot import Robot
import example_robot_data
package_dirs = [".."]


class Go2:
    def __init__(self, sim):
        self.sim = sim
        self.conf = conf

        self.go2 = example_robot_data.load("go2")
        self.model = self.go2.model
            conf.urdf,
            self.model,
            q = q_init,
            baseQuationerion = q_init[3:7],
            useFixedBase = False
        )

        # Initialize crocoddyl walking controller
        self._init_walking_controller()

        # Control loop parameters
        self.control_step= 0
        self.replan_frequency= 25  # Replan every N steps
        self.trajectory_index= 0

        self.solved= False

    def _init_walking_controller(self):
        """Initialize crocoddyl walking problem"""
        # Reduce torque limits
        lims= self.model.effortLimit
        lims *= 0.5
        self.model.effortLimit= lims

        # Define foot frame names
        lfFoot, rfFoot, lhFoot, rhFoot= "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        self.gait= SimpleQuadrupedalGaitProblem(
            self.model, lfFoot, rfFoot, lhFoot, rhFoot)

        # Walking gait parameters
        self.walking_gait= {
            "stepLength": 0.25,
            "stepHeight": 0.15,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }

        # Initialize with standing configuration
        q0 = self.model.referenceConfigurations["standing"].copy()
        v0 = pin.utils.zero(self.model.nv)

        self.current_state = np.concatenate([q0, v0])

        # Initialize solver
        self.solver = None
        self.planned_trajectory = None

    def get_current_state_from_pybullet(self):
        """Step 1: Get current robot state from PyBullet"""
        # Get base position and orientation
        base_pos = self.robot.baseWorldPosition()
        base_orn = self.robot.baseWorldOrientation()  # quaternion [x,y,z,w]

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

            plotSolution(self.solver, bounds=True, figIndex=1, show=False)
            us = np.array([np.zeros(12) if u is None or len(
                u) == 0 else np.ravel(u) for u in self.solver.us])

            plt.figure(figsize=(10, 6))
            for i in range(us.shape[1]):
                plt.plot(us[:, i], label=f'Torque {i}')
            plt.xlabel('Time step')
            plt.ylabel('Joint torque (Nm)')
            plt.title('Planned Joint Torques (u)')
            plt.legend(ncol=4, fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

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
            return np.zeros(12)

        # Get current control input
        print(f"length:{len(self.planned_trajectory['controls'])}")
        u = self.planned_trajectory['controls'][self.trajectory_index]

        # Extract joint torques (skip base coordinates)
        # Assuming the control vector is joint torques
        # This might need adjustment based on crocoddyl's control structure
        joint_torques = u

        return joint_torques

    def apply_joint_commands(self, joint_torques):
        """Step 4: Apply joint commands to PyBullet robot"""
        if joint_torques is not None:
            # Apply torques to robot joints
            self.robot.setActuatedJointTorques(joint_torques)

    def step_walking_controller(self):
        """Main control loop - combines all steps"""
        # Step 1: Get current state
        if self.solved == False:
            current_state = self.get_current_state_from_pybullet()
            print(f"current state pybullet: {current_state}")
            q0 = self.model.referenceConfigurations["standing"].copy(
            )
            v0 = pin.utils.zero(self.model.nv)
            current_state = np.concatenate([q0, v0])
            print(f"current state: {current_state}")
            self.solve_walking_problem(current_state)
            self.solved = True

        # Step 3: Extract control inputs
        joint_torques = self.extract_control_inputs()

        # Step 4: Apply joint commands
        # TODO: WHY the hell are some torques empty???????????
        if len(joint_torques) == 12:
            self.apply_joint_commands(joint_torques)
        else:
            print("[WARNING] No joint torques from solver")

        # Update indices
        self.trajectory_index += 1
        self.control_step += 1
