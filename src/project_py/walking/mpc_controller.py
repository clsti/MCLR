import sys
sys.path.insert(0, "..")  # noqa
import walking.conf_go2 as conf
import copy


class MPCController:
    """
    Model Predictive Control (MPC) Controller for quadruped walking.

    Recalculates the optimal control problem after each step execution,
    providing more reactive control compared to pre-planning all steps.
    """

    def __init__(self, robot, controller, traj_planner):
        """
        Initialize MPC Controller.

        Args:
            robot: Go2 robot instance
            controller: Go2Controller instance  
            traj_planner: TrajectoriesPlanner instance
        """
        self.robot = robot
        self.controller = controller
        self.traj_planner = traj_planner

        # MPC parameters
        self.horizon = conf.mpc_horizon
        self.replan_frequency = conf.mpc_replan_frequency  # Now actually used!
        self.warm_start = conf.mpc_warm_start

        # Control state
        self.total_controls_executed = 0  # Total controls executed since start
        self.full_step_counter = 0  # Tracks complete gait cycles
        # Tracks which leg is stepping (0-3: RR, LF, RF, LH)
        self.leg_counter = 0

        # Current control sequence and states
        self.current_controls = []
        self.current_states = []
        self.current_control_idx = 0
        self.trajectory_updated = False  # Flag to indicate new trajectory available

        # Previous solution for warm start
        self.prev_controls = None
        self.prev_states = None

        # Step execution tracking
        # In Go2 gait: One "full step" = 4 leg movements (RR->LF->RF->LH)
        # Each leg movement has n_per_step time steps + 1 foot switch action
        self.controls_per_leg = conf.n_per_step + 1  # Including foot switch
        self.controls_per_full_step = 4 * self.controls_per_leg  # 4 legs per full step
        self.replan_every_n_controls = self.replan_frequency * self.controls_per_full_step

    def should_replan(self):
        """
        Determine if we should replan the trajectory.

        Returns:
            bool: True if replanning is needed
        """
        # Replan at the start
        if self.total_controls_executed == 0:
            return True

        # Replan every N full steps (based on replan_frequency)
        if self.total_controls_executed % self.replan_every_n_controls == 0:
            return True

        # Emergency replan if we run out of controls
        if self.current_control_idx >= len(self.current_controls):
            return True

        return False

    def replan(self, current_state):
        """
        Replan the trajectory from current state.

        Args:
            current_state: Current robot state
        """
        # Get current COM position
        current_com = self.robot.get_com()

        # Reset trajectory planner state for consistent planning
        # This ensures that each replan starts with fresh foot positions
        self.traj_planner.first_step = (self.full_step_counter == 0)

        # Plan trajectories for the horizon
        foot_trajectories, com_trajectories = self.traj_planner.get_N_full_steps(
            current_state, self.horizon, current_com)

        # Create optimal control problem
        problem = self.controller.walking_problem_ocp(
            current_state, conf.time_step, foot_trajectories, com_trajectories)

        # Solve with warm start if available
        if self.warm_start and self.prev_controls is not None:
            try:
                # Shift previous solution as initial guess
                x_init = self._shift_trajectory(
                    self.prev_states, current_state)
                u_init = self._shift_controls(self.prev_controls)
                controls, states = self.controller.solve(
                    current_state, problem, x_init, u_init)
            except Exception as e:
                # Fallback to cold start if warm start fails
                controls, states = self.controller.solve(
                    current_state, problem)
        else:
            # Cold start
            controls, states = self.controller.solve(current_state, problem)

        # Store solution
        self.current_controls = controls
        self.current_states = states
        self.current_control_idx = 0
        self.trajectory_updated = True  # Mark that new trajectory is available

        # Save for next warm start
        self.prev_controls = copy.deepcopy(controls)
        self.prev_states = copy.deepcopy(states)

        # Reset control counters for this full step
        self.current_control_in_full_step = 0

    def get_current_control(self):
        """
        Get the current control to execute.

        Returns:
            tuple: (control, state) or (None, None) if no control available
        """
        if self.current_control_idx < len(self.current_controls):
            u = self.current_controls[self.current_control_idx]
            x_d = self.current_states[self.current_control_idx]
            return u, x_d
        else:
            print("Warning: No more controls available!")
            return None, None

    def step(self, current_state):
        """
        Execute one MPC step.

        Args:
            current_state: Current robot state

        Returns:
            tuple: (control, desired_state) to execute
        """
        # Check if we need to replan
        if self.should_replan():
            self.replan(current_state)

        # Get current control
        u, x_d = self.get_current_control()

        # Advance counters
        self.current_control_idx += 1
        self.total_controls_executed += 1

        # Update step tracking
        controls_in_current_step = self.total_controls_executed % self.controls_per_full_step
        self.leg_counter = controls_in_current_step // self.controls_per_leg

        # Check if we completed a full gait cycle
        if controls_in_current_step == 0 and self.total_controls_executed > 0:
            self.full_step_counter += 1

        return u, x_d

    def _shift_trajectory(self, prev_states, current_state):
        """
        Shift previous state trajectory for warm start.

        Args:
            prev_states: Previous state trajectory
            current_state: Current robot state

        Returns:
            list: Shifted trajectory for warm start
        """
        if not prev_states or len(prev_states) < 2:
            return [current_state] * 100  # Fallback

        # Simple shift: use states from one full step onwards
        shift_idx = min(self.controls_per_full_step, len(prev_states) - 1)
        shifted = prev_states[shift_idx:]

        # Pad with last state if needed
        while len(shifted) < len(prev_states):
            shifted.append(prev_states[-1])

        # Replace first state with current state
        shifted[0] = current_state

        return shifted

    def _shift_controls(self, prev_controls):
        """
        Shift previous control sequence for warm start.

        Args:
            prev_controls: Previous control sequence

        Returns:
            list: Shifted control sequence
        """
        if not prev_controls or len(prev_controls) < 2:
            return None

        # Simple shift: use controls from one full step onwards
        shift_idx = min(self.controls_per_full_step, len(prev_controls) - 1)
        shifted = prev_controls[shift_idx:]

        # Pad with last control if needed
        while len(shifted) < len(prev_controls):
            shifted.append(prev_controls[-1])

        return shifted

    def get_stats(self):
        """
        Get MPC controller statistics.

        Returns:
            dict: Statistics about the controller state
        """
        return {
            'total_controls_executed': self.total_controls_executed,
            'leg_counter': self.leg_counter,
            'full_step_counter': self.full_step_counter,
            'current_control_idx': self.current_control_idx,
            'controls_remaining': len(self.current_controls) - self.current_control_idx,
            'controls_in_current_step': self.total_controls_executed % self.controls_per_full_step,
            'trajectory_updated': self.trajectory_updated
        }
