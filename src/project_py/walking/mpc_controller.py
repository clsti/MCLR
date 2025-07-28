import sys
import numpy as np
sys.path.insert(0, "..")  # noqa
import walking.conf_go2 as conf
from walking.controller import Go2Controller
from walking.foot_trajectory_planner import TrajectoriesPlanner
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
        self.leg_counter = 0  # Tracks which leg is stepping (0-3: RR, LF, RF, LH)
        
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
        
        print(f"MPC Controller initialized:")
        print(f"  Horizon: {self.horizon} full steps")
        print(f"  Replan every: {self.replan_frequency} full steps ({self.replan_every_n_controls} controls)")
        print(f"  Controls per full step: {self.controls_per_full_step}")

    def should_replan(self):
        """
        Determine if we should replan the trajectory.
        
        Returns:
            bool: True if replanning is needed
        """
        # Debug output
        #print(f"DEBUG: total_controls={self.total_controls_executed}, "
        #      f"replan_every={self.replan_every_n_controls}, "
        #      f"current_control_idx={self.current_control_idx}, "
        #      f"total_available={len(self.current_controls) if self.current_controls else 0}")

        # Replan at the start
        if self.total_controls_executed == 0:
            print("DEBUG: Replanning - initial plan")
            return True
            
        # Replan every N full steps (based on replan_frequency)
        if self.total_controls_executed % self.replan_every_n_controls == 0:
            print(f"DEBUG: Replanning - after {self.replan_frequency} full steps")
            return True
            
        # Emergency replan if we run out of controls
        if self.current_control_idx >= len(self.current_controls):
            print("DEBUG: Emergency replanning - out of controls")
            return True
            
        return False

    def replan(self, current_state):
        """
        Replan the trajectory from current state.
        
        Args:
            current_state: Current robot state
        """
        print(f"\n=== MPC Replanning at control {self.total_controls_executed} ===")
        print(f"Full step counter: {self.full_step_counter}")
        
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
                x_init = self._shift_trajectory(self.prev_states, current_state)
                u_init = self._shift_controls(self.prev_controls)
                
                # Additional validation before using warm start
                if u_init is None:
                    print("Shifted controls are invalid, using cold start")
                    controls, states = self.controller.solve(current_state, problem)
                else:
                    # Debug info about warm start dimensions
                    print(f"Warm start: x_init length={len(x_init)}, u_init length={len(u_init)}")
                    if len(u_init) > 0:
                        print(f"First control dimension: {u_init[0].shape if hasattr(u_init[0], 'shape') else 'no shape'}")
                    
                    controls, states = self.controller.solve_with_initial_guess(
                        current_state, problem, x_init, u_init)
                    print("Used warm start successfully")
            except Exception as e:
                # Fallback to cold start if warm start fails
                print(f"Warm start failed ({e}), using cold start")
                controls, states = self.controller.solve(current_state, problem)
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
        
        print(f"Replanned with {len(controls)} control points")

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
            print(f"=== Completed full gait cycle {self.full_step_counter} ===")
        
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
        # No shifting needed - always start from the beginning
        if not prev_states:
            return [current_state] * 100  # Fallback
        
        # Convert to list and return states starting with current state
        if len(prev_states) > 1:
            # Create new list starting with current state, then rest of previous states
            result = [current_state]
            for i in range(1, len(prev_states)):
                result.append(prev_states[i])
            return result
        else:
            return [current_state]

    def _shift_controls(self, prev_controls):
        """
        Shift previous control sequence for warm start.
        
        Args:
            prev_controls: Previous control sequence
            
        Returns:
            list: Shifted control sequence
        """
        # No shifting needed - return previous controls as is
        return prev_controls if prev_controls else None

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
