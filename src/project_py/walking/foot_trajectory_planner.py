import numpy as np
import pinocchio as pin

import walking.conf_go2 as conf


class TrajectoriesPlanner():
    """"
    Trajectory Planner for a quadruped robot walking task.
    """

    def __init__(self, model, data, step_length, step_height, time_step, n_per_step):
        """
        Initialize the trajectory planner for quadruped walking.

        Args:
            model (pinocchio.Model): Robot model.
            data (pinocchio.Data): Robot data.
            step_length (float): Desired step length in the x-direction.
            step_height (float): Maximum swing foot height during step.
            time_step (float): Time between each trajectory point.
            n_per_step (int): Number of points per step.
        """
        self.model = model
        self.data = data

        self.step_length = step_length
        self.step_height = step_height
        self.time_step = time_step
        self.n_per_step = n_per_step

        self.num_legs = 4.0
        self.first_step = True

        self.lfFoot = conf.lfFoot
        self.rfFoot = conf.rfFoot
        self.lhFoot = conf.lhFoot
        self.rhFoot = conf.rhFoot

        # Getting the frame id for all the legs
        self.rhFootId = self.model.getFrameId(self.rhFoot)
        self.rfFootId = self.model.getFrameId(self.rfFoot)
        self.lhFootId = self.model.getFrameId(self.lhFoot)
        self.lfFootId = self.model.getFrameId(self.lfFoot)

    def get_foot_states(self, x0):
        """
        Compute the current foot positions for a given state.

        Args:
            x0 (np.array): Robot state

        Returns:
            Tuple of np.array: (rhFootPos0, rfFootPos0, lhFootPos0, lfFootPos0) - current positions of each foot in world frame.
        """
        q0 = x0[: self.model.nq]
        pin.forwardKinematics(self.model, self.data, q0)
        pin.updateFramePlacements(self.model, self.data)
        rhFootPos0 = self.data.oMf[self.rhFootId].translation
        rfFootPos0 = self.data.oMf[self.rfFootId].translation
        lhFootPos0 = self.data.oMf[self.lhFootId].translation
        lfFootPos0 = self.data.oMf[self.lfFootId].translation

        return rhFootPos0, rfFootPos0, lhFootPos0, lfFootPos0

    def _get_traj(self, x0_foot_positions, x0_com, swingFootIds, half_step=False):
        """
        Generate swing foot and COM trajectories for one step phase.

        Args:
            x0_foot_positions (list of np.array): Current foot positions for swing feet.
            x0_com (np.array): Current center-of-mass position.
            swingFootIds (list): List of frame IDs of the swing feet.
            half_step (bool): If True, perform a half-length step (used at start).

        Returns:
            tuple:
                - foot_traj (list): Foot trajectory per time step. Each item is a list of [frameId, SE3].
                - com_traj (list): Center-of-mass trajectory per time step.
        """
        if half_step:
            step_length = 0.5 * self.step_length
        else:
            step_length = self.step_length

        foot_traj = []
        com_traj = []
        com_percentage = len(swingFootIds) / self.num_legs
        for k in range(self.n_per_step):
            swing_foot_task_k = []
            for i, p in zip(swingFootIds, x0_foot_positions):
                phKnots = self.n_per_step / 2
                if k < phKnots:  # lift foot
                    dp = np.array(
                        [step_length * (k + 1) / self.n_per_step,
                         0.0, self.step_height * k / phKnots]
                    )
                elif k == phKnots:  # reach step height
                    dp = np.array(
                        [step_length * (k + 1) / self.n_per_step, 0.0, self.step_height])
                else:  # lower foot
                    dp = np.array(
                        [
                            step_length *
                            (k + 1) / self.n_per_step, 0.0, self.step_height *
                            (1 - float(k - phKnots) / phKnots),
                        ]
                    )
                tref = p + dp
                swing_foot_task_k += [[i, pin.SE3(np.eye(3), tref)]]

            foot_traj += [swing_foot_task_k]
            com_traj += [
                np.array([step_length * (k + 1) / self.n_per_step,
                         0.0, 0.0]) * com_percentage
                + x0_com
            ]
        # Update the current foot position & com for next step
        x0_com += np.array([step_length * com_percentage, 0.0, 0.0])
        for p in x0_foot_positions:
            p += [step_length, 0.0, 0.0]

        return foot_traj, com_traj

    def get_full_step_itr(self, x0_pos, x0_com):
        """
        Generate full walking step trajectories for all four legs.

        Args:
            x0_pos (list of np.array): Initial positions of [rh, rf, lh, lf] feet.
            x0_com (np.array): Initial center-of-mass position.

        Returns:
            tuple: (rh_traj, rf_traj, lh_traj, lf_traj)
                Each is a tuple of (foot_traj, com_traj) from `_get_traj`.
        """
        x0_rh, x0_rf, x0_lh, x0_lf = x0_pos
        if self.first_step:
            rh_traj = self._get_traj([x0_rh], x0_com, [self.rhFootId], True)
            rf_traj = self._get_traj([x0_rf], x0_com, [self.rfFootId], True)
            self.first_step = False
        else:
            rh_traj = self._get_traj([x0_rh], x0_com, [self.rhFootId])
            rf_traj = self._get_traj([x0_rf], x0_com, [self.rfFootId])
        lh_traj = self._get_traj([x0_lh], x0_com, [self.lhFootId])
        lf_traj = self._get_traj([x0_lf], x0_com, [self.lfFootId])

        return rh_traj, rf_traj, lh_traj, lf_traj

    def get_N_full_steps(self, x0, N, x0_com):
        """
        Generate N full walking step trajectories.

        Args:
            x0 (np.array): Initial full robot state.
            N (int): Number of full steps to plan.
            x0_com (np.array): Initial COM position.

        Returns:
            tuple:
                - foot_traj_N (list): Nested list of foot trajectories for N steps.
                - com_traj_N (list): Nested list of COM trajectories for N steps.
        """
        foot_traj_N = []
        com_traj_N = []

        # get foot positions
        x0_pos = self.get_foot_states(x0)

        for _ in range(N):
            rh_traj, rf_traj, lh_traj, lf_traj = self.get_full_step_itr(
                x0_pos, x0_com)
            foot_traj_N.append(
                [rh_traj[0], rf_traj[0], lh_traj[0], lf_traj[0]])
            com_traj_N.append([rh_traj[1], rf_traj[1], lh_traj[1], lf_traj[1]])

        return foot_traj_N, com_traj_N

    def visualize_trajectories(self, sim, foot_trajectories, com_trajectories):
        """
        Visualize planned foot and COM trajectories in the simulator.

        Args:
            sim: Simulator.
            foot_trajectories (list): List of planned foot trajectories.
            com_trajectories (list): List of planned COM trajectories.
        """
        for com_traj, foot_traj in zip(com_trajectories, foot_trajectories):
            for com_task, foot_pos_task in zip(com_traj, foot_traj):
                for com_pos, foot_pos in zip(com_task, foot_pos_task):
                    pos = foot_pos[0][1].translation
                    sim.addSphereMarker(
                        pos, color=[1, 0, 0, 1])  # Red for feet
                    sim.addSphereMarker(
                        com_pos, color=[0, 1, 0, 1])  # Green for COM
