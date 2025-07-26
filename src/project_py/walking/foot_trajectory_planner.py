import numpy as np
import pinocchio as pin

import walking.conf_go2 as conf


class TrajectoriesPlanner():

    def __init__(self, model, data, step_length, step_height, time_step, n_per_step):

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
        Get the trajectory for one foot and respective com task
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
        x0_pos = [x0_rh, x0_rf, x0_lh, x0_lf]
        """
        x0_rh, x0_rf, x0_lh, x0_lf = x0_pos
        if self.first_step:
            rh_traj = self._get_traj([x0_rh], x0_com, [self.rhFootId], True)
            lf_traj = self._get_traj([x0_lf], x0_com, [self.lfFootId], True)
            self.first_step = False
        else:
            rh_traj = self._get_traj([x0_rh], x0_com, [self.rhFootId])
            lf_traj = self._get_traj([x0_lf], x0_com, [self.lfFootId])
        rf_traj = self._get_traj([x0_rf], x0_com, [self.rfFootId])
        lh_traj = self._get_traj([x0_lh], x0_com, [self.lhFootId])

        return rh_traj, lf_traj, rf_traj, lh_traj

    def get_N_full_steps(self, x0, N, x0_com):
        foot_traj_N = []
        com_traj_N = []

        # get foot positions
        x0_pos = self.get_foot_states(x0)

        for _ in range(N):
            rh_traj, lf_traj, rf_traj, lh_traj = self.get_full_step_itr(
                x0_pos, x0_com)
            foot_traj_N.append(
                [rh_traj[0], lf_traj[0], rf_traj[0], lh_traj[0]])
            com_traj_N.append([rh_traj[1], lf_traj[1], rf_traj[1], lh_traj[1]])

        return foot_traj_N, com_traj_N

    def visualize_trajectories(self, sim, foot_trajectories, com_trajectories):
        for com_traj, foot_traj in zip(com_trajectories, foot_trajectories):
            for com_task, foot_pos_task in zip(com_traj, foot_traj):
                for com_pos, foot_pos in zip(com_task, foot_pos_task):
                    pos = foot_pos[0][1].translation
                    sim.addSphereMarker(
                        pos, color=[1, 0, 0, 1])  # Red for feet
                    sim.addSphereMarker(
                        com_pos, color=[0, 1, 0, 1])  # Green for COM
