import sys
import time
sys.path.insert(0, "..")  # noqa
from simulator.pybullet_wrapper import PybulletWrapper
from walking.go2 import Go2
from walking.controller import Go2Controller
from walking.foot_trajectory_planner import TrajectoriesPlanner

import pinocchio as pin
import numpy as np


def main():
    sim = PybulletWrapper()
    robot = Go2(sim)
    # com reference height
    com_h = robot.robot.baseCoMPosition()[2]
    controller = Go2Controller(robot.model, com_h)

    x0 = robot.get_state()
    q_d = x0[7:robot.nq]
    v_d = x0[robot.nq+6:robot.nq+robot.nv]
    x0_com = robot.get_com()

    # get trajectory  for N steps
    N = 4
    step_length = 0.25
    step_height = 0.15
    time_step = 1e-2
    n_per_step = 50
    traj_planner = TrajectoriesPlanner(robot.model, step_length,
                                       step_height, time_step, n_per_step)

    foot_trajectories, com_trajectories = traj_planner.get_N_full_steps(
        N, x0_com)

    def visualize_trajectories(sim, foot_trajectories, com_trajectories):
        for com_traj, foot_traj in zip(com_trajectories, foot_trajectories):
            for com_task, foot_pos_task in zip(com_traj, foot_traj):
                for com_pos, foot_pos in zip(com_task, foot_pos_task):
                    pos = foot_pos[1].translation
                    sim.addSphereMarker(
                        pos, color=[1, 0, 0, 1])  # Red for feet
                    sim.addSphereMarker(
                        com_pos, color=[0, 1, 0, 1])  # Green for COM

    visualize_trajectories(sim, foot_trajectories, com_trajectories)

    # ocp
    problem = controller.walking_problem_ocp(
        x0, time_step, foot_trajectories, com_trajectories)

    controls, states = controller.solve(problem)

    for u, x_d, in zip(controls, states):
        robot.update()
        x0 = robot.get_state()
        q = x0[7:robot.nq]
        v = x0[robot.nq+6:robot.nq+robot.nv]

        q_d = x_d[:robot.nq]
        v_d = x_d[robot.nq:]

        # ------------ TEST ------------
        """
        q = x0[:robot.nq]
        v = x0[robot.nq:robot.nq + robot.nv]
        tau = pin.rnea(controller.model, controller.data,
                       q, v, np.zeros_like(v))
        tau = tau[6:]
        robot.set_torque(tau)

        try:
            k, K = controller.get_feedback_gains(node_index=0)

            state_error = xs0 - x0
            alpha = 1.0

            if state_error.shape[0] == 37 and K.shape[1] == 36:
                # Convert state error to tangent space representation
                q_current = x0[:robot.nq]
                q_ref = xs0[:robot.nq]
                v_current = x0[robot.nq:]
                v_ref = xs0[robot.nq:]

                # configuration error in tangent space
                q_error_tangent = pin.difference(
                    controller.model, q_ref, q_current)
                v_error = v_current - v_ref
                state_error_tangent = np.concatenate(
                    [q_error_tangent, v_error])

                uk = u0 + alpha * k + K @ state_error_tangent
                # print(f"uo: {u0}\nuk: {uk}")
            else:
                uk = u0 + alpha * k + K @ state_error

        except (AttributeError, RuntimeError) as e:
            print(f"Warning: Could not get feedback gains: {e}")
            uk = u0
        """
        # ------------ TEST ------------

        robot.set_torque(u, q_d, q, v_d, v)
        # robot.set_torque(u0)
        # robot.set_position(x0[:robot.nq])

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(0.01)


if __name__ == '__main__':
    main()
