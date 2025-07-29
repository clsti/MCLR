import sys
import time
sys.path.insert(0, "..")  # noqa
from simulator.pybullet_wrapper import PybulletWrapper
from walking.go2 import Go2
from walking.controller import Go2Controller
from walking.foot_trajectory_planner import TrajectoriesPlanner
from walking.mpc_controller import MPCController
import walking.conf_go2 as conf
import pinocchio as pin
import numpy as np


def main():
    sim = PybulletWrapper()
    robot = Go2(sim)

    # com reference height
    com_h = robot.robot.baseCoMPosition()[2]
    controller = Go2Controller(robot.model, robot.data, com_h)

    # get current state and com
    x0 = robot.get_state()
    x0_com = robot.get_com()

    # Create trajectory planner
    step_length = conf.step_length
    step_height = conf.step_height
    time_step = conf.time_step
    n_per_step = conf.n_per_step

    traj_planner = TrajectoriesPlanner(robot.model, robot.data, step_length,
                                       step_height, time_step, n_per_step)

    if conf.mpc_enabled:
        mpc = MPCController(robot, controller, traj_planner)
        run_mpc_control(sim, robot, mpc)
    else:
        run_opc_control(sim, robot, controller, traj_planner, x0, x0_com)


def run_mpc_control(sim, robot, mpc):
    """
    Run the MPC control loop.
    """

    # Visualization setup
    # Set to True to enable trajectory visualization (warning: may slow down simulation!)
    VISU = True
    visualization_counter = 0

    def visualize_mpc_trajectory(states, color_offset=0):
        """Visualize MPC planned trajectory."""
        if not states or len(states) < 2:
            return

        xs_array = [np.array(x) for x in states]
        model_visu = robot.model.copy()
        data_visu = model_visu.createData()
        joint_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # Different colors for different MPC replans
        base_colors = [
            [1, 0, 0, 0.7],  # Red
            [0, 1, 0, 0.7],  # Green
            [0, 0, 1, 0.7],  # Blue
            [1, 1, 0, 0.7],  # Yellow
        ]

        # Cycle through colors for different replans
        colors = {}
        for i, frame in enumerate(joint_frames):
            color = base_colors[i].copy()
            # Modify color intensity based on replan number
            intensity = max(0.3, 1.0 - (color_offset * 0.2) % 0.7)
            color[0] *= intensity
            color[1] *= intensity
            color[2] *= intensity
            colors[frame] = color

        frame_ids = [model_visu.getFrameId(f) for f in joint_frames]

        # Visualize every 5th point to avoid clutter
        for i in range(0, len(xs_array), 7):
            q = xs_array[i]
            pin.forwardKinematics(model_visu, data_visu, q[:robot.nq])
            pin.updateFramePlacements(model_visu, data_visu)
            for f, fid in zip(joint_frames, frame_ids):
                pos = data_visu.oMf[fid].translation
                # Make spheres smaller for MPC trajectories
                sim.addSphereMarker(pos, radius=0.01, color=colors[f])

    control_counter = 0
    max_steps = 50000  # Safety limit

    for i in range(max_steps):
        # Get current robot state
        robot.update()
        current_state = robot.get_state()

        # MPC step - get control and desired state
        if control_counter == 0:
            try:
                u, x_d = mpc.step(current_state)
                if u is None or x_d is None:
                    print("MPC returned no control, stopping.")
                    break

                # Visualize newly planned trajectory (only when replanned)
                if VISU and mpc.trajectory_updated:
                    visualize_mpc_trajectory(
                        mpc.current_states, visualization_counter)
                    visualization_counter += 1
                    mpc.trajectory_updated = False  # Reset flag

            except Exception as e:
                print(f"MPC step failed: {e}")
                break

        # Apply control
        if u is not None and x_d is not None:
            q = current_state[7:robot.nq]
            v = current_state[robot.nq+6:robot.nq+robot.nv]
            q_d = x_d[7:robot.nq]
            v_d = x_d[robot.nq+6:]
            robot.set_torque(u, q_d, q, v_d, v)

        # Step simulation
        sim.step()
        sim.debug()
        time.sleep(conf.sim_time_step)

        # Update control counter
        control_counter = (control_counter + 1) % conf.control_steps_per_sim

    print("MPC control loop finished.")


def run_opc_control(sim, robot, controller, traj_planner, x0, x0_com):
    """
    Run open loop pre-planned control.
    """
    # get trajectory for N steps
    N = conf.n_steps
    foot_trajectories, com_trajectories = traj_planner.get_N_full_steps(
        x0, N, x0_com)

    VISU = False  # do not set true for large n_per_step (takes much time!)
    if VISU:
        traj_planner.visualize_trajectories(
            sim, foot_trajectories, com_trajectories)

    # ocp
    problem = controller.walking_problem_ocp(
        x0, conf.time_step, foot_trajectories, com_trajectories)

    controls, states = controller.solve(x0, problem)

    def visualize():
        xs_array = [np.array(x) for x in states]
        model_visu = robot.model.copy()
        data_visu = model_visu.createData()
        joint_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        colors = {
            "FL_foot": [1, 0, 0, 1],  # Red
            "FR_foot": [0, 1, 0, 1],  # Green
            "RL_foot": [0, 0, 1, 1],  # Blue
            "RR_foot": [1, 1, 0, 1],  # Yellow
        }

        frame_ids = [model_visu.getFrameId(f) for f in joint_frames]
        for i in range(0, len(xs_array), 7):  # only visualize every 10th point
            q = xs_array[i]
            pin.forwardKinematics(model_visu, data_visu, q[:robot.nq])
            pin.updateFramePlacements(model_visu, data_visu)
            for f, fid in zip(joint_frames, frame_ids):
                pos = data_visu.oMf[fid].translation
                sim.addSphereMarker(pos, radius=0.01, color=colors[f])

    VISU = True
    if VISU:
        visualize()

    control_counter = 0
    current_control_idx = 0
    current_torque = None
    x_d = None

    total_sim_steps = len(controls) * conf.control_steps_per_sim

    for i in range(total_sim_steps):
        if control_counter == 0:
            if current_control_idx < len(controls):
                u = controls[current_control_idx]
                x_d = states[current_control_idx]
                current_torque = u
                current_control_idx += 1
            else:
                print("No more controls available.")
                break

        robot.update()
        x0 = robot.get_state()
        q = x0[7:robot.nq]
        v = x0[robot.nq+6:robot.nq+robot.nv]

        if current_torque is not None and x_d is not None:
            q_d = x_d[7:robot.nq]
            v_d = x_d[robot.nq+6:]
            robot.set_torque(current_torque, q_d, q, v_d, v)

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(conf.sim_time_step)
        control_counter = (control_counter + 1) % conf.control_steps_per_sim


if __name__ == '__main__':
    main()
