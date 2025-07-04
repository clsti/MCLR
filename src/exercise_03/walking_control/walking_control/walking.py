"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

# simulator
from simulator.pybullet_wrapper import PybulletWrapper
from walking_control.talos import Talos

# robot configs
import walking_control.talos_conf as conf

# modules
# Import all previously coded modules
from walking_control.foot_trajectory import SwingFootTrajectory
from walking_control.footstep_planner import FootStepPlanner
from walking_control.footstep_planner import Side
from walking_control.lip_mpc import LIPInterpolator
from walking_control.lip_mpc import LIPMPC
from walking_control.lip_mpc import generate_zmp_reference

################################################################################
# main
################################################################################


def main(args=None):

    ############################################################################
    # setup
    ############################################################################

    # Setup ros
    rclpy.init(args=args)

    # setup the simulator
    sim = PybulletWrapper(sim_rate=conf.f_cntr)

    # setup the robot
    robot = Talos(sim)

    # inital footsteps
    # Set intial swing foot pose to left foot
    T_swing_w = robot.stack.get_placement_LF()
    # Set intial support foot pose to right foot
    T_support_w = robot.stack.get_placement_RF()

    # initial footsteps for logging
    LF_pos_ref = T_swing_w.translation
    RF_pos_ref = T_support_w.translation

    # setup the plan with 20 steps
    no_steps = 20
    planner = FootStepPlanner(conf)  # Create the planner
    plan = planner.planLine(T_swing_w, Side.LEFT, no_steps)  # Create the plan
    # Append the two last steps once more to the plan so our mpc horizon will never run out
    plan.append(plan[-1])
    plan.append(plan[-1])

    # generate reference
    ZMP_ref = generate_zmp_reference(
        plan, conf.no_mpc_samples_per_step)  # Generate the mpc reference
    # Plot the plan (make sure this workes first)
    planner.plot(sim)
    for zmp in ZMP_ref:
        pos = [zmp[0], zmp[1], 0.01]
        sim.addSphereMarker(pos)

    # setup the lip models
    mpc = LIPMPC(conf)  # Setup mpc

    # Assume the com is over the first support foot
    # Build the intial mpc state vector
    x0 = np.array([
        T_support_w.translation[0],
        0.0,
        T_support_w.translation[1],
        0.0
    ])
    # Create the interpolator and set the inital state
    interpolator = LIPInterpolator(x0, conf)

    # set the com task reference to the inital support foot
    com_rf = np.array(
        [T_support_w.translation[0], T_support_w.translation[1], conf.h])
    # Set the COM reference to be over supporting foot
    robot.stack.setComRefState(com_rf)

    ############################################################################
    # logging
    ############################################################################

    pre_dur = 3.0   # Time to wait before walking should start

    # Compute number of iterations:
    N_pre = int(pre_dur/conf.dt)  # Number of sim steps before walking starts
    # Total number of sim steps during walking
    N_sim = no_steps * conf.no_sim_per_step
    # Total number of mpc steps during walking
    N_mpc = no_steps * conf.no_mpc_samples_per_step

    # Create vectors to log all the data of the simulation
    # - COM_POS, COM_VEL, COM_ACC (from the planned reference, pinocchio and pybullet)
    # - Angular momentum (from pinocchio)
    # - Left and right foot POS, VEL, ACC (from planned reference, pinocchio)
    # - ZMP (from planned reference, from estimator )
    # - DCM (from estimtor)
    # - Normal forces in right and left foot (from pybullet ft sensors, from pinocchio)
    TIME = np.nan*np.empty(N_sim)
    COM_POS_ref = np.nan * np.empty((N_sim, 3))
    COM_VEL_ref = np.nan * np.empty((N_sim, 3))
    COM_ACC_ref = np.nan * np.empty((N_sim, 3))
    COM_POS_pin = np.nan * np.empty((N_sim, 3))
    COM_VEL_pin = np.nan * np.empty((N_sim, 3))
    COM_ACC_pin = np.nan * np.empty((N_sim, 3))
    COM_POS_pb = np.nan * np.empty((N_sim, 3))
    COM_VEL_pb = np.nan * np.empty((N_sim, 3))
    COM_ACC_pb = np.nan * np.empty((N_sim, 3))

    ANGULAR_MOMENTUM = np.nan * np.empty((N_sim, 3))

    LEFT_FOOT_POS_ref = np.nan * np.empty((N_sim, 3))
    LEFT_FOOT_VEL_ref = np.nan * np.empty((N_sim, 3))
    LEFT_FOOT_ACC_ref = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_POS_ref = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_VEL_ref = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_ACC_ref = np.nan * np.empty((N_sim, 3))
    LEFT_FOOT_POS_pin = np.nan * np.empty((N_sim, 3))
    LEFT_FOOT_VEL_pin = np.nan * np.empty((N_sim, 3))
    LEFT_FOOT_ACC_pin = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_POS_pin = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_VEL_pin = np.nan * np.empty((N_sim, 3))
    RIGHT_FOOT_ACC_pin = np.nan * np.empty((N_sim, 3))

    ZMP_REF = np.nan * np.empty((N_sim, 3))
    ZMP_EST = np.nan * np.empty((N_sim, 3))
    DCM = np.nan * np.empty((N_sim, 3))

    NORMAL_FORCE_RIGHT_pb = np.nan * np.empty((N_sim, 3))
    NORMAL_FORCE_LEFT_pb = np.nan * np.empty((N_sim, 3))
    NORMAL_FORCE_RIGHT_pin = np.nan * np.empty((N_sim, 3))
    NORMAL_FORCE_LEFT_pin = np.nan * np.empty((N_sim, 3))

    ############################################################################
    # logging
    ############################################################################

    k = 0                                               # current MPC index
    # current index of the step within foot step plan
    plan_idx = 1
    # elapsed time within current step (use to evaluate spline)
    t_step_elapsed = 0.0
    # last publish time (last time we published something)
    t_publish = 0.0

    for i in range(-N_pre, N_sim):
        t = sim.simTime()  # Simulator time
        dt = sim.stepTime()  # Simulator dt

        ########################################################################
        # update the mpc very no_sim_per_mpc steps
        ########################################################################

        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            # MPC update
            # Get current LIP state
            c = interpolator.x
            # Extract the ZMP reference
            ZMP_ref_k = ZMP_ref[k: k + conf.no_mpc_samples_per_horizon]
            # get terminal index
            idx_terminal_k = (no_steps - 1) * conf.no_mpc_samples_per_step - k
            # Solve mpc
            u_k = mpc.buildSolveOCP(c, ZMP_ref_k, idx_terminal_k)
            k += 1

        ########################################################################
        # update the foot spline
        ########################################################################

        if i >= 0 and i % conf.no_sim_per_step == 0:
            # Start next step

            # Get the current location of the swing foot
            step_curr = plan[plan_idx - 1]
            sw_foot_loc_curr = step_curr.poseInWorld()

            # Get next step location for swing foot
            step_next = plan[plan_idx + 1]
            sw_foot_loc_next = step_next.poseInWorld()

            # Set the swing foot of the robot
            robot.setSwingFoot(step_next.side)

            # Set the support foot for the robot
            support_foot = Side.LEFT if step_next.side == Side.RIGHT else Side.RIGHT
            robot.setSupportFoot(support_foot)

            # Plan a foot trajectory between current and next foot pose
            foot_traj = SwingFootTrajectory(
                sw_foot_loc_curr, sw_foot_loc_next, conf.step_dur)

            t_step_elapsed = 0.0
            plan_idx += 1

        ########################################################################
        # in every iteration when walking
        ########################################################################

        if i >= 0:
            # Update foot trajectory with current step time
            traj_pos, traj_vel, traj_acc = foot_traj.evaluate(t_step_elapsed)
            robot.updateSwingFootRef(traj_pos, traj_vel, traj_acc)
            if step_next.side == Side.LEFT:
                LF_pos_ref = traj_pos.translation
                LF_vel_ref = traj_vel
                LF_acc_ref = traj_acc
                RF_vel_ref = np.array([0.0, 0.0, 0.0])
                RF_acc_ref = np.array([0.0, 0.0, 0.0])
            else:
                RF_pos_ref = traj_pos.translation
                RF_vel_ref = traj_vel
                RF_acc_ref = traj_acc
                LF_vel_ref = np.array([0.0, 0.0, 0.0])
                LF_acc_ref = np.array([0.0, 0.0, 0.0])

            # Update the interpolator with the latest command u_k
            interpolator.integrate(u_k)

            # Feed the com tasks with the new com reference
            com_pos, com_vel, com_acc = interpolator.comState()
            robot.stack.setComRefState(com_pos, com_vel, com_acc)

            # Increment elapsed footstep time
            t_step_elapsed += dt

        ########################################################################
        # update the simulation
        ########################################################################

        # Update the simulator and the robot
        sim.step()
        sim.debug()
        robot.update()

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            # Publish
            robot.publish()

        # store for visualizations
        if i >= 0:
            TIME[i] = t
            # Log information
            # TODO: reference from interpolator?
            COM_POS_ref[i, :] = interpolator.comState()[0]
            COM_VEL_ref[i, :] = interpolator.comState()[1]
            COM_ACC_ref[i, :] = interpolator.comState()[2]
            COM_POS_pin[i, :] = robot.stack.comState().value()
            COM_VEL_pin[i, :] = robot.stack.comState().derivative()
            COM_ACC_pin[i, :] = robot.stack.comState().second_derivative()
            COM_POS_pb[i, :] = robot.robot.baseCoMPosition()
            COM_VEL_pb[i, :] = robot.robot.baseCoMVelocity()
            # TODO: Central difference method to get acceleration
            '''
            if len(COM_VEL_pb) >= 3:
                gt_acc = (self.plot_gt_vel[-1] - self.plot_gt_vel[-3]
                        ) / (self.plot_time[-1] - self.plot_time[-3])
            else:
                gt_acc = np.zeros_like(gt_vel)
            COM_ACC_pb[i, :] = 
            '''

            ANGULAR_MOMENTUM[i, :] = robot.stack.get_angular_momentum()

            LEFT_FOOT_POS_ref[i, :] = LF_pos_ref
            LEFT_FOOT_VEL_ref[i, :] = LF_vel_ref
            LEFT_FOOT_ACC_ref[i, :] = LF_acc_ref
            RIGHT_FOOT_POS_ref[i, :] = RF_pos_ref
            RIGHT_FOOT_VEL_ref[i, :] = RF_vel_ref
            RIGHT_FOOT_ACC_ref[i, :] = RF_acc_ref
            LF_3d_pos, LF_3d_vel, LF_3d_acc = robot.stack.get_LF_3d_pos_vel_acc(
                robot.dv)
            LEFT_FOOT_POS_pin[i, :] = LF_3d_pos
            LEFT_FOOT_VEL_pin[i, :] = LF_3d_vel
            LEFT_FOOT_ACC_pin[i, :] = LF_3d_acc
            RF_3d_pos, RF_3d_vel, RF_3d_acc = robot.stack.get_RF_3d_pos_vel_acc(
                robot.dv)
            RIGHT_FOOT_POS_pin[i, :] = RF_3d_pos
            RIGHT_FOOT_VEL_pin[i, :] = RF_3d_vel
            RIGHT_FOOT_ACC_pin[i, :] = RF_3d_acc

            ZMP_REF[i, :] = interpolator.zmp()
            ZMP_EST[i, :] = np.array([ZMP_ref_k[0][0], ZMP_ref_k[0][1], 0.0])
            DCM[i, :] = interpolator.dcm()

            NORMAL_FORCE_RIGHT_pb[i, :] = robot._get_ankle_wrenches()[0].linear
            NORMAL_FORCE_LEFT_pb[i, :] = robot._get_ankle_wrenches()[1].linear
            '''
            NORMAL_FORCE_RIGHT_pin[i, :] = robot.stack.get_RF_normal_force(
                robot.dv)
            NORMAL_FORCE_LEFT_pin[i, :] = robot.stack.get_LF_normal_force(
                robot.dv)
            '''
            # TODO
            # NORMAL_FORCE_RIGHT_pin[i, :] = robot.stack.get_RF_wrench(robot.dv)
            # NORMAL_FORCE_LEFT_pin[i, :] = robot.stack.get_LF_wrench(robot.dv)

    ########################################################################
    # enough with the simulation, lets plot
    ########################################################################

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-dark')

    # Plot everything
    def plot_3x3(data_refs, labels, title_prefix):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        components = ['x', 'y', 'z']
        types = ['Position', 'Velocity', 'Acceleration']
        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                for data, label in zip(data_refs[col], labels):
                    ax.plot(TIME, data[:, row], label=label)
                ax.set_ylabel(components[row])
                ax.set_title(f"{title_prefix} {types[col]}")
                ax.legend()
        plt.tight_layout()

    def plot_3x1(data_refs, labels, title_prefix):
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        components = ['x', 'y', 'z']
        for i in range(3):
            ax = axes[i]
            for data, label in zip(data_refs, labels):
                ax.plot(TIME, data[:, i], label=label)
            ax.set_ylabel(components[i])
            ax.set_title(f"{title_prefix} - {components[i]}")
            ax.legend()
        plt.tight_layout()

    # === COM plots ===
    plot_3x3(
        data_refs=[
            [COM_POS_ref, COM_POS_pin, COM_POS_pb],
            [COM_VEL_ref, COM_VEL_pin, COM_VEL_pb],
            [COM_ACC_ref, COM_ACC_pin],  # TODO: COM_ACC_pb
        ],
        labels=['ref', 'pinocchio', 'pybullet'],
        title_prefix='COM'
    )

    # === Foot plots ===
    plot_3x3(
        data_refs=[
            [LEFT_FOOT_POS_ref, LEFT_FOOT_POS_pin,
                RIGHT_FOOT_POS_ref, RIGHT_FOOT_POS_pin],
            [LEFT_FOOT_VEL_ref, LEFT_FOOT_VEL_pin,
                RIGHT_FOOT_VEL_ref, RIGHT_FOOT_VEL_pin],
            [LEFT_FOOT_ACC_ref, LEFT_FOOT_ACC_pin,
                RIGHT_FOOT_ACC_ref, RIGHT_FOOT_ACC_pin],
        ],
        labels=['LF_ref', 'LF_pin', 'RF_ref', 'RF_pin'],
        title_prefix='Feet'
    )

    # === ZMP, DCM & Angular Momentum ===
    plot_3x3(
        data_columns=[
            [ZMP_REF, ZMP_EST],
            [DCM],
            [ANGULAR_MOMENTUM],
        ],
        column_labels=['ZMP', 'DCM', 'Angular Momentum'],
        series_labels=[
            ['ZMP ref', 'ZMP est'],
            ['DCM'],
            ['Angular Momentum'],
        ],
        title_prefix='ZMP, DCM, and Angular Momentum'
    )

    # === Forces (normal) ===
    plot_3x1(
        data_refs=[
            np.stack([NORMAL_FORCE_LEFT_pb, NORMAL_FORCE_LEFT_pb,
                     NORMAL_FORCE_LEFT_pb], axis=1),
            np.stack([NORMAL_FORCE_RIGHT_pb, NORMAL_FORCE_RIGHT_pb,
                     NORMAL_FORCE_RIGHT_pb], axis=1),
            np.stack([NORMAL_FORCE_LEFT_pin, NORMAL_FORCE_LEFT_pin,
                     NORMAL_FORCE_LEFT_pin], axis=1),
            np.stack([NORMAL_FORCE_RIGHT_pin, NORMAL_FORCE_RIGHT_pin,
                     NORMAL_FORCE_RIGHT_pin], axis=1)
        ],
        labels=['LF_pb', 'RF_pb', 'LF_pin', 'RF_pin'],
        title_prefix='Normal Forces'
    )

    plt.show()


if __name__ == '__main__':
    rclpy.init()
    main()
