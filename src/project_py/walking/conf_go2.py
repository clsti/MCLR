import os
import numpy as np

################################################################################
# robot
################################################################################

go2_description = "/home/devel/miniconda3/envs/walking/share/example-robot-data/robots/go2_description/"
urdf = os.path.join(go2_description, "urdf/go2.urdf")
# path = os.path.join(go2_description, "meshes")

lfFoot = "FL_foot"
rfFoot = "FR_foot"
lhFoot = "RL_foot"
rhFoot = "RR_foot"

knee = -1.5
hipPitch = 1.57 / 2.0 - 0.2
hipRoll = 0.0

q_stable = np.array([
    # ───────────── floating base ─────────────
    0.00, 0.00, 0.36,
    0.00, 0.00, 0.00, 1.00,  # qx, qy, qz, qw  (identity orientation)
    # ───── front‑left (FL) leg ─────
    hipRoll,   hipPitch,   knee,    # hipRoll, hipPitch, knee
    # ───── front‑right (FR)  leg ─────
    hipRoll,   hipPitch,   knee,
    # ───── rear‑left (RL)  leg ─────
    hipRoll,   hipPitch,   knee,
    # ───── rear‑right (RR)   leg ─────
    hipRoll,   hipPitch,   knee

])

    ## Joint numbers
    # 0: hipRoll, 1: hipPitch, 2: knee
    # joint_numbers = {
    #     "FL_hipRoll": 0, "FL_hipPitch": 1, "FL_knee": 2,
    #     "FR_hipRoll": 3, "FR_hipPitch": 4, "FR_knee": 5,
    #     "RL_hipRoll": 6, "RL_hipPitch": 7, "RL_knee": 8,
    #     "RR_hipRoll": 9, "RR_hipPitch": 10, "RR_knee": 11
    # }

################################################################################
# PD gains 
################################################################################
hip_Kp = 80.0
hip_Kd = 3.0

thigh_Kp = 100.0
thigh_Kd = 4.0

calf_Kp = 120.0
calf_Kd = 5.0

Kp = np.array([
    hip_Kp, thigh_Kp, calf_Kp,    # FL leg
    hip_Kp, thigh_Kp, calf_Kp,    # FR leg  
    hip_Kp, thigh_Kp, calf_Kp,    # RL leg
    hip_Kp, thigh_Kp, calf_Kp     # RR leg
])

Kd = np.array([
    hip_Kd, thigh_Kd, calf_Kd,    # FL leg
    hip_Kd, thigh_Kd, calf_Kd,    # FR leg
    hip_Kd, thigh_Kd, calf_Kd,    # RL leg  
    hip_Kd, thigh_Kd, calf_Kd     # RR leg
])


################################################################################
# simulation frequency
################################################################################
sim_freq = 200
control_freq = 100
sim_time_step = 1.0 / sim_freq
control_time_step = 1.0 / control_freq
control_steps_per_sim = int(sim_freq / control_freq) 

################################################################################
# Step parameters
################################################################################
step_length = 0.25  # [m]
step_height = 0.10  # [m]
n_per_step = 50  # number of time steps per step
time_step = control_time_step  # [s]
n_steps = 6  # number of steps to plan

################################################################################
# MPC parameters
################################################################################
mpc_enabled = False  # Enable MPC controller
mpc_horizon = 2  
mpc_replan_frequency = 1.0  
mpc_warm_start = False  # Use previous solution as initial guess
