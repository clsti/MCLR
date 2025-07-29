import os
import numpy as np

################################################################################
# robot
################################################################################

go2_description = "/home/devel/miniconda3/envs/walking/share/example-robot-data/robots/go2_description/"
urdf = os.path.join(go2_description, "urdf/go2.urdf")

lfFoot = "FL_foot"
rfFoot = "FR_foot"
lhFoot = "RL_foot"
rhFoot = "RR_foot"

knee = -1.5
hipPitch = 1.57 / 2.0 - 0.2
hipRoll = 0.0

# Stable standing pose: floating base + each leg joint [hipRoll, hipPitch, knee]
# (not used)
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
sim_freq = 500
control_freq = 100
sim_time_step = 1.0 / sim_freq
control_time_step = 1.0 / control_freq
control_steps_per_sim = int(sim_freq / control_freq)

################################################################################
# Step parameters
################################################################################
step_length = 0.20  # [m]
step_height = 0.10  # [m]
n_per_step = 50  # number of time steps per step
time_step = control_time_step  # [s]
n_steps = 7 # number of steps to plan

################################################################################
# MPC parameters
################################################################################
mpc_enabled = True  # Enable MPC controller
mpc_horizon = 1
mpc_replan_frequency = 1.0
mpc_warm_start = True  # Use previous solution as initial guess
