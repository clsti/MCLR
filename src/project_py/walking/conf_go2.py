import os
import numpy as np

################################################################################
# robot
################################################################################

go2_description = "/home/devel/miniconda3/envs/robot_env/share/example-robot-data/robots/go2_description/"
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
