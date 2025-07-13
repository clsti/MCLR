import os
import numpy as np
from ament_index_python.packages import get_package_share_directory

################################################################################
# robot
################################################################################

talos_description = get_package_share_directory('go2_description')
urdf = os.path.join(talos_description, "urdf/go2_description.urdf")
path = os.path.join(talos_description, "meshes/../..")


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
