import os
import numpy as np
from ament_index_python.packages import get_package_share_directory

################################################################################
# robot
################################################################################

talos_description = get_package_share_directory('go2_description')
urdf = os.path.join(talos_description, "urdf/go2_description.urdf")
path = os.path.join(talos_description, "meshes/../..")


q_stable = np.array([
    # ───────────── floating base ─────────────
    # x, y, z  (≈ 31 cm height – tweak ±2 cm if the feet hover or sink)
    0.00, 0.00, 0.47,
    0.00, 0.00, 0.00, 1.00,  # qx, qy, qz, qw  (identity orientation)
    # ───── front‑right (FR) leg ─────
    0.00,   0.00,   0.00,    # hipRoll, hipPitch, knee
    # ───── front‑left (FL)  leg ─────
    0.00,   0.00,   0.00,
    # ───── rear‑right (RR)  leg ─────
    0.00,   0.00,   0.00,
    # ───── rear‑left (RL)   leg ─────
    0.00,   0.00,   0.00
])
