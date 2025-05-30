import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin

# For REEM-C robot
# urdf = "src/reemc_description/robots/reemc.urdf"
# path_meshes = "src/reemc_description/meshes/../.."

# For Talos robot
urdf = "src/talos_description/robots/talos_reduced.urdf"
path_meshes = "src/talos_description/meshes/../.."

'''
Talos
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20, 21  # left arm
22, 23, 24, 25, 26, 27, 28, 29  # right arm
30, 31                          # head

REEMC
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20,     # left arm
21, 22, 23, 24, 25, 26, 27,     # right arm
28, 29                          # head
'''

# Initial condition for the simulator an model
z_init = 1.15
q_actuated_home = np.zeros(32)
q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0])
q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0])
q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Initialization position including floating base
q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

# setup the task stack
modelWrap = pin.RobotWrapper.BuildFromURDF(urdf,                        # Model description
                                           path_meshes,                 # Model geometry descriptors
                                           pin.JointModelFreeFlyer(),   # Floating base model. Use "None" if fixed
                                           True,                        # Printout model details
                                           None)                        # Load meshes different from the descripor
# Get model from wrapper
model = modelWrap.model

# setup the simulator
simulator = PybulletWrapper(sim_rate=1000)

# Create Pybullet-Pinocchio map
robot = Robot(simulator,            # The Pybullet wrapper
              urdf,                 # Robot descriptor
              model,                # Pinocchio model
              [0, 0, z_init],       # Floating base initial position
              [0, 0, 0, 1],            # Floating base initial orientation [x,y,z,w]
              q=q_home,             # Initial state
              useFixedBase=False,   # Fixed base or not
              verbose=True)         # Printout details

# get data
data = robot._model.createData()

# Compute inertia matrix
M = pin.crba(robot._model, data, robot.q())
# Compute non-linear effects
h = pin.nonLinearEffects(robot._model, data, robot.q(), robot.v())

# print complete matrix
np.set_printoptions(
    threshold=np.inf,
    linewidth=200,
    suppress=True,
    formatter={'float_kind': lambda x: f"{x:.1f}"}
)

# Printout Inertia matrix & Non linear effects
print("Inertia matrix:")
print(M)
print("Non linear effects:")
print(h)

# Needed for compatibility
simulator.addLinkDebugFrame(-1, -1)

# Setup pybullet camera
pb.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=90,
    cameraPitch=-20,
    cameraTargetPosition=[0.0, 0.0, 0.8])

# Joint command vector
tau = q_actuated_home*0

# gain matrices
Kx_I = np.diag(np.array([3]*12 + [1]*20))
K_p = 300.0 * Kx_I  # stable: 300 / falling: 200
K_d = 0.4 * Kx_I

# desired joint state
q_d = np.zeros_like(q_actuated_home)

done = False
while not done:
    # update the simulator and the robot
    simulator.step()
    simulator.debug()
    robot.update()

    # mask floating base
    q = robot.q()[7:]
    v = robot.v()[6:]

    # PD controller
    tau = K_p @ (q_d - q) - K_d @ v

    # command to the robot
    robot.setActuatedJointTorques(tau)
