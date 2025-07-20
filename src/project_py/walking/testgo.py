import pybullet as pb
import numpy as np
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "..")
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin
from walking.go2_test import Go2

import walking.conf_go2 as conf

# For Go2 quadruped robot
urdf = conf.urdf
path_meshes = conf.path
package_dirs = [".."]


'''
Go2 Quadruped Robot Joint Mapping:
The Go2 has 12 actuated joints (3 per leg):

Front Left (FL):  0, 1, 2     # hip, thigh, calf
Front Right (FR): 3, 4, 5     # hip, thigh, calf  
Rear Left (RL):   6, 7, 8     # hip, thigh, calf
Rear Right (RR):  9, 10, 11   # hip, thigh, calf

Joint order in URDF:
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint
'''

# Initial condition for the simulator and model
# Go2 is much smaller than humanoid robots, so lower initial height
z_init = 0.35  # Reduced from 1.15 for quadruped

# Home position for actuated joints (12 joints total)
q_actuated_home = np.zeros(12)

# Set neutral standing pose for quadruped
# Hip joints (abduction/adduction) - slightly spread
q_actuated_home[0] = 0.0   # FL_hip
q_actuated_home[3] = 0.0   # FR_hip  
q_actuated_home[6] = 0.0   # RL_hip
q_actuated_home[9] = 0.0   # RR_hip

# Thigh joints (forward/backward) - slightly forward for stability
q_actuated_home[1] = 0.8   # FL_thigh
q_actuated_home[4] = 0.8   # FR_thigh
q_actuated_home[7] = 0.8   # RL_thigh  
q_actuated_home[10] = 0.8  # RR_thigh

# Calf joints (knee) - bent to support body
q_actuated_home[2] = -1.6   # FL_calf
q_actuated_home[5] = -1.6   # FR_calf
q_actuated_home[8] = -1.6   # RL_calf
q_actuated_home[11] = -1.6  # RR_calf

# Initialization position including floating base
# [x, y, z, roll, pitch, yaw, w] + actuated joints
q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

# Setup the task stack
modelWrap = pin.RobotWrapper.BuildFromURDF(urdf,                        # Model description
                                           package_dirs,                 # Model geometry descriptors
                                           pin.JointModelFreeFlyer(),   # Floating base model. Use "None" if fixed
                                           True,                        # Printout model details
                                           None)                        # Load meshes different from the descriptor

# Get model from wrapper
model = modelWrap.model

# Setup the simulator
simulator = PybulletWrapper(sim_rate=1000)

# Create Pybullet-Pinocchio map
robot = Robot(simulator,            # The Pybullet wrapper
              urdf,                 # Robot descriptor
              model,                # Pinocchio model
              [0, 0, z_init],       # Floating base initial position
              [0, 0, 0, 1],         # Floating base initial orientation [x,y,z,w]
              q=q_home,             # Initial state
              useFixedBase=False,   # Fixed base or not
              verbose=True)         # Printout details



# Needed for compatibility
simulator.addLinkDebugFrame(-1, -1)

# Setup pybullet camera for quadruped viewing
pb.resetDebugVisualizerCamera(
    cameraDistance=1.5,      # Increased distance for better quadruped view
    cameraYaw=45,            # Diagonal view angle
    cameraPitch=-30,         # Look down slightly
    cameraTargetPosition=[0.0, 0.0, 0.2])  # Lower target height

# # Joint command vector (zero torques for now)
# tau = q_actuated_home * 0
go2_controller = Go2(simulator)
go2_controller.robot = robot 

done = False
while not done:
    # Update the simulator and the robot
    go2_controller.step_walking_controller()
    simulator.step()
    simulator.debug()
    robot.update()

    # # Command to the robot (currently zero torques)
    # robot.setActuatedJointTorques(tau)
