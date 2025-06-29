"""Task2: Linear inverted pendulum Trajectory planning

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12. 

In this case we will solve the trajectory planning over the entire footstep plan
(= horizon) in one go.

Our state will be the position and velocity of the pendulum in the 2d plane.
x = [cx, vx, cy, vy]
And your control the ZMP position in the 2d plane
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

################################################################################
# settings
################################################################################

# Robot Parameters:
# --------------

h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g           = 9.81   # norm of the gravity vector
foot_length = 0.10   # foot size in the x-direction
foot_width  = 0.06   # foot size in the y-direciton

# OCP Parameters:
# --------------
T                     = 0.1                                # fixed sampling time interval of computing the ocp in [s]
STEP_TIME             = 0.8                                # fixed time needed for every foot step [s]

NO_SAMPLES_PER_STEP   = int(round(STEP_TIME/T))            # number of ocp samples per step

NO_STEPS              = 10                                 # total number of foot steps in the plan
TOTAL_NO_SAMPLES      = NO_SAMPLES_PER_STEP*NO_STEPS       # total number of ocp samples over the complete plan (= Horizon)

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                      # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                      # CoM velocity error squared cost weight (= smoothing cost)

################################################################################
# helper function for visualization and dynamics
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of step size = step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).
    
    Args:
        foot_step_0 (_type_): first footstep position (x0, y0)
        step_size_x (_type_): step size in x direction
        no_steps (_type_): number of steps to take
    """

    #>>>>TODO: generate the foot step plan with no_steps
    #>>>>Hint: Check the pdf Fig.3 for inspiration
    foot_steps = None
    return foot_steps


def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to gerneate a colored rectanges.
    Color the left and right steps different and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): the foot step plan
        XY_foot_print (_type_): the dimensions of the foot (x,y)
        ax (_type_): the axis to plot on
    """
    #>>>>TODO: Plot the the footsteps into ax 

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the ZMP
    (We need this for the tracking cost in the cost function of eq. 12)
    Remember: Our goal is to keep the ZMP at the footstep center within each step.
    So for the # of samples a step is active the zmp_ref should be at that step.
    
    Returns a vector of size (TOTAL_NO_SAMPLES, 2)

    Args:
        foot_steps (_type_): the foot step plan
        no_samples_per_step (_type_): number of sampes per step
    """
    #>>>>TODO: Generate the ZMP reference based on given foot_steps
    zmp_ref = None
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): fixed height

    Returns:
        np.array: A, B
    """
    #>>>>TODO: Generate A, B for the continous linear inverted pendulum
    #>>>>Hint: Look at Eq. 4 and rewrite as a system first order diff. eq.
    A=None; B=None
    return A, B

def discrete_LIP_dynamics(delta_t, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        delta_t (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: Generate Ad, Bd for the discretized linear inverted pendulum
    Ad=None; Bd=None
    return Ad, Bd

################################################################################
# setup the plan references and system matrices
################################################################################

# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# footprint
footprint = np.array([foot_length, foot_width])

# generate the footsteps
step_size = 0.2
#>>>>TODO: 1. generate the foot step plan using generate_foot_steps

# zmp reference trajecotry
#>>>>TODO: 2. generate the ZMP reference using generate_zmp_reference

#>>>>Note: At this point you can already start plotting things to see if they
# really make sense!

# discrete LIP dynamics
#>>>>TODO: get the static dynamic matrix Ad, Bd

# continous LIP dynamics
#>>>>TODO: get the static dynamic matrix A, B

################################################################################
# problem definition
################################################################################

# Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

################################################################################
# variables
nx = None #>>>>TODO: State dimension = ?
nu = None #>>>>TODO: control dimension = ?

state = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nx, 'state')
control = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nu, 'control')

# intial state
state_inital = None #>>>>TODO: inital state if based on first footstep (+ zero velo)

# terminal state
state_terminal = None #>>>>TODO: terminal state if based on last footstep (+ zero velo)

################################################################################
# constraints

# 1. intial constraint
#>>>>TODO: Add inital state constrain, Hint: prog.AddConstraint

# 2. terminal constraint
#>>>>TODO: Add terminal state constrain, Hint: prog.AddConstraint

# 3. at each step: respect the LIP descretized dynamics
#>>>>TODO: Enforce the dynamics at every time step

# 4. at each step: keep the ZMP within the foot sole (use the footprint and planned step position)
#>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
#Hint: first compute upper and lower bound based on zmp_ref then add constraints.
#Hint: Add constraints at every time step

################################################################################
# stepwise cost, note that the cost function is scalar!

# setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
#>>>>TODO: add the cost at each timestep, hint: prog.AddCost

################################################################################
# solve

result = Solve(prog)
if not result.is_success:
    print("failure")
print("solved")

# extract the solution
#>>>>TODO: extract your variables from the result object
t = T*np.arange(0, TOTAL_NO_SAMPLES)

# compute the acceleration
#>>>>TODO: compute the acceleration of the COM

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
#>>>>TODO: plot everything in y-axis
#>>>>TODO: plot everything in xy-plane