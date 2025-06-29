"""Task2: Linear inverted pendulum MPC

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12 but this time as a model predictive controller (MPC).

In this case we will solve the trajectory planning multiple times over 
a shorter horizon of just 2 steps (receding horizon).
Time between two MPC updates is called T_MPC.

In between MPC updates we simulate the Linear inverted pendulum at a smaller
step time T_SIM, with the lates MPC control ouput u.

Our state & control is the same as before
x = [cx, vx, cy, vy]
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve

import matplotlib.animation as animation

################################################################################
# settings
################################################################################

NO_STEPS                = 8         # total number of foot steps
STEP_TIME               = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h                       = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g                       = 9.81      # norm of the gravity vector
foot_length             = 0.10      # foot size in the x-direction
foot_width              = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
T_MPC                   = 0.1                                               # sampling time interval of the MPC
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))                       # number of mpc updates per step

NO_STEPS_PER_HORIZON  = 2                                                   # how many steps in the horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME                                  # duration of future horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))   # number of mpc updates per horizon

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                  # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                  # CoM velocity error squared cost weight (= smoothing cost)

# Simulation Parameters:
# --------------
T_SIM                   = 0.005                         # 200 Hz simulation time

NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))        # NO SIM samples between MPC updates
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC))   # Total number of MPC samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))   # Total number of Simulator samples

################################################################################
# Helper fnc
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of stepsize=step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).

    Args:
        foot_step_0 (_type_): _description_
        step_size_x (_type_): _description_
        no_steps (_type_): _description_
    """

    #>>>>TODO: copy from previous Task 2
    foot_steps = None
    return foot_steps

def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to gerneate a rectable.
    Color left and right steps differt and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): _description_
    """
    #>>>>TODO: copy from previous Task 2

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the zmp.
    Our goal is to keep the ZMP at the footstep center within each step

    Args:
        foot_steps (_type_): _description_
        no_samples_per_step (_type_): _description_
    """
    #>>>>TODO: copy from previous Task 2
    zmp_ref = None
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics():
    """returns the static matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): height

    Returns:
        np.array: A, B
    """

    #>>>>TODO: copy from previous Task 2
    A=None; B=None
    return A, B

def discrete_LIP_dynamics(dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        dt (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: copy from previous Task 2
    Ad=None; Bd=None
    return Ad, Bd

################################################################################
# Simulation
################################################################################

class Simulator:
    """Simulates the Linear inverted pendulum continous dynamics
    Uses simple euler integration to simulate LIP at sample time dt
    """
    def __init__(self, x_inital, dt):
        self.dt = dt
        self.x = x_inital
        
        self.A, self.B = continious_LIP_dynamics()
        self.D = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
        
    def simulate(self, u, d=np.zeros(2)):
        """updates the LIP state x using based on command u
        
        Optionally: Takes a disturbance acceleration d to simulate effect
        of external pushes on the LIP.
        """

        #>>>>TODO: Compute x_dot and use euler integration to approximate
        # the state at t+dt
        #>>>>TODO: The disturbance is added in x_dot as self.D@d
        
        return self.x    

################################################################################
# MPC
################################################################################

class MPC:
    """MPC for the Linear inverted pendulum
    """
    def __init__(self, dt, T_horizon):
        self.dt = dt                                        # mpc dt
        self.T_horizon = T_horizon                          # time of horizon
        self.no_samples = int(round(T_horizon/self.dt))     # mpc samples in horizon (nodes)

        self.Ad, self.Bd = discrete_LIP_dynamics(dt)
        
        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        self.ZMP_ref_k = None                               # ZMP reference over current horizon
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """ build the MathematicalProgram that solves the mpc problem and 
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)
            
        """
        
        # variables
        nx = None #>>>>TODO: State dimension = ?
        nu = None #>>>>TODO: control dimension = ?
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
    
        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step
        
        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step
    
        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        #Hint: If you are unsure, you can start testing without this first!
    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
            
        # solve
        result = Solve(prog)
        if not result.is_success:
            print("failure")
            
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0]
    
################################################################################
# run the simulation
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
foot_steps=None

# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
#>>>>TODO: 2. generate the complete ZMP reference using generate_zmp_reference

# generate mpc
mpc = MPC(T_MPC, T_HORIZON)

# generate the pendulum simulator
state_0 = np.concatenate([x_0, y_0])
sim = Simulator(state_0, T_SIM)

# setup some vectors for plotting stuff
TIME_VEC = np.nan*np.ones(NO_SIM_SAMPLES)
STATE_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 4])
ZMP_REF_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])
ZMP_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])

# time to add some disturbance
t_push = 3.2

# execution loop

k = 0   # the number of mpc update
for i in range(NO_SIM_SAMPLES):
    
    # simulation time
    t = i*T_SIM
        
    if i % NO_SIM_SAMPLES_PER_MPC == 0:
        # time to update the mpc
        
        # current state
        #>>>>TODO: get current state from the simulator
        x_k = None
    
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        ZMP_ref_k = None
    
        # check if we have terminal constraint
        idx_terminal_k = NO_MPC_SAMPLES - k
        #>>>>TODO: Update the mpc, get new command
        u_k = None
        
        k += 1
    
    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration 
    x_ddot_ext = np.array([0, 0])
    
    #>>>>TODO: when you got everything working try adding a small disturbance
    # if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
    #    x_ddot_ext = np.array([0, 1.0])
    
    #>>>>TODO: Update the simulation using the current command
    x_k = None
    
    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    ZMP_REF_VEC[i] = mpc.ZMP_ref_k[0]
    
ZMP_LB_VEC = ZMP_REF_VEC - footprint[None,:]
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None,:]

#>>>>TODO: Use the recodings in STATE_VEC and ZMP_VEC to compute the 
# LIP acceleration
#>>>>Hint: Use the continious dynamic matrices
STATE_DOT_VEC = None

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
#>>>>TODO: plot everything in y-axis
#>>>>TODO: plot everything in xy-plane
