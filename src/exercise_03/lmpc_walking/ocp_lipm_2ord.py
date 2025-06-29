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
plt.style.use('seaborn-v0_8-dark')

################################################################################
# settings
################################################################################

# Robot Parameters:
# --------------

h = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g = 9.81   # norm of the gravity vector
foot_length = 0.10   # foot size in the x-direction
foot_width = 0.06   # foot size in the y-direciton

# OCP Parameters:
# --------------
# fixed sampling time interval of computing the ocp in [s]
T = 0.1
# fixed time needed for every foot step [s]
STEP_TIME = 0.8

# number of ocp samples per step
NO_SAMPLES_PER_STEP = int(round(STEP_TIME/T))

# total number of foot steps in the plan
NO_STEPS = 10
# total number of ocp samples over the complete plan (= Horizon)
TOTAL_NO_SAMPLES = NO_SAMPLES_PER_STEP*NO_STEPS

# Cost Parameters:
# ---------------
# ZMP error squared cost weight (= tracking cost)
alpha = 10**(-1)
# CoM velocity error squared cost weight (= smoothing cost)
gamma = 10**(-3)

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
    foot_steps = np.zeros((no_steps, 2))
    for i in range(no_steps):
        if i == 0:
            foot_steps[i] = foot_step_0
        else:
            foot_steps[i, 0] = foot_steps[i-1, 0] + step_size_x
            foot_steps[i, 1] = -foot_steps[i-1, 1]
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
    # Compute corners
    foot_length, foot_width = XY_foot_print
    dx = foot_length / 2
    dy = foot_width / 2
    corners = np.array([
        [-dx, -dy],
        [-dx,  dy],
        [dx,  dy],
        [dx, -dy]
    ])

    for i, step in enumerate(foot_steps):

        translated_corners = corners + step

        # Choose color
        color = 'red' if (i % 2 == 0) else 'green'

        ax.fill(translated_corners[:, 0], translated_corners[:, 1],
                color, alpha=0.5, edgecolor='black')

    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Planned Foot Steps")


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
    zmp_ref_list = []

    for (x, y) in foot_steps:
        zmp_step = np.tile([x, y], (no_samples_per_step, 1))
        zmp_ref_list.append(zmp_step)

    zmp_ref = np.vstack(zmp_ref_list)
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
    g_h = g/h

    # State: [c_x, c_x_dot, c_y, c_y_dot]
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [g_h, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g_h, 0.0]
    ])

    # Control: [p_x, p_y]
    B = np.array([
        [0.0, 0.0],
        [-g_h, 0.0],
        [0.0, 0.0],
        [0.0, -g_h]
    ])
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
    omega = np.sqrt(g/h)
    coshwdT = np.cosh(omega * delta_t)
    sinhwdT = np.sinh(omega * delta_t)
    A_1d = np.array([
        [coshwdT, 1.0/omega * sinhwdT],
        [omega * sinhwdT, coshwdT]
    ])
    B_1d = np.array([
        [1.0 - coshwdT],
        [-omega * sinhwdT]
    ])

    # State: [c_x, c_x_dot, c_y, c_y_dot]
    Ad = np.block([
        [A_1d, np.zeros((2, 2))],
        [np.zeros((2, 2)), A_1d]
    ])

    # Control: [p_x, p_y]
    Bd = np.block([
        [B_1d, np.zeros((2, 1))],
        [np.zeros((2, 1)), B_1d]
    ])
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
# 1. generate the foot step plan using generate_foot_steps
foot_steps = generate_foot_steps(
    np.array([x_0[0], y_0[0]]), step_size, NO_STEPS)

# zmp reference trajecotry
# 2. generate the ZMP reference using generate_zmp_reference
zmp_ref = generate_zmp_reference(foot_steps, NO_SAMPLES_PER_STEP)


# discrete LIP dynamics
# get the static dynamic matrix Ad, Bd
Ad, Bd = discrete_LIP_dynamics(T, g, h)

# continous LIP dynamics
# get the static dynamic matrix A, B
A, B = continious_LIP_dynamics(g, h)

################################################################################
# problem definition
################################################################################

# Define an instance of MathematicalProgram
prog = MathematicalProgram()

################################################################################
# variables
nx = 4  # State dimension
nu = 2  # control dimension

state = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nx, 'state')
control = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nu, 'control')

# intial state
# inital state if based on first footstep (+ zero velo)
state_inital = np.hstack([x_0, y_0])

# terminal state
# terminal state if based on last footstep (+ zero velo)
state_terminal = np.array([foot_steps[-1, 0], 0.0, foot_steps[-1, 1], 0.0])

################################################################################
# constraints

# 1. intial constraint
# Add inital state constrain, Hint: prog.AddConstraint
for i in range(nx):
    prog.AddLinearConstraint(state[0, i] == state_inital[i])

# 2. terminal constraint
# Add terminal state constrain, Hint: prog.AddConstraint
for i in range(nx):
    prog.AddConstraint(state[-1, i] == state_terminal[i])

# 3. at each step: respect the LIP descretized dynamics
# Enforce the dynamics at every time step
for k in range(TOTAL_NO_SAMPLES - 1):
    # x_{k+1} - A_d * x_k - B_d * u_k == 0
    x_next = state[k+1] - Ad @ state[k] - Bd @ control[k]
    for i in range(nx):
        prog.AddLinearConstraint(x_next[i] == 0)

# 4. at each step: keep the ZMP within the foot sole (use the footprint and planned step position)
# Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
# Hint: first compute upper and lower bound based on zmp_ref then add constraints.
# Hint: Add constraints at every time step
for i in range(NO_STEPS):
    # Footstep position
    step_x, step_y = foot_steps[i]

    # Bounds for this step
    x_lower = step_x - foot_length / 2
    x_upper = step_x + foot_length / 2
    y_lower = step_y - foot_width / 2
    y_upper = step_y + foot_width / 2

    # Step sampling
    start_idx = i * NO_SAMPLES_PER_STEP
    end_idx = (i + 1) * NO_SAMPLES_PER_STEP

    for k in range(start_idx, end_idx):
        prog.AddBoundingBoxConstraint(
            x_lower, x_upper, control[k, 0])  # ZMP_x in bounds
        prog.AddBoundingBoxConstraint(
            y_lower, y_upper, control[k, 1])  # ZMP_y in bounds


################################################################################
# stepwise cost, note that the cost function is scalar!

# setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
# add the cost at each timestep, hint: prog.AddCost
for k in range(TOTAL_NO_SAMPLES):
    prog.AddQuadraticErrorCost(
        alpha * np.eye(nu),        # Weight matrix scaled by alpha
        zmp_ref[k],                # Desired control (ZMP reference)
        control[k]                 # Control variable at timestep k
    )

    vx = state[k, 1]
    vy = state[k, 3]

    prog.AddQuadraticCost(gamma * (vx**2 + vy**2))

################################################################################
# solve

result = Solve(prog)
if not result.is_success:
    print("failure")
print("solved")

# extract the solution
# extract your variables from the result object
t = T*np.arange(0, TOTAL_NO_SAMPLES)
state_sol = result.GetSolution(state)
control_sol = result.GetSolution(control)

# compute the acceleration
# compute the acceleration of the COM
acceleration = np.zeros((TOTAL_NO_SAMPLES, 2))
for k in range(TOTAL_NO_SAMPLES):
    xk = state_sol[k, :]
    uk = control_sol[k, :]
    x_dot = A @ xk + B @ uk
    # acceleration = derivatives of velocities
    acceleration[k, 0] = x_dot[1]  # ax
    acceleration[k, 1] = x_dot[3]  # ay

################################################################################
# plot something
fig, axs = plt.subplots(3, 2, figsize=(20, 12))
fig.subplots_adjust(left=0.07, right=0.97, top=0.93,
                    bottom=0.07, hspace=0.4, wspace=0.25)

# --- ZMP bounds X ---
zmp_x_lower = np.zeros(TOTAL_NO_SAMPLES)
zmp_x_upper = np.zeros(TOTAL_NO_SAMPLES)
zmp_y_lower = np.zeros(TOTAL_NO_SAMPLES)
zmp_y_upper = np.zeros(TOTAL_NO_SAMPLES)

for i in range(NO_STEPS):
    x, y = foot_steps[i]
    start = i * NO_SAMPLES_PER_STEP
    end = (i + 1) * NO_SAMPLES_PER_STEP
    zmp_x_lower[start:end] = x - foot_length / 2
    zmp_x_upper[start:end] = x + foot_length / 2
    zmp_y_lower[start:end] = y - foot_width / 2
    zmp_y_upper[start:end] = y + foot_width / 2

# ---------------- X Axis ----------------
axs[0, 0].plot(t, state_sol[:, 0], label='CoM X')
axs[0, 0].plot(t, control_sol[:, 0], label='ZMP X')
axs[0, 0].plot(t, zmp_ref[:, 0], label='ZMP Ref X')
axs[0, 0].plot(t, zmp_x_lower, 'k--', label='ZMP Lower Bound X')
axs[0, 0].plot(t, zmp_x_upper, 'k--', label='ZMP Upper Bound X')
axs[0, 0].set_title("X-Axis: CoM & ZMP")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[1, 0].plot(t, state_sol[:, 1], label='Velocity X', color='orange')
axs[1, 0].set_title("X-Axis: Velocity")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[2, 0].plot(t, acceleration[:, 0], label='Acceleration X', color='purple')
axs[2, 0].set_title("X-Axis: Acceleration")
axs[2, 0].set_xlabel("Time [s]")
axs[2, 0].legend()
axs[2, 0].grid(True)

# ---------------- Y Axis ----------------
axs[0, 1].plot(t, state_sol[:, 2], label='CoM Y')
axs[0, 1].plot(t, control_sol[:, 1], label='ZMP Y')
axs[0, 1].plot(t, zmp_ref[:, 1], label='ZMP Ref Y')
axs[0, 1].plot(t, zmp_y_lower, 'k--', label='ZMP Lower Bound Y')
axs[0, 1].plot(t, zmp_y_upper, 'k--', label='ZMP Upper Bound Y')
axs[0, 1].set_title("Y-Axis: CoM & ZMP")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 1].plot(t, state_sol[:, 3], label='Velocity Y', color='orange')
axs[1, 1].set_title("Y-Axis: Velocity")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].legend()
axs[1, 1].grid(True)

axs[2, 1].plot(t, acceleration[:, 1], label='Acceleration Y', color='purple')
axs[2, 1].set_title("Y-Axis: Acceleration")
axs[2, 1].set_xlabel("Time [s]")
axs[2, 1].legend()
axs[2, 1].grid(True)


# XY-footstep plot
fig, ax = plt.subplots(figsize=(10, 8))
plot_foot_steps(foot_steps, footprint, ax)
ax.plot(state_sol[:, 0], state_sol[:, 2], label='COM trajectory')
ax.plot(control_sol[:, 0], control_sol[:, 1], label='ZMP control inputs')
ax.scatter(foot_steps[:, 0], foot_steps[:, 1],
           color='red', marker='x', label='Foot steps')
ax.set_title('COM and ZMP trajectory in XY plane')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.legend()
ax.grid(True)
ax.axis('equal')

plt.grid(True)
plt.tight_layout()
plt.show()
