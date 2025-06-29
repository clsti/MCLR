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

NO_STEPS = 8         # total number of foot steps
STEP_TIME = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g = 9.81      # norm of the gravity vector
foot_length = 0.10      # foot size in the x-direction
foot_width = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
# sampling time interval of the MPC
T_MPC = 0.1
# number of mpc updates per step
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))

# how many steps in the horizon
NO_STEPS_PER_HORIZON = 2
# duration of future horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME
# number of mpc updates per horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))

# Cost Parameters:
# ---------------
# ZMP error squared cost weight (= tracking cost)
alpha = 10**(-1)
# CoM velocity error squared cost weight (= smoothing cost)
gamma = 10**(-3)

# Simulation Parameters:
# --------------
T_SIM = 0.005                         # 200 Hz simulation time

# NO SIM samples between MPC updates
NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC)
                     )   # Total number of MPC samples
# Total number of Simulator samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))

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
    You can use the function ax.fill() to gerneate a rectable.
    Color left and right steps differt and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): _description_
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
    """generate a function that computes a referecne trajecotry for the zmp.
    Our goal is to keep the ZMP at the footstep center within each step

    Args:
        foot_steps (_type_): _description_
        no_samples_per_step (_type_): _description_
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


def continious_LIP_dynamics():
    """returns the static matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): height

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


def discrete_LIP_dynamics(dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        dt (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    omega = np.sqrt(g/h)
    coshwdT = np.cosh(omega * dt)
    sinhwdT = np.sinh(omega * dt)
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

        # Compute x_dot and use euler integration to approximate
        x_dot = self.A @ self.x + self.B @ u
        # the state at t+dt
        # The disturbance is added in x_dot as self.D@d
        x_dot += self.D @ d
        self.x += x_dot * self.dt

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
        # mpc samples in horizon (nodes)
        self.no_samples = int(round(T_horizon/self.dt))

        self.Ad, self.Bd = discrete_LIP_dynamics(dt)

        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        # ZMP reference over current horizon
        self.ZMP_ref_k = None

    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """ build the MathematicalProgram that solves the mpc problem and
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)

        """
        # variables
        nx = 4  # State dimension
        nu = 2  # control dimension
        prog = MathematicalProgram()

        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')

        # 1. intial constraint
        # Add inital state constraint, Hint: x_k
        for i in range(nx):
            prog.AddLinearConstraint(state[0, i] == x_k[i])

        # 2. at each time step: respect the LIP descretized dynamics
        # Enforce the dynamics at every time step
        for k in range(self.no_samples - 1):
            # x_{k+1} - A_d * x_k - B_d * u_k == 0
            x_next = state[k+1] - self.Ad @ state[k] - self.Bd @ control[k]
            for i in range(nx):
                prog.AddLinearConstraint(x_next[i] == 0)

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        # Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        # Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        # Hint: Add constraints at every time step
        for i in range(self.no_samples - 1):
            # Footstep position
            step_x, step_y = ZMP_ref_k[i]

            # Bounds for this step
            x_lower = step_x - foot_length / 2
            x_upper = step_x + foot_length / 2
            y_lower = step_y - foot_width / 2
            y_upper = step_y + foot_width / 2

            prog.AddBoundingBoxConstraint(
                x_lower, x_upper, control[i, 0])  # ZMP_x in bounds
            prog.AddBoundingBoxConstraint(
                y_lower, y_upper, control[i, 1])  # ZMP_y in bounds

        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        # Add the terminal constraint if requires
        # Hint: If you are unsure, you can start testing without this first!
        if terminal_idx < self.no_samples:
            '''
            for k in range(terminal_idx, self.no_samples):
                prog.AddLinearConstraint(
                    state[k, 0] == ZMP_ref_k[terminal_idx, 0])
                prog.AddLinearConstraint(state[k, 1] == 0.0)
                prog.AddLinearConstraint(
                    state[k, 2] == ZMP_ref_k[terminal_idx, 1])
                prog.AddLinearConstraint(state[k, 3] == 0.0)
            '''
            # Use terminal soft constraints instead of terminal hard constraints to prevent overshooting
            terminal_weight_pos = 0.005
            terminal_weight_vel = 0.001

            terminal_state_ref = np.array([
                ZMP_ref_k[-1, 0],
                0.0,
                ZMP_ref_k[-1, 1],
                0.0
            ])
            # Diagonal weight matrix for terminal cost
            W_terminal = np.diag([terminal_weight_pos, terminal_weight_vel,
                                  terminal_weight_pos, terminal_weight_vel])
            # Terminal cost
            prog.AddQuadraticErrorCost(
                W_terminal, terminal_state_ref, state[-1])

        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        # add the cost at each timestep, hint: prog.AddCost
        for k in range(self.no_samples):
            prog.AddQuadraticErrorCost(
                alpha * np.eye(nu),        # Weight matrix scaled by alpha
                ZMP_ref_k[k],              # Desired control (ZMP reference)
                control[k]                 # Control variable at timestep k
            )

            vx = state[k, 1]
            vy = state[k, 3]

            prog.AddQuadraticCost(gamma * (vx**2 + vy**2))

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
# 1. generate the foot step plan using generate_foot_steps
foot_steps = generate_foot_steps(
    np.array([x_0[0], y_0[0]]), step_size, NO_STEPS)

# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
# 2. generate the complete ZMP reference using generate_zmp_reference
zmp_ref = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES_PER_STEP)

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
        # get current state from the simulator
        x_k = sim.x

        # extract the current horizon from the complete reference trajecotry ZMP_ref
        ZMP_ref_k = zmp_ref[k: k + NO_MPC_SAMPLES_HORIZON]

        # check if we have terminal constraint
        idx_terminal_k = NO_MPC_SAMPLES - k
        # Update the mpc, get new command
        u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, idx_terminal_k)

        k += 1

    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration
    x_ddot_ext = np.array([0, 0])

    # adding a small disturbance
    if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
        x_ddot_ext = np.array([0.0, 1.0])

    # Update the simulation using the current command
    x_k = sim.simulate(u_k, x_ddot_ext)

    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    ZMP_REF_VEC[i] = mpc.ZMP_ref_k[0]

ZMP_LB_VEC = ZMP_REF_VEC - footprint[None, :]
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None, :]

# Use the recodings in STATE_VEC and ZMP_VEC to compute the
# LIP acceleration
# >>>>Hint: Use the continious dynamic matrices
STATE_DOT_VEC = (sim.A @ STATE_VEC.T + sim.B @ ZMP_VEC.T).T

################################################################################
# plot something
fig, axs = plt.subplots(3, 2, figsize=(20, 12))
fig.subplots_adjust(left=0.07, right=0.97, top=0.93,
                    bottom=0.07, hspace=0.4, wspace=0.25)

# ---------------- X Axis ----------------
axs[0, 0].plot(TIME_VEC, STATE_VEC[:, 0], label='CoM X')
axs[0, 0].plot(TIME_VEC, ZMP_VEC[:, 0], label='ZMP X')
axs[0, 0].plot(TIME_VEC, ZMP_REF_VEC[:, 0], label='ZMP Ref X')
axs[0, 0].plot(TIME_VEC, ZMP_LB_VEC[:, 0], 'k--', label='ZMP Lower Bound X')
axs[0, 0].plot(TIME_VEC, ZMP_UB_VEC[:, 0], 'k--', label='ZMP Upper Bound X')
axs[0, 0].set_title("X-Axis: CoM & ZMP")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[1, 0].plot(TIME_VEC, STATE_VEC[:, 1], label='Velocity X', color='orange')
axs[1, 0].set_title("X-Axis: Velocity")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[2, 0].plot(TIME_VEC, STATE_DOT_VEC[:, 1],
               label='Acceleration X', color='purple')
axs[2, 0].set_title("X-Axis: Acceleration")
axs[2, 0].set_xlabel("Time [s]")
axs[2, 0].legend()
axs[2, 0].grid(True)

# ---------------- Y Axis ----------------
axs[0, 1].plot(TIME_VEC, STATE_VEC[:, 2], label='CoM Y')
axs[0, 1].plot(TIME_VEC, ZMP_VEC[:, 1], label='ZMP Y')
axs[0, 1].plot(TIME_VEC, ZMP_REF_VEC[:, 1], label='ZMP Ref Y')
axs[0, 1].plot(TIME_VEC, ZMP_LB_VEC[:, 1], 'k--', label='ZMP Lower Bound Y')
axs[0, 1].plot(TIME_VEC, ZMP_UB_VEC[:, 1], 'k--', label='ZMP Upper Bound Y')
axs[0, 1].set_title("Y-Axis: CoM & ZMP")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 1].plot(TIME_VEC, STATE_VEC[:, 3], label='Velocity Y', color='orange')
axs[1, 1].set_title("Y-Axis: Velocity")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].legend()
axs[1, 1].grid(True)

axs[2, 1].plot(TIME_VEC, STATE_DOT_VEC[:, 3],
               label='Acceleration Y', color='purple')
axs[2, 1].set_title("Y-Axis: Acceleration")
axs[2, 1].set_xlabel("Time [s]")
axs[2, 1].legend()
axs[2, 1].grid(True)


# Define the push interval
start_t = t_push
end_t = t_push + 0.05

# Add vertical span (highlight) to CoM & ZMP plots
axs[0, 0].axvspan(start_t, end_t, color='yellow',
                  alpha=0.5, label='Disturbance')
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].legend(handles, labels)
axs[0, 1].axvspan(start_t, end_t, color='yellow',
                  alpha=0.5, label='Disturbance')
handles, labels = axs[1, 1].get_legend_handles_labels()
axs[1, 1].legend(handles, labels)

# XY-footstep plot
fig, ax = plt.subplots(figsize=(10, 8))
plot_foot_steps(foot_steps, footprint, ax)
ax.plot(STATE_VEC[:, 0], STATE_VEC[:, 2], label='COM trajectory')
ax.plot(ZMP_VEC[:, 0], ZMP_VEC[:, 1], label='ZMP control inputs')
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
