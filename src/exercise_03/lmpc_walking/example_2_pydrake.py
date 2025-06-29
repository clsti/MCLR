"""
Example 2

The goal of this example is to use trajectory optimization on the standard
pendulum swingup problem (= getting the pendulum into an upward position)

This will show you how to use one of pydrakes solvers to find the trajectory x[]
through a system's statespace and the coresponding control signals u[]
over a horizon of N steps (also called nodes)

The problem (like always) is composed of a cost funciton C(x[],u[]) and a
series of constraints that have to be fullfiled (eigher at each step 
or at the start/end = intial/terminal constraint)

min_{x[], u[]} sum_k C_k(x_k,u_k) 
    s.t. ...

Here we will use the pendulum model as an example
State:      x = [q, q_dot] \in R^2   (angle and angle velocity)
Control:    u = tau \in R            (torque of a motor)

Please ckeck the pdf and TODOs in this file.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

from pydrake.all import (MathematicalProgram, Solve, SnoptSolver)
import pydrake.symbolic as sym

################################################################################
# settings
################################################################################

# Pendulum Parameters:
# --------------
m = 1.0         # mass of the pendulum
g = 9.81        # gravity
l = 0.8         # length of the pendulum
b = 0.1         # viscous friction of the pendulm

# Solver Parameters:
# --------------
N       = 350   # number of steps / our problem horizon

# The minimum and maximum Timeteps limits h
h_min = .002    # 500 Hz
h_max = .05     # 100 Hz

################################################################################
# helper function for visualization and dynamics
################################################################################

class Visualizer():
    """Visualize the pendulum
    """
    def __init__(self, ax, length):
        """init visualizer

        Args:
            ax (axis): axis of a figure
            length (float): length of the pendulum in meters
        """
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.link = np.vstack((0.025 * np.array([1, -1, -1, 1, 1]),
                        length * np.array([0, 0, -1, -1, 0])))

        self.link_fill = ax.fill(
                self.link[0, :], self.link[1, :], zorder=1, edgecolor="k", facecolor=[.6, .6, .6])

    def draw(self, time, q):
        R = np.array([[np.cos(q), np.sin(q)], [-np.sin(q), np.cos(q)]])
        p = np.dot(R, self.link)
        self.link_fill[0].get_path().vertices[:, 0] = p[0, :]
        self.link_fill[0].get_path().vertices[:, 1] = p[1, :]
        self.ax.set_title("t = {:.1f}".format(time))


def pendulum_continous_dynamics(x, u):
    """continous dynamics x_dot = f(x, u) for a simple one dimensional 
    pendulum. (See eq. 13)

    Args:
        x (_type_): two dimensional state x=[q, q_dot]
        u (_type_): one dimensional control input u=torque

    Returns:
        _type_: the two dimensional time derivative x_dot=[q_dot, q_ddot]
    """
    
    # use object d to get math functions such as (d.sin, d.log, etc.)
    d = sym if x.dtype == object else np

    #>>>>TODO: add continous state space equation and return x_dot
    x_dot = None
    return x_dot

def pendulum_discretized_dynamics(x, u, x_next, dt):
    """descritization of the continous dynamics.
    First, compute the derivative at the current state x(t) using the know
    dynamics of the pendulum. Then, use euler integration to compute the next
    state at t + dt. 
    Finally, return the residual between the euler integration 
    and x_next. The solver needs to make this zero to repect the dynamics.

    Args:
        x (_type_): two dimensional state x=[q, q_dot] at time t
        x_next (_type_): two dimensional state x_next=[q, q_dot] at time t + dt
        delta_t (_type_): time discretization step

    Returns:
        _type_: the residual between euler integration and x_next
    """

    #>>>>TODO: compute x_dot integrated it using x and dt. Return the
    # residual to x_next
    residuals=None
    return residuals

################################################################################
# problem definition
################################################################################

# the important dimension in this problem
nx = 2  # dimension of our state x=[q, q_dot]
nu = 1  # dimension of our control u=tau

# Define our inital state, the pendulum is in its lowest energy state,
# (angle q=0, velocity q_dot=0)
x_intial = np.array([0.0, 0.0])

# Define our goal (terminal) state, the pendulum is in its upward position
# and should have zero acceleration
x_final = np.array([np.pi, 0.0])

# Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

################################################################################
# variables

# At any of our N timesteps we want to find the state x[k] and the control u[k]
state = prog.NewContinuousVariables(N+1, nx, 'state')
control = prog.NewContinuousVariables(N, 'control')     # nu = 1
h = prog.NewContinuousVariables(N, name='h')

################################################################################
# constraints

# 1. we want our pendulum to start with the inital state x_init
# For this we can add an equality constraint to the first time step k=0
for i in range(nx):
    prog.AddConstraint(state[0,i] == x_intial[i])

# 2. we want our pendulum to end with the final state k=N
# For this we can add an equality to the last time step k=N
for i in range(nx):
    prog.AddConstraint(state[N-1,i] == x_final[i])

# 3. add any timestep we want our solution to respect the dynamics of the pendulum
# That means the next state x_k+1 should be the integral of the prev. state x_k
for k in range(N-1):
    residuals = pendulum_discretized_dynamics(state[k], control[k], state[k+1], h[k])
    for i in range(nx):
        prog.AddConstraint(residuals[i] == 0)

prog.AddBoundingBoxConstraint([h_min]*N, [h_max]*N, h)

# 4. add a constrain on the control torque
#>>>>TODO: After, you simulated the unconstraint case. 
#>>>>TODO: Add some limits on the control torque between some min and max value


################################################################################
# cost function

# in this example there are three costs:
# 1) minimize the control effort: u*R*u
# 2) get closer to the goal: (x - x_goal)^T*Q*(x - x_goal)
# 3) ####TODO: What is the meaning of the term S*sum(h) ?

Q = np.array([[500, 0],[0, 500]])
R = 10
S = 100 

for k in range(N):
    prog.AddCost(control[k]*R*control[k])
    prog.AddCost((state[k] - x_final).dot(Q.dot(state[k] - x_final)))
prog.AddCost(S*sum(h))

################################################################################
# solve
h_guess = h_max
prog.SetInitialGuess(h, [h_guess]*N)

# finally lets start the solver, if we want we could provide an inital guess
result = Solve(prog)

if not result.is_success:
    print("failure")
print("solved")

# extract the solution
state_opt = result.GetSolution(state)
control_opt = result.GetSolution(control)
h_opt = result.GetSolution(h)

# seperate into variables
t_opt = np.cumsum(h_opt)
q_opt = state_opt[:-1,0]
q_dot_opt = state_opt[:-1,1]
torque_opt = control_opt

################################################################################
# plot some stuff

fig, ax = plt.subplots(1,1)
vis = Visualizer(ax, length=l)
for k in range(N):
    if h_opt[k] > h_min:
        vis.draw(t_opt[k], q_opt[k])
        plt.pause(h_opt[k])

# x-axis pos, vel
fig, ax = plt.subplots(3,1, figsize=(12, 10))
ax[0].plot(t_opt, q_opt, linewidth=2, label="Position")
ax[0].grid();ax[0].legend();ax[0].set_ylabel("Postion [rad]")
ax[1].plot(t_opt, q_dot_opt, linewidth=2, label="Velocity")
ax[1].grid();ax[1].legend();ax[1].set_ylabel("Velocity [rad/s]")
ax[2].plot(t_opt, torque_opt, linewidth=2, label="Torque")
ax[2].grid();ax[2].legend();ax[2].set_ylabel("Torque [Nm]");ax[2].set_xlabel("Time [s]")

plt.show()




