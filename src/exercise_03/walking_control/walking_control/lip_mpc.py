import numpy as np

from pydrake.all import MathematicalProgram, Solve

################################################################################
# Helper fnc
################################################################################


def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics
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


def discrete_LIP_dynamics(g, h, dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics
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
    A_d = np.block([
        [A_1d, np.zeros((2, 2))],
        [np.zeros((2, 2)), A_1d]
    ])

    # Control: [p_x, p_y]
    B_d = np.block([
        [B_1d, np.zeros((2, 1))],
        [np.zeros((2, 1)), B_1d]
    ])

    return A_d, B_d

################################################################################
# LIPInterpolator
################################################################################


class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight 
    """

    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital
        self.g = conf.g
        self.h = conf.h

        self.A, self.B = continious_LIP_dynamics(self.g, self.h)
        self.g_h = self.g/self.h

    def integrate(self, u):
        # Integrate with dt
        self.x_dot = self.A @ self.x + self.B @ u
        self.x += self.x_dot * self.dt
        return self.x

    def comState(self):
        # Return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        c = np.array([self.x[0], self.x[2], self.h])
        c_dot = np.array([self.x[1], self.x[3], 0.0])
        c_ddot = np.array([self.x_dot[1], self.x_dot[3], 0.0])
        return c, c_dot, c_ddot

    def dcm(self):
        # Return the computed dcm
        com, com_dot, _ = self.comState()
        x_p = np.array([com[0], com[1]])
        x_p_dot = np.array([com_dot[0], com_dot[1]])

        omega = np.sqrt(self.g/com[2])

        dcm = x_p + x_p_dot/omega
        return np.array([dcm[0], dcm[1], 0.0])

    def zmp(self):
        # Return the zmp
        c, _, c_ddot = self.comState()
        zmp_x = c[0] - c_ddot[0] / self.g_h
        zmp_y = c[1] - c_ddot[1] / self.g_h
        return np.array([zmp_x, zmp_y, 0.0])


################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt_mpc
        self.no_samples = conf.no_mpc_samples_per_horizon
        self.g = conf.g
        self.h = conf.h

        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None

        # MPC parameter gains
        self.alpha = conf.alpha
        self.gamma = conf.gamma

        self.Ad, self.Bd = discrete_LIP_dynamics(self.g, self.h, self.dt)

        # Foot parameters
        self.foot_length = (conf.lfxp + conf.lfxn)/2.0
        self.foot_width = (conf.lfyp + conf.lfyn)/2.0

    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint

        Returns:
            _type_: control
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
            x_lower = step_x - self.foot_length / 2
            x_upper = step_x + self.foot_length / 2
            y_lower = step_y - self.foot_width / 2
            y_upper = step_y + self.foot_width / 2

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
                self.alpha * np.eye(nu),   # Weight matrix scaled by alpha
                ZMP_ref_k[k],              # Desired control (ZMP reference)
                control[k]                 # Control variable at timestep k
            )

            vx = state[k, 1]
            vy = state[k, 3]

            prog.AddQuadraticCost(self.gamma * (vx**2 + vy**2))

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


def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate a function that computes a referecne trajecotry for the zmp.
    Our goal is to keep the ZMP at the footstep center within each step

    Args:
        foot_steps (_type_): _description_
        no_samples_per_step (_type_): _description_
    """
    zmp_ref_list = []

    for foot in foot_steps:
        pose = foot.poseInWorld()
        center_offset = np.mean(foot.footprint, axis=1)
        x = pose.translation[0] + center_offset[0]
        y = pose.translation[1] + center_offset[1]
        zmp_step = np.tile([x, y], (no_samples_per_step, 1))
        zmp_ref_list.append(zmp_step)

    zmp_ref = np.vstack(zmp_ref_list)

    return zmp_ref


def main(args=None):
    from simulator.pybullet_wrapper import PybulletWrapper
    from walking_control.talos import Talos
    from walking_control.footstep_planner import FootStepPlanner
    from walking_control.footstep_planner import Side
    import walking_control.talos_conf as conf
    import rclpy
    import pinocchio as pin

    rclpy.init(args=args)

    sim = PybulletWrapper()

    robot = Talos(sim)

    # inital footsteps
    # Set intial swing foot pose to left foot
    T_swing_w = robot.swingFootPose()
    # Set intial support foot pose to right foot
    T_support_w = robot.supportFootPose()

    no_steps = 20
    planner = FootStepPlanner(conf)  # Create the planner
    plan = planner.planLine(T_swing_w, Side.LEFT,
                            no_steps + 2)  # Create the plan
    # Append the two last steps once more to the plan so our mpc horizon will never run out
    plan.append(plan[-1])
    plan.append(plan[-1])
    planner.plot(sim)

    # generate reference
    ZMP_ref = generate_zmp_reference(
        plan, conf.no_mpc_samples_per_step)

    # setup the lip models
    mpc = LIPMPC(conf)  # Setup mpc

    # Assume the com is over the first support foot
    # Build the intial mpc state vector
    x0 = np.array([
        T_support_w.translation[0],
        0.0,
        T_support_w.translation[1],
        0.0
    ])

    # Create the interpolator and set the inital state
    interpolator = LIPInterpolator(x0, conf)

    # set the com task reference to the inital support foot
    com_rf = np.array(
        [T_support_w.translation[0], T_support_w.translation[1], conf.h])
    # Set the COM reference to be over supporting foot
    robot.stack.setComRefState(com_rf)

    pre_dur = 3.0   # Time to wait before walking should start

    # Compute number of iterations:
    N_pre = int(pre_dur/conf.dt)  # Number of sim steps before walking starts
    # Total number of sim steps during walking
    N_sim = no_steps * conf.no_sim_per_step

    # current MPC index
    k = 1 * conf.no_mpc_samples_per_step

    for i in range(-N_pre, N_sim):
        t = sim.simTime()  # Simulator time
        dt = sim.stepTime()  # Simulator dt

        ########################################################################
        # update the mpc very no_sim_per_mpc steps
        ########################################################################

        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            # MPC update
            # Get current LIP state
            c = interpolator.x
            sim.addSphereMarker(
                np.array([c[0], c[2], 0.0]), radius=0.01, color=[0, 1, 0, 1])
            # Extract the ZMP reference
            ZMP_ref_k = ZMP_ref[k: k + conf.no_mpc_samples_per_horizon]
            sim.addSphereMarker(
                np.array([ZMP_ref_k[0][0], ZMP_ref_k[0][1], 0.0]), radius=0.01, color=[1, 0, 0, 1])
            # get terminal index
            idx_terminal_k = (no_steps - 1) * conf.no_mpc_samples_per_step - k
            # Solve mpc
            # u_k seems to be always correct & within footplan
            u_k = mpc.buildSolveOCP(c, ZMP_ref_k, idx_terminal_k)
            # print(f"x_dif: {interpolator.x[0] - u_k[0]}")
            # print(f"y_dif: {interpolator.x[2] - u_k[1]}")
            sim.addSphereMarker(
                np.array([u_k[0], u_k[1], 0.0]), radius=0.01, color=[0, 0, 1, 1])
            k += 1

        ########################################################################
        # in every iteration when walking
        ########################################################################

        if i >= 0:
            # Update the interpolator with the latest command u_k
            x_k = interpolator.integrate(u_k)

            # Feed the com tasks with the new com reference
            # com_pos, com_vel, com_acc = interpolator.comState()

        ########################################################################
        # update the simulation
        ########################################################################

        # Update the simulator and the robot
        sim.step()
        sim.debug()
        robot.update()

    try:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0)
            sim.step()
            robot.update()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
