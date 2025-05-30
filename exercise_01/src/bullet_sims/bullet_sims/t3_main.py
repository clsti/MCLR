import numpy as np
import numpy.linalg as la

# simulator (#TODO: set your own import path!)
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# modeling
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from enum import Enum
from scipy.interpolate import CubicHermiteSpline

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

################################################################################
# utility functions
################################################################################


class State(Enum):
    JOINT_SPLINE = 0,
    CART_SPLINE = 1

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, q=None, verbose=True, useFixedBase=True):
        # For Talos robot
        urdf = "src/talos_description/robots/talos_reduced.urdf"
        path_meshes = "src/talos_description/meshes/../.."

        # Initial condition for the simulator an model
        z_init = 1.15

        if useFixedBase:
            base_model = None
        else:
            base_model = pin.JointModelFreeFlyer()

        self._wrapper = pin.RobotWrapper.BuildFromURDF(urdf,             # Model description
                                                       path_meshes,      # Model geometry descriptors
                                                       base_model,       # Floating base model. Use "None" if fixed
                                                       verbose,          # Printout model details
                                                       None)             # Load meshes different from the descripor
        # print(self._wrapper.model.nq)

        super().__init__(
            simulator,
            urdf,
            self._wrapper.model,
            [0, 0, z_init],
            [0, 0, 0, 1],
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose)

    def update(self):
        # update robot
        super().update()
        # update base class, update pinocchio robot wrapper's kinematics
        self._wrapper.forwardKinematics(self.q(), self.v())

    def wrapper(self):
        return self._wrapper

    def data(self):
        return self._wrapper.data

    def publish(self):
        pass

################################################################################
# Controllers
################################################################################


class JointSpaceController:
    """JointSpaceController
    Tracking controller in jointspace
    """

    def __init__(self, robot, Kp, Kd):
        # Save gains, robot ref
        self.robot = robot
        self.Kp = Kp
        self.Kd = Kd

        self.model = self.robot._model
        self.data = self.robot.data()

    def update(self, q_r, q_r_dot, q_r_ddot):
        # Compute jointspace torque, return torque

        M = pin.crba(self.model, self.data, self.robot.q())
        h = pin.nonLinearEffects(self.model, self.data,
                                 self.robot.q(), self.robot.v())
        q_d_dot = q_r_ddot - \
            self.Kd @ (self.robot.v() - q_r_dot) - \
            self.Kp @ (self.robot.q() - q_r)
        tau = M @ q_d_dot + h
        return tau


class CartesianSpaceController:
    """CartesianSpaceController
    Tracking controller in cartspace
    """

    def __init__(self, robot, joint_name, Kp, Kd):
        # save gains, robot ref
        self.robot = robot
        self.joint_name = joint_name
        self.Kp = Kp
        self.Kd = Kd

        self.damp = 1e-12

        self.model = self.robot._model
        self.data = self.robot.data()

    def update(self, X_r, X_dot_r, X_ddot_r):
        # compute cartesian control torque, return torque

        q = self.robot.q()
        q_dot = self.robot.v()

        # Get joint id from joint name
        id = self.model.getJointId(self.joint_name)

        # Compute the Joint Jacobian
        self.l_J_id = pin.computeJointJacobian(self.model, self.data, q, id)

        # Get current Cartesian pose & velocity of controlled frame
        w_X_id = self.data.oMi[id]
        l_X_dot_id = self.l_J_id @ q_dot

        # Compute the Cartesian desired acceleration
        l_X_w = pin.SE3.Identity()
        l_X_dot_r = l_X_w * w_X_id.act(pin.Motion(X_dot_r))
        l_X_ddot_r = l_X_w * w_X_id.act(pin.Motion(X_ddot_r))

        X_dif = pin.log(X_r.actInv(w_X_id)).vector
        X_dot_dif = l_X_dot_id - l_X_dot_r

        X_ddot_d = l_X_ddot_r - self.Kd @ X_dot_dif - self.Kp @ X_dif

        # Compute J_dot_q_dot
        J_dot_q_dot = pin.getClassicalAcceleration(self.model, self.data, id)

        # Map Cartesian desired acceleration to q_ddot_d
        # q_ddot_d = J_pinv @ (X_ddot_d - J_dot_q_dot)
        inv_term = self.l_J_id.dot(self.l_J_id.T) + self.damp * np.eye(6)
        q_ddot_d = self.l_J_id.T.dot(
            la.solve(inv_term, X_ddot_d - J_dot_q_dot))

        # Dynamics model
        M = pin.crba(self.model, self.data, q)
        h = pin.nonLinearEffects(self.model, self.data, q, q_dot)

        tau = M @ q_ddot_d + h
        return tau


################################################################################
# Application
################################################################################


class Environment(Node):
    def __init__(self):
        super().__init__('env_talos_sim')
        # state
        self.cur_state = State.JOINT_SPLINE

        # create simulation
        self.simulator = PybulletWrapper()

        ########################################################################
        # spawn the robot
        ########################################################################
        self.q_home = np.zeros(32)
        self.q_home[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0])
        self.q_home[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0])

        self.q_init = np.zeros(32)

        self.robot = Talos(self.simulator, self.q_init)

        ########################################################################
        # joint space spline: init -> home
        ########################################################################

        # create a joint spline
        self.homing_duration = 5.0  # sec
        self.t_homing = 0.0
        v_q_init = np.zeros_like(self.q_init)
        v_q_home = np.zeros_like(self.q_home)
        self.splines = [
            CubicHermiteSpline(
                [0.0, self.homing_duration], [self.q_init[i], self.q_home[i]], [v_q_init[i], v_q_home[i]])
            for i in range(len(self.q_init))
        ]

        # gain matrices
        Kx_I_joint = np.diag(np.array([3]*12 + [1]*20))
        Kp_joint = 400.0 * Kx_I_joint
        Kd_joint = 2.0 * np.sqrt(Kp_joint)  # critically damped

        # create a joint controller
        self.joint_crtl = JointSpaceController(self.robot, Kp_joint, Kd_joint)

        ########################################################################
        # cart space: hand motion
        ########################################################################

        # gain matrices
        Kx_I_cart = np.eye(6)
        Kp_cart = 200.0 * Kx_I_cart
        Kd_cart = 2.0 * np.sqrt(Kp_cart)  # critically damped

        # joint name
        self.joint_name = "arm_right_7_joint"

        self.cart_crtl = CartesianSpaceController(
            self.robot, self.joint_name, Kp_cart, Kd_cart)

        # initial hand transformation
        self.X_goal = pin.SE3.Identity()

        # control targets
        self.X_r = self.X_goal
        self.X_dot_r = np.zeros(6)
        self.X_ddot_r = np.zeros(6)

        # pose subscriber
        self.sub = self.create_subscription(
            PoseStamped, "/marker_pose", self.get_marker_pose, 10)

        ########################################################################
        # logging
        ########################################################################

        # Publish robot state every 0.01 s to ROS
        self.publish_period = 0.01
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        # Publisher for home position trigger
        self.pub_home_trg = self.create_publisher(Bool, "/trigger_homing", 10)

        self.timer = self.create_timer(self.publish_period, self.publish)

        # create joint state message
        self.joint_msg = JointState()
        self.joint_msg.name = self.robot.actuatedJointNames()

    def update(self, t, dt):
        # update the robot and model
        self.robot.update()

        # update the controllers
        # Do inital jointspace, switch to cartesianspace control
        if self.cur_state == State.JOINT_SPLINE:
            self.t_homing += dt
            self.t_homing = np.clip(self.t_homing, 0.0, self.homing_duration)

            # create spline
            q_t = np.array([spline(self.t_homing) for spline in self.splines])
            dq_t = np.array([spline.derivative()(self.t_homing)
                             for spline in self.splines])
            ddq_t = np.array([spline.derivative(nu=2)(self.t_homing)
                              for spline in self.splines])

            self.tau = self.joint_crtl.update(q_t, dq_t, ddq_t)

            # Switch joint to cartesian control
            if self.t_homing == self.homing_duration:
                self.cur_state = State.CART_SPLINE
                id = self.robot._model.getJointId(self.joint_name)
                self.X_goal = self.robot.data().oMi[id]

                # send trigger for homing pose
                msg_trg = Bool()
                msg_trg.data = True
                self.pub_home_trg.publish(msg_trg)

        else:
            self.X_r = self.X_goal
            tau_joint = self.joint_crtl.update(
                self.q_home, np.zeros_like(self.q_home), np.zeros_like(self.q_home))

            tau_cart = self.cart_crtl.update(
                self.X_r, self.X_dot_r, self.X_ddot_r)

            # calculate nullspace projector
            I = np.eye(self.q_home.shape[0])
            N = I - self.cart_crtl.l_J_id.T @ la.pinv(self.cart_crtl.l_J_id).T

            self.tau = N @ tau_joint + tau_cart

        # command the robot
        self.robot.setActuatedJointTorques(self.tau)

    def publish(self):
        # publish the joint state
        self.joint_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_msg.position = self.robot.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.robot.actuatedJointVelocity().tolist()
        self.joint_msg.effort = self.tau.tolist()

        self.pub.publish(self.joint_msg)

    def get_marker_pose(self, msg):
        vec = np.array([msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w])
        self.X_goal = pin.XYZQUATToSE3(vec)


def main(args=None):
    rclpy.init(args=args)
    env = Environment()
    try:
        while rclpy.ok():
            t = env.simulator.simTime()
            dt = env.simulator.stepTime()

            env.update(t, dt)

            env.simulator.debug()
            env.simulator.step()
            # spin once for joint state publisher
            rclpy.spin_once(env)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
