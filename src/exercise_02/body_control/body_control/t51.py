import threading
import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from scipy.constants import g
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
import tsid
from body_control.tsid_wrapper import TSIDWrapper
import body_control.config_5 as conf

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

################################################################################
# settings
################################################################################

THREADING = True
DO_PLOT = True
USE_MAXLEN = True
MAXLEN = 500

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=True):
        # call base class constructor

        # Initial condition for the simulator an model
        z_init = 1.15

        super().__init__(
            simulator,
            urdf,
            model,
            basePosition=[0, 0, z_init],
            baseQuationerion=[0, 0, 0, 1],
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose)

        self.node = node

        # add publisher
        self.pub_joint = self.node.create_publisher(
            JointState, "/joint_states", 10)

        self.joint_msg = JointState()
        self.joint_msg.name = self.actuatedJointNames()

        # add tf broadcaster
        self.br = tf2_ros.TransformBroadcaster(self.node)

    def update(self):
        # update base class
        super().update()

    def publish(self, T_b_w, tau):
        # publish jointstate
        self.joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_msg.position = self.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.actuatedJointVelocity().tolist()
        self.joint_msg.effort = tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.baseName()

        tf_msg.transform.translation.x = T_b_w.translation[0]
        tf_msg.transform.translation.y = T_b_w.translation[1]
        tf_msg.transform.translation.z = T_b_w.translation[2]

        q = pin.Quaternion(T_b_w.rotation)
        q.normalize()
        tf_msg.transform.rotation.x = q.x
        tf_msg.transform.rotation.y = q.y
        tf_msg.transform.rotation.z = q.z
        tf_msg.transform.rotation.w = q.w

        self.br.sendTransform(tf_msg)

################################################################################
# External pushing force
################################################################################


class ExtPushForce():
    def __init__(self, robot, simulator, force, t_start, t_period, verbose=True, color=[1, 0, 0]):
        assert isinstance(force, np.ndarray), "Input must be a NumPy array."
        assert force.shape == (
            3,), f"Input must be of shape (3,), but got shape {force.shape}."

        self.robot = robot
        self.simulator = simulator
        self.verbose = verbose

        # define parameters
        self.force = force
        self.t_start = t_start
        self.t_end = t_start + t_period
        self.color = color

        self.line_id = None
        line_length = 0.5
        norm = np.linalg.norm(self.force)
        if norm == 0:
            self.direction = np.array([0, 0, 0])
        else:
            self.direction = line_length * self.force / norm

        # flag for line removal
        self.line_removed = False

    def apply_force(self, t):
        if t >= self.t_start and t <= self.t_end:
            # apply force
            self.robot.applyForce(self.force)
            if self.verbose:
                # update debug arrow for high CoM deviations
                p2 = np.array(self.robot.baseCoMPosition())
                if self.line_id is not None:
                    if (np.abs(p2 - self.p2_old) > 0.01).any():
                        p1 = p2 - self.direction
                        self.simulator.removeDebugItem(self.line_id)
                        self.line_id = self.simulator.addGlobalDebugLine(
                            p1, p2, -1, color=self.color)
                        self.p2_old = p2
                else:
                    p1 = p2 - self.direction
                    self.line_id = self.simulator.addGlobalDebugLine(
                        p1, p2, -1, color=self.color)
                    self.p2_old = p2

        if t > self.t_end and not self.line_removed:
            # remove debug error after period
            self.simulator.removeDebugItem(self.line_id)
            self.line_removed = True


################################################################################
# External pushing force
################################################################################

class BalanceController():
    def __init__(self, robot, tsid_wrapper, verbose=True):
        '''
        Balance controller with ankle & hip strategy
        '''
        self.robot = robot
        self.tsid_wrapper = tsid_wrapper
        self.verbose = verbose

        # ankle strategy parameters
        self.x_ref = conf.x_ref
        self.p_ref = conf.p_ref
        self.K_x_ankle = conf.kx_ankle
        self.K_p_ankle = conf.kp_ankle

        # hip strategy parameters
        self.r_ref = conf.r_ref
        self.K_gamma_hip = conf.kgamma_hip

    def ankle_strategy(self, x_d, p, x_ref_dot=None):
        '''
        Ankle balance strategy for small disturbances
        '''

        if x_ref_dot is None:
            # standing condition
            x_ref_dot = np.zeros_like(p)

        d_x = x_d - self.x_ref
        d_p = p - self.p_ref

        # x_d_dot: new desired velocity for CoM
        x_d_dot = x_ref_dot - self.K_x_ankle @ d_x + self.K_p_ankle @ d_p

        # TODO: How to use x_d_dot?
        self.tsid_wrapper.setComRefState(self.p_ref, x_d_dot)

    def hip_strategy(self, r):
        '''
        Hip balance strategy for higher disturbances
        '''
        # Gamma_d: desired angular momentum
        Gamma_d = self.K_gamma_hip @ (r - self.r_ref)

        # set angular momentum (am) task
        am_traj = tsid.TrajectorySample(3)
        am_traj.value(Gamma_d)
        self.tsid_wrapper.amTask.setReference(am_traj)


################################################################################
# Application
################################################################################


class Environment(Node):
    def __init__(self):
        super().__init__('tutorial_4_standing_node')

        # init TSIDWrapper
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        self.simulator = PybulletWrapper(sim_rate=conf.f_cntr)

        q_init = np.hstack([np.array([0, 0, 1.15, 0, 0, 0, 1]),
                           np.zeros_like(conf.q_actuated_home)])

        # init ROBOT
        self.robot = Talos(
            self.simulator,
            conf.urdf,
            self.tsid_wrapper.model,
            self,
            q=q_init,
            verbose=True,
            useFixedBase=False)

        # init Balance Controller
        self.balance_crtl = BalanceController(self.robot, self.tsid_wrapper)

        # init external forces
        force = 55.0
        f_right = force * np.array([0.0, 1.0, 0.0])
        f_left = force * np.array([0.0, -1.0, 0.0])
        f_back = force * np.array([1.0, 0.0, 0.0])
        self.f_ext_right = ExtPushForce(
            self.robot, self.simulator, f_right, 4.0, 0.25)
        self.f_ext_left = ExtPushForce(
            self.robot, self.simulator, f_left, 4.2, 0.25)
        self.f_ext_back = ExtPushForce(
            self.robot, self.simulator, f_back, 4.1, 0.3)

        # add force-torque sensors at ankles
        pb.enableJointForceTorqueSensor(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_right_6_joint'],
            True)
        pb.enableJointForceTorqueSensor(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_left_6_joint'],
            True)

        self.t_publish = 0.0

        # current estimates
        self.zmp_curr_est = None
        self.cmp_curr_est = None
        self.cp_curr_est = None
        self.f_total = None

        # logging
        if USE_MAXLEN:
            self.time = deque(maxlen=MAXLEN)
            self.zmp_x = deque(maxlen=MAXLEN)
            self.zmp_y = deque(maxlen=MAXLEN)
            self.cmp_x = deque(maxlen=MAXLEN)
            self.cmp_y = deque(maxlen=MAXLEN)
            self.cp_x = deque(maxlen=MAXLEN)
            self.cp_y = deque(maxlen=MAXLEN)
            self.com_x = deque(maxlen=MAXLEN)
            self.com_y = deque(maxlen=MAXLEN)
        else:
            self.time = []
            self.zmp_x = []
            self.zmp_y = []
            self.cmp_x = []
            self.cmp_y = []
            self.cp_x = []
            self.cp_y = []
            self.com_x = []
            self.com_y = []

        # plotting (PyQtGraph UI setup)
        # PyQt App setup
        if DO_PLOT:
            self.app = QtWidgets.QApplication([])
            self.win = pg.GraphicsLayoutWidget(
                show=True, title="ZMP/CMP/CP/CoM Plots")
            self.win.resize(1200, 800)

            # === X Component Plots ===
            self.plot_zmp_x = self.win.addPlot(
                title="ZMP X Over Time", row=0, col=0)
            self.plot_zmp_x.setLabels(left='X Value', bottom='Time (s)')
            self.plot_zmp_x.addLegend()
            self.plot_zmp_x.showGrid(x=True, y=True)
            self.curve_zmp_x = self.plot_zmp_x.plot(pen='r', name='ZMP X')

            self.plot_cmp_x = self.win.addPlot(
                title="CMP X Over Time", row=0, col=1)
            self.plot_cmp_x.setLabels(left='X Value', bottom='Time (s)')
            self.plot_cmp_x.addLegend()
            self.plot_cmp_x.showGrid(x=True, y=True)
            self.curve_cmp_x = self.plot_cmp_x.plot(pen='g', name='ZMP X')

            self.plot_cp_x = self.win.addPlot(
                title="CP X Over Time", row=1, col=0)
            self.plot_cp_x.setLabels(left='X Value', bottom='Time (s)')
            self.plot_cp_x.addLegend()
            self.plot_cp_x.showGrid(x=True, y=True)
            self.curve_cp_x = self.plot_cp_x.plot(pen='b', name='CP X')

            self.plot_com_x = self.win.addPlot(
                title="CoM X Over Time", row=1, col=1)
            self.plot_com_x.setLabels(left='X Value', bottom='Time (s)')
            self.plot_com_x.addLegend()
            self.plot_com_x.showGrid(x=True, y=True)
            self.curve_com_x = self.plot_com_x.plot(pen='m', name='CoM X')

            # === Y Component Plots ===
            self.plot_zmp_y = self.win.addPlot(
                title="ZMP Y Over Time", row=2, col=0)
            self.plot_zmp_y.setLabels(left='Y Value', bottom='Time (s)')
            self.plot_zmp_y.addLegend()
            self.plot_zmp_y.showGrid(x=True, y=True)
            self.curve_zmp_y = self.plot_zmp_y.plot(pen='r', name='ZMP Y')

            self.plot_cmp_y = self.win.addPlot(
                title="CMP Y Over Time", row=2, col=1)
            self.plot_cmp_y.setLabels(left='Y Value', bottom='Time (s)')
            self.plot_cmp_y.addLegend()
            self.plot_cmp_y.showGrid(x=True, y=True)
            self.curve_cmp_y = self.plot_cmp_y.plot(pen='g', name='ZMP Y')

            self.plot_cp_y = self.win.addPlot(
                title="CP Y Over Time", row=3, col=0)
            self.plot_cp_y.setLabels(left='Y Value', bottom='Time (s)')
            self.plot_cp_y.addLegend()
            self.plot_cp_y.showGrid(x=True, y=True)
            self.curve_cp_y = self.plot_cp_y.plot(pen='b', name='CP Y')

            self.plot_com_y = self.win.addPlot(
                title="CoM Y Over Time", row=3, col=1)
            self.plot_com_y.setLabels(left='Y Value', bottom='Time (s)')
            self.plot_com_y.addLegend()
            self.plot_com_y.showGrid(x=True, y=True)
            self.curve_com_y = self.plot_com_y.plot(pen='m', name='CoM Y')

            # Set up timer for update loop
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot)
            self.timer.start(50)  # 20 Hz

    def get_ankle_wrenches(self):
        # read ankle wrenches
        wren_r = pb.getJointState(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_right_6_joint'])[2]
        wnp_r = np.array(
            [-wren_r[0], -wren_r[1], -wren_r[2], -wren_r[3], -wren_r[4], -wren_r[5]])
        wr_rankle = pin.Force(wnp_r)

        wren_l = pb.getJointState(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wnp_l = np.array(
            [-wren_l[0], -wren_l[1], -wren_l[2], -wren_l[3], -wren_l[4], -wren_l[5]])
        wl_lankle = pin.Force(wnp_l)

        # get positions of ankles and soles
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, self.robot.q())

        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId(
            "leg_right_6_joint")]

        # transform wrenches to world frame
        wr_world = H_w_rankle.act(wr_rankle)
        wl_world = H_w_lankle.act(wl_lankle)

        # transform wrenches to sole frame
        wr_sole = H_w_rsole.actInv(wr_world)
        wl_sole = H_w_lsole.actInv(wl_world)

        return wr_sole, wl_sole

    def estimate_ZMP(self):
        # estimate the Zero Moment Point
        d = 0.1

        # get wrenches
        wr_sole, wl_sole = self.get_ankle_wrenches()
        # right ankle (R)
        tau_xR, tau_yR, tau_zR = wr_sole.angular
        f_xR, f_yR, f_zR = wr_sole.linear
        # left ankle (L)
        tau_xL, tau_yL, tau_zL = wl_sole.angular
        f_xL, f_yL, f_zL = wl_sole.linear

        # avoid division by zero
        f_zR = f_zR if abs(f_zR) > 1e-6 else 1e-6
        f_zL = f_zL if abs(f_zL) > 1e-6 else 1e-6

        # left foot
        p_xL = (-tau_yL - f_xL * d) / f_zL
        p_yL = (-tau_xL - f_yL * d) / f_zL
        p_zL = 0
        p_L = np.array([p_xL, p_yL, p_zL])

        # right foot
        p_xR = (-tau_yR - f_xR * d) / f_zR
        p_yR = (-tau_xR - f_yR * d) / f_zR
        p_zR = 0
        p_R = np.array([p_xR, p_yR, p_zR])

        # distinguish foot support
        total_fz = f_zR + f_zL
        double_support = total_fz > 1e-3 and f_zR > 1e-3 and f_zL > 1e-3

        if double_support:
            # double support
            p_x = (p_xR * f_zR + p_xL * f_zL) / (f_zR + f_zL)
            p_y = (p_yR * f_zR + p_yL * f_zL) / (f_zR + f_zL)
            p_z = 0
            self.zmp_curr_est = np.array([p_x, p_y, p_z])
            self.f_total = wr_sole.linear + wl_sole.linear
        else:
            # single support
            if f_zR > f_zL:
                self.zmp_curr_est = p_R
                self.f_total = wr_sole.linear
            else:
                self.zmp_curr_est = p_L
                self.f_total = wl_sole.linear

    def estimate_CMP(self):
        # estimate the Centroidal Moment Pivot
        p_zmp = self.get_zmp()
        f = self.get_f_total()
        X_x, X_y, X_z = p_zmp
        f_x, f_y, f_z = f

        r_x = X_x - f_x/f_z * X_z
        r_y = X_y - f_y/f_z * X_z
        r_z = 0

        self.cmp_curr_est = np.array([r_x, r_y, r_z])

    def estimate_CP(self):
        # estimate the Capture point (CP) / Divergent Component of Motion (DCM)
        x_CoM = self.robot.baseCoMPosition()
        x_p = np.array([x_CoM[0], x_CoM[1], 0.0])
        x_p_dot = self.robot.baseCoMVelocity()
        omega = np.sqrt(g/x_CoM[2])

        self.cp_curr_est = x_p + x_p_dot/omega

    def compute_estimates(self):
        # compute all estimates
        self.estimate_ZMP()
        self.estimate_CMP()
        self.estimate_CP()

    def get_zmp(self):
        return self.zmp_curr_est

    def get_cmp(self):
        return self.cmp_curr_est

    def get_cp(self):
        return self.cp_curr_est

    def get_f_total(self):
        return self.f_total

    def logging(self, t):
        # log the x and y components of ground reference points and CoM
        p_zmp = self.get_zmp()
        p_cmp = self.get_cmp()
        p_cp = self.get_cp()
        p_com = self.robot.baseCoMPosition()

        self.time.append(t)
        self.zmp_x.append(p_zmp[0])
        self.zmp_y.append(p_zmp[1])
        self.cmp_x.append(p_cmp[0])
        self.cmp_y.append(p_cmp[1])
        self.cp_x.append(p_cp[0])
        self.cp_y.append(p_cp[1])
        self.com_x.append(p_com[0])
        self.com_y.append(p_com[1])

    def update_plot(self):
        # Set data to curves
        if len(self.time) > 0 and DO_PLOT:
            self.curve_zmp_x.setData(self.time, self.zmp_x)
            self.curve_cmp_x.setData(self.time, self.cmp_x)
            self.curve_cp_x.setData(self.time, self.cp_x)
            self.curve_com_x.setData(self.time, self.com_x)

            self.curve_zmp_y.setData(self.time, self.zmp_y)
            self.curve_cmp_y.setData(self.time, self.cmp_y)
            self.curve_cp_y.setData(self.time, self.cp_y)
            self.curve_com_y.setData(self.time, self.com_y)

    def update(self):
        # elapsed time
        t = self.simulator.simTime()

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # apply forces
        self.f_ext_right.apply_force(t)
        self.f_ext_left.apply_force(t)
        self.f_ext_back.apply_force(t)

        # compute estimates
        self.compute_estimates()

        # balance strategy
        # self.balance_crtl.ankle_strategy(
        #    self.robot.baseCoMPosition(), self.get_zmp())
        # self.balance_crtl.hip_strategy(self.get_cmp())

        # update TSID controller
        tau_sol, _ = self.tsid_wrapper.update(
            self.robot.q(), self.robot.v(), t)

        # command to the robot
        self.robot.setActuatedJointTorques(tau_sol)

        # logging
        self.logging(t)

        # publish to ros
        if t - self.t_publish > 1./30.:
            self.t_publish = t
            # get current BASE Pose
            T_b_w, _ = self.tsid_wrapper.baseState()
            self.robot.publish(T_b_w, tau_sol)


################################################################################
# main
################################################################################

def ros_spin_thread(node):
    try:
        while rclpy.ok():
            node.update()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def main(args=None):
    rclpy.init(args=args)
    env = Environment()

    # threading for faster simulation with plots
    if THREADING and DO_PLOT:
        ros_thread = threading.Thread(target=ros_spin_thread, args=(env,))
        ros_thread.start()

        env.app.exec_()

        env.destroy_node()
        rclpy.shutdown()
        ros_thread.join()
    else:
        try:
            while rclpy.ok():
                env.update()

                if DO_PLOT:
                    env.app.processEvents()

        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            env.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
