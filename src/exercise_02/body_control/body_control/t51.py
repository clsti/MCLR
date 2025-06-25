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
USE_MAXLEN = False
MAXLEN = 500

################################################################################
# Robot
################################################################################


class Talos(Robot):
    def __init__(self, simulator, urdf, model, node, q=None, verbose=True, useFixedBase=True):
        # call base class constructor

        # Initial condition for the simulator an model
        z_init = 1.1

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


class ExtPushForce:
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

            self.visualize_force()

        self.remove_visualization(t)

    def visualize_force(self):
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

    def remove_visualization(self, t):
        if t > self.t_end and not self.line_removed:
            # remove debug line after period
            self.simulator.removeDebugItem(self.line_id)
            self.line_removed = True


################################################################################
# External pushing force
################################################################################


class BalanceController:
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

        self.delta_x_com = np.array([0.0, 0.0, 0.0])
        self.max_offset = np.array([0.03, 0.08, 0.0])

        # hip strategy parameters
        self.r_ref = conf.r_ref
        self.K_gamma_hip = conf.kgamma_hip

    def ankle_strategy(self, dt, x_d, p, x_ref_dot=None):
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

        # integrate velocity
        self.delta_x_com += x_d_dot * dt

        # clip delta to prevent drift
        self.delta_x_com = np.clip(
            self.delta_x_com, -self.max_offset, self.max_offset)

        x_com = self.x_ref + self.delta_x_com

        self.tsid_wrapper.setComRefState(x_com, x_d_dot)

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

        # init TSIDWrapper (including CoM & whole-body angular momentum tasks)
        self.tsid_wrapper = TSIDWrapper(conf)

        # init Simulator
        self.simulator = PybulletWrapper(sim_rate=conf.f_cntr)

        # use q_init for robot initialization, as conf.q_home results in error
        q_init = np.hstack([np.array([0, 0, 1.1, 0, 0, 0, 1]),
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
        force = 30.0
        f_right = force * np.array([0.0, 1.0, 0.0])
        f_left = force * np.array([0.0, -1.0, 0.0])
        f_back = force * np.array([1.0, 0.0, 0.0])
        self.f_ext_right = ExtPushForce(
            self.robot, self.simulator, f_right, 4.0, 0.5)
        self.f_ext_left = ExtPushForce(
            self.robot, self.simulator, f_left, 6.0, 0.5)
        self.f_ext_back = ExtPushForce(
            self.robot, self.simulator, f_back, 8.0, 0.5)

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

        return wr_rankle, wl_lankle

    def get_ankle_transf(self):

        # get positions of ankles and soles
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, self.robot.q())

        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId(
            "leg_right_6_joint")]
        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]

        return H_w_rankle, H_w_lankle, H_w_rsole, H_w_lsole

    def estimate_ZMP(self):
        # estimate the Zero Moment Point
        d = 0.1

        # get wrenches (6 joint coordinate frame)
        wr_ankle, wl_ankle = self.get_ankle_wrenches()

        # get positions of ankles and soles
        H_w_rankle, H_w_lankle, H_w_rsole, H_w_lsole = self.get_ankle_transf()

        # function for calculating the zmp of the respective foot
        def calc_foot_zmp(wrench):
            tau_x, tau_y, tau_z = wrench.angular
            f_x, f_y, f_z = wrench.linear

            # avoid division by zero
            f_z = f_z if abs(f_z) > 1e-6 else 1e-6

            p_x = (-tau_y - f_x * d) / f_z
            p_y = (tau_x - f_y * d) / f_z
            p_z = 0
            return np.array([p_x, p_y, p_z])

        # function for calculating the double support zmp
        def calc_zmp(pR, pL, fR_z, fL_z):
            pR_x, pR_y, pR_z = pR
            pL_x, pL_y, pL_z = pL

            # calculate zmp
            p_x_zmp = (pR_x * fR_z + pL_x * fL_z) / (fR_z + fL_z)
            p_y_zmp = (pR_y * fR_z + pL_y * fL_z) / (fR_z + fL_z)
            return np.array([p_x_zmp, p_y_zmp, 0.0])

        # right & left foot zmp (foot/sole coordinate frame)
        p_R = calc_foot_zmp(wr_ankle)
        p_L = calc_foot_zmp(wl_ankle)

        # Transform points (p_L & p_R) to world frame
        w_p_R = H_w_rsole * p_R
        w_p_L = H_w_lsole * p_L

        # calculate zmp for double support
        self.zmp_curr_est = calc_zmp(
            p_R, p_L, wr_ankle.linear[2], wl_ankle.linear[2])

        # Transform ground reaction force to zmp
        w_H_zmp = pin.SE3(np.eye(3), self.zmp_curr_est)
        wr_zmp = w_H_zmp.inverse() * H_w_rankle * wr_ankle
        wl_zmp = w_H_zmp.inverse() * H_w_lankle * wl_ankle

        self.f_total = wr_zmp.linear + wl_zmp.linear

    def estimate_CMP(self):
        # estimate the Centroidal Moment Pivot
        p_com = self.robot.baseCoMPosition()
        f = self.get_f_total()
        X_x, X_y, X_z = p_com
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
        x_p_dot = np.array([x_p_dot[0], x_p_dot[1], 0.0])
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
        dt = self.simulator.stepTime()

        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # compute estimates
        self.compute_estimates()

        # apply forces
        self.f_ext_right.apply_force(t)
        self.f_ext_left.apply_force(t)
        self.f_ext_back.apply_force(t)

        # balance strategy
        self.balance_crtl.ankle_strategy(
            dt, self.tsid_wrapper.comState().value(), self.get_zmp())
        self.balance_crtl.hip_strategy(self.get_cmp())

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
