import numpy as np
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.robot import Robot

# whole-body controller
from body_control.tsid_wrapper import TSIDWrapper

# robot configs
import walking_control.talos_conf as conf

from walking_control.footstep_planner import Side

# ROS visualizations
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray


class Talos(Node):
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """

    def __init__(self, simulator):
        super().__init__('talos_robot_node')
        self.conf = conf
        self.sim = simulator

        # Create the tsid wrapper for the whole body QP
        self.stack = TSIDWrapper(conf)

        # Use q_init for robot initialization, as conf.q_home results in error
        z_init = 1.1
        q_init = np.hstack([
            np.array([0, 0, z_init, 0, 0, 0, 1]),
            np.zeros_like(conf.q_actuated_home)
        ])

        # spawn robot in simulation
        # Create the pybullet robot in the simulatior
        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.stack.model,
            q=q_init,
            basePosition=[0, 0, z_init],
            baseQuationerion=[0, 0, 0, 1],
            verbose=True,
            useFixedBase=False
        )

        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT

        ########################################################################
        # estimators
        ########################################################################
        self.zmp = None
        self.dcm = None

        ########################################################################
        # sensors
        ########################################################################
        # ft sensors
        # Turn on the force torque sensor in the robots feet
        pb.enableJointForceTorqueSensor(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_right_6_joint'],
            True)
        pb.enableJointForceTorqueSensor(
            self.robot.id(),
            self.robot.jointNameIndexMap()['leg_left_6_joint'],
            True)

        ########################################################################
        # visualizations
        ########################################################################

        # Joint state publisher
        self.pub_joint = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_msg = JointState()
        self.joint_msg.name = self.robot.actuatedJointNames()

        # Floating base broadcaster
        self.br = tf2_ros.TransformBroadcaster(self)

        # Zmp and dcm point publisher
        self.marker_pub = self.create_publisher(
            MarkerArray, '/zmp_dcm_markers', 10)

        # Wrench publisher for left and right foot
        self.left_wrench_pub = self.create_publisher(
            WrenchStamped, '/left_foot_wrench', 10)
        self.right_wrench_pub = self.create_publisher(
            WrenchStamped, '/right_foot_wrench', 10)

    def update(self):
        """updates the robot
        """
        t = self.sim.simTime()
        dt = self.sim.stepTime()

        # Update the pybullet robot
        self.robot.update()

        # update the estimators
        self._update_zmp_estimate()
        self._update_dcm_estimate()

        # update wbc and send back to pybullet
        self._solve(t, dt)

    def setSupportFoot(self, side):
        """sets the the support foot of the robot on given side
        """

        # The support foot is in rigid contact with the ground and should
        # hold the weight of the robot
        self.support_foot = side
        # Activate the foot contact on the support foot & deactivate the motion task on the support foot
        if self.support_foot == Side.LEFT:
            self.stack.add_contact_LF()
            self.stack.remove_motion_LF()
        else:
            self.stack.add_contact_RF()
            self.stack.remove_motion_RF()

    def setSwingFoot(self, side):
        """sets the swing foot of the robot on given side
        """

        # The swing foot is not in contact and can move
        self.swing_foot = side
        # Deactivate the foot contact on the swing foot & turn on the motion task on the swing foot
        if self.swing_foot == Side.LEFT:
            self.stack.remove_contact_LF()
            self.stack.add_motion_LF()
        else:
            self.stack.remove_contact_RF()
            self.stack.add_motion_RF()

    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference
        """

        # Set the pose, velocity and acceleration on the swing foots
        if self.swing_foot == Side.LEFT:
            self.stack.set_LF_pose_ref(T_swing_w, V_swing_w, A_swing_w)
        else:
            self.stack.set_RF_pose_ref(T_swing_w, V_swing_w, A_swing_w)

    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        # Return correct foot pose
        if self.swing_foot == Side.LEFT:
            pose = self.stack.get_placement_LF()
        else:
            pose = self.stack.get_placement_RF()
        return pose

    def supportFootPose(self):
        """return the pose of the current support foot
        """
        # Return correct foot pose
        if self.support_foot == Side.LEFT:
            pose = self.stack.get_placement_LF()
        else:
            pose = self.stack.get_placement_RF()
        return pose

    def publish(self):
        # Publish the jointstate
        now = self.get_clock().now().to_msg()

        # Publish joint states
        self.joint_msg.header.stamp = now
        self.joint_msg.position = self.robot.actuatedJointPosition().tolist()
        self.joint_msg.velocity = self.robot.actuatedJointVelocity().tolist()
        self.joint_msg.effort = self.tau.tolist()

        self.pub_joint.publish(self.joint_msg)

        # Broadcast odometry
        T_b_w, _ = self.stack.baseState()

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = self.robot.baseName()

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

        # Get wrenches
        wr_rankle, wl_lankle = self._get_ankle_wrenches()

        # Right foot
        msg_R = WrenchStamped()
        msg_R.header.stamp = now
        msg_R.header.frame_id = 'leg_right_6_joint'

        msg_R.wrench.force.x = wr_rankle.linear[0]
        msg_R.wrench.force.y = wr_rankle.linear[1]
        msg_R.wrench.force.z = wr_rankle.linear[2]
        msg_R.wrench.torque.x = wr_rankle.angular[0]
        msg_R.wrench.torque.y = wr_rankle.angular[1]
        msg_R.wrench.torque.z = wr_rankle.angular[2]

        self.right_wrench_pub.publish(msg_R)

        # Left foot
        msg_L = WrenchStamped()
        msg_L.header.stamp = now
        msg_L.header.frame_id = 'leg_left_6_joint'

        msg_L.wrench.force.x = wl_lankle.linear[0]
        msg_L.wrench.force.y = wl_lankle.linear[1]
        msg_L.wrench.force.z = wl_lankle.linear[2]
        msg_L.wrench.torque.x = wl_lankle.angular[0]
        msg_L.wrench.torque.y = wl_lankle.angular[1]
        msg_L.wrench.torque.z = wl_lankle.angular[2]

        self.left_wrench_pub.publish(msg_L)

        # Publish dcm and zmp marker
        marker_array = MarkerArray()

        # create marker function
        def create_marker(name, position, color):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = now
            marker.ns = name
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            return marker

        # dcm
        marker_dcm = create_marker("dcm", self.dcm, [1.0, 0.0, 0.0])
        marker_array.markers.append(marker_dcm)

        # zmp
        marker_zmp = create_marker("zmp", self.zmp, [0.0, 1.0, 0.0])
        marker_array.markers.append(marker_zmp)

        self.marker_pub.publish(marker_array)

        ############################################################################
        # private functions
        ############################################################################

    def _get_ankle_wrenches(self):
        """
        Reads ankle joint wrenches and constructs pinocchio Force objects.

        Returns:
        - wr_rankle: Wrench at right ankle (pin.Force)
        - wl_lankle: Wrench at left ankle (pin.Force)
        """
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

    def _get_ankle_transf(self):
        """
        Computes transformations of ankle and sole frames to the world frame.

        Returns:
        - H_w_rankle, H_w_lankle: Right/left ankle to world transforms
        - H_w_rsole, H_w_lsole: Right/left sole to world transforms
        """
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, self.robot.q())

        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId(
            "leg_right_6_joint")]
        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]

        return H_w_rankle, H_w_lankle, H_w_rsole, H_w_lsole

    def _solve(self, t, dt):
        # get the current state
        q = self.robot.q()
        v = self.robot.v()

        # Solve the whole body qp
        self.tau, self.dv = self.stack.update(q, v, t)
        # Command the torque to pybullet robot
        self.robot.setActuatedJointTorques(self.tau)

    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        # Compute the zmp based on force torque sensor readings
        d = 0.1  # vertical offset

        # Read wrenches and frame transforms
        wr_ankle, wl_ankle = self._get_ankle_wrenches()
        _, _, H_w_rsole, H_w_lsole = self._get_ankle_transf()

        # Function for calculating the ZMP of the respective foot
        def calc_foot_zmp(wrench):
            tau_x, tau_y, _ = wrench.angular
            f_x, f_y, f_z = wrench.linear

            # avoid division by zero
            f_z = f_z if abs(f_z) > 1e-6 else 1e-6

            p_x = (-tau_y - f_x * d) / f_z
            p_y = (tau_x - f_y * d) / f_z
            p_z = 0
            return np.array([p_x, p_y, p_z])

        # Function for calculating the double support ZMP
        def calc_zmp(pR, pL, fR_z, fL_z):
            pR_x, pR_y, _ = pR
            pL_x, pL_y, _ = pL

            # Calculate zmp
            p_x_zmp = (pR_x * fR_z + pL_x * fL_z) / (fR_z + fL_z)
            p_y_zmp = (pR_y * fR_z + pL_y * fL_z) / (fR_z + fL_z)
            return np.array([p_x_zmp, p_y_zmp, 0.0])

        # Estimate per-foot ZMP in sole frames
        p_R = calc_foot_zmp(wr_ankle)
        p_L = calc_foot_zmp(wl_ankle)

        # Convert per-foot ZMP to world coordinates
        w_p_R = H_w_rsole * p_R
        w_p_L = H_w_lsole * p_L

        # Calculate ZMP for double support
        self.zmp = calc_zmp(
            w_p_R, w_p_L, wr_ankle.linear[2], wl_ankle.linear[2])

    def _update_dcm_estimate(self):
        """update the estimated dcm position
        """
        # Compute the com based on current center of mass state
        x_CoM = self.robot.baseCoMPosition()
        x_p = np.array([x_CoM[0], x_CoM[1], 0.0])
        x_p_dot = self.robot.baseCoMVelocity()
        x_p_dot = np.array([x_p_dot[0], x_p_dot[1], 0.0])
        omega = np.sqrt(self.conf.g/x_CoM[2])

        self.dcm = x_p + x_p_dot/omega


def main(args=None):
    from simulator.pybullet_wrapper import PybulletWrapper

    rclpy.init(args=args)

    sim = PybulletWrapper()

    robot = Talos(sim)

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
