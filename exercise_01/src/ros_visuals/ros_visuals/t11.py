import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import pinocchio as pin
import numpy as np


class CageTFBroadcaster(Node):
    def __init__(self):
        super().__init__('cage_tf_broadcaster')
        self.br = tf2_ros.TransformBroadcaster(self)

        # Cage parameters (half length)
        self.l_cage = 0.5
        self.w_cage = 0.3
        self.h_cage = 0.4

        self.dt = 0.01  # time discretization
        self.v = np.array([0.1, 0, 0])  # linear velocity
        self.omega = np.array([0, 0.05, 0.2])  # angular velocity
        self.H_iterator = pin.SE3(np.eye(3), np.array(
            [0, 0, 2]))  # current transformation matrix

        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Frame names
        self.frames = {
            "world": "world",
            "reference": "center",
            "ids": ["corner_1", "corner_2", "corner_3", "corner_4",
                    "corner_5", "corner_6", "corner_7", "corner_8", ]
        }

        # Create cage
        self.cage_tfs = self.build_cage()

        # Publisher for marker
        self.pub_marker_c1_p = self.create_publisher(Marker, 'marker_c1_p', 1)
        self.pub_marker_w_p = self.create_publisher(Marker, 'marker_w_p', 1)

        # Marker point parameters
        self.c1_point = {
            "ref_frame": "corner_1",
            "coords": np.array([0.2, 0.3, 0.1])
        }
        self.world_point = {
            "ref_frame": "world"
        }

        # Marker colors
        self.color_orange = [1.0, 1.0, 0.5, 0.0]
        self.color_green = [1.0, 0.0, 1.0, 0.0]

        # Flag to set markers to cubes (exercise a-10)
        self.cube_flag = False

    def get_rotation_matrix(self, roll, pitch, yaw):
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        return pin.rpy.rpyToMatrix(roll, pitch, yaw)

    def get_translation_vector(self, t_x, t_y, t_z):
        return np.array([t_x, t_y, t_z])

    def create_SE3_transformations(self, roll, pitch, yaw, t_x, t_y, t_z):
        return pin.SE3(
            self.get_rotation_matrix(roll, pitch, yaw),
            self.get_translation_vector(t_x, t_y, t_z))

    def build_cage(self):
        H_0 = self.create_SE3_transformations(0, 0, 0, 0, 0, 0)
        H_1 = self.create_SE3_transformations(
            0, 0, 0, -self.l_cage, -self.w_cage, -self.h_cage)
        H_2 = self.create_SE3_transformations(
            0, 0, 90, self.l_cage, -self.w_cage, -self.h_cage)
        H_3 = self.create_SE3_transformations(
            0, 0, 180, self.l_cage, self.w_cage, -self.h_cage)
        H_4 = self.create_SE3_transformations(
            0, 0, 270, -self.l_cage, self.w_cage, -self.h_cage)
        H_5 = self.create_SE3_transformations(
            270, 0, 0, -self.l_cage, -self.w_cage, self.h_cage)
        H_6 = self.create_SE3_transformations(
            0, 180, 0, self.l_cage, -self.w_cage, self.h_cage)
        H_7 = self.create_SE3_transformations(
            0, 180, 90, self.l_cage, self.w_cage, self.h_cage)
        H_8 = self.create_SE3_transformations(
            180, 0, 0, -self.l_cage, self.w_cage, self.h_cage)

        # Create an array of SE3 transformations
        array_SE3 = [H_0, H_1, H_2, H_3, H_4, H_5, H_6, H_7, H_8]
        return array_SE3

    def broadcast_tf(self, H, frame_reference, frame_id):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = frame_reference
        tf_msg.child_frame_id = frame_id

        tf_msg.transform.translation.x = H.translation[0]
        tf_msg.transform.translation.y = H.translation[1]
        tf_msg.transform.translation.z = H.translation[2]

        q = pin.Quaternion(H.rotation)
        q.normalize()
        tf_msg.transform.rotation.x = q.x
        tf_msg.transform.rotation.y = q.y
        tf_msg.transform.rotation.z = q.z
        tf_msg.transform.rotation.w = q.w

        self.br.sendTransform(tf_msg)

    def skew_symmetric_matrix(self, w):
        x, y, z = w
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    def integrate_rotations(self, R_t0, delta_t, omega):
        S_w_delta_t = delta_t / 2.0 * self.skew_symmetric_matrix(omega)
        return R_t0 @ (1 + S_w_delta_t) @ np.linalg.inv(1 - S_w_delta_t)

    def integrate_velocity(self, translation, v, dt, R):
        return translation + R @ (v * dt)

    def pub_visualization_marker(self, pub, frame, coords, color):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        if self.cube_flag:
            marker.type = Marker.CUBE
        else:
            marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = color[0]
        marker.color.r = color[1]
        marker.color.g = color[2]
        marker.color.b = color[3]

        marker.pose.position.x = coords[0]
        marker.pose.position.y = coords[1]
        marker.pose.position.z = coords[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        pub.publish(marker)

    def timer_callback(self):
        # Calculate current transformation
        R = self.H_iterator.rotation
        t = self.H_iterator.translation
        R_new = self.integrate_rotations(R, self.dt, self.omega)
        T_new = self.integrate_velocity(t, self.v, self.dt, R_new)
        # TODO: ASK HERE?
        H_iterator = pin.SE3(R_new, T_new)

        self.H_iterator = self.H_iterator * \
            pin.exp6(np.hstack([self.v, self.omega]) * self.dt)

        # Broadcast transformations
        self.broadcast_tf(self.H_iterator, self.frames["world"],
                          self.frames["reference"])

        for i, id in enumerate(self.frames["ids"]):
            self.broadcast_tf(self.cage_tfs[i+1], self.frames["reference"], id)

        # Publish and calculate markers
        self.pub_visualization_marker(
            self.pub_marker_c1_p, self.c1_point["ref_frame"],
            self.c1_point["coords"], self.color_orange)

        center_point = self.cage_tfs[1].act(self.c1_point["coords"])
        world_point = self.H_iterator.act(center_point)

        self.pub_visualization_marker(
            self.pub_marker_w_p, self.world_point["ref_frame"],
            world_point, self.color_green)


def main(args=None):
    rclpy.init(args=args)
    node = CageTFBroadcaster()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
