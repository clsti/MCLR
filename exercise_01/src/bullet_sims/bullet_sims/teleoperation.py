import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener
from tf2_ros import Buffer
from tf2_ros import TransformException
from rclpy.executors import ExternalShutdownException

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class InteractiveMarkerNode(Node):
    def __init__(self):
        super().__init__('interactive_marker_node')

        # Create interactive marker
        self.server = InteractiveMarkerServer(self, 'marker_server')

        # Create a pose publisher
        self.pub = self.create_publisher(PoseStamped, "/marker_pose", 10)

        # Creat subscriber for homing trigger
        self.sub_home_trg = self.create_subscription(
            Bool, "/trigger_homing", self.home_trigger, 10)

        # Flag for homing position
        self.homing_flag = False

        # tf listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Frames for transformation
        self.from_frame = "arm_right_7_link"
        self.to_frame = "base_link"

        self.trans = None
        self.create_timer(1.0, self.get_trans)

    def home_trigger(self, msg):
        self.homing_flag = msg

    def get_trans(self):
        # retrive the 'arm_right_7_link' wrt 'base_link' transformation
        # initialize when robot in home position
        if self.tf_buffer._getFrameStrings() and self.trans is None and self.homing_flag:
            self.trans = self.tf_buffer.lookup_transform(
                self.to_frame,
                self.from_frame,
                rclpy.time.Time())
            self.setup_marker()

    def setup_marker(self):

        trans = self.trans

        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position.x = trans.transform.translation.x
        int_marker.pose.position.y = trans.transform.translation.y
        int_marker.pose.position.z = trans.transform.translation.z
        int_marker.pose.orientation.x = trans.transform.rotation.x
        int_marker.pose.orientation.y = trans.transform.rotation.y
        int_marker.pose.orientation.z = trans.transform.rotation.z
        int_marker.pose.orientation.w = trans.transform.rotation.w
        int_marker.scale = 0.2
        int_marker.name = "marker"
        int_marker.description = "Interactive Marker"

        def add_control(name, mode, orientation):
            control = InteractiveMarkerControl()
            control.name = name
            control.interaction_mode = mode
            control.orientation.w = orientation[0]
            control.orientation.x = orientation[1]
            control.orientation.y = orientation[2]
            control.orientation.z = orientation[3]
            return control

        # Add controls
        visible_control = InteractiveMarkerControl()
        visible_control.always_visible = True
        visible_control.markers.append(marker)
        visible_control.interaction_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(visible_control)

        orientations = [
            ("rotate_x", InteractiveMarkerControl.ROTATE_AXIS, (1.0, 1.0, 0.0, 0.0)),
            ("move_x", InteractiveMarkerControl.MOVE_AXIS, (1.0, 1.0, 0.0, 0.0)),
            ("rotate_z", InteractiveMarkerControl.ROTATE_AXIS, (1.0, 0.0, 1.0, 0.0)),
            ("move_z", InteractiveMarkerControl.MOVE_AXIS, (1.0, 0.0, 1.0, 0.0)),
            ("rotate_y", InteractiveMarkerControl.ROTATE_AXIS, (1.0, 0.0, 0.0, 1.0)),
            ("move_y", InteractiveMarkerControl.MOVE_AXIS, (1.0, 0.0, 0.0, 1.0)),
        ]

        for name, mode, orientation in orientations:
            int_marker.controls.append(add_control(name, mode, orientation))

        self.server.insert(int_marker, feedback_callback=self.handle_feedback)
        self.server.applyChanges()

    def handle_feedback(self, feedback):
        # publish marker pose to ros
        pose_msg = PoseStamped()
        pose_msg.header = feedback.header
        pose_msg.pose = feedback.pose
        self.pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveMarkerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
