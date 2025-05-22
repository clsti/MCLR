import rclpy
from rclpy.node import Node

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
   
class InteractiveMarkerNode(Node):
    def __init__(self):
        super().__init__('interactive_marker_node')
        
        # Create interactive marker
        self.server = InteractiveMarkerServer(self, 'marker_server')
        self.setup_marker()
    
        # Create a pose publisher
        # TODO publisher for pose
        
        # TODO create a tf listener
        
    def setup_marker(self):
    
        # TODO Use the tf listner to retrive the 'arm_right_7_link' wrt 'base_link' transformation
        trans = None   # TODO get the transformation 

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
        int_marker.pose.position = trans.transform.translation
        int_marker.pose.orientation = trans.transform.rotation
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
            ("rotate_x", InteractiveMarkerControl.ROTATE_AXIS, (1, 1, 0, 0)),
            ("move_x", InteractiveMarkerControl.MOVE_AXIS, (1, 1, 0, 0)),
            ("rotate_z", InteractiveMarkerControl.ROTATE_AXIS, (1, 0, 1, 0)),
            ("move_z", InteractiveMarkerControl.MOVE_AXIS, (1, 0, 1, 0)),
            ("rotate_y", InteractiveMarkerControl.ROTATE_AXIS, (1, 0, 0, 1)),
            ("move_y", InteractiveMarkerControl.MOVE_AXIS, (1, 0, 0, 1)),
        ]

        for name, mode, orientation in orientations:
            int_marker.controls.append(add_control(name, mode, orientation))

        self.server.insert(int_marker, self.handle_feedback)
        self.server.applyChanges()

    def handle_feedback(self, feedback):
        # TODO publish marker pose to ros
        None


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
