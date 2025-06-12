import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'ros_visuals'
    rviz_config_file = os.path.join(
        'src', 'exercise_01', package_name,
        'rviz', 'talos_world.rviz'
    )

    urdf_file = os.path.join(
        'src', 'exercise_01', 'talos_description',
        'robots', 'talos_reduced_no_hands.urdf'
    )

    with open(urdf_file, 'r') as urdf_file:
        robot_description = urdf_file.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )
    ])
