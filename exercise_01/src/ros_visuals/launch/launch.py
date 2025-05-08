import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'ros_visuals'
    rviz_config_file = os.path.join(
        'src', package_name, 'rviz',
        'cage_rviz.rviz'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_file],
            output='screen'
        ),
        # Node(
        #    package='ros_visuals',
        #    executable='t11',
        #    name='t11',
        #    output='screen'
        # )
    ])
