import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("cev_pose_graph_slam"), "config", "pgo.yaml")
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="cev_pose_graph_slam",
            executable="pgo_node",
            name="pgo_backend",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
        ),
    ])
