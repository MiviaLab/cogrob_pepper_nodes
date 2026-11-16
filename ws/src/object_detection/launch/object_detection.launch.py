import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "object_detection"
    default_config = os.path.join(
        get_package_share_directory(pkg_name), "config", "object_detector_conf.yaml")
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="YAML file selecting the detector (classic / open_vocab)",
        ),
        Node(
            package=pkg_name,
            executable="object_detection",
            name="object_detector_node",
            parameters=[config_file],
            output="screen",
        ),
    ])
