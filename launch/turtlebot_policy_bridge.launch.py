from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('rosetta')
    contract = os.path.join(share, 'contracts', 'fomo_simple.yaml')

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level'
    )

    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value='/workspace/pretrained_model_1cam',
        description='Local LeRobot model directory'
    )

    return LaunchDescription([
        log_level_arg,
        policy_path_arg,
        Node(
            package='rosetta',
            executable='policy_bridge_node',
            name='policy_bridge',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'contract_path': contract},
                {'policy_path': LaunchConfiguration('policy_path')},
                {'use_sim_time': True},
                {'bridge_debug': True},
                {'use_chunks': True},
                {"actions_per_chunk": 25},
                {"max_queue_actions": 25},
                {"chunk_size_threshold": 0.8},
                {"debug_chunk_log_dir": "chunk_logs"},
            ],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
    ])