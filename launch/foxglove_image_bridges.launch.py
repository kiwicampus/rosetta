from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_hw = LaunchConfiguration("use_hw")
    max_dim = LaunchConfiguration("max_decode_dimension")
    skip_ms = LaunchConfiguration("skip_if_older_than_ms")
    metrics_period = LaunchConfiguration("metrics_period_s")
    qos_sub = LaunchConfiguration("qos_depth_sub")
    qos_pub = LaunchConfiguration("publisher_qos_depth")
    sub_rel = LaunchConfiguration("subscription_reliability")
    pub_rel = LaunchConfiguration("publisher_reliability")
    use_sim_time = LaunchConfiguration("use_sim_time")

    main_in = LaunchConfiguration("main_input")
    main_out = LaunchConfiguration("main_output")
    left_in = LaunchConfiguration("left_input")
    left_out = LaunchConfiguration("left_output")
    right_in = LaunchConfiguration("right_input")
    right_out = LaunchConfiguration("right_output")
    rear_in = LaunchConfiguration("rear_input")
    rear_out = LaunchConfiguration("rear_output")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_hw",
                default_value="true",
                description="Use NVDEC (h264_cuvid, etc.) when available",
            ),
            DeclareLaunchArgument(
                "max_decode_dimension",
                default_value="640",
                description="Max H/W for decode output (0 = no resize)",
            ),
            DeclareLaunchArgument(
                "skip_if_older_than_ms",
                default_value="800",
                description="Drop non-keyframe packets older than this (ms); -1 disables",
            ),
            DeclareLaunchArgument(
                "metrics_period_s",
                default_value="5.0",
                description="Seconds between latency metric logs (0 disables timer)",
            ),
            DeclareLaunchArgument(
                "qos_depth_sub",
                default_value="2",
                description="Subscription KEEP_LAST depth for CompressedVideo",
            ),
            DeclareLaunchArgument(
                "publisher_qos_depth",
                default_value="10",
                description="Publisher KEEP_LAST depth for Image",
            ),
            DeclareLaunchArgument(
                "subscription_reliability",
                default_value="reliable",
                description="Must match CompressedVideo publisher (rosbag play is often reliable)",
            ),
            DeclareLaunchArgument(
                "publisher_reliability",
                default_value="reliable",
                description="Image QoS reliability (match what RViz / subscribers expect)",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Must be true when playing rosbags with /clock so stamps match",
            ),
            DeclareLaunchArgument(
                "main_input",
                default_value="/camera/color/foxglove",
            ),
            DeclareLaunchArgument(
                "main_output",
                default_value="/camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "left_input",
                default_value="/video_mapping/left/foxglove",
            ),
            DeclareLaunchArgument(
                "left_output",
                default_value="/camera/color/left",
            ),
            DeclareLaunchArgument(
                "right_input",
                default_value="/video_mapping/right/foxglove",
            ),
            DeclareLaunchArgument(
                "right_output",
                default_value="/camera/color/right",
            ),
            DeclareLaunchArgument(
                "rear_input",
                default_value="/video_mapping/rear/foxglove",
            ),
            DeclareLaunchArgument(
                "rear_output",
                default_value="/camera/color/rear",
            ),
            Node(
                package="rosetta",
                executable="foxglove_image_bridge_node",
                name="foxglove_bridge_main",
                output="screen",
                parameters=[
                    {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
                    {
                        "input_topic": main_in,
                        "output_topic": main_out,
                        "use_hardware_decode": use_hw,
                        "max_decode_dimension": max_dim,
                        "skip_if_older_than_ms": skip_ms,
                        "metrics_period_s": metrics_period,
                        "qos_depth": qos_sub,
                        "publisher_qos_depth": qos_pub,
                        "subscription_reliability": sub_rel,
                        "publisher_reliability": pub_rel,
                    },
                ],
            ),
            Node(
                package="rosetta",
                executable="foxglove_image_bridge_node",
                name="foxglove_bridge_left",
                output="screen",
                parameters=[
                    {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
                    {
                        "input_topic": left_in,
                        "output_topic": left_out,
                        "use_hardware_decode": use_hw,
                        "max_decode_dimension": max_dim,
                        "skip_if_older_than_ms": skip_ms,
                        "metrics_period_s": metrics_period,
                        "qos_depth": qos_sub,
                        "publisher_qos_depth": qos_pub,
                        "subscription_reliability": sub_rel,
                        "publisher_reliability": pub_rel,
                    },
                ],
            ),
            Node(
                package="rosetta",
                executable="foxglove_image_bridge_node",
                name="foxglove_bridge_right",
                output="screen",
                parameters=[
                    {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
                    {
                        "input_topic": right_in,
                        "output_topic": right_out,
                        "use_hardware_decode": use_hw,
                        "max_decode_dimension": max_dim,
                        "skip_if_older_than_ms": skip_ms,
                        "metrics_period_s": metrics_period,
                        "qos_depth": qos_sub,
                        "publisher_qos_depth": qos_pub,
                        "subscription_reliability": sub_rel,
                        "publisher_reliability": pub_rel,
                    },
                ],
            ),
            Node(
                package="rosetta",
                executable="foxglove_image_bridge_node",
                name="foxglove_bridge_rear",
                output="screen",
                parameters=[
                    {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
                    {
                        "input_topic": rear_in,
                        "output_topic": rear_out,
                        "use_hardware_decode": use_hw,
                        "max_decode_dimension": max_dim,
                        "skip_if_older_than_ms": skip_ms,
                        "metrics_period_s": metrics_period,
                        "qos_depth": qos_sub,
                        "publisher_qos_depth": qos_pub,
                        "subscription_reliability": sub_rel,
                        "publisher_reliability": pub_rel,
                    },
                ],
            ),
        ]
    )
