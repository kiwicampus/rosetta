# Running rosetta Inside DevContainer

The devcontainer provides a pre-configured environment for running rosetta policies. This guide explains how to set up and run the model with two different data sources: **rosbag playback** or **live robot connection via rosboard**.

## Initial Setup

1. **Build and source the workspace:**

```bash
cd /workspace
bash setup.sh
```

2. **Source the ROS WS:**

```bash
cd ~/rosetta_ws
source install/setup.bash
```

## Running the Policy Bridge

The policy bridge can run with data from two sources:

### Option 1: Using a Rosbag (MCAP)

This mode replays recorded data from a rosbag file:

1. **Launch the policy bridge:**

```bash
ros2 launch rosetta turtlebot_policy_bridge.launch.py log_level:=info
```

2. **In a separate terminal, play the rosbag:**

```bash
ros2 bag play my_rosbag.mcap --loop --clock
```

The `--loop` flag repeats playback, and `--clock` publishes simulation time.

3. **Start the policy:**

```bash
ros2 action send_goal /run_policy rosetta_interfaces/action/RunPolicy \
  "{prompt: 'your task description here'}"
```

### Option 2: Live Robot Connection via Rosboard

This mode connects to a live robot through rosboard:

1. **Configure rosboard client:**

Ensure `rosboard_client` package has a `topics_to_subscribe.yml` configuration file. Example:

```yaml
url: ws://balena_uuid.balena-devices.com:80
topics:
  [
    /odometry/local,
    /imu/data_abs_heading,
    /camera/color/image_raw
  ]
topics_to_stream:
  [
    /motion_control/speed_controller/reference_cmd
  ]
```

2. **Launch the policy bridge:**

```bash
ros2 launch rosetta turtlebot_policy_bridge.launch.py log_level:=info
```

3. **Start the rosboard client** (in a separate terminal):

```bash
ros2 run rosboard_client rosboard_client
```

4. **Start the policy:**

```bash
ros2 action send_goal /run_policy rosetta_interfaces/action/RunPolicy \
  "{prompt: 'your task description here'}"
```

## Optional: Policy Visualizer

The policy visualizer node overlays velocity commands on camera images, making it easy to see what actions the policy is taking. It subscribes to:
- Camera image topic (default: `/camera/color/image_raw`)
- Velocity command topic (default: `/motion_control/speed_controller/reference_cmd`)

And publishes an overlayed image to `/policy_visualizer/image` showing the camera feed with velocity command visualization.

**Run the visualizer:**

```bash
ros2 run rosetta policy_visualizer_node
```

**Visualize in rviz2:**

1. Launch rviz2:
```bash
rviz2
```

2. Add an Image display:
   - Click "Add" → "By topic" → select `/policy_visualizer/image`
   - Or click "Add" → "Image" → set the Image Topic to `/policy_visualizer/image`

The overlay will show velocity commands (linear and angular) as visual indicators on the camera image, helping you understand what the policy is doing in real-time.

## Notes

- The policy bridge uses `use_sim_time: True` by default in the launch file, which is required for rosbag playback with `--clock`.
- For live robot connections, ensure your network can reach the rosboard server URL.
- The contract file (`fomo_test2.yaml` by default) defines which topics the policy subscribes to and publishes.
