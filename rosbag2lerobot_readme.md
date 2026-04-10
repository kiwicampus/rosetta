
# rosetta — ROS 2 ⇄ LeRobot bridge

**rosetta** is a ROS 2 package that standardizes the interface between ROS 2 topics and LeRobot policies using a small YAML **contract**. It provides:

- **bag_to_lerobot.py** — converts recorded bags into a ready-to-train **LeRobot v3** dataset 

## Install & Build

#### Create workspace + venv (system Python)

```bash
# Create a venv exclusive for this tool. The .gitignore for the venv is inside rosetta 

cd /workspace/rover/ros2/src/fomo/src/rosetta

python3.10 -m venv ./venv
touch ./venv/COLCON_IGNORE
source ./venv/bin/activate

#The venv es requiered because lerobot installs a version of torch and torchvision different from the one we use
pip install -r requirements.txt

# Source ROS and build
cd /workspace/rover/ros2
colcon build --packages-select rosetta
```

Verify after install (optional): `which lerobot-train` to confirm CLI is on `PATH`.


### How to run

To test the tool you need a rosbag with its corresponding metadata.yaml file. Make sure they are both stored in the same folder. 

Convert a single bag:

    `python bag_to_lerobot.py \\
        --bag /path/to/my_bag \\
        --contract /path/to/contract.yaml`
    
    # Output: out_lerobot/my_bag/

Convert multiple bags from a splits folder (containing split0, split1, ...):

    `python bag_to_lerobot.py \\
        --bags /path/to/session_folder \\
        --contract /path/to/contract.yaml`
    
    # Output: out_lerobot/session_folder/

Custom output root:

    `python bag_to_lerobot.py \\
        --bags /path/to/session_folder \\
        --contract /path/to/contract.yaml \\
        --out /custom/output/root`
    
Each bag folder must have an mcap file with its metadata file. In case you trimmed a rosbag and want to obtain a metadata.yaml for it, run: `ros2 bag reindex /mcap_folder/ ` (the mcap file inside mcap_folder/ must be called data_0.mcap)
---

## Contracts

A **contract** is a small YAML that declares which topics, message types, fields, rates, and timing rules a policy consumes and what it publishes. The current state of the contract should allow you to record datasets.

Important considerations: 
- All cameras of type foxglove_msgs/msg/CompressedVideo must have stamp: foxglove 
- Make sure to define the correct image size in the contract. For instance, resize: [720,1280]
- Rate_hz is interpreted as an int

---

## Tasks

Tasks are defined in the metadata.yaml of each mcap at the end: 

```bash
  custom_data: 
    lerobot.operator_prompt: rover_diff_drive
```

If not defined, the task is set as "" 

---

## Face & License-Plate Anonymization

`bag_to_lerobot.py` can automatically blur faces and license plates in every
image frame before they are written to the dataset.  The feature uses the
[offline-anonymization](https://github.com/kiwicampus/offline-anonymization)
submodule and a dedicated **TF 1.15 + CUDA 10.0** conda environment
(`anon_env`) that lives inside the Docker image.  The model is loaded
**once** at startup and reused across all episodes (GPU-accelerated,
batch-processed per episode).

### One-time setup

**1. Initialise the submodule** (if you cloned without `--recurse-submodules`):

```bash
git submodule update --init ros2_workspace/src/offline-anonymization
```

**2. Rebuild the Docker image**

The Dockerfile installs Miniconda and the `anon_env` automatically.

```bash
# Default — GPU (CUDA 10.0 managed by conda, compatible with CUDA 12 driver)
docker compose -f .devcontainer/docker-compose.yml build

# CPU-only machines (no NVIDIA GPU)
ANON_GPU=false docker compose -f .devcontainer/docker-compose.yml build
```

> **CUDA compatibility note**: the container's NVIDIA driver (≥525, required
> for CUDA 12.4) is fully backward-compatible with the CUDA 10.0 runtime that
> conda installs for TF 1.15.  No conflict with the system CUDA 12 libraries.

**3. Model weights** are downloaded automatically on first run to
`ros2_workspace/src/offline-anonymization/weights/`.

---

### Running with anonymization

```bash
python ros2_workspace/src/rosetta/scripts/bag_to_lerobot.py \
    --bags /path/to/session_folder \
    --contract /path/to/contract.yaml \
    --out /path/to/output \
    --anonymize
```

Force CPU inference (slower, no GPU required):

```bash
python bag_to_lerobot.py \
    --bags /path/to/session_folder \
    --contract /path/to/contract.yaml \
    --anonymize \
    --no-anonymize-gpu
```

**What happens internally:**

1. A persistent `anonymize_daemon.py` subprocess is spawned inside `anon_env`.
2. The face + plate TF 1.15 models are loaded **once**.
3. For each episode: all image frames are collected into a buffer, sent to the
   daemon as a single batch, and the anonymized frames are written back before
   `add_frame` is called.
4. The daemon is shut down cleanly when all episodes are processed.

**Tunable env vars** (advanced):

| Variable | Default | Description |
|---|---|---|
| `ANON_WEIGHTS_DIR` | `offline-anonymization/weights/` | Path to pre-downloaded weights |
| `ANON_FACE_THRESHOLD` | `0.3` | Detection confidence threshold for faces |
| `ANON_PLATE_THRESHOLD` | `0.3` | Detection confidence threshold for plates |