# SLAM-sliding-window

2D lidar SLAM for the Cornell Electric Vehicles car (ROS 2 Humble). A sliding-window pose
graph backend corrects the drift in the EKF's fused odometry and publishes it as the TF
`map -> odom`.

```
RPLIDAR /scan ───────────────┬──────────────────────────┐
                             v                          v
lidar_odometry (ICP) ──> EKF (+ IMU) ──> /odometry/filtered ──> pgo_node ──> TF map -> odom
                                                                          └─> /slam/path
```

- **Fusion:** the EKF (`robot_localization`) fuses ICP lidar odometry and the IMU.
- **Drift correction:** `pgo_node` scan-matches each keyframe against nearby and earlier
  keyframes, and closes loops when the car revisits a place.

In simulation (2 laps, ~115 m), position error falls from 3.09 m with raw odometry to
0.12 m with the backend.

## Packages

| Path | What it is |
|---|---|
| [`src/cev_pose_graph_slam`](src/cev_pose_graph_slam/README.md) | Sliding-window pose graph backend (`pgo_node`). Its README explains the design, fixes, tests and tuning. |
| `src/cev_lidar_odometry_ros2` (submodule) | ICP lidar odometry and the car launch files (`carv2.py`) |
| `src/icp` (submodule) | CEV ICP library |
| `src/witmotion_ros2` (submodule) | WitMotion IMU driver |

## Quick start

```bash
git clone --recurse-submodules https://github.com/cornellev/SLAM-sliding-window.git
cd SLAM-sliding-window
pixi run build      # build the workspace
pixi run launch     # full car stack (lidar, IMU, EKF, ICP odometry, PGO, Foxglove)
```

To run only the backend's tests (no ROS needed):

```bash
cmake -S src/cev_pose_graph_slam -B build/pgo_core -DPGO_CORE_ONLY=ON
cmake --build build/pgo_core -j && ./build/pgo_core/pgo_sim_test
```

## Status

Scan matching, loop closure, marginalization and the `map -> odom` fix are done and tested in
simulation. Still open: drift measured on a real closed-loop bag, data-driven information
matrices, GPS factors for IGVC, a GTSAM iSAM2 comparison, and merging with SLAMv2. The
[package README](src/cev_pose_graph_slam/README.md#4-status-and-open-work) tracks the details.
