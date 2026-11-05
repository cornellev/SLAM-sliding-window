# cev_pose_graph_slam: sliding-window pose graph SLAM

Backend for the CEV car. It takes fused odometry (`/odometry/filtered`, from the EKF fusing ICP
lidar odometry and the IMU) and raw lidar scans (`/scan`). It builds a pose graph over a fixed-size
window of keyframes, adds scan-matching and loop-closure constraints, and publishes the drift
correction as the TF `map -> odom`.

```
/odometry/filtered ─┐                     ┌─> TF map -> odom
                    ├─> pgo_node ─────────┤
/scan ──────────────┘   (sliding window)  └─> /slam/path (all keyframes, map frame)
```

Sensor fusion (ICP lidar odometry + IMU) happens upstream in the EKF (`robot_localization`).
This package does not fuse sensors. It corrects the drift left in the fused odometry by
matching scans against earlier keyframes.

This document covers four things:
1. **Where the previous version went wrong**: each bug, why it matters, and the fix.
2. **How the sliding window works**: what it is, why marginalization rather than deletion, and
   how loop closure fits in.
3. **Evidence**: tests and numbers, plus how to build and run it.
4. **Status and open work**: what still needs doing.

---

## 1. Where it went wrong (and what changed)

Ordered by impact. Line references are to the old `src/SLAM_backend_math.cpp` (commit `3d6c397`).

### 1.1 The graph had nothing to optimize (the root cause)

The only edges were sequential odometry edges `k-1 -> k`. A pose graph that is a single chain has
an exact solution: every pose equals the integrated odometry. Gauss-Newton always converged to
"exactly what odometry said", so the backend could never remove drift. That isn't a tuning
problem. With no redundant constraints there is no information to correct anything with.
(The loop-closure work lived in SLAMv2. The sliding-window branch dropped it.)

**Fix:** every keyframe now gets two extra kinds of edge, both verified with ICP:
- **Local scan-matching edges** to up to 3 *non-adjacent* keyframes in the window within 3 m.
  These cut drift inside the window.
- **Loop-closure edges** to keyframes that already left the window (the archive). These are
  checked whenever the car comes back within 3 m of somewhere it has been, at least 30 keyframes
  earlier.

### 1.2 `map -> odom` was computed by subtracting poses

```cpp
t.transform.translation.x = opt.x - fused_x;            // old
q.setRPY(0, 0, opt.th - fused_th);
```
Poses are rigid-body transforms, and they compose by rotation *and* translation, so subtracting
components is only correct when the heading is 0. The comment above it even states the right
formula (`T_map_odom = T_map_base * inv(T_odom_base)`), but the code doesn't implement it.
Example from the test: odom (5, 2, 90°) and optimized pose (5.3, 1.8, 80°). The old formula puts
the robot **0.94 m** from where the optimizer said it was.

**Fix:** `map_to_odom = compose(kf.est, inverse(kf.odom))`, a real SE(2) composition.

### 1.3 `map -> odom` compared the last *keyframe* against the *current* odometry

`publish_correction(cur_x, cur_y, cur_th)` used `W_.back()` (the last keyframe, up to 0.3 m and
15° old) against the current odometry pose. As the car drove away from a keyframe, the
"correction" grew by however far it had moved since, then snapped back at the next keyframe. The
result was a 0.3 m sawtooth on the published pose even with perfect odometry.

**Fix:** the correction is computed from the *same* keyframe on both sides (its optimized pose
and its raw odometry pose), so it only changes when the optimizer changes something.

### 1.4 Map-frame and odom-frame poses were mixed

The keyframe test and the edge measurement `z` were computed from `W_.back()` (an *optimized*,
map-frame pose) and the *raw* odometry pose. As soon as the optimizer moved anything, every
new `z` included that correction, so the odometry edges started fighting the optimizer.
Commit `3d6c397` fixed the body-frame rotation of `z` but kept this frame mix. New poses were
also initialized at the raw odometry pose instead of in the map frame.

**Fix:** each keyframe stores both `odom` (raw) and `est` (optimized).
- `z = between(last.odom, odom)` uses odom on both sides.
- The keyframe test uses the same raw delta.
- New poses start at `compose(last.est, z)`, which is in the map frame.

### 1.5 "Marginalization" was deletion plus a hard pin

```cpp
W_edges_.erase(... e.i == 0 || e.j == 0 ...);  W_.pop_front();
H.coeffRef(k, k) += 1e12;   // pin new oldest pose
```
This throws away everything the oldest pose knew, then declares the next pose perfectly known
(1e12 information). Information that should flow from old constraints into the window is lost,
and the anchor becomes falsely certain, so the window can never correct it. The commit message
calls this marginalization, but marginalization keeps that information (see §2). Measured on the
same problem (§3): truncation drifts **0.19 m** from the full-batch solution. Real
marginalization stays within **0.2 mm**.

### 1.6 Numerical and robustness issues in the solver

- **Gauge fix:** a 1e12 diagonal plus 1e-6 damping gives a condition number around 1e18, at the
  limit of double precision. That is now a real prior with sigma 1 mm on the first keyframe,
  carried forward by marginalization.
- **Iterations:** the old solver ran exactly 5 Gauss-Newton steps, with no convergence check, no
  check for a step that *increases* the error, and no NaN check. The return value was ignored.
  It is now Levenberg–Marquardt: it rejects bad steps, stops on convergence, and checks for
  non-finite values.
- **Outliers:** there was no robust kernel. A single bad ICP match is now down-weighted by a Huber
  kernel instead of yanking the whole window.

### 1.7 ROS plumbing

- **TF timestamp:** TF was stamped with `now()` instead of the odometry's own stamp, which breaks
  bag replay (`use_sim_time`) and TF interpolation. It now uses `msg->header.stamp`.
- **Yaw extraction:** it assumed `qx = qy = 0`. It now uses the full yaw formula.
- **Launch remap:** `carv2.py` remaps `/odom -> /odometry/filtered`, but the node never subscribed
  to `/odom`, so the remap did nothing. The topic is now a parameter (`odom_topic`), and the
  package ships its own `config/pgo.yaml` and `launch/pgo.launch.py`.
- **Scan/odometry timing:** this one turned up while testing the new node, and SLAMv2 has it too.
  Pairing each odometry message with "the latest scan" attaches a scan up to one scan period old
  (100 ms on the RPLIDAR) to the wrong pose. At 1 m/s that's 5–10 cm and about 2° baked into
  *every* scan edge. In the ROS smoke test with perfect odometry it produced 2.8° of false
  correction. Keyframes are now taken **at scan time**, with odometry interpolated to the scan
  stamp. The same test then gives 0.02–0.35°.

### 1.8 Build problems (SLAMv2's loop closure and the vendored icp)

- **Include path:** `#include "cev_icp/icp/icp.h"` doesn't exist anywhere in the repo or the pixi
  env. The vendored headers are at `icp/include/icp/`, and `cevicp` doesn't export that path.
- **Eigen include:** `cevicp` only uses `${EIGEN3_INCLUDE_DIRS}`, which conda/pixi's Eigen leaves
  empty, so the result is `Eigen/Core: No such file`.
- **`assert`:** `icp/lib/icp/driver.cpp` uses `assert` without `#include <cassert>`, and newer
  Eigen no longer includes it transitively.

The package's `CMakeLists.txt` works around all three without touching the submodule. The last
two should also be fixed upstream in `cornellev/icp`.

The loop-closure logic in SLAMv2 had problems of its own:
- It ran ICP from an identity guess.
- It accepted matches on ICP's own cost, which is RMS over *all* points including non-overlapping
  ones.
- It `break`s after the first nearby candidate even if that one is rejected.
- It did brute-force ICP on full-resolution scans.

The new matcher:
- starts ICP from the current pose estimate,
- verifies the result independently (≥ 60 % of points within 15 cm, inlier RMSE ≤ 6 cm),
- gates how far a match may move a pose,
- tries up to 3 candidates,
- runs on voxel-downsampled scans.

### What was right
The error function and its Jacobians were correct (verified against finite differences to 3e-6).
So were the body-frame measurement convention, the angle wrapping, and the idea of a bounded
window with keyframes. All of that is kept.

---

## 2. The sliding window, explained

### Why a window at all
Full pose-graph SLAM (SLAMv2) keeps every keyframe and re-solves the whole graph. Memory and
solve time grow for as long as the car drives. On the Pi this eventually stops being real-time.
A sliding window keeps only the last `N` keyframes (`window_size`, default 20, about 6 m of
driving) in the optimizer, so cost per keyframe is **constant**: about 2 ms in the simulation
from the first lap to the last.

### The window lifecycle (per keyframe)
1. **Insert:** add keyframe `k` with an odometry edge from `k-1`.
2. **Constrain:**
   - ICP against nearby non-adjacent window keyframes (local edges).
   - ICP against nearby *archived* keyframes (loop closure).
3. **Optimize:** run Levenberg–Marquardt over the window, using all edges plus the
   marginalization prior.
4. **Slide:** if the window holds more than `N` keyframes, marginalize the oldest one into the
   prior and move it to the archive.

### Marginalization: why you can't just delete the oldest pose
Write the linearized problem over the oldest pose `m` and the rest of the window `r` as

```
[ H_mm  H_mr ] [dx_m]   = - [ b_m ]
[ H_rm  H_rr ] [dx_r]       [ b_r ]
```

Eliminating `dx_m` (the Schur complement) gives an equivalent system over the remaining poses
only:

```
H_prior = H_rr - H_rm H_mm^-1 H_mr
b_prior = b_r  - H_rm H_mm^-1 b_m
```

Solving with `H_prior, b_prior` gives the **same** `dx_r` as solving the full system. Nothing is
lost, apart from the linearization point being frozen (see trade-offs below). So instead of
deleting pose `m`, we keep `(H_prior, b_prior)` as a **prior factor** on the poses `m` was
connected to (its Markov blanket). That prior carries everything the dropped poses and edges
said. It also replaces the 1e12 "pin": the window is anchored by real, accumulated uncertainty
instead of a fake certainty.

In code (`SlidingWindowPGO::marginalizeOldest`):
1. Collect every factor touching `m`: odometry/scan edges, map edges, and the existing prior.
   The existing prior is always folded in, so there is only ever one prior.
2. Build `H, b` over `[m, blanket]` at the current estimate.
3. Take the Schur complement and store it as `LinearPrior{ids, x0, H, g}`.
4. During optimization the prior contributes cost `2 gᵀdx + dxᵀ H dx`, with `dx = x - x0` and
   angles wrapped.

Truncation versus marginalization on the same problem (window 6, edges spanning 3 keyframes):

| window strategy | max error of newest pose vs. full batch |
|---|---|
| **marginalization (new)** | **0.0002 m** |
| truncation + hard pin (old) | 0.19 m |

### Loop closure with a sliding window
Keyframes that leave the window go to an **archive** with their final pose and scan. When the car
returns, a new keyframe is ICP-matched against nearby archived keyframes. A verified match becomes
a **map edge**: a constraint from a *fixed* archived pose to the new keyframe. This snaps the
window, and therefore `map -> odom`, back onto the map built on the previous lap.

### Trade-offs to know about
- **Archived poses are frozen.** A loop closure corrects the *current* estimate and everything
  after it. It doesn't go back and bend the trajectory from earlier laps (full SLAMv2 does).
  For a car that needs a consistent `map` pose *now*, this is usually the right trade.
  If you need an optimized historical map, run a one-shot full optimization over the archive at
  the end.
- **Fixed linearization point.** The prior is linearized when it is created, and later
  relinearization of those poses is approximate. This is the usual sliding-window caveat
  (cf. "first-estimate Jacobians"). In 2D at these window sizes the effect is in the 0.2 mm range
  shown above.
- **Prior fill-in.** The prior is dense over its blanket. With a 20-keyframe window that's at most
  a 60×60 matrix, which is negligible.
- **Archive memory.** The archive grows by one downsampled scan per keyframe, roughly 5 KB.
  An hour of driving is a few tens of MB.

### Tuning (all in `config/pgo.yaml`)
| parameter | effect |
|---|---|
| `window_size` | bigger means more smoothing and slower solves; 10–40 is reasonable |
| `keyframe_translation/rotation` | keyframe spacing; smaller means more keyframes and more ICP |
| `local_match_radius`, `max_local_matches` | how many in-window scan edges per keyframe |
| `loop_search_radius` | must exceed the drift accumulated over one lap |
| `loop_min_keyframe_gap` | stops recent keyframes from counting as a "loop" |
| `match_min_fitness`, `match_max_rmse` | ICP acceptance: tighten if you see bad loop closures |
| `use_marginalization` | `false` reproduces the old truncation behaviour, for comparison only |

---

## 3. Evidence

`test/sim_test.cpp` builds without ROS. It drives a simulated car two laps (~115 m) around a
5 m-wide corridor loop with pillars, using a 360-beam lidar (1 cm noise) and odometry with 2 %
scale error plus a +0.25°/m heading bias. It runs the real `cevicp` ICP.

```
[1] Jacobians:            max |analytic - numeric| = 3.5e-06
[2] map->odom:            old formula 0.94 m off, new formula exact
[3] window vs batch:      marginalization 0.0002 m, truncation 0.19 m
[4] end-to-end            ATE[m]  KFs  scan  loop  ms/KF(first 50)  ms/KF(last 50)
  raw odometry             3.089
  old backend              2.869
  new, no loop closure     0.285  377  1122     0        1.5            1.9
  new, truncation + loops  0.239  377  1122   201        1.8            2.8
  new, marginalization     0.121  377  1122   200        1.8            2.5
```

Error drops from 3.1 m to 0.12 m (about 26× less), and per-keyframe time stays flat.
`docs/sim_result.svg` plots the trajectories.

ROS smoke test (perfect odometry, 20 Hz; scans 10 Hz, stamped between odometry samples):
- The node runs, keyframes at scan times, and publishes `/slam/path`.
- It closes loops on every revisit.
- After 2+ laps it holds `map -> odom` at about 1 cm and 0.35°.

That 0.35° is the sampling bias of point-to-point ICP in a bare square room. It is the
same 0.3° measured without ROS.

---

## 4. Status and open work

| Item | Status |
|---|---|
| Constraints that can disagree with odometry (scan matching, loop closure) | Done |
| Schur-complement marginalization | Done |
| Robust kernel (Huber) on scan and loop edges | Done |
| Correct `map -> odom` composition | Done |
| ATE with vs. without the backend, simulation | Done: 3.09 m → 0.12 m |
| End-point drift % / ATE on a real closed-loop bag from the car | Open: the number that matters most |
| Information matrices from data instead of constants | Open. Use the covariance of each relative step between keyframes, or the ICP fit quality. The EKF's absolute pose covariance grows over time, so it can't be used directly. |
| GPS factors for the IGVC course | Open. GPS fits as a unary constraint on a pose, like the loop-closure map edges. |
| Benchmark against GTSAM iSAM2 on the same bag | Open |
| Merge `SLAM-sliding-window` and `SLAMv2` into one repo | Open |
| Point `carv2.py` at `pgo.launch.py` | Open |
| Upstream the icp build fixes (`<cassert>`, Eigen target) to `cornellev/icp` | Open |

## Build, test, run

```bash
# ROS-free core + simulation test (works anywhere with Eigen + a C++17 compiler)
cmake -S src/cev_pose_graph_slam -B build/pgo_core -DPGO_CORE_ONLY=ON
cmake --build build/pgo_core -j
./build/pgo_core/pgo_sim_test src/cev_pose_graph_slam/docs/sim_result.svg

# ROS 2 package (inside the pixi env: `pixi shell`)
colcon build --packages-select cev_pose_graph_slam --cmake-args "-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
source install/setup.bash
ros2 launch cev_pose_graph_slam pgo.launch.py            # add use_sim_time:=true for bags
```

`carv2.py` (in the `cev_lidar_odometry_ros2` submodule) still starts `pgo_node` without
parameters. It works, because the defaults match `pgo.yaml`. To get the YAML, point it at
`pgo.launch.py` or pass `parameters=[<share>/config/pgo.yaml]`.

## Files

| file | role |
|---|---|
| `include/.../sliding_window_pgo.hpp`, `src/sliding_window_pgo.cpp` | ROS-free backend: SE(2) math, LM solver, marginalization, loop closure |
| `include/.../icp_scan_matcher.hpp`, `src/icp_scan_matcher.cpp` | `cevicp` wrapper with independent match verification; scan filtering |
| `src/SLAM_backend_math.cpp` | ROS 2 node (`pgo_node`): TF, topics, scan/odometry time sync |
| `test/sim_test.cpp` | unit and end-to-end simulation tests (`pgo_sim_test`) |
| `config/pgo.yaml`, `launch/pgo.launch.py` | parameters and launch |
