# Current Architecture

Status date: 2026-08-11. This document describes the post-cleanup repository. It separates offline paper evidence, observation-only robot sessions, and explicitly gated motion. It is not a real-robot validation certificate. The current physical release status is **NO-GO** until the staged Windows/robot checks in `REAL_ROBOT_EXPERIMENT.md` are completed and reviewed.

## 1. Frozen experiment contract

The active paper experiment is supine passive hip-knee rehabilitation through an equivalent strap pull point. The model uses:

```text
theta_shank = q_hip - q_knee
ROM protocol = ROM_PROTOCOL_V2
hip ROM = 0–120 deg
knee ROM = 5–145 deg
first-trial candidate = reference_measured_asymmetric_closed_slow only
reference-freeze robot approval = false (NO-GO)
```

`reference_measured_asymmetric_closed_nominal` remains offline and fail-closed because its frozen six-dimensional local-domain coverage is 66.334%, below the unchanged 90% threshold. `reference_closed_symmetric` and `reference_closed_c2` remain legacy software comparisons only. The completed Stage 1–6 mismatch, delay, geometry, and candidate studies remain evidence, but they are not extended by this refactor.

## 2. Layer ownership

| Layer | Primary paths | Current responsibility |
|---|---|---|
| Offline mechanics | `lower_limb_sim/kinematics.py`, `jacobian.py`, `full_dynamics.py`, `force_mapping.py` | Two-link FK/dynamics, subtractive shank angle, endpoint force/torque mapping |
| Approved reference | `lower_limb_sim/reference_cycle_closure.py`, `reference_measured_asymmetric.py`, `run_reference_measured_asymmetric.py`, `run_robot_trajectory_export.py` | Full-joint natural-cycle audit, measured-branch periodic C2 slow/nominal generation, frozen-domain check, retained absolute-calibrated export |
| Relative trajectory | `control/start_anchored_relative_trajectory.py` | Recompute pull point from `L1/L2`; apply start-anchor delta and reviewed rehab axes; fixed TCP orientation |
| Anchor/frame | `control/start_anchor.py`, `config/rehab_frame_config.json` | Observation-only StartAnchor capture, robot/tool/workpiece binding and strict reviewed/draft JSON |
| SDK observation boundary | `hardware/rokae_adapter.py`, `hardware/windows/rokae_xcore.py` | Narrow connect/state/wrench/identity interface; no motion method on the public observation adapter |
| SDK motion boundary | `hardware/rokae_motion.py` | Attach to externally prepared RT Cartesian controller; send target; one stop route |
| Acquisition | `collection/real_robot_acquisition.py` | Independent state, wrench and aligned-snapshot producers with latest caches and health |
| Persistence | `collection/episode_logger.py`, `utils/clock.py`, `utils/provenance.py` | Shared monotonic clock/Git provenance, logger-ready barrier, four independent CSV streams, bounded durable command logging, atomic metadata and global fail state |
| Safety/preflight | `safety/experiment_safety.py`, `control/execution_preflight.py` | Human-reviewed identity/tool/load/limits and complete offline/live execute gate |
| Scheduler | `control/robot_trajectory_executor.py` | Single-use slow-reference timing, deadline-bounded logging-before-dispatch, cached health checks and strict stop |
| Operator CLIs | `scripts/rokae_probe.py`, `capture_start_anchor.py`, `preview_rehab_trajectory.py`, `acquire_robot_data.py`, `run_rehab_experiment.py` | Separate observation-only/offline/execute workflows |
| Offline real-data adapter | `scripts/identify_real_episode.py` | Reviewed mapping from episode data into the unchanged five-parameter estimator |

## 3. Trajectory modes

### `absolute_calibrated`

The existing Stage 6A mode remains in `lower_limb_sim/run_robot_trajectory_export.py`. It uses reviewed absolute calibration, including the hip center. This mode was not overwritten or silently redirected.

### `start_anchored_relative`

The real first-trial mode uses the current TCP as a reviewed session anchor:

```text
p_R(t)       = [x_pull_FK(t), 0, z_pull_FK(t)]
delta_p_R(t) = p_R(t) - p_R(0)
p_tcp_B(t)   = p_tcp_start_B + R_base_from_rehab @ delta_p_R(t)
```

`RehabFrameConfig` accepts approximate unit/orthogonal rehab x/z directions, checks them, then builds a right-handed orthonormal rotation by Gram-Schmidt. Draft axes may be previewed; execute requires `reviewed=true`.

The relative builder:

- loads only `reference_measured_asymmetric_closed` for start-anchored use;
- recalculates the equivalent pull point from `L1/L2` FK;
- never consumes an observed ankle or absolute hip center;
- preserves the immutable measured raw table and both measured branches, while the closed version stays inside quantified 0.5°/2.5 mm path-deviation gates;
- makes the endpoints exactly zero only after numerical closure passes;
- freezes the captured TCP Euler XYZ orientation;
- writes row-level validity/reasons and an aggregate audit;
- never marks offline output as robot-execution approved.

The tracked `reference_release/` bundle pins the slow-reference byte SHA-256 to `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881` and pins `L1=0.42 m`, `L2=0.30 m`. Build metadata records both parent reference ID/SHA and the geometry/physical definition. The freeze explicitly sets `approved_for_first_robot_trial=false`; candidate whitelisting is not physical approval. Existing anchors bound to the old C2 slow ID cannot be relabeled and reused.

## 4. ROKAE interface split

### Observation adapter

`RokaeRobotAdapter` exposes only:

```text
connect / disconnect
start_state_stream / stop_state_stream
is_connected
read_tcp_pose
read_joint_positions
read_state_frame
read_internal_wrench
read_robot_metadata
get_robot_state_summary
```

The underlying `RokaeRobot.connect()` no longer calls `setMotionControlMode` or `setMaxCacheSize`. It uses one static connection style—construct `xMateRobot()` and then call `connectToRobot(remote_ip[, local_ip])`—loads SDK v0.7.0, reads robot identity and soft limits, starts receive-only state feedback, and reads operation state. Later observation-summary calls query toolset/load metadata, available tool/workobject names and the SDK safety-event collision state. Neither path clears alarms, changes operate mode, powers servos, calibrates, drags, or commands a target.

xCoreSDK declarations say robot initialization invokes `moveReset`, and disconnect stops robot motion before disconnecting. Constructor/connect may have other session side effects. Therefore “observation-only” here means the project does not explicitly issue a target, power command or mode change; it does **not** mean that vendor construction/connect/disconnect is proven to have zero motion-side effects. Every such session remains supervised and requires an already stationary robot.

### Motion adapter

The motion adapter is not reachable from probe, capture, preview, or acquire. `attach_externally_prepared_realtime(reviewed_filter_hz=...)` requires an explicit local IP, checks queryable automatic/power/idle state, obtains the SDK RT controller and applies only the schema-v3 safety file's reviewed filter value. It does not set automatic, power, motion mode, or network tolerance. The operator/site procedure must prepare the reviewed network tolerance externally; the inspected SDK path does not provide a confirming readback.

Confirmed static RT calls are:

```text
getRtMotionController
setFilterFrequency
setControlLoopCar
startMove(RtControllerMode.cartesianPosition)
startLoop(False)
stopLoop
stopMove
```

The repository also contains confirmed NRT `MoveLCommand`/`MoveAbsJCommand` APIs, but the new first-trial executor does not use them and never moves to a start point.

## 5. State and wrench time model

The SDK RT state fields used are `tcpPoseAbc_m`, `jointPos_m`, and `keypads`. The inspected v0.7.0 declaration contains no device timestamp, measured velocity, wrench, or collision field. Consequently:

- state time is host receive time;
- TCP/joint velocities in the low-level wrapper are host finite differences;
- wrench time is bounded by host query start/end, with midpoint and publish time;
- all new state, wrench, command and alignment timing uses `time.perf_counter_ns()`;
- missing device time remains `None`/blank.

`getEndTorque()` is called with the statically confirmed argument order:

```text
ref_type,
joint_torque_measured,
external_torque_measured,
cart_torque,
cart_force,
ec
```

It can request world/flange/tool, not base. Compensation, physical point, sign/direction and synchronization with RT state remain real-robot validation items.

## 6. Concurrency and cache ownership

```text
xCore state receive thread
  -> immutable KinematicStateFrame cache
  -> state producer -> robot_state.csv

independent getEndTorque producer
  -> immutable RobotWrenchFrame cache
  -> robot_wrench.csv

alignment producer
  -> reads both latest caches only
  -> age/skew/thread/query-duration health
  -> aligned_snapshot.csv

trajectory scheduler
  -> reads cached health
  -> logs command intent
  -> atomically replaces the RT callback target
```

State and wrench native calls no longer use the same host lock, and target replacement performs no blocking SDK health query. A blocked wrench fake therefore does not directly block state or command logging. Command intent is durably acknowledged within a reviewed deadline before dispatch; a timeout permanently fails the logger and the target is not sent. The scheduler never burst-sends overdue samples. This is a host-architecture result only; SDK native thread safety and achieved physical rates remain unverified.

Cleanup is also conservative: a partially started acquisition first signals stop and joins only threads that actually started; any still-live producer causes a lock-free failure signal and refusal to stop/disconnect the SDK. The native state-stream stop retains a timed-out thread handle rather than waiting on a lock it may own. A failed `disconnectFromRobot` retains the robot handle and connected/unknown local state so a supervised retry remains possible; it is never reported as a confirmed disconnect.

## 7. Episode schema

Every real acquire/execute episode owns:

| File | Source and key provenance |
|---|---|
| `robot_state.csv` | host receive time, six robot q, base TCP pose, valid/reason |
| `robot_wrench.csv` | query start/end/publish, six measured and external joint torques, raw Cartesian force/torque, raw frame, duration, valid/reason |
| `trajectory_command.csv` | command host time, trajectory time/phase, rehab delta, target TCP, hip/knee reference, valid/reason |
| `aligned_snapshot.csv` | state/wrench time, age, skew, thread liveness, query duration, valid/reason |
| `metadata.json` | Git commit, robot/SDK if available, complete safety snapshot/config path, ROM/reference hash, frozen `L1/L2` geometry, frame/anchor and paths, mode, fixed orientation, bound live preflight/result, episode duration, host-observed stream publish rates and logger state |

`EpisodeLogger.start()` creates all headers and the initial atomic metadata before setting its ready barrier. It refuses existing outputs, and a later `close()` cannot overwrite a foreign colliding episode. Any stream/metadata write failure or bounded-write timeout publishes a lock-free global failure signal and prevents later appends. The scheduler flushes and `fsync`s command intent before target dispatch—including the initial RT hold target—uses a race-free bounded acknowledgement, and refuses cleanup/disconnect while a native producer or pending writer may still touch the episode.

## 8. Execute state machine

The only path to `send_cartesian_target` is:

```text
offline request checks before adapter construction
  -> five-file logger ready
  -> adapter connected
  -> state/wrench/alignment healthy
  -> runtime identity/tool/payload/soft-limit/collision/current-joint preflight
  -> bind exact trajectory + complete safety digests in a live-only preflight
  -> attach to externally prepared RT controller with reviewed filter
  -> recheck connected/IDLE/identity/payload/limits/current-joint/collision/streams and live TCP
  -> fsync the initial-anchor command intent
  -> atomically record execution approval in metadata
  -> repeat the live configuration/collision/stream/anchor/deadline checks
  -> start Cartesian hold at the existing anchor
  -> fsync, then recheck health/deadline/stop intent before each remaining target
  -> request_stop(reason)
```

The preflight requires exact robot model/serial/controller agreement across the live robot, StartAnchor and reviewed safety configuration; matching anchor/config tool/workpiece declarations whose names appear in the SDK's available lists; payload mass/CoG/inertia snapshots; six joint soft limits and current-joint inclusion; a valid no-collision query; workspace and anchor tolerances; reviewed RT configuration; and all configured freshness/force/torque limits. It independently checks the exact official slow hash, pinned `L1/L2`, anchor start-q, strict boolean validity, frame transform, ROM/FK/closure and derivatives recomputed from xyz/time. Its live result is sealed to trajectory and safety digests; offline/synthetic/unbound or mutated input is rejected by the executor. The inspected API does not prove which HMI tool/workobject is active, so that remains a separate human-reviewed gate. After attachment and immediately before start, the executor rechecks identity/payload/limits/current joints/collision, acquisition health, anchor and deadline.

During the hot path it uses only cached logger/thread/state/wrench age/skew/force/torque health. The execute path does not call the wrapper's `has_motion_error()` at all, and collision is queried during live preflight, after attachment and immediately before start—not continuously during dispatch—because no bounded native call duration has been demonstrated. This missing real-time native-health evidence is one reason the physical release remains NO-GO. The executor and motion facade are single-use, publish stop intent before waiting for the lifecycle lock, retry a failed native stop, refuse late catch-up dispatch, and report completion only when the final reason is exactly `trajectory_completed` and native stop is confirmed. The trajectory is feed-forward slow C2; no wrench-based slowing or force feedback was added.

## 9. Offline identification boundary

The real-episode adapter does not modify the five-parameter model. It requires a reviewed identification config because the logger cannot infer real anthropometry, raw-wrench sign/frame rotation or delay. With sufficient data it:

1. reconstructs pull-point displacement from actual TCP relative to StartAnchor;
2. uses the frozen two-link IK with the approved 0–120/5–145 ROM;
3. reuses the offline Savitzky–Golay derivative implementation;
4. reuses `StateHistoryBuffer.linear_interpolation` for time matching;
5. rotates/signs raw force only by the reviewed mapping;
6. calls the existing bounded five-parameter estimator and metrics.

Missing review/data or optimizer failure creates no fake parameter outputs.

## 10. Evidence levels

| Claim | Current evidence |
|---|---|
| C2/ROM/FK/relative math and fail-closed gates | Offline unit/regression tests |
| probe/capture/acquire no project motion calls | Fake adapter call-contract tests and static review |
| SDK type names/signatures and available RT/NRT APIs | Local `.pyi` plus vendor examples |
| Native library loads on Windows Python 3.12 | Not tested on this macOS run |
| Real connect semantics, state rate, wrench rate/skew | Needs Windows robot validation |
| Wrench compensation/frame/reference point/sign | Needs documented real setup validation |
| RT Cartesian physical behavior/stability/stop latency | Needs supervised empty-load slow test |
| Human trial readiness | Requires institutional/site safety process beyond this repository |

Final-reference-freeze verification on 2026-08-13 completed with `667 passed, 5 skipped in 100.64 s`. The five skips are platform/environment-gated native Windows integration tests. These results do not load the Windows `.pyd`, connect to a robot, validate a physical frame/wrench, or authorize motion.
