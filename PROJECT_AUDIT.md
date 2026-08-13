# PROJECT AUDIT — pre real-robot experiment cleanup

Audit date: 2026-08-09 (Asia/Shanghai)

This audit was completed before deleting source files. It records the current
working tree, import/CLI/test relationships, the frozen paper scope, and the
evidence boundary for the bundled ROKAE xCoreSDK. It is not a claim of physical
robot validation.

## 1. Git and working-tree checkpoint

- Branch: `simulation`
- HEAD: `8e0c3de507e0f08ee35818c83c259c227add7d36`
- Rollback tag: `pre_real_robot_experiment_cleanup`
- Cleanup-before baseline: `508 passed, 4 skipped in 98.53 s`
- The four skipped tests are Windows-native SDK tests. No robot was connected
  and no hardware command was sent.

The rollback tag protects the committed HEAD only. The following user work was
already uncommitted and must not be overwritten:

| Path | State before cleanup | Protection boundary |
|---|---|---|
| `requirements.txt` | modified; adds pandas, matplotlib, pytest, Pillow | not contained in the tag |
| `bone_return_3_leg.csv` | untracked source data | not contained in the tag |
| `lower_limb_sim/` | untracked current paper code | not contained in the tag |
| `lower_limb_sim/data/` | ignored, about 5.3 GB / 2,958 files | not contained in Git history or the tag |
| `logs/` | ignored, 104 user-generated logs | not contained in Git history or the tag |

The audit counted 3,284 project files after excluding `.git`, `.venv`, Python
caches, and pytest cache. Of these, 2,958 are ignored `lower_limb_sim/data`
artifacts and 104 are ignored logs. No C/C++ project source was found; native
code is supplied only as vendor `.pyd`, `.dll`, `.so`, `.lib`, and `.exp`
artifacts with `.pyi` declarations.

## 2. Frozen paper mainline

The current paper is supine passive hip-knee rehabilitation through an
equivalent strap pull point on the shank. The immutable model convention is:

```text
theta_shank = q_hip - q_knee
```

The formally approved ROM is hip 0–120 degrees and knee 5–145 degrees. At this
pre-cleanup audit snapshot, the closed C2 references were present under
`lower_limb_sim/data/reference_candidates/`:

- `reference_closed_c2_slow.csv`: 401 samples, 24 s, then-current first-trial reference.
- `reference_closed_c2_nominal.csv`: 401 samples, 12 s, closed, retained offline.

Both references preserve `theta_shank = q_hip - q_knee`, use the configured
`L1/L2` forward kinematics for `x_pull_m/z_pull_m`, explicitly state that the
observed ankle is not the pull point, and carry the 0–120 / 5–145 approval.
`ROM_PROTOCOL_V2` now supplies the single formal 0–120 / 5–145 range to the
workspace, IK, reference, candidate, identification and robot-preview gates.
Pre-migration 5–130 outputs remain inactive legacy evidence, not active inputs.

Post-audit update on 2026-08-11: both files above are now legacy software
comparisons. The active offline-approved source is
`reference_measured_asymmetric_closed_slow.csv` (401 samples, 24 s, SHA-256
`f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`), built
from the selected measured 5844→5895→5934 flexion-return cycle. The corresponding
nominal reference remains fail-closed because frozen local-domain coverage is
66.334%, below the unchanged 90% gate.

## 3. Major-module disposition

Status meanings:

- `KEEP`: part of the current mainline or required infrastructure.
- `MERGE`: useful functionality must be retained while obsolete coupling is removed.
- `DELETE`: outside the frozen scope after its callers/tests/docs are removed.
- `LEGACY_BUT_KEEP`: completed offline evidence or vendor material, not an active runtime.
- `UNKNOWN_REQUIRES_REVIEW`: user artifact whose provenance cannot be safely inferred.

| Module/path | Status | Imports / imported by | CLI and tests | Paper-mainline decision |
|---|---|---|---|---|
| `lower_limb_sim/kinematics.py`, `jacobian.py`, `full_dynamics.py`, `quasi_static_dynamics.py`, `force_mapping.py` | KEEP | NumPy and local config; used across Stage 1–6 | dedicated regression tests | current mechanics; preserve FK and subtractive shank convention |
| `lower_limb_sim/parameter_estimator.py`, `identification_dataset.py`, `identifiability_analysis.py` | KEEP | NumPy/Pandas/SciPy and five-parameter dynamics | `run_identification.py` and estimator tests | current five-parameter identification; do not replace with PINN |
| `lower_limb_sim/timestamp_alignment.py`, `state_history_buffer.py`, `causal_sample_matcher.py`, delay estimators | KEEP | local data/derivative utilities | extensive offline tests | required for real-episode offline alignment |
| `lower_limb_sim/reference_*`, `run_reference_*`, candidate evaluation | KEEP | kinematics, dynamics, Pandas/SciPy | Stage 5/6 CLIs and regression tests | current reference/C2/candidate pipeline; slow is the first-trial whitelist |
| `lower_limb_sim/robot_coordinate_transform.py`, `run_robot_trajectory_export.py`, `robot_trajectory_audit.py` | KEEP | local FK/audit and SciPy rotation | offline absolute-calibrated export tests | retain `absolute_calibrated`; add relative mode without overwriting it |
| Completed mismatch, geometry-error, variable-delay modules and artifacts | LEGACY_BUT_KEEP | only the offline Stage 1–6 package | offline experiment/test entry points | completed evidence; frozen against further scenario expansion, not deleted |
| `hardware/windows/rokae_xcore.py` | MERGE | imports project state types; lazily imports SDK | hardware tests default-skip outside Windows | keep adapter implementation, make connection truly read-only, publish required adapter interface, separate motion preparation |
| `hardware/windows/rokae_internal_wrench.py` | KEEP | project state/config; called by diagnostics/collection | fake-adapter and diagnostic tests | current independent `getEndTorque()` source; preserve raw/timing/invalid fields |
| `hardware/windows/rokae_force_sensor.py` | DELETE | compatibility alias only | no current direct caller | misleading external-sensor legacy name; remove after exports are updated |
| `hardware/windows/xcoresdk/` | KEEP | runtime loaded by `rokae_xcore.py` | Windows import tests | actual bundled CPython 3.12 x64 runtime copy |
| `hardware/xcoresdk_python-v0.7-2.0/` | LEGACY_BUT_KEEP | not imported by active wrapper | vendor examples are API evidence | retain complete vendor package/examples; directory name is not evidence of SDK 2.0 |
| `collection/state.py`, `snapshot.py` | KEEP | standard library + settings | snapshot/frame tests and diagnostics | current typed, time-qualified state/wrench contract |
| `collection/collector.py` | MERGE | snapshot + trajectory projection | old collection tests | retain logging mechanics; remove comfort/pain labeling coupling and supersede active format with real episode files |
| `collection/trajectory.py` | MERGE | NumPy + old settings | old geometry/calibration tests | keep generic projection only if useful; remove old sweep/excitation/hip-center calibration paths |
| `collection/safety_guard.py` | KEEP | snapshot + settings + robot stop abstraction | stale/collision tests | preserve runtime safety infrastructure; it is not experiment approval and must not supply guessed first-trial limits |
| `scripts/check_*`, `scripts/rokae_diagnostic_common.py` | KEEP | hardware adapter, wrench source, snapshot | cross-platform fake tests | read-only diagnostics; never enable, move, calibrate, drag, or change verification flags |
| `scripts/run_collection.py` | DELETE | old sweep/excitation collector and SafetyGuard | old automatic collection CLI | unsafe/out-of-scope entry: enables power and moves before the new anchor/frame/safety/slow gates |
| `scripts/run_control.py` | DELETE | PINN + ComfortNet + MPC + old excitation | old automatic control CLI | wrong paper trajectory and no explicit motion enable gate |
| `models/pinn.py`, `scripts/train_pinn.py` | DELETE | PyTorch; called only by old training/control/tests | old PINN CLI | explicitly frozen out of scope |
| `models/comfort_net.py`, `scripts/train_comfort.py` | DELETE | PyTorch; called only by old training/control/tests | old Comfort CLI | explicitly frozen out of scope |
| `control/mpc_controller.py` | DELETE | SciPy optimize + old models/config | old MPC test/control CLI | explicitly frozen out of scope; `control/` package remains for deterministic trajectory execution |
| `tests/test_units_and_frames.py` | MERGE | mixes valuable state/wrench/safety checks with deleted ML/MPC/old collector paths | 12 tests before cleanup | retain hardware/frame/staleness regressions; replace obsolete assertions with new experiment tests |
| `tests/test_rokae_diagnostics.py` | KEEP | fake adapter and diagnostic functions | 5 cross-platform tests | current read-only evidence |
| `tests/test_rokae_xcore.py` | MERGE | native SDK on Windows | four default-skipped tests | keep hardware import/signature tests; require explicit opt-in for any real connection test |
| `config/settings.py` | MERGE | imported by hardware/collection/diagnostics | no standalone CLI | keep hardware/acquisition defaults; remove PINN/Comfort/MPC/old trajectory settings; experiment limits move to reviewed JSON |
| `requirements.txt` | MERGE | environment specification | exercised by all offline tests | remove dead `torch`; retain NumPy/SciPy/Pandas/Matplotlib/Pytest/Pillow |
| `README.md` | MERGE | documentation | current commands are stale | rewrite around the frozen offline-to-real mainline and fail-closed gates |
| `utils/logger.py` | MERGE | imported by old runtime and SafetyGuard | import currently creates log files | retain logging helper but remove import-time filesystem side effects |
| `utils/signal_processing.py` | UNKNOWN_REQUIRES_REVIEW | no in-repository caller or direct test | no CLI | overlaps lower-limb derivative utilities, but external use is unconfirmed; do not delete in this pass |
| `bone_return_3_leg.csv` | UNKNOWN_REQUIRES_REVIEW | source for skeleton-reference work is plausible but not inferred as disposable | no active direct CLI default | user-owned untracked input; retain unchanged |
| `logs/` | UNKNOWN_REQUIRES_REVIEW | generated historical logs | no current import | ignored user artifacts; retain unchanged |

No source/data implementation was found for `physics_informed`, `upper_limb`,
`dobot`, `admittance`, `single_knee`, or `old_reference`. PINN/MPC mentions in
`lower_limb_sim` are negative scope/provenance statements and must not be
mistaken for active dependencies.

## 4. ROKAE xCoreSDK evidence

### Static facts established from local files

- Runtime directory: `hardware/windows/xcoresdk/`.
- Native build: Windows x64 CPython 3.12; SDK reports expected version 0.7.0.
- Robot class declaration: `xMateRobot(Cobot_6)`; constructors with no address
  or `(remoteIP, localIP='')` are declared.
- Both constructor styles occur in bundled vendor examples. The current
  wrapper constructs with the IP and then calls `connectToRobot(ec)`; the
  repeat-connect runtime semantics need Windows/robot verification.
- RT fields are `tcpPoseAbc_m`, `jointPos_m`, and `keypads`; TCP pose is base
  expressed in m/rad. No device timestamp, velocity, wrench, or collision field
  is declared in the RT field list.
- Actual local `getEndTorque` declaration:

  ```text
  getEndTorque(ref_type,
               joint_torque_measured,
               external_torque_measured,
               cart_torque,
               cart_force,
               ec)
  ```

  The wrapper uses this order. Its Cartesian expression frame may be world,
  flange, or tool. Compensation semantics, reference point, and hardware
  synchronization remain unverified.
- Static motion API evidence exists for NRT `MoveLCommand` / `MoveAbsJCommand`
  plus `moveAppend` / `moveStart`, and RT Cartesian callback control via
  `getRtMotionController`, `setControlLoopCar`,
  `startMove(RtControllerMode.cartesianPosition)`, `startLoop`, `stopLoop`, and
  `stopMove`. This establishes API existence, not physical suitability.

### Gaps found before refactoring

1. `RokaeRobot.connect()` changes motion mode and command cache, so the current
   diagnostic connection is not strictly query-only.
2. Public names do not yet provide the requested `RokaeRobotAdapter` contract;
   state-stream start/stop are private.
3. State, wrench, and motion checks share one SDK lock. A blocking wrench query
   can therefore delay state reads or a motion-target health check.
4. State interval 8 ms and wrench target 50 Hz are configured targets only;
   no real diagnostic report in this checkout proves achieved rates.
5. No device timestamp is available in the inspected SDK fields.
6. The present active motion CLIs lack frame/anchor/safety review, operator
   confirmation, slow-only whitelist, logging-first, and explicit
   `--enable-motion` gates.

## 5. Cleanup gates

Files marked `DELETE` may be removed only in one dependency-consistent change:
their imports, CLI documentation, tests, settings, and `torch` requirement must
be removed together. Files and ignored artifacts marked `KEEP`,
`LEGACY_BUT_KEEP`, or `UNKNOWN_REQUIRES_REVIEW` are not deletion targets.

All later hardware work must preserve these evidence boundaries:

- offline/fake tests are not real-robot evidence;
- probe, acquisition, anchor capture, and preview must never power or move;
- `BASE_WRENCH_ROTATION_VERIFIED` is never set automatically;
- missing SDK/state/wrench values remain `None`/blank with an invalid reason;
- no software stop replaces the physical emergency stop, safety controller,
  trained operator, or reviewed site-specific limits.
