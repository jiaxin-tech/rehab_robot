# Real ROKAE Experiment Procedure

This is a phased engineering checklist, not a substitute for the robot manual, emergency-stop validation, institutional approval, clinical supervision, or a trained operator. No real connection or motion was performed during this repository refactor.

## 0. Stop conditions before starting

Do not proceed if any of the following is unresolved:

- robot model/controller/SDK version is unknown or incompatible;
- Windows Python is not CPython 3.12 x64 for the bundled extension;
- for execute, a non-empty reviewed Windows RT network-interface IP is unavailable;
- tool, payload, center of mass, workobject or active HMI project is unreviewed;
- physical E-stop, safety controller, collision settings or operator supervision is unavailable;
- bed/robot layout, subject restraint or equivalent strap pull point differs from the frozen experiment;
- rehabilitation axes, StartAnchor, safety limits or wrench semantics are unreviewed;
- the pinned slow-reference SHA-256 or frozen `L1=0.42 m` / `L2=0.30 m` geometry does not match;
- reviewed RT filter/network-tolerance evidence is absent or differs from the external controller preparation;
- the robot is not already at the reviewed anchor before execute;
- the requested trajectory is not `reference_measured_asymmetric_closed_slow`;
- a diagnostic/logger/stream reports invalid, stale, dead-thread or write failure.

The software stop path is only an additional request to the SDK. It is not a safety-rated stop.

## 1. Frozen physical and mathematical scope

- Supine passive hip-knee flexion/extension.
- Equivalent pull point at the shank strap.
- No ankle-as-pull-point substitution.
- `theta_shank = q_hip - q_knee`.
- Approved ROM: hip 0–120°, knee 5–145°.
- Fixed TCP orientation captured at the session start.
- First motion: measured-flexion/measured-extension periodic C2 slow only, 24 s / 401 samples.
- Feed-forward trajectory only; no force feedback, automatic slowing, online personalization or model adaptation.

## 2. Phase A — offline software release

On the development machine:

```bash
python3 -m pip install -r requirements.txt
python3 -B -m pytest -q
```

Release gate:

- full suite passes;
- Windows/native tests are skipped unless their explicit platform/environment gate is satisfied;
- no deleted legacy import is present;
- `reference_measured_asymmetric_closed_slow` passes closure, ROM, FK, finite, C2, measured-asymmetry and subtractive-angle tests;
- config templates remain `reviewed=false` with physical limits/axes `null`.

The current slow reference is approved by the byte SHA-256 `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`, not by comparison against a second read of the same mutable file. The present `lower_limb_sim/` source and its ignored data are not protected by the cleanup tag; commit the intended source and independently preserve the source/manifest/pinned CSV before treating any Git commit field as complete experiment provenance. Any StartAnchor bound to the old `reference_closed_c2_slow` ID is invalid for this reference and must be recaptured, not relabeled.

The repository result recorded on 2026-08-11 is `640 passed, 5 skipped in 96.90 s`, with `645` tests collected. The skips are explicit Windows/platform integration gates; this is offline evidence only.

## 3. Phase B — Windows SDK import, no robot

Use Windows 10/11 x64 and CPython 3.12 x64. Confirm the runtime is loaded from `hardware/windows/xcoresdk`, not another global SDK install.

```powershell
python -B -m pytest -q tests/test_rokae_xcore.py
```

Expected static/runtime boundary:

- SDK reports 0.7.0;
- xMateRobot and RT Cartesian declarations load;
- fake controller tests pass;
- no real IP connection is attempted.

If the `.pyd` cannot load, stop. Do not replace it with a fake success branch.

## 4. Phase C — supervised observation-only connection probe

Ensure the physical robot is stationary and the operator controls the session. Then:

```powershell
python -m scripts.rokae_probe --ip ROBOT_IP
```

The project probe calls connect/disconnect, receive-state, pose/q reads, one internal-wrench query, robot identity/soft-limit/toolset/load/available-tool/workobject reads, and a safety-event collision query. It does not explicitly send a target, call power/automatic/clear-error/calibration/drag, or invoke MoveL/MoveJ/RT motion.

Release gate:

- connection and cleanup are both confirmed;
- base TCP pose and six robot joints are finite;
- wrench includes query start/end/publish and a supported raw frame;
- robot model/controller/SDK values are recorded;
- observed connection/disconnection side effects are acceptable under the vendor/site procedure.

The vendor declarations say robot initialization invokes `moveReset` and disconnect can stop motion. Constructor/connect may have further session effects. Treat this phase as supervised with an already stationary robot: “observation-only” describes the project calls, not a guarantee of zero vendor-side motion effects.

## 5. Phase D — timing and wrench diagnostics

Run the existing observation diagnostics under the same stationary/supervised vendor-session restriction and preserve their CSV/JSON reports:

```powershell
python -m scripts.check_rt_state_timing --duration 10
python -m scripts.check_wrench_query_timing --duration 10 --target-hz 50
python -m scripts.check_snapshot_alignment --duration 10
```

Review actual, not configured, values:

- state source update cadence and dropped sequence IDs;
- state age distribution;
- `getEndTorque` query duration/deadline misses;
- wrench publish cadence;
- state/wrench skew and invalid reasons;
- thread faults, future timestamps or non-increasing times.

The configured 8 ms state interval and 50 Hz wrench rate are targets only until these reports exist.

## 6. Phase E — tool/load and wrench semantics

Do not call `calibrateForceSensor()` as normal startup. With the controller tool/load already configured and the mechanism stationary/unloaded, a session-local software bias may be evaluated only after explicit operator confirmation.

Then perform small manual known-direction checks in at least two non-collinear directions:

```powershell
python -m scripts.check_wrench_frame_rotation --direction X --confirm-unloaded
python -m scripts.check_wrench_frame_rotation --direction Y --confirm-unloaded
python -m scripts.check_wrench_frame_rotation --direction Z --confirm-unloaded
python -m scripts.check_wrench_pose_dependence --poses 3 --confirm-unloaded
```

Record, but do not infer automatically:

- sign of robot-on-leg force;
- raw world/flange/tool axes;
- `baseFrame()` direction and Euler convention;
- gravity/payload compensation behavior;
- Cartesian torque reference point;
- pose dependence and query timing.

Only a human-reviewed documented procedure may change `BASE_WRENCH_ROTATION_VERIFIED`. No current script changes it.

## 7. Phase F — rehabilitation frame review

Copy/edit `config/rehab_frame_config.json`:

```json
{
  "rehab_x_axis_in_base": [null, null, null],
  "rehab_z_axis_in_base": [null, null, null],
  "reviewed": false,
  "notes": ""
}
```

Measure rehab +x/+z in robot Base, supply approximately unit and orthogonal axes, document the method, and preview them. The loader re-orthogonalizes them; it does not determine physical direction. Set `reviewed=true` only after an independent human check.

## 8. Phase G — StartAnchor capture and review

The operator first places the stationary robot at the intended subject/session start by the approved external procedure. The program never drives to this point.

```powershell
python -m scripts.capture_start_anchor `
  --ip ROBOT_IP `
  --output anchors/SUBJECT_SESSION.json `
  --anchor-id SUBJECT_SESSION `
  --tool-name REVIEWED_HMI_TOOL_NAME `
  --workpiece-name REVIEWED_HMI_WORKPIECE_NAME
```

The schema-v2 output contains creation/capture time, base TCP pose, fixed Euler orientation, six robot joints, C2 start hip/knee values, trajectory ID, robot model/serial/controller, the operator-declared tool/workpiece names, notes and `reviewed=false`. The CLI reads identity but cannot prove that the declared tool/workpiece is active. Older schema anchors are rejected; recapture rather than hand-converting them.

Review:

- subject/session/trajectory/anchor IDs;
- robot model/serial/controller and the exact HMI tool/workpiece names;
- physical strap location and limb posture;
- TCP position/orientation and robot joint posture;
- robot has not moved during capture;
- the anchor belongs to this exact frame/tool/bed arrangement.

Only then manually set `reviewed=true`. A later execute command must supply the same `--anchor-id` and the robot must already match the pose within reviewed tolerances.

## 9. Phase H — pure offline preview

```powershell
python -m scripts.preview_rehab_trajectory `
  --anchor anchors/SUBJECT_SESSION.json `
  --frame-config config/rehab_frame_config.json `
  --output-dir previews/SUBJECT_SESSION
```

Review the CSV, Git-qualified JSON and four plots:

- first and last relative displacement are exactly zero;
- first and last TCP targets equal the anchor;
- orientation is constant;
- TCP/path/derivatives are finite;
- no obvious position/velocity/acceleration jump;
- ROM is 0–120 / 5–145;
- `theta_shank=q_hip-q_knee` and pull FK pass;
- the pinned slow-reference SHA and frozen `L1/L2` geometry/physical definition match;
- workspace and direction agree with the physical bed.

Preview never connects to hardware and never approves execution.

## 10. Phase I — observation-only five-file acquisition

```powershell
python -m scripts.acquire_robot_data `
  --ip ROBOT_IP `
  --episode-dir data/SUBJECT/acquire_001 `
  --duration-s 30
```

Expected files:

```text
robot_state.csv
robot_wrench.csv
trajectory_command.csv
aligned_snapshot.csv
metadata.json
```

For acquire, the command CSV contains its header and no motion commands. Verify independent state/wrench row growth, blank-not-zero unavailable fields, query duration, ages/skew, thread liveness, metadata Git commit, episode duration, host-observed average publish rates and clean cleanup. These rates are observations, not the controller/device clock. A blocked native query causes fail-closed cleanup that refuses to race a still-live producer; do not reuse the process or robot session until the operator has recovered it.

## 11. Phase J — reviewed safety configuration

Fill schema-v3 `config/experiment_safety.json` from site-specific evidence. Required fields include:

- expected robot model, serial number and controller version;
- reviewed HMI tool and workpiece names;
- payload mass, CoG and inertia copied from the reviewed controller setup;
- six reviewed controller soft-limit pairs;
- maximum TCP speed, acceleration and command lateness;
- maximum allowed start-anchor position/orientation errors;
- force and torque magnitudes;
- maximum state age, wrench age and state/wrench skew;
- base workspace minimum/maximum;
- reviewed RT command-filter frequency and the network-tolerance percentage prepared externally;
- separate reviewed flags for identity, tool/workpiece, payload, collision configuration, joint soft limits and RT configuration;
- overall `reviewed=true` and review notes.

All repository physical defaults are `null` and all review flags are `false`; a reviewed flag alone cannot bypass a missing or malformed value. Confirm that the live robot reports the same identity, payload and soft limits, that every current joint is inside those limits, and that the collision query is valid and reports no collision. The SDK query proves only that the reviewed tool/workpiece names are available; it does not prove the active HMI selection, which must be checked separately by the operator. The program applies only `reviewed_rt_filter_hz`; it records but cannot read back the externally prepared network tolerance. Do not copy forces, workspaces, RT values or load properties from simulation.

## 12. Phase K — empty-load slow RT release

Before any human is in the robot workspace:

1. Confirm physical E-stop and safety controller behavior.
2. Confirm the robot is exactly at the reviewed StartAnchor.
3. Confirm live robot identity, payload, soft limits and collision configuration against both the StartAnchor and reviewed safety file; separately verify the active HMI tool/workpiece because the SDK only reports available names.
4. Through the approved external HMI/vendor procedure, prepare automatic mode, servo power, RT command mode and the exact reviewed network tolerance. The program does not set those values; during explicit attach it applies only the reviewed command-filter frequency.
5. Keep an operator at the E-stop.
6. Use a reviewed conservative safety config for this empty-load setup.
7. Run exactly one `reference_measured_asymmetric_closed_slow` execution.

```powershell
python -m scripts.run_rehab_experiment `
  --mode execute `
  --enable-motion `
  --ip ROBOT_IP `
  --local-ip REVIEWED_WINDOWS_RT_NIC_IP `
  --episode-dir data/empty_load/slow_001 `
  --anchor anchors/EMPTY_LOAD.json `
  --anchor-id EMPTY_LOAD `
  --frame-config config/rehab_frame_config.json `
  --safety-config config/experiment_safety.json `
  --trajectory reference_measured_asymmetric_closed_slow `
  --operator-confirmation "I CONFIRM SUPERVISED SLOW ROBOT MOTION"
```

The runner fails before connection when static review/flag/whitelist/local-IP gates fail. Its offline preflight cannot be passed to motion. The live preflight binds the exact trajectory digest and complete safety digest, pins the official slow SHA and `L1/L2`, validates anchor start-q and strict boolean rows, and recomputes velocity/acceleration from xyz/time. After connection it requires healthy streams/logger, exact live identity/payload/soft-limit/current-joint/collision evidence, matching reviewed tool/workpiece declarations, and those names in the SDK's available lists. It checks configuration/collision/health/anchor after attach, flushes and `fsync`s the initial-anchor target, records approval, then repeats those checks and the absolute deadline immediately before starting the hold. Every remaining on-time command is synchronized, followed by fresh health/deadline/stop-intent checks before dispatch; late targets are never burst-sent, and all terminal paths use `request_stop(reason)`. Stop failures are retryable but remain failures and are never reported as completion.

The execute path does not consume the wrapper's `has_motion_error()` query at all. Collision is queried during live preflight, after RT attachment and immediately before start, not continuously in the hot scheduler, because no bounded native call duration has been demonstrated. Controller collision protection must already be configured and active; this code is not its replacement. Until Windows trials establish native call behavior, RT stability and stop latency, the repository remains NO-GO for human motion.

Release evidence must include actual path tracking, achieved command/state/wrench timing, stop latency, invalid rows, frame/sign checks and operator observations. Static `.pyi` evidence is insufficient.

## 13. Phase L — non-human mechanical-surrogate release

After the empty-load evidence is independently reviewed, repeat the complete frame/anchor/tool/load/safety process with a documented dummy-leg or equivalent non-human mechanical surrogate. Validate strap retention, load direction, reachable workspace, tracking, force/torque response, stale-data/logger stops and physical E-stop behavior. Any configuration change invalidates the prior anchor and safety review.

Do not proceed to a human trial until both empty-load and surrogate evidence are accepted by the site safety process.

## 14. Phase M — human trial release

Human use is outside the automatic authority of this codebase. Proceed only after the relevant ethics, clinical, institutional, risk-assessment and trained-operator approvals are complete.

Repeat the subject-specific frame/anchor/safety review. Begin with the slow whitelist only. Do not enable nominal, fast or C1–C8 based on offline results alone.

Abort immediately on pain, unexpected posture, strap motion, tool/bed movement, stale data, dead thread, logger failure, force/torque limit, tracking error, controller error or operator concern. Preserve the incomplete episode and its stop reason.

## 15. Phase N — offline real-episode identification

After a valid episode, copy `config/real_identification_config.json` to `EPISODE_DIR/identification_config.json`. Supply reviewed subject anthropometry, raw-wrench frame, raw-to-rehab rotation, robot-on-leg sign and assumed delay. No defaults are supplied.

```bash
python -m scripts.identify_real_episode EPISODE_DIR
```

On success it writes `identified_parameters.json` and `prediction_metrics.csv`, including the source-episode and identification-code Git commits. If review/data/optimizer evidence is insufficient, it writes neither; never substitute virtual-subject values as a real identification result.
