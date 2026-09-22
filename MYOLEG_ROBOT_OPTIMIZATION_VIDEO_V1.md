# MyoLeg + robot system and offline optimization video V1

## Status and scope

MYOLEG_ROBOT_SYSTEM_VIDEO = COMPLETE_WITH_LIMITATIONS

OPTIMIZATION_DEMO = COMPLETE_WITH_LIMITATIONS

Three deterministic 1080p videos illustrate the system topology and a real four-trial offline EI run. These are prescribed-state visualizations, not controlled robot motion, strap mechanics, clinical outcomes, or evidence of patient-specific benefit. No method comparison video was produced; no methods, hyperparameters, seeds or objectives were tuned for presentation.

## Source models and coupling

ROBOT_MODEL_AVAILABLE = false. Repository inspection found no project robot MJCF/URDF, vendor mesh set or compatible local robot IK model. Hardware/SDK interfaces are not geometric robot models. ROBOT_MODEL_TYPE = SCHEMATIC, visibly labeled throughout. The arm is a position-only two-link schematic with a pedestal and wrist, not a claimed ROKAE replica or calibrated six-axis model.

The unmodified MyoLeg model is `external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml`. It retains all muscle/tendon parameters, source equalities and locked non-target joints. Its existing absolute mesh references resolve to the local MyoSuite installation under `/Users/fengjiaxin/.virtualenvs/myosuite-v2/`; that asset installation is needed to regenerate. Reuse is through `reset_to_target_state()` and the existing `prescribed_truth()` in `external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py`.

CUFF_MODEL = VISUAL_ONLY. The two cuff bands, plate, 0.15 m short connector, schematic links, bed and floor are `MjvScene` geometry only; no bodies, masses, joints, contact or forces are added to the physical model. The source RTB3 shank site defines the cuff connection. The flange target is RTB3 + `(0,-0.15,0)` m. Connector geometry terminates at RTB3. This is a visual topology, not a physically calibrated attachment or compliant strap model.

Arm shoulder = `(0.95,-0.80,1.04)` m; link lengths = 0.72 and 0.68 m. Analytic fixed-link position IK follows the target. Unreachable targets raise an error without clipping or changing the leg trajectory. No robot-orientation, joint-limit, collision-avoidance or dynamic control claim is made.

## Trajectory source and ROM

The trajectory comes from the existing native-ROM MyoLeg reference `external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv`, its frozen timing adapter and existing V3 generator. A `SubjectROMProfile` wraps those exact existing reference extrema; no new range is chosen and no source ROM file is changed.

- Hip: 28.909336926–112.025438570°.
- Knee: 18.320897422–119.500000000°.
- Original duration: 24 s, 401 source samples.
- Reference: `(0,0)`; system/contrast comparison: `(-0.03,+0.03)`.
- Both MyoLeg target coordinates use the existing identity map from project hip/knee, preserving the project convention `theta_shank=q_hip-q_knee`. MyoLeg's anatomical knee includes its original coupled coordinates; it is not replaced with an ideal planar hinge.

All video poses use original V3 sample arrays directly; playback frame skipping changes display speed only. Subtitles show source simulation time. Reference and contrast share ROM, duration and start/end q/dq/ddq. Exact q_video−q_source maxima: hip=0 rad, knee=0 rad.

## Actual optimization run

Source: `lower_limb_sim/visualization/optimization_video.py`, a narrowly scoped offline environment adapter. It calls the existing MyoLeg `prescribed_truth()` for each actually requested full trajectory. E0 uses the existing `full_cycle_dual_joint_rms()` with source timestamps. No landscape or oracle is supplied to the optimizer.

Method: `MODEL_INFORMED_BO_EI_TIMESERIES_ID` via existing `run_offline_configuration()`. Budget=4 total trajectories including reference; seed=0; default EI xi=0; baseline template unchanged. Full-series ID → five-parameter effective gray-box → residual GP → EI → next candidate. Display Trial 0 corresponds to ledger trial_index=1.

The time-series payload contains real computed MyoLeg inverse-dynamics torques transformed to *algebraically equivalent project-plane forces* using `solve(J.T,tau)` with existing project Jacobian and L1=0.42 m, L2=0.30 m. This is not a generated physical cuff sensor signal. It supports this offline software demonstration only. It does not identify physiological muscle/tendon properties. The five-parameter surrogate is structurally mismatched to MyoLeg: the reference fit reaches the existing mass-scale upper bound 1.6 and has combined torque RMSE about 3.51 N·m. Bounds and model logic were not changed.

| Display trial | beta_flex | beta_extend | Simulated E0 (N·m) | Best observed (N·m) |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 37.641921488 | 37.641921488 |
| 1 | +0.03 | -0.03 | 37.559850719 | 37.559850719 |
| 2 | -0.03 | -0.03 | 37.628306974 | 37.559850719 |
| 3 | +0.03 | +0.03 | 37.657953732 | 37.559850719 |

Relative reduction = `(reference-best)/reference*100` = **0.218030232%**. This is small and has not been established as a scientifically meaningful efficacy improvement. Accordingly, the requested `reference_vs_selected` filename contains an explicitly labeled **Reference vs V3 contrast trajectory**, not an optimized-treatment claim. Demonstrating meaningful optimization effects requires a separate controlled benchmark. The optimization video still faithfully shows the actual run and its best observed beta `(0.03,-0.03)`.

`outputs/myoleg_bo_demo_run.json` retains actual trial values, identification diagnostics and next-selection metadata. `outputs/myoleg_bo_demo_run.npz` retains each executed q/dq/ddq, time, torque and equivalent-force array. The scalar result for each trial is revealed in the animation only after completing that trajectory. No fabricated heatmap, oracle or physical sensor result is shown.

The existing scientific conclusion remains `SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED`.

## Rendering and verification

Shared encoder reused by both visualization tasks: `lower_limb_sim/visualization/encoding.py`. MuJoCo offscreen → RGB → FFmpeg libx264, H.264/yuv420p, faststart, 1920×1080 at 30 fps.

Fixed perspective camera: azimuth=55°, elevation=−28°, distance=3.65 m, lookat=(0.28,−0.20,0.96) m. Three-quarter view keeps the cuff and short connector distinguishable from the arm. No camera orbit. Only in-memory visual framebuffer settings and scene graphics are configured.

- `outputs/myoleg_robot_rehab_system_showcase.mp4`: 26 s / 780 frames; 2 s title, 11 s reference, 11 s contrast, 2 s closing.
- `outputs/myoleg_robot_reference_vs_selected.mp4`: 12 s / 360 frames; synchronized Reference vs V3 contrast.
- `outputs/model_informed_bo_optimization_demo.mp4`: 26 s / 780 frames; four 5 s trajectory replays followed by 1.5 s result holds.
- `outputs/previews/`: first, intermediate and last frames for each video.
- Same-stem JSON sidecars report source, camera, ROM, frame count and numerical error.

Six focused tests passed (three new scene/data tests plus the previous three visualization tests). They check all 401 samples across the five used beta pairs, fixed link lengths, explicit unreachable-target rejection, exact MyoLeg angles, unchanged muscle/mechanical parameters, nonblack/different renders, and recompute every displayed objective from retained executed torque arrays. Maximum visualization IK error is below 2e-16 m; the separate physical-looking flange-to-cuff connector is deliberately 0.15 m long. This numerical quantity is **visualization IK / kinematic coupling error**, not real robot tracking error.

Final MP4 decode checks passed: 780 / 360 / 780 frames respectively, all 1920×1080 H.264/yuv420p at 30 fps, zero black frames. File sizes are 3,427,960 / 2,358,243 / 3,097,993 bytes. Motion changes were detected in all three videos, and first/intermediate/final previews were visually checked.

## Regeneration

Run from the repository root. The current temporary environment contains MuJoCo 3.6.0, numpy/scipy, Pillow and imageio-ffmpeg. macOS needs access to its CoreGraphics offscreen context. No project dependency file was changed for this task. If the temporary environment is removed, recreate an isolated environment with these dependencies and the existing source MyoSuite mesh installation, then substitute its Python path.

```sh
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_myoleg_robot_video --mode system-showcase --output outputs/myoleg_robot_rehab_system_showcase.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_myoleg_robot_video --mode reference-vs-selected --output outputs/myoleg_robot_reference_vs_selected.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_myoleg_robot_video --mode optimization-demo --output outputs/model_informed_bo_optimization_demo.mp4
```

The last command reuses the recorded actual run if present. Add `--rerun-optimization` to execute the same unchanged four-trial algorithm again; `--run-data PATH` chooses its JSON/NPZ output location. Optional method comparison is not implemented or generated.

ROBOT_OR_CONTROL_CODE_MODIFIED=false; MYOLEG_DYNAMICS_MODIFIED=false; V3_MODIFIED=false; FROZEN_RESULTS_MODIFIED=false. No automatic commit.
