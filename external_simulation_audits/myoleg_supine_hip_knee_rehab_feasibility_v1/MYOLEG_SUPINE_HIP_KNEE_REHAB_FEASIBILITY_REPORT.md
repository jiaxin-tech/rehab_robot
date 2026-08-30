# MYOLEG_SUPINE_HIP_KNEE_REHAB_FEASIBILITY_V1

## Formal decision

`MYOLEG_SUPINE_REHAB_MODEL_FEASIBLE_WITH_LIMITATIONS`

The project-owned derived model is a reproducible, headless, offline feasibility model. It is not a physiological-passivity, human, clinical, robot-control, or safety validation. The unchanged formal reference is **not** replayed because the ROM precheck fails closed.

## Frozen input and provenance

- Environment: Python 3.10.19, MyoSuite 2.12.2, MuJoCo 3.6.0.
- Source XML SHA-256: `27a9bec4544acfc15fec2bda2b410820ed1cbc9fe45fcbb44cb4c2ba8de6f91e`.
- Derived XML SHA-256: `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d`.
- Frozen reference SHA-256: `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- The previous install/smoke-test checksum manifest verified 14 files with no failure.
- All 22 referenced upstream assets were hashed before and after the run; no source or asset hash changed.
- Source dimensions: {'nq': 35, 'nv': 34, 'nbody': 16, 'njnt': 29, 'neq': 14, 'nu': 80, 'ntendon': 80}.
- Derived dimensions: {'nq': 35, 'nv': 34, 'nbody': 16, 'njnt': 29, 'neq': 27, 'nu': 80, 'ntendon': 80}.

## Derived-model design

The right leg is frozen as `TARGET_LEG`: the source model is bilaterally symmetric, and the right tibia has directly auditable RTB sites that can be compared with the project's strap-equivalent `L2=0.30 m` meaning. The free root is preserved but constrained by a site-to-world weld at `[0,0,1] m` with a -90 deg world-y supine rotation. The contralateral primary joints, target hip adduction/rotation, and target ankle/subtalar/MTP are locked with single-joint equality constraints at native zero. Target `hip_flexion_r`, `knee_angle_r`, all seven right auxiliary/patella joints, all 14 source knee equalities, all 80 tendons, and all 80 muscle actuators remain present.

`SUPINE_NO_BED_CONTACT` disables only floor/terrain contact masks. Gravity and the 1 ms timestep remain unchanged; no bed contact was introduced. Weld-to-world and joint-equality locking are documented MuJoCo equality mechanisms: <https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality>.

## Coordinate gate

`COORDINATE_MAPPING_VALID = PASS`

The exact fixed mapping is:

```text
q_project_hip  = +1 * q_myoleg_hip_flexion_r + 0
q_project_knee = +1 * q_myoleg_knee_angle_r  + 0

q_myoleg_hip_flexion_r = q_project_hip
q_myoleg_knee_angle_r   = q_project_knee

theta_shank = q_project_hip - q_project_knee
```

The rehabilitation frame is MyoLeg world x-z after the root transform, with the `hip_r` site as the local origin. Across 20/60/100 deg diagnostic poses, maximum measured thigh-angle error was 0.000065 deg and maximum shank-angle error was 1.380 deg. Round-trip joint-coordinate error was zero.

## ROM compatibility gate

`REFERENCE_SLIGHTLY_EXCEEDS_MYOLEG_NATIVE_ROM`

- Project `ROM_PROTOCOL_V2`: hip 0-120 deg, knee 5-145 deg.
- Native mapped MyoLeg range: hip -30.000013 to 120.000281 deg; knee 0.000000 to 120.000281 deg.
- Common formal/native range: hip 0-120 deg; knee 5-120.000281 deg.
- Frozen reference range: hip 28.909337 to 112.025439 deg; knee 18.319125 to 124.786604 deg.
- Inside native knee range: 293/401; outside: 108/401 (26.933%).
- Maximum exceedance over the actual native upper limit: 4.786323 deg.
- Exceeding interval: indices 155-262, t=10.540-16.824 s, crossing flexion then extension. Peak knee 124.786604 deg occurs at t=15.160 s in extension.

No clipping, scaling, XML limit extension, project-ROM change, or reference modification was used. `REFERENCE_REPLAY_BLOCKED_BY_ROM`.

## Passive-state audit

P0 is explicitly a `LOW_ACTIVATION_OR_ZERO_CONTROL_MUSCULOSKELETAL_CONDITION`, not a physiological passive human. With all 80 controls zero, the run remained finite with no solver warning and retained nonzero residual mechanics: final `qfrc_passive` L2=1.008956, `qfrc_actuator` L2=205.584279, and maximum absolute muscle actuator force=380.865410. The nonzero actuator force at zero command is a model residual/passive-muscle response, not voluntary activation.

The unsupported free-space leg fell from hip/knee 28.826/30.319 deg to -30.081/56.230 deg and reached the native hip extension boundary. This is a material limitation of `SUPINE_NO_BED_CONTACT`, not evidence of physiological resting posture.

## 2-DOF motion and knee integrity

The diagnostic controller applies generalized PD torque only through `qfrc_applied`; it is separated in every row from muscle `qfrc_actuator` and is not a formal controller.

- `HIP_ONLY`: hip RMSE 1.179 deg, knee RMSE 0.327 deg, equality peak 6.077e-05, warnings 0.
- `KNEE_ONLY`: hip RMSE 1.168 deg, knee RMSE 0.299 deg, equality peak 6.077e-05, warnings 0.
- `COMBINED`: hip RMSE 1.168 deg, knee RMSE 0.304 deg, equality peak 6.077e-05, warnings 0.

All diagnostic motions were finite and continuous. Source equality definitions/hashes are unchanged. The largest source-knee equality error during controlled motion was 6.077e-05. During the 10,000-step P0/static test, `knee_angle_r_beta_rotation1_constraint` briefly reached 0.039864 at t=0.381 s as hip flexion crossed to -31.186 deg; it recovered to 6.077e-05. This recovered boundary-impact transient is retained as a limitation. `DERIVED_KNEE_MODEL_INTEGRITY = PASS` for the controlled rehabilitation-motion diagnostic, not an unconditional dynamics certification.

The 10,000-step static run covered 10.0 simulated seconds with zero warnings, root drift 1.494e-06 m / 0.006700 deg, maximum locked-joint deviation 6.559e-04 rad, and no nonfinite state or model explosion.

## Strap and external-force path

`PROVISIONAL_STRAP_SITE = RTB3` on `tibia_r`.

It was selected by geometry rather than by name: local position `[0.0114, -0.2952, 0.0554] m`, distance from `knee_r` 0.300570 m, 75.119% of the knee-to-ankle distance, and only 0.570 mm from project `L2=0.30 m`. The site remains provisional until physical strap placement is defined.

Four 2 N pulses (+x/-x/+z/-z) applied with `mj_applyFT` produced finite hip/knee acceleration and no warning. The `J_mujoco^T F` comparison matched `qfrc_applied` with maximum absolute error 0.000e+00. The project analytic two-link `J_project^T F` and 3-D MyoLeg values differed by at most 10.159% at this pose. `EXTERNAL_STRAP_FORCE_PATH_AVAILABLE = true`; `FORCE_MAPPING_FEASIBILITY = PASS`. This is a kinematic/dynamic interface smoke test, not robot-force validation.

## Direct answers

### Q1

Yes, for headless offline feasibility: a pelvis-fixed, right-target-leg, sagittal hip-knee derived MyoLeg can run without deleting or changing its muscle/tendon/knee-equality structure. The zero-control boundary transient prevents an unrestricted claim.

### Q2

Use the identity joint mapping shown above, with +1 signs, zero offsets, the fixed -90 deg world-y root transform, and `theta_shank=q_hip-q_knee`.

### Q3

No. The frozen reference is not fully compatible with the actual mapped native knee ROM.

### Q4

108 of 401 samples exceed the native knee upper limit. The maximum exceedance is 4.786323 deg, over t=10.540-16.824 s; the peak is 124.786604 deg at t=15.160 s during extension.

### Q5

Yes, P0 is finite and nontrivial, but only as a zero-command model condition. It falls to the native hip boundary and must not be called physiological passive human behavior.

### Q6

Yes. `RTB3` is a physically interpretable provisional tibial site because its knee-center distance is 0.300570 m, close to project `L2=0.30 m`.

### Q7

Yes. Small sagittal external forces can be applied at RTB3 and mapped exactly through the MyoLeg site Jacobian to hip/knee generalized force; the simplified project-Jacobian comparison is close but not identical.

### Q8

Yes, under the controlled diagnostic motions. All auxiliary/patella joints and 14 source equalities remain intact and finite; the recovered zero-control boundary transient is explicitly retained as a limitation.

### Q9

No, not for an unchanged full formal-reference replay. The derived interface is feasible, but `MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1` remains blocked by the native-ROM conflict. No replay is run here.

## Test and scope closure

- Tests: 15 passed, 0 failed.
- No MyoSuite upstream file or asset changed.
- No formal reference, `ROM_PROTOCOL_V2`, lower-limb model, five-parameter model, BO, formal artifact, hardware, control, or safety code changed.
- No BO, PINN, RL, candidate landscape, formal replay, robot connection, or visualization-based gate was run.
- Visualization remains unavailable and is not used as evidence.
