# MYOLEG_KNEE_ROM_COMPATIBILITY_AUDIT_V1

## Decision

`LIMITED_125DEG_EXTENSION_MECHANICALLY_CONTINUOUS`

This is an offline numerical and structural continuity conclusion over the limited extension required by the frozen current reference. It is **not** physiological validation to 125 deg and does not support the project's full 145 deg search domain.

## Frozen protocol and provenance

- Protocol: `ROM_EXTENSION_PROTOCOL_V1`, SHA-256 `e01ba6e0d01b85c8ff3d41e5d27db55d23dfabeb549408210ae4e1925a73810e`.
- Primary derived upper limit: 125 deg, fixed before results.
- Baseline: actual native upper limit 120.000281 deg.
- Stress-only upper limit: 130 deg; never used to authorize the formal reference.
- Native-compatible target: 119.5 deg, fixed before results.
- Frozen formal reference SHA-256: `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- Native derived XML SHA-256: `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d`.
- 125 derived XML SHA-256: `c652424679308411fb73a211ad1fc770002fd760c8339c1ed9553888c14e0d41`.
- 130 stress-only XML SHA-256: `d8007d0a65c1d49a988a181c1fd251d0766f6f25cecc89e20f7030aefa444151`.
- Environment: Python 3.10.19, MyoSuite 2.12.2, MuJoCo 3.6.0.

The 125 and 130 XML files are generated from the frozen project-owned supine XML. The only XML content change is the upper range value of `knee_angle_r`. All 80 muscle actuators, 80 tendons, bodies, joints, 14 source knee/patella equalities, and the additional supine constraints remain present.

## Common-domain invariance

`LIMIT_EXTENSION_MODEL_INTEGRITY = PASS`

Across 72 matched states (four hips, native-compatible knee grid, and zero/positive/negative velocities), the largest differences were:

```text
{
  "actuator_force": 0.0,
  "actuator_length": 0.0,
  "equality_residual": 0.0,
  "moment_arm_hip": 0.0,
  "moment_arm_knee": 0.0,
  "normalized_actuator_length": 0.0,
  "qfrc_actuator": 0.0,
  "qfrc_passive": 0.0,
  "tendon_length": 0.0
}
```

The predeclared absolute tolerance was 1.0e-10. Thus changing the joint upper limit did not alter common-domain tendon length, actuator length, normalized length, moment arms, passive/actuator forces, or source equality residuals.

## 120-125 deg continuity

- Relevant right-side hip/knee actuators checked: 31.
- Maximum geometry derivative-jump ratio at 120 deg: 0.0374139 (limit 0.25).
- Maximum moment-arm derivative-jump ratio: 0.0223789 (limit 0.5).
- Maximum muscle-force derivative-jump ratio: 0.0104752 (limit 1.0).
- Passive knee-torque derivative-jump ratio: 0.00926183 (limit 1.0).
- Maximum extension/native passive-force growth ratio: 1.29048.
- Normalized actuator-length range: -0.023995 to 0.851556.
- Minimum tendon/actuator length: 0.080581 m.
- State-grid rows: 396; warnings: 0; all finite: True.

The 130-deg stress-only grid contributed 180 rows, all finite=True, warnings=0, maximum source-equality residual=1.9984e-15, and minimum tendon length=0.072884 m. These observations are robustness context only and do not authorize 130 deg for the formal reference.

Crossing the original native range value did not introduce a detected discontinuity under the pre-frozen criteria. This means the existing geometry/equality polynomials extrapolate smoothly over the narrow interval; it does not establish anatomical validity outside the source model's native calibration.

## Patella/equality and low-control sweep

Four zero-muscle-control sweeps were driven at fixed hip 30/60/90/110 deg over `100 -> 125 -> 100 deg`. The driver used `qfrc_applied`, separately recorded from muscle actuator force.

- Total steps: 80004.
- Maximum knee tracking error: 0.757489 deg.
- Maximum source equality residual: 0.00068217.
- Maximum source equality force: 481.561.
- Maximum auxiliary round-trip inconsistency: 0.00019252.
- Maximum required diagnostic torque: 66.1036 N m.
- Solver warnings: 0; finite: True.

The auxiliary/patella mechanism remained intact under the low-speed diagnostic sweep. These constraint forces are numerical model quantities, not robot or tissue loads.

## Current formal-reference state path

`REFERENCE_STATE_PATH_VALID_IN_125_MODEL = PASS`

All 401 unchanged formal-reference states were evaluated without time integration or replay. The original 108 above-native points were explicitly tagged. Abnormal points: 0; maximum source equality residual: 2.66454e-15; maximum force/model-Fmax ratio: 0.380136; warnings: 0.

The `actuator_lengthrange`-normalized transmission coordinate crossed the context band at 123 low-flexion native-domain states and at 0 above-native states. This field is not physiological normalized muscle-fiber length and is therefore retained as context rather than used as a hard validity gate. All target-relevant physical tendon/actuator lengths stayed positive.

## Native-compatible diagnostic reference

The independent `MYOLEG_NATIVE_ROM_REFERENCE_CANDIDATE` uses the globally affine, invertible transformation:

```text
q_k,new(t) = q_k,0 + s * (q_k,formal(t) - q_k,0)
s = 0.950328715412
```

Hip, duration (24 s), 401 samples, phase columns, extrema timing, asymmetric branch topology, starting pose, endpoint closure, and C2 continuity are preserved. No pointwise clipping is used. Candidate SHA-256: `208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678`.

Distortion relative to the formal reference:

- Knee RMS / maximum difference: 3.712562 / 5.286604 deg.
- Knee velocity RMS / maximum difference: 0.623399 / 1.479679 deg/s.
- Knee acceleration RMS / maximum difference: 0.327342 / 0.913516 deg/s2.
- Pull-point RMS / maximum displacement: 19.433195 / 27.670774 mm.
- Project Jacobian maximum condition number, formal / candidate: 14.973214 / 14.971741.
- Joint, velocity, and acceleration closure errors: 0.000e+00, 0.000e+00, 0.000e+00.

This candidate is `active_reference=false`, diagnostic-only, and not robot-approved. It does not replace the frozen formal reference.

## Direct answers

### Q1

Yes. The 120-125 deg region is mechanically continuous under the limited derived-model extension according to the frozen numerical and structural criteria.

### Q2

No discontinuity was detected in common-domain mechanics, muscle/tendon geometry, moment arms, passive force, or patella constraints at the original 120 deg boundary.

### Q3

Yes, as an explicitly caveated offline modeling condition on the limited-125 derived model; it is not physiological validation.

### Q4

The 119.5-deg candidate differs by 3.712562 deg RMS and 5.286604 deg maximum at the knee, with 19.433195 mm RMS pull-point displacement.

### Q5

Recommended primary condition for the next replay stage: **original formal reference on the limited-125 derived model, with the native-compatible candidate as a required sensitivity condition**.

### Q6

Yes, an offline MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1 may proceed only with both the original-on-125 primary condition and the native-compatible sensitivity condition predeclared.

No replay, candidate landscape, BO, PINN, RL, robot connection, or 145-deg model test was performed here.

## Tests

18 passed, 0 failed. Stage-test status: `PASS`.
