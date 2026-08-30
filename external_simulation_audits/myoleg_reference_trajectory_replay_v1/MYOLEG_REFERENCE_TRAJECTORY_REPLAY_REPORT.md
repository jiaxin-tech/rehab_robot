# MYOLEG REFERENCE TRAJECTORY REPLAY V1

## Final determination

`MYOLEG_REFERENCE_REPLAY_VALID_WITH_LIMITATIONS`

This is an **offline headless virtual-model replay**, not a physiological-passive,
human, robot, safety, comfort, efficacy, or clinical result.  P0 means zero
muscle control and zero initial activation in this model; it does not mean a
physiological passive patient.

## Frozen inputs and replay

- PRIMARY: unchanged 24 s / 401-point formal reference on the frozen limited-125 XML.
- SENSITIVITY: frozen native-compatible 119.5-degree candidate on the frozen native supine XML.
- Both conditions use the model's 0.001 s timestep, zero warmup, identical P0 state,
  identical quintic q/dq/ddq interpolation, and the same pre-existing diagnostic
  generalized-PD replay. No pointwise clipping, rescaling, controller tuning, or
  stabilization adjustment occurred.
- Formal reference SHA-256: `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- PRIMARY model SHA-256: `c652424679308411fb73a211ad1fc770002fd760c8339c1ed9553888c14e0d41`.
- SENSITIVITY reference/model SHA-256: `208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678` /
  `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d`.

## Frozen generalized-force truth

`TAU_MY0LEG_REQUIRED_DRIVE` is the external hip/knee generalized drive, in N m, required to
realize the prescribed q, dq, ddq under P0 after accounting for inertia, bias,
MuJoCo passive force, zero-control muscle actuator force, and constraint force.
The full 34-DOF equation is

`r = M(q) qacc + qfrc_bias - qfrc_passive - qfrc_constraint - qfrc_actuator`.

Because seven right knee/patella coordinates are polynomially constrained to the
main knee angle, the frozen two-coordinate truth is **not** the naive hip/knee
slice of `r`. It is `tau_truth = T(q)^T r`, where T is the constraint-consistent
velocity tangent. This is the virtual-work projection into project hip/knee
coordinates. `qfrc_inverse = M qacc + bias - passive - constraint` is checked
explicitly before subtracting P0 actuator force.

MuJoCo's official documentation defines `qfrc_inverse` as the net external force
and documents its relation to applied and actuator forces:
[Computation](https://mujoco.readthedocs.io/en/latest/computation/index.html) and
[API types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).

## Independent-path validation

Method A is prescribed-state inverse dynamics plus explicit full-EOM accounting
and constraint-tangent projection. Method B is an independently integrated,
zero-control generalized-PD replay; its applied diagnostic drive is reconstructed
from the forward force balance at the actual state. The controller output is not
the frozen truth.

| condition/joint | RMSE N m | P95 abs N m | max abs N m | relative RMS | pass |
|---|---:|---:|---:|---:|---|
| PRIMARY hip | 2.695688 | 0.219325 | 52.111408 | 7.439% | True |
| PRIMARY knee | 4.273047 | 9.529833 | 17.220277 | 15.107% | True |
| SENSITIVITY hip | 2.695641 | 0.221519 | 52.111408 | 7.416% | True |
| SENSITIVITY knee | 0.935972 | 0.519774 | 17.220277 | 8.400% | True |

Exact inverse/formula, decomposition, muscle moment reconstruction, and forward
balance residuals are all reported in `FORCE_SEMANTICS_VALIDATION.csv` and
`DYNAMICS_BALANCE_AUDIT.csv`. `GENERALIZED_FORCE_TRUTH_SEMANTICS=PASS` only when
both conditions pass the predeclared limits.

## Tracking and stability

PRIMARY stable replay: `True`. SENSITIVITY stable replay:
`True`. Full q/dq/ddq RMS, P95 and maxima are in
`DYNAMICS_BALANCE_AUDIT.csv`. There were no truth-role controller gains: the
driver is a cross-check only, while the dataset truth remains prescribed-state
inverse dynamics.

## Limited extension sensitivity

The knee PRIMARY/SENSITIVITY RMS ratio is
2.539; the peak ratio is
3.598. In 10.540–16.824 s,
the knee truth difference RMSE is
54.121 N m versus a
4.015 N m SENSITIVITY RMS.
Assessment: `MATERIAL_HIGH_FLEXION_TORQUE_AMPLIFICATION_DETECTED`. This is finite and
mechanically continuous, but it is a material reference-level dynamics caveat;
it must not be hidden by calling the limited-125 extension equivalent to native
MyoLeg behavior. The dominant source is the model's soft knee joint-limit
constraint: high-flexion RMS is
50.194
N m in PRIMARY versus
0.001
N m in SENSITIVITY. The corresponding muscle-actuator internal-term difference
is only
3.832 N m.
Thus this is specifically a near-upper-limit model reaction, not an inertia
increase and not evidence of human physiology.

## Existing objective compatibility

`MYOLEG_TAU_COMPATIBLE_WITH_EXISTING_J = WITH_CAVEATS`.

The sign is project-positive hip flexion/knee flexion and the unit is N m for
both hinge generalized forces. Future use of the existing RMS-torque objective
must normalize every virtual-patient condition with its own reference replay,
so `J_truth(reference)=1` by construction. This stage does not compute a
candidate J, fit five parameters, train PINN, rank candidates, run BO, or make a
landscape.

## Runtime

- PRIMARY complete two-path replay: 2.417 s wall time,
  24000 integration steps, replay realtime
  factor 10.685x.
- SENSITIVITY complete two-path replay: 2.436 s wall time.
- One-reference engineering estimate (mean of the two conditions):
  2.426 s; 100 =
  242.6 s; 1,000 =
  2426.3 s; 21,025 =
  51012.7 s.

These are `reference-replay-based engineering estimates`, not landscape timing.

## Direct answers

### Q1 — Can the full 24 s formal reference replay stably?

Yes on the limited-125 virtual model under the frozen diagnostic replay; this is
offline model evidence only.

### Q2 — What is frozen as tau_truth?

`TAU_MY0LEG_REQUIRED_DRIVE = T(q)^T [M qacc + bias - passive - constraint - actuator(P0)]`,
with project-positive hip/knee coordinates and N m units.

### Q3 — Do two paths agree?

Yes within the predeclared research limits for both conditions; exact force
accounting also closes at numerical precision.

### Q4 — Are sign and units compatible?

Yes. The explicit ±1 N m forward-response sign audit passes for both joints and
both conditions, and the coordinate mapping remains identity with
`theta_shank=q_hip-q_knee`.

### Q5 — Does 120–124.79 degrees introduce abnormal dynamics?

Yes, relative to the native-compatible sensitivity replay it produces material
high-flexion knee-torque amplification under the predeclared comparison rule,
dominated by activation of the limited-125 model's soft joint-limit constraint.
That limitation does not invalidate the force semantics, but it limits how the
PRIMARY virtual patient may be interpreted.

### Q6 — Can existing J be applied later?

Yes, with the caveat that normalization must use each condition's own frozen
reference truth; no J landscape was generated here.

### Q7 — What is one replay's measured cost?

2.426 s for the complete
two-path engineering benchmark on this machine.

### Q8 — Is the truth interface ready?

Yes for the next **offline** cohort-design stage, with the limited-extension
torque-amplification caveat. It is not human-ready or robot-approved.

## Tests and next stage

Internal invariant tests: 22 passed, 0 failed.
The only recommended next stage is `MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1`;
this script stops without executing it.
