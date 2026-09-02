# Sequential Personalization Formal Offline Report

Status: **FORMAL OFFLINE SOFTWARE EXPERIMENT**. This is not real-robot, human, clinical, safety, effectiveness, or comfort validation.

## Frozen inputs

- Active reference SHA-256: `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`
- ROM: `ROM_PROTOCOL_V2`, hip 0--120 deg, knee 5--145 deg
- Model convention: `theta_shank = q_hip - q_knee`
- Duration: 24 s; duration optimization disabled

## Mechanical objective and deterministic ranking

`J_rms = sqrt((R_h^2 + R_k^2) / 2)`, where each joint RMS torque is normalized by the same subject/scenario's frozen reference, so the reference is exactly 1. Mechanical costs within 0.005 are equivalent; ties use smaller reference deviation, combined peak ratio, torque-rate ratio, then lexical trajectory ID. This tolerance is not a robot safety limit.

## Combined-mild formal result

| subject | executed | accepted improvements | final alpha (hip, knee, phase) | final J | reduction | stop | boundary |
|---|---:|---:|---|---:|---:|---|---|
| baseline | 2 | 1 | (0, -1, 0) | 0.993146983 | 0.685302% | STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD | False |
| hip_stiff | 2 | 1 | (0, -1, 0) | 0.993099324 | 0.690068% | STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD | False |
| knee_stiff | 1 | 0 | (0, 0, 0) | 1.000000000 | 0.000000% | STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE | False |
| heavy_leg | 2 | 1 | (0, -1, 0) | 0.992374061 | 0.762594% | STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD | False |

The `knee_stiff` result legitimately falls back to the frozen reference: no feasible candidate cleared the 0.005 predicted-improvement rule. The other three subjects accepted one knee-amplitude step of -1 deg. No final point reached an offline search bound.

## Prediction audit and stopping boundary

- Matched-case maximum absolute prediction error: `1.6364687383e-13`.
- Runs reaching a non-reference prediction audit: `6`.
- Runs stopped because no frozen reliability threshold exists: `6`.
- No model-reliability threshold was guessed. A reviewed threshold is required before additional sequential steps can be called formal.

## Failure and gate audit

- Infeasible candidate rows: `16`; observed formal reason: `domain_coverage_insufficient`.
- Executed-but-rejected behavior, model-update failure, trust-step shrink, minimum-step stop, bound saturation, parent hash failure, and legacy reference rejection are retained as regression tests, not hidden trials.

## Data isolation

Initial fitting uses only the persisted `train` role. Only actually simulated trials enter adaptation, including rejected trials. Validation and held-out rows never enter proposal, fitting, ranking, trust-region, or stopping decisions. Held-out evaluation runs once after the stop reason is fixed. See `DATA_LEAKAGE_AUDIT.md` and `data_leakage_audit.json`.

## Interpretation limit

The reported reductions are virtual-model mechanical torque reductions. They do not establish comfort, rehabilitation benefit, patient response, robot safety, or real-world effectiveness.
