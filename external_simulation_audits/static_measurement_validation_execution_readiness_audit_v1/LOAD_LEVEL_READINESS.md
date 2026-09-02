# Load-Level Readiness

## Current decision

`FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW` and `LOAD_LEVEL_BLOCKER` remain active. Both `L1_REVIEWED_LOW` and `L2_REVIEWED_HIGH` are null. No N value can be produced from current repository evidence.

## Frozen future determination pathway

A load may be selected only after all six constraints are jointly available:

1. it exceeds the independently measured zero/noise floor by a prefrozen signal criterion;
2. it is below a site-reviewed static-force limit;
3. it is inside the calibrated instrument and fixture ranges with uncertainty/margin;
4. it is far below the most conservative robot/hardware protective limit;
5. the hands-free fixture reproduces magnitude and direction;
6. magnitude is independently traceable rather than inferred from the robot output.

Use the most conservative upper bound. The SDK wrench result cannot choose or increase a load. Hand push, estimated manual force and human-subject loading are prohibited formal evidence.

## Threshold evidence split

| Threshold/evidence | Can freeze before physical validation? | Dedicated calibration dataset required? |
|---|---|---|
| instrument range/accuracy/uncertainty | yes, from current certificate | no if certificate is current and setup-relevant |
| fixture rating and site-reviewed robot safety ceiling | yes, from reviewed engineering records | no |
| geometry metrology resolution and registration tolerance | yes, from certificate plus endpoint error budget | a metrology phantom check is still required |
| minimum separation/angular endpoint-error tolerance | yes, prospectively from geometry/error budget | no formal validation outcome may tune it |
| robot wrench zero/noise floor and drift | no numeric value exists now | yes: independent PRE/POST unloaded calibration dataset |
| steady-load acceptance/SNR and sign minimum | definition can be prefrozen; numeric gate needs baseline variability | yes |
| cross-axis leakage/pose consistency gate | engineering maximum can be prefrozen | fixture registration/calibration evidence is required; not validation outcomes |
| setup repeatability PASS gate | tolerance can be prefrozen | actual ten setups test the gate but must not set it |

Calibration data and formal validation data must have separate run IDs and roles. Formal validation results may test a threshold but may not define or relax it.
