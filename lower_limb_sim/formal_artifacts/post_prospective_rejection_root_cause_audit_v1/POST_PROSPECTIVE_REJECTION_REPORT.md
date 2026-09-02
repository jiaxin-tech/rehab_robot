# Post-prospective rejection root-cause report

## Immutable evidence boundary

The original conclusion remains `P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION`. Its manifest SHA remains `94d33675b2ae51ef80154c3bba92f31b87852267f3cffbaaacc75c3ce0aa1876` and its start commit remains `d7fe80945ae625fffc7919e1735e9e2df8c8fa00`. Nothing in this audit is prospective evidence: all new results are `POST_PROSPECTIVE_DEVELOPMENT_ONLY`.

## Factorial decomposition

| Variant | Trials | EXPLORE | EXPLOIT | Missed | False | Mean final J | Mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 G0+S0 | 35 | 24 | 11 | 6 | 0 | 0.989646141 | 0.028292404 |
| A1 G2+S0 post-hoc | 29 | 24 | 5 | 22 | 0 | 0.994867132 | 0.033512263 |
| A2 G0+S2 post-hoc | 20 | 9 | 11 | 6 | 0 | 0.989646141 | 0.028292404 |
| A3 G2+S2 rejected | 12 | 7 | 5 | 10 | 0 | 0.994867132 | 0.033512263 |

The transparent mean case-wise decomposition gives A3-A0 final-J change 0.005220992, guard main effect 0.005220992, stopping main effect 0.000000000, and interaction 0.000000000. For regret the corresponding values are 0.005219860, 0.005219860, 0.000000000, and 0.000000000. Missed-round A3-A0 mean change is 0.666667.

The rejected V2A outcome is primarily a **guard effect**, not a stopping effect: A1 and A3 have the same mean final J, while A0 and A2 also have the same mean final J. S2 reduced the number of executed and low-decision-value trials, but its mean final-J and regret effects are exactly zero in this cohort. The missed-round count has a guard/stopping interaction because earlier stopping reduces the number of later rounds that can be counted; that interaction must not be mistaken for better decisions. This is not evidence that the mechanical objective, generator, or five-parameter model must change. Local uncertainty remains only a research concept and no percentile is frozen by this audit.

## Premature stops and removed trials

All 24 immutable prospective premature-stop rows were retained and classified:

- `MULTIPLE_FACTORS`: 15
- `EXPLORATION_STOPPED_BEFORE_REACHING_CANDIDATE`: 9

Across the two fixed-guard S0-vs-S2 comparisons, 6 cases removed only low-value continuation while 0 cases truncated a chain that later produced a useful S0 action; the remaining comparisons are indeterminate because S2 did not trigger early. In this cohort there is no demonstrated S2 truncation of a later useful S0 action and no separable S2 endpoint penalty.

K=1, K=2, and K=3 had the same previously observed mean final J. That result does not identify K=2 as the cause of rejection and does not justify tuning K=4 or K=5. A richer stopping criterion may still be studied as a robustness question, but it is not the primary repair supported by this factorial audit.

## Small-step accumulation

All nine previously detected paths were expanded through steps 1..5. They are same-axis, same-sign, formal-neighbor-continuous knee-negative paths; none requires a turn or mixed-axis move. 9 paths are classified `CUMULATIVE_SIGNAL_PRESENT` and 0 `CUMULATIVE_MODEL_UNRELIABLE`. Their prospective residuals remain post-hoc characterization and cannot calibrate future uncertainty.

## New validation design

The new `DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1` plan SHA is `3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84`. It independently samples 2-step and 3-step endpoint bundles, with 5-step as an optional diagnostic layer, across all generator coordinates, signs, and boundary/interior strata. It contains no truth outcomes and does not alter the old 324-pair plan.

## Minimum future revision scope

1. Retain local-uncertainty validation as a concept, but do not freeze a percentile from this rejected cohort.
2. Study bundle-aware cumulative decisions only after independent execution of the newly frozen bundle-validation plan.
3. Treat any richer decision-value stopping rule as a secondary, independently validated robustness study; do not present stopping replacement as the demonstrated repair for this rejection.

No P2 V3 policy, new threshold, objective change, generator enlargement, or model enlargement is implemented here.

## Status and future split

`NEXT_REVISION_ROOT_CAUSE_IDENTIFIED`

`DEVELOPMENT_USED_AFTER_REJECTION = true`. These six cases can never again support a claim of prospective success. Any future revision must use a new independent prospective cohort. P2 V2 remains default-off, `NOT_HUMAN_READY`, and `NOT_ROBOT_MOTION_APPROVED`.
