# P2_DECISION_RULE_SEMANTICS_AUDIT_V1

Candidate manifest SHA-256: `2e97a2b812acff15284c469756e5a0b0dedad307a7f8b8410276dd415c593b65`

## Frozen semantics

`0.005` is the unchanged mechanical-objective equivalence/minimum-meaningful-improvement magnitude. `U` is empirical model error on predicted objective differences. The current rule adds them, but the repository contains no explicit theoretical or preregistered requirement that they must be added. Audit label: `ADDITIVE_MARGIN_IS_DESIGN_ASSUMPTION`.

S1 separates a categorical calibration direction-evidence gate from the 0.005 magnitude gate. S2 uses the independent one-step P95 only to ask whether a transparent residual interval still supports a negative delta J, while independently retaining the magnitude check. S3 applies the same split to 2/3/5-step endpoints and authorizes only one next formal-grid step before refit/recompute.

## Development shadow comparison

- S0_CURRENT_ADDITIVE_MARGIN: trials=87, exploit=31, bundle authorizations=0, missed=13, false total=0, final J=0.987946424, regret=0.022990289.
- S1_TWO_GATE_DIRECTION_AND_MAGNITUDE: trials=110, exploit=54, bundle authorizations=0, missed=9, false total=0, final J=0.978450745, regret=0.013496074.
- S2_UNCERTAINTY_INTERVAL_DIRECTION_GATE: trials=110, exploit=54, bundle authorizations=0, missed=9, false total=0, final J=0.978450745, regret=0.013496074.
- S3_BUNDLE_ENDPOINT_TWO_GATE: trials=169, exploit=118, bundle authorizations=64, missed=45, false total=0, final J=0.971619623, regret=0.006665725.

The nine historical small-step paths recovered by S0/S1/S2/S3 were respectively 0/9, 0/9, 0/9, and 9/9.

S3 changed bundle authorizations from 0 to 64, missed-improvement rounds from 13 to 45, and total false-improvement events from 0 to 0. Mean final J changed by -0.016326801; mean regret changed by -0.016324564.

## Interpretation

The frozen classification criteria evaluated to `{'S3_recovers_more_of_9_paths_than_S0': True, 'S3_has_more_bundle_authorizations_than_S0': True, 'S3_missed_improvement_not_higher_than_S0': False, 'S3_total_false_improvement_not_higher_than_S0': True, 'S3_mean_final_J_not_worse_than_S0_by_0.005': True, 'S3_mean_regret_not_worse_than_S0_by_0.005': True, 'S0_small_step_recovery': 0, 'S3_small_step_recovery': 9}`. Final audit conclusion:

`MORE_EVIDENCE_REQUIRED`

This conclusion is limited to the specified synthetic DEVELOPMENT + POST_REJECTION_DEVELOPMENT shadow. It does not make S3 a policy, choose a final percentile, alter 0.005, or establish prospective, human, robot-motion, safety, or clinical readiness. The formal states remain `POLICY_DESIGN_REQUIRES_REVISION`, `NOT_HUMAN_READY`, and `NOT_ROBOT_MOTION_APPROVED`.

## Plain-language answers

A. `0.005` says an objective change must be large enough to count as meaningful; `U` says how wrong the model's predicted direction/magnitude has been on independent residual evidence.

B. They are currently added because the implemented conservative rule requires the prediction to clear both quantities in one scalar margin.

C. No repository evidence establishes that addition as a scientific necessity; it is a design assumption.

D. The S3 two-gate recovery is 9/9 versus S0 0/9.

E. Observed total false-improvement events are S0=0, S3=0.

F. The permitted final root-cause label is `MORE_EVIDENCE_REQUIRED`; no next policy is implemented.
