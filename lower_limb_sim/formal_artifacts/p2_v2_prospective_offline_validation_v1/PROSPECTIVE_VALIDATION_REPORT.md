# P2 V2 prospective offline validation report

## Frozen boundary

- Protocol: `P2_V2_PROSPECTIVE_OFFLINE_VALIDATION_V1`.
- Manifest SHA-256: `94d33675b2ae51ef80154c3bba92f31b87852267f3cffbaaacc75c3ce0aa1876`. The manifest was atomically persisted before any prospective identification, validation, personalization, or post-policy truth call.
- Cohort: 3 new matched cases and 3 new model-mismatch cases. All 9 named development cases are excluded from primary metrics; held-out final test data were not read.
- Active reference SHA, ROM, shank-angle convention, five-parameter model, objective, generator bounds, 0.005 tolerance, and 90% support gate remained frozen.

## Primary P95 result

Compared with P2 V1, P2 V2A (frozen local P95 + C0 + K=2) changed total missed-improvement rounds from 6 to 10, executed false improvements from 0 to 0, mean final J from 0.989646141 to 0.994867132, and mean global regret from 0.028292404 to 0.033512263. Prospective outcomes did not update P95.

## P99 sensitivity

The frozen P99 sensitivity produced 13 missed rounds, 0 executed false improvements, mean final J 0.998940719, and mean global regret 0.037585826. P95/P99 was not reselected from these results.

## Bundle and small-step accumulation

C2, C3, and C5 remain `SHADOW_ONLY_NOT_CALIBRATED`; the designated plan contains no pre-registered bundle residuals, and neither n-times nor square-root-n aggregation was assumed. P2 V2B was not executed. The post-policy audit found 9 pre-registered small-step accumulation paths; they remain descriptive and cannot activate a cumulative policy.

## Decision-value stopping

- `P2_V1_G0_C0_S0`: total trials 35, EXPLORE 24, mean final J 0.989646141.
- `P2_V2A_G2_C0_S2`: total trials 12, EXPLORE 7, mean final J 0.994867132.
- `P2_V2A_G3_C0_S2_SENSITIVITY`: total trials 8, EXPLORE 7, mean final J 0.998940719.
- `P2_V2A_G2_C0_S1_SENSITIVITY`: total trials 9, EXPLORE 4, mean final J 0.994867132.
- `P2_V2A_G2_C0_S3_SENSITIVITY`: total trials 15, EXPLORE 10, mean final J 0.994867132.

Support reduction is reported as provenance reduction, not automatically as performance loss. The stopping decisions used only already-observed model/prediction/guard/information/support signals and no future truth, exploit, or best trajectory.

## Subject specificity and boundaries

The 3 matched subjects produced 2 distinct objective truth optima. Every truth-boundary row is separately labelled `OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM`; a policy-only boundary is labelled `POLICY_INDUCED_BOUNDARY_COLLAPSE`. The objective was not modified.

## Failure modes

Observed failure-mode rows: 24. They are recorded in `prospective_failure_mode_audit.csv`; none caused tuning during this experiment. Any observed row is a future revision question only.

## Formal conclusion

`P2_V2_PROSPECTIVE_EVIDENCE_REJECTS_CURRENT_REVISION`

This is synthetic offline research evidence only. P2 V2 remains default-off. Initial-identification acceptance still requires review, global model reliability is not frozen for humans, and the result remains `NOT_HUMAN_READY` and `NOT_ROBOT_MOTION_APPROVED`.
