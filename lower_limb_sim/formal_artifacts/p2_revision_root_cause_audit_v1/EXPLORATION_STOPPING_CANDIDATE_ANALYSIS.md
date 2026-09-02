# Exploration stopping candidate analysis

## Frozen observation

- EXPLORE rows audited: 32.
- `SUPPORT_ONLY_EXPLORATION`: 29.
- Rows opening a new exploit eligibility: 3.
- knee_stiff EXPLORE rows: 8; all have exact-zero five-parameter and prediction-map change.
- baseline/hip_stiff/heavy_leg Trial 7--13 rows: 21; new exploit eligibility = 0.

## Future observable candidate

A future revision may study an **exploration diminishing-value stop** using only quantities already observable to the policy: repeated exact-zero parameter change, exact-zero map change, unchanged validation decision error, no newly eligible exploit within one/two rounds, declining incremental information gain, and continued support growth.  These fields should remain separate; support growth alone must not be treated as decision value.

No numeric threshold is frozen here.  The candidate is not enabled, does not use virtual truth, and is not a human/robot stopping rule.

- `candidate_enabled = false`
- `new_threshold_created = false`
- `truth_used_as_future_online_feature = false`
- diagnostic conclusion: `EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT`
- matched late-trial conclusion: `POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION`
