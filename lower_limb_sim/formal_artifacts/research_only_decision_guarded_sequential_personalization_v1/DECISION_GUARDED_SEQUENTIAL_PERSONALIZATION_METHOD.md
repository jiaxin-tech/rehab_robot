# RESEARCH_ONLY_DECISION_GUARDED_SEQUENTIAL_PERSONALIZATION_V1

## Status boundary

This is OFFLINE VIRTUAL RESEARCH ONLY. It creates no human-ready theta, formal personalization approval, robot command, clinical-safety claim, or comfort-optimization claim.

## Method

Each iteration recomputes predicted J over all 21,025 geometrically admissible points. Execution remains local: a supported trust-region neighbor may be exploited only when its predicted improvement exceeds the maximum designated-validation pairwise delta-J residual plus the existing 0.005 algorithm-equivalence tolerance. Support and distance are provenance/locality fields, never reliability scores. If no exploit passes, P2 may select one adjacent formal-grid frontier point by information gain before any truth access. Exactly one selected virtual trajectory is then executed, appended to the estimator dataset, used for one five-parameter update, and followed by another whole-map calculation.

The research decision uncertainty uses TRAIN-fitted parameters and designated VALIDATION trajectories only. P95 and P99 are reported as research diagnostics; the first guard uses the conservative maximum observed pairwise delta-J residual. Held-out final-test data are not read.

## Policy meanings

- P0 is the supported-only greedy sanity comparator and is not recommended.
- P1 permits only local candidates passing the validation-calibrated research guard.
- P2 uses the same exploit rule, then permits information-driven one-step local exploration when exploit is unavailable.

## Aggregate virtual results

- P2 matched cases: 24 trials, 9 EXPLORE, 15 EXPLOIT, mean J reduction 0.022478.
- P2 mismatch cases: 5 trials, 0 EXPLORE, 5 EXPLOIT, mean J reduction 0.005753.
- Explore trials later followed by reliable exploit: 3.
- Mean explore log-information gain: 1.33416; mean supported-region point growth: 2283.333.

Per-case and per-policy values, including false improvements, parameter changes, map changes, regret, and stop reason, are in the CSV artifacts. These virtual results do not establish a human decision threshold.

## Frozen statuses

- `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`
- `GLOBAL_MODEL_RELIABILITY_RULE_NOT_FROZEN`
- `REAL_ROBOT_HARD_SAFEGUARD = NOT_DEFINED_NOT_APPROVED`
- `FORMAL_HUMAN_READY_THETA_0 = false`
- `FORMAL_PERSONALIZATION_APPROVAL = false`
