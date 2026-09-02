# P2 V2 Formal Research Design Report

## Boundary

- Design ID: `P2_V2_FORMAL_RESEARCH_PROTOCOL_DESIGN_V1`; this is a formal research protocol design, not a frozen policy.
- P2 V1 remains unchanged and default. No formal personalization or robot connection was executed.
- active reference, ROM_PROTOCOL_V2, five-parameter model, mechanical objective, generator bounds, 0.005 tolerance, and 90% support gate remain unchanged.
- Status remains `OFFLINE_METHOD_REQUIRES_REVISION`, `NOT_HUMAN_READY`, `NOT_ROBOT_MOTION_APPROVED`.

## Part A — Designated local validation

`DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1` should enter a future formal P2 V2 as a mandatory evidence layer, but only after its 324-pair plan receives independent outcomes and reviewer approval. Pair selection is geometry/trust/hash only; final truth landscape selection is forbidden. The present protocol does not choose max/P95/P99 or create a guard threshold.

## Part B — `CUMULATIVE_DECISION_RULE_V1`

Rule A evaluates each step separately. In knee_stiff evidence, its maximum single-step improvement is 0.004467188, so 0 of 5 steps exceed the unchanged 0.005 criterion.

Rule B evaluates a same-coordinate, same-sign bundle. Candidate windows are 2, 3, and 5 steps; all three observed cumulative windows exceed 0.005, with maximum 0.022042232. This directly addresses the stepwise mechanism but is not enabled.

To avoid accumulating the wrong direction, a shadow bundle must be selected before truth, remain inside existing geometry/support at every intermediate point, keep one signed coordinate direction, reject predicted sign/ranking reversal, fix the model checkpoint within the bundle, and pass a bundle uncertainty constraint. Uncertainty aggregation candidates are worst-case sum, newly estimated block-P95, and RSS only after residual independence is demonstrated. Maximum steps and aggregation method remain unfrozen.

## Part C — `DECISION_VALUE_EXPLORATION_STOPPING_V1`

Each explore separates SUPPORT (coverage), MODEL (parameter), PREDICTION (map), and DECISION (ranking/best; exploit eligibility supplemental). Support alone is not a reason to continue.

Historical shadow results:

| consecutive zero-value candidate K | potentially avoided | later exploits | later accepted best changes |
|---:|---:|---:|---:|
| 1 | 25 | 0 | 0 |
| 2 | 21 | 0 | 0 |
| 3 | 17 | 0 | 0 |

These candidates could reduce low-value exploration in the frozen history, but no automatic stop occurred and K is not frozen. Prospective offline shadow validation and reviewed change-detection tolerance are still required.

## Final recommendation

1. **Local validation should enter formal P2 V2**, but only after independent outcomes, sample/power review, and uncertainty-statistic review.
2. **Cumulative improvement addresses the observed stepwise problem mechanistically**, provided direction/path/uncertainty constraints are retained; it is not yet a formal rule.
3. **Decision-value stopping is promising**, with historical potential reductions of 25/21/17 trials for K=1/2/3 and no later exploit in this history; it has not yet demonstrated prospective validity.
4. Minimal P2 V2 revision set:

1. `pre_registered_designated_local_pair_plan_with_immutable_hash_and_independent_outcomes`
2. `reviewed_local_uncertainty_provider_defaulting_to_P2_V1_until_approved`
3. `default_off_cumulative_bundle_evaluator_with_direction_path_and_uncertainty_guards`
4. `separate_support_model_prediction_decision_value_history`
5. `default_off_shadow_stopping_candidate_with_no_support_only_continue`
6. `preserve_reference_ROM_model_objective_generator_0p005_and_90percent_gate`

P2 V2 is not ready to replace P2 V1. Final state remains `OFFLINE_METHOD_REQUIRES_REVISION`, `NOT_HUMAN_READY`, `NOT_ROBOT_MOTION_APPROVED`.
