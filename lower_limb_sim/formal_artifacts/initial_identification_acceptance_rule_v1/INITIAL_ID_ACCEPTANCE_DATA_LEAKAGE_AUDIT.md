# Initial-identification acceptance data-leakage audit

## Allowed evidence

- Accumulated TRAIN observations from the sequential identification trials.
- Frozen VALIDATION trajectories only: hip_dominant/slow, knee_dominant/fast.
- Scenario identity and generator parameters only for post-fit simulation audit;
  they are removed by the strict estimator-input projection before fitting or
  prediction decisions.

## Prohibited evidence

- Held-out final-test trajectories: coupled/nominal, hip_dominant/fast, knee_dominant/slow.
- Truth five-parameter values, complex generator torque terms, or future trial
  outcomes in excitation selection, parameter fitting, threshold construction,
  or acceptance decisions.
- Active-reference personalization maps or explore/exploit outcomes.

## Verified boundary

The validation builder generates only the two validation specifications and
projects observations to the existing `ESTIMATOR_INPUT_COLUMNS` whitelist.
The three held-out specifications are constants used solely for a negative
membership assertion; their trajectories and files are not generated or read.

`heldout_final_test_used_for_threshold_construction = false`  
`heldout_final_test_used_for_threshold_selection = false`  
`heldout_final_test_used_for_stopping = false`  
`truth_parameters_used_by_decision = false`
