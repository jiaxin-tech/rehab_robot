# P2_MULTI_STEP_DECISION_FRAMEWORK_ANALYSIS_V1

Manifest SHA-256: `a640f7c897291bf044f2f67dd41c84af87cd867be8140030cc4304a4b57d5731`

## Frozen comparison

All four frameworks use the same research-only two-gate semantics: endpoint predicted improvement must separately exceed the unchanged 0.005 meaningful mechanical-objective tolerance, and the independently calibrated scale-P95 residual interval must still support the improvement direction. The only changed variable is the endpoint horizon: 1, 2, 3, or 5 existing formal-grid units.

Bundle candidates execute the endpoint directly. Intermediate nodes are checked for existing generator relationship, geometry, active-reference provenance, patient envelope, and unchanged 90% model support, but zero intermediate trajectories are executed. After every selected endpoint, the five-parameter model is refit and the entire prediction map is recomputed.

## Results

- SINGLE_STEP: U=0.001967623, trials=56, missed=0, false=0, recovery=0/9, final J=0.999663936, regret=0.034705854, mean decision latency=3.733333333 trials.
- BUNDLE_2: U=0.001642171, trials=56, missed=1, false=0, recovery=0/9, final J=0.999663936, regret=0.034705854, mean decision latency=3.733333333 trials.
- BUNDLE_3: U=0.002445443, trials=57, missed=5, false=0, recovery=0/9, final J=0.999665675, regret=0.034707905, mean decision latency=3.266666667 trials.
- BUNDLE_5: U=0.004004960, trials=116, missed=7, false=0, recovery=9/9, final J=0.968794100, regret=0.003840217, mean decision latency=0.466666667 trials.

The 2/3/5-step bundle P95 values increase strictly with length: `True`. Small-step recovery across those horizons is non-decreasing: `True`. This is descriptive evidence of a benefit-versus-uncertainty trade-off, not a scale law or statistical-power claim. The 2-step P95 is below the separately sampled 1-step P95, so no monotonic 1→2→3→5 uncertainty claim is made.

The pre-frozen recommendation rule selected `BUNDLE_5` for further research. It did not change any live policy.

## Subject specificity

The endpoint horizon improved the frozen mechanical objective, but it did not improve observed subject differentiation. `SINGLE_STEP`, `BUNDLE_2`, and `BUNDLE_3` produced two distinct final alpha vectors across the 15 cases; `BUNDLE_5` produced one: `(hip=0°, knee=-5°, phase=0)` for every case. Thus all BUNDLE_5 cases saturated the same generator boundary. Its mean generator-grid L1 distance from the posthoc truth optimum was 18.133 steps, compared with 37.600 for SINGLE_STEP, but the common boundary solution is not evidence of subject-specific personalization.

This limitation was evaluated after the frozen manifest and did not change the predeclared recommendation ordering. `BUNDLE_5` is selected only as the most informative horizon for a future default-off prototype, where boundary saturation and subject specificity must be explicit failure-mode outcomes. No personalization-success claim is made.

## Answers

### A. Is single-step failure mainly a short decision horizon?

Development evidence says `True` under the pre-frozen criterion: at least one direct endpoint bundle recovers more historical accumulation paths without increasing observed false improvement and preserves or improves mean final J/regret within 0.005. This identifies horizon length as a material mechanism, not necessarily the only P2 failure source.

### B. Is a bundle endpoint more consistent with the rehabilitation-trajectory optimization goal?

For the repository's frozen **mechanical** objective, a bundle endpoint can represent a meaningful cumulative trajectory-shape change that one formal step cannot. It is therefore a better decision unit for this mechanical optimization question when its calibrated direction gate passes. This does not establish comfort, human rehabilitation benefit, safety, or clinical superiority.

### C. Which of 2/3/5 steps is most worth further study?

`BUNDLE_5` according to the manifest-frozen ordering: maximize small-step recovery, then minimize final J, trials, uncertainty, and horizon length. Its mean final J is 0.968794100, regret 0.003840217, and uncertainty 0.004004960.

### D. Should the next stage enter prototype implementation?

`YES, BUT DEFAULT-OFF OFFLINE PROTOTYPE ONLY`. This recommendation is development-only and requires a separately frozen prototype manifest. It does not authorize prospective testing, human use, or robot motion.

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Independent calibration cases supply residual scales only.
- No held-out final test and no prospective cohort.
- Active reference, ROM, `theta_shank = q_hip - q_knee`, five-parameter model, mechanical objective, generator, 0.005 tolerance, and 90% support gate remain unchanged.
- P2 V1 is unchanged; no policy is implemented or enabled.
- Final states: `OFFLINE_ONLY`, `NOT_HUMAN_READY`, `NOT_ROBOT_APPROVED`.
