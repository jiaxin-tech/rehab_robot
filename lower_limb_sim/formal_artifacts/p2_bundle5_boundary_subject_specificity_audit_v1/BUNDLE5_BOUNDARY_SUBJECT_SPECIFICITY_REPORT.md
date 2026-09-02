# P2_BUNDLE5_BOUNDARY_AND_SUBJECT_SPECIFICITY_AUDIT_V1

Manifest SHA-256: `b959444e8df39a05693f873aaa3060cb5c21a4525d7f1bbda9c81aa96f1762c8`

## Plain-language findings

### A. Why did adaptive almost degenerate to BUNDLE_5?

The unchanged 1/2/3-step endpoint evidence almost never passed both frozen
decision gates. The same H5 uncertainty applies to every axis at H5, while
knee-negative had the strongest predicted endpoint improvement and remained
available. Therefore the adaptive sequence escalated to H5 for 59/60 endpoint
authorizations; this audit does not define a replacement rule.

### B. How diverse are the 15 truth optima?

There are `8` unique full alpha optima: hip has `3`
values, knee `1`, and phase `7`. Pairwise analysis
finds `83` pairs where truth optima differ but H2 final alpha is
the same.

### C. Is truth itself generally at knee=-5?

Yes: `15/15` global truth optima have knee=-5, and the negative-knee
axis profile reaches its best value at that boundary in
`15/15` cases. But only `0/15` truth
optima equal the H2 common full alpha `(0,-5,0)`.

### D. Is the 15/15 common H2 alpha caused by truth or policy?

`MIXED_TRUTH_AND_POLICY_EFFECT`. Truth supplies a uniform knee-negative boundary
direction, while truth still contains hip/phase diversity that H2 does not
preserve.

### E. Why does knee-negative dominate H5 decisions?

KNEE_NEGATIVE was selected `60` times. Its mean
predicted improvement is `0.006784035` and
availability is `0.896`. All H5
directions use the same calibrated uncertainty, so uncertainty cannot explain
the axis preference. Selected direction agrees with posthoc truth direction in
`1.000` of executions. Dominance is primarily predicted
magnitude plus the truth mechanical response, followed by deterministic
lowest-J ranking—not a favorable axis-specific uncertainty.

### F. When is the H2-over-H1 J gain obtained?

Median within-case trials to obtain 50/80/90/95% of final H2 gain are
`2.0`, `4.0`, `4.0`,
and `4.0` respectively. These are diagnostics, not stopping
thresholds.

### G. How many of 116 trials are an obvious low-decision-value tail?

`49/116`. Classification counts: {"POST_OPTIMUM_LOW_VALUE": 49, "MULTIPLE_VALUES": 39, "DIRECT_DECISION_VALUE": 28}.
Final trial-cost interpretation: `MIXED_TRIAL_VALUE`.

### H. Are there many ineffective actions after reaching the boundary/final alpha?

Yes, in the matched-model cases: actions strictly after first arrival at final
alpha total `49`. They are exploration/refit actions that can
enlarge support or alter the model, but they do not change the best trajectory,
best J, or subsequent exploit eligibility in this frozen replay. They are
therefore classified posthoc as a low-decision-value tail, not used to define a
stopping rule.

### I. Does the objective lack all subject discrimination?

No. The full truth landscape contains `8` distinct full-alpha
optima, even though knee=-5 is common. At H2 final, knee torque contributes more
normalized squared-ratio reduction than hip in `15/15`
cases. The formula normalizes each joint to its own reference and weights the
two normalized RMS ratios equally; the common knee direction is therefore a
combined torque-response/generator-geometry effect, not evidence of an unequal
hard-coded knee weight.

### J. Is objective review scientifically justified now?

`OBJECTIVE_CHANGE_NOT_JUSTIFIED` under the manifest-frozen review criterion. This task does
not change the objective.

### K. What should be studied next?

The evidence points first to **policy subject-discrimination/ranking** and
second to **trial-efficiency accounting**. It does not justify objective change
or further automatic P2 expansion. Any next study needs a new checkpoint and a
separate protocol; none is implemented here.

### L. Final status

`BUNDLE5_ROOT_CAUSE_IDENTIFIED`

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Calibration cases provide frozen uncertainty only; no outcome selection.
- No held-out final test, future prospective cohort, human use, or robot motion.
- P2 V1, H1/H2/H3, reference, ROM, theta definition, five-parameter model,
  objective, generator/bounds, 0.005 tolerance, and 90% support gate unchanged.
- Final operational states remain `OFFLINE_ONLY`, `NOT_HUMAN_READY`, and
  `NOT_ROBOT_APPROVED`.
