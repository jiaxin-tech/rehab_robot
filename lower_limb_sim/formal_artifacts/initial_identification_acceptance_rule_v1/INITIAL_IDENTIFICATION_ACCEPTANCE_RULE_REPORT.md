# Initial identification acceptance-rule report

## Formal outcome

`INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`

The repository has enough evidence to define and test a two-gate architecture,
but not enough independent scientific evidence to freeze its numerical limits.
Consequently no case receives a formal `theta_hat_0`, no personalization
prerequisite passes, and every retained temporary model is `DIAGNOSTIC_ONLY /`
`NOT_APPROVED_FOR_PERSONALIZATION`.

## Why the two gates differ

Parameter identifiability asks whether the five columns of the local sensitivity
matrix carry sufficiently independent numerical information. Model adequacy asks
whether those five equivalent parameters predict unseen validation motion. A
full-rank, well-conditioned fit can still be systematically wrong when the data
generator contains nonlinear stiffness, coupling, nonlinear damping, or a
structured residual that the five-parameter model cannot represent.

## Matched positive controls

Under the explicitly non-formal matched-Trial-2 envelope, all four matched
positive controls first satisfy both diagnostic gates at Trial(s)
`[2]`. Trial 2 adds a second distinct excitation and lifts the
minimum singular value/support while the independent validation error is at
floating-point solver scale. This shows the software can recover a model that
is structurally identical to its generator; it does not by itself justify a
human-subject acceptance threshold.

## combined_mild diagnosis

After Trial 5, `combined_mild` has rank `5`, minimum
singular value `30.2879`, condition number
`38.8242`, worst absolute correlation
`0.2667`, and maximum normalized
parameter change `0.00169218`. Independent
validation combined RMSE remains `0.787185`
N·m and validation `e_J` is `0.056861`. Its trend is
therefore `MODEL_STRUCTURE_LIMITATION`: information improves and all five
columns remain full rank, but structural prediction error does not resolve.
More excitation can improve the information matrix; it cannot create nonlinear,
coupling, or residual terms that are absent from the estimator.

The previous within-identification 20% holdout comparator was approximately
0.44 N·m for this case. The stricter split audit here uses the two predeclared,
independent VALIDATION trajectories and obtains the separately reported value
above; the two numbers answer different questions and are not interchangeable.

## Treatment of 0.20 N·m

The 0.20 N·m line remains `RESEARCH_ONLY / UNJUSTIFIED`. It is shown in the
candidate table and figure for provenance, but it is neither inherited nor
changed to 0.45 N·m. Without an independently justified acceptable-model label,
false-accept and false-reject counts are `NOT_COMPUTABLE`; candidate analysis is
diagnostic consistency analysis only.

## Recommended future rule contents

The parameter-identifiability gate should jointly review: five-column rank,
minimum singular value, condition number, worst parameter correlation,
per-parameter sensitivity, per-parameter uncertainty, and normalized change of
all five accumulated estimates. The model-adequacy gate should independently
review validation hip/knee/combined torque RMSE, formal NRMSE, validation
mechanical-objective `e_J`, and relative `e_J`. Training RMSE belongs to neither
an adequacy release decision nor a substitute for validation.

## Scenario outcomes

The only formal status is `REQUIRES_REVIEW`; the requested categorical states
below are explicitly candidate-rule diagnostics, not frozen releases:

- `baseline__matched_linear`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `INITIAL_IDENTIFICATION_COMPLETE`; theta_hat_0 frozen = no.
- `hip_stiff__matched_linear`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `INITIAL_IDENTIFICATION_COMPLETE`; theta_hat_0 frozen = no.
- `knee_stiff__matched_linear`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `INITIAL_IDENTIFICATION_COMPLETE`; theta_hat_0 frozen = no.
- `heavy_leg__matched_linear`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `INITIAL_IDENTIFICATION_COMPLETE`; theta_hat_0 frozen = no.
- `baseline__nonlinear_stiffness_mild`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `MODEL_INADEQUATE_FOR_PERSONALIZATION`; theta_hat_0 frozen = no.
- `baseline__hip_knee_coupling_mild`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `MODEL_INADEQUATE_FOR_PERSONALIZATION`; theta_hat_0 frozen = no.
- `baseline__nonlinear_damping_mild`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `MODEL_INADEQUATE_FOR_PERSONALIZATION`; theta_hat_0 frozen = no.
- `baseline__structured_residual`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `MODEL_INADEQUATE_FOR_PERSONALIZATION`; theta_hat_0 frozen = no.
- `baseline__combined_mild`: formal `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`; diagnostic candidate `MODEL_INADEQUATE_FOR_PERSONALIZATION`; theta_hat_0 frozen = no.

## Frozen boundaries

- ROM: `ROM_PROTOCOL_V2`, hip `0.0–120.0` deg, knee `5.0–145.0` deg.
- Active reference: `reference_measured_asymmetric_closed_slow` / `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- Angle identity: `q_hip - q_knee`.
- Model: unchanged five equivalent parameters `mass_scale, k_hip_nm_per_rad, k_knee_nm_per_rad, b_hip_nm_s_per_rad, b_knee_nm_s_per_rad`.
- Mechanical objective: reused unchanged as a validation-only precursor; no global reliability threshold frozen.
- Held-out final test: not generated, read, tuned against, or used.
- Robot/hardware/safety: not imported, connected, or modified.
- Explore/exploit personalization: not executed.

## Commit boundary

The prerequisite `SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1` files were
already untracked at task start. Commit that stage intentionally first (while
excluding `.DS_Store`), then commit this module, runner, tests, and this artifact
directory as a separate checkpoint. Do not use `git add .`.
