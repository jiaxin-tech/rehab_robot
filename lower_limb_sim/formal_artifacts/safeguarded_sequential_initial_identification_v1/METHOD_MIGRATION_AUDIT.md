# Sequential Initial Identification Method Migration Audit

## Active new method

- `safeguarded_sequential_initial_identification.py` is the only implementation
  of `SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1`.
- `MAX_INITIAL_IDENTIFICATION_TRIALS=5`;
  early stop is supported and a sixth trial is structurally prohibited.
- Every identification candidate owns an independent
  `excitation_duration_s`.  The tested values are explicitly
  `RESEARCH_DESIGN_RANGE_NOT_HUMAN_SAFETY_LIMIT`.
- Default completion remains fail-closed as
  `IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW`.

## Frozen 24 s content that must remain

- `reference_measured_asymmetric_closed_slow` remains a 24 s rehabilitation
  reference with SHA-256 `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`.
- `continuous_reference_neighborhood.py` keeps `TOTAL_DURATION_S=24.0` and a
  fixed time scale because it defines the frozen rehabilitation-personalization
  family, not the new identification-duration policy.
- The new identification generator first obtains a C2 geometric member of that
  family, then applies an independent linear time change with exact chain-rule
  scaling of velocity and acceleration.  It does not edit the reference file or
  the existing generator equations.

## Historical method retained, not active for this task

- `sequential_personalization.py` and
  `formal_artifacts/sequential_personalization/` preserve the earlier workflow
  in which one pre-existing training dataset seeds personalization and metadata
  records 24 s.  These are historical prior-stage evidence; this task neither
  imports nor executes that personalization path.
- Existing Stage 4 identification trajectory families and speed profiles are
  retained as earlier software evidence.  They are not silently relabelled as
  the new patient-envelope-aware 1–5 trial protocol.

## Existing identifiability threshold audit

- `identifiability_analysis.py` contains a numerical SVD rank tolerance and a
  correlation reporting threshold.  These are numerical/diagnostic mechanisms,
  not an approved multi-criterion completion rule.
- `parameter_estimator.py` reports optimizer success, singular values,
  covariance-shaped uncertainty, standard errors, and residuals, but does not
  define when a new subject is sufficiently identified.
- No approved conjunction of rank, minimum singular value, condition number,
  worst correlation, all-five-parameter uncertainty/sensitivity, and validation
  residual was found.  No real-subject threshold was invented during migration.

## Constraint audit

- Global ROM/workspace/Jacobian/force mapping/finite/C2 checks remain model
  constraints and are never called patient-safety limits.
- Patient operational envelopes are supplied inputs, never expanded by
  constraint violation, large force, error, or pain.
- Real-robot hard safeguard status remains
  `NOT_DEFINED_NOT_APPROVED`; therefore no physical execution is authorized.
