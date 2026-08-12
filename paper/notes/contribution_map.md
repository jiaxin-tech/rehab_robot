# Contribution candidates

These are candidates, not submission-ready claims. They are ordered by the
intended scientific hierarchy and limited to current repository evidence.

## Contribution 1 — Task-local subject-specific equivalent dynamics identification

- **Scientific novelty candidate:** A five-parameter gray-box identification
  method for passive supine hip–knee rehabilitation that reconstructs joint
  interaction torque from a strap-point force and separates a common
  mass/inertia scale, hip/knee equivalent passive stiffness, and hip/knee
  equivalent passive damping using limited task-relevant trajectories.
- **Repository evidence:** `parameter_estimator.py`, `identification_dataset.py`,
  `identifiability_analysis.py`, `reference_local_excitation.py`, and
  `reference_local_active_asymmetric.py`; results R-ID-001, R-ID-002,
  R-LOCAL-001 (legacy), and R-LOCAL-ACTIVE-001 (current active reference).
- **Quantitative support:** Clean global identification has maximum parameter
  error about 0.000001%. Around the current active asymmetric reference, all
  four local sensitivity matrices have rank 5, condition number 36.448--90.131,
  maximum absolute parameter correlation 0.3085 with no pair at or above 0.9,
  maximum matched-clean parameter error $3.654\times10^{-6}\%$, and maximum
  held-out torque RMSE $1.874\times10^{-8}$ N m. Exact active-slow state-box
  coverage is 99.7506%; the 10%-faster boundary case falls to 81.0474%.
- **Missing evidence:** Synchronized real wrench/state episode, real-data
  identification, model-mismatch replication around the active reference, and
  physical repeatability.
- **Reviewer risk:** Current near-zero clean errors use matched generator and
  estimator equations and primarily demonstrate numerical self-consistency.
- **Confidence:** MEDIUM.

## Contribution 2 — Reference-centered low-dimensional candidate generation and screening

- **Scientific novelty candidate:** Preserve a prescribed closed rehabilitation
  task while generating low-dimensional amplitude, phase, and duration variants
  and rejecting candidates that violate ROM, workspace, closure, Jacobian,
  force-mapping, or train-fitted local-domain constraints.
- **Repository evidence:** `reference_local_excitation.py`,
  `reference_local_active_asymmetric.py`, `run_reference_candidate_evaluation.py`,
  and active asymmetric reference generator `reference_measured_asymmetric.py`;
  results R-REF-ASYM-001, R-LOCAL-ACTIVE-001, and R-CAND-001.
- **Quantitative support:** The active slow asymmetric reference is C2 closed,
  preserves measured asymmetry at ratios above 0.99998, and has 100% frozen-domain
  coverage. In the older symmetric-reference study, all nine candidates passed
  the software gates; C4 reduced the worst-subject peak knee torque from 64.12
  to 61.26 N m and combined RMS torque from 50.12 to 48.73 N m relative to C0.
  The current active-reference study now verifies conservative amplitude,
  phase, and duration neighbors and their local-domain/identifiability basis,
  but it does not select a personalized candidate.
- **Missing evidence:** Rerun candidates about the active asymmetric reference;
  a frozen subject-specific selection objective; calibrated interaction residual;
  physical velocity/acceleration/load limits; reference-versus-selected result.
- **Reviewer risk:** Existing C0–C8 results are centered on a superseded symmetric
  reference, every feasible candidate lies on the unweighted Pareto front, and no
  final personalized candidate is selected.
- **Confidence:** LOW.

## Contribution 3 — Reliability boundaries for task-local prediction

- **Scientific novelty candidate:** A unified validation protocol showing how
  excitation identifiability, fixed/variable timing mismatch, geometry error,
  and structured model mismatch affect equivalent parameter recovery and
  unseen-trajectory prediction.
- **Repository evidence:** `delay_estimator.py`, `timestamp_alignment.py`,
  `windowed_delay_tracker.py`, `causal_sample_matcher.py`,
  `run_model_mismatch_experiment.py`, and `run_geometry_error_experiment.py`;
  results R-TIME-001, R-TIME-002, R-MM-001, R-GEO-001, and
  R-LOCAL-ACTIVE-001.
- **Quantitative support:** Fixed-delay grid search recovered all 24 injected
  0–40 ms cases exactly at 1-ms resolution with no boundary hit and reduced the
  worst test RMSE from 0.486 to 0.000635 N m. In the combined-mild mismatch case,
  the equivalent identified model reduced mean interpolation-trajectory torque
  RMSE by 63.55% versus the generic baseline. The active-reference-local audit
  adds a concrete speed boundary: 18.9526% of the 10%-faster profile lies
  outside the train-fitted 6-D box even though matched-model torque prediction
  remains nearly exact.
- **Missing evidence:** Hardware timestamps, real state/wrench skew, verified
  wrench semantics, and external replication. Geometry-error conclusions require
  careful mode-specific reporting because identification can be worse than the
  generic baseline for some assumed-geometry cases.
- **Reviewer risk:** Many variable-delay results benefit from reliable synthetic
  sample timestamps; strong coupling produces trajectory-specific failures, so
  averages cannot be reported without failure cases.
- **Confidence:** MEDIUM.

## Explicitly incomplete contribution

Wrench–tactile fusion is not a contribution. No calibrated tactile acquisition
or quantitative tactile result exists in the current repository.
