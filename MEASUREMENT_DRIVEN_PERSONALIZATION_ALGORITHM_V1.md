# Measurement-Driven Personalization Algorithm V1

## Status and scope

`PHYSICS_INFORMED_SEQUENTIAL_PERSONALIZATION_V1_IMPLEMENTED_WITH_LIMITATIONS`

This is `OFFLINE_ALGORITHM_DEVELOPMENT_ONLY`. It is not clinical validation,
human-comfort optimization, real-robot personalization, or real-subject
evidence. The current real mechanical endpoint remains `NOT_YET_VALIDATED`.
The algorithm never imports robot motion, control, power, enable, or hardware
safety mutation APIs. `RealRobotEnvironment` is a fail-closed stub returning
`REAL_ROBOT_ENVIRONMENT_DISABLED: NOT_ROBOT_APPROVED`.

`PINN_NOT_JUSTIFIED`; `PINN_TRAINING = 0`.

## Architecture

```text
Frozen P4 branch-aware V3 candidate [beta_flex, beta_extend]
    -> PersonalizationEnvironment.evaluate(candidate)
    -> EpisodeObservation (valid or explicit missing endpoint)
    -> effective five-parameter gray-box physics prediction
    -> residual r = observation - physics prediction
    -> fixed Matern-5/2 residual Gaussian process
    -> posterior mean = physics + residual mean; sigma = residual sigma
    -> lower-confidence-bound acquisition
    -> next unexecuted V3 candidate
```

The gray-box answers: “Given the currently available subject data, what
mechanical response does the effective physics model predict?” Its parameters
remain effective gray-box quantities:

```text
mass_scale, K_hip, K_knee, B_hip, B_knee
```

They are not reinterpreted as physiological parameters. The full-dynamics
adapter reuses the existing V3 generator, existing five-parameter subject
construction, and existing inverse dynamics. It derives a temporary compatible
offline endpoint as the episode RMS magnitude of the predicted hip/knee torque
vector. This projection is not frozen as the future measured endpoint.

The residual GP answers: “In which beta regions is the physics prediction
systematically wrong?” It uses only valid observations from completed trials.
Its input is `[beta_flex, beta_extend]`; its primary kernel is a fixed,
deterministic Matern-5/2 kernel. There is no per-case oracle tuning.

BO answers: “Given the remaining fixed trial budget, which unexecuted
candidate is most useful next?” The primary acquisition minimizes
`LCB = mean - 1.5 * standard_deviation`.

## Frozen candidate identity

The implementation reads and verifies the existing
`P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3` artifact. It does not rewrite the
V3 equation or artifact. The domain contains 625 candidates on a 25-by-25 grid
from -0.03 to +0.03 at step 0.0025. `MYOLEG_V3_K0312` at `[0,0]` is the exact
Trial-1 cold start.

## Observation and environment boundary

`EpisodeObservation` carries episode/trial identity, V3 identity and beta,
endpoint name/value/unit/uncertainty, validity, invalid reason, and metadata.
A valid episode requires a finite value. An invalid episode requires `None`;
zero imputation is rejected.

Available environments are:

- `AnalyticBenchmarkEnvironment`: labelled `OFFLINE_ALGORITHM_TEST_CASE`, with
  deterministic smooth-convex, anisotropic, rotated, mildly nonlinear, noise,
  biased-prior, invalid-episode, and outlier facilities. It is never called a
  virtual patient.
- `FrozenOfflineReplayEnvironment`: owns the complete replay mapping but
  reveals only the requested executed candidate observation.
- `RealRobotEnvironment`: disabled and fail closed.

The benchmark evaluator owns analytic optimum access. Selectors and fitted
models receive no environment or evaluator reference, and the run asserts that
oracle access remains zero until adaptation finishes.

## Sequential and validity policies

Primary budget is K=4; K=3 and K=5 are sensitivity settings. Trial 1 is always
the reference. Selection for Trial k is completed using only Trials 1 through
k-1. `ExecutedCandidateLedger` records the observation, model summaries,
selector, acquisition value, and selected next candidate at every step.

`INVALID_EPISODE_POLICY` is
`INVALID_CONSUMES_BUDGET_EXCLUDED_FROM_FIT_NO_COVERT_RETRY`. Invalid trials
remain in the ledger, consume K, and never enter normal physics/GP fitting.

`NO_DUPLICATE_CANDIDATE` is the adaptive default. Reference-only is the one
explicit repeated-measurement baseline. Numerical BO cannot silently return an
executed candidate.

After K trials, `best_observed_candidate` and
`model_recommended_final_candidate` are stored separately. Analytic truth is
used only afterward by `OFFLINE_ALGORITHM_DEVELOPMENT_EVALUATION` to compute
regret.

## Equal-budget comparison

Six methods use the same V3 domain, reference cold start, K, and deterministic
paired noise rule:

1. Reference
2. Random
3. Space Filling
4. Model-Only Greedy
5. Standard BO (outcome-only GP; no physics access)
6. Physics-Informed BO (physics prior plus residual GP)

The formal smoke contains four analytic landscapes, P0/P1/P2/P3 physics-prior
quality, zero/low/moderate development noise, five deterministic seeds, and all
six methods: 1,440 primary K=4 runs. Metrics include final regret, best-seen
regret, per-trial simple regret, beta distance to optimum, invalid count,
diversity, and duplicate count. K=3/K=5 sensitivity is also recorded.

At K=4, Physics-Informed BO versus Standard BO produced:

| Prior | Noise | Mean final regret: physics BO | Mean final regret: standard BO | W/T/L |
|---|---:|---:|---:|---:|
| P0 | zero | 0.00127 | 0.12620 | 20/0/0 |
| P0 | low | 0.00362 | 0.10765 | 20/0/0 |
| P0 | moderate | 0.02126 | 0.11198 | 16/0/4 |
| P1 | zero | 0.00211 | 0.12620 | 20/0/0 |
| P1 | low | 0.00325 | 0.10765 | 20/0/0 |
| P1 | moderate | 0.01983 | 0.11198 | 17/0/3 |
| P2 | zero | 0.02877 | 0.12620 | 15/0/5 |
| P2 | low | 0.03494 | 0.10765 | 15/0/5 |
| P2 | moderate | 0.04870 | 0.11198 | 14/0/6 |
| P3 | zero | 0.17776 | 0.12620 | 5/0/15 |
| P3 | low | 0.18046 | 0.10765 | 5/0/15 |
| P3 | moderate | 0.20071 | 0.11198 | 3/1/16 |

Thus the offline algorithm evidence supports physics-informed residual BO when
the prior is accurate, mildly biased, or trend-informative with local mismatch.
It also exposes the required failure boundary: a directionally poor P3 prior
dominates the first four trials and hurts relative to standard BO.

## Limitations and future boundary

- The benchmark endpoint and analytic physics adapter are development-only.
- Four trials cannot identify five effective parameters without strong prior
  regularization; a future validated episode may carry richer time-series
  features to the existing gray-box estimator.
- The benchmark is not a noise calibration, patient model, or MyoLeg
  held-out-truth result.
- No method currently detects and disables a poor physics prior online. P3 is
  deliberately reported as a failure case.
- The stage-specific test module passes 18/18. The repository-wide suite is not
  green in the merged starting tree: `1466 passed, 1 skipped, 269 failed, 183
  errors`. The first fail-fast error is an existing frozen protocol SHA
  mismatch in
  `external_simulation/test_measurement_driven_personalization_data_and_endpoint_design_v1.py`;
  many subsequent errors similarly report pre-existing formal-artifact hash
  mismatches. This stage did not modify those frozen inputs, so the requested
  repository-wide `0 failed` condition cannot be claimed or repaired by
  rewriting them here.
- A future validated real observation can enter through
  `PersonalizationEnvironment` without changing the selector/GP/ledger loop,
  but the environment, endpoint projection, uncertainty semantics, and robot
  approval must be independently implemented and validated first.

The next scientifically justified step is to validate and freeze the measured
mechanical `EpisodeObservation` endpoint and repeatability/uncertainty contract,
then connect a development-side frozen replay through the existing environment
boundary. It is not PINN training or robot execution.
