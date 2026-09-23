# MODEL_INFORMED_BO_ARCHITECTURE_V2

## Estimator contract inspected before integration

`lower_limb_sim.parameter_estimator.estimate_subject_parameters` is the sole
optimizer for the new path. Its five parameters remain `mass_scale`,
`k_hip_nm_per_rad`, `k_knee_nm_per_rad`, `b_hip_nm_s_per_rad`,
`b_knee_nm_s_per_rad`: **EFFECTIVE_FIVE_PARAMETER_GRAY_BOX**, not physiological
parameters.

Required DataFrame columns are `q_hip_rad`, `q_knee_rad`, `dq_hip_rad_s`,
`dq_knee_rad_s`, `ddq_hip_rad_s2`, `ddq_knee_rad_s2`, `fx_observed_n`,
`fz_observed_n`, `sample_valid`. Optional `force_mapping_valid` and
`wrench_is_stale` further filter samples. Nonfinite required values are excluded;
no valid samples raises. Truth-labelled columns are rejected. Angles retain
`theta_shank = q_hip - q_knee`. Fx/Fz must already represent the project planar
robot-on-leg equivalent pull-point force: measured torque is `J(q).T @ F` with
explicit L1 and strap-equivalent L2. This integration performs no robot-axis
mapping or geometry validation.

The estimator consumes already-derived q/dq/ddq. It neither reads timestamps
nor differentiates, interpolates or integrates. Concatenating independent,
already-aligned episode rows is therefore safe: timestamps can restart in each
episode; no inter-episode interval or synthetic sample is created. Each accepted
sample contributes two torque residuals. Joint scales are the pooled observed
torque standard deviations with the existing 1 N m numerical scale floor.
There is no endpoint-uncertainty weighting or parameter-prior regularizer in
this estimator. The unchanged default objective uses `soft_l1`, f_scale=1,
max_nfev=500, existing parameter scales and bounds from `lower_limb_sim/config.py`.
Thus it is a robust scaled sample-residual objective, not unweighted raw SSE.

Returned diagnostics include optimizer success/message, cost, nfev, torque
residual metrics, Jacobian singular values, covariance, parameter standard
errors, torque scales, valid sample count and loss. Numerical rank/conditioning
do not establish structural identifiability. The existing single-episode API
and legacy scalar-regularized adapter are retained.

## Implementation scope and status

This is an offline architecture integration, not a new scientific algorithm
comparison or real-measurement qualification. V3, ROM policy, estimator math,
five-leg definitions, existing E2 artifacts, GP kernel settings, robot code and
historical alpha-EI implementations were not modified. Old method names remain
LCB; `METHODS` used by historical runners is unchanged. The scalar adapter is
explicitly documented as legacy/development and remains operational.

```text
TIME_SERIES_FIVE_PARAMETER_IDENTIFICATION_INTEGRATION = COMPLETE_WITH_LIMITATIONS
CURRENT_V3_EXPECTED_IMPROVEMENT = IMPLEMENTED
LCB_SENSITIVITY_PATH = PRESERVED
E0_GRAYBOX_ENDPOINT = PRESERVED
E2_GRAYBOX_ENDPOINT = IMPLEMENTED
RESIDUAL_GP_ENDPOINT_SEMANTICS = CONSISTENT
MODEL_INFORMED_BO_EI_TIMESERIES_ID = RUNNABLE_OFFLINE
LEGACY_SCALAR_GRAYBOX = PRESERVED
FROZEN_E2_NECESSITY_RESULT_CHANGED = false
ROBOT_CODE_MODIFIED = false
```

The identification limitation is scientific/measurement scope: numerical fitting
and its diagnostics are integrated, but do not establish structural
identifiability, physical force mapping, geometry validity or real parameter
accuracy. The software does not silently substitute scalar fitting if payloads
are absent, nor declare success when the optimizer fails.

## Why the two observation channels are separate

A scalar RMS/E2 value compresses an entire trajectory. Four scalar values do not
by themselves independently constrain five unknown model parameters; the legacy
regularized scalar fit is a development surrogate, not equivalent to the
sample-level estimator. The new path does not delete that historical method.

`personalization/identification.py:TimeSeriesIdentificationPayload` is an
optional immutable payload on `EpisodeObservation.identification_payload`.
It contains episode/candidate identity, ROM profile/version/fingerprint,
reference version, beta, episode-local time, q/dq/ddq, project planar Fx/Fz,
sample-valid mask, L1/L2, force-mapping declaration, mapping provenance,
evidence classification and payload validity/reason. Arrays are copied into
immutable tuples. Nonfinite samples are filtered by the existing estimator;
JSON serialization represents missing numerical entries as `null`, not zero.
Timestamps must be finite and strictly increasing **within** each payload.

Only explicit project-defined synthetic mapping or an explicitly validated
project mapping declaration is accepted. These strings record the caller's
provenance; they do not perform physical validation. Arbitrary robot Cartesian
axes are not mapped to project planar force by this code.

- **Channel A:** valid payload sample arrays → five-parameter estimator.
- **Channel B:** valid scalar endpoint and uncertainty → endpoint residual GP.

Changing J alone leaves theta unchanged; changing force samples can change theta.
GP still has one target per valid scalar episode, not one target per time sample.
A valid scalar episode without valid identification data can still contribute to
the GP after an earlier successful ID, but contributes no physics samples.
With no usable ID episodes at all, fitting fails with
`TIMESERIES_ID_REQUIRES_VALID_IDENTIFICATION_EPISODE`.

The payload is an already-preprocessed data contract. It does not certify full
physical cycle coverage, timestamps against robot clocks, or derivative quality.
The supplied synthetic integration provider evaluates all 401 samples of each
requested V3 trajectory.

## Time-series adapters and model update

New file: `personalization/models/time_series_graybox.py`.

`TimeSeriesFiveParameterGrayBoxAdapter.fit(history)`:

1. Checks endpoint name/unit and candidate ID/beta against its subject domain.
2. Uses only valid observations with present, valid payloads; verifies payload
   episode/candidate/beta, ROM and reference identity, and L1/L2.
3. Applies existing finite/valid sample filtering independently to each episode.
4. Concatenates only those existing rows; keeps episode IDs and local timestamps.
   It adds no boundary samples. Longer valid episodes retain the estimator's
   existing per-sample contribution rather than artificial equal-episode weights.
5. Calls the original `estimate_subject_parameters`, warm-started from the last
   accepted theta. Repeated unchanged usable histories do not refit; adding valid
   ID data does. Invalid scalar trials still consume the existing trial budget.
6. Commits theta only after optimizer success, finite estimates and existing
   parameter bounds. Failure raises and preserves the previous accepted theta;
   predictions are blocked after a failed update until a successful new fit.
7. Exposes all estimator diagnostics, last-attempt diagnostics/error, accepted
   episode/sample counts, sample counts by episode, successful update count,
   bounds status and singular-value condition ratio. No condition-number or
   structural-identifiability pass threshold is invented.

`theta_hat` returns a copy as a named mapping. The successful fit state is used
by `predict_torques` to call existing `candidate_subject_from_parameters` and
`inverse_dynamics` on the subject-specific candidate q/dq/ddq. Predictions over
625 candidates are evaluated on demand by the selector; a full truth landscape
is neither passed in nor read. Known baseline template and geometry are explicit
configuration inputs, not inferred from simulator truth.

`PhysicsSubjectModel.endpoint_name` / `endpoint_unit` expose adapter identities.
Legacy adapters remain permissive for their historical endpoint names. New
adapters require exact canonical name/unit matches before fitting. New
`run_offline_configuration` also checks scalar-only Pure BO endpoint identity
before GP fitting. No alias automatically treats E0 as E2 or torque units as a
reference ratio.

## E0 and E2 endpoints

Reusable pure functions are in `lower_limb_sim/mechanical_endpoints.py`; they
have no MuJoCo, robot or cohort dependency. Frozen five-leg code/artifacts were
left intact and used for equivalence tests, not regenerated.

| Configuration | Canonical endpoint name | Unit | Adapter |
|---|---|---|---|
| E0 | `E0_FULL_CYCLE_DUAL_JOINT_RMS` | `N_m` | `TimeSeriesFiveParameterGrayBoxAdapter` |
| E2 | `E2_BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS` | `dimensionless` | `BranchBalancedE2GrayBoxEndpointAdapter` |

E0 uses the existing time-weighted full-cycle formula:

```text
E0 = sqrt(trapezoid(tau_hip(t)^2 + tau_knee(t)^2, t) / (t[-1] - t[0]))
```

The legacy `FullDynamicsGrayBoxEndpointAdapter` and subject-specific scalar
subclass retain their original names and E0-like predictions/optimizer. The new
canonical E0 identity is used by the explicit new adapter, not retroactively
written into historical logs.

`branch_rms_components` takes tau_hip/tau_knee, time and explicit
`flexion`/`extension` labels, computing trapezoidal RMS within each branch:

```text
R_beta = [hip_flex_RMS, hip_extend_RMS, knee_flex_RMS, knee_extend_RMS]
E2(beta; theta) = max(R_beta(theta) / R_reference_beta_zero(theta))
```

There is no averaging, weighting, peak term or cohort normalization. Increasing
one limiting component cannot be cancelled by decreases in other components.
Reference components must be finite and positive. The only denominator guard
is `np.finfo(float).tiny` (floating-point underflow protection); no physical
threshold/floor is invented. Nonfinite ratios fail explicitly.

`BranchRMSReference` carries a `MechanicalReferenceContext`. The pure E2 helper
rejects a different context. The E2 model constructs that context from:

```text
ROM profile ID + version + fingerprint
+ subject reference version
+ full theta + L1/L2 + known baseline template values
```

Every successful fit invalidates its reference cache; the next prediction
recomputes reference torques/RMS under current theta. Cache access also compares
the complete context. Replacing the bound ROM/reference requires a new adapter.
Tests reject cross-leg, cross-ROM, cross-reference and cross-theta normalization.
The synthetic provider separately uses its own same-leg reference response for
observed J; it never hands hidden theta to the selector/model.

Both endpoint adapters remain **MODEL_DERIVED / NOT_REAL_MEASUREMENT_VALIDATED**.
In particular, robot wrench channels do not yet directly establish the hip/knee
branch torque measurements required for a real E2 endpoint.

## EI, LCB and explicit method names

`personalization/selectors/bo.py:expected_improvement` and
`ExpectedImprovementSelector` implement **current 2D V3** EI. They do not call
or replace the legacy three-dimensional alpha runner.

For minimization:

```text
J_best = min(valid executed EpisodeObservation.endpoint_value)
a = J_best - mu - xi
z = a / sigma
EI = a * Phi(z) + sigma * phi(z)
select argmax EI
```

Default xi=0, not tuned. For sigma≤1e-12 the deterministic limit is max(a,0).
Ties use candidate_index. Without a valid measured incumbent, the selector raises
`EI_REQUIRES_VALID_INCUMBENT`; it does not invent a reference or oracle value.

LCB remains:

```text
LCB = mu - kappa * sigma
select argmin LCB; default kappa=1.5
```

EI measures expected improvement over an actual measured incumbent; LCB trades
predicted mean against uncertainty without that incumbent-dependent integral.
They can intentionally choose different points, as tested using a constructed
posterior. Neither xi nor kappa was tuned in this task.

`selectors/base.py:valid_unexecuted_predictions` is shared by EI, LCB and greedy:
only unexecuted domain members with `Prediction.valid=True`, finite mean/std and
nonnegative std are ranked. If none remain, it fails clearly. This is numerical
prediction-validity filtering, **not physical safety approval**. A domain with
no unexecuted members retains its explicit domain-exhaustion error.

Explicit names supported by `personalization/sequential.py`:

```text
PURE_BO_EI
PURE_BO_LCB
MODEL_INFORMED_BO_EI
MODEL_INFORMED_BO_LCB
MODEL_ONLY_GREEDY
MODEL_INFORMED_BO_EI_TIMESERIES_ID
```

The last name additionally requires the new time-series adapter. Historical
`Standard BO`, `Physics-Informed BO`, `Model-Only Greedy` and their runner method
lists remain available with their original acquisition meanings. LCB/greedy
behavior changes only when predictions are invalid/nonfinite (now excluded).
Historical serialized runs may gain additive acquisition metadata; frozen
artifacts were not overwritten.

## Residual GP and acquisition ledger

The existing `PhysicsInformedResidualModel.fit` is reused unchanged:

```text
physics.fit(past observations) → theta_k
for each past valid executed observation i:
    residual_i = J_observed_i - physics.predict(candidate_i; theta_k)
ResidualGaussianProcess.fit_arrays(beta_i / 0.03, residual_i, endpoint_noise_i)
```

All residual targets are recomputed after the physics update; no stale-theta
residual survives. The GP remains two-dimensional, fixed Matérn-5/2
(length_scale=.7, signal_std=.6), with the existing observation-noise/jitter
handling. Total predictive mean is physics mean + residual mean; predictive std
is residual GP std. Parameter covariance is reported but not propagated into
that std; this limitation is unchanged and explicit. Missing endpoint uncertainty
continues to use the legacy GP numerical-noise convention, not proof of zero real
measurement noise.

`Selection.metadata` records acquisition type/value, predictive mean/std,
measured incumbent, xi/kappa where applicable, candidate ID/beta and target trial
index. `LedgerEntry.next_selection_metadata` stores it on the observation entry
that caused the **next** selection. Thus entry k describes selection for k+1;
reference initialization is prescribed, not an acquisition optimization.
The final entry has no next-selection metadata.

## Integrated configuration and actual architecture map

Entry point: `personalization/integrated_v2.py:run_offline_configuration` with
`OfflineBOConfiguration(endpoint="E0" or "E2")`. Its default method is
`MODEL_INFORMED_BO_EI_TIMESERIES_ID`.

It requires a frozen `SubjectSpecificV3CandidateDomain` with 625 unique beta
pairs and an explicitly offline environment. Physics methods additionally
require the known baseline template and project L1/L2. Pure BO does not construct
a physics adapter. Domain creation remains the existing frozen V3 implementation,
25×25 points over [-.03,.03]²; neither its generator nor ROM policy was changed.

```text
SubjectROMProfile
  personalization/rom_gated_v2/rom.py:SubjectROMProfile
→ SubjectSpecificReference
  personalization/rom_gated_v2/reference.py:SubjectSpecificReferenceAdapter.adapt
→ SubjectSpecificV3CandidateDomain.from_frozen_beta_grid
  same file; uses unchanged generate_v3_trajectory
→ run_offline_configuration / run_sequential_personalization
  personalization/integrated_v2.py / sequential.py
→ environment.evaluate(full candidate trajectory)
  offline_time_series.py:ModelConsistentTimeSeriesEnvironment (test provider)
→ EpisodeObservation + TimeSeriesIdentificationPayload
  personalization/observations.py / identification.py
→ PhysicsInformedResidualModel.fit → PhysicsSubjectModel.fit
  personalization/models/residual_gp.py / physics_graybox.py
→ TimeSeriesFiveParameterGrayBoxAdapter.fit
  personalization/models/time_series_graybox.py
→ estimate_subject_parameters → theta_k
  lower_limb_sim/parameter_estimator.py (unchanged)
→ E0 or BranchBalancedE2GrayBoxEndpointAdapter prediction
  time_series_graybox.py + lower_limb_sim/mechanical_endpoints.py
→ recomputed scalar residual targets → ResidualGaussianProcess
  personalization/models/residual_gp.py
→ ExpectedImprovementSelector → Selection
  personalization/selectors/bo.py / base.py
→ ExecutedCandidateLedger (entry k holds next-selection metadata)
  personalization/ledger.py
→ next full trajectory → repeat
```

Pure BO EI:

```text
same domain + reference + endpoint + K_total
→ environment.evaluate
→ endpoint/identity checked observation; identification payload removed
→ StandardGaussianProcess.fit(scalar J only)
→ ExpectedImprovementSelector → ledger → next trial
```

`PURE_BO_*` also strips payloads at the sequential-core boundary, so directly
using those explicit method names does not pass Channel A to Pure BO.

Adaptive Greedy time-series baseline:

```text
same domain + reference + endpoint + K_total
→ same TimeSeriesFiveParameterGrayBoxAdapter.fit(same available history)
→ same theta / E0 or E2 prediction
→ ModelOnlyGreedySelector: argmin model mean
→ ledger → next trial
```

Greedy does not build the residual GP. On identical identification histories its
physics estimate is identical to the model-informed method's physics estimate;
after different selections their available histories can naturally diverge.
The LCB sensitivity configuration uses the same ID/endpoint/residual components
as the EI model-informed path. Pure EI versus model-informed EI matches
acquisition family; no new scientific method comparison was run here.

Minimal use with existing prepared objects (not a robot command):

```python
from personalization.integrated_v2 import OfflineBOConfiguration, run_offline_configuration

result = run_offline_configuration(
    offline_environment, subject_v3_domain,
    configuration=OfflineBOConfiguration(endpoint="E2", budget=4),
    baseline_template=known_template, L1=project_L1, L2=project_strap_L2,
)
```

`ModelConsistentTimeSeriesEnvironment` exists for controlled software tests only.
It evaluates requested trajectories, derives torques from one supplied synthetic
theta, maps them with the existing project Jacobian, and emits no hidden-theta or
full-landscape truth payload. It creates no new cohort or frozen MuJoCo leg and
has no robot dependency. Its exact-model recovery tests are not real validation.

## Budget and final-selection compatibility

`K_total` still includes reference; default4 means reference+3 additional trials.
`K_additional = K_total - 1`. Stage 0 cost remains outside this budget. No budget
API migration, retries or free diagnostic trials were introduced.

Both existing outputs remain:

- `best_observed_candidate`: valid measured-best executed candidate.
- `model_recommended_final_candidate`: model mean minimum, potentially unexecuted.

EI uses only the first concept's measured value as incumbent. Existing benchmark
`final_regret` still evaluates model recommendation, and `best_seen_regret` still
has its previously reviewed oracle-based semantics. Neither historical metric
was silently changed. Final model recommendation filtering/qualification is
also not redesigned; numerical acquisition filtering is not a guarantee about
that separate legacy recommendation path.

## Tests and evidence boundary

Existing directly related regression: **186 distinct existing tests passed**,
covering the estimator, legacy alpha EI, real-episode offline adapter, V3, subject
ROM, sequential GP/BO, all retained robustness variants, five-leg MuJoCo,
endpoint design/E2 necessity, discriminability and measurement validation.

New tests: **45 passed** across
`tests/test_model_informed_bo_architecture_v2.py` (43 parametrized cases) and
`tests/test_e2_shared_endpoint_v2.py` (2). Coverage includes controlled-theta
recovery, two-episode refinement, invalid/missing payload exclusion, exact
802-row two-episode concatenation with timestamps restarting at row401, failed
optimizer non-commit, strict identity/units, independent channels, future-data
causality, EI math/validity/incumbent, greedy/LCB use of the same adapter, Pure BO
isolation, and three-trial 625-domain E0/E2 integration.

E2 equivalence checks compare all **3,125 frozen feature rows** with both stored
E2 and the original endpoint function, plus the old/new branch-time RMS formula
on nonuniform time samples. Tolerance is 1e-14. The frozen necessity artifact
still reports NOT_SUPPORTED and not ready for an E2 scientific algorithm
comparison. No artifact was regenerated.

Tests were run in batches. The final rerun of newly added tests and affected
sequential/ROM/robustness tests was **150 passed in 16.09s**; combined distinct
coverage is **231 passing tests**, not a full repository run. An initial new
E2-equivalence test used a field absent from the compact frozen feature table;
its input lookup was corrected, without modifying algorithms or frozen data.

Commands used `python3` and the already prepared temporary MuJoCo3.6 environment
`/private/tmp/rehab_architecture_review_venv/bin/python -m pytest -q <test files>`.
Logs: `/private/tmp/rehab_bo_v2_new_tests.log`,
`/private/tmp/rehab_bo_v2_regression.log`,
`/private/tmp/rehab_bo_v2_final_tests.log`. Temporary logs are software checks,
not formal experiment artifacts. No full-repository pass is claimed.

## Scientific status and remaining real integration

```text
SIMULATED_MECHANICAL_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
E2 = MODEL_DERIVED
E2 = NOT_REAL_MEASUREMENT_VALIDATED
```

Correcting an architecture cannot create decision-relevant subject×trajectory
interaction evidence. No new five-leg tuning, endpoint selection, clinical
claim, comfort claim, patient personalization claim or PINN training occurred.

Before real use, the project still needs independently validated force
frame/sign/reference point and project geometry, valid timing/alignment and
q/dq/ddq reconstruction, suitable full-cycle data, endpoint justification and
uncertainty, repeatability/sensitivity evidence, and cross-subject decision
relevance before personalization is activated. The measurement-gate issues in
`ALGORITHM_ARCHITECTURE_REVIEW_V1.md` were not part of this task and remain.
Actual V3 execution/recorded-data conversion and binding to an approved real
measurement environment are still separate work. The new explicit offline
entry does not connect a robot, change thresholds or lift existing gates.

## Q1–Q12

**Q1. What exact data enter identification?** Valid past episode q/dq/ddq and
explicitly project-mapped Fx/Fz with sample-valid flags, known template and
L1/L2. Payload/candidate/ROM/reference/endpoint identities are checked first.

**Q2. Scalar or full residual time-series?** Each accepted trajectory contributes
its valid sample-level hip and knee torque residuals. Scalar J never estimates
theta in the new adapter.

**Q3. How are episodes combined?** Concatenate existing valid rows with episode
IDs/local time retained. The unchanged estimator computes pointwise residuals;
no differentiation, interpolation or integration crosses a boundary.

**Q4. Which diagnostics?** Optimizer success/message/cost/nfev, torque residual
statistics/scales, singular values/condition ratio, covariance/standard errors,
bounds status, episode/sample counts and last-attempt failure information.
No structural-identifiability pass threshold is invented.

**Q5. Exact E0?** sqrt(integral(tau_hip²+tau_knee²)dt / duration), trapezoidal
integration, N_m.

**Q6. Exact E2?** max of hip-flex, hip-extend, knee-flex and knee-extend RMS ratios
to the same subject/ROM/reference/current-theta beta-zero model response.
Dimensionless; no peaks or weights.

**Q7. Theta changes and normalization?** Successful fit clears the E2 reference
cache. On next access the reference is recomputed; the cache context additionally
contains full theta, template, L1/L2, ROM and reference identities.

**Q8. Exact EI and incumbent?** a Phi(a/sigma)+sigma phi(a/sigma),
a=min(valid measured executed J)−mu−xi, defaultxi=0. Near-zero sigma uses
max(a,0). No incumbent raises explicitly.

**Q9. Exact LCB?** mu−kappa sigma, minimize; defaultkappa=1.5, unchanged.

**Q10. Primary and sensitivity?** Intended future offline method:
MODEL_INFORMED_BO_EI_TIMESERIES_ID. Sensitivity: MODEL_INFORMED_BO_LCB with the
same time-series adapter/configured endpoint. Neither is currently a scientifically
justified deployed personalization method.

**Q11. What is offline/model-derived?** Synthetic time-series provider, theta
recovery evidence, predicted mechanics, E0/E2 landscapes and BO software tests.
They do not validate real measurement semantics, patient-specific benefit,
comfort or clinical efficacy.

**Q12. What remains for real measurements?** Validated planar force and geometry
mapping, synchronization/derivatives/full-cycle data qualification, real endpoint
validation and uncertainty, evidence gates and decision-relevant interaction,
plus an explicitly approved real execution/data adapter. None is inferred from
this successful offline integration.
