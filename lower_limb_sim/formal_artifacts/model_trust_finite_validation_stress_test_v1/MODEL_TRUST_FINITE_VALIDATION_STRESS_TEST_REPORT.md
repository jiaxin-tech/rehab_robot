# MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_V1

- Protocol SHA-256: `abaad1f7ab56c7ccd97124ec995810f21f1f331b4e87d3243feb45db0a426b04`
- Frozen baseline manifest SHA-256: `17303f532350853755dcbd33b60d6c5bd3d57b30f888a0b04fabe2c643b9d0f6`
- Integrity: `STRESS_TEST_PROTOCOL_INTEGRITY = PASS`
- Evidence: `OFFLINE_ONLY`, `NOT_HUMAN_READY`, `NOT_ROBOT_APPROVED`

## Scope and interpretation

This stage did not tune or replace the frozen V1 method. It evaluated the same
15 existing virtual cases, covering all nine existing mismatch scenario
definitions. No global mismatch severity scalar was invented; results are
reported by family and defined level. B5 Oracle was revealed only after every
baseline candidate identity and Random-3 seed had been frozen and persisted.

## Overall method comparison (Table A)

| method | validation_budget | case_count | evaluation_row_count | mean_J | mean_regret | median_regret | P95_regret | max_regret | near_optimal_at_0.001 | near_optimal_at_0.0025 | near_optimal_at_0.005 | final_harmful_selection_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_REFERENCE | 0 | 15 | 15 | 1.000000 | 0.035042 | 0.032367 | 0.055930 | 0.060978 | 0.000000 | 0.000000 | 0.000000 | 0 |
| B1_MODEL_ONLY | 0 | 15 | 15 | 0.967272 | 0.002314 | 0.001425 | 0.006624 | 0.006980 | 0.333333 | 0.733333 | 0.800000 | 0 |
| B2_RANDOM3_FINITE_VALIDATION | 3 | 15 | 1500 | 0.979757 | 0.014799 | 0.013369 | 0.032486 | 0.060978 | 0.018000 | 0.050000 | 0.141333 | 0 |
| B3_MODEL_TOP1_VALIDATION | 1 | 15 | 15 | 0.967272 | 0.002314 | 0.001425 | 0.006624 | 0.006980 | 0.333333 | 0.733333 | 0.800000 | 0 |
| B4_FROZEN_TOP3_SEQUENTIAL | 3 | 15 | 15 | 0.967272 | 0.002314 | 0.001425 | 0.006624 | 0.006980 | 0.333333 | 0.733333 | 0.800000 | 0 |
| B5_ORACLE | N/A | 15 | 15 | 0.964958 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |

Primary metric is final regret, not exact optimum hit rate. Near-optimal success
is shown at all preregistered tolerances (0.001, 0.0025, 0.005).

## Screening and ranking utility

- Model Top-3 mean regret: `0.002314`.
- Random-3 mean regret: `0.014799`.
- Mean model strict-beat percentile within Random-3 distributions:
  `95.3%`.
- Mean truth-top 1%/5%/10% enrichment: `19.928` /
  `6.662` / `4.665`.
- Median truth rank percentile of C1/C2/C3:
  `{'C1': 0.5993150684931506, 'C2': 11.28234398782344, 'C3': 21.58009893455099}`.

Screening conclusion: `MODEL_SCREENING_SUPPORTED`.

## Top-1 versus Frozen Top-3 (Tables C and D)

- Wins/ties/losses from the Top-3 perspective:
  `0/15/0`.
- Mean paired regret reduction (Top1 − Top3):
  `0.000000`.
- Median paired reduction: `0.000000`.
- Bootstrap 95% CI: `[0.000000,
  0.000000]`.
- Paired Cohen dz: `0.0`.

Scenario-family concentration:

- `combined`: Top-3 wins 0, ties 2, Top-1 wins 0.
- `hip_knee_coupling`: Top-3 wins 0, ties 2, Top-1 wins 0.
- `matched_linear`: Top-3 wins 0, ties 7, Top-1 wins 0.
- `nonlinear_damping`: Top-3 wins 0, ties 1, Top-1 wins 0.
- `nonlinear_stiffness`: Top-3 wins 0, ties 2, Top-1 wins 0.
- `structured_residual`: Top-3 wins 0, ties 1, Top-1 wins 0.

Top-1/Top-3 conclusion: `TOP1_SUFFICIENT`.

## Trial-budget ablation (Table E)

- K=0: mean regret 0.002314, median 0.001425, P95 0.006624.
- K=1: mean regret 0.002314, median 0.001425, P95 0.006624.
- K=2: mean regret 0.002314, median 0.001425, P95 0.006624.
- K=3: mean regret 0.002314, median 0.001425, P95 0.006624.

K=5 was preregistered as skipped because the frozen V1 selection helper enforces
at most three equivalence-band representatives. Extending it would redesign the
selection rule, which this stage forbids.

Budget conclusion: `FINITE_VALIDATION_BUDGET_SATURATES_AT_K_0`.

## False improvement

Across the formal model-screened candidate pools, `697/145157`
candidates had `J_pred < 1` but `J_truth >= 1`. Model-only made
`0` final harmful selections because fallback was deliberately
disabled. Validation methods retained Reference fallback; zero final harmful
selections there does not erase the model's prediction errors.

## Mismatch and trust limit

Increasing mismatch is not forced onto one scalar axis. Mild/strong comparisons
are valid only within the explicitly paired stiffness, coupling, and combined
families. The current diagnostic conclusion is
`MODEL_TRUST_LIMIT_NOT_IDENTIFIED`. A family-level trust limit is
declared only when at least two existing cases in that predefined family meet
the frozen collapse rule; isolated failures remain failure-regime observations,
not a generalized threshold.

## Direct answers

### Q1 Does the five-parameter model provide useful ranking/screening information?

`MODEL_SCREENING_SUPPORTED`. This answer uses regret,
equal-budget Random-3 distributions, and top-fraction enrichment together.

### Q2 Does model screening outperform equal-budget random finite validation?

`YES`.
Top-3's mean regret is `0.002314` versus Random-3
`0.014799`; per-case distribution evidence is in
`MODEL_TOP3_VS_RANDOM3.csv`.

### Q3 Does Frozen Top-3 provide meaningful benefit beyond Top-1?

`TOP1_SUFFICIENT`. Top-1 mean regret is
`0.002314` and Top-3 mean regret is `0.002314`.

### Q4 How does increasing model mismatch affect final regret and screening utility?

`NOT ESTABLISHED` as one global monotonic relation because the existing truth
definitions do not share a scientific scalar severity. Family-specific
mild/strong and scenario results are reported without pooling incompatible
mismatch mechanisms.

### Q5 How many full-cycle validation trials are empirically justified?

`FINITE_VALIDATION_BUDGET_SATURATES_AT_K_0`. Evidence beyond K=3 was not generated;
K=5 would require changing the frozen rule.

## Final status

- `MODEL_SCREENING_SUPPORTED`
- `TOP1_SUFFICIENT`
- `FINITE_VALIDATION_BUDGET_SATURATES_AT_K_0`
- `MODEL_TRUST_LIMIT_NOT_IDENTIFIED`
- `BO_BASELINE_REQUIRED_NEXT = true`

No BO, new optimizer, prospective cohort, human experiment, or robot connection
was performed. The frozen V1 source and artifact hashes were checked before and
after this independent stage.
