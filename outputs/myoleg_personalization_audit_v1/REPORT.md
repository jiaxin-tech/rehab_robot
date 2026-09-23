# MyoLeg personalization identifiability audit V1

Date: 2026-09-23
Scope: 24 frozen development subjects; no sealed subjects; no measurement noise.
## Result

The full evaluator landscape contains 17,064 rows: 24 subjects × 324 `BETA_TIMING` candidates and 24 subjects × 387 `KEY_POSTURE_TIMING` candidates. Each family has one feasible oracle shared by all 24 subjects:

| Family | Shared oracle | Feasible candidates per subject | Median Spearman rank | Interaction fraction |
|---|---:|---:|---:|---:|
| `BETA_TIMING` | `BETA_TIMING:288` | 131–144 | 0.999800 | 0.1547% |
| `KEY_POSTURE_TIMING` | `KEY_POSTURE_TIMING:383` | 130–141 | 0.999873 | 0.1251% |

The evaluator-selected common candidate is the same as each subject oracle, so relative common regret is zero for every subject in both families. This is a task-landscape result, not evidence that the selector is clinically personalized.

The executed-pool comparison separates exploration from personalization. At K=4, `PHYSICS_GREEDY`, `RESIDUAL_GP_GREEDY`, and `MODEL_INFORMED_BO_EI` capture the full oracle for every subject in both families. Pure BO and Random have positive regret because their small trial pool misses the oracle; this is an exploration-budget effect. `pool_vs_full_summary.csv` contains all budgets and methods.

## Scope and isolation

`scope_audit.json` reports zero sealed model-delta reads, zero sealed truth-array reads, and zero full-landscape/oracle values delivered to a learner. The evaluator reads the complete landscape only after the causal benchmark runs have finished.

## Decision

Do not expand V1 seeds or tune `mass_scale` to create oracle diversity. Keep V1 as the null-control for an evidence-gated method. The next experiment must validate held-out gray-box prediction and then use a separately frozen controlled positive cohort to test whether an observation-driven residual model can improve equal-budget regret and constraints when a real interaction exists.
