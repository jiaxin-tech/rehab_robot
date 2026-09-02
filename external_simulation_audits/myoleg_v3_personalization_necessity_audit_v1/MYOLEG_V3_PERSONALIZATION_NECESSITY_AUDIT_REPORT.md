# MyoLeg V3 Personalization Necessity Audit V1

## Formal decision

`V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED`

Recommended next scientific branch: `MYOLEG_OBJECTIVE_AND_MUSCULOSKELETAL_HETEROGENEITY_DECISION_AUDIT_V1`. This report does not execute that stage.

This is a development-only offline oracle upper-bound audit under the frozen normalized torque objective. It measures subject-specific mechanical trajectory preference in heterogeneous musculoskeletal virtual subjects. It is not achieved algorithm benefit, patient preference, comfort, clinical improvement, or safety evidence.

## Q1 — Distinct V3 oracle coordination paths

There are **1** distinct exact oracle paths among 24 development subjects, with **1** beta_flex values and **1** beta_extend values. The modal oracle is `MYOLEG_V3_K0600` and is shared by **24/24 (100.0%)**.

Boundary-oracle fraction is **100.0%**. All 24 share one boundary oracle: **True**. The beta domain was not expanded.

## Q2 — Mechanical size versus grid/equivalence effects

Of 276 subject pairs, **276** share the exact oracle, **0** are immediate grid neighbors, and **0** are separated. The median / P95 / maximum normalized 2-D oracle distance is **0 / 0 / 0**.

## Q3 — Subject-by-candidate interaction

V3 candidate-main variance is **99.863908%** and subject-by-candidate interaction is **0.135074%**, versus frozen V2 **0.033114%**. The V3/V2 interaction fold change is **4.079x** and interaction/common-effect RMS ratio is **0.0367774**. Under the preregistered >=0.25% material-increase rule, material increase is **False**.

## Q4 — Candidate-ranking similarity

Pairwise Spearman median / range is **0.999953 / [0.999553, 0.999999]**, versus V2 median 0.999834. Kendall median is **0.995256**, versus V2 0.989722. Top-5% Jaccard median is **1.000000**.

## Q5 — Common-trajectory regret

The development mean-optimal common candidate is `MYOLEG_V3_K0600` at beta **[0.03, -0.03]**. Relative common regret median / mean / P75 / P95 / max is **0.000000% / 0.000000% / 0.000000% / 0.000000% / 0.000000%**. This is an oracle upper bound on potential mechanical personalization, not an achieved algorithm benefit.

## Q6 — Universal near-oracle solution

At epsilon=0.001 the maximum shared coverage is **24/24** and a 24/24 universal near-oracle candidate exists: **True**. The median near-oracle plateau contains **15/625** candidates; the preregistered broad-plateau rule is **False**.

## Q7 — Oracle transferability

Off-diagonal relative oracle-transfer regret median / P95 / max is **0.000000% / 0.000000% / 0.000000%**. The preregistered meaningful median-transfer-loss condition is **False**.

## Q8 — Frozen subject-factor associations

The 30 preregistered exploratory Spearman tests produced **1** BH-q<0.05 associations. No predictive learner was trained. Any associations remain exploratory structural diagnostics, not causal or physiological parameter identification.

## Q9 — Fixed-ROM coordination-path personalization

Local sign counts (descent/equivalent/increase) are `{"beta_extend_negative": {"descent": 24, "equivalent": 0, "increase": 0, "majority_fraction": 1.0}, "beta_extend_positive": {"descent": 0, "equivalent": 0, "increase": 24, "majority_fraction": 1.0}, "beta_flex_negative": {"descent": 0, "equivalent": 0, "increase": 24, "majority_fraction": 1.0}, "beta_flex_positive": {"descent": 24, "equivalent": 0, "increase": 0, "majority_fraction": 1.0}}`. Across 1,200 global adjacent transitions, **0 (0.000%)** show non-equivalent cross-subject direction disagreement under the frozen 1e-12 tolerance.

The formal decision is `V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED` because the preregistered conjunction, not any single dramatic subject or metric, determines whether the fixed-ROM V3 task constitutes an algorithmically meaningful mechanical personalization problem.

## Q10 — Next branch

Proceed, if authorized, to `MYOLEG_OBJECTIVE_AND_MUSCULOSKELETAL_HETEROGENEITY_DECISION_AUDIT_V1`. Do not execute it automatically.

## Integrity boundary

- Protocol SHA-256: `d21506672ed006e5015bb92ad8ec50dce15ea2762b1b421083664ae6321b3eb3`; frozen before V3 development outcome reveal.
- V3-only numeric outputs were frozen before V2 result artifacts were opened by this execution.
- V3 truth landscape, candidate manifest/table, cohort, objective, normalization, and V2 frozen artifacts were read-only.
- Exactly 24 development subjects were read. `HELD_OUT_SCIENTIFIC_ACCESS_COUNT = 0`.
- No held-out replay/J/oracle/ranking/torque/beta statistic was accessed.
- No Five-parameter model, NN/PINN, BO, robot, hardware, human, or clinical stage was run.
