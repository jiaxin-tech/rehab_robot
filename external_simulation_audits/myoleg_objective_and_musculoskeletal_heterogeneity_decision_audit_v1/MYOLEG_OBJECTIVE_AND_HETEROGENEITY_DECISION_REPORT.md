# MyoLeg objective and musculoskeletal heterogeneity decision audit V1

## Formal outcome

**HETEROGENEITY_LIMITATION_DOMINANT**

- Objective assessment: `CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE`
- Current 6-D heterogeneity assessment: `CURRENT_HETEROGENEITY_TRAJECTORY_INTERACTION_LIMITED`
- Scientifically justified next independent stage: `MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1`
- Next stage executed: **no**

This is an offline, development-only diagnosis. The frozen objective remains `J_NORMALIZED_RMS`; raw, peak, time-local, and force-component views are diagnostic only. No objective weights were searched.

## Frozen protocol and integrity

- Protocol: `OBJECTIVE_HETEROGENEITY_DECISION_PROTOCOL_V1`
- Protocol SHA-256: `80a24b8da08b902581db05c875fb3f8532ece30737e36a98cf26778a0e467fd8` (frozen before new development diagnostics were read)
- Primary matrix: 24 development subjects × 625 frozen V3 candidates
- Replay subset: 8 geometry-selected development subjects × 17 beta-geometry-selected candidates = 136 replays
- Replay/compact RMS agreement: hip max error `2.842e-14` Nm; knee max error `1.066e-14` Nm
- Replay cache SHA-256: `4e5534f9a786ce5f9ef2831bc57f2d7c03cf519ccb9ad5874c78a63cd2a09f76`
- Held-out scientific access: **0**; held-out replay: **0**; held-out `np.load`: **0**
- V3 candidate domain, cohort, ranges, subject factors, objective, normalization, learner, and BO were not modified.

## Frozen formal baseline

The frozen V3 result is unchanged: interaction `0.135074%`, median Spearman `0.999953`, median Kendall `0.995256`, adjacent direction reversals `0`, and common regret `0`. V2 remains a historical secondary comparison (interaction about `0.033114%`); it was not used to select diagnostics.

## Raw versus normalized RMS

| Joint | Raw interaction | Normalized interaction | attenuation ratio (fraction) | raw→normalized Spearman min | adjacent sign changes | local-gradient direction changes |
|---|---:|---:|---:|---:|---:|---:|
| Hip | 0.000029% | 0.082592% | 2806.8 | 1.000000000 | 0 | 0 |
| Knee | 0.000392% | 0.159600% | 406.881 | 1.000000000 | 0 | 0 |

两个关节的 subject 内 raw→normalized 排序均逐值保留，邻接方向没有改变。 Absolute interaction RMS changes unit/scale after normalization, so the decision uses dimensionless interaction plus ordering/direction retention rather than comparing Nm directly with a ratio.

## Time-resolved and peak diagnostics

- Reference-normalized hip time/RMS interaction ratio: `2.66749`.
- Reference-normalized knee time/RMS interaction ratio: `3.51084`.
- Interpretation: 未满足预冻结的 RMS compression 条件。
- Peak representations meeting frozen ordering-evidence rule: `normalized_knee_peak`.

Time-local evidence is not a comfort, safety, tissue-load, clinical, or human-preference result. It is only a simulator required-drive diagnostic.

## Existing force-component decomposition

The decomposition reuses the frozen replay semantics: required drive = mass + bias/gravity − passive − zero-control actuator − constraint. It does not invent a new physical decomposition.

| Component | Joint | interaction | interaction/common | direction disagreement |
|---|---|---:|---:|---:|
| mass_inertia | knee | 0.226334% | 0.0672785 | 0.000000% |
| mass_inertia | hip | 0.177856% | 0.0738977 | 0.000000% |
| zero_control_actuator | knee | 0.001379% | 0.0380324 | 0.000000% |
| zero_control_actuator | hip | 0.000283% | 0.0319819 | 0.000000% |
| bias_gravity | knee | 0.000127% | 0.0580279 | 0.000000% |
| bias_gravity | hip | 0.000042% | 0.0619703 | 0.000000% |
| constraint | knee | 0.000000% | 2.46456e-14 | 0.000000% |
| passive | knee | 0.000000% | 2.2621e-14 | 0.000000% |
| passive | hip | 0.000000% | 0 | 0.000000% |

Components meeting frozen ordering-evidence rule: `none`.

## Q1–Q10 answers

### Q1. Does raw unnormalized torque exhibit materially more subject×trajectory interaction than normalized torque?

**No.** Hip raw/normalized interaction fractions are `0.000029%` / `0.082592%`; knee values are `0.000392%` / `0.159600%`. The fraction increases after normalization because subject-main scale is removed from the total-variance denominator; this is not evidence that normalization created or destroyed ordering. Dimensionless interaction changes by `0.8418×` (hip) and `1.2649×` (knee), while ordering remains identical.

### Q2. Does normalization change only magnitude scaling, or trajectory ordering?

**Only magnitude scaling changed under the frozen tests.** 两个关节的 subject 内 raw→normalized 排序均逐值保留，邻接方向没有改变。 Therefore normalization is interpreted as removing absolute subject scale unless the frozen `NORMALIZATION_ORDERING_LOSS` rule is met; current status: `False`.

### Q3. Does time-resolved torque contain subject-specific path information that RMS removes?

**Not under the preregistered conjunction.** Hip and knee time/RMS energy ratios are `2.6675` and `3.5108`, but all six windows for both joints have zero direction disagreement and no window satisfies representation-ordering evidence. 未满足预冻结的 RMS compression 条件。

### Q4. Do peak-based diagnostics reveal subject-specific ordering?

Only `normalized_knee_peak` meets the composite representation-evidence rule. For normalized knee peak, median rank correlation is `1.000000` and direction disagreement is `0.083333%`; therefore it does **not** establish robust subject-specific path preference by itself. Peak remains diagnostic-only and was not optimized.

### Q5. Which MyoLeg force components carry the strongest subject×candidate interaction?

The largest preregistered subset interaction is `mass_inertia:knee` at `0.226334%`. The complete ordered table above prevents cherry-picking a single component.

### Q6. Do passive/fpmax variations alter path preference or mainly response magnitude?

Passive component ordering evidence count is `0`; direction-evidence count is `0`. The frozen fpmax variation therefore does not change path preference in this audit; at most it rescales passive magnitude. Magnitude associations are descriptive and are not counted as preference-direction evidence.

### Q7. Do mass/inertia variations alter path preference or mainly response magnitude?

Mass/bias component direction-evidence count is `0`. These factors mainly change response magnitude, not preferred direction. Parameter associations with scale or gradient norm do not establish a different preferred path.

### Q8. Are subject-specific gradient directions different, or only magnitudes?

For frozen J, `0/24` subjects differ in local gradient sign from the pre-frozen anchor and the minimum local gradient cosine is `0.999873`. Gradient magnitudes vary, but preferred directions do not meaningfully differ.

### Q9. Is the stronger limitation the objective, musculoskeletal heterogeneity, or both?

The stronger limitation is the **current musculoskeletal heterogeneity**: `HETEROGENEITY_LIMITATION_DOMINANT` under the pre-frozen 2×2 decision matrix. Objective evidence: `CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE`. Heterogeneity evidence: `CURRENT_HETEROGENEITY_TRAJECTORY_INTERACTION_LIMITED`.

### Q10. What exact next independent stage is justified?

`MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1`. It is recommended only and was **not executed**. If the branch concerns heterogeneity, future factor classes are taxonomy entries without ranges or implementation; if it concerns objective information, future information classes are not a new objective.

## Scientific boundaries

- This audit does not establish physiological parameter values, patient comfort, rehabilitation effectiveness, safety, clinical validity, or robot validity.
- Passive simulator terms are not direct tissue-force measurements.
- Current six factors and ranges are unchanged; no future range is proposed.
- Held-out truth remains sealed and cannot support any statement above.
- Runtime for the audit build: `1.411` s (full pytest is reported separately after generation).
