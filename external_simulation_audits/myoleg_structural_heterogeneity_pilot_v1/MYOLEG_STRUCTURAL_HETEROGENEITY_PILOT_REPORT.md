# MyoLeg Structural Heterogeneity Pilot V1

## Formal decision

**STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED**

Execution protocol SHA-256: `a92b4c95b0f76337e3405a2e36fc0a2571eabffd47bf1c925199e8231f2d9b20`  
Authoritative amended S1 SHA-256: `3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763`

This was a mechanistic offline structural diagnostic, not personalization. It executed the frozen 13-trajectory geometry subset for one nominal and eight one-factor-at-a-time models. It computed no oracle, rank, regret, personalization necessity or 625-point search.

## Q1. Did all nine primary diagnostic models pass integrity?

**True.** Primary integrity PASS: **9/9**. Every model was compiled from the same nominal model, only authoritative members/fields changed, gain/bias remained synchronized, topology was unchanged, and every replay was checked against the frozen numerical and V3 trajectory gates.

## Q2. Was fallback used?

Fallback models used: **0**. It was permitted only for preregistered integrity failures. No small effect, failed scientific gate or unfavorable result could trigger it. Actual total replay count: **117**.

## Q3-Q7. Factor findings

- `S1F1_BIARTICULAR_LMAX`: `MAGNITUDE_ONLY`; non-proportional any=False, configuration any=True, gradient rotation any=False; admission `NOT_ELIGIBLE_MAGNITUDE_ONLY`.
- `S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE`: `MAGNITUDE_ONLY`; non-proportional any=False, configuration any=True, gradient rotation any=False; admission `NOT_ELIGIBLE_MAGNITUDE_ONLY`.
- `S1F3_HIP_MONO_ANTAGONIST_F0`: `MAGNITUDE_ONLY`; non-proportional any=False, configuration any=False, gradient rotation any=False; admission `NOT_ELIGIBLE_MAGNITUDE_ONLY`.
- `S1F4_KNEE_MONO_ANTAGONIST_F0`: `MAGNITUDE_ONLY`; non-proportional any=False, configuration any=True, gradient rotation any=False; admission `NOT_ELIGIBLE_MAGNITUDE_ONLY`.

Positive and negative signs are reported separately in all result tables. Opposite-delta cosine and inversion residual are descriptive only because no sign-symmetry threshold was preregistered; neither was used to change factor semantics.

## Q8. Is a defensible population-range calibration pathway already available?

**No factor has an admission-ready population calibration pathway yet.** Candidate measurement/calibration programs are specified, but the normalized `lmax`, family `fpmax`, and active-plus-passive F0 fields do not have validated human mappings or population bounds in the frozen evidence. `STRUCTURAL_MECHANISM_SUPPORTED` therefore does not mean `POPULATION_VARIABILITY_CALIBRATED`.

## Q9. Is Cohort V2 generation justified now?

**No.** Structurally informative factors, if any, are only `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`. No diagnostic z value was promoted to a population bound and no Cohort V2 was generated.

## Q10. Exact independent next stage

`MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_AUDIT_V1`

No factor met the frozen structural-informativeness logic; increasing z is prohibited and a formulation stop/pivot audit is the independent next step. The next stage was not executed automatically.

## Frozen-gate interpretation

- Non-proportionality required absolute delta RMS >= `1e-5 N*m`, relative delta RMS >= `1e-4`, proportional NRMSE > `1e-4`, and affine R2 < `0.9999` on the same response.
- Configuration dependence additionally required normalized SD >= `1e-4`, normalized range >= `2e-4`, and fixed beta-polynomial R2 >= `0.25`.
- A factor-sign required at least one required-torque response and one preregistered factor component passing both gates.
- Gradient rotation was supporting evidence only; magnitude-only gradient changes never counted as rotation.

## Stop state

- Primary models/replays planned: `9 / 117`; actual replays: `117`.
- New virtual subjects or Cohort V2: `0`.
- Held-out scientific access: `0`.
- Objective, normalization, amended S1, V3 parameterization/domain and V1 cohort: unchanged.
- Oracle/Five-parameter/NN/PINN/BO: not run.
- Robot/hardware: untouched.
