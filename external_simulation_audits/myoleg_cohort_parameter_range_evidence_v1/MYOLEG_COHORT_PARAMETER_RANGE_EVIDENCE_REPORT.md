# MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_V1

## Final outcome

`MYOLEG_COHORT_RANGES_READY_WITH_SYNTHETIC_LIMITATIONS`

Scheme A's structure and every proposed endpoint/corner pass the frozen offline
MyoLeg integrity gates, but all three `fpmax` intervals remain synthetic
model-property ranges.  The cohort may only be described as **heterogeneous
musculoskeletal virtual subjects**, never as a representative patient cohort.

## Q1 - Conservative range for every Scheme A factor

- `FEMUR_MASS_INERTIA_SCALE`: **0.88-1.12** (E2 mass-derived; inertia coupling approximate).
- `TIBIA_PATELLA_MASS_INERTIA_SCALE`: **0.87-1.13** (E2 shank-mass-derived; patella/inertia approximate).
- `FOOT_COMPLEX_MASS_INERTIA_SCALE`: **0.82-1.18** (E2 foot-mass-derived; low confidence for a small segmented body region).
- `HIP_ONLY_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).
- `KNEE_ONLY_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).
- `HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).

Extended stress-only envelopes are 0.76-1.24, 0.74-1.26 and 0.64-1.36 for
the three mass factors, and 0.90-1.10 for every passive group.  These are not
primary population claims.

## Q2 - Evidence interpretation

The fractional mass intervals are model-derived (E2) from reported adult
segment mass mean/SD.  DXA evidence directly supports that segment inertial
properties vary across and within populations but did not supply the numeric
bounds used here.  Proportional inertia scaling, fixed COM and grouped body
mapping are approximations.  The `fpmax` intervals are E4 synthetic ranges,
not direct or model-converted physiological distributions.

## Q3 - Coupled mass/inertia scaling

It is defensible only as a practical fixed-geometry V1 approximation:
`INERTIA_SCALING_IS_MODELING_APPROXIMATION`.  Literature reports substantially
different fractional variability for mass and the three inertia axes; one
scalar cannot represent all of them or replace subject-specific geometry.

## Q4 - Population-level fpmax

No.  MuJoCo `fpmax` is passive normalized force at `lmax` relative to `F0`.
Passive-joint and force-length studies show real heterogeneity, while
Rajagopal/OpenSim calibration changes different curve parameters.  No source
provides a defensible population-to-MuJoCo-`fpmax` conversion.

## Q5 - Different passive-group ranges

No distinct relative ranges are justified.  The frozen structural groups are
retained, including `bflh_r, grac_r, recfem_r, sart_r, semimem_r, semiten_r, tfl_r`, but all three groups
use the common conservative 0.95-1.05 interval.  Structural grouping creates
different torque effects without inventing different marginal widths.

## Q6 - Correlations

Thigh/shank/foot scales are `POSSIBLY_CORRELATED`; passive-group factors are
also `POSSIBLY_CORRELATED`; mass-to-`fpmax` pairs have
`NO_USEFUL_EVIDENCE`.  A global fpmax factor is `KNOWN_CORRELATED` by algebraic
overlap and remains excluded.  No quantitative covariance matrix is frozen.

## Q7 - Numerical integrity

All `30` endpoint rows (conservative and extended, including
nominal rows) and all `5` requested interaction rows pass.
The maximum peak-force ratio versus nominal is `1.100000` and the
maximum concentration-share ratio is `1.034007`.  There are no
solver warnings, nonfinite states, new contact/limit modes, equality failures or
truth-algebra failures.  This proves numerical/model integrity only.

The requested `HIGH_MASS_HIGH_PASSIVE` corner is exactly the same parameter
vector as `ALL_CONSERVATIVE_HIGH` because Scheme A contains only mass and
passive factors.  Both labels are retained and the duplicate is explicit.

## Q8 - Claim boundary

Use **structured musculoskeletal heterogeneity**.  Do not call the complete
cohort physiologically motivated or representative of patients.  Only its
mass marginals have anthropometric motivation, with modeling approximations.

## Q9 - Size and design

Freeze 32 heterogeneous profiles, split 24 development / 8 held-out, with the
nominal frozen model as a separate control.  Use the preregistered deterministic
centered maximin Latin hypercube and seed `20260830`.  This is simulation-space
coverage, not clinical power or a human joint probability model.

## Q10 - Readiness for generation

Yes, but only under the generated default-off protocol and the synthetic claim
boundary.  `MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1` was **not executed**.

## Frozen boundaries

- primary P0 semantics unchanged; no simultaneous F0/fpmax dimensions
- V2 reference / base model / truth semantics unchanged
- formal reference / ROM protocol / `theta_shank = q_hip - q_knee` unchanged
- no cohort, landscape, five-parameter fit, NN/PINN, BO or robot access
- proposal SHA: `fc826049859960654d593f7ce1c2096c3df563f92915c07758a8ea84645e37de`
- evaluation-manifest SHA: `499667344eff7d9ab1c160abfac5c3db0e1928f21d8d5b9063d900f24f5b6677`
