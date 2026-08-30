# MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1

## Final outcome

`MYOLEG_VIRTUAL_PATIENT_COHORT_V1_VALID_WITH_LIMITATIONS`

Exactly 32 preregistered **heterogeneous musculoskeletal virtual subjects**
were generated from the frozen six-dimensional Scheme-A primary ranges.  One
separate nominal MyoLeg control is retained and is not counted in the 32.

## Q1 - Exactly 32 frozen subjects?

Yes.  The unit-cube and transformed matrices were frozen before any subject
replay.  The centered maximin LHS used NumPy `2.2.6` PCG64 seed
`20260830`, 512 permutation restarts, selected restart
`94`, and minimum normalized
pairwise distance `0.427337323`.  No
duplicate or out-of-range row exists and no extended stress range was used.

## Q2 - Frozen split

- Development (24): `MYOLEG_VP_001, MYOLEG_VP_002, MYOLEG_VP_003, MYOLEG_VP_005, MYOLEG_VP_006, MYOLEG_VP_007, MYOLEG_VP_009, MYOLEG_VP_010, MYOLEG_VP_011, MYOLEG_VP_013, MYOLEG_VP_014, MYOLEG_VP_015, MYOLEG_VP_017, MYOLEG_VP_018, MYOLEG_VP_019, MYOLEG_VP_021, MYOLEG_VP_022, MYOLEG_VP_023, MYOLEG_VP_025, MYOLEG_VP_026, MYOLEG_VP_027, MYOLEG_VP_029, MYOLEG_VP_030, MYOLEG_VP_031`
- Held-out (8): `MYOLEG_VP_004, MYOLEG_VP_008, MYOLEG_VP_012, MYOLEG_VP_016, MYOLEG_VP_020, MYOLEG_VP_024, MYOLEG_VP_028, MYOLEG_VP_032`

This is the protocol-frozen zero-based index-modulo assignment.  It used no
torque, learner, landscape or difficulty outcome.

## Q3 - Integrity and replacement

All 32 subjects passed all preregistered gates.  There was no replacement
sampling.  Every replay completed 24 s / 401 samples with finite states, zero
solver warnings, retained 80 muscles and 80 tendons, exact structural arrays,
unchanged knee/patella equalities, tendon paths, sites, joint axes/ranges,
actuator length ranges, RTB3 and coordinate mapping.  Only the listed Scheme-A
mass/inertia and group `fpmax` fields changed.

## Q4 - Reference-response heterogeneity

| response | min | median | max | CV |
|---|---:|---:|---:|---:|
| hip tau RMS (N m) | 33.006220 | 36.169515 | 39.242511 | 4.305% |
| knee tau RMS (N m) | 9.663574 | 11.055870 | 12.605279 | 6.695% |
| hip peak (N m) | 47.568370 | 52.625543 | 57.430281 | 4.620% |
| knee peak (N m) | 15.684911 | 17.305771 | 18.969095 | 4.833% |

This is descriptive model-response heterogeneity only; no subject was selected,
removed or reweighted from these values.

## Q5 - Subject-specific normalization

All 32 subjects have positive finite subject-specific hip/knee reference RMS
denominators.  The frozen objective
`sqrt(((hip/ref_hip)^2 + (knee/ref_knee)^2)/2)` gives
`J_truth(reference)=1` within `1e-12` for every subject.  No nominal denominator
was used.

## Q6 - Nominal control

The nominal control matches all `49` arrays
in the previously frozen native-V2 nominal replay exactly (`np.array_equal`).

## Q7 - Permanent identities

The literal unit-cube matrix, transformed matrix, exact 24/8 split, compact
model deltas, compiled-model fingerprints, per-subject replay arrays and file
checksums are retained.  Final cohort manifest SHA-256:
`31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057`.

## Q8 - Remaining scientific limitation

Mass marginals are anthropometric/model-motivated, but proportional inertia
scaling with fixed COM/geometry remains a modeling approximation.  All three
`fpmax` factors are conservative structured synthetic heterogeneity rather than
population-derived passive mechanics.  This cohort cannot be called a patient
sample, physiological distribution, representative population or validated
digital-twin cohort.

## Q9 - Runtime

Mean model generation was `0.045365 s`; mean prescribed replay
`0.168493 s`; mean controlled replay `2.246280 s`;
mean complete reference replay `2.414774 s` per subject.  The 32
reference replays took `77.273 s`
inside the replay routines.  Candidate-domain totals remain engineering
formulas/illustrations only; no landscape was run.

## Q10 - Candidate-domain readiness

Yes, with the above synthetic limitations and mandatory manifest identity.  The
next allowed design stage is `MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1`.  This stage
did not generate a landscape or train Five-parameter, NN, PINN or BO models.

## Frozen boundary

- formal reference, ROM_PROTOCOL_V2 and `theta_shank = q_hip - q_knee`: unchanged
- V2 119.5-degree reference, base MyoLeg and truth semantics V1: unchanged
- no candidate landscape, Five-parameter fit, NN/PINN, BO or robot/hardware
- `INERTIA_SCALING_IS_MODELING_APPROXIMATION = true`
- no outcome-based removal or replacement sampling
