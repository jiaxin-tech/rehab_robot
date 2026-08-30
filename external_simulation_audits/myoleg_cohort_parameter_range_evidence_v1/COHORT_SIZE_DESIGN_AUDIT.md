# Cohort size design audit

This is a simulation-design comparison, not a clinical power analysis.  Scheme
A has six factors and no defensible joint human probability distribution.

| heterogeneous subjects | proposed split | points per factor dimension (descriptive only) | measured one-reference-replay lower-bound cost |
|---:|---|---:|---:|
| 16 | 12 development / 4 held-out | 2.67 | 38.59 s |
| 24 | 16 development / 8 held-out | 4.00 | 57.89 s |
| 32 | 24 development / 8 held-out | 5.33 | 77.18 s |

The mean unique-profile P0/V2 integrity replay time on the frozen runtime was
`2.411891 s`.  The cost column excludes future learner fitting,
candidate landscapes and repeated method evaluations and is therefore only a
hardware-specific lower bound.

- 16 is too sparse once four are held out.
- 24 preserves eight held-out models but leaves only 16 development profiles.
- 32 permits 24 development and eight held-out profiles and is a natural size
  for deterministic six-dimensional space filling without claiming clinical
  representativeness.

Recommendation: **32 heterogeneous virtual subjects (24 development / 8
held-out), plus the existing nominal base model as a separate control**.
