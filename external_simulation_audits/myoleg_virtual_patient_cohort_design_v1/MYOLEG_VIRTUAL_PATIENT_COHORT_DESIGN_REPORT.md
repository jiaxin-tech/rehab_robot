# MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1

## Outcome

`MYOLEG_COHORT_DESIGN_READY_WITH_EVIDENCE_GAPS`

This is an offline design audit. It generated no virtual subjects, no candidate landscape, no learner fit, no NN/PINN, no BO run and no robot interaction. The allowed description is **heterogeneous musculoskeletal virtual subjects** or **independent structurally mismatched virtual-patient models**—not real patients, validated digital twins or a clinically representative population.

## Frozen V2 simulation domain

- `MYOLEG_V2_REFERENCE = NATIVE_ROM_REFERENCE_CANDIDATE`
- SHA-256: `208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678`
- duration / samples: `24.0 s / 401`
- knee maximum: `119.5 deg`
- transformation: `q_new(t)=q0+s*(q_original(t)-q0)` with scale `0.9503287154119732`; globally smooth, invertible, no pointwise clipping
- model: native supine MyoLeg SHA `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d` with native knee `[0,120] deg`
- limited-125 condition: historical modeling-limit evidence only, not the V2 cohort domain

## Frozen truth semantics

Every future subject must use `MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1` / `TAU_MY0LEG_REQUIRED_DRIVE`:

`r = M*qacc + qfrc_bias - qfrc_passive - qfrc_constraint - qfrc_actuator(P0)`

`tau_truth = T(q)^T*r`

`SUBJECT_SPECIFIC_REFERENCE_NORMALIZATION = true`: each subject gets its own reference hip/knee truth RMS denominator. A nominal-subject denominator is prohibited.

## Actual model inventory and taxonomy

The loaded model has `16` bodies, `29` joints, `80` tendons, `80` muscle actuators, `384` sites, `89` geoms and `27` equality constraints. The CSV inventory records each actual body, muscle field, tendon path/wrap, joint field and equality definition. Taxonomy family counts are A=5, B=2, C=2, D=6, E=5.

## Q1 — Suitable subject-level fields

Numerically and semantically promising Class A families are coupled segment `body_mass + body_inertia`, grouped muscle XML `force`/compiled gain+bias index 2, and muscle `fpmax`/compiled gain+bias index 7 including structurally verified biarticular groups. Every factor maps to actual frozen model objects. In primary P0, same-group force and fpmax factors are mutually exclusive because their effects are exactly confounded. All still require external scientific range evidence.

## Q2 — Fields not to perturb

Do not independently perturb body/joint geometry, segment length, sites, wrap geoms, tendon path/spring length, actuator lengthrange, native joint range/axis, or knee/patella equality polynomials. Joint damping/stiffness/friction, armature, tendon elasticity and the remaining FLV-shape fields are stress-only rather than primary cohort variables.

## Q3 — Anthropometry without geometry breakage

Scale each selected segment's mass and principal inertia by the same factor and hold COM fixed. Do not change segment length in V1. Pelvis is shared and needs an explicitly bilateral/global protocol rather than silent unilateral scaling.

## Q4 — Force-capacity variability

The actual `force` field is peak active force `F0` in N. Group factors may scale the matching gain and bias slots for explicit structurally identified actuator lists. Because `F0` multiplies both active and passive FLV terms, this is not an active-only manipulation. Under zero activation it is exactly confounded with same-group fpmax scaling, so force capacity belongs in a separate active-condition design or replaces—never accompanies—that fpmax factor.

## Q5 — Passive-property variability

`fpmax` is the primary V1 candidate: passive force at `lmax` relative to `F0`. `actuator_lengthrange` is a tendon-transmission range in metres and is not normalized physiological fiber length. Operating `range` and other FLV-shape parameters stay stress-only pending evidence.

## Q6 — Natural biarticular coupling

Yes. `bflh_r, grac_r, recfem_r, sart_r, semimem_r, semiten_r, tfl_r` span both independent hip and knee coordinates according to the compiled moment matrix over all 401 states. Scaling their actual force or fpmax fields changes native tendon-transmitted coupling without an added torque equation.

## Q7 — Low activation

Treat low activation as episode-level nuisance by default, not subject identity. A fixed subject baseline needs separate physiological justification. No range is frozen here.

## Q8 — Independent 4–8 dimensional design

Yes. Scheme A has 6 factors and Scheme B has 8, using coupled segment inertia plus structurally grouped passive `fpmax` factors rather than true learner `K_hip`, `K_knee`, `B_hip` or `B_knee`. The P0 schemes deliberately exclude simultaneous same-group force/fpmax factors. `TRUTH_LEARNER_PARAMETERIZATION_INDEPENDENCE = PASS`.

## Q9 — Missing evidence

The missing evidence is the magnitude and covariance of human/validated-model variation for segment mass/inertia, muscle/group force capacity, passive fpmax and any low-activation condition; muscle-group membership also needs manual structural/anatomical review. The ±5% values below are Level-3 numerical smoke amplitudes, not physiological ranges.

## Minimal one-family sensitivity checks

- `SEGMENT_MASS_INERTIA_COUPLED_SCALE`: hip RMS -3.779% / +3.783%; knee RMS -5.025% / +5.084% (−/+ smoke).
- `MUSCLE_FORCE_CAPACITY_SCALE`: hip RMS -1.217% / +1.222%; knee RMS +0.095% / -0.036% (−/+ smoke).
- `MUSCLE_PASSIVE_FP_MAX_SCALE`: hip RMS -1.217% / +1.222%; knee RMS +0.095% / -0.036% (−/+ smoke).
- `BIARTICULAR_FORCE_CAPACITY_SCALE`: hip RMS -0.076% / +0.076%; knee RMS -0.229% / +0.230% (−/+ smoke).
- `BIARTICULAR_PASSIVE_FP_MAX_SCALE`: hip RMS -0.076% / +0.076%; knee RMS -0.229% / +0.230% (−/+ smoke).

All 15 retained rows loaded, remained finite, completed the 24-s/401-sample reference, produced no solver warnings, retained equality and native-ROM integrity, passed exact repeated fingerprints, and stayed below the predeclared smoke-only 2× peak-force screen. Nominal truth and controlled replay arrays exactly match the frozen prior V2 artifact: `True`. Global force versus global fpmax and biarticular force versus biarticular fpmax agreed within `1e-12 N*m` in both perturbation directions: `True`.

## Q10 — What should be evaluated next?

After external range evidence and manual group review, evaluate `SCHEME_A_MINIMAL_INTERPRETABLE` first with a preregistered deterministic design. A 24-subject option (16 development / 8 held-out subject models) is a candidate, not frozen. Freeze every subject manifest and split before truth reveal. Because evidence gaps remain, do **not** execute `MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1` yet.

## Final boundary

- cohort generated: no
- landscape generated: no
- five-parameter fit: no
- NN/PINN trained: no
- BO run: no
- robot/hardware accessed: no
- formal reference / ROM changed: no
- V2 reference / truth semantics changed: no
