# MyoLeg Musculoskeletal Heterogeneity Expansion Design Audit V1

## Formal outcome

**MYOLEG_HETEROGENEITY_EXPANSION_DESIGN_READY_WITH_EVIDENCE_GAPS**

This is an offline design/model-semantics/evidence audit. It generated no subject, cohort, truth landscape, learner, optimizer result, or robot action. The V1 negative results remain formal: `V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED` and `HETEROGENEITY_LIMITATION_DOMINANT`.

Protocol SHA-256: `527763494905ae55b1bb672b1e4f23594c5eeb1461552c5f358c3ffcb1db6bf0`. Held-out scientific access: **0**.

## What the model actually contains

The compiled supine MyoLeg model contains 80 muscle actuators, 80 spatial tendons, 384 sites, 89 geoms, and 27 equality constraints, including 14 bilateral knee/patella polynomial equalities. The built-in muscle fields provide normalized operating/curve parameters and muscle-specific force/fpmax values. They do **not** provide a defensible one-to-one label for physiological optimal fiber length or tendon slack length.

`actuator_lengthrange` remains protected. It is the feasible transmission-length range used by MuJoCo normalization, not a subject fiber-length measurement. Current spatial tendon stiffness and damping are both zero; nonzero `springlength` alone is not calibrated tendon elasticity. Hip/knee joint stiffness is zero, while default damping is 0.5 N m s/rad. Joint-limit solver behavior remains a simulator artifact, not subject physiology.

## Local numerical smoke test

The preregistered geometry-only subset used 9 V3 trajectories and a local magnitude of 1e-4 only. This is `LOCAL_NUMERICAL_SENSITIVITY_ONLY`, never a population bound. Three eligible probes ran; tendon elasticity was not substituted after being found ineligible. Non-proportionality threshold met by: **none**. These results establish implementation/derivative behavior only and do not demonstrate subject-specific oracles.

## Q1. Which actual parameters can change configuration-dependent mechanics?

Normalized muscle curve fields (`range`, `lmin/lmax`), muscle-family-relative force/fpmax fields, and spatial path/wrap geometry can do so. Joint damping is trajectory dependent but overlaps the future gray-box damping parameters and is therefore not a primary independent truth factor. Geometry is mechanistically strong but cannot be safely varied as isolated sites.

## Q2. Can operating-length/passive-curve structure be varied defensibly?

**At model-field level, yes; at population-range/physiological-label level, not yet.** A coherent pilot can vary exact normalized curve fields without touching geometry. It must retain their MuJoCo names and cannot rename them optimal fiber length or slack length. Bounds require external calibration evidence.

## Q3. Can biarticular coupling be represented without an artificial coefficient?

Yes. Use the seven real muscle actuators, their muscle-specific curve/force fields, and their spatial tendon paths. A low-dimensional anatomical-family balance changes existing model fields; no new coupling coefficient is needed.

## Q4. Which moment-arm geometry factors are safe?

No single attachment/via/wrap coordinate is safe as an independent subject factor. Coupled subject geometry is scientifically plausible but classified `REQUIRES_REBUILD/CALIBRATION`: neighboring sites, wrap objects, muscle-length calibration, body/joint geometry, and knee/patella consistency must be updated together.

## Q5. Can segment geometry/COM be introduced safely?

COM can be a secondary factor after coherent inertial identification. Femur/tibia length cannot enter the primary expanded cohort through a scalar edit: joints, attachments, paths, wraps, muscle calibration, inertia, and RTB3/strap geometry must all be rebuilt.

## Q6. Are tendon/joint passive mechanics primary factors?

Not now. Current tendon elasticity is uncalibrated and joint stiffness is zero. Joint damping is an actual field but is E4/secondary and not truth-learner independent. Joint-limit curves are excluded.

## Q7. What happens to the existing six V1 factors?

Mass/inertia remain background/secondary anthropometry. The three common fpmax group factors remain valid V1 synthetic factors but are magnitude-dominant and should not be the primary signal source in a personalization-focused V2. They are preserved, not retroactively relabeled.

## Q8. Which 4-8D schemes are defensible?

S1 is a 4-D minimal field-consistent pilot candidate; S2 is a 7-D moderate candidate with more calibration burden; S3 is stress-only. None has frozen population bounds. Scheme ranking used model validity, geometry consistency, semantics, evidence, and parsimony—never oracle diversity.

## Q9. Is a new cohort and split required?

Yes: `NEW_VERSION_REQUIRED = true`. Any adopted structural scheme must become `MYOLEG_VIRTUAL_PATIENT_COHORT_V2` with a newly preregistered development/held-out split. The existing 32-subject V1 and its 24/8 split remain immutable; its sealed eight subjects are not automatically confirmatory for a new structural space.

## Q10. Is a small preregistered pilot ready?

**Ready with evidence gaps.** The exact fields and geometry-consistency rules are sufficiently clear for a small preregistered structural-integrity/non-proportionality pilot, but population bounds and some field-to-physiology mappings remain unresolved. The pilot should use nominal plus a few pre-frozen profiles and the same small geometry-selected V3 subset. It must not claim personalization and was not executed here.

## Stop state

- Current objective and normalization: unchanged.
- V3 parameterization/domain: unchanged.
- V1 cohort/32 subjects: unchanged.
- New subjects or truth: none.
- Held-out access: 0.
- Hardware/control/safety: untouched.
- Next stage `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1`: **not executed**.
