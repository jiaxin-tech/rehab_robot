# MyoLeg Personalization Formulation Stop-or-Pivot Audit V1

## Primary project decision

`PIVOT_TO_MEASUREMENT_DRIVEN_PERSONALIZATION`

Protocol SHA-256: `029ffe5bcc91ca4ffc8d9db3216b7df0c92c73f0aa253a26a2d2d9b24f173503`

The current evidence does **not** justify continuing to treat MyoLeg-generated parameter heterogeneity plus the frozen mechanical torque objective as the primary source of subject-specific trajectory preference. This does not prove that real patients share one best trajectory. It shows that the current synthetic formulation has not produced the required subject-specific ordering despite independent simulator mapping, fixed-ROM V3 reformulation, objective diagnostics, and preregistered structural-factor testing.

## Route decisions

- Option A, synthetic MyoLeg personalization: `STOP` as a primary personalization-truth program.
- Option B, universal mechanical optimization: `RETAIN_AS_LIMITED_NONPERSONALIZED_SECONDARY_BRANCH`.
- Option C, measurement-driven personalization: `PRIMARY_RECOMMENDATION`.

The pivot retains V3, physics models and MyoLeg as priors/engineering tools. The individualized signal must come from subject-specific executed-trial measurements or direct preference feedback—not from repeated synthetic factor redesign.

## Why this decision follows the frozen evidence

- V2 and V3 both failed formal personalization-necessity audits.
- V3 removed task-amplitude confounding but still produced one oracle for 24/24 subjects, zero common/transfer regret, 0.135074% interaction and nearly identical ranking.
- Raw, normalized, time-local, peak and component diagnostics did not reveal a strong ordering signal hidden by RMS; the objective was judged information-retaining and heterogeneity limiting.
- All four amended/preregistered structural factors were integrity-valid but `MAGNITUDE_ONLY`; no fallback was used.
- Population ranges for those S1 factors remain unavailable.

## Mechanical versus comfort personalization

Mechanical-load personalization can target prespecified force, torque or pressure metrics measured during executed trials. Comfort/preference personalization requires direct subject feedback. Torque and pressure may be features or constraints, but neither is comfort truth.

## Future research questions

**Mechanical-only:** Can physics-informed adaptation plus low-budget exploration use subject-specific interaction measurements to select a mechanically preferable V3 coordination path over equal-budget non-personalized and non-adaptive baselines?

**Preference/comfort:** Can direct ratings or pairwise choices, combined with mechanical constraints, support low-budget selection of a subject-reported preferable V3 coordination path?

These are separate studies with different data requirements and claim boundaries.

## PINN and BO

- PINN: `PINN_NOT_YET_JUSTIFIED`. It becomes meaningful only after measured executed-trial residual data exist and simpler models provide a baseline.
- BO: `PERSONALIZED_BO_NOT_YET_JUSTIFIED_WITHOUT_SUBJECT_FEEDBACK`. BO selects against a supplied outcome; it cannot invent a personalized objective. Mechanical BO requires measured mechanical outcomes, while preference BO requires direct feedback.

## Existing work retained

The 2-DOF model, measured reference, fixed-ROM V3 family, MyoLeg mapping, truth semantics and negative audits remain useful. V3 and the reference can anchor a future formulation; MyoLeg remains useful for offline feasibility, physics-prior checks and stress testing. Past negative results remain evidence explaining the pivot and are not deleted.

## Explicit stops

- No larger S1 z or outcome-triggered S2/S3.
- No Cohort V2 with uncalibrated synthetic bounds.
- No V4/V5 or objective-weight search solely to induce different oracles.
- No unsupported PINN/BO personalization claim.

## Exact next independent stage

`PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2`

That stage should freeze the research question, primary outcome, data role, physics/PINN role, BO role, tactile role, trial budget and validation hierarchy. It was **not** executed here.

## Integrity and status

- Frozen evidence files verified: 13.
- Scientific arrays/replays/new optimizer runs: 0.
- Held-out scientific access: 0.
- New cohort/virtual subjects: 0.
- Objective, normalization, V3 domain, S1 and historical artifacts: unchanged.
- Human ready: no.
- Robot approved: no.
