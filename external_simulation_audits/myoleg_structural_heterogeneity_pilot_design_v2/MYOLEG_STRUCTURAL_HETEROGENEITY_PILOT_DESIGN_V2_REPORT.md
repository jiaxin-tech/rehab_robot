# MyoLeg Structural Heterogeneity Pilot Design V2

## Formal outcome

**MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_READY_WITH_EVIDENCE_GAPS**

Authoritative definition: `S1_STRUCTURAL_DEFINITION_AMENDED_V1`  
Authoritative SHA-256: `3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763`

This stage froze an executable, outcome-independent design. It performed parameter arithmetic, compilation, declared-field mutation checks and one-state finite forwards only. It did **not** run any scientific trajectory replay, reveal multi-trajectory torque response, create a virtual subject, generate a cohort/landscape, or access held-out truth.

## Q1. Were all four amended factors reconstructed exactly?

**Yes.** Exact factor IDs, authoritative members, gain/bias fields, mathematical operators, `z=0` identities, semantics and V1 relationships were read from the SHA-pinned amended artifact. All nominal/primary/fallback operator-only checks passed. The old V1 design failure remains immutable: `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY / S1_DEFINITION_INCOMPLETE`.

## Q2. What primary synthetic z-levels are frozen?

- `S1F1_BIARTICULAR_LMAX`: primary z=`-0.01/+0.01`, fallback z=`-0.005/+0.005`; `DIAGNOSTIC_LEVEL_READY`.
- `S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE`: primary z=`-0.025/+0.025`, fallback z=`-0.0125/+0.0125`; `DIAGNOSTIC_LEVEL_READY`.
- `S1F3_HIP_MONO_ANTAGONIST_F0`: primary z=`-0.025/+0.025`, fallback z=`-0.0125/+0.0125`; `DIAGNOSTIC_LEVEL_READY`.
- `S1F4_KNEE_MONO_ANTAGONIST_F0`: primary z=`-0.025/+0.025`, fallback z=`-0.0125/+0.0125`; `DIAGNOSTIC_LEVEL_READY`.

The 1% lmax level is smaller because `lmax` is an indirect normalized curve field. The force-scale/family-balance factors use a symmetric 2.5% log displacement. These are synthetic diagnostic coordinates, **not human population ranges, patient distributions, validated physiological parameters, or Cohort V2 bounds**. Population range remains `NOT_AVAILABLE` for all four.

## Q3. What one-step fallback is frozen?

Exactly one half-magnitude fallback is frozen per factor/sign. It may be used only after a primary **integrity** failure. A small or uninteresting response cannot trigger fallback. A failed fallback makes that factor `INVALID_FOR_PILOT`; no additional levels may be added.

## Q4. What V3 subset is frozen?

- REFERENCE: `MYOLEG_V3_K0312` at `[+0, +0]`.
- CORNER_NEG_NEG: `MYOLEG_V3_K0000` at `[-0.03, -0.03]`.
- CORNER_NEG_POS: `MYOLEG_V3_K0024` at `[-0.03, +0.03]`.
- CORNER_POS_NEG: `MYOLEG_V3_K0600` at `[+0.03, -0.03]`.
- CORNER_POS_POS: `MYOLEG_V3_K0624` at `[+0.03, +0.03]`.
- FLEX_NEG_AXIS: `MYOLEG_V3_K0012` at `[-0.03, +0]`.
- FLEX_POS_AXIS: `MYOLEG_V3_K0612` at `[+0.03, +0]`.
- EXTEND_NEG_AXIS: `MYOLEG_V3_K0300` at `[+0, -0.03]`.
- EXTEND_POS_AXIS: `MYOLEG_V3_K0324` at `[+0, +0.03]`.
- INTERIOR_NEG_NEG: `MYOLEG_V3_K0156` at `[-0.015, -0.015]`.
- INTERIOR_NEG_POS: `MYOLEG_V3_K0168` at `[-0.015, +0.015]`.
- INTERIOR_POS_NEG: `MYOLEG_V3_K0456` at `[+0.015, -0.015]`.
- INTERIOR_POS_POS: `MYOLEG_V3_K0468` at `[+0.015, +0.015]`.

All 13 coordinates exist exactly on the 625-point grid. Selection used only candidate ID/index, beta coordinates and frozen kinematic inclusion bits—never J, oracle, rank or personalized outcomes.

## Q5. What response representations are primary?

Hip and knee required-torque RMS are required for every factor. Factor-specific signed actuator-contribution RMS responses are also primary: affected biarticular hip/knee contributions for lmax; rectus, hamstring and their net hip/knee contributions for balance; declared hip antagonist hip contribution; and declared knee antagonist knee contribution. Every primary scalar receives proportional/affine fits, delta-configuration metrics and beta gradients. Frozen combined J is secondary only and cannot be the sole endpoint.

## Q6. What non-proportionality gate is frozen?

On the same preregistered response, delta RMS must be at least `1e-5 N*m` and `1e-4` of nominal RMS, proportional NRMSE must be strictly above `1e-4`, and affine R2 must be strictly below `0.9999`. A factor-sign needs evidence in at least one required-torque response and one factor-specific mechanistic component; floating residuals and pure scale/offset effects do not pass.

## Q7. What configuration-dependence gate is frozen?

For `delta_y`, normalized population SD must be at least `1e-4`, normalized range at least `2e-4`, and the fixed six-term beta polynomial must have R2 at least `0.25`, with the effect-resolution gate also passing. Configuration dependence cannot replace the separate non-proportionality gate.

## Q8. What gradient-rotation gate is frozen?

The four `±0.015` diagonal interior points form a centered stencil. Both gradients must clear a relative resolution of `1e-5`; direction evidence is cosine at or below `0.995`, a resolved component sign change, or maximum unit-direction component change of at least `0.05`. A magnitude-only change is not direction evidence. Gradient rotation is supporting evidence and full sign reversal is not mandatory.

## Q9. What Cohort V2 admission rule is frozen?

Admission to `MYOLEG_VIRTUAL_PATIENT_COHORT_V2_RANGE_AND_DESIGN` requires the conjunction of defensible exact semantics, pilot integrity PASS, frozen-gate non-proportional configuration dependence, and a future population-range calibration pathway. If only the first three pass, status is `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`; no cohort is generated. Old V1 held-out data are not automatic Cohort V2 confirmation and had zero scientific access here.

V1 relations remain unchanged: femur, tibia/patella and foot mass/inertia are background; hip/knee common fpmax are secondary only; common biarticular fpmax is removed from the personalization-focused cohort.

## Q10. Is the protocol ready for `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1`?

**Yes, with evidence gaps.** All four synthetic levels and operators passed outcome-independent prechecks. Population ranges remain unavailable, so this is not Cohort V2 readiness. The pilot must be invoked separately under this frozen protocol and must stop after factor classification.

## Exact future workload

- Primary structural diagnostic models: **9** (`1 nominal + 8 perturbed`).
- Trajectories/model: **13**.
- Primary replays: **117**.
- Optional fallback maximum: **8 models / 104 replays**.
- Absolute maximum including fallback: **17 models / 221 replays**.
- Scientific models/replays executed now: **0 / 0**.

## Stop state

- Design only; scientific pilot not executed.
- Cohort V1, amended S1, V3 domain, objective and normalization unchanged.
- New virtual subjects/cohort/landscape: none.
- Held-out scientific access: 0.
- Five-parameter/NN/PINN/BO: not run.
- Robot/hardware: untouched.
