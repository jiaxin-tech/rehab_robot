# MyoLeg Structural Heterogeneity Pilot Design V1

## Formal outcome

**MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY**

Blocking status: **S1_DEFINITION_INCOMPLETE**.

This is a scientifically required fail-closed result, not a failed simulation. No pilot truth was executed. The authoritative S1 artifact supplies four factor names, four field families and a declared dimensionality of four, but it does not fully specify the target members and signed/coherent update operation needed to create deterministic models.

## Why the design stopped

- Factor 1 identifies the biarticular `lmax` field and companion audit identifies seven muscles, but the authoritative S1 does not freeze how their heterogeneous nominal values are shifted or normalized coherently.
- Factor 2 identifies rectus-versus-hamstring `fpmax` members through a companion audit, but the positive/negative balance convention and normalization constraint are not frozen.
- Factors 3 and 4 name hip/knee monoarticular antagonist relative F0, but no exact muscle membership, agonist/antagonist side, or update convention exists in the frozen artifacts.
- The frozen S1 does not explicitly say which factors replace or supplement V1 factors.

Guessing these definitions from anatomy would create a new S1 after seeing prior evidence, contrary to the authoritative-source and no-redefinition rules. Therefore no diagnostic level, trajectory subset, numeric scientific gate, fallback or replay count was made executable.

## Q1. Exact frozen S1 factors and fields

The four exact names and field families are preserved in `S1_EXACT_FACTOR_DEFINITION.csv`. Exact target/update definitions are incomplete, so S1 cannot be reconstructed deterministically.

## Q2. Which factors have defensible model semantics?

All four point to real MuJoCo fields. Factors 1 and 2 have partially resolved muscle groups. Field semantics alone are insufficient: factors 3/4 lack targets, and factors 1/2 lack complete coherent operations.

## Q3. Population-range evidence versus diagnostic perturbations

No S1 factor has an evidence-backed population range. All remain `RANGE_REQUIRES_EXTERNAL_EVIDENCE`. Synthetic diagnostic perturbations were not frozen because the exact factor operators are incomplete.

## Q4. Exact perturbation levels

**Not frozen.** Any numeric levels now would attach to ambiguous operators and violate `S1_DEFINITION_INCOMPLETE` fail-closed handling.

## Q5. Deterministic V3 subset

**Not frozen as executable.** The geometry-only selection rule can be designed later, after S1 is made deterministic; no candidate ID was selected from J, oracle or rank here.

## Q6. Scientific gates

**Numeric gates not frozen.** Required endpoint families are documented in the blocked gate artifact, but they are deliberately non-executable until exact factor reconstruction succeeds.

## Q7. Integrity and fallback

Required integrity categories are documented. No fallback level exists because neither primary level nor exact factor operation is frozen. The fail-closed action is zero execution.

## Q8. Model and replay count

`0` structural diagnostic models and `0` replays are authorized or executed in this stage.

## Q9. Cohort V2 admission rule

A future factor must have defensible semantics, integrity PASS, demonstrated configuration-dependent non-proportional response, and a range-calibration pathway. A missing direct range may yield `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`, never automatic cohort admission. `NEW_VERSION_REQUIRED = true` remains frozen.

## Q10. Is the pilot ready?

**No.** A new versioned S1-definition amendment must first freeze exact target membership, sign/direction, coherent normalization and V1 relationship for all four factors. The original S1/V1 artifacts must not be overwritten.

## Stop state

- Pilot executed: **no**.
- Structural models/replays: **0 / 0**.
- New subjects/cohort/landscape: **none**.
- Held-out scientific access: **0**.
- V1 negative evidence: preserved.
- Robot/hardware: untouched.
