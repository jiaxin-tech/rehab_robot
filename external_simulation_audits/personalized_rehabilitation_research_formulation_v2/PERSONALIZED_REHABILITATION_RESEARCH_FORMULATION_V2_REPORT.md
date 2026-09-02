# Personalized Rehabilitation Research Formulation V2

## Formal decision

`PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_READY_WITH_LIMITATIONS`

Protocol SHA-256: `41da6efac092da267a0e8477ff8453fe4790b5adc0e2bfd23d1ab7671cc58a45`

The project primary line is now mechanical measurement-driven personalization. The candidate family remains the frozen V3 P4 branch-aware two-parameter family. Subject specificity comes from the current subject's measured executed trials, not a MyoLeg parameter vector. Preference/comfort is retained only as a future direct-feedback extension.

## Q1. New primary research question

Under the same low adaptation-trial budget, can subject-specific measured interaction trials adapt a reduced-order model and select a final fixed-task V3 trajectory with lower independently evaluated mechanical interaction than fixed, common, random/space-filling and non-adaptive baselines?

## Q2. Primary thesis

`MECHANICAL_MEASUREMENT_DRIVEN_PERSONALIZATION`. Comfort/preference is optional future work because no direct feedback channel, human protocol, robot readiness or ethics approval currently exists.

## Q3. Subject-specific information source

`SUBJECT_SPECIFIC_MEASURED_INTERACTION_TRIALS`: time-qualified robot state and independently validated interaction mechanics, with tactile features only after sensor validation. Direct ratings/pairwise labels are required for a future preference branch.

## Q4. What updates after each trial?

A versioned episode-feature record, effective gray-box parameter/posterior state, prediction uncertainty, optional gated residual state, observed-candidate ledger and selector/surrogate state. Only trials already executed by the same subject may be used.

## Q5. MyoLeg role

`OFFLINE_PHYSICS_PRIOR_STRESS_TEST_AND_FEASIBILITY_SUPPORT_ONLY`. It supports offline prior checks, feasibility and mismatch stress tests; it is not patient/preference truth.

## Q6. PINN task and gate

Potential task: predict the subject-specific residual between measured mechanical response and gray-box physics. It is justified only after repeated measured trials, evaluated gray-box baseline, systematic repeatable residual, sufficient data and an equal-budget benchmark. Current status: `PINN_NOT_JUSTIFIED`.

## Q7. BO objective and gate

Mechanical BO would optimize a calibrated measured mechanical endpoint; preference BO would optimize latent utility from direct feedback. BO is a selector, not the source of personalization. It remains unjustified until domain, endpoint/feedback, complete-trial semantics and independent safety constraints are frozen.

## Q8. Tactile role

Tactile pressure is a planned measured interaction feature and possible comfort correlate. It needs calibration, rate/latency/synchronization, repeatability and spatial mapping validation. `pressure != comfort`.

## Q9. Baselines and validation

Reference, common trajectory, random/space-filling, model-free BO, gray-box plus BO, and residual/PINN plus BO only if justified; every adaptive method receives K=4 complete adaptation trials in the primary hypothesis, with K=3/5 sensitivity analyses. Adaptation and final evaluation are separate, and no method receives extra truth.

## Q10. Single next stage

`MEASUREMENT_DRIVEN_PERSONALIZATION_DATA_AND_ENDPOINT_DESIGN_V1`

It should freeze measurement channels, synchronization, episode features, primary mechanical endpoint and repeated-trial design. It was not executed.

## Why READY_WITH_LIMITATIONS

- The primary scientific question, data source, V3 family, episode semantics, model/selector hierarchy and validation logic are coherent and frozen.
- No robot channel is formally research-ready: safety and identification manifests remain unreviewed and base-wrench rotation is unverified.
- Tactile/direct feedback are not implemented.
- The exact mechanical endpoint and K budget remain next-stage hypotheses.
- No PINN, BO, robot or human study was run.

`ALGORITHM_FORMULATION_READY != ROBOT_EXECUTION_READY`; status remains `NOT_HUMAN_READY / NOT_ROBOT_APPROVED`.
