# Paper outline and frozen scientific story

## Frozen story

The manuscript asks:

> Given a prescribed passive hip–knee rehabilitation trajectory, can the robot
> identify subject-specific equivalent mechanical characteristics from limited
> task-relevant interaction data and use them to generate a better task-local
> candidate trajectory while preserving rehabilitation and safety constraints?

The causal chain is fixed as:

```text
prescribed rehabilitation trajectory
  -> 2-DOF sagittal-plane lower-limb model
  -> subject-specific equivalent dynamics identification
  -> mechanical interaction evaluation
  -> reference-local constrained trajectory personalization
```

Parameter identification is an enabling method. Timing alignment,
identifiability, geometry error, and model mismatch are supporting validation.
The paper is not framed as a simulator, delay-compensation, pure-identification,
MPC, PINN, or comfort-prediction paper.

## Current evidence-limited stopping point

The repository currently supports the story through offline identification,
identifiability, timing robustness, geometry error, model mismatch, and a
fixed-candidate screening demonstration. It does **not** yet support the full
personalization claim because:

1. the active reference is `reference_measured_asymmetric_closed_slow`;
2. the stored reference-local identification and C0–C8 candidate study are
   centered on the now-legacy symmetric reference;
3. the candidate study provides feasibility and an unweighted Pareto set, not a
   finalized subject-specific selector;
4. tactile calibration/integration and a calibrated interaction residual are
   absent; and
5. no physical dummy, robot episode, real-data identification, or human result
   is present.

Until the P0 work in `experiment_todo.md` is completed, the defensible
manuscript endpoint is **identification and task-local generalization, with
personalization presented as an implemented but incomplete software
framework**.

## IEEE-style structure

- Abstract — skeleton only.
- I. Introduction — skeleton only.
- II. Related Work — skeleton only.
- III. System Modeling and Problem Formulation — complete draft.
  - A. Rehabilitation Scenario and Coordinate Definition
  - B. Two-Link Lower-Limb Kinematics
  - C. Jacobian and Mechanical Interaction Mapping
  - D. Subject-Specific Equivalent Dynamic Model
  - E. Reference-Trajectory-Centered Personalization Problem
- IV. Subject-Specific Equivalent Dynamics Identification — complete draft.
  - A. Identification Data and Excitation Design
  - B. Equivalent Parameter Estimation
  - C. Parameter Identifiability
  - D. Timing Alignment and Delay Robustness
  - E. Model-Mismatch and Generalization Analysis
- V. Interaction-Aware Personalized Trajectory Optimization — evidence-aware skeleton.
- VI. Experiments and Results — skeleton only.
- VII. Discussion — skeleton only.
- VIII. Conclusion — skeleton only.

Stage labels remain internal provenance labels and must not become manuscript
section names.

## Repository conflicts recorded during the 2026-08-11 audit

| Conflict | Current implementation/source of truth | Manuscript handling |
|---|---|---|
| Root README says active reference is measured asymmetric; older Stage-5 outputs center a symmetric reference. | `reference_version_manifest.csv` makes the asymmetric slow profile active and marks symmetric/C2 versions legacy. | Active reference is asymmetric; legacy candidate results cannot be presented as active-reference personalization. |
| Earlier workspace/IK outputs used knee ROM 5–130 deg. | `ROM_PROTOCOL_V2` now supplies hip 0–120 deg and knee 5–145 deg to every formal active gate; old outputs remain inactive legacy provenance. | Report 5–145 deg as the one formal range and label 5–130 deg historical wherever cited. |
| Older processed/retimed reference has 0.19998 m closure error and unknown source timing. | It is superseded by the periodic C2 asymmetric reference generated from a better full-joint cycle boundary. | Register older retiming as SUPERSEDED; do not use it for a closure claim. |
| README reports a 640-passed offline run. | The 2026-08-11 audit independently reproduced 640 passed and 5 skipped in 100.80 s with bytecode/cache writes disabled. | The count remains DEBUG software verification and must never be cited as experimental evidence. |
| Robot execution architecture exists, but no episode/result directory exists and robot-trajectory export metadata says `generation_status=blocked`. | Root README and `CURRENT_ARCHITECTURE.md` state physical status `NO-GO`. | Section VI-F remains TODO; no robot or physical claim. |

## Claim release gates

1. **Identification claim:** may use FORMAL offline synthetic results with an
   explicit virtual/software evidence label.
2. **Active-reference task-local claim:** requires regeneration around the
   active asymmetric reference.
3. **Personalization claim:** additionally requires a frozen subject-specific
   selection rule and reference-versus-selected-candidate comparison.
4. **Physical robot claim:** requires a reviewed real episode and physical
   tracking/timing/frame evidence.
5. **Human/clinical claim:** requires appropriate approvals and human evidence;
   nothing in the current repository reaches this gate.
