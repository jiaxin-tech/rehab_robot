# P2_REVISION_ROOT_CAUSE_AUDIT_V1

Final recommendation: `P2_POLICY_REVISION_JUSTIFIED`

Implementation readiness remains `REVISION_DESIGN_NOT_FROZEN`: the audit justifies changing the method, but does not supply a reviewed G1/G2 numeric bound or a stopping threshold.

## Plain-language answers

### A. Why baseline / hip_stiff / heavy_leg all reached knee -5

All four matched truth landscapes place the knee component of the global optimum at the generator's -5 deg boundary, so the common knee march is a real `OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM` tendency, not merely an optimizer selection failure.  However, their complete truth-global alpha values contain 3 distinct hip/phase combinations while frozen P2 ends at `(0,-5,0)` for the three cited subjects.  The missing hip/phase differences are therefore `POLICY_COLLAPSES_SUBJECT_DIFFERENCES`.

### B/C. Objective discrimination and its source

The unchanged objective retains some subject discrimination: the four matched subjects have 3 complete global optima, and the registered-value synthetic scan has 5.  It nevertheless rewards knee-amplitude reduction to the lower generator boundary in 8/8 scanned combinations.  Per-subject normalization intentionally removes absolute scale (heavy_leg reference hip/knee scales versus baseline are 1.157308/0.901268); it compresses mass/stiffness differences but does not erase all optimum differences.  The shared boundary is best explained by the combination of the mechanical torque-ratio objective and the available generator direction; the current four virtual subjects also do not span every optimum seen in the registered-value product scan.  The five-parameter model is not the primary matched-case cause.

The evidence does **not** establish that the objective is incapable of subject-specific personalization, so this audit does not issue `OBJECTIVE_REQUIRES_SCIENTIFIC_REVIEW` and does not change the objective.  Its common boundary behavior still requires scientific interpretation before a future policy is frozen.

### D/E. Why four mismatch cases stopped prematurely

In all four cases the relevant candidate was fully supported and the model predicted the correct improving direction, but the current validation-pair maximum made the margin negative.  That bound comes from an identification-excitation comparison without personalization alpha coordinates, so it is not calibrated to the formal one-step local decision scale: `GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH`.

### F. Local-guard counterfactual

There are zero designated validation pairs on the formal local-alpha scale.  G1 local-max and G2 local-P95 are therefore unavailable rather than zero; G1 rows estimable = 0.  This audit cannot honestly claim fewer missed improvements or unchanged false improvements for a local guard: `LOCAL_CALIBRATION_NOT_SUFFICIENT`.  The only allowed next diagnostic is a predeclared, designated local-pair validation design; it must not use adaptation truth or held-out final test.

### G/H/I. Exploration value

Of 32 EXPLORE trials, 29 increased support/information without exact parameter, map, validation-error, best-J, or exploit-eligibility change; 3 opened exploit eligibility.  knee_stiff continued eight times because each valid unsupported adjacent frontier point remained rankable by information gain while P2 had no diminishing-decision-value stop.  baseline/hip_stiff/heavy_leg Trial 7--13 added support but opened no exploit and changed neither theta nor the map: `POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION`.

### J. What a future P2 revision may change

1. Replace the structurally mismatched guard evidence only after a separately reviewed **local-decision-matched designated validation** protocol exists.  This task does not choose max/P95/P99.
2. Study a **decision-value-aware exploration continuation/stopping** rule using observable parameter/map/validation/eligibility/information/support traces.  This task does not freeze a numeric threshold.

Do not change the mechanical objective, five-parameter model, generator bounds, reference, ROM, 0.005 equivalence tolerance, or 90% support gate in this audit.

## Frozen status

- `OFFLINE_METHOD_REQUIRES_REVISION`
- `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`
- `GLOBAL_MODEL_RELIABILITY_RULE_NOT_FROZEN`
- `NOT_HUMAN_READY`
- `NOT_ROBOT_MOTION_APPROVED`
- `counterfactual_trajectory_executed = false`
- `real_robot_connected = false`
