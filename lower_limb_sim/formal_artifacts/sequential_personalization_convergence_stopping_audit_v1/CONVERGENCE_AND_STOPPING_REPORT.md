# SEQUENTIAL_PERSONALIZATION_CONVERGENCE_AND_STOPPING_AUDIT_V1

## Plain-language findings

- P2 naturally stopped in 9/9 cases; 0 reached the 20-trial diagnostic cap.
- The previous six-trial best already equalled the extended final best in 9/9 cases, although six trials did not always expose the natural stopping decision.
- Boundary-optimum diagnostics: baseline__matched_linear, baseline__nonlinear_damping_mild, heavy_leg__matched_linear, hip_stiff__matched_linear.
- Mean first versus last accepted-EXPLOIT marginal improvement: 0.00610981 versus 0.00576189; this is characterization, not a new stopping threshold.
- Missed-opportunity rounds: 7 of 27 rounds with a true >0.005 local improvement.
- Exploration trials with observed decision value within two rounds: 3; informative but low-decision-value trials: 29.
- Executed false improvements across extended P0/P1/P2: 0.
- Freeze-readiness result: `OFFLINE_METHOD_REQUIRES_REVISION`.

## Correct-stop audit

- baseline__matched_linear: `CORRECT_CONSERVATIVE_STOP`; true local improvement at stop = false.
- hip_stiff__matched_linear: `CORRECT_CONSERVATIVE_STOP`; true local improvement at stop = false.
- knee_stiff__matched_linear: `CORRECT_CONSERVATIVE_STOP`; true local improvement at stop = false.
- heavy_leg__matched_linear: `CORRECT_CONSERVATIVE_STOP`; true local improvement at stop = false.
- baseline__nonlinear_stiffness_mild: `PREMATURE_CONSERVATIVE_STOP`; true local improvement at stop = true.
- baseline__hip_knee_coupling_mild: `PREMATURE_CONSERVATIVE_STOP`; true local improvement at stop = true.
- baseline__nonlinear_damping_mild: `CORRECT_CONSERVATIVE_STOP`; true local improvement at stop = false.
- baseline__structured_residual: `PREMATURE_CONSERVATIVE_STOP`; true local improvement at stop = true.
- baseline__combined_mild: `PREMATURE_CONSERVATIVE_STOP`; true local improvement at stop = true.

## Evidence boundary

All truth-based missed-opportunity and correct-stop checks were computed after the policy decision and were not fed back into proposal, ranking, fitting, stopping, or threshold selection.

- `INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW`
- `GLOBAL_MODEL_RELIABILITY_RULE_NOT_FROZEN`
- `REAL_ROBOT_HARD_SAFEGUARD = NOT_DEFINED_NOT_APPROVED`
- `NOT_HUMAN_READY = true`
- `NOT_ROBOT_MOTION_APPROVED = true`
