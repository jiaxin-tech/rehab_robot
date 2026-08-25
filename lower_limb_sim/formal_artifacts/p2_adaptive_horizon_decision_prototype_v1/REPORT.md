# P2_ADAPTIVE_HORIZON_DECISION_PROTOTYPE_V1

Manifest SHA-256: `ace9586f98bfc5142ee310539e6c42b02d7164a3fbb91fdad8d7352110c96f9b`

## Frozen research rule

The default-off adaptive prototype evaluates horizons in the frozen order
`1 → 2 → 3 → 5`. It uses the unchanged independent 0.005 magnitude gate and
the horizon-specific calibrated direction interval. Escalation is allowed only
when every existing generator-grid node along one coordinate and one signed
direction has a finite, strictly improving predicted objective. The first
horizon with an eligible endpoint is selected. No intermediate trajectory is
executed; every endpoint is followed by model refit and full-map recomputation.

## Development shadow results

- H1_SINGLE_STEP_ONLY: trials=56, explore=56, missed=0, false=0, small-step=0/9, mean final J=0.999663936, regret=0.034705854, unique final alpha=2, boundary cases=1/15.
- H2_FIXED_BUNDLE_5: trials=116, explore=56, missed=7, false=0, small-step=9/9, mean final J=0.968794100, regret=0.003840217, unique final alpha=1, boundary cases=15/15.
- H3_ADAPTIVE_HORIZON: trials=116, explore=56, missed=0, false=0, small-step=9/9, mean final J=0.969000562, regret=0.004046678, unique final alpha=2, boundary cases=14/15.

Adaptive endpoint horizon usage: `H1=0, H2=0, H3=1, H5=59`.

## Questions

### A. Does adaptive horizon approach fixed BUNDLE_5 performance?

`True` under the pre-frozen criterion: equal 9-path recovery and final
J/regret no worse than BUNDLE_5 by more than the unchanged 0.005 tolerance.

### B. Does adaptive horizon reduce fixed BUNDLE_5 trial cost?

`False`. Adaptive used 116 trials versus
116 for fixed BUNDLE_5.

### C. Does adaptive horizon reduce unified boundary optimum collapse?

`True`. Adaptive produced
2 unique final alpha vectors and
14/15 boundary-saturated cases, versus
1 and
15/15 for fixed BUNDLE_5.

### D. Which subject/scenario shadows required longer horizons?

- baseline__matched_linear: H5 × 4
- hip_stiff__matched_linear: H5 × 4
- knee_stiff__matched_linear: H5 × 4
- heavy_leg__matched_linear: H3 × 1
- heavy_leg__matched_linear: H5 × 3
- baseline__nonlinear_stiffness_mild: H5 × 4
- baseline__hip_knee_coupling_mild: H5 × 4
- baseline__nonlinear_damping_mild: H5 × 4
- baseline__structured_residual: H5 × 4
- baseline__combined_mild: H5 × 4
- prospective_subject_001__matched_linear: H5 × 4
- prospective_subject_001__nonlinear_stiffness_strong: H5 × 4
- prospective_subject_002__matched_linear: H5 × 4
- prospective_subject_002__hip_knee_coupling_strong: H5 × 4
- prospective_subject_003__matched_linear: H5 × 4
- prospective_subject_003__combined_strong: H5 × 4

These are development-only diagnostics. They do not establish human benefit,
robot safety, or clinical effectiveness.

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Independent calibration supplies residual scales only.
- No held-out final test and no prospective cohort.
- P2 V1, objective, five-parameter model, generator, ROM, active reference,
  0.005 tolerance, and 90% support gate are unchanged.
- Final state: `OFFLINE_ONLY`, `DEFAULT_OFF`, `NOT_HUMAN_READY`,
  `NOT_ROBOT_APPROVED`.
