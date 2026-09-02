# Static Strap Pull Geometry Validation Protocol V1

## Formal status

`STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED`

This stage froze a future static, nonhuman geometry protocol. It executed no physical measurement and cannot output a validated class.

## Answers to the ten protocol questions

1. **Robot-side point:** the actual eyelet/hook/fixture load-transfer center, expressed as a measured fixed offset `p_attach_TCP`; it is not automatically TCP or flange origin.
2. **Unique limb-side point:** the current wide cuff does not have an evidenced unique physical point. Contact is distributed and may also transmit a net moment.
3. **Proposed equivalent:** use the observed contact-to-free-span exit point, or a fitted taut free-span line, as an operational line point. This is a model approximation, not the true pressure resultant point.
4. **Common frame:** compute both endpoints in robot base `B`, linked through a physical `REHAB_SETUP_FRAME R`.
5. **Transform:** obtain `T_B_R` by redundant rigid fiducial registration using independent metrology; no numeric transform is invented and robot probing needs separate authorization.
6. **Repeatability:** `10` complete remove/reattach setups and `3` point measurements per setup, separated into within- and between-setup covariance; no post-result extension.
7. **Static or dynamic:** configuration-dependent unless future evidence supports a declared limited static approximation.
8. **Dynamic minimum:** `p_attach_TCP`, synchronized `T_B_TCP(t)`, calibrated `T_B_R`, limb pose or direct free-span tracking, exit mapping, strap state/placement and timing/uncertainty.
9. **2-D mapping:** `NOT_YET_CALIBRATED`. L2 stays a knee-to-equivalent traction-point length and is neither ankle nor measured attachment.
10. **Before `J_force`:** authorized nonhuman execution must validate common-frame endpoints/free-span, routing, transform, ten-setup repeatability and propagated angular uncertainty under prefrozen thresholds; separately, the static wrench protocol must validate frame/sign sufficiently.

## Mechanics and sign boundary

The positive geometric direction remains limb exit toward robot attachment. This does not resolve the physical sign of the reported wrench; that remains a separate static wrench validation dependency. `TCP_TRAJECTORY_TANGENT` and fixed bed/model directions are diagnostics only, never fail-open fallbacks.

## Current state

All thresholds, `T_B_R`, attachment offset, jig coordinates and geometry preload are null pending independent review. Therefore `TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION`, `PRIMARY_ENDPOINT_FINALIZED=false`, `PRIMARY_ENDPOINT_VALIDATED=false`, `NOT_HUMAN_READY` and `NOT_ROBOT_APPROVED` remain unchanged.
