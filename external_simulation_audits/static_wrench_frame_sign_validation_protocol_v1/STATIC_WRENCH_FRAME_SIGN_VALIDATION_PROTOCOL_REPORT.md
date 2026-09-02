# Static Wrench Frame/Sign Validation Protocol V1

## Formal status

`STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED`

This stage froze a future force-only, static, non-human protocol. It performed no physical validation.

## Hypotheses and design

H1-H5 are frozen unchanged: axis/direction response, paired sign reversal, cross-pose world consistency, zero/pose bias characterization and host-timing-only interpretation. The full matrix contains three pose roles, six world directions and two load-level roles. Each cell has `5` independent load applications and `100` host queries per PRE/LOAD/POST window.

P0 is only the current stationary-pose role. P1/P2 contain no coordinates and require separate positioning approval. If only P0 is possible, H3 becomes `POSE_DEPENDENCE_NOT_YET_VALIDATED` and full world-frame validation is prohibited.

## Load and thresholds

Loads must come from calibrated non-human equipment, never a hand or subject. The two N values remain null: `FORCE_MAGNITUDE_REQUIRES_SAFETY_REVIEW`. Direction-angle, leakage, sign, drift, pose-consistency and SNR thresholds remain null: `THRESHOLD_REQUIRES_CALIBRATION_EVIDENCE`. They must be frozen before physical result reveal; the SDK result cannot tune them.

## Metrics

Raw PRE/LOAD/POST windows remain immutable. The preregistered response contrast is `DeltaF=mean(load)-0.5*(mean(pre)+mean(post))`. Both same-direction and opposite-direction angle errors are reported until paired +/- cells establish at most one global sign convention. Cross-axis leakage is orthogonal norm divided by dominant-axis magnitude. Magnitude linearity is secondary and available only with two approved known levels.

## Safety and current authorization

The current safety file is unreviewed, force/workspace/tool values are null, world/bed registration is absent and physical equipment is unspecified. Therefore future physical execution is `NOT_AUTHORIZED`. This protocol does not authorize connection, power, enable, positioning, load or motion.

## State preservation

`REQUESTED_WRENCH_FRAME=world`; `VERIFIED_WRENCH_FRAME=NONE_PHYSICALLY_VERIFIED`; `WRENCH_FORCE_SIGN_VERIFIED=false`; `BASE_WRENCH_ROTATION_VERIFIED=false`; `TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION`. Moment remains `NOT_FULLY_VALIDATED`, and no endpoint was computed/finalized/validated.

## Next dependency

`STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1`. It remains separate because wrench response and strap geometry require different evidence. It was not executed.
