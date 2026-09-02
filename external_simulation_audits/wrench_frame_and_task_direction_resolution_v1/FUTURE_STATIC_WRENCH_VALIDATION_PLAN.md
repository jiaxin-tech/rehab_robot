# Future Static Wrench Validation Plan

This is a protocol concept only. It is not executed and does not itself approve connection, power, force application or motion.

## Preconditions

An independent site safety review must authorize a stationary, non-human fixture; exact robot/controller/tool/load identity; safe operation state; calibrated external reference instrument and loading fixture; force/moment limits; stop procedure; operator roles; and read-only query/connect side effects. No human supplies the test load.

## Controlled evidence blocks

1. Capture unloaded raw output at multiple stationary poses before and after tests to characterize offset, drift and pose dependence without calling it compensation.
2. Apply independently measured forces in positive and negative directions along at least three non-collinear/orthogonal axes. Predefine source frame, load point and sign.
3. Query world and, where independently safe and semantically supported, tool/flange expressions without changing pose; compare the reported vectors with known rotation predictions.
4. Repeat at multiple non-degenerate base/tool orientations to distinguish transpose/sign/axis errors.
5. Apply known forces at known lever arms and sign reversals to identify moment origin/shift behavior using `M_B=R M_A+r x F_B`.
6. Record query start/end/publish times and repeated identical values to estimate query duration, effective source-update cadence and state-wrench skew; do not infer device latency from host midpoint.
7. Compare raw and session software-zero outputs across poses to assess bias, hysteresis and whether controller compensation remains unexplained.

Freeze acceptance rules, force levels, repetitions and uncertainty budgets before data inspection. Required outputs include axis/sign confusion matrix, rotation residuals, cross-axis ratio, linearity/reversal residuals, reference-point moment residuals, bias/drift/pose-dependence results and timing distributions. Any ambiguous result stays fail-closed and must not set `BASE_WRENCH_ROTATION_VERIFIED=true`.
