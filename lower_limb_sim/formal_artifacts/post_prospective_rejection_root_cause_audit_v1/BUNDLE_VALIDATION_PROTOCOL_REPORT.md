# Designated bundle validation protocol report

`DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1` freezes a geometry-only plan containing 648 endpoint pairs across 54 balanced strata. The plan SHA-256 is `3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84`.

The existing 324-pair local plan is insufficient for bundle uncertainty because it did not predeclare bundle identity, all continuous intermediate formal neighbors, or 2/3/5-step endpoint-residual strata. Even where an old pair has the same endpoint distance, it was not selected as a direction-consistent bundle calibration unit.

The new plan crosses hip/knee/phase, positive/negative direction, 2/3/5 steps, and lower-boundary/interior/upper-boundary locations, with 12 SHA-selected pairs per stratum. Selection uses only generator geometry and IDs. It uses no predicted J, prospective error, successful prospective location, or future truth.

No bundle truth or `e_deltaJ_bundle` is generated here. Every outcome field remains blank with status `PENDING_FUTURE_INDEPENDENT_BUNDLE_CALIBRATION`. A future, independent calibration task must evaluate this frozen plan without reselection.
