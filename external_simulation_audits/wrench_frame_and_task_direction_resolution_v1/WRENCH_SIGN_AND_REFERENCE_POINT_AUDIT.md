# Wrench Sign and Reference-Point Audit

## Formal statuses

- `WRENCH_SIGN_STATUS = WRENCH_FORCE_SIGN_NOT_VERIFIED`
- `WRENCH_REFERENCE_POINT_STATUS = WRENCH_REFERENCE_POINT_PARTIALLY_DOCUMENTED_NOT_PHYSICALLY_VERIFIED`
- `WRENCH_COMPENSATION_STATUS = WRENCH_COMPENSATION_NOT_DOCUMENTED`
- `WRENCH_TIMESTAMP_STATUS = HOST_QUERY_BOUNDS_VERIFIED_DEVICE_SOURCE_TIMESTAMP_UNAVAILABLE`

The Cartesian positive-force meaning is not stated as robot-on-environment or environment-on-robot. It must not be inferred from `external_torque_measured`, function naming or a plot. A later controlled test must apply known load directions and explicit sign reversals.

Force is a free vector for coordinate-expression rotation: `F_B=R_B_from_A F_A`; translating the reference origin does not change that 3D force vector. Moment is origin-dependent. If an A-origin wrench is re-expressed at a B-origin, the verified convention must implement `M_B=R M_A+r x F_B`. Consequently a future moment endpoint or six-dimensional wrench norm cannot use rotation-only torque unless the reference point is identical or the shift is known and applied.

Software bias subtraction in the repository is a session reference offset, not proof of controller compensation. `getEndTorque` documentation does not state gravity/tool/load/friction compensation for Cartesian outputs. Host query start/end/midpoint are provenance bounds, not a controller measurement timestamp or transport-delay estimate.
