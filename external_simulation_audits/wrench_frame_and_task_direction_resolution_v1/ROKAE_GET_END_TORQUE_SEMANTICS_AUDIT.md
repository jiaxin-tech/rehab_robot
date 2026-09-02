# ROKAE `getEndTorque` Semantics Audit

## SDK identity and exact call

- Repository wrapper expectation: xCoreSDK `0.7.0` (`VERIFIED` from source); it rejects a different runtime version.
- Actual Windows native library loaded in this macOS audit: not loaded (`REQUIRES_PHYSICAL_VALIDATION`).
- Exact local declaration: `getEndTorque(ref_type, joint_torque_measured, external_torque_measured, cart_torque, cart_force, ec)` (`VERIFIED`).
- Project call uses that order and validates at least 6/6/3/3 finite values (`VERIFIED`).

| Semantic item | Local evidence | Status |
|---|---|---|
| joint measured torque | 6 axes; force-sensor-measured joint torque; N*m | VERIFIED |
| external joint torque | 6 axes; controller model plus measurement derived; N*m | VERIFIED |
| Cartesian torque | XYZ; N*m | VERIFIED for shape/unit; physical semantics partial |
| Cartesian force | XYZ; N | VERIFIED for shape/unit; physical semantics partial |
| accepted API request frames | getEndTorque documentation lists world/flange/tool | VERIFIED |
| current request argument | settings/default/call request `world` | VERIFIED |
| returned force/moment coordinate expression | documentation says relative to requested world/flange/tool | PARTIALLY_VERIFIED; axes are documented, physical experiment absent |
| force reference point | no unambiguous statement | NOT_DOCUMENTED |
| moment reference point | tool text mentions TCP; world/flange wording does not unambiguously fix all origins | PARTIALLY_VERIFIED |
| force sign | robot-on-environment versus environment-on-robot is not stated | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |
| compensation | gravity/tool/load/bias/friction compensation for Cartesian output is not stated | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |
| measured versus estimated Cartesian wrench | joint channels are distinguished; Cartesian channel origin is not explicitly classified | PARTIALLY_VERIFIED |
| update/query semantics | one synchronous API query fills arrays; project records host bounds | PARTIALLY_VERIFIED; controller source-update cadence unknown |
| source/device timestamp | not returned by signature/arrays | NOT_DOCUMENTED; current value is host midpoint only |
| RT synchronization | no SDK synchronization contract with `tcpPoseAbc_m` | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |

## Requested versus verified

`REQUESTED_WRENCH_FRAME = world`

`VERIFIED_WRENCH_FRAME = NONE_PHYSICALLY_VERIFIED`

The request and documented expression label are real software facts; they do not prove the physical load sign, compensation state, reference point, controller tool/TCP configuration or physical world/base registration. Therefore `WORLD_WRENCH_VERIFIED` is false.
