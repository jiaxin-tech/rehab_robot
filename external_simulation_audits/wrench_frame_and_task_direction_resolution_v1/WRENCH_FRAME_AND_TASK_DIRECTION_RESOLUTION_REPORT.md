# Wrench Frame and Task Direction Resolution V1

## Formal decision

`WRENCH_AND_TASK_DIRECTION_PARTIALLY_RESOLVED_REQUIRES_STATIC_VALIDATION`

The stage resolved software/API and mathematical semantics as far as local evidence permits. It did not create physical frame, sign, compensation, timing or strap-geometry evidence.

## Wrench result

- SDK contract in repository: xCoreSDK `0.7.0`; exact 6/6/3/3 `getEndTorque` call and units are documented.
- `REQUESTED_WRENCH_FRAME = world`.
- `VERIFIED_WRENCH_FRAME = NONE_PHYSICALLY_VERIFIED`.
- World is a documented request/expression label, not a physically verified world wrench.
- Force sign is not documented; Cartesian compensation and exact force/reference-point semantics remain incomplete.
- Moment point is partially documented only; full point shifts require `M_B=R M_A+r x F_B`.
- Only host query bounds/midpoint exist; no device/source timestamp or RT synchronization contract exists.

## Rotation result

All `9/9` offline canonical math cases passed, including identity, +90 degree x/y/z, inverse transpose and moment shift. This confirms internal active-column-vector algebra under the source convention. It does not confirm the physical SDK frame convention, so `BASE_WRENCH_ROTATION_VERIFIED=false` remains unchanged.

## Frame and geometry result

`baseFrame()` and `toolset.end` are runtime-queryable but have no current frozen values. Active HMI tool/workobject is explicitly unverified. Bed axes are null/unreviewed. TCP-to-strap and limb attachment points are unmeasured. L2 remains the configured equivalent strap traction point, not an observed ankle or automatically the physical cuff attachment.

## Task direction result

`TASK_DIRECTION_DEFINITION = ACTUAL_STRAP_PULL_LINE_OF_ACTION` is the physical target. With `p_limb_attach_B` and `p_robot_attach_B` measured in one validated base frame:

`d_task_B(t)=normalize(p_robot_attach_B(t)-p_limb_attach_B(t))`

The positive geometric direction is limb-to-robot. Both points and their variation must be validated. TCP tangent is command/motion direction, not automatically interaction line of action. Because wrench force sign is unresolved, positive/negative `F_task` physical meaning is also unresolved.

## What was not done

No robot connection, enable, motion, human loading, endpoint value, trajectory comparison, repeatability/sensitivity, model/PINN training or BO was performed. Hardware/control/safety code and frozen results were not modified.

## Single next stage

`STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1`

It should preregister the exact non-human stationary fixture, known loads, signs, frames, lever arms, timing evidence and fail-closed acceptance rules. It was not executed.
