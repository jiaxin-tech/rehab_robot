# Wrench Semantics Audit

## Formal finding

`WRENCH_FRAME_SEMANTICS_NOT_VERIFIED`

The locally installed xCoreSDK stub documents the exact call order
`getEndTorque(ref_type, joint_torque_measured, external_torque_measured, cart_torque, cart_force, ec)`.
It defines six joint measured torques in N*m, six controller-model-derived external joint torques in N*m, Cartesian torque XYZ in N*m, and Cartesian force XYZ in N. For this API it lists `world`, `flange`, and `tool`; the project currently requests `world`. The wider SDK enum also contains base/user-like frames, but this does not prove that `getEndTorque` accepts them.

## What is not established

- No documented physical sign convention: the code cannot say whether positive force is robot-on-patient or patient-on-robot.
- No documented controller compensation/bias state or tool/load dependence.
- The Chinese stub calls world/flange results "relative to" the frame and tool results relative to the TCP point. It does not unambiguously state the moment reference point for every option, nor whether a frame request changes axes only or also the point.
- No device/source timestamp, update cadence, source age, or synchronization contract with RT state.
- No formal validated physical result file was available to this stage. Existing diagnostics are read-only instruments, not evidence that the checks were performed and approved.

Therefore variable names and offline rotation unit tests cannot close the semantics gap. Cartesian force may only become a primary endpoint source after controlled frame/sign/bias/timing validation. Cartesian moment needs both rotation and reference-point translation semantics.

## `BASE_WRENCH_ROTATION_VERIFIED`

It remains `false`. To change it, a future independently approved, stationary/read-only validation must use known robot/world/tool orientations and known-direction applied loads, compare raw world/tool outputs, and verify the declared rotation convention in multiple non-degenerate orientations. It must check `F_b = R_b_from_w F_w`.

For moments, a separate known lever arm must test the full relation `tau_b = R tau_w + p x (R F_w)`. Rotation-only is not a point transform. If the primary endpoint ultimately consumes only Cartesian force, the moment-reference translation term does not enter the force projection; this permits excluding moment from that endpoint, but it does not resolve force sign, force expression axes, bias, contact line of action, or synchronization.
