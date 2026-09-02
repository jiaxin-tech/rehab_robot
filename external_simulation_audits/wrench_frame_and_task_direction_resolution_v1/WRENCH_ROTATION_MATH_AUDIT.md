# Wrench Rotation Math Audit

## Conventions in current code

- Vectors are 3-element column-vector semantics; matrix multiplication is `result[row] = sum(R[row,column]*v[column])`.
- `rpy_euler_xyz_rotation_matrix` uses `Rz(yaw) @ Ry(pitch) @ Rx(roll)` as an active XYZ-Euler rotation.
- SDK `baseFrame()` is interpreted as `^world T_base`, hence the source code builds `R_world_from_base` and transposes it to `R_base_from_world`.
- Corrected world force is expressed in base as `F_base = R_base_from_world F_world`.
- The same pure rotation is currently applied to torque but deliberately labelled rotation-only pending validation. A full point change requires `M_B = R M_A + r x (R F_A)` using a verified displacement and convention.

## Offline results

`9/9` deterministic canonical tests passed at tolerance `1e-12`: identity, +90 degree rotations about x/y/z, transpose inverse, and a known reference-point moment shift.

These results verify implementation algebra under the declared convention. Every row deliberately records `physical_frame_verified=false`. The SDK phrase "Euler XYZ" and the wrapper's direction are consistent with the implemented convention, but only static known-direction evidence on the exact robot/tool/base setup can verify the physical mapping. Therefore `BASE_WRENCH_ROTATION_VERIFIED` remains false.
