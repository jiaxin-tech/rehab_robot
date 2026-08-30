# MODEL_MODIFICATION_DIFF

The upstream XML/assets are read-only. The generated derived XML makes only these changes:

1. Add `derived_root_anchor` on `root` and `derived_world_supine_anchor` on world.
2. Add site-based weld `derived_root_supine_weld` at [0,0,1] m with -90 deg world-y rotation.
3. Add single-joint equality locks at native zero for:
   - `hip_adduction_r`
   - `hip_rotation_r`
   - `ankle_angle_r`
   - `subtalar_angle_r`
   - `mtp_angle_r`
   - `hip_flexion_l`
   - `hip_adduction_l`
   - `hip_rotation_l`
   - `knee_angle_l`
   - `ankle_angle_l`
   - `subtalar_angle_l`
   - `mtp_angle_l`
4. Preserve `hip_flexion_r`, `knee_angle_r`, all original auxiliary/patella joints and all 14 source equalities.
5. Preserve every original body, muscle actuator, tendon and native joint range.
6. Set floor/terrain contact masks to zero for `SUPINE_NO_BED_CONTACT`; do not add a bed.
7. Convert mesh/texture references to absolute, hashed references inside the frozen external environment; assets are not copied or edited.
8. Keep gravity and the 0.001 s integration timestep unchanged.

No joint, muscle, tendon, equality, mesh or body was deleted. No knee range was changed.
