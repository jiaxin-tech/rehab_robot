# Biarticular coupling variation audit

The structurally verified right-side actuators that have non-zero transmission moment arms about both independent hip flexion and knee flexion over the frozen 401-state path are:

`bflh_r, grac_r, recfem_r, sart_r, semimem_r, semiten_r, tfl_r`

Membership is based on `mjData.actuator_moment @ T(q)` with a 1e-07 m numerical threshold, not name guessing. A high-level `BIARTICULAR_FORCE_SCALE` maps exactly to `gainprm[2]` and `biasprm[2]` of this frozen list. A passive-coupling factor can analogously map to `gainprm[7]` and `biasprm[7]`.

Therefore native tendon forces can create different hip-knee coupling without adding a synthetic coupling torque equation. For primary P0, use a biarticular `fpmax` factor; the same-group `force` factor is observationally equivalent at zero activation. The group list and scientific scale range still require external/manual review before cohort generation.
