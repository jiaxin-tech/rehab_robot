# Anthropometry variation audit

## Decision

`body_mass` is numerically perturbable, but V1 must scale each selected segment's `body_inertia` by the same mass factor and keep `body_ipos` (COM) fixed. This is a deliberately conservative mass-only scaling assumption; it does not claim geometrically scaled anatomy.

- Femur: `femur_r` mass + inertia together.
- Tibia/knee assembly: `tibia_r` and, if the factor is defined as an assembly, `patella_r` mass + inertia together.
- Foot complex: `talus_r`, `calcn_r`, and `toes_r` mass + inertia together.
- Pelvis: the actual `pelvis` body exists, but it is shared by both limbs. A unilateral target-leg cohort must not silently perturb it; a bilateral/global anthropometry protocol is required first.

COM is held fixed in V1 because independently moving it without a segment-scaling model can be inertially inconsistent. Segment length is `DO NOT PERTURB_IN_V1`: body/joint positions, 384 attachment sites, wrapping geoms, tendon paths and moment arms would all require coordinated reconstruction and revalidation.
