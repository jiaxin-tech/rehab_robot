# Next Dependency Plan

The wrench protocol and actual strap geometry use different reference equipment, uncertainties and scientific gates, so they remain independent. A future successful static wrench result would establish force expression/sign only; it would not measure the two physical strap attachment points or validate their pose dependence.

Exact next dependency: `STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1`.

That protocol should register `p_limb_attach_B(t)` and `p_robot_attach_B(t)`, fixture/bed/base frames, placement repeatability, tautness/routing and direction uncertainty. Only after both wrench and geometry validation can the primary endpoint definition be reconsidered. This dependency was not executed.
