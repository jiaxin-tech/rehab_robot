# Dynamic Task-Direction Reconstruction Plan

Static validation establishes geometry at registered fixture configurations; it does not make the line constant during the 24 s trajectory.

| Candidate | Evidence meaning | Current disposition |
|---|---|---|
| A: TCP-derived robot point + fixed limb point | assumes limb/exit fixed | invalid for a moving limb unless future variation is below a prefrozen bound |
| B: both endpoints from kinematics | reproducible model proxy | requires physical mapping calibration; cannot be called actual geometry now |
| C: robot point from TCP + limb point from measured limb pose | practical state reconstruction | preferred online candidate after pose/exit mapping and timing validation |
| D: direct external tracking of free span/endpoints | strongest physical geometry reference | preferred validation reference where feasible |
| E: one static direction | approximation | allowed only for a declared limited range after validation |

Minimum dynamic information is: measured `p_attach_TCP`; synchronized valid `T_B_TCP(t)`; calibrated `T_B_R`; measured limb/surrogate pose `T_R_S(t)`; configuration-dependent exit mapping or directly tracked free-span line; strap/cuff/routing/tautness state; placement identifier; transform and timing uncertainty. Recompute per valid sample after all gates. Missing information yields `d_task(t)=null`.

`TCP_TRAJECTORY_TANGENT != STRAP_PULL_LINE_OF_ACTION`. Future angles between strap direction and TCP tangent, fixed bed direction or model direction are diagnostics only and cannot select the endpoint-favorable approximation.
