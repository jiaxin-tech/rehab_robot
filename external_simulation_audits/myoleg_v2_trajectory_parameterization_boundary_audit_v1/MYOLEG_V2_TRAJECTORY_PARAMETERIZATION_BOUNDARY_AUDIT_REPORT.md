# MyoLeg V2 trajectory parameterization boundary audit V1

## Status and scope

- Stage: `MYOLEG_V2_TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_V1`
- Protocol SHA-256: `5699e75a73a28d9df037a01de2c047a241c9cb5a4c0598e16d66e8f8d2f708a5`
- Development subjects: `24`
- Frozen candidates: `16,675`
- Held-out scientific truth access: `0`
- Outcome: `TRAJECTORY_PARAMETERIZATION_ROOT_CAUSE_SUPPORTED`
- Current V2 decision: `NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION`
- Offline diagnosis/design only; no V3 landscape, learner, BO, robot, or hardware operation was run.

## Q1 — What the three current parameters actually change

`delta_hip_amp` adds the frozen minimum-jerk amplitude basis to hip. It leaves the cycle start/end, duration, branch durations, closure and C2 continuity fixed, but moves the hip turning maximum and therefore changes hip ROM, q-space geometry, velocity and acceleration.

`delta_knee_amp` does the same for knee on top of the phase-warped knee trajectory. It moves the knee turning maximum and knee ROM. Its admitted upper coordinate is only `+0.5 deg`, because higher original proposals cross the frozen MyoLeg 120-degree trusted-domain gate.

`knee_phase_shift` applies `r' = r + shift*64*r^3*(1-r)^3` within both branches. It fixes branch endpoints, extrema, duration, closure and C2, but changes knee timing relative to hip, q-space loop geometry, dq and ddq. It is the only current approximately fixed-ROM coordinate.

## Q2 — How much common candidate effect is associated with ROM/extrema?

`ROM_EXTREMA_EXPLAINED_COMMON_EFFECT = 0.997742` (`DOMINANT`). This is the preregistered ROM-only quadratic descriptive OLS R2. Adding phase raises R2 to `0.999985`, and the fixed kinematic descriptor model reaches `0.999985`.

This is an **association/decomposition result, not a causal estimate**. Nevertheless, together with the exact generator semantics it shows that the dominant common ordering is largely aligned with changing task amplitude/ROM rather than isolated subject-specific path shape.

## Q3 — What remains when ROM/extrema are fixed?

There are `667` complete matched-ROM groups, each containing `25` phase values. But only one independent alpha coordinate varies within any group. Formal status:

`CURRENT_GRID_CANNOT_IDENTIFY_FIXED_ROM_PATH_EFFECT`

The current grid can identify the behavior of its single phase warp at fixed ROM; it cannot identify whether a richer two-or-more-dimensional interior path family would be subject-specific.

## Q4 — Is phase-only variation common-monotonic, and why?

Across all `6` preregistered amplitude pairs, phase-only J is common-monotonic: `True`. Direction: `phase increase worsens J`. The six-subject deterministic replay subset shows that moving from phase `-0.03` to `+0.03` changes the time placement of q/dq/ddq while extrema remain fixed. Mean high-minus-low component RMS changes include hip bias/gravity `+0.026056 Nm`, knee bias/gravity `+0.022804 Nm`, hip inertia `-0.011092 Nm`, and knee inertia `-0.004211 Nm`.

Thus phase monotonicity is consistent with a shared deterministic timing effect on gravity/bias and inertial demand under the unchanged RMS objective. This mechanism audit does not change phase semantics and does not claim physiology.

## Q5 — Why is the oracle on three boundaries?

- Hip `+2`: `A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE`; the generated hip maximum remains separated from the native/trusted hip upper range.
- Knee `+0.5`: `B_MYOLEG_SIMULATOR_VALIDITY_LIMIT`; original proposals extended to `+2`, but the frozen all-model knee upper-domain admission gate removes them.
- Phase `-0.03`: `A_ARBITRARY_ORIGINAL_PROPOSAL_RANGE`; all 25 frozen phase values pass the phase-warp integrity checks.

None of the three is classified as a prescribed rehabilitation-task constraint. The audit makes no claim about an optimum outside the frozen domain and does not recommend bound widening as the primary fix.

## Q6 — Is V2 optimizing task amplitude rather than pure path shape?

Yes. `CURRENT_PARAMETERIZATION_CHANGES_REHABILITATION_TASK_AMPLITUDE = true`. Two of three coordinates explicitly change target maxima and ROM. V2 therefore mixes task/ROM dosage modification with timing/path modification.

## Q7 — What should be invariant in a cleaner problem?

The future fixed-task problem should **MUST PRESERVE** hip and knee extrema, duration, branch endpoints, q/dq/ddq closure, C2 continuity, finite native simulator validity, and no-clipping generation. It **SHOULD PRESERVE** branch duration and the measured flexion/extension asymmetry. Interior relative coordination, curvature/loop area, and branch-specific smooth deviations may vary.

## Q8 — Which low-dimensional parameterization is structurally best?

The preregistered equal-weight structural scores are P1 `27/30`, P2 `27/30`, P3 `20/30`, and P4 `30/30`.

Primary: `P4_BRANCH_AWARE_COORDINATION_FUNCTION` (2 parameters). It leaves hip unchanged as the prescribed task coordinate and changes only knee's smooth branch-interior coordination relation using separate normalized-phase functions for flexion and extension. Direct `q_knee=f(q_hip)` is not used because the measured hip branch is not strictly single-valued. At the diagnostic kinematic point its maximum extrema change is `0.000278102 deg`; no clipping or objective evaluation was used.

Fallback: `P2_INTERIOR_BSPLINE_JOINT_PERTURBATION` (4 parameters). Its branch/joint coefficients are interpretable and C2-compatible, but a future independent stage must constrain its local coefficient range so extrema remain fixed without clipping.

P3 is not primary because the Euclidean angle-space normal has coordinate-scale/mechanical-meaning ambiguity and a measured low-speed normal degeneracy fraction of `0.007481`. P1 is retained as a baseline but is too limited to answer the broader path-shape question.

## Q9 — Is current V2 adequate for the stated personalization question?

`NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION`

V2 remains interpretable as a task-amplitude/timing optimization family, but it is not scientifically adequate as a clean test of subject-specific path optimization at fixed rehabilitation ROM.

## Q10 — Next branch

Recommend Branch A: `MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_V1`. The next stage should formally design and preflight the P4 primary and P2 fallback ranges using kinematic/simulator validity only. It must not be executed automatically and must not select ranges by new oracle diversity.

## Integrity

- Frozen root-cause, truth-landscape, candidate-domain, cohort, objective, normalization, reference and model inputs passed their SHA checks.
- Development-only scientific access: `24` subjects.
- Held-out scientific access: `0`.
- Five-parameter / NN / PINN / BO: not run.
- Robot / hardware: not accessed.
- Full pytest: `1548 passed, 5 skipped, 0 failed, 0 warnings reported` in `284.04 s` (`python3 -m pytest -q`).
- Runtime: `9.743 s`.
