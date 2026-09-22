# V3 trajectory / objective discriminability audit V1

## Scope and evidence boundary

This stage is a diagnostic-only replay of the frozen
`FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1`. It reuses, without modification:

- the five frozen synthetic MuJoCo legs and SubjectROMProfiles;
- the frozen 25 x 25 V3 beta grid and beta range;
- the existing 5 x 625 primary landscapes;
- the existing simulated required joint-torque-vector RMS endpoint;
- `theta_shank = q_hip - q_knee`.

It is offline synthetic simulation evidence only. It is not robot, human,
clinical, comfort, safety, or effectiveness validation. No BO or other
personalization algorithm was run, no PINN was trained, and no robot interface
was used.

## Executive diagnosis

The primary limitation is endpoint compression, not an absence of measurable
V3 kinematic variation. Across the beta domain, V3 produces 8.30--9.96 deg
maximum instantaneous knee-angle separation and up to 0.391 s displacement of
selected knee progression events. Local mechanical separation reaches 46.30
N m, but the unchanged full-cycle aggregate endpoint spans only 0.036--2.166%
of its reference value across the five legs.

The aggregate also combines opposing joint responses. For legs 0--3, hip-only
and knee-only RMS rankings have Spearman correlations from -0.937 to -0.997;
the hip-only optimum is `(+0.03,-0.03)` while the knee-only and aggregate
optima are `(-0.03,+0.03)`. The knee response and passive-load exposure govern
the aggregate descent direction.

The full endpoint nevertheless has a nearly universal normalized shape:
pairwise gradient-field cosine similarity is 0.99322--0.99989, every evaluated
grid point descends toward negative beta_flex and positive beta_extend, and no
strict interior local minimum exists. This is important secondary evidence of
shared mechanics, but it does not satisfy H-C's requirement that the primary
endpoint first have a reasonable dynamic range.

## Q1. How large is the V3 trajectory change at beta +/-0.03?

V3 leaves the hip q/dq/ddq array-identical for every candidate; all measured
spread is in the knee coordination trajectory.

| Quantity across 625 candidates | Median pairwise RMS | P95 pairwise RMS | Maximum pairwise RMS | Maximum instantaneous |
|---|---:|---:|---:|---:|
| knee q | 0.907--1.089 deg | 1.863--2.236 deg | 2.528--3.033 deg | 8.302--9.962 deg |
| knee dq | 0.677--0.812 deg/s | 1.472--1.766 deg/s | 1.942--2.330 deg/s | 5.996--7.195 deg/s |
| knee ddq | 2.019--2.423 deg/s2 | 3.224--3.869 deg/s2 | 4.058--4.870 deg/s2 | 24.898--29.877 deg/s2 |

For the reference-to-`(-0.03,+0.03)` comparison, joint-path mean distance is
0.680--0.816 deg, RMS distance is 1.239--1.486 deg, and maximum distance is
4.159--4.990 deg across the five ROM profiles. The opposite boundary has a
similar magnitude but opposite coordination direction.

The warp is `w(s)=s+beta*64*s^3*(1-s)^3`; its basis reaches one at branch
progression 0.5. Thus beta `+/-0.03` is a `+/-3` percentage-point branch-phase
perturbation at the warp maximum, or six percentage points between domain
extremes.

Measured event-time effects in the 24 s trajectory are:

- flexion knee 25/50/75% progression: 0.0368/0.1677/0.3913 s full-domain range;
- extension knee 25/50/75% progression: 0.2913/0.3204/0.1497 s full-domain range;
- reference to `(-0.03,+0.03)` at 50% progression: +0.0909 s during flexion
  (delay) and -0.1656 s during extension (advance);
- reference to `(+0.03,-0.03)`: -0.0768 s during flexion and +0.1547 s during
  extension;
- knee peak-velocity argmax range: 0.068 s in flexion and 0.468 s in extension;
- knee peak-acceleration argmax range: 1.836 s in flexion and 1.820 s in
  extension. These last values reflect switching among local acceleration
  peaks, so progression-crossing times are the more stable timing measure.

Conclusion for H-A: V3 is restricted to knee timing/coordination and its
pairwise RMS changes are modest, but its boundary and timing changes are not
negligible enough to explain the nearly flat scalar endpoint by themselves.

## Q2. What is the full primary-landscape dynamic range?

| Leg | J_min (N m) | J_max (N m) | J_reference (N m) | Absolute range (N m) | Range/reference | CV | Near-oracle counts at 0.1/0.5/1/2/5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEG_0_NOMINAL | 32.1811 | 32.2451 | 32.2138 | 0.0641 | 0.1988% | 0.0424% | 304 / 625 / 625 / 625 / 625 |
| LEG_1_HEAVY_HIP_STIFF | 98.5049 | 98.5407 | 98.5221 | 0.0358 | 0.0363% | 0.0078% | 625 / 625 / 625 / 625 / 625 |
| LEG_2_KNEE_DOMINANT | 173.9157 | 177.7233 | 175.7863 | 3.8076 | 2.1660% | 0.4602% | 6 / 75 / 272 / 612 / 625 |
| LEG_3_NONLINEAR_COUPLED | 317.3970 | 322.0677 | 319.6692 | 4.6707 | 1.4611% | 0.3105% | 10 / 160 / 500 / 625 / 625 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 163.6813 | 164.4505 | 164.0675 | 0.7691 | 0.4688% | 0.0996% | 62 / 625 / 625 / 625 / 625 |

## Q3. Why are all 625 candidates within 5% of the oracle?

Primarily because 5% is wider than every observed landscape. The complete
domain spans at most 2.166% of reference and only 0.036% in the flattest leg.
The tighter counts show that legs 2 and 3 do have measurable ordering, while
legs 0, 1, and 4 are already nearly saturated by the 0.5% threshold. The 5%
result therefore should not be interpreted as 625 independently strong
solutions; it mainly describes a broad threshold relative to a flat endpoint.

## Mechanical component decomposition

The unchanged inverse-dynamics load was reconstructed as:

`M(q)qdd + gravity + Coriolis + required passive stiffness + required damping + required coupling + required joint-limit constraint`.

All requested terms are strictly available in the current model. The extra
joint-limit term is required because trajectories touch the frozen ROM extrema
and MuJoCo can activate the declared joint limit there. Its maximum
instantaneous magnitude is only 0.0257 N m. The recomputed primary endpoint
matches all frozen landscape rows exactly; the maximum component reconstruction
error is `1.14e-13 N m`.

Across legs, component RMS range/reference is:

- inertia: 2.35--5.23%;
- gravity: 0.053--0.511%;
- Coriolis: 1.01--4.98%;
- passive stiffness: 0.109--2.072%;
- damping: 0.197--2.370%;
- coupling: 0.744--6.129% where coupling is present; it is exactly zero in the
  nominal leg.

The component responses therefore contain more beta sensitivity than the final
aggregate scalar. Their vector sum can cancel, and their RMS values cannot be
added as scalars.

## Q4. Does the aggregate hide different hip and knee ordering?

Yes.

- Legs 0--3: hip-vs-knee Spearman is -0.937, -0.991, -0.964, and -0.997.
  Hip-only RMS prefers `(+0.03,-0.03)`; knee-only RMS prefers
  `(-0.03,+0.03)`.
- Leg 4: hip-vs-knee Spearman is +0.998 and both prefer
  `(-0.03,+0.03)`.
- Hip-only range/reference is 0.036--1.086%, while knee-only is
  1.602--2.310%.

Although V3 does not directly change hip kinematics, hip torque changes through
two-link inertial, gravity, and structural coupling. In four legs the aggregate
suppresses an opposing hip response and follows the stronger knee ordering.

## Q5. Are flexion/extension differences hidden?

They are separated by construction and partially obscured by a single scalar,
but they do not create conflicting branch optima.

- Flexion RMS always prefers `beta_flex=-0.03` and is independent of
  beta_extend; its range/reference is 0.033--1.879%.
- Extension RMS always prefers `beta_extend=+0.03` and is independent of
  beta_flex; its range/reference is 0.042--2.586%.
- Flexion-vs-extension Spearman over the Cartesian grid is approximately zero
  (`7.09e-06`) because the two metrics vary along orthogonal beta axes.

The apparent exact flexion oracle `(-0.03,-0.03)` merely reflects deterministic
tie-breaking along its inactive beta_extend axis. The full-cycle optimum
combines the two branchwise directions into `(-0.03,+0.03)`; there is no
evidence that one branch reverses the other.

## Q6. Are time-local responses more discriminative than the final scalar?

Yes, especially for the knee-dominant and nonlinear-coupled systems.

| Leg | Largest total separation (N m @ s) | Largest passive stiffness+coupling separation (N m @ s) | Largest inertia separation (N m @ s) |
|---|---:|---:|---:|
| LEG_0 | 0.424 @ 18.020 | 1.422 @ 18.748 | 0.103 @ 18.748 |
| LEG_1 | 0.418 @ 18.020 | 0.928 @ 18.384 | 0.082 @ 18.748 |
| LEG_2 | 30.880 @ 18.436 | 30.747 @ 18.540 | 0.174 @ 18.748 |
| LEG_3 | 46.296 @ 18.176 | 47.492 @ 18.228 | 0.113 @ 18.748 |
| LEG_4 | 7.917 @ 18.592 | 8.926 @ 18.644 | 0.055 @ 18.748 |

The total spread remains above half of its maximum for only 7.5--14.2% of
samples. The most sensitive 10% of time samples contain 47.2--52.4% of the
integrated spread. Full-cycle RMS therefore materially dilutes short localized
differences.

## Q7. Are normalized gradient fields still the same across legs?

Yes at the first-order level:

- pairwise normalized gradient-field cosine similarity: 0.99322--0.99989;
- normalized landscape Spearman: 0.99198--0.99986;
- 100% of grid points on every leg have positive dJ/dbeta_flex and negative
  dJ/dbeta_extend, hence descent toward `(-0.03,+0.03)`;
- all five surfaces have the same boundary oracle and zero strict interior
  local minima.

Curvature is less universal: pairwise curvature-trace Spearman ranges from
-0.580 to 0.901, and boundary Hessians are not positive definite. The evidence
supports a common monotone descent field, not an identical second-order
surface.

The shared direction is explained mechanically: delaying knee progression on
flexion (`beta_flex<0`) and advancing it on extension (`beta_extend>0`) reduce
exposure to the high-knee-angle passive-load region. Every frozen leg has
non-negative knee stiffness and damping, and the knee-only and branchwise
orderings are therefore shared. Opposing hip responses in legs 0--3 are too
small to reverse the aggregate.

## Q8. What is the primary source of the missing personalization signal?

`MECHANICAL_ENDPOINT_INSUFFICIENTLY_DISCRIMINATIVE`.

- H-A is not primary: the V3 family is narrow and knee-only, but it produces
  clearly measurable path, derivative, and timing differences.
- H-B is supported: joint, component, branch, and time-local responses retain
  differences that the full-cycle two-joint vector RMS compresses.
- H-C is a real secondary observation: normalized first-order fields are nearly
  universal. It is not selected as primary because the current endpoint's
  dynamic range is not reasonable across all five legs.

## Q9. What is the next scientifically justified direction?

`REVISIT_MECHANICAL_ENDPOINT`.

This means a separate future study should test mechanistically justified
joint-resolved, branch-resolved, or time-local summaries and their robustness;
it does not authorize selecting whichever metric creates oracle diversity.
Real measured subject data remains necessary before claiming that any revised
endpoint represents subject preference, comfort, benefit, or clinical need.
No endpoint is changed in this stage.

## Q10. Is another personalization algorithm supported now?

No. An algorithm cannot recover subject-specific information that the primary
objective does not contain. Secondary diagnostics do produce different exact
oracles (notably hip-only and peak metrics), but peak ranges are essentially
zero in several legs and branch metrics contain inactive-axis ties. These are
diagnostic clues, not a validated replacement target. Developing another BO,
failover, prior, or arbitration method now would optimize the same insufficient
signal.

## Outputs and validation

- Machine-readable summary: `lower_limb_sim/five_leg_mujoco_v1/results_discriminability_audit_v1/audit_summary.json`
- Full per-candidate diagnostic table and 11 focused tables: same result directory
- Figure 1: selected joint-space paths
- Figure 2: q/dq/ddq spread versus time
- Figure 3: normalized primary landscapes
- Figure 4: component sensitivity across beta
- Figure 5: flexion/extension and hip/knee response
- Targeted test: `6 passed`
- Offline diagnostic replays: `3125`; algorithm runs: `0`; robot actions: `0`;
  PINN training: `0`

V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1 = COMPLETE

PRIMARY_LIMITATION = MECHANICAL_ENDPOINT_INSUFFICIENTLY_DISCRIMINATIVE

NEXT_SCIENTIFIC_DIRECTION = REVISIT_MECHANICAL_ENDPOINT
