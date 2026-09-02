# MyoLeg V3 Trajectory Parameterization Design V1

## Formal result

`MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_VALID_WITH_LIMITATIONS`

This is an **offline, default-off, kinematic design result**. It is not a human result, robot-motion approval, clinical finding, safety validation, or personalized-trajectory outcome. No development or held-out scientific truth, mechanical objective, subject model, Five-parameter model, learner, PINN, NN, BO, hardware, control, collection, or safety code was used.

## Frozen semantics

The two coordinates are `beta_flex` and `beta_extend`. For each measured branch, the normalized phase is transformed by

`w(s; beta) = s + beta * 64 s^3 (1-s)^3`.

Positive beta advances knee progression along that measured branch relative to the frozen hip/reference phase; negative beta delays it. The hip trajectory is copied exactly. The basis and its first two derivatives are zero at both branch endpoints, so the map returns to identity through second derivative. Pointwise clipping is absent.

## Reference recovery and task invariance

At `[0, 0]`, q, dq and ddq are array-exact copies of the frozen V2 reference. The cycle remains 24 s and 401 samples. Across all 625 candidates, the frozen hip/knee extrema and ROM are preserved within 0.001 deg; hip q/dq/ddq remain exactly unchanged; branch anchors, cycle closure and C2 endpoint conditions pass. The parameterization changes only the interior hip-knee path and timing relationship.

## Range and grid

The preregistered outcome-free axis sweep covered [-0.25, +0.25] at step 0.0025. The largest origin-connected symmetric interval satisfying all frozen kinematic gates is [-0.030, +0.030]. The first failed positive axis value was {'flex': 0.045, 'extend': 0.0325}; the table records every failure reason. These bounds are structural research bounds, not human or robot safety thresholds.

Coarse, medium and fine grids were compared before candidate-domain freeze. The selected `FINE` grid has step 0.0025, 25 values per axis, and 625 total candidates. Stable IDs follow beta-flex outer ascending and beta-extend inner ascending order.

## Actual interior variation

Because hip is exact, matched-phase joint-space displacement from reference is entirely the knee displacement. Across the domain, maximum displacement spans 0.000000--4.704586 deg and time-weighted RMS displacement spans 0.000000--1.393700 deg. Joint-space path length spans 265.980358--269.030561 deg, signed path area spans 460.852218--951.836592 deg2, flexion knee-midpoint time spans 4.493186--4.660852 s, and extension knee-midpoint time spans 18.948749--19.268771 s. Thus the domain produces nonzero interior coordination/path variation without amplitude variation.

## Nominal MyoLeg smoke

After manifest freeze, 13 geometry-selected candidates (reference, corners, axes and interior points) were replayed once on the unmodified nominal MyoLeg model. All passed the frozen simulator-artifact integrity screen. Maximum joint-limit knee contribution was 0.00262078 N m, maximum relative contribution 0.000200074, maximum source equality residual 2.66454e-15, maximum algebraic residual 1.06581e-14 N m, and maximum solver warnings 0. No trajectory was ranked and no objective was evaluated.

## P4 versus P2

P4 remains the primary structure because two branch-specific coefficients provide direct, reversible semantics, disjoint branch support, exact hip preservation and low dimensionality. P2 remains a plausible fallback if later development evidence shows that one scalar warp mode per branch is too restrictive; it would require a new knot/basis/range freeze and a larger four-dimensional domain. That fallback was not implemented here.

## Limitations and next stage

The result is `VALID_WITH_LIMITATIONS` because simulator coverage is deliberately sparse and nominal-only, and because no development truth landscape has been generated. It establishes a valid candidate space, not personalization benefit or subject specificity. The only recommended next stage is `MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_V1`, using the frozen manifest SHA. That stage was not run here. Held-out scientific access remained zero.
