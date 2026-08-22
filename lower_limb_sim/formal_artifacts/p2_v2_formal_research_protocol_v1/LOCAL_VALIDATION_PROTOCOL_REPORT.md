# Local Validation Protocol Report

## Protocol

`DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1` defines an independent, pre-registered local validation plan. Candidate pairs are drawn only from the existing geometry-valid generator lattice. A pair changes exactly one of hip/knee/phase by one existing initial, half, or minimum trust step; clipping and bounds expansion are prohibited.

The complete geometry/trust pair universe contains 173188 pairs. The pilot plan contains 324 pairs: coordinate × trust-level × lower/interior/upper strata, with 12 deterministic SHA-selected pairs per stratum. Pilot sample count still requires power/reviewer approval and is not a decision threshold.

Pair plan SHA-256:

```text
ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55
```

This file must be frozen before predictions or independent truth outcomes are evaluated. The final truth landscape is forbidden as a selection source. Current `predicted_delta_J`, `truth_delta_J`, and `e_delta_J` fields are intentionally blank.

## Global vs designated local

| class | pairs | alpha mappable | trust levels | outcomes | P95 | max |
|---|---:|---|---|---|---:|---:|
| current global identification validation | 61 | no | none | available | 0.00753215325 | 0.216052983 |
| designated local plan | 324 | yes | initial/half/minimum | pending | pending | pending |

The current global pair measures differences between identification excitations and cannot be mapped to generator alpha or a trust step. The designated plan measures exactly the future local decision relationship, but it has no result yet. Therefore it should become a mandatory research evidence layer for a future P2 V2, while no local uncertainty statistic or threshold can be selected in this task.

## Required outcome attachment

After plan freeze, a predeclared model checkpoint records predicted ΔJ and a newly generated independent designated offline evaluation records truth ΔJ. Outcomes must match the frozen pair-ID set exactly; error is `abs(predicted_delta_J - truth_delta_J)`. The data cannot be used for fitting, adaptation update, held-out final test, P2 V1, or an unreviewed P2 V2 guard.
