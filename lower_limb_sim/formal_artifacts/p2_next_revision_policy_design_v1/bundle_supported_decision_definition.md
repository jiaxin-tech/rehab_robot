# Bundle-supported one-step decision definition

`BUNDLE_SUPPORTED_ONE_STEP_COMMITMENT_V1` evaluates only straight, same-axis, same-direction 2/3/5-step formal-grid paths. Every start, intermediate, first-step, and endpoint node must exist in the unchanged generator lattice, retain the active-reference provenance, satisfy the synthetic patient-envelope fixture, and be model-supported at the unchanged 90% gate.

The research evidence margin is `-deltaJ_pred(start, endpoint) - U_bundle(scale[,axis]) - 0.005`. The 0.005 term remains the meaningful-improvement tolerance; it is not redefined as a safety threshold. `U_bundle` comes directly from independent 2/3/5-step residual distributions. Neither `n*U1` nor `sqrt(n)*U1` is used.

An eligible endpoint authorizes exactly the next adjacent formal-grid step. It never queues the remaining endpoint path. Immediately after that one simulated trajectory, the five-parameter model is refit once, the entire prediction map is recomputed, and the authorization expires. A new round must independently re-evaluate all bundle evidence.

This is default-off synthetic offline research. It is not a human or robot execution rule.
