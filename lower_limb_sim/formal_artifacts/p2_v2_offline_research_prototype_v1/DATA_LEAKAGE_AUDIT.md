# Data Leakage Audit

- Pair set comes exclusively from `DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1` with frozen SHA `ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55`.
- Case assignment uses only protocol ID, pair ID, SHA-256 ordering, and the nine pre-existing case IDs. Prediction, truth, objective values, subject outcomes, and final truth landscape are not assignment inputs.
- Each pair prediction is computed before its fresh offline virtual-truth outcome is attached. Truth does not reselect a pair or case.
- Local max/P95/P99 are evaluated only in a historical shadow table; no statistic is frozen or exposed to P2 V1.
- knee_stiff direction and sequence lengths are predeclared research candidates. Truth is a post-hoc outcome label and cannot extend or reselect a sequence.
- K=1/2/3 use model/prediction/decision history fields only. Support is reported separately; truth is not a stopping feature.
- No formal personalization, held-out release claim, human threshold, robot connection, motion approval, or safety modification was created.
