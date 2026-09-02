# MyoLeg V3 Development Truth Landscape Generation V1

## Decision

`MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_VALID`

This is a hidden **development-only offline virtual truth landscape**. It is not a human result, robot-motion approval, clinical result, learner training set, personalization conclusion, or safety validation.

## Frozen identity and coverage

- V3 candidate manifest SHA-256: `6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745`.
- Cohort manifest SHA-256: `31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057`.
- Truth semantic: `MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1` / `TAU_MY0LEG_REQUIRED_DRIVE`.
- Development subjects: `24`; candidates: `625`.
- Evaluated pairs: `15,000` / `15,000`.
- Duplicate pairs: `0`; missing pairs: `0`.
- All 24 reference objectives are within `1e-12` of 1: `True`.

## Compact storage and runtime

The frozen dataset contains `24` deterministic compressed NPZ subject shards. It stores only one-dimensional compact scalar columns and occupies `501,833` bytes plus `2,016` bytes of checksum sidecars. No bulk 401-point replay arrays were stored; full prescribed arrays remain reproducible through the development-only on-demand replay API.

The on-demand API was invoked twice for the frozen development reference pair and returned array-identical full payloads; a held-out ID was rejected before simulator replay.

- Formal workers: `4`.
- Formal generation wall time: `358.195` s.
- Throughput: `41.877` pairs/s.

## Integrity validation

- All-pair compact integrity: `True`.
- All-pair task invariants: `True`.
- Detailed repeated prescribed replays: `36` / pass `True`.
- Controlled diagnostic: `2` pairs and `4` joint rows / pass `True`.

Prescribed replay remains primary truth. Controlled replay was used only for the two frozen consistency diagnostics and did not replace or alter any compact truth value.

## Information boundary

Held-out scientific truth access was exactly `0`. No held-out replay, J, torque, oracle, rank, or candidate preference was read. No per-subject minimum, candidate ranking, common optimum, distinct-oracle count, Top-K overlap, regret, cross-transfer, or V2/V3 performance comparison was computed. `ORACLE_NOT_REVEALED_DURING_GENERATION_STAGE = true`.

## Boundaries and next step

No Five-parameter model, NN/PINN, BO, cohort/domain/objective modification, hardware/control/collection/safety change, or personalization interpretation occurred. The only allowed next stage is `MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_V1`, under a separately frozen protocol; it was not executed here.
