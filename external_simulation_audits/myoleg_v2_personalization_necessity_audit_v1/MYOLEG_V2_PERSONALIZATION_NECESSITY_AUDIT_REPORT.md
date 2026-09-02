# MyoLeg V2 Personalization Necessity Audit V1

## Decision

`PERSONALIZATION_NECESSITY_NOT_SUPPORTED`

This is a development-only offline oracle upper-bound audit under the frozen normalized torque objective. It is not achieved algorithm performance, a patient optimum, human evidence, clinical benefit, comfort, or safety evidence. All eight confirmatory subjects remained sealed.

## Development oracle geometry

- Distinct exact oracle candidates: **1 / 24**.
- Unique hip / knee / phase oracle values: **1 / 1 / 1**.
- Any candidate-domain boundary fraction: **1.000**.
- Hip lower / upper edge fractions: **0.000 / 1.000**.
- Knee lower / upper edge fractions: **0.000 / 1.000**.
- Phase edge fraction: **1.000**.
- Clearly separated oracle-pair fraction: **0.000**.

Different exact candidate IDs are not by themselves evidence of useful personalization. The pair classifications and near-oracle plateaus below determine whether the differences exceed grid/tie effects.

## Ranking and top sets

- Pairwise Spearman: median **0.999834**, mean **0.999771**, range **[0.998646, 0.999998]**.
- Kendall tau-b: median **0.989722**.
- Top-5% Jaccard: median **0.964664**, P5/P95 **0.915040/0.995215**.

## Common trajectory versus subject oracle

- DEV_MEAN_OPTIMAL_COMMON: `MYOLEG_V2_P20850`, alpha `[2.0, 0.5, -0.03]`.
- Common J mean / median / worst: **0.994487566 / 0.994457290 / 0.994840871**.
- Relative common regret median / mean / P75 / P95 / max: **0.000000% / 0.000000% / 0.000000% / 0.000000% / 0.000000%**.

These gaps are oracle upper bounds on potential subject-specific mechanical-objective benefit, not achieved few-trial improvement.

## Near-oracle plateau and transfer

- At epsilon=0.001, maximum common coverage: **24 / 24**.
- Universal near-oracle candidate exists: **True**.
- Near-oracle count at epsilon=0.001, median / P95 / max: **43 / 48 / 48**.
- Broad plateau rule triggered: **False**.
- Foreign-oracle regret median / P95 / max: **0.000000000 / 0.000000000 / 0.000000000** J.

## Parameter associations

Thirty preregistered descriptive Spearman tests were run across six frozen parameters and five oracle/gap outcomes. Associations with BH q<0.05: **1**. These are exploratory associations only; no predictive model was trained and no cohort or range was changed.

## Interpretation boundary

The decision follows the protocol frozen before development truth reveal. Held-out shards were only stream-hashed and row-count checked; no held-out NPZ array, oracle row, J value, ranking, figure, or statistic was read. No Five-parameter model, NN/PINN, BO, candidate-domain update, objective update, cohort update, robot, or hardware operation occurred.
