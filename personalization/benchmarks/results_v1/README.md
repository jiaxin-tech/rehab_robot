# Physics-Informed Sequential Personalization V1 — Offline Benchmark

Status: `OFFLINE_ALGORITHM_EVIDENCE`

This directory contains 1,440 primary K=4 analytic development runs over four
landscapes, four prior-quality settings, three development-noise settings, five
deterministic seeds, and six equal-budget methods. It also contains K=3/K=5
sensitivity summaries. These files are not real-subject, clinical, comfort,
robot, or MyoLeg held-out scientific evidence.

- `benchmark_summary.json`: complete metrics, aggregates, sensitivity, and
  win/tie/loss results.
- `run_metrics.csv`: one row per K=4 method run.
- `aggregate_metrics.csv`: mean/median/P95 aggregation.
- `budget_sensitivity.csv`: K=3/K=5 aggregation.
- `prior_pairwise_win_tie_loss.csv`: overall physics-BO comparisons.
- Figures 1–4: landscape/samples, regret by trial, prior-residual correction,
  and prior-quality stress test.

The P3 result is intentionally retained: a poor prior hurts physics-informed
BO under K=4 relative to standard BO.

Stage-specific pytest: `18 passed`. Repository-wide pytest on the merged input
tree: `1466 passed, 1 skipped, 269 failed, 183 errors`; the earliest failure is
an existing frozen-protocol SHA mismatch outside this stage. No frozen artifact
was rewritten to conceal that baseline condition.
