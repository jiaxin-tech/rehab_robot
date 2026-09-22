# Adaptive Trust V1 Offline Benchmark

This directory contains deterministic, pure-offline algorithm-development
evidence for the frozen
`PRIMARY_CAUSAL_OFFSET_INVARIANT_RANKING_TRUST_V1` rule.

```text
ADAPTIVE_PHYSICS_PRIOR_TRUST_PERSONALIZATION_V1_IMPLEMENTED_WITH_LIMITATIONS
READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK = false
```

The benchmark has 240 paired conditions and 720 method runs at K=4. It uses the
existing frozen P0-P3 analytic cases, zero/low/moderate noise, seeds 0...4, and
one identical frozen ROM fingerprint per comparison. It performs no robot or
human actions, does not train a PINN, and is not physical, safety, comfort, or
clinical validation.

`benchmark_summary.json` is the complete machine-readable record.
`run_metrics.csv`, `aggregate_metrics.csv`, `paired_comparisons.csv`,
`negative_transfer_summary.csv`, and `trust_trajectories.csv` provide tabular
views. `adversarial_trust_audit.json` records the frozen semantic cases. The
three PNG files are the only requested visualizations. Verify the formal
artifacts with:

```bash
shasum -a 256 -c checksums.sha256
```

The frozen rule reduced mean P3 negative transfer in zero and low noise but
worsened it in moderate noise. Adaptive trust was worse than both baselines in
17 of 240 paired conditions. These limitations are why the readiness flag is
false; they were not used to retune the primary trust rule.
