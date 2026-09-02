# Formal Artifact Reproduction

## NORMAL_REGRESSION

Normal regression checks algorithm invariants, small summaries, frozen SHA values, and manifest provenance. It does not require any of the five large CSVs:

```bash
python3 -B -m pytest -q -p no:cacheprovider \
  lower_limb_sim/test_decision_relevant_global_model_reliability.py \
  lower_limb_sim/test_p2_revision_root_cause_audit.py \
  lower_limb_sim/test_p2_revision_v2_research_prototype.py \
  lower_limb_sim/test_large_artifact_reproduction.py
```

If an optional local cache exists, its SHA and schema are verified. Absence is not a skip and does not change scientific expected values.

A clean checkout must include the four small frozen prerequisites explicitly listed in Checkpoint 2 (`reference_full_angles.csv`, `detected_cycles.csv`, its Stage 5A `metadata.json`, and `state_domain_bounds.json`). They are source/provenance inputs, not large generated truth caches.

## Representative deterministic verification

```bash
python3 -m lower_limb_sim.large_artifact_reproduction \
  --artifact truth_landscape_baseline \
  --output-dir /tmp/p2-large-reproduction-a \
  --representative-subset
python3 -m lower_limb_sim.large_artifact_reproduction \
  --artifact truth_landscape_baseline \
  --output-dir /tmp/p2-large-reproduction-b \
  --representative-subset
```

Compare the printed SHA-256 values. This uses the existing model/cache/truth evaluator and does not copy scientific equations.

## FORMAL_ARTIFACT_REPRODUCTION

The following commands regenerate complete formal outputs and verify row count, schema, and SHA against the tracked manifest. Use an empty/cache directory or the original stage directory only when the target file is absent; the wrapper refuses overwrite.

```bash
python3 -m lower_limb_sim.large_artifact_reproduction --artifact global_prediction_truth_comparison --output-dir <cache-dir> --verify-manifest
python3 -m lower_limb_sim.large_artifact_reproduction --artifact truth_landscape_baseline --output-dir <cache-dir> --verify-manifest
python3 -m lower_limb_sim.large_artifact_reproduction --artifact truth_landscape_hip_stiff --output-dir <cache-dir> --verify-manifest
python3 -m lower_limb_sim.large_artifact_reproduction --artifact truth_landscape_knee_stiff --output-dir <cache-dir> --verify-manifest
python3 -m lower_limb_sim.large_artifact_reproduction --artifact truth_landscape_heavy_leg --output-dir <cache-dir> --verify-manifest
```

To regenerate the downstream V2 research prototype from a fresh checkout, first place the verified knee artifact at its manifest `expected_path`, then run `python3 -m lower_limb_sim.run_p2_revision_v2_research_prototype --output-directory <empty-dir>`.

The 132 MB full regeneration is intentionally separate from normal pytest. The final task report states whether it was actually executed; representative verification must never be presented as a full SHA reproduction.

## Verification performed for this checkpoint task (2026-08-23)

- Two independent representative baseline regenerations were byte-deterministic.
- Two independent 900-row/nine-case representative global-comparison regenerations were byte-deterministic at SHA-256 `194e2cf0397839c0cb8c3155e833d1686704fc7e1e9fa0ccb4017839cd859561`.
- One complete 21,025-row baseline landscape regeneration matched manifest SHA-256 `2cc2519ee04a3804f17cf81e30c2350b27adfb6ca07ead650782572e4a322ba0`.
- A temporary clean-room copy containing the frozen small prerequisites but none of the five large CSVs passed the selected normal P2 regression (`152 passed`).
- `global_prediction_truth_comparison.csv` full 132 MB SHA regeneration was **not executed** in this task: `FULL_132MB_SHA_REGENERATION_NOT_EXECUTED`. Its existing formal runner and representative mode remain verified separately; do not describe this as full reproduction.
