# ROM protocol migration audit

Audit baseline: branch `paper`, commit `18d0cb7`, before this migration.
Scope: tracked Python/JSON/YAML/Markdown/TeX/config text plus persisted ROM
metadata. Ordinary numeric samples equal to 130 in raw or generated CSV files
are not ROM references and were intentionally excluded.

## Decision

The only formal active protocol is now:

```text
ROM_PROTOCOL_V2
hip_deg  = [0.0, 120.0]
knee_deg = [5.0, 145.0]
theta_shank = q_hip - q_knee
```

`config/formal_experiment_manifest.json` is the single editable protocol
source. `lower_limb_sim/formal_protocol.py` validates and exposes it to the
runtime. The old public `hip_range_deg` / `knee_range_deg` names remain only as
compatibility aliases to the same immutable values.

## Baseline scan and classification

The baseline tracked scan found 36 standalone `130` tokens in 15 files. Two
were hip-angle negative test values and were not knee-ROM uses. The remaining
34 knee-ROM-related references classify as follows:

| Class | Count | Treatment |
|---|---:|---|
| `ACTIVE_MUST_MIGRATE` | 23 | Migrated to V2 or to the shared runtime source. |
| `LEGACY_RESULT_KEEP` | 10 | Kept only in explicit legacy mapping/mismatch tests. |
| `HISTORICAL_TEXT` | 1 | Retained with explicit historical/legacy wording. |
| `SAFETY_REQUIRES_REVIEW` | 0 | No physical safety threshold was inferred or changed. |
| `UNKNOWN` | 0 | Every semantic hit was resolved. |

The post-migration tracked-text scan contains 24 standalone `130` tokens in
13 files, but zero production-Python numeric `130` constants and zero active
ROM gates. Those 24 lines are: 13 explicit legacy regression/test lines, eight
historical/provenance text lines, two hip-angle negative-test lines, and one
V2 result line reporting the newly admitted knee-above-130 region.

Three additional ignored persisted metadata files contain the old upper bound;
they are counted as legacy *files* rather than added to the tracked line count:
`data/reference_trajectories/processed/metadata.json`,
`data/reference_trajectories/retimed/metadata.json`, and
`data/reference_candidates/metadata.json`.

### ACTIVE_MUST_MIGRATE

- `lower_limb_sim/config.py`: the actual active 5--130 definition.
- `lower_limb_sim/test_full_dynamics.py` and `test_mismatch_scenarios.py`:
  configured-range sampling now reads V2 rather than embedding 130.
- `lower_limb_sim/test_reference_candidate_evaluation.py` and
  `test_reference_trajectory_retiming.py`: active assertions now expect 145;
  the Stage-5C public runner now defaults to and only accepts the manifest ROM.
- `lower_limb_sim/README.md`, `PROJECT_AUDIT.md`, and paper notes: formal method,
  result, TODO and conflict descriptions now state V2.
- Workspace, IK, reference, candidate, identification, preview/preflight and
  real-episode identification import the same formal values.

### LEGACY_RESULT_KEEP

- Low-level affine-mapping tests retain 5--130 inputs to verify historical
  behavior without making them active.
- Robot-export tests retain 5--130 calibration/conflict inputs to prove that a
  legacy calibration cannot pass a V2 source ROM match.
- Reversed or invalid pairs retain 130 to test input validation.
- `lower_limb_sim/data/reference_trajectories/{processed,retimed}/metadata.json`
  and `lower_limb_sim/data/reference_candidates/metadata.json` are immutable
  pre-migration Stage-5 artifacts. They remain ignored legacy data and are not
  default active loaders.
- `lower_limb_sim/data/workspace/` and `data/force_maps/` retain the old outputs
  without overwrite. Their replacement defaults live under
  `lower_limb_sim/formal_artifacts/rom_protocol_v2/`.

### HISTORICAL_TEXT

- `lower_limb_sim/dynamic_force_audit.md` labels its old numerical basis as
  pre-migration historical evidence.
- Architecture/paper audit notes mention 5--130 only to explain superseded
  provenance, never as a formal range.

### Non-ROM numeric 130

`test_reference_trajectory_import.py` uses hip=130 as an intentionally invalid
hip-angle test. Raw skeleton CSV and simulation CSV values containing the
digits `130` are measurements, frame indices or calculated values, not gates.

## Regenerated V2 artifacts

All directly ROM-dependent formal outputs were regenerated without overwriting
legacy data:

- `lower_limb_sim/formal_artifacts/rom_protocol_v2/workspace/`
  - `workspace_atlas.csv` / `.npy`
  - `workspace_metadata.json`
  - hip/knee workspace figures and sample-posture figure
- `lower_limb_sim/formal_artifacts/rom_protocol_v2/force_maps/`
  - four subject CSV/NPZ maps
  - six figures per virtual subject
  - aggregate comparison CSV and figure
- `rom_migration_summary.json` records protocol, active SHA, legacy boundary,
  Stage-3/4/4.5 dependency review and the active-reference audit.

The V2 atlas has 17,061 grid points and 11,993 geometrically reachable points.
The knee-above-130 region contains 1,815 points; 1,127 are geometrically
reachable and all 1,815 pass the existing Jacobian numerical gate. No Jacobian
threshold or force anomaly threshold was changed.

Stage 3/4/4.5 configured trajectories top out at knee 120 degrees, so their
stored numerical results do not change solely because the formal ceiling moved
from 130 to 145. Their old metadata remains historical; every future formal
regeneration reads and records V2.

## Active reference compatibility

The active file was not regenerated, mapped, clipped or edited:

```text
id     = reference_measured_asymmetric_closed_slow
cycle  = 5844 -> 5895 -> 5934
sha256 = f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881
```

For all 401 samples under V2:

- hip range: 28.909337--112.025439 deg;
- knee range: 18.319125--124.786604 deg;
- ROM valid: true;
- workspace valid (`x_pull>=0`, `z_pull>=0`): true;
- Jacobian valid: true;
- maximum Jacobian condition number: 14.973214;
- `max(abs(theta_shank - (q_hip-q_knee)))`: `4.996e-16 rad`.

Nominal remains offline/fail-closed for its independent local-domain coverage
reason. ROM migration does not change that decision.

## Safety and hardware boundary

No hardware source, SDK API, real-robot safety JSON, force/torque threshold,
workspace safety box, collision setting, speed/acceleration limit, soft limit,
payload or realtime setting was changed. `ROM_PROTOCOL_V2` is an offline/model
and trajectory-validity protocol; it does not approve real robot motion.

## Verification

- Full offline test suite: 658 passed, 5 skipped, 0 failed in 101.91 s.
- Pytest emitted no warning summary.
- `git diff --check`: passed.
- Active reference SHA-256: unchanged and equal to the pinned manifest value.
- Hardware/safety diff: empty.
