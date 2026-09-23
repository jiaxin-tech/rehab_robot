# Code Cleanup Report

Cleanup date: 2026-08-11 (Asia/Shanghai)

This report records the real-experiment consolidation performed after `PROJECT_AUDIT.md`. It distinguishes repository edits from pre-existing untracked/ignored user data.

## 1. Checkpoint and data boundary

- Branch: `simulation`
- Starting/current HEAD: `8e0c3de507e0f08ee35818c83c259c227add7d36`
- Lightweight rollback tag: `pre_real_robot_experiment_cleanup`
- No new commit was created.

The tag points to the committed HEAD only. It does not protect the dirty working tree, the untracked paper package or ignored experiment artifacts.

Using the same count boundary as the initial audit (excluding `.git`, `.venv`, Python caches and pytest cache), the tree changed from 3,284 files to 3,314 files: nine tracked legacy source files were deleted and 39 non-`lower_limb_sim` refactor files were added. The final worktree has 11 modified tracked paths, nine deleted tracked paths, 39 new refactor paths, and nothing staged. Those new paths are still untracked and are not protected by the checkpoint tag.

Pre-existing user work preserved:

| Path | Initial state | Cleanup action |
|---|---|---|
| `requirements.txt` | already modified before this task | preserved the pre-existing dependency edit in full; this cleanup made no additional requirements edit |
| `bone_return_3_leg.csv` | untracked | retained |
| `lower_limb_sim/` | untracked current paper code | retained; only integration docs/two obsolete rollout-local worktree assertions were updated |
| `lower_limb_sim/data/` | ignored, about 5.3 GB / 2,958 files | not deleted or moved |
| `logs/` | ignored, 104 historical files at audit | all 104 pre-existing files retained; 10 transient files created by an early audit import side effect were removed, restoring the initial count |

Because `lower_limb_sim/data/` and `logs/` are ignored, neither the tag nor a normal future commit backs them up. They require an independent data-backup decision.

## 2. Deleted legacy source

Nine tracked files (2,013 source lines at HEAD) were removed as one dependency-consistent old feature island:

| Deleted path | Reason |
|---|---|
| `models/pinn.py` | old physics-informed model, outside frozen paper scope |
| `models/comfort_net.py` | old comfort classifier, outside frozen scope |
| `models/__init__.py` | empty package after model removal |
| `control/mpc_controller.py` | old MPC controller, outside frozen scope |
| `scripts/train_pinn.py` | CLI for deleted model |
| `scripts/train_comfort.py` | CLI for deleted model |
| `scripts/run_control.py` | old excitation + model + MPC runtime; no explicit safe execute gate |
| `scripts/run_collection.py` | old entry automatically enabled/moved before the new frame/anchor/safety gate |
| `hardware/windows/rokae_force_sensor.py` | deprecated/misleading compatibility alias; no external sensor is used |

Eight ignored stale `.pyc` files corresponding to deleted functional model/train/runtime/alias source were also deleted after exact enumeration. One later-discovered orphan cache for the deleted empty `models/__init__.py` package and an old `.pytest_cache` remain because the final destructive-cache command could not obtain environment approval; neither is active source/import/test evidence. Five additional current-module caches produced by an earlier syntax check were removed as generated artifacts. All of these caches contain no experiment data and can be regenerated or safely removed later.

## 3. Surgically retained and decoupled code

- `collection/collector.py`: retained generic background collection and provenance; removed comfort/pain label schema and episode labelling.
- `collection/trajectory.py`: retained generic trajectory geometry/tangent projection; removed old sinusoid, sweep, drag/circle calibration generators.
- `tests/test_units_and_frames.py`: retained state, wrench, snapshot, collector and safety tests; removed deleted-model/MPC loader tests.
- `config/settings.py`: retained SDK/acquisition settings; removed old excitation, tactile, model and controller settings.
- `utils/logger.py`: removed import-time file creation; logging is console-only until explicitly configured.
- `hardware/windows/rokae_xcore.py`: retained real SDK calls; made connect receive-only with respect to project motion APIs, exposed state stream lifecycle, split state/wrench locks and removed blocking SDK error polling from target replacement.
- `lower_limb_sim/run_robot_trajectory_export.py` and the existing `absolute_calibrated` mode were retained unchanged.
- Completed Stage 1–6 offline studies, audits and data were retained as evidence, not advertised as the active runtime.

Two lower-limb tests previously asserted the entire working tree was Git-clean at a historical rollout boundary. Those assertions became invalid once the user deliberately added the next stage in the same untracked tree. They were replaced with durable AST checks that the completed offline modules do not import the hardware/runtime stack. Numerical tolerances and scientific assertions were not relaxed.

## 4. Added real-experiment surfaces

The consolidation added strict, testable boundaries rather than another simulation stage:

- start-anchored relative C2 trajectory and frame audit;
- strict StartAnchor JSON/capture;
- reviewed/null safety schema v3, including explicit RT filter/network preparation evidence;
- narrow observation-only project adapter and separate motion adapter;
- connection/state/wrench health probe with explicit vendor-session side-effect disclosure;
- independent state/wrench/alignment acquisition;
- fail-closed five-file episode logger;
- shared Git provenance for episode, preview and real-identification metadata;
- pure offline preview and plots;
- offline/static and live execution preflight;
- pinned slow-reference SHA and `L1/L2` first-trial manifest;
- live-only trajectory/safety-bound, externally prepared, single-use RT Cartesian scheduler with repeated configuration/health/collision checks, race-free bounded durable command logging, deadline rejection and retryable strict unified stop;
- unified preview/acquire/execute runner;
- reviewed real-episode adapter for the unchanged five-parameter estimator;
- cross-platform fake tests and explicit hardware opt-in test.

At report time, 39 new non-`lower_limb_sim` files are present after adding this report: Python source/tests, JSON templates and four Markdown deliverables. The untracked status is inherited from the user's working tree; this report does not claim they are protected by Git.

## 5. Dependency result

Current `requirements.txt`:

```text
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
pytest>=7.0
pillow>=9.0
```

- The pre-existing requirements edit already omitted `torch`; the final import graph confirms that no retained runtime needs it.
- NumPy/SciPy remain required by mechanics, FK/dynamics, C2 and estimation.
- Pandas remains required by references, logging adapters and identification.
- Matplotlib/Pillow remain required by offline preview/visualization outputs.
- Pytest remains the repository regression runner.

No packaging console scripts exist; operator entry points use `python -m scripts.<module>`.

## 6. Static cleanup checks

Post-cleanup scans established:

- active Python imports of `models.pinn`, `models.comfort_net` and `control.mpc_controller`: zero;
- executable implementations/callers of the deleted legacy stack: zero;
- root README references to deleted commands: zero;
- remaining PINN/MPC strings under `lower_limb_sim` are negative `scope_excludes`/audit provenance only;
- `lower_limb_sim` does not import `hardware`, xCoreSDK, the motion executor or collection runtime;
- primary packages import on macOS without loading the Windows native extension;
- no C/C++ project source was added; vendor native binaries remain untouched.

Useful verification commands:

```bash
rg -n 'models\.(pinn|comfort_net)|control\.mpc_controller' --glob '*.py' .
rg -n 'setPowerState|setOperateMode|move_l|move_j|start_realtime' scripts hardware/rokae_adapter.py
python3 -B -m pytest --collect-only -q -p no:cacheprovider
python3 -B -m pytest -q
git diff --check
```

## 7. Regression history

| Gate | Result | Interpretation |
|---|---|---|
| Before deletion | `508 passed, 4 skipped in 98.53 s` | offline baseline only |
| After legacy cleanup | `506 passed, 4 skipped in 96.14 s` | two obsolete model/MPC tests removed; no scientific tolerance change |
| Relative trajectory + initial adapter | `524 passed, 4 skipped in 97.83 s` | new start-anchor and observation interfaces included |
| Pre-document final implementation run | `565 passed, 5 skipped in 100.93 s` | includes full new runner/logger/executor/identification surfaces |
| Cleanup final reviewed run | `622 passed, 5 skipped in 101.56 s` | 627 tests collected; no failures or warning summary |
| Measured-asymmetric reference final run | `640 passed, 5 skipped in 96.90 s` | 645 tests collected; no failures; no robot connection or motion |

All runs used offline/fake adapters on macOS. They did not load the Windows `.pyd`, connect to a robot or validate physical motion.

## 8. Intentionally unresolved items

- Vendor initialization/connect/disconnect side effects on the real controller. The wrapper now uses only `xMateRobot()` followed by `connectToRobot(remoteIP[, localIP])`, but physical semantics remain unverified.
- Actual state/wrench rates, query latency, skew and native thread safety.
- Tool/load, wrench compensation, frame direction, sign and reference point.
- RT Cartesian stability, tracking and stop latency.
- Runtime behavior and latency of the SDK motion-error/safety-event queries under a real RT session; no synchronous SDK health query is placed in the target-update hot path.
- Site-specific safety limits and start-anchor tolerances (templates remain null).
- Site-reviewed RT command-filter and externally prepared network-tolerance values (schema-v3 defaults remain null; the latter has no confirmed SDK readback).
- Subject-specific anthropometry and real identification mapping (template remains null).
- Whether `utils/signal_processing.py` has an external consumer; retained rather than guessed disposable.
- Independent backup/provenance policy for ignored 5.3 GB data and logs.
- Generated-cache cleanup: `models/__pycache__/__init__.cpython-313.pyc` and the stale `.pytest_cache/` remain after the environment rejected the exact deletion command; they are excluded from all audit/test counts.
