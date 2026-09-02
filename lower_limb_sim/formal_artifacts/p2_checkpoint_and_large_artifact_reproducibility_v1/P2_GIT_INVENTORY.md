# P2 Git Inventory

## Snapshot

- Branch: `sync/mac-20260813`
- HEAD: `0ae022c0f8a3cd8b154de83f0606c30ceb0850cf`
- Git-tracked `lower_limb_sim` paths at HEAD/worktree: 309.
- Currently visible untracked `lower_limb_sim` paths: 187.
- Reconstructed task-start P2 paths: 172 (must equal 172).
- Generated data-like files larger than 1,000,000 bytes: 660.
- `.DS_Store` and the five registered large truth CSVs are ignored by exact rules; their on-disk files were retained.

## Reconstructed ownership of the 172 task-start P2 paths

| Checkpoint | Scientific scope | Source/runner/test + artifact paths | Git candidates after excluding manifest-managed large CSVs |
|---|---|---:|---:|
| 1 | P2 V1 research foundation | 55 | 55 |
| 2 | Revision root cause, V2 design, V2 research prototype | 79 | 75 |
| 3 | Frozen research protocol and 324-pair plan | 21 | 21 |
| 4 | Default-off offline prototype evaluation | 17 | 17 |

The four 3.2 MB truth landscapes remain scientific outputs owned by Checkpoint 2, but are deliberately absent from its Git pathspec because they are content-addressed regenerated artifacts. The 132 MB global comparison predates these four P2 checkpoints and is likewise manifest-managed, not silently reassigned.

## Classification

- Scientific source, runner, and test files are listed exactly once in the four checkpoint manifests.
- Small formal artifacts remain ordinary Git candidates in their scientific stage.
- Large generated truth outputs are recorded in `GENERATED_LARGE_ARTIFACT_MANIFEST.json` and `large_generated_artifact_inventory.csv`.
- Most ignored `lower_limb_sim/data/**` belongs to earlier simulation stages and is not reassigned to P2.
- Four small previously ignored frozen prerequisites are packaged in Checkpoint 2 solely so the tracked active-reference loader and generator work after clone; they retain their earlier scientific provenance and are not counted among the 172 P2 paths.
- No file was moved or deleted to manufacture a clean status.

## Final checkpoint path counts (including this task's infrastructure)

- Checkpoint 1: 55 paths.
- Checkpoint 2: 96 paths.
- Checkpoint 3: 21 paths.
- Checkpoint 4: 17 paths.
