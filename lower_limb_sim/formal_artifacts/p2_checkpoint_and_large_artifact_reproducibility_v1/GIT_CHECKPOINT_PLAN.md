# Selective Git Checkpoint Plan

This file is instructions only. The audit runner does not stage or commit anything. Review each pathspec file before use; it contains one explicit repository-relative path per line and no wildcard.

Run from repository root, in order:

## Checkpoint 1

```bash
sed -n '1,240p' lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_1_files.txt
git add --pathspec-from-file=lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_1_files.txt
git diff --cached --stat
git diff --cached --name-status
git status --short
git commit -m "implement decision-guarded sequential personalization and convergence audit"
```

## Checkpoint 2

```bash
sed -n '1,240p' lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_2_files.txt
git add --pathspec-from-file=lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_2_files.txt
git diff --cached --stat
git diff --cached --name-status
git status --short
git commit -m "add P2 revision root-cause and design evidence"
```

## Checkpoint 3

```bash
sed -n '1,240p' lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_3_files.txt
git add --pathspec-from-file=lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_3_files.txt
git diff --cached --stat
git diff --cached --name-status
git status --short
git commit -m "freeze P2 V2 offline research protocol"
```

## Checkpoint 4

```bash
sed -n '1,240p' lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_4_files.txt
git add --pathspec-from-file=lower_limb_sim/formal_artifacts/p2_checkpoint_and_large_artifact_reproducibility_v1/checkpoint_4_files.txt
git diff --cached --stat
git diff --cached --name-status
git status --short
git commit -m "evaluate P2 V2 offline research prototype"
```

The five manifest-managed large CSVs and `.DS_Store` do not appear in any pathspec. Do not use `git add .` or a directory-wide add. If any reviewed staged set differs from the pathspec, stop and resolve it before commit.

Only after all four commits and a fresh status/test review may the separately defined prospective-validation task be reconsidered; this plan does not start it.
