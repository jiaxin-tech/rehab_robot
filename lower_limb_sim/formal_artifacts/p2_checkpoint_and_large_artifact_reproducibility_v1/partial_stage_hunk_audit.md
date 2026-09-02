# Partial-stage hunk audit

No scientific source, runner, test, or formal artifact contains independent hunks belonging to more than one of the four checkpoints. Therefore there are no `PARTIAL_STAGE_FILE` entries.

Two pre-existing tracked files receive Checkpoint 2 reproducibility-only edits:

- `.gitignore`: one independent exact-rule block for `.DS_Store` and five generated large CSVs.
- `lower_limb_sim/test_decision_relevant_global_model_reliability.py`: normal-regression assertions now use the content-addressed manifest instead of the 132 MB local file.

Each is wholly assigned to Checkpoint 2 for this worktree change; neither changes scientific policy. The Checkpoint 2 root-cause and prototype tests are new/untracked stage files and their manifest-only refactor remains in Checkpoint 2.

Four small prior-stage files are additionally listed as `PRIOR_FROZEN_CROSS_STAGE_PREREQUISITE`, not as P2 outputs: the Stage 5A full-angle/cycle/metadata provenance triplet and `state_domain_bounds.json`. They are independent whole files, not partial hunks; including them resolves a real clean-checkout fail-closed dependency.
