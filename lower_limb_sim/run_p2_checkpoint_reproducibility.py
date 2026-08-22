"""Generate checkpoint-boundary and large-artifact reproducibility audits.

This is repository/provenance tooling only.  It does not import prospective
validation code, generate subjects or outcomes, or modify any scientific rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .large_artifact_reproduction import load_large_artifact_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "lower_limb_sim"
OUTPUT_DIR = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_checkpoint_and_large_artifact_reproducibility_v1"
)
PAIR_PLAN_SHA256 = "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"
PAIR_PLAN_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_v2_formal_research_protocol_v1"
    / "designated_local_validation_pair_plan.csv"
)

CHECKPOINT_CORE = {
    1: (
        "lower_limb_sim/research_decision_guarded_sequential_personalization.py",
        "lower_limb_sim/run_research_decision_guarded_sequential_personalization.py",
        "lower_limb_sim/test_research_decision_guarded_sequential_personalization.py",
        "lower_limb_sim/sequential_personalization_convergence_stopping_audit.py",
        "lower_limb_sim/run_sequential_personalization_convergence_stopping_audit.py",
        "lower_limb_sim/test_sequential_personalization_convergence_stopping_audit.py",
    ),
    2: (
        "lower_limb_sim/p2_revision_root_cause_audit.py",
        "lower_limb_sim/run_p2_revision_root_cause_audit.py",
        "lower_limb_sim/test_p2_revision_root_cause_audit.py",
        "lower_limb_sim/p2_revision_v2_design.py",
        "lower_limb_sim/run_p2_revision_v2_design.py",
        "lower_limb_sim/test_p2_revision_v2_design.py",
        "lower_limb_sim/p2_revision_v2_research_prototype.py",
        "lower_limb_sim/run_p2_revision_v2_research_prototype.py",
        "lower_limb_sim/test_p2_revision_v2_research_prototype.py",
    ),
    3: (
        "lower_limb_sim/p2_v2_formal_research_protocol.py",
        "lower_limb_sim/run_p2_v2_formal_research_protocol.py",
        "lower_limb_sim/test_p2_v2_formal_research_protocol.py",
    ),
    4: (
        "lower_limb_sim/p2_v2_offline_research_prototype.py",
        "lower_limb_sim/run_p2_v2_offline_research_prototype.py",
        "lower_limb_sim/test_p2_v2_offline_research_prototype.py",
    ),
}
CHECKPOINT_ARTIFACT_DIRS = {
    1: (
        "research_only_decision_guarded_sequential_personalization_v1",
        "sequential_personalization_convergence_stopping_audit_v1",
    ),
    2: (
        "p2_revision_root_cause_audit_v1",
        "p2_revision_v2_design_analysis_v1",
        "p2_revision_v2_research_prototype_v1",
    ),
    3: ("p2_v2_formal_research_protocol_v1",),
    4: ("p2_v2_offline_research_prototype_v1",),
}
OUTPUT_NAMES = (
    "P2_GIT_INVENTORY.md",
    "checkpoint_1_files.txt",
    "checkpoint_2_files.txt",
    "checkpoint_3_files.txt",
    "checkpoint_4_files.txt",
    "partial_stage_hunk_audit.md",
    "large_generated_artifact_inventory.csv",
    "large_artifact_dependency_graph.csv",
    "FORMAL_ARTIFACT_REPRODUCTION.md",
    "CHECKPOINT_DATA_PROVENANCE_AUDIT.md",
    "GIT_CHECKPOINT_PLAN.md",
)
LARGE_EXTENSIONS = {
    ".csv", ".npy", ".npz", ".parquet", ".feather", ".h5", ".hdf5",
    ".pt", ".pth", ".onnx",
}


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _stage_artifact_files(checkpoint: int, *, include_large: bool) -> list[str]:
    large_paths = {
        entry["expected_path"] for entry in load_large_artifact_manifest()["artifacts"]
    }
    paths: list[str] = []
    base = MODULE_DIR / "formal_artifacts"
    for directory_name in CHECKPOINT_ARTIFACT_DIRS[checkpoint]:
        for path in sorted((base / directory_name).rglob("*")):
            if not path.is_file():
                continue
            relative = _repo_path(path)
            if include_large or relative not in large_paths:
                paths.append(relative)
    return paths


def initial_stage_paths(checkpoint: int, *, include_large: bool) -> list[str]:
    return sorted(
        set(CHECKPOINT_CORE[checkpoint])
        | set(_stage_artifact_files(checkpoint, include_large=include_large))
    )


def checkpoint_paths() -> dict[int, list[str]]:
    paths = {
        checkpoint: initial_stage_paths(checkpoint, include_large=False)
        for checkpoint in range(1, 5)
    }
    infrastructure = {
        ".gitignore",
        "lower_limb_sim/data/reference_trajectories/processed/reference_full_angles.csv",
        "lower_limb_sim/data/reference_trajectories/processed/detected_cycles.csv",
        "lower_limb_sim/data/reference_trajectories/processed/metadata.json",
        "lower_limb_sim/data/reference_local_active_asymmetric/state_domain_bounds.json",
        "lower_limb_sim/test_decision_relevant_global_model_reliability.py",
        "lower_limb_sim/large_artifact_reproduction.py",
        "lower_limb_sim/test_large_artifact_reproduction.py",
        "lower_limb_sim/run_p2_checkpoint_reproducibility.py",
        "lower_limb_sim/formal_artifacts/GENERATED_LARGE_ARTIFACT_MANIFEST.json",
        *{
            _repo_path(OUTPUT_DIR / output_name)
            for output_name in OUTPUT_NAMES
        },
    }
    paths[2] = sorted(set(paths[2]) | infrastructure)
    return paths


def _write_checkpoint_manifests(paths: dict[int, list[str]]) -> None:
    all_paths: list[str] = []
    for checkpoint, checkpoint_files in paths.items():
        manifest_path = OUTPUT_DIR / f"checkpoint_{checkpoint}_files.txt"
        manifest_path.write_text("\n".join(checkpoint_files) + "\n", encoding="utf-8")
        all_paths.extend(checkpoint_files)
    duplicates = sorted({path for path in all_paths if all_paths.count(path) > 1})
    if duplicates:
        raise RuntimeError(f"cross-checkpoint duplicate paths: {duplicates}")


def _source_consumers() -> dict[str, tuple[list[str], list[str]]]:
    sources = sorted(MODULE_DIR.glob("*.py"))
    output: dict[str, tuple[list[str], list[str]]] = {}
    for entry in load_large_artifact_manifest()["artifacts"]:
        filename = entry["expected_filename"]
        logical_id = entry["logical_artifact_id"]
        tests: list[str] = []
        scripts: list[str] = []
        for source in sources:
            text = source.read_text(encoding="utf-8")
            if filename not in text and logical_id not in text:
                continue
            relative = _repo_path(source)
            if source.name.startswith("test_"):
                tests.append(relative)
            else:
                scripts.append(relative)
        output[entry["expected_path"]] = (tests, scripts)
    return output


def _producer_for(path: str) -> str:
    entries = {
        entry["expected_path"]: entry for entry in load_large_artifact_manifest()["artifacts"]
    }
    if path in entries:
        return entries[path]["generator_module"]
    if "/data/delay_compensation/" in path:
        return "lower_limb_sim.run_delay_compensation_experiment"
    if "/data/variable_delay/" in path:
        return "lower_limb_sim.run_variable_delay_experiment"
    if "/data/model_mismatch/" in path:
        return "lower_limb_sim.run_model_mismatch_experiment"
    if "/data/geometry_error/" in path:
        return "lower_limb_sim.run_geometry_error_experiment"
    if "/formal_artifacts/" in path:
        return "EXISTING_FORMAL_STAGE_RUNNER_SEE_DIRECTORY_METADATA"
    return "UNKNOWN_OUTSIDE_P2_CHECKPOINT_SCOPE"


def _ignored_paths(paths: list[str]) -> set[str]:
    completed = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=PROJECT_ROOT,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr)
    return {line for line in completed.stdout.splitlines() if line}


def _write_large_inventory() -> list[dict[str, object]]:
    manifest = load_large_artifact_manifest()
    registered = {entry["expected_path"]: entry for entry in manifest["artifacts"]}
    files = sorted(
        path
        for path in MODULE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in LARGE_EXTENSIONS
        and path.stat().st_size > 1_000_000
    )
    relative_paths = [_repo_path(path) for path in files]
    tracked = set(_git("ls-files").splitlines())
    ignored = _ignored_paths(relative_paths)
    consumers = _source_consumers()
    rows: list[dict[str, object]] = []
    for path, relative in zip(files, relative_paths):
        tests, scripts = consumers.get(relative, ([], []))
        entry = registered.get(relative)
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "tracked": str(relative in tracked).lower(),
                "ignored": str(relative in ignored).lower(),
                "producer": _producer_for(relative),
                "consumer_tests": ";".join(tests),
                "consumer_scripts": ";".join(scripts),
                "sha256_if_present": _sha256(path),
                "regenerable": "true" if entry else "UNKNOWN_OUTSIDE_P2_SCOPE",
                "deterministic": "true" if entry else "UNKNOWN_OUTSIDE_P2_SCOPE",
                "required_for_normal_pytest": (
                    str(entry["required_for_normal_pytest"]).lower()
                    if entry else "NOT_AUDITED_OUTSIDE_P2_SCOPE"
                ),
                "required_for_formal_reproduction": (
                    str(entry["required_for_formal_reproduction"]).lower()
                    if entry else "STAGE_SPECIFIC_OUTSIDE_P2_SCOPE"
                ),
            }
        )
    fieldnames = list(rows[0]) if rows else []
    with (OUTPUT_DIR / "large_generated_artifact_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_dependency_graph() -> None:
    rows = [
        ("global_prediction_truth_comparison", "lower_limb_sim.run_decision_relevant_global_model_reliability", "formal producer"),
        ("global_prediction_truth_comparison", "lower_limb_sim.test_decision_relevant_global_model_reliability", "manifest-only normal regression; former direct row-count read removed"),
        ("global_prediction_truth_comparison", "lower_limb_sim.large_artifact_reproduction", "full or representative deterministic regeneration"),
    ]
    for subject in ("baseline", "hip_stiff", "knee_stiff", "heavy_leg"):
        artifact = f"truth_landscape_{subject}"
        rows.extend(
            [
                (artifact, "lower_limb_sim.run_p2_revision_root_cause_audit", "formal producer"),
                (artifact, "lower_limb_sim.test_p2_revision_root_cause_audit", "manifest and small-summary normal regression; former direct CSV read removed"),
                (artifact, "lower_limb_sim.large_artifact_reproduction", "full or representative deterministic regeneration"),
            ]
        )
    rows.extend(
        [
            ("truth_landscape_knee_stiff", "lower_limb_sim.run_p2_revision_v2_research_prototype", "direct formal-development input; regenerate root artifact first"),
            ("truth_landscape_knee_stiff", "lower_limb_sim.test_p2_revision_v2_research_prototype", "manifest-only source-SHA regression; former direct file hash removed"),
        ]
    )
    with (OUTPUT_DIR / "large_artifact_dependency_graph.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("artifact", "consumer", "usage_type"))
        writer.writerows(rows)


def _write_inventory_doc(paths: dict[int, list[str]], large_rows: list[dict[str, object]]) -> None:
    initial_counts = {
        checkpoint: len(initial_stage_paths(checkpoint, include_large=True))
        for checkpoint in range(1, 5)
    }
    initial_total = sum(initial_counts.values())
    visible_untracked = [line for line in _git("ls-files", "--others", "--exclude-standard", "lower_limb_sim").splitlines() if line]
    tracked_lower = [line for line in _git("ls-files", "lower_limb_sim").splitlines() if line]
    lines = [
        "# P2 Git Inventory",
        "",
        "## Snapshot",
        "",
        f"- Branch: `{_git('branch', '--show-current')}`",
        f"- HEAD: `{_git('rev-parse', 'HEAD')}`",
        f"- Git-tracked `lower_limb_sim` paths at HEAD/worktree: {len(tracked_lower)}.",
        f"- Currently visible untracked `lower_limb_sim` paths: {len(visible_untracked)}.",
        f"- Reconstructed task-start P2 paths: {initial_total} (must equal 172).",
        f"- Generated data-like files larger than 1,000,000 bytes: {len(large_rows)}.",
        "- `.DS_Store` and the five registered large truth CSVs are ignored by exact rules; their on-disk files were retained.",
        "",
        "## Reconstructed ownership of the 172 task-start P2 paths",
        "",
        "| Checkpoint | Scientific scope | Source/runner/test + artifact paths | Git candidates after excluding manifest-managed large CSVs |",
        "|---|---|---:|---:|",
        f"| 1 | P2 V1 research foundation | {initial_counts[1]} | {len(initial_stage_paths(1, include_large=False))} |",
        f"| 2 | Revision root cause, V2 design, V2 research prototype | {initial_counts[2]} | {len(initial_stage_paths(2, include_large=False))} |",
        f"| 3 | Frozen research protocol and 324-pair plan | {initial_counts[3]} | {len(initial_stage_paths(3, include_large=False))} |",
        f"| 4 | Default-off offline prototype evaluation | {initial_counts[4]} | {len(initial_stage_paths(4, include_large=False))} |",
        "",
        "The four 3.2 MB truth landscapes remain scientific outputs owned by Checkpoint 2, but are deliberately absent from its Git pathspec because they are content-addressed regenerated artifacts. The 132 MB global comparison predates these four P2 checkpoints and is likewise manifest-managed, not silently reassigned.",
        "",
        "## Classification",
        "",
        "- Scientific source, runner, and test files are listed exactly once in the four checkpoint manifests.",
        "- Small formal artifacts remain ordinary Git candidates in their scientific stage.",
        "- Large generated truth outputs are recorded in `GENERATED_LARGE_ARTIFACT_MANIFEST.json` and `large_generated_artifact_inventory.csv`.",
        "- Most ignored `lower_limb_sim/data/**` belongs to earlier simulation stages and is not reassigned to P2.",
        "- Four small previously ignored frozen prerequisites are packaged in Checkpoint 2 solely so the tracked active-reference loader and generator work after clone; they retain their earlier scientific provenance and are not counted among the 172 P2 paths.",
        "- No file was moved or deleted to manufacture a clean status.",
        "",
        "## Final checkpoint path counts (including this task's infrastructure)",
        "",
    ]
    lines.extend(f"- Checkpoint {number}: {len(items)} paths." for number, items in paths.items())
    (OUTPUT_DIR / "P2_GIT_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if initial_total != 172:
        raise RuntimeError(f"task-start P2 ownership reconstruction is {initial_total}, not 172")


def _write_partial_hunk_audit() -> None:
    text = """# Partial-stage hunk audit

No scientific source, runner, test, or formal artifact contains independent hunks belonging to more than one of the four checkpoints. Therefore there are no `PARTIAL_STAGE_FILE` entries.

Two pre-existing tracked files receive Checkpoint 2 reproducibility-only edits:

- `.gitignore`: one independent exact-rule block for `.DS_Store` and five generated large CSVs.
- `lower_limb_sim/test_decision_relevant_global_model_reliability.py`: normal-regression assertions now use the content-addressed manifest instead of the 132 MB local file.

Each is wholly assigned to Checkpoint 2 for this worktree change; neither changes scientific policy. The Checkpoint 2 root-cause and prototype tests are new/untracked stage files and their manifest-only refactor remains in Checkpoint 2.

Four small prior-stage files are additionally listed as `PRIOR_FROZEN_CROSS_STAGE_PREREQUISITE`, not as P2 outputs: the Stage 5A full-angle/cycle/metadata provenance triplet and `state_domain_bounds.json`. They are independent whole files, not partial hunks; including them resolves a real clean-checkout fail-closed dependency.
"""
    (OUTPUT_DIR / "partial_stage_hunk_audit.md").write_text(text, encoding="utf-8")


def _write_reproduction_doc() -> None:
    text = """# Formal Artifact Reproduction

## NORMAL_REGRESSION

Normal regression checks algorithm invariants, small summaries, frozen SHA values, and manifest provenance. It does not require any of the five large CSVs:

```bash
python3 -B -m pytest -q -p no:cacheprovider \\
  lower_limb_sim/test_decision_relevant_global_model_reliability.py \\
  lower_limb_sim/test_p2_revision_root_cause_audit.py \\
  lower_limb_sim/test_p2_revision_v2_research_prototype.py \\
  lower_limb_sim/test_large_artifact_reproduction.py
```

If an optional local cache exists, its SHA and schema are verified. Absence is not a skip and does not change scientific expected values.

A clean checkout must include the four small frozen prerequisites explicitly listed in Checkpoint 2 (`reference_full_angles.csv`, `detected_cycles.csv`, its Stage 5A `metadata.json`, and `state_domain_bounds.json`). They are source/provenance inputs, not large generated truth caches.

## Representative deterministic verification

```bash
python3 -m lower_limb_sim.large_artifact_reproduction \\
  --artifact truth_landscape_baseline \\
  --output-dir /tmp/p2-large-reproduction-a \\
  --representative-subset
python3 -m lower_limb_sim.large_artifact_reproduction \\
  --artifact truth_landscape_baseline \\
  --output-dir /tmp/p2-large-reproduction-b \\
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
"""
    (OUTPUT_DIR / "FORMAL_ARTIFACT_REPRODUCTION.md").write_text(text, encoding="utf-8")


def _write_provenance_audit() -> None:
    protocol_metadata = json.loads(
        (PAIR_PLAN_PATH.parent / "metadata.json").read_text(encoding="utf-8")
    )
    pair_hash = _sha256(PAIR_PLAN_PATH)
    if pair_hash != PAIR_PLAN_SHA256:
        raise RuntimeError("frozen designated pair plan SHA changed")
    if protocol_metadata["designated_local_pair_plan_count"] != 324:
        raise RuntimeError("frozen designated pair count changed")
    if protocol_metadata["designated_local_outcomes_available"] is not False:
        raise RuntimeError("Checkpoint 3 improperly contains outcomes")
    text = f"""# Checkpoint Data Provenance Audit

## Frozen identities

- Active reference SHA-256: `{ACTIVE_REFERENCE_SHA256}`.
- ROM: `ROM_PROTOCOL_V2`.
- Shank convention: `q_hip - q_knee`.
- Designated local validation pair-plan SHA-256: `{pair_hash}`.
- Pair count: 324.

## Checkpoint 3 independence

Checkpoint 3 contains its protocol core, runner, tests, three protocol JSON files, the 324-row designated pair plan, its SHA provenance, reports, tables, and figures. Its metadata says `designated_local_outcomes_available=false`; it does not need Checkpoint 4 results to define the pair plan. This supports the statement: **the P2 V2 evaluation protocol was frozen before later evaluation outcomes**. This is an offline development chronology statement, not a human/robot approval.

Checkpoint 4 contains retrospective/default-off calculated outcomes only. It must not rewrite the Checkpoint 3 pair plan or its SHA.

## Large truth boundary

The five registered large files are retrospective virtual/development truth. They may reproduce earlier development evidence, but they must not select a future prospective manifest, subject/cohort, candidate, or threshold. They are not human, robot, or clinical evidence.

No prospective manifest, cohort, truth, case, outcome, or P2 V2 candidate was created by this checkpoint task. Prospective validation remains stopped pending the user's commits.
"""
    (OUTPUT_DIR / "CHECKPOINT_DATA_PROVENANCE_AUDIT.md").write_text(text, encoding="utf-8")


def _write_git_plan(paths: dict[int, list[str]]) -> None:
    messages = {
        1: "implement decision-guarded sequential personalization and convergence audit",
        2: "add P2 revision root-cause and design evidence",
        3: "freeze P2 V2 offline research protocol",
        4: "evaluate P2 V2 offline research prototype",
    }
    lines = [
        "# Selective Git Checkpoint Plan",
        "",
        "This file is instructions only. The audit runner does not stage or commit anything. Review each pathspec file before use; it contains one explicit repository-relative path per line and no wildcard.",
        "",
        "Run from repository root, in order:",
        "",
    ]
    for checkpoint in range(1, 5):
        manifest = _repo_path(OUTPUT_DIR / f"checkpoint_{checkpoint}_files.txt")
        lines.extend(
            [
                f"## Checkpoint {checkpoint}",
                "",
                "```bash",
                f"sed -n '1,240p' {manifest}",
                f"git add --pathspec-from-file={manifest}",
                "git diff --cached --stat",
                "git diff --cached --name-status",
                "git status --short",
                f"git commit -m \"{messages[checkpoint]}\"",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "The five manifest-managed large CSVs and `.DS_Store` do not appear in any pathspec. Do not use `git add .` or a directory-wide add. If any reviewed staged set differs from the pathspec, stop and resolve it before commit.",
            "",
            "Only after all four commits and a fresh status/test review may the separately defined prospective-validation task be reconsidered; this plan does not start it.",
        ]
    )
    (OUTPUT_DIR / "GIT_CHECKPOINT_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_audits() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths()
    _write_checkpoint_manifests(paths)
    large_rows = _write_large_inventory()
    _write_dependency_graph()
    _write_inventory_doc(paths, large_rows)
    _write_partial_hunk_audit()
    _write_reproduction_doc()
    _write_provenance_audit()
    _write_git_plan(paths)
    observed = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if observed != set(OUTPUT_NAMES):
        raise RuntimeError(f"audit output set mismatch: {sorted(observed)}")
    return {
        "output_directory": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "checkpoint_path_counts": {str(k): len(v) for k, v in paths.items()},
        "large_artifact_count": len(large_rows),
        "prospective_validation_started": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    if argv:
        raise ValueError("this runner accepts no arguments")
    result = generate_audits()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
