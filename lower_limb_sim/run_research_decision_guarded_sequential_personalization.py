"""Generate the research-only sequential personalization study artifacts.

The runner is intentionally offline.  It never imports the hardware, control,
collection, or safety packages and never creates a robot/human approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rehab_robot_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    build_trajectory_component_cache,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import ACTIVE_REFERENCE_PATH, sha256_file
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    POLICY_IDS,
    PROTOCOL_ID,
    RESEARCH_ONLY,
    RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET,
    SUPPORT_ROLE,
    build_initial_research_state,
    frozen_baseline_metadata,
    policy_definitions,
    run_policy,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "research_decision_guarded_sequential_personalization.py"
DEFAULT_PARAMETER_MAP_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "admissible_personalization_region_v1"
    / "parameter_space_admissibility.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "research_only_decision_guarded_sequential_personalization_v1"
)

ANALYSIS_CASES = (
    ("baseline", "matched_linear", "MATCHED_POSITIVE_CONTROL"),
    ("hip_stiff", "matched_linear", "MATCHED_POSITIVE_CONTROL"),
    ("knee_stiff", "matched_linear", "MATCHED_POSITIVE_CONTROL"),
    ("heavy_leg", "matched_linear", "MATCHED_POSITIVE_CONTROL"),
    ("baseline", "nonlinear_stiffness_mild", "MILD_MODEL_MISMATCH"),
    ("baseline", "hip_knee_coupling_mild", "MILD_MODEL_MISMATCH"),
    ("baseline", "nonlinear_damping_mild", "MILD_MODEL_MISMATCH"),
    ("baseline", "structured_residual", "MILD_MODEL_MISMATCH"),
    ("baseline", "combined_mild", "MILD_MODEL_MISMATCH"),
)

CSV_FILENAMES = (
    "research_decision_uncertainty.csv",
    "decision_guard_candidate_audit.csv",
    "sequential_trial_history.csv",
    "model_parameter_history.csv",
    "prediction_map_history_summary.csv",
    "known_region_history.csv",
    "exploration_information_gain.csv",
    "false_improvement_execution_audit.csv",
    "policy_comparison.csv",
    "scenario_sequential_summary.csv",
)
FIGURE_FILENAMES = (
    "sequential_personalization_flowchart.png",
    "trial_history_J.png",
    "parameter_evolution_by_trial.png",
    "known_region_growth_by_trial.png",
    "prediction_landscape_iteration_0.png",
    "prediction_landscape_final.png",
    "explore_exploit_path_in_parameter_space.png",
    "policy_comparison.png",
)
REPORT_FILENAMES = (
    "DECISION_GUARDED_SEQUENTIAL_PERSONALIZATION_METHOD.md",
    "DATA_LEAKAGE_AUDIT.md",
)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return completed.stdout.rstrip("\n")


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    selected = [frame for frame in frames if not frame.empty]
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def _policy_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, group in summary.groupby("policy_id", sort=False):
        rows.append(
            {
                "policy_id": policy_id,
                "scenario_count": int(len(group)),
                "executed_trial_count": int(group["number_of_executed_trials"].sum()),
                "exploit_trial_count": int(group["number_of_exploit_trials"].sum()),
                "explore_trial_count": int(group["number_of_explore_trials"].sum()),
                "accepted_improvement_count": int(
                    group["number_of_accepted_improvements"].sum()
                ),
                "executed_false_improvement_count": int(
                    group["number_of_executed_false_improvements"].sum()
                ),
                "mean_final_best_actual_J": float(group["final_best_actual_J"].mean()),
                "mean_actual_J_reduction_from_reference": float(
                    group["actual_J_reduction_from_reference"].mean()
                ),
                "mean_cumulative_regret": float(
                    group["cumulative_regret_vs_best_before"].mean()
                ),
                "mean_final_local_regret": float(group["final_local_regret"].mean()),
                "mean_known_region_growth": float(group["known_region_growth"].mean()),
                "research_only": True,
                "recommended_for_human_or_robot": False,
            }
        )
    return pd.DataFrame(rows)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_flow(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(15, 3.2))
    axis.axis("off")
    labels = (
        "Diagnostic\ninitial model",
        "21,025-point\nprediction map",
        "Validation-only\ndecision guard",
        "EXPLOIT / EXPLORE\nselect exactly one",
        "Selection-gated\nvirtual truth",
        "Five-parameter\nmodel update",
        "Recompute\nwhole map",
    )
    xs = np.linspace(0.06, 0.94, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        axis.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "fc": "#e8f1f8", "ec": "#315b7d"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(xs[index + 1] - 0.055, 0.55),
                xytext=(x + 0.055, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#315b7d"},
            )
    axis.text(
        0.5,
        0.08,
        "OFFLINE VIRTUAL RESEARCH ONLY — no robot execution or human-ready threshold",
        ha="center",
        fontsize=10,
        color="#a33a2b",
    )
    _save(figure, path)


def _plot_trial_history(history: pd.DataFrame, path: Path) -> None:
    p2 = history.loc[history["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    figure, axis = plt.subplots(figsize=(9, 5))
    if not p2.empty:
        for case_id, group in p2.groupby("case_id", sort=False):
            ordered = group.sort_values("iteration")
            axis.plot(ordered["iteration"], ordered["actual_J"], alpha=0.28, lw=1)
        means = p2.groupby("iteration", as_index=False)[
            ["J_pred", "actual_J", "best_actual_J_after"]
        ].mean()
        axis.plot(means["iteration"], means["J_pred"], "--o", label="mean predicted J")
        axis.plot(means["iteration"], means["actual_J"], "-o", label="mean actual J")
        axis.plot(
            means["iteration"], means["best_actual_J_after"], "-s", label="mean best-so-far J"
        )
        explores = p2.loc[p2["trial_purpose"].eq("EXPLORE")]
        axis.scatter(explores["iteration"], explores["actual_J"], marker="D", s=45, label="EXPLORE")
    axis.set(xlabel="personalization iteration", ylabel="mechanical objective J", title="P2 trial history across nine virtual cases")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save(figure, path)


def _plot_parameters(parameters: pd.DataFrame, path: Path) -> None:
    p2 = parameters.loc[parameters["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for axis, name in zip(axes.flat, PARAMETER_NAMES):
        values = p2.groupby("iteration", as_index=False)[f"{name}_after"].mean()
        axis.plot(values["iteration"], values[f"{name}_after"], "-o", ms=3)
        axis.set_title(name)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.suptitle("P2 diagnostic five-parameter evolution (case mean)")
    figure.supxlabel("iteration")
    _save(figure, path)


def _plot_known(known: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for policy_id, group in known.groupby("policy_id", sort=False):
        values = group.groupby("iteration", as_index=False)["supported_point_count"].mean()
        axis.plot(values["iteration"], values["supported_point_count"], "-o", label=policy_id[:2])
    axis.set(xlabel="iteration", ylabel="mean model-supported map points", title="Known/support region growth (provenance only)")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path)


def _plot_landscape(table: pd.DataFrame, path: Path, title: str) -> None:
    selected = table.loc[np.isclose(table["phase_delta"], 0.0)].copy()
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        selected["hip_delta"], selected["knee_delta"], c=selected["J_pred"],
        s=18, cmap="viridis_r"
    )
    figure.colorbar(scatter, ax=axis, label="predicted J")
    axis.set(xlabel="hip amplitude delta (deg)", ylabel="knee amplitude delta (deg)", title=title)
    axis.grid(alpha=0.2)
    _save(figure, path)


def _plot_paths(history: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    p2 = history.loc[history["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter([0], [0], marker="*", s=180, color="black", label="reference")
    for _, group in p2.groupby("case_id", sort=False):
        ordered = group.sort_values("iteration")
        axis.plot(ordered["alpha_hip"], ordered["alpha_knee"], color="#888888", alpha=0.45)
    for purpose, marker, color in (("EXPLORE", "D", "#d77b00"), ("EXPLOIT", "o", "#2166ac")):
        selected = p2.loc[p2["trial_purpose"].eq(purpose)]
        axis.scatter(selected["alpha_hip"], selected["alpha_knee"], marker=marker, color=color, label=purpose)
    p2_summary = summary.loc[summary["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    axis.scatter(p2_summary["final_best_alpha_hip"], p2_summary["final_best_alpha_knee"], marker="X", s=85, color="#2a9d55", label="final accepted best")
    axis.set(xlabel="hip amplitude delta (deg)", ylabel="knee amplitude delta (deg)", title="P2 local explore/exploit paths")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path)


def _plot_policy(comparison: pd.DataFrame, path: Path) -> None:
    metrics = (
        ("mean_final_best_actual_J", "mean final J"),
        ("executed_false_improvement_count", "false improvements"),
        ("executed_trial_count", "executed trials"),
        ("mean_final_local_regret", "mean local regret"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    labels = comparison["policy_id"].str[:2]
    for axis, (column, label) in zip(axes.flat, metrics):
        axis.bar(labels, comparison[column], color=("#8da0cb", "#66c2a5", "#fc8d62"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Offline diagnostic policy comparison")
    _save(figure, path)


def _method_report(summary: pd.DataFrame, comparison: pd.DataFrame, exploration: pd.DataFrame) -> str:
    p2 = summary.loc[summary["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    matched = p2.loc[p2["scenario_name"].eq("matched_linear")]
    mismatch = p2.loc[~p2["scenario_name"].eq("matched_linear")]
    explore_then_exploit = 0
    if not exploration.empty:
        explore_then_exploit = int(exploration["later_enabled_reliable_exploit"].astype(bool).sum())
    mean_info = float(exploration["incremental_log_information_gain"].mean()) if not exploration.empty else float("nan")
    mean_growth = float(exploration["new_supported_point_count"].mean()) if not exploration.empty else float("nan")
    lines = [
        f"# {PROTOCOL_ID}",
        "",
        "## Status boundary",
        "",
        "This is OFFLINE VIRTUAL RESEARCH ONLY. It creates no human-ready theta, formal personalization approval, robot command, clinical-safety claim, or comfort-optimization claim.",
        "",
        "## Method",
        "",
        "Each iteration recomputes predicted J over all 21,025 geometrically admissible points. Execution remains local: a supported trust-region neighbor may be exploited only when its predicted improvement exceeds the maximum designated-validation pairwise delta-J residual plus the existing 0.005 algorithm-equivalence tolerance. Support and distance are provenance/locality fields, never reliability scores. If no exploit passes, P2 may select one adjacent formal-grid frontier point by information gain before any truth access. Exactly one selected virtual trajectory is then executed, appended to the estimator dataset, used for one five-parameter update, and followed by another whole-map calculation.",
        "",
        "The research decision uncertainty uses TRAIN-fitted parameters and designated VALIDATION trajectories only. P95 and P99 are reported as research diagnostics; the first guard uses the conservative maximum observed pairwise delta-J residual. Held-out final-test data are not read.",
        "",
        "## Policy meanings",
        "",
        "- P0 is the supported-only greedy sanity comparator and is not recommended.",
        "- P1 permits only local candidates passing the validation-calibrated research guard.",
        "- P2 uses the same exploit rule, then permits information-driven one-step local exploration when exploit is unavailable.",
        "",
        "## Aggregate virtual results",
        "",
        f"- P2 matched cases: {int(matched['number_of_executed_trials'].sum())} trials, {int(matched['number_of_explore_trials'].sum())} EXPLORE, {int(matched['number_of_exploit_trials'].sum())} EXPLOIT, mean J reduction {float(matched['actual_J_reduction_from_reference'].mean()):.6f}.",
        f"- P2 mismatch cases: {int(mismatch['number_of_executed_trials'].sum())} trials, {int(mismatch['number_of_explore_trials'].sum())} EXPLORE, {int(mismatch['number_of_exploit_trials'].sum())} EXPLOIT, mean J reduction {float(mismatch['actual_J_reduction_from_reference'].mean()):.6f}.",
        f"- Explore trials later followed by reliable exploit: {explore_then_exploit}.",
        f"- Mean explore log-information gain: {mean_info:.6g}; mean supported-region point growth: {mean_growth:.3f}.",
        "",
        "Per-case and per-policy values, including false improvements, parameter changes, map changes, regret, and stop reason, are in the CSV artifacts. These virtual results do not establish a human decision threshold.",
        "",
        "## Frozen statuses",
        "",
        f"- `{INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS}`",
        f"- `{GLOBAL_MODEL_RELIABILITY_RULE_STATUS}`",
        "- `REAL_ROBOT_HARD_SAFEGUARD = NOT_DEFINED_NOT_APPROVED`",
        "- `FORMAL_HUMAN_READY_THETA_0 = false`",
        "- `FORMAL_PERSONALIZATION_APPROVAL = false`",
        "",
    ]
    return "\n".join(lines)


def _leakage_report(results: Sequence[Any]) -> str:
    proposal_clean = all(
        bool(result.truth_access_audit["truth_calls_unchanged_during_every_proposal"])
        for result in results
    )
    one_per_iteration = all(
        bool(result.truth_access_audit["exactly_one_trajectory_per_iteration"])
        for result in results
    )
    return f"""# Data leakage audit

- Model fitting: initial TRAIN plus valid selected ADAPTATION_EXECUTED observations.
- Decision-uncertainty calibration: designated VALIDATION only.
- Held-out final test: not loaded, not fitted, not ranked, not calibrated, not used for stopping.
- Virtual truth: selection-token gated; proposal/ranking truth-call count remained unchanged: `{str(proposal_clean).lower()}`.
- Exactly one selected virtual trajectory per iteration: `{str(one_per_iteration).lower()}`.
- Full-map truth: absent from policy inputs. Final local regret truth is a separately labelled post-policy evaluation.
- Exploration ranking: information metrics first; predicted J, truth, support distance are not the primary score.
- Support: `{SUPPORT_ROLE}`.
- Human/robot approval created: `false`.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
    *,
    analysis_cases: Sequence[tuple[str, str, str]] = ANALYSIS_CASES,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parameter_map = pd.read_csv(parameter_map_path)
    lattice = geometrically_valid_parameter_lattice(parameter_map)
    if len(lattice) != 21025:
        raise RuntimeError(f"formal geometrically admissible map must contain 21025 points, got {len(lattice)}")
    cache = build_trajectory_component_cache(lattice)

    results: list[Any] = []
    case_classes: dict[str, str] = {}
    for subject_id, scenario_name, case_class in analysis_cases:
        case_id = f"{subject_id}__{scenario_name}"
        case_classes[case_id] = case_class
        initial = build_initial_research_state(subject_id, scenario_name)
        for policy_id in POLICY_IDS:
            results.append(run_policy(initial, policy_id, lattice, cache))

    summaries = pd.DataFrame([dict(result.summary) for result in results])
    summaries["case_class"] = summaries["case_id"].map(case_classes)
    histories = _concat(result.trial_history for result in results)
    guards = _concat(result.decision_guard_audit for result in results)
    parameters = _concat(result.parameter_history for result in results)
    maps = _concat(result.prediction_map_history for result in results)
    known = _concat(result.known_region_history for result in results)
    uncertainty_summary = _concat(result.uncertainty_history for result in results)
    uncertainty_pairs = _concat(result.uncertainty_pairwise_audit for result in results)
    uncertainty = uncertainty_pairs.merge(
        uncertainty_summary,
        on=["case_id", "subject_id", "scenario_name", "policy_id", "iteration"],
        how="left",
        validate="many_to_one",
    )
    exploration = _concat(result.exploration_information_gain for result in results)
    false_audit = _concat(result.false_improvement_audit for result in results)
    comparison = _policy_comparison(summaries)

    tables = {
        "research_decision_uncertainty.csv": uncertainty,
        "decision_guard_candidate_audit.csv": guards,
        "sequential_trial_history.csv": histories,
        "model_parameter_history.csv": parameters,
        "prediction_map_history_summary.csv": maps,
        "known_region_history.csv": known,
        "exploration_information_gain.csv": exploration,
        "false_improvement_execution_audit.csv": false_audit,
        "policy_comparison.csv": comparison,
        "scenario_sequential_summary.csv": summaries,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_json(output / "policy_definition.json", policy_definitions())

    representative = next(
        result
        for result in results
        if result.subject_id == "baseline"
        and result.scenario_name == "matched_linear"
        and result.policy_id == POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
    )
    _plot_flow(output / FIGURE_FILENAMES[0])
    _plot_trial_history(histories, output / FIGURE_FILENAMES[1])
    _plot_parameters(parameters, output / FIGURE_FILENAMES[2])
    _plot_known(known, output / FIGURE_FILENAMES[3])
    _plot_landscape(
        representative.initial_prediction_map,
        output / FIGURE_FILENAMES[4],
        "Baseline matched P2 prediction landscape: iteration 0, phase delta 0",
    )
    _plot_landscape(
        representative.final_prediction_map,
        output / FIGURE_FILENAMES[5],
        "Baseline matched P2 prediction landscape: final, phase delta 0",
    )
    _plot_paths(histories, summaries, output / FIGURE_FILENAMES[6])
    _plot_policy(comparison, output / FIGURE_FILENAMES[7])

    (output / REPORT_FILENAMES[0]).write_text(
        _method_report(summaries, comparison, exploration), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _leakage_report(results), encoding="utf-8"
    )

    generated_names = (
        *CSV_FILENAMES,
        "policy_definition.json",
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
    )
    output_hashes = {name: _sha256(output / name) for name in generated_names}
    metadata = {
        **frozen_baseline_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "core_source_sha256": sha256_file(CORE_SOURCE_PATH),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "parameter_map_path": str(parameter_map_path.relative_to(PROJECT_ROOT)),
        "parameter_map_sha256": sha256_file(parameter_map_path),
        "active_reference_path": str(ACTIVE_REFERENCE_PATH.relative_to(PROJECT_ROOT)),
        "active_reference_sha256_observed": sha256_file(ACTIVE_REFERENCE_PATH),
        "analysis_cases": [
            {"subject_id": subject, "scenario_name": scenario, "case_class": case_class}
            for subject, scenario, case_class in analysis_cases
        ],
        "policy_ids": list(POLICY_IDS),
        "geometrically_admissible_point_count": int(len(lattice)),
        "whole_map_recomputed_after_every_successful_update": bool(
            (summaries["whole_map_recomputation_count"] == summaries["model_update_count"] + 1).all()
        ),
        "trial_budget": RESEARCH_ONLY_PERSONALIZATION_TRIAL_BUDGET,
        "trial_budget_is_human_safety_threshold": False,
        "heldout_final_test_used": False,
        "virtual_truth_used_before_selection": False,
        "hardware_control_collection_safety_imported_or_modified": False,
        "real_robot_connected": False,
        "formal_human_ready_theta_0_created": False,
        "formal_personalization_approval_created": False,
        "research_status": RESEARCH_ONLY,
        "approval_status": NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": output_hashes,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_directory, arguments.parameter_map)
    print(f"protocol={PROTOCOL_ID}")
    print(f"output={arguments.output_directory}")
    print(f"cases={len(metadata['analysis_cases'])}")
    print(f"policies={len(metadata['policy_ids'])}")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    print("status=OFFLINE_VIRTUAL_RESEARCH_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
