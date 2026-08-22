"""Generate the offline convergence and stopping audit for frozen policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
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
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    build_trajectory_component_cache,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
    validate_active_reference_file,
)
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
    POLICY_IDS,
    RESEARCH_ONLY,
    apply_research_decision_guard,
    build_initial_research_state,
    policy_definitions,
    rank_exploration_frontier,
    run_policy,
    select_exploit_candidate,
)
from .run_research_decision_guarded_sequential_personalization import (
    ANALYSIS_CASES,
    DEFAULT_OUTPUT_DIRECTORY as PREVIOUS_STAGE_DIRECTORY,
    DEFAULT_PARAMETER_MAP_PATH,
)
from .sequential_personalization_convergence_stopping_audit import (
    AUDIT_PROTOCOL_ID,
    BOUNDARY_OPTIMUM_DIAGNOSTIC,
    EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
    HORIZON_STATUS,
    INFORMATIVE_BUT_LOW_DECISION_VALUE,
    OFFLINE_METHOD_REQUIRES_REVISION,
    TRIAL_BUDGETS,
    audit_post_decision_local_truth,
    build_best_trajectory_stability,
    build_boundary_chasing_audit,
    build_exploration_decision_value,
    build_marginal_improvement,
    build_natural_stopping_summary,
    build_prediction_landscape_evolution,
    build_repeated_exploration_audit,
    build_subject_path_divergence,
    build_trial_budget_sensitivity,
    classify_knee_stiff,
    freeze_readiness_audit,
    missed_improvement_summary,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_POLICY_SOURCE = MODULE_DIR / "research_decision_guarded_sequential_personalization.py"
AUDIT_SOURCE = MODULE_DIR / "sequential_personalization_convergence_stopping_audit.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "sequential_personalization_convergence_stopping_audit_v1"
)

CSV_FILENAMES = (
    "natural_stopping_summary.csv",
    "boundary_chasing_audit.csv",
    "marginal_improvement_by_trial.csv",
    "knee_stiff_extended_audit.csv",
    "missed_improvement_cases.csv",
    "missed_improvement_summary.csv",
    "correct_stop_audit.csv",
    "exploration_decision_value.csv",
    "repeated_exploration_audit.csv",
    "subject_path_divergence.csv",
    "prediction_landscape_evolution.csv",
    "model_parameter_evolution.csv",
    "best_trajectory_stability.csv",
    "trial_budget_sensitivity.csv",
    "false_improvement_extended_audit.csv",
)
REPORT_FILENAMES = (
    "OFFLINE_METHOD_FREEZE_READINESS_AUDIT.md",
    "CONVERGENCE_AND_STOPPING_REPORT.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "extended_trial_J_history.png",
    "alpha_path_to_boundary.png",
    "marginal_improvement_vs_trial.png",
    "exploration_information_vs_decision_value.png",
    "missed_improvement_by_scenario.png",
    "trial_budget_sensitivity.png",
    "subject_sequential_path_comparison.png",
    "offline_method_freeze_readiness_overview.png",
)


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    selected = [frame for frame in frames if not frame.empty]
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
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


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_j_history(best: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for case_id, group in best.groupby("case_id", sort=False):
        axis.plot(group["iteration"], group["best_actual_J"], "-o", ms=3, label=case_id)
    axis.axvline(6, ls="--", color="black", alpha=0.55, label="previous 6-trial budget")
    axis.set(xlabel="executed trial", ylabel="actual best J", title="Extended P2 best-objective history")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    _save(figure, output)


def _plot_alpha_boundary(boundary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for case_id, group in boundary.groupby("case_id", sort=False):
        ordered = group.sort_values("iteration")
        axis.plot(ordered["alpha_hip"], ordered["alpha_knee"], "-o", ms=3, label=case_id)
    axis.axvline(-5, color="black", lw=1)
    axis.axvline(2, color="black", lw=1)
    axis.axhline(-5, color="black", lw=1)
    axis.axhline(2, color="black", lw=1)
    axis.set(xlabel="hip amplitude delta (deg)", ylabel="knee amplitude delta (deg)", title="Executed P2 paths and generator parameter bounds")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    _save(figure, output)


def _plot_marginal(marginal: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    accepted = marginal.loc[marginal["accepted_exploit"].astype(bool)]
    for case_id, group in accepted.groupby("case_id", sort=False):
        axis.plot(group["iteration"], group["marginal_best_J_improvement"], "-o", ms=3, label=case_id)
    axis.set(xlabel="trial", ylabel="marginal actual best-J improvement", title="Accepted EXPLOIT marginal improvement (no new threshold)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6)
    _save(figure, output)


def _plot_exploration(value: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    if not value.empty:
        colors = np.where(
            value["decision_value_observed_within_2_rounds"].astype(bool),
            "#2a9d55",
            "#d77b00",
        )
        axis.scatter(
            value["incremental_log_information_gain"],
            value["new_supported_point_count"],
            c=colors,
            s=55,
        )
    axis.set(xlabel="incremental log-information gain", ylabel="new supported points", title="Information gain versus observed decision value")
    axis.grid(alpha=0.25)
    axis.text(0.02, 0.98, "green: future exploit/best change within 2 rounds\norange: informative but low observed decision value", transform=axis.transAxes, va="top", fontsize=8)
    _save(figure, output)


def _plot_missed(rounds: pd.DataFrame, output: Path) -> None:
    counts = rounds.groupby("case_id", as_index=False)["missed_opportunity_round"].sum()
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(counts["case_id"], counts["missed_opportunity_round"], color="#8c6bb1")
    axis.tick_params(axis="x", rotation=55, labelsize=7)
    axis.set(ylabel="missed-improvement rounds", title="Post-decision missed local improvements by case")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output)


def _plot_budget(budget: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for case_id, group in budget.groupby("case_id", sort=False):
        axis.plot(group["offline_research_budget"], group["final_best_J"], "-o", label=case_id)
    axis.set(xlabel="offline diagnostic trial budget", ylabel="final actual best J", title="P2 trial-budget sensitivity")
    axis.set_xticks(TRIAL_BUDGETS)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=2)
    _save(figure, output)


def _plot_subject_paths(best: pd.DataFrame, output: Path) -> None:
    matched = best.loc[
        best["subject_id"].isin(("baseline", "hip_stiff", "knee_stiff", "heavy_leg"))
        & best["scenario_name"].eq("matched_linear")
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    for subject, group in matched.groupby("subject_id", sort=False):
        axis.plot(group["best_alpha_hip"], group["best_alpha_knee"], "-o", ms=4, label=subject)
    axis.set(xlabel="best hip delta (deg)", ylabel="best knee delta (deg)", title="Matched-subject accepted-best paths")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, output)


def _plot_readiness(readiness: dict[str, Any], output: Path) -> None:
    labels = ("missed rounds", "low-value explores", "boundary optima", "false improvements", "cap cases")
    values = (
        readiness["missed_opportunity_round_count"],
        readiness["informative_but_low_decision_value_explore_count"],
        readiness["boundary_optimum_case_count"],
        readiness["executed_false_improvement_count"],
        readiness["diagnostic_cap_case_count"],
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#4c78a8")
    axis.set(ylabel="count", title=f"Offline method freeze readiness: {readiness['status']}")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output)


def _freeze_report(readiness: dict[str, Any]) -> str:
    reasons = readiness["revision_reasons"] or ["none"]
    return "\n".join(
        [
            "# Offline method freeze readiness audit",
            "",
            f"Final status: `{readiness['status']}`",
            "",
            "This status concerns the offline method architecture only. It is not a human-readiness, safety, clinical, comfort, or robot-motion approval.",
            "",
            "## Evidence checklist",
            "",
            f"- Explore → update → future exploit observations: {readiness['explore_update_future_exploit_chain_observed_count']}.",
            f"- Whole-map recomputation architecture stable: {str(readiness['whole_map_recomputation_architecture_stable']).lower()}.",
            f"- Missed-opportunity rounds: {readiness['missed_opportunity_round_count']}.",
            f"- Informative but low-decision-value explores: {readiness['informative_but_low_decision_value_explore_count']}.",
            f"- Boundary-optimum diagnostic cases: {readiness['boundary_optimum_case_count']}.",
            f"- Executed false improvements: {readiness['executed_false_improvement_count']}.",
            f"- Cases reaching the 20-trial diagnostic cap: {readiness['diagnostic_cap_case_count']}.",
            "",
            "## Reasons requiring attention",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
            "Thresholds were not tuned. Any revision must be a future, separately reviewed research task.",
            "",
            "- `NOT_HUMAN_READY = true`",
            "- `NOT_ROBOT_MOTION_APPROVED = true`",
            "",
        ]
    )


def _main_report(
    natural: pd.DataFrame,
    boundary: pd.DataFrame,
    marginal: pd.DataFrame,
    rounds: pd.DataFrame,
    correct_stop: pd.DataFrame,
    exploration: pd.DataFrame,
    best: pd.DataFrame,
    false_audit: pd.DataFrame,
    readiness: dict[str, Any],
) -> str:
    p2_natural = natural.loc[natural["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)]
    boundary_cases = sorted(
        boundary.loc[
            boundary["final_optimum_diagnostic_status"].eq(BOUNDARY_OPTIMUM_DIAGNOSTIC),
            "case_id",
        ].unique()
    )
    accepted = marginal.loc[marginal["accepted_exploit"].astype(bool)]
    first_mean = float(accepted.groupby("case_id").first()["marginal_best_J_improvement"].mean()) if not accepted.empty else float("nan")
    last_mean = float(accepted.groupby("case_id").last()["marginal_best_J_improvement"].mean()) if not accepted.empty else float("nan")
    low_value = int(exploration["exploration_decision_value_status"].eq(INFORMATIVE_BUT_LOW_DECISION_VALUE).sum()) if not exploration.empty else 0
    decision_value = int(exploration["decision_value_observed_within_2_rounds"].astype(bool).sum()) if not exploration.empty else 0
    stable_six = best.groupby("case_id")["six_trial_best_equals_extended_final_best"].first()
    lines = [
        f"# {AUDIT_PROTOCOL_ID}",
        "",
        "## Plain-language findings",
        "",
        f"- P2 naturally stopped in {int(p2_natural['natural_stop_reached'].sum())}/{len(p2_natural)} cases; {int(p2_natural['diagnostic_cap_reached'].sum())} reached the 20-trial diagnostic cap.",
        f"- The previous six-trial best already equalled the extended final best in {int(stable_six.sum())}/{len(stable_six)} cases, although six trials did not always expose the natural stopping decision.",
        f"- Boundary-optimum diagnostics: {', '.join(boundary_cases) if boundary_cases else 'none'}.",
        f"- Mean first versus last accepted-EXPLOIT marginal improvement: {first_mean:.6g} versus {last_mean:.6g}; this is characterization, not a new stopping threshold.",
        f"- Missed-opportunity rounds: {int(rounds['missed_opportunity_round'].sum())} of {int(rounds['true_local_improvement_available'].sum())} rounds with a true >0.005 local improvement.",
        f"- Exploration trials with observed decision value within two rounds: {decision_value}; informative but low-decision-value trials: {low_value}.",
        f"- Executed false improvements across extended P0/P1/P2: {int(false_audit['executed_false_improvement'].astype(bool).sum())}.",
        f"- Freeze-readiness result: `{readiness['status']}`.",
        "",
        "## Correct-stop audit",
        "",
    ]
    for row in correct_stop.to_dict(orient="records"):
        lines.append(
            f"- {row['case_id']}: `{row['conservative_stop_classification']}`; true local improvement at stop = {str(row['true_local_improvement_available_at_stop']).lower()}."
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "All truth-based missed-opportunity and correct-stop checks were computed after the policy decision and were not fed back into proposal, ranking, fitting, stopping, or threshold selection.",
            "",
            f"- `{INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS}`",
            f"- `{GLOBAL_MODEL_RELIABILITY_RULE_STATUS}`",
            "- `REAL_ROBOT_HARD_SAFEGUARD = NOT_DEFINED_NOT_APPROVED`",
            "- `NOT_HUMAN_READY = true`",
            "- `NOT_ROBOT_MOTION_APPROVED = true`",
            "",
        ]
    )
    return "\n".join(lines)


def _leakage_report(rounds: pd.DataFrame) -> str:
    frozen = bool(rounds["policy_decision_frozen_before_truth"].astype(bool).all())
    no_feedback = bool((~rounds["truth_fed_back_to_policy"].astype(bool)).all())
    return f"""# Data leakage audit

- Extended policy execution used the unchanged prediction, support, decision guard, exploration ranking, model fitting, and stop logic.
- Virtual truth for missed-improvement/correct-stop analysis was attached only after each EXPLOIT, EXPLORE, or STOP decision was frozen: `{str(frozen).lower()}`.
- Post-decision truth was fed back to policy: `{str(not no_feedback).lower()}`.
- Held-out final-test data were not loaded or used.
- Truth was not used for proposal, guard calibration, frontier ranking, fitting, stopping, or threshold tuning.
- The 20-trial cap is an offline virtual diagnostic horizon, not a human trial recommendation.
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
    validate_active_reference_file()

    previous_policy_path = PREVIOUS_STAGE_DIRECTORY / "policy_definition.json"
    previous_policy = json.loads(previous_policy_path.read_text(encoding="utf-8"))
    current_policy = policy_definitions()
    if previous_policy != current_policy:
        raise RuntimeError("frozen P0/P1/P2 policy definition changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("algorithm equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("model support gate changed")

    raw_map = pd.read_csv(parameter_map_path)
    lattice = geometrically_valid_parameter_lattice(raw_map)
    if len(lattice) != 21025:
        raise RuntimeError("formal geometric map no longer contains 21,025 points")
    cache = build_trajectory_component_cache(lattice)
    case_classes = {
        f"{subject}__{scenario}": case_class
        for subject, scenario, case_class in analysis_cases
    }
    states: dict[str, Any] = {}
    results = []
    for subject_id, scenario_name, _ in analysis_cases:
        case_id = f"{subject_id}__{scenario_name}"
        state = build_initial_research_state(subject_id, scenario_name)
        states[case_id] = state
        for policy_id in POLICY_IDS:
            results.append(
                run_policy(
                    state,
                    policy_id,
                    lattice,
                    cache,
                    trial_budget=EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
                    allow_extended_offline_diagnostic_horizon=True,
                )
            )
    p2_results = [
        result
        for result in results
        if result.policy_id == POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT
    ]

    natural = build_natural_stopping_summary(results, case_classes)
    truth_audits = [
        audit_post_decision_local_truth(
            result,
            states[result.summary["case_id"]],
            cache,
            case_class=case_classes[result.summary["case_id"]],
        )
        for result in p2_results
    ]
    candidate_truth = _concat(item.candidate_rows for item in truth_audits)
    truth_rounds = _concat(item.round_rows for item in truth_audits)
    missed_cases = _concat(item.missed_cases for item in truth_audits)
    missed_summary = missed_improvement_summary(truth_rounds)
    correct_stop = _concat(item.correct_stop for item in truth_audits)

    boundary = build_boundary_chasing_audit(p2_results)
    all_history = _concat(result.trial_history for result in p2_results)
    marginal = build_marginal_improvement(all_history)
    landscape_frames = []
    landscape_by_case: dict[str, pd.DataFrame] = {}
    for result in p2_results:
        landscape = build_prediction_landscape_evolution(
            result, states[result.summary["case_id"]], lattice, cache
        )
        landscape_frames.append(landscape)
        landscape_by_case[result.summary["case_id"]] = landscape
    landscapes = _concat(landscape_frames)
    exploration_frames = []
    repeated_frames = []
    for result in p2_results:
        value = build_exploration_decision_value(
            result, landscape_by_case[result.summary["case_id"]]
        )
        exploration_frames.append(value)
        repeated_frames.append(build_repeated_exploration_audit(result, value))
    exploration_value = _concat(exploration_frames)
    repeated = _concat(repeated_frames)
    divergence = build_subject_path_divergence(p2_results)
    parameters = _concat(result.parameter_history for result in p2_results)
    parameters["parameter_interpretation"] = (
        "LOCAL_EQUIVALENT_DYNAMICS_NOT_PHYSIOLOGICAL_TISSUE_CHANGE"
    )
    best = _concat(build_best_trajectory_stability(result) for result in p2_results)
    budget = _concat(build_trial_budget_sensitivity(result) for result in p2_results)
    false_audit = _concat(result.false_improvement_audit for result in results)

    knee_candidates = [
        result
        for result in p2_results
        if result.subject_id == "knee_stiff" and result.scenario_name == "matched_linear"
    ]
    knee = pd.DataFrame()
    if knee_candidates:
        knee_result = knee_candidates[0]
        knee_rounds = truth_rounds.loc[
            truth_rounds["case_id"].eq(knee_result.summary["case_id"])
        ]
        knee_status = classify_knee_stiff(knee_result, knee_rounds)
        knee = knee_rounds.merge(
            knee_result.trial_history,
            on=["case_id", "subject_id", "scenario_name", "iteration"],
            how="left",
            suffixes=("_truth_audit", "_executed"),
        )
        knee = knee.merge(
            knee_result.known_region_history[
                ["case_id", "iteration", "supported_point_count", "new_supported_point_count"]
            ],
            on=["case_id", "iteration"],
            how="left",
        )
        knee = knee.merge(
            landscape_by_case[knee_result.summary["case_id"]][
                ["case_id", "iteration", "RMS_map_change", "max_abs_map_change"]
            ],
            on=["case_id", "iteration"],
            how="left",
        )
        knee["knee_stiff_diagnostic_status"] = knee_status

    readiness = freeze_readiness_audit(
        all_rounds=truth_rounds,
        exploration_value=exploration_value,
        boundary=boundary,
        natural_stopping=natural.loc[
            natural["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)
        ],
        false_improvement=false_audit,
    )

    tables = {
        "natural_stopping_summary.csv": natural,
        "boundary_chasing_audit.csv": boundary,
        "marginal_improvement_by_trial.csv": marginal,
        "knee_stiff_extended_audit.csv": knee,
        "missed_improvement_cases.csv": missed_cases,
        "missed_improvement_summary.csv": missed_summary,
        "correct_stop_audit.csv": correct_stop,
        "exploration_decision_value.csv": exploration_value,
        "repeated_exploration_audit.csv": repeated,
        "subject_path_divergence.csv": divergence,
        "prediction_landscape_evolution.csv": landscapes,
        "model_parameter_evolution.csv": parameters,
        "best_trajectory_stability.csv": best,
        "trial_budget_sensitivity.csv": budget,
        "false_improvement_extended_audit.csv": false_audit,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)

    (output / REPORT_FILENAMES[0]).write_text(
        _freeze_report(readiness), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _main_report(
            natural,
            boundary,
            marginal,
            truth_rounds,
            correct_stop,
            exploration_value,
            best,
            false_audit,
            readiness,
        ),
        encoding="utf-8",
    )
    (output / REPORT_FILENAMES[2]).write_text(
        _leakage_report(truth_rounds), encoding="utf-8"
    )

    _plot_j_history(best, output / FIGURE_FILENAMES[0])
    _plot_alpha_boundary(boundary, output / FIGURE_FILENAMES[1])
    _plot_marginal(marginal, output / FIGURE_FILENAMES[2])
    _plot_exploration(exploration_value, output / FIGURE_FILENAMES[3])
    _plot_missed(truth_rounds, output / FIGURE_FILENAMES[4])
    _plot_budget(budget, output / FIGURE_FILENAMES[5])
    _plot_subject_paths(best, output / FIGURE_FILENAMES[6])
    _plot_readiness(readiness, output / FIGURE_FILENAMES[7])

    generated = (*CSV_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    output_hashes = {name: _sha256(output / name) for name in generated}
    metadata = {
        "protocol_id": AUDIT_PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "active_reference_sha256_observed": sha256_file(ACTIVE_REFERENCE_PATH),
        "five_parameter_names": list(PARAMETER_NAMES),
        "mechanical_objective_version": MECHANICAL_OBJECTIVE_VERSION,
        "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "support_coverage_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "extended_offline_diagnostic_horizon": EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
        "diagnostic_horizon_status": HORIZON_STATUS,
        "trial_budget_sensitivity_values": list(TRIAL_BUDGETS),
        "policy_definition_changed": False,
        "policy_definition_source_sha256": sha256_file(previous_policy_path),
        "decision_guard_source_sha256": _text_sha256(
            inspect.getsource(apply_research_decision_guard)
        ),
        "exploit_selector_source_sha256": _text_sha256(
            inspect.getsource(select_exploit_candidate)
        ),
        "exploration_ranker_source_sha256": _text_sha256(
            inspect.getsource(rank_exploration_frontier)
        ),
        "core_policy_source_sha256": sha256_file(CORE_POLICY_SOURCE),
        "audit_source_sha256": sha256_file(AUDIT_SOURCE),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "parameter_map_sha256": sha256_file(parameter_map_path),
        "geometrically_admissible_point_count": int(len(lattice)),
        "analysis_case_count": len(analysis_cases),
        "policy_count": len(POLICY_IDS),
        "readiness": readiness,
        "initial_identification_acceptance_status": INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
        "global_model_reliability_rule_status": GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
        "real_robot_hard_safeguard_status": "NOT_DEFINED_NOT_APPROVED",
        "research_status": RESEARCH_ONLY,
        "approval_status": NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
        "heldout_final_test_used": False,
        "post_decision_truth_fed_back_to_policy": False,
        "thresholds_tuned": False,
        "formal_human_ready_model_created": False,
        "robot_motion_approved": False,
        "real_robot_connected": False,
        "hardware_control_collection_safety_modified": False,
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
    print(f"protocol={AUDIT_PROTOCOL_ID}")
    print(f"output={arguments.output_directory}")
    print(f"readiness={metadata['readiness']['status']}")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    print("status=OFFLINE_VIRTUAL_CHARACTERIZATION_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
