"""Generate the offline global decision-reliability characterization artifacts.

The runner deliberately keeps proposal-time prediction separate from virtual
truth evaluation.  It does not freeze a reliability rule, approve a diagnostic
model, read held-out final-test data, or execute a trajectory/robot command.
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
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rehab_robot_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .decision_relevant_global_model_reliability import (
    DIAGNOSTIC_INITIAL_MODEL,
    DIAGNOSTIC_ONLY,
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS,
    GRID_DISTANCE_DEFINITION,
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    IMPROVE,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    NEUTRAL,
    NOT_APPROVED_FOR_PERSONALIZATION,
    NOT_HUMAN_READY,
    PROTOCOL_ID,
    RELATIVE_ERROR_DEFINITION,
    WORSE,
    build_predicted_map,
    build_trajectory_component_cache,
    decision_sign_agreement_summary,
    diagnostic_model_from_sequential_result,
    evaluate_truth_map,
    false_improvement_cases,
    frozen_baseline_metadata,
    geometrically_valid_parameter_lattice,
    global_rank_consistency,
    local_decision_regret,
    local_rank_consistency,
    predicted_best_regret,
    reliability_vs_domain_coverage,
    reliability_vs_support_distance,
    scenario_reliability_summary,
    select_predicted_best,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
    validate_active_reference_file,
)
from .initial_identification_acceptance_rule import (
    MODEL_STRUCTURE_LIMITATION,
    build_parameter_identifiability_table,
    build_validation_observations,
    diagnose_model_structure_limitation,
    evaluate_validation_by_trial,
    parameter_stability_by_trial,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .mismatch_scenarios import get_mismatch_scenario
from .parameter_estimator import PARAMETER_NAMES
from .safeguarded_sequential_initial_identification import (
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    run_sequential_initial_identification,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_PARAMETER_MAP_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "admissible_personalization_region_v1"
    / "parameter_space_admissibility.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "decision_relevant_global_model_reliability_v1"
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
    "global_prediction_truth_comparison.csv",
    "decision_sign_agreement.csv",
    "false_improvement_cases.csv",
    "global_rank_consistency.csv",
    "local_rank_consistency_by_distance.csv",
    "predicted_best_regret.csv",
    "local_decision_regret.csv",
    "reliability_vs_support_distance.csv",
    "reliability_vs_domain_coverage.csv",
    "scenario_decision_reliability_summary.csv",
    "explore_exploit_implication_table.csv",
)
FIGURE_FILENAMES = (
    "J_pred_vs_J_truth.png",
    "decision_sign_confusion.png",
    "false_improvement_vs_support_distance.png",
    "rank_consistency_vs_distance.png",
    "predicted_best_vs_truth_best.png",
    "global_reliability_characterization_overview.png",
)
REPORT_FILENAMES = (
    "GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS.md",
    "DECISION_RELEVANT_RELIABILITY_REPORT.md",
    "DATA_LEAKAGE_AUDIT.md",
)

# Captured before this stage was created.  These paths are intentionally not
# committed or staged by this runner, and remain three explicit checkpoint
# boundaries for the operator.
RECORDED_TASK_START_GIT_STATUS = (
    "?? lower_limb_sim/.DS_Store",
    "?? lower_limb_sim/formal_artifacts/initial_identification_acceptance_rule_v1/",
    "?? lower_limb_sim/formal_artifacts/safeguarded_sequential_initial_identification_v1/",
    "?? lower_limb_sim/initial_identification_acceptance_rule.py",
    "?? lower_limb_sim/run_initial_identification_acceptance_rule.py",
    "?? lower_limb_sim/run_safeguarded_sequential_initial_identification.py",
    "?? lower_limb_sim/safeguarded_sequential_initial_identification.py",
    "?? lower_limb_sim/test_initial_identification_acceptance_rule.py",
    "?? lower_limb_sim/test_safeguarded_sequential_initial_identification.py",
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


def _case_id(subject_id: str, scenario_name: str) -> str:
    return f"{subject_id}__{scenario_name}"


def _limited_lattice_for_test(lattice: pd.DataFrame, maximum_points: int) -> pd.DataFrame:
    if maximum_points < 27:
        raise ValueError("maximum_points must be at least 27")
    normalized = lattice.loc[:, ["hip_delta", "knee_delta", "phase_delta"]].to_numpy(
        dtype=float
    ) / np.asarray([GRID_HIP_STEP_DEG, GRID_KNEE_STEP_DEG, GRID_PHASE_STEP])
    ordered = lattice.assign(
        _radius=np.max(np.abs(normalized), axis=1),
        _l1=np.sum(np.abs(normalized), axis=1),
    ).sort_values(["_radius", "_l1", "trajectory_id"], kind="mergesort")
    local = ordered.loc[ordered["_radius"] <= 1.0 + 1e-12]
    remaining_count = maximum_points - len(local)
    if remaining_count > 0:
        spread_indices = np.linspace(
            0, len(lattice) - 1, num=remaining_count, dtype=int
        )
        spread = lattice.iloc[np.unique(spread_indices)]
        selected = pd.concat((local, spread), ignore_index=True).drop_duplicates(
            "trajectory_id"
        )
        if len(selected) < maximum_points:
            fill = ordered.loc[
                ~ordered["trajectory_id"].isin(selected["trajectory_id"])
            ].head(maximum_points - len(selected))
            selected = pd.concat((selected, fill), ignore_index=True)
    else:
        selected = local.head(maximum_points)
    return selected.head(maximum_points).drop(
        columns=["_radius", "_l1"], errors="ignore"
    ).reset_index(drop=True)


def _selected_validation_diagnostic(result: Any, selected_trial_id: int) -> dict[str, Any]:
    validation = build_validation_observations(result.subject_id, result.truth_scenario)
    adequacy = evaluate_validation_by_trial(result, validation)
    selected = adequacy.loc[adequacy["trial_id"].astype(int).eq(selected_trial_id)]
    if len(selected) != 1:
        raise RuntimeError("selected validation diagnostic is not unique")
    stability = parameter_stability_by_trial(result)
    identifiability = build_parameter_identifiability_table(result, stability)
    diagnosis = diagnose_model_structure_limitation(identifiability, adequacy)
    row = selected.iloc[0]
    precise_status = (
        MODEL_INADEQUATE_FOR_PRECISE_DYNAMICS
        if diagnosis == MODEL_STRUCTURE_LIMITATION
        else "NO_PRECISE_DYNAMICS_LIMITATION_DIAGNOSED_FROM_EXISTING_TREND"
    )
    return {
        "validation_combined_rmse_nm": float(row["validation_combined_rmse_nm"]),
        "validation_combined_nrmse_percent": float(
            row["validation_combined_nrmse_percent"]
        ),
        "validation_e_J": float(row["validation_e_j"]),
        "validation_relative_e_J_percent": float(
            row["validation_relative_e_j_percent"]
        ),
        "existing_model_adequacy_trend_diagnosis": diagnosis,
        "precise_dynamics_status": precise_status,
        "validation_split_only": True,
        "heldout_final_test_read": False,
    }


def _coverage_gate_discontinuity_audit(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, case in table.groupby("case_id", sort=False):
        coverage = case["domain_coverage"].to_numpy(dtype=float)
        below_values = coverage[coverage < MODEL_SUPPORT_COVERAGE_GATE_PERCENT]
        above_values = coverage[coverage >= MODEL_SUPPORT_COVERAGE_GATE_PERCENT]
        if below_values.size == 0 or above_values.size == 0:
            rows.append(
                {
                    "case_id": case_id,
                    "auditable": False,
                    "reason": "one_side_of_existing_90_percent_gate_is_empty",
                }
            )
            continue
        below_value = float(np.max(below_values))
        above_value = float(np.min(above_values))
        below = case.loc[np.isclose(case["domain_coverage"], below_value)]
        above = case.loc[np.isclose(case["domain_coverage"], above_value)]
        rows.append(
            {
                "case_id": case_id,
                "auditable": True,
                "nearest_below_coverage_percent": below_value,
                "nearest_above_coverage_percent": above_value,
                "below_mean_e_J_abs": float(below["e_J_abs"].mean()),
                "above_mean_e_J_abs": float(above["e_J_abs"].mean()),
                "mean_e_J_abs_step_above_minus_below": float(
                    above["e_J_abs"].mean() - below["e_J_abs"].mean()
                ),
                "below_sign_accuracy": float(below["decision_sign_agreement"].mean()),
                "above_sign_accuracy": float(above["decision_sign_agreement"].mean()),
                "sign_accuracy_step_above_minus_below": float(
                    above["decision_sign_agreement"].mean()
                    - below["decision_sign_agreement"].mean()
                ),
                "below_false_improvement_rate": float(
                    below["false_improvement"].mean()
                ),
                "above_false_improvement_rate": float(
                    above["false_improvement"].mean()
                ),
                "coverage_gate_modified": False,
                "formal_threshold_inferred": False,
            }
        )
    return rows


def build_explore_exploit_implication_table(
    evaluated: pd.DataFrame,
    local_regrets: pd.DataFrame,
    validation_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Create conceptual interpretations, never an executable policy."""

    rows: list[dict[str, Any]] = []
    for case_id, case in evaluated.groupby("case_id", sort=False):
        local = local_regrets.loc[local_regrets["case_id"].eq(case_id)].iloc[0]
        validation = validation_diagnostics.loc[
            validation_diagnostics["case_id"].eq(case_id)
        ].iloc[0]
        groups = (
            (
                "SUPPORTED_CURRENT_REGION",
                case.loc[case["model_supported"].astype(bool)],
                (
                    "CANDIDATE_FOR_FUTURE_EXPLOIT_ELIGIBILITY_REVIEW_ONLY"
                    if bool(local["predicted_best_equals_truth_best"])
                    and int(local["local_false_improvement_count"]) == 0
                    else "REQUIRES_MODEL_REVIEW_NOT_EXPLOIT_READY"
                ),
            ),
            (
                "UNSUPPORTED_NEAREST_GRID_FRONTIER",
                case.loc[
                    ~case["model_supported"].astype(bool)
                    & (case["distance_to_supported_region"] <= 1.0 + 1e-12)
                ],
                "CANDIDATE_FOR_FUTURE_EXPLORE_DESIGN_REVIEW_ONLY",
            ),
            (
                "UNSUPPORTED_BEYOND_NEAREST_GRID_FRONTIER",
                case.loc[
                    ~case["model_supported"].astype(bool)
                    & (case["distance_to_supported_region"] > 1.0 + 1e-12)
                ],
                "NOT_DIRECTLY_ACTIONABLE",
            ),
        )
        for context, selected, implication in groups:
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": case["subject_id"].iloc[0],
                    "scenario_name": case["scenario_name"].iloc[0],
                    "context": context,
                    "point_count": int(len(selected)),
                    "validation_e_J": float(validation["validation_e_J"]),
                    "mean_domain_coverage_percent": float(
                        selected["domain_coverage"].mean()
                    )
                    if len(selected)
                    else float("nan"),
                    "mean_distance_to_supported_region": float(
                        selected["distance_to_supported_region"].mean()
                    )
                    if len(selected)
                    else float("nan"),
                    "decision_sign_accuracy": float(
                        selected["decision_sign_agreement"].mean()
                    )
                    if len(selected)
                    else float("nan"),
                    "false_improvement_rate": float(
                        selected["false_improvement"].mean()
                    )
                    if len(selected)
                    else float("nan"),
                    "local_rank_or_decision_evidence": str(
                        local["diagnostic_local_utility_label"]
                    ),
                    "conceptual_implication": implication,
                    "is_executable_policy": False,
                    "trajectory_proposed": False,
                    "trajectory_executed": False,
                    "requires_future_human_review": True,
                }
            )
    return pd.DataFrame(rows)


def _spearman_factor(table: pd.DataFrame, left: str, right: str) -> float:
    selected = table.loc[
        np.isfinite(table[left].to_numpy(dtype=float))
        & np.isfinite(table[right].to_numpy(dtype=float))
    ]
    if len(selected) < 2 or selected[left].nunique() < 2 or selected[right].nunique() < 2:
        return float("nan")
    return float(selected[left].corr(selected[right], method="spearman"))


def _candidate_factor_diagnostics(
    evaluated: pd.DataFrame,
    local_ranks: pd.DataFrame,
    local_regrets: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    scenario = local_regrets.merge(validation, on=["case_id", "subject_id", "scenario_name"])
    radius_one = local_ranks.loc[local_ranks["radius_grid_steps"].eq(1), [
        "case_id", "spearman_rank_correlation"
    ]]
    scenario = scenario.merge(radius_one, on="case_id", validate="one_to_one")
    return {
        "analysis_status": GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS,
        "rule_frozen": False,
        "point_level_spearman": {
            "domain_coverage_vs_e_J_abs": _spearman_factor(
                evaluated, "domain_coverage", "e_J_abs"
            ),
            "distance_to_support_vs_e_J_abs": _spearman_factor(
                evaluated, "distance_to_supported_region", "e_J_abs"
            ),
            "domain_coverage_vs_false_improvement": _spearman_factor(
                evaluated.assign(_false=evaluated["false_improvement"].astype(float)),
                "domain_coverage",
                "_false",
            ),
            "distance_to_support_vs_false_improvement": _spearman_factor(
                evaluated.assign(_false=evaluated["false_improvement"].astype(float)),
                "distance_to_supported_region",
                "_false",
            ),
        },
        "case_level_spearman": {
            "validation_e_J_vs_local_decision_regret": _spearman_factor(
                scenario, "validation_e_J", "local_decision_regret"
            ),
            "radius_one_rank_vs_local_decision_regret": _spearman_factor(
                scenario, "spearman_rank_correlation", "local_decision_regret"
            ),
        },
        "candidate_factors_only": [
            "model_supported",
            "domain_coverage",
            "distance_to_supported_region",
            "validation_e_J",
            "local_ranking_consistency",
            "false_improvement_history",
        ],
        "complex_machine_learning_used": False,
        "heldout_final_test_used": False,
        "threshold_candidate_generated": False,
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_figures(
    output_directory: Path,
    evaluated: pd.DataFrame,
    sign_summary: pd.DataFrame,
    support_distance: pd.DataFrame,
    local_ranks: pd.DataFrame,
    regrets: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> None:
    colors = {True: "#2878B5", False: "#D95319"}

    plt.figure(figsize=(8.0, 6.5))
    for supported, label in ((True, "SUPPORTED"), (False, "UNSUPPORTED")):
        selected = evaluated.loc[evaluated["model_supported"].eq(supported)]
        stride = max(1, len(selected) // 35000)
        selected = selected.iloc[::stride]
        plt.scatter(
            selected["J_truth"],
            selected["J_pred"],
            s=5,
            alpha=0.28,
            color=colors[supported],
            label=f"{label} (display n={len(selected):,})",
        )
    limits = [
        float(min(evaluated["J_truth"].min(), evaluated["J_pred"].min())),
        float(max(evaluated["J_truth"].max(), evaluated["J_pred"].max())),
    ]
    plt.plot(limits, limits, color="black", linewidth=1.2, linestyle="--")
    plt.xlabel("Virtual truth objective J")
    plt.ylabel("Diagnostic model objective J")
    plt.title("Decision objective prediction over the geometric lattice")
    plt.legend()
    plt.grid(alpha=0.2)
    _save_figure(output_directory / "J_pred_vs_J_truth.png")

    directions = (IMPROVE, NEUTRAL, WORSE)
    confusion = np.zeros((3, 3), dtype=int)
    for i, predicted in enumerate(directions):
        for j, truth in enumerate(directions):
            confusion[i, j] = int(
                (
                    evaluated["predicted_direction"].eq(predicted)
                    & evaluated["truth_direction"].eq(truth)
                ).sum()
            )
    plt.figure(figsize=(7.0, 5.8))
    image = plt.imshow(confusion, cmap="Blues")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{confusion[i, j]:,}", ha="center", va="center")
    plt.xticks(range(3), directions)
    plt.yticks(range(3), directions)
    plt.xlabel("Truth direction")
    plt.ylabel("Predicted direction")
    plt.title("Prediction/truth improvement-sign confusion")
    plt.colorbar(image, label="point count")
    _save_figure(output_directory / "decision_sign_confusion.png")

    plt.figure(figsize=(9.0, 6.2))
    for case_id, group in support_distance.groupby("case_id", sort=False):
        group = group.sort_values("distance_to_supported_region")
        plt.plot(
            group["distance_to_supported_region"],
            group["false_improvement_rate"],
            linewidth=1.2,
            alpha=0.8,
            label=case_id,
        )
    plt.xlabel("Distance to supported region (formal grid-step units)")
    plt.ylabel("False-improvement rate")
    plt.title("False improvements versus distance from identified support")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    _save_figure(output_directory / "false_improvement_vs_support_distance.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for case_id, group in local_ranks.groupby("case_id", sort=False):
        axes[0].plot(
            group["radius_grid_steps"],
            group["spearman_rank_correlation"],
            marker="o",
            linewidth=1.4,
            label=case_id,
        )
    for case_id, group in support_distance.groupby("case_id", sort=False):
        group = group.sort_values("distance_to_supported_region")
        axes[1].plot(
            group["distance_to_supported_region"],
            group["mean_absolute_rank_percentile_error"],
            linewidth=1.2,
            label=case_id,
        )
    axes[0].set_xticks((1, 2, 3))
    axes[0].set_ylim(0.985, 1.0005)
    axes[0].set_xlabel("Chebyshev radius from reference (formal grid steps)")
    axes[0].set_ylabel("Local Spearman rank correlation")
    axes[0].set_title("Local rank consistency")
    axes[1].set_xlabel("Distance to supported region (formal grid-step units)")
    axes[1].set_ylabel("Mean absolute rank-percentile error")
    axes[1].set_title("Rank error versus identified support distance")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6, ncol=2)
    fig.suptitle("Ranking reliability versus parameter/support distance", fontsize=13)
    _save_figure(output_directory / "rank_consistency_vs_distance.png")

    cases = list(dict.fromkeys(regrets["case_id"].astype(str)))
    x = np.arange(len(cases), dtype=float)
    width = 0.25
    truth_best = []
    predicted_best = []
    supported_best = []
    for case_id in cases:
        global_row = regrets.loc[
            regrets["case_id"].eq(case_id) & regrets["scope"].eq("GLOBAL")
        ].iloc[0]
        support_row = regrets.loc[
            regrets["case_id"].eq(case_id)
            & regrets["scope"].eq("SUPPORTED_ONLY")
        ].iloc[0]
        truth_best.append(float(global_row["J_truth_at_truth_best"]))
        predicted_best.append(float(global_row["J_truth_at_predicted_best"]))
        supported_best.append(float(support_row["J_truth_at_predicted_best"]))
    plt.figure(figsize=(11.5, 6.2))
    plt.bar(x - width, truth_best, width, label="truth global best")
    plt.bar(x, predicted_best, width, label="predicted global best, truth-evaluated")
    plt.bar(
        x + width,
        supported_best,
        width,
        label="predicted supported best, truth-evaluated",
    )
    plt.xticks(x, cases, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Truth objective J")
    plt.title("Predicted-best versus truth-best diagnostic outcomes")
    visible_values = np.asarray(truth_best + predicted_best + supported_best, dtype=float)
    padding = max(0.001, 0.08 * float(np.ptp(visible_values)))
    plt.ylim(float(np.min(visible_values) - padding), float(np.max(visible_values) + padding))
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    _save_figure(output_directory / "predicted_best_vs_truth_best.png")

    overall = scenario_summary.loc[scenario_summary["scope"].eq("OVERALL")].copy()
    supported = scenario_summary.loc[scenario_summary["scope"].eq("SUPPORTED")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0, 0].bar(overall["case_id"], overall["e_J_abs_p95"], color="#2878B5")
    axes[0, 0].set_title("P95 absolute J error")
    axes[0, 1].bar(
        overall["case_id"], overall["decision_sign_agreement_rate"], color="#54A24B"
    )
    axes[0, 1].set_title("Overall sign agreement")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[1, 0].bar(
        supported["case_id"], supported["false_improvement_rate"], color="#E45756"
    )
    axes[1, 0].set_title("Supported false-improvement rate")
    axes[1, 1].bar(
        overall["case_id"], overall["predicted_best_regret"], color="#F2CF5B"
    )
    axes[1, 1].set_title("Global predicted-best diagnostic regret")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=50, labelsize=7)
    fig.suptitle("Global reliability characterization overview", fontsize=14)
    _save_figure(output_directory / "global_reliability_characterization_overview.png")


def _fmt(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _write_candidate_analysis(
    path: Path,
    factors: Mapping[str, Any],
    scenario_summary: pd.DataFrame,
    local_regrets: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    point = factors["point_level_spearman"]
    case = factors["case_level_spearman"]
    combined_summary = scenario_summary.loc[
        scenario_summary["case_id"].eq("baseline__combined_mild")
        & scenario_summary["scope"].eq("OVERALL")
    ].iloc[0]
    combined_local = local_regrets.loc[
        local_regrets["case_id"].eq("baseline__combined_mild")
    ].iloc[0]
    combined_validation = validation.loc[
        validation["case_id"].eq("baseline__combined_mild")
    ].iloc[0]
    lines = [
        "# GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS",
        "",
        f"Status: `{GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS}` / rule not frozen.",
        "",
        "This is a deterministic factor characterization, not a classifier, threshold search, or executable gate.",
        "",
        "## Candidate factors",
        "",
        "- model support and continuous domain coverage",
        "- formal-grid distance to the nearest supported point",
        "- validation-only objective error from the selected diagnostic trial",
        "- local rank consistency and one-step local regret",
        "- false-improvement history",
        "",
        "## Descriptive associations",
        "",
        f"- coverage vs absolute J error Spearman: `{_fmt(point['domain_coverage_vs_e_J_abs'])}`",
        f"- distance vs absolute J error Spearman: `{_fmt(point['distance_to_support_vs_e_J_abs'])}`",
        f"- coverage vs false-improvement indicator Spearman: `{_fmt(point['domain_coverage_vs_false_improvement'])}`",
        f"- distance vs false-improvement indicator Spearman: `{_fmt(point['distance_to_support_vs_false_improvement'])}`",
        f"- validation e_J vs local regret across cases Spearman: `{_fmt(case['validation_e_J_vs_local_decision_regret'])}`",
        f"- radius-one local rank vs local regret across cases Spearman: `{_fmt(case['radius_one_rank_vs_local_decision_regret'])}`",
        "",
        "`NA` case-level associations mean the observed local-regret/rank outcome lacked enough variation for a correlation; it is not evidence of no relationship. These values are descriptive only. No factor cutoff or multivariable rule was selected.",
        "",
        "## combined_mild diagnostic",
        "",
        f"- existing adequacy trend: `{combined_validation['existing_model_adequacy_trend_diagnosis']}`",
        f"- precise dynamics status: `{combined_validation['precise_dynamics_status']}`",
        f"- validation combined torque RMSE: `{_fmt(combined_validation['validation_combined_rmse_nm'])} N m`",
        f"- improvement-sign agreement: `{_fmt(combined_summary['decision_sign_agreement_rate'] * 100)}%`",
        f"- false-improvement rate: `{_fmt(combined_summary['false_improvement_rate'] * 100)}%`",
        f"- global rank correlation: `{_fmt(combined_summary['spearman_rank_correlation'])}`",
        f"- predicted-best regret: `{_fmt(combined_summary['predicted_best_regret'])}`",
        f"- one-step local regret: `{_fmt(combined_local['local_decision_regret'])}`",
        f"- diagnostic local utility: `{combined_local['diagnostic_local_utility_label']}`",
        "",
        "Even a favorable local label would remain diagnostic-only and would not approve personalization.",
        "",
        "## Leakage and decision boundary",
        "",
        "Held-out final-test data was not read or used. Virtual truth was attached only after prediction, support, and predicted-best IDs were fixed. The existing 90% support gate and 0.005 algorithmic equivalence tolerance were not modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_leakage_audit(path: Path) -> None:
    lines = [
        "# DATA_LEAKAGE_AUDIT",
        "",
        "## Isolation result",
        "",
        "- Diagnostic model fitting used only actually executed virtual sequential-identification trials.",
        "- Matched controls use the temporary Trial 2 estimate; mismatch cases use the final actually executed temporary trial.",
        "- Predeclared validation observations were used only as a diagnostic candidate factor, never for fitting or trajectory ranking.",
        "- Held-out final-test trajectories/data were not read, generated, joined, or hashed by this stage.",
        "- The geometrically admissible lattice and support construction contain no virtual truth objective.",
        "- Global and supported predicted-best trajectory IDs were selected from a table that explicitly rejects any `J_truth` column.",
        "- Virtual truth was generated only in the post-prediction evaluation layer.",
        "- Truth was not used for fitting, proposal, pre-evaluation ranking, support construction, reliability-factor selection, or threshold generation.",
        "",
        "## Frozen boundaries",
        "",
        "No reliability threshold was frozen. The initial-identification acceptance rule remains `REQUIRES_REVIEW`; no human-ready theta_hat_0 exists. No personalization, explore/exploit policy, hardware connection, or trajectory execution occurred.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_scope(table: pd.DataFrame, supported: bool) -> dict[str, float]:
    selected = table.loc[table["model_supported"].eq(supported)]
    return {
        "point_count": int(len(selected)),
        "mean_e_J_abs": float(selected["e_J_abs"].mean()),
        "p95_e_J_abs": float(selected["e_J_abs"].quantile(0.95)),
        "sign_agreement_rate": float(selected["decision_sign_agreement"].mean()),
        "false_improvement_rate": float(selected["false_improvement"].mean()),
    }


def _write_reliability_report(
    path: Path,
    evaluated: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    ranks: pd.DataFrame,
    regrets: pd.DataFrame,
    local_regrets: pd.DataFrame,
    validation: pd.DataFrame,
    distance_table: pd.DataFrame,
    coverage_audit: Sequence[Mapping[str, Any]],
) -> None:
    matched = scenario_summary.loc[
        scenario_summary["scenario_name"].eq("matched_linear")
        & scenario_summary["scope"].eq("OVERALL")
    ]
    combined = scenario_summary.loc[
        scenario_summary["case_id"].eq("baseline__combined_mild")
        & scenario_summary["scope"].eq("OVERALL")
    ].iloc[0]
    combined_local = local_regrets.loc[
        local_regrets["case_id"].eq("baseline__combined_mild")
    ].iloc[0]
    supported = _aggregate_scope(evaluated, True)
    unsupported = _aggregate_scope(evaluated, False)
    distance_error_corr = _spearman_factor(
        evaluated, "distance_to_supported_region", "e_J_abs"
    )
    distance_false_corr = _spearman_factor(
        evaluated.assign(_false=evaluated["false_improvement"].astype(float)),
        "distance_to_supported_region",
        "_false",
    )
    coverage_steps = [row for row in coverage_audit if row.get("auditable")]
    mean_error_step = float(
        np.mean([row["mean_e_J_abs_step_above_minus_below"] for row in coverage_steps])
    )
    mean_sign_step = float(
        np.mean([row["sign_accuracy_step_above_minus_below"] for row in coverage_steps])
    )
    false_count = int(evaluated["false_improvement"].sum())
    favorable_local = local_regrets.loc[
        local_regrets["diagnostic_local_utility_label"].eq(
            "POTENTIALLY_USEFUL_FOR_LOCAL_DECISION"
        ),
        "case_id",
    ].astype(str).tolist()
    nonfavorable_local = local_regrets.loc[
        ~local_regrets["diagnostic_local_utility_label"].eq(
            "POTENTIALLY_USEFUL_FOR_LOCAL_DECISION"
        ),
        ["case_id", "diagnostic_local_utility_label"],
    ]
    lines = [
        "# DECISION_RELEVANT_RELIABILITY_REPORT",
        "",
        f"Protocol: `{PROTOCOL_ID}`. Evidence level: offline virtual research only.",
        "",
        "## A. Why imperfect torque prediction need not imply a wrong decision",
        "",
        "Personalization compares candidate objectives and directions. A model can have a systematic torque error while preserving relative ordering or the improve/neutral/worse sign near the current trajectory. This stage therefore audits error, sign, rank, false improvements, and regret separately; none is treated as approval by itself.",
        "",
        "## B. Matched positive controls",
        "",
        f"Across four matched controls, maximum mean absolute J error was `{_fmt(matched['e_J_abs_mean'].max())}`, minimum sign agreement was `{_fmt(100 * matched['decision_sign_agreement_rate'].min())}%`, minimum Spearman rank correlation was `{_fmt(matched['spearman_rank_correlation'].min())}`, and maximum predicted-best regret was `{_fmt(matched['predicted_best_regret'].max())}`. These near-ideal controls support implementation consistency.",
        "",
        "## C. combined_mild",
        "",
        f"Its existing adequacy trend remains `{validation.loc[validation['case_id'].eq('baseline__combined_mild'), 'existing_model_adequacy_trend_diagnosis'].iloc[0]}`. Improvement-sign accuracy was `{_fmt(100 * combined['decision_sign_agreement_rate'])}%`, false-improvement rate `{_fmt(100 * combined['false_improvement_rate'])}%`, global rank correlation `{_fmt(combined['spearman_rank_correlation'])}`, predicted-best regret `{_fmt(combined['predicted_best_regret'])}`, and one-step supported local regret `{_fmt(combined_local['local_decision_regret'])}`. Local diagnostic label: `{combined_local['diagnostic_local_utility_label']}`.",
        "",
        "## D. Mild mismatch local decision utility",
        "",
        f"Cases meeting the exact diagnostic condition (truth-best local choice matched and zero local false improvements): `{', '.join(favorable_local) if favorable_local else 'none'}`.",
        "",
        "Other cases remain review-only: "
        + (
            "; ".join(
                f"{row.case_id}={row.diagnostic_local_utility_label}"
                for row in nonfavorable_local.itertuples(index=False)
            )
            if len(nonfavorable_local)
            else "none"
        )
        + ".",
        "",
        "## E. Supported versus unsupported",
        "",
        f"Supported (`n={supported['point_count']:,}`): mean/P95 absolute J error `{_fmt(supported['mean_e_J_abs'])}` / `{_fmt(supported['p95_e_J_abs'])}`, sign agreement `{_fmt(100 * supported['sign_agreement_rate'])}%`, false-improvement rate `{_fmt(100 * supported['false_improvement_rate'])}%`. Unsupported (`n={unsupported['point_count']:,}`): `{_fmt(unsupported['mean_e_J_abs'])}` / `{_fmt(unsupported['p95_e_J_abs'])}`, `{_fmt(100 * unsupported['sign_agreement_rate'])}%`, `{_fmt(100 * unsupported['false_improvement_rate'])}%`.",
        "",
        "## F. Distance from support",
        "",
        f"Across all diagnostic maps, formal-grid distance had Spearman association `{_fmt(distance_error_corr)}` with absolute J error and `{_fmt(distance_false_corr)}` with false-improvement occurrence. The distance metric is descriptive grid geometry, not a physical or safety threshold.",
        "",
        "## G. Existing 90% domain-coverage gate",
        "",
        f"At the closest observed coverage levels below and above 90%, the mean across-case error step (above minus below) was `{_fmt(mean_error_step)}` and sign-accuracy step was `{_fmt(mean_sign_step)}`. This does not establish a causal or sharp reliability discontinuity. The 90% gate remains unchanged.",
        "",
        "## H. Candidate ingredients for a future reliability rule",
        "",
        "Support state, continuous coverage, distance to support, independent validation e_J, local rank consistency, local regret, and false-improvement history all remain candidate factors. No rule, weights, or cutoffs were selected.",
        "",
        "## I. Precision error versus local decision ranking",
        "",
        "The validation torque error and local decision outcomes are reported side-by-side in the scenario summary. Any mismatch case with a structure-limitation diagnosis but favorable local diagnostic label demonstrates that precise dynamics adequacy and local decision utility are distinct questions; it still is not approved.",
        "",
        "## J. False improvements",
        "",
        f"There were `{false_count:,}` points where the diagnostic model predicted improvement while virtual truth was neutral or worse under the existing 0.005 research equivalence band. Exact cases are preserved in `false_improvement_cases.csv`.",
        "",
        "## K. Reliability status",
        "",
        f"No model-reliability threshold was frozen: `{GLOBAL_MODEL_RELIABILITY_RULE_STATUS}`.",
        "",
        "## L. Initial-identification acceptance",
        "",
        f"The acceptance rule remains `{INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS}` and was not modified.",
        "",
        "## M. Human-ready model status",
        "",
        "There is no formal or human-ready theta_hat_0. Every model in this stage is `DIAGNOSTIC_ONLY`, not approved for personalization, and not human-ready.",
        "",
        "## N. Execution status",
        "",
        "No personalization, explore/exploit action, robot connection, or trajectory execution occurred.",
        "",
        "## O. Tests",
        "",
        "The final pytest counts and runtime are reported by the task handoff after the complete suite is run; they are not fabricated into this offline artifact before that run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_characterization(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: str | Path = DEFAULT_PARAMETER_MAP_PATH,
    *,
    analysis_cases: Sequence[tuple[str, str, str]] = ANALYSIS_CASES,
    maximum_points: int | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference changed before reliability analysis")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("formal ROM protocol changed")
    if FORMAL_HIP_ROM_DEG != (0.0, 120.0) or FORMAL_KNEE_ROM_DEG != (5.0, 145.0):
        raise RuntimeError("formal ROM bounds changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    if tuple(PARAMETER_NAMES) != (
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    ):
        raise RuntimeError("five-parameter model changed")

    parameter_path = Path(parameter_map_path)
    parameter_map = pd.read_csv(parameter_path)
    lattice = geometrically_valid_parameter_lattice(parameter_map)
    formal_lattice_point_count = int(len(lattice))
    if maximum_points is not None:
        lattice = _limited_lattice_for_test(lattice, int(maximum_points))
    cache = build_trajectory_component_cache(lattice)

    evaluated_frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    predicted_best_ids: dict[tuple[str, str], str] = {}
    prediction_metadata: list[dict[str, Any]] = []
    truth_metadata: list[dict[str, Any]] = []

    for subject_id, scenario_name, evidence_role in analysis_cases:
        result = run_sequential_initial_identification(
            VirtualIdentificationOracle(subject_id, scenario_name),
            default_virtual_patient_envelope(),
            stop_rule=None,
        )
        model = diagnostic_model_from_sequential_result(result)
        validation = _selected_validation_diagnostic(result, model.selected_trial_id)
        case_id = _case_id(subject_id, scenario_name)
        validation_rows.append(
            {
                "case_id": case_id,
                "subject_id": subject_id,
                "scenario_name": scenario_name,
                "evidence_role": evidence_role,
                "diagnostic_model_trial_id": model.selected_trial_id,
                **validation,
            }
        )
        predicted, pred_meta = build_predicted_map(
            model, lattice, cache, batch_size=batch_size
        )
        global_best = select_predicted_best(predicted, supported_only=False)
        supported_best = select_predicted_best(predicted, supported_only=True)
        predicted_best_ids[(case_id, "GLOBAL")] = str(global_best["trajectory_id"])
        predicted_best_ids[(case_id, "SUPPORTED_ONLY")] = str(
            supported_best["trajectory_id"]
        )
        evaluated, truth_meta = evaluate_truth_map(
            predicted, model, cache, batch_size=batch_size
        )
        evaluated["evidence_role"] = evidence_role
        evaluated["truth_scenario_role"] = (
            "POSITIVE_CONTROL" if scenario_name == "matched_linear" else "MISMATCH_DIAGNOSTIC"
        )
        evaluated_frames.append(evaluated)
        prediction_metadata.append({"case_id": case_id, **pred_meta})
        truth_metadata.append({"case_id": case_id, **truth_meta})
        model_records.append(
            {
                "case_id": case_id,
                "subject_id": subject_id,
                "scenario_name": scenario_name,
                "evidence_role": evidence_role,
                "model_type": DIAGNOSTIC_INITIAL_MODEL,
                "model_status": DIAGNOSTIC_ONLY,
                "approval_status": NOT_APPROVED_FOR_PERSONALIZATION,
                "human_readiness": NOT_HUMAN_READY,
                "selected_actual_trial_id": model.selected_trial_id,
                "parameters": dict(model.parameters),
                "identification_domain": {
                    "columns": list(model.identification_domain.columns),
                    "lower": list(model.identification_domain.lower),
                    "upper": list(model.identification_domain.upper),
                    "valid_training_samples": model.identification_domain.valid_training_samples,
                },
                "identification_dataset_sha256": model.identification_dataset_sha256,
                "truth_scenario_definition": get_mismatch_scenario(
                    scenario_name
                ).as_metadata_dict(),
            }
        )

    evaluated = pd.concat(evaluated_frames, ignore_index=True)
    validation_diagnostics = pd.DataFrame(validation_rows)
    ranks = global_rank_consistency(evaluated)
    local_ranks = local_rank_consistency(evaluated)
    regrets = predicted_best_regret(evaluated, predicted_best_ids)
    local_regrets = local_decision_regret(evaluated)
    sign_summary = decision_sign_agreement_summary(evaluated)
    false_cases = false_improvement_cases(evaluated)
    support_distance = reliability_vs_support_distance(evaluated)
    domain_coverage = reliability_vs_domain_coverage(evaluated)
    scenario_summary = scenario_reliability_summary(
        evaluated, ranks, regrets, local_regrets
    ).merge(
        validation_diagnostics,
        on=["case_id", "subject_id", "scenario_name"],
        how="left",
        validate="many_to_one",
    )
    implications = build_explore_exploit_implication_table(
        evaluated, local_regrets, validation_diagnostics
    )
    coverage_audit = _coverage_gate_discontinuity_audit(evaluated)
    factor_diagnostics = _candidate_factor_diagnostics(
        evaluated, local_ranks, local_regrets, validation_diagnostics
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "global_prediction_truth_comparison.csv": evaluated,
        "decision_sign_agreement.csv": sign_summary,
        "false_improvement_cases.csv": false_cases,
        "global_rank_consistency.csv": ranks,
        "local_rank_consistency_by_distance.csv": local_ranks,
        "predicted_best_regret.csv": regrets,
        "local_decision_regret.csv": local_regrets,
        "reliability_vs_support_distance.csv": support_distance,
        "reliability_vs_domain_coverage.csv": domain_coverage,
        "scenario_decision_reliability_summary.csv": scenario_summary,
        "explore_exploit_implication_table.csv": implications,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)

    _plot_figures(
        output,
        evaluated,
        sign_summary,
        support_distance,
        local_ranks,
        regrets,
        scenario_summary,
    )
    _write_candidate_analysis(
        output / "GLOBAL_RELIABILITY_CANDIDATE_ANALYSIS.md",
        factor_diagnostics,
        scenario_summary,
        local_regrets,
        validation_diagnostics,
    )
    _write_leakage_audit(output / "DATA_LEAKAGE_AUDIT.md")
    _write_reliability_report(
        output / "DECISION_RELEVANT_RELIABILITY_REPORT.md",
        evaluated,
        scenario_summary,
        ranks,
        regrets,
        local_regrets,
        validation_diagnostics,
        support_distance,
        coverage_audit,
    )

    expected_without_metadata = set(CSV_FILENAMES + FIGURE_FILENAMES + REPORT_FILENAMES)
    observed_without_metadata = {
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name != "metadata.json"
    }
    unexpected = observed_without_metadata.difference(expected_without_metadata)
    missing = expected_without_metadata.difference(observed_without_metadata)
    if unexpected or missing:
        raise RuntimeError(
            f"artifact set mismatch before metadata: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    artifact_manifest = {
        filename: {
            "sha256": _sha256(output / filename),
            "size_bytes": (output / filename).stat().st_size,
        }
        for filename in sorted(expected_without_metadata)
    }
    metadata = {
        **frozen_baseline_metadata(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "OFFLINE_VIRTUAL_RESEARCH_ONLY",
        "analysis_status": "FORMAL_CHARACTERIZATION_OUTPUT_NOT_A_RELEASE_GATE",
        "parameter_map_path": str(parameter_path.relative_to(PROJECT_ROOT)),
        "parameter_map_sha256": _sha256(parameter_path),
        "formal_geometrically_admissible_point_count": formal_lattice_point_count,
        "evaluated_point_count_per_case": int(len(lattice)),
        "test_lattice_truncation_applied": maximum_points is not None,
        "analysis_case_count": len(analysis_cases),
        "matched_positive_control_count": sum(
            scenario == "matched_linear" for _, scenario, _ in analysis_cases
        ),
        "mild_mismatch_case_count": sum(
            scenario != "matched_linear" for _, scenario, _ in analysis_cases
        ),
        "diagnostic_initial_models": model_records,
        "validation_diagnostics": validation_rows,
        "prediction_stage_audit": prediction_metadata,
        "truth_evaluation_stage_audit": truth_metadata,
        "truth_attached_only_after_prediction_and_predicted_best_selection": True,
        "heldout_final_test_read": False,
        "heldout_final_test_used_for_factor_selection": False,
        "relative_error_definition": RELATIVE_ERROR_DEFINITION,
        "distance_to_supported_region_definition": GRID_DISTANCE_DEFINITION,
        "distance_is_physical_or_safety_threshold": False,
        "formal_grid_steps": {
            "hip_deg": GRID_HIP_STEP_DEG,
            "knee_deg": GRID_KNEE_STEP_DEG,
            "phase": GRID_PHASE_STEP,
        },
        "coverage_gate_discontinuity_audit": coverage_audit,
        "coverage_gate_modified": False,
        "reliability_candidate_factor_diagnostics": factor_diagnostics,
        "global_reliability_threshold_frozen": False,
        "initial_identification_acceptance_rule_modified": False,
        "formal_theta_hat_0_frozen": False,
        "human_ready_theta_hat_0_frozen": False,
        "explore_exploit_policy_implemented": False,
        "personalization_proposed": False,
        "personalization_executed": False,
        "hardware_connected": False,
        "hardware_or_safety_code_modified_by_runner": False,
        "active_reference_verified_sha256": sha256_file(ACTIVE_REFERENCE_PATH),
        "recorded_task_start_git_status": list(RECORDED_TASK_START_GIT_STATUS),
        "git_branch": _git_output("branch", "--show-current"),
        "git_head": _git_output("rev-parse", "HEAD"),
        "checkpoint_boundaries": [
            "SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1",
            "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_V1",
            PROTOCOL_ID,
        ],
        "checkpoint_commit_created_by_runner": False,
        "ds_store_included": False,
        "artifact_file_count_including_metadata": len(artifact_manifest) + 1,
        "artifact_manifest_excluding_metadata": artifact_manifest,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metadata.json", metadata)
    observed = {path.name for path in output.iterdir() if path.is_file()}
    expected = expected_without_metadata | {"metadata.json"}
    if observed != expected:
        raise RuntimeError("final artifact directory must contain exactly 21 files")
    return {
        "output_directory": output,
        "metadata": metadata,
        "tables": tables,
        "runtime_seconds": float(metadata["runtime_seconds"]),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate offline decision-relevant global model reliability artifacts."
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_characterization(
        args.output_directory,
        args.parameter_map,
        batch_size=args.batch_size,
    )
    print(f"protocol={PROTOCOL_ID}")
    print(f"output_directory={result['output_directory']}")
    print(f"artifact_file_count={result['metadata']['artifact_file_count_including_metadata']}")
    print(f"runtime_seconds={result['runtime_seconds']:.3f}")
    print(f"reliability_rule_status={GLOBAL_MODEL_RELIABILITY_RULE_STATUS}")
    print(f"initial_id_acceptance_status={INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS}")
    print("human_ready_theta_hat_0=false")
    print("personalization_executed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
