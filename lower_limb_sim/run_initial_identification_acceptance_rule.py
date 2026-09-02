"""Generate formal offline acceptance-rule research artifacts.

The generated result is intentionally ``REQUIRES_REVIEW``.  Candidate
thresholds are descriptive TRAIN+VALIDATION analyses, not release limits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    DIAGNOSTIC_ONLY,
    HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW,
    INITIAL_IDENTIFICATION_COMPLETE,
    INITIAL_IDENTIFICATION_INSUFFICIENT,
    MODEL_INADEQUATE_FOR_PERSONALIZATION,
    MODEL_STRUCTURE_LIMITATION,
    NOT_APPROVED_FOR_PERSONALIZATION,
    PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW,
    MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
    PROTOCOL_ID,
    VALIDATION_TRAJECTORY_SPECS,
    build_parameter_identifiability_table,
    build_validation_observations,
    diagnose_model_structure_limitation,
    evaluate_validation_by_trial,
    frozen_baseline_metadata,
    identification_marginal_gain_table,
    parameter_stability_by_trial,
)
from .parameter_estimator import PARAMETER_NAMES
from .safeguarded_sequential_initial_identification import (
    MAX_INITIAL_IDENTIFICATION_TRIALS,
    VIRTUAL_RESEARCH_STOP_RULE_STATUS,
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    run_sequential_initial_identification,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "initial_identification_acceptance_rule_v1"
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

RECORDED_TASK_START_GIT_STATUS = (
    "?? lower_limb_sim/.DS_Store",
    "?? lower_limb_sim/formal_artifacts/safeguarded_sequential_initial_identification_v1/",
    "?? lower_limb_sim/run_safeguarded_sequential_initial_identification.py",
    "?? lower_limb_sim/safeguarded_sequential_initial_identification.py",
    "?? lower_limb_sim/test_safeguarded_sequential_initial_identification.py",
)

LEGACY_VIRTUAL_COMPARATOR = {
    "minimum_rank": 5,
    "minimum_singular_value": 20.0,
    "maximum_condition_number": 50.0,
    "maximum_abs_parameter_correlation": 0.30,
    "maximum_uncertainty_proxy": 0.05,
    "minimum_parameter_sensitivity": 20.0,
    "maximum_validation_rmse_nm": 0.20,
    "source_status": VIRTUAL_RESEARCH_STOP_RULE_STATUS,
}


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
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=MODULE_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return completed.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(subject_id: str, scenario_name: str) -> str:
    return f"{subject_id}__{scenario_name}"


def run_diagnostic_cases() -> tuple[
    list[Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Run all five virtual trials solely to measure marginal evidence."""

    results = []
    stability_frames = []
    identifiability_frames = []
    adequacy_frames = []
    gain_frames = []
    for subject_id, scenario_name, evidence_role in ANALYSIS_CASES:
        result = run_sequential_initial_identification(
            VirtualIdentificationOracle(subject_id, scenario_name),
            default_virtual_patient_envelope(),
            stop_rule=None,
        )
        stability = parameter_stability_by_trial(result)
        identifiability = build_parameter_identifiability_table(result, stability)
        validation = build_validation_observations(subject_id, scenario_name)
        adequacy = evaluate_validation_by_trial(result, validation)
        gains = identification_marginal_gain_table(
            identifiability,
            stability,
            adequacy,
            result.incremental_information_gain,
        )
        for table in (stability, identifiability, adequacy, gains):
            table["evidence_role"] = evidence_role
            table["diagnostic_five_trial_rollout"] = True
            table["operational_stop_rule_applied"] = False
        results.append(result)
        stability_frames.append(stability)
        identifiability_frames.append(identifiability)
        adequacy_frames.append(adequacy)
        gain_frames.append(gains)
    return (
        results,
        pd.concat(identifiability_frames, ignore_index=True),
        pd.concat(stability_frames, ignore_index=True),
        pd.concat(adequacy_frames, ignore_index=True),
        pd.concat(gain_frames, ignore_index=True),
    )


def _trial_level_identifiability(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(
        ["case_id", "subject_id", "scenario_name", "trial_id", "evidence_role"],
        as_index=False,
    ).agg(
        rank=("rank", "first"),
        minimum_singular_value=("minimum_singular_value", "first"),
        condition_number=("condition_number", "first"),
        maximum_abs_parameter_correlation=(
            "maximum_abs_parameter_correlation",
            "first",
        ),
        maximum_uncertainty_proxy=("uncertainty_proxy", "max"),
        minimum_parameter_sensitivity=("sensitivity", "min"),
        maximum_normalized_parameter_change=(
            "maximum_normalized_parameter_change_in_trial",
            "first",
        ),
        within_identification_validation_residual_rmse_nm=(
            "within_identification_validation_residual_rmse_nm",
            "first",
        ),
    )


def _threshold_envelope(
    ident: pd.DataFrame,
    adequacy: pd.DataFrame,
    mask: pd.Series,
    *,
    trial_id: int,
) -> dict[str, float]:
    selected_ident = ident.loc[mask & ident["trial_id"].eq(trial_id)]
    selected_cases = set(selected_ident["case_id"].astype(str))
    selected_adequacy = adequacy.loc[
        adequacy["case_id"].astype(str).isin(selected_cases)
        & adequacy["trial_id"].eq(trial_id)
    ]
    if selected_ident.empty or selected_adequacy.empty:
        raise ValueError("candidate envelope has no evidence rows")
    return {
        "minimum_rank": int(selected_ident["rank"].min()),
        "minimum_singular_value": float(selected_ident["minimum_singular_value"].min()),
        "maximum_condition_number": float(selected_ident["condition_number"].max()),
        "maximum_abs_parameter_correlation": float(selected_ident["maximum_abs_parameter_correlation"].max()),
        "maximum_uncertainty_proxy": float(selected_ident["maximum_uncertainty_proxy"].max()),
        "minimum_parameter_sensitivity": float(selected_ident["minimum_parameter_sensitivity"].min()),
        "maximum_normalized_parameter_change": float(selected_ident["maximum_normalized_parameter_change"].max()),
        "maximum_hip_rmse_nm": float(selected_adequacy["validation_hip_rmse_nm"].max()),
        "maximum_knee_rmse_nm": float(selected_adequacy["validation_knee_rmse_nm"].max()),
        "maximum_combined_rmse_nm": float(selected_adequacy["validation_combined_rmse_nm"].max()),
        "maximum_combined_nrmse_percent": float(selected_adequacy["validation_combined_nrmse_percent"].max()),
        "maximum_validation_e_j": float(selected_adequacy["validation_e_j"].max()),
        "maximum_validation_relative_e_j_percent": float(selected_adequacy["validation_relative_e_j_percent"].max()),
    }


def _candidate_passes(row: Mapping[str, Any], rule: Mapping[str, float]) -> tuple[bool, bool]:
    parameter_pass = bool(
        int(row["rank"]) >= int(rule["minimum_rank"])
        and float(row["minimum_singular_value"]) >= rule["minimum_singular_value"]
        and float(row["condition_number"]) <= rule["maximum_condition_number"]
        and float(row["maximum_abs_parameter_correlation"])
        <= rule["maximum_abs_parameter_correlation"]
        and float(row["maximum_uncertainty_proxy"])
        <= rule["maximum_uncertainty_proxy"]
        and float(row["minimum_parameter_sensitivity"])
        >= rule["minimum_parameter_sensitivity"]
        and float(row["maximum_normalized_parameter_change"])
        <= rule["maximum_normalized_parameter_change"]
    )
    adequacy_pass = bool(
        float(row["validation_hip_rmse_nm"]) <= rule["maximum_hip_rmse_nm"]
        and float(row["validation_knee_rmse_nm"]) <= rule["maximum_knee_rmse_nm"]
        and float(row["validation_combined_rmse_nm"])
        <= rule["maximum_combined_rmse_nm"]
        and float(row["validation_combined_nrmse_percent"])
        <= rule["maximum_combined_nrmse_percent"]
        and float(row["validation_e_j"]) <= rule["maximum_validation_e_j"]
        and float(row["validation_relative_e_j_percent"])
        <= rule["maximum_validation_relative_e_j_percent"]
    )
    return parameter_pass, adequacy_pass


def _candidate_case_outcomes(
    merged: pd.DataFrame,
    rule: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for case_id, group in merged.groupby("case_id", sort=False):
        state = INITIAL_IDENTIFICATION_INSUFFICIENT
        earliest: int | None = None
        parameter_at_final = False
        adequacy_at_final = False
        for row in group.sort_values("trial_id").to_dict("records"):
            parameter_pass, adequacy_pass = _candidate_passes(row, rule)
            parameter_at_final = parameter_pass
            adequacy_at_final = adequacy_pass
            if parameter_pass:
                earliest = int(row["trial_id"])
                state = (
                    INITIAL_IDENTIFICATION_COMPLETE
                    if adequacy_pass
                    else MODEL_INADEQUATE_FOR_PERSONALIZATION
                )
                break
        outcomes[str(case_id)] = {
            "state": state,
            "earliest_trial": earliest,
            "parameter_pass": parameter_at_final,
            "adequacy_pass": adequacy_at_final,
        }
    return outcomes


def build_candidate_table(
    identifiability: pd.DataFrame,
    adequacy: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, float]]:
    ident = _trial_level_identifiability(identifiability)
    merged = ident.merge(
        adequacy.drop(columns=["subject_id", "scenario_name", "evidence_role"], errors="ignore"),
        on=["case_id", "trial_id"],
        validate="one_to_one",
    )
    matched_mask = ident["evidence_role"].eq("MATCHED_POSITIVE_CONTROL")
    matched_envelope = _threshold_envelope(
        ident, adequacy, matched_mask, trial_id=2
    )
    all_mask = pd.Series(True, index=ident.index)
    all_case_envelope = _threshold_envelope(ident, adequacy, all_mask, trial_id=2)
    separated_envelope = dict(all_case_envelope)
    for key in (
        "maximum_hip_rmse_nm",
        "maximum_knee_rmse_nm",
        "maximum_combined_rmse_nm",
        "maximum_combined_nrmse_percent",
        "maximum_validation_e_j",
        "maximum_validation_relative_e_j_percent",
    ):
        separated_envelope[key] = matched_envelope[key]

    candidates: list[dict[str, Any]] = []
    outcomes_by_candidate: dict[str, dict[str, Any]] = {}

    legacy_id = "legacy_virtual_comparator_0p20_nm"
    legacy_outcomes: dict[str, Any] = {}
    for case_id, group in merged.groupby("case_id", sort=False):
        original = group.sort_values("trial_id")
        first = original.loc[
            (original["rank"] >= LEGACY_VIRTUAL_COMPARATOR["minimum_rank"])
            & (
                original["minimum_singular_value"]
                >= LEGACY_VIRTUAL_COMPARATOR["minimum_singular_value"]
            )
            & (
                original["condition_number"]
                <= LEGACY_VIRTUAL_COMPARATOR["maximum_condition_number"]
            )
            & (
                original["maximum_abs_parameter_correlation"]
                <= LEGACY_VIRTUAL_COMPARATOR["maximum_abs_parameter_correlation"]
            )
            & (
                original["maximum_uncertainty_proxy"]
                <= LEGACY_VIRTUAL_COMPARATOR["maximum_uncertainty_proxy"]
            )
            & (
                original["minimum_parameter_sensitivity"]
                >= LEGACY_VIRTUAL_COMPARATOR["minimum_parameter_sensitivity"]
            )
            & (
                original["within_identification_validation_residual_rmse_nm"]
                <= LEGACY_VIRTUAL_COMPARATOR["maximum_validation_rmse_nm"]
            )
        ]
        legacy_outcomes[str(case_id)] = (
            int(first.iloc[0]["trial_id"]) if not first.empty else None
        )
    candidates.append(
        {
            "candidate_rule_id": legacy_id,
            "derivation": "preexisting_virtual_software_comparator",
            "evidence_splits": "TRAIN+within_trial_validation_comparator",
            "threshold_freezing_status": "RESEARCH_ONLY_UNJUSTIFIED_NOT_FORMAL",
            **LEGACY_VIRTUAL_COMPARATOR,
            "maximum_normalized_parameter_change": np.nan,
            "maximum_hip_rmse_nm": np.nan,
            "maximum_knee_rmse_nm": np.nan,
            "maximum_combined_rmse_nm": 0.20,
            "maximum_combined_nrmse_percent": np.nan,
            "maximum_validation_e_j": np.nan,
            "maximum_validation_relative_e_j_percent": np.nan,
            "complete_two_gate_rule": False,
            "matched_cases_accepted": sum(
                trial is not None
                for case, trial in legacy_outcomes.items()
                if case.endswith("matched_linear")
            ),
            "mild_mismatch_cases_accepted": sum(
                trial is not None
                for case, trial in legacy_outcomes.items()
                if not case.endswith("matched_linear")
            ),
            "median_trials_required_when_accepted": float(
                np.median([trial for trial in legacy_outcomes.values() if trial is not None])
            ),
            "clearly_inadequate_ground_truth_defined": False,
            "false_accept_count": np.nan,
            "false_reject_count": np.nan,
            "diagnostic_consistency_conclusion": (
                "incomplete_one_gate_comparator;0.20_Nm_has_no_frozen_source"
            ),
        }
    )

    for candidate_id, rule, derivation in (
        (
            "matched_trial2_positive_control_envelope",
            matched_envelope,
            "exact_worst_case_envelope_of_four_matched_positive_controls_at_trial2",
        ),
        (
            "separated_gate_trial2_diagnostic_envelope",
            separated_envelope,
            "all_case_identifiability_envelope_plus_matched_positive_control_adequacy_envelope_at_trial2",
        ),
        (
            "all_validation_cases_trial2_envelope",
            all_case_envelope,
            "exact_worst_case_envelope_of_all_matched_and_mild_mismatch_cases_at_trial2",
        ),
    ):
        outcomes = _candidate_case_outcomes(merged, rule)
        outcomes_by_candidate[candidate_id] = outcomes
        accepted = [
            value["earliest_trial"]
            for value in outcomes.values()
            if value["state"] == INITIAL_IDENTIFICATION_COMPLETE
        ]
        candidates.append(
            {
                "candidate_rule_id": candidate_id,
                "derivation": derivation,
                "evidence_splits": "TRAIN+VALIDATION_ONLY",
                "threshold_freezing_status": "DIAGNOSTIC_CANDIDATE_REQUIRES_REVIEW",
                **rule,
                "source_status": "RESEARCH_ONLY_NOT_FROZEN",
                "complete_two_gate_rule": True,
                "matched_cases_accepted": sum(
                    value["state"] == INITIAL_IDENTIFICATION_COMPLETE
                    for case, value in outcomes.items()
                    if case.endswith("matched_linear")
                ),
                "mild_mismatch_cases_accepted": sum(
                    value["state"] == INITIAL_IDENTIFICATION_COMPLETE
                    for case, value in outcomes.items()
                    if not case.endswith("matched_linear")
                ),
                "median_trials_required_when_accepted": (
                    float(np.median(accepted)) if accepted else np.nan
                ),
                "clearly_inadequate_ground_truth_defined": False,
                "false_accept_count": np.nan,
                "false_reject_count": np.nan,
                "diagnostic_consistency_conclusion": (
                    "positive_control_sanity_lens_not_a_scientifically_frozen_rule"
                    if candidate_id.startswith("matched")
                    else "separated_gates_diagnose_stable_mismatch_as_model_inadequacy_but_limits_remain_unfrozen"
                    if candidate_id.startswith("separated")
                    else "non_discriminating_envelope_demonstrates_why_distribution_maxima_cannot_define_adequacy"
                ),
            }
        )
    return pd.DataFrame(candidates), outcomes_by_candidate, matched_envelope


def build_summary(
    results: Iterable[Any],
    identifiability: pd.DataFrame,
    stability: pd.DataFrame,
    adequacy: pd.DataFrame,
    diagnostic_outcomes: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        case_id = _case_id(result.subject_id, result.truth_scenario)
        last_ident = identifiability.loc[
            identifiability["case_id"].eq(case_id)
            & identifiability["trial_id"].eq(MAX_INITIAL_IDENTIFICATION_TRIALS)
        ]
        last_stability = stability.loc[
            stability["case_id"].eq(case_id)
            & stability["trial_id"].eq(MAX_INITIAL_IDENTIFICATION_TRIALS)
        ]
        last_model = adequacy.loc[
            adequacy["case_id"].eq(case_id)
            & adequacy["trial_id"].eq(MAX_INITIAL_IDENTIFICATION_TRIALS)
        ].iloc[0]
        candidate = diagnostic_outcomes[case_id]
        diagnosis = diagnose_model_structure_limitation(
            identifiability.loc[identifiability["case_id"].eq(case_id)],
            adequacy.loc[adequacy["case_id"].eq(case_id)],
        )
        rows.append(
            {
                "case_id": case_id,
                "subject_id": result.subject_id,
                "scenario_name": result.truth_scenario,
                "evidence_role": last_ident["evidence_role"].iloc[0],
                "trials_executed_for_diagnostic_marginal_analysis": result.trials_required,
                "trials_executed": result.trials_required,
                "trials_required_under_positive_control_candidate": candidate["earliest_trial"],
                "parameter_identifiability_status": PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW,
                "model_adequacy_status": MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
                "rank": int(last_ident["rank"].iloc[0]),
                "minimum_singular_value": float(last_ident["minimum_singular_value"].iloc[0]),
                "condition_number": float(last_ident["condition_number"].iloc[0]),
                "worst_parameter_correlation": float(last_ident["maximum_abs_parameter_correlation"].iloc[0]),
                "maximum_uncertainty_proxy": float(last_ident["uncertainty_proxy"].max()),
                "minimum_parameter_sensitivity": float(last_ident["sensitivity"].min()),
                "parameter_stability": float(last_stability["normalized_parameter_change"].max()),
                "validation_hip_rmse_nm": float(last_model["validation_hip_rmse_nm"]),
                "validation_knee_rmse_nm": float(last_model["validation_knee_rmse_nm"]),
                "validation_combined_rmse_nm": float(last_model["validation_combined_rmse_nm"]),
                "validation_combined_nrmse_percent": float(last_model["validation_combined_nrmse_percent"]),
                "validation_e_J": float(last_model["validation_e_j"]),
                "validation_relative_e_J_percent": float(last_model["validation_relative_e_j_percent"]),
                "trend_diagnosis": diagnosis,
                "candidate_rule_diagnostic_state": candidate["state"],
                "final_status": INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW,
                "theta_hat_0_frozen": False,
                "personalization_prerequisite": False,
                "temporary_model_status": DIAGNOSTIC_ONLY,
                "temporary_model_personalization_status": NOT_APPROVED_FOR_PERSONALIZATION,
                "heldout_final_test_used": False,
                "truth_parameters_used_by_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_identifiability(identifiability: pd.DataFrame, path: Path) -> None:
    table = _trial_level_identifiability(identifiability)
    grouped = table.groupby("trial_id", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].plot(grouped["trial_id"], grouped["rank"], marker="o", color="#2166ac", label="All 9 cases overlap")
    axes[1].plot(grouped["trial_id"], grouped["minimum_singular_value"], marker="o", color="#2166ac")
    axes[2].plot(grouped["trial_id"], grouped["condition_number"], marker="o", color="#2166ac")
    axes[0].set_ylabel("Numerical rank")
    axes[1].set_ylabel("Minimum singular value")
    axes[2].set_ylabel("Condition number")
    for axis in axes:
        axis.set_xlabel("Accumulated identification trial")
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Parameter identifiability diagnostics across accumulated trials")
    _save_figure(path)


def plot_stability(stability: pd.DataFrame, path: Path) -> None:
    selected = stability.loc[
        stability["case_id"].isin(
            ["baseline__matched_linear", "baseline__combined_mild"]
        )
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharex=True)
    for axis, parameter in zip(axes, PARAMETER_NAMES):
        data = selected.loc[selected["parameter"].eq(parameter)]
        for case_id, color, label in (
            ("baseline__matched_linear", "#2166ac", "matched"),
            ("baseline__combined_mild", "#b2182b", "combined mild"),
        ):
            case = data.loc[data["case_id"].eq(case_id)]
            axis.plot(case["trial_id"], case["estimate"], marker="o", color=color, label=label)
        axis.set_title(parameter.replace("_nm_per_rad", "").replace("_nm_s_per_rad", ""), fontsize=9)
        axis.set_xlabel("Trial")
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Temporary parameter estimate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Temporary five-parameter estimates (diagnostic only)")
    _save_figure(path)


def plot_validation_error(adequacy: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    matched = adequacy.loc[adequacy["evidence_role"].eq("MATCHED_POSITIVE_CONTROL")]
    matched_mean = matched.groupby("trial_id")["validation_combined_rmse_nm"].mean()
    axis.plot(matched_mean.index, matched_mean.values, marker="o", linewidth=2.5, color="#2166ac", label="Matched controls (mean)")
    colors = plt.cm.OrRd(np.linspace(0.45, 0.95, 5))
    mismatch = adequacy.loc[adequacy["evidence_role"].eq("MILD_MODEL_MISMATCH")]
    for color, (scenario, group) in zip(colors, mismatch.groupby("scenario_name", sort=True)):
        axis.plot(group["trial_id"], group["validation_combined_rmse_nm"], marker="o", color=color, label=scenario)
    axis.axhline(0.20, color="#555555", linestyle="--", linewidth=1.2, label="Legacy 0.20 N·m comparator")
    axis.set_xlabel("Accumulated identification trial")
    axis.set_ylabel("Independent validation combined RMSE (N·m)")
    axis.set_xticks(range(1, 6))
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.set_title("Validation error is distinct from identifiability")
    _save_figure(path)


def plot_information_vs_error(gains: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    for role, color, label in (
        ("MATCHED_POSITIVE_CONTROL", "#2166ac", "Matched positive controls"),
        ("MILD_MODEL_MISMATCH", "#b2182b", "Mild mismatch cases"),
    ):
        data = gains.loc[gains["evidence_role"].eq(role)]
        axis.scatter(
            data["selected_candidate_incremental_log_information_gain"],
            data["validation_rmse_improvement_nm"],
            color=color,
            alpha=0.75,
            label=label,
        )
    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.set_xlabel("Incremental log information gain")
    axis.set_ylabel("Validation RMSE improvement (N·m; positive is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("More information does not guarantee lower model error")
    _save_figure(path)


def plot_identifiability_vs_adequacy(
    identifiability: pd.DataFrame, adequacy: pd.DataFrame, path: Path
) -> None:
    ident = _trial_level_identifiability(identifiability)
    final = ident.loc[ident["trial_id"].eq(5)].merge(
        adequacy.loc[adequacy["trial_id"].eq(5), ["case_id", "validation_combined_rmse_nm"]],
        on="case_id",
        validate="one_to_one",
    )
    final["minimum_singular_value_display"] = np.round(
        final["minimum_singular_value"].to_numpy(dtype=float), 6
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    for role, color, marker, label in (
        (
            "MATCHED_POSITIVE_CONTROL",
            "#2166ac",
            "o",
            "Identifiable + candidate-adequate",
        ),
        (
            "MILD_MODEL_MISMATCH",
            "#b2182b",
            "s",
            "Identifiable + candidate-inadequate",
        ),
    ):
        data = final.loc[final["evidence_role"].eq(role)]
        axis.scatter(
            data["minimum_singular_value_display"],
            data["validation_combined_rmse_nm"],
            color=color,
            marker=marker,
            s=65,
            label=label,
        )
        if role == "MATCHED_POSITIVE_CONTROL":
            axis.annotate(
                "4 matched controls",
                (
                    float(data["minimum_singular_value_display"].mean()),
                    float(data["validation_combined_rmse_nm"].max()),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        else:
            label_y = {
                "nonlinear_stiffness_mild": 1.35,
                "combined_mild": 0.72,
                "hip_knee_coupling_mild": 0.36,
                "structured_residual": 0.16,
                "nonlinear_damping_mild": 0.012,
            }
            for row in data.itertuples(index=False):
                scenario_label = str(row.case_id).replace("baseline__", "")
                axis.annotate(
                    scenario_label,
                    (
                        row.minimum_singular_value_display,
                        row.validation_combined_rmse_nm,
                    ),
                    xytext=(
                        row.minimum_singular_value_display + 0.20,
                        label_y.get(
                            scenario_label, row.validation_combined_rmse_nm
                        ),
                    ),
                    textcoords="data",
                    fontsize=7,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#777777",
                        "linewidth": 0.6,
                    },
                )
    axis.set_xlabel("Minimum singular value after Trial 5")
    axis.set_ylabel("Independent validation combined RMSE (N·m)")
    axis.scatter(
        [], [], color="#666666", marker="x", label="Unidentifiable (none observed)"
    )
    center = float(final["minimum_singular_value_display"].mean())
    axis.set_xlim(center - 5.0, center + 5.0)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Identifiable parameters can still yield an inadequate model")
    _save_figure(path)


def _current_rule_audit() -> str:
    return """# Current initial-identification stop-rule audit

## Scope

This audit covers the temporary comparator in
`SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1`. It does not approve a
real-subject or real-robot stopping rule.

| Existing item | Value | Classification | Finding |
|---|---:|---|---|
| Five-parameter numerical rank | 5 | FROZEN | Structural requirement: all five equivalent parameters must be supported. Numerical rank alone is not model adequacy. |
| Minimum singular value | 20 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; no frozen statistical, physical, or prospective source was found. |
| Maximum condition number | 50 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Maximum absolute parameter correlation | 0.30 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Maximum uncertainty proxy | 0.05 | RESEARCH_ONLY / UNJUSTIFIED | Dimensionless design-matrix proxy; not a confidence interval and not formally frozen. |
| Minimum per-parameter sensitivity | 20 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Parameter stability limit | not defined | NOT_DEFINED | The previous comparator had no independent accumulated-estimate stability threshold. |
| Validation residual | 0.20 N·m | RESEARCH_ONLY / UNJUSTIFIED | A convenient virtual software comparator declared in code; no clinical, hardware, statistical, or preregistered source exists. It must not be inherited as a formal threshold. |
| Mechanical-objective validation error limit | not defined | NOT_DEFINED | The previous comparator did not gate on validation `e_J`. |
| NRMSE validation limit | not defined | NOT_DEFINED | The formal metric exists, but no acceptance limit is frozen. |

## 0.20 N·m conclusion

The value is retained only as an explicitly labeled comparison line. The
observed TRAIN+VALIDATION distribution cannot establish whether it is
scientifically too strict or too loose because the repository has no
independent definition of an acceptable mismatched model. It is therefore not
promoted, loosened to 0.45 N·m, or used to freeze this protocol.
"""


def _data_leakage_audit() -> str:
    validation = ", ".join(f"{a}/{b}" for a, b in VALIDATION_TRAJECTORY_SPECS)
    heldout = ", ".join(f"{a}/{b}" for a, b in HELD_OUT_FINAL_TEST_TRAJECTORY_SPECS)
    return f"""# Initial-identification acceptance data-leakage audit

## Allowed evidence

- Accumulated TRAIN observations from the sequential identification trials.
- Frozen VALIDATION trajectories only: {validation}.
- Scenario identity and generator parameters only for post-fit simulation audit;
  they are removed by the strict estimator-input projection before fitting or
  prediction decisions.

## Prohibited evidence

- Held-out final-test trajectories: {heldout}.
- Truth five-parameter values, complex generator torque terms, or future trial
  outcomes in excitation selection, parameter fitting, threshold construction,
  or acceptance decisions.
- Active-reference personalization maps or explore/exploit outcomes.

## Verified boundary

The validation builder generates only the two validation specifications and
projects observations to the existing `ESTIMATOR_INPUT_COLUMNS` whitelist.
The three held-out specifications are constants used solely for a negative
membership assertion; their trajectories and files are not generated or read.

`heldout_final_test_used_for_threshold_construction = false`  
`heldout_final_test_used_for_threshold_selection = false`  
`heldout_final_test_used_for_stopping = false`  
`truth_parameters_used_by_decision = false`
"""


def _report(summary: pd.DataFrame, candidates: pd.DataFrame) -> str:
    combined = summary.loc[summary["scenario_name"].eq("combined_mild")].iloc[0]
    matched = summary.loc[summary["scenario_name"].eq("matched_linear")]
    matched_trials = sorted(
        set(matched["trials_required_under_positive_control_candidate"].dropna().astype(int))
    )
    scenarios = "\n".join(
        f"- `{row.case_id}`: formal `{row.final_status}`; diagnostic candidate "
        f"`{row.candidate_rule_diagnostic_state}`; theta_hat_0 frozen = no."
        for row in summary.itertuples(index=False)
    )
    return f"""# Initial identification acceptance-rule report

## Formal outcome

`{INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REQUIRES_REVIEW}`

The repository has enough evidence to define and test a two-gate architecture,
but not enough independent scientific evidence to freeze its numerical limits.
Consequently no case receives a formal `theta_hat_0`, no personalization
prerequisite passes, and every retained temporary model is `DIAGNOSTIC_ONLY /`
`NOT_APPROVED_FOR_PERSONALIZATION`.

## Why the two gates differ

Parameter identifiability asks whether the five columns of the local sensitivity
matrix carry sufficiently independent numerical information. Model adequacy asks
whether those five equivalent parameters predict unseen validation motion. A
full-rank, well-conditioned fit can still be systematically wrong when the data
generator contains nonlinear stiffness, coupling, nonlinear damping, or a
structured residual that the five-parameter model cannot represent.

## Matched positive controls

Under the explicitly non-formal matched-Trial-2 envelope, all four matched
positive controls first satisfy both diagnostic gates at Trial(s)
`{matched_trials}`. Trial 2 adds a second distinct excitation and lifts the
minimum singular value/support while the independent validation error is at
floating-point solver scale. This shows the software can recover a model that
is structurally identical to its generator; it does not by itself justify a
human-subject acceptance threshold.

## combined_mild diagnosis

After Trial 5, `combined_mild` has rank `{int(combined['rank'])}`, minimum
singular value `{combined['minimum_singular_value']:.6g}`, condition number
`{combined['condition_number']:.6g}`, worst absolute correlation
`{combined['worst_parameter_correlation']:.6g}`, and maximum normalized
parameter change `{combined['parameter_stability']:.6g}`. Independent
validation combined RMSE remains `{combined['validation_combined_rmse_nm']:.6g}`
N·m and validation `e_J` is `{combined['validation_e_J']:.6g}`. Its trend is
therefore `{combined['trend_diagnosis']}`: information improves and all five
columns remain full rank, but structural prediction error does not resolve.
More excitation can improve the information matrix; it cannot create nonlinear,
coupling, or residual terms that are absent from the estimator.

The previous within-identification 20% holdout comparator was approximately
0.44 N·m for this case. The stricter split audit here uses the two predeclared,
independent VALIDATION trajectories and obtains the separately reported value
above; the two numbers answer different questions and are not interchangeable.

## Treatment of 0.20 N·m

The 0.20 N·m line remains `RESEARCH_ONLY / UNJUSTIFIED`. It is shown in the
candidate table and figure for provenance, but it is neither inherited nor
changed to 0.45 N·m. Without an independently justified acceptable-model label,
false-accept and false-reject counts are `NOT_COMPUTABLE`; candidate analysis is
diagnostic consistency analysis only.

## Recommended future rule contents

The parameter-identifiability gate should jointly review: five-column rank,
minimum singular value, condition number, worst parameter correlation,
per-parameter sensitivity, per-parameter uncertainty, and normalized change of
all five accumulated estimates. The model-adequacy gate should independently
review validation hip/knee/combined torque RMSE, formal NRMSE, validation
mechanical-objective `e_J`, and relative `e_J`. Training RMSE belongs to neither
an adequacy release decision nor a substitute for validation.

## Scenario outcomes

The only formal status is `REQUIRES_REVIEW`; the requested categorical states
below are explicitly candidate-rule diagnostics, not frozen releases:

{scenarios}

## Frozen boundaries

- ROM: `{ROM_PROTOCOL_VERSION}`, hip `{FORMAL_HIP_ROM_DEG[0]}–{FORMAL_HIP_ROM_DEG[1]}` deg, knee `{FORMAL_KNEE_ROM_DEG[0]}–{FORMAL_KNEE_ROM_DEG[1]}` deg.
- Active reference: `{ACTIVE_REFERENCE_ID}` / `{ACTIVE_REFERENCE_SHA256}`.
- Angle identity: `{THETA_SHANK_DEFINITION}`.
- Model: unchanged five equivalent parameters `{', '.join(PARAMETER_NAMES)}`.
- Mechanical objective: reused unchanged as a validation-only precursor; no global reliability threshold frozen.
- Held-out final test: not generated, read, tuned against, or used.
- Robot/hardware/safety: not imported, connected, or modified.
- Explore/exploit personalization: not executed.

## Commit boundary

The prerequisite `SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1` files were
already untracked at task start. Commit that stage intentionally first (while
excluding `.DS_Store`), then commit this module, runner, tests, and this artifact
directory as a separate checkpoint. Do not use `git add .`.
"""


def run(output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY) -> dict[str, Any]:
    started = time.perf_counter()
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference changed")
    before_status = _git_output("status", "--short")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    results, identifiability, stability, adequacy, gains = run_diagnostic_cases()
    candidates, outcomes, matched_envelope = build_candidate_table(
        identifiability, adequacy
    )
    positive_control_outcomes = outcomes[
        "separated_gate_trial2_diagnostic_envelope"
    ]
    summary = build_summary(
        results,
        identifiability,
        stability,
        adequacy,
        positive_control_outcomes,
    )

    csv_tables = {
        "parameter_identifiability_by_trial.csv": identifiability,
        "parameter_stability_by_trial.csv": stability,
        "model_adequacy_by_trial.csv": adequacy,
        "identification_marginal_gain_by_trial.csv": gains,
        "acceptance_rule_candidate_table.csv": candidates,
        "initial_identification_acceptance_summary.csv": summary,
    }
    for name, table in csv_tables.items():
        table.to_csv(destination / name, index=False, lineterminator="\n")

    (destination / "CURRENT_INITIAL_ID_STOP_RULE_AUDIT.md").write_text(
        _current_rule_audit(), encoding="utf-8"
    )
    (destination / "INITIAL_ID_ACCEPTANCE_DATA_LEAKAGE_AUDIT.md").write_text(
        _data_leakage_audit(), encoding="utf-8"
    )
    (destination / "INITIAL_IDENTIFICATION_ACCEPTANCE_RULE_REPORT.md").write_text(
        _report(summary, candidates), encoding="utf-8"
    )

    plot_identifiability(
        identifiability, destination / "parameter_identifiability_vs_trial.png"
    )
    plot_stability(stability, destination / "parameter_stability_vs_trial.png")
    plot_validation_error(adequacy, destination / "validation_error_vs_trial.png")
    plot_information_vs_error(
        gains, destination / "information_gain_vs_model_error.png"
    )
    plot_identifiability_vs_adequacy(
        identifiability,
        adequacy,
        destination / "identifiability_vs_model_adequacy.png",
    )

    artifact_paths = sorted(
        path for path in destination.iterdir() if path.name != "metadata.json"
    )
    metadata = {
        **frozen_baseline_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_version_or_git_commit": _git_output("rev-parse", "HEAD"),
        "task_start_git_status_short": list(RECORDED_TASK_START_GIT_STATUS),
        "task_start_git_status_source": "captured_before_task_edits",
        "artifact_generation_start_git_status_short": before_status.splitlines(),
        "prerequisite_checkpoint_committed_at_task_start": False,
        "prerequisite_untracked_paths": [
            "lower_limb_sim/formal_artifacts/safeguarded_sequential_initial_identification_v1/",
            "lower_limb_sim/run_safeguarded_sequential_initial_identification.py",
            "lower_limb_sim/safeguarded_sequential_initial_identification.py",
            "lower_limb_sim/test_safeguarded_sequential_initial_identification.py",
        ],
        "ds_store_excluded_from_task": True,
        "analysis_case_count": len(ANALYSIS_CASES),
        "analysis_cases": [
            {
                "subject_id": subject,
                "scenario_name": scenario,
                "evidence_role": role,
            }
            for subject, scenario, role in ANALYSIS_CASES
        ],
        "diagnostic_trials_executed_per_case": MAX_INITIAL_IDENTIFICATION_TRIALS,
        "diagnostic_five_trial_rollout_reason": "marginal_gain_analysis_only",
        "formal_theta_hat_0_count": int(summary["theta_hat_0_frozen"].sum()),
        "personalization_prerequisite_pass_count": int(
            summary["personalization_prerequisite"].sum()
        ),
        "parameter_identifiability_threshold_status": PARAMETER_IDENTIFIABILITY_THRESHOLD_REQUIRES_REVIEW,
        "model_adequacy_threshold_status": MODEL_ADEQUACY_THRESHOLD_REQUIRES_REVIEW,
        "legacy_0p20_nm_status": "RESEARCH_ONLY_UNJUSTIFIED_NOT_FORMAL",
        "legacy_0p20_nm_changed_to_0p45": False,
        "candidate_threshold_selected_or_frozen": False,
        "candidate_false_accept_false_reject_computable": False,
        "matched_trial2_candidate_envelope": matched_envelope,
        "diagnostic_state_lens_candidate_id": (
            "separated_gate_trial2_diagnostic_envelope"
        ),
        "validation_mechanical_objective_definition": (
            "duration_weighted_mean_of_per_trajectory_J; actual trajectory torque metrics are each validation reference"
        ),
        "validation_mechanical_objective_status": (
            "PRECURSOR_ONLY_NOT_GLOBAL_RELIABILITY_RULE"
        ),
        "truth_parameters_used_by_decision": False,
        "heldout_final_test_used": False,
        "heldout_final_test_files_read": False,
        "heldout_final_test_trajectories_generated": False,
        "estimator_modified": False,
        "excitation_selection_modified": False,
        "mechanical_objective_modified": False,
        "hardware_or_safety_modified": False,
        "real_robot_connected": False,
        "explore_exploit_personalization_executed": False,
        "artifact_sha256": {
            path.name: _sha256(path) for path in artifact_paths
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(destination / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    metadata = run(args.output_dir)
    print(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
