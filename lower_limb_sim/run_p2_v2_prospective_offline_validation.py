"""Generate the pre-registered P2 V2 prospective offline validation artifacts."""

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
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rehab_robot_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
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
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_v2_prospective_offline_validation import (
    DEVELOPMENT_CASES,
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    FINAL_INSUFFICIENT,
    FINAL_REJECTS,
    FINAL_SUPPORTS,
    GLOBAL_MODEL_RELIABILITY_STATUS,
    HELD_OUT_FINAL_TEST,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    LOCAL_RESULTS_PATH,
    MANIFEST_ID,
    MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_REQUIRES_REVISION,
    P2_V2_DEFAULT_ENABLED,
    PAIR_PLAN_PATH,
    POLICY_VARIANTS,
    PROTOCOL_ID,
    FrozenManifestGate,
    audit_bundle_uncertainty,
    build_prospective_manifest,
    classify_final_status,
    dynamic_subject_for_id,
    evaluate_full_truth_landscape,
    post_policy_local_truth_audit,
    prospective_case_rows,
    registered_prospective_subject,
    run_prospective_policy,
    small_step_accumulation_audit,
    stable_manifest_sha256,
    validate_frozen_local_evidence,
)
from .research_decision_guarded_sequential_personalization import (
    STOP_MAX_PERSONALIZATION_TRIALS,
    build_initial_research_state,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_v2_prospective_offline_validation.py"
P2_V1_SOURCE_PATH = MODULE_DIR / "research_decision_guarded_sequential_personalization.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_v2_prospective_offline_validation_v1"
)

MANIFEST_FILENAME = "P2_V2_PROSPECTIVE_EVALUATION_MANIFEST_V1.json"
CSV_FILENAMES = (
    "prospective_case_manifest.csv",
    "designated_local_validation_provenance.csv",
    "bundle_uncertainty_audit.csv",
    "prospective_trial_history.csv",
    "prospective_policy_summary.csv",
    "guard_prospective_comparison.csv",
    "cumulative_prospective_comparison.csv",
    "stopping_prospective_comparison.csv",
    "prospective_false_improvement_audit.csv",
    "prospective_missed_improvement_audit.csv",
    "prospective_trial_efficiency.csv",
    "prospective_subject_specificity.csv",
    "prospective_failure_mode_audit.csv",
    "prospective_exploration_value_history.csv",
)
REPORT_FILENAMES = (
    "PROSPECTIVE_VALIDATION_REPORT.md",
    "DATA_SPLIT_AND_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "prospective_trial_efficiency.png",
    "prospective_guard_outcomes.png",
    "prospective_final_J_and_regret.png",
)


def _json_safe(value: Any) -> Any:
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _atomic_write_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    canonical = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    expected = stable_manifest_sha256(payload)
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise RuntimeError("manifest canonical hash is not deterministic")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(path) != expected:
        raise RuntimeError("frozen prospective manifest hash mismatch")
    return expected


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return completed.stdout.rstrip("\n")


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _protected_source_hashes() -> dict[str, str]:
    paths = {
        "active_reference": ACTIVE_REFERENCE_PATH,
        "formal_protocol": MODULE_DIR / "formal_protocol.py",
        "generator": MODULE_DIR / "continuous_reference_neighborhood.py",
        "mechanical_objective": MODULE_DIR / "mechanical_objective.py",
        "five_parameter_estimator": MODULE_DIR / "parameter_estimator.py",
        "P2_V1_core": P2_V1_SOURCE_PATH,
    }
    output = {
        name: sha256_file(path) for name, path in paths.items()
    }
    for directory_name in ("hardware", "control", "collection", "safety"):
        output[f"tree:{directory_name}"] = _tree_sha256(PROJECT_ROOT / directory_name)
    return output


def _result_summary_with_truth(
    summary: Mapping[str, Any],
    rounds: pd.DataFrame,
    truth_landscape: pd.DataFrame,
) -> dict[str, Any]:
    output = dict(summary)
    final_key = (
        round(float(output["final_best_alpha_hip"]), 12),
        round(float(output["final_best_alpha_knee"]), 12),
        round(float(output["final_best_alpha_phase"]), 12),
    )
    by_key = {
        (
            round(float(row["hip_delta"]), 12),
            round(float(row["knee_delta"]), 12),
            round(float(row["phase_delta"]), 12),
        ): row
        for row in truth_landscape.to_dict(orient="records")
    }
    selected_truth = float(by_key[final_key]["J_truth"])
    truth_best = truth_landscape.sort_values(
        ["J_truth", "trajectory_id"], kind="mergesort"
    ).iloc[0]
    local_keys = [final_key]
    for delta in (
        (GRID_HIP_STEP_DEG, 0.0, 0.0),
        (-GRID_HIP_STEP_DEG, 0.0, 0.0),
        (0.0, GRID_KNEE_STEP_DEG, 0.0),
        (0.0, -GRID_KNEE_STEP_DEG, 0.0),
        (0.0, 0.0, GRID_PHASE_STEP),
        (0.0, 0.0, -GRID_PHASE_STEP),
    ):
        key = tuple(round(final_key[index] + delta[index], 12) for index in range(3))
        if key in by_key:
            local_keys.append(key)
    local_best = min(float(by_key[key]["J_truth"]) for key in local_keys)
    natural_stop = output["stop_reason"] != STOP_MAX_PERSONALIZATION_TRIALS
    stop_round = rounds.sort_values("iteration").iloc[-1] if not rounds.empty else None
    true_at_stop = bool(stop_round["true_local_improvement_available"]) if natural_stop and stop_round is not None else False
    output.update(
        {
            "evaluated_decision_rounds": int(len(rounds)),
            "missed_improvement_rounds": int(rounds["missed_improvement_round"].sum()) if not rounds.empty else 0,
            "premature_conservative_stops": int(natural_stop and true_at_stop),
            "correct_local_stops": int(natural_stop and not true_at_stop),
            "final_selected_J_truth": selected_truth,
            "truth_optimum_trajectory_id": str(truth_best["trajectory_id"]),
            "truth_optimum_J": float(truth_best["J_truth"]),
            "truth_optimum_alpha_hip": float(truth_best["hip_delta"]),
            "truth_optimum_alpha_knee": float(truth_best["knee_delta"]),
            "truth_optimum_alpha_phase": float(truth_best["phase_delta"]),
            "global_truth_regret": max(selected_truth - float(truth_best["J_truth"]), 0.0),
            "final_local_regret": max(selected_truth - local_best, 0.0),
            "post_policy_truth_fed_back": False,
        }
    )
    return output


def _boundary_status(row: Mapping[str, Any], prefix: str) -> bool:
    values = {
        "hip_amplitude_delta_deg": float(row[f"{prefix}_alpha_hip"]),
        "knee_amplitude_delta_deg": float(row[f"{prefix}_alpha_knee"]),
        "knee_phase_shift": float(row[f"{prefix}_alpha_phase"]),
    }
    return any(
        np.isclose(value, bounds[0], atol=1e-12, rtol=0.0)
        or np.isclose(value, bounds[1], atol=1e-12, rtol=0.0)
        for (name, value), bounds in zip(
            values.items(),
            (OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name] for name in values),
        )
    )


def _subject_specificity(summary: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    matched_ids = set(
        cases.loc[cases["case_class"].eq("PROSPECTIVE_MATCHED"), "case_id"].astype(str)
    )
    selected = summary.loc[summary["case_id"].astype(str).isin(matched_ids)].copy()
    rows: list[dict[str, Any]] = []
    for item in selected.to_dict(orient="records"):
        row = {
            "case_id": item["case_id"],
            "subject_id": item["subject_id"],
            "policy_id": item["policy_id"],
            "truth_alpha_hip": item["truth_optimum_alpha_hip"],
            "truth_alpha_knee": item["truth_optimum_alpha_knee"],
            "truth_alpha_phase": item["truth_optimum_alpha_phase"],
            "truth_optimum_J": item["truth_optimum_J"],
            "final_alpha_hip": item["final_best_alpha_hip"],
            "final_alpha_knee": item["final_best_alpha_knee"],
            "final_alpha_phase": item["final_best_alpha_phase"],
            "global_truth_regret": item["global_truth_regret"],
            "objective_modified": False,
            "truth_used_to_modify_policy": False,
        }
        truth_boundary = _boundary_status(row, "truth")
        final_boundary = _boundary_status(row, "final")
        row["truth_optimum_on_generator_boundary"] = truth_boundary
        row["final_alpha_on_generator_boundary"] = final_boundary
        row["boundary_classification"] = (
            "OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM"
            if truth_boundary
            else (
                "POLICY_INDUCED_BOUNDARY_COLLAPSE"
                if final_boundary
                else "INTERIOR_TRUTH_AND_POLICY"
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_comparisons(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    guard = (
        summary.groupby(["policy_id", "guard_id"], as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            executed_trials=("number_of_executed_trials", "sum"),
            missed_improvement_rounds=("missed_improvement_rounds", "sum"),
            executed_false_improvements=("number_of_executed_false_improvements", "sum"),
            mean_final_J=("final_best_actual_J", "mean"),
            mean_global_regret=("global_truth_regret", "mean"),
        )
    )
    stopping = (
        summary.groupby(["policy_id", "stopping_rule_id", "stopping_k"], dropna=False, as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            total_trials=("number_of_executed_trials", "sum"),
            explore_trials=("number_of_explore_trials", "sum"),
            low_decision_value_explorations=("low_decision_value_exploration_count", "sum"),
            mean_final_J=("final_best_actual_J", "mean"),
            mean_global_regret=("global_truth_regret", "mean"),
        )
    )
    return guard, stopping


def _trial_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    v1 = summary.loc[summary["policy_id"].eq("P2_V1_G0_C0_S0")].set_index("case_id")
    variants = (
        "P2_V2A_G2_C0_S2",
        "P2_V2A_G3_C0_S2_SENSITIVITY",
        "P2_V2A_G2_C0_S1_SENSITIVITY",
        "P2_V2A_G2_C0_S3_SENSITIVITY",
    )
    rows: list[dict[str, Any]] = []
    for policy_id in variants:
        table = summary.loc[summary["policy_id"].eq(policy_id)].set_index("case_id")
        for case_id in sorted(v1.index.intersection(table.index)):
            left, right = v1.loc[case_id], table.loc[case_id]
            rows.append(
                {
                    "case_id": case_id,
                    "comparison_policy_id": policy_id,
                    "trials_saved_vs_P2_V1": int(left["number_of_executed_trials"] - right["number_of_executed_trials"]),
                    "explore_trials_saved_vs_P2_V1": int(left["number_of_explore_trials"] - right["number_of_explore_trials"]),
                    "final_J_difference_vs_P2_V1": float(right["final_best_actual_J"] - left["final_best_actual_J"]),
                    "global_regret_difference_vs_P2_V1": float(right["global_truth_regret"] - left["global_truth_regret"]),
                    "support_growth_difference_vs_P2_V1": int(right["known_region_growth"] - left["known_region_growth"]),
                    "support_reduction_interpreted_as_performance_loss": False,
                }
            )
    rows.append(
        {
            "case_id": "ALL_PROSPECTIVE_CASES",
            "comparison_policy_id": "P2_V2B_G2_C2_S2",
            "trials_saved_vs_P2_V1": np.nan,
            "explore_trials_saved_vs_P2_V1": np.nan,
            "final_J_difference_vs_P2_V1": np.nan,
            "global_regret_difference_vs_P2_V1": np.nan,
            "support_growth_difference_vs_P2_V1": np.nan,
            "support_reduction_interpreted_as_performance_loss": False,
            "status": "NOT_ACTIVE_BUNDLE_UNCERTAINTY_NOT_CALIBRATED",
        }
    )
    return pd.DataFrame(rows)


def _failure_modes(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    cumulative: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in summary.to_dict(orient="records"):
        selected = history.loc[
            history["case_id"].eq(item["case_id"])
            & history["policy_id"].eq(item["policy_id"])
        ].sort_values("iteration")
        exploit = selected.loc[selected["trial_purpose"].eq("EXPLOIT")]
        explore = selected.loc[selected["trial_purpose"].eq("EXPLORE")]

        def oscillates(table: pd.DataFrame) -> bool:
            if len(table) < 3:
                return False
            values = table[["alpha_hip", "alpha_knee", "alpha_phase"]].to_numpy(float)
            changes = np.diff(values, axis=0)
            return bool(np.any(changes[1:] * changes[:-1] < 0.0))

        checks = {
            "NEW_FALSE_IMPROVEMENT": int(item["number_of_executed_false_improvements"]) > 0,
            "OSCILLATORY_EXPLOIT": oscillates(exploit),
            "OSCILLATORY_EXPLORE": oscillates(explore),
            "PREMATURE_CONSERVATIVE_STOP": int(item["premature_conservative_stops"]) > 0,
            "EXCESSIVE_EXPLORATION": int(item["number_of_explore_trials"]) >= int(0.75 * MAXIMUM_RESEARCH_DIAGNOSTIC_HORIZON),
            "SUPPORT_COLLAPSE": int(item["final_supported_point_count"]) < int(item["initial_supported_point_count"]),
            "MODEL_UPDATE_INSTABILITY": str(item["stop_reason"]) == "STOP_MODEL_UPDATE_FAILURE",
            "CUMULATIVE_RULE_DRIFT": False,
            "PREMATURE_CUMULATIVE_COMMITMENT": False,
        }
        for failure_mode, observed in checks.items():
            rows.append(
                {
                    "case_id": item["case_id"],
                    "policy_id": item["policy_id"],
                    "failure_mode": failure_mode,
                    "observed": observed,
                    "policy_modified_in_response": False,
                    "future_revision_required": bool(observed),
                }
            )
    return pd.DataFrame(rows)


def _plot_artifacts(summary: pd.DataFrame, output: Path) -> None:
    order = [spec.policy_variant_id for spec in POLICY_VARIANTS]
    grouped = summary.groupby("policy_id", as_index=True).mean(numeric_only=True).reindex(order)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(np.arange(len(grouped)), grouped["number_of_executed_trials"], color="#3b82f6")
    axis.set_xticks(np.arange(len(grouped)), [value.replace("P2_", "") for value in grouped.index], rotation=25, ha="right")
    axis.set(ylabel="mean executed trials", title="Prospective trial efficiency (pre-registered variants)")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[0], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(grouped))
    axis.bar(x - 0.18, grouped["missed_improvement_rounds"], width=0.36, label="missed rounds")
    axis.bar(x + 0.18, grouped["number_of_executed_false_improvements"], width=0.36, label="false improvements")
    axis.set_xticks(x, [value.replace("P2_", "") for value in grouped.index], rotation=25, ha="right")
    axis.set(title="Prospective guard outcomes", ylabel="mean count per case")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[1], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - 0.18, grouped["final_best_actual_J"], width=0.36, label="final best actual J")
    axis.bar(x + 0.18, grouped["global_truth_regret"], width=0.36, label="global truth regret")
    axis.set_xticks(x, [value.replace("P2_", "") for value in grouped.index], rotation=25, ha="right")
    axis.set(title="Prospective outcome and post-policy regret")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[2], dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{float(value):.9f}"


def _report(
    manifest_sha: str,
    cases: pd.DataFrame,
    summary: pd.DataFrame,
    guard: pd.DataFrame,
    stopping: pd.DataFrame,
    cumulative: pd.DataFrame,
    specificity: pd.DataFrame,
    failures: pd.DataFrame,
    final_status: str,
) -> str:
    matched = int(cases["case_class"].eq("PROSPECTIVE_MATCHED").sum())
    mismatch = int(cases["case_class"].eq("PROSPECTIVE_MODEL_MISMATCH").sum())
    primary = guard.set_index("policy_id")
    v1 = primary.loc["P2_V1_G0_C0_S0"]
    v2 = primary.loc["P2_V2A_G2_C0_S2"]
    p99 = primary.loc["P2_V2A_G3_C0_S2_SENSITIVITY"]
    small = int(cumulative["small_step_accumulation_case"].sum()) if not cumulative.empty else 0
    observed_failures = failures.loc[failures["observed"].astype(bool)]
    stop_lines = []
    for row in stopping.to_dict(orient="records"):
        stop_lines.append(
            f"- `{row['policy_id']}`: total trials {int(row['total_trials'])}, "
            f"EXPLORE {int(row['explore_trials'])}, mean final J {_fmt(row['mean_final_J'])}."
        )
    subject_truth_count = int(
        specificity.loc[
            specificity["policy_id"].eq("P2_V1_G0_C0_S0"),
            ["truth_alpha_hip", "truth_alpha_knee", "truth_alpha_phase"],
        ].drop_duplicates().shape[0]
    )
    return f"""# P2 V2 prospective offline validation report

## Frozen boundary

- Protocol: `{PROTOCOL_ID}`.
- Manifest SHA-256: `{manifest_sha}`. The manifest was atomically persisted before any prospective identification, validation, personalization, or post-policy truth call.
- Cohort: {matched} new matched cases and {mismatch} new model-mismatch cases. All {len(DEVELOPMENT_CASES)} named development cases are excluded from primary metrics; held-out final test data were not read.
- Active reference SHA, ROM, shank-angle convention, five-parameter model, objective, generator bounds, 0.005 tolerance, and 90% support gate remained frozen.

## Primary P95 result

Compared with P2 V1, P2 V2A (frozen local P95 + C0 + K=2) changed total missed-improvement rounds from {int(v1['missed_improvement_rounds'])} to {int(v2['missed_improvement_rounds'])}, executed false improvements from {int(v1['executed_false_improvements'])} to {int(v2['executed_false_improvements'])}, mean final J from {_fmt(v1['mean_final_J'])} to {_fmt(v2['mean_final_J'])}, and mean global regret from {_fmt(v1['mean_global_regret'])} to {_fmt(v2['mean_global_regret'])}. Prospective outcomes did not update P95.

## P99 sensitivity

The frozen P99 sensitivity produced {int(p99['missed_improvement_rounds'])} missed rounds, {int(p99['executed_false_improvements'])} executed false improvements, mean final J {_fmt(p99['mean_final_J'])}, and mean global regret {_fmt(p99['mean_global_regret'])}. P95/P99 was not reselected from these results.

## Bundle and small-step accumulation

C2, C3, and C5 remain `SHADOW_ONLY_NOT_CALIBRATED`; the designated plan contains no pre-registered bundle residuals, and neither n-times nor square-root-n aggregation was assumed. P2 V2B was not executed. The post-policy audit found {small} pre-registered small-step accumulation paths; they remain descriptive and cannot activate a cumulative policy.

## Decision-value stopping

{chr(10).join(stop_lines)}

Support reduction is reported as provenance reduction, not automatically as performance loss. The stopping decisions used only already-observed model/prediction/guard/information/support signals and no future truth, exploit, or best trajectory.

## Subject specificity and boundaries

The {matched} matched subjects produced {subject_truth_count} distinct objective truth optima. Every truth-boundary row is separately labelled `OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM`; a policy-only boundary is labelled `POLICY_INDUCED_BOUNDARY_COLLAPSE`. The objective was not modified.

## Failure modes

Observed failure-mode rows: {len(observed_failures)}. They are recorded in `prospective_failure_mode_audit.csv`; none caused tuning during this experiment. Any observed row is a future revision question only.

## Formal conclusion

`{final_status}`

This is synthetic offline research evidence only. P2 V2 remains default-off. Initial-identification acceptance still requires review, global model reliability is not frozen for humans, and the result remains `{NOT_HUMAN_READY}` and `{NOT_ROBOT_MOTION_APPROVED}`.
"""


def _leakage_report(manifest_sha: str, case_count: int) -> str:
    return f"""# Data split and leakage audit

- The immutable prospective manifest SHA is `{manifest_sha}`.
- DEVELOPMENT contains the {len(DEVELOPMENT_CASES)} predeclared prior cases and contributes no primary prospective metric.
- PROSPECTIVE contains {case_count} newly pre-registered case IDs selected without truth outcomes.
- HELD_OUT_FINAL_TEST status is `{HELD_OUT_FINAL_TEST}`; no held-out loader is called by this runner.
- Frozen local P95/P99 come only from the 324-row designated artifact; no prospective case contributes to calibration.
- Every prospective policy is rerun from fresh initial identification. No development execution history is reused.
- Proposal and ranking complete before one selection token is issued; only then can the virtual execution oracle reveal truth.
- Full truth landscapes and missed-improvement labels are computed only after a policy path is complete and are never fed back.
- P2 V2B is absent because bundle-scale uncertainty is not calibrated. No n-times or square-root-n assumption is used.
- Prospective outcomes did not change a threshold, K, bundle length, objective, generator, model, support gate, or equivalence tolerance.
- No hardware, control, collection, safety, or robot connection code is imported or called.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
    *,
    prospective_start_commit_sha: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    start_sha = prospective_start_commit_sha or _git_output("rev-parse", "HEAD")
    if not start_sha or start_sha == "UNAVAILABLE":
        raise RuntimeError("prospective start commit SHA is unavailable")
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference SHA changed")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("formal ROM protocol changed")
    if tuple(FORMAL_HIP_ROM_DEG) != (0.0, 120.0) or tuple(FORMAL_KNEE_ROM_DEG) != (5.0, 145.0):
        raise RuntimeError("formal ROM values changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("equivalence tolerance changed")

    provenance, local_metrics = validate_frozen_local_evidence()
    pair_plan = pd.read_csv(PAIR_PLAN_PATH)
    bundle = audit_bundle_uncertainty(pair_plan)
    if bool(bundle.loc[bundle["bundle_length"].gt(1), "active_prospective_policy"].any()):
        raise RuntimeError("uncalibrated cumulative policy was activated")
    cases = prospective_case_rows()
    raw_map = pd.read_csv(parameter_map_path)
    lattice = geometrically_valid_parameter_lattice(raw_map)
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometrically admissible lattice changed")
    cache = build_trajectory_component_cache(lattice)

    protected_before = _protected_source_hashes()
    manifest_payload = build_prospective_manifest(
        start_sha, protected_source_sha256=protected_before
    )
    manifest_path = output / MANIFEST_FILENAME
    manifest_sha = _atomic_write_manifest(manifest_path, manifest_payload)
    manifest_frozen_at = datetime.now(timezone.utc).isoformat()
    gate = FrozenManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    # These provenance artifacts are written before any new prospective truth.
    _write_csv(output / "prospective_case_manifest.csv", cases)
    _write_csv(output / "designated_local_validation_provenance.csv", provenance)
    _write_csv(output / "bundle_uncertainty_audit.csv", bundle)

    results = []
    state_by_result: dict[tuple[str, str], Any] = {}
    local_candidates: list[pd.DataFrame] = []
    local_rounds: list[pd.DataFrame] = []
    landscapes: dict[str, pd.DataFrame] = {}
    cumulative_frames: list[pd.DataFrame] = []
    for case in cases.to_dict(orient="records"):
        subject = dynamic_subject_for_id(str(case["subject_id"]))
        with registered_prospective_subject(subject):
            case_results = []
            for spec in POLICY_VARIANTS:
                # Manifest gate opens before the first initial-ID truth and the
                # unchanged ID selection protocol then selects before execute.
                gate.record_truth_access()
                state = build_initial_research_state(
                    str(case["subject_id"]), str(case["scenario_name"])
                )
                result = run_prospective_policy(state, spec, lattice, cache, gate)
                results.append(result)
                case_results.append((result, state))
                state_by_result[(str(case["case_id"]), spec.policy_variant_id)] = state
            # Post-policy analyses start only after every variant for this case
            # has completed, preventing a truth landscape from reaching a later
            # policy implementation.
            first_result, first_state = case_results[0]
            gate.record_truth_access()
            landscape = evaluate_full_truth_landscape(first_result, first_state, cache)
            landscapes[str(case["case_id"])] = landscape
            cumulative_frames.append(
                small_step_accumulation_audit(landscape, str(case["case_id"]))
            )
            for result, state in case_results:
                gate.record_truth_access()
                candidates, rounds = post_policy_local_truth_audit(result, state, cache)
                if not candidates.empty:
                    local_candidates.append(candidates)
                if not rounds.empty:
                    local_rounds.append(rounds)

    history = pd.concat(
        [result.trial_history for result in results], ignore_index=True, sort=False
    )
    false_audit = pd.concat(
        [result.false_improvement_audit for result in results],
        ignore_index=True,
        sort=False,
    )
    exploration_frames = [
        result.exploration_information_gain
        for result in results
        if not result.exploration_information_gain.empty
    ]
    exploration = (
        pd.concat(exploration_frames, ignore_index=True, sort=False)
        if exploration_frames
        else pd.DataFrame()
    )
    missed_candidates = (
        pd.concat(local_candidates, ignore_index=True, sort=False)
        if local_candidates
        else pd.DataFrame()
    )
    rounds = (
        pd.concat(local_rounds, ignore_index=True, sort=False)
        if local_rounds
        else pd.DataFrame()
    )
    summary_rows = []
    for result in results:
        case_rounds = rounds.loc[
            rounds["case_id"].eq(result.summary["case_id"])
            & rounds["policy_id"].eq(result.policy_id)
        ]
        summary_rows.append(
            _result_summary_with_truth(
                result.summary,
                case_rounds,
                landscapes[str(result.summary["case_id"])],
            )
        )
    summary = pd.DataFrame(summary_rows)
    cumulative = pd.concat(cumulative_frames, ignore_index=True, sort=False)
    guard, stopping = _policy_comparisons(summary)
    efficiency = _trial_efficiency(summary)
    specificity = _subject_specificity(summary, cases)
    failures = _failure_modes(summary, history, cumulative)
    final_status = classify_final_status(summary)

    _write_csv(output / "prospective_trial_history.csv", history)
    _write_csv(output / "prospective_policy_summary.csv", summary)
    _write_csv(output / "guard_prospective_comparison.csv", guard)
    _write_csv(output / "cumulative_prospective_comparison.csv", cumulative)
    _write_csv(output / "stopping_prospective_comparison.csv", stopping)
    _write_csv(output / "prospective_false_improvement_audit.csv", false_audit)
    _write_csv(output / "prospective_missed_improvement_audit.csv", missed_candidates)
    _write_csv(output / "prospective_trial_efficiency.csv", efficiency)
    _write_csv(output / "prospective_subject_specificity.csv", specificity)
    _write_csv(output / "prospective_failure_mode_audit.csv", failures)
    _write_csv(output / "prospective_exploration_value_history.csv", exploration)
    _plot_artifacts(summary, output)
    (output / "PROSPECTIVE_VALIDATION_REPORT.md").write_text(
        _report(
            manifest_sha,
            cases,
            summary,
            guard,
            stopping,
            cumulative,
            specificity,
            failures,
            final_status,
        ),
        encoding="utf-8",
    )
    (output / "DATA_SPLIT_AND_LEAKAGE_AUDIT.md").write_text(
        _leakage_report(manifest_sha, len(cases)), encoding="utf-8"
    )

    protected_after = _protected_source_hashes()
    if protected_after != protected_before:
        raise RuntimeError("protected source changed during prospective validation")
    artifact_names = [
        MANIFEST_FILENAME,
        *CSV_FILENAMES,
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
    ]
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "manifest_id": MANIFEST_ID,
        "prospective_manifest_sha256": manifest_sha,
        "manifest_frozen_at_utc": manifest_frozen_at,
        "first_truth_after_manifest_freeze": True,
        "prospective_start_commit_sha": start_sha,
        "git_branch": _git_output("branch", "--show-current"),
        "git_log_at_start": _git_output("log", "--oneline", "-10").splitlines(),
        "checkpoint_preflight_status": "PASSED_CLEAN_BEFORE_PROSPECTIVE_IMPLEMENTATION",
        "case_count": len(cases),
        "matched_case_count": int(cases["case_class"].eq("PROSPECTIVE_MATCHED").sum()),
        "mismatch_case_count": int(cases["case_class"].eq("PROSPECTIVE_MODEL_MISMATCH").sum()),
        "policy_variant_count": len(POLICY_VARIANTS),
        "executed_case_policy_runs": len(results),
        "prospective_truth_gate_access_count": gate.truth_access_count,
        "local_uncertainty_metrics": local_metrics,
        "bundle_uncertainty_calibration": {
            "C2": "SHADOW_ONLY_NOT_CALIBRATED",
            "C3": "SHADOW_ONLY_NOT_CALIBRATED",
            "C5": "SHADOW_ONLY_NOT_CALIBRATED",
        },
        "P2_V2B_executed": False,
        "final_status": final_status,
        "offline_method_status": OFFLINE_METHOD_REQUIRES_REVISION,
        "P2_V2_default_enabled": P2_V2_DEFAULT_ENABLED,
        "human_readiness": NOT_HUMAN_READY,
        "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        "initial_identification_acceptance_rule": INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
        "global_model_reliability_rule": GLOBAL_MODEL_RELIABILITY_STATUS,
        "heldout_final_test": HELD_OUT_FINAL_TEST,
        "hardware_control_collection_safety_unchanged": True,
        "robot_connected": False,
        "prospective_outcome_used_to_tune_policy": False,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "artifacts": {
            name: {
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        },
        "runtime_seconds": time.perf_counter() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered P2 V2 prospective offline validation."
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    parser.add_argument("--prospective-start-commit-sha")
    args = parser.parse_args(argv)
    metadata = generate_artifacts(
        args.output_directory,
        args.parameter_map,
        prospective_start_commit_sha=args.prospective_start_commit_sha,
    )
    print(f"protocol: {metadata['protocol_id']}")
    print(f"manifest_sha256: {metadata['prospective_manifest_sha256']}")
    print(f"case_policy_runs: {metadata['executed_case_policy_runs']}")
    print(f"final_status: {metadata['final_status']}")
    print(f"runtime_seconds: {metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
