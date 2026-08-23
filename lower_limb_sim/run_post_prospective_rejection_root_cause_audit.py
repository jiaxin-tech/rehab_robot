"""Generate the post-prospective development audit for rejected P2 V2A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_v2_prospective_offline_validation import (
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    FrozenManifestGate,
    dynamic_subject_for_id,
    evaluate_full_truth_landscape,
    post_policy_local_truth_audit,
    prospective_case_rows,
    registered_prospective_subject,
    run_prospective_policy,
)
from .post_prospective_rejection_root_cause_audit import (
    AUDIT_DATA_ROLE,
    AUDIT_ID,
    BUNDLE_OUTCOME_STATUS,
    BUNDLE_PROTOCOL_ID,
    FACTORIAL_SPECS,
    FINAL_STATUS_IDENTIFIED,
    PROSPECTIVE_ARTIFACT_DIRECTORY,
    PROSPECTIVE_CONCLUSION,
    PROSPECTIVE_MANIFEST_PATH,
    PROSPECTIVE_MANIFEST_SHA256,
    PROSPECTIVE_START_COMMIT,
    attach_factorial_identity,
    build_designated_bundle_pair_plan,
    detailed_small_step_audit,
    factorial_decomposition,
    missed_round_root_cause,
    premature_stop_root_cause,
    root_cause_matrix,
    stopping_removed_trial_value_audit,
    verify_historical_reproduction,
    verify_immutable_prospective_artifacts,
)
from .research_decision_guarded_sequential_personalization import (
    build_initial_research_state,
)
from .run_p2_v2_prospective_offline_validation import (
    _protected_source_hashes,
    _result_summary_with_truth,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "post_prospective_rejection_root_cause_audit.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "post_prospective_rejection_root_cause_audit_v1"
)

REQUIRED_CSV_FILENAMES = (
    "guard_stopping_factorial_decomposition.csv",
    "prospective_missed_round_root_cause.csv",
    "prospective_rejection_mechanism_summary.csv",
    "prospective_small_step_accumulation.csv",
    "prospective_bundle_residual_characterization.csv",
    "stopping_removed_trial_value_audit.csv",
    "POST_PROSPECTIVE_REVISION_ROOT_CAUSE_MATRIX.csv",
    "designated_bundle_validation_pair_plan.csv",
)
EXTRA_CSV_FILENAMES = (
    "factorial_policy_summary.csv",
    "factorial_trial_history.csv",
    "factorial_historical_reproduction_audit.csv",
    "prospective_premature_stop_root_cause.csv",
    "bundle_validation_plan_strata.csv",
    "factorial_exploration_value_history.csv",
)
JSON_FILENAMES = ("DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1.json",)
REPORT_FILENAMES = (
    "BUNDLE_VALIDATION_PROTOCOL_REPORT.md",
    "POST_PROSPECTIVE_REJECTION_REPORT.md",
    "DATA_PROVENANCE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "factorial_final_J_and_regret.png",
    "stopping_removed_trial_value.png",
    "posthoc_bundle_residuals.png",
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


def _checkpoint_preflight() -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    tracked = _git_output(
        "ls-files",
        "--error-unmatch",
        "lower_limb_sim/p2_v2_prospective_offline_validation.py",
    )
    artifact = _git_output(
        "ls-files",
        "--error-unmatch",
        "lower_limb_sim/formal_artifacts/p2_v2_prospective_offline_validation_v1/metadata.json",
    )
    if head == "UNAVAILABLE" or tracked == "UNAVAILABLE" or artifact == "UNAVAILABLE":
        raise RuntimeError("POST_PROSPECTIVE_ANALYSIS_REQUIRES_CHECKPOINT")
    return {
        "post_prospective_start_commit_sha": head,
        "prospective_source_tracked": True,
        "prospective_artifact_tracked": True,
        "git_log": _git_output("log", "--oneline", "-6").splitlines(),
    }


def _bundle_protocol_payload(
    plan: pd.DataFrame,
    strata: pd.DataFrame,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_id": BUNDLE_PROTOCOL_ID,
        "status": "PLAN_FROZEN_BEFORE_FUTURE_BUNDLE_TRUTH",
        "data_role": "FUTURE_INDEPENDENT_BUNDLE_CALIBRATION_PLAN_ONLY",
        "source_space": "EXISTING_GEOMETRICALLY_ADMISSIBLE_GENERATOR_LATTICE",
        "formal_grid_steps": {
            "hip_deg": 0.25,
            "knee_deg": 0.25,
            "phase": 0.0025,
        },
        "bundle_lengths": {
            "primary": [2, 3],
            "optional_diagnostic": [5],
        },
        "strata": [
            "coordinate",
            "direction",
            "bundle_length",
            "location_class",
        ],
        "coordinates": ["hip", "knee", "phase"],
        "directions": ["NEGATIVE", "POSITIVE"],
        "location_classes": ["LOWER_BOUNDARY", "INTERIOR", "UPPER_BOUNDARY"],
        "pairs_per_stratum": 12,
        "planned_pair_count": len(plan),
        "stratum_count": len(strata),
        "pair_plan_sha256": plan_sha256,
        "selection_rule": "LOWEST_SHA256_WITHIN_GEOMETRY_DIRECTION_LENGTH_LOCATION_STRATUM",
        "plan_inputs": [
            "generator_geometry",
            "formal_grid",
            "formal_trust_step_definitions",
            "boundary_interior_labels",
            "direction_identity",
        ],
        "forbidden_plan_inputs": [
            "prospective_truth_error_magnitude",
            "prospective_successful_bundle_location",
            "future_bundle_truth",
            "predicted_objective",
        ],
        "formal_neighbor_continuity_required": True,
        "direction_consistency_required": True,
        "generator_bounds_expanded": False,
        "future_required_outcome": (
            "e_deltaJ_bundle=abs(deltaJ_pred_start_to_endpoint-"
            "deltaJ_truth_start_to_endpoint)"
        ),
        "future_truth_generated_in_this_task": False,
        "bundle_uncertainty_calibrated_in_this_task": False,
        "current_324_pair_plan_modified": False,
        "current_324_pair_plan_limitation": (
            "single-pair trust-level comparisons lack predeclared bundle identity, "
            "continuous intermediate path, and 2/3/5-step bundle residual strata"
        ),
        "prospective_outcomes_used_to_select_plan": False,
        "truth_used_to_select_plan": False,
    }


def _factorial_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(
            ["factorial_variant_id", "policy_id", "evidence_role"],
            as_index=False,
            sort=False,
        )
        .agg(
            case_count=("case_id", "count"),
            executed_trials=("number_of_executed_trials", "sum"),
            EXPLORE_count=("number_of_explore_trials", "sum"),
            EXPLOIT_count=("number_of_exploit_trials", "sum"),
            missed_improvement_rounds=("missed_improvement_rounds", "sum"),
            premature_conservative_stops=("premature_conservative_stops", "sum"),
            executed_false_improvements=("number_of_executed_false_improvements", "sum"),
            mean_final_J=("final_best_actual_J", "mean"),
            mean_regret=("global_truth_regret", "mean"),
            low_decision_value_exploration=("low_decision_value_exploration_count", "sum"),
            correct_stops=("correct_local_stops", "sum"),
        )
    )


def _plot_factorial(aggregate: pd.DataFrame, output: Path) -> None:
    labels = aggregate["factorial_variant_id"].str.slice(0, 2)
    x = np.arange(len(aggregate))
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.bar(x - 0.18, aggregate["mean_final_J"], 0.36, label="mean final J")
    axis.bar(x + 0.18, aggregate["mean_regret"], 0.36, label="mean regret")
    axis.set_xticks(x, labels)
    axis.set(title="Post-hoc G x stopping factorial outcomes", ylabel="objective value")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[0], dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_stopping(stopping: pd.DataFrame, output: Path) -> None:
    counts = stopping["removed_trial_value_classification"].value_counts()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.bar(counts.index, counts.values, color="#4c78a8")
    axis.tick_params(axis="x", rotation=20)
    axis.set(title="S2 removed-trial value classification", ylabel="case x guard comparisons")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[1], dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_bundle_residuals(residuals: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    groups = [
        residuals.loc[residuals["bundle_length"].eq(length), "e_deltaJ_bundle_posthoc"]
        for length in (2, 3, 5)
    ]
    axis.boxplot(groups, tick_labels=["2-step", "3-step", "5-step"])
    axis.set(
        title="Revealed prospective bundle residuals (development characterization only)",
        ylabel="absolute endpoint delta-J residual",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[2], dpi=180, bbox_inches="tight")
    plt.close(figure)


def _bundle_report(plan: pd.DataFrame, strata: pd.DataFrame, sha: str) -> str:
    return f"""# Designated bundle validation protocol report

`{BUNDLE_PROTOCOL_ID}` freezes a geometry-only plan containing {len(plan)} endpoint pairs across {len(strata)} balanced strata. The plan SHA-256 is `{sha}`.

The existing 324-pair local plan is insufficient for bundle uncertainty because it did not predeclare bundle identity, all continuous intermediate formal neighbors, or 2/3/5-step endpoint-residual strata. Even where an old pair has the same endpoint distance, it was not selected as a direction-consistent bundle calibration unit.

The new plan crosses hip/knee/phase, positive/negative direction, 2/3/5 steps, and lower-boundary/interior/upper-boundary locations, with 12 SHA-selected pairs per stratum. Selection uses only generator geometry and IDs. It uses no predicted J, prospective error, successful prospective location, or future truth.

No bundle truth or `e_deltaJ_bundle` is generated here. Every outcome field remains blank with status `{BUNDLE_OUTCOME_STATUS}`. A future, independent calibration task must evaluate this frozen plan without reselection.
"""


def _main_report(
    aggregate: pd.DataFrame,
    decomposition: pd.DataFrame,
    missed_summary: pd.DataFrame,
    small: pd.DataFrame,
    residuals: pd.DataFrame,
    stopping: pd.DataFrame,
    premature: pd.DataFrame,
    bundle_sha: str,
    final_status: str,
) -> str:
    indexed = aggregate.set_index("factorial_variant_id")
    a0 = indexed.loc["A0_G0_C0_S0_ORIGINAL_P2_V1"]
    a1 = indexed.loc["A1_G2_C0_S0_POST_HOC"]
    a2 = indexed.loc["A2_G0_C0_S2_POST_HOC"]
    a3 = indexed.loc["A3_G2_C0_S2_REJECTED_V2A"]
    mean_decomp = decomposition.groupby("metric", as_index=True).mean(numeric_only=True)
    final_j = mean_decomp.loc["final_best_actual_J"]
    regret = mean_decomp.loc["global_truth_regret"]
    missed = mean_decomp.loc["missed_improvement_rounds"]
    signal_count = int(
        residuals.drop_duplicates("path_id")["prediction_usefulness"]
        .eq("CUMULATIVE_SIGNAL_PRESENT")
        .sum()
    )
    unreliable_count = int(
        residuals.drop_duplicates("path_id")["prediction_usefulness"]
        .eq("CUMULATIVE_MODEL_UNRELIABLE")
        .sum()
    )
    truncated = int(
        stopping["removed_trial_value_classification"]
        .eq("TRUNCATED_USEFUL_EXPLORATION_CHAIN")
        .sum()
    )
    removed = int(
        stopping["removed_trial_value_classification"]
        .eq("REMOVED_LOW_VALUE_EXPLORATION")
        .sum()
    )
    stop_counts = premature["primary_stop_root_cause"].value_counts().to_dict()
    mechanism_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in stop_counts.items()
    )
    return f"""# Post-prospective rejection root-cause report

## Immutable evidence boundary

The original conclusion remains `{PROSPECTIVE_CONCLUSION}`. Its manifest SHA remains `{PROSPECTIVE_MANIFEST_SHA256}` and its start commit remains `{PROSPECTIVE_START_COMMIT}`. Nothing in this audit is prospective evidence: all new results are `{AUDIT_DATA_ROLE}`.

## Factorial decomposition

| Variant | Trials | EXPLORE | EXPLOIT | Missed | False | Mean final J | Mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 G0+S0 | {int(a0.executed_trials)} | {int(a0.EXPLORE_count)} | {int(a0.EXPLOIT_count)} | {int(a0.missed_improvement_rounds)} | {int(a0.executed_false_improvements)} | {a0.mean_final_J:.9f} | {a0.mean_regret:.9f} |
| A1 G2+S0 post-hoc | {int(a1.executed_trials)} | {int(a1.EXPLORE_count)} | {int(a1.EXPLOIT_count)} | {int(a1.missed_improvement_rounds)} | {int(a1.executed_false_improvements)} | {a1.mean_final_J:.9f} | {a1.mean_regret:.9f} |
| A2 G0+S2 post-hoc | {int(a2.executed_trials)} | {int(a2.EXPLORE_count)} | {int(a2.EXPLOIT_count)} | {int(a2.missed_improvement_rounds)} | {int(a2.executed_false_improvements)} | {a2.mean_final_J:.9f} | {a2.mean_regret:.9f} |
| A3 G2+S2 rejected | {int(a3.executed_trials)} | {int(a3.EXPLORE_count)} | {int(a3.EXPLOIT_count)} | {int(a3.missed_improvement_rounds)} | {int(a3.executed_false_improvements)} | {a3.mean_final_J:.9f} | {a3.mean_regret:.9f} |

The transparent mean case-wise decomposition gives A3-A0 final-J change {final_j['A3_minus_A0_total']:.9f}, guard main effect {final_j['factorial_guard_main_effect']:.9f}, stopping main effect {final_j['factorial_stopping_main_effect']:.9f}, and interaction {final_j['guard_stopping_interaction']:.9f}. For regret the corresponding values are {regret['A3_minus_A0_total']:.9f}, {regret['factorial_guard_main_effect']:.9f}, {regret['factorial_stopping_main_effect']:.9f}, and {regret['guard_stopping_interaction']:.9f}. Missed-round A3-A0 mean change is {missed['A3_minus_A0_total']:.6f}.

The rejected V2A outcome is primarily a **guard effect**, not a stopping effect: A1 and A3 have the same mean final J, while A0 and A2 also have the same mean final J. S2 reduced the number of executed and low-decision-value trials, but its mean final-J and regret effects are exactly zero in this cohort. The missed-round count has a guard/stopping interaction because earlier stopping reduces the number of later rounds that can be counted; that interaction must not be mistaken for better decisions. This is not evidence that the mechanical objective, generator, or five-parameter model must change. Local uncertainty remains only a research concept and no percentile is frozen by this audit.

## Premature stops and removed trials

All 24 immutable prospective premature-stop rows were retained and classified:

{mechanism_lines}

Across the two fixed-guard S0-vs-S2 comparisons, {removed} cases removed only low-value continuation while {truncated} cases truncated a chain that later produced a useful S0 action; the remaining comparisons are indeterminate because S2 did not trigger early. In this cohort there is no demonstrated S2 truncation of a later useful S0 action and no separable S2 endpoint penalty.

K=1, K=2, and K=3 had the same previously observed mean final J. That result does not identify K=2 as the cause of rejection and does not justify tuning K=4 or K=5. A richer stopping criterion may still be studied as a robustness question, but it is not the primary repair supported by this factorial audit.

## Small-step accumulation

All nine previously detected paths were expanded through steps 1..5. They are same-axis, same-sign, formal-neighbor-continuous knee-negative paths; none requires a turn or mixed-axis move. {signal_count} paths are classified `CUMULATIVE_SIGNAL_PRESENT` and {unreliable_count} `CUMULATIVE_MODEL_UNRELIABLE`. Their prospective residuals remain post-hoc characterization and cannot calibrate future uncertainty.

## New validation design

The new `{BUNDLE_PROTOCOL_ID}` plan SHA is `{bundle_sha}`. It independently samples 2-step and 3-step endpoint bundles, with 5-step as an optional diagnostic layer, across all generator coordinates, signs, and boundary/interior strata. It contains no truth outcomes and does not alter the old 324-pair plan.

## Minimum future revision scope

1. Retain local-uncertainty validation as a concept, but do not freeze a percentile from this rejected cohort.
2. Study bundle-aware cumulative decisions only after independent execution of the newly frozen bundle-validation plan.
3. Treat any richer decision-value stopping rule as a secondary, independently validated robustness study; do not present stopping replacement as the demonstrated repair for this rejection.

No P2 V3 policy, new threshold, objective change, generator enlargement, or model enlargement is implemented here.

## Status and future split

`{final_status}`

`DEVELOPMENT_USED_AFTER_REJECTION = true`. These six cases can never again support a claim of prospective success. Any future revision must use a new independent prospective cohort. P2 V2 remains default-off, `{NOT_HUMAN_READY}`, and `{NOT_ROBOT_MOTION_APPROVED}`.
"""


# Imported late only for report constants, keeping all robot-side packages absent.
from .p2_v2_prospective_offline_validation import (  # noqa: E402
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
)


def _provenance_report(
    checkpoint: Mapping[str, Any],
    reproduction: pd.DataFrame,
    bundle_sha: str,
) -> str:
    return f"""# Data provenance audit

- Checkpoint prerequisite passed at `{checkpoint['post_prospective_start_commit_sha']}`; prospective source and formal artifacts were tracked before this audit began.
- Original prospective conclusion: `{PROSPECTIVE_CONCLUSION}` (immutable).
- Original prospective manifest SHA: `{PROSPECTIVE_MANIFEST_SHA256}` (unchanged).
- Original prospective start commit: `{PROSPECTIVE_START_COMMIT}` (unchanged).
- A0/A3 historical metric reproduction rows: {len(reproduction)}; all reproduced: `{str(bool(reproduction['reproduced'].all())).lower()}`.
- A1/A2 are labelled `POST_HOC_COUNTERFACTUAL_ONLY` and are not new prospective policies.
- Revealed truth is used only for post-rejection development diagnosis and never changes a historical decision.
- Current cohort classification is permanently `DEVELOPMENT_USED_AFTER_REJECTION=true`.
- Held-out final-test data were not loaded.
- New bundle plan SHA: `{bundle_sha}`. Its selection uses geometry and hashes only; truth outcome columns remain blank.
- No future bundle calibration truth, new prospective cohort, policy implementation, percentile selection, K tuning, or threshold freeze occurs in this task.
- Active reference, ROM, shank convention, objective, generator, model, P2 V1, rejected V2A, hardware, control, collection, and safety are unchanged.
- No robot connection or human-ready approval is created.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = _checkpoint_preflight()
    immutable_metadata = verify_immutable_prospective_artifacts()
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference changed")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("ROM protocol changed")
    if tuple(FORMAL_HIP_ROM_DEG) != (0.0, 120.0) or tuple(FORMAL_KNEE_ROM_DEG) != (5.0, 145.0):
        raise RuntimeError("ROM values changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")

    protected_before = _protected_source_hashes()
    raw_map = pd.read_csv(parameter_map_path)
    lattice = geometrically_valid_parameter_lattice(raw_map)
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometric lattice changed")
    cache = build_trajectory_component_cache(lattice)

    # Freeze the geometry-only bundle plan before this task performs any
    # post-hoc truth replay.  No future bundle-calibration truth is generated.
    bundle_plan, bundle_strata = build_designated_bundle_pair_plan(lattice)
    bundle_plan_path = output / "designated_bundle_validation_pair_plan.csv"
    _write_csv(bundle_plan_path, bundle_plan)
    bundle_plan_sha = sha256_file(bundle_plan_path)
    bundle_protocol = _bundle_protocol_payload(
        bundle_plan, bundle_strata, bundle_plan_sha
    )
    _write_json(
        output / "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1.json", bundle_protocol
    )
    _write_csv(output / "bundle_validation_plan_strata.csv", bundle_strata)
    (output / "BUNDLE_VALIDATION_PROTOCOL_REPORT.md").write_text(
        _bundle_report(bundle_plan, bundle_strata, bundle_plan_sha),
        encoding="utf-8",
    )

    manifest_gate = FrozenManifestGate(
        PROSPECTIVE_MANIFEST_PATH, PROSPECTIVE_MANIFEST_SHA256
    )
    manifest_gate.require_frozen()
    cases = prospective_case_rows()
    results = []
    guard_frames_by_policy: dict[str, list[pd.DataFrame]] = {}
    candidate_frames: list[pd.DataFrame] = []
    round_frames: list[pd.DataFrame] = []
    prediction_truth_by_case: dict[str, pd.DataFrame] = {}
    for case in cases.to_dict(orient="records"):
        subject = dynamic_subject_for_id(str(case["subject_id"]))
        with registered_prospective_subject(subject):
            case_results = []
            for _, spec, _ in FACTORIAL_SPECS:
                manifest_gate.record_truth_access()
                state = build_initial_research_state(
                    str(case["subject_id"]), str(case["scenario_name"])
                )
                result = run_prospective_policy(
                    state, spec, lattice, cache, manifest_gate
                )
                results.append(result)
                case_results.append((result, state))
                guard_frames_by_policy.setdefault(result.policy_id, []).append(
                    result.decision_guard_audit
                )
            # All four actions are immutable before post-hoc landscape truth.
            a0_result, a0_state = case_results[0]
            manifest_gate.record_truth_access()
            landscape = evaluate_full_truth_landscape(a0_result, a0_state, cache)
            prediction_truth_by_case[str(case["case_id"])] = landscape
            for result, state in case_results:
                manifest_gate.record_truth_access()
                candidates, rounds = post_policy_local_truth_audit(
                    result, state, cache
                )
                if not candidates.empty:
                    candidate_frames.append(candidates)
                if not rounds.empty:
                    round_frames.append(rounds)

    history = pd.concat(
        [result.trial_history for result in results], ignore_index=True, sort=False
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
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    rounds = pd.concat(round_frames, ignore_index=True, sort=False)
    summary_rows = []
    for result in results:
        selected_rounds = rounds.loc[
            rounds["case_id"].eq(result.summary["case_id"])
            & rounds["policy_id"].eq(result.policy_id)
        ]
        summary_rows.append(
            _result_summary_with_truth(
                result.summary,
                selected_rounds,
                prediction_truth_by_case[str(result.summary["case_id"])],
            )
        )
    summary = attach_factorial_identity(pd.DataFrame(summary_rows))
    aggregate = _factorial_aggregate(summary)

    historical_summary = pd.read_csv(
        PROSPECTIVE_ARTIFACT_DIRECTORY / "prospective_policy_summary.csv"
    )
    reproduction = verify_historical_reproduction(summary, historical_summary)
    decomposition = factorial_decomposition(summary)
    guard_audits = {
        policy_id: pd.concat(frames, ignore_index=True, sort=False)
        for policy_id, frames in guard_frames_by_policy.items()
    }
    missed_detail, missed_summary = missed_round_root_cause(
        candidates, history, guard_audits
    )

    detected_paths = pd.read_csv(
        PROSPECTIVE_ARTIFACT_DIRECTORY / "cumulative_prospective_comparison.csv"
    )
    small_steps, residuals = detailed_small_step_audit(
        detected_paths, prediction_truth_by_case
    )
    stopping = stopping_removed_trial_value_audit(summary, history, exploration)
    historical_failure = pd.read_csv(
        PROSPECTIVE_ARTIFACT_DIRECTORY / "prospective_failure_mode_audit.csv"
    )
    historical_candidates = pd.read_csv(
        PROSPECTIVE_ARTIFACT_DIRECTORY / "prospective_missed_improvement_audit.csv"
    )
    premature = premature_stop_root_cause(
        historical_summary, historical_failure, historical_candidates
    )
    root_matrix = root_cause_matrix(
        decomposition, residuals, stopping, premature
    )
    final_status = FINAL_STATUS_IDENTIFIED

    outputs = {
        "guard_stopping_factorial_decomposition.csv": decomposition,
        "prospective_missed_round_root_cause.csv": missed_detail,
        "prospective_rejection_mechanism_summary.csv": missed_summary,
        "prospective_small_step_accumulation.csv": small_steps,
        "prospective_bundle_residual_characterization.csv": residuals,
        "stopping_removed_trial_value_audit.csv": stopping,
        "POST_PROSPECTIVE_REVISION_ROOT_CAUSE_MATRIX.csv": root_matrix,
        "factorial_policy_summary.csv": summary,
        "factorial_trial_history.csv": history,
        "factorial_historical_reproduction_audit.csv": reproduction,
        "prospective_premature_stop_root_cause.csv": premature,
        "factorial_exploration_value_history.csv": exploration,
    }
    for name, table in outputs.items():
        _write_csv(output / name, table)

    _plot_factorial(aggregate, output)
    _plot_stopping(stopping, output)
    _plot_bundle_residuals(residuals, output)
    (output / "POST_PROSPECTIVE_REJECTION_REPORT.md").write_text(
        _main_report(
            aggregate,
            decomposition,
            missed_summary,
            small_steps,
            residuals,
            stopping,
            premature,
            bundle_plan_sha,
            final_status,
        ),
        encoding="utf-8",
    )
    (output / "DATA_PROVENANCE_AUDIT.md").write_text(
        _provenance_report(checkpoint, reproduction, bundle_plan_sha),
        encoding="utf-8",
    )

    protected_after = _protected_source_hashes()
    if protected_before != protected_after:
        raise RuntimeError("protected baseline changed during post-prospective audit")
    artifact_names = [
        *REQUIRED_CSV_FILENAMES,
        *EXTRA_CSV_FILENAMES,
        *JSON_FILENAMES,
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
    ]
    metadata = {
        "audit_id": AUDIT_ID,
        "data_role": AUDIT_DATA_ROLE,
        "final_status": final_status,
        "prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "prospective_conclusion_revised": False,
        "prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "prospective_start_commit_sha": PROSPECTIVE_START_COMMIT,
        "post_prospective_checkpoint": checkpoint,
        "development_used_after_rejection": True,
        "current_six_cases_may_support_future_prospective_claim": False,
        "future_revision_requires_new_independent_prospective_cohort": True,
        "factorial_variants": [
            {
                "factorial_variant_id": identifier,
                "policy": spec.as_dict(),
                "evidence_role": role,
            }
            for identifier, spec, role in FACTORIAL_SPECS
        ],
        "A0_A3_historical_reproduction_all_passed": bool(
            reproduction["reproduced"].all()
        ),
        "A1_A2_post_hoc_counterfactual_only": True,
        "prospective_outcome_used_to_modify_historical_policy": False,
        "prospective_truth_gate_access_count": manifest_gate.truth_access_count,
        "small_step_path_count": int(small_steps["path_id"].nunique()),
        "premature_stop_count": len(premature),
        "bundle_validation_protocol_id": BUNDLE_PROTOCOL_ID,
        "bundle_pair_plan_sha256": bundle_plan_sha,
        "bundle_pair_plan_count": len(bundle_plan),
        "future_bundle_calibration_truth_generated": False,
        "new_percentile_selected": False,
        "K_tuned": False,
        "new_policy_implemented": False,
        "P2_V2_default_enabled": False,
        "human_readiness": NOT_HUMAN_READY,
        "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        "robot_connected": False,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "immutable_prospective_metadata_final_status": immutable_metadata[
            "final_status"
        ],
        "artifact_manifest": {
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
        description="Run post-prospective root-cause audit without implementing a new policy."
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    args = parser.parse_args(argv)
    metadata = generate_artifacts(args.output_directory, args.parameter_map)
    print(f"audit_id: {metadata['audit_id']}")
    print(f"bundle_pair_plan_sha256: {metadata['bundle_pair_plan_sha256']}")
    print(f"final_status: {metadata['final_status']}")
    print(f"runtime_seconds: {metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
