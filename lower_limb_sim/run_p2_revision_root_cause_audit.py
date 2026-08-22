"""Generate P2 revision root-cause diagnostics without changing P2.

This runner is offline-only.  It replays the frozen P2 implementation, then
attaches virtual truth strictly for post-hoc landscape and counterfactual
analysis.  It never imports robot packages or creates a human/robot approval.
"""

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

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
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
from .p2_revision_root_cause_audit import (
    AUDIT_PROTOCOL_ID,
    EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT,
    GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH,
    LOCAL_CALIBRATION_NOT_SUFFICIENT,
    LOCAL_PAIR_UNAVAILABLE,
    OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM,
    OFFLINE_METHOD_REQUIRES_REVISION,
    P2_POLICY_REVISION_JUSTIFIED,
    POLICY_COLLAPSES_SUBJECT_DIFFERENCES,
    POST_HOC_TRUTH_ROLE,
    POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION,
    SUPPORT_ONLY_EXPLORATION,
    SYNTHETIC_SCAN_ROLE,
    audit_subject_truth_landscape,
    build_counterfactual_guard_comparison,
    build_current_guard_uncertainty_provenance,
    build_exploration_value_decomposition,
    build_knee_stiff_exploration_audit,
    build_objective_normalization_audit,
    classify_root_causes,
    scan_registered_parameter_sensitivity,
    summarize_validation_pair_scales,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
    INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
    NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
    POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
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
    CORE_SOURCE_PATH as POLICY_CORE_SOURCE,
    DEFAULT_OUTPUT_DIRECTORY as POLICY_ARTIFACT_DIRECTORY,
    DEFAULT_PARAMETER_MAP_PATH,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)
from .sequential_personalization_convergence_stopping_audit import (
    EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
    audit_post_decision_local_truth,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_revision_root_cause_audit.py"
MECHANICAL_OBJECTIVE_SOURCE = MODULE_DIR / "mechanical_objective.py"
GENERATOR_SOURCE = MODULE_DIR / "continuous_reference_neighborhood.py"
ESTIMATOR_SOURCE = MODULE_DIR / "parameter_estimator.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_revision_root_cause_audit_v1"
)

TRUTH_SUBJECTS = ("baseline", "hip_stiff", "knee_stiff", "heavy_leg")
TRUTH_LANDSCAPE_FILENAMES = tuple(
    f"truth_landscape_{subject}.csv" for subject in TRUTH_SUBJECTS
)
CSV_FILENAMES = (
    *TRUTH_LANDSCAPE_FILENAMES,
    "truth_landscape_summary.csv",
    "truth_axis_profiles.csv",
    "subject_truth_local_sensitivity.csv",
    "objective_normalization_subject_effect.csv",
    "synthetic_subject_optimum_map.csv",
    "current_guard_uncertainty_provenance.csv",
    "global_vs_local_validation_uncertainty.csv",
    "counterfactual_guard_comparison.csv",
    "exploration_value_decomposition.csv",
    "knee_stiff_exploration_audit.csv",
    "P2_REVISION_ROOT_CAUSE_MATRIX.csv",
)
REPORT_FILENAMES = (
    "EXPLORATION_STOPPING_CANDIDATE_ANALYSIS.md",
    "P2_REVISION_RECOMMENDATION.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "subject_truth_landscape_comparison.png",
    "truth_hip_profile_by_subject.png",
    "truth_knee_profile_by_subject.png",
    "truth_phase_profile_by_subject.png",
    "truth_local_gradient_by_subject.png",
    "global_vs_local_validation_uncertainty.png",
    "counterfactual_guard_comparison.png",
    "exploration_value_decomposition.png",
    "knee_stiff_exploration_value.png",
    "root_cause_overview.png",
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


def _write_json(path: Path, payload: Any) -> None:
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
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return completed.stdout.rstrip("\n")


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    selected = [frame for frame in frames if not frame.empty]
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_landscape_comparison(
    landscapes: Mapping[str, pd.DataFrame], path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    selected_frames = {
        subject: table.loc[np.isclose(table["phase_delta"], 0.0)]
        for subject, table in landscapes.items()
    }
    low = min(float(frame["J_truth"].min()) for frame in selected_frames.values())
    high = max(float(frame["J_truth"].max()) for frame in selected_frames.values())
    scatter = None
    for axis, subject in zip(axes.flat, TRUTH_SUBJECTS):
        frame = selected_frames[subject]
        scatter = axis.scatter(
            frame["hip_delta"],
            frame["knee_delta"],
            c=frame["J_truth"],
            s=14,
            cmap="viridis_r",
            vmin=low,
            vmax=high,
        )
        best = frame.sort_values(["J_truth", "trajectory_id"]).iloc[0]
        axis.scatter(
            [best["hip_delta"]], [best["knee_delta"]], marker="X", s=90, color="red"
        )
        axis.set_title(f"{subject}, phase=0")
        axis.grid(alpha=0.2)
    figure.supxlabel("hip amplitude delta (deg)")
    figure.supylabel("knee amplitude delta (deg)")
    figure.suptitle("Post-hoc matched truth landscapes (not policy input)")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes.ravel().tolist(), label="J truth", shrink=0.8)
    _save(figure, path)


def _plot_profile(profiles: pd.DataFrame, axis_name: str, path: Path) -> None:
    selected = profiles.loc[profiles["profile_axis"].eq(axis_name)]
    figure, axis = plt.subplots(figsize=(9, 6))
    for subject, group in selected.groupby("subject_id", sort=False):
        ordered = group.sort_values("axis_value")
        axis.plot(ordered["axis_value"], ordered["J_truth"], "-o", ms=2.5, label=subject)
    label = {
        "hip_delta": "hip amplitude delta (deg)",
        "knee_delta": "knee amplitude delta (deg)",
        "phase_delta": "knee phase delta",
    }[axis_name]
    axis.axvline(0.0, color="black", lw=1, alpha=0.5)
    axis.set(xlabel=label, ylabel="J truth", title=f"Truth {axis_name} profile by subject")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path)


def _plot_gradients(sensitivity: pd.DataFrame, path: Path) -> None:
    columns = (
        ("dJ_d_hip_at_reference", "hip"),
        ("dJ_d_knee_at_reference", "knee"),
        ("dJ_d_phase_at_reference", "phase"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (column, label) in zip(axes, columns):
        axis.bar(sensitivity["subject_id"], sensitivity[column], color="#4c78a8")
        axis.axhline(0.0, color="black", lw=1)
        axis.tick_params(axis="x", rotation=30)
        axis.set_title(f"dJ/d{label} at reference")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Formal-grid finite differences (not physiological gradients)")
    _save(figure, path)


def _plot_validation_uncertainty(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    labels = summary["validation_pair_scale_class"].str.replace("_", " ").str.slice(0, 24)
    values = summary["max_e_delta_J"].fillna(0.0)
    colors = ["#8c6bb1" if count else "#cccccc" for count in summary["pair_instance_count"]]
    axis.bar(labels, values, color=colors)
    for index, row in summary.reset_index(drop=True).iterrows():
        axis.text(index, float(values.iloc[index]), f"n={int(row['pair_instance_count'])}", ha="center", va="bottom", fontsize=8)
    axis.set(ylabel="max pairwise delta-J error", title="Current guard evidence has no formal local-alpha validation pair")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _counterfactual_counts(counterfactual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for guard_id, group in counterfactual.groupby("guard_id", sort=False):
        available = group["would_exploit"].notna().any()
        rows.append(
            {
                "guard_id": guard_id,
                "available": available,
                "would_exploit_count": int(group["would_exploit"].eq(True).sum()),
                "false_improvement_count": int(group["false_improvement"].eq(True).sum()),
                "missed_improvement_count": int(group["missed_improvement"].eq(True).sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_counterfactual(counterfactual: pd.DataFrame, path: Path) -> None:
    counts = _counterfactual_counts(counterfactual)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(counts))
    width = 0.25
    for offset, column, label, color in (
        (-width, "would_exploit_count", "would exploit", "#4c78a8"),
        (0.0, "missed_improvement_count", "missed improvement", "#e45756"),
        (width, "false_improvement_count", "false improvement", "#f2cf5b"),
    ):
        axis.bar(x + offset, counts[column], width=width, label=label, color=color)
    axis.set_xticks(x, counts["guard_id"].str.replace("_", " "), rotation=15)
    axis.set(ylabel="candidate opportunities", title="Guard counterfactual (G1/G2 unavailable, not zero)")
    for index, available in enumerate(counts["available"]):
        if not available:
            axis.text(index, 0.25, "not estimable", ha="center", rotation=90, fontsize=8)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_exploration(exploration: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = np.where(exploration["DECISION_VALUE"].astype(bool), "#2a9d55", "#d77b00")
    axis.scatter(
        exploration["information_gain"],
        exploration["new_supported_points"],
        c=colors,
        s=60,
    )
    axis.set(
        xlabel="incremental log-information gain",
        ylabel="new supported points",
        title="Support provenance versus observed decision value",
    )
    axis.text(0.02, 0.98, "green: new exploit eligibility\norange: support-only", transform=axis.transAxes, va="top", fontsize=8)
    axis.grid(alpha=0.25)
    _save(figure, path)


def _plot_knee(knee: pd.DataFrame, path: Path) -> None:
    figure, left = plt.subplots(figsize=(9, 6))
    left.plot(knee["iteration"], knee["information_gain"], "-o", color="#4c78a8", label="information gain")
    left.set(xlabel="knee_stiff EXPLORE iteration", ylabel="information gain")
    right = left.twinx()
    right.plot(knee["iteration"], knee["new_supported_points"], "-s", color="#f28e2b", label="new support")
    right.set_ylabel("new supported points")
    left.set_title("Eight knee_stiff explores: support grows while model/map/decision stay fixed")
    left.grid(alpha=0.25)
    _save(figure, path)


def _plot_root_overview(matrix: pd.DataFrame, path: Path) -> None:
    problems = ("same_subject_path", "premature_mismatch_stop", "low_value_exploration")
    evidence_counts = (
        int(matrix["problem"].eq(problems[0]).sum()),
        int(matrix["problem"].eq(problems[1]).sum()),
        int(matrix["problem"].eq(problems[2]).sum()),
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        ("same path", "premature stop", "low-value explore"),
        evidence_counts,
        color=("#4c78a8", "#e45756", "#f2cf5b"),
    )
    axis.set(ylabel="audited root-cause evidence rows", title=f"{P2_POLICY_REVISION_JUSTIFIED}: diagnostic overview")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _stopping_report(exploration: pd.DataFrame, knee: pd.DataFrame) -> str:
    support_only = int(exploration["support_only_exploration"].astype(bool).sum())
    decision = int(exploration["DECISION_VALUE"].astype(bool).sum())
    matched_late = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ]
    return f"""# Exploration stopping candidate analysis

## Frozen observation

- EXPLORE rows audited: {len(exploration)}.
- `{SUPPORT_ONLY_EXPLORATION}`: {support_only}.
- Rows opening a new exploit eligibility: {decision}.
- knee_stiff EXPLORE rows: {len(knee)}; all have exact-zero five-parameter and prediction-map change.
- baseline/hip_stiff/heavy_leg Trial 7--13 rows: {len(matched_late)}; new exploit eligibility = {int(matched_late['newly_enabled_exploit_candidates'].sum())}.

## Future observable candidate

A future revision may study an **exploration diminishing-value stop** using only quantities already observable to the policy: repeated exact-zero parameter change, exact-zero map change, unchanged validation decision error, no newly eligible exploit within one/two rounds, declining incremental information gain, and continued support growth.  These fields should remain separate; support growth alone must not be treated as decision value.

No numeric threshold is frozen here.  The candidate is not enabled, does not use virtual truth, and is not a human/robot stopping rule.

- `candidate_enabled = false`
- `new_threshold_created = false`
- `truth_used_as_future_online_feature = false`
- diagnostic conclusion: `{EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT}`
- matched late-trial conclusion: `{POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION}`
"""


def _recommendation_report(
    truth_summary: pd.DataFrame,
    normalization: pd.DataFrame,
    synthetic: pd.DataFrame,
    scale_summary: pd.DataFrame,
    counterfactual: pd.DataFrame,
    exploration: pd.DataFrame,
    knee: pd.DataFrame,
) -> str:
    global_alphas = truth_summary[
        ["alpha_truth_global_hip", "alpha_truth_global_knee", "alpha_truth_global_phase"]
    ].drop_duplicates()
    synthetic_alphas = synthetic[
        ["alpha_truth_global_hip", "alpha_truth_global_knee", "alpha_truth_global_phase"]
    ].drop_duplicates()
    g0 = counterfactual.loc[counterfactual["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")]
    mismatch_g0 = g0.loc[~g0["scenario_name"].eq("matched_linear")]
    g1 = counterfactual.loc[counterfactual["guard_id"].eq("G1_LOCAL_PAIRWISE_MAX")]
    support_only = int(exploration["support_only_exploration"].astype(bool).sum())
    decision = int(exploration["DECISION_VALUE"].astype(bool).sum())
    reference_scale = normalization.loc[normalization["candidate_role"].eq("REFERENCE")]
    heavy = reference_scale.loc[reference_scale["subject_id"].eq("heavy_leg")].iloc[0]
    return f"""# {AUDIT_PROTOCOL_ID}

Final recommendation: `{P2_POLICY_REVISION_JUSTIFIED}`

Implementation readiness remains `REVISION_DESIGN_NOT_FROZEN`: the audit justifies changing the method, but does not supply a reviewed G1/G2 numeric bound or a stopping threshold.

## Plain-language answers

### A. Why baseline / hip_stiff / heavy_leg all reached knee -5

All four matched truth landscapes place the knee component of the global optimum at the generator's -5 deg boundary, so the common knee march is a real `{OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM}` tendency, not merely an optimizer selection failure.  However, their complete truth-global alpha values contain {len(global_alphas)} distinct hip/phase combinations while frozen P2 ends at `(0,-5,0)` for the three cited subjects.  The missing hip/phase differences are therefore `{POLICY_COLLAPSES_SUBJECT_DIFFERENCES}`.

### B/C. Objective discrimination and its source

The unchanged objective retains some subject discrimination: the four matched subjects have {len(global_alphas)} complete global optima, and the registered-value synthetic scan has {len(synthetic_alphas)}.  It nevertheless rewards knee-amplitude reduction to the lower generator boundary in {int(synthetic['global_knee_at_lower_generator_bound'].sum())}/{len(synthetic)} scanned combinations.  Per-subject normalization intentionally removes absolute scale (heavy_leg reference hip/knee scales versus baseline are {float(heavy['reference_hip_scale_vs_baseline']):.6f}/{float(heavy['reference_knee_scale_vs_baseline']):.6f}); it compresses mass/stiffness differences but does not erase all optimum differences.  The shared boundary is best explained by the combination of the mechanical torque-ratio objective and the available generator direction; the current four virtual subjects also do not span every optimum seen in the registered-value product scan.  The five-parameter model is not the primary matched-case cause.

The evidence does **not** establish that the objective is incapable of subject-specific personalization, so this audit does not issue `OBJECTIVE_REQUIRES_SCIENTIFIC_REVIEW` and does not change the objective.  Its common boundary behavior still requires scientific interpretation before a future policy is frozen.

### D/E. Why four mismatch cases stopped prematurely

In all four cases the relevant candidate was fully supported and the model predicted the correct improving direction, but the current validation-pair maximum made the margin negative.  That bound comes from an identification-excitation comparison without personalization alpha coordinates, so it is not calibrated to the formal one-step local decision scale: `{GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH}`.

### F. Local-guard counterfactual

There are zero designated validation pairs on the formal local-alpha scale.  G1 local-max and G2 local-P95 are therefore unavailable rather than zero; G1 rows estimable = {int(g1['would_exploit'].notna().sum())}.  This audit cannot honestly claim fewer missed improvements or unchanged false improvements for a local guard: `{LOCAL_CALIBRATION_NOT_SUFFICIENT}`.  The only allowed next diagnostic is a predeclared, designated local-pair validation design; it must not use adaptation truth or held-out final test.

### G/H/I. Exploration value

Of 32 EXPLORE trials, {support_only} increased support/information without exact parameter, map, validation-error, best-J, or exploit-eligibility change; {decision} opened exploit eligibility.  knee_stiff continued eight times because each valid unsupported adjacent frontier point remained rankable by information gain while P2 had no diminishing-decision-value stop.  baseline/hip_stiff/heavy_leg Trial 7--13 added support but opened no exploit and changed neither theta nor the map: `{POST_OPTIMUM_LOW_DECISION_VALUE_EXPLORATION}`.

### J. What a future P2 revision may change

1. Replace the structurally mismatched guard evidence only after a separately reviewed **local-decision-matched designated validation** protocol exists.  This task does not choose max/P95/P99.
2. Study a **decision-value-aware exploration continuation/stopping** rule using observable parameter/map/validation/eligibility/information/support traces.  This task does not freeze a numeric threshold.

Do not change the mechanical objective, five-parameter model, generator bounds, reference, ROM, 0.005 equivalence tolerance, or 90% support gate in this audit.

## Frozen status

- `{OFFLINE_METHOD_REQUIRES_REVISION}`
- `{INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS}`
- `{GLOBAL_MODEL_RELIABILITY_RULE_STATUS}`
- `NOT_HUMAN_READY`
- `NOT_ROBOT_MOTION_APPROVED`
- `counterfactual_trajectory_executed = false`
- `real_robot_connected = false`
"""


def _leakage_report() -> str:
    return f"""# Data leakage audit

- Virtual truth role: `{POST_HOC_TRUTH_ROLE}`.
- Truth landscapes and local finite differences were computed only after the frozen P2 implementation and reference/model definitions were fixed.
- Current P2 was replayed unchanged; truth was not supplied to proposal, candidate ranking, support, guard calibration, model fitting, exploration ranking, or stopping.
- Counterfactual G0 outcomes use truth only to label post-hoc true/false/missed outcomes.
- G1/G2 were not constructed from truth and remain unavailable because there is no designated validation pair on the formal local-alpha scale.
- Synthetic parameter scan role: `{SYNTHETIC_SCAN_ROLE}`; it uses only parameter values already present in registered repository virtual subjects.
- Held-out final-test data were not loaded.
- Future stopping candidates contain no truth feature.
- No robot or human trajectory was executed; no human threshold or robot approval was created.
"""


def _verify_previous_replay(results: Sequence[Any]) -> dict[str, Any]:
    natural = pd.read_csv(CONVERGENCE_ARTIFACT_DIRECTORY / "natural_stopping_summary.csv")
    natural = natural.loc[
        natural["policy_id"].eq(POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT)
    ].set_index("case_id")
    mismatches: list[str] = []
    for result in results:
        case_id = str(result.summary["case_id"])
        row = natural.loc[case_id]
        checks = {
            "executed_trial_count": int(result.summary["number_of_executed_trials"]),
            "stop_reason": str(result.summary["stop_reason"]),
        }
        for column, observed in checks.items():
            if str(row[column]) != str(observed):
                mismatches.append(f"{case_id}:{column}")
    if mismatches:
        raise RuntimeError(f"frozen P2 replay differs from convergence artifact: {mismatches}")
    return {
        "case_count": len(results),
        "replay_matches_previous_convergence_summary": True,
        "mismatch_fields": [],
    }


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validate_active_reference_file()
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    previous_policy = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    if previous_policy != policy_definitions():
        raise RuntimeError("frozen P0/P1/P2 policy definitions changed")

    parameter_lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(parameter_map_path)
    )
    if len(parameter_lattice) != 21025:
        raise RuntimeError("formal geometrically admissible space must contain 21,025 points")
    cache = build_trajectory_component_cache(parameter_lattice)

    states: dict[str, Any] = {}
    p2_results = []
    case_classes: dict[str, str] = {}
    for subject_id, scenario_name, case_class in ANALYSIS_CASES:
        case_id = f"{subject_id}__{scenario_name}"
        state = build_initial_research_state(subject_id, scenario_name)
        states[case_id] = state
        case_classes[case_id] = case_class
        p2_results.append(
            run_policy(
                state,
                POLICY_DECISION_GUARDED_EXPLORE_EXPLOIT,
                parameter_lattice,
                cache,
                trial_budget=EXTENDED_OFFLINE_DIAGNOSTIC_HORIZON,
                allow_extended_offline_diagnostic_horizon=True,
            )
        )
    replay = _verify_previous_replay(p2_results)

    truth_audits = []
    for result in p2_results:
        truth_audits.append(
            audit_post_decision_local_truth(
                result,
                states[result.summary["case_id"]],
                cache,
                case_class=case_classes[result.summary["case_id"]],
            )
        )
    post_truth_candidates = _concat(item.candidate_rows for item in truth_audits)

    matched_result = {
        result.subject_id: result
        for result in p2_results
        if result.scenario_name == "matched_linear"
    }
    subject_audits = {}
    for subject_id in TRUTH_SUBJECTS:
        result = matched_result[subject_id]
        p2_alpha = (
            float(result.summary["final_best_alpha_hip"]),
            float(result.summary["final_best_alpha_knee"]),
            float(result.summary["final_best_alpha_phase"]),
        )
        subject_audits[subject_id] = audit_subject_truth_landscape(
            subject_id, parameter_lattice, cache, p2_final_alpha=p2_alpha
        )
    truth_summary = pd.DataFrame(
        [audit.summary for audit in subject_audits.values()]
    )
    profiles = _concat(audit.profiles for audit in subject_audits.values())
    sensitivity = pd.DataFrame(
        [audit.local_sensitivity for audit in subject_audits.values()]
    )
    normalization = build_objective_normalization_audit(
        truth_summary, parameter_lattice, cache
    )
    synthetic = scan_registered_parameter_sensitivity(parameter_lattice, cache)

    provenance = build_current_guard_uncertainty_provenance(p2_results)
    scale_summary = summarize_validation_pair_scales(provenance)
    counterfactual = build_counterfactual_guard_comparison(
        p2_results, post_truth_candidates
    )
    previous_exploration = pd.read_csv(
        CONVERGENCE_ARTIFACT_DIRECTORY / "exploration_decision_value.csv"
    )
    exploration = build_exploration_value_decomposition(
        p2_results, previous_exploration
    )
    knee = build_knee_stiff_exploration_audit(exploration)
    matrix = classify_root_causes(
        truth_summary, synthetic, counterfactual, exploration
    )

    tables = {
        **{
            f"truth_landscape_{subject}.csv": audit.landscape
            for subject, audit in subject_audits.items()
        },
        "truth_landscape_summary.csv": truth_summary,
        "truth_axis_profiles.csv": profiles,
        "subject_truth_local_sensitivity.csv": sensitivity,
        "objective_normalization_subject_effect.csv": normalization,
        "synthetic_subject_optimum_map.csv": synthetic,
        "current_guard_uncertainty_provenance.csv": provenance,
        "global_vs_local_validation_uncertainty.csv": scale_summary,
        "counterfactual_guard_comparison.csv": counterfactual,
        "exploration_value_decomposition.csv": exploration,
        "knee_stiff_exploration_audit.csv": knee,
        "P2_REVISION_ROOT_CAUSE_MATRIX.csv": matrix,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)

    (output / REPORT_FILENAMES[0]).write_text(
        _stopping_report(exploration, knee), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _recommendation_report(
            truth_summary,
            normalization,
            synthetic,
            scale_summary,
            counterfactual,
            exploration,
            knee,
        ),
        encoding="utf-8",
    )
    (output / REPORT_FILENAMES[2]).write_text(
        _leakage_report(), encoding="utf-8"
    )

    landscapes = {
        subject: audit.landscape for subject, audit in subject_audits.items()
    }
    _plot_landscape_comparison(
        landscapes, output / "subject_truth_landscape_comparison.png"
    )
    _plot_profile(profiles, "hip_delta", output / "truth_hip_profile_by_subject.png")
    _plot_profile(profiles, "knee_delta", output / "truth_knee_profile_by_subject.png")
    _plot_profile(profiles, "phase_delta", output / "truth_phase_profile_by_subject.png")
    _plot_gradients(sensitivity, output / "truth_local_gradient_by_subject.png")
    _plot_validation_uncertainty(
        scale_summary, output / "global_vs_local_validation_uncertainty.png"
    )
    _plot_counterfactual(
        counterfactual, output / "counterfactual_guard_comparison.png"
    )
    _plot_exploration(
        exploration, output / "exploration_value_decomposition.png"
    )
    _plot_knee(knee, output / "knee_stiff_exploration_value.png")
    _plot_root_overview(matrix, output / "root_cause_overview.png")

    generated = (*CSV_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    missing = [name for name in generated if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    output_hashes = {name: _sha256(output / name) for name in generated}
    g0 = counterfactual.loc[
        counterfactual["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")
    ]
    g1 = counterfactual.loc[
        counterfactual["guard_id"].eq("G1_LOCAL_PAIRWISE_MAX")
    ]
    matched_late = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ]
    protected_diff = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )
    metadata = {
        "protocol_id": AUDIT_PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_boundary_note": (
            "policy_and_convergence_sources/artifacts_are_untracked_after_HEAD_0ae022c;"
            "this audit_did_not_stage_or_commit"
        ),
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
        "generator_bounds": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
        "formal_grid_steps": {
            "hip_deg": GRID_HIP_STEP_DEG,
            "knee_deg": GRID_KNEE_STEP_DEG,
            "phase": GRID_PHASE_STEP,
        },
        "geometrically_admissible_point_count": len(parameter_lattice),
        "truth_subjects": list(TRUTH_SUBJECTS),
        "truth_role": POST_HOC_TRUTH_ROLE,
        "truth_fed_back_to_policy": False,
        "truth_used_for_model_fitting": False,
        "truth_used_for_validation_uncertainty": False,
        "truth_used_for_candidate_proposal": False,
        "truth_used_for_future_stopping_feature": False,
        "heldout_final_test_used": False,
        "synthetic_scan_role": SYNTHETIC_SCAN_ROLE,
        "synthetic_scan_uses_only_registered_repository_values": True,
        "policy_replay": replay,
        "policy_behavior_modified": False,
        "mechanical_objective_modified": False,
        "generator_modified": False,
        "five_parameter_model_modified": False,
        "current_research_guard_modified": False,
        "counterfactual_trajectory_executed": False,
        "local_pair_validation_count": 0,
        "local_guard_candidate_status": LOCAL_PAIR_UNAVAILABLE,
        "G1_G2_estimable": False,
        "G1_evaluated_candidate_count": int(g1["would_exploit"].notna().sum()),
        "G0_missed_improvement_candidate_count": int(
            g0["missed_improvement"].astype(bool).sum()
        ),
        "explore_trial_count": len(exploration),
        "support_only_explore_count": int(
            exploration["support_only_exploration"].astype(bool).sum()
        ),
        "decision_value_explore_count": int(
            exploration["DECISION_VALUE"].astype(bool).sum()
        ),
        "knee_stiff_explore_count": len(knee),
        "matched_trial_7_to_13_count": len(matched_late),
        "matched_trial_7_to_13_new_exploit_count": int(
            matched_late["newly_enabled_exploit_candidates"].sum()
        ),
        "root_cause_statuses": {
            "common_knee_boundary": OBJECTIVE_TRUTH_BOUNDARY_OPTIMUM,
            "lost_complete_alpha_subject_difference": POLICY_COLLAPSES_SUBJECT_DIFFERENCES,
            "guard_calibration": GLOBAL_TO_LOCAL_CALIBRATION_MISMATCH,
            "local_guard_evidence": LOCAL_CALIBRATION_NOT_SUFFICIENT,
            "exploration_continuation": EXPLORATION_CONTINUATION_OVERVALUE_SUPPORT,
        },
        "final_recommendation": P2_POLICY_REVISION_JUSTIFIED,
        "revision_implementation_readiness": "REVISION_DESIGN_NOT_FROZEN",
        "allowed_future_revision_candidates": [
            "local_decision_matched_designated_validation_uncertainty_calibration_after_new_evidence",
            "decision_value_aware_exploration_continuation_or_stopping_after_rule_review",
        ],
        "objective_requires_scientific_review_status": (
            "OBJECTIVE_RETAINS_SOME_SUBJECT_DISCRIMINATION_NO_AUTOMATIC_REVISION"
        ),
        "offline_method_status": OFFLINE_METHOD_REQUIRES_REVISION,
        "initial_identification_acceptance_status": INITIAL_IDENTIFICATION_ACCEPTANCE_STATUS,
        "global_model_reliability_rule_status": GLOBAL_MODEL_RELIABILITY_RULE_STATUS,
        "research_status": RESEARCH_ONLY,
        "approval_status": NOT_APPROVED_FOR_ROBOT_PERSONALIZATION,
        "not_human_ready": True,
        "not_robot_motion_approved": True,
        "real_robot_connected": False,
        "human_threshold_created": False,
        "protected_package_git_diff_empty": protected_diff == "",
        "protected_package_git_diff": protected_diff,
        "policy_definition_source_sha256": sha256_file(
            POLICY_ARTIFACT_DIRECTORY / "policy_definition.json"
        ),
        "decision_guard_source_sha256": _text_sha256(
            inspect.getsource(apply_research_decision_guard)
        ),
        "exploit_selector_source_sha256": _text_sha256(
            inspect.getsource(select_exploit_candidate)
        ),
        "exploration_ranker_source_sha256": _text_sha256(
            inspect.getsource(rank_exploration_frontier)
        ),
        "policy_core_source_sha256": sha256_file(POLICY_CORE_SOURCE),
        "mechanical_objective_source_sha256": sha256_file(
            MECHANICAL_OBJECTIVE_SOURCE
        ),
        "generator_source_sha256": sha256_file(GENERATOR_SOURCE),
        "estimator_source_sha256": sha256_file(ESTIMATOR_SOURCE),
        "audit_source_sha256": sha256_file(CORE_SOURCE_PATH),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "parameter_map_sha256": sha256_file(parameter_map_path),
        "previous_convergence_metadata_sha256": sha256_file(
            CONVERGENCE_ARTIFACT_DIRECTORY / "metadata.json"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": output_hashes,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_directory, arguments.parameter_map)
    print(f"protocol={AUDIT_PROTOCOL_ID}")
    print(f"output={arguments.output_directory}")
    print(f"recommendation={metadata['final_recommendation']}")
    print(f"offline_method_status={metadata['offline_method_status']}")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    print("robot_connected=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
