"""Generate endpoint-only P2 multi-step framework analysis artifacts."""

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
    validate_active_reference_file,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_multi_step_decision_framework_analysis import (
    ANALYSIS_ID,
    DEFAULT_ENABLED,
    FRAMEWORKS,
    MANIFEST_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED,
    OFFLINE_ONLY,
    PRIOR_SEMANTICS_MANIFEST_PATH,
    PRIOR_SEMANTICS_MANIFEST_SHA256,
    FrozenFrameworkManifestGate,
    canonical_json_bytes,
    framework_uncertainty,
    load_semantics_calibration,
    manifest_payload,
    run_framework_shadow,
    small_step_recovery,
)
from .p2_v2_prospective_offline_validation import (
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    dynamic_subject_for_id,
    evaluate_full_truth_landscape,
    post_policy_local_truth_audit,
    prospective_case_rows,
    registered_prospective_subject,
)
from .post_prospective_rejection_root_cause_audit import (
    PROSPECTIVE_CONCLUSION,
    PROSPECTIVE_MANIFEST_SHA256,
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
    ANALYSIS_CASES,
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_multi_step_decision_framework_analysis.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_multi_step_decision_framework_analysis.py"
PRIOR_SEMANTICS_CORE_PATH = MODULE_DIR / "p2_decision_rule_semantics_audit.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_multi_step_decision_framework_analysis_v1"
)
REQUIRED_OUTPUT_FILENAMES = (
    "MANIFEST.json",
    "DECISION_FRAMEWORK_REPORT.md",
    "single_vs_bundle_comparison.csv",
    "small_step_recovery.csv",
    "subject_specificity_analysis.csv",
    "trial_efficiency_analysis.csv",
    "metadata.json",
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


def _write_json(path: Path, payload: Mapping[str, Any], *, canonical: bool = False) -> None:
    data = (
        canonical_json_bytes(_json_safe(dict(payload)))
        if canonical
        else (
            json.dumps(
                _json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    return completed.stdout.rstrip("\n")


def _checkpoint_preflight() -> dict[str, Any]:
    required = (
        "lower_limb_sim/p2_next_revision_independent_calibration.py",
        "lower_limb_sim/run_p2_next_revision_independent_calibration.py",
        (
            "lower_limb_sim/formal_artifacts/"
            "p2_next_revision_independent_calibration_v1/"
            "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
        ),
        (
            "lower_limb_sim/formal_artifacts/"
            "post_prospective_rejection_root_cause_audit_v1/"
            "prospective_small_step_accumulation.csv"
        ),
    )
    for relative in required:
        try:
            _git_output("ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("MULTI_STEP_ANALYSIS_REQUIRES_CHECKPOINT") from exc
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != sha256_file(PROJECT_ROOT / relative):
            raise RuntimeError("MULTI_STEP_ANALYSIS_REQUIRES_CHECKPOINT")
    if sha256_file(PRIOR_SEMANTICS_MANIFEST_PATH) != PRIOR_SEMANTICS_MANIFEST_SHA256:
        raise RuntimeError("prior decision semantics manifest changed")
    prior_relative = str(PRIOR_SEMANTICS_MANIFEST_PATH.relative_to(PROJECT_ROOT))
    prior_tracked = bool(_git_output("ls-files", "--", prior_relative))
    return {
        "checkpoint_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "tracked_calibration_inputs_verified": True,
        "prior_semantics_manifest_sha_verified": True,
        "prior_semantics_artifact_git_tracked": prior_tracked,
        "required_tracked_inputs": list(required),
    }


def _case_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, scenario, _case_class in ANALYSIS_CASES:
        rows.append(
            {
                "case_id": f"{subject}__{scenario}",
                "subject_id": subject,
                "scenario_name": scenario,
                "case_class": "MATCHED" if scenario == "matched_linear" else "MISMATCH",
                "development_origin": "ORIGINAL_P2_DEVELOPMENT",
            }
        )
    for row in prospective_case_rows().to_dict(orient="records"):
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "scenario_name": str(row["scenario_name"]),
                "case_class": (
                    "MATCHED"
                    if str(row["scenario_name"]) == "matched_linear"
                    else "MISMATCH"
                ),
                "development_origin": "POST_REJECTION_DEVELOPMENT",
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 15 or output["case_id"].duplicated().any():
        raise RuntimeError("development case roles changed")
    return output


def _aggregate(
    case_summary: pd.DataFrame, small: pd.DataFrame
) -> pd.DataFrame:
    output = (
        case_summary.groupby(
            ["framework_id", "horizon_steps", "calibrated_uncertainty"],
            as_index=False,
            sort=False,
        )
        .agg(
            case_count=("case_id", "count"),
            trials=("number_of_executed_trials", "sum"),
            explore=("number_of_explore_trials", "sum"),
            exploit=("number_of_exploit_trials", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            false_improvement=("number_of_executed_false_improvements", "sum"),
            conservative_stop=("premature_conservative_stops", "sum"),
            mean_decision_latency_trials=("decision_latency_trials", "mean"),
            median_decision_latency_trials=("decision_latency_trials", "median"),
            latent_intermediate_nodes_skipped=(
                "latent_intermediate_nodes_skipped",
                "sum",
            ),
            intermediate_trajectory_executions=(
                "intermediate_trajectory_executions",
                "sum",
            ),
            endpoint_executions=("endpoint_executions", "sum"),
            final_J=("final_best_actual_J", "mean"),
            global_regret=("global_truth_regret", "mean"),
        )
    )
    recovery = (
        small.groupby("framework_id")["recovered_small_step_path"].sum().to_dict()
    )
    output["small_step_recovery"] = output["framework_id"].map(recovery).astype(int)
    order = [spec.framework_id for spec in FRAMEWORKS]
    output = output.set_index("framework_id").reindex(order).reset_index()
    single = output.loc[output["framework_id"].eq("SINGLE_STEP")].iloc[0]
    output["trial_change_vs_single"] = output["trials"] - int(single["trials"])
    output["missed_change_vs_single"] = (
        output["missed_improvement"] - int(single["missed_improvement"])
    )
    output["false_change_vs_single"] = (
        output["false_improvement"] - int(single["false_improvement"])
    )
    output["final_J_change_vs_single"] = output["final_J"] - float(single["final_J"])
    output["global_regret_change_vs_single"] = (
        output["global_regret"] - float(single["global_regret"])
    )
    output["small_step_recovery_change_vs_single"] = (
        output["small_step_recovery"] - int(single["small_step_recovery"])
    )
    output["truth_used_for_framework_selection"] = False
    output["new_policy_enabled"] = False
    return output


def _trial_efficiency(
    case_summary: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    accepted = (
        history.groupby(["case_id", "framework_id"])
        .agg(
            accepted_meaningful_improvements=("accepted_meaningful_improvement", "sum"),
            wrong_direction_executions=(
                "delta_J_actual_vs_operating",
                lambda values: int((values.astype(float) > 0.0).sum()),
            ),
        )
        .reset_index()
    )
    columns = [
        "case_id",
        "subject_id",
        "scenario_name",
        "framework_id",
        "horizon_steps",
        "number_of_executed_trials",
        "number_of_explore_trials",
        "number_of_exploit_trials",
        "first_exploit_iteration",
        "decision_latency_trials",
        "latent_intermediate_nodes_skipped",
        "intermediate_trajectory_executions",
        "endpoint_executions",
    ]
    output = case_summary[columns].merge(
        accepted, on=["case_id", "framework_id"], how="left", validate="one_to_one"
    )
    output["accepted_meaningful_improvements"] = output[
        "accepted_meaningful_improvements"
    ].fillna(0).astype(int)
    output["wrong_direction_executions"] = output[
        "wrong_direction_executions"
    ].fillna(0).astype(int)
    output["accepted_improvements_per_trial"] = np.divide(
        output["accepted_meaningful_improvements"],
        output["number_of_executed_trials"],
        out=np.zeros(len(output), dtype=float),
        where=output["number_of_executed_trials"].to_numpy(dtype=float) > 0.0,
    )
    output["intermediate_execution_policy"] = "ZERO_INTERMEDIATE_EXECUTIONS"
    output["evidence_level"] = OFFLINE_ONLY
    return output


def _subject_specificity(
    case_summary: pd.DataFrame, truth_landscapes: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in case_summary.to_dict(orient="records"):
        case_id = str(item["case_id"])
        truth = truth_landscapes[case_id]
        optimum = truth.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort"
        ).iloc[0]
        final = np.asarray(
            (
                item["final_best_alpha_hip"],
                item["final_best_alpha_knee"],
                item["final_best_alpha_phase"],
            ),
            dtype=float,
        )
        optimum_alpha = np.asarray(
            (
                optimum["hip_delta"],
                optimum["knee_delta"],
                optimum["phase_delta"],
            ),
            dtype=float,
        )
        grid = np.asarray((0.25, 0.25, 0.0025), dtype=float)
        signed_steps = (final - optimum_alpha) / grid
        rows.append(
            {
                "case_id": case_id,
                "subject_id": item["subject_id"],
                "scenario_name": item["scenario_name"],
                "framework_id": item["framework_id"],
                "horizon_steps": item["horizon_steps"],
                "final_alpha_hip": final[0],
                "final_alpha_knee": final[1],
                "final_alpha_phase": final[2],
                "truth_optimum_alpha_hip_posthoc": optimum_alpha[0],
                "truth_optimum_alpha_knee_posthoc": optimum_alpha[1],
                "truth_optimum_alpha_phase_posthoc": optimum_alpha[2],
                "hip_grid_steps_from_truth_optimum": signed_steps[0],
                "knee_grid_steps_from_truth_optimum": signed_steps[1],
                "phase_grid_steps_from_truth_optimum": signed_steps[2],
                "generator_grid_L1_steps_from_truth_optimum": float(
                    np.abs(signed_steps).sum()
                ),
                "final_alpha_is_neutral": bool(np.allclose(final, 0.0, atol=1e-12)),
                "final_J": item["final_best_actual_J"],
                "global_regret": item["global_truth_regret"],
                "truth_used_for_selection": False,
                "truth_attached_posthoc_only": True,
                "distance_is_physical": False,
                "distance_definition": "L1_COUNT_OF_EXISTING_FORMAL_GENERATOR_GRID_UNITS",
            }
        )
    output = pd.DataFrame(rows)
    unique_counts = (
        output.assign(
            final_key=output[["final_alpha_hip", "final_alpha_knee", "final_alpha_phase"]]
            .astype(str)
            .agg("|".join, axis=1)
        )
        .groupby("framework_id")["final_key"]
        .nunique()
        .to_dict()
    )
    output["unique_final_alpha_count_within_framework"] = output["framework_id"].map(
        unique_counts
    )
    return output


def _recommendation(
    comparison: pd.DataFrame,
) -> tuple[str | None, pd.DataFrame]:
    single = comparison.loc[comparison["framework_id"].eq("SINGLE_STEP")].iloc[0]
    candidates = comparison.loc[comparison["framework_id"].ne("SINGLE_STEP")].copy()
    candidates["prototype_eligible"] = (
        candidates["small_step_recovery"].gt(int(single["small_step_recovery"]))
        & candidates["false_improvement"].le(int(single["false_improvement"]))
        & candidates["final_J"].le(
            float(single["final_J"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        & candidates["global_regret"].le(
            float(single["global_regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
    )
    eligible = candidates.loc[candidates["prototype_eligible"]].copy()
    if eligible.empty:
        selected = None
    else:
        selected = str(
            eligible.sort_values(
                [
                    "small_step_recovery",
                    "final_J",
                    "trials",
                    "calibrated_uncertainty",
                    "horizon_steps",
                ],
                ascending=[False, True, True, True, True],
                kind="mergesort",
            ).iloc[0]["framework_id"]
        )
    candidates["selected_for_further_research"] = candidates["framework_id"].eq(
        selected
    )
    candidates["selection_rule_frozen_before_shadow"] = True
    return selected, candidates


def _fmt(value: Any) -> str:
    return f"{float(value):.9f}"


def _report(
    manifest_sha: str,
    comparison: pd.DataFrame,
    recommendation: pd.DataFrame,
    selected: str | None,
) -> str:
    indexed = comparison.set_index("framework_id")
    lines = []
    for spec in FRAMEWORKS:
        row = indexed.loc[spec.framework_id]
        lines.append(
            f"- {spec.framework_id}: U={_fmt(row['calibrated_uncertainty'])}, "
            f"trials={int(row['trials'])}, missed={int(row['missed_improvement'])}, "
            f"false={int(row['false_improvement'])}, recovery={int(row['small_step_recovery'])}/9, "
            f"final J={_fmt(row['final_J'])}, regret={_fmt(row['global_regret'])}, "
            f"mean decision latency={_fmt(row['mean_decision_latency_trials'])} trials."
        )
    single = indexed.loc["SINGLE_STEP"]
    bundle = comparison.loc[comparison["framework_id"].str.startswith("BUNDLE_")]
    uncertainty_increases_2_3_5 = bool(
        np.all(np.diff(bundle.sort_values("horizon_steps")["calibrated_uncertainty"]) > 0.0)
    )
    recovery_non_decreasing = bool(
        np.all(np.diff(bundle.sort_values("horizon_steps")["small_step_recovery"]) >= 0)
    )
    selected_row = (
        indexed.loc[selected] if selected is not None else None
    )
    horizon_limited = bool(
        recommendation["prototype_eligible"].astype(bool).any()
    )
    prototype = selected is not None
    return f"""# {ANALYSIS_ID}

Manifest SHA-256: `{manifest_sha}`

## Frozen comparison

All four frameworks use the same research-only two-gate semantics: endpoint predicted improvement must separately exceed the unchanged 0.005 meaningful mechanical-objective tolerance, and the independently calibrated scale-P95 residual interval must still support the improvement direction. The only changed variable is the endpoint horizon: 1, 2, 3, or 5 existing formal-grid units.

Bundle candidates execute the endpoint directly. Intermediate nodes are checked for existing generator relationship, geometry, active-reference provenance, patient envelope, and unchanged 90% model support, but zero intermediate trajectories are executed. After every selected endpoint, the five-parameter model is refit and the entire prediction map is recomputed.

## Results

{chr(10).join(lines)}

The 2/3/5-step bundle P95 values increase strictly with length: `{uncertainty_increases_2_3_5}`. Small-step recovery across those horizons is non-decreasing: `{recovery_non_decreasing}`. This is descriptive evidence of a benefit-versus-uncertainty trade-off, not a scale law or statistical-power claim. The 2-step P95 is below the separately sampled 1-step P95, so no monotonic 1→2→3→5 uncertainty claim is made.

The pre-frozen recommendation rule selected `{selected or 'NONE'}` for further research. It did not change any live policy.

## Answers

### A. Is single-step failure mainly a short decision horizon?

Development evidence says `{horizon_limited}` under the pre-frozen criterion: at least one direct endpoint bundle recovers more historical accumulation paths without increasing observed false improvement and preserves or improves mean final J/regret within 0.005. This identifies horizon length as a material mechanism, not necessarily the only P2 failure source.

### B. Is a bundle endpoint more consistent with the rehabilitation-trajectory optimization goal?

For the repository's frozen **mechanical** objective, a bundle endpoint can represent a meaningful cumulative trajectory-shape change that one formal step cannot. It is therefore a better decision unit for this mechanical optimization question when its calibrated direction gate passes. This does not establish comfort, human rehabilitation benefit, safety, or clinical superiority.

### C. Which of 2/3/5 steps is most worth further study?

`{selected or 'NONE'}` according to the manifest-frozen ordering: maximize small-step recovery, then minimize final J, trials, uncertainty, and horizon length. {('Its mean final J is ' + _fmt(selected_row['final_J']) + ', regret ' + _fmt(selected_row['global_regret']) + ', and uncertainty ' + _fmt(selected_row['calibrated_uncertainty']) + '.') if selected_row is not None else 'No bundle satisfied the frozen eligibility criteria.'}

### D. Should the next stage enter prototype implementation?

`{'YES, BUT DEFAULT-OFF OFFLINE PROTOTYPE ONLY' if prototype else 'NO'}`. This recommendation is development-only and requires a separately frozen prototype manifest. It does not authorize prospective testing, human use, or robot motion.

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Independent calibration cases supply residual scales only.
- No held-out final test and no prospective cohort.
- Active reference, ROM, `theta_shank = q_hip - q_knee`, five-parameter model, mechanical objective, generator, 0.005 tolerance, and 90% support gate remain unchanged.
- P2 V1 is unchanged; no policy is implemented or enabled.
- Final states: `{OFFLINE_ONLY}`, `{NOT_HUMAN_READY}`, `{NOT_ROBOT_APPROVED}`.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = _checkpoint_preflight()
    verify_immutable_prospective_artifacts()
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference SHA changed")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("ROM protocol changed")
    if tuple(FORMAL_HIP_ROM_DEG) != (0.0, 120.0):
        raise RuntimeError("formal hip ROM changed")
    if tuple(FORMAL_KNEE_ROM_DEG) != (5.0, 145.0):
        raise RuntimeError("formal knee ROM changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    if DEFAULT_ENABLED:
        raise PermissionError("framework analysis must remain default-off")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_semantics_calibration()
    protected_before = _protected_source_hashes()
    protected_before.update(
        {
            "prior_semantics_manifest": sha256_file(PRIOR_SEMANTICS_MANIFEST_PATH),
            "prior_semantics_core": sha256_file(PRIOR_SEMANTICS_CORE_PATH),
            "multi_step_core": sha256_file(CORE_SOURCE_PATH),
            "multi_step_runner": sha256_file(RUNNER_SOURCE_PATH),
        }
    )

    # The manifest is the first artifact; no development state/truth was loaded above.
    manifest = manifest_payload(
        calibration,
        checkpoint_commit=str(checkpoint["checkpoint_commit"]),
        protected_source_sha256=protected_before,
    )
    manifest_path = output / "MANIFEST.json"
    _write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    gate = FrozenFrameworkManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    cases = _case_table()
    lattice = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometrically admissible lattice changed")
    cache = build_trajectory_component_cache(lattice)
    patient_cache: dict[tuple[float, float, float], bool] = {}
    results = []
    truth_landscapes: dict[str, pd.DataFrame] = {}
    truth_round_frames: list[pd.DataFrame] = []

    for case in cases.to_dict(orient="records"):
        case_id = str(case["case_id"])
        post_rejection = case["development_origin"] == "POST_REJECTION_DEVELOPMENT"

        def run_case() -> None:
            gate.record_truth_access("INITIAL_IDENTIFICATION_AFTER_MANIFEST_FREEZE")
            state = build_initial_research_state(
                str(case["subject_id"]), str(case["scenario_name"])
            )
            case_results = []
            for spec in FRAMEWORKS:
                result = run_framework_shadow(
                    state,
                    spec,
                    lattice,
                    cache,
                    gate,
                    calibration,
                    patient_validity_cache=patient_cache,
                )
                results.append(result)
                case_results.append(result)
            gate.record_truth_access("POST_FRAMEWORK_FULL_TRUTH_LANDSCAPE")
            landscape = evaluate_full_truth_landscape(case_results[0], state, cache)
            truth_landscapes[case_id] = landscape
            for result in case_results:
                gate.record_truth_access("POST_FRAMEWORK_LOCAL_TRUTH_AUDIT")
                _candidate_truth, rounds = post_policy_local_truth_audit(
                    result, state, cache
                )
                if not rounds.empty:
                    truth_round_frames.append(rounds)

        if post_rejection:
            subject = dynamic_subject_for_id(str(case["subject_id"]))
            with registered_prospective_subject(subject):
                run_case()
        else:
            run_case()

    history = pd.concat(
        [result.trial_history for result in results], ignore_index=True, sort=False
    )
    truth_rounds = pd.concat(truth_round_frames, ignore_index=True, sort=False)
    case_summary_rows = []
    for result in results:
        rounds = truth_rounds.loc[
            truth_rounds["case_id"].eq(result.summary["case_id"])
            & truth_rounds["policy_id"].eq(result.policy_id)
        ]
        case_summary_rows.append(
            _result_summary_with_truth(
                result.summary,
                rounds,
                truth_landscapes[str(result.summary["case_id"])],
            )
        )
    case_summary = pd.DataFrame(case_summary_rows)
    small = small_step_recovery(calibration)
    comparison = _aggregate(case_summary, small)
    trial_efficiency = _trial_efficiency(case_summary, history)
    subject_specificity = _subject_specificity(case_summary, truth_landscapes)
    selected, recommendation = _recommendation(comparison)

    _write_csv(output / "single_vs_bundle_comparison.csv", comparison)
    _write_csv(output / "small_step_recovery.csv", small)
    _write_csv(output / "subject_specificity_analysis.csv", subject_specificity)
    _write_csv(output / "trial_efficiency_analysis.csv", trial_efficiency)
    _write_csv(output / "framework_case_summary.csv", case_summary)
    _write_csv(output / "framework_trial_history.csv", history)
    _write_csv(output / "framework_truth_rounds.csv", truth_rounds)
    _write_csv(output / "framework_recommendation_audit.csv", recommendation)
    _write_text(
        output / "DECISION_FRAMEWORK_REPORT.md",
        _report(manifest_sha, comparison, recommendation, selected),
    )

    protected_after = _protected_source_hashes()
    protected_after.update(
        {
            "prior_semantics_manifest": sha256_file(PRIOR_SEMANTICS_MANIFEST_PATH),
            "prior_semantics_core": sha256_file(PRIOR_SEMANTICS_CORE_PATH),
            "multi_step_core": sha256_file(CORE_SOURCE_PATH),
            "multi_step_runner": sha256_file(RUNNER_SOURCE_PATH),
        }
    )
    if protected_before != protected_after:
        raise RuntimeError("protected source changed during framework shadow")
    if sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("framework manifest changed after development truth")
    if not history["intermediate_execution_count"].eq(0).all():
        raise RuntimeError("an intermediate trajectory was executed")
    if not history["model_refit_after_execution"].astype(bool).all():
        raise RuntimeError("model was not refit after every execution")
    if not history["full_map_recomputed_after_execution"].astype(bool).all():
        raise RuntimeError("map was not recomputed after every execution")
    if history["truth_accessed_before_selection"].astype(bool).any():
        raise RuntimeError("future truth entered framework selection")

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    bundles = comparison.loc[comparison["framework_id"].str.startswith("BUNDLE_")]
    metadata = {
        "analysis_id": ANALYSIS_ID,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "prior_semantics_manifest_sha256": PRIOR_SEMANTICS_MANIFEST_SHA256,
        "checkpoint": checkpoint,
        "framework_ids": [spec.framework_id for spec in FRAMEWORKS],
        "selected_framework_for_further_research": selected,
        "prototype_implementation_recommended": selected is not None,
        "prototype_scope_if_recommended": "DEFAULT_OFF_OFFLINE_ONLY",
        "bundle_uncertainty_strictly_increases_2_to_3_to_5": bool(
            np.all(
                np.diff(
                    bundles.sort_values("horizon_steps")["calibrated_uncertainty"]
                )
                > 0.0
            )
        ),
        "development_case_count": len(cases),
        "original_development_case_count": int(
            cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum()
        ),
        "post_rejection_development_case_count": int(
            cases["development_origin"].eq("POST_REJECTION_DEVELOPMENT").sum()
        ),
        "independent_calibration_cases_used_for_performance": False,
        "independent_calibration_used_for_residual_scale_only": True,
        "manifest_frozen_before_development_truth": True,
        "manifest_truth_access_count": gate.truth_access_count,
        "manifest_truth_access_stages": gate.truth_access_stages,
        "future_truth_used_for_authorization": False,
        "held_out_final_test_read": False,
        "prospective_cohort_run": False,
        "intermediate_trajectories_executed": False,
        "total_intermediate_trajectory_executions": int(
            history["intermediate_execution_count"].sum()
        ),
        "model_refit_after_every_execution": True,
        "full_map_recomputed_after_every_execution": True,
        "P2_V1_modified": False,
        "new_policy_implemented": False,
        "new_policy_default_enabled": False,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "immutable_prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "immutable_prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "robot_connected": False,
        "artifact_manifest": artifacts,
        "runtime_seconds": time.perf_counter() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    missing = [name for name in REQUIRED_OUTPUT_FILENAMES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing framework outputs: {missing}")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    args = parser.parse_args(argv)
    metadata = generate_artifacts(args.output_dir, args.parameter_map)
    print(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

