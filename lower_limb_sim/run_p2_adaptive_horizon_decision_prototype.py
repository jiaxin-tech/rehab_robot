"""Generate formal offline artifacts for the adaptive-horizon prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import (
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
)
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
from .p2_adaptive_horizon_decision_prototype import (
    ADAPTIVE_HORIZON_SEQUENCE,
    DEFAULT_ENABLED,
    H1_ID,
    H2_ID,
    H3_ID,
    MANIFEST_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED,
    OFFLINE_ONLY,
    PRIOR_FRAMEWORK_DIRECTORY,
    PRIOR_FRAMEWORK_MANIFEST_PATH,
    PRIOR_FRAMEWORK_MANIFEST_SHA256,
    PROTOTYPE_ID,
    adaptive_small_step_recovery,
    manifest_payload,
    run_adaptive_shadow,
)
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_multi_step_decision_framework_analysis import (
    FrozenFrameworkManifestGate,
    canonical_json_bytes,
    load_semantics_calibration,
)
from .p2_v2_prospective_offline_validation import (
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    dynamic_subject_for_id,
    evaluate_full_truth_landscape,
    post_policy_local_truth_audit,
    registered_prospective_subject,
)
from .post_prospective_rejection_root_cause_audit import (
    verify_immutable_prospective_artifacts,
)
from .research_decision_guarded_sequential_personalization import (
    TRIAL_PURPOSE_EXPLOIT,
    build_initial_research_state,
)
from .run_p2_multi_step_decision_framework_analysis import (
    _case_table,
    _subject_specificity,
    _write_csv,
    _write_json,
    _write_text,
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
CORE_SOURCE_PATH = MODULE_DIR / "p2_adaptive_horizon_decision_prototype.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_adaptive_horizon_decision_prototype.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_adaptive_horizon_decision_prototype_v1"
)
REQUIRED_OUTPUT_FILENAMES = (
    "REPORT.md",
    "MANIFEST.json",
    "adaptive_vs_fixed_comparison.csv",
    "horizon_usage.csv",
    "subject_specificity.csv",
    "boundary_analysis.csv",
    "metadata.json",
)

_PRIOR_REQUIRED = (
    "MANIFEST.json",
    "framework_case_summary.csv",
    "framework_trial_history.csv",
    "subject_specificity_analysis.csv",
    "small_step_recovery.csv",
)
_VARIANT_MAP = {
    "SINGLE_STEP": H1_ID,
    "BUNDLE_5": H2_ID,
}


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
    if sha256_file(PRIOR_FRAMEWORK_MANIFEST_PATH) != PRIOR_FRAMEWORK_MANIFEST_SHA256:
        raise RuntimeError("ADAPTIVE_HORIZON_REQUIRES_FROZEN_MULTI_STEP_CHECKPOINT")
    verified: dict[str, str] = {}
    for name in _PRIOR_REQUIRED:
        path = PRIOR_FRAMEWORK_DIRECTORY / name
        relative = str(path.relative_to(PROJECT_ROOT))
        try:
            _git_output("ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ADAPTIVE_HORIZON_REQUIRES_FROZEN_MULTI_STEP_CHECKPOINT"
            ) from exc
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise RuntimeError(
                "ADAPTIVE_HORIZON_REQUIRES_FROZEN_MULTI_STEP_CHECKPOINT"
            )
        verified[name] = sha256_file(path)
    return {
        "checkpoint_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "prior_framework_artifacts_tracked_and_verified": True,
        "prior_framework_artifact_sha256": verified,
    }


def _prototype_source_hashes() -> dict[str, str]:
    hashes = _protected_source_hashes()
    hashes.update(
        {
            "adaptive_core": sha256_file(CORE_SOURCE_PATH),
            "adaptive_runner": sha256_file(RUNNER_SOURCE_PATH),
            "prior_framework_manifest": sha256_file(
                PRIOR_FRAMEWORK_MANIFEST_PATH
            ),
        }
    )
    for name in _PRIOR_REQUIRED:
        hashes[f"prior_framework_artifact:{name}"] = sha256_file(
            PRIOR_FRAMEWORK_DIRECTORY / name
        )
    return hashes


def _fixed_case_summary() -> pd.DataFrame:
    table = pd.read_csv(PRIOR_FRAMEWORK_DIRECTORY / "framework_case_summary.csv")
    output = table.loc[table["framework_id"].isin(_VARIANT_MAP)].copy()
    output["source_framework_id"] = output["framework_id"]
    output["prototype_variant_id"] = output["framework_id"].map(_VARIANT_MAP)
    output["framework_id"] = output["prototype_variant_id"]
    output["policy_id"] = output["prototype_variant_id"]
    if len(output) != 30:
        raise RuntimeError("fixed comparator case count changed")
    return output


def _fixed_history() -> pd.DataFrame:
    table = pd.read_csv(PRIOR_FRAMEWORK_DIRECTORY / "framework_trial_history.csv")
    output = table.loc[table["framework_id"].isin(_VARIANT_MAP)].copy()
    output["source_framework_id"] = output["framework_id"]
    output["prototype_variant_id"] = output["framework_id"].map(_VARIANT_MAP)
    output["framework_id"] = output["prototype_variant_id"]
    output["policy_id"] = output["prototype_variant_id"]
    return output


def _fixed_subject_specificity() -> pd.DataFrame:
    table = pd.read_csv(
        PRIOR_FRAMEWORK_DIRECTORY / "subject_specificity_analysis.csv"
    )
    output = table.loc[table["framework_id"].isin(_VARIANT_MAP)].copy()
    output["source_framework_id"] = output["framework_id"]
    output["prototype_variant_id"] = output["framework_id"].map(_VARIANT_MAP)
    output["framework_id"] = output["prototype_variant_id"]
    return output


def _fixed_small_step() -> pd.DataFrame:
    table = pd.read_csv(PRIOR_FRAMEWORK_DIRECTORY / "small_step_recovery.csv")
    output = table.loc[table["framework_id"].isin(_VARIANT_MAP)].copy()
    output["source_framework_id"] = output["framework_id"]
    output["prototype_variant_id"] = output["framework_id"].map(_VARIANT_MAP)
    return output


def _boundary_table(subject_specificity: pd.DataFrame) -> pd.DataFrame:
    output = subject_specificity.copy()
    bounds = {
        "hip": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[
            "hip_amplitude_delta_deg"
        ],
        "knee": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[
            "knee_amplitude_delta_deg"
        ],
        "phase": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_phase_shift"],
    }
    for axis, (lower, upper) in bounds.items():
        values = output[f"final_alpha_{axis}"].to_numpy(dtype=float)
        output[f"{axis}_lower_boundary"] = np.isclose(
            values, lower, atol=1e-12, rtol=0.0
        )
        output[f"{axis}_upper_boundary"] = np.isclose(
            values, upper, atol=1e-12, rtol=0.0
        )
    boundary_columns = [
        f"{axis}_{side}_boundary"
        for axis in ("hip", "knee", "phase")
        for side in ("lower", "upper")
    ]
    output["any_generator_boundary_saturated"] = output[
        boundary_columns
    ].any(axis=1)
    output["generator_boundary_count"] = output[boundary_columns].sum(axis=1)
    output["generator_bounds_changed"] = False
    return output


def _comparison(
    case_summary: pd.DataFrame,
    boundary: pd.DataFrame,
    fixed_small: pd.DataFrame,
    adaptive_small: pd.DataFrame,
) -> pd.DataFrame:
    recovery = (
        fixed_small.groupby("prototype_variant_id")["recovered_small_step_path"]
        .sum()
        .to_dict()
    )
    recovery[H3_ID] = int(adaptive_small["recovered_small_step_path"].sum())
    boundary_summary = (
        boundary.groupby("prototype_variant_id", sort=False)
        .agg(
            boundary_saturated_case_count=(
                "any_generator_boundary_saturated",
                "sum",
            ),
            boundary_saturation_fraction=(
                "any_generator_boundary_saturated",
                "mean",
            ),
        )
        .reset_index()
    )
    finals = (
        boundary.assign(
            final_key=boundary[
                ["final_alpha_hip", "final_alpha_knee", "final_alpha_phase"]
            ]
            .astype(str)
            .agg("|".join, axis=1)
        )
        .groupby("prototype_variant_id")["final_key"]
        .nunique()
        .rename("unique_final_alpha_count")
        .reset_index()
    )
    output = (
        case_summary.groupby("prototype_variant_id", as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            trial_count=("number_of_executed_trials", "sum"),
            exploration_count=("number_of_explore_trials", "sum"),
            exploit_count=("number_of_exploit_trials", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            false_improvement=("number_of_executed_false_improvements", "sum"),
            conservative_stop=("premature_conservative_stops", "sum"),
            mean_final_J=("final_best_actual_J", "mean"),
            mean_global_regret=("global_truth_regret", "mean"),
            mean_decision_latency_trials=("decision_latency_trials", "mean"),
            intermediate_trajectory_executions=(
                "intermediate_trajectory_executions",
                "sum",
            ),
            endpoint_executions=("endpoint_executions", "sum"),
        )
        .merge(boundary_summary, on="prototype_variant_id", validate="one_to_one")
        .merge(finals, on="prototype_variant_id", validate="one_to_one")
    )
    output["small_step_recovery"] = output["prototype_variant_id"].map(
        recovery
    ).astype(int)
    order = [H1_ID, H2_ID, H3_ID]
    output = output.set_index("prototype_variant_id").reindex(order).reset_index()
    h1 = output.loc[output["prototype_variant_id"].eq(H1_ID)].iloc[0]
    h2 = output.loc[output["prototype_variant_id"].eq(H2_ID)].iloc[0]
    output["trial_change_vs_H1"] = output["trial_count"] - int(h1["trial_count"])
    output["trial_change_vs_H2"] = output["trial_count"] - int(h2["trial_count"])
    output["final_J_change_vs_H2"] = output["mean_final_J"] - float(
        h2["mean_final_J"]
    )
    output["regret_change_vs_H2"] = output["mean_global_regret"] - float(
        h2["mean_global_regret"]
    )
    output["truth_used_for_authorization"] = False
    output["default_enabled"] = False
    return output


def _horizon_usage(
    fixed_history: pd.DataFrame, adaptive_history: pd.DataFrame, cases: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    histories = {
        H1_ID: fixed_history.loc[fixed_history["prototype_variant_id"].eq(H1_ID)],
        H2_ID: fixed_history.loc[fixed_history["prototype_variant_id"].eq(H2_ID)],
        H3_ID: adaptive_history,
    }
    for variant, history in histories.items():
        for case in cases.to_dict(orient="records"):
            case_id = str(case["case_id"])
            selected = history.loc[
                history["case_id"].eq(case_id)
                & history["trial_purpose"].eq(TRIAL_PURPOSE_EXPLOIT)
            ]
            for horizon in ADAPTIVE_HORIZON_SEQUENCE:
                rows.append(
                    {
                        "row_scope": "CASE",
                        "prototype_variant_id": variant,
                        "case_id": case_id,
                        "subject_id": case["subject_id"],
                        "scenario_name": case["scenario_name"],
                        "horizon_steps": horizon,
                        "endpoint_authorization_count": int(
                            selected["horizon_steps"].eq(horizon).sum()
                        ),
                        "intermediate_trajectory_execution_count": 0,
                    }
                )
        case_rows = pd.DataFrame(rows)
        case_rows = case_rows.loc[
            case_rows["prototype_variant_id"].eq(variant)
            & case_rows["row_scope"].eq("CASE")
        ]
        for horizon in ADAPTIVE_HORIZON_SEQUENCE:
            rows.append(
                {
                    "row_scope": "ALL_CASES",
                    "prototype_variant_id": variant,
                    "case_id": "ALL_CASES",
                    "subject_id": "ALL_SUBJECTS",
                    "scenario_name": "ALL_SCENARIOS",
                    "horizon_steps": horizon,
                    "endpoint_authorization_count": int(
                        case_rows.loc[
                            case_rows["horizon_steps"].eq(horizon),
                            "endpoint_authorization_count",
                        ].sum()
                    ),
                    "intermediate_trajectory_execution_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    return f"{float(value):.9f}"


def _report(
    manifest_sha: str,
    comparison: pd.DataFrame,
    horizon_usage: pd.DataFrame,
) -> str:
    indexed = comparison.set_index("prototype_variant_id")
    h1 = indexed.loc[H1_ID]
    h2 = indexed.loc[H2_ID]
    h3 = indexed.loc[H3_ID]
    approaches = bool(
        int(h3["small_step_recovery"]) == int(h2["small_step_recovery"])
        and float(h3["mean_final_J"])
        <= float(h2["mean_final_J"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
        and float(h3["mean_global_regret"])
        <= float(h2["mean_global_regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
    )
    reduces_cost = bool(int(h3["trial_count"]) < int(h2["trial_count"]))
    reduces_boundary = bool(
        int(h3["boundary_saturated_case_count"])
        < int(h2["boundary_saturated_case_count"])
        or int(h3["unique_final_alpha_count"])
        > int(h2["unique_final_alpha_count"])
    )
    adaptive_usage = horizon_usage.loc[
        horizon_usage["prototype_variant_id"].eq(H3_ID)
        & horizon_usage["row_scope"].eq("ALL_CASES")
    ]
    usage_text = ", ".join(
        f"H{int(row.horizon_steps)}={int(row.endpoint_authorization_count)}"
        for row in adaptive_usage.itertuples()
    )
    long_cases = horizon_usage.loc[
        horizon_usage["prototype_variant_id"].eq(H3_ID)
        & horizon_usage["row_scope"].eq("CASE")
        & horizon_usage["horizon_steps"].isin((3, 5))
        & horizon_usage["endpoint_authorization_count"].gt(0)
    ]
    long_lines = [
        f"- {row.case_id}: H{int(row.horizon_steps)} × "
        f"{int(row.endpoint_authorization_count)}"
        for row in long_cases.itertuples()
    ] or ["- None"]
    rows = []
    for variant in (H1_ID, H2_ID, H3_ID):
        row = indexed.loc[variant]
        rows.append(
            f"- {variant}: trials={int(row['trial_count'])}, "
            f"explore={int(row['exploration_count'])}, "
            f"missed={int(row['missed_improvement'])}, "
            f"false={int(row['false_improvement'])}, "
            f"small-step={int(row['small_step_recovery'])}/9, "
            f"mean final J={_fmt(row['mean_final_J'])}, "
            f"regret={_fmt(row['mean_global_regret'])}, "
            f"unique final alpha={int(row['unique_final_alpha_count'])}, "
            f"boundary cases={int(row['boundary_saturated_case_count'])}/15."
        )
    return f"""# {PROTOTYPE_ID}

Manifest SHA-256: `{manifest_sha}`

## Frozen research rule

The default-off adaptive prototype evaluates horizons in the frozen order
`1 → 2 → 3 → 5`. It uses the unchanged independent 0.005 magnitude gate and
the horizon-specific calibrated direction interval. Escalation is allowed only
when every existing generator-grid node along one coordinate and one signed
direction has a finite, strictly improving predicted objective. The first
horizon with an eligible endpoint is selected. No intermediate trajectory is
executed; every endpoint is followed by model refit and full-map recomputation.

## Development shadow results

{chr(10).join(rows)}

Adaptive endpoint horizon usage: `{usage_text}`.

## Questions

### A. Does adaptive horizon approach fixed BUNDLE_5 performance?

`{approaches}` under the pre-frozen criterion: equal 9-path recovery and final
J/regret no worse than BUNDLE_5 by more than the unchanged 0.005 tolerance.

### B. Does adaptive horizon reduce fixed BUNDLE_5 trial cost?

`{reduces_cost}`. Adaptive used {int(h3['trial_count'])} trials versus
{int(h2['trial_count'])} for fixed BUNDLE_5.

### C. Does adaptive horizon reduce unified boundary optimum collapse?

`{reduces_boundary}`. Adaptive produced
{int(h3['unique_final_alpha_count'])} unique final alpha vectors and
{int(h3['boundary_saturated_case_count'])}/15 boundary-saturated cases, versus
{int(h2['unique_final_alpha_count'])} and
{int(h2['boundary_saturated_case_count'])}/15 for fixed BUNDLE_5.

### D. Which subject/scenario shadows required longer horizons?

{chr(10).join(long_lines)}

These are development-only diagnostics. They do not establish human benefit,
robot safety, or clinical effectiveness.

## Evidence boundary

- DEVELOPMENT + POST_REJECTION_DEVELOPMENT only.
- Independent calibration supplies residual scales only.
- No held-out final test and no prospective cohort.
- P2 V1, objective, five-parameter model, generator, ROM, active reference,
  0.005 tolerance, and 90% support gate are unchanged.
- Final state: `OFFLINE_ONLY`, `DEFAULT_OFF`, `NOT_HUMAN_READY`,
  `NOT_ROBOT_APPROVED`.
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
        raise PermissionError("adaptive prototype must remain default-off")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_semantics_calibration()
    protected_before = _prototype_source_hashes()
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
    adaptive_results = []
    adaptive_case_rows: list[dict[str, Any]] = []
    adaptive_histories: list[pd.DataFrame] = []
    adaptive_guards: list[pd.DataFrame] = []
    truth_landscapes: dict[str, pd.DataFrame] = {}

    for case in cases.to_dict(orient="records"):
        case_id = str(case["case_id"])
        post_rejection = case["development_origin"] == "POST_REJECTION_DEVELOPMENT"

        def run_case() -> None:
            gate.record_truth_access("INITIAL_IDENTIFICATION_AFTER_MANIFEST_FREEZE")
            state = build_initial_research_state(
                str(case["subject_id"]), str(case["scenario_name"])
            )
            result = run_adaptive_shadow(
                state,
                lattice,
                cache,
                gate,
                calibration,
                patient_validity_cache=patient_cache,
            )
            adaptive_results.append(result)
            adaptive_histories.append(result.trial_history)
            adaptive_guards.append(result.decision_guard_audit)
            gate.record_truth_access("POST_ADAPTIVE_FULL_TRUTH_LANDSCAPE")
            landscape = evaluate_full_truth_landscape(result, state, cache)
            truth_landscapes[case_id] = landscape
            gate.record_truth_access("POST_ADAPTIVE_LOCAL_TRUTH_AUDIT")
            _candidate_truth, rounds = post_policy_local_truth_audit(
                result, state, cache
            )
            adaptive_case_rows.append(
                _result_summary_with_truth(result.summary, rounds, landscape)
            )

        if post_rejection:
            subject = dynamic_subject_for_id(str(case["subject_id"]))
            with registered_prospective_subject(subject):
                run_case()
        else:
            run_case()

    adaptive_case = pd.DataFrame(adaptive_case_rows)
    adaptive_case["prototype_variant_id"] = H3_ID
    adaptive_history = pd.concat(
        adaptive_histories, ignore_index=True, sort=False
    )
    adaptive_guard = pd.concat(adaptive_guards, ignore_index=True, sort=False)
    fixed_case = _fixed_case_summary()
    fixed_history = _fixed_history()
    all_cases = pd.concat((fixed_case, adaptive_case), ignore_index=True, sort=False)

    adaptive_specificity = _subject_specificity(
        adaptive_case, truth_landscapes
    )
    adaptive_specificity["prototype_variant_id"] = H3_ID
    fixed_specificity = _fixed_subject_specificity()
    specificity = pd.concat(
        (fixed_specificity, adaptive_specificity), ignore_index=True, sort=False
    )
    boundary = _boundary_table(specificity)
    adaptive_small = adaptive_small_step_recovery(calibration)
    fixed_small = _fixed_small_step()
    comparison = _comparison(all_cases, boundary, fixed_small, adaptive_small)
    horizon_usage = _horizon_usage(fixed_history, adaptive_history, cases)

    _write_csv(output / "adaptive_vs_fixed_comparison.csv", comparison)
    _write_csv(output / "horizon_usage.csv", horizon_usage)
    _write_csv(output / "subject_specificity.csv", specificity)
    _write_csv(output / "boundary_analysis.csv", boundary)
    _write_csv(output / "adaptive_small_step_recovery.csv", adaptive_small)
    _write_csv(output / "adaptive_case_summary.csv", adaptive_case)
    _write_csv(output / "adaptive_trial_history.csv", adaptive_history)
    _write_csv(output / "adaptive_guard_audit.csv", adaptive_guard)
    _write_text(output / "REPORT.md", _report(manifest_sha, comparison, horizon_usage))

    protected_after = _prototype_source_hashes()
    if protected_before != protected_after:
        raise RuntimeError("protected source changed during adaptive shadow")
    if sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("adaptive manifest changed after development truth")
    if adaptive_history["intermediate_execution_count"].sum() != 0:
        raise RuntimeError("an intermediate trajectory was executed")
    if not adaptive_history["model_refit_after_execution"].astype(bool).all():
        raise RuntimeError("model was not refit after every execution")
    if not adaptive_history["full_map_recomputed_after_execution"].astype(bool).all():
        raise RuntimeError("prediction map was not recomputed after every execution")
    if adaptive_history["truth_accessed_before_selection"].astype(bool).any():
        raise RuntimeError("truth entered adaptive horizon selection")

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    indexed = comparison.set_index("prototype_variant_id")
    h2 = indexed.loc[H2_ID]
    h3 = indexed.loc[H3_ID]
    metadata = {
        "prototype_id": PROTOTYPE_ID,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "checkpoint": checkpoint,
        "prior_framework_manifest_sha256": PRIOR_FRAMEWORK_MANIFEST_SHA256,
        "adaptive_horizon_sequence": list(ADAPTIVE_HORIZON_SEQUENCE),
        "development_case_count": len(cases),
        "approaches_fixed_bundle_5_performance": bool(
            int(h3["small_step_recovery"]) == int(h2["small_step_recovery"])
            and float(h3["mean_final_J"])
            <= float(h2["mean_final_J"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE
            and float(h3["mean_global_regret"])
            <= float(h2["mean_global_regret"])
            + OBJECTIVE_EQUIVALENCE_TOLERANCE
        ),
        "reduces_fixed_bundle_5_trial_cost": bool(
            int(h3["trial_count"]) < int(h2["trial_count"])
        ),
        "reduces_uniform_boundary_collapse": bool(
            int(h3["boundary_saturated_case_count"])
            < int(h2["boundary_saturated_case_count"])
            or int(h3["unique_final_alpha_count"])
            > int(h2["unique_final_alpha_count"])
        ),
        "manifest_frozen_before_development_truth": True,
        "manifest_truth_access_count": gate.truth_access_count,
        "future_truth_used_for_authorization": False,
        "held_out_final_test_read": False,
        "prospective_cohort_run": False,
        "intermediate_trajectories_executed": False,
        "model_refit_after_every_execution": True,
        "full_map_recomputed_after_every_execution": True,
        "P2_V1_modified": False,
        "objective_modified": False,
        "five_parameter_model_modified": False,
        "generator_modified": False,
        "ROM_modified": False,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "default_enabled": False,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "artifact_manifest": artifacts,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metadata.json", metadata)
    missing = [name for name in REQUIRED_OUTPUT_FILENAMES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing adaptive prototype artifacts: {missing}")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate default-off P2 adaptive-horizon offline artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH
    )
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_dir, arguments.parameter_map)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

