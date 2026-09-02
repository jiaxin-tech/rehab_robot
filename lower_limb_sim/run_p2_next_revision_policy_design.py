"""Generate the default-off P2 next-revision policy-design shadow artifacts."""

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
from .p2_next_revision_policy_design import (
    BUNDLE_POLICY_ID,
    CALIBRATED_BUNDLE_LENGTHS,
    CANDIDATE_MANIFEST_ID,
    DESIGN_ID,
    EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
    EXPECTED_CALIBRATION_MANIFEST_SHA256,
    EXPECTED_LOCAL_PAIR_PLAN_SHA256,
    FINAL_READY,
    FINAL_REVISE,
    FrozenPolicyDesignManifestGate,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OLD_DEVELOPMENT_LOCAL_P95,
    POLICY_VARIANTS,
    aggregate_policy_summary,
    attach_bundle_posthoc_truth,
    candidate_manifest_payload,
    canonical_json_bytes,
    load_calibration_uncertainty,
    run_policy_design_shadow,
    sha256_file,
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
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_next_revision_policy_design_v1"
)
CORE_SOURCE_PATH = MODULE_DIR / "p2_next_revision_policy_design.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_next_revision_policy_design.py"
CALIBRATION_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_next_revision_independent_calibration_v1"
)
POST_REJECTION_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "post_prospective_rejection_root_cause_audit_v1"
)

REQUIRED_OUTPUT_FILENAMES = (
    "POLICY_DESIGN_CANDIDATE_MANIFEST_V1.json",
    "policy_candidate_manifest_sha.txt",
    "POLICY_DESIGN_DATA_ROLE_AUDIT.md",
    "bundle_supported_decision_definition.md",
    "policy_shadow_trial_history.csv",
    "policy_shadow_summary.csv",
    "bundle_authorization_history.csv",
    "small_step_recovery_audit.csv",
    "guard_failure_repair_comparison.csv",
    "stopping_shadow_comparison.csv",
    "matched_mismatch_policy_comparison.csv",
    "axis_scale_uncertainty_shadow.csv",
    "new_failure_mode_audit.csv",
    "NEXT_REVISION_POLICY_DESIGN_REPORT.md",
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
                _json_safe(dict(payload)),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
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
        "lower_limb_sim/test_p2_next_revision_independent_calibration.py",
        (
            "lower_limb_sim/formal_artifacts/"
            "p2_next_revision_independent_calibration_v1/"
            "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
        ),
        (
            "lower_limb_sim/formal_artifacts/"
            "p2_next_revision_independent_calibration_v1/"
            "independent_bundle_5step_residuals.csv"
        ),
    )
    for path in required:
        try:
            _git_output("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "NEXT_REVISION_POLICY_DESIGN_REQUIRES_CHECKPOINT"
            ) from exc
        committed = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        live = PROJECT_ROOT / path
        if hashlib.sha256(committed).hexdigest() != sha256_file(live):
            raise RuntimeError("NEXT_REVISION_POLICY_DESIGN_REQUIRES_CHECKPOINT")
    artifact_files = _git_output(
        "ls-files",
        "--",
        "lower_limb_sim/formal_artifacts/p2_next_revision_independent_calibration_v1",
    ).splitlines()
    if len(artifact_files) < 30:
        raise RuntimeError("NEXT_REVISION_POLICY_DESIGN_REQUIRES_CHECKPOINT")
    commit = _git_output("rev-parse", "HEAD")
    calibration_commit = _git_output(
        "log", "-1", "--format=%H", "--", required[0]
    )
    if not calibration_commit:
        raise RuntimeError("NEXT_REVISION_POLICY_DESIGN_REQUIRES_CHECKPOINT")
    return {
        "checkpoint_commit": commit,
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "calibration_commit": calibration_commit,
        "calibration_artifact_tracked_count": len(artifact_files),
        "checkpoint_verified": True,
    }


def _case_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, scenario, case_class in ANALYSIS_CASES:
        rows.append(
            {
                "case_id": f"{subject}__{scenario}",
                "subject_id": subject,
                "scenario_name": scenario,
                "case_class": (
                    "MATCHED" if scenario == "matched_linear" else "MISMATCH"
                ),
                "development_origin": "ORIGINAL_P2_DEVELOPMENT",
                "data_role": "DEVELOPMENT_POLICY_SHADOW_ONLY",
                "may_support_future_prospective_claim": False,
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
                "development_origin": "REJECTED_PROSPECTIVE_NOW_DEVELOPMENT",
                "data_role": "POST_REJECTION_DEVELOPMENT_POLICY_SHADOW_ONLY",
                "may_support_future_prospective_claim": False,
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 15 or output["case_id"].duplicated().any():
        raise RuntimeError("policy shadow development case roles changed")
    return output


def _data_role_report(cases: pd.DataFrame) -> str:
    original = int(cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum())
    rejected = int(
        cases["development_origin"].eq(
            "REJECTED_PROSPECTIVE_NOW_DEVELOPMENT"
        ).sum()
    )
    return f"""# Policy design data-role audit

- DEVELOPMENT policy-shadow cases: {len(cases)} ({original} original P2 development + {rejected} rejected-prospective cases permanently reclassified as development).
- INDEPENDENT_CALIBRATION: 12 cases; residual distributions only. Their case IDs are never passed to the policy runner or outcome summarizer.
- FUTURE_PROSPECTIVE: not generated.
- HELD_OUT_FINAL_TEST: not read and no loader is called.
- Candidate manifest is frozen before initial identification or any development truth call.
- Development truth labels outcomes only after each policy path is complete and cannot alter candidate, percentile, bundle length, formula, or stopping K.
- The six rejected-prospective cases cannot support a new prospective success claim; immutable conclusion remains `{PROSPECTIVE_CONCLUSION}`.
"""


def _bundle_definition_report() -> str:
    return f"""# Bundle-supported one-step decision definition

`{BUNDLE_POLICY_ID}` evaluates only straight, same-axis, same-direction 2/3/5-step formal-grid paths. Every start, intermediate, first-step, and endpoint node must exist in the unchanged generator lattice, retain the active-reference provenance, satisfy the synthetic patient-envelope fixture, and be model-supported at the unchanged {MODEL_SUPPORT_COVERAGE_GATE_PERCENT:.0f}% gate.

The research evidence margin is `-deltaJ_pred(start, endpoint) - U_bundle(scale[,axis]) - {OBJECTIVE_EQUIVALENCE_TOLERANCE:.3f}`. The {OBJECTIVE_EQUIVALENCE_TOLERANCE:.3f} term remains the meaningful-improvement tolerance; it is not redefined as a safety threshold. `U_bundle` comes directly from independent 2/3/5-step residual distributions. Neither `n*U1` nor `sqrt(n)*U1` is used.

An eligible endpoint authorizes exactly the next adjacent formal-grid step. It never queues the remaining endpoint path. Immediately after that one simulated trajectory, the five-parameter model is refit once, the entire prediction map is recomputed, and the authorization expires. A new round must independently re-evaluate all bundle evidence.

This is default-off synthetic offline research. It is not a human or robot execution rule.
"""


def _small_step_recovery(uncertainty: Any) -> pd.DataFrame:
    source = pd.read_csv(
        POST_REJECTION_DIRECTORY / "prospective_small_step_accumulation.csv"
    )
    rows: list[dict[str, Any]] = []
    variants = [spec for spec in POLICY_VARIANTS if spec.bundle_decision_id]
    for path_id, group in source.groupby("path_id", sort=True):
        ordered = group.sort_values("step_number")
        first = ordered.iloc[0]
        old_margin = (
            -float(first["single_step_deltaJ_pred"])
            - OLD_DEVELOPMENT_LOCAL_P95
            - OBJECTIVE_EQUIVALENCE_TOLERANCE
        )
        for spec in variants:
            candidates: list[dict[str, Any]] = []
            for length in CALIBRATED_BUNDLE_LENGTHS:
                row = ordered.loc[ordered["step_number"].eq(length)]
                if row.empty:
                    continue
                item = row.iloc[0]
                bound = uncertainty.bundle_bound(
                    spec, length, str(item["coordinate"])
                )
                predicted = float(item["cumulative_endpoint_deltaJ_pred"])
                margin = -predicted - bound - OBJECTIVE_EQUIVALENCE_TOLERANCE
                candidates.append(
                    {
                        "length": length,
                        "predicted": predicted,
                        "truth": float(item["cumulative_endpoint_deltaJ_truth"]),
                        "uncertainty": bound,
                        "margin": margin,
                        "authorized": margin >= -1e-15,
                    }
                )
            eligible = [item for item in candidates if item["authorized"]]
            selected = (
                sorted(
                    eligible,
                    key=lambda item: (-item["margin"], item["length"]),
                )[0]
                if eligible
                else (
                    sorted(
                        candidates,
                        key=lambda item: (-item["margin"], item["length"]),
                    )[0]
                    if candidates
                    else None
                )
            )
            rows.append(
                {
                    "path_id": path_id,
                    "case_id": str(first["case_id"]),
                    "coordinate": str(first["coordinate"]),
                    "direction": str(first["direction"]),
                    "policy_id": spec.policy_id,
                    "old_G2_single_step_margin": old_margin,
                    "old_G2_first_step_authorized": old_margin >= -1e-15,
                    "bundle_scale_used": selected["length"] if selected else np.nan,
                    "predicted_cumulative_improvement": (
                        -selected["predicted"] if selected else np.nan
                    ),
                    "calibrated_uncertainty": (
                        selected["uncertainty"] if selected else np.nan
                    ),
                    "bundle_lower_bound_margin": (
                        selected["margin"] if selected else np.nan
                    ),
                    "first_step_authorized": bool(
                        selected is not None and selected["authorized"]
                    ),
                    "posthoc_truth_cumulative_delta_J": (
                        selected["truth"] if selected else np.nan
                    ),
                    "posthoc_truth_supports_bundle": bool(
                        selected is not None
                        and selected["truth"] < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                    ),
                    "recovered_true_accumulation_path": bool(
                        selected is not None
                        and selected["authorized"]
                        and selected["truth"] < -OBJECTIVE_EQUIVALENCE_TOLERANCE
                        and old_margin < 0.0
                    ),
                    "truth_used_for_authorization": False,
                    "rule_modified_for_path": False,
                }
            )
    return pd.DataFrame(rows)


def _direct_guard_comparison(
    comparator: pd.DataFrame,
    truth_candidates: pd.DataFrame,
) -> pd.DataFrame:
    if comparator.empty:
        return pd.DataFrame()
    truth = truth_candidates.loc[
        truth_candidates["policy_id"].eq(POLICY_VARIANTS[0].policy_id)
    ][
        ["case_id", "iteration", "trajectory_id", "delta_J_truth"]
    ].copy()
    joined = comparator.merge(
        truth,
        on=["case_id", "iteration", "trajectory_id"],
        how="inner",
        validate="one_to_one",
    )
    true = joined["delta_J_truth"].lt(-OBJECTIVE_EQUIVALENCE_TOLERANCE)
    rows = []
    for identifier, column in (
        ("D0_CURRENT_GLOBAL_GUARD", "D0_current_guard_authorized"),
        ("OLD_G2_DEVELOPMENT_LOCAL_P95", "old_G2_development_P95_authorized"),
        ("D1_INDEPENDENT_ONE_STEP_P95", "D1_independent_P95_authorized"),
    ):
        authorized = joined[column].astype(bool)
        rows.append(
            {
                "comparison_id": identifier,
                "comparison_type": "DIRECT_ONE_STEP",
                "evaluated_candidate_count": len(joined),
                "authorized_count": int(authorized.sum()),
                "true_meaningful_opportunity_count": int(true.sum()),
                "missed_improvement": int((true & ~authorized).sum()),
                "false_improvement": int((~true & authorized).sum()),
                "development_truth_used_to_define_rule": False,
            }
        )
    return pd.DataFrame(rows)


def _bundle_guard_comparison(bundle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_policies = (
        POLICY_VARIANTS[1].policy_id,
        POLICY_VARIANTS[3].policy_id,
        POLICY_VARIANTS[4].policy_id,
    )
    for policy_id in selected_policies:
        frame = bundle.loc[bundle["policy_id"].eq(policy_id)]
        round_rows = []
        for _, group in frame.groupby(["case_id", "iteration"], sort=False):
            true = group["truth_bundle_meaningful_improvement"].astype(bool)
            selected = group["selected_authorization"].astype(bool)
            round_rows.append(
                {
                    "true": bool(true.any()),
                    "authorized": bool(selected.any()),
                    "false": bool(
                        (selected & ~group["truth_bundle_meaningful_improvement"].astype(bool)).any()
                    ),
                }
            )
        table = pd.DataFrame(round_rows)
        rows.append(
            {
                "comparison_id": policy_id,
                "comparison_type": "BUNDLE_SUPPORTED",
                "evaluated_candidate_count": len(frame),
                "authorized_count": int(table["authorized"].sum()) if not table.empty else 0,
                "true_meaningful_opportunity_count": int(table["true"].sum()) if not table.empty else 0,
                "missed_improvement": int((table["true"] & ~table["authorized"]).sum()) if not table.empty else 0,
                "false_improvement": int(table["false"].sum()) if not table.empty else 0,
                "development_truth_used_to_define_rule": False,
            }
        )
    return pd.DataFrame(rows)


def _matched_mismatch_summary(
    case_summary: pd.DataFrame, cases: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    selected_false = (
        bundle.loc[bundle["selected_authorization"].astype(bool)]
        .groupby(["case_id", "policy_id"], as_index=False)
        .agg(
            bundle_endpoint_false_improvement=(
                "bundle_induced_false_improvement",
                "sum",
            )
        )
    )
    merged = case_summary.merge(
        cases[["case_id", "case_class", "development_origin"]],
        on="case_id",
        validate="many_to_one",
    ).merge(selected_false, on=["case_id", "policy_id"], how="left")
    merged["bundle_endpoint_false_improvement"] = merged[
        "bundle_endpoint_false_improvement"
    ].fillna(0).astype(int)
    return (
        merged.groupby(["policy_id", "case_class"], as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            trials=("number_of_executed_trials", "sum"),
            EXPLORE=("number_of_explore_trials", "sum"),
            EXPLOIT=("number_of_exploit_trials", "sum"),
            bundle_authorizations=("number_of_bundle_authorized_trials", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            executed_false_improvement=(
                "number_of_executed_false_improvements",
                "sum",
            ),
            bundle_endpoint_false_improvement=(
                "bundle_endpoint_false_improvement",
                "sum",
            ),
            final_J=("final_best_actual_J", "mean"),
            regret=("global_truth_regret", "mean"),
        )
    )


def _stopping_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        summary["policy_id"].isin(
            (POLICY_VARIANTS[1].policy_id, POLICY_VARIANTS[2].policy_id)
        )
    ].copy()
    output = (
        selected.groupby(
            ["policy_id", "stopping_rule_id", "stopping_k"],
            dropna=False,
            as_index=False,
            sort=False,
        )
        .agg(
            trials=("number_of_executed_trials", "sum"),
            EXPLORE=("number_of_explore_trials", "sum"),
            EXPLOIT=("number_of_exploit_trials", "sum"),
            low_value_exploration=("low_decision_value_exploration_count", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            false_improvement=("number_of_executed_false_improvements", "sum"),
            final_J=("final_best_actual_J", "mean"),
            regret=("global_truth_regret", "mean"),
        )
    )
    s0 = output.loc[output["policy_id"].eq(POLICY_VARIANTS[1].policy_id)].iloc[0]
    output["trials_removed_vs_R1"] = int(s0["trials"]) - output["trials"]
    output["final_J_change_vs_R1"] = output["final_J"] - float(s0["final_J"])
    output["regret_change_vs_R1"] = output["regret"] - float(s0["regret"])
    output["S2_K_tuned_from_shadow"] = False
    return output


def _axis_scale_shadow(uncertainty: Any, policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for length in CALIBRATED_BUNDLE_LENGTHS:
        for axis in ("hip", "knee", "phase"):
            rows.append(
                {
                    "row_type": "UNCERTAINTY_CANDIDATE",
                    "decision_scale": f"{length}-step",
                    "coordinate": axis,
                    "scale_only_P95": uncertainty.scale_p95[length],
                    "scale_axis_P95": uncertainty.scale_axis_p95[length][axis],
                    "difference_axis_minus_scale": (
                        uncertainty.scale_axis_p95[length][axis]
                        - uncertainty.scale_p95[length]
                    ),
                    "policy_id": "",
                    "trials": np.nan,
                    "missed_improvement": np.nan,
                    "false_improvement": np.nan,
                    "final_J": np.nan,
                    "regret": np.nan,
                }
            )
    for policy_id in (POLICY_VARIANTS[2].policy_id, POLICY_VARIANTS[4].policy_id):
        row = policy_summary.loc[policy_summary["policy_id"].eq(policy_id)].iloc[0]
        rows.append(
            {
                "row_type": "POLICY_OUTCOME",
                "decision_scale": "ALL",
                "coordinate": "ALL",
                "scale_only_P95": np.nan,
                "scale_axis_P95": np.nan,
                "difference_axis_minus_scale": np.nan,
                "policy_id": policy_id,
                "trials": row["trials"],
                "missed_improvement": row["missed_improvement"],
                "false_improvement": row["false_improvement"],
                "final_J": row["final_J"],
                "regret": row["regret"],
            }
        )
    output = pd.DataFrame(rows)
    output["winner_selected"] = False
    return output


def _failure_mode_audit(
    case_summary: pd.DataFrame,
    history: pd.DataFrame,
    bundle: pd.DataFrame,
) -> pd.DataFrame:
    selected_bundle = bundle.loc[bundle["selected_authorization"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    bounds = (
        OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["hip_amplitude_delta_deg"],
        OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_amplitude_delta_deg"],
        OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_phase_shift"],
    )
    for item in case_summary.to_dict(orient="records"):
        key = (str(item["case_id"]), str(item["policy_id"]))
        trials = history.loc[
            history["case_id"].astype(str).eq(key[0])
            & history["policy_id"].astype(str).eq(key[1])
        ].sort_values("iteration")
        authorizations = selected_bundle.loc[
            selected_bundle["case_id"].astype(str).eq(key[0])
            & selected_bundle["policy_id"].astype(str).eq(key[1])
        ].sort_values("iteration")
        auth_directions = list(
            zip(
                authorizations["coordinate"].astype(str),
                authorizations["direction"].astype(str),
            )
        )
        repeated_no_progress = False
        bundle_trials = trials.loc[
            trials["selection_mode"].eq("BUNDLE_SUPPORTED_ONE_STEP")
        ]
        if len(bundle_trials) >= 2:
            values = bundle_trials["best_actual_J_after"].to_numpy(dtype=float)
            repeated_no_progress = bool(np.any(np.isclose(np.diff(values), 0.0)))
        direction_lock = any(
            auth_directions[index] == auth_directions[index + 1] == auth_directions[index + 2]
            for index in range(max(0, len(auth_directions) - 2))
        )
        reversal = any(
            auth_directions[index][0] == auth_directions[index + 1][0]
            and auth_directions[index][1] != auth_directions[index + 1][1]
            for index in range(max(0, len(auth_directions) - 1))
        )
        lengths = authorizations["bundle_length"].astype(int).tolist()
        scale_oscillation = sum(
            lengths[index] != lengths[index + 1]
            for index in range(max(0, len(lengths) - 1))
        ) >= 2
        final_values = (
            float(item["final_operating_alpha_hip"]),
            float(item["final_operating_alpha_knee"]),
            float(item["final_operating_alpha_phase"]),
        )
        boundary = bool(
            authorizations.shape[0]
            and any(
                np.isclose(value, low, atol=1e-12, rtol=0.0)
                or np.isclose(value, high, atol=1e-12, rtol=0.0)
                for value, (low, high) in zip(final_values, bounds)
            )
        )
        checks = {
            "bundle_induced_false_improvement": bool(
                authorizations["bundle_induced_false_improvement"].astype(bool).any()
            ),
            "repeated_bundle_authorization_without_progress": repeated_no_progress,
            "direction_lock_in": direction_lock,
            "boundary_chasing": boundary,
            "model_update_oscillation": reversal,
            "bundle_scale_oscillation": scale_oscillation,
            "excessive_authorization_followed_by_reversal": reversal,
            "premature_S2_stop": bool(
                str(item["stopping_rule_id"]) == "S2_DECISION_VALUE_K2"
                and int(item["premature_conservative_stops"]) > 0
            ),
        }
        for mode, observed in checks.items():
            rows.append(
                {
                    "case_id": key[0],
                    "policy_id": key[1],
                    "failure_mode": mode,
                    "observed": observed,
                    "policy_modified_in_response": False,
                    "future_design_review_required": observed,
                }
            )
    return pd.DataFrame(rows)


def _classify(
    summary: pd.DataFrame,
    small: pd.DataFrame,
    mismatch: pd.DataFrame,
    stopping: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    by_id = summary.set_index("policy_id")
    r0 = by_id.loc[POLICY_VARIANTS[0].policy_id]
    r2 = by_id.loc[POLICY_VARIANTS[2].policy_id]
    recovered = int(
        small.loc[
            small["policy_id"].eq(POLICY_VARIANTS[2].policy_id),
            "recovered_true_accumulation_path",
        ].sum()
    )
    mismatch_rows = mismatch.loc[mismatch["case_class"].eq("MISMATCH")].set_index(
        "policy_id"
    )
    r0_mismatch_false = int(
        mismatch_rows.loc[POLICY_VARIANTS[0].policy_id, "executed_false_improvement"]
    )
    r2_mismatch_false = int(
        mismatch_rows.loc[POLICY_VARIANTS[2].policy_id, "executed_false_improvement"]
        + mismatch_rows.loc[
            POLICY_VARIANTS[2].policy_id,
            "bundle_endpoint_false_improvement",
        ]
    )
    s2 = stopping.loc[
        stopping["policy_id"].eq(POLICY_VARIANTS[2].policy_id)
    ].iloc[0]
    criteria = {
        "small_step_recovery_present": recovered > 0,
        "missed_improvement_not_worse": int(r2["missed_improvement"])
        <= int(r0["missed_improvement"]),
        "executed_false_improvement_not_worse": int(r2["false_improvement"])
        <= int(r0["false_improvement"]),
        "mean_final_J_not_worse_by_0p005": float(r2["final_J"])
        <= float(r0["final_J"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "mean_regret_not_worse_by_0p005": float(r2["regret"])
        <= float(r0["regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "mismatch_false_improvement_not_increased": r2_mismatch_false
        <= r0_mismatch_false,
        "S2_not_materially_harmful": float(s2["final_J_change_vs_R1"])
        <= OBJECTIVE_EQUIVALENCE_TOLERANCE,
    }
    status = FINAL_READY if all(criteria.values()) else FINAL_REVISE
    return status, {
        **criteria,
        "recovered_path_count": recovered,
        "R0_mismatch_false_count": r0_mismatch_false,
        "R2_mismatch_false_count": r2_mismatch_false,
        "mismatch_reliability_label": (
            "BUNDLE_DECISION_NOT_RELIABLE_UNDER_MISMATCH"
            if r2_mismatch_false > r0_mismatch_false
            else "NO_CLEAR_MISMATCH_FALSE_IMPROVEMENT_INCREASE"
        ),
    }


def _plot_outputs(summary: pd.DataFrame, small: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - 0.18, summary["final_J"], 0.36, label="mean final J")
    axis.bar(x + 0.18, summary["regret"], 0.36, label="mean global regret")
    axis.set_xticks(x, summary["policy_id"].str.slice(0, 2))
    axis.set(title="Development-only P2 next-revision shadow", ylabel="objective")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "policy_shadow_final_J_and_regret.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    grouped = small.groupby("policy_id", as_index=False).agg(
        recovered=("recovered_true_accumulation_path", "sum")
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(grouped["policy_id"].str.slice(0, 2), grouped["recovered"], color="#4c78a8")
    axis.set(title="Recovery of nine historical small-step paths", ylabel="recovered paths")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "small_step_recovery.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: Any) -> str:
    return f"{float(value):.9f}"


def _final_report(
    manifest_sha: str,
    summary: pd.DataFrame,
    small: pd.DataFrame,
    stopping: pd.DataFrame,
    matched: pd.DataFrame,
    failure: pd.DataFrame,
    final_status: str,
    criteria: Mapping[str, Any],
) -> str:
    by_id = summary.set_index("policy_id")
    lines = []
    for spec in POLICY_VARIANTS:
        row = by_id.loc[spec.policy_id]
        lines.append(
            f"- {spec.policy_id}: trials={int(row['trials'])}, EXPLORE={int(row['EXPLORE'])}, "
            f"EXPLOIT={int(row['EXPLOIT'])}, missed={int(row['missed_improvement'])}, "
            f"false={int(row['false_improvement'])}, final J={_fmt(row['final_J'])}, "
            f"regret={_fmt(row['regret'])}."
        )
    recovery = small.groupby("policy_id")["recovered_true_accumulation_path"].sum()
    p95_recovery = int(recovery.get(POLICY_VARIANTS[2].policy_id, 0))
    p99_recovery = int(recovery.get(POLICY_VARIANTS[3].policy_id, 0))
    observed = failure.loc[failure["observed"].astype(bool)]
    r1 = stopping.loc[stopping["policy_id"].eq(POLICY_VARIANTS[1].policy_id)].iloc[0]
    r2 = stopping.loc[stopping["policy_id"].eq(POLICY_VARIANTS[2].policy_id)].iloc[0]
    r2_mismatch = matched.loc[
        matched["policy_id"].eq(POLICY_VARIANTS[2].policy_id)
        & matched["case_class"].eq("MISMATCH")
    ].iloc[0]
    return f"""# {DESIGN_ID}

Candidate manifest SHA-256: `{manifest_sha}`

## 通俗结论

下一版候选没有把旧 P95 简单替换为新 P95。直接一步分支仍是原 P2 V1 的 G0；新增的是 bundle-supported one-step commitment：当一步本身不足以跨过 0.005 时，只查看同轴、同方向、全部节点合规且有独立 2/3/5-step residual calibration 的累计 endpoint。即使 endpoint 证据通过，本轮也只执行第一个 formal-grid 邻点。随后立即 refit 一次五参数模型、重算整张 prediction map，并使旧授权失效。

这样保留了 trial-by-trial 更新；不存在一次执行整个 bundle、预排后续轨迹或 analytic uncertainty scaling。

## R0--R4 development shadow

{chr(10).join(lines)}

## Small-step repair

- P95 scale bundle recovered {p95_recovery}/9 historical direction-consistent paths.
- P99 sensitivity recovered {p99_recovery}/9.
- Old G2 remains a direct one-step comparator; no path-specific rule or percentile was added after truth.

## Scale-only versus scale-by-axis

R2 is scale-only P95 and R4 is scale-by-axis P95. Their outcomes are reported without selecting a winner: R2 final J `{_fmt(by_id.loc[POLICY_VARIANTS[2].policy_id, 'final_J'])}`, R4 `{_fmt(by_id.loc[POLICY_VARIANTS[4].policy_id, 'final_J'])}`; R2 missed `{int(by_id.loc[POLICY_VARIANTS[2].policy_id, 'missed_improvement'])}`, R4 `{int(by_id.loc[POLICY_VARIANTS[4].policy_id, 'missed_improvement'])}`.

## Matched/mismatch and stopping

- R2 mismatch executed false improvements: {int(r2_mismatch['executed_false_improvement'])}; selected bundle endpoint false improvements: {int(r2_mismatch['bundle_endpoint_false_improvement'])}.
- Mismatch label: `{criteria['mismatch_reliability_label']}`.
- S2 K=2 changed total trials from {int(r1['trials'])} to {int(r2['trials'])}, mean final J by {_fmt(r2['final_J_change_vs_R1'])}, and regret by {_fmt(r2['regret_change_vs_R1'])}. K was not tuned.

## New failure modes

Observed case-policy failure-mode rows: {len(observed)}. They are preserved in `new_failure_mode_audit.csv`; none caused an in-task policy change.

## Data and evidence boundary

- Policy outcomes use only 9 original development and 6 rejected-prospective-now-development cases.
- The 12 independent calibration cases supply residual distributions only and never enter policy outcome selection.
- No new prospective cohort was generated; held-out final test was not read.
- P2 V1 remains unchanged and the new candidate is default-off.
- Active reference, ROM, `theta_shank = q_hip - q_knee`, five-parameter model, objective, generator, 0.005 tolerance, and 90% support gate are unchanged.
- Hardware/control/collection/safety are unchanged; no robot was connected.

## Formal status

`{final_status}`

This is synthetic offline development evidence, not prospective validation, human readiness, robot-motion approval, safety validation, or clinical evidence. Status remains `{NOT_HUMAN_READY}` and `{NOT_ROBOT_MOTION_APPROVED}`.
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
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    uncertainty = load_calibration_uncertainty()
    protected_before = _protected_source_hashes()

    # This exact candidate set is the first persisted artifact.  No initial
    # identification or development truth call occurs above this line.
    manifest = candidate_manifest_payload(
        uncertainty,
        checkpoint_commit=str(checkpoint["checkpoint_commit"]),
        protected_source_sha256=protected_before,
    )
    manifest_path = output / "POLICY_DESIGN_CANDIDATE_MANIFEST_V1.json"
    _write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    (output / "policy_candidate_manifest_sha.txt").write_text(
        manifest_sha + "\n", encoding="utf-8"
    )
    gate = FrozenPolicyDesignManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    cases = _case_table()
    (output / "POLICY_DESIGN_DATA_ROLE_AUDIT.md").write_text(
        _data_role_report(cases), encoding="utf-8"
    )
    (output / "bundle_supported_decision_definition.md").write_text(
        _bundle_definition_report(), encoding="utf-8"
    )

    lattice = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometrically admissible lattice changed")
    cache = build_trajectory_component_cache(lattice)
    patient_cache: dict[tuple[float, float, float], bool] = {}

    results = []
    states: dict[str, Any] = {}
    bundle_frames: list[pd.DataFrame] = []
    comparator_frames: list[pd.DataFrame] = []
    truth_landscapes: dict[str, pd.DataFrame] = {}
    truth_candidate_frames: list[pd.DataFrame] = []
    truth_round_frames: list[pd.DataFrame] = []

    for case in cases.to_dict(orient="records"):
        case_id = str(case["case_id"])
        is_rejected = case["development_origin"] == "REJECTED_PROSPECTIVE_NOW_DEVELOPMENT"

        def run_case() -> None:
            gate.record_truth_access("INITIAL_IDENTIFICATION_AFTER_MANIFEST_FREEZE")
            state = build_initial_research_state(
                str(case["subject_id"]), str(case["scenario_name"])
            )
            states[case_id] = state
            case_results = []
            for spec in POLICY_VARIANTS:
                result, bundle_history, comparator = run_policy_design_shadow(
                    state,
                    spec,
                    lattice,
                    cache,
                    gate,
                    uncertainty,
                    patient_validity_cache=patient_cache,
                )
                results.append(result)
                case_results.append(result)
                if not bundle_history.empty:
                    bundle_frames.append(bundle_history)
                if not comparator.empty:
                    comparator_frames.append(comparator)
            gate.record_truth_access("POST_POLICY_FULL_TRUTH_LANDSCAPE")
            landscape = evaluate_full_truth_landscape(case_results[0], state, cache)
            truth_landscapes[case_id] = landscape
            for result in case_results:
                gate.record_truth_access("POST_POLICY_LOCAL_TRUTH_AUDIT")
                candidates_frame, rounds_frame = post_policy_local_truth_audit(
                    result, state, cache
                )
                if not candidates_frame.empty:
                    truth_candidate_frames.append(candidates_frame)
                if not rounds_frame.empty:
                    truth_round_frames.append(rounds_frame)

        if is_rejected:
            subject = dynamic_subject_for_id(str(case["subject_id"]))
            with registered_prospective_subject(subject):
                run_case()
        else:
            run_case()

    history = pd.concat(
        [result.trial_history for result in results], ignore_index=True, sort=False
    )
    bundle_raw = (
        pd.concat(bundle_frames, ignore_index=True, sort=False)
        if bundle_frames
        else pd.DataFrame()
    )
    bundle = attach_bundle_posthoc_truth(bundle_raw, truth_landscapes)
    comparator = (
        pd.concat(comparator_frames, ignore_index=True, sort=False)
        if comparator_frames
        else pd.DataFrame()
    )
    truth_candidates = pd.concat(
        truth_candidate_frames, ignore_index=True, sort=False
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
    policy_summary = aggregate_policy_summary(case_summary)
    small = _small_step_recovery(uncertainty)
    guard = pd.concat(
        (
            _direct_guard_comparison(comparator, truth_candidates),
            _bundle_guard_comparison(bundle),
        ),
        ignore_index=True,
        sort=False,
    )
    stopping = _stopping_comparison(case_summary)
    matched = _matched_mismatch_summary(case_summary, cases, bundle)
    axis = _axis_scale_shadow(uncertainty, policy_summary)
    failure = _failure_mode_audit(case_summary, history, bundle)
    final_status, criteria = _classify(policy_summary, small, matched, stopping)

    outputs = {
        "policy_shadow_trial_history.csv": history,
        "policy_shadow_summary.csv": policy_summary,
        "bundle_authorization_history.csv": bundle,
        "small_step_recovery_audit.csv": small,
        "guard_failure_repair_comparison.csv": guard,
        "stopping_shadow_comparison.csv": stopping,
        "matched_mismatch_policy_comparison.csv": matched,
        "axis_scale_uncertainty_shadow.csv": axis,
        "new_failure_mode_audit.csv": failure,
        "policy_shadow_case_summary.csv": case_summary,
        "policy_shadow_truth_rounds.csv": truth_rounds,
        "direct_one_step_comparator_history.csv": comparator,
    }
    for name, table in outputs.items():
        _write_csv(output / name, table)

    _plot_outputs(policy_summary, small, output)
    (output / "NEXT_REVISION_POLICY_DESIGN_REPORT.md").write_text(
        _final_report(
            manifest_sha,
            policy_summary,
            small,
            stopping,
            matched,
            failure,
            final_status,
            criteria,
        ),
        encoding="utf-8",
    )

    protected_after = _protected_source_hashes()
    if protected_before != protected_after:
        raise RuntimeError("protected scientific or robot-side source changed")
    if sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("candidate manifest changed after shadow truth")
    if EXPECTED_CALIBRATION_MANIFEST_SHA256 != sha256_file(
        CALIBRATION_DIRECTORY / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
    ):
        raise RuntimeError("calibration manifest changed during policy shadow")

    artifact_manifest = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            artifact_manifest[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    metadata = {
        "design_id": DESIGN_ID,
        "candidate_manifest_id": CANDIDATE_MANIFEST_ID,
        "candidate_manifest_sha256": manifest_sha,
        "final_status": final_status,
        "classification_criteria": criteria,
        "checkpoint": checkpoint,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA256,
        "local_pair_plan_sha256": EXPECTED_LOCAL_PAIR_PLAN_SHA256,
        "bundle_pair_plan_sha256": EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
        "policy_variant_ids": [spec.policy_id for spec in POLICY_VARIANTS],
        "development_case_count": len(cases),
        "original_development_case_count": int(
            cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum()
        ),
        "post_rejection_development_case_count": int(
            cases["development_origin"].eq(
                "REJECTED_PROSPECTIVE_NOW_DEVELOPMENT"
            ).sum()
        ),
        "independent_calibration_case_count": 12,
        "calibration_cases_used_for_policy_outcome_selection": False,
        "future_prospective_generated": False,
        "held_out_final_test_read": False,
        "candidate_manifest_frozen_before_truth": True,
        "manifest_truth_gate_access_count": gate.truth_access_count,
        "manifest_truth_gate_stages": gate.truth_access_stages,
        "development_truth_modified_candidate": False,
        "percentile_search_performed": False,
        "analytic_uncertainty_scaling_used": False,
        "bundle_lengths_used": list(CALIBRATED_BUNDLE_LENGTHS),
        "bundle_authorizes_next_one_step_only": True,
        "model_refit_after_every_execution": bool(
            history["model_refit_after_execution"].astype(bool).all()
        ),
        "full_map_recomputed_after_every_execution": bool(
            history["full_map_recomputed_after_execution"].astype(bool).all()
        ),
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "P2_V1_replaced": False,
        "new_policy_default_enabled": False,
        "human_ready": NOT_HUMAN_READY,
        "robot_motion_approved": NOT_ROBOT_MOTION_APPROVED,
        "robot_connected": False,
        "prospective_conclusion_revised": False,
        "immutable_prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "immutable_prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "artifact_manifest": artifact_manifest,
        "runtime_seconds": time.perf_counter() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    missing = [name for name in REQUIRED_OUTPUT_FILENAMES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing policy-design outputs: {missing}")
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
