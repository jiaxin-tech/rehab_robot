"""Generate P2 decision-rule semantics audit artifacts (offline only)."""

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
from .p2_decision_rule_semantics_audit import (
    ADDITIVE_ASSUMPTION,
    AUDIT_ID,
    EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
    EXPECTED_CALIBRATION_MANIFEST_SHA256,
    MANIFEST_ID,
    MORE_EVIDENCE,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    NOT_SUFFICIENT,
    PRIMARY_BLOCKER,
    SEMANTIC_VARIANTS,
    FrozenSemanticsManifestGate,
    attach_bundle_posthoc_truth,
    candidate_manifest_payload,
    canonical_json_bytes,
    load_semantics_calibration,
    run_semantic_shadow,
    sha256_file,
    small_step_semantic_recovery,
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
CORE_SOURCE_PATH = MODULE_DIR / "p2_decision_rule_semantics_audit.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_decision_rule_semantics_audit.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_decision_rule_semantics_audit_v1"
)
REQUIRED_OUTPUT_FILENAMES = (
    "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1.json",
    "current_rule_semantics.md",
    "semantic_shadow_comparison.csv",
    "small_step_semantic_recovery.csv",
    "false_improvement_semantic_audit.csv",
    "DECISION_RULE_SEMANTICS_REPORT.md",
    "DATA_ROLE_AUDIT.md",
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
    content = (
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
            stream.write(content)
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
        "lower_limb_sim/test_p2_next_revision_independent_calibration.py",
        (
            "lower_limb_sim/formal_artifacts/"
            "p2_next_revision_independent_calibration_v1/"
            "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
        ),
        (
            "lower_limb_sim/formal_artifacts/"
            "post_prospective_rejection_root_cause_audit_v1/"
            "designated_bundle_validation_pair_plan.csv"
        ),
    )
    for relative in required:
        try:
            _git_output("ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("SEMANTICS_AUDIT_REQUIRES_CHECKPOINT") from exc
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != sha256_file(PROJECT_ROOT / relative):
            raise RuntimeError("SEMANTICS_AUDIT_REQUIRES_CHECKPOINT")
    return {
        "checkpoint_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "checkpoint_verified": True,
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
                "data_role": "DEVELOPMENT_POLICY_SHADOW_ONLY",
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
                "data_role": "POST_REJECTION_DEVELOPMENT_POLICY_SHADOW_ONLY",
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 15 or output["case_id"].duplicated().any():
        raise RuntimeError("development case roles changed")
    return output


def _current_rule_report(calibration: Any) -> str:
    support = int(calibration.direction_evidence["direction_support_count"].sum())
    contradict = int(
        calibration.direction_evidence["direction_contradiction_count"].sum()
    )
    return f"""# Current P2 decision-rule semantics

## Direct rule

The active P2 V1 research guard computes `I_pred = -deltaJ_pred` and authorizes a supported local candidate only when `I_pred - U_global_max - 0.005 > 0` (`research_decision_guarded_sequential_personalization.py`, lines 502-550). The unchanged `0.005` originates as `OBJECTIVE_EQUIVALENCE_TOLERANCE` (`mechanical_objective.py`, lines 18-19). The same value defines mechanically equivalent ranking and actual-trial acceptance; it is a minimum meaningful objective-magnitude convention, not an uncertainty estimate, probability, or robot-safety threshold.

`U` is an empirical absolute error statistic for predicted objective difference. In current P2 V1 it is the maximum error from the current global validation-pair audit. In the default-off bundle candidate it is an independently calibrated scale-specific P95 endpoint residual. It describes model-decision disagreement; it does not change what 0.005 means.

## Bundle candidate rule

The current default-off bundle comparator uses `I_endpoint_pred - U_scale_P95 - 0.005 > 0`, with 2/3/5-step residual evidence of {calibration.bundle_scale_p95}. A passing endpoint authorizes only the next adjacent formal-grid step, after which the five-parameter model is refit and the full prediction map is recomputed.

## Evidence for addition

The repository contains the additive formula as an implementation/design choice, but the searched code, protocol reports, calibration reports, and manifests contain no theorem, loss-derived risk allocation, physical law, or preregistered scientific argument requiring `required_margin = 0.005 + U`. The calibration report explicitly says it selected no threshold or policy and inferred no universal scale law. Therefore the correct audit label is:

`{ADDITIVE_ASSUMPTION}`

This does not prove the additive rule is wrong. It means its necessity is not established. The shadow comparators therefore separate magnitude from direction without changing either the 0.005 value or the residual evidence.

Independent one-step direction evidence is descriptive: {support}/324 pairs agree in sign and {contradict}/324 reverse. S1 calls a stratum direction-supported only when supporting pairs outnumber contradicting pairs; it makes no probability claim. S2 uses the transparent research interval `[deltaJ_pred-U_P95, deltaJ_pred+U_P95]` and requires its upper endpoint to remain below zero.
"""


def _data_role_report(cases: pd.DataFrame) -> str:
    original = int(cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum())
    post = int(cases["development_origin"].eq("POST_REJECTION_DEVELOPMENT").sum())
    return f"""# Data-role audit

- DEVELOPMENT shadow outcomes: {len(cases)} cases ({original} original P2 development + {post} rejected prospective cases already reclassified as post-rejection development).
- INDEPENDENT_CALIBRATION: 12 cases; used only for 324 one-step residual/direction observations and 216 direct endpoint residuals at each of 2/3/5 steps.
- The 12 calibration case IDs are not passed to the policy runner, final-J calculation, regret calculation, missed-improvement calculation, or false-improvement calculation.
- Calibration truth selects no semantic candidate and contributes no policy-performance row.
- The exact S0-S3 manifest is persisted and SHA-gated before initial identification or any development truth access.
- Development truth is attached only after each complete semantic path; it cannot alter gate definitions, percentile, 0.005, bundle lengths, or interpretation criteria.
- No new prospective cohort is generated. The immutable prior prospective conclusion remains `{PROSPECTIVE_CONCLUSION}`.
- HELD_OUT_FINAL_TEST is not loaded or read.
- This task is synthetic offline research only; no hardware, control, collection, safety, or robot connector is imported by the semantics core/runner.
"""


def _false_audit(
    history: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in history.loc[history["trial_purpose"].eq("EXPLOIT")].to_dict(
        orient="records"
    ):
        rows.append(
            {
                "case_id": row["case_id"],
                "semantic_id": row["policy_id"],
                "iteration": row["iteration"],
                "event_type": (
                    "BUNDLE_AUTHORIZED_NEXT_STEP"
                    if row["selection_mode"] == "BUNDLE_SUPPORTED_ONE_STEP"
                    else "DIRECT_ONE_STEP"
                ),
                "predicted_delta_J": row["delta_J_pred_one_step"],
                "truth_delta_J": row["delta_J_actual_vs_operating"],
                "predicted_meaningful_improvement": float(row["delta_J_pred_one_step"])
                < -OBJECTIVE_EQUIVALENCE_TOLERANCE,
                "truth_meaningful_improvement": float(
                    row["delta_J_actual_vs_operating"]
                )
                < -OBJECTIVE_EQUIVALENCE_TOLERANCE,
                "false_improvement": bool(row["executed_false_improvement"]),
                "truth_attached_posthoc": False,
                "truth_used_for_selection": False,
            }
        )
    if not bundle.empty:
        for row in bundle.loc[bundle["selected_authorization"].astype(bool)].to_dict(
            orient="records"
        ):
            rows.append(
                {
                    "case_id": row["case_id"],
                    "semantic_id": row["semantic_id"],
                    "iteration": row["iteration"],
                    "event_type": "BUNDLE_ENDPOINT_AUTHORIZATION_POSTHOC",
                    "predicted_delta_J": row["predicted_cumulative_delta_J"],
                    "truth_delta_J": row["truth_cumulative_delta_J_posthoc"],
                    "predicted_meaningful_improvement": float(
                        row["predicted_cumulative_delta_J"]
                    )
                    < -OBJECTIVE_EQUIVALENCE_TOLERANCE,
                    "truth_meaningful_improvement": bool(
                        row["truth_bundle_meaningful_improvement"]
                    ),
                    "false_improvement": bool(
                        row["bundle_endpoint_false_improvement"]
                    ),
                    "truth_attached_posthoc": True,
                    "truth_used_for_selection": False,
                }
            )
    return pd.DataFrame(rows)


def _aggregate(
    case_summary: pd.DataFrame,
    bundle: pd.DataFrame,
    false_audit: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        case_summary.groupby("policy_id", as_index=False, sort=False)
        .agg(
            case_count=("case_id", "count"),
            trials=("number_of_executed_trials", "sum"),
            explore=("number_of_explore_trials", "sum"),
            exploit=("number_of_exploit_trials", "sum"),
            bundle_authorizations=("number_of_bundle_authorized_trials", "sum"),
            missed_improvement=("missed_improvement_rounds", "sum"),
            direct_executed_false_improvement=(
                "number_of_executed_false_improvements",
                "sum",
            ),
            conservative_stop=("premature_conservative_stops", "sum"),
            final_J=("final_best_actual_J", "mean"),
            regret=("global_truth_regret", "mean"),
        )
        .rename(columns={"policy_id": "semantic_id"})
    )
    endpoint_false = (
        false_audit.loc[
            false_audit["event_type"].eq("BUNDLE_ENDPOINT_AUTHORIZATION_POSTHOC")
        ]
        .groupby("semantic_id")["false_improvement"]
        .sum()
        .to_dict()
    )
    selected_endpoint_count = (
        bundle.loc[bundle["selected_authorization"].astype(bool)]
        .groupby("semantic_id")["authorization_id"]
        .count()
        .to_dict()
        if not bundle.empty
        else {}
    )
    output["bundle_endpoint_false_improvement"] = output["semantic_id"].map(
        endpoint_false
    ).fillna(0).astype(int)
    output["selected_bundle_endpoint_audits"] = output["semantic_id"].map(
        selected_endpoint_count
    ).fillna(0).astype(int)
    output["total_false_improvement"] = (
        output["direct_executed_false_improvement"]
        + output["bundle_endpoint_false_improvement"]
    )
    order = [item.semantic_id for item in SEMANTIC_VARIANTS]
    output = output.set_index("semantic_id").reindex(order).reset_index()
    output["policy_modified"] = False
    output["data_role"] = "DEVELOPMENT_AND_POST_REJECTION_DEVELOPMENT_SHADOW_ONLY"
    return output


def _classify(
    summary: pd.DataFrame, small: pd.DataFrame
) -> tuple[str, dict[str, Any]]:
    values = summary.set_index("semantic_id")
    s0 = values.loc[SEMANTIC_VARIANTS[0].semantic_id]
    s3 = values.loc[SEMANTIC_VARIANTS[3].semantic_id]
    recovery = small.groupby("semantic_id")["recovered_small_step_path"].sum()
    s0_recovery = int(recovery.get(SEMANTIC_VARIANTS[0].semantic_id, 0))
    s3_recovery = int(recovery.get(SEMANTIC_VARIANTS[3].semantic_id, 0))
    criteria = {
        "S3_recovers_more_of_9_paths_than_S0": s3_recovery > s0_recovery,
        "S3_has_more_bundle_authorizations_than_S0": int(
            s3["bundle_authorizations"]
        )
        > int(s0["bundle_authorizations"]),
        "S3_missed_improvement_not_higher_than_S0": int(
            s3["missed_improvement"]
        )
        <= int(s0["missed_improvement"]),
        "S3_total_false_improvement_not_higher_than_S0": int(
            s3["total_false_improvement"]
        )
        <= int(s0["total_false_improvement"]),
        "S3_mean_final_J_not_worse_than_S0_by_0.005": float(s3["final_J"])
        <= float(s0["final_J"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "S3_mean_regret_not_worse_than_S0_by_0.005": float(s3["regret"])
        <= float(s0["regret"]) + OBJECTIVE_EQUIVALENCE_TOLERANCE,
    }
    if all(criteria.values()):
        status = PRIMARY_BLOCKER
    elif not criteria["S3_recovers_more_of_9_paths_than_S0"]:
        status = NOT_SUFFICIENT
    else:
        status = MORE_EVIDENCE
    return status, {
        **criteria,
        "S0_small_step_recovery": s0_recovery,
        "S3_small_step_recovery": s3_recovery,
    }


def _fmt(value: Any) -> str:
    return f"{float(value):.9f}"


def _final_report(
    manifest_sha: str,
    summary: pd.DataFrame,
    small: pd.DataFrame,
    final_status: str,
    criteria: Mapping[str, Any],
) -> str:
    by_id = summary.set_index("semantic_id")
    lines = []
    for spec in SEMANTIC_VARIANTS:
        row = by_id.loc[spec.semantic_id]
        lines.append(
            f"- {spec.semantic_id}: trials={int(row['trials'])}, exploit={int(row['exploit'])}, "
            f"bundle authorizations={int(row['bundle_authorizations'])}, missed={int(row['missed_improvement'])}, "
            f"false total={int(row['total_false_improvement'])}, final J={_fmt(row['final_J'])}, "
            f"regret={_fmt(row['regret'])}."
        )
    recovery = small.groupby("semantic_id")["recovered_small_step_path"].sum()
    s0 = by_id.loc[SEMANTIC_VARIANTS[0].semantic_id]
    s3 = by_id.loc[SEMANTIC_VARIANTS[3].semantic_id]
    return f"""# {AUDIT_ID}

Candidate manifest SHA-256: `{manifest_sha}`

## Frozen semantics

`0.005` is the unchanged mechanical-objective equivalence/minimum-meaningful-improvement magnitude. `U` is empirical model error on predicted objective differences. The current rule adds them, but the repository contains no explicit theoretical or preregistered requirement that they must be added. Audit label: `{ADDITIVE_ASSUMPTION}`.

S1 separates a categorical calibration direction-evidence gate from the 0.005 magnitude gate. S2 uses the independent one-step P95 only to ask whether a transparent residual interval still supports a negative delta J, while independently retaining the magnitude check. S3 applies the same split to 2/3/5-step endpoints and authorizes only one next formal-grid step before refit/recompute.

## Development shadow comparison

{chr(10).join(lines)}

The nine historical small-step paths recovered by S0/S1/S2/S3 were respectively {int(recovery.get(SEMANTIC_VARIANTS[0].semantic_id, 0))}/9, {int(recovery.get(SEMANTIC_VARIANTS[1].semantic_id, 0))}/9, {int(recovery.get(SEMANTIC_VARIANTS[2].semantic_id, 0))}/9, and {int(recovery.get(SEMANTIC_VARIANTS[3].semantic_id, 0))}/9.

S3 changed bundle authorizations from {int(s0['bundle_authorizations'])} to {int(s3['bundle_authorizations'])}, missed-improvement rounds from {int(s0['missed_improvement'])} to {int(s3['missed_improvement'])}, and total false-improvement events from {int(s0['total_false_improvement'])} to {int(s3['total_false_improvement'])}. Mean final J changed by {_fmt(float(s3['final_J']) - float(s0['final_J']))}; mean regret changed by {_fmt(float(s3['regret']) - float(s0['regret']))}.

## Interpretation

The frozen classification criteria evaluated to `{dict(criteria)}`. Final audit conclusion:

`{final_status}`

This conclusion is limited to the specified synthetic DEVELOPMENT + POST_REJECTION_DEVELOPMENT shadow. It does not make S3 a policy, choose a final percentile, alter 0.005, or establish prospective, human, robot-motion, safety, or clinical readiness. The formal states remain `POLICY_DESIGN_REQUIRES_REVISION`, `{NOT_HUMAN_READY}`, and `{NOT_ROBOT_MOTION_APPROVED}`.

## Plain-language answers

A. `0.005` says an objective change must be large enough to count as meaningful; `U` says how wrong the model's predicted direction/magnitude has been on independent residual evidence.

B. They are currently added because the implemented conservative rule requires the prediction to clear both quantities in one scalar margin.

C. No repository evidence establishes that addition as a scientific necessity; it is a design assumption.

D. The S3 two-gate recovery is {int(recovery.get(SEMANTIC_VARIANTS[3].semantic_id, 0))}/9 versus S0 {int(recovery.get(SEMANTIC_VARIANTS[0].semantic_id, 0))}/9.

E. Observed total false-improvement events are S0={int(s0['total_false_improvement'])}, S3={int(s3['total_false_improvement'])}.

F. The permitted final root-cause label is `{final_status}`; no next policy is implemented.
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
    calibration = load_semantics_calibration()
    protected_before = _protected_source_hashes()
    protected_before["semantics_audit_core"] = sha256_file(CORE_SOURCE_PATH)
    protected_before["semantics_audit_runner"] = sha256_file(RUNNER_SOURCE_PATH)

    # First persisted artifact. No development state or truth is loaded above.
    manifest = candidate_manifest_payload(
        calibration,
        checkpoint_commit=str(checkpoint["checkpoint_commit"]),
        protected_source_sha256=protected_before,
    )
    manifest_path = output / "DECISION_RULE_SEMANTICS_CANDIDATE_MANIFEST_V1.json"
    _write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    gate = FrozenSemanticsManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    cases = _case_table()
    _write_text(output / "current_rule_semantics.md", _current_rule_report(calibration))
    _write_text(output / "DATA_ROLE_AUDIT.md", _data_role_report(cases))
    _write_csv(output / "calibration_direction_evidence.csv", calibration.direction_evidence)

    lattice = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometrically admissible lattice changed")
    cache = build_trajectory_component_cache(lattice)
    patient_cache: dict[tuple[float, float, float], bool] = {}
    results = []
    bundle_frames: list[pd.DataFrame] = []
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
            for spec in SEMANTIC_VARIANTS:
                result, bundle_history = run_semantic_shadow(
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
                if not bundle_history.empty:
                    bundle_frames.append(bundle_history)
            gate.record_truth_access("POST_POLICY_FULL_TRUTH_LANDSCAPE")
            landscape = evaluate_full_truth_landscape(case_results[0], state, cache)
            truth_landscapes[case_id] = landscape
            for result in case_results:
                gate.record_truth_access("POST_POLICY_LOCAL_TRUTH_AUDIT")
                _candidates, rounds = post_policy_local_truth_audit(result, state, cache)
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
    bundle_raw = (
        pd.concat(bundle_frames, ignore_index=True, sort=False)
        if bundle_frames
        else pd.DataFrame()
    )
    bundle = attach_bundle_posthoc_truth(bundle_raw, truth_landscapes)
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
    false_audit = _false_audit(history, bundle)
    summary = _aggregate(case_summary, bundle, false_audit)
    small = small_step_semantic_recovery(calibration)
    final_status, criteria = _classify(summary, small)

    _write_csv(output / "semantic_shadow_comparison.csv", summary)
    _write_csv(output / "small_step_semantic_recovery.csv", small)
    _write_csv(output / "false_improvement_semantic_audit.csv", false_audit)
    _write_csv(output / "semantic_shadow_case_summary.csv", case_summary)
    _write_csv(output / "semantic_shadow_trial_history.csv", history)
    _write_csv(output / "semantic_bundle_authorization_history.csv", bundle)
    _write_csv(output / "semantic_shadow_truth_rounds.csv", truth_rounds)
    _write_text(
        output / "DECISION_RULE_SEMANTICS_REPORT.md",
        _final_report(manifest_sha, summary, small, final_status, criteria),
    )

    protected_after = _protected_source_hashes()
    protected_after["semantics_audit_core"] = sha256_file(CORE_SOURCE_PATH)
    protected_after["semantics_audit_runner"] = sha256_file(RUNNER_SOURCE_PATH)
    if protected_before != protected_after:
        raise RuntimeError("protected source changed during semantics shadow")
    if sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("semantic manifest changed after shadow truth")
    if not history["queued_later_bundle_steps"].eq(False).all():
        raise RuntimeError("bundle queued more than one execution")
    if not history["model_refit_after_execution"].astype(bool).all():
        raise RuntimeError("model was not refit after every execution")
    if not history["full_map_recomputed_after_execution"].astype(bool).all():
        raise RuntimeError("prediction map was not recomputed after every execution")

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    metadata = {
        "audit_id": AUDIT_ID,
        "manifest_id": MANIFEST_ID,
        "candidate_manifest_sha256": manifest_sha,
        "current_additive_semantics_classification": ADDITIVE_ASSUMPTION,
        "final_status": final_status,
        "classification_criteria": criteria,
        "checkpoint": checkpoint,
        "development_case_count": len(cases),
        "original_development_case_count": int(
            cases["development_origin"].eq("ORIGINAL_P2_DEVELOPMENT").sum()
        ),
        "post_rejection_development_case_count": int(
            cases["development_origin"].eq("POST_REJECTION_DEVELOPMENT").sum()
        ),
        "independent_calibration_case_count": 12,
        "calibration_cases_used_for_policy_performance": False,
        "calibration_used_for_residual_and_direction_evidence_only": True,
        "candidate_manifest_frozen_before_truth": True,
        "manifest_truth_gate_access_count": gate.truth_access_count,
        "manifest_truth_gate_stages": gate.truth_access_stages,
        "development_truth_modified_semantics": False,
        "percentile_tuned_after_results": False,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA256,
        "bundle_pair_plan_sha256": EXPECTED_BUNDLE_PAIR_PLAN_SHA256,
        "bundle_authorizes_next_one_step_only": True,
        "prospective_cohort_run": False,
        "held_out_final_test_read": False,
        "P2_V1_replaced": False,
        "new_policy_implemented": False,
        "new_policy_default_enabled": False,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "immutable_prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "immutable_prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "human_ready": NOT_HUMAN_READY,
        "robot_motion_approved": NOT_ROBOT_MOTION_APPROVED,
        "robot_connected": False,
        "artifact_manifest": artifacts,
        "runtime_seconds": time.perf_counter() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    missing = [name for name in REQUIRED_OUTPUT_FILENAMES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing semantics-audit outputs: {missing}")
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

