"""Run MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_V1 offline.

The runner freezes every baseline identity and seed on disk before opening the
existing virtual-truth layer.  Frozen Top-3 and the K=2 ablation call the
existing MODEL_SCREENED_FINITE_SEQUENTIAL_VALIDATION_V1 evaluator directly.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import run_final_model_screened_finite_sequential_validation as frozen_v1
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    build_trajectory_component_cache,
    geometrically_valid_parameter_lattice,
)
from .final_model_screened_finite_sequential_validation import (
    METHOD_ID as FROZEN_V1_METHOD_ID,
    FrozenShortlist,
    freeze_model_screened_shortlist,
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
from .mismatch_scenarios import MISMATCH_SCENARIOS, get_mismatch_scenario
from .model_trust_finite_validation_stress_test import (
    BASELINE_MANIFEST_ID,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    DEFAULT_ENABLED,
    MACHINE_COMPARISON_TOLERANCE,
    NEAR_OPTIMAL_TOLERANCES,
    NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED,
    N_RANDOM_REPEATS,
    OFFLINE_ONLY,
    PROTOCOL_ID,
    PersistedTruthGate,
    RANDOM_BASE_SEED,
    STAGE_ID,
    bootstrap_mean_ci,
    canonical_json_bytes,
    empirical_p95,
    final_regret,
    freeze_random3_candidates,
    geometry_candidate_universe,
    near_optimal_rates,
    select_model_only,
    select_validated_with_reference,
    truth_rank_percentile,
)
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_v2_prospective_offline_validation import (
    dynamic_subject_for_id,
    registered_prospective_subject,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "model_trust_finite_validation_stress_test.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_model_trust_finite_validation_stress_test.py"
TEST_SOURCE_PATH = MODULE_DIR / "test_model_trust_finite_validation_stress_test.py"
FROZEN_V1_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "final_model_screened_finite_sequential_validation_v1"
)
FROZEN_V1_MANIFEST_PATH = FROZEN_V1_DIRECTORY / "FINAL_METHOD_MANIFEST_V1.json"
FROZEN_V1_METADATA_PATH = FROZEN_V1_DIRECTORY / "metadata.json"
FROZEN_V1_MANIFEST_SHA256 = (
    "7576e5a545878292f2eb1846e9cae780325a2e44bb58093dfb04bae982827498"
)
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "model_trust_finite_validation_stress_test_v1"
)
FIGURE_FILENAMES = (
    "figures/MODEL_MISMATCH_VS_FINAL_REGRET.png",
    "figures/TRIAL_BUDGET_VS_FINAL_REGRET.png",
    "figures/MODEL_SCREENING_VS_RANDOM3.png",
    "figures/TOP1_VS_TOP3_PAIRED_REGRET.png",
    "figures/MODEL_PREDICTION_RANKING_UTILITY.png",
)
REQUIRED_OUTPUTS = (
    "STRESS_TEST_PROTOCOL.json",
    "FROZEN_BASELINE_MANIFEST.json",
    "PER_CASE_RESULTS.csv",
    "RANDOM3_RESULTS.csv",
    "METHOD_SUMMARY.csv",
    "SCENARIO_SUMMARY.csv",
    "TRIAL_BUDGET_RESULTS.csv",
    "TOP1_TOP3_PAIRED.csv",
    "MODEL_TOP3_VS_RANDOM3.csv",
    "RANKING_UTILITY.csv",
    "FALSE_IMPROVEMENT_SUMMARY.csv",
    "SUBJECT_SPECIFICITY.csv",
    "TRUTH_ACCESS_AUDIT.csv",
    "CODE_AND_MISMATCH_AUDIT.md",
    "MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_REPORT.md",
    *FIGURE_FILENAMES,
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any], *, canonical: bool = False) -> None:
    safe = _json_safe(dict(payload))
    data = (
        canonical_json_bytes(safe)
        if canonical
        else (
            json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    _atomic_bytes(path, data)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    _atomic_bytes(
        path,
        table.to_csv(
            index=False, lineterminator="\n", float_format="%.12g"
        ).encode("utf-8"),
    )


def _write_text(path: Path, content: str) -> None:
    _atomic_bytes(path, content.encode("utf-8"))


def _directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "stress_test_core": CORE_SOURCE_PATH,
        "stress_test_runner": RUNNER_SOURCE_PATH,
        "frozen_v1_core": frozen_v1.CORE_SOURCE_PATH,
        "frozen_v1_runner": frozen_v1.RUNNER_SOURCE_PATH,
        "mismatch_scenarios": MODULE_DIR / "mismatch_scenarios.py",
        "mechanical_objective": MODULE_DIR / "mechanical_objective.py",
        "candidate_generator": MODULE_DIR / "continuous_reference_neighborhood.py",
        "model_support": MODULE_DIR / "decision_relevant_global_model_reliability.py",
        "five_parameter_estimator": MODULE_DIR / "parameter_estimator.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _preflight() -> dict[str, Any]:
    if sha256_file(FROZEN_V1_MANIFEST_PATH) != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError("frozen V1 manifest changed")
    metadata = json.loads(FROZEN_V1_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["method_id"] != FROZEN_V1_METHOD_ID:
        raise RuntimeError("frozen V1 method identity changed")
    if metadata["manifest_sha256"] != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError("frozen V1 metadata/manifest mismatch")
    for relative, record in metadata["artifact_manifest"].items():
        if sha256_file(FROZEN_V1_DIRECTORY / relative) != record["sha256"]:
            raise RuntimeError(f"frozen V1 artifact changed: {relative}")
    validate_active_reference_file(ACTIVE_REFERENCE_PATH)
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference SHA changed")
    cases = frozen_v1._case_table()
    observed = set(cases["scenario_name"].astype(str))
    if observed != set(MISMATCH_SCENARIOS):
        raise RuntimeError(
            "stress-test case plan does not include every existing mismatch scenario"
        )
    return {
        "frozen_v1_method_id": FROZEN_V1_METHOD_ID,
        "frozen_v1_manifest_sha256": FROZEN_V1_MANIFEST_SHA256,
        "frozen_v1_artifact_count": len(metadata["artifact_manifest"]),
        "all_existing_mismatch_scenarios_present": True,
        "case_count": len(cases),
        "scenario_count": len(observed),
    }


def _scenario_role(scenario_name: str) -> dict[str, Any]:
    scenario = get_mismatch_scenario(scenario_name)
    name = str(scenario_name)
    if name == "matched_linear":
        family = "matched_linear"
        level = "MATCHED"
    elif name.startswith("nonlinear_stiffness_"):
        family = "nonlinear_stiffness"
        level = name.rsplit("_", 1)[1].upper()
    elif name.startswith("hip_knee_coupling_"):
        family = "hip_knee_coupling"
        level = name.rsplit("_", 1)[1].upper()
    elif name.startswith("nonlinear_damping_"):
        family = "nonlinear_damping"
        level = name.rsplit("_", 1)[1].upper()
    elif name == "structured_residual":
        family = "structured_residual"
        level = "SINGLE_DEFINED_LEVEL"
    elif name.startswith("combined_"):
        family = "combined"
        level = name.rsplit("_", 1)[1].upper()
    else:
        raise RuntimeError(f"unclassified frozen mismatch scenario: {name}")
    return {
        "mismatch_family": family,
        "mismatch_level": level,
        "model_mismatch_terms": ";".join(scenario.model_mismatch_terms),
        "truth_generator_parameters_json": json.dumps(
            dict(scenario.generator_parameters),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "truth_generator_seed": int(scenario.random_seed),
    }


def _case_plan() -> pd.DataFrame:
    cases = frozen_v1._case_table().copy()
    roles = pd.DataFrame(
        [_scenario_role(value) for value in cases["scenario_name"].astype(str)]
    )
    output = pd.concat((cases.reset_index(drop=True), roles), axis=1)
    output["global_scalar_severity_order_used"] = False
    output["case_selection_used_candidate_truth"] = False
    return output


def _identity_lattice(parameter_map: pd.DataFrame) -> pd.DataFrame:
    lattice = geometrically_valid_parameter_lattice(parameter_map)
    hashes = parameter_map.loc[
        :, ["trajectory_id", "trajectory_sha256"]
    ].drop_duplicates()
    identity = lattice.merge(hashes, on="trajectory_id", validate="one_to_one")
    if len(identity) != 21025:
        raise RuntimeError("formal 21,025-point geometry lattice changed")
    return identity


def _prepare_all(cases: pd.DataFrame, lattice: pd.DataFrame, cache: Any) -> list[Any]:
    preparation_roles = cases.loc[
        :, ["case_id", "subject_id", "scenario_name", "case_class", "development_origin"]
    ]
    return frozen_v1._prepare_cases(preparation_roles, lattice, cache)


def _shortlist_identity_rows(
    prepared: Sequence[Any], shortlists_k2: Mapping[str, FrozenShortlist]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for item in prepared:
        case_id = str(item.role["case_id"])
        for baseline_id, shortlist in (
            ("B1_MODEL_ONLY", item.shortlist),
            ("B3_MODEL_TOP1_VALIDATION", item.shortlist),
            ("B4_FROZEN_TOP3_SEQUENTIAL", item.shortlist),
            ("TRIAL_BUDGET_K2", shortlists_k2[case_id]),
        ):
            limit = 1 if baseline_id in {
                "B1_MODEL_ONLY", "B3_MODEL_TOP1_VALIDATION"
            } else len(shortlist.candidates)
            for candidate in shortlist.candidates[:limit]:
                logical_order += 1
                rows.append(
                    {
                        "case_id": case_id,
                        "baseline_id": baseline_id,
                        "random_repeat": None,
                        "seed": None,
                        "candidate_ordinal": candidate.shortlist_ordinal,
                        "candidate_id": f"C{candidate.shortlist_ordinal}",
                        "trajectory_id": candidate.trajectory_id,
                        "trajectory_sha256": candidate.trajectory_sha256,
                        "hip_delta": candidate.hip_delta,
                        "knee_delta": candidate.knee_delta,
                        "phase_delta": candidate.phase_delta,
                        "prediction_rank": candidate.initial_prediction_rank,
                        "freeze_logical_order_global": logical_order,
                        "freeze_timestamp": f"PRETRUTH_LOGICAL_T{logical_order:06d}",
                        "truth_read_before_freeze": False,
                    }
                )
    return rows


def _protocol_payload(
    *,
    preflight: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    cases: pd.DataFrame,
    candidate_universe_size: int,
    v1_artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    scenarios = [
        {
            **get_mismatch_scenario(name).as_metadata_dict(),
            **_scenario_role(name),
        }
        for name in MISMATCH_SCENARIOS
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "default_enabled": DEFAULT_ENABLED,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "preflight": dict(preflight),
        "frozen_v1_manifest_sha256": FROZEN_V1_MANIFEST_SHA256,
        "source_sha256": dict(source_hashes),
        "frozen_v1_artifact_sha256": dict(v1_artifact_hashes),
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "formal_hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "formal_knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "candidate_lattice_size": 21025,
        "shared_nonzero_geometry_candidate_universe_size": candidate_universe_size,
        "case_plan": cases.to_dict(orient="records"),
        "mismatch_scenarios": scenarios,
        "single_global_mismatch_severity_scalar_defined": False,
        "severity_ordering_scope": "WITHIN_EXPLICIT_MILD_STRONG_FAMILY_ONLY",
        "random_repeat_count": N_RANDOM_REPEATS,
        "random_base_seed": RANDOM_BASE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "near_optimal_tolerances": list(NEAR_OPTIMAL_TOLERANCES),
        "primary_metric": "final_regret=J_final_truth-J_oracle_truth",
        "baselines": {
            "B0_REFERENCE": "J_final=1;K=0",
            "B1_MODEL_ONLY": "frozen model C1;no reference fallback;K=0",
            "B2_RANDOM3_FINITE_VALIDATION": (
                "three uniform candidates from shared nonzero geometry universe;"
                "Reference fallback;K=3"
            ),
            "B3_MODEL_TOP1_VALIDATION": "frozen model C1;Reference fallback;K=1",
            "B4_FROZEN_TOP3_SEQUENTIAL": (
                "direct call to frozen V1 evaluator;Reference fallback;K<=3"
            ),
            "B5_ORACLE": "post-freeze full 21,025 truth lower bound only",
        },
        "trial_budget_ablation": {
            "K0": "B1 model-only",
            "K1": "B3 Top1 with Reference fallback",
            "K2": "same frozen V1 construction with max_candidates=2",
            "K3": "formal frozen V1 Top3",
            "K5": "SKIP_IF_FROZEN_V1_CANNOT_NATURALLY_EXTEND_WITHOUT_REDESIGN",
            "K5_pretruth_decision": "SKIPPED_REQUIRES_FROZEN_SELECTION_RULE_REDESIGN",
        },
        "conclusion_rules_frozen_before_truth": {
            "MODEL_SCREENING_SUPPORTED": (
                "Top3 mean regret < Random3 mean regret AND mean per-case strict-"
                "beat percentile >50% AND mean truth-top5 enrichment >1"
            ),
            "TOP1_SUFFICIENT": "all paired Top1/Top3 regrets tie within 1e-12",
            "TOP3_MISMATCH_RESILIENCE_SUPPORTED": (
                "mean paired benefit >0 AND bootstrap95 lower>0 AND no Top3 loss"
            ),
            "FINITE_VALIDATION_BUDGET_SATURATES_AT_K_X": (
                "smallest K whose all later through K3 per-case regrets are equal"
            ),
            "MODEL_TRUST_LIMIT_IDENTIFIED": (
                "at least two cases in one predefined family have Top3 no better"
                "than Random3 median and truth-top5 enrichment<=1"
            ),
        },
        "no_posthoc_algorithm_changes": True,
        "candidate_truth_may_open_only_after_both_manifests_persist": True,
        "bayesian_optimization_implemented": False,
        "hardware_robot_safety_scope": "UNTOUCHED",
    }


def _baseline_manifest_payload(
    *,
    protocol_sha256: str,
    cases: pd.DataFrame,
    identity_rows: list[dict[str, Any]],
    random_rows: pd.DataFrame,
    shortlists_k2: Mapping[str, FrozenShortlist],
) -> dict[str, Any]:
    all_rows = [*identity_rows, *random_rows.to_dict(orient="records")]
    for order, row in enumerate(all_rows, start=1):
        row.setdefault("freeze_logical_order_global", order)
    return {
        "manifest_id": BASELINE_MANIFEST_ID,
        "stage_id": STAGE_ID,
        "protocol_sha256": protocol_sha256,
        "manifest_role": "ALL_BASELINE_IDENTITIES_AND_SEEDS_FROZEN_PRETRUTH",
        "truth_read_before_manifest_persist": False,
        "case_count": len(cases),
        "random_repeat_count_per_case": N_RANDOM_REPEATS,
        "random_candidate_count_per_repeat": 3,
        "K5_skipped_before_truth": True,
        "K5_skip_reason": "FROZEN_V1_MAXIMUM_IS_3_AND_EXTENSION_WOULD_REDESIGN_SELECTION_RULE",
        "k2_shortlists": {
            case_id: shortlist.as_manifest()
            for case_id, shortlist in sorted(shortlists_k2.items())
        },
        "candidate_freeze_records": all_rows,
    }


def _registered_context(role: Mapping[str, Any]):
    if str(role["development_origin"]) == "POST_REJECTION_DEVELOPMENT":
        subject = dynamic_subject_for_id(str(role["subject_id"]))
        return registered_prospective_subject(subject)
    return nullcontext()


def _prediction_eligible(table: pd.DataFrame) -> pd.DataFrame:
    neutral = (
        np.isclose(table["hip_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(table["knee_delta"].to_numpy(dtype=float), 0.0)
        & np.isclose(table["phase_delta"].to_numpy(dtype=float), 0.0)
    )
    return table.loc[
        ~neutral
        & table["geometrically_admissible"].astype(bool).to_numpy()
        & table["model_supported"].astype(bool).to_numpy()
        & table["J_pred"].lt(1.0).to_numpy()
    ].copy()


def _alpha_json(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [float(row["hip_delta"]), float(row["knee_delta"]), float(row["phase_delta"])],
        separators=(",", ":"),
    )


def _method_row(
    *,
    role: Mapping[str, Any],
    method: str,
    budget: int | None,
    candidate_ids: Sequence[str],
    selected: Mapping[str, Any],
    oracle: Mapping[str, Any],
    initial_theta_hat_json: str,
    predicted_j: float | None = None,
    final_harmful_selection: bool = False,
    reference_fallback_available: bool = False,
) -> dict[str, Any]:
    final_j = float(selected["J_truth"])
    return {
        **dict(role),
        "method": method,
        "validation_budget": budget,
        "initial_identified_theta_hat": initial_theta_hat_json,
        "candidate_ids_json": json.dumps(list(candidate_ids), separators=(",", ":")),
        "predicted_J_selected": predicted_j,
        "final_selected_trajectory": str(selected["trajectory_id"]),
        "final_alpha_json": _alpha_json(selected),
        "J_final_truth": final_j,
        "oracle_trajectory_id": str(oracle["trajectory_id"]),
        "oracle_J": float(oracle["J_truth"]),
        "final_regret": final_regret(final_j, float(oracle["J_truth"])),
        "final_harmful_selection": bool(final_harmful_selection),
        "reference_fallback_available": bool(reference_fallback_available),
        "candidate_truth_used_for_candidate_generation_or_ordering": False,
    }


def _evaluate_random(
    frozen_random: pd.DataFrame,
    truth_map: pd.DataFrame,
    role: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> pd.DataFrame:
    lookup = truth_map.set_index("trajectory_id")
    rows: list[dict[str, Any]] = []
    for repeat, group in frozen_random.groupby("random_repeat", sort=True):
        candidates = group.merge(
            truth_map[
                [
                    "trajectory_id",
                    "J_truth",
                    "hip_delta",
                    "knee_delta",
                    "phase_delta",
                ]
            ],
            on=["trajectory_id", "hip_delta", "knee_delta", "phase_delta"],
            validate="one_to_one",
        )
        selected = select_validated_with_reference(candidates)
        candidate_truth = [
            float(lookup.loc[value, "J_truth"])
            for value in group["trajectory_id"].astype(str)
        ]
        rows.append(
            {
                **dict(role),
                "method": "B2_RANDOM3_FINITE_VALIDATION",
                "random_repeat": int(repeat),
                "seed": int(group["seed"].iloc[0]),
                "candidate_ids_json": json.dumps(
                    group["trajectory_id"].astype(str).tolist(), separators=(",", ":")
                ),
                "candidate_truth_J_json": json.dumps(candidate_truth, separators=(",", ":")),
                "final_selected_trajectory": str(selected["trajectory_id"]),
                "final_alpha_json": _alpha_json(selected),
                "J_final_truth": float(selected["J_truth"]),
                "oracle_J": float(oracle["J_truth"]),
                "final_regret": final_regret(
                    float(selected["J_truth"]), float(oracle["J_truth"])
                ),
                "harmful_candidate_exposure_count": int(
                    np.sum(np.asarray(candidate_truth) >= 1.0)
                ),
                "final_harmful_selection": bool(float(selected["J_truth"]) > 1.0),
                "reference_fallback_available": True,
                "sampling_uses_J_pred": False,
                "sampling_uses_J_truth": False,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_all(
    *,
    prepared: Sequence[Any],
    shortlists_k2: Mapping[str, FrozenShortlist],
    frozen_random: pd.DataFrame,
    cache: Any,
    baseline_manifest_sha256: str,
) -> dict[str, pd.DataFrame]:
    per_case_rows: list[dict[str, Any]] = []
    random_frames: list[pd.DataFrame] = []
    budget_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    truth_access_rows: list[dict[str, Any]] = []
    global_truth_order = 0
    for item in prepared:
        role = {**item.role, **_scenario_role(str(item.role["scenario_name"]))}
        case_id = str(role["case_id"])
        with _registered_context(role):
            top3 = frozen_v1._evaluate_case(
                item,
                cache=cache,
                global_manifest_sha256=baseline_manifest_sha256,
            )
        k2_prepared = frozen_v1.PreparedCase(
            role=dict(item.role),
            state=item.state,
            initial_model=item.initial_model,
            initial_prediction_map=item.initial_prediction_map.copy(deep=True),
            shortlist=shortlists_k2[case_id],
        )
        with _registered_context(role):
            k2 = frozen_v1._evaluate_case(
                k2_prepared,
                cache=cache,
                global_manifest_sha256=baseline_manifest_sha256,
            )
        truth = top3.truth_map.copy(deep=True)
        if not np.allclose(
            truth["J_truth"].to_numpy(dtype=float),
            k2.truth_map["J_truth"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("K2 and frozen V1 do not share the same truth landscape")
        oracle = truth.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort"
        ).iloc[0].to_dict()
        c1 = item.shortlist.candidates[0]
        c1_truth = truth.loc[truth["trajectory_id"].eq(c1.trajectory_id)].iloc[0].to_dict()
        c1_truth["trajectory_id"] = c1.trajectory_id
        model_only = select_model_only(c1_truth)
        top1 = select_validated_with_reference(pd.DataFrame([c1_truth]))
        top3_selected = truth.loc[
            truth["trajectory_id"].eq(top3.summary["best_validated_trajectory_id"])
        ]
        if top3_selected.empty:
            top3_selected_dict = {
                "trajectory_id": "REFERENCE",
                "J_truth": 1.0,
                "hip_delta": 0.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
            }
        else:
            top3_selected_dict = top3_selected.iloc[0].to_dict()
        k2_selected = truth.loc[
            truth["trajectory_id"].eq(k2.summary["best_validated_trajectory_id"])
        ]
        if k2_selected.empty:
            k2_selected_dict = {
                "trajectory_id": "REFERENCE",
                "J_truth": 1.0,
                "hip_delta": 0.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
            }
        else:
            k2_selected_dict = k2_selected.iloc[0].to_dict()
        theta_json = json.dumps(
            {key: float(value) for key, value in sorted(item.state.parameters.items())},
            separators=(",", ":"),
        )
        reference = {
            "trajectory_id": "REFERENCE",
            "J_truth": 1.0,
            "hip_delta": 0.0,
            "knee_delta": 0.0,
            "phase_delta": 0.0,
        }
        exact_rows = [
            _method_row(
                role=role,
                method="B0_REFERENCE",
                budget=0,
                candidate_ids=[],
                selected=reference,
                oracle=oracle,
                initial_theta_hat_json=theta_json,
                reference_fallback_available=True,
            ),
            _method_row(
                role=role,
                method="B1_MODEL_ONLY",
                budget=0,
                candidate_ids=[c1.trajectory_id],
                selected=model_only,
                oracle=oracle,
                initial_theta_hat_json=theta_json,
                predicted_j=float(c1.initial_J_pred),
                final_harmful_selection=float(model_only["J_truth"]) > 1.0,
                reference_fallback_available=False,
            ),
            _method_row(
                role=role,
                method="B3_MODEL_TOP1_VALIDATION",
                budget=1,
                candidate_ids=[c1.trajectory_id],
                selected=top1,
                oracle=oracle,
                initial_theta_hat_json=theta_json,
                predicted_j=float(c1.initial_J_pred),
                reference_fallback_available=True,
            ),
            _method_row(
                role=role,
                method="B4_FROZEN_TOP3_SEQUENTIAL",
                budget=int(top3.summary["executed_validation_count"]),
                candidate_ids=item.shortlist.trajectory_ids,
                selected=top3_selected_dict,
                oracle=oracle,
                initial_theta_hat_json=theta_json,
                reference_fallback_available=True,
            ),
            _method_row(
                role=role,
                method="B5_ORACLE",
                budget=None,
                candidate_ids=[],
                selected=oracle,
                oracle=oracle,
                initial_theta_hat_json=theta_json,
                reference_fallback_available=False,
            ),
        ]
        per_case_rows.extend(exact_rows)
        random_case = _evaluate_random(
            frozen_random.loc[frozen_random["case_id"].eq(case_id)],
            truth,
            role,
            oracle,
        )
        random_frames.append(random_case)
        random_summary = {
            **dict(role),
            "method": "B2_RANDOM3_FINITE_VALIDATION",
            "validation_budget": 3,
            "initial_identified_theta_hat": theta_json,
            "candidate_ids_json": "RANDOM_DISTRIBUTION_SEE_RANDOM3_RESULTS",
            "predicted_J_selected": None,
            "final_selected_trajectory": "RANDOM_DISTRIBUTION",
            "final_alpha_json": "RANDOM_DISTRIBUTION",
            "J_final_truth": float(random_case["J_final_truth"].mean()),
            "oracle_trajectory_id": str(oracle["trajectory_id"]),
            "oracle_J": float(oracle["J_truth"]),
            "final_regret": float(random_case["final_regret"].mean()),
            "final_harmful_selection": bool(random_case["final_harmful_selection"].any()),
            "reference_fallback_available": True,
            "candidate_truth_used_for_candidate_generation_or_ordering": False,
            "distribution_repeat_count": len(random_case),
            "distribution_median_regret": float(random_case["final_regret"].median()),
            "distribution_p5_regret": float(random_case["final_regret"].quantile(0.05)),
            "distribution_p95_regret": empirical_p95(random_case["final_regret"]),
        }
        per_case_rows.append(random_summary)

        method_lookup = {row["method"]: row for row in exact_rows}
        for budget, selected, source in (
            (0, model_only, "B1_MODEL_ONLY_NO_REFERENCE_FALLBACK"),
            (1, top1, "B3_MODEL_TOP1_WITH_REFERENCE_FALLBACK"),
            (2, k2_selected_dict, "FROZEN_V1_CONSTRUCTION_MAX_CANDIDATES_2"),
            (3, top3_selected_dict, "B4_DIRECT_FROZEN_V1"),
        ):
            budget_rows.append(
                {
                    **dict(role),
                    "validation_budget_K": budget,
                    "status": "EVALUATED",
                    "construction": source,
                    "final_selected_trajectory": str(selected["trajectory_id"]),
                    "J_final_truth": float(selected["J_truth"]),
                    "oracle_J": float(oracle["J_truth"]),
                    "final_regret": final_regret(
                        float(selected["J_truth"]), float(oracle["J_truth"])
                    ),
                    "reference_fallback_available": budget > 0,
                }
            )
        budget_rows.append(
            {
                **dict(role),
                "validation_budget_K": 5,
                "status": "SKIPPED_REQUIRES_FROZEN_SELECTION_RULE_REDESIGN",
                "construction": "NOT_RUN_PRETRUTH_DECISION",
                "final_selected_trajectory": "",
                "J_final_truth": np.nan,
                "oracle_J": float(oracle["J_truth"]),
                "final_regret": np.nan,
                "reference_fallback_available": True,
            }
        )

        eligible = _prediction_eligible(item.initial_prediction_map).merge(
            truth[["trajectory_id", "J_truth"]], on="trajectory_id", validate="one_to_one"
        )
        false_mask = eligible["J_pred"].lt(1.0) & eligible["J_truth"].ge(1.0)
        false_rows.append(
            {
                **dict(role),
                "predicted_improvement_candidate_count": len(eligible),
                "predicted_false_improvement_count": int(false_mask.sum()),
                "predicted_false_improvement_rate": float(false_mask.mean()),
                "model_only_final_harmful_selection": bool(float(model_only["J_truth"]) > 1.0),
                "top1_final_harmful_selection": bool(float(top1["J_truth"]) > 1.0),
                "top3_final_harmful_selection": bool(float(top3_selected_dict["J_truth"]) > 1.0),
                "reference_fallback_masks_model_error": bool(
                    false_mask.any()
                    and float(top3_selected_dict["J_truth"]) <= 1.0
                ),
            }
        )
        truth_nonzero = truth.loc[
            ~(
                np.isclose(truth["hip_delta"], 0.0)
                & np.isclose(truth["knee_delta"], 0.0)
                & np.isclose(truth["phase_delta"], 0.0)
            )
        ].copy()
        ranked = truth_nonzero.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort"
        ).reset_index(drop=True)
        for candidate in item.shortlist.candidates:
            rank, percentile = truth_rank_percentile(truth_nonzero, candidate.trajectory_id)
            ranking_rows.append(
                {
                    **dict(role),
                    "record_type": "FROZEN_CANDIDATE_TRUTH_RANK",
                    "candidate_id": f"C{candidate.shortlist_ordinal}",
                    "trajectory_id": candidate.trajectory_id,
                    "prediction_rank": candidate.initial_prediction_rank,
                    "J_pred": candidate.initial_J_pred,
                    "J_truth": float(
                        truth.loc[
                            truth["trajectory_id"].eq(candidate.trajectory_id), "J_truth"
                        ].iloc[0]
                    ),
                    "truth_rank": rank,
                    "truth_rank_percentile": percentile,
                    "truth_top_1_percent": percentile <= 1.0,
                    "truth_top_5_percent": percentile <= 5.0,
                    "truth_top_10_percent": percentile <= 10.0,
                    "top_fraction": np.nan,
                    "screened_hit_rate": np.nan,
                    "uniform_random_base_rate": np.nan,
                    "enrichment_ratio": np.nan,
                    "spearman_rank_correlation": float(
                        eligible["J_pred"].corr(eligible["J_truth"], method="spearman")
                    ),
                    "kendall_rank_correlation": float(
                        eligible["J_pred"].corr(eligible["J_truth"], method="kendall")
                    ),
                }
            )
        shortlist_ids = set(item.shortlist.trajectory_ids)
        shortlist_truth = ranked.loc[ranked["trajectory_id"].isin(shortlist_ids)]
        for fraction in (0.01, 0.05, 0.10):
            truth_top_count = max(1, int(np.ceil(len(ranked) * fraction)))
            truth_top_ids = set(ranked.head(truth_top_count)["trajectory_id"].astype(str))
            hit_rate = len(shortlist_ids.intersection(truth_top_ids)) / len(shortlist_ids)
            random_rate = truth_top_count / len(ranked)
            ranking_rows.append(
                {
                    **dict(role),
                    "record_type": "TOP_FRACTION_ENRICHMENT",
                    "candidate_id": "C1_C2_C3",
                    "trajectory_id": "",
                    "prediction_rank": np.nan,
                    "J_pred": np.nan,
                    "J_truth": np.nan,
                    "truth_rank": np.nan,
                    "truth_rank_percentile": np.nan,
                    "truth_top_1_percent": np.nan,
                    "truth_top_5_percent": np.nan,
                    "truth_top_10_percent": np.nan,
                    "top_fraction": fraction,
                    "screened_hit_rate": hit_rate,
                    "uniform_random_base_rate": random_rate,
                    "enrichment_ratio": hit_rate / random_rate,
                    "spearman_rank_correlation": float(
                        eligible["J_pred"].corr(eligible["J_truth"], method="spearman")
                    ),
                    "kendall_rank_correlation": float(
                        eligible["J_pred"].corr(eligible["J_truth"], method="kendall")
                    ),
                }
            )

        global_truth_order += 1
        truth_access_rows.append(
            {
                "case_id": case_id,
                "baseline_or_role": "B4_REFERENCE",
                "trajectory_id": "REFERENCE",
                "truth_access_order_within_case": 1,
                "truth_access_order_global_case_block": global_truth_order,
                "candidate_frozen_before_access": True,
            }
        )
        for index, row in enumerate(
            top3.execution_history.sort_values("round").to_dict(orient="records"),
            start=2,
        ):
            truth_access_rows.append(
                {
                    "case_id": case_id,
                    "baseline_or_role": "B4_FROZEN_TOP3_SEQUENTIAL",
                    "trajectory_id": str(row["trajectory_id"]),
                    "truth_access_order_within_case": index,
                    "truth_access_order_global_case_block": global_truth_order,
                    "candidate_frozen_before_access": True,
                }
            )
        truth_access_rows.append(
            {
                "case_id": case_id,
                "baseline_or_role": "B5_FULL_LANDSCAPE_POST_B4",
                "trajectory_id": "ALL_21025",
                "truth_access_order_within_case": 2 + len(top3.execution_history),
                "truth_access_order_global_case_block": global_truth_order,
                "candidate_frozen_before_access": True,
            }
        )
    return {
        "per_case": pd.DataFrame(per_case_rows),
        "random": pd.concat(random_frames, ignore_index=True),
        "budget": pd.DataFrame(budget_rows),
        "ranking": pd.DataFrame(ranking_rows),
        "false": pd.DataFrame(false_rows),
        "truth_access": pd.DataFrame(truth_access_rows),
    }


def _method_summary(per_case: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [
        "B0_REFERENCE",
        "B1_MODEL_ONLY",
        "B2_RANDOM3_FINITE_VALIDATION",
        "B3_MODEL_TOP1_VALIDATION",
        "B4_FROZEN_TOP3_SEQUENTIAL",
        "B5_ORACLE",
    ]
    for method in methods:
        if method == "B2_RANDOM3_FINITE_VALIDATION":
            table = random
            budgets: Any = 3
        else:
            table = per_case.loc[per_case["method"].eq(method)]
            budgets = table["validation_budget"].dropna().unique()
            budgets = int(budgets[0]) if len(budgets) == 1 else "N/A"
        regret = table["final_regret"].to_numpy(dtype=float)
        final_j = table["J_final_truth"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "validation_budget": budgets,
                "case_count": int(table["case_id"].nunique()),
                "evaluation_row_count": len(table),
                "mean_J": float(np.mean(final_j)),
                "mean_regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "P95_regret": empirical_p95(regret),
                "max_regret": float(np.max(regret)),
                **near_optimal_rates(regret),
                "final_harmful_selection_count": int(
                    table["final_harmful_selection"].astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _scenario_summary(per_case: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact = per_case.loc[~per_case["method"].eq("B2_RANDOM3_FINITE_VALIDATION")]
    for (scenario, family, level, method), group in exact.groupby(
        ["scenario_name", "mismatch_family", "mismatch_level", "method"], sort=False
    ):
        rows.append(
            {
                "scenario_name": scenario,
                "mismatch_family": family,
                "mismatch_level": level,
                "method": method,
                "case_count": group["case_id"].nunique(),
                "evaluation_row_count": len(group),
                "mean_final_J": float(group["J_final_truth"].mean()),
                "mean_regret": float(group["final_regret"].mean()),
                "median_regret": float(group["final_regret"].median()),
                "P95_regret": empirical_p95(group["final_regret"]),
                "max_regret": float(group["final_regret"].max()),
            }
        )
    for (scenario, family, level), group in random.groupby(
        ["scenario_name", "mismatch_family", "mismatch_level"], sort=False
    ):
        rows.append(
            {
                "scenario_name": scenario,
                "mismatch_family": family,
                "mismatch_level": level,
                "method": "B2_RANDOM3_FINITE_VALIDATION",
                "case_count": group["case_id"].nunique(),
                "evaluation_row_count": len(group),
                "mean_final_J": float(group["J_final_truth"].mean()),
                "mean_regret": float(group["final_regret"].mean()),
                "median_regret": float(group["final_regret"].median()),
                "P95_regret": empirical_p95(group["final_regret"]),
                "max_regret": float(group["final_regret"].max()),
            }
        )
    return pd.DataFrame(rows)


def _paired_tables(
    per_case: pd.DataFrame, random: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    exact = per_case.pivot(index="case_id", columns="method", values="final_regret")
    roles = per_case.drop_duplicates("case_id").set_index("case_id")
    difference = (
        exact["B3_MODEL_TOP1_VALIDATION"]
        - exact["B4_FROZEN_TOP3_SEQUENTIAL"]
    )
    ci_low, ci_high = bootstrap_mean_ci(difference)
    paired_rows: list[dict[str, Any]] = []
    for case_id, value in difference.items():
        if value > MACHINE_COMPARISON_TOLERANCE:
            outcome = "TOP3_WIN"
        elif value < -MACHINE_COMPARISON_TOLERANCE:
            outcome = "TOP1_WIN"
        else:
            outcome = "TIE"
        paired_rows.append(
            {
                "case_id": case_id,
                "scenario_name": roles.loc[case_id, "scenario_name"],
                "mismatch_family": roles.loc[case_id, "mismatch_family"],
                "mismatch_level": roles.loc[case_id, "mismatch_level"],
                "top1_regret": float(exact.loc[case_id, "B3_MODEL_TOP1_VALIDATION"]),
                "top3_regret": float(exact.loc[case_id, "B4_FROZEN_TOP3_SEQUENTIAL"]),
                "delta_regret_top1_minus_top3": float(value),
                "paired_outcome": outcome,
            }
        )
    paired = pd.DataFrame(paired_rows)
    random_rows: list[dict[str, Any]] = []
    for case_id, group in random.groupby("case_id", sort=False):
        model_regret = float(exact.loc[case_id, "B4_FROZEN_TOP3_SEQUENTIAL"])
        values = group["final_regret"].to_numpy(dtype=float)
        random_rows.append(
            {
                "case_id": case_id,
                "scenario_name": roles.loc[case_id, "scenario_name"],
                "mismatch_family": roles.loc[case_id, "mismatch_family"],
                "mismatch_level": roles.loc[case_id, "mismatch_level"],
                "model_top3_regret": model_regret,
                "random3_mean_regret": float(np.mean(values)),
                "random3_median_regret": float(np.median(values)),
                "random3_P5_regret": float(np.quantile(values, 0.05)),
                "random3_P95_regret": empirical_p95(values),
                "model_strictly_beats_random_draw_percent": float(
                    100.0 * np.mean(model_regret < values)
                ),
                "model_ties_random_draw_percent": float(
                    100.0 * np.mean(np.isclose(model_regret, values, atol=1e-12, rtol=0.0))
                ),
                "paired_model_minus_random_mean_regret": model_regret - float(np.mean(values)),
                "model_harmful_candidate_exposure_count": int(
                    per_case.loc[
                        per_case["case_id"].eq(case_id)
                        & per_case["method"].eq("B4_FROZEN_TOP3_SEQUENTIAL"),
                        "final_harmful_selection",
                    ].sum()
                ),
                "random_mean_harmful_candidate_exposure_count": float(
                    group["harmful_candidate_exposure_count"].mean()
                ),
            }
        )
    random_comparison = pd.DataFrame(random_rows)
    standard_deviation = float(np.std(difference, ddof=1))
    effect = (
        float(np.mean(difference) / standard_deviation)
        if standard_deviation > 0.0
        else (float("inf") if float(np.mean(difference)) > 0.0 else 0.0)
    )
    statistics = {
        "top3_wins": int((paired["paired_outcome"] == "TOP3_WIN").sum()),
        "ties": int((paired["paired_outcome"] == "TIE").sum()),
        "top1_wins": int((paired["paired_outcome"] == "TOP1_WIN").sum()),
        "mean_paired_improvement": float(difference.mean()),
        "median_paired_improvement": float(difference.median()),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "paired_effect_size_cohen_dz": effect,
    }
    return paired, random_comparison, statistics


def _subject_specificity(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact = per_case.loc[
        ~per_case["method"].isin(["B2_RANDOM3_FINITE_VALIDATION"])
    ]
    for method, group in exact.groupby("method", sort=False):
        alpha = group["final_alpha_json"].astype(str)
        rows.append(
            {
                "method": method,
                "case_count": len(group),
                "unique_final_alpha_count": alpha.nunique(),
                "unique_final_alpha_fraction": float(alpha.nunique() / len(group)),
                "boundary_definition": "ANY_GENERATOR_COORDINATE_AT_FROZEN_BOUND",
                "boundary_selected_count": int(
                    alpha.str.contains(r"\[-5(?:\.0)?[,\]]|,2(?:\.0)?[,\]]|-0\.03|0\.03").sum()
                ),
                "truth_used_to_create_subject_specificity": False,
            }
        )
    return pd.DataFrame(rows)


def _budget_summary(budget: pd.DataFrame) -> pd.DataFrame:
    evaluated = budget.loc[budget["status"].eq("EVALUATED")]
    rows = []
    for k, group in evaluated.groupby("validation_budget_K", sort=True):
        low, high = bootstrap_mean_ci(
            group["final_regret"], seed=BOOTSTRAP_SEED + int(k)
        )
        rows.append(
            {
                "validation_budget_K": int(k),
                "case_count": len(group),
                "mean_regret": float(group["final_regret"].mean()),
                "median_regret": float(group["final_regret"].median()),
                "P95_regret": empirical_p95(group["final_regret"]),
                "mean_regret_bootstrap95_low": low,
                "mean_regret_bootstrap95_high": high,
            }
        )
    output = pd.DataFrame(rows)
    output["marginal_mean_regret_improvement_from_previous_K"] = -output[
        "mean_regret"
    ].diff()
    return output


def _conclusions(
    *,
    method_summary: pd.DataFrame,
    paired: pd.DataFrame,
    paired_statistics: Mapping[str, Any],
    random_comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    budget: pd.DataFrame,
) -> dict[str, Any]:
    methods = method_summary.set_index("method")
    top3_mean = float(methods.loc["B4_FROZEN_TOP3_SEQUENTIAL", "mean_regret"])
    random_mean = float(methods.loc["B2_RANDOM3_FINITE_VALIDATION", "mean_regret"])
    enrichment = ranking.loc[
        ranking["record_type"].eq("TOP_FRACTION_ENRICHMENT")
        & np.isclose(ranking["top_fraction"].astype(float), 0.05)
    ]
    mean_enrichment = float(enrichment["enrichment_ratio"].mean())
    mean_beat = float(
        random_comparison["model_strictly_beats_random_draw_percent"].mean()
    )
    screening_supported = (
        top3_mean < random_mean - MACHINE_COMPARISON_TOLERANCE
        and mean_beat > 50.0
        and mean_enrichment > 1.0
    )
    top1_sufficient = bool((paired["paired_outcome"] == "TIE").all())
    top3_resilience = (
        float(paired_statistics["mean_paired_improvement"])
        > MACHINE_COMPARISON_TOLERANCE
        and float(paired_statistics["bootstrap_95_ci_low"]) > 0.0
        and int(paired_statistics["top1_wins"]) == 0
    )
    evaluated_budget = budget.loc[budget["status"].eq("EVALUATED")]
    pivot = evaluated_budget.pivot(
        index="case_id", columns="validation_budget_K", values="final_regret"
    )
    saturation_k: int | None = None
    for candidate in (0, 1, 2):
        later = [value for value in (candidate + 1, 2, 3) if value > candidate]
        if all(
            np.allclose(
                pivot[candidate].to_numpy(dtype=float),
                pivot[value].to_numpy(dtype=float),
                atol=MACHINE_COMPARISON_TOLERANCE,
                rtol=0.0,
            )
            for value in sorted(set(later))
        ):
            saturation_k = candidate
            break
    limit_cases = random_comparison.merge(
        enrichment.loc[:, ["case_id", "enrichment_ratio"]],
        on="case_id",
        validate="one_to_one",
    )
    limit_cases["diagnostic_collapse"] = (
        limit_cases["model_top3_regret"]
        >= limit_cases["random3_median_regret"] - MACHINE_COMPARISON_TOLERANCE
    ) & limit_cases["enrichment_ratio"].le(1.0)
    limit_counts = limit_cases.groupby("mismatch_family")["diagnostic_collapse"].sum()
    trust_limit = bool((limit_counts >= 2).any())
    top_conclusion_clear = top1_sufficient or top3_resilience
    return {
        "model_screening_conclusion": (
            "MODEL_SCREENING_SUPPORTED"
            if screening_supported
            else "MODEL_SCREENING_NOT_SUPPORTED"
        ),
        "top1_top3_conclusion": (
            "TOP1_SUFFICIENT"
            if top1_sufficient
            else (
                "TOP3_MISMATCH_RESILIENCE_SUPPORTED"
                if top3_resilience
                else "TOP1_VS_TOP3_NOT_ESTABLISHED"
            )
        ),
        "budget_conclusion": (
            f"FINITE_VALIDATION_BUDGET_SATURATES_AT_K_{saturation_k}"
            if saturation_k is not None
            else "FINITE_VALIDATION_BUDGET_SATURATION_NOT_ESTABLISHED"
        ),
        "model_trust_limit_conclusion": (
            "MODEL_TRUST_LIMIT_IDENTIFIED"
            if trust_limit
            else "MODEL_TRUST_LIMIT_NOT_IDENTIFIED"
        ),
        "BO_BASELINE_REQUIRED_NEXT": bool(screening_supported and top_conclusion_clear),
        "mean_truth_top5_enrichment": mean_enrichment,
        "mean_model_top3_strict_beat_random_percent": mean_beat,
        "diagnostic_limit_cases": limit_cases.loc[
            limit_cases["diagnostic_collapse"], "case_id"
        ].astype(str).tolist(),
    }


def _plot_figures(
    output: Path,
    per_case: pd.DataFrame,
    random: pd.DataFrame,
    budget: pd.DataFrame,
    paired: pd.DataFrame,
    random_comparison: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    scenarios = list(dict.fromkeys(per_case["scenario_name"].astype(str)))
    x = np.arange(len(scenarios))
    methods = (
        ("B1_MODEL_ONLY", "Model-only", "o"),
        ("B3_MODEL_TOP1_VALIDATION", "Top-1", "s"),
        ("B4_FROZEN_TOP3_SEQUENTIAL", "Frozen Top-3", "D"),
    )
    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    random_stats = random.groupby("scenario_name")["final_regret"].agg(
        median="median",
        p5=lambda value: value.quantile(0.05),
        p95=lambda value: value.quantile(0.95),
    ).reindex(scenarios)
    ax.fill_between(x, random_stats["p5"], random_stats["p95"], alpha=0.18, label="Random-3 P5–P95")
    ax.plot(x, random_stats["median"], color="0.35", marker="^", label="Random-3 median")
    for method, label, marker in methods:
        values = (
            per_case.loc[per_case["method"].eq(method)]
            .groupby("scenario_name")["final_regret"]
            .mean()
            .reindex(scenarios)
        )
        ax.plot(x, values, marker=marker, linewidth=1.5, label=label)
    ax.axhline(0.0, color="0.2", linewidth=0.8, linestyle="--", label="Oracle regret = 0")
    ax.set_xticks(x, scenarios, rotation=32, ha="right")
    ax.set_ylabel("Final regret, J − J_oracle")
    ax.set_xlabel("Existing mismatch scenario (categorical; no invented global severity scalar)")
    ax.set_title("Model mismatch versus final regret")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures / "MODEL_MISMATCH_VS_FINAL_REGRET.png")
    plt.close(fig)

    budget_summary = _budget_summary(budget)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        budget_summary["validation_budget_K"],
        budget_summary["mean_regret"],
        marker="o",
        label="Mean regret",
    )
    ax.fill_between(
        budget_summary["validation_budget_K"],
        budget_summary["mean_regret_bootstrap95_low"],
        budget_summary["mean_regret_bootstrap95_high"],
        alpha=0.20,
        label="Bootstrap 95% CI of mean",
    )
    ax.plot(
        budget_summary["validation_budget_K"],
        budget_summary["median_regret"],
        marker="s",
        linestyle="--",
        label="Median regret",
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Full-cycle validation budget K")
    ax.set_ylabel("Final regret, J − J_oracle")
    ax.set_title("Trial budget versus final regret (K=5 preregistered skip)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(figures / "TRIAL_BUDGET_VS_FINAL_REGRET.png")
    plt.close(fig)

    ordered = random_comparison.sort_values("model_strictly_beats_random_draw_percent")
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    y = np.arange(len(ordered))
    ax.barh(y, ordered["model_strictly_beats_random_draw_percent"], alpha=0.75)
    ax.axvline(50.0, color="0.2", linestyle="--", linewidth=1.0, label="Random median")
    ax.set_yticks(y, ordered["case_id"], fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Random-3 draws with regret worse than model Top-3 (%)")
    ax.set_ylabel("Virtual case")
    ax.set_title("Model-screened Top-3 versus equal-budget Random-3")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.savefig(figures / "MODEL_SCREENING_VS_RANDOM3.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    colors = {"TOP3_WIN": "tab:blue", "TIE": "0.45", "TOP1_WIN": "tab:red"}
    for outcome, group in paired.groupby("paired_outcome", sort=False):
        ax.scatter(
            group["top1_regret"],
            group["top3_regret"],
            label=outcome.replace("_", " ").title(),
            color=colors[outcome],
            s=38,
            alpha=0.85,
        )
    maximum = float(max(paired["top1_regret"].max(), paired["top3_regret"].max()))
    ax.plot([0, maximum], [0, maximum], color="0.2", linestyle="--", linewidth=1.0)
    for row in paired.itertuples(index=False):
        if row.paired_outcome != "TIE":
            ax.annotate(row.case_id, (row.top1_regret, row.top3_regret), fontsize=6)
    ax.set_xlabel("Top-1 final regret")
    ax.set_ylabel("Frozen Top-3 final regret")
    ax.set_title("Top-1 versus Frozen Top-3 paired regret")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(figures / "TOP1_VS_TOP3_PAIRED_REGRET.png")
    plt.close(fig)

    candidate_ranks = ranking.loc[
        ranking["record_type"].eq("FROZEN_CANDIDATE_TRUTH_RANK")
    ]
    enrichment = ranking.loc[
        ranking["record_type"].eq("TOP_FRACTION_ENRICHMENT")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    data = [
        candidate_ranks.loc[candidate_ranks["candidate_id"].eq(candidate), "truth_rank_percentile"]
        for candidate in ("C1", "C2", "C3")
    ]
    axes[0].boxplot(data, tick_labels=["C1", "C2", "C3"], showfliers=True)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Truth rank percentile (lower is better; log scale)")
    axes[0].set_xlabel("Frozen prediction candidate")
    axes[0].set_title("Truth rank of model-screened candidates")
    axes[0].grid(axis="y", alpha=0.25)
    enrich_summary = enrichment.groupby("top_fraction")["enrichment_ratio"].agg(["mean", "median"])
    labels = [f"Top {100 * value:g}%" for value in enrich_summary.index]
    positions = np.arange(len(labels))
    axes[1].bar(positions - 0.18, enrich_summary["mean"], width=0.36, label="Mean")
    axes[1].bar(positions + 0.18, enrich_summary["median"], width=0.36, label="Median")
    axes[1].axhline(1.0, color="0.2", linestyle="--", linewidth=1.0, label="Uniform random")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Top-fraction enrichment ratio")
    axes[1].set_xlabel("Truth near-optimal region")
    axes[1].set_title("Model screening enrichment")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(figures / "MODEL_PREDICTION_RANKING_UTILITY.png")
    plt.close(fig)


def _audit_document(cases: pd.DataFrame, preflight: Mapping[str, Any]) -> str:
    scenario_lines = []
    for name in MISMATCH_SCENARIOS:
        scenario = get_mismatch_scenario(name)
        role = _scenario_role(name)
        active = {
            key: value
            for key, value in scenario.generator_parameters.items()
            if float(value) != 0.0
        }
        scenario_lines.append(
            f"- `{name}`: family `{role['mismatch_family']}`, level "
            f"`{role['mismatch_level']}`, terms {list(scenario.model_mismatch_terms)}, "
            f"parameters `{json.dumps(active, sort_keys=True)}`."
        )
    return f"""# CODE_AND_MISMATCH_AUDIT

## Existing mismatch definitions

{chr(10).join(scenario_lines)}

The actual generator definitions support mild/strong ordering only within the
nonlinear-stiffness, hip-knee-coupling, and combined families. Nonlinear
damping has only a mild definition and structured residual has one defined
level. Therefore this stage does not create a synthetic global severity scalar.

All nine scenario definitions are present in the unchanged 15-case V1 case
plan (`{len(cases)}` cases total, including matched subject-specific and the
previously preregistered strong-mismatch cases).

## Truth-access audit

`build_predicted_map` explicitly reports `truth_evaluated_during_prediction =
false`. The existing V1 freezes its shortlist, persists a manifest, and then
requires `FrozenShortlistTruthGate` authorization. This stress stage additionally
persists `STRESS_TEST_PROTOCOL.json` and `FROZEN_BASELINE_MANIFEST.json`, including
all Random-3 seeds and identities, before the shared truth-open token is issued.
The full 21,025-point truth landscape is read only after frozen V1 candidate
execution and is used as B5/post-selection evaluation, never candidate ordering.

## Frozen inputs

- V1 manifest SHA-256: `{preflight['frozen_v1_manifest_sha256']}`
- Candidate lattice: 21,025 points.
- Model-domain coverage gate: 90%.
- Mechanical equivalence tolerance: 0.005.
- Active reference SHA-256: `{ACTIVE_REFERENCE_SHA256}`.
- No robot, hardware, SDK, control, collection, wrench, or safety code is imported.
"""


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a small Markdown table without pandas' optional tabulate extra."""

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _report(
    *,
    conclusions: Mapping[str, Any],
    method_summary: pd.DataFrame,
    paired_statistics: Mapping[str, Any],
    paired: pd.DataFrame,
    random_comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    false_summary: pd.DataFrame,
    budget_summary: pd.DataFrame,
    protocol_sha256: str,
    baseline_manifest_sha256: str,
) -> str:
    methods = method_summary.set_index("method")
    top1 = methods.loc["B3_MODEL_TOP1_VALIDATION"]
    top3 = methods.loc["B4_FROZEN_TOP3_SEQUENTIAL"]
    random = methods.loc["B2_RANDOM3_FINITE_VALIDATION"]
    model_only = methods.loc["B1_MODEL_ONLY"]
    ranking_candidates = ranking.loc[
        ranking["record_type"].eq("FROZEN_CANDIDATE_TRUTH_RANK")
    ]
    enrichment = ranking.loc[
        ranking["record_type"].eq("TOP_FRACTION_ENRICHMENT")
    ]
    by_fraction = enrichment.groupby("top_fraction")["enrichment_ratio"].mean()
    family = paired.groupby("mismatch_family")["paired_outcome"].value_counts().unstack(fill_value=0)
    family_lines = "\n".join(
        f"- `{index}`: Top-3 wins {int(row.get('TOP3_WIN', 0))}, ties {int(row.get('TIE', 0))}, Top-1 wins {int(row.get('TOP1_WIN', 0))}."
        for index, row in family.iterrows()
    )
    budget_lines = "\n".join(
        f"- K={int(row.validation_budget_K)}: mean regret {row.mean_regret:.6f}, median {row.median_regret:.6f}, P95 {row.P95_regret:.6f}."
        for row in budget_summary.itertuples(index=False)
    )
    false_count = int(false_summary["predicted_false_improvement_count"].sum())
    predicted_count = int(false_summary["predicted_improvement_candidate_count"].sum())
    harmful_model_only = int(false_summary["model_only_final_harmful_selection"].sum())
    return f"""# MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_V1

- Protocol SHA-256: `{protocol_sha256}`
- Frozen baseline manifest SHA-256: `{baseline_manifest_sha256}`
- Integrity: `STRESS_TEST_PROTOCOL_INTEGRITY = PASS`
- Evidence: `OFFLINE_ONLY`, `NOT_HUMAN_READY`, `NOT_ROBOT_APPROVED`

## Scope and interpretation

This stage did not tune or replace the frozen V1 method. It evaluated the same
15 existing virtual cases, covering all nine existing mismatch scenario
definitions. No global mismatch severity scalar was invented; results are
reported by family and defined level. B5 Oracle was revealed only after every
baseline candidate identity and Random-3 seed had been frozen and persisted.

## Overall method comparison (Table A)

{_markdown_table(method_summary)}

Primary metric is final regret, not exact optimum hit rate. Near-optimal success
is shown at all preregistered tolerances (0.001, 0.0025, 0.005).

## Screening and ranking utility

- Model Top-3 mean regret: `{top3.mean_regret:.6f}`.
- Random-3 mean regret: `{random.mean_regret:.6f}`.
- Mean model strict-beat percentile within Random-3 distributions:
  `{conclusions['mean_model_top3_strict_beat_random_percent']:.1f}%`.
- Mean truth-top 1%/5%/10% enrichment: `{by_fraction.get(0.01, float('nan')):.3f}` /
  `{by_fraction.get(0.05, float('nan')):.3f}` / `{by_fraction.get(0.10, float('nan')):.3f}`.
- Median truth rank percentile of C1/C2/C3:
  `{ranking_candidates.groupby('candidate_id')['truth_rank_percentile'].median().to_dict()}`.

Screening conclusion: `{conclusions['model_screening_conclusion']}`.

## Top-1 versus Frozen Top-3 (Tables C and D)

- Wins/ties/losses from the Top-3 perspective:
  `{paired_statistics['top3_wins']}/{paired_statistics['ties']}/{paired_statistics['top1_wins']}`.
- Mean paired regret reduction (Top1 − Top3):
  `{paired_statistics['mean_paired_improvement']:.6f}`.
- Median paired reduction: `{paired_statistics['median_paired_improvement']:.6f}`.
- Bootstrap 95% CI: `[{paired_statistics['bootstrap_95_ci_low']:.6f},
  {paired_statistics['bootstrap_95_ci_high']:.6f}]`.
- Paired Cohen dz: `{paired_statistics['paired_effect_size_cohen_dz']}`.

Scenario-family concentration:

{family_lines}

Top-1/Top-3 conclusion: `{conclusions['top1_top3_conclusion']}`.

## Trial-budget ablation (Table E)

{budget_lines}

K=5 was preregistered as skipped because the frozen V1 selection helper enforces
at most three equivalence-band representatives. Extending it would redesign the
selection rule, which this stage forbids.

Budget conclusion: `{conclusions['budget_conclusion']}`.

## False improvement

Across the formal model-screened candidate pools, `{false_count}/{predicted_count}`
candidates had `J_pred < 1` but `J_truth >= 1`. Model-only made
`{harmful_model_only}` final harmful selections because fallback was deliberately
disabled. Validation methods retained Reference fallback; zero final harmful
selections there does not erase the model's prediction errors.

## Mismatch and trust limit

Increasing mismatch is not forced onto one scalar axis. Mild/strong comparisons
are valid only within the explicitly paired stiffness, coupling, and combined
families. The current diagnostic conclusion is
`{conclusions['model_trust_limit_conclusion']}`. A family-level trust limit is
declared only when at least two existing cases in that predefined family meet
the frozen collapse rule; isolated failures remain failure-regime observations,
not a generalized threshold.

## Direct answers

### Q1 Does the five-parameter model provide useful ranking/screening information?

`{conclusions['model_screening_conclusion']}`. This answer uses regret,
equal-budget Random-3 distributions, and top-fraction enrichment together.

### Q2 Does model screening outperform equal-budget random finite validation?

`{'YES' if conclusions['model_screening_conclusion'] == 'MODEL_SCREENING_SUPPORTED' else 'NO / NOT ESTABLISHED'}`.
Top-3's mean regret is `{top3.mean_regret:.6f}` versus Random-3
`{random.mean_regret:.6f}`; per-case distribution evidence is in
`MODEL_TOP3_VS_RANDOM3.csv`.

### Q3 Does Frozen Top-3 provide meaningful benefit beyond Top-1?

`{conclusions['top1_top3_conclusion']}`. Top-1 mean regret is
`{top1.mean_regret:.6f}` and Top-3 mean regret is `{top3.mean_regret:.6f}`.

### Q4 How does increasing model mismatch affect final regret and screening utility?

`NOT ESTABLISHED` as one global monotonic relation because the existing truth
definitions do not share a scientific scalar severity. Family-specific
mild/strong and scenario results are reported without pooling incompatible
mismatch mechanisms.

### Q5 How many full-cycle validation trials are empirically justified?

`{conclusions['budget_conclusion']}`. Evidence beyond K=3 was not generated;
K=5 would require changing the frozen rule.

## Final status

- `{conclusions['model_screening_conclusion']}`
- `{conclusions['top1_top3_conclusion']}`
- `{conclusions['budget_conclusion']}`
- `{conclusions['model_trust_limit_conclusion']}`
- `BO_BASELINE_REQUIRED_NEXT = {str(conclusions['BO_BASELINE_REQUIRED_NEXT']).lower()}`

No BO, new optimizer, prospective cohort, human experiment, or robot connection
was performed. The frozen V1 source and artifact hashes were checked before and
after this independent stage.
"""


def _checksums(output: Path, filenames: Sequence[str]) -> str:
    return "".join(
        f"{sha256_file(output / name)}  {name}\n" for name in sorted(filenames)
    )


def generate_artifacts(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preflight = _preflight()
    source_before = _source_hashes()
    v1_artifacts_before = _directory_hashes(FROZEN_V1_DIRECTORY)
    parameter_map = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    identity_lattice = _identity_lattice(parameter_map)
    cache = build_trajectory_component_cache(identity_lattice)
    cases = _case_plan()
    prepared = _prepare_all(cases, identity_lattice, cache)
    prepared_lookup = {str(item.role["case_id"]): item for item in prepared}
    if set(prepared_lookup) != set(cases["case_id"].astype(str)):
        raise RuntimeError("prepared stress cases differ from frozen case plan")
    shortlists_k2 = {
        case_id: freeze_model_screened_shortlist(
            item.initial_prediction_map,
            case_id=case_id,
            max_candidates=2,
        )
        for case_id, item in prepared_lookup.items()
    }
    random_universe = geometry_candidate_universe(
        identity_lattice.loc[
            :,
            [
                "trajectory_id",
                "trajectory_sha256",
                "hip_delta",
                "knee_delta",
                "phase_delta",
                "geometrically_admissible",
            ],
        ]
    )
    random_frames = [
        freeze_random3_candidates(random_universe, case_id=case_id)
        for case_id in cases["case_id"].astype(str)
    ]
    frozen_random = pd.concat(random_frames, ignore_index=True)
    identity_rows = _shortlist_identity_rows(prepared, shortlists_k2)
    protocol = _protocol_payload(
        preflight=preflight,
        source_hashes=source_before,
        cases=cases,
        candidate_universe_size=len(random_universe),
        v1_artifact_hashes=v1_artifacts_before,
    )
    protocol_path = output / "STRESS_TEST_PROTOCOL.json"
    _write_json(protocol_path, protocol, canonical=True)
    protocol_sha = sha256_file(protocol_path)
    baseline_manifest = _baseline_manifest_payload(
        protocol_sha256=protocol_sha,
        cases=cases,
        identity_rows=identity_rows,
        random_rows=frozen_random,
        shortlists_k2=shortlists_k2,
    )
    baseline_path = output / "FROZEN_BASELINE_MANIFEST.json"
    _write_json(baseline_path, baseline_manifest, canonical=True)
    baseline_sha = sha256_file(baseline_path)
    truth_gate = PersistedTruthGate()
    truth_gate.mark_persisted(
        protocol_sha256=protocol_sha,
        baseline_manifest_sha256=baseline_sha,
    )
    truth_open_token = truth_gate.authorize_truth()

    evaluated = _evaluate_all(
        prepared=prepared,
        shortlists_k2=shortlists_k2,
        frozen_random=frozen_random,
        cache=cache,
        baseline_manifest_sha256=baseline_sha,
    )
    per_case = evaluated["per_case"]
    random_results = evaluated["random"]
    budget = evaluated["budget"]
    ranking = evaluated["ranking"]
    false_summary = evaluated["false"]
    truth_access = evaluated["truth_access"]
    method_summary = _method_summary(per_case, random_results)
    scenario_summary = _scenario_summary(per_case, random_results)
    paired, random_comparison, paired_statistics = _paired_tables(
        per_case, random_results
    )
    specificity = _subject_specificity(per_case)
    budget_summary = _budget_summary(budget)
    conclusions = _conclusions(
        method_summary=method_summary,
        paired=paired,
        paired_statistics=paired_statistics,
        random_comparison=random_comparison,
        ranking=ranking,
        budget=budget,
    )
    tables = {
        "PER_CASE_RESULTS.csv": per_case,
        "RANDOM3_RESULTS.csv": random_results,
        "METHOD_SUMMARY.csv": method_summary,
        "SCENARIO_SUMMARY.csv": scenario_summary,
        "TRIAL_BUDGET_RESULTS.csv": budget,
        "TOP1_TOP3_PAIRED.csv": paired,
        "MODEL_TOP3_VS_RANDOM3.csv": random_comparison,
        "RANKING_UTILITY.csv": ranking,
        "FALSE_IMPROVEMENT_SUMMARY.csv": false_summary,
        "SUBJECT_SPECIFICITY.csv": specificity,
        "TRUTH_ACCESS_AUDIT.csv": truth_access,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_text(output / "CODE_AND_MISMATCH_AUDIT.md", _audit_document(cases, preflight))
    _plot_figures(
        output,
        per_case,
        random_results,
        budget,
        paired,
        random_comparison,
        ranking,
    )
    report = _report(
        conclusions=conclusions,
        method_summary=method_summary,
        paired_statistics=paired_statistics,
        paired=paired,
        random_comparison=random_comparison,
        ranking=ranking,
        false_summary=false_summary,
        budget_summary=budget_summary,
        protocol_sha256=protocol_sha,
        baseline_manifest_sha256=baseline_sha,
    )
    _write_text(output / "MODEL_TRUST_FINITE_VALIDATION_STRESS_TEST_REPORT.md", report)
    source_after = _source_hashes()
    v1_artifacts_after = _directory_hashes(FROZEN_V1_DIRECTORY)
    if source_after != source_before:
        raise RuntimeError("protected scientific source changed during stress test")
    if v1_artifacts_after != v1_artifacts_before:
        raise RuntimeError("frozen V1 artifacts changed during stress test")
    scientific_files = [*REQUIRED_OUTPUTS]
    scientific_result_sha = hashlib.sha256(
        "".join(
            f"{name}:{sha256_file(output / name)}\n"
            for name in sorted(scientific_files)
        ).encode("utf-8")
    ).hexdigest()
    _write_text(output / "checksums.sha256", _checksums(output, scientific_files))
    metadata = {
        "stage_id": STAGE_ID,
        "default_enabled": DEFAULT_ENABLED,
        "evidence_level": OFFLINE_ONLY,
        "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "protocol_sha256": protocol_sha,
        "frozen_baseline_manifest_sha256": baseline_sha,
        "truth_open_token": truth_open_token,
        "STRESS_TEST_PROTOCOL_INTEGRITY": "PASS",
        "candidate_truth_opened_after_both_manifests_persisted": True,
        "case_count": len(cases),
        "scenario_count": cases["scenario_name"].nunique(),
        "random_repeat_count_per_case": N_RANDOM_REPEATS,
        "random_result_count": len(random_results),
        "candidate_lattice_size": len(identity_lattice),
        "shared_random_candidate_universe_size": len(random_universe),
        "conclusions": conclusions,
        "paired_statistics": dict(paired_statistics),
        "scientific_result_sha256": scientific_result_sha,
        "frozen_v1_artifacts_unchanged": True,
        "frozen_v1_source_unchanged": True,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "held_out_final_test_read": False,
        "new_prospective_cohort_generated": False,
        "bayesian_optimization_implemented": False,
        "robot_connected": False,
        "hardware_control_collection_safety_modified": False,
        "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            name: sha256_file(output / name) for name in scientific_files
        },
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
