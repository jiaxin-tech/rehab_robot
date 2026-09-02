"""Independent decision-error calibration for the next P2 research revision.

This module defines a calibration-only cohort and truth-free pair assignments.
It does not run personalization, select a guard percentile, or implement a new
P2 policy.  Calibration endpoint truth may be attached only through a frozen
manifest gate after every case and pair assignment has been persisted.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .decision_relevant_global_model_reliability import (
    build_predicted_map,
    build_trajectory_component_cache,
    evaluate_truth_map,
)
from .dynamic_subject import DYNAMIC_SUBJECTS, DynamicVirtualSubject, get_dynamic_subject
from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .p2_v2_prospective_offline_validation import (
    DEVELOPMENT_CASES,
    prospective_case_rows,
    prospective_subject_definitions,
)
from .post_prospective_rejection_root_cause_audit import (
    BUNDLE_PROTOCOL_ID,
    PROSPECTIVE_CONCLUSION,
    PROSPECTIVE_MANIFEST_SHA256,
    PROSPECTIVE_START_COMMIT,
)
from .research_decision_guarded_sequential_personalization import (
    _model_for_iteration,
    build_initial_research_state,
)


CALIBRATION_ID = "P2_NEXT_REVISION_INDEPENDENT_CALIBRATION_V1"
CALIBRATION_MANIFEST_ID = "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1"
CALIBRATION_DATA_ROLE = "INDEPENDENT_CALIBRATION_DECISION_ERROR_ONLY"
CALIBRATION_SUBJECT_SELECTION_SEED = 20260824
CALIBRATION_SUBJECT_COUNT = 6
CALIBRATION_CASE_COUNT = 12
LOCAL_CALIBRATION_PLAN_ID = "INDEPENDENT_LOCAL_CALIBRATION_PAIR_PLAN_V1"
LOCAL_SOURCE_PROTOCOL_ID = "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1"
LOCAL_SOURCE_PAIR_PLAN_SHA256 = (
    "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"
)
BUNDLE_SOURCE_PAIR_PLAN_SHA256 = (
    "3808bfe8819ded263a1cac847e3234e39878623ed5332e57b2bb4bd17e26ee84"
)
HELD_OUT_STATUS = "HELD_OUT_FINAL_TEST_NOT_READ"
FUTURE_PROSPECTIVE_STATUS = "NOT_CREATED_IN_THIS_TASK"
BUNDLE_SCALE_CALIBRATED = "BUNDLE_SCALE_CALIBRATED_FOR_RESEARCH"
BUNDLE_SCALE_INSUFFICIENT = "BUNDLE_SCALE_CALIBRATION_INSUFFICIENT"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_MOTION_APPROVED = "NOT_ROBOT_MOTION_APPROVED"

MODULE_DIR = Path(__file__).resolve().parent
LOCAL_SOURCE_PAIR_PLAN_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "p2_v2_formal_research_protocol_v1"
    / "designated_local_validation_pair_plan.csv"
)
BUNDLE_SOURCE_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "post_prospective_rejection_root_cause_audit_v1"
)
BUNDLE_SOURCE_PAIR_PLAN_PATH = (
    BUNDLE_SOURCE_DIRECTORY / "designated_bundle_validation_pair_plan.csv"
)
BUNDLE_SOURCE_PROTOCOL_PATH = (
    BUNDLE_SOURCE_DIRECTORY / "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1.json"
)

_PARAMETER_GRID = tuple(
    itertools.product(
        (0.88, 0.96, 1.04, 1.12),
        (11.0, 18.0, 24.0, 27.0),
        (9.0, 16.0, 21.0, 26.0),
        (0.9, 1.1),
    )
)
_MISMATCH_ASSIGNMENT = (
    "nonlinear_stiffness_strong",
    "hip_knee_coupling_strong",
    "combined_strong",
    "nonlinear_stiffness_strong",
    "hip_knee_coupling_strong",
    "combined_strong",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(table: pd.DataFrame) -> str:
    payload = table.to_csv(
        index=False, lineterminator="\n", float_format="%.12g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grid_signature(values: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 12) for value in values)  # type: ignore[return-value]


def _dynamic_subject_grid_signature(
    subject: DynamicVirtualSubject,
    baseline: DynamicVirtualSubject,
) -> tuple[float, float, float, float]:
    return _grid_signature(
        (
            subject.mass_thigh_kg / baseline.mass_thigh_kg,
            subject.k_hip_nm_per_rad,
            subject.k_knee_nm_per_rad,
            subject.b_hip_nm_s_per_rad / baseline.b_hip_nm_s_per_rad,
        )
    )


def calibration_subject_definitions() -> tuple[dict[str, Any], ...]:
    """Hash-select six unused parameter combinations without truth outcomes."""

    baseline = get_dynamic_subject("baseline")
    excluded = {
        _grid_signature(
            (
                item["mass_scale"],
                item["parameters"]["k_hip_nm_per_rad"],
                item["parameters"]["k_knee_nm_per_rad"],
                item["damping_scale"],
            )
        )
        for item in prospective_subject_definitions()
    }
    excluded.update(
        _dynamic_subject_grid_signature(subject, baseline)
        for subject in DYNAMIC_SUBJECTS.values()
    )
    ranked: list[tuple[str, tuple[float, float, float, float]]] = []
    for values in _PARAMETER_GRID:
        signature = _grid_signature(values)
        if signature in excluded:
            continue
        identity = "|".join(
            (
                CALIBRATION_ID,
                str(CALIBRATION_SUBJECT_SELECTION_SEED),
                *(f"{value:.12g}" for value in signature),
            )
        )
        ranked.append((hashlib.sha256(identity.encode("utf-8")).hexdigest(), signature))
    selected = sorted(ranked)[:CALIBRATION_SUBJECT_COUNT]
    if len(selected) != CALIBRATION_SUBJECT_COUNT:
        raise RuntimeError("independent calibration subject grid is too small")
    output: list[dict[str, Any]] = []
    for index, (selection_hash, values) in enumerate(selected, start=1):
        mass_scale, k_hip, k_knee, damping_scale = values
        subject_id = f"calibration_subject_{index:03d}"
        subject = DynamicVirtualSubject(
            subject_id=subject_id,
            mass_thigh_kg=baseline.mass_thigh_kg * mass_scale,
            mass_shank_kg=baseline.mass_shank_kg * mass_scale,
            com_thigh_m=baseline.com_thigh_m,
            com_shank_m=baseline.com_shank_m,
            inertia_thigh_kg_m2=baseline.inertia_thigh_kg_m2 * mass_scale,
            inertia_shank_kg_m2=baseline.inertia_shank_kg_m2 * mass_scale,
            b_hip_nm_s_per_rad=baseline.b_hip_nm_s_per_rad * damping_scale,
            b_knee_nm_s_per_rad=baseline.b_knee_nm_s_per_rad * damping_scale,
            k_hip_nm_per_rad=k_hip,
            k_knee_nm_per_rad=k_knee,
            q0_hip_rad=baseline.q0_hip_rad,
            q0_knee_rad=baseline.q0_knee_rad,
            gravity_m_s2=baseline.gravity_m_s2,
        )
        output.append(
            {
                "subject_id": subject_id,
                "selection_hash": selection_hash,
                "selection_seed": CALIBRATION_SUBJECT_SELECTION_SEED,
                "selection_rule": (
                    "LOWEST_SHA256_OVER_FIXED_UNUSED_SYNTHETIC_GRID_AFTER_"
                    "EXCLUDING_DEVELOPMENT_AND_REJECTED_PROSPECTIVE_SIGNATURES"
                ),
                "candidate_grid_size_before_exclusion": len(_PARAMETER_GRID),
                "eligible_grid_size_after_exclusion": len(ranked),
                "mass_scale": mass_scale,
                "damping_scale": damping_scale,
                "k_hip_nm_per_rad": k_hip,
                "k_knee_nm_per_rad": k_knee,
                "parameter_signature": "|".join(f"{value:.12g}" for value in values),
                "parameters": subject.as_metadata_dict(),
                "virtual_parameter_source": (
                    "PREEXISTING_FIXED_RESEARCH_SYNTHETIC_PARAMETER_GRID_UNUSED_COMBINATION"
                ),
                "parameter_interpretation": "OFFLINE_EQUIVALENT_SYNTHETIC_NOT_CLINICAL",
                "truth_used_for_selection": False,
                "truth_optimum_used_for_selection": False,
                "error_magnitude_used_for_selection": False,
                "subject_specificity_used_for_selection": False,
            }
        )
    return tuple(output)


def calibration_subject_for_id(subject_id: str) -> DynamicVirtualSubject:
    definitions = {
        str(item["subject_id"]): item for item in calibration_subject_definitions()
    }
    try:
        parameters = definitions[str(subject_id)]["parameters"]
    except KeyError as exc:
        raise ValueError(f"unknown calibration subject: {subject_id}") from exc
    return DynamicVirtualSubject(**parameters)


@contextmanager
def registered_calibration_subject(
    subject: DynamicVirtualSubject,
) -> Iterator[None]:
    if subject.subject_id in DYNAMIC_SUBJECTS:
        raise RuntimeError("calibration subject ID collides with existing registry")
    before_keys = tuple(DYNAMIC_SUBJECTS)
    DYNAMIC_SUBJECTS[subject.subject_id] = subject
    try:
        yield
    finally:
        current = DYNAMIC_SUBJECTS.pop(subject.subject_id, None)
        if current is not subject or tuple(DYNAMIC_SUBJECTS) != before_keys:
            raise RuntimeError("temporary calibration subject registry was not restored")


def calibration_case_manifest() -> pd.DataFrame:
    """Create 6 matched and 6 mismatch cases before any calibration truth."""

    old_development = set(DEVELOPMENT_CASES)
    old_prospective = set(prospective_case_rows()["case_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(calibration_subject_definitions()):
        subject_id = str(definition["subject_id"])
        for scenario_name, category in (
            ("matched_linear", "MATCHED"),
            (_MISMATCH_ASSIGNMENT[index], "MISMATCH"),
        ):
            case_id = f"{subject_id}__{scenario_name}"
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "scenario_name": scenario_name,
                    "calibration_category": category,
                    "data_split": "INDEPENDENT_CALIBRATION",
                    "data_role": CALIBRATION_DATA_ROLE,
                    "subject_selection_hash": definition["selection_hash"],
                    "subject_selection_seed": definition["selection_seed"],
                    "parameter_signature": definition["parameter_signature"],
                    "mass_scale": definition["mass_scale"],
                    "damping_scale": definition["damping_scale"],
                    "k_hip_nm_per_rad": definition["k_hip_nm_per_rad"],
                    "k_knee_nm_per_rad": definition["k_knee_nm_per_rad"],
                    "virtual_parameter_source": definition["virtual_parameter_source"],
                    "mismatch_mechanism_source": (
                        "NONE_MATCHED_LINEAR"
                        if category == "MATCHED"
                        else f"PREEXISTING_FROZEN_MISMATCH_SCENARIO:{scenario_name}"
                    ),
                    "truth_used_for_case_selection": False,
                    "truth_optimum_used_for_case_selection": False,
                    "error_used_for_case_selection": False,
                    "old_development_case": case_id in old_development,
                    "rejected_prospective_case": case_id in old_prospective,
                    "reserved_for_future_prospective": False,
                    "heldout_final_test": False,
                    "calibration_only": True,
                }
            )
    output = pd.DataFrame(rows).sort_values("case_id", kind="mergesort").reset_index(drop=True)
    if len(output) != CALIBRATION_CASE_COUNT or output["case_id"].duplicated().any():
        raise RuntimeError("calibration case manifest identity failure")
    if output[["old_development_case", "rejected_prospective_case", "heldout_final_test"]].astype(bool).any().any():
        raise RuntimeError("calibration case overlaps a protected data split")
    if set(output["calibration_category"]) != {"MATCHED", "MISMATCH"}:
        raise RuntimeError("calibration case categories are incomplete")
    return output


def assign_pairs_to_calibration_cases(
    plan: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    pair_id_column: str,
    strata_columns: Sequence[str],
    assignment_id: str,
) -> pd.DataFrame:
    """Assign exactly one pair per case in every 12-pair stratum by hash."""

    if plan[pair_id_column].duplicated().any():
        raise ValueError("pair plan contains duplicate identities")
    if len(cases) != CALIBRATION_CASE_COUNT:
        raise ValueError("calibration assignment requires the frozen 12 cases")
    case_ids = tuple(sorted(cases["case_id"].astype(str)))
    case_lookup = cases.set_index("case_id")
    rows: list[dict[str, Any]] = []
    for keys, group in plan.groupby(list(strata_columns), sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        if len(group) != len(case_ids):
            raise RuntimeError(
                f"assignment stratum must contain one pair per case: {key_tuple}"
            )
        stratum_id = "|".join(str(value) for value in key_tuple)
        ordered_pairs = group.sort_values(
            ["selection_hash", pair_id_column], kind="mergesort"
        )
        ordered_cases = sorted(
            case_ids,
            key=lambda case_id: hashlib.sha256(
                f"{CALIBRATION_ID}|{assignment_id}|{stratum_id}|{case_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        for rank, (pair, case_id) in enumerate(
            zip(ordered_pairs.to_dict(orient="records"), ordered_cases), start=1
        ):
            case = case_lookup.loc[case_id]
            rows.append(
                {
                    pair_id_column: pair[pair_id_column],
                    "case_id": case_id,
                    "subject_id": case["subject_id"],
                    "scenario_name": case["scenario_name"],
                    "calibration_category": case["calibration_category"],
                    "assignment_id": assignment_id,
                    "assignment_stratum_id": stratum_id,
                    "within_stratum_assignment_rank": rank,
                    "case_assignment_rule": (
                        "ONE_PAIR_PER_FROZEN_CASE_PER_STRATUM_BY_SHA256_CASE_PERMUTATION"
                    ),
                    "prediction_used_for_assignment": False,
                    "truth_used_for_assignment": False,
                    "error_used_for_assignment": False,
                    "truth_optimum_used_for_assignment": False,
                    "future_prospective_used_for_assignment": False,
                    "heldout_final_test_used_for_assignment": False,
                    "data_role": CALIBRATION_DATA_ROLE,
                }
            )
    output = pd.DataFrame(rows).sort_values(pair_id_column, kind="mergesort").reset_index(drop=True)
    if len(output) != len(plan) or output[pair_id_column].duplicated().any():
        raise RuntimeError("calibration assignment is incomplete")
    counts = output.groupby("assignment_stratum_id")["case_id"].nunique()
    if set(counts) != {CALIBRATION_CASE_COUNT}:
        raise RuntimeError("calibration assignment lost per-stratum case balance")
    return output


class FrozenCalibrationManifestGate:
    """Fail closed unless the exact calibration manifest bytes stay frozen."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.access_records: list[dict[str, Any]] = []

    def require_frozen(self) -> None:
        if not self.path.is_file() or sha256_file(self.path) != self.expected_sha256:
            raise PermissionError("calibration truth requires the frozen manifest")

    def record_truth_access(self, *, case_id: str, stage: str) -> None:
        self.require_frozen()
        self.access_records.append(
            {
                "case_id": str(case_id),
                "stage": str(stage),
                "manifest_sha256": self.expected_sha256,
                "manifest_verified_before_truth": True,
                "truth_used_for_selection_or_assignment": False,
            }
        )


def _reference_id(parameter_lattice: pd.DataFrame) -> str:
    reference = parameter_lattice.loc[
        np.isclose(parameter_lattice["hip_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(parameter_lattice["knee_delta"], 0.0, atol=1e-12, rtol=0.0)
        & np.isclose(parameter_lattice["phase_delta"], 0.0, atol=1e-12, rtol=0.0)
    ]
    if len(reference) != 1:
        raise RuntimeError("formal lattice lacks one active reference")
    return str(reference.iloc[0]["trajectory_id"])


def _selected_points(parameter_lattice: pd.DataFrame, identifiers: Sequence[str]) -> pd.DataFrame:
    lookup = parameter_lattice.set_index("trajectory_id", drop=False)
    unique = sorted({str(value) for value in identifiers})
    missing = set(unique).difference(lookup.index.astype(str))
    if missing:
        raise RuntimeError(f"assigned calibration trajectories missing: {sorted(missing)}")
    return lookup.loc[unique].reset_index(drop=True).copy()


def evaluate_frozen_calibration_assignments(
    cases: pd.DataFrame,
    local_plan: pd.DataFrame,
    local_assignment: pd.DataFrame,
    bundle_plan: pd.DataFrame,
    bundle_assignment: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
    gate: FrozenCalibrationManifestGate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate direct pair endpoint residuals after the manifest freeze."""

    gate.require_frozen()
    reference_id = _reference_id(parameter_lattice)
    local = local_plan.merge(local_assignment, on="pair_id", validate="one_to_one")
    bundles = bundle_plan.merge(
        bundle_assignment, on="bundle_pair_id", validate="one_to_one"
    )
    all_ids = [
        reference_id,
        *local["trajectory_i"].astype(str),
        *local["trajectory_j"].astype(str),
        *bundles["start_trajectory_id"].astype(str),
        *bundles["endpoint_trajectory_id"].astype(str),
    ]
    master_points = _selected_points(parameter_lattice, all_ids)
    cache = build_trajectory_component_cache(master_points)
    local_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for case in cases.to_dict(orient="records"):
        case_id = str(case["case_id"])
        subject = calibration_subject_for_id(str(case["subject_id"]))
        case_local = local.loc[local["case_id"].eq(case_id)].copy()
        case_bundles = bundles.loc[bundles["case_id"].eq(case_id)].copy()
        identifiers = [
            reference_id,
            *case_local["trajectory_i"].astype(str),
            *case_local["trajectory_j"].astype(str),
            *case_bundles["start_trajectory_id"].astype(str),
            *case_bundles["endpoint_trajectory_id"].astype(str),
        ]
        points = _selected_points(parameter_lattice, identifiers)
        with registered_calibration_subject(subject):
            gate.record_truth_access(
                case_id=case_id, stage="FROZEN_SEQUENTIAL_IDENTIFICATION_FIXTURE"
            )
            state = build_initial_research_state(
                str(case["subject_id"]), str(case["scenario_name"])
            )
            model = _model_for_iteration(state, state.parameters, state.domain_data, 0)
            predicted, prediction_audit = build_predicted_map(model, points, cache)
            if prediction_audit["truth_evaluated_during_prediction"]:
                raise RuntimeError("calibration truth leaked into prediction")
            gate.record_truth_access(
                case_id=case_id, stage="FROZEN_PAIR_ENDPOINT_TRUTH_EVALUATION"
            )
            evaluated, truth_audit = evaluate_truth_map(predicted, model, cache)
            if truth_audit["truth_used_for_pre_evaluation_ranking"]:
                raise RuntimeError("calibration truth leaked into assignment")
        lookup = evaluated.set_index("trajectory_id")
        model_rows.append(
            {
                "case_id": case_id,
                "subject_id": case["subject_id"],
                "scenario_name": case["scenario_name"],
                "calibration_category": case["calibration_category"],
                "selected_identification_trial_id": model.selected_trial_id,
                "identification_dataset_sha256": model.identification_dataset_sha256,
                "identification_domain_valid_samples": model.identification_domain.valid_training_samples,
                **{f"estimated_{name}": value for name, value in model.parameters.items()},
                "truth_used_for_pair_assignment": False,
                "pair_endpoint_truth_used_for_model_fitting": False,
                "personalization_executed": False,
                "data_role": CALIBRATION_DATA_ROLE,
            }
        )
        for item in case_local.to_dict(orient="records"):
            first = lookup.loc[str(item["trajectory_i"])]
            second = lookup.loc[str(item["trajectory_j"])]
            delta_pred = float(second["J_pred"] - first["J_pred"])
            delta_truth = float(second["J_truth"] - first["J_truth"])
            local_rows.append(
                {
                    "pair_id": item["pair_id"],
                    "case_id": case_id,
                    "subject_id": case["subject_id"],
                    "scenario_name": case["scenario_name"],
                    "calibration_category": case["calibration_category"],
                    "coordinate": item["coordinate"],
                    "direction": "POSITIVE_CANONICAL",
                    "trust_level": item["trust_level"],
                    "trust_step": item["trust_step"],
                    "location_class": item["location_class"],
                    "trajectory_i": item["trajectory_i"],
                    "trajectory_j": item["trajectory_j"],
                    "deltaJ_pred": delta_pred,
                    "deltaJ_truth": delta_truth,
                    "e_deltaJ_1": abs(delta_pred - delta_truth),
                    "reverse_deltaJ_pred": -delta_pred,
                    "reverse_deltaJ_truth": -delta_truth,
                    "reverse_e_deltaJ_1": abs(delta_pred - delta_truth),
                    "residual_computation": "DIRECT_PAIRED_ENDPOINT_DIFFERENCE",
                    "reverse_pair_error_is_symmetric": bool(item["reverse_pair_error_is_symmetric"]),
                    "negative_direction_is_independent_pair": False,
                    "truth_read_after_manifest_freeze": True,
                    "truth_used_for_selection_or_assignment": False,
                    "used_to_select_threshold": False,
                    "used_by_policy": False,
                    "heldout_final_test_used": False,
                    "data_role": CALIBRATION_DATA_ROLE,
                }
            )
        for item in case_bundles.to_dict(orient="records"):
            first = lookup.loc[str(item["start_trajectory_id"])]
            second = lookup.loc[str(item["endpoint_trajectory_id"])]
            delta_pred = float(second["J_pred"] - first["J_pred"])
            delta_truth = float(second["J_truth"] - first["J_truth"])
            length = int(item["bundle_length"])
            bundle_rows.append(
                {
                    "bundle_pair_id": item["bundle_pair_id"],
                    "case_id": case_id,
                    "subject_id": case["subject_id"],
                    "scenario_name": case["scenario_name"],
                    "calibration_category": case["calibration_category"],
                    "coordinate": item["coordinate"],
                    "direction": item["direction"],
                    "bundle_length": length,
                    "location_class": item["location_class"],
                    "start_trajectory_id": item["start_trajectory_id"],
                    "endpoint_trajectory_id": item["endpoint_trajectory_id"],
                    "intermediate_trajectory_ids": item["intermediate_trajectory_ids"],
                    "deltaJ_pred": delta_pred,
                    "deltaJ_truth": delta_truth,
                    "e_deltaJ_bundle": abs(delta_pred - delta_truth),
                    "residual_column": f"e_deltaJ_{length}",
                    "residual_computation": "DIRECT_START_TO_ENDPOINT_DIFFERENCE",
                    "n_times_one_step_uncertainty_used": False,
                    "sqrt_n_times_one_step_uncertainty_used": False,
                    "analytic_scaling_formula_used": False,
                    "truth_read_after_manifest_freeze": True,
                    "truth_used_for_selection_or_assignment": False,
                    "used_to_select_threshold": False,
                    "used_by_policy": False,
                    "heldout_final_test_used": False,
                    "data_role": CALIBRATION_DATA_ROLE,
                }
            )
    local_output = pd.DataFrame(local_rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    bundle_output = pd.DataFrame(bundle_rows).sort_values("bundle_pair_id", kind="mergesort").reset_index(drop=True)
    model_output = pd.DataFrame(model_rows).sort_values("case_id", kind="mergesort").reset_index(drop=True)
    if len(local_output) != len(local_plan) or len(bundle_output) != len(bundle_plan):
        raise RuntimeError("calibration residual evaluation is incomplete")
    if not np.isfinite(local_output["e_deltaJ_1"]).all() or not np.isfinite(bundle_output["e_deltaJ_bundle"]).all():
        raise RuntimeError("calibration residuals must be finite")
    return local_output, bundle_output, model_output


def residual_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("residual distribution requires finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "P90": float(np.percentile(array, 90)),
        "P95": float(np.percentile(array, 95)),
        "P99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def grouped_residual_summary(
    table: pd.DataFrame,
    *,
    residual_column: str,
    groups: Sequence[str],
    decision_scale: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    iterator = table.groupby(list(groups), sort=True, dropna=False) if groups else [((), table)]
    for keys, selected in iterator:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {name: value for name, value in zip(groups, key_tuple)}
        row.update(residual_distribution(selected[residual_column]))
        row["decision_scale"] = decision_scale or "MIXED"
        row["percentile_method"] = "NUMPY_LINEAR"
        row["threshold_selected"] = False
        row["data_role"] = CALIBRATION_DATA_ROLE
        rows.append(row)
    return pd.DataFrame(rows)


def bundle_scale_feasibility(bundle_residuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for length in (2, 3, 5):
        selected = bundle_residuals.loc[bundle_residuals["bundle_length"].eq(length)]
        stratum_counts = selected.groupby(
            ["coordinate", "direction", "location_class"]
        ).size()
        complete = bool(
            len(selected) == 216
            and len(stratum_counts) == 18
            and set(stratum_counts) == {12}
            and np.isfinite(selected["e_deltaJ_bundle"]).all()
            and selected["residual_computation"].eq(
                "DIRECT_START_TO_ENDPOINT_DIFFERENCE"
            ).all()
        )
        rows.append(
            {
                "bundle_length": length,
                "pair_count": len(selected),
                "stratum_count": len(stratum_counts),
                "pairs_per_stratum_min": int(stratum_counts.min()),
                "pairs_per_stratum_max": int(stratum_counts.max()),
                "all_residuals_finite": bool(np.isfinite(selected["e_deltaJ_bundle"]).all()),
                "direct_endpoint_residual_only": bool(
                    selected["residual_computation"].eq(
                        "DIRECT_START_TO_ENDPOINT_DIFFERENCE"
                    ).all()
                ),
                "calibration_status": (
                    BUNDLE_SCALE_CALIBRATED if complete else BUNDLE_SCALE_INSUFFICIENT
                ),
                "research_uncertainty_candidate_may_be_designed_later": complete,
                "formal_threshold_ready": False,
                "policy_enabled": False,
                "statistical_power_claimed": False,
                "data_role": CALIBRATION_DATA_ROLE,
            }
        )
    return pd.DataFrame(rows)


def calibration_manifest_payload(
    *,
    checkpoint_commit: str,
    case_manifest_sha256: str,
    local_plan_sha256: str,
    local_assignment_sha256: str,
    bundle_plan_sha256: str,
    bundle_assignment_sha256: str,
    protected_source_sha256: Mapping[str, str],
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_id": CALIBRATION_MANIFEST_ID,
        "calibration_id": CALIBRATION_ID,
        "status": "FROZEN_BEFORE_ANY_NEW_CALIBRATION_TRUTH",
        "checkpoint_commit": checkpoint_commit,
        "data_role": CALIBRATION_DATA_ROLE,
        "data_split": {
            "development": (
                "EARLY_P2_DEVELOPMENT_PLUS_SIX_REJECTED_PROSPECTIVE_CASES_"
                "DEVELOPMENT_USED"
            ),
            "independent_calibration": "TWELVE_NEW_CASES_THIS_MANIFEST_ONLY",
            "future_prospective": FUTURE_PROSPECTIVE_STATUS,
            "held_out_final_test": HELD_OUT_STATUS,
        },
        "case_ids": [str(row["case_id"]) for row in case_rows],
        "case_count": len(case_rows),
        "matched_case_count": sum(
            str(row["calibration_category"]) == "MATCHED" for row in case_rows
        ),
        "mismatch_case_count": sum(
            str(row["calibration_category"]) == "MISMATCH" for row in case_rows
        ),
        "case_selection": {
            "seed": CALIBRATION_SUBJECT_SELECTION_SEED,
            "virtual_parameter_source": (
                "FIXED_REPOSITORY_SYNTHETIC_GRID_UNUSED_COMBINATIONS"
            ),
            "mismatch_mechanism_source": (
                "PREEXISTING_STRONG_MISMATCH_SCENARIOS_ASSIGNED_BY_SUBJECT_INDEX"
            ),
            "selection_uses_truth": False,
            "selection_uses_truth_optimum": False,
            "selection_uses_error_magnitude": False,
            "selection_uses_subject_specificity": False,
        },
        "patient_envelope_fixture": {
            "envelope_id": "VIRTUAL_RESEARCH_ENVELOPE_DEFAULT",
            "hip_deg": [20.0, 115.0],
            "knee_deg": [15.0, 135.0],
            "status": "SYNTHETIC_TEST_FIXTURE_NOT_CLINICAL_SAFETY_LIMIT",
            "auto_expand": False,
        },
        "identification_procedure": (
            "SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1_WITH_"
            "STOP_RULE_NONE_THEN_DIAGNOSTIC_MODEL_SELECTION"
        ),
        "local_validation_protocol": {
            "protocol_id": LOCAL_CALIBRATION_PLAN_ID,
            "source_protocol_id": LOCAL_SOURCE_PROTOCOL_ID,
            "source_geometry_unchanged": True,
            "source_pair_plan_sha256": LOCAL_SOURCE_PAIR_PLAN_SHA256,
            "independent_plan_sha256": local_plan_sha256,
            "assignment_sha256": local_assignment_sha256,
            "assignment_uses_truth": False,
        },
        "bundle_validation_protocol": {
            "protocol_id": BUNDLE_PROTOCOL_ID,
            "source_pair_plan_sha256": BUNDLE_SOURCE_PAIR_PLAN_SHA256,
            "verified_pair_plan_sha256": bundle_plan_sha256,
            "assignment_sha256": bundle_assignment_sha256,
            "assignment_uses_truth": False,
            "bundle_lengths": [2, 3, 5],
        },
        "case_manifest_sha256": case_manifest_sha256,
        "evaluation_metrics": [
            "e_deltaJ_1=abs(deltaJ_pred-deltaJ_truth)",
            "e_deltaJ_2=abs(deltaJ_pred_start_endpoint-deltaJ_truth_start_endpoint)",
            "e_deltaJ_3=abs(deltaJ_pred_start_endpoint-deltaJ_truth_start_endpoint)",
            "e_deltaJ_5=abs(deltaJ_pred_start_endpoint-deltaJ_truth_start_endpoint)",
            "mean",
            "median",
            "P90",
            "P95",
            "P99",
            "max",
        ],
        "truth_access_discipline": [
            "CASE_MANIFEST_PERSISTED",
            "LOCAL_AND_BUNDLE_PAIR_ASSIGNMENTS_PERSISTED",
            "THIS_MANIFEST_PERSISTED_AND_SHA256_VERIFIED",
            "ONLY_THEN_IDENTIFICATION_AND_ENDPOINT_TRUTH",
        ],
        "truth_may_change_case_or_pair_selection": False,
        "truth_may_update_policy": False,
        "threshold_selection_allowed": False,
        "prospective_personalization_allowed": False,
        "heldout_final_test_read_allowed": False,
        "reserved_for_future_prospective": False,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "old_prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "old_prospective_start_commit": PROSPECTIVE_START_COMMIT,
        "old_prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "protected_source_sha256": dict(protected_source_sha256),
        "P2_V2_default_enabled": False,
        "human_ready": NOT_HUMAN_READY,
        "robot_motion_approved": NOT_ROBOT_MOTION_APPROVED,
    }


__all__ = [
    "BUNDLE_SCALE_CALIBRATED",
    "BUNDLE_SCALE_INSUFFICIENT",
    "BUNDLE_SOURCE_PAIR_PLAN_PATH",
    "BUNDLE_SOURCE_PAIR_PLAN_SHA256",
    "BUNDLE_SOURCE_PROTOCOL_PATH",
    "CALIBRATION_CASE_COUNT",
    "CALIBRATION_DATA_ROLE",
    "CALIBRATION_ID",
    "CALIBRATION_MANIFEST_ID",
    "FUTURE_PROSPECTIVE_STATUS",
    "FrozenCalibrationManifestGate",
    "HELD_OUT_STATUS",
    "LOCAL_CALIBRATION_PLAN_ID",
    "LOCAL_SOURCE_PAIR_PLAN_PATH",
    "LOCAL_SOURCE_PAIR_PLAN_SHA256",
    "NOT_HUMAN_READY",
    "NOT_ROBOT_MOTION_APPROVED",
    "assign_pairs_to_calibration_cases",
    "bundle_scale_feasibility",
    "calibration_case_manifest",
    "calibration_manifest_payload",
    "calibration_subject_definitions",
    "dataframe_sha256",
    "evaluate_frozen_calibration_assignments",
    "grouped_residual_summary",
    "residual_distribution",
    "sha256_file",
]
