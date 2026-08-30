"""Generate the preregistered MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1.

The stage freezes a deterministic six-dimensional centered maximin Latin
hypercube, reconstructs 32 compact parameter-delta MyoLeg subjects, and runs
the frozen P0/V2 reference replay for every subject.  It never generates a
candidate landscape, fits a learner, runs BO, or accesses robot-facing code.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np


STAGE_ID = "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1"
PROTOCOL_ID = "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1"
OUTCOME = "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_VALID_WITH_LIMITATIONS"
COHORT_ID = "MYOLEG_VIRTUAL_PATIENT_COHORT_V1"
SCHEME_ID = "SCHEME_A_MINIMAL_INTERPRETABLE"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"
REFERENCE_ID = "NATIVE_ROM_REFERENCE_CANDIDATE"
SUBJECT_PREFIX = "MYOLEG_VP_"
NOMINAL_ID = "SUBJECT_NOMINAL_CONTROL"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_virtual_patient_cohort_generation_v1"
)
COHORT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation"
    / "cohorts"
    / "myoleg_virtual_patient_cohort_v1"
)
SUBJECT_DIRECTORY = COHORT_DIRECTORY / "subjects"
NOMINAL_DIRECTORY = COHORT_DIRECTORY / "nominal_control"

MODEL_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_supine_rehab_v1"
    / "myoleg_supine_right_v1.xml"
)
V2_REFERENCE_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)
TRUTH_SEMANTICS_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
    / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
)
FROZEN_NOMINAL_REPLAY_PATH = TRUTH_SEMANTICS_PATH.with_name(
    "SENSITIVITY_REFERENCE_REPLAY.npz"
)
FORMAL_REFERENCE_PATH = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST_PATH = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
COORDINATE_MAPPING_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
    / "PROJECT_MYOLEG_COORDINATE_MAPPING.json"
)

DESIGN_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_virtual_patient_cohort_design_v1"
)
RANGE_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_cohort_parameter_range_evidence_v1"
)
PROPOSAL_PATH = RANGE_ARTIFACT_DIRECTORY / "PROPOSED_PARAMETER_RANGES.json"
PROTOCOL_PATH = (
    RANGE_ARTIFACT_DIRECTORY
    / "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1.json"
)
DESIGN_BUILDER_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_virtual_patient_cohort_design_v1"
    / "build_design_audit.py"
)
REPLAY_BUILDER_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_reference_trajectory_replay_v1"
    / "build_and_replay.py"
)

FROZEN_SHA256 = {
    "base_myoleg_model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "frozen_nominal_replay": "ffb389f22ad586a1cbe3f18fbaca9a4bf2cc7964d94336bb5e398a38b64e6cde",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "coordinate_mapping": "83798958fd0b12f5c5314bc32df898f1d6e56d8e224f7673d8aca5c457ce713c",
    "design_checksums": "1f307917a08e64dd81e779a88838043cc81cc4dac9a347c1b6aacc3e4b5edfa6",
    "range_checksums": "492213ea82680efc2457c29b9bdfc82c3af58395347ae2b5bf3f7e69c74fa3c8",
    "proposal_file": "10135f17cb0780da586bf466c3ae53b1fb0ed64a86afea5af656c0173cbeb134",
    "protocol_file": "94abf249d19e7ec820ba6c24b6be665686b0434379b27c5c281620e10405fc48",
}

FROZEN_INPUT_PATHS = {
    "base_myoleg_model": MODEL_PATH,
    "v2_reference": V2_REFERENCE_PATH,
    "truth_semantics": TRUTH_SEMANTICS_PATH,
    "frozen_nominal_replay": FROZEN_NOMINAL_REPLAY_PATH,
    "formal_reference": FORMAL_REFERENCE_PATH,
    "formal_manifest": FORMAL_MANIFEST_PATH,
    "coordinate_mapping": COORDINATE_MAPPING_PATH,
    "design_checksums": DESIGN_ARTIFACT_DIRECTORY / "checksums.sha256",
    "range_checksums": RANGE_ARTIFACT_DIRECTORY / "checksums.sha256",
    "proposal_file": PROPOSAL_PATH,
    "protocol_file": PROTOCOL_PATH,
}

MODEL_FINGERPRINT_ARRAYS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_pos",
    "body_quat",
    "body_parentid",
    "body_jntadr",
    "body_jntnum",
    "jnt_type",
    "jnt_bodyid",
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "jnt_qposadr",
    "jnt_dofadr",
    "site_bodyid",
    "site_pos",
    "site_quat",
    "tendon_adr",
    "tendon_num",
    "tendon_range",
    "tendon_lengthspring",
    "wrap_type",
    "wrap_objid",
    "wrap_prm",
    "actuator_trntype",
    "actuator_trnid",
    "actuator_lengthrange",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_dynprm",
    "eq_type",
    "eq_obj1id",
    "eq_obj2id",
    "eq_data",
    "eq_solref",
    "eq_solimp",
)

STRUCTURAL_ARRAYS = tuple(
    name
    for name in MODEL_FINGERPRINT_ARRAYS
    if name not in {"body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checksum_manifest(path: Path) -> dict[str, str]:
    checked: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        candidate = path.parent / relative.strip()
        actual = sha256_file(candidate)
        if actual != expected:
            raise RuntimeError(f"frozen checksum failed: {candidate}")
        checked[relative.strip()] = actual
    return checked


def frozen_input_hashes() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path in FROZEN_INPUT_PATHS.items()}
    failures = {
        name: {"expected": FROZEN_SHA256[name], "actual": value}
        for name, value in actual.items()
        if value != FROZEN_SHA256[name]
    }
    if failures:
        raise RuntimeError(f"frozen input changed: {failures}")
    verify_checksum_manifest(DESIGN_ARTIFACT_DIRECTORY / "checksums.sha256")
    verify_checksum_manifest(RANGE_ARTIFACT_DIRECTORY / "checksums.sha256")
    formal = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["hip_rom_deg"] == [0.0, 120.0]
        and formal["knee_rom_deg"] == [5.0, 145.0]
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA256["formal_reference"]
    ):
        raise RuntimeError("formal ROM/reference convention changed")
    truth = json.loads(TRUTH_SEMANTICS_PATH.read_text(encoding="utf-8"))
    if truth["semantic_version"] != TRUTH_SEMANTIC_VERSION or truth["truth_field"] != TRUTH_FIELD:
        raise RuntimeError("truth semantics changed")
    return actual


def runtime_environment() -> dict[str, Any]:
    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "myosuite": importlib.metadata.version("myosuite"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    expected = {"python": "3.10.19", "myosuite": "2.12.2", "mujoco": "3.6.0", "numpy": "2.2.6"}
    result["frozen_expected"] = expected
    result["frozen_match"] = all(result[key] == value for key, value in expected.items())
    if not result["frozen_match"]:
        raise RuntimeError("frozen MyoLeg generation runtime changed")
    return result


def model_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or ""


def array_bundle_sha256(model: mujoco.MjModel, names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    dimensions = np.asarray(
        [model.nbody, model.njnt, model.nq, model.nv, model.ntendon, model.nu, model.nsite, model.neq],
        dtype=np.int64,
    )
    digest.update(dimensions.tobytes())
    for name in names:
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pairwise_values(matrix: np.ndarray) -> np.ndarray:
    delta = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.sum(delta**2, axis=2))
    return distances[np.triu_indices(len(matrix), k=1)]


def generate_centered_maximin_lhs(
    *, sample_count: int, dimension: int, restarts: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    best: np.ndarray | None = None
    best_score = -math.inf
    best_key: tuple[float, ...] | None = None
    best_restart = -1
    scores = []
    for restart in range(restarts):
        candidate = np.column_stack(
            [(rng.permutation(sample_count) + 0.5) / sample_count for _ in range(dimension)]
        )
        score = float(np.min(pairwise_values(candidate)))
        key = tuple(float(value) for value in candidate.ravel())
        scores.append(score)
        if score > best_score + 1.0e-15 or (
            abs(score - best_score) <= 1.0e-15 and (best_key is None or key < best_key)
        ):
            best = candidate.copy()
            best_score = score
            best_key = key
            best_restart = restart
    if best is None:
        raise RuntimeError("LHS generation produced no candidate")
    return best, {
        "selected_restart_zero_based": best_restart,
        "selected_min_pairwise_distance": best_score,
        "restart_score_min": float(np.min(scores)),
        "restart_score_median": float(np.median(scores)),
        "restart_score_max": float(np.max(scores)),
    }


def lhs_audit(matrix: np.ndarray) -> dict[str, Any]:
    sample_count, dimension = matrix.shape
    dimensions = []
    stratified = True
    for axis in range(dimension):
        strata = np.floor(matrix[:, axis] * sample_count).astype(int)
        axis_stratified = np.array_equal(np.sort(strata), np.arange(sample_count))
        stratified = stratified and axis_stratified
        dimensions.append(
            {
                "dimension_zero_based": axis,
                "min": float(np.min(matrix[:, axis])),
                "max": float(np.max(matrix[:, axis])),
                "mean": float(np.mean(matrix[:, axis])),
                "unique_strata": int(len(np.unique(strata))),
                "centered_stratification_pass": bool(axis_stratified),
            }
        )
    unique = np.unique(matrix, axis=0)
    distances = pairwise_values(matrix)
    return {
        "shape": [sample_count, dimension],
        "dimensions": dimensions,
        "all_dimensions_centered_stratified": bool(stratified),
        "duplicate_row_count": int(sample_count - len(unique)),
        "min_pairwise_distance": float(np.min(distances)),
        "median_pairwise_distance": float(np.median(distances)),
        "max_pairwise_distance": float(np.max(distances)),
        "boundary_behavior": "centered strata; theoretical unit-cube min/max are 0.5/N and 1-0.5/N; no exact 0 or 1",
        "all_inside_open_unit_cube": bool(np.all(matrix > 0.0) and np.all(matrix < 1.0)),
    }


def time_weighted_rms(time_s: np.ndarray, values: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s) / duration))


def replay_metrics(
    reference: dict[str, np.ndarray],
    prescribed: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
    controlled_runtime: dict[str, Any],
) -> dict[str, Any]:
    time_s = np.asarray(reference["time_s"], dtype=float)
    tau = np.asarray(prescribed["tau_truth_nm"], dtype=float)
    hip_rms = time_weighted_rms(time_s, tau[:, 0])
    knee_rms = time_weighted_rms(time_s, tau[:, 1])
    hip_rate = time_weighted_rms(time_s, np.gradient(tau[:, 0], time_s, edge_order=2))
    knee_rate = time_weighted_rms(time_s, np.gradient(tau[:, 1], time_s, edge_order=2))
    actuator = np.asarray(prescribed["actuator_internal_nm"], dtype=float)
    passive = np.asarray(prescribed["passive_internal_nm"], dtype=float)
    constraint = np.asarray(prescribed["constraint_internal_nm"], dtype=float)
    force = np.abs(np.asarray(prescribed["actuator_force_n"], dtype=float))
    q_error_deg = np.degrees(controlled["actual_q_rad"] - reference["q"])
    knee_deg = np.degrees(controlled["actual_q_rad"][:, 1])
    algebraic = max(
        float(np.max(np.abs(prescribed["inverse_formula_residual_nm"]))),
        float(np.max(np.abs(prescribed["decomposition_residual_nm"]))),
        float(np.max(np.abs(prescribed["muscle_reconstruction_residual_nm"]))),
    )
    arrays: Iterable[np.ndarray] = (
        tau,
        prescribed["actuator_force_n"],
        prescribed["tendon_length_m"],
        prescribed["mass_term_nm"],
        prescribed["bias_term_nm"],
        prescribed["passive_internal_nm"],
        prescribed["actuator_internal_nm"],
        prescribed["constraint_internal_nm"],
        controlled["actual_q_rad"],
        controlled["actual_dq_rad_s"],
        controlled["actual_ddq_rad_s2"],
    )
    return {
        "duration_s": float(time_s[-1]),
        "sample_count": len(time_s),
        "hip_tau_rms_nm": hip_rms,
        "knee_tau_rms_nm": knee_rms,
        "hip_tau_peak_abs_nm": float(np.max(np.abs(tau[:, 0]))),
        "knee_tau_peak_abs_nm": float(np.max(np.abs(tau[:, 1]))),
        "hip_tau_rate_rms_nm_s": hip_rate,
        "knee_tau_rate_rms_nm_s": knee_rate,
        "hip_actuator_internal_rms_nm": time_weighted_rms(time_s, actuator[:, 0]),
        "knee_actuator_internal_rms_nm": time_weighted_rms(time_s, actuator[:, 1]),
        "hip_passive_internal_rms_nm": time_weighted_rms(time_s, passive[:, 0]),
        "knee_passive_internal_rms_nm": time_weighted_rms(time_s, passive[:, 1]),
        "hip_constraint_internal_rms_nm": time_weighted_rms(time_s, constraint[:, 0]),
        "knee_constraint_internal_rms_nm": time_weighted_rms(time_s, constraint[:, 1]),
        "maximum_actuator_force_abs_n": float(np.max(force)),
        "source_equality_residual_max": max(
            float(np.max(np.abs(prescribed["source_equality_residual"]))),
            float(np.max(np.abs(controlled["source_equality_residual"]))),
        ),
        "algebraic_residual_max_nm": algebraic,
        "tracking_q_max_abs_deg": float(np.max(np.abs(q_error_deg))),
        "controlled_knee_min_deg": float(np.min(knee_deg)),
        "controlled_knee_max_deg": float(np.max(knee_deg)),
        "joint_limit_active_max": int(np.max(prescribed["constraint_joint_limit_active_count"])),
        "tendon_limit_active_max": int(np.max(prescribed["constraint_tendon_limit_active_count"])),
        "contact_active_max": int(np.max(prescribed["constraint_contact_active_count"])),
        "warning_count": max(
            int(np.max(prescribed["warning_count"])),
            int(np.max(controlled["warning_count"])),
            int(controlled_runtime["warning_count"]),
        ),
        "all_replay_arrays_finite": all(bool(np.isfinite(array).all()) for array in arrays),
        "muscle_states_finite": bool(np.isfinite(prescribed["actuator_force_n"]).all()),
        "tendon_states_finite": bool(np.isfinite(prescribed["tendon_length_m"]).all()),
    }


def exact_array_match(path: Path, payload: dict[str, np.ndarray]) -> dict[str, Any]:
    mismatches = []
    with np.load(path, allow_pickle=False) as frozen:
        missing = sorted(set(frozen.files).symmetric_difference(payload))
        for key in sorted(set(frozen.files).intersection(payload)):
            if not np.array_equal(frozen[key], payload[key]):
                mismatches.append(key)
    return {
        "exact": not missing and not mismatches,
        "missing_or_extra_keys": missing,
        "mismatched_keys": mismatches,
        "compared_array_count": len(payload),
    }


def apply_factor_profile(
    model: mujoco.MjModel,
    base: mujoco.MjModel,
    factor_order: list[str],
    factor_specs: dict[str, dict[str, Any]],
    factor_values: dict[str, float],
    structural: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    modifications: list[dict[str, Any]] = []
    group_members: dict[str, list[str]] = {}
    used_actuators: set[int] = set()
    for factor_id in factor_order:
        value = float(factor_values[factor_id])
        spec = factor_specs[factor_id]
        if spec["factor_type"] == "ANTHROPOMETRY":
            for body_name in spec["targets"]:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise RuntimeError(f"missing body {body_name}")
                before_mass = float(base.body_mass[body_id])
                before_inertia = np.asarray(base.body_inertia[body_id], dtype=float).copy()
                model.body_mass[body_id] = before_mass * value
                model.body_inertia[body_id] = before_inertia * value
                modifications.append(
                    {
                        "factor_id": factor_id,
                        "object_type": "body",
                        "object_name": body_name,
                        "fields": {
                            "body_mass": {"before": before_mass, "after": float(model.body_mass[body_id])},
                            "body_inertia": {"before": before_inertia.tolist(), "after": model.body_inertia[body_id].tolist()},
                        },
                        "scale": value,
                        "body_ipos_com_unchanged": bool(np.array_equal(model.body_ipos[body_id], base.body_ipos[body_id])),
                        "inertia_scaling_is_modeling_approximation": True,
                    }
                )
        elif spec["factor_type"] == "PASSIVE_FPMAX":
            group = spec["structural_group"]
            selected = [
                index
                for index in range(model.nu)
                if model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
                and structural[index]["structural_group"] == group
            ]
            if not selected or used_actuators.intersection(selected):
                raise RuntimeError(f"invalid/disjoint structural group {group}")
            used_actuators.update(selected)
            members = [model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in selected]
            group_members[group] = members
            for actuator, actuator_name in zip(selected, members):
                before_gain = float(base.actuator_gainprm[actuator, 7])
                before_bias = float(base.actuator_biasprm[actuator, 7])
                model.actuator_gainprm[actuator, 7] = before_gain * value
                model.actuator_biasprm[actuator, 7] = before_bias * value
                modifications.append(
                    {
                        "factor_id": factor_id,
                        "object_type": "actuator",
                        "object_name": actuator_name,
                        "structural_group": group,
                        "fields": {
                            "actuator_gainprm_7_fpmax": {"before": before_gain, "after": float(model.actuator_gainprm[actuator, 7])},
                            "actuator_biasprm_7_fpmax": {"before": before_bias, "after": float(model.actuator_biasprm[actuator, 7])},
                        },
                        "scale": value,
                        "group_basis": structural[actuator]["mapping_basis"],
                    }
                )
        else:
            raise RuntimeError(f"unrecognized frozen factor type {spec['factor_type']}")
    return modifications, group_members


def structural_integrity(
    base: mujoco.MjModel,
    subject: mujoco.MjModel,
    factor_order: list[str],
    factor_specs: dict[str, dict[str, Any]],
    factor_values: dict[str, float],
    structural: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dimensions = {
        "nbody": subject.nbody,
        "njnt": subject.njnt,
        "nq": subject.nq,
        "nv": subject.nv,
        "ntendon": subject.ntendon,
        "nu": subject.nu,
        "nsite": subject.nsite,
        "neq": subject.neq,
    }
    base_dimensions = {
        "nbody": base.nbody,
        "njnt": base.njnt,
        "nq": base.nq,
        "nv": base.nv,
        "ntendon": base.ntendon,
        "nu": base.nu,
        "nsite": base.nsite,
        "neq": base.neq,
    }
    structural_exact = {name: bool(np.array_equal(getattr(subject, name), getattr(base, name))) for name in STRUCTURAL_ARRAYS}
    expected_mass = np.asarray(base.body_mass).copy()
    expected_inertia = np.asarray(base.body_inertia).copy()
    expected_gain = np.asarray(base.actuator_gainprm).copy()
    expected_bias = np.asarray(base.actuator_biasprm).copy()
    for factor_id in factor_order:
        spec = factor_specs[factor_id]
        scale = float(factor_values[factor_id])
        if spec["factor_type"] == "ANTHROPOMETRY":
            for body_name in spec["targets"]:
                body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, body_name)
                expected_mass[body_id] *= scale
                expected_inertia[body_id] *= scale
        else:
            group = spec["structural_group"]
            selected = [
                index
                for index in range(base.nu)
                if model_name(base, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
                and structural[index]["structural_group"] == group
            ]
            expected_gain[selected, 7] *= scale
            expected_bias[selected, 7] *= scale
    rtb3 = mujoco.mj_name2id(subject, mujoco.mjtObj.mjOBJ_SITE, "RTB3")
    checks = {
        "same_dimensions": dimensions == base_dimensions,
        "same_80_muscles": subject.nu == base.nu == 80,
        "same_80_tendons": subject.ntendon == base.ntendon == 80,
        "all_structural_arrays_exact": all(structural_exact.values()),
        "only_expected_body_mass_changed": bool(np.array_equal(subject.body_mass, expected_mass)),
        "only_expected_body_inertia_changed": bool(np.array_equal(subject.body_inertia, expected_inertia)),
        "only_expected_actuator_gain_changed": bool(np.array_equal(subject.actuator_gainprm, expected_gain)),
        "only_expected_actuator_bias_changed": bool(np.array_equal(subject.actuator_biasprm, expected_bias)),
        "rtb3_site_present": rtb3 >= 0,
        "coordinate_mapping_sha_frozen": sha256_file(COORDINATE_MAPPING_PATH) == FROZEN_SHA256["coordinate_mapping"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "dimensions": dimensions,
        "structural_array_exact": structural_exact,
        "structural_fingerprint_sha256": array_bundle_sha256(subject, STRUCTURAL_ARRAYS),
        "model_fingerprint_sha256": array_bundle_sha256(subject, MODEL_FINGERPRINT_ARRAYS),
        "model_fingerprint_definition": list(MODEL_FINGERPRINT_ARRAYS),
    }


def add_integrity_gates(
    metrics: dict[str, Any],
    nominal: dict[str, Any],
    structure: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    force_ratio = float(metrics["maximum_actuator_force_abs_n"]) / float(nominal["maximum_actuator_force_abs_n"])
    gates = {
        "full_24s_401_samples": metrics["duration_s"] == 24.0 and metrics["sample_count"] == 401,
        "all_finite": bool(metrics["all_replay_arrays_finite"] and metrics["muscle_states_finite"] and metrics["tendon_states_finite"]),
        "no_solver_warning": metrics["warning_count"] == 0,
        "equality_residual": metrics["source_equality_residual_max"] <= float(thresholds["source_equality_residual_max"]),
        "algebraic_residual": metrics["algebraic_residual_max_nm"] <= float(thresholds["algebraic_residual_max_nm"]),
        "tracking": metrics["tracking_q_max_abs_deg"] <= float(thresholds["tracking_q_max_abs_deg"]),
        "native_knee_rom": metrics["controlled_knee_min_deg"] >= float(thresholds["native_knee_min_deg"]) - 1.0e-10
        and metrics["controlled_knee_max_deg"] <= float(thresholds["native_knee_max_deg"]) + 1.0e-10,
        "no_force_explosion": force_ratio <= float(thresholds["peak_force_ratio_vs_nominal_max"]),
        "no_new_contact_or_limit_mode": metrics["joint_limit_active_max"] <= nominal["joint_limit_active_max"]
        and metrics["tendon_limit_active_max"] <= nominal["tendon_limit_active_max"]
        and metrics["contact_active_max"] <= nominal["contact_active_max"],
        "structure": bool(structure["pass"]),
    }
    result = dict(metrics)
    result["peak_force_ratio_vs_nominal"] = force_ratio
    result["integrity_gate_results"] = gates
    result["subject_integrity"] = "PASS" if all(gates.values()) else "FAIL"
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def subset_coverage(matrix: np.ndarray, indices: list[int]) -> dict[str, Any]:
    subset = matrix[indices]
    distances = pairwise_values(subset)
    ranges = np.ptp(subset, axis=0)
    radii = np.sqrt(np.sum((subset - 0.5) ** 2, axis=1))
    return {
        "count": len(indices),
        "dimension_min": np.min(subset, axis=0).tolist(),
        "dimension_max": np.max(subset, axis=0).tolist(),
        "dimension_mean": np.mean(subset, axis=0).tolist(),
        "normalized_bounding_box_volume": float(np.prod(ranges)),
        "min_pairwise_distance": float(np.min(distances)),
        "median_pairwise_distance": float(np.median(distances)),
        "max_pairwise_distance": float(np.max(distances)),
        "centroid_radius_min": float(np.min(radii)),
        "centroid_radius_median": float(np.median(radii)),
        "centroid_radius_max": float(np.max(radii)),
        "extreme_profile_count_any_dimension_outside_0p1_0p9": int(np.sum(np.any((subset < 0.1) | (subset > 0.9), axis=1))),
        "central_profile_count_all_dimensions_inside_0p3_0p7": int(np.sum(np.all((subset >= 0.3) & (subset <= 0.7), axis=1))),
    }


def recursive_checksums(directory: Path, excluded: set[str]) -> list[str]:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded:
            lines.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    return lines


def main() -> None:
    started = time.perf_counter()
    if ARTIFACT_DIRECTORY.exists() or COHORT_DIRECTORY.exists():
        raise RuntimeError("generation outputs already exist; refusing replacement/overwrite")
    ARTIFACT_DIRECTORY.mkdir(parents=True)
    SUBJECT_DIRECTORY.mkdir(parents=True)
    NOMINAL_DIRECTORY.mkdir(parents=True)

    inputs_before = frozen_input_hashes()
    environment = runtime_environment()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID or protocol["scheme_id"] != SCHEME_ID:
        raise RuntimeError("frozen protocol identity changed")
    sampling = protocol["sampling"]
    if not (
        sampling["algorithm"] == "deterministic centered maximin Latin hypercube"
        and sampling["seed"] == 20260830
        and sampling["dimension"] == 6
        and sampling["heterogeneous_subject_count"] == 32
        and len(sampling["development_indices_zero_based"]) == 24
        and len(sampling["held_out_indices_zero_based"]) == 8
    ):
        raise RuntimeError("frozen sampling protocol changed")
    factor_order = list(protocol["factor_order"])
    proposal_specs = {item["factor_id"]: item for item in proposal["factors"]}
    if factor_order != list(proposal_specs) or len(factor_order) != 6:
        raise RuntimeError("frozen Scheme A factor order changed")
    for factor in factor_order:
        if protocol["ranges"][factor]["primary"] != proposal_specs[factor]["conservative"]:
            raise RuntimeError(f"primary range mismatch for {factor}")

    design = load_module(DESIGN_BUILDER_PATH, "frozen_cohort_design_builder_generation")
    replay = load_module(REPLAY_BUILDER_PATH, "frozen_myoleg_replay_builder_generation")
    base_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    reference = replay.load_reference(V2_REFERENCE_PATH, "MYOLEG_V2_PRIMARY")
    reference_audit = replay.reference_audit(reference, base_model)
    if not (
        reference_audit["duration_s"] == 24.0
        and reference_audit["sample_count"] == 401
        and abs(reference_audit["q_range_deg"]["knee"][1] - 119.5) <= 1.0e-10
    ):
        raise RuntimeError("frozen V2 reference changed")
    structural = design.structural_muscle_map(base_model, reference, replay)
    biarticular = tuple(
        model_name(base_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(base_model.nu)
        if model_name(base_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
        and structural[index]["structural_group"] == "HIP_KNEE_BIARTICULAR"
    )
    frozen_biarticular = ("bflh_r", "grac_r", "recfem_r", "sart_r", "semimem_r", "semiten_r", "tfl_r")
    if biarticular != frozen_biarticular:
        raise RuntimeError("frozen compiled-transmission biarticular group changed")

    matrix, search_audit = generate_centered_maximin_lhs(
        sample_count=32, dimension=6, restarts=512, seed=20260830
    )
    repeat_matrix, repeat_search_audit = generate_centered_maximin_lhs(
        sample_count=32, dimension=6, restarts=512, seed=20260830
    )
    lhs_deterministic = bool(
        np.array_equal(matrix, repeat_matrix) and search_audit == repeat_search_audit
    )
    if not lhs_deterministic:
        raise RuntimeError("centered maximin LHS is not deterministic")
    matrix_audit = lhs_audit(matrix)
    if not (
        matrix_audit["all_dimensions_centered_stratified"]
        and matrix_audit["duplicate_row_count"] == 0
        and matrix_audit["all_inside_open_unit_cube"]
    ):
        raise RuntimeError("generated LHS fails frozen design")

    lower = np.asarray([protocol["ranges"][factor]["primary"][0] for factor in factor_order])
    upper = np.asarray([protocol["ranges"][factor]["primary"][2] for factor in factor_order])
    transformed = lower[None, :] + matrix * (upper - lower)[None, :]
    if not (np.all(transformed >= lower[None, :]) and np.all(transformed <= upper[None, :])):
        raise RuntimeError("transformed parameters exceed primary ranges")
    if np.any(np.all(transformed == 1.0, axis=1)):
        raise RuntimeError("nominal control appeared inside heterogeneous LHS")

    development_indices = list(sampling["development_indices_zero_based"])
    held_out_indices = list(sampling["held_out_indices_zero_based"])
    if sorted(development_indices + held_out_indices) != list(range(32)) or set(development_indices).intersection(held_out_indices):
        raise RuntimeError("frozen split is not a disjoint complete 24/8 partition")
    subject_ids = [f"{SUBJECT_PREFIX}{index + 1:03d}" for index in range(32)]
    split_labels = ["DEVELOPMENT" if index in development_indices else "HELD_OUT" for index in range(32)]

    lhs_rows = []
    parameter_rows = []
    split_rows = []
    for index, subject_id in enumerate(subject_ids):
        lhs_row: dict[str, Any] = {"sample_index_zero_based": index, "subject_id": subject_id}
        parameter_row: dict[str, Any] = {"sample_index_zero_based": index, "subject_id": subject_id, "split": split_labels[index]}
        for axis, factor in enumerate(factor_order):
            lhs_row[f"u_{axis + 1}_{factor}"] = format(float(matrix[index, axis]), ".17g")
            parameter_row[factor] = format(float(transformed[index, axis]), ".17g")
        lhs_rows.append(lhs_row)
        parameter_rows.append(parameter_row)
        split_rows.append(
            {
                "sample_index_zero_based": index,
                "subject_id": subject_id,
                "split": split_labels[index],
                "assignment_rule": "HELD_OUT iff zero-based sample index modulo 4 equals 3; otherwise DEVELOPMENT",
                "frozen_before_replay": True,
            }
        )
    write_csv(ARTIFACT_DIRECTORY / "LHS_UNIT_CUBE_MATRIX.csv", lhs_rows)
    write_csv(ARTIFACT_DIRECTORY / "SUBJECT_PARAMETER_MATRIX.csv", parameter_rows)
    write_csv(ARTIFACT_DIRECTORY / "SUBJECT_SPLIT.csv", split_rows)
    sampling_freeze = {
        "manifest_id": "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_SAMPLING_FREEZE",
        "frozen_before_any_subject_replay_or_learner_outcome": True,
        "algorithm": sampling["algorithm"],
        "implementation": "repository-native NumPy PCG64 centered permutation LHS; no third-party LHS package",
        "numpy_version": np.__version__,
        "seed": 20260830,
        "seed_semantics": "one NumPy Generator(PCG64(seed)) consumed sequentially across 512 restarts and six dimension permutations per restart",
        "sample_count": 32,
        "dimension": 6,
        "restart_count": 512,
        "selection": "maximize minimum pairwise Euclidean distance in unit cube; lexicographically smaller flattened matrix breaks ties within 1e-15",
        "search_audit": search_audit,
        "lhs_audit": matrix_audit,
        "lhs_deterministic_repeat_exact": lhs_deterministic,
        "factor_order": factor_order,
        "primary_bounds": {factor: [float(lower[i]), float(upper[i])] for i, factor in enumerate(factor_order)},
        "development_indices_zero_based": development_indices,
        "held_out_indices_zero_based": held_out_indices,
        "files": {
            "LHS_UNIT_CUBE_MATRIX.csv": sha256_file(ARTIFACT_DIRECTORY / "LHS_UNIT_CUBE_MATRIX.csv"),
            "SUBJECT_PARAMETER_MATRIX.csv": sha256_file(ARTIFACT_DIRECTORY / "SUBJECT_PARAMETER_MATRIX.csv"),
            "SUBJECT_SPLIT.csv": sha256_file(ARTIFACT_DIRECTORY / "SUBJECT_SPLIT.csv"),
        },
        "sampling_used_torque_or_learner_outcomes": False,
        "replacement_sampling_allowed": False,
    }
    sampling_freeze["content_sha256"] = canonical_sha256(sampling_freeze)
    write_json(ARTIFACT_DIRECTORY / "SAMPLING_FREEZE_MANIFEST.json", sampling_freeze)
    sampling_freeze_file_sha = sha256_file(ARTIFACT_DIRECTORY / "SAMPLING_FREEZE_MANIFEST.json")

    # Nominal control is evaluated before subjects only to establish frozen
    # integrity baselines; it is not a cohort row and does not affect sampling.
    nominal_prescribed, nominal_prescribed_runtime = replay.prescribed_truth(base_model, reference)
    nominal_controlled, nominal_controlled_runtime = replay.controlled_replay(base_model, reference)
    nominal_payload = replay.dataset_payload(reference, nominal_prescribed, nominal_controlled)
    nominal_match = exact_array_match(FROZEN_NOMINAL_REPLAY_PATH, nominal_payload)
    if not nominal_match["exact"]:
        raise RuntimeError(f"nominal control differs from frozen V2 replay: {nominal_match}")
    np.savez_compressed(NOMINAL_DIRECTORY / "reference_replay_truth.npz", **nominal_payload)
    nominal_metrics = replay_metrics(reference, nominal_prescribed, nominal_controlled, nominal_controlled_runtime)
    nominal_structure = structural_integrity(
        base_model,
        base_model,
        factor_order,
        proposal_specs,
        {factor: 1.0 for factor in factor_order},
        structural,
    )
    nominal_integrity = add_integrity_gates(
        nominal_metrics, nominal_metrics, nominal_structure, protocol["integrity_gates"]
    )
    if nominal_integrity["subject_integrity"] != "PASS":
        raise RuntimeError("nominal control failed frozen integrity gates")
    nominal_metadata = {
        "subject_id": NOMINAL_ID,
        "cohort_member": False,
        "factor_values": {factor: 1.0 for factor in factor_order},
        "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
        "generated_model_fingerprint_sha256": nominal_structure["model_fingerprint_sha256"],
        "v2_reference_sha256": FROZEN_SHA256["v2_reference"],
        "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "exact_array_match_to_frozen_nominal": nominal_match,
        "reference_denominators": {
            "subject_reference_tau_hip_rms_nm": nominal_metrics["hip_tau_rms_nm"],
            "subject_reference_tau_knee_rms_nm": nominal_metrics["knee_tau_rms_nm"],
        },
        "subject_specific_reference_normalization": "PASS",
        "j_truth_reference": 1.0,
        "integrity": nominal_integrity,
        "reference_replay_truth_sha256": sha256_file(NOMINAL_DIRECTORY / "reference_replay_truth.npz"),
        "runtime": {
            "prescribed_s": nominal_prescribed_runtime["wall_time_s"],
            "controlled_s": nominal_controlled_runtime["wall_time_s"],
        },
    }
    write_json(NOMINAL_DIRECTORY / "metadata.json", nominal_metadata)

    subject_records: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    generation_times = []
    prescribed_times = []
    controlled_times = []
    no_replacement_sampling = True
    base_structural_fingerprint = array_bundle_sha256(base_model, STRUCTURAL_ARRAYS)

    for index, subject_id in enumerate(subject_ids):
        subject_started = time.perf_counter()
        subject_dir = SUBJECT_DIRECTORY / subject_id
        subject_dir.mkdir()
        factors = {factor: float(transformed[index, axis]) for axis, factor in enumerate(factor_order)}
        unit_vector = [float(value) for value in matrix[index]]
        subject_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        modifications, group_members = apply_factor_profile(
            subject_model, base_model, factor_order, proposal_specs, factors, structural
        )
        structure = structural_integrity(
            base_model, subject_model, factor_order, proposal_specs, factors, structural
        )
        if structure["structural_fingerprint_sha256"] != base_structural_fingerprint:
            structure["pass"] = False
            structure["checks"]["common_structural_fingerprint"] = False
        else:
            structure["checks"]["common_structural_fingerprint"] = True
        repeat_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        repeat_modifications, repeat_groups = apply_factor_profile(
            repeat_model, base_model, factor_order, proposal_specs, factors, structural
        )
        model_generation_deterministic = bool(
            structure["model_fingerprint_sha256"]
            == array_bundle_sha256(repeat_model, MODEL_FINGERPRINT_ARRAYS)
            and modifications == repeat_modifications
            and group_members == repeat_groups
        )
        if not model_generation_deterministic:
            structure["pass"] = False
            structure["checks"]["model_generation_deterministic"] = False
        else:
            structure["checks"]["model_generation_deterministic"] = True
        generation_time = time.perf_counter() - subject_started
        generation_times.append(generation_time)

        delta_payload = {
            "subject_id": subject_id,
            "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
            "cohort_protocol_sha256": FROZEN_SHA256["protocol_file"],
            "sampling_freeze_manifest_sha256": sampling_freeze_file_sha,
            "unit_cube_vector": unit_vector,
            "factor_order": factor_order,
            "factor_values": factors,
            "modifications": modifications,
            "compiled_transmission_group_members": group_members,
            "generated_model_fingerprint_sha256": structure["model_fingerprint_sha256"],
            "structural_fingerprint_sha256": structure["structural_fingerprint_sha256"],
            "reconstruction": "load frozen base XML, apply exact listed double-precision field deltas in factor_order, then verify model fingerprint",
            "compact_delta_used_instead_of_copied_mjb_or_assets": True,
            "inertia_scaling_is_modeling_approximation": True,
        }
        write_json(subject_dir / "model_delta.json", delta_payload)

        prescribed, prescribed_runtime = replay.prescribed_truth(subject_model, reference)
        controlled, controlled_runtime = replay.controlled_replay(subject_model, reference)
        prescribed_times.append(float(prescribed_runtime["wall_time_s"]))
        controlled_times.append(float(controlled_runtime["wall_time_s"]))
        payload = replay.dataset_payload(reference, prescribed, controlled)
        np.savez_compressed(subject_dir / "reference_replay_truth.npz", **payload)
        metrics = replay_metrics(reference, prescribed, controlled, controlled_runtime)
        integrity = add_integrity_gates(
            metrics, nominal_metrics, structure, protocol["integrity_gates"]
        )
        hip_denominator = float(metrics["hip_tau_rms_nm"])
        knee_denominator = float(metrics["knee_tau_rms_nm"])
        hip_ratio = hip_denominator / hip_denominator
        knee_ratio = knee_denominator / knee_denominator
        j_reference = float(math.sqrt((hip_ratio**2 + knee_ratio**2) / 2.0))
        normalization_pass = bool(
            math.isfinite(hip_denominator)
            and math.isfinite(knee_denominator)
            and hip_denominator > 0.0
            and knee_denominator > 0.0
            and abs(j_reference - 1.0) <= 1.0e-12
        )
        if not normalization_pass:
            integrity["subject_integrity"] = "FAIL"
            integrity["integrity_gate_results"]["subject_specific_reference_normalization"] = False
        else:
            integrity["integrity_gate_results"]["subject_specific_reference_normalization"] = True

        replay_sha = sha256_file(subject_dir / "reference_replay_truth.npz")
        delta_sha = sha256_file(subject_dir / "model_delta.json")
        metadata = {
            "subject_id": subject_id,
            "sample_index_zero_based": index,
            "split": split_labels[index],
            "claim_class": "heterogeneous musculoskeletal virtual subject",
            "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
            "cohort_protocol_sha256": FROZEN_SHA256["protocol_file"],
            "sampling_freeze_manifest_sha256": sampling_freeze_file_sha,
            "unit_cube_vector": unit_vector,
            "factor_order": factor_order,
            "factor_values": factors,
            "generated_model_fingerprint_sha256": structure["model_fingerprint_sha256"],
            "model_delta_sha256": delta_sha,
            "v2_reference_sha256": FROZEN_SHA256["v2_reference"],
            "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
            "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
            "truth_field": TRUTH_FIELD,
            "reference_denominators": {
                "subject_reference_tau_hip_rms_nm": hip_denominator,
                "subject_reference_tau_knee_rms_nm": knee_denominator,
            },
            "subject_specific_reference_normalization": "PASS" if normalization_pass else "FAIL",
            "j_truth_reference": j_reference,
            "integrity": integrity,
            "structure": structure,
            "reference_replay_truth_sha256": replay_sha,
            "runtime": {
                "model_generation_s": generation_time,
                "prescribed_replay_s": prescribed_runtime["wall_time_s"],
                "controlled_replay_s": controlled_runtime["wall_time_s"],
                "complete_reference_replay_s": prescribed_runtime["wall_time_s"] + controlled_runtime["wall_time_s"],
            },
            "replacement_sample": False,
        }
        write_json(subject_dir / "metadata.json", metadata)
        metadata_sha = sha256_file(subject_dir / "metadata.json")

        flat_integrity = {
            "subject_id": subject_id,
            "split": split_labels[index],
            "subject_integrity": integrity["subject_integrity"],
            "full_24s_401_samples": integrity["integrity_gate_results"]["full_24s_401_samples"],
            "all_finite": integrity["integrity_gate_results"]["all_finite"],
            "no_solver_warning": integrity["integrity_gate_results"]["no_solver_warning"],
            "equality_residual_pass": integrity["integrity_gate_results"]["equality_residual"],
            "algebraic_residual_pass": integrity["integrity_gate_results"]["algebraic_residual"],
            "tracking_pass": integrity["integrity_gate_results"]["tracking"],
            "native_knee_rom_pass": integrity["integrity_gate_results"]["native_knee_rom"],
            "no_force_explosion": integrity["integrity_gate_results"]["no_force_explosion"],
            "no_new_contact_or_limit_mode": integrity["integrity_gate_results"]["no_new_contact_or_limit_mode"],
            "structure_pass": integrity["integrity_gate_results"]["structure"],
            "normalization_pass": normalization_pass,
            "warning_count": metrics["warning_count"],
            "source_equality_residual_max": metrics["source_equality_residual_max"],
            "algebraic_residual_max_nm": metrics["algebraic_residual_max_nm"],
            "tracking_q_max_abs_deg": metrics["tracking_q_max_abs_deg"],
            "peak_force_ratio_vs_nominal": integrity["peak_force_ratio_vs_nominal"],
            "replacement_sample": False,
        }
        integrity_rows.append(flat_integrity)
        normalization_rows.append(
            {
                "subject_id": subject_id,
                "split": split_labels[index],
                "subject_reference_tau_hip_rms_nm": hip_denominator,
                "subject_reference_tau_knee_rms_nm": knee_denominator,
                "hip_reference_ratio": hip_ratio,
                "knee_reference_ratio": knee_ratio,
                "j_truth_reference": j_reference,
                "subject_specific_reference_normalization": "PASS" if normalization_pass else "FAIL",
                "nominal_denominator_used": False,
                "objective_formula": "sqrt(((hip_rms/subject_hip_reference_rms)^2 + (knee_rms/subject_knee_reference_rms)^2)/2)",
            }
        )
        response_rows.append({"subject_id": subject_id, "split": split_labels[index], **{key: metrics[key] for key in (
            "hip_tau_rms_nm",
            "knee_tau_rms_nm",
            "hip_tau_peak_abs_nm",
            "knee_tau_peak_abs_nm",
            "hip_actuator_internal_rms_nm",
            "knee_actuator_internal_rms_nm",
            "hip_passive_internal_rms_nm",
            "knee_passive_internal_rms_nm",
        )}})
        subject_records.append(
            {
                "subject_id": subject_id,
                "sample_index_zero_based": index,
                "split": split_labels[index],
                "unit_cube_vector": unit_vector,
                "factor_values": factors,
                "generated_model_fingerprint_sha256": structure["model_fingerprint_sha256"],
                "model_delta_path": str((subject_dir / "model_delta.json").relative_to(PROJECT_ROOT)),
                "model_delta_sha256": delta_sha,
                "metadata_path": str((subject_dir / "metadata.json").relative_to(PROJECT_ROOT)),
                "metadata_sha256": metadata_sha,
                "reference_replay_truth_path": str((subject_dir / "reference_replay_truth.npz").relative_to(PROJECT_ROOT)),
                "reference_replay_truth_sha256": replay_sha,
                "subject_reference_tau_hip_rms_nm": hip_denominator,
                "subject_reference_tau_knee_rms_nm": knee_denominator,
                "j_truth_reference": j_reference,
                "subject_integrity": integrity["subject_integrity"],
                "replacement_sample": False,
            }
        )

    write_csv(ARTIFACT_DIRECTORY / "SUBJECT_INTEGRITY_RESULTS.csv", integrity_rows)
    write_csv(ARTIFACT_DIRECTORY / "SUBJECT_REFERENCE_NORMALIZATION.csv", normalization_rows)

    failed_subjects = [row["subject_id"] for row in integrity_rows if row["subject_integrity"] != "PASS"]
    if failed_subjects:
        no_replacement_sampling = True
        raise RuntimeError(f"COHORT_GENERATION_BLOCKED; no replacement sampled; failed={failed_subjects}")

    response_metrics = [key for key in response_rows[0] if key not in {"subject_id", "split"}]
    response_summary_rows = []
    for metric in response_metrics:
        values = np.asarray([float(row[metric]) for row in response_rows])
        minimum = int(np.argmin(values))
        maximum = int(np.argmax(values))
        mean = float(np.mean(values))
        response_summary_rows.append(
            {
                "metric": metric,
                "min": float(np.min(values)),
                "min_subject_id": subject_ids[minimum],
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "max_subject_id": subject_ids[maximum],
                "mean": mean,
                "sd_population": float(np.std(values, ddof=0)),
                "cv_percent": float(100.0 * np.std(values, ddof=0) / abs(mean)),
                "purpose": "descriptive heterogeneity only; no subject filtering",
            }
        )
    write_csv(ARTIFACT_DIRECTORY / "COHORT_REFERENCE_RESPONSE_SUMMARY.csv", response_summary_rows)

    diagnostic_rows = []
    diagnostic_outputs = (
        "hip_tau_rms_nm",
        "knee_tau_rms_nm",
        "hip_tau_peak_abs_nm",
        "knee_tau_peak_abs_nm",
    )
    for axis, factor in enumerate(factor_order):
        x = transformed[:, axis]
        for output in diagnostic_outputs:
            y = np.asarray([float(row[output]) for row in response_rows])
            pearson, spearman = correlation(x, y)
            diagnostic_rows.append(
                {
                    "factor_id": factor,
                    "response": output,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "sample_count": 32,
                    "analysis_role": "POST_FREEZE_DESCRIPTIVE_ONLY",
                    "used_to_modify_ranges_sampling_or_subjects": False,
                }
            )
    write_csv(ARTIFACT_DIRECTORY / "COHORT_PARAMETER_RESPONSE_DIAGNOSTICS.csv", diagnostic_rows)

    coverage = {
        "audit_id": "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_PARAMETER_SPACE_COVERAGE",
        "sampling_freeze_manifest_sha256": sampling_freeze_file_sha,
        "overall_lhs": matrix_audit,
        "development": subset_coverage(matrix, development_indices),
        "held_out": subset_coverage(matrix, held_out_indices),
        "split_rule": "protocol-frozen index modulo rule; independent of torque, learner or future landscape",
        "held_out_all_extreme": bool(
            subset_coverage(matrix, held_out_indices)["extreme_profile_count_any_dimension_outside_0p1_0p9"] == 8
        ),
        "held_out_all_central": bool(
            subset_coverage(matrix, held_out_indices)["central_profile_count_all_dimensions_inside_0p3_0p7"] == 8
        ),
        "convex_hull_not_used": "small split and no need to introduce SciPy/Qhull version dependence; normalized bounding-box and pairwise metrics retained",
        "learner_performance_compared": False,
    }
    write_json(ARTIFACT_DIRECTORY / "COHORT_COVERAGE_AUDIT.json", coverage)

    mean_generation = float(np.mean(generation_times))
    mean_prescribed = float(np.mean(prescribed_times))
    mean_controlled = float(np.mean(controlled_times))
    mean_complete = mean_prescribed + mean_controlled
    runtime = {
        "runtime_id": "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_RUNTIME",
        "subject_count": 32,
        "model_generation_s": {
            "mean": mean_generation,
            "median": float(np.median(generation_times)),
            "min": float(np.min(generation_times)),
            "max": float(np.max(generation_times)),
            "total": float(np.sum(generation_times)),
        },
        "per_subject_reference_replay_s": {
            "prescribed_mean": mean_prescribed,
            "controlled_mean": mean_controlled,
            "complete_mean": mean_complete,
            "complete_median": float(np.median(np.asarray(prescribed_times) + np.asarray(controlled_times))),
        },
        "cohort_reference_replay_total_s": float(np.sum(prescribed_times) + np.sum(controlled_times)),
        "engineering_estimates": {
            "basis": "mean per-subject frozen complete prescribed+controlled reference replay; conservative and hardware-specific",
            "per_subject_100_candidates_s": 100.0 * mean_complete,
            "whole_32_subject_cohort_100_candidates_s": 32.0 * 100.0 * mean_complete,
            "per_subject_1000_candidates_s": 1000.0 * mean_complete,
            "whole_32_subject_cohort_1000_candidates_s": 32.0 * 1000.0 * mean_complete,
            "full_candidate_domain_status": "NOT_FROZEN_UNTIL_MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1",
            "formula_for_domain_size_n": "32 * N_domain * mean_complete_reference_replay_s",
            "illustrative_only_if_N_domain_21025_s": 32.0 * 21025.0 * mean_complete,
            "illustrative_21025_is_frozen_myoleg_domain": False,
        },
        "candidate_landscape_executed": False,
    }
    write_json(ARTIFACT_DIRECTORY / "RUNTIME_BENCHMARK.json", runtime)

    cohort_readme = f"""# {COHORT_ID}

This directory contains 32 frozen **heterogeneous musculoskeletal virtual
subjects** and one separate nominal control.  It is not a representative
patient cohort or physiological population sample.

Each subject is stored compactly as:

- `model_delta.json`: exact double-precision Scheme-A field changes relative to
  frozen base model SHA `{FROZEN_SHA256['base_myoleg_model']}`;
- `metadata.json`: identity, split, hashes, denominators and integrity result;
- `reference_replay_truth.npz`: complete frozen P0/V2 prescribed and controlled
  replay arrays.

No copied MJB or upstream meshes are retained.  Reconstruct by loading the
frozen base XML, applying `model_delta.json` in factor order, and verifying the
compiled model fingerprint.  Do not resample or replace subjects.
"""
    (COHORT_DIRECTORY / "README.md").write_text(cohort_readme, encoding="utf-8")
    cohort_checksum_lines = recursive_checksums(COHORT_DIRECTORY, {"checksums.sha256"})
    (COHORT_DIRECTORY / "checksums.sha256").write_text(
        "\n".join(cohort_checksum_lines) + "\n", encoding="utf-8"
    )
    cohort_checksums_sha = sha256_file(COHORT_DIRECTORY / "checksums.sha256")

    manifest = {
        "manifest_id": "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST",
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "claim_boundary": "heterogeneous musculoskeletal virtual subjects; not representative patients or a physiological population sample",
        "scheme_id": SCHEME_ID,
        "cohort_size": 32,
        "development_count": 24,
        "held_out_count": 8,
        "nominal_control_counted_in_cohort": False,
        "factor_order": factor_order,
        "primary_ranges": {factor: protocol["ranges"][factor]["primary"] for factor in factor_order},
        "extended_ranges_used": False,
        "sampling_freeze_manifest_path": str((ARTIFACT_DIRECTORY / "SAMPLING_FREEZE_MANIFEST.json").relative_to(PROJECT_ROOT)),
        "sampling_freeze_manifest_sha256": sampling_freeze_file_sha,
        "protocol_path": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "protocol_sha256": FROZEN_SHA256["protocol_file"],
        "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
        "v2_reference_sha256": FROZEN_SHA256["v2_reference"],
        "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "development_subject_ids": [subject_ids[index] for index in development_indices],
        "held_out_subject_ids": [subject_ids[index] for index in held_out_indices],
        "subjects": subject_records,
        "nominal_control": {
            "subject_id": NOMINAL_ID,
            "cohort_member": False,
            "metadata_path": str((NOMINAL_DIRECTORY / "metadata.json").relative_to(PROJECT_ROOT)),
            "metadata_sha256": sha256_file(NOMINAL_DIRECTORY / "metadata.json"),
            "reference_replay_truth_path": str((NOMINAL_DIRECTORY / "reference_replay_truth.npz").relative_to(PROJECT_ROOT)),
            "reference_replay_truth_sha256": sha256_file(NOMINAL_DIRECTORY / "reference_replay_truth.npz"),
            "exact_array_match_to_frozen_nominal": True,
        },
        "all_subject_integrity_pass": True,
        "all_subject_specific_reference_normalization_pass": True,
        "replacement_sampling_used": not no_replacement_sampling,
        "cohort_directory_checksums_sha256": cohort_checksums_sha,
        "candidate_landscape_generated": False,
        "learner_performance_revealed": False,
    }
    write_json(ARTIFACT_DIRECTORY / "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json", manifest)
    manifest_sha = sha256_file(ARTIFACT_DIRECTORY / "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json")

    response_summary = {row["metric"]: row for row in response_summary_rows}
    dev_ids = manifest["development_subject_ids"]
    held_ids = manifest["held_out_subject_ids"]
    report = f"""# {STAGE_ID}

## Final outcome

`{OUTCOME}`

Exactly 32 preregistered **heterogeneous musculoskeletal virtual subjects**
were generated from the frozen six-dimensional Scheme-A primary ranges.  One
separate nominal MyoLeg control is retained and is not counted in the 32.

## Q1 - Exactly 32 frozen subjects?

Yes.  The unit-cube and transformed matrices were frozen before any subject
replay.  The centered maximin LHS used NumPy `{np.__version__}` PCG64 seed
`20260830`, 512 permutation restarts, selected restart
`{search_audit['selected_restart_zero_based']}`, and minimum normalized
pairwise distance `{search_audit['selected_min_pairwise_distance']:.9f}`.  No
duplicate or out-of-range row exists and no extended stress range was used.

## Q2 - Frozen split

- Development (24): `{', '.join(dev_ids)}`
- Held-out (8): `{', '.join(held_ids)}`

This is the protocol-frozen zero-based index-modulo assignment.  It used no
torque, learner, landscape or difficulty outcome.

## Q3 - Integrity and replacement

All 32 subjects passed all preregistered gates.  There was no replacement
sampling.  Every replay completed 24 s / 401 samples with finite states, zero
solver warnings, retained 80 muscles and 80 tendons, exact structural arrays,
unchanged knee/patella equalities, tendon paths, sites, joint axes/ranges,
actuator length ranges, RTB3 and coordinate mapping.  Only the listed Scheme-A
mass/inertia and group `fpmax` fields changed.

## Q4 - Reference-response heterogeneity

| response | min | median | max | CV |
|---|---:|---:|---:|---:|
| hip tau RMS (N m) | {float(response_summary['hip_tau_rms_nm']['min']):.6f} | {float(response_summary['hip_tau_rms_nm']['median']):.6f} | {float(response_summary['hip_tau_rms_nm']['max']):.6f} | {float(response_summary['hip_tau_rms_nm']['cv_percent']):.3f}% |
| knee tau RMS (N m) | {float(response_summary['knee_tau_rms_nm']['min']):.6f} | {float(response_summary['knee_tau_rms_nm']['median']):.6f} | {float(response_summary['knee_tau_rms_nm']['max']):.6f} | {float(response_summary['knee_tau_rms_nm']['cv_percent']):.3f}% |
| hip peak (N m) | {float(response_summary['hip_tau_peak_abs_nm']['min']):.6f} | {float(response_summary['hip_tau_peak_abs_nm']['median']):.6f} | {float(response_summary['hip_tau_peak_abs_nm']['max']):.6f} | {float(response_summary['hip_tau_peak_abs_nm']['cv_percent']):.3f}% |
| knee peak (N m) | {float(response_summary['knee_tau_peak_abs_nm']['min']):.6f} | {float(response_summary['knee_tau_peak_abs_nm']['median']):.6f} | {float(response_summary['knee_tau_peak_abs_nm']['max']):.6f} | {float(response_summary['knee_tau_peak_abs_nm']['cv_percent']):.3f}% |

This is descriptive model-response heterogeneity only; no subject was selected,
removed or reweighted from these values.

## Q5 - Subject-specific normalization

All 32 subjects have positive finite subject-specific hip/knee reference RMS
denominators.  The frozen objective
`sqrt(((hip/ref_hip)^2 + (knee/ref_knee)^2)/2)` gives
`J_truth(reference)=1` within `1e-12` for every subject.  No nominal denominator
was used.

## Q6 - Nominal control

The nominal control matches all `{nominal_match['compared_array_count']}` arrays
in the previously frozen native-V2 nominal replay exactly (`np.array_equal`).

## Q7 - Permanent identities

The literal unit-cube matrix, transformed matrix, exact 24/8 split, compact
model deltas, compiled-model fingerprints, per-subject replay arrays and file
checksums are retained.  Final cohort manifest SHA-256:
`{manifest_sha}`.

## Q8 - Remaining scientific limitation

Mass marginals are anthropometric/model-motivated, but proportional inertia
scaling with fixed COM/geometry remains a modeling approximation.  All three
`fpmax` factors are conservative structured synthetic heterogeneity rather than
population-derived passive mechanics.  This cohort cannot be called a patient
sample, physiological distribution, representative population or validated
digital-twin cohort.

## Q9 - Runtime

Mean model generation was `{mean_generation:.6f} s`; mean prescribed replay
`{mean_prescribed:.6f} s`; mean controlled replay `{mean_controlled:.6f} s`;
mean complete reference replay `{mean_complete:.6f} s` per subject.  The 32
reference replays took `{runtime['cohort_reference_replay_total_s']:.3f} s`
inside the replay routines.  Candidate-domain totals remain engineering
formulas/illustrations only; no landscape was run.

## Q10 - Candidate-domain readiness

Yes, with the above synthetic limitations and mandatory manifest identity.  The
next allowed design stage is `MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1`.  This stage
did not generate a landscape or train Five-parameter, NN, PINN or BO models.

## Frozen boundary

- formal reference, ROM_PROTOCOL_V2 and `theta_shank = q_hip - q_knee`: unchanged
- V2 119.5-degree reference, base MyoLeg and truth semantics V1: unchanged
- no candidate landscape, Five-parameter fit, NN/PINN, BO or robot/hardware
- `INERTIA_SCALING_IS_MODELING_APPROXIMATION = true`
- no outcome-based removal or replacement sampling
"""
    (ARTIFACT_DIRECTORY / "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    inputs_after = frozen_input_hashes()
    if inputs_before != inputs_after:
        raise RuntimeError("a frozen source changed during cohort generation")
    artifact_size = sum(path.stat().st_size for path in ARTIFACT_DIRECTORY.rglob("*") if path.is_file())
    cohort_size = sum(path.stat().st_size for path in COHORT_DIRECTORY.rglob("*") if path.is_file())
    runtime["artifact_size_bytes_before_metadata_and_checksums"] = artifact_size
    runtime["cohort_directory_size_bytes"] = cohort_size
    write_json(ARTIFACT_DIRECTORY / "RUNTIME_BENCHMARK.json", runtime)

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(ARTIFACT_DIRECTORY.iterdir())
        if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    }
    metadata = {
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "evidence_level": "OFFLINE_SYNTHETIC_MUSCULOSKELETAL_COHORT_GENERATION_AND_P0_V2_REPLAY",
        "claim_boundary": "heterogeneous musculoskeletal virtual subjects",
        "runtime_environment": environment,
        "runtime_s": time.perf_counter() - started,
        "cohort_size": 32,
        "development_count": 24,
        "held_out_count": 8,
        "nominal_control_counted_in_cohort": False,
        "manifest_sha256": manifest_sha,
        "sampling_freeze_manifest_sha256": sampling_freeze_file_sha,
        "cohort_checksums_sha256": cohort_checksums_sha,
        "all_subject_integrity_pass": True,
        "all_subject_normalization_pass": True,
        "nominal_exact_array_match": True,
        "lhs_deterministic_repeat_exact": lhs_deterministic,
        "duplicate_subject_count": matrix_audit["duplicate_row_count"],
        "replacement_sampling_used": False,
        "input_sha256_before": inputs_before,
        "input_sha256_after": inputs_after,
        "builder_script_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "artifact_sha256": artifact_hashes,
        "cohort_generated": True,
        "subject_instances_generated": 32,
        "candidate_landscape_generated": False,
        "five_parameter_fit": False,
        "nn_trained": False,
        "pinn_trained": False,
        "bo_run": False,
        "robot_connected": False,
        "hardware_accessed": False,
        "control_modified": False,
        "safety_modified": False,
        "formal_reference_modified": False,
        "v2_reference_modified": False,
        "rom_protocol_modified": False,
        "truth_semantics_modified": False,
        "next_stage_executed": False,
    }
    write_json(ARTIFACT_DIRECTORY / "metadata.json", metadata)
    audit_checksum_lines = recursive_checksums(ARTIFACT_DIRECTORY, {"checksums.sha256"})
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text(
        "\n".join(audit_checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage_id": STAGE_ID,
                "outcome": OUTCOME,
                "subject_count": 32,
                "development": 24,
                "held_out": 8,
                "failed_subjects": failed_subjects,
                "replacement_sampling": False,
                "manifest_sha256": manifest_sha,
                "runtime_s": metadata["runtime_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
