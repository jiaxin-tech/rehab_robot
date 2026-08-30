"""Offline MyoLeg knee-ROM compatibility audit for the frozen active reference.

This script depends on the frozen project-owned supine derived model.  It never
edits MyoSuite, the native supine XML, or the formal reference.  The primary
125-degree extension and all decision thresholds are written before any model
results are evaluated.
"""

from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAGE_ID = "MYOLEG_KNEE_ROM_COMPATIBILITY_AUDIT_V1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
)
FIGURE_DIRECTORY = ARTIFACT_DIRECTORY / "figures"
PRIOR_STAGE_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
)
PRIOR_MANIFEST_PATH = PRIOR_STAGE_DIRECTORY / "DERIVED_MODEL_MANIFEST.json"
NATIVE_DERIVED_XML = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_supine_rehab_v1"
    / "myoleg_supine_right_v1.xml"
)
PRIMARY_125_XML = DERIVED_DIRECTORY / "myoleg_supine_right_knee125_v1.xml"
STRESS_130_XML = DERIVED_DIRECTORY / "myoleg_supine_right_knee130_stress_only_v1.xml"
REFERENCE_PATH = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST_PATH = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"

REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
NATIVE_DERIVED_SHA256 = "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d"
TARGET_HIP = "hip_flexion_r"
TARGET_KNEE = "knee_angle_r"
SOURCE_KNEE_EQUALITY_COUNT = 14
PRIMARY_LIMIT_DEG = 125.0
STRESS_LIMIT_DEG = 130.0
NATIVE_REFERENCE_TARGET_DEG = 119.5
REFERENCE_MAX_DEG = 124.78660371882795
HIP_GRID_DEG = (30.0, 60.0, 90.0, 110.0)
KNEE_PRIMARY_GRID_DEG = (
    100.0,
    105.0,
    110.0,
    115.0,
    118.0,
    120.0,
    121.0,
    122.0,
    123.0,
    124.0,
    REFERENCE_MAX_DEG,
    125.0,
)
KNEE_STRESS_EXTRA_DEG = (126.0, 128.0, 130.0)
VELOCITY_GRID_DEG_S = (0.0, 5.0, -5.0)
CONTINUITY_GRID_DEG = tuple(np.linspace(115.0, 125.0, 41).tolist())
MOMENT_ARM_EPS_RAD = 1.0e-5
SWEEP_HALF_DURATION_S = 10.0
SWEEP_DT_S = 0.001
SWEEP_RECORD_STRIDE = 20
SWEEP_HIP_GRID_DEG = (30.0, 60.0, 90.0, 110.0)
SWEEP_KP = 5000.0
SWEEP_KD = 150.0
SWEEP_TORQUE_LIMIT_NM = 3000.0
L1_PROJECT_M = 0.42
L2_PROJECT_M = 0.30
KEY_MUSCLES = (
    "recfem_r",
    "bflh_r",
    "semimem_r",
    "gasmed_r",
    "vaslat_r",
)

THRESHOLDS = {
    "common_domain_abs_tolerance": 1.0e-10,
    "geometry_crossing_derivative_jump_ratio_max": 0.25,
    "moment_arm_crossing_derivative_jump_ratio_max": 0.50,
    "force_crossing_derivative_jump_ratio_max": 1.00,
    "passive_torque_crossing_derivative_jump_ratio_max": 1.00,
    "extended_to_native_force_growth_ratio_max": 5.0,
    "force_to_model_fmax_ratio_max": 10.0,
    "normalized_actuator_length_min": -0.25,
    "normalized_actuator_length_max": 1.25,
    "source_equality_residual_max": 1.0e-3,
    "sweep_auxiliary_hysteresis_max": 5.0e-3,
    "sweep_tracking_error_max_deg": 2.0,
    "state_length_positive_min_m": 0.0,
}

PROTOCOL = {
    "stage_id": STAGE_ID,
    "protocol_version": "ROM_EXTENSION_PROTOCOL_V1",
    "evidence_level": "OFFLINE_HEADLESS_DIAGNOSTIC_ONLY",
    "primary_extension_deg": PRIMARY_LIMIT_DEG,
    "primary_extension_selection": (
        "preselected minimum practical integer upper bound above frozen reference maximum"
    ),
    "native_baseline": "actual knee_angle_r native range upper bound",
    "stress_only_extension_deg": STRESS_LIMIT_DEG,
    "stress_only_not_formal_reference_eligible": True,
    "project_full_145_domain_tested": False,
    "hip_grid_deg": list(HIP_GRID_DEG),
    "knee_primary_grid_deg": list(KNEE_PRIMARY_GRID_DEG),
    "knee_stress_extra_deg": list(KNEE_STRESS_EXTRA_DEG),
    "velocity_grid_deg_s": list(VELOCITY_GRID_DEG_S),
    "continuity_grid_deg": list(CONTINUITY_GRID_DEG),
    "low_control_sweep": {
        "hip_grid_deg": list(SWEEP_HIP_GRID_DEG),
        "knee_path_deg": [100.0, 125.0, 100.0],
        "half_duration_s": SWEEP_HALF_DURATION_S,
        "maximum_minimum_jerk_speed_deg_s": 4.6875,
        "muscle_control": 0.0,
        "driver": "qfrc_applied generalized PD torque, separate from muscle actuator force",
    },
    "representative_key_muscles_preselected": list(KEY_MUSCLES),
    "native_compatible_reference": {
        "candidate_id": "MYOLEG_NATIVE_ROM_REFERENCE_CANDIDATE",
        "target_peak_knee_deg": NATIVE_REFERENCE_TARGET_DEG,
        "anchor": "original first-sample knee angle",
        "formula": "q_new(t)=q0+s*(q_original(t)-q0)",
        "scale_formula": "s=(119.5-q0)/(max(q_original)-q0)",
        "derivative_formula": "dq_new=s*dq_original; ddq_new=s*ddq_original",
        "pointwise_clipping": False,
        "selection_uses_audit_results": False,
    },
    "continuity_metrics": {
        "crossing": "left finite-difference slope at 119.75-120 vs right slope at 120-120.25 deg",
        "normalization": "max P95 absolute derivative across 115-125 and a predeclared signal floor",
        "native_segment": "115-120 deg",
        "extension_segment": "120-125 deg",
    },
    "thresholds_frozen_before_results": THRESHOLDS,
    "normalized_length_and_force_threshold_scope": (
        "right-side actuators with nonzero target hip or knee moment arm; all-model values retained as context only"
    ),
    "normalized_muscle_fiber_length_available": False,
    "actuator_lengthrange_normalized_coordinate_role": (
        "context-only transmission-range coordinate; it is not physiological normalized muscle fiber length and is not a hard validity gate"
    ),
    "implementation_scope_correction_before_formal_freeze": (
        "non-retained dry runs exposed two implementation semantics issues: all-model normalization included locked contralateral/distal actuators, and actuator_lengthrange was not physiological normalized fiber length; numeric continuity/force/equality thresholds, extension limits, and reference transformation were unchanged"
    ),
    "forbidden": {
        "pointwise_clipping": True,
        "formal_reference_modification": True,
        "upstream_modification": True,
        "145_degree_test": True,
        "candidate_landscape": True,
        "bo": True,
        "pinn": True,
        "rl": True,
        "robot_connection": True,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def name(model: mujoco.MjModel, objtype: Any, identifier: int) -> str:
    return mujoco.mj_id2name(model, objtype, int(identifier)) or ""


def jid(model: mujoco.MjModel, joint_name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if value < 0:
        raise RuntimeError(f"missing joint {joint_name}")
    return int(value)


def qadr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[jid(model, joint_name)])


def dadr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_dofadr[jid(model, joint_name)])


def aid(model: mujoco.MjModel, actuator_name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if value < 0:
        raise RuntimeError(f"missing actuator {actuator_name}")
    return int(value)


def object_dimensions(model: mujoco.MjModel) -> dict[str, int]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "neq": int(model.neq),
        "nu": int(model.nu),
        "ntendon": int(model.ntendon),
    }


def warning_count(data: mujoco.MjData) -> int:
    return int(np.asarray(data.warning.number, dtype=np.int64).sum())


def finite_data(data: mujoco.MjData) -> bool:
    arrays = (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qfrc_passive,
        data.qfrc_actuator,
        data.qfrc_constraint,
        data.actuator_length,
        data.actuator_velocity,
        data.actuator_force,
        data.ten_length,
    )
    return all(bool(np.isfinite(np.asarray(value)).all()) for value in arrays)


def verify_checksum_manifest(path: Path) -> dict[str, Any]:
    failures = []
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        candidate = path.parent / relative.strip()
        checked += 1
        if not candidate.is_file():
            failures.append({"file": relative.strip(), "reason": "missing"})
        elif sha256_file(candidate) != expected:
            failures.append({"file": relative.strip(), "reason": "sha256_mismatch"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_file_count": checked,
        "failures": failures,
    }


def frozen_identity() -> dict[str, Any]:
    prior_manifest = json.loads(PRIOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    formal_manifest = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if prior_manifest["stage_feasibility_status"] != "MYOLEG_SUPINE_REHAB_MODEL_FEASIBLE_WITH_LIMITATIONS":
        raise RuntimeError("prior supine feasibility status changed")
    if sha256_file(NATIVE_DERIVED_XML) != NATIVE_DERIVED_SHA256:
        raise RuntimeError("frozen native supine derived XML changed")
    if sha256_file(REFERENCE_PATH) != REFERENCE_SHA256:
        raise RuntimeError("formal reference changed")
    if (
        formal_manifest["rom_protocol_version"] != "ROM_PROTOCOL_V2"
        or formal_manifest["hip_rom_deg"] != [0.0, 120.0]
        or formal_manifest["knee_rom_deg"] != [5.0, 145.0]
        or formal_manifest["theta_shank_definition"] != "q_hip - q_knee"
        or formal_manifest["active_reference_sha256"] != REFERENCE_SHA256
    ):
        raise RuntimeError("formal ROM/reference convention changed")
    prior_checks = verify_checksum_manifest(PRIOR_STAGE_DIRECTORY / "checksums.sha256")
    if prior_checks["status"] != "PASS":
        raise RuntimeError("prior stage checksum verification failed")
    upstream_assets = prior_manifest["source_identity"]["upstream_asset_sha256"]
    current_assets = {path: sha256_file(Path(path)) for path in upstream_assets}
    if current_assets != upstream_assets:
        raise RuntimeError("upstream MyoLeg asset changed")
    return {
        "formal_manifest_sha256": sha256_file(FORMAL_MANIFEST_PATH),
        "formal_reference_sha256": sha256_file(REFERENCE_PATH),
        "native_derived_xml_sha256": sha256_file(NATIVE_DERIVED_XML),
        "prior_stage_manifest_sha256": sha256_file(PRIOR_MANIFEST_PATH),
        "prior_stage_checksum_verification": prior_checks,
        "upstream_source_model_path": prior_manifest["source_identity"]["source_model_path"],
        "upstream_source_model_sha256": prior_manifest["source_identity"]["source_model_sha256"],
        "upstream_asset_count": len(upstream_assets),
        "upstream_asset_sha256": current_assets,
    }


def runtime_environment() -> dict[str, Any]:
    environment = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "myosuite": importlib.metadata.version("myosuite"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("pillow"),
    }
    frozen = {"python": "3.10.19", "myosuite": "2.12.2", "mujoco": "3.6.0"}
    environment["frozen_expected"] = frozen
    environment["frozen_match"] = all(environment[key] == value for key, value in frozen.items())
    if not environment["frozen_match"]:
        raise RuntimeError("frozen MyoLeg environment changed")
    return environment


def build_extension(limit_deg: float, output: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(NATIVE_DERIVED_XML))
    joint = spec.joint(TARGET_KNEE)
    native_lower = float(joint.range[0])
    joint.range = [native_lower, math.radians(limit_deg)]
    model = spec.compile()
    DERIVED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    spec.to_file(str(output))
    reloaded = mujoco.MjModel.from_xml_path(str(output))
    if object_dimensions(model) != object_dimensions(reloaded):
        raise RuntimeError("extension XML round-trip changed model dimensions")
    return reloaded


def xml_limit_only_diff(native_path: Path, extended_path: Path) -> dict[str, Any]:
    native_lines = native_path.read_text(encoding="utf-8").splitlines()
    extended_lines = extended_path.read_text(encoding="utf-8").splitlines()
    diff = list(
        difflib.unified_diff(
            native_lines,
            extended_lines,
            fromfile=native_path.name,
            tofile=extended_path.name,
            lineterm="",
        )
    )
    changed = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    only_knee_range = (
        len(changed) == 2
        and all('joint name="knee_angle_r"' in line and 'range="0 ' in line for line in changed)
    )
    return {
        "status": "PASS" if only_knee_range else "FAIL",
        "changed_content_lines": changed,
        "unified_diff": diff,
    }


def source_equality_hash(model: mujoco.MjModel) -> str:
    digest = hashlib.sha256()
    for array in (
        model.eq_type[:SOURCE_KNEE_EQUALITY_COUNT],
        model.eq_obj1id[:SOURCE_KNEE_EQUALITY_COUNT],
        model.eq_obj2id[:SOURCE_KNEE_EQUALITY_COUNT],
        model.eq_data[:SOURCE_KNEE_EQUALITY_COUNT],
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def polynomial(coefficients: np.ndarray, value: float) -> float:
    return float(sum(float(coefficients[index]) * value**index for index in range(5)))


def polynomial_derivative(coefficients: np.ndarray, value: float) -> float:
    return float(sum(index * float(coefficients[index]) * value ** (index - 1) for index in range(1, 5)))


def project_source_knee_equalities(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for equality_id in range(SOURCE_KNEE_EQUALITY_COUNT):
        first_joint = int(model.eq_obj1id[equality_id])
        second_joint = int(model.eq_obj2id[equality_id])
        first_qadr = int(model.jnt_qposadr[first_joint])
        second_qadr = int(model.jnt_qposadr[second_joint])
        first_dadr = int(model.jnt_dofadr[first_joint])
        second_dadr = int(model.jnt_dofadr[second_joint])
        parent_q = float(data.qpos[second_qadr])
        coefficients = model.eq_data[equality_id]
        data.qpos[first_qadr] = polynomial(coefficients, parent_q)
        data.qvel[first_dadr] = (
            polynomial_derivative(coefficients, parent_q) * data.qvel[second_dadr]
        )


def state_data(
    model: mujoco.MjModel,
    hip_deg: float,
    knee_deg: float,
    hip_velocity_deg_s: float = 0.0,
    knee_velocity_deg_s: float = 0.0,
    control: float = 0.0,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[:7] = np.asarray(
        [0.0, 0.0, 1.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    )
    data.qpos[qadr(model, TARGET_HIP)] = math.radians(hip_deg)
    data.qpos[qadr(model, TARGET_KNEE)] = math.radians(knee_deg)
    data.qvel[dadr(model, TARGET_HIP)] = math.radians(hip_velocity_deg_s)
    data.qvel[dadr(model, TARGET_KNEE)] = math.radians(knee_velocity_deg_s)
    project_source_knee_equalities(model, data)
    if model.na:
        data.act[:] = 0.0
    data.ctrl[:] = control
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def source_equality_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    rows = np.flatnonzero(
        np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    )
    source_rows = rows[np.asarray(data.efc_id)[rows] < SOURCE_KNEE_EQUALITY_COUNT]
    if not source_rows.size:
        return {"max_abs_residual": 0.0, "max_abs_force": 0.0}
    return {
        "max_abs_residual": float(np.max(np.abs(np.asarray(data.efc_pos)[source_rows]))),
        "max_abs_force": float(np.max(np.abs(np.asarray(data.efc_force)[source_rows]))),
    }


def moment_arms(
    model: mujoco.MjModel, hip_deg: float, knee_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    epsilon_deg = math.degrees(MOMENT_ARM_EPS_RAD)
    hip_plus = state_data(model, hip_deg + epsilon_deg, knee_deg)
    hip_minus = state_data(model, hip_deg - epsilon_deg, knee_deg)
    knee_plus = state_data(model, hip_deg, knee_deg + epsilon_deg)
    knee_minus = state_data(model, hip_deg, knee_deg - epsilon_deg)
    hip_moment = -(
        np.asarray(hip_plus.actuator_length) - np.asarray(hip_minus.actuator_length)
    ) / (2.0 * MOMENT_ARM_EPS_RAD)
    knee_moment = -(
        np.asarray(knee_plus.actuator_length) - np.asarray(knee_minus.actuator_length)
    ) / (2.0 * MOMENT_ARM_EPS_RAD)
    return hip_moment, knee_moment


def state_snapshot(
    model: mujoco.MjModel,
    hip_deg: float,
    knee_deg: float,
    knee_velocity_deg_s: float = 0.0,
) -> dict[str, Any]:
    data = state_data(
        model,
        hip_deg,
        knee_deg,
        knee_velocity_deg_s=knee_velocity_deg_s,
    )
    hip_moment, knee_moment = moment_arms(model, hip_deg, knee_deg)
    lower = np.asarray(model.actuator_lengthrange[:, 0], dtype=float)
    upper = np.asarray(model.actuator_lengthrange[:, 1], dtype=float)
    normalized = (np.asarray(data.actuator_length) - lower) / np.maximum(upper - lower, 1e-12)
    equality = source_equality_metrics(model, data)
    return {
        "data": data,
        "tendon_length": np.asarray(data.ten_length, dtype=float).copy(),
        "actuator_length": np.asarray(data.actuator_length, dtype=float).copy(),
        "normalized_actuator_length": normalized,
        "actuator_force": np.asarray(data.actuator_force, dtype=float).copy(),
        "hip_moment_arm": hip_moment,
        "knee_moment_arm": knee_moment,
        "qfrc_passive": np.asarray(data.qfrc_passive, dtype=float).copy(),
        "qfrc_actuator": np.asarray(data.qfrc_actuator, dtype=float).copy(),
        "qfrc_constraint": np.asarray(data.qfrc_constraint, dtype=float).copy(),
        "equality_residual": equality["max_abs_residual"],
        "equality_force": equality["max_abs_force"],
        "finite": finite_data(data),
        "warning_count": warning_count(data),
    }


def right_relevant_actuators(model: mujoco.MjModel) -> list[int]:
    related = set()
    for knee_deg in (115.0, 120.0, 125.0):
        hip_moment, knee_moment = moment_arms(model, 60.0, knee_deg)
        for index in range(model.nu):
            actuator_name = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            if actuator_name.endswith("_r") and (
                abs(hip_moment[index]) > 1e-8 or abs(knee_moment[index]) > 1e-8
            ):
                related.add(index)
    return sorted(related)


def common_domain_audit(
    native: mujoco.MjModel, primary: mujoco.MjModel
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    maximum = {
        "tendon_length": 0.0,
        "actuator_length": 0.0,
        "normalized_actuator_length": 0.0,
        "moment_arm_hip": 0.0,
        "moment_arm_knee": 0.0,
        "actuator_force": 0.0,
        "qfrc_passive": 0.0,
        "qfrc_actuator": 0.0,
        "equality_residual": 0.0,
    }
    native_knees = [value for value in KNEE_PRIMARY_GRID_DEG if value <= 120.0]
    for hip_deg in HIP_GRID_DEG:
        for knee_deg in native_knees:
            for velocity in VELOCITY_GRID_DEG_S:
                first = state_snapshot(native, hip_deg, knee_deg, velocity)
                second = state_snapshot(primary, hip_deg, knee_deg, velocity)
                differences = {
                    "tendon_length": float(np.max(np.abs(first["tendon_length"] - second["tendon_length"]))),
                    "actuator_length": float(np.max(np.abs(first["actuator_length"] - second["actuator_length"]))),
                    "normalized_actuator_length": float(np.max(np.abs(first["normalized_actuator_length"] - second["normalized_actuator_length"]))),
                    "moment_arm_hip": float(np.max(np.abs(first["hip_moment_arm"] - second["hip_moment_arm"]))),
                    "moment_arm_knee": float(np.max(np.abs(first["knee_moment_arm"] - second["knee_moment_arm"]))),
                    "actuator_force": float(np.max(np.abs(first["actuator_force"] - second["actuator_force"]))),
                    "qfrc_passive": float(np.max(np.abs(first["qfrc_passive"] - second["qfrc_passive"]))),
                    "qfrc_actuator": float(np.max(np.abs(first["qfrc_actuator"] - second["qfrc_actuator"]))),
                    "equality_residual": abs(first["equality_residual"] - second["equality_residual"]),
                }
                for key, value in differences.items():
                    maximum[key] = max(maximum[key], value)
                rows.append(
                    {
                        "hip_deg": hip_deg,
                        "knee_deg": knee_deg,
                        "knee_velocity_deg_s": velocity,
                        **{f"max_abs_diff_{key}": value for key, value in differences.items()},
                        "native_finite": first["finite"],
                        "extended_125_finite": second["finite"],
                        "deterministic_match": max(differences.values()) <= THRESHOLDS["common_domain_abs_tolerance"],
                    }
                )
    status = "PASS" if max(maximum.values()) <= THRESHOLDS["common_domain_abs_tolerance"] else "FAIL"
    return rows, {
        "status": status,
        "state_count": len(rows),
        "maximum_absolute_differences": maximum,
        "absolute_tolerance": THRESHOLDS["common_domain_abs_tolerance"],
    }


def state_grid_audit(models: dict[str, mujoco.MjModel]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    variants = {
        "NATIVE": [value for value in KNEE_PRIMARY_GRID_DEG if value <= 120.0],
        "PRIMARY_125": list(KNEE_PRIMARY_GRID_DEG),
        "STRESS_ONLY_130": list(KNEE_PRIMARY_GRID_DEG) + list(KNEE_STRESS_EXTRA_DEG),
    }
    for variant, knee_values in variants.items():
        model = models[variant]
        for hip_deg in HIP_GRID_DEG:
            for knee_deg in knee_values:
                for velocity in VELOCITY_GRID_DEG_S:
                    snapshot = state_snapshot(model, hip_deg, knee_deg, velocity)
                    actuator_force = snapshot["actuator_force"]
                    normalized = snapshot["normalized_actuator_length"]
                    rows.append(
                        {
                            "variant": variant,
                            "formal_role": "STRESS_ONLY" if variant == "STRESS_ONLY_130" else "PRIMARY_OR_NATIVE",
                            "hip_deg": hip_deg,
                            "knee_deg": knee_deg,
                            "knee_velocity_deg_s": velocity,
                            "tendon_length_min_m": float(np.min(snapshot["tendon_length"])),
                            "tendon_length_max_m": float(np.max(snapshot["tendon_length"])),
                            "actuator_length_min_m": float(np.min(snapshot["actuator_length"])),
                            "actuator_length_max_m": float(np.max(snapshot["actuator_length"])),
                            "normalized_actuator_length_min": float(np.min(normalized)),
                            "normalized_actuator_length_max": float(np.max(normalized)),
                            "actuator_force_max_abs_n": float(np.max(np.abs(actuator_force))),
                            "hip_moment_arm_max_abs_m": float(np.max(np.abs(snapshot["hip_moment_arm"]))),
                            "knee_moment_arm_max_abs_m": float(np.max(np.abs(snapshot["knee_moment_arm"]))),
                            "hip_qfrc_passive": float(snapshot["qfrc_passive"][dadr(model, TARGET_HIP)]),
                            "knee_qfrc_passive": float(snapshot["qfrc_passive"][dadr(model, TARGET_KNEE)]),
                            "hip_qfrc_actuator": float(snapshot["qfrc_actuator"][dadr(model, TARGET_HIP)]),
                            "knee_qfrc_actuator": float(snapshot["qfrc_actuator"][dadr(model, TARGET_KNEE)]),
                            "source_equality_max_abs_residual": snapshot["equality_residual"],
                            "source_equality_max_abs_force": snapshot["equality_force"],
                            "qfrc_constraint_l2": float(np.linalg.norm(snapshot["qfrc_constraint"])),
                            "finite": snapshot["finite"],
                            "warning_count": snapshot["warning_count"],
                        }
                    )
    stress_rows = [row for row in rows if row["formal_role"] == "STRESS_ONLY"]
    return rows, {
        "row_count": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "warning_count": sum(int(row["warning_count"]) for row in rows),
        "maximum_source_equality_residual": max(row["source_equality_max_abs_residual"] for row in rows),
        "stress_only_rows": len(stress_rows),
        "stress_only_all_finite": all(row["finite"] for row in stress_rows),
        "stress_only_warning_count": sum(int(row["warning_count"]) for row in stress_rows),
        "stress_only_maximum_source_equality_residual": max(
            row["source_equality_max_abs_residual"] for row in stress_rows
        ),
        "stress_only_minimum_tendon_length_m": min(
            row["tendon_length_min_m"] for row in stress_rows
        ),
        "stress_only_maximum_actuator_force_abs_n": max(
            row["actuator_force_max_abs_n"] for row in stress_rows
        ),
    }


def crossing_metrics(
    angles_deg: np.ndarray,
    values: np.ndarray,
    derivative_floor: float,
) -> dict[str, Any]:
    radians = np.radians(angles_deg)
    derivative = np.gradient(values, radians)
    index_120 = int(np.argmin(np.abs(angles_deg - 120.0)))
    left_slope = (values[index_120] - values[index_120 - 1]) / (
        radians[index_120] - radians[index_120 - 1]
    )
    right_slope = (values[index_120 + 1] - values[index_120]) / (
        radians[index_120 + 1] - radians[index_120]
    )
    scale = max(float(np.percentile(np.abs(derivative), 95)), derivative_floor)
    jump = float(abs(right_slope - left_slope))
    native_steps = np.abs(np.diff(values[angles_deg <= 120.0]))
    extended_steps = np.abs(np.diff(values[angles_deg >= 120.0]))
    native_step_max = float(np.max(native_steps, initial=0.0))
    extended_step_max = float(np.max(extended_steps, initial=0.0))
    step_ratio = extended_step_max / max(native_step_max, derivative_floor * math.radians(0.25))
    sign_flips = int(np.sum(np.signbit(values[:-1]) != np.signbit(values[1:])))
    return {
        "left_slope": float(left_slope),
        "right_slope": float(right_slope),
        "crossing_derivative_jump_abs": jump,
        "derivative_scale": scale,
        "crossing_derivative_jump_ratio": jump / scale,
        "native_max_adjacent_step": native_step_max,
        "extended_max_adjacent_step": extended_step_max,
        "extended_to_native_step_ratio": step_ratio,
        "sign_flip_count": sign_flips,
        "finite": bool(np.isfinite(values).all() and np.isfinite(derivative).all()),
    }


def continuity_audit(
    model: mujoco.MjModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    related = right_relevant_actuators(model)
    snapshots = [state_snapshot(model, 60.0, angle, 0.0) for angle in CONTINUITY_GRID_DEG]
    angles = np.asarray(CONTINUITY_GRID_DEG, dtype=float)
    muscle_rows = []
    moment_rows = []
    geometry_jump_max = 0.0
    moment_jump_max = 0.0
    force_jump_max = 0.0
    force_growth_max = 0.0
    force_fmax_ratio_max = 0.0
    normalized_min = math.inf
    normalized_max = -math.inf
    minimum_length = math.inf
    for actuator_index in related:
        actuator_name = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index)
        lengths = np.asarray([item["actuator_length"][actuator_index] for item in snapshots])
        tendon_lengths = np.asarray([item["tendon_length"][actuator_index] for item in snapshots])
        normalized = np.asarray([item["normalized_actuator_length"][actuator_index] for item in snapshots])
        forces = np.asarray([item["actuator_force"][actuator_index] for item in snapshots])
        hip_moments = np.asarray([item["hip_moment_arm"][actuator_index] for item in snapshots])
        knee_moments = np.asarray([item["knee_moment_arm"][actuator_index] for item in snapshots])
        length_metric = crossing_metrics(angles, lengths, 1.0e-5)
        tendon_metric = crossing_metrics(angles, tendon_lengths, 1.0e-5)
        force_metric = crossing_metrics(angles, forces, 1.0)
        hip_metric = crossing_metrics(angles, hip_moments, 1.0e-4)
        knee_metric = crossing_metrics(angles, knee_moments, 1.0e-4)
        geometry_jump_max = max(
            geometry_jump_max,
            length_metric["crossing_derivative_jump_ratio"],
            tendon_metric["crossing_derivative_jump_ratio"],
        )
        moment_jump_max = max(
            moment_jump_max,
            hip_metric["crossing_derivative_jump_ratio"],
            knee_metric["crossing_derivative_jump_ratio"],
        )
        force_jump_max = max(force_jump_max, force_metric["crossing_derivative_jump_ratio"])
        native_force = float(np.max(np.abs(forces[angles <= 120.0]), initial=0.0))
        extended_force = float(np.max(np.abs(forces[angles >= 120.0]), initial=0.0))
        growth = extended_force / max(native_force, 1.0)
        force_growth_max = max(force_growth_max, growth)
        fmax = max(float(model.actuator_gainprm[actuator_index, 2]), 1.0)
        fmax_ratio = float(np.max(np.abs(forces), initial=0.0)) / fmax
        force_fmax_ratio_max = max(force_fmax_ratio_max, fmax_ratio)
        normalized_min = min(normalized_min, float(np.min(normalized)))
        normalized_max = max(normalized_max, float(np.max(normalized)))
        minimum_length = min(minimum_length, float(np.min(lengths)), float(np.min(tendon_lengths)))
        biarticular = bool(
            np.max(np.abs(hip_moments)) > 1e-8 and np.max(np.abs(knee_moments)) > 1e-8
        )
        for sample_index, angle in enumerate(angles):
            muscle_rows.append(
                {
                    "actuator_id": actuator_index,
                    "actuator": actuator_name,
                    "preselected_key_muscle": actuator_name in KEY_MUSCLES,
                    "biarticular_hip_knee": biarticular,
                    "hip_deg": 60.0,
                    "knee_deg": float(angle),
                    "native_limit_crossed": bool(angle > 120.0),
                    "tendon_length_m": float(tendon_lengths[sample_index]),
                    "actuator_length_m": float(lengths[sample_index]),
                    "normalized_actuator_length": float(normalized[sample_index]),
                    "zero_control_actuator_force_n": float(forces[sample_index]),
                    "length_crossing_derivative_jump_ratio": length_metric["crossing_derivative_jump_ratio"],
                    "tendon_crossing_derivative_jump_ratio": tendon_metric["crossing_derivative_jump_ratio"],
                    "force_crossing_derivative_jump_ratio": force_metric["crossing_derivative_jump_ratio"],
                    "extended_to_native_force_growth_ratio": growth,
                    "force_to_model_fmax_ratio": fmax_ratio,
                    "finite": bool(
                        np.isfinite(lengths[sample_index])
                        and np.isfinite(forces[sample_index])
                    ),
                }
            )
            moment_rows.append(
                {
                    "actuator_id": actuator_index,
                    "actuator": actuator_name,
                    "preselected_key_muscle": actuator_name in KEY_MUSCLES,
                    "biarticular_hip_knee": biarticular,
                    "hip_deg": 60.0,
                    "knee_deg": float(angle),
                    "native_limit_crossed": bool(angle > 120.0),
                    "hip_moment_arm_m": float(hip_moments[sample_index]),
                    "knee_moment_arm_m": float(knee_moments[sample_index]),
                    "hip_crossing_derivative_jump_ratio": hip_metric["crossing_derivative_jump_ratio"],
                    "knee_crossing_derivative_jump_ratio": knee_metric["crossing_derivative_jump_ratio"],
                    "hip_sign_flip_count": hip_metric["sign_flip_count"],
                    "knee_sign_flip_count": knee_metric["sign_flip_count"],
                    "finite": bool(
                        np.isfinite(hip_moments[sample_index])
                        and np.isfinite(knee_moments[sample_index])
                    ),
                }
            )
    passive_knee = np.asarray(
        [item["qfrc_passive"][dadr(model, TARGET_KNEE)] for item in snapshots]
    )
    actuator_knee = np.asarray(
        [item["qfrc_actuator"][dadr(model, TARGET_KNEE)] for item in snapshots]
    )
    total_knee = passive_knee + actuator_knee
    torque_metric = crossing_metrics(angles, total_knee, 1.0)
    passive_rows = []
    for index, angle in enumerate(angles):
        passive_rows.append(
            {
                "hip_deg": 60.0,
                "knee_deg": float(angle),
                "native_limit_crossed": bool(angle > 120.0),
                "knee_qfrc_passive": float(passive_knee[index]),
                "knee_qfrc_actuator_zero_control": float(actuator_knee[index]),
                "knee_total_passive_residual": float(total_knee[index]),
                "total_crossing_derivative_jump_ratio": torque_metric["crossing_derivative_jump_ratio"],
                "source_equality_max_abs_residual": snapshots[index]["equality_residual"],
                "source_equality_max_abs_force": snapshots[index]["equality_force"],
                "finite": snapshots[index]["finite"],
                "warning_count": snapshots[index]["warning_count"],
            }
        )
    summary = {
        "related_right_actuator_count": len(related),
        "related_right_actuators": [
            name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in related
        ],
        "preselected_key_muscles": list(KEY_MUSCLES),
        "maximum_geometry_crossing_derivative_jump_ratio": geometry_jump_max,
        "maximum_moment_arm_crossing_derivative_jump_ratio": moment_jump_max,
        "maximum_force_crossing_derivative_jump_ratio": force_jump_max,
        "passive_torque_crossing_derivative_jump_ratio": torque_metric[
            "crossing_derivative_jump_ratio"
        ],
        "maximum_extended_to_native_force_growth_ratio": force_growth_max,
        "maximum_force_to_model_fmax_ratio": force_fmax_ratio_max,
        "normalized_actuator_length_min": normalized_min,
        "normalized_actuator_length_max": normalized_max,
        "minimum_tendon_or_actuator_length_m": minimum_length,
        "maximum_source_equality_residual": max(item["equality_residual"] for item in snapshots),
        "all_finite": all(item["finite"] for item in snapshots),
        "warning_count": sum(item["warning_count"] for item in snapshots),
    }
    summary["geometry_continuity_pass"] = (
        geometry_jump_max <= THRESHOLDS["geometry_crossing_derivative_jump_ratio_max"]
        and minimum_length > THRESHOLDS["state_length_positive_min_m"]
    )
    summary["moment_arm_continuity_pass"] = (
        moment_jump_max <= THRESHOLDS["moment_arm_crossing_derivative_jump_ratio_max"]
    )
    summary["force_continuity_pass"] = (
        force_jump_max <= THRESHOLDS["force_crossing_derivative_jump_ratio_max"]
        and torque_metric["crossing_derivative_jump_ratio"]
        <= THRESHOLDS["passive_torque_crossing_derivative_jump_ratio_max"]
        and force_growth_max <= THRESHOLDS["extended_to_native_force_growth_ratio_max"]
        and force_fmax_ratio_max <= THRESHOLDS["force_to_model_fmax_ratio_max"]
    )
    return muscle_rows, moment_rows, passive_rows, summary


def minimum_jerk(value: float) -> tuple[float, float]:
    value = min(1.0, max(0.0, value))
    position = 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5
    derivative = 30.0 * value**2 - 60.0 * value**3 + 30.0 * value**4
    return position, derivative


def sweep_target(elapsed_s: float) -> tuple[str, float, float]:
    if elapsed_s <= SWEEP_HALF_DURATION_S:
        phase, derivative = minimum_jerk(elapsed_s / SWEEP_HALF_DURATION_S)
        return (
            "ASCENDING_100_TO_125",
            100.0 + 25.0 * phase,
            25.0 * derivative / SWEEP_HALF_DURATION_S,
        )
    phase, derivative = minimum_jerk(
        (elapsed_s - SWEEP_HALF_DURATION_S) / SWEEP_HALF_DURATION_S
    )
    return (
        "DESCENDING_125_TO_100",
        125.0 - 25.0 * phase,
        -25.0 * derivative / SWEEP_HALF_DURATION_S,
    )


def set_sweep_driver(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_target_deg: float,
    knee_target_deg: float,
    knee_target_velocity_deg_s: float,
) -> tuple[float, float]:
    hip_q = qadr(model, TARGET_HIP)
    knee_q = qadr(model, TARGET_KNEE)
    hip_v = dadr(model, TARGET_HIP)
    knee_v = dadr(model, TARGET_KNEE)
    tau_hip = np.clip(
        SWEEP_KP * (math.radians(hip_target_deg) - data.qpos[hip_q])
        + SWEEP_KD * (0.0 - data.qvel[hip_v]),
        -SWEEP_TORQUE_LIMIT_NM,
        SWEEP_TORQUE_LIMIT_NM,
    )
    tau_knee = np.clip(
        SWEEP_KP * (math.radians(knee_target_deg) - data.qpos[knee_q])
        + SWEEP_KD
        * (math.radians(knee_target_velocity_deg_s) - data.qvel[knee_v]),
        -SWEEP_TORQUE_LIMIT_NM,
        SWEEP_TORQUE_LIMIT_NM,
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[hip_v] = tau_hip
    data.qfrc_applied[knee_v] = tau_knee
    return float(tau_hip), float(tau_knee)


def auxiliary_positions(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    values = []
    for equality_id in range(7):
        joint = int(model.eq_obj1id[equality_id])
        values.append(float(data.qpos[int(model.jnt_qposadr[joint])]))
    return values


def low_control_sweep(
    model: mujoco.MjModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sweep_rows = []
    patella_rows = []
    total_steps = int(round(2.0 * SWEEP_HALF_DURATION_S / SWEEP_DT_S)) + 1
    peak_tracking_error = 0.0
    peak_equality_residual = 0.0
    peak_constraint_force = 0.0
    peak_applied_torque = 0.0
    all_finite = True
    warnings = 0
    hysteresis_by_hip: dict[str, float] = {}
    for hip_deg in SWEEP_HIP_GRID_DEG:
        data = state_data(model, hip_deg, 100.0)
        branch_records: dict[str, list[dict[str, Any]]] = {
            "ASCENDING_100_TO_125": [],
            "DESCENDING_125_TO_100": [],
        }
        for step in range(total_steps):
            elapsed = min(step * SWEEP_DT_S, 2.0 * SWEEP_HALF_DURATION_S)
            direction, knee_target, knee_velocity_target = sweep_target(elapsed)
            tau_hip, tau_knee = set_sweep_driver(
                model, data, hip_deg, knee_target, knee_velocity_target
            )
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            equality = source_equality_metrics(model, data)
            actual_hip = math.degrees(float(data.qpos[qadr(model, TARGET_HIP)]))
            actual_knee = math.degrees(float(data.qpos[qadr(model, TARGET_KNEE)]))
            tracking_error = abs(actual_knee - knee_target)
            peak_tracking_error = max(peak_tracking_error, tracking_error)
            peak_equality_residual = max(
                peak_equality_residual, equality["max_abs_residual"]
            )
            peak_constraint_force = max(
                peak_constraint_force, equality["max_abs_force"]
            )
            peak_applied_torque = max(
                peak_applied_torque, abs(tau_hip), abs(tau_knee)
            )
            all_finite = all_finite and finite_data(data)
            warnings = max(warnings, warning_count(data))
            aux = auxiliary_positions(model, data)
            branch_records[direction].append(
                {
                    "desired_knee_deg": knee_target,
                    "actual_knee_deg": actual_knee,
                    "auxiliary": aux,
                }
            )
            if step % SWEEP_RECORD_STRIDE == 0 or step == total_steps - 1:
                common = {
                    "hip_target_deg": hip_deg,
                    "actual_hip_deg": actual_hip,
                    "direction": direction,
                    "time_s": elapsed,
                    "desired_knee_deg": knee_target,
                    "actual_knee_deg": actual_knee,
                    "desired_knee_velocity_deg_s": knee_velocity_target,
                    "actual_knee_velocity_deg_s": math.degrees(
                        float(data.qvel[dadr(model, TARGET_KNEE)])
                    ),
                    "knee_tracking_error_deg": actual_knee - knee_target,
                    "hip_diagnostic_torque_nm": tau_hip,
                    "knee_diagnostic_torque_nm": tau_knee,
                    "hip_qfrc_passive": float(data.qfrc_passive[dadr(model, TARGET_HIP)]),
                    "knee_qfrc_passive": float(data.qfrc_passive[dadr(model, TARGET_KNEE)]),
                    "hip_qfrc_actuator": float(data.qfrc_actuator[dadr(model, TARGET_HIP)]),
                    "knee_qfrc_actuator": float(data.qfrc_actuator[dadr(model, TARGET_KNEE)]),
                    "source_equality_max_abs_residual": equality["max_abs_residual"],
                    "source_equality_max_abs_force": equality["max_abs_force"],
                    "qfrc_constraint_l2": float(np.linalg.norm(data.qfrc_constraint)),
                    "tendon_length_min_m": float(np.min(data.ten_length)),
                    "tendon_length_max_m": float(np.max(data.ten_length)),
                    "muscle_force_max_abs_n": float(np.max(np.abs(data.actuator_force))),
                    "individual_muscle_forces_json": json.dumps(
                        np.asarray(data.actuator_force, dtype=float).tolist(),
                        separators=(",", ":"),
                    ),
                    "finite": finite_data(data),
                    "warning_count": warning_count(data),
                }
                sweep_rows.append(common)
                patella_rows.append(
                    {
                        "hip_target_deg": hip_deg,
                        "direction": direction,
                        "time_s": elapsed,
                        "desired_knee_deg": knee_target,
                        "actual_knee_deg": actual_knee,
                        "source_equality_max_abs_residual": equality[
                            "max_abs_residual"
                        ],
                        "source_equality_max_abs_force": equality["max_abs_force"],
                        "auxiliary_joint_positions_json": json.dumps(
                            aux, separators=(",", ":")
                        ),
                        "finite": finite_data(data),
                        "warning_count": warning_count(data),
                    }
                )
        comparisons = []
        ascending = branch_records["ASCENDING_100_TO_125"]
        descending = branch_records["DESCENDING_125_TO_100"]
        for target in np.linspace(110.0, 125.0, 31):
            up = min(ascending, key=lambda row: abs(row["desired_knee_deg"] - target))
            down = min(descending, key=lambda row: abs(row["desired_knee_deg"] - target))
            comparisons.append(
                max(
                    abs(a - b)
                    for a, b in zip(up["auxiliary"], down["auxiliary"])
                )
            )
        hysteresis_by_hip[str(hip_deg)] = max(comparisons)
    summary = {
        "hip_angles_deg": list(SWEEP_HIP_GRID_DEG),
        "simulated_time_per_hip_s": 2.0 * SWEEP_HALF_DURATION_S,
        "total_steps": total_steps * len(SWEEP_HIP_GRID_DEG),
        "recorded_rows": len(sweep_rows),
        "maximum_tracking_error_deg": peak_tracking_error,
        "maximum_source_equality_residual": peak_equality_residual,
        "maximum_source_equality_force": peak_constraint_force,
        "maximum_required_diagnostic_torque_nm": peak_applied_torque,
        "maximum_auxiliary_roundtrip_inconsistency": max(hysteresis_by_hip.values()),
        "auxiliary_roundtrip_inconsistency_by_hip": hysteresis_by_hip,
        "all_finite": all_finite,
        "warning_count": warnings,
    }
    summary["pass"] = (
        all_finite
        and warnings == 0
        and peak_tracking_error <= THRESHOLDS["sweep_tracking_error_max_deg"]
        and peak_equality_residual <= THRESHOLDS["source_equality_residual_max"]
        and max(hysteresis_by_hip.values())
        <= THRESHOLDS["sweep_auxiliary_hysteresis_max"]
    )
    return sweep_rows, patella_rows, summary


def read_reference() -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def reference_state_path_audit(
    model: mujoco.MjModel, reference_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    relevant_indices = right_relevant_actuators(model)
    native_model = mujoco.MjModel.from_xml_path(str(NATIVE_DERIVED_XML))
    native_knee_upper_deg = math.degrees(
        float(native_model.jnt_range[jid(native_model, TARGET_KNEE), 1])
    )
    knee_upper_deg = math.degrees(float(model.jnt_range[jid(model, TARGET_KNEE), 1]))
    hip_range_deg = np.degrees(model.jnt_range[jid(model, TARGET_HIP)]).tolist()
    for index, source in enumerate(reference_rows):
        hip_deg = math.degrees(float(source["q_hip_rad"]))
        knee_deg = math.degrees(float(source["q_knee_rad"]))
        hip_velocity_deg_s = math.degrees(float(source["dq_hip_rad_s"]))
        knee_velocity_deg_s = math.degrees(float(source["dq_knee_rad_s"]))
        data = state_data(
            model,
            hip_deg,
            knee_deg,
            hip_velocity_deg_s,
            knee_velocity_deg_s,
        )
        equality = source_equality_metrics(model, data)
        lower = np.asarray(model.actuator_lengthrange[:, 0])
        upper = np.asarray(model.actuator_lengthrange[:, 1])
        normalized = (np.asarray(data.actuator_length) - lower) / np.maximum(
            upper - lower, 1e-12
        )
        fmax = np.maximum(np.asarray(model.actuator_gainprm[:, 2]), 1.0)
        force_ratio = np.abs(np.asarray(data.actuator_force)) / fmax
        relevant_normalized = normalized[relevant_indices]
        relevant_force_ratio = force_ratio[relevant_indices]
        relevant_tendon_lengths = np.asarray(data.ten_length)[relevant_indices]
        violation_reasons = []
        context_warnings = []
        if not finite_data(data):
            violation_reasons.append("nonfinite_state")
        if knee_deg > knee_upper_deg + 1e-9:
            violation_reasons.append("knee_above_125_model_limit")
        if not (hip_range_deg[0] - 1e-9 <= hip_deg <= hip_range_deg[1] + 1e-9):
            violation_reasons.append("hip_outside_native_limit")
        if float(np.min(relevant_tendon_lengths)) <= THRESHOLDS["state_length_positive_min_m"]:
            violation_reasons.append("nonpositive_tendon_length")
        if float(np.min(relevant_normalized)) < THRESHOLDS["normalized_actuator_length_min"]:
            context_warnings.append("actuator_lengthrange_coordinate_below_context_band")
        if float(np.max(relevant_normalized)) > THRESHOLDS["normalized_actuator_length_max"]:
            context_warnings.append("actuator_lengthrange_coordinate_above_context_band")
        if float(np.max(relevant_force_ratio)) > THRESHOLDS["force_to_model_fmax_ratio_max"]:
            violation_reasons.append("force_growth_above_protocol")
        if equality["max_abs_residual"] > THRESHOLDS["source_equality_residual_max"]:
            violation_reasons.append("source_equality_residual_above_protocol")
        if warning_count(data):
            violation_reasons.append("solver_warning")
        key_forces = {
            muscle: float(data.actuator_force[aid(model, muscle)])
            for muscle in KEY_MUSCLES
        }
        rows.append(
            {
                "index": index,
                "time_s": float(source["time_s"]),
                "cycle_phase": source["cycle_phase"],
                "hip_deg": hip_deg,
                "knee_deg": knee_deg,
                "originally_above_native_120": knee_deg
                > native_knee_upper_deg,
                "within_125_model_limit": knee_deg <= knee_upper_deg + 1e-9,
                "target_relevant_actuator_count": len(relevant_indices),
                "target_relevant_tendon_length_min_m": float(np.min(relevant_tendon_lengths)),
                "target_relevant_tendon_length_max_m": float(np.max(relevant_tendon_lengths)),
                "target_relevant_actuator_length_min_m": float(np.min(np.asarray(data.actuator_length)[relevant_indices])),
                "target_relevant_actuator_length_max_m": float(np.max(np.asarray(data.actuator_length)[relevant_indices])),
                "target_relevant_normalized_actuator_length_min": float(np.min(relevant_normalized)),
                "target_relevant_normalized_actuator_length_max": float(np.max(relevant_normalized)),
                "target_relevant_actuator_force_max_abs_n": float(np.max(np.abs(np.asarray(data.actuator_force)[relevant_indices]))),
                "target_relevant_force_to_model_fmax_ratio_max": float(np.max(relevant_force_ratio)),
                "all_model_normalized_actuator_length_min_context_only": float(np.min(normalized)),
                "all_model_normalized_actuator_length_max_context_only": float(np.max(normalized)),
                "key_muscle_forces_json": json.dumps(
                    key_forces, sort_keys=True, separators=(",", ":")
                ),
                "source_equality_max_abs_residual": equality["max_abs_residual"],
                "source_equality_max_abs_force": equality["max_abs_force"],
                "finite": finite_data(data),
                "warning_count": warning_count(data),
                "abnormal_condition": bool(violation_reasons),
                "abnormal_reasons": ";".join(violation_reasons),
                "context_warning": bool(context_warnings),
                "context_warning_reasons": ";".join(context_warnings),
            }
        )
    abnormal = [row for row in rows if row["abnormal_condition"]]
    context_warning_rows = [row for row in rows if row["context_warning"]]
    summary = {
        "status": "PASS" if not abnormal else "FAIL",
        "formal_status_label": (
            "REFERENCE_STATE_PATH_VALID_IN_125_MODEL = PASS"
            if not abnormal
            else "REFERENCE_STATE_PATH_VALID_IN_125_MODEL = FAIL"
        ),
        "sample_count": len(rows),
        "originally_above_native_count": sum(
            row["originally_above_native_120"] for row in rows
        ),
        "abnormal_count": len(abnormal),
        "abnormal_indices": [row["index"] for row in abnormal],
        "context_warning_count": len(context_warning_rows),
        "context_warning_indices": [row["index"] for row in context_warning_rows],
        "context_warnings_above_native_count": sum(
            row["context_warning"] and row["originally_above_native_120"]
            for row in rows
        ),
        "normalized_muscle_fiber_length_available": False,
        "actuator_lengthrange_coordinate_is_context_only": True,
        "maximum_knee_deg": max(row["knee_deg"] for row in rows),
        "maximum_source_equality_residual": max(
            row["source_equality_max_abs_residual"] for row in rows
        ),
        "maximum_force_to_model_fmax_ratio": max(
            row["target_relevant_force_to_model_fmax_ratio_max"] for row in rows
        ),
        "target_relevant_actuator_count": len(relevant_indices),
        "target_relevant_actuators": [
            name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in relevant_indices
        ],
        "all_finite": all(row["finite"] for row in rows),
        "warning_count": max(row["warning_count"] for row in rows),
        "formal_reference_replayed": False,
        "state_evaluation_only": True,
    }
    return rows, summary


def project_kinematics(hip: np.ndarray, knee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shank = hip - knee
    x = L1_PROJECT_M * np.sin(hip) + L2_PROJECT_M * np.sin(shank)
    z = L1_PROJECT_M * np.cos(hip) + L2_PROJECT_M * np.cos(shank)
    return x, z


def project_jacobian(hip: float, knee: float) -> np.ndarray:
    shank = hip - knee
    return np.asarray(
        [
            [
                L1_PROJECT_M * math.cos(hip) + L2_PROJECT_M * math.cos(shank),
                -L2_PROJECT_M * math.cos(shank),
            ],
            [
                -L1_PROJECT_M * math.sin(hip) - L2_PROJECT_M * math.sin(shank),
                L2_PROJECT_M * math.sin(shank),
            ],
        ]
    )


def native_reference_candidate(
    reference_rows: list[dict[str, str]], output_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    time_values = np.asarray([float(row["time_s"]) for row in reference_rows])
    hip = np.asarray([float(row["q_hip_rad"]) for row in reference_rows])
    knee = np.asarray([float(row["q_knee_rad"]) for row in reference_rows])
    dhip = np.asarray([float(row["dq_hip_rad_s"]) for row in reference_rows])
    dknee = np.asarray([float(row["dq_knee_rad_s"]) for row in reference_rows])
    ddhip = np.asarray([float(row["ddq_hip_rad_s2"]) for row in reference_rows])
    ddknee = np.asarray([float(row["ddq_knee_rad_s2"]) for row in reference_rows])
    anchor = float(knee[0])
    original_max = float(np.max(knee))
    target = math.radians(NATIVE_REFERENCE_TARGET_DEG)
    scale = (target - anchor) / (original_max - anchor)
    knee_new = anchor + scale * (knee - anchor)
    dknee_new = scale * dknee
    ddknee_new = scale * ddknee
    theta_new = hip - knee_new
    x_new, z_new = project_kinematics(hip, knee_new)
    candidate_rows = []
    for index, source in enumerate(reference_rows):
        candidate_rows.append(
            {
                "candidate_id": "MYOLEG_NATIVE_ROM_REFERENCE_CANDIDATE",
                "parent_reference_id": "reference_measured_asymmetric_closed_slow",
                "parent_reference_sha256": REFERENCE_SHA256,
                "time_s": time_values[index],
                "cycle_phase": source["cycle_phase"],
                "segment_phase": source["segment_phase"],
                "global_phase": source["global_phase"],
                "q_hip_rad": hip[index],
                "q_knee_rad": knee_new[index],
                "dq_hip_rad_s": dhip[index],
                "dq_knee_rad_s": dknee_new[index],
                "ddq_hip_rad_s2": ddhip[index],
                "ddq_knee_rad_s2": ddknee_new[index],
                "theta_shank_rad": theta_new[index],
                "x_pull_m": x_new[index],
                "z_pull_m": z_new[index],
                "transformation": "global_affine_knee_amplitude_about_first_sample",
                "amplitude_scale": scale,
                "target_peak_knee_deg": NATIVE_REFERENCE_TARGET_DEG,
                "pointwise_clipped": False,
                "active_reference": False,
                "diagnostic_simulation_candidate_only": True,
                "robot_execution_approved": False,
            }
        )
    write_csv(output_path, candidate_rows, list(candidate_rows[0].keys()))
    candidate_sha = sha256_file(output_path)
    x_original, z_original = project_kinematics(hip, knee)
    jac_original = [project_jacobian(h, k) for h, k in zip(hip, knee)]
    jac_new = [project_jacobian(h, k) for h, k in zip(hip, knee_new)]
    original_det = np.asarray([np.linalg.det(value) for value in jac_original])
    new_det = np.asarray([np.linalg.det(value) for value in jac_new])
    original_condition = np.asarray([np.linalg.cond(value) for value in jac_original])
    new_condition = np.asarray([np.linalg.cond(value) for value in jac_new])
    transformation = {
        "candidate_id": "MYOLEG_NATIVE_ROM_REFERENCE_CANDIDATE",
        "role": "DIAGNOSTIC_SIMULATION_CANDIDATE_ONLY",
        "active_reference": False,
        "robot_execution_approved": False,
        "parent_reference_sha256": REFERENCE_SHA256,
        "target_peak_knee_deg_preselected": NATIVE_REFERENCE_TARGET_DEG,
        "anchor_knee_rad": anchor,
        "anchor_knee_deg": math.degrees(anchor),
        "original_peak_knee_rad": original_max,
        "original_peak_knee_deg": math.degrees(original_max),
        "amplitude_scale": scale,
        "formula": "q_new(t)=q0+s*(q_original(t)-q0)",
        "derivative_formula": "dq_new=s*dq_original; ddq_new=s*ddq_original",
        "inverse_formula": "q_original(t)=q0+(q_new(t)-q0)/s",
        "globally_smooth": True,
        "invertible": bool(scale > 0.0),
        "pointwise_clipping_used": False,
        "selection_uses_extension_results": False,
        "duration_s": float(time_values[-1] - time_values[0]),
        "sample_count": len(reference_rows),
        "candidate_csv_sha256": candidate_sha,
    }
    comparison = {
        "hip_trajectory_max_abs_difference_deg": math.degrees(
            float(np.max(np.abs(hip - hip)))
        ),
        "knee_rms_difference_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(knee_new - knee))))
        ),
        "knee_max_abs_difference_deg": math.degrees(
            float(np.max(np.abs(knee_new - knee)))
        ),
        "original_knee_peak_deg": math.degrees(float(np.max(knee))),
        "candidate_knee_peak_deg": math.degrees(float(np.max(knee_new))),
        "knee_velocity_rms_difference_deg_s": math.degrees(
            float(np.sqrt(np.mean(np.square(dknee_new - dknee))))
        ),
        "knee_velocity_max_abs_difference_deg_s": math.degrees(
            float(np.max(np.abs(dknee_new - dknee)))
        ),
        "knee_acceleration_rms_difference_deg_s2": math.degrees(
            float(np.sqrt(np.mean(np.square(ddknee_new - ddknee))))
        ),
        "knee_acceleration_max_abs_difference_deg_s2": math.degrees(
            float(np.max(np.abs(ddknee_new - ddknee)))
        ),
        "start_knee_difference_deg": math.degrees(float(knee_new[0] - knee[0])),
        "candidate_joint_closure_error_rad": float(
            max(abs(hip[-1] - hip[0]), abs(knee_new[-1] - knee_new[0]))
        ),
        "candidate_velocity_closure_error_rad_s": float(
            max(abs(dhip[-1] - dhip[0]), abs(dknee_new[-1] - dknee_new[0]))
        ),
        "candidate_acceleration_closure_error_rad_s2": float(
            max(abs(ddhip[-1] - ddhip[0]), abs(ddknee_new[-1] - ddknee_new[0]))
        ),
        "c2_preserved_by_affine_transformation": True,
        "phase_columns_unchanged": True,
        "extrema_indices_unchanged": bool(
            int(np.argmax(knee_new)) == int(np.argmax(knee))
            and int(np.argmin(knee_new)) == int(np.argmin(knee))
        ),
        "pull_point_rms_displacement_mm": 1000.0
        * float(
            np.sqrt(
                np.mean(np.square(x_new - x_original) + np.square(z_new - z_original))
            )
        ),
        "pull_point_max_displacement_mm": 1000.0
        * float(np.max(np.hypot(x_new - x_original, z_new - z_original))),
        "jacobian_determinant_rms_difference": float(
            np.sqrt(np.mean(np.square(new_det - original_det)))
        ),
        "jacobian_condition_original_max": float(np.max(original_condition)),
        "jacobian_condition_candidate_max": float(np.max(new_condition)),
        "jacobian_condition_max_abs_difference": float(
            np.max(np.abs(new_condition - original_condition))
        ),
        "duration_preserved": bool(time_values[-1] == 24.0 and len(time_values) == 401),
        "sample_count_preserved": len(time_values) == 401,
        "pointwise_clipping_used": False,
    }
    return transformation, comparison


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def line_figure(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]],
    vertical_references: list[tuple[float, str, tuple[int, int, int]]] | None = None,
    horizontal_references: list[tuple[float, str, tuple[int, int, int]]] | None = None,
) -> None:
    width, height = 1400, 820
    margin_left, margin_right, margin_top, margin_bottom = 115, 55, 75, 95
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    all_x = np.concatenate([np.asarray(item[1], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item[2], dtype=float) for item in series])
    if vertical_references:
        all_x = np.concatenate([all_x, np.asarray([item[0] for item in vertical_references])])
    if horizontal_references:
        all_y = np.concatenate([all_y, np.asarray([item[0] for item in horizontal_references])])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if math.isclose(xmin, xmax):
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if math.isclose(ymin, ymax):
        ymin, ymax = ymin - 1.0, ymax + 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def pixel_x(value: float) -> float:
        return margin_left + (value - xmin) / (xmax - xmin) * (
            width - margin_left - margin_right
        )

    def pixel_y(value: float) -> float:
        return height - margin_bottom - (value - ymin) / (ymax - ymin) * (
            height - margin_top - margin_bottom
        )

    draw.text((margin_left, 22), title, fill="black", font=font)
    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill="black",
        width=2,
    )
    draw.line(
        [
            (margin_left, height - margin_bottom),
            (width - margin_right, height - margin_bottom),
        ],
        fill="black",
        width=2,
    )
    for tick in np.linspace(xmin, xmax, 6):
        x = pixel_x(float(tick))
        draw.line([(x, height - margin_bottom), (x, height - margin_bottom + 6)], fill="black")
        draw.text((x - 20, height - margin_bottom + 10), f"{tick:.2f}", fill="black", font=font)
    for tick in np.linspace(ymin, ymax, 6):
        y = pixel_y(float(tick))
        draw.line([(margin_left - 6, y), (margin_left, y)], fill="black")
        draw.text((8, y - 6), f"{tick:.3g}", fill="black", font=font)
    draw.text((width // 2 - 50, height - 35), x_label, fill="black", font=font)
    draw.text((10, 48), y_label, fill="black", font=font)
    if vertical_references:
        for value, label, color in vertical_references:
            x = pixel_x(value)
            for y in range(margin_top, height - margin_bottom, 12):
                draw.line([(x, y), (x, min(y + 6, height - margin_bottom))], fill=color, width=2)
            near_right = x >= width - margin_right - 210
            label_x = x + 4 if not near_right else x - 190
            label_y = margin_top + 5 if not near_right else height - margin_bottom - 20
            draw.text((label_x, label_y), label, fill=color, font=font)
    if horizontal_references:
        for value, label, color in horizontal_references:
            y = pixel_y(value)
            for x in range(margin_left, width - margin_right, 12):
                draw.line([(x, y), (min(x + 6, width - margin_right), y)], fill=color, width=2)
            draw.text((margin_left + 5, y - 15), label, fill=color, font=font)
    legend_x = width - margin_right - 250
    legend_y = margin_top + 15
    for index, (label, x_values, y_values, color) in enumerate(series):
        points = [
            (pixel_x(float(x)), pixel_y(float(y)))
            for x, y in zip(x_values, y_values)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points[:: max(1, len(points) // 40)]:
            draw.ellipse(
                [point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2],
                fill=color,
            )
        y_legend = legend_y + 20 * index
        draw.line([(legend_x, y_legend), (legend_x + 30, y_legend)], fill=color, width=3)
        draw.text((legend_x + 38, y_legend - 6), label, fill=color, font=font)
    image.save(path)


def generate_figures(
    passive_rows: list[dict[str, Any]],
    muscle_rows: list[dict[str, Any]],
    moment_rows: list[dict[str, Any]],
    patella_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, str]],
    candidate_path: Path,
) -> list[Path]:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    native_line = [(120.0, "native 120 deg", (190, 30, 45))]
    reference_line = [(REFERENCE_MAX_DEG, "reference max 124.7866", (230, 125, 20))]
    angles = np.asarray([row["knee_deg"] for row in passive_rows])
    torque = np.asarray([row["knee_total_passive_residual"] for row in passive_rows])
    paths = []
    path = FIGURE_DIRECTORY / "passive_knee_torque_vs_knee_angle.png"
    line_figure(
        path,
        "Zero-control residual knee torque across limited extension (hip=60 deg)",
        "knee angle (deg)",
        "knee generalized torque (N m)",
        [("passive+muscle residual", angles, torque, (30, 100, 190))],
        native_line + reference_line,
    )
    paths.append(path)
    colors = [(34, 139, 34), (128, 0, 128), (0, 139, 139), (210, 80, 20), (40, 80, 180)]
    force_series = []
    moment_series = []
    for muscle, color in zip(KEY_MUSCLES, colors):
        selected_force = [row for row in muscle_rows if row["actuator"] == muscle]
        selected_moment = [row for row in moment_rows if row["actuator"] == muscle]
        force_series.append(
            (
                muscle,
                np.asarray([row["knee_deg"] for row in selected_force]),
                np.asarray([row["zero_control_actuator_force_n"] for row in selected_force]),
                color,
            )
        )
        moment_series.append(
            (
                muscle,
                np.asarray([row["knee_deg"] for row in selected_moment]),
                np.asarray([row["knee_moment_arm_m"] for row in selected_moment]),
                color,
            )
        )
    path = FIGURE_DIRECTORY / "key_muscle_passive_force_vs_knee_angle.png"
    line_figure(
        path,
        "Preselected key-muscle zero-control force",
        "knee angle (deg)",
        "actuator force (N)",
        force_series,
        native_line + reference_line,
    )
    paths.append(path)
    path = FIGURE_DIRECTORY / "key_muscle_moment_arm_vs_knee_angle.png"
    line_figure(
        path,
        "Preselected key-muscle knee moment arms",
        "knee angle (deg)",
        "moment arm (m)",
        moment_series,
        native_line + reference_line,
    )
    paths.append(path)
    patella_selected = [
        row
        for row in patella_rows
        if row["hip_target_deg"] == 60.0
        and row["direction"] == "ASCENDING_100_TO_125"
    ]
    path = FIGURE_DIRECTORY / "equality_residual_vs_knee_angle.png"
    line_figure(
        path,
        "Source knee/patella equality residual during low-speed ascent (hip=60 deg)",
        "actual knee angle (deg)",
        "max absolute equality residual",
        [
            (
                "source equality residual",
                np.asarray([row["actual_knee_deg"] for row in patella_selected]),
                np.asarray(
                    [row["source_equality_max_abs_residual"] for row in patella_selected]
                ),
                (90, 50, 170),
            )
        ],
        native_line + reference_line,
    )
    paths.append(path)
    with candidate_path.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    time_values = np.asarray([float(row["time_s"]) for row in reference_rows])
    original_knee = np.degrees(
        np.asarray([float(row["q_knee_rad"]) for row in reference_rows])
    )
    candidate_knee = np.degrees(
        np.asarray([float(row["q_knee_rad"]) for row in candidate_rows])
    )
    path = FIGURE_DIRECTORY / "original_vs_native_compatible_knee_trajectory.png"
    line_figure(
        path,
        "Formal reference vs native-compatible diagnostic candidate",
        "time (s)",
        "knee flexion (deg)",
        [
            ("formal unchanged", time_values, original_knee, (30, 100, 190)),
            ("native candidate", time_values, candidate_knee, (34, 139, 34)),
        ],
        horizontal_references=[
            (120.0, "native 120 deg", (190, 30, 45)),
            (REFERENCE_MAX_DEG, "reference max 124.7866", (230, 125, 20)),
        ],
    )
    paths.append(path)
    return paths


def determine_outcome(
    common_summary: dict[str, Any],
    continuity_summary: dict[str, Any],
    state_summary: dict[str, Any],
    sweep_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> str:
    required = (
        common_summary["status"] == "PASS",
        continuity_summary["geometry_continuity_pass"],
        continuity_summary["moment_arm_continuity_pass"],
        continuity_summary["force_continuity_pass"],
        continuity_summary["all_finite"],
        continuity_summary["warning_count"] == 0,
        state_summary["all_finite"],
        state_summary["warning_count"] == 0,
        sweep_summary["pass"],
        reference_summary["status"] == "PASS",
    )
    if all(required):
        return "LIMITED_125DEG_EXTENSION_MECHANICALLY_CONTINUOUS"
    clear_failure = (
        common_summary["status"] == "FAIL"
        or not continuity_summary["all_finite"]
        or not state_summary["all_finite"]
        or not sweep_summary["all_finite"]
        or reference_summary["status"] == "FAIL"
    )
    return (
        "LIMITED_EXTENSION_NOT_SUPPORTED"
        if clear_failure
        else "LIMITED_EXTENSION_VALIDITY_INCONCLUSIVE"
    )


def run_tests(
    identity_before: dict[str, Any],
    identity_after: dict[str, Any],
    native: mujoco.MjModel,
    primary: mujoco.MjModel,
    diff_125: dict[str, Any],
    common_summary: dict[str, Any],
    state_summary: dict[str, Any],
    continuity_summary: dict[str, Any],
    sweep_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    transformation: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    tests = [
        ("upstream_myoleg_unchanged", identity_before == identity_after),
        ("formal_reference_unchanged", identity_after["formal_reference_sha256"] == REFERENCE_SHA256),
        (
            "rom_protocol_v2_unchanged",
            identity_before["formal_manifest_sha256"]
            == identity_after["formal_manifest_sha256"],
        ),
        (
            "native_derived_model_unchanged",
            identity_after["native_derived_xml_sha256"] == NATIVE_DERIVED_SHA256,
        ),
        ("primary_125_diff_is_only_knee_limit", diff_125["status"] == "PASS"),
        (
            "muscles_and_tendons_preserved",
            native.nu == primary.nu == 80 and native.ntendon == primary.ntendon == 80,
        ),
        (
            "fourteen_source_knee_equalities_preserved",
            source_equality_hash(native) == source_equality_hash(primary)
            and native.neq >= SOURCE_KNEE_EQUALITY_COUNT
            and primary.neq >= SOURCE_KNEE_EQUALITY_COUNT,
        ),
        ("common_domain_deterministic", common_summary["status"] == "PASS"),
        (
            "no_nan_or_inf",
            state_summary["all_finite"]
            and continuity_summary["all_finite"]
            and sweep_summary["all_finite"]
            and reference_summary["all_finite"],
        ),
        (
            "candidate_is_not_pointwise_clipped",
            not transformation["pointwise_clipping_used"]
            and transformation["globally_smooth"]
            and transformation["invertible"],
        ),
        (
            "candidate_duration_and_sample_count_preserved",
            comparison["duration_preserved"] and comparison["sample_count_preserved"],
        ),
        (
            "candidate_c2_and_closure_explicit",
            comparison["c2_preserved_by_affine_transformation"]
            and comparison["candidate_joint_closure_error_rad"] < 1e-12
            and comparison["candidate_velocity_closure_error_rad_s"] < 1e-12
            and comparison["candidate_acceleration_closure_error_rad_s2"] < 1e-12,
        ),
        (
            "reference_state_path_valid_in_125_model",
            reference_summary["status"] == "PASS"
            and reference_summary["abnormal_count"] == 0,
        ),
        (
            "context_proxy_warning_not_created_by_extension",
            reference_summary["context_warnings_above_native_count"] == 0,
        ),
        (
            "no_robot_or_hardware_import",
            imports.isdisjoint({"hardware", "control", "safety", "collection"}),
        ),
        (
            "stress_130_is_not_formal_condition",
            PROTOCOL["stress_only_not_formal_reference_eligible"],
        ),
        (
            "full_145_degree_domain_not_tested",
            not PROTOCOL["project_full_145_domain_tested"],
        ),
        (
            "formal_reference_not_replayed",
            not reference_summary["formal_reference_replayed"]
            and reference_summary["state_evaluation_only"],
        ),
    ]
    return {
        "status": "PASS" if all(value for _, value in tests) else "FAIL",
        "passed": sum(bool(value) for _, value in tests),
        "failed": sum(not bool(value) for _, value in tests),
        "tests": [
            {"test": test, "status": "PASS" if value else "FAIL"}
            for test, value in tests
        ],
    }


def report_text(
    outcome: str,
    metadata: dict[str, Any],
    common_summary: dict[str, Any],
    state_summary: dict[str, Any],
    continuity_summary: dict[str, Any],
    sweep_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    transformation: dict[str, Any],
    comparison: dict[str, Any],
    tests: dict[str, Any],
) -> str:
    original_condition = comparison["jacobian_condition_original_max"]
    candidate_condition = comparison["jacobian_condition_candidate_max"]
    recommended_primary = (
        "original formal reference on the limited-125 derived model, with the native-compatible candidate as a required sensitivity condition"
        if outcome == "LIMITED_125DEG_EXTENSION_MECHANICALLY_CONTINUOUS"
        else "native-compatible 119.5-degree diagnostic candidate only"
    )
    proceed = outcome == "LIMITED_125DEG_EXTENSION_MECHANICALLY_CONTINUOUS"
    return f"""# MYOLEG_KNEE_ROM_COMPATIBILITY_AUDIT_V1

## Decision

`{outcome}`

This is an offline numerical and structural continuity conclusion over the limited extension required by the frozen current reference. It is **not** physiological validation to 125 deg and does not support the project's full 145 deg search domain.

## Frozen protocol and provenance

- Protocol: `ROM_EXTENSION_PROTOCOL_V1`, SHA-256 `{metadata['protocol_sha256']}`.
- Primary derived upper limit: 125 deg, fixed before results.
- Baseline: actual native upper limit {metadata['native_knee_upper_deg']:.6f} deg.
- Stress-only upper limit: 130 deg; never used to authorize the formal reference.
- Native-compatible target: 119.5 deg, fixed before results.
- Frozen formal reference SHA-256: `{REFERENCE_SHA256}`.
- Native derived XML SHA-256: `{NATIVE_DERIVED_SHA256}`.
- 125 derived XML SHA-256: `{metadata['primary_125_xml_sha256']}`.
- 130 stress-only XML SHA-256: `{metadata['stress_130_xml_sha256']}`.
- Environment: Python {metadata['environment']['python']}, MyoSuite {metadata['environment']['myosuite']}, MuJoCo {metadata['environment']['mujoco']}.

The 125 and 130 XML files are generated from the frozen project-owned supine XML. The only XML content change is the upper range value of `knee_angle_r`. All 80 muscle actuators, 80 tendons, bodies, joints, 14 source knee/patella equalities, and the additional supine constraints remain present.

## Common-domain invariance

`LIMIT_EXTENSION_MODEL_INTEGRITY = {common_summary['status']}`

Across {common_summary['state_count']} matched states (four hips, native-compatible knee grid, and zero/positive/negative velocities), the largest differences were:

```text
{json.dumps(common_summary['maximum_absolute_differences'], indent=2, sort_keys=True)}
```

The predeclared absolute tolerance was {common_summary['absolute_tolerance']:.1e}. Thus changing the joint upper limit did not alter common-domain tendon length, actuator length, normalized length, moment arms, passive/actuator forces, or source equality residuals.

## 120-125 deg continuity

- Relevant right-side hip/knee actuators checked: {continuity_summary['related_right_actuator_count']}.
- Maximum geometry derivative-jump ratio at 120 deg: {continuity_summary['maximum_geometry_crossing_derivative_jump_ratio']:.6g} (limit {THRESHOLDS['geometry_crossing_derivative_jump_ratio_max']}).
- Maximum moment-arm derivative-jump ratio: {continuity_summary['maximum_moment_arm_crossing_derivative_jump_ratio']:.6g} (limit {THRESHOLDS['moment_arm_crossing_derivative_jump_ratio_max']}).
- Maximum muscle-force derivative-jump ratio: {continuity_summary['maximum_force_crossing_derivative_jump_ratio']:.6g} (limit {THRESHOLDS['force_crossing_derivative_jump_ratio_max']}).
- Passive knee-torque derivative-jump ratio: {continuity_summary['passive_torque_crossing_derivative_jump_ratio']:.6g} (limit {THRESHOLDS['passive_torque_crossing_derivative_jump_ratio_max']}).
- Maximum extension/native passive-force growth ratio: {continuity_summary['maximum_extended_to_native_force_growth_ratio']:.6g}.
- Normalized actuator-length range: {continuity_summary['normalized_actuator_length_min']:.6f} to {continuity_summary['normalized_actuator_length_max']:.6f}.
- Minimum tendon/actuator length: {continuity_summary['minimum_tendon_or_actuator_length_m']:.6f} m.
- State-grid rows: {state_summary['row_count']}; warnings: {state_summary['warning_count']}; all finite: {state_summary['all_finite']}.

The 130-deg stress-only grid contributed {state_summary['stress_only_rows']} rows, all finite={state_summary['stress_only_all_finite']}, warnings={state_summary['stress_only_warning_count']}, maximum source-equality residual={state_summary['stress_only_maximum_source_equality_residual']:.6g}, and minimum tendon length={state_summary['stress_only_minimum_tendon_length_m']:.6f} m. These observations are robustness context only and do not authorize 130 deg for the formal reference.

Crossing the original native range value did not introduce a detected discontinuity under the pre-frozen criteria. This means the existing geometry/equality polynomials extrapolate smoothly over the narrow interval; it does not establish anatomical validity outside the source model's native calibration.

## Patella/equality and low-control sweep

Four zero-muscle-control sweeps were driven at fixed hip 30/60/90/110 deg over `100 -> 125 -> 100 deg`. The driver used `qfrc_applied`, separately recorded from muscle actuator force.

- Total steps: {sweep_summary['total_steps']}.
- Maximum knee tracking error: {sweep_summary['maximum_tracking_error_deg']:.6f} deg.
- Maximum source equality residual: {sweep_summary['maximum_source_equality_residual']:.6g}.
- Maximum source equality force: {sweep_summary['maximum_source_equality_force']:.6g}.
- Maximum auxiliary round-trip inconsistency: {sweep_summary['maximum_auxiliary_roundtrip_inconsistency']:.6g}.
- Maximum required diagnostic torque: {sweep_summary['maximum_required_diagnostic_torque_nm']:.6g} N m.
- Solver warnings: {sweep_summary['warning_count']}; finite: {sweep_summary['all_finite']}.

The auxiliary/patella mechanism remained intact under the low-speed diagnostic sweep. These constraint forces are numerical model quantities, not robot or tissue loads.

## Current formal-reference state path

`{reference_summary['formal_status_label']}`

All {reference_summary['sample_count']} unchanged formal-reference states were evaluated without time integration or replay. The original {reference_summary['originally_above_native_count']} above-native points were explicitly tagged. Abnormal points: {reference_summary['abnormal_count']}; maximum source equality residual: {reference_summary['maximum_source_equality_residual']:.6g}; maximum force/model-Fmax ratio: {reference_summary['maximum_force_to_model_fmax_ratio']:.6g}; warnings: {reference_summary['warning_count']}.

The `actuator_lengthrange`-normalized transmission coordinate crossed the context band at {reference_summary['context_warning_count']} low-flexion native-domain states and at {reference_summary['context_warnings_above_native_count']} above-native states. This field is not physiological normalized muscle-fiber length and is therefore retained as context rather than used as a hard validity gate. All target-relevant physical tendon/actuator lengths stayed positive.

## Native-compatible diagnostic reference

The independent `MYOLEG_NATIVE_ROM_REFERENCE_CANDIDATE` uses the globally affine, invertible transformation:

```text
q_k,new(t) = q_k,0 + s * (q_k,formal(t) - q_k,0)
s = {transformation['amplitude_scale']:.12f}
```

Hip, duration (24 s), 401 samples, phase columns, extrema timing, asymmetric branch topology, starting pose, endpoint closure, and C2 continuity are preserved. No pointwise clipping is used. Candidate SHA-256: `{transformation['candidate_csv_sha256']}`.

Distortion relative to the formal reference:

- Knee RMS / maximum difference: {comparison['knee_rms_difference_deg']:.6f} / {comparison['knee_max_abs_difference_deg']:.6f} deg.
- Knee velocity RMS / maximum difference: {comparison['knee_velocity_rms_difference_deg_s']:.6f} / {comparison['knee_velocity_max_abs_difference_deg_s']:.6f} deg/s.
- Knee acceleration RMS / maximum difference: {comparison['knee_acceleration_rms_difference_deg_s2']:.6f} / {comparison['knee_acceleration_max_abs_difference_deg_s2']:.6f} deg/s2.
- Pull-point RMS / maximum displacement: {comparison['pull_point_rms_displacement_mm']:.6f} / {comparison['pull_point_max_displacement_mm']:.6f} mm.
- Project Jacobian maximum condition number, formal / candidate: {original_condition:.6f} / {candidate_condition:.6f}.
- Joint, velocity, and acceleration closure errors: {comparison['candidate_joint_closure_error_rad']:.3e}, {comparison['candidate_velocity_closure_error_rad_s']:.3e}, {comparison['candidate_acceleration_closure_error_rad_s2']:.3e}.

This candidate is `active_reference=false`, diagnostic-only, and not robot-approved. It does not replace the frozen formal reference.

## Direct answers

### Q1

{'Yes. The 120-125 deg region is mechanically continuous under the limited derived-model extension according to the frozen numerical and structural criteria.' if proceed else 'Not conclusively. The frozen criteria were not all satisfied.'}

### Q2

{'No discontinuity was detected in common-domain mechanics, muscle/tendon geometry, moment arms, passive force, or patella constraints at the original 120 deg boundary.' if proceed else 'At least one required continuity or integrity gate did not pass; the original limit cannot be treated as a smooth crossing.'}

### Q3

{'Yes, as an explicitly caveated offline modeling condition on the limited-125 derived model; it is not physiological validation.' if proceed else 'No. The unchanged reference should not be replayed on the limited extension.'}

### Q4

The 119.5-deg candidate differs by {comparison['knee_rms_difference_deg']:.6f} deg RMS and {comparison['knee_max_abs_difference_deg']:.6f} deg maximum at the knee, with {comparison['pull_point_rms_displacement_mm']:.6f} mm RMS pull-point displacement.

### Q5

Recommended primary condition for the next replay stage: **{recommended_primary}**.

### Q6

{'Yes, an offline MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1 may proceed only with both the original-on-125 primary condition and the native-compatible sensitivity condition predeclared.' if proceed else 'No. Do not proceed to original-reference replay; resolve the failed or inconclusive gate first.'}

No replay, candidate landscape, BO, PINN, RL, robot connection, or 145-deg model test was performed here.

## Tests

{tests['passed']} passed, {tests['failed']} failed. Stage-test status: `{tests['status']}`.
"""


def write_checksums() -> None:
    files = sorted(
        path
        for path in ARTIFACT_DIRECTORY.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    content = "".join(
        f"{sha256_file(path)}  {path.relative_to(ARTIFACT_DIRECTORY)}\n"
        for path in files
    )
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text(content, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACT_DIRECTORY / "ROM_EXTENSION_PROTOCOL.json", PROTOCOL)
    protocol_sha = sha256_file(ARTIFACT_DIRECTORY / "ROM_EXTENSION_PROTOCOL.json")
    environment = runtime_environment()
    identity_before = frozen_identity()
    native = mujoco.MjModel.from_xml_path(str(NATIVE_DERIVED_XML))
    native_knee_upper_deg = math.degrees(
        float(native.jnt_range[jid(native, TARGET_KNEE), 1])
    )
    primary = build_extension(PRIMARY_LIMIT_DEG, PRIMARY_125_XML)
    stress = build_extension(STRESS_LIMIT_DEG, STRESS_130_XML)
    diff_125 = xml_limit_only_diff(NATIVE_DERIVED_XML, PRIMARY_125_XML)
    diff_130 = xml_limit_only_diff(NATIVE_DERIVED_XML, STRESS_130_XML)
    write_json(
        ARTIFACT_DIRECTORY / "MODEL_EXTENSION_DIFF.json",
        {"primary_125": diff_125, "stress_only_130": diff_130},
    )
    if diff_125["status"] != "PASS" or diff_130["status"] != "PASS":
        raise RuntimeError("derived XML includes non-limit changes")
    if not (
        native.nu == primary.nu == stress.nu == 80
        and native.ntendon == primary.ntendon == stress.ntendon == 80
        and source_equality_hash(native)
        == source_equality_hash(primary)
        == source_equality_hash(stress)
    ):
        raise RuntimeError("extension destroyed muscle/tendon/equality structure")

    common_rows, common_summary = common_domain_audit(native, primary)
    write_csv(
        ARTIFACT_DIRECTORY / "NATIVE_VS_125_COMMON_DOMAIN.csv",
        common_rows,
        list(common_rows[0].keys()),
    )
    if common_summary["status"] != "PASS":
        raise RuntimeError("LIMIT_EXTENSION_MODEL_INTEGRITY = FAIL")

    state_rows, state_summary = state_grid_audit(
        {"NATIVE": native, "PRIMARY_125": primary, "STRESS_ONLY_130": stress}
    )
    write_csv(
        ARTIFACT_DIRECTORY / "KNEE_STATE_GRID_RESULTS.csv",
        state_rows,
        list(state_rows[0].keys()),
    )
    muscle_rows, moment_rows, passive_rows, continuity_summary = continuity_audit(
        primary
    )
    write_csv(
        ARTIFACT_DIRECTORY / "MUSCLE_TENDON_CONTINUITY.csv",
        muscle_rows,
        list(muscle_rows[0].keys()),
    )
    write_csv(
        ARTIFACT_DIRECTORY / "MOMENT_ARM_CONTINUITY.csv",
        moment_rows,
        list(moment_rows[0].keys()),
    )
    write_csv(
        ARTIFACT_DIRECTORY / "PASSIVE_FORCE_CONTINUITY.csv",
        passive_rows,
        list(passive_rows[0].keys()),
    )
    write_json(
        ARTIFACT_DIRECTORY / "CONTINUITY_SUMMARY.json", continuity_summary
    )

    sweep_rows, patella_rows, sweep_summary = low_control_sweep(primary)
    write_csv(
        ARTIFACT_DIRECTORY / "LOW_CONTROL_PASSIVE_SWEEP.csv",
        sweep_rows,
        list(sweep_rows[0].keys()),
    )
    write_csv(
        ARTIFACT_DIRECTORY / "PATELLA_EQUALITY_AUDIT.csv",
        patella_rows,
        list(patella_rows[0].keys()),
    )
    write_json(ARTIFACT_DIRECTORY / "SWEEP_SUMMARY.json", sweep_summary)

    reference_rows = read_reference()
    reference_audit_rows, reference_summary = reference_state_path_audit(
        primary, reference_rows
    )
    write_csv(
        ARTIFACT_DIRECTORY / "REFERENCE_125_STATE_PATH_AUDIT.csv",
        reference_audit_rows,
        list(reference_audit_rows[0].keys()),
    )
    write_json(
        ARTIFACT_DIRECTORY / "REFERENCE_125_STATE_PATH_SUMMARY.json",
        reference_summary,
    )
    candidate_path = ARTIFACT_DIRECTORY / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
    transformation, comparison = native_reference_candidate(
        reference_rows, candidate_path
    )
    write_json(
        ARTIFACT_DIRECTORY / "NATIVE_REFERENCE_TRANSFORMATION.json",
        transformation,
    )
    write_json(
        ARTIFACT_DIRECTORY / "REFERENCE_COMPARISON_METRICS.json", comparison
    )
    figure_paths = generate_figures(
        passive_rows,
        muscle_rows,
        moment_rows,
        patella_rows,
        reference_rows,
        candidate_path,
    )

    outcome = determine_outcome(
        common_summary,
        continuity_summary,
        state_summary,
        sweep_summary,
        reference_summary,
    )
    identity_after = frozen_identity()
    tests = run_tests(
        identity_before,
        identity_after,
        native,
        primary,
        diff_125,
        common_summary,
        state_summary,
        continuity_summary,
        sweep_summary,
        reference_summary,
        transformation,
        comparison,
    )
    write_json(ARTIFACT_DIRECTORY / "TEST_RESULTS.json", tests)
    metadata = {
        "stage_id": STAGE_ID,
        "evidence_level": "OFFLINE_ONLY_NOT_HUMAN_READY_NOT_ROBOT_APPROVED",
        "outcome": outcome,
        "protocol_sha256": protocol_sha,
        "builder_script_path": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "environment": environment,
        "frozen_identity": identity_after,
        "native_knee_upper_deg": native_knee_upper_deg,
        "primary_125_xml_path": str(PRIMARY_125_XML),
        "primary_125_xml_sha256": sha256_file(PRIMARY_125_XML),
        "stress_130_xml_path": str(STRESS_130_XML),
        "stress_130_xml_sha256": sha256_file(STRESS_130_XML),
        "native_dimensions": object_dimensions(native),
        "primary_125_dimensions": object_dimensions(primary),
        "stress_130_dimensions": object_dimensions(stress),
        "source_knee_equality_hash": source_equality_hash(native),
        "common_domain_summary": common_summary,
        "state_grid_summary": state_summary,
        "continuity_summary": continuity_summary,
        "sweep_summary": sweep_summary,
        "reference_path_summary": reference_summary,
        "native_reference_candidate_sha256": transformation[
            "candidate_csv_sha256"
        ],
        "figure_sha256": {
            str(path.relative_to(ARTIFACT_DIRECTORY)): sha256_file(path)
            for path in figure_paths
        },
        "formal_reference_modified": False,
        "formal_reference_replayed": False,
        "rom_protocol_v2_modified": False,
        "upstream_modified": False,
        "primary_extension_limit_selected_after_results": False,
        "stress_130_used_for_formal_decision": False,
        "full_145_domain_claimed": False,
        "candidate_landscape_generated": False,
        "bo_run": False,
        "pinn_trained": False,
        "rl_run": False,
        "robot_connected": False,
        "tests": tests,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(ARTIFACT_DIRECTORY / "metadata.json", metadata)
    report = report_text(
        outcome,
        metadata,
        common_summary,
        state_summary,
        continuity_summary,
        sweep_summary,
        reference_summary,
        transformation,
        comparison,
        tests,
    )
    (ARTIFACT_DIRECTORY / "MYOLEG_KNEE_ROM_COMPATIBILITY_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    write_checksums()
    if tests["status"] != "PASS":
        raise RuntimeError("stage validation failed")


if __name__ == "__main__":
    main()
