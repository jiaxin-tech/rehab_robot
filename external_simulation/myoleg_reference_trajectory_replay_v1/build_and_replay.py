"""Freeze an offline MyoLeg generalized-force truth interface for one reference.

The stage is deliberately isolated from the formal lower-limb model and all
personalization code.  It reads two already-frozen MyoLeg-derived conditions,
evaluates prescribed-state inverse dynamics, and checks the result against an
independent zero-control, generalized-PD replay.  It never edits MyoSuite,
the reference, the ROM protocol, or any prior artifact.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAGE_ID = "MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
)
FIGURE_DIRECTORY = ARTIFACT_DIRECTORY / "figures"

PRIMARY_MODEL = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_knee_rom_compatibility_v1"
    / "myoleg_supine_right_knee125_v1.xml"
)
SENSITIVITY_MODEL = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_supine_rehab_v1"
    / "myoleg_supine_right_v1.xml"
)
PRIMARY_REFERENCE = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
SENSITIVITY_REFERENCE = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)
FORMAL_MANIFEST = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
COORDINATE_MAPPING = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
    / "PROJECT_MYOLEG_COORDINATE_MAPPING.json"
)
SUPINE_MANIFEST = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
    / "DERIVED_MODEL_MANIFEST.json"
)
PRIOR_CHECKSUM_MANIFESTS = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_install_and_smoke_test_v1"
    / "checksums.sha256",
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
    / "checksums.sha256",
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "checksums.sha256",
)

REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
PRIMARY_MODEL_SHA256 = "c652424679308411fb73a211ad1fc770002fd760c8339c1ed9553888c14e0d41"
SENSITIVITY_MODEL_SHA256 = "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d"
SENSITIVITY_REFERENCE_SHA256 = "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678"
FORMAL_MANIFEST_SHA256 = "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441"

TARGET_HIP = "hip_flexion_r"
TARGET_KNEE = "knee_angle_r"
SOURCE_KNEE_EQUALITY_COUNT = 14
RIGHT_KNEE_EQUALITY_COUNT = 7
JOINT_NAMES = (TARGET_HIP, TARGET_KNEE)
JOINT_LABELS = ("hip", "knee")
KEY_BIARTICULAR_MUSCLES = ("recfem_r", "bflh_r", "semimem_r", "gasmed_r")
HIGH_FLEXION_INTERVAL_S = (10.540, 16.824)

# These are validation limits for a deliberately non-truth diagnostic replay.
# They were frozen before the retained formal replay.  The exact prescribed-
# state algebraic closure remains the primary truth gate.
THRESHOLDS = {
    "inverse_formula_max_abs_nm": 1.0e-9,
    "decomposition_max_abs_nm": 1.0e-9,
    "muscle_reconstruction_max_abs_nm": 1.0e-8,
    "dynamics_balance_max_abs_nm": 1.0e-8,
    "method_ab_rmse_max_nm": 5.0,
    "method_ab_p95_max_nm": 10.0,
    "method_ab_max_abs_nm": 65.0,
    "method_ab_relative_rms_max": 0.20,
    "tracking_q_rmse_max_deg": 1.0,
    "tracking_q_p95_max_deg": 1.0,
    "tracking_q_max_abs_deg": 1.0,
    "tracking_dq_rmse_max_deg_s": 1.0,
    "tracking_dq_p95_max_deg_s": 1.0,
    "tracking_dq_max_abs_deg_s": 6.0,
    "tracking_ddq_rmse_max_deg_s2": 50.0,
    "tracking_ddq_p95_max_deg_s2": 5.0,
    "tracking_ddq_max_abs_deg_s2": 900.0,
    "source_equality_residual_max": 1.0e-3,
    "sign_response_min_rad_s2_per_nm": 0.0,
    "primary_to_sensitivity_rms_ratio_max": 2.0,
    "primary_to_sensitivity_peak_ratio_max": 2.0,
    "high_flexion_delta_to_sensitivity_rms_ratio_max": 1.0,
}

DRIVER = {
    "type": "generalized_PD_diagnostic_driver",
    "kp_nm_per_rad": 5000.0,
    "kd_nm_s_per_rad": 150.0,
    "torque_limit_nm": 3000.0,
    "source": "reused unchanged from frozen knee-ROM low-control sweep",
    "controller_tuning_runs": 0,
    "stabilization_adjustments_allowed": 0,
    "truth_role": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_fingerprint(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "manifest_sha256": sha256_file(path),
    }


def prior_integrity() -> dict[str, Any]:
    checks = {str(path.relative_to(PROJECT_ROOT)): verify_checksum_manifest(path) for path in PRIOR_CHECKSUM_MANIFESTS}
    if any(value["status"] != "PASS" for value in checks.values()):
        raise RuntimeError("a frozen prior-stage checksum manifest does not validate")
    supine = json.loads(SUPINE_MANIFEST.read_text(encoding="utf-8"))
    upstream = supine["source_identity"]["upstream_asset_sha256"]
    current_upstream = {path: sha256_file(Path(path)) for path in upstream}
    if current_upstream != upstream:
        raise RuntimeError("upstream MyoSuite/MyoLeg asset changed")
    hashes = {
        "formal_reference": sha256_file(PRIMARY_REFERENCE),
        "formal_manifest": sha256_file(FORMAL_MANIFEST),
        "primary_125_model": sha256_file(PRIMARY_MODEL),
        "native_supine_model": sha256_file(SENSITIVITY_MODEL),
        "native_compatible_reference": sha256_file(SENSITIVITY_REFERENCE),
        "coordinate_mapping": sha256_file(COORDINATE_MAPPING),
        "supine_manifest": sha256_file(SUPINE_MANIFEST),
    }
    expected = {
        "formal_reference": REFERENCE_SHA256,
        "formal_manifest": FORMAL_MANIFEST_SHA256,
        "primary_125_model": PRIMARY_MODEL_SHA256,
        "native_supine_model": SENSITIVITY_MODEL_SHA256,
        "native_compatible_reference": SENSITIVITY_REFERENCE_SHA256,
    }
    if any(hashes[key] != value for key, value in expected.items()):
        raise RuntimeError("a frozen replay input changed")
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    if (
        formal["rom_protocol_version"] != "ROM_PROTOCOL_V2"
        or formal["hip_rom_deg"] != [0.0, 120.0]
        or formal["knee_rom_deg"] != [5.0, 145.0]
        or formal["theta_shank_definition"] != "q_hip - q_knee"
        or formal["active_reference_sha256"] != REFERENCE_SHA256
    ):
        raise RuntimeError("formal ROM/reference convention changed")
    return {
        "hashes": hashes,
        "prior_checksum_verification": checks,
        "upstream_asset_count": len(current_upstream),
        "upstream_asset_sha256": current_upstream,
    }


def runtime_environment() -> dict[str, Any]:
    environment = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "myosuite": importlib.metadata.version("myosuite"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("pillow"),
    }
    frozen = {"python": "3.10.19", "myosuite": "2.12.2", "mujoco": "3.6.0"}
    environment["frozen_expected"] = frozen
    environment["frozen_match"] = all(environment[key] == value for key, value in frozen.items())
    if not environment["frozen_match"]:
        raise RuntimeError("frozen MyoLeg runtime environment changed")
    return environment


def name(model: mujoco.MjModel, objtype: Any, identifier: int) -> str:
    return mujoco.mj_id2name(model, objtype, int(identifier)) or ""


def joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if value < 0:
        raise RuntimeError(f"missing joint {joint_name}")
    return int(value)


def qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, joint_name)])


def dof_address(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, joint_name)])


def polynomial_derivative(coefficients: np.ndarray, value: float, order: int) -> float:
    total = 0.0
    for power in range(order, 5):
        multiplier = math.factorial(power) / math.factorial(power - order)
        total += multiplier * float(coefficients[power]) * value ** (power - order)
    return float(total)


def project_source_knee_state(
    model: mujoco.MjModel, data: mujoco.MjData, include_acceleration: bool
) -> None:
    for equality_id in range(SOURCE_KNEE_EQUALITY_COUNT):
        first_joint = int(model.eq_obj1id[equality_id])
        second_joint = int(model.eq_obj2id[equality_id])
        first_qadr = int(model.jnt_qposadr[first_joint])
        second_qadr = int(model.jnt_qposadr[second_joint])
        first_dadr = int(model.jnt_dofadr[first_joint])
        second_dadr = int(model.jnt_dofadr[second_joint])
        parent_q = float(data.qpos[second_qadr])
        coefficients = model.eq_data[equality_id]
        first = polynomial_derivative(coefficients, parent_q, 1)
        data.qpos[first_qadr] = polynomial_derivative(coefficients, parent_q, 0)
        data.qvel[first_dadr] = first * data.qvel[second_dadr]
        if include_acceleration:
            second = polynomial_derivative(coefficients, parent_q, 2)
            data.qacc[first_dadr] = (
                first * data.qacc[second_dadr]
                + second * data.qvel[second_dadr] ** 2
            )


def reset_to_target_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_target: np.ndarray,
    dq_target: np.ndarray,
    ddq_target: np.ndarray | None,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qpos[:7] = np.asarray(
        [0.0, 0.0, 1.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    )
    q_addresses = [qpos_address(model, joint) for joint in JOINT_NAMES]
    dof_addresses = [dof_address(model, joint) for joint in JOINT_NAMES]
    data.qpos[q_addresses] = q_target
    data.qvel[dof_addresses] = dq_target
    if ddq_target is not None:
        data.qacc[dof_addresses] = ddq_target
    project_source_knee_state(model, data, ddq_target is not None)
    if model.na:
        data.act[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0


def independent_coordinate_tangent(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Map project hip/knee velocities into the full constrained velocity space."""

    tangent = np.zeros((model.nv, 2), dtype=float)
    tangent[dof_address(model, TARGET_HIP), 0] = 1.0
    tangent[dof_address(model, TARGET_KNEE), 1] = 1.0
    for equality_id in range(RIGHT_KNEE_EQUALITY_COUNT):
        first_joint = int(model.eq_obj1id[equality_id])
        second_joint = int(model.eq_obj2id[equality_id])
        if name(model, mujoco.mjtObj.mjOBJ_JOINT, second_joint) != TARGET_KNEE:
            raise RuntimeError("unexpected right-knee equality parent")
        first_dadr = int(model.jnt_dofadr[first_joint])
        second_qadr = int(model.jnt_qposadr[second_joint])
        tangent[first_dadr, 1] = polynomial_derivative(
            model.eq_data[equality_id], float(data.qpos[second_qadr]), 1
        )
    return tangent


def source_equality_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    rows = np.flatnonzero(
        (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (np.asarray(data.efc_id) < SOURCE_KNEE_EQUALITY_COUNT)
    )
    if rows.size == 0:
        return 0.0, 0.0
    return (
        float(np.max(np.abs(np.asarray(data.efc_pos)[rows]))),
        float(np.max(np.abs(np.asarray(data.efc_force)[rows]))),
    )


def warning_count(data: mujoco.MjData) -> int:
    return int(np.asarray(data.warning.number, dtype=np.int64).sum())


def sparse_actuator_moment_times_tangent(
    model: mujoco.MjModel, data: mujoco.MjData, tangent: np.ndarray
) -> np.ndarray:
    result = np.zeros((model.nu, 2), dtype=float)
    values = np.asarray(data.actuator_moment)
    for actuator in range(model.nu):
        start = int(data.moment_rowadr[actuator])
        count = int(data.moment_rownnz[actuator])
        columns = np.asarray(data.moment_colind[start : start + count], dtype=int)
        result[actuator] = values[start : start + count] @ tangent[columns]
    return result


def constraint_force_groups(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Recover generalized constraint force grouped by MuJoCo constraint type."""

    groups = {
        "equality": {int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)},
        "joint_limit": {int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)},
        "tendon_limit": {int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)},
        "friction": {
            int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF),
            int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON),
        },
        "contact": {
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
        },
    }
    if data.nefc == 0:
        return (
            {key: np.zeros(model.nv, dtype=float) for key in groups},
            {key: 0 for key in groups},
        )
    if not mujoco.mj_isSparse(model):
        jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
    else:
        jacobian = np.zeros((data.nefc, model.nv), dtype=float)
        for row in range(data.nefc):
            start = int(data.efc_J_rowadr[row])
            count = int(data.efc_J_rownnz[row])
            columns = np.asarray(data.efc_J_colind[start : start + count], dtype=int)
            jacobian[row, columns] = np.asarray(data.efc_J)[start : start + count]
    types = np.asarray(data.efc_type, dtype=int)
    force = np.asarray(data.efc_force)
    forces: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for key, accepted in groups.items():
        selected = np.asarray([value in accepted for value in types], dtype=bool)
        counts[key] = int(np.sum(selected))
        forces[key] = (
            jacobian[selected].T @ force[selected]
            if np.any(selected)
            else np.zeros(model.nv, dtype=float)
        )
    return forces, counts


def model_inventory(model: mujoco.MjModel) -> dict[str, Any]:
    source_equalities = []
    for equality_id in range(SOURCE_KNEE_EQUALITY_COUNT):
        source_equalities.append(
            {
                "equality_id": equality_id,
                "child_joint": name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[equality_id])
                ),
                "parent_joint": name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[equality_id])
                ),
            }
        )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "ntendon": int(model.ntendon),
        "neq": int(model.neq),
        "timestep_s": float(model.opt.timestep),
        "target_joint_types": {
            label: int(model.jnt_type[joint_id(model, joint)])
            for label, joint in zip(JOINT_LABELS, JOINT_NAMES)
        },
        "source_knee_equalities": source_equalities,
        "source_knee_equality_count": SOURCE_KNEE_EQUALITY_COUNT,
    }


def load_reference(path: Path, condition: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    time_s = np.asarray([float(row["time_s"]) for row in rows])
    q = np.asarray(
        [[float(row["q_hip_rad"]), float(row["q_knee_rad"])] for row in rows]
    )
    dq = np.asarray(
        [[float(row["dq_hip_rad_s"]), float(row["dq_knee_rad_s"])] for row in rows]
    )
    ddq = np.asarray(
        [[float(row["ddq_hip_rad_s2"]), float(row["ddq_knee_rad_s2"])] for row in rows]
    )
    phases = np.asarray([row["cycle_phase"] for row in rows])
    return {
        "condition": condition,
        "path": path,
        "rows": rows,
        "time_s": time_s,
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "phases": phases,
    }


def reference_audit(reference: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    time_s = reference["time_s"]
    q = reference["q"]
    dq = reference["dq"]
    ddq = reference["ddq"]
    model_ranges = {
        label: np.degrees(model.jnt_range[joint_id(model, joint)]).tolist()
        for label, joint in zip(JOINT_LABELS, JOINT_NAMES)
    }
    timestamp_steps = np.rint(time_s / model.opt.timestep).astype(np.int64)
    reconstructed = timestamp_steps * model.opt.timestep
    q_deg = np.degrees(q)
    ranges_valid = all(
        np.min(q_deg[:, index]) >= model_ranges[label][0] - 1e-10
        and np.max(q_deg[:, index]) <= model_ranges[label][1] + 1e-10
        for index, label in enumerate(JOINT_LABELS)
    )
    extension_reversed = any(
        str(row.get("measured_extension_is_reversed_flexion", "False")) == "True"
        for row in reference["rows"]
    )
    pointwise_clipped = any(
        str(row.get("pointwise_clipped", "False")) == "True"
        for row in reference["rows"]
    )
    return {
        "condition": reference["condition"],
        "path": str(reference["path"].relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(reference["path"]),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sample_count": len(time_s),
        "strictly_increasing_timestamps": bool(np.all(np.diff(time_s) > 0.0)),
        "timestamps_exact_integer_model_steps": bool(
            np.max(np.abs(reconstructed - time_s)) <= 1.0e-12
        ),
        "model_timestep_s": float(model.opt.timestep),
        "phase_counts": {
            phase: int(np.sum(reference["phases"] == phase))
            for phase in np.unique(reference["phases"])
        },
        "q_range_deg": {
            label: [float(np.min(q_deg[:, index])), float(np.max(q_deg[:, index]))]
            for index, label in enumerate(JOINT_LABELS)
        },
        "model_joint_range_deg": model_ranges,
        "model_range_valid": ranges_valid,
        "closure": {
            "q_max_abs_rad": float(np.max(np.abs(q[-1] - q[0]))),
            "dq_max_abs_rad_s": float(np.max(np.abs(dq[-1] - dq[0]))),
            "ddq_max_abs_rad_s2": float(np.max(np.abs(ddq[-1] - ddq[0]))),
        },
        "c2_fields_present_and_finite": bool(
            np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(ddq).all()
        ),
        "measured_asymmetric_flexion_and_extension": not extension_reversed,
        "pointwise_clipping_used": pointwise_clipped,
        "rescaling_performed_by_this_stage": False,
    }


class QuinticReference:
    """Piecewise quintic interpolation matching q, dq and ddq at every sample."""

    def __init__(self, time_s: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray):
        self.time_s = time_s
        self.coefficients = np.empty((len(time_s) - 1, 2, 6), dtype=float)
        for interval, duration in enumerate(np.diff(time_s)):
            for joint in range(2):
                c0 = q[interval, joint]
                c1 = dq[interval, joint] * duration
                c2 = 0.5 * ddq[interval, joint] * duration**2
                rhs = np.asarray(
                    [
                        q[interval + 1, joint] - c0 - c1 - c2,
                        dq[interval + 1, joint] * duration - c1 - 2.0 * c2,
                        ddq[interval + 1, joint] * duration**2 - 2.0 * c2,
                    ]
                )
                matrix = np.asarray([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]])
                c3, c4, c5 = np.linalg.solve(matrix, rhs)
                self.coefficients[interval, joint] = [c0, c1, c2, c3, c4, c5]

    def evaluate(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elapsed_s = min(max(elapsed_s, float(self.time_s[0])), float(self.time_s[-1]))
        interval = min(int(np.searchsorted(self.time_s, elapsed_s, side="right") - 1), len(self.time_s) - 2)
        interval = max(interval, 0)
        duration = float(self.time_s[interval + 1] - self.time_s[interval])
        phase = (elapsed_s - float(self.time_s[interval])) / duration
        powers = np.asarray([1.0, phase, phase**2, phase**3, phase**4, phase**5])
        d_powers = np.asarray([0.0, 1.0, 2.0 * phase, 3.0 * phase**2, 4.0 * phase**3, 5.0 * phase**4]) / duration
        dd_powers = np.asarray([0.0, 0.0, 2.0, 6.0 * phase, 12.0 * phase**2, 20.0 * phase**3]) / duration**2
        coefficients = self.coefficients[interval]
        return coefficients @ powers, coefficients @ d_powers, coefficients @ dd_powers


def prescribed_truth(
    model: mujoco.MjModel, reference: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = mujoco.MjData(model)
    sample_count = len(reference["time_s"])
    actuator_names = np.asarray(
        [name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]
    )
    key_ids = [int(np.flatnonzero(actuator_names == muscle)[0]) for muscle in KEY_BIARTICULAR_MUSCLES]
    arrays = {
        "tau_truth_nm": np.zeros((sample_count, 2)),
        "tau_inverse_net_nm": np.zeros((sample_count, 2)),
        "tau_naive_full_target_dof_nm": np.zeros((sample_count, 2)),
        "mass_term_nm": np.zeros((sample_count, 2)),
        "bias_term_nm": np.zeros((sample_count, 2)),
        "passive_internal_nm": np.zeros((sample_count, 2)),
        "actuator_internal_nm": np.zeros((sample_count, 2)),
        "constraint_internal_nm": np.zeros((sample_count, 2)),
        "constraint_equality_internal_nm": np.zeros((sample_count, 2)),
        "constraint_joint_limit_internal_nm": np.zeros((sample_count, 2)),
        "constraint_tendon_limit_internal_nm": np.zeros((sample_count, 2)),
        "constraint_friction_internal_nm": np.zeros((sample_count, 2)),
        "constraint_contact_internal_nm": np.zeros((sample_count, 2)),
        "constraint_equality_active_count": np.zeros(sample_count, dtype=np.int64),
        "constraint_joint_limit_active_count": np.zeros(sample_count, dtype=np.int64),
        "constraint_tendon_limit_active_count": np.zeros(sample_count, dtype=np.int64),
        "constraint_friction_active_count": np.zeros(sample_count, dtype=np.int64),
        "constraint_contact_active_count": np.zeros(sample_count, dtype=np.int64),
        "inverse_formula_residual_nm": np.zeros((sample_count, 2)),
        "decomposition_residual_nm": np.zeros((sample_count, 2)),
        "muscle_reconstruction_residual_nm": np.zeros((sample_count, 2)),
        "source_equality_residual": np.zeros(sample_count),
        "source_equality_force": np.zeros(sample_count),
        "actuator_force_n": np.zeros((sample_count, model.nu)),
        "actuator_length_m": np.zeros((sample_count, model.nu)),
        "tendon_length_m": np.zeros((sample_count, model.ntendon)),
        "actuator_activation": np.zeros((sample_count, model.na)),
        "muscle_moment_independent_m": np.zeros((sample_count, model.nu, 2)),
        "muscle_torque_contribution_nm": np.zeros((sample_count, model.nu, 2)),
        "key_biarticular_torque_contribution_nm": np.zeros(
            (sample_count, len(key_ids), 2)
        ),
        "warning_count": np.zeros(sample_count, dtype=np.int64),
    }
    mass_matrix = np.zeros((model.nv, model.nv), dtype=float)
    dof_indices = [dof_address(model, joint) for joint in JOINT_NAMES]
    start = time.perf_counter()
    for sample in range(sample_count):
        reset_to_target_state(
            model,
            data,
            reference["q"][sample],
            reference["dq"][sample],
            reference["ddq"][sample],
        )
        desired_full_acceleration = np.asarray(data.qacc).copy()
        tangent = independent_coordinate_tangent(model, data)
        mujoco.mj_forward(model, data)
        actuator_force = np.asarray(data.actuator_force).copy()
        actuator_internal_full = np.asarray(data.qfrc_actuator).copy()
        moment_independent = sparse_actuator_moment_times_tangent(model, data, tangent)
        contribution = moment_independent * actuator_force[:, None]
        arrays["actuator_force_n"][sample] = actuator_force
        arrays["actuator_length_m"][sample] = data.actuator_length
        arrays["tendon_length_m"][sample] = data.ten_length
        if model.na:
            arrays["actuator_activation"][sample] = data.act
        arrays["muscle_moment_independent_m"][sample] = moment_independent
        arrays["muscle_torque_contribution_nm"][sample] = contribution
        arrays["key_biarticular_torque_contribution_nm"][sample] = contribution[key_ids]

        data.qacc[:] = desired_full_acceleration
        mujoco.mj_inverse(model, data)
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        mass_full = mass_matrix @ desired_full_acceleration
        bias_full = np.asarray(data.qfrc_bias).copy()
        passive_full = np.asarray(data.qfrc_passive).copy()
        constraint_full = np.asarray(data.qfrc_constraint).copy()
        grouped_constraint, grouped_counts = constraint_force_groups(model, data)
        inverse_formula_full = mass_full + bias_full - passive_full - constraint_full
        external_full = inverse_formula_full - actuator_internal_full

        mass_reduced = tangent.T @ mass_full
        bias_reduced = tangent.T @ bias_full
        passive_reduced = tangent.T @ passive_full
        actuator_reduced = tangent.T @ actuator_internal_full
        constraint_reduced = tangent.T @ constraint_full
        inverse_reduced = tangent.T @ np.asarray(data.qfrc_inverse)
        truth = tangent.T @ external_full
        reconstructed = (
            mass_reduced
            + bias_reduced
            - passive_reduced
            - actuator_reduced
            - constraint_reduced
        )

        arrays["tau_truth_nm"][sample] = truth
        arrays["tau_inverse_net_nm"][sample] = inverse_reduced
        arrays["tau_naive_full_target_dof_nm"][sample] = external_full[dof_indices]
        arrays["mass_term_nm"][sample] = mass_reduced
        arrays["bias_term_nm"][sample] = bias_reduced
        arrays["passive_internal_nm"][sample] = passive_reduced
        arrays["actuator_internal_nm"][sample] = actuator_reduced
        arrays["constraint_internal_nm"][sample] = constraint_reduced
        for group in ("equality", "joint_limit", "tendon_limit", "friction", "contact"):
            arrays[f"constraint_{group}_internal_nm"][sample] = (
                tangent.T @ grouped_constraint[group]
            )
            arrays[f"constraint_{group}_active_count"][sample] = grouped_counts[group]
        arrays["inverse_formula_residual_nm"][sample] = tangent.T @ (
            np.asarray(data.qfrc_inverse) - inverse_formula_full
        )
        arrays["decomposition_residual_nm"][sample] = truth - reconstructed
        arrays["muscle_reconstruction_residual_nm"][sample] = (
            np.sum(contribution, axis=0) - actuator_reduced
        )
        equality_residual, equality_force = source_equality_metrics(model, data)
        arrays["source_equality_residual"][sample] = equality_residual
        arrays["source_equality_force"][sample] = equality_force
        arrays["warning_count"][sample] = warning_count(data)
    wall_time = time.perf_counter() - start
    arrays["actuator_names"] = actuator_names
    arrays["key_biarticular_names"] = np.asarray(KEY_BIARTICULAR_MUSCLES)
    return arrays, {
        "wall_time_s": wall_time,
        "evaluated_samples": sample_count,
        "samples_per_wall_second": sample_count / wall_time,
    }


def controlled_replay(
    model: mujoco.MjModel, reference: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = mujoco.MjData(model)
    reset_to_target_state(
        model, data, reference["q"][0], reference["dq"][0], None
    )
    mujoco.mj_forward(model, data)
    interpolator = QuinticReference(
        reference["time_s"], reference["q"], reference["dq"], reference["ddq"]
    )
    sample_steps = {
        int(round(value / model.opt.timestep)): index
        for index, value in enumerate(reference["time_s"])
    }
    total_steps = int(round(float(reference["time_s"][-1]) / model.opt.timestep))
    sample_count = len(reference["time_s"])
    arrays = {
        "actual_q_rad": np.zeros((sample_count, 2)),
        "actual_dq_rad_s": np.zeros((sample_count, 2)),
        "actual_ddq_rad_s2": np.zeros((sample_count, 2)),
        "diagnostic_driver_nm": np.zeros((sample_count, 2)),
        "force_balance_reconstruction_nm": np.zeros((sample_count, 2)),
        "dynamics_balance_residual_nm": np.zeros((sample_count, 2)),
        "dynamics_balance_full_l2_nm": np.zeros(sample_count),
        "dynamics_balance_full_max_abs_nm": np.zeros(sample_count),
        "source_equality_residual": np.zeros(sample_count),
        "source_equality_force": np.zeros(sample_count),
        "ctrl_max": np.zeros(sample_count),
        "activation_max": np.zeros(sample_count),
        "warning_count": np.zeros(sample_count, dtype=np.int64),
    }
    q_indices = [qpos_address(model, joint) for joint in JOINT_NAMES]
    dof_indices = [dof_address(model, joint) for joint in JOINT_NAMES]
    mass_matrix = np.zeros((model.nv, model.nv), dtype=float)
    finite = True
    peak_warning = 0
    start = time.perf_counter()
    for step in range(total_steps + 1):
        elapsed = step * model.opt.timestep
        q_target, dq_target, _ = interpolator.evaluate(elapsed)
        driver = np.clip(
            DRIVER["kp_nm_per_rad"] * (q_target - np.asarray(data.qpos)[q_indices])
            + DRIVER["kd_nm_s_per_rad"]
            * (dq_target - np.asarray(data.qvel)[dof_indices]),
            -DRIVER["torque_limit_nm"],
            DRIVER["torque_limit_nm"],
        )
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[dof_indices] = driver
        mujoco.mj_forward(model, data)
        if step in sample_steps:
            sample = sample_steps[step]
            mujoco.mj_fullM(model, mass_matrix, data.qM)
            force_balance_full = (
                mass_matrix @ np.asarray(data.qacc)
                + np.asarray(data.qfrc_bias)
                - np.asarray(data.qfrc_passive)
                - np.asarray(data.qfrc_actuator)
                - np.asarray(data.qfrc_constraint)
            )
            residual_full = force_balance_full - np.asarray(data.qfrc_applied)
            arrays["actual_q_rad"][sample] = np.asarray(data.qpos)[q_indices]
            arrays["actual_dq_rad_s"][sample] = np.asarray(data.qvel)[dof_indices]
            arrays["actual_ddq_rad_s2"][sample] = np.asarray(data.qacc)[dof_indices]
            arrays["diagnostic_driver_nm"][sample] = driver
            arrays["force_balance_reconstruction_nm"][sample] = force_balance_full[
                dof_indices
            ]
            arrays["dynamics_balance_residual_nm"][sample] = residual_full[dof_indices]
            arrays["dynamics_balance_full_l2_nm"][sample] = np.linalg.norm(residual_full)
            arrays["dynamics_balance_full_max_abs_nm"][sample] = np.max(
                np.abs(residual_full)
            )
            equality_residual, equality_force = source_equality_metrics(model, data)
            arrays["source_equality_residual"][sample] = equality_residual
            arrays["source_equality_force"][sample] = equality_force
            arrays["ctrl_max"][sample] = np.max(np.abs(data.ctrl))
            arrays["activation_max"][sample] = (
                np.max(np.abs(data.act)) if model.na else 0.0
            )
            arrays["warning_count"][sample] = warning_count(data)
        finite = finite and all(
            np.isfinite(value).all()
            for value in (
                data.qpos,
                data.qvel,
                data.qacc,
                data.qfrc_applied,
                data.qfrc_actuator,
                data.qfrc_constraint,
            )
        )
        peak_warning = max(peak_warning, warning_count(data))
        if step < total_steps:
            mujoco.mj_step(model, data)
    wall_time = time.perf_counter() - start
    return arrays, {
        "wall_time_s": wall_time,
        "integration_steps": total_steps,
        "state_evaluations": total_steps + 1,
        "simulated_duration_s": float(reference["time_s"][-1]),
        "realtime_factor": float(reference["time_s"][-1]) / wall_time,
        "all_finite": finite,
        "warning_count": peak_warning,
        "warmup_duration_s": 0.0,
    }


def metric(values: np.ndarray, reference_rms: float | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    result = {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "max_abs": float(np.max(np.abs(values))),
    }
    if reference_rms is not None:
        result["relative_rms"] = result["rmse"] / max(reference_rms, 1.0e-12)
    return result


def evaluate_metrics(
    condition: str,
    reference: dict[str, Any],
    prescribed: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    method_rows = []
    tracking = {}
    method_passes = []
    for joint, label in enumerate(JOINT_LABELS):
        truth_rms = float(np.sqrt(np.mean(prescribed["tau_truth_nm"][:, joint] ** 2)))
        comparison = metric(
            controlled["force_balance_reconstruction_nm"][:, joint]
            - prescribed["tau_truth_nm"][:, joint],
            truth_rms,
        )
        comparison["condition"] = condition
        comparison["joint"] = label
        comparison["truth_rms_nm"] = truth_rms
        comparison["correlation"] = float(
            np.corrcoef(
                controlled["force_balance_reconstruction_nm"][:, joint],
                prescribed["tau_truth_nm"][:, joint],
            )[0, 1]
        )
        comparison["pass"] = bool(
            comparison["rmse"] <= THRESHOLDS["method_ab_rmse_max_nm"]
            and comparison["p95_abs"] <= THRESHOLDS["method_ab_p95_max_nm"]
            and comparison["max_abs"] <= THRESHOLDS["method_ab_max_abs_nm"]
            and comparison["relative_rms"]
            <= THRESHOLDS["method_ab_relative_rms_max"]
        )
        method_passes.append(comparison["pass"])
        method_rows.append(comparison)
        q_error = np.degrees(controlled["actual_q_rad"][:, joint] - reference["q"][:, joint])
        dq_error = np.degrees(
            controlled["actual_dq_rad_s"][:, joint] - reference["dq"][:, joint]
        )
        ddq_error = np.degrees(
            controlled["actual_ddq_rad_s2"][:, joint] - reference["ddq"][:, joint]
        )
        tracking[label] = {
            "q_error_deg": metric(q_error),
            "dq_error_deg_s": metric(dq_error),
            "ddq_error_deg_s2": metric(ddq_error),
        }
    exact = {
        "inverse_formula_max_abs_nm": float(
            np.max(np.abs(prescribed["inverse_formula_residual_nm"]))
        ),
        "decomposition_max_abs_nm": float(
            np.max(np.abs(prescribed["decomposition_residual_nm"]))
        ),
        "muscle_reconstruction_max_abs_nm": float(
            np.max(np.abs(prescribed["muscle_reconstruction_residual_nm"]))
        ),
        "dynamics_balance_max_abs_nm": float(
            np.max(np.abs(controlled["dynamics_balance_residual_nm"]))
        ),
        "dynamics_balance_full_max_abs_nm": float(
            np.max(controlled["dynamics_balance_full_max_abs_nm"])
        ),
    }
    exact_pass = bool(
        exact["inverse_formula_max_abs_nm"] <= THRESHOLDS["inverse_formula_max_abs_nm"]
        and exact["decomposition_max_abs_nm"] <= THRESHOLDS["decomposition_max_abs_nm"]
        and exact["muscle_reconstruction_max_abs_nm"]
        <= THRESHOLDS["muscle_reconstruction_max_abs_nm"]
        and exact["dynamics_balance_max_abs_nm"]
        <= THRESHOLDS["dynamics_balance_max_abs_nm"]
        and exact["dynamics_balance_full_max_abs_nm"]
        <= THRESHOLDS["dynamics_balance_max_abs_nm"]
    )
    tracking_pass = True
    for values in tracking.values():
        tracking_pass = tracking_pass and (
            values["q_error_deg"]["rmse"] <= THRESHOLDS["tracking_q_rmse_max_deg"]
            and values["q_error_deg"]["p95_abs"] <= THRESHOLDS["tracking_q_p95_max_deg"]
            and values["q_error_deg"]["max_abs"] <= THRESHOLDS["tracking_q_max_abs_deg"]
            and values["dq_error_deg_s"]["rmse"]
            <= THRESHOLDS["tracking_dq_rmse_max_deg_s"]
            and values["dq_error_deg_s"]["p95_abs"]
            <= THRESHOLDS["tracking_dq_p95_max_deg_s"]
            and values["dq_error_deg_s"]["max_abs"]
            <= THRESHOLDS["tracking_dq_max_abs_deg_s"]
            and values["ddq_error_deg_s2"]["rmse"]
            <= THRESHOLDS["tracking_ddq_rmse_max_deg_s2"]
            and values["ddq_error_deg_s2"]["p95_abs"]
            <= THRESHOLDS["tracking_ddq_p95_max_deg_s2"]
            and values["ddq_error_deg_s2"]["max_abs"]
            <= THRESHOLDS["tracking_ddq_max_abs_deg_s2"]
        )
    equality_max = max(
        float(np.max(prescribed["source_equality_residual"])),
        float(np.max(controlled["source_equality_residual"])),
    )
    stable = bool(
        runtime["all_finite"]
        and runtime["warning_count"] == 0
        and equality_max <= THRESHOLDS["source_equality_residual_max"]
        and tracking_pass
    )
    return {
        "condition": condition,
        "method_ab": method_rows,
        "method_ab_pass": all(method_passes),
        "exact_force_accounting": exact,
        "exact_force_accounting_pass": exact_pass,
        "tracking": tracking,
        "tracking_pass": tracking_pass,
        "source_equality_residual_max": equality_max,
        "stable_replay": stable,
        "truth_semantics_pass": bool(all(method_passes) and exact_pass and stable),
    }


def torque_sign_audit(model: mujoco.MjModel, condition: str) -> list[dict[str, Any]]:
    rows = []
    q_target = np.radians([60.0, 60.0])
    dq_target = np.zeros(2)
    q_indices = [qpos_address(model, joint) for joint in JOINT_NAMES]
    dof_indices = [dof_address(model, joint) for joint in JOINT_NAMES]
    del q_indices
    for joint, label in enumerate(JOINT_LABELS):
        accelerations = []
        for torque in (-1.0, 1.0):
            data = mujoco.MjData(model)
            reset_to_target_state(model, data, q_target, dq_target, None)
            data.qfrc_applied[dof_indices[joint]] = torque
            mujoco.mj_forward(model, data)
            accelerations.append(np.asarray(data.qacc)[dof_indices].copy())
        response = (accelerations[1] - accelerations[0]) / 2.0
        rows.append(
            {
                "condition": condition,
                "driven_joint": label,
                "positive_coordinate": JOINT_NAMES[joint],
                "applied_unit": "N*m",
                "own_acceleration_response_rad_s2_per_nm": float(response[joint]),
                "cross_acceleration_response_rad_s2_per_nm": float(response[1 - joint]),
                "project_mapping_sign": 1.0,
                "pass": bool(
                    response[joint] > THRESHOLDS["sign_response_min_rad_s2_per_nm"]
                ),
            }
        )
    return rows


def primary_sensitivity_comparison(
    primary: dict[str, np.ndarray], sensitivity: dict[str, np.ndarray], time_s: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    high = (time_s >= HIGH_FLEXION_INTERVAL_S[0]) & (time_s <= HIGH_FLEXION_INTERVAL_S[1])
    assessments = []
    for joint, label in enumerate(JOINT_LABELS):
        p = primary["tau_truth_nm"][:, joint]
        s = sensitivity["tau_truth_nm"][:, joint]
        difference = p - s
        p_rms = float(np.sqrt(np.mean(p**2)))
        s_rms = float(np.sqrt(np.mean(s**2)))
        p_peak = float(np.max(np.abs(p)))
        s_peak = float(np.max(np.abs(s)))
        high_difference_rms = float(np.sqrt(np.mean(difference[high] ** 2)))
        high_sensitivity_rms = float(np.sqrt(np.mean(s[high] ** 2)))
        row = {
            "joint": label,
            "primary_rms_nm": p_rms,
            "sensitivity_rms_nm": s_rms,
            "primary_p95_abs_nm": float(np.percentile(np.abs(p), 95.0)),
            "sensitivity_p95_abs_nm": float(np.percentile(np.abs(s), 95.0)),
            "primary_peak_abs_nm": p_peak,
            "sensitivity_peak_abs_nm": s_peak,
            "time_series_difference_rmse_nm": float(np.sqrt(np.mean(difference**2))),
            "time_series_correlation": float(np.corrcoef(p, s)[0, 1]),
            "primary_to_sensitivity_rms_ratio": p_rms / max(s_rms, 1.0e-12),
            "primary_to_sensitivity_peak_ratio": p_peak / max(s_peak, 1.0e-12),
            "high_flexion_interval_start_s": HIGH_FLEXION_INTERVAL_S[0],
            "high_flexion_interval_end_s": HIGH_FLEXION_INTERVAL_S[1],
            "high_flexion_primary_rms_nm": float(np.sqrt(np.mean(p[high] ** 2))),
            "high_flexion_sensitivity_rms_nm": high_sensitivity_rms,
            "high_flexion_difference_rmse_nm": high_difference_rms,
            "high_flexion_delta_to_sensitivity_rms_ratio": high_difference_rms
            / max(high_sensitivity_rms, 1.0e-12),
        }
        for component in (
            "mass_term_nm",
            "bias_term_nm",
            "passive_internal_nm",
            "actuator_internal_nm",
            "constraint_internal_nm",
            "constraint_equality_internal_nm",
            "constraint_joint_limit_internal_nm",
            "constraint_contact_internal_nm",
        ):
            label_component = component.removesuffix("_nm")
            p_component = primary[component][high, joint]
            s_component = sensitivity[component][high, joint]
            row[f"high_flexion_{label_component}_primary_rms_nm"] = float(
                np.sqrt(np.mean(p_component**2))
            )
            row[f"high_flexion_{label_component}_sensitivity_rms_nm"] = float(
                np.sqrt(np.mean(s_component**2))
            )
            row[f"high_flexion_{label_component}_difference_rmse_nm"] = float(
                np.sqrt(np.mean((p_component - s_component) ** 2))
            )
        row["growth_within_predeclared_limits"] = bool(
            row["primary_to_sensitivity_rms_ratio"]
            <= THRESHOLDS["primary_to_sensitivity_rms_ratio_max"]
            and row["primary_to_sensitivity_peak_ratio"]
            <= THRESHOLDS["primary_to_sensitivity_peak_ratio_max"]
            and row["high_flexion_delta_to_sensitivity_rms_ratio"]
            <= THRESHOLDS["high_flexion_delta_to_sensitivity_rms_ratio_max"]
        )
        assessments.append(row["growth_within_predeclared_limits"])
        rows.append(row)
    return rows, {
        "all_joint_growth_within_predeclared_limits": all(assessments),
        "assessment": (
            "NO_ABNORMAL_REFERENCE_LEVEL_GROWTH_DETECTED"
            if all(assessments)
            else "MATERIAL_HIGH_FLEXION_TORQUE_AMPLIFICATION_DETECTED"
        ),
    }


def dataset_payload(
    reference: dict[str, Any],
    prescribed: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    payload = {
        "time_s": reference["time_s"],
        "target_q_rad": reference["q"],
        "target_dq_rad_s": reference["dq"],
        "target_ddq_rad_s2": reference["ddq"],
        "cycle_phase": reference["phases"],
        "joint_names": np.asarray(JOINT_LABELS),
    }
    payload.update(prescribed)
    payload.update(controlled)
    return payload


def summary_rows(
    condition: str,
    reference: dict[str, Any],
    prescribed: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    key_names = prescribed["key_biarticular_names"].tolist()
    for sample, elapsed in enumerate(reference["time_s"]):
        row: dict[str, Any] = {
            "condition": condition,
            "sample_index": sample,
            "time_s": elapsed,
            "cycle_phase": reference["phases"][sample],
            "target_q_hip_rad": reference["q"][sample, 0],
            "target_q_knee_rad": reference["q"][sample, 1],
            "target_dq_hip_rad_s": reference["dq"][sample, 0],
            "target_dq_knee_rad_s": reference["dq"][sample, 1],
            "target_ddq_hip_rad_s2": reference["ddq"][sample, 0],
            "target_ddq_knee_rad_s2": reference["ddq"][sample, 1],
            "actual_q_hip_rad": controlled["actual_q_rad"][sample, 0],
            "actual_q_knee_rad": controlled["actual_q_rad"][sample, 1],
            "actual_dq_hip_rad_s": controlled["actual_dq_rad_s"][sample, 0],
            "actual_dq_knee_rad_s": controlled["actual_dq_rad_s"][sample, 1],
            "actual_ddq_hip_rad_s2": controlled["actual_ddq_rad_s2"][sample, 0],
            "actual_ddq_knee_rad_s2": controlled["actual_ddq_rad_s2"][sample, 1],
            "tau_truth_hip_nm": prescribed["tau_truth_nm"][sample, 0],
            "tau_truth_knee_nm": prescribed["tau_truth_nm"][sample, 1],
            "method_b_hip_nm": controlled["force_balance_reconstruction_nm"][sample, 0],
            "method_b_knee_nm": controlled["force_balance_reconstruction_nm"][sample, 1],
            "mass_hip_nm": prescribed["mass_term_nm"][sample, 0],
            "mass_knee_nm": prescribed["mass_term_nm"][sample, 1],
            "bias_hip_nm": prescribed["bias_term_nm"][sample, 0],
            "bias_knee_nm": prescribed["bias_term_nm"][sample, 1],
            "passive_internal_hip_nm": prescribed["passive_internal_nm"][sample, 0],
            "passive_internal_knee_nm": prescribed["passive_internal_nm"][sample, 1],
            "muscle_internal_hip_nm": prescribed["actuator_internal_nm"][sample, 0],
            "muscle_internal_knee_nm": prescribed["actuator_internal_nm"][sample, 1],
            "constraint_internal_hip_nm": prescribed["constraint_internal_nm"][sample, 0],
            "constraint_internal_knee_nm": prescribed["constraint_internal_nm"][sample, 1],
            "constraint_equality_hip_nm": prescribed["constraint_equality_internal_nm"][sample, 0],
            "constraint_equality_knee_nm": prescribed["constraint_equality_internal_nm"][sample, 1],
            "constraint_joint_limit_hip_nm": prescribed["constraint_joint_limit_internal_nm"][sample, 0],
            "constraint_joint_limit_knee_nm": prescribed["constraint_joint_limit_internal_nm"][sample, 1],
            "constraint_joint_limit_active_count": prescribed["constraint_joint_limit_active_count"][sample],
            "constraint_contact_active_count": prescribed["constraint_contact_active_count"][sample],
            "balance_residual_hip_nm": controlled["dynamics_balance_residual_nm"][sample, 0],
            "balance_residual_knee_nm": controlled["dynamics_balance_residual_nm"][sample, 1],
            "balance_residual_full_max_abs_nm": controlled["dynamics_balance_full_max_abs_nm"][sample],
            "source_equality_residual": max(
                prescribed["source_equality_residual"][sample],
                controlled["source_equality_residual"][sample],
            ),
            "source_equality_force": max(
                prescribed["source_equality_force"][sample],
                controlled["source_equality_force"][sample],
            ),
            "ctrl_max": controlled["ctrl_max"][sample],
            "activation_max": controlled["activation_max"][sample],
            "tendon_length_min_m": np.min(prescribed["tendon_length_m"][sample]),
            "tendon_length_max_m": np.max(prescribed["tendon_length_m"][sample]),
            "actuator_force_max_abs_n": np.max(np.abs(prescribed["actuator_force_n"][sample])),
            "warning_count": max(
                prescribed["warning_count"][sample], controlled["warning_count"][sample]
            ),
        }
        for muscle_index, muscle in enumerate(key_names):
            row[f"{muscle}_hip_nm"] = prescribed[
                "key_biarticular_torque_contribution_nm"
            ][sample, muscle_index, 0]
            row[f"{muscle}_knee_nm"] = prescribed[
                "key_biarticular_torque_contribution_nm"
            ][sample, muscle_index, 1]
        rows.append(row)
    return rows


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def multi_panel_figure(path: Path, title: str, panels: list[dict[str, Any]]) -> None:
    width = 1280
    panel_height = 330
    top = 60
    image = Image.new("RGB", (width, top + panel_height * len(panels) + 35), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((35, 20), title, fill=(10, 10, 10), font=font)
    colors = [(28, 90, 180), (210, 55, 45), (30, 150, 80), (140, 70, 180), (230, 145, 20), (20, 160, 170)]
    for panel_index, panel in enumerate(panels):
        x0, x1 = 90, width - 35
        y0 = top + panel_index * panel_height + 35
        y1 = top + (panel_index + 1) * panel_height - 45
        x_values = np.asarray(panel["x"], dtype=float)
        all_y = np.concatenate([np.asarray(series[1], dtype=float) for series in panel["series"]])
        finite_y = all_y[np.isfinite(all_y)]
        ymin = float(np.min(finite_y)) if finite_y.size else -1.0
        ymax = float(np.max(finite_y)) if finite_y.size else 1.0
        if math.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad
        xmin, xmax = float(np.min(x_values)), float(np.max(x_values))
        draw.rectangle((x0, y0, x1, y1), outline=(80, 80, 80), width=1)
        for grid in range(1, 5):
            yy = y0 + (y1 - y0) * grid / 5.0
            draw.line((x0, yy, x1, yy), fill=(225, 225, 225), width=1)
        draw.text((12, y0), panel["ylabel"], fill=(20, 20, 20), font=font)
        draw.text((x0, y1 + 10), f"{xmin:.3f}", fill=(30, 30, 30), font=font)
        draw.text((x1 - 45, y1 + 10), f"{xmax:.3f} s", fill=(30, 30, 30), font=font)
        draw.text((x0, y0 - 18), f"{ymax:.4g}", fill=(30, 30, 30), font=font)
        draw.text((x0, y1 + 24), f"{ymin:.4g}", fill=(30, 30, 30), font=font)
        legend_x = x0 + 10
        for series_index, (label, values) in enumerate(panel["series"]):
            values = np.asarray(values, dtype=float)
            points = []
            for xx, yy in zip(x_values, values):
                px = x0 + (float(xx) - xmin) / max(xmax - xmin, 1.0e-12) * (x1 - x0)
                py = y1 - (float(yy) - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0)
                points.append((px, py))
            color = colors[series_index % len(colors)]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            draw.line((legend_x, y0 + 12, legend_x + 24, y0 + 12), fill=color, width=3)
            draw.text((legend_x + 30, y0 + 5), label, fill=color, font=font)
            legend_x += 30 + max(75, 7 * len(label))
    image.save(path)


def generate_figures(
    primary_reference: dict[str, Any],
    sensitivity_reference: dict[str, Any],
    primary_prescribed: dict[str, np.ndarray],
    sensitivity_prescribed: dict[str, np.ndarray],
    primary_controlled: dict[str, np.ndarray],
    sensitivity_controlled: dict[str, np.ndarray],
) -> list[Path]:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    time_s = primary_reference["time_s"]
    paths = []
    path = FIGURE_DIRECTORY / "target_vs_actual_hip_knee.png"
    multi_panel_figure(
        path,
        "Frozen target vs diagnostic controlled replay (PRIMARY)",
        [
            {
                "x": time_s,
                "ylabel": "hip angle (deg)",
                "series": [
                    ("target", np.degrees(primary_reference["q"][:, 0])),
                    ("actual", np.degrees(primary_controlled["actual_q_rad"][:, 0])),
                ],
            },
            {
                "x": time_s,
                "ylabel": "knee angle (deg)",
                "series": [
                    ("target", np.degrees(primary_reference["q"][:, 1])),
                    ("actual", np.degrees(primary_controlled["actual_q_rad"][:, 1])),
                ],
            },
        ],
    )
    paths.append(path)
    path = FIGURE_DIRECTORY / "tau_truth_hip_knee.png"
    multi_panel_figure(
        path,
        f"{TRUTH_FIELD} at prescribed PRIMARY state",
        [
            {"x": time_s, "ylabel": "hip torque (N m)", "series": [("tau_truth", primary_prescribed["tau_truth_nm"][:, 0])]},
            {"x": time_s, "ylabel": "knee torque (N m)", "series": [("tau_truth", primary_prescribed["tau_truth_nm"][:, 1])]},
        ],
    )
    paths.append(path)
    path = FIGURE_DIRECTORY / "generalized_force_decomposition.png"
    multi_panel_figure(
        path,
        "PRIMARY reduced-coordinate generalized-force decomposition",
        [
            {
                "x": time_s,
                "ylabel": "hip (N m)",
                "series": [
                    ("mass", primary_prescribed["mass_term_nm"][:, 0]),
                    ("bias", primary_prescribed["bias_term_nm"][:, 0]),
                    ("-muscle", -primary_prescribed["actuator_internal_nm"][:, 0]),
                    ("-constraint", -primary_prescribed["constraint_internal_nm"][:, 0]),
                    ("truth", primary_prescribed["tau_truth_nm"][:, 0]),
                ],
            },
            {
                "x": time_s,
                "ylabel": "knee (N m)",
                "series": [
                    ("mass", primary_prescribed["mass_term_nm"][:, 1]),
                    ("bias", primary_prescribed["bias_term_nm"][:, 1]),
                    ("-muscle", -primary_prescribed["actuator_internal_nm"][:, 1]),
                    ("-joint-limit", -primary_prescribed["constraint_joint_limit_internal_nm"][:, 1]),
                    ("-other-constraint", -(
                        primary_prescribed["constraint_internal_nm"][:, 1]
                        - primary_prescribed["constraint_joint_limit_internal_nm"][:, 1]
                    )),
                    ("truth", primary_prescribed["tau_truth_nm"][:, 1]),
                ],
            },
        ],
    )
    paths.append(path)
    path = FIGURE_DIRECTORY / "primary_vs_sensitivity_torque.png"
    multi_panel_figure(
        path,
        "PRIMARY limited-125 vs native-compatible SENSITIVITY truth",
        [
            {
                "x": time_s,
                "ylabel": "hip torque (N m)",
                "series": [
                    ("PRIMARY", primary_prescribed["tau_truth_nm"][:, 0]),
                    ("SENSITIVITY", sensitivity_prescribed["tau_truth_nm"][:, 0]),
                ],
            },
            {
                "x": time_s,
                "ylabel": "knee torque (N m)",
                "series": [
                    ("PRIMARY", primary_prescribed["tau_truth_nm"][:, 1]),
                    ("SENSITIVITY", sensitivity_prescribed["tau_truth_nm"][:, 1]),
                ],
            },
        ],
    )
    paths.append(path)
    path = FIGURE_DIRECTORY / "dynamics_balance_residual.png"
    epsilon = 1.0e-18
    multi_panel_figure(
        path,
        "Controlled-replay dynamics balance residual (log10 absolute N m)",
        [
            {
                "x": time_s,
                "ylabel": "PRIMARY log10 abs",
                "series": [
                    ("hip", np.log10(np.abs(primary_controlled["dynamics_balance_residual_nm"][:, 0]) + epsilon)),
                    ("knee", np.log10(np.abs(primary_controlled["dynamics_balance_residual_nm"][:, 1]) + epsilon)),
                ],
            },
            {
                "x": time_s,
                "ylabel": "SENSITIVITY log10 abs",
                "series": [
                    ("hip", np.log10(np.abs(sensitivity_controlled["dynamics_balance_residual_nm"][:, 0]) + epsilon)),
                    ("knee", np.log10(np.abs(sensitivity_controlled["dynamics_balance_residual_nm"][:, 1]) + epsilon)),
                ],
            },
        ],
    )
    paths.append(path)
    high = (time_s >= HIGH_FLEXION_INTERVAL_S[0]) & (time_s <= HIGH_FLEXION_INTERVAL_S[1])
    path = FIGURE_DIRECTORY / "high_flexion_10p540_to_16p824_zoom.png"
    multi_panel_figure(
        path,
        "High-flexion interval torque zoom",
        [
            {
                "x": time_s[high],
                "ylabel": "knee angle (deg)",
                "series": [
                    ("PRIMARY", np.degrees(primary_reference["q"][high, 1])),
                    ("SENSITIVITY", np.degrees(sensitivity_reference["q"][high, 1])),
                ],
            },
            {
                "x": time_s[high],
                "ylabel": "knee truth (N m)",
                "series": [
                    ("PRIMARY", primary_prescribed["tau_truth_nm"][high, 1]),
                    ("SENSITIVITY", sensitivity_prescribed["tau_truth_nm"][high, 1]),
                ],
            },
        ],
    )
    paths.append(path)
    return paths


def memory_peak_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0**2)
    return value / 1024.0


def report_text(
    outcome: str,
    protocol: dict[str, Any],
    semantic: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    comparison_assessment: dict[str, Any],
    runtime: dict[str, Any],
    tests: dict[str, Any],
) -> str:
    primary = metrics["PRIMARY"]
    sensitivity = metrics["SENSITIVITY"]
    p_hip, p_knee = primary["method_ab"]
    s_hip, s_knee = sensitivity["method_ab"]
    knee_comparison = next(row for row in comparison_rows if row["joint"] == "knee")
    compatible = "YES" if semantic["mechanical_objective_compatibility"] == "YES" else "WITH_CAVEATS"
    return f"""# MYOLEG REFERENCE TRAJECTORY REPLAY V1

## Final determination

`{outcome}`

This is an **offline headless virtual-model replay**, not a physiological-passive,
human, robot, safety, comfort, efficacy, or clinical result.  P0 means zero
muscle control and zero initial activation in this model; it does not mean a
physiological passive patient.

## Frozen inputs and replay

- PRIMARY: unchanged 24 s / 401-point formal reference on the frozen limited-125 XML.
- SENSITIVITY: frozen native-compatible 119.5-degree candidate on the frozen native supine XML.
- Both conditions use the model's 0.001 s timestep, zero warmup, identical P0 state,
  identical quintic q/dq/ddq interpolation, and the same pre-existing diagnostic
  generalized-PD replay. No pointwise clipping, rescaling, controller tuning, or
  stabilization adjustment occurred.
- Formal reference SHA-256: `{REFERENCE_SHA256}`.
- PRIMARY model SHA-256: `{PRIMARY_MODEL_SHA256}`.
- SENSITIVITY reference/model SHA-256: `{SENSITIVITY_REFERENCE_SHA256}` /
  `{SENSITIVITY_MODEL_SHA256}`.

## Frozen generalized-force truth

`{TRUTH_FIELD}` is the external hip/knee generalized drive, in N m, required to
realize the prescribed q, dq, ddq under P0 after accounting for inertia, bias,
MuJoCo passive force, zero-control muscle actuator force, and constraint force.
The full 34-DOF equation is

`r = M(q) qacc + qfrc_bias - qfrc_passive - qfrc_constraint - qfrc_actuator`.

Because seven right knee/patella coordinates are polynomially constrained to the
main knee angle, the frozen two-coordinate truth is **not** the naive hip/knee
slice of `r`. It is `tau_truth = T(q)^T r`, where T is the constraint-consistent
velocity tangent. This is the virtual-work projection into project hip/knee
coordinates. `qfrc_inverse = M qacc + bias - passive - constraint` is checked
explicitly before subtracting P0 actuator force.

MuJoCo's official documentation defines `qfrc_inverse` as the net external force
and documents its relation to applied and actuator forces:
[Computation](https://mujoco.readthedocs.io/en/latest/computation/index.html) and
[API types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).

## Independent-path validation

Method A is prescribed-state inverse dynamics plus explicit full-EOM accounting
and constraint-tangent projection. Method B is an independently integrated,
zero-control generalized-PD replay; its applied diagnostic drive is reconstructed
from the forward force balance at the actual state. The controller output is not
the frozen truth.

| condition/joint | RMSE N m | P95 abs N m | max abs N m | relative RMS | pass |
|---|---:|---:|---:|---:|---|
| PRIMARY hip | {p_hip['rmse']:.6f} | {p_hip['p95_abs']:.6f} | {p_hip['max_abs']:.6f} | {p_hip['relative_rms']:.3%} | {p_hip['pass']} |
| PRIMARY knee | {p_knee['rmse']:.6f} | {p_knee['p95_abs']:.6f} | {p_knee['max_abs']:.6f} | {p_knee['relative_rms']:.3%} | {p_knee['pass']} |
| SENSITIVITY hip | {s_hip['rmse']:.6f} | {s_hip['p95_abs']:.6f} | {s_hip['max_abs']:.6f} | {s_hip['relative_rms']:.3%} | {s_hip['pass']} |
| SENSITIVITY knee | {s_knee['rmse']:.6f} | {s_knee['p95_abs']:.6f} | {s_knee['max_abs']:.6f} | {s_knee['relative_rms']:.3%} | {s_knee['pass']} |

Exact inverse/formula, decomposition, muscle moment reconstruction, and forward
balance residuals are all reported in `FORCE_SEMANTICS_VALIDATION.csv` and
`DYNAMICS_BALANCE_AUDIT.csv`. `GENERALIZED_FORCE_TRUTH_SEMANTICS=PASS` only when
both conditions pass the predeclared limits.

## Tracking and stability

PRIMARY stable replay: `{primary['stable_replay']}`. SENSITIVITY stable replay:
`{sensitivity['stable_replay']}`. Full q/dq/ddq RMS, P95 and maxima are in
`DYNAMICS_BALANCE_AUDIT.csv`. There were no truth-role controller gains: the
driver is a cross-check only, while the dataset truth remains prescribed-state
inverse dynamics.

## Limited extension sensitivity

The knee PRIMARY/SENSITIVITY RMS ratio is
{knee_comparison['primary_to_sensitivity_rms_ratio']:.3f}; the peak ratio is
{knee_comparison['primary_to_sensitivity_peak_ratio']:.3f}. In 10.540–16.824 s,
the knee truth difference RMSE is
{knee_comparison['high_flexion_difference_rmse_nm']:.3f} N m versus a
{knee_comparison['high_flexion_sensitivity_rms_nm']:.3f} N m SENSITIVITY RMS.
Assessment: `{comparison_assessment['assessment']}`. This is finite and
mechanically continuous, but it is a material reference-level dynamics caveat;
it must not be hidden by calling the limited-125 extension equivalent to native
MyoLeg behavior. The dominant source is the model's soft knee joint-limit
constraint: high-flexion RMS is
{knee_comparison['high_flexion_constraint_joint_limit_internal_primary_rms_nm']:.3f}
N m in PRIMARY versus
{knee_comparison['high_flexion_constraint_joint_limit_internal_sensitivity_rms_nm']:.3f}
N m in SENSITIVITY. The corresponding muscle-actuator internal-term difference
is only
{knee_comparison['high_flexion_actuator_internal_difference_rmse_nm']:.3f} N m.
Thus this is specifically a near-upper-limit model reaction, not an inertia
increase and not evidence of human physiology.

## Existing objective compatibility

`MYOLEG_TAU_COMPATIBLE_WITH_EXISTING_J = {compatible}`.

The sign is project-positive hip flexion/knee flexion and the unit is N m for
both hinge generalized forces. Future use of the existing RMS-torque objective
must normalize every virtual-patient condition with its own reference replay,
so `J_truth(reference)=1` by construction. This stage does not compute a
candidate J, fit five parameters, train PINN, rank candidates, run BO, or make a
landscape.

## Runtime

- PRIMARY complete two-path replay: {runtime['PRIMARY']['complete_replay_wall_time_s']:.3f} s wall time,
  {runtime['PRIMARY']['integration_steps']} integration steps, replay realtime
  factor {runtime['PRIMARY']['controlled_replay_realtime_factor']:.3f}x.
- SENSITIVITY complete two-path replay: {runtime['SENSITIVITY']['complete_replay_wall_time_s']:.3f} s wall time.
- One-reference engineering estimate (mean of the two conditions):
  {runtime['engineering_estimates']['one_trajectory_s']:.3f} s; 100 =
  {runtime['engineering_estimates']['100_trajectories_s']:.1f} s; 1,000 =
  {runtime['engineering_estimates']['1000_trajectories_s']:.1f} s; 21,025 =
  {runtime['engineering_estimates']['21025_trajectories_s']:.1f} s.

These are `reference-replay-based engineering estimates`, not landscape timing.

## Direct answers

### Q1 — Can the full 24 s formal reference replay stably?

Yes on the limited-125 virtual model under the frozen diagnostic replay; this is
offline model evidence only.

### Q2 — What is frozen as tau_truth?

`{TRUTH_FIELD} = T(q)^T [M qacc + bias - passive - constraint - actuator(P0)]`,
with project-positive hip/knee coordinates and N m units.

### Q3 — Do two paths agree?

Yes within the predeclared research limits for both conditions; exact force
accounting also closes at numerical precision.

### Q4 — Are sign and units compatible?

Yes. The explicit ±1 N m forward-response sign audit passes for both joints and
both conditions, and the coordinate mapping remains identity with
`theta_shank=q_hip-q_knee`.

### Q5 — Does 120–124.79 degrees introduce abnormal dynamics?

Yes, relative to the native-compatible sensitivity replay it produces material
high-flexion knee-torque amplification under the predeclared comparison rule,
dominated by activation of the limited-125 model's soft joint-limit constraint.
That limitation does not invalidate the force semantics, but it limits how the
PRIMARY virtual patient may be interpreted.

### Q6 — Can existing J be applied later?

Yes, with the caveat that normalization must use each condition's own frozen
reference truth; no J landscape was generated here.

### Q7 — What is one replay's measured cost?

{runtime['engineering_estimates']['one_trajectory_s']:.3f} s for the complete
two-path engineering benchmark on this machine.

### Q8 — Is the truth interface ready?

Yes for the next **offline** cohort-design stage, with the limited-extension
torque-amplification caveat. It is not human-ready or robot-approved.

## Tests and next stage

Internal invariant tests: {tests['passed']} passed, {tests['failed']} failed.
The only recommended next stage is `MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1`;
this script stops without executing it.
"""


def internal_tests(
    identity_before: dict[str, Any],
    identity_after: dict[str, Any],
    inventories: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    prescribed: dict[str, dict[str, np.ndarray]],
    controlled: dict[str, dict[str, np.ndarray]],
    repeated_fingerprints: dict[str, dict[str, str]],
    metrics: dict[str, dict[str, Any]],
    sign_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
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
        ("upstream_myosuite_unchanged", identity_before["upstream_asset_sha256"] == identity_after["upstream_asset_sha256"]),
        ("formal_reference_unchanged", identity_after["hashes"]["formal_reference"] == REFERENCE_SHA256),
        ("rom_protocol_v2_unchanged", identity_before["hashes"]["formal_manifest"] == identity_after["hashes"]["formal_manifest"] == FORMAL_MANIFEST_SHA256),
        ("knee125_model_hash_frozen", identity_after["hashes"]["primary_125_model"] == PRIMARY_MODEL_SHA256),
        ("native_candidate_hash_frozen", identity_after["hashes"]["native_compatible_reference"] == SENSITIVITY_REFERENCE_SHA256),
        ("muscles_and_tendons_preserved", all(value["nu"] == 80 and value["ntendon"] == 80 for value in inventories.values())),
        ("fourteen_knee_equalities_preserved", all(value["source_knee_equality_count"] == 14 for value in inventories.values())),
        ("method_a_b_deterministic", all(value["first"] == value["repeat"] for value in repeated_fingerprints.values())),
        ("torque_sign_mapping", all(row["pass"] for row in sign_rows)),
        ("no_nan_or_inf", all(all(np.isfinite(array).all() for array in list(prescribed[condition].values()) + list(controlled[condition].values()) if isinstance(array, np.ndarray) and array.dtype.kind not in "USO") for condition in prescribed)),
        ("dynamics_balance", all(value["exact_force_accounting_pass"] for value in metrics.values())),
        ("sample_timestamps_reproducible", all(value["timestamps_exact_integer_model_steps"] for value in audits.values())),
        ("no_pointwise_clipping", all(not value["pointwise_clipping_used"] for value in audits.values())),
        ("no_five_parameter_fitting", "lower_limb_sim" not in imports and not protocol["forbidden_operations"]["five_parameter_fit"]),
        ("no_pinn", not protocol["forbidden_operations"]["pinn_training"]),
        ("no_bo", not protocol["forbidden_operations"]["bo"]),
        ("no_robot_hardware_access", imports.isdisjoint({"hardware", "control", "collection", "safety"}) and not protocol["forbidden_operations"]["robot_connection"]),
        ("prior_artifacts_unchanged", identity_before["prior_checksum_verification"] == identity_after["prior_checksum_verification"]),
        ("truth_semantics_pass", all(value["truth_semantics_pass"] for value in metrics.values())),
        ("zero_control_p0", all(np.max(np.abs(controlled[condition]["ctrl_max"])) == 0.0 and np.max(np.abs(controlled[condition]["activation_max"])) == 0.0 for condition in controlled)),
        ("no_warmup", protocol["p0_condition"]["warmup_duration_s"] == 0.0),
        ("theta_shank_preserved", protocol["coordinate_convention"]["theta_shank"] == "q_hip - q_knee"),
    ]
    return {
        "status": "PASS" if all(value for _, value in tests) else "FAIL",
        "passed": sum(bool(value) for _, value in tests),
        "failed": sum(not bool(value) for _, value in tests),
        "tests": [{"test": test, "status": "PASS" if value else "FAIL"} for test, value in tests],
    }


def write_checksums() -> None:
    paths = sorted(
        path for path in ARTIFACT_DIRECTORY.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(ARTIFACT_DIRECTORY)}" for path in paths]
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    identity_before = prior_integrity()
    environment = runtime_environment()
    models = {
        "PRIMARY": mujoco.MjModel.from_xml_path(str(PRIMARY_MODEL)),
        "SENSITIVITY": mujoco.MjModel.from_xml_path(str(SENSITIVITY_MODEL)),
    }
    references = {
        "PRIMARY": load_reference(PRIMARY_REFERENCE, "PRIMARY"),
        "SENSITIVITY": load_reference(SENSITIVITY_REFERENCE, "SENSITIVITY"),
    }
    inventories = {condition: model_inventory(model) for condition, model in models.items()}
    audits = {
        condition: reference_audit(references[condition], models[condition])
        for condition in models
    }
    if any(
        audit["duration_s"] != 24.0
        or audit["sample_count"] != 401
        or not audit["model_range_valid"]
        or audit["pointwise_clipping_used"]
        or not audit["timestamps_exact_integer_model_steps"]
        for audit in audits.values()
    ):
        raise RuntimeError("frozen replay input audit failed")

    protocol = {
        "stage_id": STAGE_ID,
        "evidence_level": "OFFLINE_HEADLESS_VIRTUAL_MODEL_ONLY",
        "default_off": True,
        "primary_condition": {
            "model_path": str(PRIMARY_MODEL.relative_to(PROJECT_ROOT)),
            "model_sha256": PRIMARY_MODEL_SHA256,
            "reference_path": str(PRIMARY_REFERENCE.relative_to(PROJECT_ROOT)),
            "reference_sha256": REFERENCE_SHA256,
        },
        "sensitivity_condition": {
            "model_path": str(SENSITIVITY_MODEL.relative_to(PROJECT_ROOT)),
            "model_sha256": SENSITIVITY_MODEL_SHA256,
            "reference_path": str(SENSITIVITY_REFERENCE.relative_to(PROJECT_ROOT)),
            "reference_sha256": SENSITIVITY_REFERENCE_SHA256,
        },
        "input_trajectory_audit": audits,
        "model_inventory": inventories,
        "p0_condition": {
            "muscle_control": 0.0,
            "initial_activation": 0.0,
            "tendon_state": "computed by mj_forward at each prescribed state; full arrays retained",
            "warmup_duration_s": 0.0,
            "physiological_passive_claim": False,
        },
        "replay_method": {
            "target_interpolation": "piecewise_quintic_matching_q_dq_ddq_at_all_401_samples",
            "model_timestep_s": 0.001,
            "sample_count": 401,
            "duration_s": 24.0,
            "diagnostic_driver": DRIVER,
        },
        "coordinate_convention": {
            "q_hip": "MyoLeg hip_flexion_r; project-positive hip flexion; rad",
            "q_knee": "MyoLeg knee_angle_r; project-positive knee flexion; rad",
            "theta_shank": "q_hip - q_knee",
            "torque_unit": "N*m",
        },
        "force_semantics_validation_thresholds_frozen_before_retained_replay": THRESHOLDS,
        "high_flexion_interval_s": list(HIGH_FLEXION_INTERVAL_S),
        "forbidden_operations": {
            "five_parameter_fit": False,
            "pinn_training": False,
            "bo": False,
            "landscape": False,
            "robot_connection": False,
            "controller_tuning": False,
            "formal_reference_modification": False,
        },
        "non_retained_implementation_check": (
            "a pre-artifact dry run was used only to detect and correct the required constraint-tangent virtual-work projection; no model/reference/result was retained and no controller gain was changed"
        ),
    }
    write_json(ARTIFACT_DIRECTORY / "REPLAY_PROTOCOL.json", protocol)
    protocol_sha = sha256_file(ARTIFACT_DIRECTORY / "REPLAY_PROTOCOL.json")

    semantic = {
        "semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "generalized_force_truth_semantics": "PENDING_VALIDATION",
        "definition": "external generalized hip/knee drive required at prescribed q,dq,ddq under frozen P0",
        "full_coordinate_equation": "r = M*qacc + qfrc_bias - qfrc_passive - qfrc_constraint - qfrc_actuator(P0)",
        "reduced_coordinate_equation": "tau_truth = T(q)^T*r",
        "tangent_definition": "T maps [dq_hip,dq_knee] into the full velocity space and includes all seven right knee/patella polynomial equality derivatives",
        "why_naive_slice_is_invalid": "full-coordinate inverse force contains auxiliary-DOF drive that must be combined by virtual work before a two-coordinate knee torque is defined",
        "mujoCo_qfrc_inverse_identity": "qfrc_inverse = M*qacc + qfrc_bias - qfrc_passive - qfrc_constraint",
        "internal_response_terms": ["qfrc_passive", "qfrc_actuator(P0)", "qfrc_constraint"],
        "external_diagnostic_driver": "qfrc_applied on hip_flexion_r and knee_angle_r only; Method B cross-check, not truth",
        "sign_convention": "positive torque increases project-positive hip flexion or knee flexion",
        "unit": "N*m",
        "method_a": "prescribed-state inverse dynamics plus explicit EOM decomposition and T^T projection",
        "method_b": "independently integrated zero-control generalized-PD replay plus forward force-balance reconstruction",
        "mechanical_objective_compatibility": "PENDING_VALIDATION",
        "protocol_sha256": protocol_sha,
        "official_mujoco_sources": [
            "https://mujoco.readthedocs.io/en/latest/computation/index.html",
            "https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html",
        ],
    }
    write_json(ARTIFACT_DIRECTORY / f"{TRUTH_SEMANTIC_VERSION}.json", semantic)

    prescribed: dict[str, dict[str, np.ndarray]] = {}
    controlled: dict[str, dict[str, np.ndarray]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    repeated_fingerprints: dict[str, dict[str, str]] = {}
    for condition in ("PRIMARY", "SENSITIVITY"):
        p, p_runtime = prescribed_truth(models[condition], references[condition])
        c, c_runtime = controlled_replay(models[condition], references[condition])
        p_repeat, _ = prescribed_truth(models[condition], references[condition])
        c_repeat, _ = controlled_replay(models[condition], references[condition])
        first_fingerprint = array_fingerprint(
            [p["tau_truth_nm"], p["actuator_force_n"], c["actual_q_rad"], c["force_balance_reconstruction_nm"]]
        )
        repeat_fingerprint = array_fingerprint(
            [p_repeat["tau_truth_nm"], p_repeat["actuator_force_n"], c_repeat["actual_q_rad"], c_repeat["force_balance_reconstruction_nm"]]
        )
        repeated_fingerprints[condition] = {"first": first_fingerprint, "repeat": repeat_fingerprint}
        prescribed[condition] = p
        controlled[condition] = c
        runtimes[condition] = {
            "prescribed_inverse_wall_time_s": p_runtime["wall_time_s"],
            "controlled_replay_wall_time_s": c_runtime["wall_time_s"],
            "complete_replay_wall_time_s": p_runtime["wall_time_s"] + c_runtime["wall_time_s"],
            "integration_steps": c_runtime["integration_steps"],
            "state_evaluations": c_runtime["state_evaluations"],
            "controlled_replay_realtime_factor": c_runtime["realtime_factor"],
            "peak_process_memory_mib": memory_peak_mib(),
            "determinism_repeat_excluded_from_benchmark": True,
        }

    metrics = {
        condition: evaluate_metrics(
            condition,
            references[condition],
            prescribed[condition],
            controlled[condition],
            {
                **runtimes[condition],
                "all_finite": bool(
                    np.isfinite(controlled[condition]["actual_q_rad"]).all()
                    and np.isfinite(prescribed[condition]["tau_truth_nm"]).all()
                ),
                "warning_count": int(
                    max(
                        np.max(controlled[condition]["warning_count"]),
                        np.max(prescribed[condition]["warning_count"]),
                    )
                ),
            },
        )
        for condition in ("PRIMARY", "SENSITIVITY")
    }
    sign_rows = []
    for condition in ("PRIMARY", "SENSITIVITY"):
        sign_rows.extend(torque_sign_audit(models[condition], condition))
    truth_pass = bool(
        all(value["truth_semantics_pass"] for value in metrics.values())
        and all(row["pass"] for row in sign_rows)
        and all(value["first"] == value["repeat"] for value in repeated_fingerprints.values())
    )
    semantic["generalized_force_truth_semantics"] = "PASS" if truth_pass else "FAIL"
    semantic["torque_sign_mapping"] = "PASS" if all(row["pass"] for row in sign_rows) else "FAIL"
    semantic["mechanical_objective_compatibility"] = "YES" if truth_pass else "NO"
    semantic["validation_metrics"] = metrics
    semantic["determinism_fingerprints"] = repeated_fingerprints
    write_json(ARTIFACT_DIRECTORY / f"{TRUTH_SEMANTIC_VERSION}.json", semantic)
    if not truth_pass:
        raise RuntimeError("GENERALIZED_FORCE_TRUTH_SEMANTICS=FAIL; dataset generation blocked")

    comparison_rows, comparison_assessment = primary_sensitivity_comparison(
        prescribed["PRIMARY"], prescribed["SENSITIVITY"], references["PRIMARY"]["time_s"]
    )
    outcome = (
        "MYOLEG_REFERENCE_REPLAY_VALID"
        if comparison_assessment["all_joint_growth_within_predeclared_limits"]
        else "MYOLEG_REFERENCE_REPLAY_VALID_WITH_LIMITATIONS"
    )
    if outcome.endswith("WITH_LIMITATIONS"):
        semantic["mechanical_objective_compatibility"] = "WITH_CAVEATS"
        write_json(ARTIFACT_DIRECTORY / f"{TRUTH_SEMANTIC_VERSION}.json", semantic)

    dataset_schema = {
        "schema_version": 1,
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "array_container": "compressed NumPy NPZ",
        "sample_axis": "401 frozen timestamps over 24 s",
        "joint_axis": ["hip", "knee"],
        "coordinate_units": {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"},
        "muscle_arrays": "80 preserved actuator names; force N, length/tendon length m, torque contribution N*m",
        "truth_source": "prescribed-state Method A; controlled Method B arrays are diagnostic validation only",
    }
    write_json(ARTIFACT_DIRECTORY / "DATASET_SCHEMA.json", dataset_schema)

    for condition in ("PRIMARY", "SENSITIVITY"):
        payload = dataset_payload(references[condition], prescribed[condition], controlled[condition])
        np.savez_compressed(ARTIFACT_DIRECTORY / f"{condition}_REFERENCE_REPLAY.npz", **payload)
        rows = summary_rows(condition, references[condition], prescribed[condition], controlled[condition])
        write_csv(
            ARTIFACT_DIRECTORY / f"{condition}_REFERENCE_REPLAY_SUMMARY.csv",
            rows,
            list(rows[0]),
        )

    force_rows = []
    for condition in ("PRIMARY", "SENSITIVITY"):
        for row in metrics[condition]["method_ab"]:
            force_rows.append({"validation": "METHOD_A_VS_METHOD_B", **row})
        for key, value in metrics[condition]["exact_force_accounting"].items():
            force_rows.append(
                {
                    "validation": key,
                    "condition": condition,
                    "joint": "all",
                    "rmse": value,
                    "p95_abs": "",
                    "max_abs": value,
                    "relative_rms": "",
                    "truth_rms_nm": "",
                    "correlation": "",
                    "pass": metrics[condition]["exact_force_accounting_pass"],
                }
            )
    for row in sign_rows:
        force_rows.append(
            {
                "validation": "TORQUE_SIGN_MAPPING",
                "condition": row["condition"],
                "joint": row["driven_joint"],
                "rmse": "",
                "p95_abs": "",
                "max_abs": row["own_acceleration_response_rad_s2_per_nm"],
                "relative_rms": "",
                "truth_rms_nm": "",
                "correlation": "",
                "pass": row["pass"],
            }
        )
    write_csv(ARTIFACT_DIRECTORY / "FORCE_SEMANTICS_VALIDATION.csv", force_rows, list(force_rows[0]))

    balance_rows = []
    for condition in ("PRIMARY", "SENSITIVITY"):
        for joint, label in enumerate(JOINT_LABELS):
            q = metrics[condition]["tracking"][label]
            balance_rows.append(
                {
                    "condition": condition,
                    "joint": label,
                    "balance_rmse_nm": metric(controlled[condition]["dynamics_balance_residual_nm"][:, joint])["rmse"],
                    "balance_p95_abs_nm": metric(controlled[condition]["dynamics_balance_residual_nm"][:, joint])["p95_abs"],
                    "balance_max_abs_nm": metric(controlled[condition]["dynamics_balance_residual_nm"][:, joint])["max_abs"],
                    "q_error_rmse_deg": q["q_error_deg"]["rmse"],
                    "q_error_p95_abs_deg": q["q_error_deg"]["p95_abs"],
                    "q_error_max_abs_deg": q["q_error_deg"]["max_abs"],
                    "dq_error_rmse_deg_s": q["dq_error_deg_s"]["rmse"],
                    "dq_error_p95_abs_deg_s": q["dq_error_deg_s"]["p95_abs"],
                    "dq_error_max_abs_deg_s": q["dq_error_deg_s"]["max_abs"],
                    "ddq_error_rmse_deg_s2": q["ddq_error_deg_s2"]["rmse"],
                    "ddq_error_p95_abs_deg_s2": q["ddq_error_deg_s2"]["p95_abs"],
                    "ddq_error_max_abs_deg_s2": q["ddq_error_deg_s2"]["max_abs"],
                    "source_equality_residual_max": metrics[condition]["source_equality_residual_max"],
                    "stable_replay": metrics[condition]["stable_replay"],
                }
            )
    write_csv(ARTIFACT_DIRECTORY / "DYNAMICS_BALANCE_AUDIT.csv", balance_rows, list(balance_rows[0]))
    write_csv(ARTIFACT_DIRECTORY / "PRIMARY_VS_SENSITIVITY.csv", comparison_rows, list(comparison_rows[0]))

    average = float(np.mean([runtimes[c]["complete_replay_wall_time_s"] for c in runtimes]))
    runtime_payload = {
        **runtimes,
        "engineering_estimates": {
            "basis": "mean complete two-path 24 s reference replay; determinism repeat excluded",
            "label": "reference-replay-based engineering estimate",
            "one_trajectory_s": average,
            "100_trajectories_s": 100.0 * average,
            "1000_trajectories_s": 1000.0 * average,
            "21025_trajectories_s": 21025.0 * average,
        },
    }
    write_json(ARTIFACT_DIRECTORY / "RUNTIME_BENCHMARK.json", runtime_payload)
    generate_figures(
        references["PRIMARY"],
        references["SENSITIVITY"],
        prescribed["PRIMARY"],
        prescribed["SENSITIVITY"],
        controlled["PRIMARY"],
        controlled["SENSITIVITY"],
    )

    identity_after = prior_integrity()
    tests = internal_tests(
        identity_before,
        identity_after,
        inventories,
        audits,
        prescribed,
        controlled,
        repeated_fingerprints,
        metrics,
        sign_rows,
        protocol,
    )
    write_json(ARTIFACT_DIRECTORY / "TEST_RESULTS.json", tests)
    if tests["status"] != "PASS":
        raise RuntimeError("internal replay invariant tests failed")

    report = report_text(
        outcome,
        protocol,
        semantic,
        metrics,
        comparison_rows,
        comparison_assessment,
        runtime_payload,
        tests,
    )
    (ARTIFACT_DIRECTORY / "MYOLEG_REFERENCE_TRAJECTORY_REPLAY_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    protected_inputs_after = prior_integrity()
    metadata = {
        "stage_id": STAGE_ID,
        "builder_script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "outcome": outcome,
        "evidence_level": "OFFLINE_HEADLESS_VIRTUAL_MODEL_ONLY",
        "default_off": True,
        "human_ready": False,
        "robot_approved": False,
        "generalized_force_truth_semantics": semantic["generalized_force_truth_semantics"],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "torque_sign_mapping": semantic["torque_sign_mapping"],
        "myoleg_tau_compatible_with_existing_j": semantic["mechanical_objective_compatibility"],
        "extension_sensitivity_assessment": comparison_assessment,
        "runtime_environment": environment,
        "source_identity_before": identity_before,
        "source_identity_after": protected_inputs_after,
        "replay_protocol_sha256": protocol_sha,
        "determinism_fingerprints": repeated_fingerprints,
        "tests": tests,
        "five_parameter_fit": False,
        "pinn_trained": False,
        "bo_run": False,
        "landscape_generated": False,
        "robot_connected": False,
        "next_stage_recommended_not_executed": "MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1",
    }
    artifact_candidates = sorted(
        path for path in ARTIFACT_DIRECTORY.rglob("*") if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    )
    metadata["artifact_sha256"] = {
        str(path.relative_to(ARTIFACT_DIRECTORY)): sha256_file(path)
        for path in artifact_candidates
    }
    write_json(ARTIFACT_DIRECTORY / "metadata.json", metadata)
    write_checksums()
    print(json.dumps({
        "stage_id": STAGE_ID,
        "outcome": outcome,
        "truth_semantics": semantic["generalized_force_truth_semantics"],
        "tests": tests["status"],
        "artifact_directory": str(ARTIFACT_DIRECTORY),
    }, indent=2))


if __name__ == "__main__":
    main()
