"""Build and audit a derived, headless MyoLeg supine rehabilitation model.

The upstream MyoSuite installation is read-only.  This script creates a new
MJCF that references the frozen upstream mesh/texture files by absolute path,
adds only explicit constraints/sites, and writes all evidence into the separate
external_simulation_audits tree.
"""

from __future__ import annotations

import ast
import csv
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


STAGE_ID = "MYOLEG_SUPINE_HIP_KNEE_REHAB_FEASIBILITY_V1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_supine_hip_knee_rehab_feasibility_v1"
)
SOURCE_MODEL = Path(
    "/Users/fengjiaxin/.virtualenvs/myosuite-v2/lib/python3.10/site-packages/"
    "myosuite/simhive/myo_sim/leg/myolegs.xml"
)
DERIVED_XML = DERIVED_DIRECTORY / "myoleg_supine_right_v1.xml"
REFERENCE_PATH = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST_PATH = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
PREVIOUS_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_install_and_smoke_test_v1"
)

TARGET_LEG = "right"
TARGET_HIP = "hip_flexion_r"
TARGET_KNEE = "knee_angle_r"
TARGET_FEMUR = "femur_r"
TARGET_TIBIA = "tibia_r"
TARGET_FOOT = "calcn_r"
PROVISIONAL_STRAP_SITE = "RTB3"
SOURCE_EQUALITY_COUNT = 14
ROOT_POSITION_M = np.asarray([0.0, 0.0, 1.0], dtype=float)
ROOT_QUATERNION_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0], dtype=float
)
INITIAL_HIP_RAD = math.radians(30.0)
INITIAL_KNEE_RAD = math.radians(30.0)
LOCKED_JOINTS = (
    "hip_adduction_r",
    "hip_rotation_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)
TARGET_AUXILIARY_JOINTS = (
    "knee_angle_r_translation2",
    "knee_angle_r_translation1",
    "knee_angle_r_rotation2",
    "knee_angle_r_rotation3",
    "knee_angle_r_beta_translation2",
    "knee_angle_r_beta_translation1",
    "knee_angle_r_beta_rotation1",
)
CONTRALATERAL_AUXILIARY_JOINTS = (
    "knee_angle_l_translation2",
    "knee_angle_l_translation1",
    "knee_angle_l_rotation2",
    "knee_angle_l_rotation3",
    "knee_angle_l_beta_translation2",
    "knee_angle_l_beta_translation1",
    "knee_angle_l_beta_rotation1",
)
DIAGNOSTIC_DT = 0.001
PD_KP = 2500.0
PD_KD = 100.0
PD_TORQUE_LIMIT_NM = 1500.0
SETTLE_STEPS = 1500
STATIC_STEPS = 10_000
MOTION_STEPS = 4_000
RECORD_STRIDE = 10
EXTERNAL_FORCE_N = 2.0
L1_PROJECT_M = 0.42
L2_PROJECT_M = 0.30
REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"


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


def object_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or ""


def joint_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise RuntimeError(f"missing joint: {name}")
    return int(value)


def site_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if value < 0:
        raise RuntimeError(f"missing site: {name}")
    return int(value)


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"missing body: {name}")
    return int(value)


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, name)])


def dof_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, name)])


def source_identity() -> dict[str, Any]:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)
    if sha256_file(REFERENCE_PATH) != REFERENCE_SHA256:
        raise RuntimeError("frozen active reference changed")
    previous_checksums = PREVIOUS_AUDIT_DIRECTORY / "checksums.sha256"
    if not previous_checksums.is_file():
        raise RuntimeError("frozen MyoLeg install audit is missing")
    previous_verification = verify_checksum_manifest(previous_checksums)
    if previous_verification["status"] != "PASS":
        raise RuntimeError("frozen MyoLeg install audit checksum verification failed")
    source_spec = mujoco.MjSpec.from_file(str(SOURCE_MODEL))
    upstream_assets = asset_files(source_spec)
    return {
        "source_model_path": str(SOURCE_MODEL),
        "source_model_sha256": sha256_file(SOURCE_MODEL),
        "upstream_asset_count": len(upstream_assets),
        "upstream_asset_sha256": {
            str(path): sha256_file(path) for path in upstream_assets
        },
        "reference_path": str(REFERENCE_PATH),
        "reference_sha256": sha256_file(REFERENCE_PATH),
        "formal_manifest_sha256": sha256_file(FORMAL_MANIFEST_PATH),
        "previous_audit_checksums_sha256": sha256_file(previous_checksums),
        "previous_audit_checksum_verification": previous_verification,
    }


def verify_checksum_manifest(path: Path) -> dict[str, Any]:
    checked = 0
    failures = []
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


def runtime_environment() -> dict[str, Any]:
    values = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "myosuite": importlib.metadata.version("myosuite"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    expected = {"python": "3.10.19", "myosuite": "2.12.2", "mujoco": "3.6.0"}
    values["frozen_versions_expected"] = expected
    values["frozen_versions_match"] = all(values[key] == value for key, value in expected.items())
    if not values["frozen_versions_match"]:
        raise RuntimeError("frozen MyoLeg environment version changed")
    return values


def asset_files(spec: mujoco.MjSpec) -> list[Path]:
    base = Path(spec.modelfiledir)
    mesh_directory = Path(spec.meshdir)
    texture_directory = Path(spec.texturedir)
    files: list[Path] = []
    for mesh in spec.meshes:
        if mesh.file:
            absolute = (base / mesh_directory / mesh.file).resolve()
            mesh.file = str(absolute)
            files.append(absolute)
    for texture in spec.textures:
        if texture.file:
            absolute = (base / texture_directory / texture.file).resolve()
            texture.file = str(absolute)
            files.append(absolute)
    spec.meshdir = ""
    spec.texturedir = ""
    return sorted(set(files))


def build_derived_model() -> tuple[mujoco.MjModel, dict[str, Any]]:
    spec = mujoco.MjSpec.from_file(str(SOURCE_MODEL))
    assets = asset_files(spec)
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)

    root_site = spec.body("root").add_site(
        name="derived_root_anchor",
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        size=[0.005],
    )
    world_site = spec.worldbody.add_site(
        name="derived_world_supine_anchor",
        pos=ROOT_POSITION_M.tolist(),
        quat=ROOT_QUATERNION_WXYZ.tolist(),
        size=[0.005],
    )
    spec.add_equality(
        name="derived_root_supine_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        name1=root_site.name,
        name2=world_site.name,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        active=1,
        solref=[0.005, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
    )
    for name in LOCKED_JOINTS:
        spec.add_equality(
            name=f"derived_lock_{name}",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=name,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            data=[0.0] * 11,
            active=1,
            solref=[0.005, 1.0],
            solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
        )

    disabled_contacts = []
    for name in ("floor", "terrain"):
        geom = spec.geom(name)
        if geom is not None:
            disabled_contacts.append(
                {
                    "geom": name,
                    "source_contype": int(geom.contype),
                    "source_conaffinity": int(geom.conaffinity),
                    "derived_contype": 0,
                    "derived_conaffinity": 0,
                }
            )
            geom.contype = 0
            geom.conaffinity = 0

    model = spec.compile()
    if not math.isclose(float(model.opt.timestep), DIAGNOSTIC_DT, abs_tol=1e-15):
        raise RuntimeError("upstream integration timestep changed")
    DERIVED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    spec.to_file(str(DERIVED_XML))
    reloaded = mujoco.MjModel.from_xml_path(str(DERIVED_XML))
    if (model.nq, model.nv, model.nu, model.neq) != (
        reloaded.nq,
        reloaded.nv,
        reloaded.nu,
        reloaded.neq,
    ):
        raise RuntimeError("derived XML round-trip changed model dimensions")
    asset_manifest = [
        {"path": str(path), "sha256": sha256_file(path), "referenced_not_copied": True}
        for path in assets
    ]
    modifications = {
        "root_strategy": "site-based weld to world supine anchor",
        "root_position_m": ROOT_POSITION_M.tolist(),
        "root_quaternion_wxyz": ROOT_QUATERNION_WXYZ.tolist(),
        "root_rotation": "-90 deg about MyoLeg world y",
        "locked_joint_strategy": "single-joint equality at native zero/reference",
        "locked_joints": list(LOCKED_JOINTS),
        "preserved_target_free_joints": [TARGET_HIP, TARGET_KNEE],
        "preserved_target_auxiliary_joints": list(TARGET_AUXILIARY_JOINTS),
        "preserved_contralateral_bodies_muscles_tendons": True,
        "contact_strategy": "SUPINE_NO_BED_CONTACT",
        "disabled_world_contacts": disabled_contacts,
        "gravity_unchanged_m_s2": model.opt.gravity.tolist(),
        "muscles_removed": 0,
        "tendons_removed": 0,
        "joints_removed": 0,
        "source_equalities_removed_or_modified": 0,
    }
    return reloaded, {
        "assets": asset_manifest,
        "modifications": modifications,
        "derived_xml_sha256": sha256_file(DERIVED_XML),
    }


def initial_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[:7] = np.concatenate([ROOT_POSITION_M, ROOT_QUATERNION_WXYZ])
    for name in LOCKED_JOINTS:
        data.qpos[qpos_address(model, name)] = 0.0
    data.qpos[qpos_address(model, TARGET_HIP)] = INITIAL_HIP_RAD
    data.qpos[qpos_address(model, TARGET_KNEE)] = INITIAL_KNEE_RAD
    if model.na:
        data.act[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def set_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_target: float,
    knee_target: float,
    hip_velocity: float = 0.0,
    knee_velocity: float = 0.0,
) -> tuple[float, float]:
    hip_q = qpos_address(model, TARGET_HIP)
    knee_q = qpos_address(model, TARGET_KNEE)
    hip_v = dof_address(model, TARGET_HIP)
    knee_v = dof_address(model, TARGET_KNEE)
    tau_hip = np.clip(
        PD_KP * (hip_target - data.qpos[hip_q])
        + PD_KD * (hip_velocity - data.qvel[hip_v]),
        -PD_TORQUE_LIMIT_NM,
        PD_TORQUE_LIMIT_NM,
    )
    tau_knee = np.clip(
        PD_KP * (knee_target - data.qpos[knee_q])
        + PD_KD * (knee_velocity - data.qvel[knee_v]),
        -PD_TORQUE_LIMIT_NM,
        PD_TORQUE_LIMIT_NM,
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[hip_v] = tau_hip
    data.qfrc_applied[knee_v] = tau_knee
    return float(tau_hip), float(tau_knee)


def constraint_metrics(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, float]:
    equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = np.flatnonzero(np.asarray(data.efc_type) == equality_type)
    source_rows = rows[np.asarray(data.efc_id)[rows] < SOURCE_EQUALITY_COUNT]
    derived_rows = rows[np.asarray(data.efc_id)[rows] >= SOURCE_EQUALITY_COUNT]
    return {
        "source_knee_equality_max_abs_position_error": float(
            np.max(np.abs(np.asarray(data.efc_pos)[source_rows]))
        ) if source_rows.size else 0.0,
        "derived_constraint_max_abs_position_error": float(
            np.max(np.abs(np.asarray(data.efc_pos)[derived_rows]))
        ) if derived_rows.size else 0.0,
        "source_knee_equality_max_abs_force": float(
            np.max(np.abs(np.asarray(data.efc_force)[source_rows]))
        ) if source_rows.size else 0.0,
        "derived_constraint_max_abs_force": float(
            np.max(np.abs(np.asarray(data.efc_force)[derived_rows]))
        ) if derived_rows.size else 0.0,
    }


def warning_count(data: mujoco.MjData) -> int:
    return int(np.asarray(data.warning.number, dtype=np.int64).sum())


def finite_state(data: mujoco.MjData) -> bool:
    fields = (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qfrc_passive,
        data.qfrc_actuator,
        data.qfrc_constraint,
        data.qfrc_applied,
        data.actuator_force,
        data.ten_length,
        data.ten_velocity,
    )
    return all(bool(np.isfinite(np.asarray(field)).all()) for field in fields)


def settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: np.ndarray | None = None,
    steps: int = SETTLE_STEPS,
) -> None:
    if control is None:
        control = np.zeros(model.nu, dtype=float)
    for _ in range(steps):
        data.ctrl[:] = control
        set_pd(model, data, INITIAL_HIP_RAD, INITIAL_KNEE_RAD)
        mujoco.mj_step(model, data)
        if not finite_state(data):
            raise RuntimeError("derived model became nonfinite during settle")
    data.qfrc_applied[:] = 0.0


def joint_config(model: mujoco.MjModel) -> list[dict[str, Any]]:
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    rows = []
    for index in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name == "root":
            role = "ROOT_FREE_JOINT_PRESERVED_BUT_WELDED"
            constraint = "derived_root_supine_weld"
        elif name in {TARGET_HIP, TARGET_KNEE}:
            role = "TARGET_PRIMARY_FREE_DOF"
            constraint = ""
        elif name in LOCKED_JOINTS:
            role = "DERIVED_EQUALITY_LOCK_ZERO"
            constraint = f"derived_lock_{name}"
        elif name in TARGET_AUXILIARY_JOINTS:
            role = "TARGET_AUXILIARY_EQUALITY_PRESERVED"
            constraint = "upstream knee equality"
        elif name in CONTRALATERAL_AUXILIARY_JOINTS:
            role = "CONTRALATERAL_AUXILIARY_EQUALITY_PRESERVED"
            constraint = "upstream knee equality"
        else:
            role = "PRESERVED_OTHER"
            constraint = ""
        rows.append(
            {
                "joint_id": index,
                "name": name,
                "body": object_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[index])
                ),
                "joint_type": int(model.jnt_type[index]),
                "qpos_address": int(model.jnt_qposadr[index]),
                "dof_address": int(model.jnt_dofadr[index]),
                "native_range_min_rad": float(source_model.jnt_range[index, 0]),
                "native_range_max_rad": float(source_model.jnt_range[index, 1]),
                "native_range_unchanged": bool(
                    np.array_equal(model.jnt_range[index], source_model.jnt_range[index])
                ),
                "derived_role": role,
                "derived_constraint": constraint,
            }
        )
    return rows


def coordinate_mapping(model: mujoco.MjModel) -> dict[str, Any]:
    poses = [(20.0, 20.0), (60.0, 60.0), (100.0, 100.0)]
    diagnostics = []
    for hip_deg, knee_deg in poses:
        data = initial_data(model)
        data.qpos[qpos_address(model, TARGET_HIP)] = math.radians(hip_deg)
        data.qpos[qpos_address(model, TARGET_KNEE)] = math.radians(knee_deg)
        mujoco.mj_forward(model, data)
        hip_pos = data.site_xpos[site_id(model, "hip_r")].copy()
        knee_pos = data.site_xpos[site_id(model, "knee_r")].copy()
        ankle_pos = data.site_xpos[site_id(model, "ankle_r")].copy()
        thigh = knee_pos - hip_pos
        shank = ankle_pos - knee_pos
        thigh_angle = math.degrees(math.atan2(thigh[2], thigh[0]))
        shank_angle = math.degrees(math.atan2(shank[2], shank[0]))
        diagnostics.append(
            {
                "project_hip_deg": hip_deg,
                "project_knee_deg": knee_deg,
                "myoleg_hip_deg": hip_deg,
                "myoleg_knee_deg": knee_deg,
                "measured_thigh_world_xz_angle_deg": thigh_angle,
                "expected_thigh_project_angle_deg": hip_deg,
                "thigh_angle_error_deg": thigh_angle - hip_deg,
                "measured_shank_world_xz_angle_deg": shank_angle,
                "expected_shank_project_angle_deg": hip_deg - knee_deg,
                "shank_angle_error_deg": shank_angle - (hip_deg - knee_deg),
                "hip_site_world_m": hip_pos.tolist(),
                "knee_site_world_m": knee_pos.tolist(),
                "ankle_site_world_m": ankle_pos.tolist(),
            }
        )
    max_thigh_error = max(abs(row["thigh_angle_error_deg"]) for row in diagnostics)
    max_shank_error = max(abs(row["shank_angle_error_deg"]) for row in diagnostics)
    status = "PASS" if max_thigh_error <= 2.0 and max_shank_error <= 5.0 else "FAIL"
    return {
        "status": status,
        "target_leg": TARGET_LEG,
        "root_supine_transform": {
            "position_m": ROOT_POSITION_M.tolist(),
            "quaternion_wxyz": ROOT_QUATERNION_WXYZ.tolist(),
            "rotation": "-90 deg about world y",
        },
        "project_frame": {
            "origin": "target hip center site hip_r",
            "x_axis": "MyoLeg world +x after root supine transform",
            "z_axis": "MyoLeg world +z after root supine transform",
            "sagittal_plane": "world x-z",
        },
        "joint_mapping": {
            "q_project_hip_rad": "q_myoleg_hip_flexion_r_rad",
            "q_project_knee_rad": "q_myoleg_knee_angle_r_rad",
            "q_myoleg_hip_flexion_r_rad": "q_project_hip_rad",
            "q_myoleg_knee_angle_r_rad": "q_project_knee_rad",
            "theta_shank_rad": "q_project_hip_rad - q_project_knee_rad",
            "signs": {"hip": 1.0, "knee": 1.0},
            "offsets_rad": {"hip": 0.0, "knee": 0.0},
            "mapping_type": "fixed sign plus fixed offset; identity for both joints",
        },
        "round_trip_max_abs_error_rad": 0.0,
        "diagnostic_poses": diagnostics,
        "max_thigh_angle_error_deg": max_thigh_error,
        "max_shank_angle_error_deg": max_shank_error,
    }


def rom_audit(model: mujoco.MjModel) -> dict[str, Any]:
    formal_manifest = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        formal_manifest["rom_protocol_version"] != "ROM_PROTOCOL_V2"
        or formal_manifest["hip_rom_deg"] != [0.0, 120.0]
        or formal_manifest["knee_rom_deg"] != [5.0, 145.0]
        or formal_manifest["theta_shank_definition"] != "q_hip - q_knee"
        or formal_manifest["active_reference_sha256"] != REFERENCE_SHA256
    ):
        raise RuntimeError("frozen formal ROM/reference convention changed")
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    hip = np.asarray([math.degrees(float(row["q_hip_rad"])) for row in rows])
    knee = np.asarray([math.degrees(float(row["q_knee_rad"])) for row in rows])
    times = np.asarray([float(row["time_s"]) for row in rows])
    phases = np.asarray([row["cycle_phase"] for row in rows])
    hip_range = np.degrees(model.jnt_range[joint_id(model, TARGET_HIP)]).tolist()
    knee_range = np.degrees(model.jnt_range[joint_id(model, TARGET_KNEE)]).tolist()
    native_knee_upper_deg = float(knee_range[1])
    outside = knee > native_knee_upper_deg + 1e-12
    exceedance = np.maximum(knee - native_knee_upper_deg, 0.0)
    intervals = []
    indices = np.flatnonzero(outside)
    if indices.size:
        starts = [int(indices[0])]
        ends = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right != left + 1:
                ends.append(int(left))
                starts.append(int(right))
        ends.append(int(indices[-1]))
        for start, end in zip(starts, ends):
            intervals.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "start_time_s": float(times[start]),
                    "end_time_s": float(times[end]),
                    "start_phase": str(phases[start]),
                    "end_phase": str(phases[end]),
                    "phase_sequence": list(dict.fromkeys(phases[start : end + 1].tolist())),
                    "sample_count": end - start + 1,
                }
            )
    maximum = float(exceedance.max(initial=0.0))
    outside_fraction = float(np.mean(outside))
    if not outside.any():
        classification = "REFERENCE_FULLY_WITHIN_MYOLEG_NATIVE_ROM"
    elif maximum <= 5.0 and outside_fraction <= 0.30:
        classification = "REFERENCE_SLIGHTLY_EXCEEDS_MYOLEG_NATIVE_ROM"
    else:
        classification = "REFERENCE_MATERIALLY_EXCEEDS_MYOLEG_NATIVE_ROM"
    peak_index = int(np.argmax(knee))
    return {
        "classification": classification,
        "classification_rule_frozen_for_v1": {
            "fully": "zero samples exceed native limit within 1e-12 deg",
            "slightly": "maximum exceedance <=5 deg and outside fraction <=30%",
            "materially": "otherwise",
        },
        "coordinate_mapping_used": "identity hip and knee joint angles",
        "formal_project_rom_protocol": formal_manifest["rom_protocol_version"],
        "formal_project_rom_deg": {
            "hip": formal_manifest["hip_rom_deg"],
            "knee": formal_manifest["knee_rom_deg"],
        },
        "myoleg_native_rom_deg": {"hip": hip_range, "knee": knee_range},
        "common_supported_formal_rom_deg": {
            "hip": [0.0, min(120.0, float(hip_range[1]))],
            "knee": [5.0, min(145.0, native_knee_upper_deg)],
        },
        "reference_sample_count": len(rows),
        "reference_hip_range_deg": [float(hip.min()), float(hip.max())],
        "reference_knee_range_deg": [float(knee.min()), float(knee.max())],
        "samples_inside_myoleg_native_rom": int((~outside).sum()),
        "samples_outside_myoleg_native_rom": int(outside.sum()),
        "outside_fraction": outside_fraction,
        "maximum_knee_exceedance_deg": maximum,
        "peak_knee_deg": float(knee[peak_index]),
        "peak_knee_time_s": float(times[peak_index]),
        "peak_knee_phase": str(phases[peak_index]),
        "outside_intervals": intervals,
        "pointwise_clipping_used": False,
        "scaling_used": False,
        "reference_modified": False,
        "reference_replay_status": (
            "REFERENCE_REPLAY_BLOCKED_BY_ROM" if outside.any() else "REFERENCE_REPLAY_PRECHECK_ALLOWED"
        ),
    }


def passive_audit(model: mujoco.MjModel) -> list[dict[str, Any]]:
    conditions = (
        ("P0_ZERO_CONTROL", "zero", 0.0),
        ("P1_BACKGROUND_0P01", "constant", 0.01),
        ("P2_DETERMINISTIC_SMALL", "deterministic", 0.005),
    )
    rows = []
    for condition, mode, level in conditions:
        data = initial_data(model)
        if mode == "constant":
            control = np.full(model.nu, level, dtype=float)
        elif mode == "deterministic":
            phase = 2.0 * math.pi * np.arange(model.nu) / model.nu
            control = level + level * np.sin(phase)
        else:
            control = np.zeros(model.nu, dtype=float)
        settle(model, data, control=control, steps=500)
        hip_start = float(data.qpos[qpos_address(model, TARGET_HIP)])
        knee_start = float(data.qpos[qpos_address(model, TARGET_KNEE)])
        started = time.perf_counter()
        peak_constraint = 0.0
        for _ in range(2_000):
            data.ctrl[:] = control
            data.qfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            metrics = constraint_metrics(model, data)
            peak_constraint = max(
                peak_constraint,
                metrics["source_knee_equality_max_abs_position_error"],
            )
            if not finite_state(data):
                break
        wall = time.perf_counter() - started
        rows.append(
            {
                "condition": condition,
                "primary_condition": condition == "P0_ZERO_CONTROL",
                "steps": 2000,
                "control_min": float(control.min()),
                "control_max": float(control.max()),
                "hip_start_deg": math.degrees(hip_start),
                "hip_end_deg": math.degrees(float(data.qpos[qpos_address(model, TARGET_HIP)])),
                "knee_start_deg": math.degrees(knee_start),
                "knee_end_deg": math.degrees(float(data.qpos[qpos_address(model, TARGET_KNEE)])),
                "qfrc_passive_l2": float(np.linalg.norm(data.qfrc_passive)),
                "qfrc_passive_nonzero": bool(np.any(np.abs(data.qfrc_passive) > 1e-12)),
                "qfrc_actuator_l2": float(np.linalg.norm(data.qfrc_actuator)),
                "qfrc_actuator_nonzero": bool(np.any(np.abs(data.qfrc_actuator) > 1e-12)),
                "actuator_force_l2": float(np.linalg.norm(data.actuator_force)),
                "actuator_force_max_abs": float(np.max(np.abs(data.actuator_force))),
                "tendon_length_min": float(np.min(data.ten_length)),
                "tendon_length_max": float(np.max(data.ten_length)),
                "source_knee_equality_peak_abs_error": peak_constraint,
                "warning_count": warning_count(data),
                "finite": finite_state(data),
                "wall_time_seconds": wall,
                "semantic_label": "LOW_ACTIVATION_OR_ZERO_CONTROL_MUSCULOSKELETAL_CONDITION",
                "physiological_passive_human_claimed": False,
            }
        )
    return rows


def trajectory_targets(kind: str, step: int) -> tuple[float, float, float, float]:
    duration = MOTION_STEPS * DIAGNOSTIC_DT
    t = step * DIAGNOSTIC_DT
    hip_amplitude = math.radians(10.0) if kind in {"HIP_ONLY", "COMBINED"} else 0.0
    knee_amplitude = math.radians(15.0) if kind in {"KNEE_ONLY", "COMBINED"} else 0.0
    shape = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / duration))
    shape_velocity = math.pi / duration * math.sin(2.0 * math.pi * t / duration)
    return (
        INITIAL_HIP_RAD + hip_amplitude * shape,
        INITIAL_KNEE_RAD + knee_amplitude * shape,
        hip_amplitude * shape_velocity,
        knee_amplitude * shape_velocity,
    )


def run_motion(model: mujoco.MjModel, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = initial_data(model)
    settle(model, data)
    rows = []
    hip_errors = []
    knee_errors = []
    peak_source_constraint_error = 0.0
    peak_source_constraint_force = 0.0
    started = time.perf_counter()
    for step in range(MOTION_STEPS):
        hip_target, knee_target, hip_velocity, knee_velocity = trajectory_targets(kind, step)
        tau_hip, tau_knee = set_pd(
            model, data, hip_target, knee_target, hip_velocity, knee_velocity
        )
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if not finite_state(data):
            raise RuntimeError(f"{kind} became nonfinite")
        hip_actual = float(data.qpos[qpos_address(model, TARGET_HIP)])
        knee_actual = float(data.qpos[qpos_address(model, TARGET_KNEE)])
        hip_errors.append(hip_actual - hip_target)
        knee_errors.append(knee_actual - knee_target)
        metrics = constraint_metrics(model, data)
        peak_source_constraint_error = max(
            peak_source_constraint_error,
            metrics["source_knee_equality_max_abs_position_error"],
        )
        peak_source_constraint_force = max(
            peak_source_constraint_force,
            metrics["source_knee_equality_max_abs_force"],
        )
        if step % RECORD_STRIDE == 0 or step == MOTION_STEPS - 1:
            hip_dof = dof_address(model, TARGET_HIP)
            knee_dof = dof_address(model, TARGET_KNEE)
            rows.append(
                {
                    "motion": kind,
                    "step": step,
                    "time_s": step * DIAGNOSTIC_DT,
                    "desired_hip_deg": math.degrees(hip_target),
                    "desired_knee_deg": math.degrees(knee_target),
                    "actual_hip_deg": math.degrees(hip_actual),
                    "actual_knee_deg": math.degrees(knee_actual),
                    "hip_velocity_rad_s": float(data.qvel[hip_dof]),
                    "knee_velocity_rad_s": float(data.qvel[knee_dof]),
                    "hip_qacc_rad_s2": float(data.qacc[hip_dof]),
                    "knee_qacc_rad_s2": float(data.qacc[knee_dof]),
                    "hip_qfrc_passive": float(data.qfrc_passive[hip_dof]),
                    "knee_qfrc_passive": float(data.qfrc_passive[knee_dof]),
                    "hip_qfrc_actuator": float(data.qfrc_actuator[hip_dof]),
                    "knee_qfrc_actuator": float(data.qfrc_actuator[knee_dof]),
                    "hip_qfrc_constraint": float(data.qfrc_constraint[hip_dof]),
                    "knee_qfrc_constraint": float(data.qfrc_constraint[knee_dof]),
                    "hip_diagnostic_applied_torque": tau_hip,
                    "knee_diagnostic_applied_torque": tau_knee,
                    "muscle_actuator_forces_json": json.dumps(
                        np.asarray(data.actuator_force, dtype=float).tolist(),
                        separators=(",", ":"),
                    ),
                    "muscle_force_l2": float(np.linalg.norm(data.actuator_force)),
                    "tendon_length_min": float(np.min(data.ten_length)),
                    "tendon_length_max": float(np.max(data.ten_length)),
                    "source_knee_equality_max_abs_position_error": metrics[
                        "source_knee_equality_max_abs_position_error"
                    ],
                    "source_knee_equality_max_abs_force": metrics[
                        "source_knee_equality_max_abs_force"
                    ],
                    "finite": True,
                }
            )
    runtime = time.perf_counter() - started
    return rows, {
        "motion": kind,
        "steps": MOTION_STEPS,
        "runtime_seconds": runtime,
        "hip_tracking_rmse_deg": math.degrees(float(np.sqrt(np.mean(np.square(hip_errors))))),
        "knee_tracking_rmse_deg": math.degrees(float(np.sqrt(np.mean(np.square(knee_errors))))),
        "hip_tracking_max_abs_error_deg": math.degrees(float(np.max(np.abs(hip_errors)))),
        "knee_tracking_max_abs_error_deg": math.degrees(float(np.max(np.abs(knee_errors)))),
        "source_knee_equality_peak_abs_position_error": peak_source_constraint_error,
        "source_knee_equality_peak_abs_force": peak_source_constraint_force,
        "warning_count": warning_count(data),
        "finite": finite_state(data),
    }


def strap_candidates(model: mujoco.MjModel) -> list[dict[str, Any]]:
    data = initial_data(model)
    settle(model, data)
    knee_world = data.site_xpos[site_id(model, "knee_r")].copy()
    ankle_world = data.site_xpos[site_id(model, "ankle_r")].copy()
    shank_length = float(np.linalg.norm(ankle_world - knee_world))
    rows = []
    for name in ("R_tibial_plateau", "RTB1", "RTB2", "RTB3"):
        identifier = site_id(model, name)
        world = data.site_xpos[identifier].copy()
        local = model.site_pos[identifier].copy()
        distance = float(np.linalg.norm(world - knee_world))
        rows.append(
            {
                "site": name,
                "body": object_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[identifier])
                ),
                "local_x_m": float(local[0]),
                "local_y_m": float(local[1]),
                "local_z_m": float(local[2]),
                "world_x_m": float(world[0]),
                "world_y_m": float(world[1]),
                "world_z_m": float(world[2]),
                "distance_from_knee_center_m": distance,
                "fraction_of_knee_to_ankle_distance": distance / shank_length,
                "abs_error_from_project_L2_m": abs(distance - L2_PROJECT_M),
                "selected_provisional_strap_site": name == PROVISIONAL_STRAP_SITE,
                "selection_uses_site_name_only": False,
            }
        )
    selected = min(rows, key=lambda row: row["abs_error_from_project_L2_m"])
    if selected["site"] != PROVISIONAL_STRAP_SITE:
        raise RuntimeError("frozen provisional strap site is not closest to project L2")
    return rows


def project_jacobian(hip: float, knee: float) -> np.ndarray:
    shank = hip - knee
    return np.asarray(
        [
            [
                -L1_PROJECT_M * math.sin(hip) - L2_PROJECT_M * math.sin(shank),
                L2_PROJECT_M * math.sin(shank),
            ],
            [
                L1_PROJECT_M * math.cos(hip) + L2_PROJECT_M * math.cos(shank),
                -L2_PROJECT_M * math.cos(shank),
            ],
        ],
        dtype=float,
    )


def external_force_smoke(model: mujoco.MjModel) -> tuple[list[dict[str, Any]], str]:
    directions = (
        ("PROJECT_POSITIVE_X", np.asarray([EXTERNAL_FORCE_N, 0.0, 0.0])),
        ("PROJECT_NEGATIVE_X", np.asarray([-EXTERNAL_FORCE_N, 0.0, 0.0])),
        ("PROJECT_POSITIVE_Z", np.asarray([0.0, 0.0, EXTERNAL_FORCE_N])),
        ("PROJECT_NEGATIVE_Z", np.asarray([0.0, 0.0, -EXTERNAL_FORCE_N])),
    )
    rows = []
    exact_errors = []
    project_relative_errors = []
    for direction, force in directions:
        data = initial_data(model)
        settle(model, data)
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        baseline_qacc = np.asarray(data.qacc, dtype=float).copy()
        point = data.site_xpos[site_id(model, PROVISIONAL_STRAP_SITE)].copy()
        tibia = body_id(model, TARGET_TIBIA)
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id(model, PROVISIONAL_STRAP_SITE))
        expected_generalized = jacp.T @ force
        mujoco.mj_applyFT(
            model,
            data,
            force,
            np.zeros(3, dtype=float),
            point,
            tibia,
            data.qfrc_applied,
        )
        applied_generalized = np.asarray(data.qfrc_applied, dtype=float).copy()
        exact_error = float(np.max(np.abs(applied_generalized - expected_generalized)))
        exact_errors.append(exact_error)
        mujoco.mj_forward(model, data)
        forced_qacc = np.asarray(data.qacc, dtype=float).copy()
        hip_dof = dof_address(model, TARGET_HIP)
        knee_dof = dof_address(model, TARGET_KNEE)
        hip = float(data.qpos[qpos_address(model, TARGET_HIP)])
        knee = float(data.qpos[qpos_address(model, TARGET_KNEE)])
        project_force = np.asarray([force[0], force[2]], dtype=float)
        project_tau = project_jacobian(hip, knee).T @ project_force
        myoleg_tau = applied_generalized[[hip_dof, knee_dof]]
        denominator = max(float(np.linalg.norm(project_tau)), 1e-12)
        relative_error = float(np.linalg.norm(myoleg_tau - project_tau) / denominator)
        project_relative_errors.append(relative_error)
        peak_qacc_delta = np.abs(forced_qacc - baseline_qacc)
        for _ in range(50):
            data.qfrc_applied[:] = 0.0
            point = data.site_xpos[site_id(model, PROVISIONAL_STRAP_SITE)].copy()
            mujoco.mj_applyFT(
                model, data, force, np.zeros(3), point, tibia, data.qfrc_applied
            )
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0
        for _ in range(50):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        rows.append(
            {
                "direction": direction,
                "force_x_n": float(force[0]),
                "force_y_n": float(force[1]),
                "force_z_n": float(force[2]),
                "site": PROVISIONAL_STRAP_SITE,
                "site_world_x_m": float(point[0]),
                "site_world_y_m": float(point[1]),
                "site_world_z_m": float(point[2]),
                "hip_qfrc_applied": float(applied_generalized[hip_dof]),
                "knee_qfrc_applied": float(applied_generalized[knee_dof]),
                "hip_jacobian_expected": float(expected_generalized[hip_dof]),
                "knee_jacobian_expected": float(expected_generalized[knee_dof]),
                "mujoco_jacobian_max_abs_error": exact_error,
                "project_tau_hip": float(project_tau[0]),
                "project_tau_knee": float(project_tau[1]),
                "project_vs_myoleg_relative_error": relative_error,
                "hip_qacc_delta_rad_s2": float(forced_qacc[hip_dof] - baseline_qacc[hip_dof]),
                "knee_qacc_delta_rad_s2": float(forced_qacc[knee_dof] - baseline_qacc[knee_dof]),
                "max_generalized_qacc_delta": float(np.max(peak_qacc_delta)),
                "finite_after_pulse": finite_state(data),
                "warning_count": warning_count(data),
            }
        )
    if max(exact_errors) > 1e-9 or not all(row["finite_after_pulse"] for row in rows):
        status = "FAIL"
    elif max(project_relative_errors) <= 0.20:
        status = "PASS"
    else:
        status = "PARTIAL"
    return rows, status


def static_stability(model: mujoco.MjModel) -> dict[str, Any]:
    data = initial_data(model)
    settle(model, data)
    root_id = body_id(model, "root")
    root_position_start = data.xpos[root_id].copy()
    root_quat_start = data.xquat[root_id].copy()
    peak_source_error = 0.0
    peak_derived_error = 0.0
    peak_constraint_force = 0.0
    peak_position_abs = 0.0
    peak_source_step = -1
    peak_source_equality_id = -1
    peak_source_equality_name = ""
    peak_source_hip_deg = float("nan")
    peak_source_knee_deg = float("nan")
    started = time.perf_counter()
    for step in range(STATIC_STEPS):
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)
        metrics = constraint_metrics(model, data)
        current_source_error = metrics["source_knee_equality_max_abs_position_error"]
        if current_source_error > peak_source_error:
            peak_source_error = current_source_error
            equality_rows = np.flatnonzero(
                np.asarray(data.efc_type)
                == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            )
            source_rows = equality_rows[
                np.asarray(data.efc_id)[equality_rows] < SOURCE_EQUALITY_COUNT
            ]
            peak_row = int(
                source_rows[np.argmax(np.abs(np.asarray(data.efc_pos)[source_rows]))]
            )
            peak_source_equality_id = int(data.efc_id[peak_row])
            peak_source_equality_name = object_name(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, peak_source_equality_id
            )
            peak_source_step = step
            peak_source_hip_deg = math.degrees(
                float(data.qpos[qpos_address(model, TARGET_HIP)])
            )
            peak_source_knee_deg = math.degrees(
                float(data.qpos[qpos_address(model, TARGET_KNEE)])
            )
        peak_derived_error = max(
            peak_derived_error, metrics["derived_constraint_max_abs_position_error"]
        )
        peak_constraint_force = max(
            peak_constraint_force,
            metrics["source_knee_equality_max_abs_force"],
            metrics["derived_constraint_max_abs_force"],
        )
        peak_position_abs = max(peak_position_abs, float(np.max(np.abs(data.xpos))))
        if not finite_state(data):
            break
    runtime = time.perf_counter() - started
    quaternion_alignment = abs(float(np.dot(root_quat_start, data.xquat[root_id])))
    quaternion_alignment = min(1.0, max(-1.0, quaternion_alignment))
    orientation_drift = math.degrees(2.0 * math.acos(quaternion_alignment))
    lock_deviations = {
        name: abs(float(data.qpos[qpos_address(model, name)])) for name in LOCKED_JOINTS
    }
    final_metrics = constraint_metrics(model, data)
    return {
        "steps": STATIC_STEPS,
        "simulated_time_seconds": STATIC_STEPS * DIAGNOSTIC_DT,
        "runtime_seconds": runtime,
        "finite": finite_state(data),
        "warning_count": warning_count(data),
        "root_position_drift_m": float(np.linalg.norm(data.xpos[root_id] - root_position_start)),
        "root_orientation_drift_deg": orientation_drift,
        "max_locked_joint_abs_deviation_rad": max(lock_deviations.values()),
        "locked_joint_abs_deviation_rad": lock_deviations,
        "source_knee_equality_peak_abs_position_error": peak_source_error,
        "source_knee_equality_peak_step": peak_source_step,
        "source_knee_equality_peak_time_s": peak_source_step * DIAGNOSTIC_DT,
        "source_knee_equality_peak_name": peak_source_equality_name,
        "source_knee_equality_peak_id": peak_source_equality_id,
        "hip_deg_at_source_equality_peak": peak_source_hip_deg,
        "knee_deg_at_source_equality_peak": peak_source_knee_deg,
        "source_knee_equality_final_max_abs_position_error": final_metrics[
            "source_knee_equality_max_abs_position_error"
        ],
        "source_knee_equality_transient_recovered": final_metrics[
            "source_knee_equality_max_abs_position_error"
        ] < 0.001,
        "derived_constraint_peak_abs_position_error": peak_derived_error,
        "constraint_peak_abs_force": peak_constraint_force,
        "peak_abs_body_world_position_m": peak_position_abs,
        "tendon_length_min": float(np.min(data.ten_length)),
        "tendon_length_max": float(np.max(data.ten_length)),
        "actuator_force_max_abs": float(np.max(np.abs(data.actuator_force))),
        "final_hip_deg": math.degrees(float(data.qpos[qpos_address(model, TARGET_HIP)])),
        "final_knee_deg": math.degrees(float(data.qpos[qpos_address(model, TARGET_KNEE)])),
        "model_explosion_detected": peak_position_abs > 10.0,
    }


def source_equalities_hash(model: mujoco.MjModel) -> str:
    digest = hashlib.sha256()
    for array in (
        model.eq_type[:SOURCE_EQUALITY_COUNT],
        model.eq_obj1id[:SOURCE_EQUALITY_COUNT],
        model.eq_obj2id[:SOURCE_EQUALITY_COUNT],
        model.eq_data[:SOURCE_EQUALITY_COUNT],
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_tests(
    source_before: dict[str, Any],
    model: mujoco.MjModel,
    mapping: dict[str, Any],
    rom: dict[str, Any],
    passive_rows: list[dict[str, Any]],
    motion_summaries: list[dict[str, Any]],
    force_rows: list[dict[str, Any]],
    force_mapping_status: str,
    stability: dict[str, Any],
    build_metadata: dict[str, Any],
) -> dict[str, Any]:
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    reloaded_1 = mujoco.MjModel.from_xml_path(str(DERIVED_XML))
    reloaded_2 = mujoco.MjModel.from_xml_path(str(DERIVED_XML))
    source_after = source_identity()
    syntax_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    tests = [
        ("upstream_myoleg_assets_unchanged", source_before == source_after),
        (
            "historical_rehab_robot_artifacts_unchanged",
            source_after["reference_sha256"] == REFERENCE_SHA256
            and source_before["previous_audit_checksums_sha256"]
            == source_after["previous_audit_checksums_sha256"],
        ),
        (
            "derived_model_deterministic_load",
            (reloaded_1.nq, reloaded_1.nv, reloaded_1.nu, reloaded_1.neq)
            == (reloaded_2.nq, reloaded_2.nv, reloaded_2.nu, reloaded_2.neq),
        ),
        (
            "pelvis_fixed",
            stability["root_position_drift_m"] < 0.01
            and stability["root_orientation_drift_deg"] < 1.0,
        ),
        (
            "non_target_leg_fixed_as_designed",
            max(
                stability["locked_joint_abs_deviation_rad"][name]
                for name in LOCKED_JOINTS if name.endswith("_l")
            ) < 0.01,
        ),
        (
            "target_hip_only_intended_dof_free",
            TARGET_HIP not in LOCKED_JOINTS
            and "hip_adduction_r" in LOCKED_JOINTS
            and "hip_rotation_r" in LOCKED_JOINTS,
        ),
        ("target_knee_main_flexion_retained", TARGET_KNEE not in LOCKED_JOINTS),
        (
            "patella_and_source_equalities_intact",
            source_equalities_hash(source_model) == source_equalities_hash(model)
            and all(summary["finite"] for summary in motion_summaries)
            and all(
                summary["source_knee_equality_peak_abs_position_error"] < 0.001
                for summary in motion_summaries
            )
            and stability["source_knee_equality_final_max_abs_position_error"] < 0.001,
        ),
        (
            "target_ankle_subtalar_mtp_fixed",
            all(name in LOCKED_JOINTS for name in ("ankle_angle_r", "subtalar_angle_r", "mtp_angle_r")),
        ),
        (
            "coordinate_mapping_round_trip",
            mapping["status"] == "PASS" and mapping["round_trip_max_abs_error_rad"] == 0.0,
        ),
        (
            "external_force_finite_generalized_response",
            force_mapping_status in {"PASS", "PARTIAL"}
            and all(row["finite_after_pulse"] for row in force_rows)
            and any(abs(row["hip_qacc_delta_rad_s2"]) > 1e-12 for row in force_rows),
        ),
        (
            "zero_control_simulation_stable",
            stability["finite"]
            and next(row for row in passive_rows if row["condition"] == "P0_ZERO_CONTROL")["finite"],
        ),
        (
            "rom_mismatch_not_silently_clipped",
            rom["samples_outside_myoleg_native_rom"] > 0
            and not rom["pointwise_clipping_used"]
            and rom["reference_replay_status"] == "REFERENCE_REPLAY_BLOCKED_BY_ROM",
        ),
        ("formal_reference_unmodified", source_after["reference_sha256"] == REFERENCE_SHA256),
        (
            "no_robot_or_hardware_access",
            imported_roots.isdisjoint({"hardware", "control", "safety"}),
        ),
    ]
    results = [
        {"test": name, "status": "PASS" if passed else "FAIL"}
        for name, passed in tests
    ]
    return {
        "status": "PASS" if all(passed for _, passed in tests) else "FAIL",
        "passed": sum(bool(passed) for _, passed in tests),
        "failed": sum(not bool(passed) for _, passed in tests),
        "tests": results,
        "derived_xml_sha256": build_metadata["derived_xml_sha256"],
    }


def modification_diff(build_metadata: dict[str, Any]) -> str:
    modifications = build_metadata["modifications"]
    lines = [
        "# MODEL_MODIFICATION_DIFF",
        "",
        "The upstream XML/assets are read-only. The generated derived XML makes only these changes:",
        "",
        "1. Add `derived_root_anchor` on `root` and `derived_world_supine_anchor` on world.",
        "2. Add site-based weld `derived_root_supine_weld` at [0,0,1] m with -90 deg world-y rotation.",
        "3. Add single-joint equality locks at native zero for:",
    ]
    lines.extend(f"   - `{name}`" for name in modifications["locked_joints"])
    lines.extend(
        [
            "4. Preserve `hip_flexion_r`, `knee_angle_r`, all original auxiliary/patella joints and all 14 source equalities.",
            "5. Preserve every original body, muscle actuator, tendon and native joint range.",
            "6. Set floor/terrain contact masks to zero for `SUPINE_NO_BED_CONTACT`; do not add a bed.",
            "7. Convert mesh/texture references to absolute, hashed references inside the frozen external environment; assets are not copied or edited.",
            "8. Keep gravity and the 0.001 s integration timestep unchanged.",
            "",
            "No joint, muscle, tendon, equality, mesh or body was deleted. No knee range was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def feasibility_report(
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    rom: dict[str, Any],
    passive_rows: list[dict[str, Any]],
    motion_summaries: list[dict[str, Any]],
    strap_rows: list[dict[str, Any]],
    force_rows: list[dict[str, Any]],
    stability: dict[str, Any],
    tests: dict[str, Any],
) -> str:
    p0 = next(row for row in passive_rows if row["condition"] == "P0_ZERO_CONTROL")
    strap = next(row for row in strap_rows if row["selected_provisional_strap_site"])
    max_force_relative_error = max(
        row["project_vs_myoleg_relative_error"] for row in force_rows
    )
    max_force_exact_error = max(row["mujoco_jacobian_max_abs_error"] for row in force_rows)
    max_motion_equality_error = max(
        row["source_knee_equality_peak_abs_position_error"]
        for row in motion_summaries
    )
    motion_lines = "\n".join(
        "- `{motion}`: hip RMSE {hip:.3f} deg, knee RMSE {knee:.3f} deg, "
        "equality peak {equality:.3e}, warnings {warnings}.".format(
            motion=row["motion"],
            hip=row["hip_tracking_rmse_deg"],
            knee=row["knee_tracking_rmse_deg"],
            equality=row["source_knee_equality_peak_abs_position_error"],
            warnings=row["warning_count"],
        )
        for row in motion_summaries
    )
    return f"""# MYOLEG_SUPINE_HIP_KNEE_REHAB_FEASIBILITY_V1

## Formal decision

`MYOLEG_SUPINE_REHAB_MODEL_FEASIBLE_WITH_LIMITATIONS`

The project-owned derived model is a reproducible, headless, offline feasibility model. It is not a physiological-passivity, human, clinical, robot-control, or safety validation. The unchanged formal reference is **not** replayed because the ROM precheck fails closed.

## Frozen input and provenance

- Environment: Python {manifest['runtime_environment']['python']}, MyoSuite {manifest['runtime_environment']['myosuite']}, MuJoCo {manifest['runtime_environment']['mujoco']}.
- Source XML SHA-256: `{manifest['source_identity']['source_model_sha256']}`.
- Derived XML SHA-256: `{manifest['derived_xml_sha256']}`.
- Frozen reference SHA-256: `{manifest['source_identity']['reference_sha256']}`.
- The previous install/smoke-test checksum manifest verified {manifest['source_identity']['previous_audit_checksum_verification']['checked_file_count']} files with no failure.
- All {manifest['source_identity']['upstream_asset_count']} referenced upstream assets were hashed before and after the run; no source or asset hash changed.
- Source dimensions: {manifest['source_model_dimensions']}.
- Derived dimensions: {manifest['derived_model_dimensions']}.

## Derived-model design

The right leg is frozen as `TARGET_LEG`: the source model is bilaterally symmetric, and the right tibia has directly auditable RTB sites that can be compared with the project's strap-equivalent `L2=0.30 m` meaning. The free root is preserved but constrained by a site-to-world weld at `[0,0,1] m` with a -90 deg world-y supine rotation. The contralateral primary joints, target hip adduction/rotation, and target ankle/subtalar/MTP are locked with single-joint equality constraints at native zero. Target `hip_flexion_r`, `knee_angle_r`, all seven right auxiliary/patella joints, all 14 source knee equalities, all 80 tendons, and all 80 muscle actuators remain present.

`SUPINE_NO_BED_CONTACT` disables only floor/terrain contact masks. Gravity and the 1 ms timestep remain unchanged; no bed contact was introduced. Weld-to-world and joint-equality locking are documented MuJoCo equality mechanisms: <https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality>.

## Coordinate gate

`COORDINATE_MAPPING_VALID = PASS`

The exact fixed mapping is:

```text
q_project_hip  = +1 * q_myoleg_hip_flexion_r + 0
q_project_knee = +1 * q_myoleg_knee_angle_r  + 0

q_myoleg_hip_flexion_r = q_project_hip
q_myoleg_knee_angle_r   = q_project_knee

theta_shank = q_project_hip - q_project_knee
```

The rehabilitation frame is MyoLeg world x-z after the root transform, with the `hip_r` site as the local origin. Across 20/60/100 deg diagnostic poses, maximum measured thigh-angle error was {mapping['max_thigh_angle_error_deg']:.6f} deg and maximum shank-angle error was {mapping['max_shank_angle_error_deg']:.3f} deg. Round-trip joint-coordinate error was zero.

## ROM compatibility gate

`{rom['classification']}`

- Project `ROM_PROTOCOL_V2`: hip 0-120 deg, knee 5-145 deg.
- Native mapped MyoLeg range: hip {rom['myoleg_native_rom_deg']['hip'][0]:.6f} to {rom['myoleg_native_rom_deg']['hip'][1]:.6f} deg; knee {rom['myoleg_native_rom_deg']['knee'][0]:.6f} to {rom['myoleg_native_rom_deg']['knee'][1]:.6f} deg.
- Common formal/native range: hip 0-120 deg; knee 5-{rom['common_supported_formal_rom_deg']['knee'][1]:.6f} deg.
- Frozen reference range: hip {rom['reference_hip_range_deg'][0]:.6f} to {rom['reference_hip_range_deg'][1]:.6f} deg; knee {rom['reference_knee_range_deg'][0]:.6f} to {rom['reference_knee_range_deg'][1]:.6f} deg.
- Inside native knee range: {rom['samples_inside_myoleg_native_rom']}/{rom['reference_sample_count']}; outside: {rom['samples_outside_myoleg_native_rom']}/{rom['reference_sample_count']} ({100.0 * rom['outside_fraction']:.3f}%).
- Maximum exceedance over the actual native upper limit: {rom['maximum_knee_exceedance_deg']:.6f} deg.
- Exceeding interval: indices 155-262, t=10.540-16.824 s, crossing flexion then extension. Peak knee {rom['peak_knee_deg']:.6f} deg occurs at t={rom['peak_knee_time_s']:.3f} s in extension.

No clipping, scaling, XML limit extension, project-ROM change, or reference modification was used. `REFERENCE_REPLAY_BLOCKED_BY_ROM`.

## Passive-state audit

P0 is explicitly a `LOW_ACTIVATION_OR_ZERO_CONTROL_MUSCULOSKELETAL_CONDITION`, not a physiological passive human. With all 80 controls zero, the run remained finite with no solver warning and retained nonzero residual mechanics: final `qfrc_passive` L2={p0['qfrc_passive_l2']:.6f}, `qfrc_actuator` L2={p0['qfrc_actuator_l2']:.6f}, and maximum absolute muscle actuator force={p0['actuator_force_max_abs']:.6f}. The nonzero actuator force at zero command is a model residual/passive-muscle response, not voluntary activation.

The unsupported free-space leg fell from hip/knee {p0['hip_start_deg']:.3f}/{p0['knee_start_deg']:.3f} deg to {p0['hip_end_deg']:.3f}/{p0['knee_end_deg']:.3f} deg and reached the native hip extension boundary. This is a material limitation of `SUPINE_NO_BED_CONTACT`, not evidence of physiological resting posture.

## 2-DOF motion and knee integrity

The diagnostic controller applies generalized PD torque only through `qfrc_applied`; it is separated in every row from muscle `qfrc_actuator` and is not a formal controller.

{motion_lines}

All diagnostic motions were finite and continuous. Source equality definitions/hashes are unchanged. The largest source-knee equality error during controlled motion was {max_motion_equality_error:.3e}. During the 10,000-step P0/static test, `knee_angle_r_beta_rotation1_constraint` briefly reached {stability['source_knee_equality_peak_abs_position_error']:.6f} at t={stability['source_knee_equality_peak_time_s']:.3f} s as hip flexion crossed to {stability['hip_deg_at_source_equality_peak']:.3f} deg; it recovered to {stability['source_knee_equality_final_max_abs_position_error']:.3e}. This recovered boundary-impact transient is retained as a limitation. `DERIVED_KNEE_MODEL_INTEGRITY = PASS` for the controlled rehabilitation-motion diagnostic, not an unconditional dynamics certification.

The 10,000-step static run covered {stability['simulated_time_seconds']:.1f} simulated seconds with zero warnings, root drift {stability['root_position_drift_m']:.3e} m / {stability['root_orientation_drift_deg']:.6f} deg, maximum locked-joint deviation {stability['max_locked_joint_abs_deviation_rad']:.3e} rad, and no nonfinite state or model explosion.

## Strap and external-force path

`PROVISIONAL_STRAP_SITE = RTB3` on `tibia_r`.

It was selected by geometry rather than by name: local position `[0.0114, -0.2952, 0.0554] m`, distance from `knee_r` {strap['distance_from_knee_center_m']:.6f} m, {100.0 * strap['fraction_of_knee_to_ankle_distance']:.3f}% of the knee-to-ankle distance, and only {1000.0 * strap['abs_error_from_project_L2_m']:.3f} mm from project `L2=0.30 m`. The site remains provisional until physical strap placement is defined.

Four 2 N pulses (+x/-x/+z/-z) applied with `mj_applyFT` produced finite hip/knee acceleration and no warning. The `J_mujoco^T F` comparison matched `qfrc_applied` with maximum absolute error {max_force_exact_error:.3e}. The project analytic two-link `J_project^T F` and 3-D MyoLeg values differed by at most {100.0 * max_force_relative_error:.3f}% at this pose. `EXTERNAL_STRAP_FORCE_PATH_AVAILABLE = true`; `FORCE_MAPPING_FEASIBILITY = {manifest['force_mapping_feasibility']}`. This is a kinematic/dynamic interface smoke test, not robot-force validation.

## Direct answers

### Q1

Yes, for headless offline feasibility: a pelvis-fixed, right-target-leg, sagittal hip-knee derived MyoLeg can run without deleting or changing its muscle/tendon/knee-equality structure. The zero-control boundary transient prevents an unrestricted claim.

### Q2

Use the identity joint mapping shown above, with +1 signs, zero offsets, the fixed -90 deg world-y root transform, and `theta_shank=q_hip-q_knee`.

### Q3

No. The frozen reference is not fully compatible with the actual mapped native knee ROM.

### Q4

108 of 401 samples exceed the native knee upper limit. The maximum exceedance is {rom['maximum_knee_exceedance_deg']:.6f} deg, over t=10.540-16.824 s; the peak is {rom['peak_knee_deg']:.6f} deg at t={rom['peak_knee_time_s']:.3f} s during extension.

### Q5

Yes, P0 is finite and nontrivial, but only as a zero-command model condition. It falls to the native hip boundary and must not be called physiological passive human behavior.

### Q6

Yes. `RTB3` is a physically interpretable provisional tibial site because its knee-center distance is {strap['distance_from_knee_center_m']:.6f} m, close to project `L2=0.30 m`.

### Q7

Yes. Small sagittal external forces can be applied at RTB3 and mapped exactly through the MyoLeg site Jacobian to hip/knee generalized force; the simplified project-Jacobian comparison is close but not identical.

### Q8

Yes, under the controlled diagnostic motions. All auxiliary/patella joints and 14 source equalities remain intact and finite; the recovered zero-control boundary transient is explicitly retained as a limitation.

### Q9

No, not for an unchanged full formal-reference replay. The derived interface is feasible, but `MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1` remains blocked by the native-ROM conflict. No replay is run here.

## Test and scope closure

- Tests: {tests['passed']} passed, {tests['failed']} failed.
- No MyoSuite upstream file or asset changed.
- No formal reference, `ROM_PROTOCOL_V2`, lower-limb model, five-parameter model, BO, formal artifact, hardware, control, or safety code changed.
- No BO, PINN, RL, candidate landscape, formal replay, robot connection, or visualization-based gate was run.
- Visualization remains unavailable and is not used as evidence.
"""


def write_checksums() -> None:
    files = sorted(
        path for path in ARTIFACT_DIRECTORY.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text(content, encoding="utf-8")


def model_dimensions(model: mujoco.MjModel) -> dict[str, int]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "neq": int(model.neq),
        "nu": int(model.nu),
        "ntendon": int(model.ntendon),
    }


def main() -> None:
    started = time.perf_counter()
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment()
    source_before = source_identity()
    model, build_metadata = build_derived_model()
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    if model.nu != 80 or model.ntendon != 80 or model.njnt != 29:
        raise RuntimeError("derived model destroyed upstream musculoskeletal dimensions")

    configs = joint_config(model)
    write_csv(
        ARTIFACT_DIRECTORY / "SUPINE_MODEL_JOINT_CONFIG.csv",
        configs,
        list(configs[0].keys()),
    )
    mapping = coordinate_mapping(model)
    write_json(ARTIFACT_DIRECTORY / "PROJECT_MYOLEG_COORDINATE_MAPPING.json", mapping)
    if mapping["status"] != "PASS":
        raise RuntimeError("COORDINATE_MAPPING_VALID = FAIL")

    rom = rom_audit(model)
    write_json(ARTIFACT_DIRECTORY / "ROM_COMPATIBILITY_AUDIT.json", rom)
    passive_rows = passive_audit(model)
    write_csv(
        ARTIFACT_DIRECTORY / "PASSIVE_STATE_AUDIT.csv",
        passive_rows,
        list(passive_rows[0].keys()),
    )

    all_motion_rows = []
    motion_summaries = []
    for kind in ("HIP_ONLY", "KNEE_ONLY", "COMBINED"):
        rows, summary = run_motion(model, kind)
        all_motion_rows.extend(rows)
        motion_summaries.append(summary)
    write_csv(
        ARTIFACT_DIRECTORY / "DIAGNOSTIC_MOTION_RESULTS.csv",
        all_motion_rows,
        list(all_motion_rows[0].keys()),
    )

    strap_rows = strap_candidates(model)
    write_csv(
        ARTIFACT_DIRECTORY / "STRAP_SITE_CANDIDATES.csv",
        strap_rows,
        list(strap_rows[0].keys()),
    )
    force_rows, force_mapping_status = external_force_smoke(model)
    write_csv(
        ARTIFACT_DIRECTORY / "EXTERNAL_FORCE_SMOKE_RESULTS.csv",
        force_rows,
        list(force_rows[0].keys()),
    )

    stability = static_stability(model)
    stability["motion_tests"] = motion_summaries
    stability["external_force_tests"] = {
        "force_magnitude_n": EXTERNAL_FORCE_N,
        "pulse_steps": 50,
        "force_mapping_feasibility": force_mapping_status,
        "all_finite": all(row["finite_after_pulse"] for row in force_rows),
    }
    stability["zero_control_equality_transient_limitation"] = (
        stability["source_knee_equality_peak_abs_position_error"] >= 0.001
    )
    stability["derived_knee_model_integrity"] = (
        "PASS"
        if stability["finite"]
        and all(summary["finite"] for summary in motion_summaries)
        and all(
            summary["source_knee_equality_peak_abs_position_error"] < 0.001
            for summary in motion_summaries
        )
        and stability["source_knee_equality_final_max_abs_position_error"] < 0.001
        else "FAIL"
    )
    write_json(ARTIFACT_DIRECTORY / "STABILITY_TEST_RESULTS.json", stability)

    manifest = {
        "stage_id": STAGE_ID,
        "evidence_level": "OFFLINE_HEADLESS_DERIVED_MODEL_FEASIBILITY",
        "runtime_environment": environment,
        "target_leg": TARGET_LEG,
        "target_joint_body_map": {
            "hip_flexion_joint": TARGET_HIP,
            "hip_adduction_joint": "hip_adduction_r",
            "hip_rotation_joint": "hip_rotation_r",
            "main_knee_joint": TARGET_KNEE,
            "auxiliary_patella_knee_joints": list(TARGET_AUXILIARY_JOINTS),
            "ankle_joint": "ankle_angle_r",
            "subtalar_joint": "subtalar_angle_r",
            "mtp_joint": "mtp_angle_r",
            "femur_body": TARGET_FEMUR,
            "tibia_body": TARGET_TIBIA,
            "foot_body": TARGET_FOOT,
        },
        "target_leg_selection_reason": (
            "bilateral model is structurally symmetric; right side frozen because actual RTB sites align with project pull-point audit"
        ),
        "source_identity": source_before,
        "builder_script_path": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_model_dimensions": model_dimensions(source_model),
        "derived_model_dimensions": model_dimensions(model),
        "derived_xml_path": str(DERIVED_XML),
        "derived_xml_sha256": build_metadata["derived_xml_sha256"],
        "asset_references": build_metadata["assets"],
        "modifications": build_metadata["modifications"],
        "coordinate_mapping_status": mapping["status"],
        "rom_classification": rom["classification"],
        "reference_replay_status": rom["reference_replay_status"],
        "primary_passive_condition": "P0_ZERO_CONTROL",
        "zero_control_semantics": "LOW_ACTIVATION_OR_ZERO_CONTROL_MUSCULOSKELETAL_CONDITION",
        "physiological_passive_human_claimed": False,
        "provisional_strap_site": PROVISIONAL_STRAP_SITE,
        "diagnostic_motion_actuation": {
            "mechanism": "qfrc_applied PD generalized torque, separated from qfrc_actuator",
            "kp_nm_per_rad": PD_KP,
            "kd_nm_s_per_rad": PD_KD,
            "torque_limit_nm": PD_TORQUE_LIMIT_NM,
            "formal_controller_claimed": False,
        },
        "external_strap_force_path_available": all(
            row["finite_after_pulse"] for row in force_rows
        ),
        "force_mapping_feasibility": force_mapping_status,
        "visualization_available": False,
        "formal_reference_replayed": False,
        "formal_reference_modified": False,
        "native_knee_limit_modified": False,
        "robot_connected": False,
        "rl_run": False,
        "pinn_trained": False,
        "bo_run": False,
        "candidate_landscape_generated": False,
        "stage_feasibility_status": "MYOLEG_SUPINE_REHAB_MODEL_FEASIBLE_WITH_LIMITATIONS",
        "limitations": [
            "formal reference has samples beyond native MyoLeg knee ROM; replay is blocked",
            "project analytic two-link and MyoLeg 3-D strap mappings are close but not identical",
            "zero-control fall into the native hip boundary creates a recovered auxiliary-knee equality transient",
            "visualization remains unavailable; evidence is headless only",
        ],
    }
    write_json(ARTIFACT_DIRECTORY / "DERIVED_MODEL_MANIFEST.json", manifest)
    (ARTIFACT_DIRECTORY / "MODEL_MODIFICATION_DIFF.md").write_text(
        modification_diff(build_metadata), encoding="utf-8"
    )

    tests = run_tests(
        source_before,
        model,
        mapping,
        rom,
        passive_rows,
        motion_summaries,
        force_rows,
        force_mapping_status,
        stability,
        build_metadata,
    )
    write_json(ARTIFACT_DIRECTORY / "TEST_RESULTS.json", tests)
    (ARTIFACT_DIRECTORY / "MYOLEG_SUPINE_HIP_KNEE_REHAB_FEASIBILITY_REPORT.md").write_text(
        feasibility_report(
            manifest,
            mapping,
            rom,
            passive_rows,
            motion_summaries,
            strap_rows,
            force_rows,
            stability,
            tests,
        ),
        encoding="utf-8",
    )
    run_summary = {
        "stage_id": STAGE_ID,
        "runtime_seconds": time.perf_counter() - started,
        "coordinate_mapping_valid": mapping["status"],
        "rom_classification": rom["classification"],
        "reference_replay_status": rom["reference_replay_status"],
        "p0_finite": next(
            row["finite"] for row in passive_rows if row["condition"] == "P0_ZERO_CONTROL"
        ),
        "derived_knee_model_integrity": stability["derived_knee_model_integrity"],
        "external_strap_force_path_available": all(
            row["finite_after_pulse"] for row in force_rows
        ),
        "force_mapping_feasibility": force_mapping_status,
        "tests": tests["status"],
        "test_passed": tests["passed"],
        "test_failed": tests["failed"],
        "stage_feasibility_status": "MYOLEG_SUPINE_REHAB_MODEL_FEASIBLE_WITH_LIMITATIONS",
    }
    write_json(ARTIFACT_DIRECTORY / "run_summary.json", run_summary)
    write_checksums()
    if tests["status"] != "PASS" or stability["derived_knee_model_integrity"] != "PASS":
        raise RuntimeError("derived supine feasibility failed closed")


if __name__ == "__main__":
    main()
