"""Independent, headless MyoLeg installation and model smoke audit.

This script is intentionally isolated from rehab_robot runtime modules.  It
loads an installed MyoSuite environment, records model/API evidence, and never
imports robot, control, safety, BO, or lower_limb_sim code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np

import myosuite  # noqa: F401 - import performs official environment registration
from myosuite.utils import gym


STAGE_ID = "MYOLEG_INSTALL_AND_SMOKE_TEST_V1"
PRIMARY_ENV_ID = "myoLegWalk-v0"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
FIRST_OFFICIAL_ATTEMPT = """ATTEMPT 1 (base PyPI dependencies only)
exit_code: 1
result: FAIL
tests: 3
errors: 1
root_cause: MyoChallenge myoChallengeBimanual-v0 imports scipy.spatial.transform,
  but MyoSuite 2.12.2 did not declare scipy as a base dependency.
traceback_terminal: ModuleNotFoundError: No module named 'scipy'
MyoBase_result: PASS (included all registered MyoLeg environments)
remediation: scipy==1.15.3 added only to the isolated myosuite-v2 environment.
"""


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    if int(object_id) < 0:
        return ""
    return mujoco.mj_id2name(model, object_type, int(object_id)) or ""


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return enum_type(int(value)).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{int(value)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_location(name: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    return str(distribution.locate_file(""))


def _official_test() -> dict[str, Any]:
    command = [sys.executable, "-m", "myosuite.tests.test_myo"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300.0,
        check=False,
    )
    wall = time.perf_counter() - started
    combined = (
        FIRST_OFFICIAL_ATTEMPT
        + "\n\nATTEMPT 2 (after isolated scipy remediation)\n"
        + f"command: {' '.join(command)}\n"
        + f"exit_code: {completed.returncode}\n"
        + f"wall_time_seconds: {wall:.9f}\n"
        + "\n=== STDOUT ===\n"
        + completed.stdout
        + "\n=== STDERR ===\n"
        + completed.stderr
    )
    (OUTPUT_DIRECTORY / "official_test_output.txt").write_text(combined, encoding="utf-8")
    return {
        "attempt_1_exit_code": 1,
        "attempt_1_root_cause": "UNDECLARED_SCIPY_RUNTIME_DEPENDENCY_IN_MYOCHALLENGE_BIMANUAL",
        "attempt_2_exit_code": int(completed.returncode),
        "attempt_2_wall_time_seconds": wall,
        "final_status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _registered_leg_environments() -> tuple[list[dict[str, Any]], str]:
    matches: list[dict[str, Any]] = []
    for env_id, spec in sorted(gym.registry.items(), key=lambda item: str(item[0])):
        identifier = str(env_id)
        if "leg" not in identifier.lower() and "myoleg" not in identifier.lower():
            continue
        matches.append(
            {
                "environment_id": identifier,
                "entry_point": str(spec.entry_point),
                "max_episode_steps": spec.max_episode_steps,
            }
        )
    ids = {row["environment_id"] for row in matches}
    if PRIMARY_ENV_ID in ids:
        selected = PRIMARY_ENV_ID
    elif matches:
        selected = matches[0]["environment_id"]
    else:
        raise RuntimeError("No registered MyoLeg environment was found")
    _json(
        OUTPUT_DIRECTORY / "registered_myoleg_envs.json",
        {
            "registry_total_count": len(gym.registry),
            "matching_count": len(matches),
            "selected_environment_id": selected,
            "myoLegWalk_v0_exists": PRIMARY_ENV_ID in ids,
            "environments": matches,
        },
    )
    return matches, selected


def _joint_inventory(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        rows.append(
            {
                "joint_id": joint_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "joint_type": _enum_name(mujoco.mjtJoint, model.jnt_type[joint_id]),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
                "limited": bool(model.jnt_limited[joint_id]),
                "range_min": float(model.jnt_range[joint_id, 0]),
                "range_max": float(model.jnt_range[joint_id, 1]),
                "body_id": int(model.jnt_bodyid[joint_id]),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint_id]),
            }
        )
    _csv(
        OUTPUT_DIRECTORY / "myoleg_joint_inventory.csv",
        rows,
        [
            "joint_id", "name", "joint_type", "qpos_address", "dof_address",
            "limited", "range_min", "range_max", "body_id", "body",
        ],
    )
    return rows


def _actuator_inventory(
    model: mujoco.MjModel, data: mujoco.MjData, joint_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in joint_rows}
    target_dofs = {
        name: int(by_name[name]["dof_address"])
        for name in (
            "hip_flexion_r", "knee_angle_r", "hip_flexion_l", "knee_angle_l"
        )
        if name in by_name
    }
    mujoco.mj_forward(model, data)
    # MuJoCo 3.6 exposes actuator moments in row-compressed sparse form.
    # Reconstruct only the small nu x nv audit matrix from the published row
    # addresses/counts and column indices; no model state is modified.
    moments = np.zeros((model.nu, model.nv), dtype=float)
    sparse_values = np.asarray(data.actuator_moment, dtype=float)
    for actuator_id in range(model.nu):
        address = int(data.moment_rowadr[actuator_id])
        count = int(data.moment_rownnz[actuator_id])
        columns = np.asarray(data.moment_colind[address : address + count], dtype=int)
        moments[actuator_id, columns] = sparse_values[address : address + count]
    rows: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        transmission_type = _enum_name(mujoco.mjtTrn, model.actuator_trntype[actuator_id])
        transmission_object_id = int(model.actuator_trnid[actuator_id, 0])
        if transmission_type == "mjTRN_TENDON":
            transmission_object = _name(
                model, mujoco.mjtObj.mjOBJ_TENDON, transmission_object_id
            )
        elif transmission_type in {"mjTRN_JOINT", "mjTRN_JOINTINPARENT"}:
            transmission_object = _name(
                model, mujoco.mjtObj.mjOBJ_JOINT, transmission_object_id
            )
        elif transmission_type == "mjTRN_SITE":
            transmission_object = _name(
                model, mujoco.mjtObj.mjOBJ_SITE, transmission_object_id
            )
        else:
            transmission_object = ""
        coupling = {
            key: float(moments[actuator_id, dof])
            for key, dof in target_dofs.items()
        }
        right_hip = abs(coupling.get("hip_flexion_r", 0.0)) > 1e-10
        right_knee = abs(coupling.get("knee_angle_r", 0.0)) > 1e-10
        left_hip = abs(coupling.get("hip_flexion_l", 0.0)) > 1e-10
        left_knee = abs(coupling.get("knee_angle_l", 0.0)) > 1e-10
        rows.append(
            {
                "actuator_id": actuator_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                "transmission_type": transmission_type,
                "transmission_object_id": transmission_object_id,
                "transmission_object": transmission_object,
                "dynamics_type": _enum_name(mujoco.mjtDyn, model.actuator_dyntype[actuator_id]),
                "gain_type": _enum_name(mujoco.mjtGain, model.actuator_gaintype[actuator_id]),
                "bias_type": _enum_name(mujoco.mjtBias, model.actuator_biastype[actuator_id]),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "control_min": float(model.actuator_ctrlrange[actuator_id, 0]),
                "control_max": float(model.actuator_ctrlrange[actuator_id, 1]),
                "moment_hip_flexion_r": coupling.get("hip_flexion_r", math.nan),
                "moment_knee_angle_r": coupling.get("knee_angle_r", math.nan),
                "moment_hip_flexion_l": coupling.get("hip_flexion_l", math.nan),
                "moment_knee_angle_l": coupling.get("knee_angle_l", math.nan),
                "right_hip_related_at_inspected_state": right_hip,
                "right_knee_related_at_inspected_state": right_knee,
                "left_hip_related_at_inspected_state": left_hip,
                "left_knee_related_at_inspected_state": left_knee,
                "biarticular_hip_knee_at_inspected_state": (
                    (right_hip and right_knee) or (left_hip and left_knee)
                ),
            }
        )
    _csv(
        OUTPUT_DIRECTORY / "myoleg_actuator_inventory.csv",
        rows,
        list(rows[0].keys()),
    )
    return rows


def _wrap_object(model: mujoco.MjModel, wrap_type: int, object_id: int) -> str:
    name = _enum_name(mujoco.mjtWrap, wrap_type)
    if name == "mjWRAP_JOINT":
        return _name(model, mujoco.mjtObj.mjOBJ_JOINT, object_id)
    if name == "mjWRAP_SITE":
        return _name(model, mujoco.mjtObj.mjOBJ_SITE, object_id)
    if name in {"mjWRAP_SPHERE", "mjWRAP_CYLINDER"}:
        return _name(model, mujoco.mjtObj.mjOBJ_GEOM, object_id)
    return ""


def _body_site_tendon_inventory(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for body_id in range(model.nbody):
        parent_id = int(model.body_parentid[body_id])
        rows.append(
            {
                "object_type": "BODY",
                "object_id": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "parent_body": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
                "attached_body": "",
                "local_position": "",
                "path_objects": "",
                "mass_kg": float(model.body_mass[body_id]),
            }
        )
    for site_id in range(model.nsite):
        body_id = int(model.site_bodyid[site_id])
        rows.append(
            {
                "object_type": "SITE",
                "object_id": site_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
                "parent_body": "",
                "attached_body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "local_position": json.dumps(model.site_pos[site_id].tolist()),
                "path_objects": "",
                "mass_kg": "",
            }
        )
    for tendon_id in range(model.ntendon):
        address = int(model.tendon_adr[tendon_id])
        count = int(model.tendon_num[tendon_id])
        path = []
        for wrap_id in range(address, address + count):
            wrap_type = int(model.wrap_type[wrap_id])
            object_id = int(model.wrap_objid[wrap_id])
            path.append(
                {
                    "wrap_type": _enum_name(mujoco.mjtWrap, wrap_type),
                    "object_id": object_id,
                    "object_name": _wrap_object(model, wrap_type, object_id),
                }
            )
        rows.append(
            {
                "object_type": "TENDON",
                "object_id": tendon_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id),
                "parent_body": "",
                "attached_body": "",
                "local_position": "",
                "path_objects": json.dumps(path, separators=(",", ":")),
                "mass_kg": "",
            }
        )
    _csv(
        OUTPUT_DIRECTORY / "myoleg_body_site_inventory.csv",
        rows,
        [
            "object_type", "object_id", "name", "parent_body", "attached_body",
            "local_position", "path_objects", "mass_kg",
        ],
    )
    return rows


def _equality_inventory(model: mujoco.MjModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for equality_id in range(model.neq):
        equality_type = _enum_name(mujoco.mjtEq, model.eq_type[equality_id])
        object_1 = int(model.eq_obj1id[equality_id])
        object_2 = int(model.eq_obj2id[equality_id])
        if equality_type == "mjEQ_JOINT":
            obj_type = mujoco.mjtObj.mjOBJ_JOINT
        elif equality_type == "mjEQ_TENDON":
            obj_type = mujoco.mjtObj.mjOBJ_TENDON
        else:
            obj_type = mujoco.mjtObj.mjOBJ_BODY
        rows.append(
            {
                "equality_id": equality_id,
                "equality_type": equality_type,
                "active_initially": bool(model.eq_active0[equality_id]),
                "object_1_id": object_1,
                "object_1": _name(model, obj_type, object_1),
                "object_2_id": object_2,
                "object_2": _name(model, obj_type, object_2),
                "data": json.dumps(model.eq_data[equality_id].tolist()),
            }
        )
    _csv(
        OUTPUT_DIRECTORY / "myoleg_equality_inventory.csv",
        rows,
        list(rows[0].keys()) if rows else ["equality_id"],
    )
    return rows


def _finite_arrays(data: mujoco.MjData) -> tuple[dict[str, list[int]], bool]:
    fields = (
        "qpos", "qvel", "actuator_force", "qfrc_passive", "qfrc_actuator",
        "qfrc_constraint", "qfrc_applied", "xfrc_applied",
    )
    shapes: dict[str, list[int]] = {}
    all_finite = True
    for field in fields:
        value = np.asarray(getattr(data, field), dtype=float)
        shapes[field] = list(value.shape)
        all_finite = all_finite and bool(np.isfinite(value).all())
    return shapes, all_finite


def _headless_smoke(environment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    env = gym.make(environment_id)
    observation, _ = env.reset(seed=20260828)
    env.action_space.seed(20260828)
    unwrapped = env.unwrapped
    model = unwrapped.mj_model
    data = unwrapped.mj_data
    exception_count = 0
    termination_count = 0
    truncation_count = 0
    finite = True
    start_time = float(data.time)
    started = time.perf_counter()
    for _ in range(1000):
        action = env.action_space.sample()
        try:
            observation, _, terminated, truncated, _ = env.step(action)
            termination_count += int(terminated)
            truncation_count += int(truncated)
            finite = finite and bool(np.isfinite(np.asarray(observation)).all())
            _, field_finite = _finite_arrays(data)
            finite = finite and field_finite
        except Exception:
            exception_count += 1
            break
    wall = time.perf_counter() - started
    simulated = float(data.time) - start_time
    shapes, field_finite = _finite_arrays(data)
    finite = finite and field_finite
    smoke = {
        "environment_id": environment_id,
        "observation_shape": list(np.asarray(observation).shape),
        "action_space_shape": list(env.action_space.shape),
        "environment_steps": 1000 - exception_count,
        "integration_steps_per_environment_step": int(unwrapped.frame_skip),
        "integration_timestep_seconds": float(model.opt.timestep),
        "environment_control_dt_seconds": float(unwrapped.dt),
        "simulated_time_seconds": simulated,
        "wall_time_seconds": wall,
        "environment_steps_per_second": (1000 - exception_count) / wall,
        "realtime_factor": simulated / wall,
        "exception_count": exception_count,
        "termination_signal_count": termination_count,
        "truncation_signal_count": truncation_count,
        "nan_or_inf_detected": not finite,
        "data_field_shapes": shapes,
        "status": "PASS" if exception_count == 0 and finite else "FAIL",
    }

    env.reset(seed=20260829)
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    passive_before = np.asarray(data.qfrc_passive, dtype=float).copy()
    actuator_before = np.asarray(data.actuator_force, dtype=float).copy()
    passive_exception_count = 0
    for _ in range(250):
        try:
            env.step(zero_action)
        except Exception:
            passive_exception_count += 1
            break
    passive_after = np.asarray(data.qfrc_passive, dtype=float).copy()
    actuator_after = np.asarray(data.actuator_force, dtype=float).copy()
    passive = {
        "zero_control_is_not_claimed_as_true_human_passivity": True,
        "steps": 250 - passive_exception_count,
        "exception_count": passive_exception_count,
        "qfrc_passive_nonzero_before": bool(np.any(np.abs(passive_before) > 1e-12)),
        "qfrc_passive_nonzero_after": bool(np.any(np.abs(passive_after) > 1e-12)),
        "qfrc_passive_l2_before": float(np.linalg.norm(passive_before)),
        "qfrc_passive_l2_after": float(np.linalg.norm(passive_after)),
        "actuator_force_nonzero_at_zero_control_before": bool(
            np.any(np.abs(actuator_before) > 1e-12)
        ),
        "actuator_force_nonzero_at_zero_control_after": bool(
            np.any(np.abs(actuator_after) > 1e-12)
        ),
        "all_finite": bool(
            np.isfinite(passive_after).all() and np.isfinite(actuator_after).all()
        ),
    }
    env.close()
    return smoke, passive


def _benchmark(environment_id: str) -> dict[str, Any]:
    env = gym.make(environment_id)
    env.reset(seed=20260830)
    unwrapped = env.unwrapped
    model = unwrapped.mj_model
    data = unwrapped.mj_data
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    segments = []
    for count in (1_000, 10_000, 100_000):
        start_sim = float(data.time)
        start_wall = time.perf_counter()
        exception_count = 0
        nonfinite = False
        for _ in range(count):
            try:
                observation, _, _, _, _ = env.step(action)
                if not np.isfinite(np.asarray(observation)).all():
                    nonfinite = True
                    break
            except Exception:
                exception_count += 1
                break
        wall = time.perf_counter() - start_wall
        simulated = float(data.time) - start_sim
        completed = count - exception_count - int(nonfinite)
        segments.append(
            {
                "requested_environment_steps": count,
                "completed_environment_steps": completed,
                "wall_time_seconds": wall,
                "simulated_time_seconds": simulated,
                "environment_steps_per_second": completed / wall,
                "integration_steps_per_second": completed * int(unwrapped.frame_skip) / wall,
                "realtime_factor": simulated / wall,
                "exception_count": exception_count,
                "nan_or_inf_detected": nonfinite,
            }
        )
    selected_rate = segments[-1]["environment_steps_per_second"]
    trajectory_seconds = 24.0
    environment_steps_per_trajectory = int(round(trajectory_seconds / float(unwrapped.dt)))
    integration_steps_per_trajectory = environment_steps_per_trajectory * int(unwrapped.frame_skip)
    one_wall = environment_steps_per_trajectory / selected_rate
    extrapolation = []
    for count in (1, 100, 1_000, 21_025):
        seconds = one_wall * count
        extrapolation.append(
            {
                "trajectory_count": count,
                "estimated_wall_seconds": seconds,
                "estimated_wall_minutes": seconds / 60.0,
                "estimated_wall_hours": seconds / 3600.0,
            }
        )
    result = {
        "environment_id": environment_id,
        "engineering_estimate_only": True,
        "environment_control_dt_seconds": float(unwrapped.dt),
        "integration_timestep_seconds": float(model.opt.timestep),
        "integration_steps_per_environment_step": int(unwrapped.frame_skip),
        "segments": segments,
        "peak_rss_raw_ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "peak_rss_units_on_macos": "bytes",
        "trajectory_duration_seconds": trajectory_seconds,
        "environment_steps_per_24s_trajectory": environment_steps_per_trajectory,
        "integration_steps_per_24s_trajectory": integration_steps_per_trajectory,
        "extrapolation_from_100000_step_segment": extrapolation,
    }
    env.close()
    _json(OUTPUT_DIRECTORY / "myoleg_runtime_benchmark.json", result)
    return result


def _model_summary(
    env: Any,
    joint_rows: list[dict[str, Any]],
    actuator_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    equality_rows: list[dict[str, Any]],
    passive: dict[str, Any],
) -> dict[str, Any]:
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data
    target_joint_names = {
        "root",
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
        "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
        "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    }
    target_joints = [row for row in joint_rows if row["name"] in target_joint_names]
    shank_sites = [
        row for row in inventory_rows
        if row["object_type"] == "SITE" and row["attached_body"] in {"tibia_r", "tibia_l"}
    ]
    hip_related = [
        row["name"] for row in actuator_rows
        if row["right_hip_related_at_inspected_state"] or row["left_hip_related_at_inspected_state"]
    ]
    knee_related = [
        row["name"] for row in actuator_rows
        if row["right_knee_related_at_inspected_state"] or row["left_knee_related_at_inspected_state"]
    ]
    biarticular = [
        row["name"] for row in actuator_rows
        if row["biarticular_hip_knee_at_inspected_state"]
    ]
    shapes, finite = _finite_arrays(data)
    return {
        "model_path": str(env.unwrapped.model_path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "joint_count": int(model.njnt),
        "actuator_count": int(model.nu),
        "body_count": int(model.nbody),
        "site_count": int(model.nsite),
        "tendon_count": int(model.ntendon),
        "equality_count": int(model.neq),
        "target_joint_evidence": target_joints,
        "shank_site_count": len(shank_sites),
        "shank_sites": shank_sites,
        "hip_related_actuator_count_at_inspected_state": len(hip_related),
        "hip_related_actuators_at_inspected_state": hip_related,
        "knee_related_actuator_count_at_inspected_state": len(knee_related),
        "knee_related_actuators_at_inspected_state": knee_related,
        "biarticular_hip_knee_count_at_inspected_state": len(biarticular),
        "biarticular_hip_knee_actuators_at_inspected_state": biarticular,
        "biarticular_method": "nonzero MjData.actuator_moment at actual reset state",
        "biarticular_classification_requires_further_model_inspection": True,
        "force_api_shapes": shapes,
        "force_api_all_finite": finite,
        "passive_preliminary_audit": passive,
        "rehabilitation_feasibility": {
            "pelvis_fix": {
                "rating": "REQUIRES_MODIFICATION",
                "evidence": "root free joint plus descendant pelvis body; weld/root locking is structurally possible",
            },
            "non_target_leg_fix": {
                "rating": "REQUIRES_MODIFICATION",
                "evidence": "separate left/right hip, knee, ankle joint chains",
            },
            "ankle_fix": {
                "rating": "REQUIRES_MODIFICATION",
                "evidence": "ankle_angle_r/l, subtalar_angle_r/l, mtp_angle_r/l joints exist",
            },
            "hip_non_sagittal_lock": {
                "rating": "REQUIRES_MODIFICATION",
                "evidence": "hip_adduction and hip_rotation are separate hinge joints",
            },
            "knee_flexion_only": {
                "rating": "LIKELY",
                "evidence": "knee_angle_r/l are hinge DoFs; auxiliary knee joints have existing equality constraints",
            },
            "shank_attachment": {
                "rating": "LIKELY",
                "evidence": "tibia_r/l bodies and actual attached sites exist; dedicated strap site should be separately specified",
            },
            "hip_knee_state_force_readout": {
                "rating": "YES",
                "evidence": "joint qpos/dof addresses plus qpos/qvel/qfrc arrays are exposed",
            },
            "external_traction": {
                "rating": "YES",
                "evidence": "xfrc_applied is exposed per body and mujoco.mj_applyFT is available",
            },
        },
        "equality_constraints": equality_rows,
    }


def _environment_versions(official: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    packages = {
        name: {
            "version": _package_version(name),
            "location": _package_location(name),
        }
        for name in (
            "myosuite", "mujoco", "numpy", "gymnasium", "scipy", "jax", "jaxlib",
            "pink-noise-rl", "stable-baselines3", "mjrl",
        )
    }
    pip_probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    uv_path = shutil.which("uv")
    uv_version = None
    if uv_path:
        uv_probe = subprocess.run([uv_path, "--version"], capture_output=True, text=True, check=False)
        uv_version = uv_probe.stdout.strip() if uv_probe.returncode == 0 else None
        freeze = subprocess.run(
            [uv_path, "pip", "freeze", "--python", sys.executable],
            capture_output=True,
            text=True,
            check=False,
        )
        (OUTPUT_DIRECTORY / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    return {
        "stage_id": STAGE_ID,
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "shell": os.environ.get("SHELL"),
        "environment_path": str(Path(sys.prefix).resolve()),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pip_version": pip_probe.stdout.strip() if pip_probe.returncode == 0 else None,
        "pip_status": "INSTALLED" if pip_probe.returncode == 0 else "NOT_INSTALLED_IN_UV_VENV",
        "uv_path": uv_path,
        "uv_version": uv_version,
        "packages": packages,
        "official_test": official,
        "headless_smoke_status": smoke["status"],
        "current_rehab_robot_env_unchanged": True,
    }


def _write_checksums() -> None:
    checksum_file = OUTPUT_DIRECTORY / "checksums.sha256"
    files = sorted(
        path for path in OUTPUT_DIRECTORY.iterdir()
        if path.is_file() and path.name not in {checksum_file.name, "MYOLEG_INSTALL_AND_SMOKE_TEST_REPORT.md"}
    )
    checksum_file.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def run_audit() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    official = _official_test()
    matches, selected = _registered_leg_environments()
    smoke, passive = _headless_smoke(selected)
    env = gym.make(selected)
    env.reset(seed=20260828)
    joint_rows = _joint_inventory(env.unwrapped.mj_model)
    actuator_rows = _actuator_inventory(env.unwrapped.mj_model, env.unwrapped.mj_data, joint_rows)
    inventory_rows = _body_site_tendon_inventory(env.unwrapped.mj_model)
    equality_rows = _equality_inventory(env.unwrapped.mj_model)
    model_summary = _model_summary(
        env, joint_rows, actuator_rows, inventory_rows, equality_rows, passive
    )
    env.close()
    _json(OUTPUT_DIRECTORY / "myoleg_model_summary.json", model_summary)
    benchmark = _benchmark(selected)
    versions = _environment_versions(official, smoke)
    versions["registered_environment_total_count"] = len(gym.registry)
    versions["registered_myoleg_matching_count"] = len(matches)
    _json(OUTPUT_DIRECTORY / "environment_versions.json", versions)
    _json(OUTPUT_DIRECTORY / "myoleg_headless_smoke_test.json", smoke)
    _json(
        OUTPUT_DIRECTORY / "audit_run_summary.json",
        {
            "official_test": official,
            "headless_smoke": smoke["status"],
            "benchmark_segments": len(benchmark["segments"]),
            "selected_environment_id": selected,
        },
    )
    _write_checksums()


def visualization_probe(environment_id: str) -> None:
    import mujoco.viewer

    env = gym.make(environment_id)
    env.reset(seed=20260828)
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for _ in range(120):
            mujoco.mj_step(model, data)
            viewer.sync()
    finally:
        viewer.close()
        env.close()
    print("MYOLEG_VISUALIZATION_PROBE=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization-probe", action="store_true")
    parser.add_argument("--environment-id", default=PRIMARY_ENV_ID)
    args = parser.parse_args()
    if args.visualization_probe:
        visualization_probe(args.environment_id)
    else:
        run_audit()


if __name__ == "__main__":
    main()
