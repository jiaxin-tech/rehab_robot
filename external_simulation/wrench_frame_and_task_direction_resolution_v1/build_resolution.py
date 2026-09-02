"""Freeze wrench and task-direction semantics without hardware access.

The builder reads local source, configuration, SDK declarations and frozen
artifacts.  Its only executable calculation is deterministic offline rotation
math using the existing pure helpers in :mod:`collection.state`.  It never
constructs or connects a robot, enables power, sends motion, or evaluates an
episode endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

from collection.state import (
    rpy_euler_xyz_rotation_matrix,
    rotate_vector,
    transform_wrench,
    transpose_rotation,
)


STAGE_ID = "WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_V1"
PROTOCOL_ID = "WRENCH_RESOLUTION_PROTOCOL"
FORMAL_STATUS = "WRENCH_AND_TASK_DIRECTION_PARTIALLY_RESOLVED_REQUIRES_STATIC_VALIDATION"
PARENT_STATUS = "PRIMARY_MECHANICAL_ENDPOINT_DEFINITION_INCOMPLETE"
REQUESTED_WRENCH_FRAME = "world"
VERIFIED_WRENCH_FRAME = "NONE_PHYSICALLY_VERIFIED"
WRENCH_FRAME_STATUS = "PARTIALLY_VERIFIED_API_WORLD_EXPRESSION_PHYSICAL_SEMANTICS_PENDING"
WRENCH_SIGN_STATUS = "WRENCH_FORCE_SIGN_NOT_VERIFIED"
WRENCH_REFERENCE_POINT_STATUS = "WRENCH_REFERENCE_POINT_PARTIALLY_DOCUMENTED_NOT_PHYSICALLY_VERIFIED"
WRENCH_COMPENSATION_STATUS = "WRENCH_COMPENSATION_NOT_DOCUMENTED"
WRENCH_TIMESTAMP_STATUS = "HOST_QUERY_BOUNDS_VERIFIED_DEVICE_SOURCE_TIMESTAMP_UNAVAILABLE"
TASK_DIRECTION_STATUS = "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
TASK_DIRECTION_TARGET = "ACTUAL_STRAP_PULL_LINE_OF_ACTION"
NEXT_STAGE = "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1"
SDK_VERSION = "0.7.0"
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
PARENT_PROTOCOL_SHA256 = "de27c80d3ca93cd299c016ccb5d80032a8af417a2d06b91e2a01e5f0b2680f9e"
PARENT_ENDPOINT_SHA256 = "0957f15738d44360d941214d192de6a42d9241aee99ffe36281019fa82e90422"
FROZEN_PROTOCOL_SHA256 = "e23c99636816b771a4233d35fe13cdeee46e57de84f5972aa764de5db775a52b"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1"
PROTOCOL_PATH = OUTPUT / "WRENCH_RESOLUTION_PROTOCOL.json"


INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "PARENT_PROTOCOL",
        "path": "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json",
        "markers": (PARENT_STATUS, "WRENCH_FRAME_SEMANTICS_NOT_VERIFIED", "TASK_DIRECTION_REQUIRES_EXPERIMENTAL_VALIDATION"),
        "exact_sha256": PARENT_PROTOCOL_SHA256,
    },
    {
        "id": "PARENT_ENDPOINT",
        "path": "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/PRIMARY_MECHANICAL_ENDPOINT_DEFINITION.json",
        "markers": (PARENT_STATUS, "EPISODE_RMS_VALIDATED_TASK_DIRECTION_INTERACTION_FORCE", '"validated": false'),
        "exact_sha256": PARENT_ENDPOINT_SHA256,
    },
    {
        "id": "PARENT_WRENCH_AUDIT",
        "path": "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/WRENCH_SEMANTICS_AUDIT.md",
        "markers": ("WRENCH_FRAME_SEMANTICS_NOT_VERIFIED", "world`, `flange`, and `tool`", "Rotation-only is not a point transform"),
    },
    {
        "id": "PARENT_TASK_DIRECTION_AUDIT",
        "path": "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/TASK_DIRECTION_DEFINITION_AUDIT.md",
        "markers": ("TASK_DIRECTION_REQUIRES_EXPERIMENTAL_VALIDATION", "strap/pull line of action", "never from whichever direction produces the smallest RMS"),
    },
    {
        "id": "FORMAL_EXPERIMENT_MANIFEST",
        "path": "config/formal_experiment_manifest.json",
        "markers": ("ROM_PROTOCOL_V2", "q_hip - q_knee", ACTIVE_REFERENCE_SHA256),
    },
    {
        "id": "ACTIVE_REFERENCE",
        "path": "reference_release/reference_measured_asymmetric_closed_slow.csv",
        "markers": (),
        "exact_sha256": ACTIVE_REFERENCE_SHA256,
    },
    {
        "id": "LOCAL_XCORE_SDK_STUB",
        "path": "hardware/windows/xcoresdk/xCoreSDK_python/__init__.pyi",
        "markers": ("def getEndTorque", "FrameType::world", "FrameType::flange", "FrameType::tool", "欧拉角XYZ", "行优先齐次变换矩阵"),
    },
    {
        "id": "XCORE_WRAPPER",
        "path": "hardware/windows/rokae_xcore.py",
        "markers": ('_EXPECTED_SDK_VERSION = "0.7.0"', "get_world_to_base_rotation", "transpose_rotation(world_from_base)", "getEndTorque"),
    },
    {
        "id": "OBSERVATION_ADAPTER",
        "path": "hardware/rokae_adapter.py",
        "markers": ("Observation-only project adapter", 'read_internal_wrench(self, reference_frame: str = "world")', "does not calibrate, bias, compensate"),
    },
    {
        "id": "INTERNAL_WRENCH_SOURCE",
        "path": "hardware/windows/rokae_internal_wrench.py",
        "markers": ("rotation_only_pending_robot_validation", "get_world_to_base_rotation", "rotate_vector", "software reference bias"),
    },
    {
        "id": "WRENCH_MATH",
        "path": "collection/state.py",
        "markers": ("rpy_euler_xyz_rotation_matrix", "Rz(yaw) @ Ry(pitch) @ Rx(roll)", "Full wrench transform", "p×(R F_a)"),
    },
    {
        "id": "SNAPSHOT_FAIL_CLOSED",
        "path": "collection/snapshot.py",
        "markers": ("base_wrench_rotation_requires_robot_validation", "base_wrench_unavailable"),
    },
    {
        "id": "PROJECT_SETTINGS",
        "path": "config/settings.py",
        "markers": ('ROBOT_FORCE_RAW_FRAME          = "world"', 'BASE_WRENCH_TRANSFORM_KIND     = "rotation_only"', "BASE_WRENCH_ROTATION_VERIFIED  = False", "TOOL_NAME = None"),
    },
    {
        "id": "SAFETY_CONFIG",
        "path": "config/experiment_safety.json",
        "markers": ('"reviewed_tool_name": null', '"tool_workpiece_reviewed": false', '"reviewed": false'),
    },
    {
        "id": "IDENTIFICATION_CONFIG",
        "path": "config/real_identification_config.json",
        "markers": ('"raw_wrench_frame": null', '"R_rehab_from_raw_wrench": null', '"force_sign_robot_on_leg": null', '"reviewed": false'),
    },
    {
        "id": "REHAB_FRAME_CONFIG",
        "path": "config/rehab_frame_config.json",
        "markers": ('"rehab_x_axis_in_base": null', '"rehab_z_axis_in_base": null', '"reviewed": false'),
    },
    {
        "id": "START_ANCHOR_SCHEMA",
        "path": "control/start_anchor.py",
        "markers": ("tcp_pose_base", "tool_name", "workpiece_name", "reviewed: bool = False"),
    },
    {
        "id": "RELATIVE_TRAJECTORY_GEOMETRY",
        "path": "control/start_anchored_relative_trajectory.py",
        "markers": ("equivalent strap pull point", "L2_knee_to_equivalent_shank_strap_pull_point_m", "hip_center_required", "observed_ankle_used_as_pull_point"),
    },
    {
        "id": "LIMB_GEOMETRY_CONFIG",
        "path": "lower_limb_sim/config.py",
        "markers": ("L1 = 0.42", "L2 = 0.30", "L2 仍只表示膝关节到束缚带等效牵引点"),
    },
    {
        "id": "LIMB_FORWARD_KINEMATICS",
        "path": "lower_limb_sim/kinematics.py",
        "markers": ("x_pull", "z_pull", "shank_angle = q_hip_array - q_knee_array"),
    },
    {
        "id": "EXISTING_ROTATION_TESTS",
        "path": "tests/test_units_and_frames.py",
        "markers": ("test_wrench_rotation_and_full_point_transform", "transform_wrench", "rotation_only_pending_robot_validation"),
    },
    {
        "id": "FRAME_DIAGNOSTIC",
        "path": "scripts/check_wrench_frame_rotation.py",
        "markers": ("never sends a motion command", "expected_axis_positive", "does not set BASE_WRENCH_ROTATION_VERIFIED"),
    },
    {
        "id": "POSE_BIAS_DIAGNOSTIC",
        "path": "scripts/check_wrench_pose_dependence.py",
        "markers": ("no motion is issued", "does not determine whether xCoreSDK performs gravity compensation"),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("CSV rows must not be empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def verify_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INPUT_SPECS:
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing input: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"semantic marker mismatch {spec['id']}: {missing}")
        digest = sha256_file(path)
        expected = spec.get("exact_sha256")
        if expected is not None and digest != expected:
            raise RuntimeError(f"exact SHA mismatch {spec['id']}: {digest}")
        rows.append({
            "input_id": spec["id"], "path": spec["path"], "sha256": digest,
            "semantic_markers": list(spec["markers"]), "semantic_markers_pass": True,
        })
    parent = read_json(ROOT / INPUT_SPECS[1]["path"])
    formal = read_json(ROOT / "config/formal_experiment_manifest.json")
    safety = read_json(ROOT / "config/experiment_safety.json")
    rehab = read_json(ROOT / "config/rehab_frame_config.json")
    if not (parent["decision_state"] == PARENT_STATUS and parent["validated"] is False):
        raise RuntimeError("parent endpoint state changed")
    if not (formal["rom_protocol_version"] == "ROM_PROTOCOL_V2" and formal["theta_shank_definition"] == "q_hip - q_knee" and formal["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256):
        raise RuntimeError("formal ROM/reference/angle convention changed")
    if safety["reviewed"] is not False or rehab["reviewed"] is not False:
        raise RuntimeError("unexpected safety or rehab-frame approval")
    return rows


def protocol_payload(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_endpoint_sha256": PARENT_ENDPOINT_SHA256,
        "parent_endpoint_status": PARENT_STATUS,
        "allowed_readiness_decisions": [
            "WRENCH_AND_TASK_DIRECTION_RESOLVED_FOR_VALIDATION",
            FORMAL_STATUS,
            "WRENCH_AND_TASK_DIRECTION_NOT_RESOLVED",
        ],
        "semantic_status_vocabulary": ["VERIFIED", "PARTIALLY_VERIFIED", "NOT_DOCUMENTED", "REQUIRES_PHYSICAL_VALIDATION"],
        "decision_rules": {
            "api_request_frame_is_not_physical_verification": True,
            "rotation_math_pass_is_not_physical_frame_verification": True,
            "task_direction_selected_by_mechanical_meaning_not_outcome": True,
            "unmeasured_strap_line_cannot_be_labelled_known": True,
            "primary_endpoint_may_not_be_computed_or_validated": True,
        },
        "offline_rotation_cases": ["identity", "+90deg_x", "+90deg_y", "+90deg_z", "inverse_world_to_base", "full_moment_shift"],
        "input_files": inputs,
        "forbidden_operations": [
            "construct_or_connect_robot", "power_or_enable_robot", "send_motion",
            "execute_rehabilitation_trajectory", "collect_human_data", "apply_human_load",
            "calibrate_or_change_controller", "modify_hardware_control_or_safety",
            "compute_final_J_force", "compare_trajectories", "run_repeatability_or_sensitivity",
            "run_PINN", "run_BO", "modify_V3_or_MyoLeg", "execute_next_stage",
        ],
        "protocol_frozen_before_resolution_results": True,
    }


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"prepare requires empty output directory: {OUTPUT}")
    inputs = verify_inputs()
    atomic_json(PROTOCOL_PATH, protocol_payload(inputs))
    atomic_json(OUTPUT / "INPUT_VERIFICATION.json", {
        "stage_id": STAGE_ID, "input_count": len(inputs),
        "all_inputs_present_and_semantically_verified": True, "inputs": inputs,
        "new_physical_evidence_files_used": [], "robot_access_count": 0,
        "human_data_access_count": 0, "endpoint_evaluation_count": 0,
    })
    atomic_json(OUTPUT / "HARDWARE_ACCESS_AUDIT.json", {
        "offline_source_and_math_audit_only": True,
        "robot_adapter_constructed": False, "robot_connected": False,
        "power_or_enable_count": 0, "motion_command_count": 0,
        "calibration_call_count": 0, "human_loading_count": 0,
        "read_only_diagnostic_scripts_inspected_not_executed": [
            "scripts/check_wrench_frame_rotation.py", "scripts/check_wrench_pose_dependence.py"
        ],
    })
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "input_count": len(inputs)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen")
    if sha256_file(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("resolution protocol SHA mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["input_files"] != verify_inputs():
        raise RuntimeError("frozen evidence inputs changed")
    return protocol


def _fmt_vector(values: tuple[float, float, float] | list[float]) -> str:
    cleaned = [0.0 if abs(float(value)) < 5e-15 else float(value) for value in values]
    return json.dumps(cleaned, separators=(",", ":"))


def rotation_rows() -> list[dict[str, Any]]:
    cases = [
        ("IDENTITY_X", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ("RX90_Y_TO_Z", (math.pi / 2, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ("RX90_Z_TO_NEG_Y", (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ("RY90_Z_TO_X", (0.0, math.pi / 2, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ("RY90_X_TO_NEG_Z", (0.0, math.pi / 2, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ("RZ90_X_TO_Y", (0.0, 0.0, math.pi / 2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("RZ90_Y_TO_NEG_X", (0.0, 0.0, math.pi / 2), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    ]
    rows: list[dict[str, Any]] = []
    for test_id, rpy, vector, expected in cases:
        rotation = rpy_euler_xyz_rotation_matrix(rpy)
        actual = rotate_vector(rotation, vector)
        if actual is None:
            raise RuntimeError(f"rotation returned None: {test_id}")
        error = max(abs(actual[index] - expected[index]) for index in range(3))
        rows.append({
            "test_id": test_id, "test_type": "CANONICAL_ACTIVE_XYZ_ROTATION",
            "rpy_rad": _fmt_vector(rpy), "input_vector": _fmt_vector(vector),
            "expected_vector": _fmt_vector(expected), "actual_vector": _fmt_vector(actual),
            "max_abs_error": f"{error:.17g}", "tolerance": "1e-12", "passed": error <= 1e-12,
            "physical_frame_verified": False,
        })

    world_from_base = rpy_euler_xyz_rotation_matrix((0.0, 0.0, math.pi / 2))
    base_from_world = transpose_rotation(world_from_base)
    actual_inverse = rotate_vector(base_from_world, (0.0, 1.0, 0.0))
    expected_inverse = (1.0, 0.0, 0.0)
    assert actual_inverse is not None
    error = max(abs(actual_inverse[index] - expected_inverse[index]) for index in range(3))
    rows.append({
        "test_id": "INVERSE_WORLD_TO_BASE", "test_type": "TRANSPOSE_INVERSE_EXPRESSION_ROTATION",
        "rpy_rad": _fmt_vector((0.0, 0.0, math.pi / 2)), "input_vector": _fmt_vector((0.0, 1.0, 0.0)),
        "expected_vector": _fmt_vector(expected_inverse), "actual_vector": _fmt_vector(actual_inverse),
        "max_abs_error": f"{error:.17g}", "tolerance": "1e-12", "passed": error <= 1e-12,
        "physical_frame_verified": False,
    })

    force, moment = transform_wrench((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), rpy_euler_xyz_rotation_matrix((0.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
    expected_moment = (0.0, 0.0, -1.0)
    error = max(abs(moment[index] - expected_moment[index]) for index in range(3))
    rows.append({
        "test_id": "FULL_MOMENT_REFERENCE_SHIFT", "test_type": "FULL_WRENCH_TRANSLATION_TERM",
        "rpy_rad": _fmt_vector((0.0, 0.0, 0.0)), "input_vector": "F=[1,0,0];M=[0,0,0];p=[0,1,0]",
        "expected_vector": "F=[1,0,0];M=[0,0,-1]", "actual_vector": f"F={_fmt_vector(force)};M={_fmt_vector(moment)}",
        "max_abs_error": f"{error:.17g}", "tolerance": "1e-12", "passed": error <= 1e-12,
        "physical_frame_verified": False,
    })
    return rows


def sdk_semantics_audit() -> str:
    return f"""# ROKAE `getEndTorque` Semantics Audit

## SDK identity and exact call

- Repository wrapper expectation: xCoreSDK `{SDK_VERSION}` (`VERIFIED` from source); it rejects a different runtime version.
- Actual Windows native library loaded in this macOS audit: not loaded (`REQUIRES_PHYSICAL_VALIDATION`).
- Exact local declaration: `getEndTorque(ref_type, joint_torque_measured, external_torque_measured, cart_torque, cart_force, ec)` (`VERIFIED`).
- Project call uses that order and validates at least 6/6/3/3 finite values (`VERIFIED`).

| Semantic item | Local evidence | Status |
|---|---|---|
| joint measured torque | 6 axes; force-sensor-measured joint torque; N*m | VERIFIED |
| external joint torque | 6 axes; controller model plus measurement derived; N*m | VERIFIED |
| Cartesian torque | XYZ; N*m | VERIFIED for shape/unit; physical semantics partial |
| Cartesian force | XYZ; N | VERIFIED for shape/unit; physical semantics partial |
| accepted API request frames | getEndTorque documentation lists world/flange/tool | VERIFIED |
| current request argument | settings/default/call request `world` | VERIFIED |
| returned force/moment coordinate expression | documentation says relative to requested world/flange/tool | PARTIALLY_VERIFIED; axes are documented, physical experiment absent |
| force reference point | no unambiguous statement | NOT_DOCUMENTED |
| moment reference point | tool text mentions TCP; world/flange wording does not unambiguously fix all origins | PARTIALLY_VERIFIED |
| force sign | robot-on-environment versus environment-on-robot is not stated | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |
| compensation | gravity/tool/load/bias/friction compensation for Cartesian output is not stated | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |
| measured versus estimated Cartesian wrench | joint channels are distinguished; Cartesian channel origin is not explicitly classified | PARTIALLY_VERIFIED |
| update/query semantics | one synchronous API query fills arrays; project records host bounds | PARTIALLY_VERIFIED; controller source-update cadence unknown |
| source/device timestamp | not returned by signature/arrays | NOT_DOCUMENTED; current value is host midpoint only |
| RT synchronization | no SDK synchronization contract with `tcpPoseAbc_m` | NOT_DOCUMENTED / REQUIRES_PHYSICAL_VALIDATION |

## Requested versus verified

`REQUESTED_WRENCH_FRAME = {REQUESTED_WRENCH_FRAME}`

`VERIFIED_WRENCH_FRAME = {VERIFIED_WRENCH_FRAME}`

The request and documented expression label are real software facts; they do not prove the physical load sign, compensation state, reference point, controller tool/TCP configuration or physical world/base registration. Therefore `WORLD_WRENCH_VERIFIED` is false.
"""


def frame_chain_audit() -> str:
    return """# Robot Frame Chain Audit

## Current chain

```text
world --baseFrame()--> robot base --RT tcpPoseAbc_m / FK--> flange or configured TCP
flange --toolset.end--> tool/TCP
tool/TCP --UNKNOWN physical attachment transform--> robot-side strap attachment
limb equivalent pull point / physical cuff --UNKNOWN setup measurement--> limb-side strap attachment
rehab bed frame --rehab_frame_config--> robot base
```

| Transform/entity | Source | Evidence class | Current status |
|---|---|---|---|
| `^world T_base` | future read of xCoreSDK `baseFrame()` | CONFIGURED_GEOMETRY | API and units documented; current value not captured; physical convention unvalidated |
| `R_base_from_world` | transpose of `R_world_from_base` built from SDK XYZ Euler | CONFIGURED_GEOMETRY + math | internal math verified; physical orientation pending |
| base-to-flange/TCP pose | RT `tcpPoseAbc_m`, project treats it as base TCP | CONFIGURED_GEOMETRY | source path exists; active tool semantics require runtime validation |
| flange-to-tool/TCP | future read `toolset.end.trans/rpy` | CONFIGURED_GEOMETRY | query path exists, no frozen value |
| active HMI tool/workobject | controller/HMI state | MEASURED/CONFIGURED_GEOMETRY | explicitly unverified; available-name list is not active selection proof |
| TCP-to-robot strap eye/attachment | no repository measurement | ASSUMED_GEOMETRY | unknown |
| limb/cuff strap load-transfer point | no repository measurement | ASSUMED_GEOMETRY | unknown |
| rehab bed x/z axes in base | `config/rehab_frame_config.json` | CONFIGURED_GEOMETRY | both null, reviewed=false |
| start TCP anchor | per-session StartAnchor schema | MEASURED_GEOMETRY candidate | no active captured anchor file in repository |
| `L1=0.42 m` hip-to-knee | formal lower-limb configuration | CONFIGURED_GEOMETRY | model geometry, not a current patient measurement |
| `L2=0.30 m` knee-to-equivalent strap pull point | formal lower-limb configuration | CONFIGURED_GEOMETRY | equivalent traction point; not observed ankle and not proven physical strap attachment |
| hip origin in robot base | absent in start-anchored mode | ASSUMED/UNAVAILABLE | not measured; required for endpoint-to-hip task direction |

No active tool name, TCP offset, flange-to-tool transform, world/base pose, bed axes or physical strap attachment transform is frozen in the current unreviewed configs. This audit does not modify them.
"""


def rotation_math_audit(rows: list[dict[str, Any]]) -> str:
    passed = sum(bool(row["passed"]) for row in rows)
    return f"""# Wrench Rotation Math Audit

## Conventions in current code

- Vectors are 3-element column-vector semantics; matrix multiplication is `result[row] = sum(R[row,column]*v[column])`.
- `rpy_euler_xyz_rotation_matrix` uses `Rz(yaw) @ Ry(pitch) @ Rx(roll)` as an active XYZ-Euler rotation.
- SDK `baseFrame()` is interpreted as `^world T_base`, hence the source code builds `R_world_from_base` and transposes it to `R_base_from_world`.
- Corrected world force is expressed in base as `F_base = R_base_from_world F_world`.
- The same pure rotation is currently applied to torque but deliberately labelled rotation-only pending validation. A full point change requires `M_B = R M_A + r x (R F_A)` using a verified displacement and convention.

## Offline results

`{passed}/{len(rows)}` deterministic canonical tests passed at tolerance `1e-12`: identity, +90 degree rotations about x/y/z, transpose inverse, and a known reference-point moment shift.

These results verify implementation algebra under the declared convention. Every row deliberately records `physical_frame_verified=false`. The SDK phrase "Euler XYZ" and the wrapper's direction are consistent with the implemented convention, but only static known-direction evidence on the exact robot/tool/base setup can verify the physical mapping. Therefore `BASE_WRENCH_ROTATION_VERIFIED` remains false.
"""


def sign_reference_audit() -> str:
    return f"""# Wrench Sign and Reference-Point Audit

## Formal statuses

- `WRENCH_SIGN_STATUS = {WRENCH_SIGN_STATUS}`
- `WRENCH_REFERENCE_POINT_STATUS = {WRENCH_REFERENCE_POINT_STATUS}`
- `WRENCH_COMPENSATION_STATUS = {WRENCH_COMPENSATION_STATUS}`
- `WRENCH_TIMESTAMP_STATUS = {WRENCH_TIMESTAMP_STATUS}`

The Cartesian positive-force meaning is not stated as robot-on-environment or environment-on-robot. It must not be inferred from `external_torque_measured`, function naming or a plot. A later controlled test must apply known load directions and explicit sign reversals.

Force is a free vector for coordinate-expression rotation: `F_B=R_B_from_A F_A`; translating the reference origin does not change that 3D force vector. Moment is origin-dependent. If an A-origin wrench is re-expressed at a B-origin, the verified convention must implement `M_B=R M_A+r x F_B`. Consequently a future moment endpoint or six-dimensional wrench norm cannot use rotation-only torque unless the reference point is identical or the shift is known and applied.

Software bias subtraction in the repository is a session reference offset, not proof of controller compensation. `getEndTorque` documentation does not state gravity/tool/load/friction compensation for Cartesian outputs. Host query start/end/midpoint are provenance bounds, not a controller measurement timestamp or transport-delay estimate.
"""


def strap_geometry_audit() -> str:
    return """# Strap/Pull Geometry Audit

| Geometry item | Repository meaning | Evidence class | Resolution |
|---|---|---|---|
| robot TCP | controller-configured endpoint represented by RT pose | CONFIGURED_GEOMETRY | active tool/TCP transform not frozen |
| robot-side strap attachment | physical eyelet/cuff connection on end effector | ASSUMED_GEOMETRY | no offset from TCP is measured |
| limb-side strap attachment | physical center/line of load transfer at the cuff | ASSUMED_GEOMETRY | no base-frame measurement or placement repeatability evidence |
| equivalent pull point | 2-DOF point at L2=0.30 m from knee | CONFIGURED_GEOMETRY | formal model point only; not an ankle and not automatically actual attachment |
| shank orientation | `theta_shank=q_hip-q_knee` | FROZEN_MODEL_SEMANTICS | preserved |
| bed plane | rehab x/z axes expressed in robot base | CONFIGURED_GEOMETRY | config values null and unreviewed |
| hip coordinate | origin of 2-DOF model | ASSUMED/UNAVAILABLE in robot base | start-anchored trajectory intentionally does not require it |

The path command applies the model equivalent-pull-point displacement at the captured TCP start anchor. This establishes command geometry, not the instantaneous physical strap line. Strap routing, slack/tension, attachment widths and cuff load distribution can make the real line differ from TCP tangent or endpoint-to-hip direction.

`MEASURED_GEOMETRY` currently contains only schema-capable future observations, not an approved setup record. The actual line of action remains unknown until both physical attachment points are measured in one validated frame over representative static poses.
"""


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "A_TCP_PATH_TANGENT", "physical_interpretation": "instantaneous direction of executed TCP path", "computability": "derivable from synchronized TCP/path", "required_measurements": "validated TCP pose/time and nonzero local path derivative", "posture_dependence": "yes", "actual_strap_relation": "not equivalent unless geometry proves tangent aligns with strap", "sign_stability": "flips between flexion/extension; undefined at zero speed", "noise_sensitivity": "high near turning points/differentiation", "answers_research_question": "partial motion-direction component only", "status": "SECONDARY_DIAGNOSTIC_NOT_PRIMARY"},
        {"candidate_id": "B_ACTUAL_STRAP_PULL_LINE", "physical_interpretation": "physical load-transfer line between limb and robot attachments", "computability": "requires measured/registered attachment points", "required_measurements": "p_limb_attach^B(t), p_robot_attach^B(t), placement/tautness validity", "posture_dependence": "yes", "actual_strap_relation": "direct", "sign_stability": "stable after fixed point order; fails if slack/ambiguous routing", "noise_sensitivity": "set by point metrology and attachment compliance", "answers_research_question": "yes: force along actual rehabilitation pull", "status": "PRIMARY_RECOMMENDATION_REQUIRES_GEOMETRIC_VALIDATION"},
        {"candidate_id": "C_ENDPOINT_TO_HIP", "physical_interpretation": "line between endpoint/equivalent pull point and hip center", "computability": "requires hip and endpoint registered in same frame", "required_measurements": "hip location, physical endpoint/attachment, patient registration", "posture_dependence": "yes", "actual_strap_relation": "only a proxy if straight strap follows this line", "sign_stability": "stable after fixed point order", "noise_sensitivity": "registration and soft-tissue placement sensitive", "answers_research_question": "conditional proxy", "status": "VALIDATION_CANDIDATE_NOT_PRIMARY"},
        {"candidate_id": "D_FIXED_BED_PULL_AXIS", "physical_interpretation": "force component along one reviewed bed-plane axis", "computability": "easy after bed-to-base registration", "required_measurements": "validated rehab frame axes", "posture_dependence": "no", "actual_strap_relation": "ignores changing strap line", "sign_stability": "stable", "noise_sensitivity": "low", "answers_research_question": "only if apparatus mechanically constrains line to fixed axis", "status": "SECONDARY_DIAGNOSTIC"},
        {"candidate_id": "E_EQUIVALENT_PULL_POINT_TO_MODEL_HIP", "physical_interpretation": "2-DOF L2 equivalent point to model hip line", "computability": "from q/L1/L2 plus patient-to-base registration", "required_measurements": "validated q mapping, hip registration, proof equivalent point matches attachment", "posture_dependence": "yes", "actual_strap_relation": "model-based proxy; L2 is not observed ankle", "sign_stability": "stable after fixed point order", "noise_sensitivity": "q and registration sensitive", "answers_research_question": "mechanics-model proxy only", "status": "VALIDATION_CANDIDATE_NOT_PRIMARY"},
    ]


def task_definition_payload() -> dict[str, Any]:
    return {
        "status": TASK_DIRECTION_STATUS,
        "target_definition": TASK_DIRECTION_TARGET,
        "currently_computable_as_validated_physical_direction": False,
        "coordinate_frame": "robot base B after independent frame-chain validation",
        "point_A": {"id": "p_limb_attach_B(t)", "meaning": "measured physical center of limb/cuff strap load transfer in base", "current_status": "NOT_MEASURED"},
        "point_B": {"id": "p_robot_attach_B(t)", "meaning": "measured physical robot-side strap attachment point in base, not automatically TCP origin", "current_status": "NOT_MEASURED"},
        "construction": "d_task_B(t) = normalize(p_robot_attach_B(t) - p_limb_attach_B(t))",
        "positive_geometric_direction": "limb attachment toward robot attachment (robotward pull direction)",
        "time_variation": "both points may vary with posture; recompute per valid synchronized sample or use a validated static approximation",
        "validity": ["both points expressed in same validated base frame", "finite separation above future metrology threshold", "strap taut and single line of action", "attachment placement valid", "geometry timestamp/pose alignment valid"],
        "force_projection_candidate": "F_task(t) = dot(F_interaction_B(t), d_task_B(t))",
        "positive_force_meaning": None,
        "positive_force_status": "PARTIALLY_UNRESOLVED_UNTIL_WRENCH_FORCE_SIGN_VALIDATED",
        "commanded_tcp_tangent_equivalent": False,
        "equivalence_condition": "only after geometry demonstrates alignment within a preregistered tolerance over representative poses",
        "selection_used_mechanical_outcomes": False,
    }


def geometry_validation_plan() -> str:
    return """# Future Static Geometry Validation Plan

This plan is not executed and does not authorize robot or human exposure.

## Required points and frames

Use one independently verified robot-base coordinate system and record: base/world registration; flange origin; configured controller TCP; robot-side strap eye/attachment center; limb-side attachment center on a rigid rehabilitation fixture/phantom; bed-plane origin/x/z axes; and, only for testing candidates C/E, model hip and equivalent L2 point. Do not substitute TCP for the physical eyelet or L2 for an ankle/attachment without measured evidence.

## Static pose and setup design

After separate site approval, arrange representative stationary poses spanning the intended flexion/extension geometry without human loading. At each pose, measure both strap attachment points with an independently calibrated metrology method, record tool/TCP/base/bed configuration, verify strap routing and tautness, and repeat complete removal/reinstallation to quantify placement repeatability. No rehabilitation motion is part of this protocol.

Compute the line unit vector, its pose-dependent angular change, point-location repeatability and disagreement with TCP tangent, fixed bed axis, endpoint-to-hip and model-equivalent directions. Propagate point uncertainty into task-direction angular uncertainty and projected-force uncertainty.

No numeric tolerance is invented here. Freeze tolerances prospectively from metrology accuracy and acceptable propagated endpoint uncertainty before viewing trajectory outcomes. Failure, slack, multi-contact routing or inconsistent attachment geometry leaves `TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION`.
"""


def wrench_validation_plan() -> str:
    return """# Future Static Wrench Validation Plan

This is a protocol concept only. It is not executed and does not itself approve connection, power, force application or motion.

## Preconditions

An independent site safety review must authorize a stationary, non-human fixture; exact robot/controller/tool/load identity; safe operation state; calibrated external reference instrument and loading fixture; force/moment limits; stop procedure; operator roles; and read-only query/connect side effects. No human supplies the test load.

## Controlled evidence blocks

1. Capture unloaded raw output at multiple stationary poses before and after tests to characterize offset, drift and pose dependence without calling it compensation.
2. Apply independently measured forces in positive and negative directions along at least three non-collinear/orthogonal axes. Predefine source frame, load point and sign.
3. Query world and, where independently safe and semantically supported, tool/flange expressions without changing pose; compare the reported vectors with known rotation predictions.
4. Repeat at multiple non-degenerate base/tool orientations to distinguish transpose/sign/axis errors.
5. Apply known forces at known lever arms and sign reversals to identify moment origin/shift behavior using `M_B=R M_A+r x F_B`.
6. Record query start/end/publish times and repeated identical values to estimate query duration, effective source-update cadence and state-wrench skew; do not infer device latency from host midpoint.
7. Compare raw and session software-zero outputs across poses to assess bias, hysteresis and whether controller compensation remains unexplained.

Freeze acceptance rules, force levels, repetitions and uncertainty budgets before data inspection. Required outputs include axis/sign confusion matrix, rotation residuals, cross-axis ratio, linearity/reversal residuals, reference-point moment residuals, bias/drift/pose-dependence results and timing distributions. Any ambiguous result stays fail-closed and must not set `BASE_WRENCH_ROTATION_VERIFIED=true`.
"""


def final_status_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "readiness_decision": FORMAL_STATUS,
        "WRENCH_FRAME_STATUS": WRENCH_FRAME_STATUS,
        "REQUESTED_WRENCH_FRAME": REQUESTED_WRENCH_FRAME,
        "VERIFIED_WRENCH_FRAME": VERIFIED_WRENCH_FRAME,
        "WORLD_WRENCH_VERIFIED": False,
        "WRENCH_SIGN_STATUS": WRENCH_SIGN_STATUS,
        "WRENCH_FORCE_SIGN_VERIFIED": False,
        "WRENCH_REFERENCE_POINT_STATUS": WRENCH_REFERENCE_POINT_STATUS,
        "WRENCH_COMPENSATION_STATUS": WRENCH_COMPENSATION_STATUS,
        "WRENCH_TIMESTAMP_STATUS": WRENCH_TIMESTAMP_STATUS,
        "BASE_WRENCH_ROTATION_VERIFIED": False,
        "ROTATION_MATH_INTERNALLY_VERIFIED": all(bool(row["passed"]) for row in rows),
        "ROTATION_MATH_TEST_COUNT": len(rows),
        "TASK_DIRECTION_STATUS": TASK_DIRECTION_STATUS,
        "TASK_DIRECTION_DEFINITION": TASK_DIRECTION_TARGET,
        "TASK_DIRECTION_CURRENTLY_COMPUTABLE": False,
        "TASK_DIRECTION_POSITIVE_GEOMETRIC_DIRECTION": "limb attachment toward robot attachment",
        "TASK_DIRECTION_FORCE_SIGN_MEANING": None,
        "PRIMARY_ENDPOINT_VALIDATED": False,
        "PRIMARY_ENDPOINT_FINALIZED": False,
        "PRIMARY_ENDPOINT_VALUE_COMPUTED": False,
        "NOT_HUMAN_READY": True,
        "NOT_ROBOT_APPROVED": True,
        "next_stage": NEXT_STAGE,
        "next_stage_executed": False,
    }


def report(rows: list[dict[str, Any]]) -> str:
    return f"""# Wrench Frame and Task Direction Resolution V1

## Formal decision

`{FORMAL_STATUS}`

The stage resolved software/API and mathematical semantics as far as local evidence permits. It did not create physical frame, sign, compensation, timing or strap-geometry evidence.

## Wrench result

- SDK contract in repository: xCoreSDK `{SDK_VERSION}`; exact 6/6/3/3 `getEndTorque` call and units are documented.
- `REQUESTED_WRENCH_FRAME = {REQUESTED_WRENCH_FRAME}`.
- `VERIFIED_WRENCH_FRAME = {VERIFIED_WRENCH_FRAME}`.
- World is a documented request/expression label, not a physically verified world wrench.
- Force sign is not documented; Cartesian compensation and exact force/reference-point semantics remain incomplete.
- Moment point is partially documented only; full point shifts require `M_B=R M_A+r x F_B`.
- Only host query bounds/midpoint exist; no device/source timestamp or RT synchronization contract exists.

## Rotation result

All `{len(rows)}/{len(rows)}` offline canonical math cases passed, including identity, +90 degree x/y/z, inverse transpose and moment shift. This confirms internal active-column-vector algebra under the source convention. It does not confirm the physical SDK frame convention, so `BASE_WRENCH_ROTATION_VERIFIED=false` remains unchanged.

## Frame and geometry result

`baseFrame()` and `toolset.end` are runtime-queryable but have no current frozen values. Active HMI tool/workobject is explicitly unverified. Bed axes are null/unreviewed. TCP-to-strap and limb attachment points are unmeasured. L2 remains the configured equivalent strap traction point, not an observed ankle or automatically the physical cuff attachment.

## Task direction result

`TASK_DIRECTION_DEFINITION = {TASK_DIRECTION_TARGET}` is the physical target. With `p_limb_attach_B` and `p_robot_attach_B` measured in one validated base frame:

`d_task_B(t)=normalize(p_robot_attach_B(t)-p_limb_attach_B(t))`

The positive geometric direction is limb-to-robot. Both points and their variation must be validated. TCP tangent is command/motion direction, not automatically interaction line of action. Because wrench force sign is unresolved, positive/negative `F_task` physical meaning is also unresolved.

## What was not done

No robot connection, enable, motion, human loading, endpoint value, trajectory comparison, repeatability/sensitivity, model/PINN training or BO was performed. Hardware/control/safety code and frozen results were not modified.

## Single next stage

`{NEXT_STAGE}`

It should preregister the exact non-human stationary fixture, known loads, signs, frames, lever arms, timing evidence and fail-closed acceptance rules. It was not executed.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files) + "\n")


def execute() -> None:
    protocol = verify_freeze()
    rows = rotation_rows()
    if not all(bool(row["passed"]) for row in rows):
        raise RuntimeError("offline rotation math audit failed")
    atomic_text(OUTPUT / "ROKAE_GET_END_TORQUE_SEMANTICS_AUDIT.md", sdk_semantics_audit())
    atomic_text(OUTPUT / "ROBOT_FRAME_CHAIN_AUDIT.md", frame_chain_audit())
    atomic_text(OUTPUT / "WRENCH_ROTATION_MATH_AUDIT.md", rotation_math_audit(rows))
    atomic_csv(OUTPUT / "WRENCH_ROTATION_UNIT_TEST_RESULTS.csv", rows)
    atomic_text(OUTPUT / "WRENCH_SIGN_AND_REFERENCE_POINT_AUDIT.md", sign_reference_audit())
    atomic_text(OUTPUT / "STRAP_PULL_GEOMETRY_AUDIT.md", strap_geometry_audit())
    atomic_csv(OUTPUT / "TASK_DIRECTION_CANDIDATE_COMPARISON.csv", candidate_rows())
    atomic_json(OUTPUT / "TASK_DIRECTION_FORMAL_DEFINITION.json", task_definition_payload())
    atomic_text(OUTPUT / "FUTURE_STATIC_GEOMETRY_VALIDATION_PLAN.md", geometry_validation_plan())
    atomic_text(OUTPUT / "FUTURE_STATIC_WRENCH_VALIDATION_PLAN.md", wrench_validation_plan())
    atomic_json(OUTPUT / "FINAL_WRENCH_TASK_DIRECTION_STATUS.json", final_status_payload(rows))
    atomic_text(OUTPUT / "WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_REPORT.md", report(rows))
    atomic_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID, "formal_status": FORMAL_STATUS,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "input_count": len(protocol["input_files"]), "sdk_version_expected": SDK_VERSION,
        "requested_wrench_frame": REQUESTED_WRENCH_FRAME,
        "verified_wrench_frame": VERIFIED_WRENCH_FRAME,
        "base_wrench_rotation_verified": False,
        "rotation_math_test_count": len(rows), "rotation_math_failed_count": 0,
        "primary_endpoint_validated": False, "primary_endpoint_value_computed": False,
        "robot_access_count": 0, "motion_command_count": 0, "human_data_access_count": 0,
        "pinn_training_count": 0, "bo_run_count": 0,
        "hardware_control_safety_modified": False, "frozen_artifacts_modified": False,
        "not_human_ready": True, "not_robot_approved": True,
        "next_stage": NEXT_STAGE, "next_stage_executed": False,
    })
    write_checksums()
    print(json.dumps(final_status_payload(rows), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute()


if __name__ == "__main__":
    main()
