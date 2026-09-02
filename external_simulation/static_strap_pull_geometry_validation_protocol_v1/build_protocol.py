"""Freeze a non-human strap-geometry validation protocol without executing it.

The builder reads pinned repository evidence and writes protocol-design
artifacts only.  It does not import robot/control/safety code, connect to
hardware, move a robot, apply a load, use a human, or calculate a validation
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


STAGE_ID = "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1"
PROTOCOL_ID = "STRAP_GEOMETRY_VALIDATION_PROTOCOL_V1"
FORMAL_STATUS = "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED"
PARENT_STATUS = "WRENCH_AND_TASK_DIRECTION_PARTIALLY_RESOLVED_REQUIRES_STATIC_VALIDATION"
PARENT_WRENCH_PROTOCOL_STATUS = "STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL_V1_READY_EXECUTION_NOT_AUTHORIZED"
TASK_DIRECTION_STATUS = "TASK_DIRECTION_REQUIRES_GEOMETRIC_VALIDATION"
TARGET = "ACTUAL_STRAP_PULL_LINE_OF_ACTION"
MODEL_MAPPING_STATUS = "NOT_YET_CALIBRATED"
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
PARENT_WRENCH_PROTOCOL_SHA256 = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"
PARENT_TASK_DEFINITION_SHA256 = "17e7e42a24393a3c1d0d4bb2f9ddfdfde6e0bbe664b01a7e242ba32ca2a37995"
FROZEN_PROTOCOL_SHA256 = "4da84b5ffde2bb5c7dc7b3baccf81cb1dbd8a7a0cb8c3bdd8d457a4f3399e337"
SETUP_REPETITIONS = 10
METROLOGY_REPEATS_PER_SETUP = 3
MONTE_CARLO_SAMPLES = 100_000
MONTE_CARLO_SEED = 20260901
RESULT_CLASSES = (
    "STRAP_PULL_GEOMETRY_VALIDATED",
    "STRAP_PULL_GEOMETRY_VALIDATED_WITH_LIMITATIONS",
    "STRAP_PULL_GEOMETRY_NOT_VALIDATED",
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/static_strap_pull_geometry_validation_protocol_v1"
PROTOCOL_PATH = OUTPUT / "STRAP_GEOMETRY_VALIDATION_PROTOCOL.json"


INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "PARENT_STATIC_WRENCH_PROTOCOL",
        "path": "external_simulation_audits/static_wrench_frame_sign_validation_protocol_v1/STATIC_WRENCH_VALIDATION_PROTOCOL.json",
        "markers": (PARENT_WRENCH_PROTOCOL_STATUS, TASK_DIRECTION_STATUS, "this protocol changes no wrench flag"),
        "exact_sha256": PARENT_WRENCH_PROTOCOL_SHA256,
    },
    {
        "id": "PARENT_TASK_DIRECTION_DEFINITION",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/TASK_DIRECTION_FORMAL_DEFINITION.json",
        "markers": (TARGET, "p_robot_attach_B(t)", "p_limb_attach_B(t)", TASK_DIRECTION_STATUS),
        "exact_sha256": PARENT_TASK_DEFINITION_SHA256,
    },
    {
        "id": "PARENT_STRAP_GEOMETRY_AUDIT",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/STRAP_PULL_GEOMETRY_AUDIT.md",
        "markers": ("physical eyelet/cuff connection", "not an ankle", "actual line of action remains unknown"),
        "exact_sha256": "d3e48f38d7102d5da2f01aeae4b86e0f95141610f586254af6a886e8ae23fc30",
    },
    {
        "id": "PARENT_STATIC_GEOMETRY_PLAN",
        "path": "external_simulation_audits/wrench_frame_and_task_direction_resolution_v1/FUTURE_STATIC_GEOMETRY_VALIDATION_PLAN.md",
        "markers": ("rigid rehabilitation fixture/phantom", "robot-side strap eye/attachment center", "task-direction angular uncertainty"),
        "exact_sha256": "b8cf4a3ff4818ca7531e231e619925451a4ad76f8f4c0dd885f99229f89ae9a9",
    },
    {
        "id": "FORMAL_EXPERIMENT_MANIFEST",
        "path": "config/formal_experiment_manifest.json",
        "markers": ("ROM_PROTOCOL_V2", "q_hip - q_knee", ACTIVE_REFERENCE_SHA256),
        "exact_sha256": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    },
    {
        "id": "ACTIVE_REFERENCE",
        "path": "reference_release/reference_measured_asymmetric_closed_slow.csv",
        "markers": ("reference_measured_asymmetric_closed_slow", "theta_shank = q_hip - q_knee", "knee_to_strap_equivalent_pull_point"),
        "exact_sha256": ACTIVE_REFERENCE_SHA256,
    },
    {
        "id": "REHAB_FRAME_CONFIG",
        "path": "config/rehab_frame_config.json",
        "markers": ('"rehab_x_axis_in_base": null', '"rehab_z_axis_in_base": null', '"reviewed": false'),
        "exact_sha256": "eeb4b28a359acc74ab9cf9dcdf5ebbf41bf8f40e4eae1020045011aecdd57606",
    },
    {
        "id": "START_ANCHORED_TRAJECTORY",
        "path": "control/start_anchored_relative_trajectory.py",
        "markers": ("L2_knee_to_equivalent_shank_strap_pull_point_m", '"hip_center_required": False', '"observed_ankle_used_as_pull_point": False'),
        "exact_sha256": "f9a9ec92183fe490ea64e909dcd1a00a1819daf369e1ed4b05403d50f51ca8e6",
    },
    {
        "id": "MODEL_CONFIG",
        "path": "lower_limb_sim/config.py",
        "markers": ("L2 = 0.30", "完整小腿长度不等于 L2", "膝关节到束缚带等效牵引点"),
        "exact_sha256": "5f4da8438ad7776b374eee6870307a0b5cf81ef6bb21c29b2dc964c540a8d234",
    },
    {
        "id": "MODEL_KINEMATICS",
        "path": "lower_limb_sim/kinematics.py",
        "markers": ("shank_angle = q_hip_array - q_knee_array", "x_pull", "z_pull"),
        "exact_sha256": "cc18b3d62b491c009959f72601f15b838f9cb2dbafcaabd06df0906af6de0b94",
    },
    {
        "id": "CURRENT_ARCHITECTURE",
        "path": "CURRENT_ARCHITECTURE.md",
        "markers": ("equivalent strap pull point", "q_hip - q_knee", "L2"),
        "exact_sha256": "ea05150712f5422c71b0226e0e1e739364a2e36793766b79642430f7a2b61cfb",
    },
    {
        "id": "REAL_ROBOT_EXPERIMENT_BOUNDARY",
        "path": "REAL_ROBOT_EXPERIMENT.md",
        "markers": ("strap", "dummy", "subject"),
        "exact_sha256": "0404269be7a2428ab5c81873468cfdde7b281ef905dd97911de95a39a161514b",
    },
    {
        "id": "PROJECT_README",
        "path": "README.md",
        "markers": ("仰卧位", "等效", "束带"),
        "exact_sha256": "3fcd4e2c546eef1dc5d2365971d9cc0233e14d7cad47e7861a87df6826e76417",
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def verify_inputs() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for spec in INPUT_SPECS:
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing input: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"semantic marker mismatch {spec['id']}: {missing}")
        digest = sha256_file(path)
        if digest != spec["exact_sha256"]:
            raise RuntimeError(f"exact SHA mismatch {spec['id']}: {digest}")
        verified.append({
            "input_id": spec["id"],
            "path": spec["path"],
            "sha256": digest,
            "semantic_markers": list(spec["markers"]),
            "semantic_markers_pass": True,
        })

    parent = read_json(ROOT / INPUT_SPECS[0]["path"])
    task = read_json(ROOT / INPUT_SPECS[1]["path"])
    rehab = read_json(ROOT / "config/rehab_frame_config.json")
    if parent["formal_status"] != PARENT_WRENCH_PROTOCOL_STATUS:
        raise RuntimeError("parent static wrench protocol status changed")
    if task["status"] != TASK_DIRECTION_STATUS or task["target_definition"] != TARGET:
        raise RuntimeError("parent task-direction definition changed")
    if rehab["reviewed"] is not False or rehab["rehab_x_axis_in_base"] is not None:
        raise RuntimeError("unexpected rehab setup frame approval")
    return verified


def topology_payload() -> dict[str, Any]:
    return {
        "topology_id": "STRAP_MECHANICAL_TOPOLOGY_V1",
        "ordered_chain": [
            "robot flange and configured tool/TCP",
            "physical robot-side strap eyelet/hook/fixture load-transfer center",
            "taut free strap segment",
            "strap exit/tangent region at wide cuff",
            "wide cuff wrap and distributed shank contact",
            "rigid shank surrogate during initial validation",
        ],
        "PHYSICAL_HARDWARE_DEFINED": [
            "component roles exist in the intended setup",
            "hardware-specific dimensions, offsets, routing and identifiers are not yet recorded",
        ],
        "MODEL_EQUIVALENT": [
            "L1/L2 planar kinematics",
            "L2=0.30 m knee-to-strap-equivalent traction point",
            "start-anchored TCP displacement generated from that equivalent point",
        ],
        "ASSUMED_NOT_VALIDATED": [
            "TCP origin equals robot-side physical attachment",
            "wide cuff has one fixed physical limb-side attachment point",
            "bed/setup frame is already aligned with robot base/world",
            "static line remains constant throughout the trajectory",
        ],
    }


def attachment_payload() -> dict[str, Any]:
    return {
        "robot_side": {
            "selected_definition": "PHYSICAL_STRAP_EYELET_OR_HOOK_LOAD_TRANSFER_CENTER",
            "not_selected_as_default": ["TCP_ORIGIN", "FLANGE_ORIGIN"],
            "local_point_id": "p_attach_TCP",
            "local_coordinates_m": None,
            "future_equation": "p_robot_attach_B(t) = T_B_TCP(t) * homogeneous(p_attach_TCP)",
            "measurement": "independent caliper/CMM/3-D digitization or rigid fixture survey; repeat after remove/reinstall if removable",
            "tcp_origin_allowed_only_if": "measured fixed offset is consistent with zero within preregistered metrology uncertainty",
        },
        "limb_side": {
            "selected_operational_definition": "EQUIVALENT_LINE_POINT_AT_OBSERVED_STRAP_EXIT",
            "point_id": "p_limb_attach_B(t)",
            "meaning": "one observable point on the validated taut free-span line at the contact-to-free-span boundary; not claimed as the physical resultant application point",
            "future_equation": "p_limb_attach_B(t) = T_B_R * T_R_S(t) * homogeneous(p_exit_S(configuration))",
            "direct_line_alternative": "fit the taut free-span line from at least two independently digitized free-segment fiducials; orient limb-to-robot",
            "configuration_dependent": True,
            "global_constant_allowed": False,
            "invalid_if": ["slack", "multiple free-span lines", "broad or indeterminate exit region", "routing contact outside frozen cuff", "slip", "placement mismatch"],
        },
        "direction": {
            "construction": "d_task_B(t) = normalize(p_robot_attach_B(t) - p_limb_attach_B(t))",
            "positive_geometric_direction": "limb/cuff exit toward robot attachment (robotward)",
            "wrench_physical_sign_resolved": False,
        },
    }


def frame_payload() -> dict[str, Any]:
    return {
        "frame_id": "REHAB_SETUP_FRAME",
        "symbol": "R",
        "origin": "preregistered durable origin fiducial on the rigid bed/setup base",
        "axes": {
            "+x_R": "bed-plane longitudinal vector from proximal-side fiducial toward distal/foot-side fiducial",
            "+z_R": "upward unit normal to the fitted rigid bed/setup reference plane",
            "+y_R": "normalize(z_R cross x_R)",
        },
        "handedness": "right-handed; x_R cross y_R = z_R",
        "physical_landmarks": ["origin fiducial", "distal longitudinal fiducial", "at least two additional non-collinear bed-plane fiducials"],
        "transform_to_robot_base": {
            "id": "T_B_R",
            "value": None,
            "status": "ROBOT_TO_REHAB_SETUP_FRAME_CALIBRATION_REQUIRED",
            "method": "rigid registration from >=3 non-collinear reference points observed in both base-associated and setup metrology frames",
        },
        "frame_chain": {
            "robot_side": "p_attach_TCP -> T_B_TCP(t) -> p_robot_attach_B(t)",
            "limb_side": "p_exit_S -> T_R_S(t) -> T_B_R -> p_limb_attach_B(t)",
            "common_computation_frame": "robot base B",
            "world_optional": "only through a separately validated T_W_B; not assumed",
        },
        "no_unmeasured_transform_filled": True,
    }


def static_measurement_payload() -> dict[str, Any]:
    return {
        "plan_id": "STATIC_NONHUMAN_STRAP_GEOMETRY_MEASUREMENT_V1",
        "execution_authorized": False,
        "subject": "rigid cylindrical shank surrogate/mannequin fixture; no human",
        "setup_repetitions": SETUP_REPETITIONS,
        "metrology_repeats_per_setup": METROLOGY_REPEATS_PER_SETUP,
        "post_result_repeat_extension_allowed": False,
        "configuration_roles": [
            {"id": "G0_NOMINAL", "role": "nominal frozen rehab setup", "exact_fixture_coordinates": None},
            {"id": "G1_FLEXION_SIDE", "role": "representative flexion-side surrogate configuration", "exact_fixture_coordinates": None},
            {"id": "G2_EXTENSION_SIDE", "role": "representative extension-side surrogate configuration", "exact_fixture_coordinates": None},
        ],
        "configuration_status": "EXACT_JIG_COORDINATES_MUST_BE_FROZEN_IN_AUTHORIZED_EXECUTION_MANIFEST_BEFORE_RESULTS",
        "per_repetition_sequence": [
            "confirm equipment IDs/calibration and rigid frame fiducials",
            "remove cuff/strap completely",
            "reset surrogate in the preregistered jig configuration",
            "reattach cuff using preregistered landmarks, wrap order and closure procedure",
            "establish a nonhuman fixture geometry state; preload value remains null pending independent review",
            "digitize setup fiducials, robot attachment, surrogate pose, cuff landmarks, strap exit and >=2 free-span fiducials",
            "repeat metrology three times without reinstalling to separate measurement noise from setup variability",
            "remove strap before the next setup repetition",
        ],
        "geometry_preload_n": None,
        "geometry_preload_status": "GEOMETRY_PRELOAD_REQUIRES_INDEPENDENT_FIXTURE_AND_SAFETY_REVIEW",
        "robot_tcp_probing": "OPTIONAL_ONLY_AFTER_INDEPENDENT_ROBOT_SAFETY_AUTHORIZATION; not required by this protocol",
        "minimum_equipment": [
            "rigid shank surrogate or mannequin fixture",
            "actual identified strap/cuff and fixed robot-side attachment fixture",
            "repeatable jig and setup reference markers/fiducials",
            "calibrated ruler/caliper only where its uncertainty is demonstrably adequate",
            "3-D digitizer, photogrammetry or tracked pointer if required by the uncertainty budget",
            "independent registration artifacts for robot base and rehab setup frames",
        ],
    }


def uncertainty_payload() -> dict[str, Any]:
    return {
        "model_id": "TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL",
        "definitions": [
            "v = p_r - p_l",
            "L = norm(v)",
            "d = v / L",
            "J_d_v = (I - d d^T) / L",
            "Sigma_v = Sigma_r + Sigma_l - Sigma_rl - Sigma_lr",
            "Sigma_d ~= J_d_v Sigma_v J_d_v^T",
            "small_angle_E_theta2_rad2 ~= trace(Sigma_d)",
        ],
        "covariance_components": [
            "instrument/metrology repeatability",
            "robot-attachment localization and removable-fixture reinstall",
            "T_B_R rigid-registration covariance",
            "surrogate-pose localization",
            "cuff placement and strap exit setup-to-setup covariance",
            "free-span line fit residual and point cross-covariance",
        ],
        "estimation": {
            "within_setup": "from 3 metrology repeats per setup",
            "between_setup": "from 10 complete remove/reattach repetitions",
            "analytic": "first-order covariance propagation above",
            "nonlinear_check": "deterministic Monte Carlo using measured covariance and the exact normalization",
            "monte_carlo_samples": MONTE_CARLO_SAMPLES,
            "monte_carlo_seed": MONTE_CARLO_SEED,
        },
        "report": ["mean direction", "angular SD", "P95 angular error", "maximum observed setup angular deviation", "endpoint displacement covariance", "line-fit residual"],
        "thresholds": {
            "minimum_endpoint_separation_m": None,
            "maximum_P95_angular_error_deg": None,
            "maximum_setup_displacement_m": None,
            "maximum_line_fit_residual_m": None,
        },
        "threshold_status": "THRESHOLDS_REQUIRE_METROLOGY_AND_ENDPOINT_ERROR_BUDGET_REVIEW_BEFORE_RESULTS",
        "scientific_endpoint_outcomes_used": False,
    }


def validity_gate_payload() -> dict[str, Any]:
    checks = [
        "required robot attachment, exit/free-span and setup points available",
        "T_B_TCP and calibrated T_B_R available with versioned uncertainty",
        "all points expressed in common robot base frame",
        "identified strap/cuff/fixture and frozen placement procedure match",
        "strap is taut and has one observable free-span line",
        "no slip, extra routing contact, split path or ambiguous broad exit",
        "endpoint separation exceeds prefrozen metrology threshold",
        "direction is finite and unit-normalizable",
        "ten setup repetitions completed without posthoc extension",
        "repeatability and propagated uncertainty meet prefrozen thresholds",
        "dynamic use additionally has synchronized valid endpoint/pose data",
    ]
    return {
        "gate_id": "STRAP_PULL_GEOMETRY_VALIDITY_GATE_V1",
        "current_status": "FAIL_CLOSED_NOT_EXECUTED",
        "all_checks_required": True,
        "checks": [{"check_id": f"G{index:02d}", "requirement": text, "current_pass": False} for index, text in enumerate(checks, start=1)],
        "on_any_failure": {"d_task": None, "reason": "GEOMETRY_INVALID_OR_UNVALIDATED", "fallback_direction_allowed": False},
        "explicitly_prohibited_fallbacks": ["TCP_TRAJECTORY_TANGENT", "FIXED_BED_DIRECTION", "MODEL_L2_DIRECTION", "GUESSED_ATTACHMENT_POINT"],
    }


def protocol_payload(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "formal_status": FORMAL_STATUS,
        "protocol_scope": "STATIC_NONHUMAN_GEOMETRY_PROTOCOL_DESIGN_ONLY",
        "parent_status": PARENT_STATUS,
        "parent_wrench_protocol_status": PARENT_WRENCH_PROTOCOL_STATUS,
        "parent_wrench_protocol_sha256": PARENT_WRENCH_PROTOCOL_SHA256,
        "parent_task_direction_definition_sha256": PARENT_TASK_DEFINITION_SHA256,
        "target": TARGET,
        "task_direction_status": TASK_DIRECTION_STATUS,
        "task_direction_construction": "d_task_B(t) = normalize(p_robot_attach_B(t) - p_limb_attach_B(t))",
        "topology": topology_payload(),
        "attachments": attachment_payload(),
        "rehab_setup_frame": frame_payload(),
        "static_measurement": static_measurement_payload(),
        "uncertainty_model": uncertainty_payload(),
        "validity_gate": validity_gate_payload(),
        "model_pull_point_mapping_status": MODEL_MAPPING_STATUS,
        "future_result_classes": list(RESULT_CLASSES),
        "primary_endpoint_finalized": False,
        "primary_endpoint_validated": False,
        "input_files": inputs,
        "protocol_frozen_before_any_physical_result": True,
        "forbidden_operations": [
            "connect_robot", "power_or_enable_robot", "send_motion_or_probe",
            "execute_rehabilitation_trajectory", "use_human_subject",
            "apply_formal_traction_load", "execute_geometry_experiment",
            "modify_hardware_control_or_safety", "modify_reference_ROM_or_L2",
            "run_PINN", "run_BO", "compute_or_finalize_primary_endpoint",
            "use_TCP_tangent_or_bed_direction_as_fallback",
            "change_repeat_count_threshold_or_definition_after_results",
        ],
    }


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"prepare requires empty output directory: {OUTPUT}")
    inputs = verify_inputs()
    atomic_json(PROTOCOL_PATH, protocol_payload(inputs))
    atomic_json(OUTPUT / "INPUT_VERIFICATION.json", {
        "stage_id": STAGE_ID,
        "input_count": len(inputs),
        "all_inputs_present_and_semantically_verified": True,
        "inputs": inputs,
        "physical_geometry_result_files_read": [],
        "scientific_endpoint_outcomes_read": [],
        "robot_access_count": 0,
        "human_data_access_count": 0,
    })
    atomic_json(OUTPUT / "HARDWARE_ACCESS_AUDIT.json", {
        "protocol_design_only": True,
        "robot_constructed": False,
        "robot_connected": False,
        "power_or_enable_count": 0,
        "motion_or_probing_command_count": 0,
        "formal_traction_load_count": 0,
        "geometry_experiment_count": 0,
        "human_subject_count": 0,
        "endpoint_computation_count": 0,
    })
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "input_count": len(inputs)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen")
    if sha256_file(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("frozen geometry protocol SHA mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["input_files"] != verify_inputs():
        raise RuntimeError("frozen inputs changed")
    return protocol


def mechanical_topology_audit() -> str:
    return """# Strap Mechanical Topology Audit

## Actual intended chain and evidence class

The intended load path is: robot flange/configured tool and TCP -> physical eyelet, hook or fixture load-transfer center -> taut free strap segment -> exit/tangent region at a wide cuff -> distributed cuff/shank contact -> rigid shank surrogate for initial validation.

`PHYSICAL_HARDWARE_DEFINED` currently means only that those component roles belong to the intended apparatus. Repository evidence does **not** contain the actual eyelet offset, cuff dimensions, routing, tension, identifiers or installed coordinates. `MODEL_EQUIVALENT` comprises the planar L1/L2 kinematics, the 0.30 m knee-to-strap-equivalent point, and the start-anchored TCP displacement derived from it. It does not make the L2 point an ankle or an observed attachment.

The following remain `ASSUMED`, not measured: TCP origin equals the eyelet; the wide cuff has one fixed physical point; the bed frame is aligned with base/world; a static pull direction represents the full trajectory. No available diagram or metadata resolves these quantities.
"""


def robot_attachment_definition() -> str:
    return """# Robot-Side Attachment Definition

The primary physical point is `PHYSICAL_STRAP_EYELET_OR_HOOK_LOAD_TRANSFER_CENTER`: the center through which the taut free strap transfers load to the rigid robot fixture. TCP and flange origins are frame references, not default attachment points.

For a fixed local point `p_attach_TCP`, future reconstruction is

`p_robot_attach_B(t) = T_B_TCP(t) * homogeneous(p_attach_TCP)`.

`p_attach_TCP` remains null. It must be measured independently by calibrated caliper/CMM/3-D digitization or a rigid fixture survey. A removable fixture requires remove/reinstall repetitions. TCP may substitute only if the measured offset is consistent with zero inside a preregistered uncertainty bound. No offset is invented here.
"""


def limb_attachment_audit() -> str:
    return """# Limb-Side Equivalent Attachment Audit

| Candidate | Physical meaning | Observability/repeatability | Wide-cuff and line compatibility | Decision |
|---|---|---|---|---|
| cuff geometric center | geometric marker, not necessarily load transfer | easy to mark; placement-dependent | ignores distributed pressure and wrap | diagnostic only |
| strap exit/tangent point | boundary where contact becomes the taut free span | observable with fiducials/digitization; varies with pose, tension, wrap and placement | directly supports free-span direction if one line exists | primary operational line point |
| equivalent resultant-force application point | point/line giving the distributed contact resultant | mechanically meaningful, not directly observable from geometry alone | can represent net force but may omit a contact moment | future mechanics target |
| model L2 point | fixed 2-D knee-to-equivalent-point construction | exactly computable in model coordinates | not physical evidence | model diagnostic only |
| configuration-dependent contact/resultant | actual state-dependent distributed transfer | requires pressure/force/moment evidence | most realistic | future extension |

A wide cuff does not currently admit a defensible unique physical `p_limb_attach`. The proposed operational definition is `EQUIVALENT_LINE_POINT_AT_OBSERVED_STRAP_EXIT`: one point on the validated taut free span at its exit boundary, or equivalently a line fitted from at least two free-span fiducials. It defines direction but is not declared the true resultant application point.

Exit geometry must be measured across configuration and ten remove/reattach setups. It must not become a global constant if it varies materially. Slack, multiple spans, broad/ambiguous exit, unexpected routing contact or slip fails the gate.
"""


def point_force_audit() -> str:
    return """# Point-Force Approximation Audit

**Answer:** a single strap line-of-action is mechanically defensible only as a `MODEL APPROXIMATION`, not as a statement that every pressure/contact force under the wide cuff is collinear.

Distributed cuff pressure can have a net resultant force and, in general, a net moment. Geometry of a single taut free strap segment can identify the transmitted free-span direction. It cannot by itself prove the cuff resultant application point, exclude a contact moment, or recover the complete pressure field.

The approximation may be used later only when the free span is taut and single, routing/placement is repeatable, propagated angular uncertainty passes a prefrozen gate, and separate evidence shows any omitted moment/contact variation is acceptable for the intended endpoint. Otherwise `d_task=null`. A tactile array may characterize contact region, pressure centroid and changes, but without independent calibration it is neither geometry ground truth nor comfort truth.
"""


def rehab_frame_definition() -> str:
    return """# Rehab Setup Frame Definition

`REHAB_SETUP_FRAME` (`R`) is fixed to the rigid bed/setup base, not to a human hip. Its origin is a preregistered durable setup fiducial. `+x_R` is the bed-plane longitudinal direction from proximal to distal/foot-side fiducials. `+z_R` is the upward normal to the fitted rigid reference plane. `+y_R=normalize(z_R cross x_R)`, so `x_R cross y_R=z_R`.

At least four labelled landmarks are required: origin, distal longitudinal and at least two other non-collinear bed-plane fiducials. Their geometry, coordinate values and uncertainty must be frozen in the authorized execution manifest. No transform is populated here.

Robot chain: `p_attach_TCP -> T_B_TCP(t) -> p_robot_attach_B(t)`. Limb chain: `p_exit_S -> T_R_S(t) -> T_B_R -> p_limb_attach_B(t)`. Both points enter `d_task` only in common base frame `B`. Controller world may be used only through a separately validated `T_W_B`.
"""


def calibration_plan() -> str:
    return """# Robot-to-Setup Frame Calibration Plan

Estimate `T_B_R` by rigid registration from at least three non-collinear reference points that are physically tied to the rehab setup and observed in a base-associated metrology frame. Prefer redundant fiducials and report registration residuals, leave-one-out error and transform covariance.

Possible non-robot methods include a calibrated 3-D digitizer/photogrammetry system that observes both rigid robot-base fiducials and setup fiducials, or a surveyed rigid jig with certified coordinates. Robot TCP probing is optional only after independent robot safety authorization; this protocol neither requires nor authorizes it.

Freeze device IDs, calibration certificates, point correspondence, transform convention, fiducial coordinates, fit algorithm and thresholds before geometry results. Bed axes are not assumed equal to base/world axes. Missing or failed registration makes `T_B_R=null` and `d_task=null`.
"""


def static_measurement_plan() -> str:
    return f"""# Static Geometry Measurement Plan

This is a future nonhuman procedure using the actual identified strap/cuff, a rigid shank surrogate and fixed robot-side attachment fixture. It contains `{SETUP_REPETITIONS}` complete remove/reattach repetitions and `{METROLOGY_REPEATS_PER_SETUP}` metrology repeats within each setup. The counts are frozen before results and cannot be extended until PASS.

For each preregistered configuration role (`G0_NOMINAL`, `G1_FLEXION_SIDE`, `G2_EXTENSION_SIDE`), record setup fiducials, robot attachment, surrogate pose/axis, cuff placement landmarks, strap exit region and at least two points on the free span. Completely remove and reinstall the cuff between setup repetitions using a frozen wrap/closure/landmark procedure. Exact jig coordinates must be frozen in a separately authorized execution manifest before any result; they are null here because the apparatus has not been physically surveyed.

The line requires a tensioned nonhuman geometry state, but no preload is selected: `GEOMETRY_PRELOAD_REQUIRES_INDEPENDENT_FIXTURE_AND_SAFETY_REVIEW`. No formal traction load, subject or robot motion is authorized. Calipers/rulers are allowed only when their calibrated uncertainty meets the budget; otherwise use 3-D digitization/tracked fiducials.

Preserve every attempt, invalid state and raw metrology record. Do not delete a failed configuration or choose a different exit definition after seeing results.
"""


def repeatability_plan() -> str:
    return f"""# Strap Setup Repeatability Plan

For each of `{SETUP_REPETITIONS}` independent setups: remove the strap/cuff completely, reset the surrogate jig, reinstall from the same landmarks with the same wrap/closure sequence, establish the reviewed nonhuman geometry state, and digitize the complete geometry `{METROLOGY_REPEATS_PER_SETUP}` times without reinstalling.

Within-setup repeats estimate instrument/point-picking noise. Between-setup variation estimates cuff placement, exit and direction repeatability. Report robot attachment covariance (including fixture reinstall where applicable), exit/line covariance, setup-to-setup displacement, free-span line-fit residual, direction angular SD/P95/max and failures by reason.

The repeat count must not be increased after results. Repeatability, displacement and angular thresholds remain null until metrology accuracy and acceptable endpoint error are independently reviewed and frozen. A failed threshold yields `STRAP_PULL_GEOMETRY_NOT_VALIDATED` or a preregistered limitation; it never triggers a guessed direction.
"""


def uncertainty_model_md() -> str:
    return f"""# Task-Direction Geometry Uncertainty Model

Let `v=p_r-p_l`, `L=norm(v)` and `d=v/L`. First-order propagation uses

`J=(I-d d^T)/L`, `Sigma_v=Sigma_r+Sigma_l-Sigma_rl-Sigma_lr`, and `Sigma_d ~= J Sigma_v J^T`.

For small errors, `E[theta^2] ~= trace(Sigma_d)`. Include point-picking noise, eyelet localization/reinstall, `T_B_R`, surrogate pose, cuff remove/reattach, exit location and free-span fit residual; retain cross-covariance when points share a registration.

Estimate within-setup and between-setup components separately. Confirm the linearization with `{MONTE_CARLO_SAMPLES}` deterministic samples (seed `{MONTE_CARLO_SEED}`) through exact normalization. Report angular SD, P95 and maximum observed setup deviation plus endpoint/line uncertainties. Minimum separation, angular, displacement and line-fit thresholds remain null pending prospective metrology/endpoint-error-budget review. No scientific endpoint outcome may tune them.
"""


def dynamic_reconstruction_plan() -> str:
    return """# Dynamic Task-Direction Reconstruction Plan

Static validation establishes geometry at registered fixture configurations; it does not make the line constant during the 24 s trajectory.

| Candidate | Evidence meaning | Current disposition |
|---|---|---|
| A: TCP-derived robot point + fixed limb point | assumes limb/exit fixed | invalid for a moving limb unless future variation is below a prefrozen bound |
| B: both endpoints from kinematics | reproducible model proxy | requires physical mapping calibration; cannot be called actual geometry now |
| C: robot point from TCP + limb point from measured limb pose | practical state reconstruction | preferred online candidate after pose/exit mapping and timing validation |
| D: direct external tracking of free span/endpoints | strongest physical geometry reference | preferred validation reference where feasible |
| E: one static direction | approximation | allowed only for a declared limited range after validation |

Minimum dynamic information is: measured `p_attach_TCP`; synchronized valid `T_B_TCP(t)`; calibrated `T_B_R`; measured limb/surrogate pose `T_R_S(t)`; configuration-dependent exit mapping or directly tracked free-span line; strap/cuff/routing/tautness state; placement identifier; transform and timing uncertainty. Recompute per valid sample after all gates. Missing information yields `d_task(t)=null`.

`TCP_TRAJECTORY_TANGENT != STRAP_PULL_LINE_OF_ACTION`. Future angles between strap direction and TCP tangent, fixed bed direction or model direction are diagnostics only and cannot select the endpoint-favorable approximation.
"""


def model_mapping_plan() -> str:
    return """# Model Pull Point to Physical Strap Mapping

Current classification: `MODEL_PULL_POINT_TO_PHYSICAL_STRAP_MAPPING = NOT_YET_CALIBRATED`.

The 2-D point uses `L2=0.30 m` from the knee and `theta_shank=q_hip-q_knee`. It is an equivalent traction point used for planar FK and start-anchored TCP displacement; it is not the ankle or a measured cuff attachment.

A future mapping audit should transform the L2 point into `REHAB_SETUP_FRAME`, compare it against the measured free-span line at every registered configuration, and report: nearest-point distance, direction-angle difference, configuration dependence and—if independently measured—resultant moment equivalence. Thresholds must be frozen before outcomes. The mapping may become `APPROXIMATE` only if its limited range passes; `DIRECTLY_MATCHED` requires direct geometric/mechanical evidence. This protocol changes neither L2 nor model kinematics.
"""


def future_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Future static strap pull geometry result",
        "type": "object",
        "required": ["protocol_sha256", "execution_manifest_sha256", "raw_data_checksums", "configuration_results", "decision"],
        "properties": {
            "protocol_sha256": {"const": FROZEN_PROTOCOL_SHA256},
            "execution_manifest_sha256": {"type": "string"},
            "raw_data_checksums": {"type": "object"},
            "configuration_results": {"type": "array", "minItems": 3},
            "repeat_count": {"const": SETUP_REPETITIONS},
            "metrology_repeats_per_setup": {"const": METROLOGY_REPEATS_PER_SETUP},
            "T_B_R": {"type": ["array", "null"]},
            "p_attach_TCP_m": {"type": ["array", "null"]},
            "uncertainty_results": {"type": "object"},
            "decision": {"enum": list(RESULT_CLASSES)},
            "task_direction_update": {"enum": ["ELIGIBLE_FOR_ENDPOINT_REEVALUATION", "LIMITED_RANGE_ONLY", "REMAINS_UNVALIDATED"]},
            "wrench_sign_status": {"const": "REQUIRES_SEPARATE_STATIC_WRENCH_FRAME_SIGN_VALIDATION"},
        },
        "full_validation_requirements": [
            "all required configurations and ten setup repetitions retained",
            "physical eyelet and operational exit/free-span line measured in common frame",
            "T_B_R calibration and uncertainty valid",
            "single taut line/routing/placement gate passes",
            "repeatability and propagated uncertainty pass all prefrozen thresholds",
            "no post-result definition, threshold or repeat-count change",
        ],
        "protocol_itself_must_not_emit_validated": True,
    }


def dependency_graph() -> str:
    return """# Next-Stage Dependency Graph

```text
STATIC_WRENCH_FRAME_SIGN_VALIDATION_PROTOCOL
                    +
STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL
                    |
        future safety / execution authorization
                    |
  physical static wrench validation + physical static geometry validation
                    |
PRIMARY_MECHANICAL_ENDPOINT_FINALIZATION_AND_VALIDATION_PROTOCOL
```

The two branches remain independent: wrench response/sign evidence does not locate the strap, and geometry evidence does not establish wrench frame/sign. Only sufficiently successful physical results from both branches permit endpoint reevaluation. Geometry protocol readiness alone cannot finalize or validate `J_force`. No downstream stage was executed.
"""


def report() -> str:
    return f"""# Static Strap Pull Geometry Validation Protocol V1

## Formal status

`{FORMAL_STATUS}`

This stage froze a future static, nonhuman geometry protocol. It executed no physical measurement and cannot output a validated class.

## Answers to the ten protocol questions

1. **Robot-side point:** the actual eyelet/hook/fixture load-transfer center, expressed as a measured fixed offset `p_attach_TCP`; it is not automatically TCP or flange origin.
2. **Unique limb-side point:** the current wide cuff does not have an evidenced unique physical point. Contact is distributed and may also transmit a net moment.
3. **Proposed equivalent:** use the observed contact-to-free-span exit point, or a fitted taut free-span line, as an operational line point. This is a model approximation, not the true pressure resultant point.
4. **Common frame:** compute both endpoints in robot base `B`, linked through a physical `REHAB_SETUP_FRAME R`.
5. **Transform:** obtain `T_B_R` by redundant rigid fiducial registration using independent metrology; no numeric transform is invented and robot probing needs separate authorization.
6. **Repeatability:** `{SETUP_REPETITIONS}` complete remove/reattach setups and `{METROLOGY_REPEATS_PER_SETUP}` point measurements per setup, separated into within- and between-setup covariance; no post-result extension.
7. **Static or dynamic:** configuration-dependent unless future evidence supports a declared limited static approximation.
8. **Dynamic minimum:** `p_attach_TCP`, synchronized `T_B_TCP(t)`, calibrated `T_B_R`, limb pose or direct free-span tracking, exit mapping, strap state/placement and timing/uncertainty.
9. **2-D mapping:** `{MODEL_MAPPING_STATUS}`. L2 stays a knee-to-equivalent traction-point length and is neither ankle nor measured attachment.
10. **Before `J_force`:** authorized nonhuman execution must validate common-frame endpoints/free-span, routing, transform, ten-setup repeatability and propagated angular uncertainty under prefrozen thresholds; separately, the static wrench protocol must validate frame/sign sufficiently.

## Mechanics and sign boundary

The positive geometric direction remains limb exit toward robot attachment. This does not resolve the physical sign of the reported wrench; that remains a separate static wrench validation dependency. `TCP_TRAJECTORY_TANGENT` and fixed bed/model directions are diagnostics only, never fail-open fallbacks.

## Current state

All thresholds, `T_B_R`, attachment offset, jig coordinates and geometry preload are null pending independent review. Therefore `{TASK_DIRECTION_STATUS}`, `PRIMARY_ENDPOINT_FINALIZED=false`, `PRIMARY_ENDPOINT_VALIDATED=false`, `NOT_HUMAN_READY` and `NOT_ROBOT_APPROVED` remain unchanged.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files) + "\n")


def execute() -> None:
    protocol = verify_freeze()
    atomic_text(OUTPUT / "STRAP_MECHANICAL_TOPOLOGY_AUDIT.md", mechanical_topology_audit())
    atomic_text(OUTPUT / "ROBOT_SIDE_ATTACHMENT_DEFINITION.md", robot_attachment_definition())
    atomic_text(OUTPUT / "LIMB_SIDE_EQUIVALENT_ATTACHMENT_AUDIT.md", limb_attachment_audit())
    atomic_text(OUTPUT / "POINT_FORCE_APPROXIMATION_AUDIT.md", point_force_audit())
    atomic_text(OUTPUT / "REHAB_SETUP_FRAME_DEFINITION.md", rehab_frame_definition())
    atomic_text(OUTPUT / "ROBOT_TO_SETUP_FRAME_CALIBRATION_PLAN.md", calibration_plan())
    atomic_text(OUTPUT / "STATIC_GEOMETRY_MEASUREMENT_PLAN.md", static_measurement_plan())
    atomic_text(OUTPUT / "STRAP_SETUP_REPEATABILITY_PLAN.md", repeatability_plan())
    atomic_text(OUTPUT / "TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL.md", uncertainty_model_md())
    atomic_json(OUTPUT / "TASK_DIRECTION_GEOMETRY_UNCERTAINTY_MODEL.json", uncertainty_payload())
    atomic_text(OUTPUT / "DYNAMIC_TASK_DIRECTION_RECONSTRUCTION_PLAN.md", dynamic_reconstruction_plan())
    atomic_text(OUTPUT / "MODEL_PULL_POINT_TO_PHYSICAL_STRAP_MAPPING.md", model_mapping_plan())
    atomic_json(OUTPUT / "GEOMETRY_VALIDITY_GATE.json", validity_gate_payload())
    atomic_json(OUTPUT / "FUTURE_GEOMETRY_RESULT_SCHEMA.json", future_schema())
    atomic_text(OUTPUT / "NEXT_DEPENDENCY_GRAPH.md", dependency_graph())
    atomic_text(OUTPUT / "STATIC_STRAP_PULL_GEOMETRY_VALIDATION_PROTOCOL_REPORT.md", report())
    atomic_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID,
        "formal_status": FORMAL_STATUS,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "input_count": len(protocol["input_files"]),
        "target": TARGET,
        "task_direction_status": TASK_DIRECTION_STATUS,
        "model_mapping_status": MODEL_MAPPING_STATUS,
        "setup_repetitions": SETUP_REPETITIONS,
        "metrology_repeats_per_setup": METROLOGY_REPEATS_PER_SETUP,
        "thresholds_frozen": False,
        "geometry_preload_frozen": False,
        "future_physical_execution_authorized": False,
        "physical_geometry_validation_performed": False,
        "geometry_result_class": None,
        "primary_endpoint_finalized": False,
        "primary_endpoint_validated": False,
        "wrench_frame_sign_validated_by_this_stage": False,
        "robot_access_count": 0,
        "motion_or_probing_command_count": 0,
        "formal_traction_load_count": 0,
        "human_subject_count": 0,
        "pinn_run_count": 0,
        "bo_run_count": 0,
        "hardware_control_safety_modified": False,
        "reference_ROM_L2_modified": False,
        "not_human_ready": True,
        "not_robot_approved": True,
        "next_stage_executed": False,
    })
    write_checksums()
    print(json.dumps({
        "stage_id": STAGE_ID,
        "formal_status": FORMAL_STATUS,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "future_execution": "NOT_AUTHORIZED",
        "task_direction_status": TASK_DIRECTION_STATUS,
    }, indent=2))


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
