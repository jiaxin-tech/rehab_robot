"""Freeze the future real-measurement episode and endpoint design.

This builder performs source/document inspection only.  It never imports the
robot SDK, constructs an adapter, connects to hardware, sends motion, trains a
model, or runs an optimizer.  ``--prepare`` freezes evidence and protocol;
``--execute`` verifies that freeze before writing the remaining artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


STAGE_ID = "MEASUREMENT_DRIVEN_PERSONALIZATION_DATA_AND_ENDPOINT_DESIGN_V1"
PROTOCOL_ID = "DATA_AND_ENDPOINT_DESIGN_PROTOCOL"
PRIMARY_DIRECTION = "MECHANICAL_MEASUREMENT_DRIVEN_PERSONALIZATION"
INHERITED_STATUS = "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_READY_WITH_LIMITATIONS"
ENDPOINT_ID = "EPISODE_RMS_VALIDATED_TASK_DIRECTION_INTERACTION_FORCE"
ENDPOINT_STATE = "PRIMARY_MECHANICAL_ENDPOINT_DEFINITION_INCOMPLETE"
WRENCH_STATUS = "WRENCH_FRAME_SEMANTICS_NOT_VERIFIED"
TASK_DIRECTION_STATUS = "TASK_DIRECTION_REQUIRES_EXPERIMENTAL_VALIDATION"
FILTER_STATUS = "FILTER_NOT_YET_FROZEN"
MASTER_TIMEBASE = "HOST_MONOTONIC_PERF_COUNTER_NS"
NEXT_STAGE = "WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_V1"
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
PARENT_PROTOCOL_SHA256 = "41da6efac092da267a0e8477ff8453fe4790b5adc0e2bfd23d1ab7671cc58a45"
FROZEN_PROTOCOL_SHA256 = "de27c80d3ca93cd299c016ccb5d80032a8af417a2d06b91e2a01e5f0b2680f9e"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1"
PROTOCOL_PATH = OUTPUT / "DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json"


INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "PARENT_PROTOCOL",
        "path": "external_simulation_audits/personalized_rehabilitation_research_formulation_v2/RESEARCH_FORMULATION_V2_PROTOCOL.json",
        "markers": (PRIMARY_DIRECTION, ENDPOINT_ID),
        "exact_sha256": PARENT_PROTOCOL_SHA256,
    },
    {
        "id": "PARENT_DECISION",
        "path": "external_simulation_audits/personalized_rehabilitation_research_formulation_v2/FINAL_RESEARCH_FORMULATION_V2.json",
        "markers": (INHERITED_STATUS, '"PRIMARY_CANDIDATE_OUTCOME_FINALIZED": false'),
    },
    {
        "id": "PARENT_CHANNEL_INVENTORY",
        "path": "external_simulation_audits/personalized_rehabilitation_research_formulation_v2/MEASUREMENT_CHANNELS_AND_ROLES.csv",
        "markers": ("GETENDTORQUE_PATH_PRESENT_UNVALIDATED", "HOST_DIFFERENCE_ESTIMATE_PATH_PRESENT"),
    },
    {
        "id": "PARENT_HARDWARE_BOUNDARY",
        "path": "external_simulation_audits/personalized_rehabilitation_research_formulation_v2/HARDWARE_READINESS_BOUNDARY.json",
        "markers": ('"base_wrench_rotation_verified": false', '"robot_approved": false'),
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
        "markers": ("def getEndTorque", "FrameType::world", "FrameType::flange", "FrameType::tool", "单位N"),
    },
    {
        "id": "HARDWARE_SETTINGS",
        "path": "config/settings.py",
        "markers": ('ROBOT_FORCE_RAW_FRAME          = "world"', "BASE_WRENCH_ROTATION_VERIFIED  = False", 'BASE_WRENCH_TRANSFORM_KIND     = "rotation_only"'),
    },
    {
        "id": "EXPERIMENT_SAFETY_REVIEW",
        "path": "config/experiment_safety.json",
        "markers": ('"reviewed": false', '"max_force_n": null', '"max_state_wrench_skew_s": null'),
    },
    {
        "id": "REAL_IDENTIFICATION_REVIEW",
        "path": "config/real_identification_config.json",
        "markers": ('"reviewed": false', '"raw_wrench_frame": null', '"assumed_wrench_delay_s": null'),
    },
    {
        "id": "TYPED_STATE_SCHEMA",
        "path": "collection/state.py",
        "markers": ("InternalWrenchFrame", "host_monotonic_time_s", "transform_wrench", "p\u00d7(R F_a)"),
    },
    {
        "id": "SNAPSHOT_ALIGNMENT",
        "path": "collection/snapshot.py",
        "markers": ("base_wrench_rotation_requires_robot_validation", "force_query_duration_ms", "state_internal_skew_ms"),
    },
    {
        "id": "REAL_ACQUISITION",
        "path": "collection/real_robot_acquisition.py",
        "markers": ("Independent state, wrench, and alignment producers", "getEndTorque", "state_wrench_skew_s"),
    },
    {
        "id": "EPISODE_LOGGER",
        "path": "collection/episode_logger.py",
        "markers": ("robot_state", "robot_wrench", "trajectory_command", "aligned_snapshot"),
    },
    {
        "id": "OBSERVATION_ADAPTER",
        "path": "hardware/rokae_adapter.py",
        "markers": ("Observation-only project adapter", "host_query_start_s", "read_internal_wrench"),
    },
    {
        "id": "XCORE_WRENCH_CALL",
        "path": "hardware/windows/rokae_xcore.py",
        "markers": ("getEndTorque", "joint_measured", "cart_torque", "cart_force", "time.perf_counter_ns"),
    },
    {
        "id": "WRENCH_QUERY_DIAGNOSTIC",
        "path": "scripts/check_wrench_query_timing.py",
        "markers": ("without robot motion", "perf_counter_ns", "query_duration_ms"),
    },
    {
        "id": "SNAPSHOT_ALIGNMENT_DIAGNOSTIC",
        "path": "scripts/check_snapshot_alignment.py",
        "markers": ("perf_counter_ns", "state_internal_skew_ms", "force_query_duration_ms"),
    },
    {
        "id": "WRENCH_FRAME_DIAGNOSTIC",
        "path": "scripts/check_wrench_frame_rotation.py",
        "markers": ("perf_counter_ns", "raw_force_frame", "base_transform_kind"),
    },
    {
        "id": "WRENCH_POSE_DIAGNOSTIC",
        "path": "scripts/check_wrench_pose_dependence.py",
        "markers": ("perf_counter_ns", "pose_index", "does not determine whether xCoreSDK performs gravity compensation"),
    },
    {
        "id": "OFFLINE_FRAME_UNIT_TESTS",
        "path": "tests/test_units_and_frames.py",
        "markers": ("transform_wrench", "rotation_only_pending_robot_validation", "base_wrench_rotation_requires_robot_validation"),
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
    identification = read_json(ROOT / "config/real_identification_config.json")
    if not (
        parent["formal_decision"] == INHERITED_STATUS
        and parent["PRIMARY_THESIS_FORMULATION"] == PRIMARY_DIRECTION
        and parent["PRIMARY_CANDIDATE_OUTCOME"] == ENDPOINT_ID
        and parent["PRIMARY_CANDIDATE_OUTCOME_FINALIZED"] is False
    ):
        raise RuntimeError("parent formulation semantics changed")
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    ):
        raise RuntimeError("formal ROM/reference/angle convention changed")
    if safety["reviewed"] is not False or identification["reviewed"] is not False:
        raise RuntimeError("unexpected robot/identification approval")
    return rows


def protocol_payload(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "inherited_direction": PRIMARY_DIRECTION,
        "inherited_status": INHERITED_STATUS,
        "candidate_primary_endpoint": ENDPOINT_ID,
        "candidate_primary_endpoint_finalized_before_stage": False,
        "allowed_decision_states": [
            "PRIMARY_MECHANICAL_ENDPOINT_READY_FOR_VALIDATION",
            ENDPOINT_STATE,
        ],
        "decision_rules": {
            "wrench_semantics_unproved": WRENCH_STATUS,
            "task_direction_unproved": TASK_DIRECTION_STATUS,
            "either_primary_dependency_unproved": ENDPOINT_STATE,
            "endpoint_may_never_be_labelled_validated_in_this_stage": True,
        },
        "master_timebase_candidate": MASTER_TIMEBASE,
        "numeric_measurement_thresholds_may_be_invented": False,
        "safety_metrics_are_constraints_not_objective_terms": True,
        "real_personalization_causal_boundary": "beta_k uses valid episodes 1..k-1 only",
        "input_files": inputs,
        "forbidden_operations": [
            "construct_or_connect_robot_adapter", "power_on_robot", "enable_robot",
            "send_motion", "execute_trajectory", "collect_human_data", "calibrate_sensor",
            "modify_hardware_control_collection_or_safety", "train_model_or_PINN",
            "run_BO", "modify_V3", "modify_MyoLeg", "use_simulator_oracle",
            "modify_frozen_artifacts", "execute_next_stage",
        ],
        "protocol_frozen_before_endpoint_decision_outputs": True,
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
        "validated_physical_wrench_result_files_used": [],
        "physical_wrench_evidence_available_to_this_stage": False,
        "held_out_outcome_access_count": 0,
        "robot_access_count": 0,
        "human_data_access_count": 0,
    })
    atomic_json(OUTPUT / "HARDWARE_ACCESS_AUDIT.json", {
        "design_only": True,
        "robot_adapter_imported": False,
        "robot_constructed": False,
        "robot_connected": False,
        "robot_powered_or_enabled": False,
        "motion_command_count": 0,
        "calibration_call_count": 0,
        "human_collection_count": 0,
        "pinn_training_count": 0,
        "bo_run_count": 0,
        "read_only_diagnostic_scripts_inspected_not_executed": [
            "scripts/check_wrench_query_timing.py",
            "scripts/check_snapshot_alignment.py",
            "scripts/check_wrench_frame_rotation.py",
            "scripts/check_wrench_pose_dependence.py",
        ],
    })
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "input_count": len(inputs)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen")
    if sha256_file(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("protocol SHA mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["input_files"] != verify_inputs():
        raise RuntimeError("frozen evidence inputs changed")
    return protocol


def measurement_rows() -> list[dict[str, Any]]:
    common = {"formal_research_ready_now": False, "requires_future_validation": True}
    return [
        {"channel_id": "ROBOT_Q", "quantity": "joint position", "sdk_source": "RT jointPos_m", "dimension": "6", "unit": "rad", "frame_or_reference": "joint coordinates", "sign_semantics": "SDK joint convention; setup identity pending", "timestamp_source": "host RT receive perf_counter", "nominal_software_rate_hz": "from configured RT interval; not physical evidence", "latency_semantics": "device/source latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "kinematic input and validity", "blocker": "physical identity/rate/accuracy not validated"},
        {"channel_id": "ROBOT_DQ", "quantity": "joint velocity", "sdk_source": "host finite difference of RT q", "dimension": "6", "unit": "rad/s", "frame_or_reference": "joint coordinates", "sign_semantics": "inherits q convention", "timestamp_source": "host difference times", "nominal_software_rate_hz": "state update dependent", "latency_semantics": "differentiation delay/noise unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "candidate model input", "blocker": "not device-measured velocity; derivative policy unvalidated"},
        {"channel_id": "ROBOT_DDQ", "quantity": "derived joint/TCP acceleration", "sdk_source": "future numerical derivative; current typed TCP estimate only", "dimension": "6 joints or 3 TCP", "unit": "rad/s^2 or m/s^2", "frame_or_reference": "joint or base TCP", "sign_semantics": "inherits source", "timestamp_source": "host derivative times", "nominal_software_rate_hz": "not frozen", "latency_semantics": "filter/difference delay unknown", "code_path": "collection/state.py; collection/snapshot.py", **common, "future_role": "conditional dynamics input", "blocker": "derivative/filter semantics not frozen"},
        {"channel_id": "TCP_POSE", "quantity": "TCP position and Euler XYZ orientation", "sdk_source": "RT tcpPoseAbc_m", "dimension": "3+3", "unit": "m, rad", "frame_or_reference": "documented by project as base-frame TCP; setup validation pending", "sign_semantics": "SDK axis convention", "timestamp_source": "host RT receive perf_counter", "nominal_software_rate_hz": "configured RT interval; observed rate required", "latency_semantics": "device/source latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "task geometry, tracking and validity", "blocker": "physical frame/registration and timing unvalidated"},
        {"channel_id": "TCP_VELOCITY", "quantity": "TCP linear/angular velocity", "sdk_source": "host finite difference of RT pose", "dimension": "3+3", "unit": "m/s, rad/s", "frame_or_reference": "base expression candidate", "sign_semantics": "inherits pose axes", "timestamp_source": "host difference times", "nominal_software_rate_hz": "state update dependent", "latency_semantics": "difference delay unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "task tangent diagnostic only until validated", "blocker": "not device velocity; zero-speed tangent unstable"},
        {"channel_id": "TRACKING_ERROR", "quantity": "q/TCP reference-minus-measured error", "sdk_source": "derived from command and state ledgers", "dimension": "trajectory dependent", "unit": "rad and/or m", "frame_or_reference": "joint/base", "sign_semantics": "must be explicitly reference-minus-measured", "timestamp_source": "aligned master timebase", "nominal_software_rate_hz": "not frozen", "latency_semantics": "command-state alignment required", "code_path": "collection/episode_logger.py; future derivation", **common, "future_role": "validity/safety diagnostic, not objective", "blocker": "alignment and reviewed bounds absent"},
        {"channel_id": "CART_FORCE", "quantity": "controller-returned Cartesian force XYZ", "sdk_source": "forceControl().getEndTorque cart_force", "dimension": "3", "unit": "N", "frame_or_reference": "requested world/flange/tool expression; current request world", "sign_semantics": "not documented/physically verified", "timestamp_source": "host query start/end/midpoint; no device time", "nominal_software_rate_hz": "configured query target 50, not observed source cadence", "latency_semantics": "query duration bounded; source/transport latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> hardware/rokae_adapter.py", **common, "future_role": "primary endpoint source candidate", "blocker": WRENCH_STATUS},
        {"channel_id": "CART_TORQUE", "quantity": "controller-returned Cartesian torque XYZ", "sdk_source": "forceControl().getEndTorque cart_torque", "dimension": "3", "unit": "N*m", "frame_or_reference": "requested world/flange/tool; exact moment reference point unresolved", "sign_semantics": "not documented/physically verified", "timestamp_source": "same host query bounds as force", "nominal_software_rate_hz": "configured target 50; not validated", "latency_semantics": "source/transport latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> hardware/rokae_adapter.py", **common, "future_role": "secondary diagnostic only", "blocker": "rotation-only is not full moment transform when origin changes"},
        {"channel_id": "JOINT_TAU_MEASURED", "quantity": "joint measured torque", "sdk_source": "getEndTorque joint_torque_measured", "dimension": "6", "unit": "N*m", "frame_or_reference": "joint axes", "sign_semantics": "SDK joint convention not physically validated", "timestamp_source": "same host query bounds", "nominal_software_rate_hz": "configured target 50; not validated", "latency_semantics": "source/transport latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "secondary diagnostic/model input candidate", "blocker": "physical semantics, compensation and timing unvalidated"},
        {"channel_id": "JOINT_TAU_EXTERNAL", "quantity": "controller model-derived external joint torque", "sdk_source": "getEndTorque external_torque_measured", "dimension": "6", "unit": "N*m", "frame_or_reference": "joint axes", "sign_semantics": "SDK joint convention not physically validated", "timestamp_source": "same host query bounds", "nominal_software_rate_hz": "configured target 50; not validated", "latency_semantics": "source/transport latency unknown", "code_path": "hardware/windows/rokae_xcore.py -> collection/state.py", **common, "future_role": "secondary diagnostic/model input candidate", "blocker": "controller model/compensation semantics incomplete"},
        {"channel_id": "TACTILE_RAW", "quantity": "future pressure array", "sdk_source": "not implemented", "dimension": "sensor dependent", "unit": "raw counts; calibrated pressure later", "frame_or_reference": "sensor grid and strap placement", "sign_semantics": "nonnegative after validated calibration", "timestamp_source": "future host/device timestamp", "nominal_software_rate_hz": "unknown", "latency_semantics": "unknown", "code_path": "none", **common, "future_role": "placeholder/secondary diagnostics only", "blocker": "sensor acquisition and validation absent"},
    ]


def wrench_audit() -> str:
    return f"""# Wrench Semantics Audit

## Formal finding

`{WRENCH_STATUS}`

The locally installed xCoreSDK stub documents the exact call order
`getEndTorque(ref_type, joint_torque_measured, external_torque_measured, cart_torque, cart_force, ec)`.
It defines six joint measured torques in N*m, six controller-model-derived external joint torques in N*m, Cartesian torque XYZ in N*m, and Cartesian force XYZ in N. For this API it lists `world`, `flange`, and `tool`; the project currently requests `world`. The wider SDK enum also contains base/user-like frames, but this does not prove that `getEndTorque` accepts them.

## What is not established

- No documented physical sign convention: the code cannot say whether positive force is robot-on-patient or patient-on-robot.
- No documented controller compensation/bias state or tool/load dependence.
- The Chinese stub calls world/flange results "relative to" the frame and tool results relative to the TCP point. It does not unambiguously state the moment reference point for every option, nor whether a frame request changes axes only or also the point.
- No device/source timestamp, update cadence, source age, or synchronization contract with RT state.
- No formal validated physical result file was available to this stage. Existing diagnostics are read-only instruments, not evidence that the checks were performed and approved.

Therefore variable names and offline rotation unit tests cannot close the semantics gap. Cartesian force may only become a primary endpoint source after controlled frame/sign/bias/timing validation. Cartesian moment needs both rotation and reference-point translation semantics.

## `BASE_WRENCH_ROTATION_VERIFIED`

It remains `false`. To change it, a future independently approved, stationary/read-only validation must use known robot/world/tool orientations and known-direction applied loads, compare raw world/tool outputs, and verify the declared rotation convention in multiple non-degenerate orientations. It must check `F_b = R_b_from_w F_w`.

For moments, a separate known lever arm must test the full relation `tau_b = R tau_w + p x (R F_w)`. Rotation-only is not a point transform. If the primary endpoint ultimately consumes only Cartesian force, the moment-reference translation term does not enter the force projection; this permits excluding moment from that endpoint, but it does not resolve force sign, force expression axes, bias, contact line of action, or synchronization.
"""


def task_direction_audit() -> str:
    return f"""# Task Direction Definition Audit

## Decision

`{TASK_DIRECTION_STATUS}`

Direction must be selected from mechanics and a prespecified validation, never from whichever direction produces the smallest RMS.

| Candidate | Mechanical meaning | Posture/measured-state dependence | Sign/noise | Relation to strap/task | Decision |
|---|---|---|---|---|
| A instantaneous TCP tangent | component along instantaneous executed path | posture-aware; computable after validated TCP timing | undefined near zero speed and reversals; differentiation is noisy | motion direction is not necessarily strap tension direction | retain diagnostic candidate, not primary definition |
| B strap/pull line of action | force along the physical load-transfer line | needs registered attachment endpoints or a validated direct direction measurement | stable if attachment is taut; sign can be fixed physically | closest to actual strap mechanics | preferred physical semantics, but evidence missing |
| C endpoint-to-hip direction | straight line from traction point to registered hip | posture-dependent; hip registration and traction point required | potentially stable away from zero length | plausible only if it represents the real taut strap | validation candidate |
| D fixed bed-plane axis | fixed component in a reviewed bed frame | independent of posture and easy to compute | stable and noise-robust | can miss changing line of action | secondary diagnostic only |
| E 2-DOF equivalent traction direction | line from modeled hip to `L2` strap-equivalent traction point | computable from q and frozen geometry after patient/robot registration | stable if registration valid | consistent with formal lower-limb model but not a measured ankle | validation candidate, never relabel `L2` as ankle |

The defensible target is B: the measured/registered strap pull line of action, with C/E usable only if an experiment proves that their registered geometry represents B throughout the task. Until that evidence exists, no production `d_task(t)` is frozen.

Future rules: express force and the unit direction in the same validated frame; normalize after checking finite nonzero norm; predefine direction orientation (for example hip-to-traction-point) and independently establish force sign. Log the signed projection. RMS is sign-invariant mathematically, but signed projection and resistive/assistive decomposition remain diagnostics and must not be created by an arbitrary absolute value.
"""


def endpoint_payload() -> dict[str, Any]:
    return {
        "endpoint_id": ENDPOINT_ID,
        "decision_state": ENDPOINT_STATE,
        "validated": False,
        "ready_for_physical_validation": False,
        "blocking_dependencies": [WRENCH_STATUS, TASK_DIRECTION_STATUS, FILTER_STATUS, "BIAS_POLICY_REQUIRES_VALIDATION", "TIMING_THRESHOLDS_NOT_YET_FROZEN"],
        "provisional_mathematical_definition": {
            "force_projection": "F_task(t_i) = dot(F_interaction^R(t_i), d_task^R(t_i)); ||d_task^R||_2 = 1",
            "time_weighted_rms": "J_force = sqrt(sum_i w_i F_task(t_i)^2 / sum_i w_i) over valid mask M",
            "weight_definition": "w_i is valid represented time on the frozen master grid; exact quadrature/interpolation rule pending timing validation",
            "unit": "N",
            "sign": "signed F_task retained; RMS itself is sign-invariant; no arbitrary abs preprocessing",
        },
        "valid_sample_mask_requires": ["episode gate PASS", "finite synchronized state and force", "validated force frame/sign/bias semantics", "validated finite unit task direction", "sample age/skew/gap rules PASS", "not within an excluded interval if a transient policy is later frozen"],
        "episode_boundary_candidate": "one complete 24 s commanded trial after independently approved start acknowledgement through confirmed completion",
        "transient_exclusion": None,
        "transient_status": "NOT_YET_FROZEN_NO_EXCLUSION_SELECTED",
        "bias_rule": None,
        "bias_status": "BIAS_POLICY_REQUIRES_VALIDATION",
        "filter_rule": None,
        "filter_status": FILTER_STATUS,
        "missing_data_rule": "never zero-fill or silently impute; missing remains null with reason; endpoint unavailable if future frozen gate fails",
        "safety_metrics_in_objective": False,
        "secondary_diagnostics_not_weighted_into_primary": True,
    }


def bias_filter_delay_policy() -> str:
    return f"""# Bias, Filter and Delay Policy

## Bias

Raw force is always retained. Three research candidates are compared by validation design, not trajectory outcome: raw; pre-episode zero-subtracted; and model-based compensated. Pre-episode zero subtraction is the leading simple candidate only when an independently verified unloaded condition exists. A leg already attached under strap preload is not a zero-force condition. Static offset, pose-dependent tool/load/controller bias, temperature/session drift, and run-order drift must be measured across multiple relevant postures. Model-based compensation remains secondary until it predicts independent held-out static checks.

Status: `BIAS_POLICY_REQUIRES_VALIDATION`. No numeric bias or drift threshold is invented.

## Filtering

`{FILTER_STATUS}`. Raw samples and timestamps remain immutable. A filter may be frozen only after source-update cadence, alias content, query latency, robot motion bandwidth, and expected human-interaction bandwidth are measured. Its record must include type, cutoff, order, causal/noncausal, phase/group delay, initialization, and online/offline compatibility. A zero-phase offline filter cannot silently become a causal online filter. Filtering may not be chosen because it makes a trajectory look better.

## Delay

`MEASUREMENT_DELAY_VALIDATION_REQUIREMENT` distinguishes:

1. host query duration (`query_finished - query_started`), which current code records;
2. source update/transport latency and source age, currently unknown without device time or a controlled event;
3. RT-state/wrench alignment skew on the host master clock;
4. command-to-observation response delay, which requires a controlled validation.

No fixed delay from old simulation may be inherited as hardware delay. Future validation must report distributions, stationarity and pose/load dependence before freezing compensation or thresholds.
"""


def synchronization_payload() -> dict[str, Any]:
    return {
        "schema_id": "FUTURE_SYNCHRONIZED_EPISODE_TIMEBASE_V1",
        "master_timebase": MASTER_TIMEBASE,
        "master_timebase_unit": "integer nanoseconds; derived seconds permitted without changing origin",
        "wall_clock_role": "metadata/display only; never sample alignment",
        "robot_device_timestamp_available": False,
        "streams": {
            "robot_state": {"timestamps": ["host_receive_perf_counter_ns", "robot_device_time_if_future_available"], "current_device_time": None},
            "wrench": {"timestamps": ["host_query_start_perf_counter_ns", "host_query_end_perf_counter_ns", "host_query_midpoint_perf_counter_ns", "host_publish_perf_counter_ns"], "current_device_time": None},
            "trajectory_command": {"timestamps": ["host_intent_perf_counter_ns", "host_durable_publish_perf_counter_ns", "trajectory_elapsed_s"]},
            "tactile_placeholder": {"timestamps": ["host_receive_perf_counter_ns", "device_time_if_available"], "implemented": False},
        },
        "alignment": {
            "online": "latest causally available valid sample only; retain age and skew",
            "offline_state_interpolation": "only between two valid bracketing state samples and only within a future frozen maximum gap",
            "offline_wrench_interpolation": "not allowed beyond validated source interval/freshness; exact rule pending source-cadence validation",
            "fixed_delay_assumption": None,
        },
        "limits": {"maximum_state_age_s": None, "maximum_wrench_age_s": None, "maximum_state_wrench_skew_s": None, "maximum_interpolation_gap_s": None, "maximum_missing_fraction": None},
        "limit_status": "NOT_YET_FROZEN_REQUIRES_MEASUREMENT_AND_REVIEW",
        "missing_behavior": "null plus invalid_reason and mask; never zero-fill; gate episode when frozen limit is exceeded",
        "compatibility_note": "settings.py contains generic software defaults, but unreviewed safety/identification JSON means they are not frozen scientific endpoint thresholds",
    }


def future_episode_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "FUTURE_MEASUREMENT_DRIVEN_PERSONALIZATION_EPISODE_V1",
        "title": "Future 24 s measurement-driven personalization episode",
        "type": "object",
        "required": ["identity", "frozen_contract", "candidate", "streams", "timing", "diagnostics", "validity", "derived"],
        "properties": {
            "identity": {"type": "object", "required": ["subject_id", "session_id", "episode_id"], "properties": {"subject_id": {"type": "string"}, "session_id": {"type": "string"}, "episode_id": {"type": "string"}, "episode_order": {"type": "integer", "minimum": 1}}},
            "frozen_contract": {"type": "object", "properties": {"active_reference_id": {"const": "reference_measured_asymmetric_closed_slow"}, "active_reference_sha256": {"const": ACTIVE_REFERENCE_SHA256}, "rom_protocol": {"const": "ROM_PROTOCOL_V2"}, "theta_shank": {"const": "q_hip - q_knee"}, "planned_duration_s": {"const": 24.0}}},
            "candidate": {"type": "object", "required": ["candidate_id", "beta_flex", "beta_extend"], "properties": {"candidate_id": {"type": "string"}, "beta_flex": {"type": "number"}, "beta_extend": {"type": "number"}}},
            "streams": {"type": "object", "properties": {"reference": {"description": "q_ref, dq_ref, optional ddq_ref with master timestamps"}, "robot_state": {"description": "q, derived dq, TCP, source timestamps and validity"}, "wrench": {"description": "raw force/torque, joint torque channels, raw frame, query bounds and validity"}, "trajectory_command": {"description": "command intent/publish timestamps and reference identity"}, "tactile": {"description": "future nullable placeholder; raw matrix and calibration provenance"}}},
            "timing": {"type": "object", "properties": {"master_timebase": {"const": MASTER_TIMEBASE}, "ages_ms": {"type": "array"}, "skews_ms": {"type": "array"}, "missing_intervals": {"type": "array"}, "validity_flags": {"type": "array"}}},
            "diagnostics": {"type": "object", "properties": {"tracking_error": {"type": ["object", "null"]}, "safety_events": {"type": "array"}, "operation_state": {"type": "array"}, "dropped_samples": {"type": "object"}, "abort_reason": {"type": ["string", "null"]}}},
            "validity": {"type": "object", "required": ["gate_version", "pass", "reasons"], "properties": {"gate_version": {"const": "EPISODE_DATA_VALIDITY_GATE_V1"}, "pass": {"type": "boolean"}, "reasons": {"type": "array", "items": {"type": "string"}}}},
            "derived": {"type": "object", "properties": {"task_direction_unit_vector": {"type": ["array", "null"]}, "projected_force_n": {"type": ["array", "null"]}, "primary_endpoint_n": {"type": ["number", "null"]}, "endpoint_status": {"type": "string"}, "secondary_diagnostics": {"type": "object"}}},
        },
        "forbidden_oracle_fields": ["future_episode_outcome", "unexecuted_candidate_truth", "MyoLeg_oracle", "future_beta_selection", "held_out_final_evaluation"],
    }


def validity_gate() -> dict[str, Any]:
    checks = [
        ("FULL_DURATION_COMPLETED", "confirmed complete 24 s trial and normal completion", None),
        ("NO_SAFETY_ABORT", "no safety abort/event requiring invalidation", None),
        ("VALID_OPERATION_STATE", "all required operation-state samples valid", None),
        ("TIMING_VALID", "timestamps monotonic and provenance complete", None),
        ("MISSING_DATA_ACCEPTABLE", "missing/dropped intervals below future frozen limit", None),
        ("STATE_FRESHNESS_VALID", "state age below future frozen limit", None),
        ("WRENCH_FRESHNESS_VALID", "wrench age/source validity below future frozen limit", None),
        ("SYNCHRONIZATION_VALID", "state-wrench-command skew/gap below future frozen limits", None),
        ("WRENCH_SEMANTICS_VALIDATED", "frame/sign/bias semantics valid for endpoint version", False),
        ("TASK_DIRECTION_VALIDATED", "finite unit direction in same validated frame", False),
        ("TRACKING_WITHIN_REVIEWED_BOUNDS", "tracking within independently reviewed constraint", None),
        ("FINITE_REQUIRED_CHANNELS", "all samples used by endpoint finite", None),
    ]
    return {
        "gate_id": "EPISODE_DATA_VALIDITY_GATE_V1",
        "current_status": "DEFINITION_READY_THRESHOLDS_AND_PHYSICAL_SEMANTICS_PENDING",
        "fail_closed": True,
        "checks": [{"check_id": item, "meaning": meaning, "current_pass": current} for item, meaning, current in checks],
        "numeric_thresholds": {"missing_fraction": None, "max_state_age_s": None, "max_wrench_age_s": None, "max_skew_s": None, "max_interpolation_gap_s": None, "tracking_bounds": None},
        "on_failure": "do not compute/use a normal-looking primary endpoint; store null endpoint, gate failure and reasons; do not update gray-box or BO objective",
        "safety_or_validity_values_are_objective_terms": False,
    }


def repeatability_plan() -> str:
    return """# Endpoint Repeatability Validation Plan

This is a future protocol design, not an experiment. Use one independently approved safe candidate and identical frozen setup/attachment/command conditions. The repeat count is `N_REPEATS_NOT_YET_FROZEN`; choose it prospectively from precision/reliability requirements, not from observed trajectory ranking.

Before collection freeze episode definition, wrench/task-direction semantics, bias/filter policy, sampling/synchronization gates, setup factors and run-order strategy. Report valid repeat count, mean, SD, CV where the mean makes CV interpretable, within-session drift, between-block/session bias, and ICC (with the exact ICC model) only when the design supports it. Plot time profiles and residuals without changing preprocessing after viewing results.

`ENDPOINT_REPEATABILITY_GATE` thresholds remain null until justified by measurement requirements. Fail or indeterminate repeatability blocks personalization use; it must not be repaired by removing inconvenient repeats after outcome inspection.
"""


def sensitivity_plan() -> str:
    return """# Endpoint Sensitivity Validation Plan

Goal: determine whether the endpoint separates prespecified small trajectory perturbations from repeatability noise. It is independent of robot approval and is not executed here.

After repeatability passes, preregister a small set of V3 `beta_flex/beta_extend` perturbations from the unchanged family. Select them from geometry/domain and approved exposure considerations, never from force outcomes or MyoLeg oracle ranking. Counterbalance order and repeat the reference/perturbations under the same setup. Estimate paired endpoint differences with uncertainty and compare their scale with within-candidate repeatability noise. Retain tracking and full force profiles as diagnostics.

No numeric effect threshold, sample size or robot-safe beta subset is frozen here. Failure to distinguish the prespecified perturbations means `ENDPOINT_NOT_SENSITIVE_ENOUGH_FOR_PERSONALIZATION`; it does not justify tuning preprocessing on the same data.
"""


def secondary_diagnostics() -> dict[str, Any]:
    return {
        "policy": "SECONDARY_DIAGNOSTICS_NOT_AUTOMATIC_PRIMARY_OBJECTIVE_TERMS",
        "diagnostics": [
            {"id": "PEAK_ABS_TASK_FORCE", "unit": "N", "status": "depends on validated task force", "role": "secondary mechanics"},
            {"id": "SIGNED_TASK_FORCE_PROFILE", "unit": "N", "status": "depends on sign validation", "role": "assistive/resistive diagnostic"},
            {"id": "FULL_FORCE_NORM", "unit": "N", "status": "depends on validated common frame/bias", "role": "secondary mechanics"},
            {"id": "CARTESIAN_TORQUE_RMS", "unit": "N*m", "status": "reference-point semantics unresolved", "role": "secondary only"},
            {"id": "JOINT_MEASURED_EXTERNAL_TORQUE_RMS", "unit": "N*m", "status": "controller semantics require validation", "role": "secondary only"},
            {"id": "TRACKING_ERROR", "unit": "rad/m", "status": "bounds and alignment pending", "role": "validity/safety diagnostic"},
            {"id": "MECHANICAL_POWER_OR_WORK", "unit": "W/J", "status": "sign and synchronized velocity required", "role": "exploratory secondary"},
            {"id": "TACTILE_FEATURES", "unit": "sensor dependent", "status": "not implemented/validated", "role": "future secondary"},
        ],
        "safety_thresholds_are_constraints": True,
        "weighted_composite_objective_created": False,
    }


def tactile_document() -> str:
    return """# Tactile Future Interface and Validation

Tactile is a nullable synchronized episode stream, not a current measurement channel and not a primary endpoint. Each frame must carry host monotonic receive time, optional device time, sensor/frame ID, raw matrix and raw unit, calibration ID, calibrated pressure matrix/unit when valid, per-cell validity/missing mask, saturation mask, sensor validity and invalid reason.

Candidate secondary features after independent validation are mean/peak pressure, spatial concentration, center of pressure, active pressure area and temporal stability. They are not comfort labels.

Minimum future evidence: calibration curve and units; zero/bias and drift; repeatability; spatial consistency/orientation; saturation; effective sampling/source-update rate; latency and timestamp provenance; synchronization with robot streams; missing-cell behavior; and strap-placement repeatability. Until these pass, values remain raw/invalid and features remain null. `PRESSURE_IS_NOT_COMFORT_TRUTH`.
"""


def causal_policy() -> dict[str, Any]:
    return {
        "policy_id": "REAL_PERSONALIZATION_CAUSAL_DATA_POLICY_V1",
        "decision_rule": "when selecting beta_k, use frozen priors plus valid completed episodes 1..k-1 only",
        "episode_available_after": "validity gate PASS and endpoint versioned computation completes",
        "invalid_episode": "record exposure and failure, but do not add a normal objective observation or fit target",
        "allowed_model_inputs": ["past valid measured state/interaction time series", "past valid episode endpoints", "past executed beta", "frozen reference/task/V3 descriptors"],
        "forbidden": ["current/future episode outcome", "unexecuted candidate truth", "offline MyoLeg oracle", "held-out final evaluation", "future participant data", "post-hoc endpoint variant chosen by ranking"],
        "final_evaluation": "sealed from adaptation; cannot update the selected beta or adaptation model",
        "bo_observation_interface": {"candidate": ["beta_flex", "beta_extend"], "value": "validated episode mechanical endpoint", "quality": ["gate_version", "endpoint_version", "uncertainty_or_quality_flag"], "invalid_value": None},
        "gray_box_interface": {"inputs": ["trajectory/state descriptors", "validated measured episode time series/features"], "outputs": ["mechanical response prediction", "prediction uncertainty/quality"], "endpoint_relationship": "time-series prediction may be projected/aggregated by the separately frozen endpoint; endpoint-only model is a simpler comparator", "five_parameter_model_modified": False},
    }


def readiness_graph() -> str:
    return f"""# Readiness Dependency Graph

```mermaid
flowchart LR
  M[Measurement semantics] --> F[Frame, sign, point and bias validation]
  F --> T[Task-direction validation]
  T --> S[Synchronization and delay validation]
  S --> R[Repeated identical-trial repeatability]
  R --> E[Endpoint sensitivity and validation]
  E --> D[Repeated safe measured episodes]
  D --> G[Gray-box identification]
  G --> X[Residual analysis]
  X --> P{{PINN stop/go gate}}
  E --> B{{BO stop/go gate}}
  G --> B
```

Current stop is before frame/task-direction validation. The exact next stage is `{NEXT_STAGE}`. It must resolve physical wrench frame/sign/reference-point semantics and validate a task line of action before endpoint repeatability, identification, PINN or BO. This stage does not execute it.
"""


def report() -> str:
    return f"""# Measurement-Driven Personalization Data and Endpoint Design V1

## Formal outcome

`{ENDPOINT_STATE}`

`{WRENCH_STATUS}`  
`{TASK_DIRECTION_STATUS}`  
`BASE_WRENCH_ROTATION_VERIFIED = false`  
`{FILTER_STATUS}`

This was a source/document design audit only: zero robot connection, zero motion, zero human collection, zero model/PINN training and zero BO.

## Q1. Channels sufficiently defined for future research use

The software schema and provenance paths for q, host-derived dq, TCP, raw wrench/joint-torque arrays and timing are sufficiently specified to plan future validation. None is currently a formally validated physical research outcome. q/TCP are future state/context candidates after setup validation; dq/ddq are derived candidates. Cartesian force is blocked as primary endpoint input. Joint/cartesian torque remain secondary candidates. Tactile is only a placeholder.

## Q2. Exact current wrench semantics

The local SDK defines 3 Cartesian force values in N, 3 Cartesian torque values in N*m, 6 measured joint torques and 6 controller-model-derived external joint torques in N*m. `getEndTorque` accepts documented world/flange/tool requests and current code requests world. Sign, compensation/bias, exact moment point for all frames, source timestamp/cadence and synchronization are not proved.

## Q3. Base rotation

Not justified. Offline math verifies only a convention, not the physical SDK convention. Known-direction, multiple-orientation force tests and known-lever-arm moment tests are required. Force projection can omit moment translation only if moment is excluded; all force semantics still need validation.

## Q4. Task direction

The physical strap/pull line of action is the most defensible. A registered equivalent traction-point-to-hip line can approximate it only after experimental geometry validation. TCP tangent and fixed bed axes remain diagnostics. No direction is selected from lower RMS.

## Q5. Mathematical candidate

For valid samples in a common validated frame, `F_task(t_i)=dot(F_interaction(t_i), d_task(t_i))`, with unit `d_task`; then `J_force=sqrt(sum(w_i F_task(t_i)^2)/sum(w_i))`, in N. Signed projection is retained, while RMS is naturally sign-invariant. Exact mask, quadrature, transient, bias and filter remain unfinished dependencies.

## Q6. Bias/filter/delay/synchronization

Always retain raw data. Validate an unloaded pre-episode zero candidate against static, pose-dependent and drift behavior before selection; do not call strap preload zero. No filter cutoff is frozen. Host `perf_counter_ns` is the master clock; retain all query bounds/ages/skews, never zero-fill, and do not reuse simulation delay. Numeric age/gap/skew/missing limits remain null pending real evidence and review.

## Q7. Episode validity

A 24 s episode must complete normally, have no invalidating safety event, valid operation/tracking, finite wrench/state, monotonic timing, and pass future frozen freshness/skew/gap/missing rules plus validated wrench/task-direction semantics. Failure produces a null endpoint and no gray-box/BO observation.

## Q8. Future evidence

Preregister repeated identical approved trials to estimate mean/SD/CV, drift and design-appropriate ICC, then test prespecified small V3 perturbations against repeatability noise. Repeat count and thresholds are not invented here.

## Q9. Tactile integration

Use a nullable timestamped raw/calibrated matrix stream with per-cell missing/saturation masks, calibration and placement provenance. Pressure features remain secondary and pressure is not comfort.

## Q10. Single next stage

`{NEXT_STAGE}`

Definition is not ready for endpoint validation until frame/sign/point and physical task direction are resolved. Status remains `NOT_HUMAN_READY / NOT_ROBOT_APPROVED`; the next stage was not executed.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files) + "\n")


def execute() -> None:
    protocol = verify_freeze()
    atomic_csv(OUTPUT / "REAL_MEASUREMENT_CHANNEL_INVENTORY.csv", measurement_rows())
    atomic_text(OUTPUT / "WRENCH_SEMANTICS_AUDIT.md", wrench_audit())
    atomic_text(OUTPUT / "TASK_DIRECTION_DEFINITION_AUDIT.md", task_direction_audit())
    atomic_json(OUTPUT / "PRIMARY_MECHANICAL_ENDPOINT_DEFINITION.json", endpoint_payload())
    atomic_text(OUTPUT / "BIAS_FILTER_DELAY_POLICY.md", bias_filter_delay_policy())
    atomic_json(OUTPUT / "SYNCHRONIZATION_SCHEMA.json", synchronization_payload())
    atomic_json(OUTPUT / "FUTURE_EPISODE_DATA_SCHEMA.json", future_episode_schema())
    atomic_json(OUTPUT / "EPISODE_VALIDITY_GATE.json", validity_gate())
    atomic_text(OUTPUT / "ENDPOINT_REPEATABILITY_VALIDATION_PLAN.md", repeatability_plan())
    atomic_text(OUTPUT / "ENDPOINT_SENSITIVITY_VALIDATION_PLAN.md", sensitivity_plan())
    atomic_json(OUTPUT / "SECONDARY_MECHANICAL_DIAGNOSTICS.json", secondary_diagnostics())
    atomic_text(OUTPUT / "TACTILE_FUTURE_INTERFACE_AND_VALIDATION.md", tactile_document())
    atomic_json(OUTPUT / "REAL_PERSONALIZATION_CAUSAL_DATA_POLICY_V1.json", causal_policy())
    atomic_text(OUTPUT / "READINESS_DEPENDENCY_GRAPH.md", readiness_graph())
    atomic_text(OUTPUT / "MEASUREMENT_DRIVEN_PERSONALIZATION_DATA_AND_ENDPOINT_DESIGN_REPORT.md", report())
    atomic_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID,
        "formal_status": ENDPOINT_STATE,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "input_count": len(protocol["input_files"]),
        "wrench_semantics_status": WRENCH_STATUS,
        "task_direction_status": TASK_DIRECTION_STATUS,
        "base_wrench_rotation_verified": False,
        "filter_status": FILTER_STATUS,
        "primary_endpoint_id": ENDPOINT_ID,
        "primary_endpoint_validated": False,
        "primary_endpoint_ready_for_validation": False,
        "master_timebase": MASTER_TIMEBASE,
        "numeric_endpoint_thresholds_frozen": False,
        "hardware_collection_control_safety_modified": False,
        "robot_access_count": 0,
        "motion_command_count": 0,
        "human_data_access_count": 0,
        "pinn_training_count": 0,
        "bo_run_count": 0,
        "myoleg_run_count": 0,
        "v3_modified": False,
        "frozen_artifacts_modified": False,
        "not_human_ready": True,
        "not_robot_approved": True,
        "next_stage": NEXT_STAGE,
        "next_stage_executed": False,
    })
    write_checksums()
    print(json.dumps({"stage_id": STAGE_ID, "formal_status": ENDPOINT_STATE, "next_stage": NEXT_STAGE}, indent=2))


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
