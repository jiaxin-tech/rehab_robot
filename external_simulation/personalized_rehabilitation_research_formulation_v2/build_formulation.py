"""Freeze the measurement-driven personalization research formulation.

The builder is evidence- and source-inspection only.  It does not import or
execute simulator, control, acquisition, learner, optimizer, robot, or human
study code.  ``--prepare`` freezes inputs and the formulation protocol;
``--execute`` verifies that freeze before generating the formal documents.
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


STAGE_ID = "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2"
FORMAL_STATUS = "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_READY_WITH_LIMITATIONS"
PROTOCOL_ID = "RESEARCH_FORMULATION_V2_PROTOCOL"
PRIMARY_DIRECTION = "PIVOT_TO_MEASUREMENT_DRIVEN_PERSONALIZATION"
PRIMARY_THESIS = "MECHANICAL_MEASUREMENT_DRIVEN_PERSONALIZATION"
OPTIONAL_EXTENSION = "PREFERENCE_AWARE_PERSONALIZATION_WITH_DIRECT_HUMAN_FEEDBACK"
PRIMARY_PERSONALIZATION_SOURCE = "SUBJECT_SPECIFIC_MEASURED_INTERACTION_TRIALS"
PRIMARY_OUTCOME_TYPE = "MEASURED_MECHANICAL_INTERACTION_ENDPOINT_PENDING_INDEPENDENT_CALIBRATION"
PRIMARY_CANDIDATE_OUTCOME = "EPISODE_RMS_VALIDATED_TASK_DIRECTION_INTERACTION_FORCE"
TRAJECTORY_PARAMETERIZATION = "P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3"
PINN_ROLE = "CONDITIONAL_SUBJECT_SPECIFIC_MECHANICAL_RESIDUAL_MODEL_NOT_YET_JUSTIFIED"
BO_ROLE = "CONDITIONAL_SELECTOR_OVER_OBSERVABLE_SUBJECT_SPECIFIC_OUTCOME_NOT_YET_JUSTIFIED"
MYOLEG_ROLE = "OFFLINE_PHYSICS_PRIOR_STRESS_TEST_AND_FEASIBILITY_SUPPORT_ONLY"
TACTILE_ROLE = "PLANNED_MEASURED_INTERACTION_FEATURE_POSSIBLE_COMFORT_CORRELATE_NOT_COMFORT_TRUTH"
HUMAN_FEEDBACK_REQUIREMENT = "REQUIRED_ONLY_FOR_COMFORT_OR_PREFERENCE_CLAIMS"
NEXT_STAGE = "MEASUREMENT_DRIVEN_PERSONALIZATION_DATA_AND_ENDPOINT_DESIGN_V1"
PRIMARY_ADAPTATION_BUDGET = 4
SENSITIVITY_BUDGETS = (3, 5)
ACTIVE_REFERENCE_SHA256 = "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
STOP_OR_PIVOT_PROTOCOL_SHA256 = "029ffe5bcc91ca4ffc8d9db3216b7df0c92c73f0aa253a26a2d2d9b24f173503"
FROZEN_PROTOCOL_SHA256 = "41da6efac092da267a0e8477ff8453fe4790b5adc0e2bfd23d1ab7671cc58a45"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/personalized_rehabilitation_research_formulation_v2"
PROTOCOL_PATH = OUTPUT / "RESEARCH_FORMULATION_V2_PROTOCOL.json"


INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "STOP_OR_PIVOT_DECISION",
        "path": "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1/FINAL_PROJECT_DIRECTION_DECISION.json",
        "markers": ("PIVOT_TO_MEASUREMENT_DRIVEN_PERSONALIZATION", "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2"),
    },
    {
        "id": "STOP_OR_PIVOT_REPORT",
        "path": "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1/MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_REPORT.md",
        "markers": ("PINN_NOT_YET_JUSTIFIED", "PERSONALIZED_BO_NOT_YET_JUSTIFIED_WITHOUT_SUBJECT_FEEDBACK"),
    },
    {
        "id": "STOP_OR_PIVOT_PROTOCOL",
        "path": "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1/STOP_OR_PIVOT_DECISION_PROTOCOL.json",
        "markers": ("SCIENTIFIC_FORMULATION_PROJECT_DECISION_AUDIT", "generate_Cohort_V2"),
        "exact_sha256": STOP_OR_PIVOT_PROTOCOL_SHA256,
    },
    {
        "id": "V3_PARAMETERIZATION_SEMANTICS",
        "path": "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_PARAMETERIZATION_SEMANTICS.json",
        "markers": ("beta_flex", "beta_extend", "w_b(s; beta_b)=s+beta_b*64*s^3*(1-s)^3"),
    },
    {
        "id": "V3_CANDIDATE_MANIFEST",
        "path": "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json",
        "markers": ("formal_parent_reference_sha256", "beta_flex", "beta_extend"),
    },
    {
        "id": "V3_PARAMETERIZATION_SOURCE",
        "path": "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py",
        "markers": ("PARAMETER_ORDER = (\"beta_flex\", \"beta_extend\")", "64.0"),
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
        "id": "ROBOT_STATE_SCHEMA",
        "path": "collection/state.py",
        "markers": ("joint_position_rad", "joint_velocity_radps", "tcp_linear_acceleration_est_mps2", "InternalWrenchFrame"),
    },
    {
        "id": "SNAPSHOT_ALIGNMENT",
        "path": "collection/snapshot.py",
        "markers": ("state_internal_skew_ms", "base_wrench_rotation_requires_robot_validation", "force_query_duration_ms"),
    },
    {
        "id": "REAL_ROBOT_ACQUISITION",
        "path": "collection/real_robot_acquisition.py",
        "markers": ("Independent state, wrench, and alignment producers", "read_internal_wrench", "state_wrench_skew_s"),
    },
    {
        "id": "ROKAE_ADAPTER",
        "path": "hardware/rokae_adapter.py",
        "markers": ("Observation-only project adapter", "read_joint_positions", "read_internal_wrench"),
    },
    {
        "id": "XCORE_STATE_WRENCH_SOURCE",
        "path": "hardware/windows/rokae_xcore.py",
        "markers": ("getEndTorque", "joint_position_rad", "joint_velocity_radps"),
    },
    {
        "id": "HARDWARE_SETTINGS",
        "path": "config/settings.py",
        "markers": ("BASE_WRENCH_ROTATION_VERIFIED  = False", "ROBOT_FORCE_SOURCE", "REQUIRE_WORKSPACE_LIMITS = True"),
    },
    {
        "id": "EXPERIMENT_SAFETY_REVIEW",
        "path": "config/experiment_safety.json",
        "markers": ('"reviewed": false', '"max_force_n": null', '"workspace_min_base_m": null'),
    },
    {
        "id": "REAL_IDENTIFICATION_CONFIG",
        "path": "config/real_identification_config.json",
        "markers": ('"reviewed": false', '"raw_wrench_frame": null', '"assumed_wrench_delay_s": null'),
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
    columns = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
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
            raise RuntimeError(f"missing frozen input: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"input semantic marker mismatch {spec['id']}: {missing}")
        digest = sha256_file(path)
        expected = spec.get("exact_sha256")
        if expected is not None and digest != expected:
            raise RuntimeError(f"exact SHA mismatch {spec['id']}: {digest}")
        rows.append(
            {
                "input_id": spec["id"],
                "path": spec["path"],
                "sha256": digest,
                "semantic_markers": list(spec["markers"]),
                "semantic_markers_pass": True,
            }
        )

    stop = read_json(ROOT / INPUT_SPECS[0]["path"])
    if not (
        stop["primary_recommendation"] == PRIMARY_DIRECTION
        and stop["option_a_continue_synthetic_myoleg_personalization"] == "STOP"
        and stop["option_c_measurement_driven_personalization"] == "PRIMARY_RECOMMENDATION"
        and stop["next_stage_executed"] is False
    ):
        raise RuntimeError("stop-or-pivot decision semantics changed")
    formal = read_json(ROOT / "config/formal_experiment_manifest.json")
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    ):
        raise RuntimeError("formal reference/ROM/angle semantics changed")
    safety = read_json(ROOT / "config/experiment_safety.json")
    identification = read_json(ROOT / "config/real_identification_config.json")
    if safety["reviewed"] is not False or identification["reviewed"] is not False:
        raise RuntimeError("unexpected hardware or identification readiness change")
    return rows


def active_channel_scan() -> dict[str, Any]:
    roots = (ROOT / "hardware", ROOT / "collection", ROOT / "control")
    files = sorted(path for root in roots for path in root.rglob("*.py") if path.is_file())
    tactile_hits: list[str] = []
    preference_hits: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        if "tactile" in lowered or "pressure map" in lowered or "pressure array" in lowered:
            tactile_hits.append(str(path.relative_to(ROOT)))
        if "pairwise preference" in lowered or "subject rating" in lowered:
            preference_hits.append(str(path.relative_to(ROOT)))
    inventory = "\n".join(f"{path.relative_to(ROOT)}:{sha256_file(path)}" for path in files)
    return {
        "active_source_file_count": len(files),
        "active_source_inventory_sha256": hashlib.sha256(inventory.encode("utf-8")).hexdigest(),
        "tactile_acquisition_source_hits": tactile_hits,
        "direct_preference_acquisition_source_hits": preference_hits,
        "tactile_implemented": bool(tactile_hits),
        "direct_preference_implemented": bool(preference_hits),
    }


def protocol_payload(inputs: list[dict[str, Any]], scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "formal_status_options": [
            "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_READY",
            FORMAL_STATUS,
            "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_NOT_READY",
        ],
        "inherited_primary_direction": PRIMARY_DIRECTION,
        "primary_thesis_formulation": PRIMARY_THESIS,
        "optional_extension": OPTIONAL_EXTENSION,
        "primary_personalization_source": PRIMARY_PERSONALIZATION_SOURCE,
        "primary_outcome_type": PRIMARY_OUTCOME_TYPE,
        "primary_candidate_outcome": PRIMARY_CANDIDATE_OUTCOME,
        "candidate_outcome_is_not_final_calibrated_endpoint": True,
        "trajectory_parameterization": {
            "id": TRAJECTORY_PARAMETERIZATION,
            "parameters": ["beta_flex", "beta_extend"],
            "mathematical_definition": "w_b(s;beta_b)=s+beta_b*64*s^3*(1-s)^3; q_hip=q_hip_ref; q_knee=q_knee_ref_branch(w_b)",
            "cold_start": {"beta_flex": 0.0, "beta_extend": 0.0},
            "offline_bounds_are_robot_approved": False,
            "v4_redesign_allowed": False,
        },
        "trial_budget_hypothesis": {
            "primary_complete_adaptation_trials": PRIMARY_ADAPTATION_BUDGET,
            "sensitivity_complete_adaptation_trials": list(SENSITIVITY_BUDGETS),
            "final_evaluation_is_separate": True,
            "hardware_approved": False,
        },
        "validation_principles": [
            "equal adaptation trial budget",
            "same causal information boundary",
            "adaptation trials separated from final evaluation",
            "no method receives future or held-out outcomes",
            "generalization unit is a new real subject only after independent approvals",
        ],
        "hardware_boundary": "ALGORITHM_FORMULATION_READY != ROBOT_EXECUTION_READY",
        "current_readiness": {"human_ready": False, "robot_approved": False},
        "input_files": inputs,
        "active_channel_source_scan": scan,
        "forbidden_operations": [
            "generate_Cohort_V2", "expand_S2_or_S3", "increase_structural_z",
            "search_new_synthetic_oracles", "redesign_V4_or_V5_for_oracle_diversity",
            "search_objective_weights_for_oracle_diversity", "run_PINN", "run_BO",
            "execute_hardware", "execute_human_study", "read_held_out_scientific_truth",
            "modify_frozen_artifacts", "execute_next_stage",
        ],
        "protocol_frozen_before_formulation_outputs_written": True,
    }


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"prepare requires an empty output directory: {OUTPUT}")
    inputs = verify_inputs()
    scan = active_channel_scan()
    atomic_json(PROTOCOL_PATH, protocol_payload(inputs, scan))
    atomic_json(
        OUTPUT / "INPUT_VERIFICATION.json",
        {
            "stage_id": STAGE_ID,
            "input_count": len(inputs),
            "all_inputs_present_and_semantically_verified": True,
            "inputs": inputs,
            "active_channel_source_scan": scan,
            "held_out_scientific_access_count": 0,
            "simulator_experiment_count": 0,
            "robot_access_count": 0,
            "human_study_count": 0,
        },
    )
    atomic_json(
        OUTPUT / "HARDWARE_READINESS_BOUNDARY.json",
        {
            "algorithm_formulation_ready_does_not_equal_robot_execution_ready": True,
            "algorithm_status": "FORMULATION_ONLY",
            "human_ready": False,
            "robot_approved": False,
            "experiment_safety_reviewed": False,
            "real_identification_config_reviewed": False,
            "base_wrench_rotation_verified": False,
            "offline_v3_bounds_robot_approved": False,
            "hardware_or_human_action_performed": False,
        },
    )
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "input_count": len(inputs)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen")
    if sha256_file(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("formulation protocol SHA mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["input_files"] != verify_inputs():
        raise RuntimeError("frozen formulation inputs changed")
    if protocol["active_channel_source_scan"] != active_channel_scan():
        raise RuntimeError("active channel source inventory changed")
    return protocol


def measurement_rows() -> list[dict[str, Any]]:
    common = {"available_now_for_formal_research": False, "requires_validation": True}
    return [
        {"channel_id": "R_STATE_Q", "channel": "robot joint position", "software_status": "CODE_PATH_PRESENT_FAIL_CLOSED", **common, "planned": False, "model_input_role": "PRIMARY_KINEMATIC_INPUT_CANDIDATE", "optimization_target_role": "NO", "context_or_safety_role": "STATE_VALIDITY_AND_TRACKING", "evidence_path": "collection/state.py;hardware/rokae_adapter.py", "limitation": "requires actual robot identity, timing and accuracy validation"},
        {"channel_id": "R_STATE_DQ", "channel": "robot joint velocity", "software_status": "HOST_DIFFERENCE_ESTIMATE_PATH_PRESENT", **common, "planned": False, "model_input_role": "KINEMATIC_INPUT_CANDIDATE", "optimization_target_role": "NO", "context_or_safety_role": "MOTION_QUALITY", "evidence_path": "hardware/windows/rokae_xcore.py;collection/state.py", "limitation": "not an SDK device velocity; differentiation timing/noise need validation"},
        {"channel_id": "R_STATE_DDQ", "channel": "derived acceleration", "software_status": "DERIVED_SCHEMA_ONLY_NOT_ENDPOINT_READY", **common, "planned": True, "model_input_role": "CONDITIONAL_DYNAMICS_INPUT", "optimization_target_role": "NO", "context_or_safety_role": "MOTION_QUALITY", "evidence_path": "collection/state.py", "limitation": "derivative method, filtering, delay and repeatability not frozen"},
        {"channel_id": "R_TCP", "channel": "TCP pose/state", "software_status": "CODE_PATH_PRESENT_FAIL_CLOSED", **common, "planned": False, "model_input_role": "CONTEXT_AND_GEOMETRY", "optimization_target_role": "NO", "context_or_safety_role": "PRIMARY_CONTEXT_AND_SAFETY", "evidence_path": "hardware/rokae_adapter.py;collection/state.py", "limitation": "base-frame/state cadence need setup-specific verification"},
        {"channel_id": "R_TRACK", "channel": "trajectory tracking error", "software_status": "DERIVABLE_NOT_FROZEN_AS_EPISODE_FEATURE", **common, "planned": True, "model_input_role": "QUALITY_COVARIATE", "optimization_target_role": "SECONDARY_PENALTY_CANDIDATE_ONLY", "context_or_safety_role": "EXECUTION_QUALITY_GATE", "evidence_path": "collection/state.py;control/robot_trajectory_executor.py", "limitation": "command-observation alignment and acceptable tracking criteria not frozen"},
        {"channel_id": "R_WRENCH", "channel": "internal estimated Cartesian wrench", "software_status": "GETENDTORQUE_PATH_PRESENT_UNVALIDATED", **common, "planned": False, "model_input_role": "PRIMARY_MECHANICAL_INPUT_CANDIDATE", "optimization_target_role": "PRIMARY_ENDPOINT_SOURCE_CANDIDATE", "context_or_safety_role": "MECHANICAL_SAFETY_DIAGNOSTIC", "evidence_path": "hardware/rokae_adapter.py;collection/snapshot.py", "limitation": "frame rotation, reference point, bias, sign, delay, timing and physical validity require real-setup validation"},
        {"channel_id": "R_JOINT_TAU", "channel": "joint measured/external torque estimates", "software_status": "TYPED_SDK_RESULT_PATH_PRESENT_UNVALIDATED", **common, "planned": False, "model_input_role": "MECHANICAL_INPUT_CANDIDATE", "optimization_target_role": "SECONDARY_ENDPOINT_CANDIDATE", "context_or_safety_role": "SAFETY_DIAGNOSTIC", "evidence_path": "collection/state.py;hardware/windows/rokae_xcore.py", "limitation": "availability and controller semantics require setup-specific confirmation"},
        {"channel_id": "T_RAW", "channel": "tactile raw pressure map", "software_status": "NOT_IMPLEMENTED", **common, "planned": True, "model_input_role": "PLANNED_INTERACTION_INPUT", "optimization_target_role": "NO_UNTIL_CALIBRATED", "context_or_safety_role": "PLANNED_CONTACT_DIAGNOSTIC", "evidence_path": "none in active hardware/collection/control source", "limitation": "sensor, calibration, acquisition, timing and spatial mapping absent"},
        {"channel_id": "T_FEATURES", "channel": "peak/mean/concentration/centroid/temporal pressure features", "software_status": "NOT_IMPLEMENTED_DEPENDS_ON_T_RAW", **common, "planned": True, "model_input_role": "PLANNED_EPISODE_FEATURES", "optimization_target_role": "SECONDARY_OR_FUTURE_ENDPOINT_CANDIDATES", "context_or_safety_role": "CONTACT_DISTRIBUTION_DIAGNOSTIC", "evidence_path": "none", "limitation": "feature semantics and repeatability cannot precede raw-sensor validation"},
        {"channel_id": "H_RATING", "channel": "direct scalar subject rating", "software_status": "NOT_IMPLEMENTED_OPTIONAL_EXTENSION", **common, "planned": False, "model_input_role": "PREFERENCE_LABEL_IF_APPROVED", "optimization_target_role": "PREFERENCE_BRANCH_ONLY", "context_or_safety_role": "NO", "evidence_path": "none", "limitation": "requires human protocol, ethics, scale design and repeatability evidence"},
        {"channel_id": "H_PAIRWISE", "channel": "direct pairwise trajectory preference", "software_status": "NOT_IMPLEMENTED_OPTIONAL_EXTENSION", **common, "planned": False, "model_input_role": "PREFERENCE_LABEL_IF_APPROVED", "optimization_target_role": "PREFERENCE_BO_BRANCH_ONLY", "context_or_safety_role": "NO", "evidence_path": "none", "limitation": "requires human protocol, ethics, comparison design and burden analysis"},
    ]


def primary_questions() -> str:
    return """# Primary Research Questions V2

## `PRIMARY_RESEARCH_QUESTION_V2` — mechanical primary

Given a fixed rehabilitation task and the offline-feasibility-screened two-dimensional V3 coordination-path family, can a small number of subject-specific measured interaction trials adapt a reduced-order subject model and select a final trajectory with lower independently evaluated mechanical interaction than fixed, common, random/space-filling and non-adaptive baselines under the same adaptation-trial budget?

This question concerns measured force/torque/pressure/state interaction only. It does not contain a comfort, preference, safety, effectiveness or clinical claim.

## Optional preference-aware extension

If direct human feedback and all independent approvals later exist: under the same trial budget, can preference-aware adaptation use ratings or pairwise choices to select a trajectory with higher directly reported subject preference than equal-budget baselines while satisfying independently reviewed mechanical constraints?

The extension requires direct labels. Mechanical measurements may be covariates or constraints but cannot replace preference feedback.

## Thesis formulation

- Current primary thesis: `MECHANICAL_MEASUREMENT_DRIVEN_PERSONALIZATION`.
- Optional stronger future thesis: `PREFERENCE_AWARE_PERSONALIZATION_WITH_DIRECT_HUMAN_FEEDBACK`.
- Current formulation status: `READY_WITH_LIMITATIONS`; no algorithm or physical experiment has validated either hypothesis.
"""


def scope_document() -> str:
    return """# Mechanical versus Preference Scope

## Primary: mechanical measurement-driven personalization

This is selected because it matches current observable engineering channels, has the smaller conceptual and ethical expansion, preserves the fixed-task V3 work, and is more defensible for the present undergraduate-project scope. The primary claim to test later is an equal-budget reduction in an independently evaluated, prespecified measured mechanical-interaction endpoint.

The primary candidate endpoint class is `EPISODE_RMS_VALIDATED_TASK_DIRECTION_INTERACTION_FORCE`. It is **not yet the final objective**: task direction, sign, frame, bias, delay, synchronization, repeatability and physical meaning must be calibrated independently. Until that stage finishes, `PRIMARY_OUTCOME_TYPE = MEASURED_MECHANICAL_INTERACTION_ENDPOINT_PENDING_INDEPENDENT_CALIBRATION`.

Secondary diagnostics may include peak interaction force/torque, time-profile features, pressure peak/concentration/centroid, tracking error and model residual. Safety limits and data-validity gates remain constraints, never reward terms. No arbitrary all-signal weighted score is frozen here.

## Optional extension: preference/comfort

| Label | Burden | Repeatability/bias | Low-budget and BO compatibility |
|---|---|---|---|
| Scalar rating | one response per trajectory; relatively low | scale drift, anchoring and inter-session calibration require study | simple regression/ordinal models; absolute scale may be noisy |
| Pairwise preference | requires explicit comparisons; burden can rise | relative judgments may be easier but order/context bias remains | natural fit for preference BO, but comparison graph must be designed |

If a preference branch is later approved, pairwise preference is the more direct methodological candidate for preference BO, with scalar rating as a possible secondary measure. This is not a human-study decision. `HUMAN_FEEDBACK_REQUIRED` applies to every comfort/preference claim.

Pressure is a possible comfort correlate, not comfort truth. Mechanical improvement and reported comfort may disagree and must be reported separately.
"""


def episode_protocol() -> str:
    return f"""# Future Personalization Episode Protocol

This is a causal conceptual protocol, not hardware approval.

## Cold start and research budget

- Trial 1 starts at the frozen reference: `beta=[0,0]`.
- Primary adaptation-budget hypothesis: `K={PRIMARY_ADAPTATION_BUDGET}` complete trials, including the reference cold start.
- Sensitivity budgets: `K={SENSITIVITY_BUDGETS[0]}` and `K={SENSITIVITY_BUDGETS[1]}`.
- The budget is motivated by the two-dimensional family, one interpretable baseline plus a small number of updates, and subject-burden control. It is an experimental-design hypothesis, not an approved robot or human exposure.

## One sequential episode

1. Select `beta_k` from the future independently approved V3 subset/domain.
2. Execute one complete, separately approved rehabilitation trial.
3. Acquire time-qualified robot state and validated interaction measurements; tactile/direct feedback are included only if independently available.
4. Apply frozen quality gates and compute one versioned episode-feature record.
5. Update the subject-specific gray-box parameter/posterior state using trials `1..k` only.
6. If justified, update a residual/observation model using the same causal history only.
7. Update the selector/surrogate and select `beta_(k+1)`.
8. Repeat until the fixed K-trial adaptation budget is exhausted.

## Information boundary

Known before trial: fixed reference/task, V3 mathematical family, future approved domain, fixed physics prior, reviewed safety constraints, algorithm settings, and this subject's past valid trials only.

Measured during trial: synchronized state/tracking and validated mechanical interaction; optional tactile or direct feedback only when the corresponding protocol exists.

Updated after trial: episode features, effective gray-box parameters/posterior, predictive uncertainty, optional residual model state, BO/surrogate state, and the observed-candidate ledger.

Forbidden before execution: the current/future trial outcome, final-evaluation outcome, held-out subject outcome, synthetic MyoLeg oracle/preference, or extra observations unavailable to comparison baselines.

## Identification versus evaluation

The K adaptation trials cannot also be claimed as final performance evidence. After adaptation freezes the selected beta, a separate final-evaluation block must compare the selected trajectory with reference and common/non-personalized controls under a preregistered, counterbalanced and equal-observation protocol. Repetition count/order is deliberately left to the next data/endpoint design stage.
"""


def subject_models() -> str:
    return """# Subject Model Hierarchy

| Model | Subject-specific state | Observations | Role and gate |
|---|---|---|---|
| M0 — no subject model | measured outcome ledger only | validated episode outcomes | direct/model-free baseline |
| M1 — gray-box | effective mass/stiffness/damping-like parameters or posterior; never physiological truth | q, dq, valid ddq, beta and measured mechanics from executed trials | primary low-data model candidate; must be revalidated for real measurements |
| M2 — data-driven residual | residual parameters/function on top of frozen physics baseline | causal executed-trial features and residual targets | allowed only after M1 residual is defined and train/validation separation exists |
| M3 — physics-informed/residual NN | learned subject-specific residual state | sufficient repeated measured trials | benchmark only after all PINN stop/go gates pass |

## Exact possible PINN task

`measured mechanical response = gray-box physics prediction + subject-specific residual`

Conceptual causal inputs: q, dq, validated ddq, beta/path descriptor, trial context, and past/current measured force state where temporal causality is preserved. Output: time-resolved interaction force/torque residual or a separately calibrated episode-endpoint residual.

The PINN does not infer comfort, generate a personalized objective, replace direct preference labels, or turn effective parameters into physiological truth.

## PINN stop/go gate

Enter a PINN benchmark only if: repeated measured trials exist; M1 is evaluated; a systematic residual exists; it is repeatable across repeated trials; data volume supports a learning split; and an equal-budget comparison against simpler models is frozen. Otherwise: `PINN_NOT_JUSTIFIED`.
"""


def selectors() -> str:
    return """# Trajectory Selector Hierarchy

| Selector | Required observations | Personalization source |
|---|---|---|
| S0 fixed reference | none | none |
| S1 common/population trajectory | frozen non-personalized prior only | none for current subject |
| S2 random/space-filling | candidate geometry and previously executed set | subject outcomes are recorded but not modeled for selection |
| S3 model-only greedy | updated subject model predictions | subject's executed mechanical trials via M1/M2/M3 |
| S4 standard mechanical BO | measured mechanical endpoint and uncertainty | subject's executed endpoint observations |
| S5 physics/model-informed BO | physics/gray-box prior plus measured endpoint | subject's executed measurements update the prior/surrogate |
| S6 preference BO | direct rating/pairwise labels plus constraints | explicit human feedback only |

Mechanical BO optimizes one independently calibrated observable mechanical endpoint. Preference BO optimizes latent preference utility derived from direct labels. `BO is a selector, not the source of personalization.` The source is `subject-specific observations`.

## BO stop/go gate

`PERSONALIZED_BO_JUSTIFIED` requires a frozen candidate domain, observable endpoint/direct feedback, complete-trial semantics, independent safety constraints, and an equal-budget comparator. None is inferred from synthetic MyoLeg preference. Until the next design stage freezes the endpoint and physical domain, personalized BO remains not justified and is not run.
"""


def pinn_bo_roles() -> str:
    return f"""# PINN and BO Role Definition

- `PINN_ROLE = {PINN_ROLE}`
- `BO_ROLE = {BO_ROLE}`

PINN/model residual learning addresses **prediction mismatch** between a gray-box physics baseline and measured interaction. BO addresses **low-budget trajectory selection** against a separately defined outcome. Neither supplies comfort truth or patient preference by itself.

The formulation recommends M1 gray-box plus a simple selector as the first implementable scientific baseline. M2/M3 and S4/S5/S6 are conditional benchmarks, not the declared final method.
"""


def myoleg_role() -> str:
    return """# MyoLeg Future Role V2

## KEEP

- independent offline mechanics integration and regression tests;
- feasibility/method sanity checks for the fixed V3 family;
- simulated model-mismatch and failure-mode stress tests;
- prior-model implementation checks and offline baseline construction.

## DOWNGRADE

MyoLeg is not a patient-population distribution, patient-preference truth, comfort oracle, final personalization-necessity proof, human safety result, or clinical result.

## STOP

Do not generate Cohort V2 from uncalibrated S1 ranges, increase structural z after the negative pilot, invent S2/S3 to obtain diverse oracles, redesign V4/V5 for oracle diversity, or tune objective weights to manufacture personalization.

Analytical/gray-box mechanics supplies the interpretable low-data adaptation prior. MyoLeg remains an offline test environment around that method; actual subject-specific information must enter through executed-trial observations.
"""


def tactile_role() -> str:
    return """# Tactile Role and Validation Needs

Future software path:

`raw pressure map -> timestamped preprocessing -> versioned episode features -> subject model / candidate endpoint diagnostics / safety diagnostics`

Potential features include mean/peak pressure, spatial concentration, pressure centroid/center-of-pressure where geometrically meaningful, temporal peak and distribution stability. Before any scientific use, the minimum validation set is:

- sensor calibration and units;
- sampling rate and dropped-sample behavior;
- acquisition latency and timestamp provenance;
- synchronization/skew with robot state and wrench;
- within-session and between-session repeatability;
- spatial mapping, orientation, contact area and sensor placement reproducibility.

Current active hardware/collection/control source contains no tactile acquisition implementation. Tactile is therefore planned, not available now. `pressure != comfort`: pressure is a measured interaction feature or possible comfort correlate. Direct feedback remains required for a comfort/preference target.
"""


def validation_plan() -> str:
    return f"""# Future Baseline and Validation Plan

## Equal-budget baseline hierarchy

Future comparison should include S0 reference, S1 common trajectory, S2 random/space-filling exploration, model-free mechanical BO, gray-box plus BO, and residual/PINN plus BO only if justified. Every adaptive method receives exactly `K={PRIMARY_ADAPTATION_BUDGET}` complete adaptation trials in the primary hypothesis; K=3 and K=5 are sensitivity budgets. No method receives oracle, future, held-out, simulator-preference or extra-trial information.

Non-adaptive baselines must receive a matched exposure/evaluation schedule defined before data collection; they cannot be advantaged or disadvantaged by silently changing the number or duration of executed trials.

## Primary future success form

Mechanical primary: after the equal K-trial adaptation phase, does the frozen subject-adaptive selection produce a lower independently evaluated, prespecified measured mechanical endpoint than reference, common and non-adaptive baselines?

Optional preference extension: with direct human labels, does equal-budget preference-aware selection produce higher independently evaluated direct preference than baselines while satisfying mechanical constraints?

## Validation hierarchy

1. Offline unit/integration and causal-information tests.
2. Sensor/state/wrench timing, frame, calibration and repeatability validation without a personalization claim.
3. Endpoint formulation and repeated-trial reliability study.
4. Equal-budget algorithm comparison on development participants only after independent robot/human approvals.
5. Locked confirmatory evaluation on new real subjects; the generalization unit is the `new real subject`, not a new MyoLeg parameter vector.

Adaptation and final evaluation data remain separate. Safety events, invalid data and constraint breaches are reported independently and never folded into an arbitrary reward weight.
"""


def architecture() -> str:
    return """# Future Method Architecture

The diagram is conceptual. It does not authorize robot motion or a human trial.

```mermaid
flowchart LR
  subgraph OFF[Offline and fixed]
    REF[Measured reference and fixed task]
    V3[V3 P4 family: beta_flex, beta_extend]
    PHY[Analytical or gray-box physics prior]
    MYO[MyoLeg stress tests and feasibility support]
    SAFE[Independent reviewed safety and domain gate]
    REF --> V3
    MYO -. offline checks only .-> PHY
  end

  subgraph ON[Online per subject]
    SEL[Trajectory selector]
    EXEC[One complete robot trial]
    MEAS[Robot state and validated wrench; optional tactile or direct feedback]
    FEAT[Episode feature extraction and quality gates]
    OBS[Mechanical endpoint or direct preference observation model]
    SUBJ[Subject-specific gray-box posterior; optional gated residual]
    SEL -->|beta_k| SAFE
    SAFE -->|only if independently approved| EXEC
    EXEC --> MEAS --> FEAT
    FEAT --> OBS
    FEAT --> SUBJ
    SUBJ --> SEL
    OBS --> SEL
  end

  V3 --> SEL
  PHY --> SUBJ
```

## State ownership

- Fixed physics prior: analytical/gray-box structure; MyoLeg is offline-only support.
- Subject-specific state: valid episode ledger, effective parameter/posterior state, predictive uncertainty and optional residual state.
- Online selector state: evaluated beta set and mechanical/preference surrogate based only on causal observations.
- Safety layer: independent of the optimizer and unable to be relaxed by predicted reward.

`ALGORITHM_FORMULATION_READY != ROBOT_EXECUTION_READY`.
"""


def retention_map() -> str:
    return """# Existing Work Retention Map V2

| Work | Placement | Role |
|---|---|---|
| 2-DOF supine model, measured asymmetric reference and ROM convention | MAIN_TEXT_CANDIDATE | fixed task and interpretable mechanics |
| V3 P4 fixed-ROM branch-aware parameterization | MAIN_TEXT_CANDIDATE | primary low-dimensional candidate family |
| reduced-order gray-box identification concept | MAIN_TEXT_CANDIDATE_WITH_REAL_DATA_REVALIDATION | low-data subject-adaptation candidate, not physiological truth |
| independent MyoLeg integration and truth semantics | MAIN_TEXT_SUPPORTING_CANDIDATE | motivates independence and simulator support role |
| stop-or-pivot evidence and measurement-driven rationale | MAIN_TEXT_CONCISE_MOTIVATION | explains the formulation change without dominating the method |
| Cohort V1 generation and V2/V3 full synthetic landscapes | SUPPLEMENTARY_OR_APPENDIX | reproducible synthetic negative evidence |
| V2 parameterization root cause and objective/heterogeneity audits | SUPPLEMENTARY_OR_APPENDIX | formulation diagnostics |
| amended S1 and structural heterogeneity pilot | SUPPLEMENTARY_OR_APPENDIX | preregistered negative mechanism evidence |
| long sequence of intermediate negative diagnostic stages | DO_NOT_MAKE_CENTRAL_FINAL_METHOD_CLAIM | retain provenance but summarize, do not let it overwhelm final method |
| old same-family five-parameter virtual truth | HISTORICAL_DEVELOPMENT_ONLY | circularity concern; not primary validation |
| PINN/BO without measured endpoint or feedback | NOT_COMPLETED_EVIDENCE | future conditional methods only |

The pivot changes the claim hierarchy, not the historical record. No frozen artifact is deleted or rewritten.
"""


def final_payload() -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "formal_decision": FORMAL_STATUS,
        "PRIMARY_RESEARCH_QUESTION_V2": "Can K low-budget subject-specific measured interaction trials adapt a reduced-order subject model and select a final fixed-task V3 coordination trajectory with lower independently evaluated mechanical interaction than equal-budget non-personalized or non-adaptive baselines?",
        "PRIMARY_THESIS_FORMULATION": PRIMARY_THESIS,
        "OPTIONAL_STRONGER_FUTURE_CLAIM": OPTIONAL_EXTENSION,
        "PRIMARY_PERSONALIZATION_SOURCE": PRIMARY_PERSONALIZATION_SOURCE,
        "PRIMARY_OUTCOME_TYPE": PRIMARY_OUTCOME_TYPE,
        "PRIMARY_CANDIDATE_OUTCOME": PRIMARY_CANDIDATE_OUTCOME,
        "PRIMARY_CANDIDATE_OUTCOME_FINALIZED": False,
        "TRAJECTORY_PARAMETERIZATION": {
            "id": TRAJECTORY_PARAMETERIZATION,
            "parameters": ["beta_flex", "beta_extend"],
            "cold_start": [0.0, 0.0],
            "offline_v3_bounds_robot_approved": False,
        },
        "SUBJECT_MODEL_HIERARCHY": ["M0_NO_SUBJECT_MODEL", "M1_GRAY_BOX_PRIMARY_CANDIDATE", "M2_DATA_DRIVEN_RESIDUAL_CONDITIONAL", "M3_PHYSICS_INFORMED_RESIDUAL_NN_CONDITIONAL"],
        "TRAJECTORY_SELECTOR_HIERARCHY": ["S0_FIXED_REFERENCE", "S1_COMMON_TRAJECTORY", "S2_RANDOM_SPACE_FILLING", "S3_MODEL_ONLY_GREEDY", "S4_STANDARD_MECHANICAL_BO", "S5_PHYSICS_MODEL_INFORMED_BO", "S6_PREFERENCE_BO_WITH_DIRECT_FEEDBACK"],
        "PINN_ROLE": PINN_ROLE,
        "BO_ROLE": BO_ROLE,
        "MYOLEG_ROLE": MYOLEG_ROLE,
        "TACTILE_ROLE": TACTILE_ROLE,
        "HUMAN_FEEDBACK_REQUIREMENT": HUMAN_FEEDBACK_REQUIREMENT,
        "VALIDATION_HIERARCHY": ["OFFLINE_CAUSAL_AND_INTEGRATION", "MEASUREMENT_VALIDATION", "ENDPOINT_REPEATABILITY", "EQUAL_BUDGET_DEVELOPMENT", "LOCKED_NEW_REAL_SUBJECT_CONFIRMATION"],
        "PRIMARY_ADAPTATION_BUDGET_HYPOTHESIS": PRIMARY_ADAPTATION_BUDGET,
        "SENSITIVITY_BUDGET_HYPOTHESES": list(SENSITIVITY_BUDGETS),
        "GENERALIZATION_UNIT": "NEW_REAL_SUBJECT_AFTER_INDEPENDENT_APPROVALS",
        "ALGORITHM_FORMULATION_READY_NE_ROBOT_EXECUTION_READY": True,
        "NOT_HUMAN_READY": True,
        "NOT_ROBOT_APPROVED": True,
        "next_stage": NEXT_STAGE,
        "next_stage_executed": False,
    }


def report() -> str:
    return f"""# Personalized Rehabilitation Research Formulation V2

## Formal decision

`{FORMAL_STATUS}`

Protocol SHA-256: `{FROZEN_PROTOCOL_SHA256}`

The project primary line is now mechanical measurement-driven personalization. The candidate family remains the frozen V3 P4 branch-aware two-parameter family. Subject specificity comes from the current subject's measured executed trials, not a MyoLeg parameter vector. Preference/comfort is retained only as a future direct-feedback extension.

## Q1. New primary research question

Under the same low adaptation-trial budget, can subject-specific measured interaction trials adapt a reduced-order model and select a final fixed-task V3 trajectory with lower independently evaluated mechanical interaction than fixed, common, random/space-filling and non-adaptive baselines?

## Q2. Primary thesis

`{PRIMARY_THESIS}`. Comfort/preference is optional future work because no direct feedback channel, human protocol, robot readiness or ethics approval currently exists.

## Q3. Subject-specific information source

`{PRIMARY_PERSONALIZATION_SOURCE}`: time-qualified robot state and independently validated interaction mechanics, with tactile features only after sensor validation. Direct ratings/pairwise labels are required for a future preference branch.

## Q4. What updates after each trial?

A versioned episode-feature record, effective gray-box parameter/posterior state, prediction uncertainty, optional gated residual state, observed-candidate ledger and selector/surrogate state. Only trials already executed by the same subject may be used.

## Q5. MyoLeg role

`{MYOLEG_ROLE}`. It supports offline prior checks, feasibility and mismatch stress tests; it is not patient/preference truth.

## Q6. PINN task and gate

Potential task: predict the subject-specific residual between measured mechanical response and gray-box physics. It is justified only after repeated measured trials, evaluated gray-box baseline, systematic repeatable residual, sufficient data and an equal-budget benchmark. Current status: `PINN_NOT_JUSTIFIED`.

## Q7. BO objective and gate

Mechanical BO would optimize a calibrated measured mechanical endpoint; preference BO would optimize latent utility from direct feedback. BO is a selector, not the source of personalization. It remains unjustified until domain, endpoint/feedback, complete-trial semantics and independent safety constraints are frozen.

## Q8. Tactile role

Tactile pressure is a planned measured interaction feature and possible comfort correlate. It needs calibration, rate/latency/synchronization, repeatability and spatial mapping validation. `pressure != comfort`.

## Q9. Baselines and validation

Reference, common trajectory, random/space-filling, model-free BO, gray-box plus BO, and residual/PINN plus BO only if justified; every adaptive method receives K={PRIMARY_ADAPTATION_BUDGET} complete adaptation trials in the primary hypothesis, with K=3/5 sensitivity analyses. Adaptation and final evaluation are separate, and no method receives extra truth.

## Q10. Single next stage

`{NEXT_STAGE}`

It should freeze measurement channels, synchronization, episode features, primary mechanical endpoint and repeated-trial design. It was not executed.

## Why READY_WITH_LIMITATIONS

- The primary scientific question, data source, V3 family, episode semantics, model/selector hierarchy and validation logic are coherent and frozen.
- No robot channel is formally research-ready: safety and identification manifests remain unreviewed and base-wrench rotation is unverified.
- Tactile/direct feedback are not implemented.
- The exact mechanical endpoint and K budget remain next-stage hypotheses.
- No PINN, BO, robot or human study was run.

`ALGORITHM_FORMULATION_READY != ROBOT_EXECUTION_READY`; status remains `NOT_HUMAN_READY / NOT_ROBOT_APPROVED`.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(
        OUTPUT / "checksums.sha256",
        "\n".join(f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files) + "\n",
    )


def execute() -> None:
    protocol = verify_freeze()
    atomic_text(OUTPUT / "PRIMARY_RESEARCH_QUESTIONS.md", primary_questions())
    atomic_text(OUTPUT / "MECHANICAL_VS_PREFERENCE_SCOPE.md", scope_document())
    atomic_csv(OUTPUT / "MEASUREMENT_CHANNELS_AND_ROLES.csv", measurement_rows())
    atomic_text(OUTPUT / "SUBJECT_MODEL_HIERARCHY.md", subject_models())
    atomic_text(OUTPUT / "TRAJECTORY_SELECTOR_HIERARCHY.md", selectors())
    atomic_text(OUTPUT / "PINN_AND_BO_ROLE_DEFINITION.md", pinn_bo_roles())
    atomic_text(OUTPUT / "MYOLEG_FUTURE_ROLE_V2.md", myoleg_role())
    atomic_text(OUTPUT / "TACTILE_ROLE_AND_VALIDATION_NEEDS.md", tactile_role())
    atomic_text(OUTPUT / "FUTURE_PERSONALIZATION_EPISODE_PROTOCOL.md", episode_protocol())
    atomic_text(OUTPUT / "FUTURE_BASELINE_AND_VALIDATION_PLAN.md", validation_plan())
    atomic_text(OUTPUT / "FUTURE_METHOD_ARCHITECTURE.md", architecture())
    atomic_text(OUTPUT / "EXISTING_WORK_RETENTION_MAP_V2.md", retention_map())
    atomic_json(OUTPUT / "FINAL_RESEARCH_FORMULATION_V2.json", final_payload())
    atomic_text(OUTPUT / "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2_REPORT.md", report())
    atomic_json(
        OUTPUT / "metadata.json",
        {
            "stage_id": STAGE_ID,
            "formal_status": FORMAL_STATUS,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "input_count": len(protocol["input_files"]),
            "primary_thesis": PRIMARY_THESIS,
            "primary_personalization_source": PRIMARY_PERSONALIZATION_SOURCE,
            "primary_outcome_type": PRIMARY_OUTCOME_TYPE,
            "primary_candidate_outcome_finalized": False,
            "measurement_channel_count": len(measurement_rows()),
            "formulation_only": True,
            "held_out_scientific_access_count": 0,
            "simulator_experiment_count": 0,
            "pinn_training_count": 0,
            "bo_run_count": 0,
            "robot_access_count": 0,
            "human_study_count": 0,
            "cohort_v2_generated": False,
            "s2_s3_expansion_run": False,
            "v4_v5_redesign_run": False,
            "objective_weight_search_run": False,
            "frozen_artifacts_modified": False,
            "human_ready": False,
            "robot_approved": False,
            "next_stage": NEXT_STAGE,
            "next_stage_executed": False,
        },
    )
    write_checksums()
    print(json.dumps(final_payload(), indent=2, sort_keys=True))


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
