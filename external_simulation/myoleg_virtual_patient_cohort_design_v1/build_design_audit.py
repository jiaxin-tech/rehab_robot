"""Build the offline MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1 audit.

This stage inventories the frozen native-ROM MyoLeg model and performs only
small, one-family-at-a-time numerical sensitivity checks.  It deliberately
does not generate virtual subjects, fit any learner, evaluate a candidate
landscape, or touch robot-facing code.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import mujoco
import numpy as np


STAGE_ID = "MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1"
OUTCOME = "MYOLEG_COHORT_DESIGN_READY_WITH_EVIDENCE_GAPS"
PARAMETER_SEMANTIC_VERSION = "MYOLEG_COHORT_PARAMETER_SEMANTICS_V1"
V2_REFERENCE_ID = "NATIVE_ROM_REFERENCE_CANDIDATE"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_virtual_patient_cohort_design_v1"
)

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
TRANSFORMATION_PATH = V2_REFERENCE_PATH.with_name("NATIVE_REFERENCE_TRANSFORMATION.json")
TRUTH_SEMANTICS_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
    / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
)
PRIOR_REPLAY_DATASET = TRUTH_SEMANTICS_PATH.with_name("SENSITIVITY_REFERENCE_REPLAY.npz")
PRIOR_REPLAY_BUILDER = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_reference_trajectory_replay_v1"
    / "build_and_replay.py"
)
FORMAL_REFERENCE_PATH = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST_PATH = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"

FROZEN_SHA256 = {
    "base_myoleg_model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
}

TARGET_HIP = "hip_flexion_r"
TARGET_KNEE = "knee_angle_r"
RIGHT_LOWER_LIMB_BODIES = (
    "femur_r",
    "tibia_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "patella_r",
)
MOMENT_ARM_THRESHOLD_M = 1.0e-7
SMOKE_SCALE = 0.05
INTEGRITY_THRESHOLDS = {
    "source_equality_residual_max": 1.0e-3,
    "algebraic_residual_max_nm": 1.0e-8,
    "tracking_q_max_abs_deg": 1.0,
    "peak_force_ratio_vs_nominal_max": 2.0,
    "native_knee_min_deg": 0.0,
    "native_knee_max_deg": 120.0,
}

OFFICIAL_MUJOCO_SOURCES = {
    "muscle_model": "https://mujoco.readthedocs.io/en/3.6.0/modeling.html#muscle-actuators",
    "muscle_xml": "https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#actuator-muscle",
}

TAXONOMY_ROWS = [
    {
        "family_id": "SEGMENT_MASS_INERTIA_COUPLED_SCALE",
        "actual_fields": "body_mass; body_inertia",
        "classification": "A",
        "classification_name": "PRIMARY SUBJECT HETEROGENEITY",
        "reason": "Direct segment inertial properties with a clear subject-level interpretation when mass and inertia are changed together and COM is fixed.",
        "eligible_for_cohort_generation": "YES_WITH_RANGE_EVIDENCE",
        "learner_overlap": "partial mass-related behavior but structurally richer than one learner mass_scale",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": True,
    },
    {
        "family_id": "SEGMENT_COM_LOCATION",
        "actual_fields": "body_ipos",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "COM is interpretable but independent displacement without a segment-scaling protocol can be inertially inconsistent.",
        "eligible_for_cohort_generation": "NO_IN_V1",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "SEGMENT_LENGTH_AND_GEOMETRY",
        "actual_fields": "body_pos; geom_pos; geom_size; site_pos",
        "classification": "E",
        "classification_name": "DO NOT PERTURB",
        "reason": "Length changes require coordinated relocation of attachment sites, wrap geometry, tendon paths, joint centers and moment-arm revalidation.",
        "eligible_for_cohort_generation": "NO",
        "learner_overlap": "not directly represented",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
    {
        "family_id": "MUSCLE_FORCE_CAPACITY_SCALE",
        "actual_fields": "actuator_gainprm[2]; actuator_biasprm[2] (XML force)",
        "classification": "A",
        "classification_name": "PRIMARY SUBJECT HETEROGENEITY",
        "reason": "The frozen XML force field is MuJoCo peak active force F0; grouped scaling remains mapped to named actuators and their tendon transmissions.",
        "eligible_for_cohort_generation": "ACTIVE_CONDITION_ONLY_OR_MUTUALLY_EXCLUSIVE_WITH_FP_MAX_IN_P0",
        "learner_overlap": "independent structural muscle law; no true K or B is assigned",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": True,
    },
    {
        "family_id": "BIARTICULAR_FORCE_CAPACITY_SCALE",
        "actual_fields": "actuator_gainprm[2]; actuator_biasprm[2] for structurally verified hip+knee spanning actuators",
        "classification": "A",
        "classification_name": "PRIMARY SUBJECT HETEROGENEITY",
        "reason": "Changes native tendon-transmitted hip-knee coupling without adding a synthetic coupling torque equation.",
        "eligible_for_cohort_generation": "ACTIVE_CONDITION_ONLY_OR_MUTUALLY_EXCLUSIVE_WITH_BIARTICULAR_FP_MAX_IN_P0",
        "learner_overlap": "not a five-parameter learner truth coordinate",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": True,
    },
    {
        "family_id": "MUSCLE_PASSIVE_FP_MAX_SCALE",
        "actual_fields": "actuator_gainprm[7]; actuator_biasprm[7] (XML fpmax)",
        "classification": "A",
        "classification_name": "PRIMARY SUBJECT HETEROGENEITY",
        "reason": "fpmax directly controls passive FLV magnitude and can encode passive-property heterogeneity while retaining the native muscle law.",
        "eligible_for_cohort_generation": "YES_WITH_RANGE_EVIDENCE",
        "learner_overlap": "independent nonlinear muscle law; no true learner stiffness is assigned",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": True,
    },
    {
        "family_id": "BIARTICULAR_PASSIVE_FP_MAX_SCALE",
        "actual_fields": "actuator_gainprm[7]; actuator_biasprm[7] for structurally verified hip+knee spanning actuators",
        "classification": "A",
        "classification_name": "PRIMARY SUBJECT HETEROGENEITY",
        "reason": "Provides a direct passive-property factor on the native biarticular transmissions without adding a coupling equation.",
        "eligible_for_cohort_generation": "YES_WITH_RANGE_EVIDENCE",
        "learner_overlap": "independent nonlinear passive coupling; no true learner stiffness is assigned",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": True,
    },
    {
        "family_id": "MUSCLE_OPERATING_RANGE",
        "actual_fields": "actuator_gainprm[0:2]; actuator_biasprm[0:2] (XML range)",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "Scientifically meaningful but high-impact and model-specific; changing it alters L0/LT mapping and needs muscle-specific evidence.",
        "eligible_for_cohort_generation": "NO_UNTIL_MUSCLE_SPECIFIC_EVIDENCE",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "ACTUATOR_LENGTH_RANGE",
        "actual_fields": "actuator_lengthrange",
        "classification": "E",
        "classification_name": "DO NOT PERTURB",
        "reason": "Compiled tendon-transmission length limits define the normalization map; they are not physiological normalized fiber length and must follow geometry.",
        "eligible_for_cohort_generation": "NO",
        "learner_overlap": "not directly represented",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
    {
        "family_id": "MUSCLE_FLV_SHAPE_OTHER",
        "actual_fields": "gainprm/biasprm lmin,lmax,vmax,fvmax",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "These alter active/passive FLV shape; official semantics exist but no project-specific subject ranges or coupled time-constant protocol are frozen.",
        "eligible_for_cohort_generation": "NO_UNTIL_RANGE_EVIDENCE",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "MUSCLE_ACTIVATION_TIME_CONSTANTS",
        "actual_fields": "actuator_dynprm[0:2]",
        "classification": "B",
        "classification_name": "SECONDARY PHYSIOLOGICAL-LIKE VARIABILITY",
        "reason": "Activation/deactivation time constants have fiber-state meaning but do not affect frozen zero-activation P0 truth and need a separate active-condition protocol.",
        "eligible_for_cohort_generation": "NO_IN_PRIMARY_P0",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "LOW_BACKGROUND_ACTIVATION_STATE",
        "actual_fields": "mjData.ctrl; mjData.act within actuator_ctrlrange [0,1]",
        "classification": "B",
        "classification_name": "SECONDARY PHYSIOLOGICAL-LIKE VARIABILITY",
        "reason": "Represents trial state rather than fixed musculoskeletal structure; keep it episode-level unless an independently justified subject baseline is frozen.",
        "eligible_for_cohort_generation": "EPISODE_NUISANCE_ONLY",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "TENDON_PATH_ATTACHMENTS_AND_WRAPS",
        "actual_fields": "tendon path; wrap_objid; site_pos; geom_pos/size; tendon_lengthspring",
        "classification": "E",
        "classification_name": "DO NOT PERTURB",
        "reason": "Direct edits can invalidate moment arms, wrapping topology and the converted musculoskeletal geometry.",
        "eligible_for_cohort_generation": "NO",
        "learner_overlap": "not directly represented",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
    {
        "family_id": "TENDON_ELASTICITY",
        "actual_fields": "tendon_stiffness; tendon_damping (both zero in frozen model)",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "Enabling nonzero elasticity changes the frozen inelastic-tendon model class and lacks calibrated values.",
        "eligible_for_cohort_generation": "NO_IN_V1",
        "learner_overlap": "not directly represented",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "JOINT_DAMPING_STIFFNESS_FRICTION",
        "actual_fields": "dof_damping; jnt_stiffness; dof_frictionloss",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "Numerically perturbable, but direct K/B-like truth would overlap the simplified learner and risk circular validation.",
        "eligible_for_cohort_generation": "NO_PRIMARY_COHORT",
        "learner_overlap": "directly overlaps learner K/B semantics",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "JOINT_ARMATURE",
        "actual_fields": "dof_armature",
        "classification": "D",
        "classification_name": "STRESS TEST ONLY",
        "reason": "A numerical rotor/inertial regularizer rather than an established patient musculoskeletal factor here.",
        "eligible_for_cohort_generation": "NO_PRIMARY_COHORT",
        "learner_overlap": "mass-like numerical effect",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "JOINT_RANGE_AND_COORDINATE_SEMANTICS",
        "actual_fields": "jnt_range; jnt_axis; jnt_pos",
        "classification": "E",
        "classification_name": "DO NOT PERTURB",
        "reason": "Would alter the frozen native domain, coordinate definition or physical joint mapping.",
        "eligible_for_cohort_generation": "NO",
        "learner_overlap": "not directly represented",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
    {
        "family_id": "KNEE_PATELLA_EQUALITY_MECHANISM",
        "actual_fields": "eq_data polynomial coefficients; solref; solimp",
        "classification": "E",
        "classification_name": "DO NOT PERTURB",
        "reason": "Defines the frozen coupled knee/patella coordinate manifold required by the truth projection.",
        "eligible_for_cohort_generation": "NO",
        "learner_overlap": "not directly represented",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
    {
        "family_id": "MEASUREMENT_INTERFACE_NOISE",
        "actual_fields": "future observation/measurement layer only; not an mjModel subject parameter",
        "classification": "C",
        "classification_name": "ROBUSTNESS / NUISANCE ONLY",
        "reason": "Useful for robustness testing but must never be described as patient musculoskeletal heterogeneity.",
        "eligible_for_cohort_generation": "SEPARATE_OBSERVATION_PROTOCOL_ONLY",
        "learner_overlap": "observation rather than truth model",
        "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "smoke_tested": False,
    },
    {
        "family_id": "SOLVER_DRIVER_NUMERICS",
        "actual_fields": "mjOption solver/timestep; diagnostic qfrc_applied driver",
        "classification": "C",
        "classification_name": "ROBUSTNESS / NUISANCE ONLY",
        "reason": "Numerical/replay nuisance, not a subject parameter; frozen for comparable cohort truth generation.",
        "eligible_for_cohort_generation": "NO_VARIATION_DURING_TRUTH_GENERATION",
        "learner_overlap": "none",
        "range_status": "NOT_APPLICABLE",
        "smoke_tested": False,
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or ""


def load_prior_builder() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_myoleg_replay_builder", PRIOR_REPLAY_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen replay builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_inputs() -> dict[str, Any]:
    actual = {
        "base_myoleg_model": sha256_file(MODEL_PATH),
        "v2_reference": sha256_file(V2_REFERENCE_PATH),
        "truth_semantics": sha256_file(TRUTH_SEMANTICS_PATH),
        "formal_reference": sha256_file(FORMAL_REFERENCE_PATH),
        "formal_manifest": sha256_file(FORMAL_MANIFEST_PATH),
    }
    failures = {key: [FROZEN_SHA256[key], value] for key, value in actual.items() if value != FROZEN_SHA256[key]}
    if failures:
        raise RuntimeError(f"frozen input changed: {failures}")
    semantic = json.loads(TRUTH_SEMANTICS_PATH.read_text(encoding="utf-8"))
    if semantic["semantic_version"] != TRUTH_SEMANTIC_VERSION or semantic["truth_field"] != TRUTH_FIELD:
        raise RuntimeError("frozen truth semantics changed")
    manifest = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["rom_protocol_version"] != "ROM_PROTOCOL_V2":
        raise RuntimeError("formal ROM changed")
    if manifest["hip_rom_deg"] != [0.0, 120.0] or manifest["knee_rom_deg"] != [5.0, 145.0]:
        raise RuntimeError("formal ROM bounds changed")
    if manifest["theta_shank_definition"] != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    return {"hashes": actual, "formal_manifest": manifest, "truth_semantics": semantic}


def runtime_environment() -> dict[str, Any]:
    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "myosuite": importlib.metadata.version("myosuite"),
        "numpy": np.__version__,
    }
    expected = {"python": "3.10.19", "mujoco": "3.6.0", "myosuite": "2.12.2"}
    result["frozen_expected"] = expected
    result["frozen_match"] = all(result[key] == value for key, value in expected.items())
    if not result["frozen_match"]:
        raise RuntimeError("frozen MyoLeg runtime environment changed")
    return result


def structural_muscle_map(
    model: mujoco.MjModel, reference: dict[str, Any], prior: Any
) -> dict[int, dict[str, Any]]:
    moment = np.zeros((len(reference["time_s"]), model.nu, 2), dtype=float)
    data = mujoco.MjData(model)
    for sample in range(len(reference["time_s"])):
        prior.reset_to_target_state(
            model,
            data,
            reference["q"][sample],
            reference["dq"][sample],
            reference["ddq"][sample],
        )
        mujoco.mj_forward(model, data)
        tangent = prior.independent_coordinate_tangent(model, data)
        moment[sample] = prior.sparse_actuator_moment_times_tangent(model, data, tangent)

    def action(values: np.ndarray) -> str:
        contractile_torque = -values[np.abs(values) > MOMENT_ARM_THRESHOLD_M]
        if contractile_torque.size == 0:
            return "NONE"
        if np.all(contractile_torque > 0.0):
            return "FLEXOR"
        if np.all(contractile_torque < 0.0):
            return "EXTENSOR"
        return "MIXED_REQUIRES_REVIEW"

    result: dict[int, dict[str, Any]] = {}
    for actuator in range(model.nu):
        hip_max = float(np.max(np.abs(moment[:, actuator, 0])))
        knee_max = float(np.max(np.abs(moment[:, actuator, 1])))
        spans_hip = hip_max > MOMENT_ARM_THRESHOLD_M
        spans_knee = knee_max > MOMENT_ARM_THRESHOLD_M
        if spans_hip and spans_knee:
            group = "HIP_KNEE_BIARTICULAR"
        elif spans_hip:
            group = "HIP_ONLY"
        elif spans_knee:
            group = "KNEE_ONLY"
        else:
            group = "OUTSIDE_TARGET_COORDINATES"
        result[actuator] = {
            "spans_hip": spans_hip,
            "spans_knee": spans_knee,
            "structural_group": group,
            "hip_contractile_action": action(moment[:, actuator, 0]),
            "knee_contractile_action": action(moment[:, actuator, 1]),
            "hip_moment_arm_max_abs_m": hip_max,
            "knee_moment_arm_max_abs_m": knee_max,
            "mapping_basis": "compiled tendon transmission moment matrix over all 401 frozen V2 states",
        }
    return result


def tendon_path(model: mujoco.MjModel, tendon_id: int) -> list[dict[str, Any]]:
    result = []
    start = int(model.tendon_adr[tendon_id])
    count = int(model.tendon_num[tendon_id])
    wrap_names = {
        int(mujoco.mjtWrap.mjWRAP_JOINT): "JOINT",
        int(mujoco.mjtWrap.mjWRAP_PULLEY): "PULLEY",
        int(mujoco.mjtWrap.mjWRAP_SITE): "SITE",
        int(mujoco.mjtWrap.mjWRAP_SPHERE): "SPHERE",
        int(mujoco.mjtWrap.mjWRAP_CYLINDER): "CYLINDER",
    }
    for address in range(start, start + count):
        wrap_type = int(model.wrap_type[address])
        object_id = int(model.wrap_objid[address])
        if wrap_type == int(mujoco.mjtWrap.mjWRAP_SITE):
            object_name = model_name(model, mujoco.mjtObj.mjOBJ_SITE, object_id)
        elif wrap_type in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)):
            object_name = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, object_id)
        elif wrap_type == int(mujoco.mjtWrap.mjWRAP_JOINT):
            object_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, object_id)
        else:
            object_name = ""
        result.append(
            {
                "type": wrap_names.get(wrap_type, str(wrap_type)),
                "object_id": object_id,
                "object_name": object_name,
                "parameter": float(model.wrap_prm[address]),
            }
        )
    return result


def inventory_rows(
    model: mujoco.MjModel, structural: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    columns = {
        "category": "",
        "object_type": "",
        "object_id": "",
        "object_name": "",
        "field": "",
        "current_value": "",
        "unit_or_normalization": "",
        "xml_or_runtime_source": "",
        "target_leg": "",
        "structural_group": "",
        "spans_hip": "",
        "spans_knee": "",
        "hip_contractile_action": "",
        "knee_contractile_action": "",
        "hip_moment_arm_max_abs_m": "",
        "knee_moment_arm_max_abs_m": "",
        "taxonomy_family": "",
        "taxonomy_class": "",
        "interpretation": "",
        "notes": "",
    }
    result: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        row = dict(columns)
        for key, value in values.items():
            row[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
        result.append(row)

    for body_id in range(1, model.nbody):
        body = model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        is_right = body.endswith("_r")
        common = {
            "category": "BODY_ANTHROPOMETRY",
            "object_type": "body",
            "object_id": body_id,
            "object_name": body,
            "target_leg": is_right,
        }
        add(
            **common,
            field="body_mass",
            current_value=float(model.body_mass[body_id]),
            unit_or_normalization="kg",
            xml_or_runtime_source="compiled mjModel.body_mass",
            taxonomy_family="SEGMENT_MASS_INERTIA_COUPLED_SCALE",
            taxonomy_class="A",
            interpretation="segment mass; only eligible with coupled inertia scaling",
        )
        add(
            **common,
            field="body_inertia",
            current_value=model.body_inertia[body_id].tolist(),
            unit_or_normalization="kg*m^2 principal inertia",
            xml_or_runtime_source="compiled mjModel.body_inertia",
            taxonomy_family="SEGMENT_MASS_INERTIA_COUPLED_SCALE",
            taxonomy_class="A",
            interpretation="principal segment inertia; scale with mass in V1",
        )
        add(
            **common,
            field="body_ipos",
            current_value=model.body_ipos[body_id].tolist(),
            unit_or_normalization="m in body frame",
            xml_or_runtime_source="compiled mjModel.body_ipos",
            taxonomy_family="SEGMENT_COM_LOCATION",
            taxonomy_class="D",
            interpretation="inertial frame / COM location",
        )
        add(
            **common,
            field="body_pos",
            current_value=model.body_pos[body_id].tolist(),
            unit_or_normalization="m relative to parent body",
            xml_or_runtime_source="compiled mjModel.body_pos",
            taxonomy_family="SEGMENT_LENGTH_AND_GEOMETRY",
            taxonomy_class="E",
            interpretation="kinematic geometry, not safe as an isolated scalar perturbation",
        )

    actuator_to_tendon = {}
    for actuator in range(model.nu):
        tendon_id = int(model.actuator_trnid[actuator, 0])
        actuator_to_tendon[actuator] = tendon_id
        muscle = model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        tendon = model_name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
        mapping = structural[actuator]
        common = {
            "category": "MUSCLE_ACTUATOR",
            "object_type": "actuator",
            "object_id": actuator,
            "object_name": muscle,
            "target_leg": muscle.endswith("_r"),
            "structural_group": mapping["structural_group"],
            "spans_hip": mapping["spans_hip"],
            "spans_knee": mapping["spans_knee"],
            "hip_contractile_action": mapping["hip_contractile_action"],
            "knee_contractile_action": mapping["knee_contractile_action"],
            "hip_moment_arm_max_abs_m": mapping["hip_moment_arm_max_abs_m"],
            "knee_moment_arm_max_abs_m": mapping["knee_moment_arm_max_abs_m"],
        }
        add(
            **common,
            field="force",
            current_value={"gainprm_2": float(model.actuator_gainprm[actuator, 2]), "biasprm_2": float(model.actuator_biasprm[actuator, 2])},
            unit_or_normalization="N peak active force F0; also scales passive bias",
            xml_or_runtime_source="XML general gainprm/biasprm index 2; muscle shortcut attribute force",
            taxonomy_family="BIARTICULAR_FORCE_CAPACITY_SCALE" if mapping["structural_group"] == "HIP_KNEE_BIARTICULAR" else "MUSCLE_FORCE_CAPACITY_SCALE",
            taxonomy_class="A",
            interpretation="actual MuJoCo muscle force capacity field",
            notes=f"transmission={tendon}",
        )
        add(
            **common,
            field="fpmax",
            current_value={"gainprm_7": float(model.actuator_gainprm[actuator, 7]), "biasprm_7": float(model.actuator_biasprm[actuator, 7])},
            unit_or_normalization="passive force at lmax relative to F0",
            xml_or_runtime_source="XML general gainprm/biasprm index 7; muscle shortcut attribute fpmax",
            taxonomy_family="BIARTICULAR_PASSIVE_FP_MAX_SCALE" if mapping["structural_group"] == "HIP_KNEE_BIARTICULAR" else "MUSCLE_PASSIVE_FP_MAX_SCALE",
            taxonomy_class="A",
            interpretation="passive FLV magnitude parameter",
        )
        add(
            **common,
            field="range",
            current_value=model.actuator_gainprm[actuator, 0:2].tolist(),
            unit_or_normalization="scaled muscle length in units of L0",
            xml_or_runtime_source="gainprm/biasprm indices 0:2; XML muscle range",
            taxonomy_family="MUSCLE_OPERATING_RANGE",
            taxonomy_class="D",
            interpretation="normalized muscle operating range used to infer L0/LT",
        )
        add(
            **common,
            field="actuator_lengthrange",
            current_value=model.actuator_lengthrange[actuator].tolist(),
            unit_or_normalization="m of tendon transmission length",
            xml_or_runtime_source="compiled mjModel.actuator_lengthrange",
            taxonomy_family="ACTUATOR_LENGTH_RANGE",
            taxonomy_class="E",
            interpretation="transmission length range; not physiological normalized fiber length",
        )
        add(
            **common,
            field="lmin_lmax_vmax_fvmax",
            current_value={
                "lmin": float(model.actuator_gainprm[actuator, 4]),
                "lmax": float(model.actuator_gainprm[actuator, 5]),
                "vmax": float(model.actuator_gainprm[actuator, 6]),
                "fvmax": float(model.actuator_gainprm[actuator, 8]),
            },
            unit_or_normalization="MuJoCo normalized FLV parameters",
            xml_or_runtime_source="gainprm/biasprm indices 4,5,6,8",
            taxonomy_family="MUSCLE_FLV_SHAPE_OTHER",
            taxonomy_class="D",
            interpretation="active/passive FLV shape context",
        )
        add(
            **common,
            field="activation_time_constants",
            current_value=model.actuator_dynprm[actuator, 0:3].tolist(),
            unit_or_normalization="s, s, smoothing control unit",
            xml_or_runtime_source="mjModel.actuator_dynprm; XML timeconst/tausmooth",
            taxonomy_family="MUSCLE_ACTIVATION_TIME_CONSTANTS",
            taxonomy_class="B",
            interpretation="activation and deactivation dynamics",
        )
        add(
            **common,
            field="control_range",
            current_value=model.actuator_ctrlrange[actuator].tolist(),
            unit_or_normalization="dimensionless neural control [0,1]",
            xml_or_runtime_source="mjModel.actuator_ctrlrange",
            taxonomy_family="LOW_BACKGROUND_ACTIVATION_STATE",
            taxonomy_class="B",
            interpretation="state/episode condition, not structural identity",
        )
        add(
            **common,
            field="tendon_transmission",
            current_value={"tendon_id": tendon_id, "tendon_name": tendon},
            unit_or_normalization="named structural mapping",
            xml_or_runtime_source="mjModel.actuator_trnid",
            taxonomy_family="TENDON_PATH_ATTACHMENTS_AND_WRAPS",
            taxonomy_class="E",
            interpretation="frozen muscle-tendon geometry mapping",
        )

    for tendon_id in range(model.ntendon):
        tendon = model_name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
        common = {
            "category": "TENDON_GEOMETRY",
            "object_type": "tendon",
            "object_id": tendon_id,
            "object_name": tendon,
            "target_leg": tendon.endswith("_r_tendon"),
        }
        add(
            **common,
            field="path_and_wraps",
            current_value=tendon_path(model, tendon_id),
            unit_or_normalization="ordered site/wrap topology",
            xml_or_runtime_source="mjModel tendon_adr/tendon_num/wrap_*",
            taxonomy_family="TENDON_PATH_ATTACHMENTS_AND_WRAPS",
            taxonomy_class="E",
            interpretation="attachment and wrap geometry",
        )
        add(
            **common,
            field="spring_reference_length",
            current_value=model.tendon_lengthspring[tendon_id].tolist(),
            unit_or_normalization="m",
            xml_or_runtime_source="mjModel.tendon_lengthspring; XML springlength",
            taxonomy_family="TENDON_PATH_ATTACHMENTS_AND_WRAPS",
            taxonomy_class="E",
            interpretation="reference for tendon spring; stiffness is zero in frozen model",
        )
        add(
            **common,
            field="stiffness_and_damping",
            current_value={"stiffness": float(model.tendon_stiffness[tendon_id]), "damping": float(model.tendon_damping[tendon_id])},
            unit_or_normalization="MuJoCo tendon stiffness/damping units",
            xml_or_runtime_source="mjModel.tendon_stiffness/tendon_damping",
            taxonomy_family="TENDON_ELASTICITY",
            taxonomy_class="D",
            interpretation="both zero: frozen model uses inelastic muscle-tendon semantics",
        )

    for joint_id in range(model.njnt):
        joint = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        dof_start = int(model.jnt_dofadr[joint_id])
        dof_end = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else model.nv
        dof_slice = slice(dof_start, dof_end)
        common = {
            "category": "JOINT",
            "object_type": "joint",
            "object_id": joint_id,
            "object_name": joint,
            "target_leg": joint.endswith("_r"),
        }
        add(
            **common,
            field="damping_stiffness_friction",
            current_value={
                "dof_damping": model.dof_damping[dof_slice].tolist(),
                "joint_stiffness": float(model.jnt_stiffness[joint_id]),
                "dof_frictionloss": model.dof_frictionloss[dof_slice].tolist(),
            },
            unit_or_normalization="joint-type dependent",
            xml_or_runtime_source="mjModel dof_damping/jnt_stiffness/dof_frictionloss",
            taxonomy_family="JOINT_DAMPING_STIFFNESS_FRICTION",
            taxonomy_class="D",
            interpretation="numerically valid stress field but circular for primary learner validation",
        )
        add(
            **common,
            field="armature",
            current_value=model.dof_armature[dof_slice].tolist(),
            unit_or_normalization="joint-type dependent reflected inertia",
            xml_or_runtime_source="mjModel.dof_armature",
            taxonomy_family="JOINT_ARMATURE",
            taxonomy_class="D",
            interpretation="numerical/actuation inertia term",
        )
        add(
            **common,
            field="range_axis_position",
            current_value={"range": model.jnt_range[joint_id].tolist(), "axis": model.jnt_axis[joint_id].tolist(), "pos": model.jnt_pos[joint_id].tolist()},
            unit_or_normalization="rad or m; local-frame vectors",
            xml_or_runtime_source="mjModel jnt_range/jnt_axis/jnt_pos",
            taxonomy_family="JOINT_RANGE_AND_COORDINATE_SEMANTICS",
            taxonomy_class="E",
            interpretation="frozen native domain and coordinate geometry",
        )

    for equality_id in range(model.neq):
        add(
            category="EQUALITY_CONSTRAINT",
            object_type="equality",
            object_id=equality_id,
            object_name=model_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id),
            field="equality_definition",
            current_value={
                "object1": int(model.eq_obj1id[equality_id]),
                "object2": int(model.eq_obj2id[equality_id]),
                "data": model.eq_data[equality_id].tolist(),
                "solref": model.eq_solref[equality_id].tolist(),
                "solimp": model.eq_solimp[equality_id].tolist(),
            },
            unit_or_normalization="constraint-specific",
            xml_or_runtime_source="mjModel.eq_*",
            target_leg=equality_id < 7,
            taxonomy_family="KNEE_PATELLA_EQUALITY_MECHANISM",
            taxonomy_class="E",
            interpretation="frozen coordinate/equality manifold",
        )
    return result


def perturb_model(
    model: mujoco.MjModel,
    family_id: str,
    scale: float,
    structural: dict[int, dict[str, Any]],
) -> list[str]:
    targets: list[str] = []
    if family_id == "SEGMENT_MASS_INERTIA_COUPLED_SCALE":
        for body in RIGHT_LOWER_LIMB_BODIES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0:
                raise RuntimeError(f"missing body {body}")
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale
            targets.append(body)
    elif family_id in {
        "MUSCLE_FORCE_CAPACITY_SCALE",
        "MUSCLE_PASSIVE_FP_MAX_SCALE",
        "BIARTICULAR_FORCE_CAPACITY_SCALE",
        "BIARTICULAR_PASSIVE_FP_MAX_SCALE",
    }:
        selected = []
        for actuator in range(model.nu):
            muscle = model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if not muscle.endswith("_r"):
                continue
            if family_id in {"BIARTICULAR_FORCE_CAPACITY_SCALE", "BIARTICULAR_PASSIVE_FP_MAX_SCALE"} and structural[actuator]["structural_group"] != "HIP_KNEE_BIARTICULAR":
                continue
            selected.append(actuator)
            targets.append(muscle)
        if family_id in {"MUSCLE_FORCE_CAPACITY_SCALE", "BIARTICULAR_FORCE_CAPACITY_SCALE"}:
            model.actuator_gainprm[selected, 2] *= scale
            model.actuator_biasprm[selected, 2] *= scale
        else:
            model.actuator_gainprm[selected, 7] *= scale
            model.actuator_biasprm[selected, 7] *= scale
    else:
        raise RuntimeError(f"unknown perturbation family {family_id}")
    return targets


def fingerprint(prescribed: dict[str, np.ndarray], controlled: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in (
        "tau_truth_nm",
        "actuator_force_n",
        "tendon_length_m",
        "source_equality_residual",
        "inverse_formula_residual_nm",
    ):
        digest.update(np.ascontiguousarray(prescribed[key]).view(np.uint8))
    for key in ("actual_q_rad", "actual_dq_rad_s", "source_equality_residual", "warning_count"):
        digest.update(np.ascontiguousarray(controlled[key]).view(np.uint8))
    return digest.hexdigest()


def run_variant(
    family_id: str,
    perturbation: str,
    scale: float,
    structural: dict[int, dict[str, Any]],
    reference: dict[str, Any],
    prior: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    targets = perturb_model(model, family_id, scale, structural) if scale != 1.0 else []
    model_loaded = model.nu == model.ntendon == 80 and model.neq == 27
    prescribed, prescribed_runtime = prior.prescribed_truth(model, reference)
    controlled, controlled_runtime = prior.controlled_replay(model, reference)
    prescribed_repeat, _ = prior.prescribed_truth(model, reference)
    controlled_repeat, _ = prior.controlled_replay(model, reference)
    first_fingerprint = fingerprint(prescribed, controlled)
    repeat_fingerprint = fingerprint(prescribed_repeat, controlled_repeat)
    q_error_deg = np.degrees(controlled["actual_q_rad"] - reference["q"])
    knee_deg = np.degrees(controlled["actual_q_rad"][:, 1])
    finite = all(
        np.isfinite(array).all()
        for array in (
            prescribed["tau_truth_nm"],
            prescribed["actuator_force_n"],
            prescribed["tendon_length_m"],
            controlled["actual_q_rad"],
            controlled["actual_dq_rad_s"],
        )
    )
    algebraic_residual = max(
        float(np.max(np.abs(prescribed["inverse_formula_residual_nm"]))),
        float(np.max(np.abs(prescribed["decomposition_residual_nm"]))),
        float(np.max(np.abs(prescribed["muscle_reconstruction_residual_nm"]))),
    )
    row = {
        "family_id": family_id,
        "perturbation": perturbation,
        "smoke_scale": scale,
        "smoke_delta_fraction": scale - 1.0,
        "smoke_value_is_scientific_range": False,
        "target_count": len(targets),
        "target_ids": ";".join(targets),
        "reference_id": V2_REFERENCE_ID,
        "reference_sha256": FROZEN_SHA256["v2_reference"],
        "duration_s": float(reference["time_s"][-1]),
        "sample_count": len(reference["time_s"]),
        "tau_truth_hip_rms_nm": float(np.sqrt(np.mean(prescribed["tau_truth_nm"][:, 0] ** 2))),
        "tau_truth_knee_rms_nm": float(np.sqrt(np.mean(prescribed["tau_truth_nm"][:, 1] ** 2))),
        "tau_truth_hip_peak_abs_nm": float(np.max(np.abs(prescribed["tau_truth_nm"][:, 0]))),
        "tau_truth_knee_peak_abs_nm": float(np.max(np.abs(prescribed["tau_truth_nm"][:, 1]))),
        "muscle_force_peak_abs_n": float(np.max(np.abs(prescribed["actuator_force_n"]))),
        "muscle_force_all_finite": bool(np.isfinite(prescribed["actuator_force_n"]).all()),
        "tendon_length_min_m": float(np.min(prescribed["tendon_length_m"])),
        "tendon_length_max_m": float(np.max(prescribed["tendon_length_m"])),
        "tendon_state_all_finite": bool(np.isfinite(prescribed["tendon_length_m"]).all()),
        "source_equality_residual_max": max(
            float(np.max(np.abs(prescribed["source_equality_residual"]))),
            float(np.max(np.abs(controlled["source_equality_residual"]))),
        ),
        "algebraic_residual_max_nm": algebraic_residual,
        "tracking_q_max_abs_deg": float(np.max(np.abs(q_error_deg))),
        "controlled_knee_min_deg": float(np.min(knee_deg)),
        "controlled_knee_max_deg": float(np.max(knee_deg)),
        "warning_count": max(
            int(np.max(prescribed["warning_count"])),
            int(np.max(controlled["warning_count"])),
        ),
        "all_state_finite": finite,
        "model_loads": model_loaded,
        "truth_semantics_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "determinism_first_sha256": first_fingerprint,
        "determinism_repeat_sha256": repeat_fingerprint,
        "deterministic": first_fingerprint == repeat_fingerprint,
        "prescribed_runtime_s": prescribed_runtime["wall_time_s"],
        "controlled_runtime_s": controlled_runtime["wall_time_s"],
        "total_runtime_s": prescribed_runtime["wall_time_s"] + controlled_runtime["wall_time_s"],
    }
    return row, prescribed, controlled


def range_evidence(structural: dict[int, dict[str, Any]], model: mujoco.MjModel) -> dict[str, Any]:
    biarticular = [
        model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        for actuator, mapping in structural.items()
        if mapping["structural_group"] == "HIP_KNEE_BIARTICULAR"
        and model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator).endswith("_r")
    ]
    return {
        "evidence_policy": {
            "LEVEL_1": "official/model field semantics or a model-verified allowed range",
            "LEVEL_2": "public human/musculoskeletal literature may support a range but this project has not verified it",
            "LEVEL_3": "synthetic numerical stress range only",
            "rule": "No scientific cohort range is frozen by this stage.",
        },
        "families": [
            {
                "family_id": "SEGMENT_MASS_INERTIA_COUPLED_SCALE",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "smoke_values": [0.95, 1.0, 1.05],
                "smoke_values_are_scientific_range": False,
            },
            {
                "family_id": "MUSCLE_FORCE_CAPACITY_SCALE",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "observed_native_force_n": [float(np.min(model.actuator_gainprm[:, 2])), float(np.max(model.actuator_gainprm[:, 2]))],
                "smoke_values": [0.95, 1.0, 1.05],
                "smoke_values_are_scientific_range": False,
            },
            {
                "family_id": "BIARTICULAR_FORCE_CAPACITY_SCALE",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "structurally_verified_actuators": biarticular,
                "smoke_values": [0.95, 1.0, 1.05],
                "smoke_values_are_scientific_range": False,
            },
            {
                "family_id": "MUSCLE_PASSIVE_FP_MAX_SCALE",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "observed_native_fpmax": [float(np.min(model.actuator_biasprm[:, 7])), float(np.max(model.actuator_biasprm[:, 7]))],
                "smoke_values": [0.95, 1.0, 1.05],
                "smoke_values_are_scientific_range": False,
            },
            {
                "family_id": "BIARTICULAR_PASSIVE_FP_MAX_SCALE",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "structurally_verified_actuators": biarticular,
                "smoke_values": [0.95, 1.0, 1.05],
                "smoke_values_are_scientific_range": False,
            },
            {
                "family_id": "MUSCLE_ACTIVATION_TIME_CONSTANTS",
                "field_semantics_evidence": "LEVEL_1",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "smoke_values": [],
            },
            {
                "family_id": "LOW_BACKGROUND_ACTIVATION_STATE",
                "field_semantics_evidence": "LEVEL_1_ALLOWED_CONTROL_RANGE_ONLY",
                "range_evidence": "LEVEL_2_NOT_YET_COLLECTED",
                "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
                "smoke_values": [],
                "note": "Negative activation is outside the frozen [0,1] muscle control domain; this stage therefore did not force a nominal/negative/positive perturbation triplet.",
            },
        ],
        "official_sources": OFFICIAL_MUJOCO_SOURCES,
    }


def proposed_schemes(biarticular: list[str]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "final_scheme_frozen": False,
        "all_ranges_frozen": False,
        "schemes": [
            {
                "scheme_id": "SCHEME_A_MINIMAL_INTERPRETABLE",
                "status": "CANDIDATE_AFTER_RANGE_EVIDENCE",
                "subject_factor_count": 6,
                "subject_level_factors": [
                    "FEMUR_MASS_INERTIA_SCALE",
                    "TIBIA_PATELLA_MASS_INERTIA_SCALE",
                    "FOOT_COMPLEX_MASS_INERTIA_SCALE",
                    "HIP_ONLY_PASSIVE_FP_MAX_SCALE",
                    "KNEE_ONLY_PASSIVE_FP_MAX_SCALE",
                    "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
                ],
                "biarticular_underlying_actuators": biarticular,
                "episode_level_nuisance": [],
                "purpose": "first low-dimensional interpretable cohort candidate",
            },
            {
                "scheme_id": "SCHEME_B_MODERATE_HETEROGENEITY",
                "status": "CANDIDATE_AFTER_RANGE_AND_GROUP_REVIEW",
                "subject_factor_count": 8,
                "subject_level_factors": [
                    "FEMUR_MASS_INERTIA_SCALE",
                    "TIBIA_PATELLA_MASS_INERTIA_SCALE",
                    "FOOT_COMPLEX_MASS_INERTIA_SCALE",
                    "HIP_ONLY_FLEXOR_PASSIVE_FP_MAX_SCALE",
                    "HIP_ONLY_EXTENSOR_PASSIVE_FP_MAX_SCALE",
                    "KNEE_ONLY_FLEXOR_PASSIVE_FP_MAX_SCALE",
                    "KNEE_ONLY_EXTENSOR_PASSIVE_FP_MAX_SCALE",
                    "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
                ],
                "biarticular_underlying_actuators": biarticular,
                "episode_level_nuisance": [],
                "purpose": "separate structurally verified muscle transmission families without 80 independent random multipliers",
                "p0_identifiability_rule": "Do not include force-capacity and fpmax scales on the same actuator group as independent P0 factors.",
            },
            {
                "scheme_id": "SCHEME_C_STRESS_RICH",
                "status": "STRESS_ONLY_NOT_PRIMARY_COHORT",
                "subject_factor_count": 8,
                "subject_level_factors": [
                    "RIGHT_LOWER_LIMB_MASS_INERTIA_SCALE",
                    "GLOBAL_RIGHT_TARGET_MUSCLE_FORCE_SCALE",
                    "HIP_KNEE_BIARTICULAR_FORCE_SCALE",
                    "GLOBAL_RIGHT_TARGET_FP_MAX_SCALE",
                    "MUSCLE_OPERATING_RANGE_STRESS_FACTOR",
                    "MUSCLE_ACTIVATION_TIME_CONSTANT_STRESS_FACTOR",
                    "JOINT_PASSIVE_STRESS_FACTOR",
                    "TENDON_ELASTICITY_STRESS_FACTOR",
                ],
                "episode_level_nuisance": ["LOW_BACKGROUND_ACTIVATION_STATE"],
                "purpose": "robustness boundary only; contains Class B/D factors and is not a physiologically motivated cohort",
            },
        ],
        "future_size_options": [
            {"subject_count": 18, "role": "small method-comparison pilot", "warning": "limited split stability for 6-8 factors"},
            {"subject_count": 24, "role": "preferred candidate for first balanced evaluation", "suggested_split": "16 development / 8 held-out subject models"},
            {"subject_count": 30, "role": "larger sensitivity/ablation option", "suggested_split": "20 development / 10 held-out subject models"},
        ],
        "sampling_candidates": [
            "deterministic maximin Latin hypercube with preregistered seed and nominal anchor",
            "deterministic predefined profiles for interpretable ablation",
            "small factorial corners only for selected factors; full factorial is rejected as combinatorial",
        ],
        "freeze_rule": "Every subject manifest, split and generated-model SHA must be frozen before any learner truth outcome is revealed.",
        "recommended_next_evaluation_after_evidence": "Evaluate Scheme A first; do not generate it until factor ranges and group membership receive external evidence review.",
        "p0_force_fpmax_identifiability": "At zero activation, F0 and fpmax multiply the same passive term. Same-group factors are mutually exclusive in primary P0 schemes.",
    }


def subject_manifest_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "MYOLEG_VIRTUAL_SUBJECT_MANIFEST_SCHEMA_V1",
        "title": "MyoLeg heterogeneous musculoskeletal virtual-subject manifest",
        "description": "Schema only. This design stage creates no subject instances.",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subject_id",
            "base_myoleg_model_sha256",
            "v2_reference_sha256",
            "parameter_semantic_version",
            "body_anthropometry_modifications",
            "muscle_strength_modifications",
            "passive_property_modifications",
            "low_activation_state",
            "random_seed",
            "generated_model_sha256",
            "truth_semantic_version",
            "truth_field",
            "subject_specific_reference_normalization",
            "reference_tau_rms_nm",
            "integrity_gate_results",
            "split_role",
            "frozen_before_truth_reveal",
        ],
        "properties": {
            "subject_id": {"type": "string", "minLength": 1},
            "base_myoleg_model_sha256": {"const": FROZEN_SHA256["base_myoleg_model"]},
            "v2_reference_sha256": {"const": FROZEN_SHA256["v2_reference"]},
            "parameter_semantic_version": {"const": PARAMETER_SEMANTIC_VERSION},
            "body_anthropometry_modifications": {"type": "array", "items": {"$ref": "#/$defs/modification"}},
            "muscle_strength_modifications": {"type": "array", "items": {"$ref": "#/$defs/modification"}},
            "passive_property_modifications": {"type": "array", "items": {"$ref": "#/$defs/modification"}},
            "low_activation_state": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["role", "control_or_activation", "value"],
                        "properties": {
                            "role": {"enum": ["EPISODE_NUISANCE", "SUBJECT_BASELINE_WITH_EXTERNAL_JUSTIFICATION"]},
                            "control_or_activation": {"enum": ["ctrl", "act"]},
                            "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                    },
                ]
            },
            "random_seed": {"type": "integer", "minimum": 0},
            "generated_model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "truth_semantic_version": {"const": TRUTH_SEMANTIC_VERSION},
            "truth_field": {"const": TRUTH_FIELD},
            "subject_specific_reference_normalization": {"const": True},
            "reference_tau_rms_nm": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hip", "knee"],
                "properties": {"hip": {"type": "number"}, "knee": {"type": "number"}},
            },
            "integrity_gate_results": {"type": "object", "additionalProperties": {"type": "boolean"}},
            "split_role": {"enum": ["DEVELOPMENT_SUBJECT", "HELD_OUT_SUBJECT"]},
            "frozen_before_truth_reveal": {"const": True},
        },
        "$defs": {
            "modification": {
                "type": "object",
                "additionalProperties": False,
                "required": ["factor_id", "underlying_fields", "target_objects", "value", "unit"],
                "properties": {
                    "factor_id": {"type": "string"},
                    "underlying_fields": {"type": "array", "items": {"type": "string"}},
                    "target_objects": {"type": "array", "items": {"type": "string"}},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
            }
        },
    }


def integrity_gate_spec() -> dict[str, Any]:
    return {
        "protocol": "MYOLEG_COHORT_INTEGRITY_GATES_V1",
        "applies_to": "every future generated subject independently",
        "fail_closed": True,
        "gates": [
            {"gate": "model_loads", "criterion": True},
            {"gate": "all_state_finite", "criterion": True},
            {"gate": "solver_warning_count", "criterion": 0},
            {"gate": "reference_replay_duration_s", "criterion": 24.0},
            {"gate": "reference_sample_count", "criterion": 401},
            {"gate": "source_equality_residual_max", "criterion": INTEGRITY_THRESHOLDS["source_equality_residual_max"]},
            {"gate": "native_knee_rom_deg", "criterion": [0.0, 120.0]},
            {"gate": "muscle_and_tendon_state_finite", "criterion": True},
            {"gate": "pathological_force_explosion_screen", "criterion": {"peak_force_ratio_vs_nominal_max": 2.0, "scope": "small +/-5% smoke only; not a scientific range"}},
            {"gate": "truth_semantic_version", "criterion": TRUTH_SEMANTIC_VERSION},
            {"gate": "truth_field", "criterion": TRUTH_FIELD},
            {"gate": "deterministic_replay", "criterion": "exact repeated fingerprint"},
            {"gate": "subject_specific_reference_normalization", "criterion": True},
        ],
        "unchanged_inputs": {
            "v2_reference_sha256": FROZEN_SHA256["v2_reference"],
            "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
            "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
        },
    }


def write_markdown_audits(
    structural: dict[int, dict[str, Any]],
    model: mujoco.MjModel,
    sensitivity: list[dict[str, Any]],
) -> None:
    right_biarticular = [
        model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        for actuator, mapping in structural.items()
        if mapping["structural_group"] == "HIP_KNEE_BIARTICULAR"
        and model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator).endswith("_r")
    ]
    (ARTIFACT_DIRECTORY / "ANTHROPOMETRY_VARIATION_AUDIT.md").write_text(
        """# Anthropometry variation audit

## Decision

`body_mass` is numerically perturbable, but V1 must scale each selected segment's `body_inertia` by the same mass factor and keep `body_ipos` (COM) fixed. This is a deliberately conservative mass-only scaling assumption; it does not claim geometrically scaled anatomy.

- Femur: `femur_r` mass + inertia together.
- Tibia/knee assembly: `tibia_r` and, if the factor is defined as an assembly, `patella_r` mass + inertia together.
- Foot complex: `talus_r`, `calcn_r`, and `toes_r` mass + inertia together.
- Pelvis: the actual `pelvis` body exists, but it is shared by both limbs. A unilateral target-leg cohort must not silently perturb it; a bilateral/global anthropometry protocol is required first.

COM is held fixed in V1 because independently moving it without a segment-scaling model can be inertially inconsistent. Segment length is `DO NOT PERTURB_IN_V1`: body/joint positions, 384 attachment sites, wrapping geoms, tendon paths and moment arms would all require coordinated reconstruction and revalidation.
""",
        encoding="utf-8",
    )
    (ARTIFACT_DIRECTORY / "MUSCLE_STRENGTH_VARIATION_AUDIT.md").write_text(
        f"""# Muscle strength / force-capacity audit

The frozen XML contains 80 `general` muscle actuators. Each has identical `gainprm[2]` and `biasprm[2]`, originating from the XML `force` field. In MuJoCo 3.6 this is peak active force `F0` in newtons. The observed compiled range is {float(np.min(model.actuator_gainprm[:, 2])):.3f}–{float(np.max(model.actuator_gainprm[:, 2])):.3f} N.

The force law is `actuator_force = -F0 * (FL*FV*activation + FP)`. Consequently, scaling the actual XML `force` field changes both active capacity and the absolute passive force. An “active-only” runtime multiplier would no longer be the native XML field and is not adopted here.

Acceptable high-level factors must map to explicit actuator lists. The inventory derives hip-only, knee-only and hip+knee-spanning membership from the compiled tendon transmission moment matrix over all 401 V2 reference states, not from names. Global right-side scaling targets all 40 `_r` actuators. Group scaling remains provisional until the structural list is manually reviewed.

Official semantics: {OFFICIAL_MUJOCO_SOURCES['muscle_model']}
""",
        encoding="utf-8",
    )
    (ARTIFACT_DIRECTORY / "PASSIVE_PROPERTY_VARIATION_AUDIT.md").write_text(
        f"""# Passive muscle-property audit

## Actual fields

- `fpmax` = `gainprm[7]` / `biasprm[7]`: passive force at `lmax`, relative to peak rest force `F0`. The compiled model range is {float(np.min(model.actuator_biasprm[:, 7])):.6g}–{float(np.max(model.actuator_biasprm[:, 7])):.6g}.
- `range` = `gainprm[0:2]`: scaled muscle operating range used with the transmission range to infer `L0` and `LT`.
- `actuator_lengthrange`: physical range of the tendon transmission in metres. It is not physiological normalized muscle-fiber length.
- `lmin/lmax/vmax/fvmax`: remaining FLV shape parameters.

`fpmax` is the only V1 primary passive-property candidate. It changes passive force magnitude, does not enter the active `FL*FV` term, retains native operating geometry, and nevertheless requires external range evidence. The smoke test changes the corresponding gain and bias slots together so a future XML `fpmax` edit remains representable.

At frozen P0, `F0` and `fpmax` multiply the same passive term. The smoke results confirm exact same-group equivalence. They must therefore be mutually exclusive factors in a P0 cohort; separating force capacity from passive magnitude requires an independently designed nonzero-activation condition.

`range` and the other FLV-shape fields are Class D until muscle-specific evidence exists. They can strongly move operating lengths and may cause force growth near the edge. `actuator_lengthrange` is Class E because it follows geometry rather than constituting a free physiological fiber-length parameter.

Official semantics: {OFFICIAL_MUJOCO_SOURCES['muscle_model']} and {OFFICIAL_MUJOCO_SOURCES['muscle_xml']}
""",
        encoding="utf-8",
    )
    (ARTIFACT_DIRECTORY / "BIARTICULAR_VARIATION_AUDIT.md").write_text(
        f"""# Biarticular coupling variation audit

The structurally verified right-side actuators that have non-zero transmission moment arms about both independent hip flexion and knee flexion over the frozen 401-state path are:

`{', '.join(right_biarticular)}`

Membership is based on `mjData.actuator_moment @ T(q)` with a {MOMENT_ARM_THRESHOLD_M:g} m numerical threshold, not name guessing. A high-level `BIARTICULAR_FORCE_SCALE` maps exactly to `gainprm[2]` and `biasprm[2]` of this frozen list. A passive-coupling factor can analogously map to `gainprm[7]` and `biasprm[7]`.

Therefore native tendon forces can create different hip-knee coupling without adding a synthetic coupling torque equation. For primary P0, use a biarticular `fpmax` factor; the same-group `force` factor is observationally equivalent at zero activation. The group list and scientific scale range still require external/manual review before cohort generation.
""",
        encoding="utf-8",
    )
    (ARTIFACT_DIRECTORY / "LOW_ACTIVATION_VARIABILITY_AUDIT.md").write_text(
        """# Low-activation variability audit

The primary truth condition remains `P0` with zero control and zero initial activation. Low background activation is not a structural musculoskeletal identity by default; it is an episode-level nuisance describing incomplete relaxation during a passive-rehabilitation trial.

- Keep structural subject parameters fixed across every episode for that subject.
- If used, predefine low activation per episode and keep it separate from the subject manifest's structural factors.
- Only treat a fixed subject baseline as identity after independent physiological justification.
- Group-specific activation needs its own preregistered mapping and cannot be selected using downstream learner performance.

The current control domain is `[0,1]`, so a symmetric negative/positive smoke perturbation about zero is invalid. This design stage did not invent a negative activation or freeze a positive range.
""",
        encoding="utf-8",
    )


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_before = verify_frozen_inputs()
    environment = runtime_environment()
    prior = load_prior_builder()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    reference = prior.load_reference(V2_REFERENCE_PATH, "MYOLEG_V2_PRIMARY")
    transformation = json.loads(TRANSFORMATION_PATH.read_text(encoding="utf-8"))
    reference_audit = prior.reference_audit(reference, model)
    if reference_audit["duration_s"] != 24.0 or reference_audit["sample_count"] != 401:
        raise RuntimeError("V2 reference identity changed")
    if reference_audit["q_range_deg"]["knee"][1] > 119.5 + 1.0e-10:
        raise RuntimeError("V2 reference knee maximum changed")

    structural = structural_muscle_map(model, reference, prior)
    inventory = inventory_rows(model, structural)
    inventory_fields = list(inventory[0])
    taxonomy_fields = list(TAXONOMY_ROWS[0])
    write_csv(ARTIFACT_DIRECTORY / "MYOLEG_PARAMETER_INVENTORY.csv", inventory, inventory_fields)
    write_csv(ARTIFACT_DIRECTORY / "PARAMETER_TAXONOMY.csv", TAXONOMY_ROWS, taxonomy_fields)

    family_ids = (
        "SEGMENT_MASS_INERTIA_COUPLED_SCALE",
        "MUSCLE_FORCE_CAPACITY_SCALE",
        "MUSCLE_PASSIVE_FP_MAX_SCALE",
        "BIARTICULAR_FORCE_CAPACITY_SCALE",
        "BIARTICULAR_PASSIVE_FP_MAX_SCALE",
    )
    cache: dict[tuple[str, float], tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    sensitivity: list[dict[str, Any]] = []
    prescribed_by_key: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    controlled_by_key: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    nominal_result: tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
    for family in family_ids:
        for perturbation, scale in (("SMALL_NEGATIVE", 1.0 - SMOKE_SCALE), ("NOMINAL", 1.0), ("SMALL_POSITIVE", 1.0 + SMOKE_SCALE)):
            if scale == 1.0 and nominal_result is not None:
                base_row, prescribed, controlled = nominal_result
                row = dict(base_row)
                row["family_id"] = family
                row["perturbation"] = perturbation
                row["target_count"] = 0
                row["target_ids"] = ""
            else:
                row, prescribed, controlled = run_variant(family, perturbation, scale, structural, reference, prior)
                if scale == 1.0:
                    nominal_result = (row, prescribed, controlled)
            sensitivity.append(row)
            prescribed_by_key[(family, scale)] = prescribed
            controlled_by_key[(family, scale)] = controlled

    nominal_row = next(row for row in sensitivity if row["perturbation"] == "NOMINAL")
    nominal_force_peak = float(nominal_row["muscle_force_peak_abs_n"])
    with np.load(PRIOR_REPLAY_DATASET) as prior_dataset:
        nominal_prior_exact = bool(
            np.array_equal(
                prescribed_by_key[(family_ids[0], 1.0)]["tau_truth_nm"],
                prior_dataset["tau_truth_nm"],
            )
            and np.array_equal(
                controlled_by_key[(family_ids[0], 1.0)]["actual_q_rad"],
                prior_dataset["actual_q_rad"],
            )
        )

    for row in sensitivity:
        row["peak_force_ratio_vs_nominal"] = float(row["muscle_force_peak_abs_n"]) / nominal_force_peak
        gates = {
            "model_loads": bool(row["model_loads"]),
            "finite": bool(row["all_state_finite"] and row["muscle_force_all_finite"] and row["tendon_state_all_finite"]),
            "no_solver_warning": int(row["warning_count"]) == 0,
            "reference_24s_401": float(row["duration_s"]) == 24.0 and int(row["sample_count"]) == 401,
            "equality_integrity": float(row["source_equality_residual_max"]) <= INTEGRITY_THRESHOLDS["source_equality_residual_max"],
            "native_knee_rom": float(row["controlled_knee_min_deg"]) >= -1.0e-10 and float(row["controlled_knee_max_deg"]) <= 120.0 + 1.0e-10,
            "no_force_explosion": float(row["peak_force_ratio_vs_nominal"]) <= INTEGRITY_THRESHOLDS["peak_force_ratio_vs_nominal_max"],
            "truth_semantics": row["truth_semantics_version"] == TRUTH_SEMANTIC_VERSION and float(row["algebraic_residual_max_nm"]) <= INTEGRITY_THRESHOLDS["algebraic_residual_max_nm"],
            "deterministic": bool(row["deterministic"]),
        }
        row["integrity_gate_results_json"] = json.dumps(gates, sort_keys=True, separators=(",", ":"))
        row["all_integrity_gates_pass"] = all(gates.values())
    if not all(bool(row["all_integrity_gates_pass"]) for row in sensitivity):
        raise RuntimeError("one or more small perturbation smoke tests failed closed")
    if not nominal_prior_exact:
        raise RuntimeError("nominal V2 replay no longer matches frozen prior artifact")

    sensitivity_fields = list(sensitivity[0])
    write_csv(
        ARTIFACT_DIRECTORY / "SINGLE_PARAMETER_SENSITIVITY_RESULTS.csv",
        sensitivity,
        sensitivity_fields,
    )

    evidence = range_evidence(structural, model)
    write_json(ARTIFACT_DIRECTORY / "PARAMETER_RANGE_EVIDENCE.json", evidence)
    biarticular = next(
        item["structurally_verified_actuators"]
        for item in evidence["families"]
        if item["family_id"] == "BIARTICULAR_FORCE_CAPACITY_SCALE"
    )
    schemes = proposed_schemes(biarticular)
    write_json(ARTIFACT_DIRECTORY / "PROPOSED_COHORT_SCHEMES.json", schemes)
    write_json(ARTIFACT_DIRECTORY / "PROPOSED_SUBJECT_MANIFEST_SCHEMA.json", subject_manifest_schema())
    gates = integrity_gate_spec()
    family_eligibility = {}
    for family in family_ids:
        family_rows = [row for row in sensitivity if row["family_id"] == family]
        family_eligibility[family] = {
            "numerical_smoke_pass": all(bool(row["all_integrity_gates_pass"]) for row in family_rows),
            "eligible_for_cohort_generation": (
                "ACTIVE_CONDITION_ONLY_OR_MUTUALLY_EXCLUSIVE_IN_P0"
                if family in {"MUSCLE_FORCE_CAPACITY_SCALE", "BIARTICULAR_FORCE_CAPACITY_SCALE"}
                else "YES_AFTER_RANGE_EVIDENCE"
            ),
            "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        }
    gates["tested_family_results"] = family_eligibility
    write_json(ARTIFACT_DIRECTORY / "COHORT_INTEGRITY_GATES.json", gates)
    write_markdown_audits(structural, model, sensitivity)

    family_summary = []
    for family in family_ids:
        rows = {row["perturbation"]: row for row in sensitivity if row["family_id"] == family}
        family_summary.append(
            {
                "family": family,
                "hip_rms_change_negative_pct": 100.0 * (float(rows["SMALL_NEGATIVE"]["tau_truth_hip_rms_nm"]) / float(rows["NOMINAL"]["tau_truth_hip_rms_nm"]) - 1.0),
                "hip_rms_change_positive_pct": 100.0 * (float(rows["SMALL_POSITIVE"]["tau_truth_hip_rms_nm"]) / float(rows["NOMINAL"]["tau_truth_hip_rms_nm"]) - 1.0),
                "knee_rms_change_negative_pct": 100.0 * (float(rows["SMALL_NEGATIVE"]["tau_truth_knee_rms_nm"]) / float(rows["NOMINAL"]["tau_truth_knee_rms_nm"]) - 1.0),
                "knee_rms_change_positive_pct": 100.0 * (float(rows["SMALL_POSITIVE"]["tau_truth_knee_rms_nm"]) / float(rows["NOMINAL"]["tau_truth_knee_rms_nm"]) - 1.0),
            }
        )
    p0_equivalence = {
        "global_force_vs_global_fpmax_negative_within_1e_12_nm": bool(
            np.allclose(
                prescribed_by_key[("MUSCLE_FORCE_CAPACITY_SCALE", 0.95)]["tau_truth_nm"],
                prescribed_by_key[("MUSCLE_PASSIVE_FP_MAX_SCALE", 0.95)]["tau_truth_nm"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "global_force_vs_global_fpmax_positive_within_1e_12_nm": bool(
            np.allclose(
                prescribed_by_key[("MUSCLE_FORCE_CAPACITY_SCALE", 1.05)]["tau_truth_nm"],
                prescribed_by_key[("MUSCLE_PASSIVE_FP_MAX_SCALE", 1.05)]["tau_truth_nm"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "biarticular_force_vs_fpmax_negative_within_1e_12_nm": bool(
            np.allclose(
                prescribed_by_key[("BIARTICULAR_FORCE_CAPACITY_SCALE", 0.95)]["tau_truth_nm"],
                prescribed_by_key[("BIARTICULAR_PASSIVE_FP_MAX_SCALE", 0.95)]["tau_truth_nm"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "biarticular_force_vs_fpmax_positive_within_1e_12_nm": bool(
            np.allclose(
                prescribed_by_key[("BIARTICULAR_FORCE_CAPACITY_SCALE", 1.05)]["tau_truth_nm"],
                prescribed_by_key[("BIARTICULAR_PASSIVE_FP_MAX_SCALE", 1.05)]["tau_truth_nm"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "global_negative_max_abs_difference_nm": float(np.max(np.abs(
            prescribed_by_key[("MUSCLE_FORCE_CAPACITY_SCALE", 0.95)]["tau_truth_nm"]
            - prescribed_by_key[("MUSCLE_PASSIVE_FP_MAX_SCALE", 0.95)]["tau_truth_nm"]
        ))),
        "global_positive_max_abs_difference_nm": float(np.max(np.abs(
            prescribed_by_key[("MUSCLE_FORCE_CAPACITY_SCALE", 1.05)]["tau_truth_nm"]
            - prescribed_by_key[("MUSCLE_PASSIVE_FP_MAX_SCALE", 1.05)]["tau_truth_nm"]
        ))),
        "biarticular_negative_max_abs_difference_nm": float(np.max(np.abs(
            prescribed_by_key[("BIARTICULAR_FORCE_CAPACITY_SCALE", 0.95)]["tau_truth_nm"]
            - prescribed_by_key[("BIARTICULAR_PASSIVE_FP_MAX_SCALE", 0.95)]["tau_truth_nm"]
        ))),
        "biarticular_positive_max_abs_difference_nm": float(np.max(np.abs(
            prescribed_by_key[("BIARTICULAR_FORCE_CAPACITY_SCALE", 1.05)]["tau_truth_nm"]
            - prescribed_by_key[("BIARTICULAR_PASSIVE_FP_MAX_SCALE", 1.05)]["tau_truth_nm"]
        ))),
        "interpretation": "At P0, actuator force is -F0*FP(L); same-group scaling of F0 or fpmax is equivalent to floating-point precision and cannot define two independent cohort factors.",
    }
    if not all(value for key, value in p0_equivalence.items() if key.endswith("_within_1e_12_nm")):
        raise RuntimeError("expected P0 force/fpmax equivalence was not reproduced")
    summary_lines = "\n".join(
        f"- `{item['family']}`: hip RMS {item['hip_rms_change_negative_pct']:+.3f}% / {item['hip_rms_change_positive_pct']:+.3f}%; knee RMS {item['knee_rms_change_negative_pct']:+.3f}% / {item['knee_rms_change_positive_pct']:+.3f}% (−/+ smoke)."
        for item in family_summary
    )
    taxonomy_counts = {label: sum(row["classification"] == label for row in TAXONOMY_ROWS) for label in "ABCDE"}
    report = f"""# MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1

## Outcome

`{OUTCOME}`

This is an offline design audit. It generated no virtual subjects, no candidate landscape, no learner fit, no NN/PINN, no BO run and no robot interaction. The allowed description is **heterogeneous musculoskeletal virtual subjects** or **independent structurally mismatched virtual-patient models**—not real patients, validated digital twins or a clinically representative population.

## Frozen V2 simulation domain

- `MYOLEG_V2_REFERENCE = {V2_REFERENCE_ID}`
- SHA-256: `{FROZEN_SHA256['v2_reference']}`
- duration / samples: `{reference_audit['duration_s']:.1f} s / {reference_audit['sample_count']}`
- knee maximum: `{reference_audit['q_range_deg']['knee'][1]:.12g} deg`
- transformation: `{transformation['formula']}` with scale `{transformation['amplitude_scale']:.16g}`; globally smooth, invertible, no pointwise clipping
- model: native supine MyoLeg SHA `{FROZEN_SHA256['base_myoleg_model']}` with native knee `[0,120] deg`
- limited-125 condition: historical modeling-limit evidence only, not the V2 cohort domain

## Frozen truth semantics

Every future subject must use `{TRUTH_SEMANTIC_VERSION}` / `{TRUTH_FIELD}`:

`r = M*qacc + qfrc_bias - qfrc_passive - qfrc_constraint - qfrc_actuator(P0)`

`tau_truth = T(q)^T*r`

`SUBJECT_SPECIFIC_REFERENCE_NORMALIZATION = true`: each subject gets its own reference hip/knee truth RMS denominator. A nominal-subject denominator is prohibited.

## Actual model inventory and taxonomy

The loaded model has `{model.nbody}` bodies, `{model.njnt}` joints, `{model.ntendon}` tendons, `{model.nu}` muscle actuators, `{model.nsite}` sites, `{model.ngeom}` geoms and `{model.neq}` equality constraints. The CSV inventory records each actual body, muscle field, tendon path/wrap, joint field and equality definition. Taxonomy family counts are A={taxonomy_counts['A']}, B={taxonomy_counts['B']}, C={taxonomy_counts['C']}, D={taxonomy_counts['D']}, E={taxonomy_counts['E']}.

## Q1 — Suitable subject-level fields

Numerically and semantically promising Class A families are coupled segment `body_mass + body_inertia`, grouped muscle XML `force`/compiled gain+bias index 2, and muscle `fpmax`/compiled gain+bias index 7 including structurally verified biarticular groups. Every factor maps to actual frozen model objects. In primary P0, same-group force and fpmax factors are mutually exclusive because their effects are exactly confounded. All still require external scientific range evidence.

## Q2 — Fields not to perturb

Do not independently perturb body/joint geometry, segment length, sites, wrap geoms, tendon path/spring length, actuator lengthrange, native joint range/axis, or knee/patella equality polynomials. Joint damping/stiffness/friction, armature, tendon elasticity and the remaining FLV-shape fields are stress-only rather than primary cohort variables.

## Q3 — Anthropometry without geometry breakage

Scale each selected segment's mass and principal inertia by the same factor and hold COM fixed. Do not change segment length in V1. Pelvis is shared and needs an explicitly bilateral/global protocol rather than silent unilateral scaling.

## Q4 — Force-capacity variability

The actual `force` field is peak active force `F0` in N. Group factors may scale the matching gain and bias slots for explicit structurally identified actuator lists. Because `F0` multiplies both active and passive FLV terms, this is not an active-only manipulation. Under zero activation it is exactly confounded with same-group fpmax scaling, so force capacity belongs in a separate active-condition design or replaces—never accompanies—that fpmax factor.

## Q5 — Passive-property variability

`fpmax` is the primary V1 candidate: passive force at `lmax` relative to `F0`. `actuator_lengthrange` is a tendon-transmission range in metres and is not normalized physiological fiber length. Operating `range` and other FLV-shape parameters stay stress-only pending evidence.

## Q6 — Natural biarticular coupling

Yes. `{', '.join(biarticular)}` span both independent hip and knee coordinates according to the compiled moment matrix over all 401 states. Scaling their actual force or fpmax fields changes native tendon-transmitted coupling without an added torque equation.

## Q7 — Low activation

Treat low activation as episode-level nuisance by default, not subject identity. A fixed subject baseline needs separate physiological justification. No range is frozen here.

## Q8 — Independent 4–8 dimensional design

Yes. Scheme A has 6 factors and Scheme B has 8, using coupled segment inertia plus structurally grouped passive `fpmax` factors rather than true learner `K_hip`, `K_knee`, `B_hip` or `B_knee`. The P0 schemes deliberately exclude simultaneous same-group force/fpmax factors. `TRUTH_LEARNER_PARAMETERIZATION_INDEPENDENCE = PASS`.

## Q9 — Missing evidence

The missing evidence is the magnitude and covariance of human/validated-model variation for segment mass/inertia, muscle/group force capacity, passive fpmax and any low-activation condition; muscle-group membership also needs manual structural/anatomical review. The ±5% values below are Level-3 numerical smoke amplitudes, not physiological ranges.

## Minimal one-family sensitivity checks

{summary_lines}

All {len(sensitivity)} retained rows loaded, remained finite, completed the 24-s/401-sample reference, produced no solver warnings, retained equality and native-ROM integrity, passed exact repeated fingerprints, and stayed below the predeclared smoke-only 2× peak-force screen. Nominal truth and controlled replay arrays exactly match the frozen prior V2 artifact: `{nominal_prior_exact}`. Global force versus global fpmax and biarticular force versus biarticular fpmax agreed within `1e-12 N*m` in both perturbation directions: `{all(value for key, value in p0_equivalence.items() if key.endswith('_within_1e_12_nm'))}`.

## Q10 — What should be evaluated next?

After external range evidence and manual group review, evaluate `SCHEME_A_MINIMAL_INTERPRETABLE` first with a preregistered deterministic design. A 24-subject option (16 development / 8 held-out subject models) is a candidate, not frozen. Freeze every subject manifest and split before truth reveal. Because evidence gaps remain, do **not** execute `MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1` yet.

## Final boundary

- cohort generated: no
- landscape generated: no
- five-parameter fit: no
- NN/PINN trained: no
- BO run: no
- robot/hardware accessed: no
- formal reference / ROM changed: no
- V2 reference / truth semantics changed: no
"""
    (ARTIFACT_DIRECTORY / "MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_REPORT.md").write_text(report, encoding="utf-8")

    source_after = verify_frozen_inputs()
    if source_before["hashes"] != source_after["hashes"]:
        raise RuntimeError("frozen inputs changed during audit")
    metadata_without_artifacts = {
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "evidence_level": "OFFLINE_MODEL_PARAMETER_DESIGN_AND_SMALL_PERTURBATION_SMOKE",
        "runtime_environment": environment,
        "myoleg_v2_reference": {
            "id": V2_REFERENCE_ID,
            "sha256": FROZEN_SHA256["v2_reference"],
            "duration_s": reference_audit["duration_s"],
            "sample_count": reference_audit["sample_count"],
            "knee_max_deg": reference_audit["q_range_deg"]["knee"][1],
            "transformation_identity": transformation,
            "regenerated": False,
        },
        "truth_semantics": {
            "version": TRUTH_SEMANTIC_VERSION,
            "field": TRUTH_FIELD,
            "sha256": FROZEN_SHA256["truth_semantics"],
            "subject_specific_reference_normalization": True,
            "nominal_denominator_for_all_subjects": False,
        },
        "model_inventory_counts": {
            "bodies": model.nbody,
            "joints": model.njnt,
            "tendons": model.ntendon,
            "muscle_actuators": model.nu,
            "sites": model.nsite,
            "geoms": model.ngeom,
            "equalities": model.neq,
            "inventory_rows": len(inventory),
        },
        "taxonomy_counts": taxonomy_counts,
        "truth_learner_parameterization_independence": "PASS",
        "nominal_prior_replay_exact_match": nominal_prior_exact,
        "smoke_test": {
            "families": list(family_ids),
            "row_count": len(sensitivity),
            "all_integrity_gates_pass": all(bool(row["all_integrity_gates_pass"]) for row in sensitivity),
            "all_deterministic": all(bool(row["deterministic"]) for row in sensitivity),
            "amplitude_is_scientific_range": False,
            "p0_force_fpmax_equivalence": p0_equivalence,
        },
        "source_identity_before": source_before["hashes"],
        "source_identity_after": source_after["hashes"],
        "builder_script_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "official_mujoco_sources": OFFICIAL_MUJOCO_SOURCES,
        "cohort_generated": False,
        "candidate_landscape_generated": False,
        "five_parameter_fit": False,
        "nn_trained": False,
        "pinn_trained": False,
        "bo_run": False,
        "robot_connected": False,
        "hardware_accessed": False,
        "formal_reference_modified": False,
        "rom_protocol_modified": False,
        "v2_reference_modified": False,
        "truth_semantics_modified": False,
        "next_stage_executed": False,
    }
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(ARTIFACT_DIRECTORY.iterdir())
        if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    }
    metadata = dict(metadata_without_artifacts)
    metadata["artifact_sha256"] = artifact_hashes
    metadata["design_content_sha256"] = canonical_sha256(
        {
            "taxonomy": TAXONOMY_ROWS,
            "range_evidence": evidence,
            "schemes": schemes,
            "integrity_gates": gates,
        }
    )
    write_json(ARTIFACT_DIRECTORY / "metadata.json", metadata)
    checksum_lines = []
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage_id": STAGE_ID, "outcome": OUTCOME, "artifact_count": len(checksum_lines) + 1, "sensitivity_rows": len(sensitivity), "all_gates_pass": True}, indent=2))


if __name__ == "__main__":
    main()
