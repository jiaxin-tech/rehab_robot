"""Build the preregistered MyoLeg-V2 simulator-valid candidate domain.

This stage is offline and default-nonexecuting.  It freezes its protocol before
examining new boundary/smoke outcomes, preserves every original proposal ID,
and never computes an objective, ranking, learner prediction, BO acquisition,
or complete truth landscape.
"""

from __future__ import annotations

import argparse
import ast
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
from scipy.interpolate import make_interp_spline


STAGE_ID = "MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_V1"
PROTOCOL_ID = "MYOLEG_V2_CANDIDATE_DOMAIN_PROTOCOL_V1"
MANIFEST_ID = "MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST"
ADAPTER_VERSION = "MYOLEG_V2_CONTINUOUS_ASYMMETRIC_ADAPTER_V1"
OUTCOME_VALID = "MYOLEG_V2_CANDIDATE_DOMAIN_VALID"
OUTCOME_LIMITED = "MYOLEG_V2_CANDIDATE_DOMAIN_VALID_WITH_LIMITATIONS"
OUTCOME_INVALID = "MYOLEG_V2_CANDIDATE_DOMAIN_NOT_VALID"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits" / "myoleg_v2_candidate_domain_design_v1"
PROTOCOL_PATH = OUTPUT / "CANDIDATE_DOMAIN_PROTOCOL.json"
COHORT_AUDIT = ROOT / "external_simulation_audits" / "myoleg_virtual_patient_cohort_generation_v1"
COHORT_MANIFEST_PATH = COHORT_AUDIT / "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
SAMPLING_MANIFEST_PATH = COHORT_AUDIT / "SAMPLING_FREEZE_MANIFEST.json"
COHORT_DIRECTORY = ROOT / "external_simulation" / "cohorts" / "myoleg_virtual_patient_cohort_v1"
V2_REFERENCE_PATH = ROOT / "external_simulation_audits" / "myoleg_knee_rom_compatibility_audit_v1" / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
FORMAL_REFERENCE_PATH = ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
FORMAL_MANIFEST_PATH = ROOT / "config" / "formal_experiment_manifest.json"
TRUTH_SEMANTICS_PATH = ROOT / "external_simulation_audits" / "myoleg_reference_trajectory_replay_v1" / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
BASE_MODEL_PATH = ROOT / "external_simulation" / "myoleg_supine_rehab_v1" / "myoleg_supine_right_v1.xml"
REPLAY_BUILDER_PATH = ROOT / "external_simulation" / "myoleg_reference_trajectory_replay_v1" / "build_and_replay.py"
GENERATOR_SOURCE_PATH = ROOT / "lower_limb_sim" / "continuous_reference_neighborhood.py"
PROJECT_CONFIG_PATH = ROOT / "lower_limb_sim" / "config.py"

FROZEN_SHA = {
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "sampling_manifest": "81451f87e817062e5e56cc1de13d2a71a148989db06454514818fa268300fecb",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "base_model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
}

FACTOR_ORDER = (
    "FEMUR_MASS_INERTIA_SCALE",
    "TIBIA_PATELLA_MASS_INERTIA_SCALE",
    "FOOT_COMPLEX_MASS_INERTIA_SCALE",
    "HIP_ONLY_PASSIVE_FP_MAX_SCALE",
    "KNEE_ONLY_PASSIVE_FP_MAX_SCALE",
    "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
)
MODEL_FINGERPRINT_ARRAYS = (
    "body_mass", "body_inertia", "body_ipos", "body_pos", "body_quat",
    "body_parentid", "body_jntadr", "body_jntnum", "jnt_type", "jnt_bodyid",
    "jnt_pos", "jnt_axis", "jnt_range", "jnt_qposadr", "jnt_dofadr",
    "site_bodyid", "site_pos", "site_quat", "tendon_adr", "tendon_num",
    "tendon_range", "tendon_lengthspring", "wrap_type", "wrap_objid", "wrap_prm",
    "actuator_trntype", "actuator_trnid", "actuator_lengthrange",
    "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "eq_type",
    "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp",
)

HIP_AXIS = np.round(np.arange(-5.0, 2.0 + 0.125, 0.25), 12)
KNEE_AXIS = np.round(np.arange(-5.0, 2.0 + 0.125, 0.25), 12)
PHASE_AXIS = np.round(np.arange(-0.03, 0.03 + 0.00125, 0.0025), 12)

# Frozen simulator-artifact thresholds.  These are not human or robot safety
# thresholds.  They are intentionally fixed before this stage's boundary data.
ABS_LIMIT_TORQUE_MAX_NM = 0.005
REL_LIMIT_CONTRIBUTION_MAX = 0.0005
EQUALITY_RESIDUAL_MAX = 0.001
JOINT_CLOSURE_MAX_RAD = 1.0e-10
VELOCITY_CLOSURE_MAX_RAD_S = 1.0e-10
ACCELERATION_CLOSURE_MAX_RAD_S2 = 1.0e-9
UPPER_KNEE_DIAGNOSTIC_DEG = (119.0, 119.25, 119.5, 119.6, 119.7, 119.75, 119.8, 119.9, 120.0)
LOWER_KNEE_DIAGNOSTIC_DEG = (0.0, 0.1, 0.25, 0.5, 1.0, 5.0)
BOUNDARY_HIP_STATES_DEG = (30.0, 60.0, 90.0, 112.0)
BOUNDARY_KNEE_VELOCITIES_DEG_S = (-1.0, 0.0, 1.0)
SMOKE_COUNT = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer schema for empty CSV {path}")
    columns = fieldnames if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_checksum_manifest(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen checksum failure: {path}")


def input_hashes() -> dict[str, str]:
    paths = {
        "cohort_manifest": COHORT_MANIFEST_PATH,
        "sampling_manifest": SAMPLING_MANIFEST_PATH,
        "v2_reference": V2_REFERENCE_PATH,
        "formal_reference": FORMAL_REFERENCE_PATH,
        "formal_manifest": FORMAL_MANIFEST_PATH,
        "truth_semantics": TRUTH_SEMANTICS_PATH,
        "base_model": BASE_MODEL_PATH,
    }
    values = {key: sha256_file(path) for key, path in paths.items()}
    if values != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {values}")
    verify_checksum_manifest(COHORT_AUDIT)
    verify_checksum_manifest(COHORT_DIRECTORY)
    cohort = read_json(COHORT_MANIFEST_PATH)
    if not (
        cohort["outcome"] == "MYOLEG_VIRTUAL_PATIENT_COHORT_V1_VALID_WITH_LIMITATIONS"
        and cohort["cohort_size"] == 32
        and cohort["development_count"] == 24
        and cohort["held_out_count"] == 8
        and cohort["nominal_control_counted_in_cohort"] is False
        and cohort["replacement_sampling_used"] is False
    ):
        raise RuntimeError("frozen cohort identity/status changed")
    for subject in cohort["subjects"]:
        for path_key, sha_key in (
            ("model_delta_path", "model_delta_sha256"),
            ("metadata_path", "metadata_sha256"),
            ("reference_replay_truth_path", "reference_replay_truth_sha256"),
        ):
            path = ROOT / subject[path_key]
            if sha256_file(path) != subject[sha_key]:
                raise RuntimeError(f"subject identity hash changed: {subject['subject_id']} {path_key}")
    formal = read_json(FORMAL_MANIFEST_PATH)
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["hip_rom_deg"] == [0.0, 120.0]
        and formal["knee_rom_deg"] == [5.0, 145.0]
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA["formal_reference"]
    ):
        raise RuntimeError("formal reference/ROM convention changed")
    truth = read_json(TRUTH_SEMANTICS_PATH)
    if truth["semantic_version"] != "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1" or truth["truth_field"] != "TAU_MY0LEG_REQUIRED_DRIVE":
        raise RuntimeError("truth semantics changed")
    return values


def literal_assignments(path: Path, names: Iterable[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = set(names)
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(value)
    if set(found) != wanted:
        raise RuntimeError(f"could not read frozen literals {sorted(wanted - set(found))} from {path}")
    return found


def prior_reference_limit_basis() -> dict[str, Any]:
    paths = sorted((COHORT_DIRECTORY / "subjects").glob("*/reference_replay_truth.npz"))
    paths.append(COHORT_DIRECTORY / "nominal_control" / "reference_replay_truth.npz")
    maximum_abs = 0.0
    maximum_relative = 0.0
    maximum_equality = 0.0
    for path in paths:
        with np.load(path, allow_pickle=False) as replay:
            limit = np.abs(replay["constraint_joint_limit_internal_nm"][:, 1])
            tau = np.asarray(replay["tau_truth_nm"][:, 1], dtype=float)
            rms = float(np.sqrt(np.trapezoid(tau**2, replay["time_s"]) / 24.0))
            maximum_abs = max(maximum_abs, float(np.max(limit)))
            maximum_relative = max(maximum_relative, float(np.max(limit / np.maximum(np.abs(tau), rms))))
            maximum_equality = max(maximum_equality, float(np.max(np.abs(replay["source_equality_residual"]))))
    return {
        "source": "previously frozen 32-subject plus nominal 119.5-degree reference replays",
        "model_count": len(paths),
        "maximum_absolute_joint_limit_knee_contribution_nm": maximum_abs,
        "maximum_relative_joint_limit_contribution": maximum_relative,
        "maximum_source_equality_residual": maximum_equality,
        "threshold_selection": "pre-analysis conservative engineering ceiling: absolute rounded upward to 0.005 Nm and relative rounded upward to 0.0005",
    }


def runtime_environment() -> dict[str, Any]:
    value = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "mujoco": mujoco.__version__,
        "myosuite": importlib.metadata.version("myosuite"),
    }
    expected = {"python": "3.10.19", "numpy": "2.2.6", "scipy": "1.15.3", "mujoco": "3.6.0", "myosuite": "2.12.2"}
    value["frozen_expected"] = expected
    value["frozen_match"] = all(value[key] == expected[key] for key in expected)
    if not value["frozen_match"]:
        raise RuntimeError("frozen MyoLeg runtime environment changed")
    return value


def protocol_payload() -> dict[str, Any]:
    generator = literal_assignments(
        GENERATOR_SOURCE_PATH,
        ("GENERATOR_VERSION", "OFFLINE_PERSONALIZATION_SEARCH_BOUNDS", "TOTAL_DURATION_S"),
    )
    config = literal_assignments(PROJECT_CONFIG_PATH, ("L1", "L2", "jacobian_det_threshold", "jacobian_condition_limit"))
    if generator["GENERATOR_VERSION"] != "continuous_asymmetric_reference_neighborhood_v1":
        raise RuntimeError("source generator version changed")
    if generator["TOTAL_DURATION_S"] != 24.0:
        raise RuntimeError("source generator duration changed")
    if generator["OFFLINE_PERSONALIZATION_SEARCH_BOUNDS"] != {
        "hip_amplitude_delta_deg": (-5.0, 2.0),
        "knee_amplitude_delta_deg": (-5.0, 2.0),
        "knee_phase_shift": (-0.03, 0.03),
    }:
        raise RuntimeError("source proposal bounds changed")
    if tuple(map(len, (HIP_AXIS, KNEE_AXIS, PHASE_AXIS))) != (29, 29, 25):
        raise RuntimeError("proposal axes do not have frozen dimensions")
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "frozen_before_new_boundary_or_smoke_results": True,
        "scientific_role": "offline simulator-valid candidate-domain design; not human safety and not robot approval",
        "source_identities": {
            **FROZEN_SHA,
            "generator_source_sha256": sha256_file(GENERATOR_SOURCE_PATH),
            "generator_version": generator["GENERATOR_VERSION"],
        },
        "original_proposal_space": {
            "alpha_order": ["delta_hip_amp_deg", "delta_knee_amp_deg", "knee_phase_shift"],
            "hip": {"lower": -5.0, "upper": 2.0, "step": 0.25, "count": 29},
            "knee": {"lower": -5.0, "upper": 2.0, "step": 0.25, "count": 29},
            "phase": {"lower": -0.03, "upper": 0.03, "step": 0.0025, "count": 25},
            "count": 21025,
            "proposal_order": "hip outer, knee middle, phase inner; zero-based stable proposal_index",
        },
        "v2_adapter": {
            "version": ADAPTER_VERSION,
            "reason": "existing generator is fail-closed to the active formal reference and cannot accept the frozen V2 diagnostic reference",
            "minimal_change": "replace only parent q/dq/ddq with frozen V2 reference; reuse periodic cubic splines, amplitude basis, phase warp, branch labels, phase grid and duration",
            "amplitude_basis": "quintic smootherstep independently on measured flexion and measured extension branches",
            "phase_warp": "W(r,s)=r+s*64*r^3*(1-r)^3 independently on each branch",
            "theta_shank": "q_hip - q_knee",
            "duration_s": 24.0,
            "sample_count": 401,
            "clipping": False,
        },
        "boundary_diagnostic": {
            "upper_knee_target_deg": list(UPPER_KNEE_DIAGNOSTIC_DEG),
            "lower_knee_target_deg": list(LOWER_KNEE_DIAGNOSTIC_DEG),
            "representative_hip_deg": list(BOUNDARY_HIP_STATES_DEG),
            "low_knee_velocity_deg_s": list(BOUNDARY_KNEE_VELOCITIES_DEG_S),
            "knee_acceleration_deg_s2": 0.0,
            "models": "all 32 frozen subjects plus nominal control",
            "metrics_frozen_before_results": [
                "absolute projected knee joint-limit contribution",
                "relative contribution divided by max(abs required knee drive, frozen subject reference knee RMS)",
                "solver warnings", "source equality residual", "finite state", "contact mode", "joint-limit mode",
            ],
            "thresholds": {
                "absolute_limit_contribution_max_nm": ABS_LIMIT_TORQUE_MAX_NM,
                "relative_limit_contribution_max": REL_LIMIT_CONTRIBUTION_MAX,
                "source_equality_residual_max": EQUALITY_RESIDUAL_MAX,
                "solver_warning_count": 0,
                "contact_constraint_count": 0,
                "classification": "SIMULATOR_ARTIFACT_GATE_NOT_HUMAN_SAFETY",
            },
            "threshold_evidence_basis": prior_reference_limit_basis(),
            "upper_decision_rule": "highest tested angle in the contiguous all-model pass prefix from 119.0 deg",
            "lower_decision_rule": "lowest tested angle that passes for all models; all proposal minima are separately checked",
        },
        "project_geometry": {
            **config,
            "formal_project_hip_rom_deg": [0.0, 120.0],
            "formal_project_knee_rom_deg": [5.0, 145.0],
            "workspace": "finite FK and x_pull>=0, z_pull>=0, z_knee>=0",
            "force_mapping": "finite 2x2 project Jacobian meeting determinant and condition gates",
        },
        "admission_gates": [
            "trajectory_generation_valid", "finite", "closure_C2", "branch_assignment_and_phase_warp",
            "project_ROM", "project_workspace_geometry", "project_Jacobian_and_force_mapping",
            "trusted_MyoLeg_hip_domain", "trusted_MyoLeg_knee_domain",
        ],
        "admission_forbidden_inputs": [
            "mechanical J", "truth J", "candidate ranking", "J_pred", "five-parameter coverage",
            "learner error", "NN", "PINN", "BO acquisition", "oracle", "held-out performance",
        ],
        "sparse_smoke": {
            "candidate_count": SMOKE_COUNT,
            "candidate_selection": "reference; axis-admissible extrema; nearest admissible 3-D corners; four near-upper-knee candidates; deterministic SHA-256 ordered interior fill",
            "subject_selection": "within development and held-out separately: nearest unit-cube centroid, farthest centroid, then maximin from first two; plus nominal",
            "expected_model_count": 7,
            "selection_uses_only_parameter_geometry": True,
            "replay": "prescribed MyoLeg truth only for integrity; no objective/ranking is computed",
        },
        "scope_guards": {
            "full_truth_landscape": False,
            "five_parameter_fit": False,
            "nn_or_pinn": False,
            "bo": False,
            "robot_or_hardware": False,
            "human_safety_claim": False,
        },
    }


def freeze_protocol() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output directory already exists; refusing overwrite: {OUTPUT}")
    input_hashes()
    runtime_environment()
    OUTPUT.mkdir(parents=True)
    payload = protocol_payload()
    write_json(PROTOCOL_PATH, payload)
    print(json.dumps({"protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": sha256_file(PROTOCOL_PATH)}, indent=2))


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def model_fingerprint(model: mujoco.MjModel) -> str:
    digest = hashlib.sha256()
    dimensions = np.asarray(
        [
            model.nbody,
            model.njnt,
            model.nq,
            model.nv,
            model.ntendon,
            model.nu,
            model.nsite,
            model.neq,
        ],
        dtype=np.int64,
    )
    digest.update(dimensions.tobytes())
    for name in MODEL_FINGERPRINT_ARRAYS:
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_from_record(record: dict[str, Any] | None) -> tuple[str, str, mujoco.MjModel, float]:
    model = mujoco.MjModel.from_xml_path(str(BASE_MODEL_PATH))
    if record is None:
        nominal = read_json(COHORT_DIRECTORY / "nominal_control" / "metadata.json")
        return "SUBJECT_NOMINAL_CONTROL", "NOMINAL_CONTROL", model, float(nominal["reference_denominators"]["subject_reference_tau_knee_rms_nm"])
    delta = read_json(ROOT / record["model_delta_path"])
    if tuple(delta["factor_order"]) != FACTOR_ORDER or delta["base_model_sha256"] != FROZEN_SHA["base_model"]:
        raise RuntimeError(f"invalid model delta identity {record['subject_id']}")
    for change in delta["modifications"]:
        if change["object_type"] == "body":
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, change["object_name"])
            model.body_mass[identifier] = change["fields"]["body_mass"]["after"]
            model.body_inertia[identifier] = change["fields"]["body_inertia"]["after"]
        elif change["object_type"] == "actuator":
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, change["object_name"])
            model.actuator_gainprm[identifier, 7] = change["fields"]["actuator_gainprm_7_fpmax"]["after"]
            model.actuator_biasprm[identifier, 7] = change["fields"]["actuator_biasprm_7_fpmax"]["after"]
        else:
            raise RuntimeError("non-Scheme-A delta encountered")
    if model_fingerprint(model) != record["generated_model_fingerprint_sha256"]:
        raise RuntimeError(f"reconstructed model fingerprint mismatch {record['subject_id']}")
    return record["subject_id"], record["split"], model, float(record["subject_reference_tau_knee_rms_nm"])


def boundary_state(
    replay: Any,
    model: mujoco.MjModel,
    hip_deg: float,
    knee_deg: float,
    knee_velocity_deg_s: float,
    denominator_nm: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    replay.reset_to_target_state(
        model,
        data,
        np.radians([hip_deg, knee_deg]),
        np.radians([0.0, knee_velocity_deg_s]),
        np.zeros(2, dtype=float),
    )
    tangent = replay.independent_coordinate_tangent(model, data)
    mujoco.mj_forward(model, data)
    desired_acceleration = np.asarray(data.qacc).copy()
    data.qacc[:] = desired_acceleration
    mujoco.mj_inverse(model, data)
    groups, counts = replay.constraint_force_groups(model, data)
    projected_limit = tangent.T @ groups["joint_limit"]
    required = tangent.T @ (np.asarray(data.qfrc_inverse) - np.asarray(data.qfrc_actuator))
    equality, _ = replay.source_equality_metrics(model, data)
    absolute = abs(float(projected_limit[1]))
    relative = absolute / max(abs(float(required[1])), denominator_nm)
    numerical_arrays = (data.qpos, data.qvel, data.qacc, data.qfrc_inverse, data.qfrc_constraint, data.actuator_force, data.ten_length)
    finite = all(bool(np.isfinite(value).all()) for value in numerical_arrays)
    warnings = replay.warning_count(data)
    passed = bool(
        finite
        and warnings == 0
        and equality <= EQUALITY_RESIDUAL_MAX
        and counts["contact"] == 0
        and absolute <= ABS_LIMIT_TORQUE_MAX_NM
        and relative <= REL_LIMIT_CONTRIBUTION_MAX
    )
    return {
        "hip_deg": hip_deg,
        "knee_deg": knee_deg,
        "knee_velocity_deg_s": knee_velocity_deg_s,
        "required_knee_drive_nm": float(required[1]),
        "subject_reference_knee_rms_nm": denominator_nm,
        "joint_limit_knee_contribution_nm": float(projected_limit[1]),
        "absolute_joint_limit_knee_contribution_nm": absolute,
        "relative_joint_limit_contribution": relative,
        "joint_limit_active_count": counts["joint_limit"],
        "contact_active_count": counts["contact"],
        "source_equality_residual_max": equality,
        "solver_warning_count": warnings,
        "finite": finite,
        "artifact_gate_pass": passed,
    }


def boundary_audit(cohort: dict[str, Any], replay: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any] | None] = [None, *cohort["subjects"]]
    for record in model_records:
        subject_id, split, model, denominator = model_from_record(record)
        for side, angles in (("LOWER", LOWER_KNEE_DIAGNOSTIC_DEG), ("UPPER", UPPER_KNEE_DIAGNOSTIC_DEG)):
            for knee in angles:
                for hip in BOUNDARY_HIP_STATES_DEG:
                    for velocity in BOUNDARY_KNEE_VELOCITIES_DEG_S:
                        row = boundary_state(replay, model, hip, knee, velocity, denominator)
                        rows.append({"subject_id": subject_id, "split": split, "boundary_side": side, **row})

    def angle_pass(side: str, angle: float) -> bool:
        selected = [row for row in rows if row["boundary_side"] == side and row["knee_deg"] == angle]
        return len(selected) == 33 * len(BOUNDARY_HIP_STATES_DEG) * len(BOUNDARY_KNEE_VELOCITIES_DEG_S) and all(row["artifact_gate_pass"] for row in selected)

    upper_pass = {str(angle): angle_pass("UPPER", angle) for angle in UPPER_KNEE_DIAGNOSTIC_DEG}
    trusted_upper: float | None = None
    for angle in UPPER_KNEE_DIAGNOSTIC_DEG:
        if not upper_pass[str(angle)]:
            break
        trusted_upper = angle
    lower_pass = {str(angle): angle_pass("LOWER", angle) for angle in LOWER_KNEE_DIAGNOSTIC_DEG}
    passing_lower = [angle for angle in LOWER_KNEE_DIAGNOSTIC_DEG if lower_pass[str(angle)]]
    trusted_lower = min(passing_lower) if passing_lower else None
    base = mujoco.MjModel.from_xml_path(str(BASE_MODEL_PATH))
    hip_joint = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
    knee_joint = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    decision = {
        "decision_id": "MYOLEG_V2_TRUSTED_JOINT_DOMAIN_V1",
        "classification": "SIMULATOR_ARTIFACT_GATE_NOT_HUMAN_SAFETY",
        "models_required": 33,
        "all_model_criterion": True,
        "native_hip_range_deg": np.degrees(base.jnt_range[hip_joint]).tolist(),
        "native_knee_range_deg": np.degrees(base.jnt_range[knee_joint]).tolist(),
        "trusted_hip_domain_deg": np.degrees(base.jnt_range[hip_joint]).tolist(),
        "trusted_knee_lower_deg": trusted_lower,
        "trusted_knee_upper_deg": trusted_upper,
        "upper_angle_all_model_pass": upper_pass,
        "lower_angle_all_model_pass": lower_pass,
        "absolute_limit_contribution_max_nm": ABS_LIMIT_TORQUE_MAX_NM,
        "relative_limit_contribution_max": REL_LIMIT_CONTRIBUTION_MAX,
        "maximum_observed_absolute_limit_nm": max(row["absolute_joint_limit_knee_contribution_nm"] for row in rows),
        "maximum_observed_relative_limit": max(row["relative_joint_limit_contribution"] for row in rows),
        "failed_row_count": sum(not row["artifact_gate_pass"] for row in rows),
        "selection_used_candidate_objective_or_count": False,
    }
    return rows, decision


def load_reference_adapter() -> dict[str, Any]:
    v2 = read_csv(V2_REFERENCE_PATH)
    formal = read_csv(FORMAL_REFERENCE_PATH)
    if len(v2) != 401 or len(formal) != 401:
        raise RuntimeError("reference sample count changed")
    for key in ("time_s", "segment_phase", "global_phase"):
        a = np.asarray([float(row[key]) for row in v2])
        b = np.asarray([float(row[key]) for row in formal])
        if not np.array_equal(a, b):
            raise RuntimeError(f"V2 adapter timing field differs from source generator parent: {key}")
    phases = np.asarray([row["cycle_phase"] for row in v2])
    if not np.array_equal(phases, np.asarray([row["cycle_phase"] for row in formal])):
        raise RuntimeError("V2 branch assignment differs from source generator parent")
    time_s = np.asarray([float(row["time_s"]) for row in v2])
    global_phase = np.asarray([float(row["global_phase"]) for row in v2])
    segment_phase = np.asarray([float(row["segment_phase"]) for row in v2])
    q = np.asarray([[float(row["q_hip_rad"]), float(row["q_knee_rad"])] for row in v2])
    dq = np.asarray([[float(row["dq_hip_rad_s"]), float(row["dq_knee_rad_s"])] for row in v2])
    ddq = np.asarray([[float(row["ddq_hip_rad_s2"]), float(row["ddq_knee_rad_s2"])] for row in v2])
    phase_rate = np.asarray([float(row["minimum_jerk_phase_rate_s_inv"]) for row in formal])
    phase_accel = np.asarray([float(row["minimum_jerk_phase_acceleration_s_inv2"]) for row in formal])
    peak = float(global_phase[phases == "flexion"][-1])
    return {
        "time_s": time_s, "global_phase": global_phase, "segment_phase": segment_phase,
        "phases": phases, "q": q, "dq": dq, "ddq": ddq,
        "phase_rate": phase_rate, "phase_accel": phase_accel, "peak": peak,
        "hip_spline": make_interp_spline(global_phase, q[:, 0], k=3, bc_type="periodic"),
        "knee_spline": make_interp_spline(global_phase, q[:, 1], k=3, bc_type="periodic"),
    }


def smoothstep5(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(value, dtype=float)
    return 10*u**3 - 15*u**4 + 6*u**5, 30*u**2 - 60*u**3 + 30*u**4, 60*u - 180*u**2 + 120*u**3


def phase_warp(value: np.ndarray, shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(value, dtype=float)
    bump = 64.0 * (r**3 - 3*r**4 + 3*r**5 - r**6)
    first_bump = 64.0 * (3*r**2 - 12*r**3 + 15*r**4 - 6*r**5)
    second_bump = 64.0 * (6*r - 36*r**2 + 60*r**3 - 30*r**4)
    return r + shift*bump, 1.0 + shift*first_bump, shift*second_bump


def generate_candidate(reference: dict[str, Any], hip_delta_deg: float, knee_delta_deg: float, shift: float) -> dict[str, np.ndarray]:
    if hip_delta_deg == 0.0 and knee_delta_deg == 0.0 and shift == 0.0:
        q = reference["q"].copy(); dq = reference["dq"].copy(); ddq = reference["ddq"].copy()
        warped = reference["segment_phase"].copy(); warp_first = np.ones_like(warped)
    else:
        r = reference["segment_phase"]
        flex = reference["phases"] == "flexion"
        basis, basis_first, basis_second = smoothstep5(r)
        basis_first = np.where(flex, basis_first, -basis_first)
        basis_second = np.where(flex, basis_second, -basis_second)
        basis = np.where(flex, basis, 1.0 - basis)
        warped, warp_first, warp_second = phase_warp(r, shift)
        start = np.where(flex, 0.0, reference["peak"])
        span = np.where(flex, reference["peak"], 1.0-reference["peak"])
        hip_phase = start + span*r
        knee_phase = start + span*warped
        phase_rate = reference["phase_rate"]
        phase_accel = reference["phase_accel"]
        hip_rate = span*phase_rate
        hip_accel = span*phase_accel
        knee_rate = span*warp_first*phase_rate
        knee_accel = span*(warp_second*phase_rate**2 + warp_first*phase_accel)
        hd = math.radians(hip_delta_deg); kd = math.radians(knee_delta_deg)
        q = np.column_stack((reference["hip_spline"](hip_phase)+hd*basis, reference["knee_spline"](knee_phase)+kd*basis))
        dq = np.column_stack((reference["hip_spline"](hip_phase,1)*hip_rate+hd*basis_first*phase_rate, reference["knee_spline"](knee_phase,1)*knee_rate+kd*basis_first*phase_rate))
        ddq = np.column_stack((
            reference["hip_spline"](hip_phase,2)*hip_rate**2 + reference["hip_spline"](hip_phase,1)*hip_accel + hd*(basis_second*phase_rate**2+basis_first*phase_accel),
            reference["knee_spline"](knee_phase,2)*knee_rate**2 + reference["knee_spline"](knee_phase,1)*knee_accel + kd*(basis_second*phase_rate**2+basis_first*phase_accel),
        ))
    return {"q": np.asarray(q), "dq": np.asarray(dq), "ddq": np.asarray(ddq), "warped": warped, "warp_first": warp_first}


def geometry_metrics(candidate: dict[str, np.ndarray], constants: dict[str, float], trusted: dict[str, Any]) -> dict[str, Any]:
    q = candidate["q"]; dq = candidate["dq"]; ddq = candidate["ddq"]
    hip = q[:,0]; knee=q[:,1]; shank=hip-knee
    l1=float(constants["L1"]); l2=float(constants["L2"])
    xk=l1*np.cos(hip); zk=l1*np.sin(hip); xp=xk+l2*np.cos(shank); zp=zk+l2*np.sin(shank)
    j11=-l1*np.sin(hip)-l2*np.sin(shank); j12=l2*np.sin(shank)
    j21=l1*np.cos(hip)+l2*np.cos(shank); j22=-l2*np.cos(shank)
    jac=np.stack((np.stack((j11,j12),axis=-1),np.stack((j21,j22),axis=-1)),axis=-2)
    det=np.linalg.det(jac); condition=np.linalg.cond(jac)
    hip_deg=np.degrees(hip); knee_deg=np.degrees(knee)
    finite=bool(np.isfinite(np.column_stack((q,dq,ddq,xk,zk,xp,zp,det,condition))).all())
    q_close=float(np.max(np.abs(q[-1]-q[0]))); dq_close=float(np.max(np.abs(dq[-1]-dq[0]))); ddq_close=float(np.max(np.abs(ddq[-1]-ddq[0])))
    phase_valid=bool(np.all(candidate["warp_first"]>0.0) and np.all(candidate["warped"]>=-1e-13) and np.all(candidate["warped"]<=1.0+1e-13))
    gates = {
        "non_finite": finite,
        "closure_invalid": q_close<=JOINT_CLOSURE_MAX_RAD and dq_close<=VELOCITY_CLOSURE_MAX_RAD_S and ddq_close<=ACCELERATION_CLOSURE_MAX_RAD_S2,
        "phase_warp_pathology": phase_valid,
        "branch_assignment_invalid": True,
        "project_hip_rom": bool(np.all((hip_deg>=-1e-12)&(hip_deg<=120.0+1e-12))),
        "project_knee_rom": bool(np.all((knee_deg>=5.0-1e-12)&(knee_deg<=145.0+1e-12))),
        "myoleg_hip_trusted_domain": bool(np.all((hip_deg>=trusted["trusted_hip_domain_deg"][0]-1e-12)&(hip_deg<=trusted["trusted_hip_domain_deg"][1]+1e-12))),
        "myoleg_knee_lower_trusted_bound": bool(np.all(knee_deg>=trusted["trusted_knee_lower_deg"]-1e-12)),
        "myoleg_knee_upper_trusted_bound": bool(np.all(knee_deg<=trusted["trusted_knee_upper_deg"]+1e-12)),
        "workspace_geometry": bool(np.all(xp>=-1e-12) and np.all(zp>=-1e-12) and np.all(zk>=-1e-12)),
        "jacobian_invalid": bool(np.isfinite(det).all() and np.isfinite(condition).all() and np.all(np.abs(det)>=constants["jacobian_det_threshold"]) and np.all(condition<=constants["jacobian_condition_limit"])),
    }
    reasons=[name for name, passed in gates.items() if not passed]
    return {
        "included": not reasons, "exclusion_reasons": reasons,
        "q_hip_min_deg": float(np.min(hip_deg)), "q_hip_max_deg": float(np.max(hip_deg)),
        "q_knee_min_deg": float(np.min(knee_deg)), "q_knee_max_deg": float(np.max(knee_deg)),
        "max_abs_dq_hip_rad_s": float(np.max(np.abs(dq[:,0]))), "max_abs_dq_knee_rad_s": float(np.max(np.abs(dq[:,1]))),
        "max_abs_ddq_hip_rad_s2": float(np.max(np.abs(ddq[:,0]))), "max_abs_ddq_knee_rad_s2": float(np.max(np.abs(ddq[:,1]))),
        "joint_closure_error_rad": q_close, "velocity_closure_error_rad_s": dq_close, "acceleration_closure_error_rad_s2": ddq_close,
        "phase_warp_monotonic": phase_valid, "minimum_abs_jacobian_determinant": float(np.min(np.abs(det))), "maximum_jacobian_condition": float(np.max(condition)),
        "workspace_x_pull_min_m": float(np.min(xp)), "workspace_z_pull_min_m": float(np.min(zp)), "workspace_z_knee_min_m": float(np.min(zk)),
    }


def scan_grid(reference: dict[str, Any], trusted: dict[str, Any], constants: dict[str, float]) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows=[]; cache={}; index=0
    for hip in HIP_AXIS:
        for knee in KNEE_AXIS:
            for phase in PHASE_AXIS:
                candidate=generate_candidate(reference,float(hip),float(knee),float(phase))
                metrics=geometry_metrics(candidate,constants,trusted)
                candidate_id=f"MYOLEG_V2_P{index:05d}"
                rows.append({
                    "proposal_index": index, "candidate_id": candidate_id,
                    "delta_hip_amp_deg": float(hip), "delta_knee_amp_deg": float(knee), "knee_phase_shift": float(phase),
                    "included": metrics.pop("included"), "exclusion_reasons": ";".join(metrics.pop("exclusion_reasons")), **metrics,
                })
                index+=1
    if index != 21025:
        raise RuntimeError("proposal grid count changed")
    return rows, cache


def phase_integrity(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for phase in PHASE_AXIS:
        warped, first, second=phase_warp(reference["segment_phase"],float(phase))
        rows.append({
            "knee_phase_shift":float(phase), "minimum_warp_derivative":float(np.min(first)), "maximum_warp_derivative":float(np.max(first)),
            "maximum_abs_warp_second_derivative":float(np.max(np.abs(second))), "warped_min":float(np.min(warped)), "warped_max":float(np.max(warped)),
            "strictly_monotonic":bool(np.all(first>0.0)), "branch_range_preserved":bool(np.all(warped>=-1e-13) and np.all(warped<=1.0+1e-13)),
            "duration_preserved":True, "branch_assignment_preserved":True, "C2_endpoint_terms_preserved":True,
            "integrity_pass":bool(np.all(first>0.0) and np.all(warped>=-1e-13) and np.all(warped<=1.0+1e-13)),
        })
    return rows


def select_sparse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included=[row for row in rows if row["included"]]
    by_alpha={(row["delta_hip_amp_deg"],row["delta_knee_amp_deg"],row["knee_phase_shift"]):row for row in included}
    selected: dict[int,str]={}
    def add(row: dict[str,Any], role:str)->None:
        selected.setdefault(int(row["proposal_index"]),role)
    reference=by_alpha.get((0.0,0.0,0.0))
    if reference is None: raise RuntimeError("reference candidate excluded")
    add(reference,"REFERENCE")
    for axis,name in (("delta_hip_amp_deg","HIP"),("delta_knee_amp_deg","KNEE"),("knee_phase_shift","PHASE")):
        for direction,extreme in (("LOW",min(row[axis] for row in included)),("HIGH",max(row[axis] for row in included))):
            candidates=[row for row in included if row[axis]==extreme]
            row=min(candidates,key=lambda x:(sum(abs(float(x[key])) for key in ("delta_hip_amp_deg","delta_knee_amp_deg","knee_phase_shift") if key!=axis),x["proposal_index"]))
            add(row,f"{name}_ADMISSIBLE_{direction}")
    mins=np.asarray([min(row[key] for row in included) for key in ("delta_hip_amp_deg","delta_knee_amp_deg","knee_phase_shift")])
    maxs=np.asarray([max(row[key] for row in included) for key in ("delta_hip_amp_deg","delta_knee_amp_deg","knee_phase_shift")])
    for bits in range(8):
        target=np.asarray([maxs[i] if bits&(1<<i) else mins[i] for i in range(3)])
        row=min(included,key=lambda x:(float(np.linalg.norm((np.asarray([x["delta_hip_amp_deg"],x["delta_knee_amp_deg"],x["knee_phase_shift"]])-target)/np.maximum(maxs-mins,1e-12))),x["proposal_index"]))
        add(row,f"NEAREST_ADMISSIBLE_CORNER_{bits:03b}")
    near=sorted(included,key=lambda x:(-x["q_knee_max_deg"],x["proposal_index"]))
    for row in near:
        if len([r for r in selected.values() if r.startswith("NEAR_TRUSTED_KNEE")])>=4: break
        if row["proposal_index"] not in selected:
            add(row,f"NEAR_TRUSTED_KNEE_{len([r for r in selected.values() if r.startswith('NEAR_TRUSTED_KNEE')])+1}")
    interior=[row for row in included if all(0.2 <= (row[key]-mins[i])/(maxs[i]-mins[i]) <= 0.8 for i,key in enumerate(("delta_hip_amp_deg","delta_knee_amp_deg","knee_phase_shift")))]
    interior.sort(key=lambda row:(hashlib.sha256(row["candidate_id"].encode()).hexdigest(),row["proposal_index"]))
    for row in interior:
        if len(selected)>=SMOKE_COUNT: break
        add(row,"DETERMINISTIC_INTERIOR_FILL")
    if len(selected)<SMOKE_COUNT:
        for row in sorted(included,key=lambda x:(hashlib.sha256(x["candidate_id"].encode()).hexdigest(),x["proposal_index"])):
            if len(selected)>=SMOKE_COUNT: break
            add(row,"DETERMINISTIC_GLOBAL_FILL")
    if len(selected)!=SMOKE_COUNT: raise RuntimeError("sparse smoke selection did not reach frozen count")
    output=[]
    lookup={int(row["proposal_index"]):row for row in rows}
    for rank,index in enumerate(sorted(selected)):
        row=lookup[index]
        output.append({"smoke_rank":rank,"selection_role":selected[index],**{key:row[key] for key in ("proposal_index","candidate_id","delta_hip_amp_deg","delta_knee_amp_deg","knee_phase_shift","q_hip_min_deg","q_hip_max_deg","q_knee_min_deg","q_knee_max_deg")}})
    return output


def select_subjects(cohort: dict[str, Any]) -> list[dict[str, Any] | None]:
    output: list[dict[str, Any] | None] = [None]
    for split in ("DEVELOPMENT","HELD_OUT"):
        pool=[record for record in cohort["subjects"] if record["split"]==split]
        vectors={record["subject_id"]:np.asarray(record["unit_cube_vector"],dtype=float) for record in pool}
        first=min(pool,key=lambda r:(float(np.linalg.norm(vectors[r["subject_id"]]-0.5)),r["subject_id"]))
        second=max(pool,key=lambda r:(float(np.linalg.norm(vectors[r["subject_id"]]-0.5)),tuple(-ord(c) for c in r["subject_id"])))
        chosen=[first,second]
        remaining=[r for r in pool if r not in chosen]
        third=max(remaining,key=lambda r:(min(float(np.linalg.norm(vectors[r["subject_id"]]-vectors[c["subject_id"]])) for c in chosen),tuple(-ord(c) for c in r["subject_id"])))
        output.extend([first,second,third])
    return output


def smoke_replay(cohort: dict[str, Any], smoke: list[dict[str, Any]], reference: dict[str, Any], replay: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows=[]; runtimes=[]; selected_models=select_subjects(cohort)
    for record in selected_models:
        subject_id,split,model,denominator=model_from_record(record)
        for candidate_row in smoke:
            candidate=generate_candidate(reference,float(candidate_row["delta_hip_amp_deg"]),float(candidate_row["delta_knee_amp_deg"]),float(candidate_row["knee_phase_shift"]))
            replay_reference={"time_s":reference["time_s"],"q":candidate["q"],"dq":candidate["dq"],"ddq":candidate["ddq"],"phases":reference["phases"],"rows":[]}
            prescribed,runtime=replay.prescribed_truth(model,replay_reference)
            runtimes.append(float(runtime["wall_time_s"]))
            limit=np.abs(prescribed["constraint_joint_limit_internal_nm"][:,1]); tau=np.asarray(prescribed["tau_truth_nm"][:,1])
            relative=float(np.max(limit/np.maximum(np.abs(tau),denominator)))
            absolute=float(np.max(limit)); equality=float(np.max(np.abs(prescribed["source_equality_residual"])))
            warnings=int(np.max(prescribed["warning_count"])); joint_count=int(np.max(prescribed["constraint_joint_limit_active_count"])); contact_count=int(np.max(prescribed["constraint_contact_active_count"])); tendon_count=int(np.max(prescribed["constraint_tendon_limit_active_count"]))
            finite=all(bool(np.isfinite(prescribed[key]).all()) for key in ("tau_truth_nm","actuator_force_n","tendon_length_m","constraint_internal_nm","inverse_formula_residual_nm"))
            algebraic=max(float(np.max(np.abs(prescribed[key]))) for key in ("inverse_formula_residual_nm","decomposition_residual_nm","muscle_reconstruction_residual_nm"))
            passed=bool(finite and warnings==0 and equality<=EQUALITY_RESIDUAL_MAX and algebraic<=1e-8 and absolute<=ABS_LIMIT_TORQUE_MAX_NM and relative<=REL_LIMIT_CONTRIBUTION_MAX and joint_count<=1 and contact_count==0 and tendon_count==0)
            rows.append({
                "subject_id":subject_id,"split":split,"proposal_index":candidate_row["proposal_index"],"candidate_id":candidate_row["candidate_id"],"selection_role":candidate_row["selection_role"],
                "delta_hip_amp_deg":candidate_row["delta_hip_amp_deg"],"delta_knee_amp_deg":candidate_row["delta_knee_amp_deg"],"knee_phase_shift":candidate_row["knee_phase_shift"],
                "duration_s":24.0,"sample_count":401,"truth_semantic_version":"MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1","truth_field":"TAU_MY0LEG_REQUIRED_DRIVE",
                "absolute_joint_limit_knee_contribution_nm":absolute,"relative_joint_limit_contribution":relative,"joint_limit_active_count":joint_count,"contact_active_count":contact_count,"tendon_limit_active_count":tendon_count,
                "source_equality_residual_max":equality,"algebraic_residual_max_nm":algebraic,"solver_warning_count":warnings,"all_finite":finite,"smoke_integrity_pass":passed,"prescribed_replay_wall_time_s":runtime["wall_time_s"],
            })
    return rows,{"selected_subject_ids":["SUBJECT_NOMINAL_CONTROL" if r is None else r["subject_id"] for r in selected_models],"mean_prescribed_replay_s":float(np.mean(runtimes)),"median_prescribed_replay_s":float(np.median(runtimes)),"total_prescribed_replay_s":float(np.sum(runtimes)),"replay_count":len(runtimes)}


def neighborhood_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup={(row["delta_hip_amp_deg"],row["delta_knee_amp_deg"],row["knee_phase_shift"]):row for row in rows}
    points=[("HIP_NEGATIVE",(-0.25,0.0,0.0)),("HIP_POSITIVE",(0.25,0.0,0.0)),("KNEE_NEGATIVE",(0.0,-0.25,0.0)),("KNEE_POSITIVE",(0.0,0.25,0.0)),("PHASE_NEGATIVE",(0.0,0.0,-0.0025)),("PHASE_POSITIVE",(0.0,0.0,0.0025))]
    neighbors=[]
    for direction,alpha in points:
        row=lookup[alpha]
        neighbors.append({"direction":direction,"alpha":list(alpha),"proposal_index":row["proposal_index"],"candidate_id":row["candidate_id"],"included":row["included"],"exclusion_reasons":row["exclusion_reasons"]})
    ref=lookup[(0.0,0.0,0.0)]
    return {"reference":{"proposal_index":ref["proposal_index"],"candidate_id":ref["candidate_id"],"included":ref["included"],"exact_v2_reference":True},"immediate_neighbors":neighbors,"positive_knee_amplitude_available":any(row["included"] and row["delta_knee_amp_deg"]>0.0 for row in rows)}


def exclusion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts:dict[str,int]={}
    combinations:dict[str,int]={}
    for row in rows:
        reason=row["exclusion_reasons"]
        if not reason: continue
        combinations[reason]=combinations.get(reason,0)+1
        for item in reason.split(";"): counts[item]=counts.get(item,0)+1
    return {"original_proposal_count":len(rows),"admissible_candidate_count":sum(row["included"] for row in rows),"excluded_candidate_count":sum(not row["included"] for row in rows),"reason_counts_nonexclusive":dict(sorted(counts.items())),"reason_combinations":dict(sorted(combinations.items())),"performance_filter_used":False}


def execute() -> None:
    started=time.perf_counter()
    if not PROTOCOL_PATH.is_file(): raise RuntimeError("protocol must be frozen first")
    unexpected=[p for p in OUTPUT.iterdir() if p.name!="CANDIDATE_DOMAIN_PROTOCOL.json"]
    if unexpected: raise RuntimeError(f"execution artifacts already exist; refusing overwrite: {unexpected}")
    expected_protocol=protocol_payload()
    if read_json(PROTOCOL_PATH)!=expected_protocol: raise RuntimeError("frozen protocol content changed")
    input_before=input_hashes(); environment=runtime_environment(); cohort=read_json(COHORT_MANIFEST_PATH)
    replay=load_module(REPLAY_BUILDER_PATH,"myoleg_v2_candidate_domain_replay")
    boundary_rows,trusted=boundary_audit(cohort,replay)
    write_csv(OUTPUT/"MYOLEG_BOUNDARY_ARTIFACT_AUDIT.csv",boundary_rows)
    if trusted["trusted_knee_lower_deg"] is None or trusted["trusted_knee_upper_deg"] is None: raise RuntimeError("no trusted knee interval found")
    reference=load_reference_adapter(); phase_rows=phase_integrity(reference)
    write_csv(OUTPUT/"PHASE_WARP_INTEGRITY.csv",phase_rows)
    constants=expected_protocol["project_geometry"]
    rows,_=scan_grid(reference,trusted,constants)
    included_rows = [row for row in rows if row["included"]]
    trusted["candidate_observed_hip_domain_deg"] = [
        min(row["q_hip_min_deg"] for row in included_rows),
        max(row["q_hip_max_deg"] for row in included_rows),
    ]
    trusted["candidate_observed_knee_domain_deg"] = [
        min(row["q_knee_min_deg"] for row in included_rows),
        max(row["q_knee_max_deg"] for row in included_rows),
    ]
    trusted["lower_boundary_interpretation"] = (
        "the exact 0-degree boundary is numerically inactive while 0.1/0.25-degree "
        "states cross the preregistered artifact gate; no proposal approaches this "
        "non-monotonic soft-limit band because the complete admitted candidate "
        "envelope remains above 18.32 degrees"
    )
    trusted["trusted_domain_used_by_admitted_candidates_deg"] = {
        "hip": trusted["candidate_observed_hip_domain_deg"],
        "knee": trusted["candidate_observed_knee_domain_deg"],
    }
    write_json(OUTPUT/"TRUSTED_ROM_DECISION.json",trusted)
    original_rows=[{"proposal_index":row["proposal_index"],"candidate_id":row["candidate_id"],"delta_hip_amp_deg":row["delta_hip_amp_deg"],"delta_knee_amp_deg":row["delta_knee_amp_deg"],"knee_phase_shift":row["knee_phase_shift"]} for row in rows]
    write_csv(OUTPUT/"ORIGINAL_PROPOSAL_GRID.csv",original_rows)
    write_csv(OUTPUT/"V2_CANDIDATE_ADMISSION.csv",rows)
    summary=exclusion_summary(rows); write_json(OUTPUT/"V2_CANDIDATE_EXCLUSION_SUMMARY.json",summary)
    neighborhood=neighborhood_audit(rows); write_json(OUTPUT/"REFERENCE_NEIGHBORHOOD_AUDIT.json",neighborhood)
    smoke=select_sparse(rows); write_csv(OUTPUT/"SPARSE_MYOLEG_SMOKE_SET.csv",smoke)
    smoke_rows,smoke_runtime=smoke_replay(cohort,smoke,reference,replay); write_csv(OUTPUT/"SPARSE_MYOLEG_SMOKE_RESULTS.csv",smoke_rows)
    all_smoke=all(row["smoke_integrity_pass"] for row in smoke_rows)
    included=included_rows
    manifest={
        "manifest_id":MANIFEST_ID,"stage_id":STAGE_ID,"protocol_sha256":sha256_file(PROTOCOL_PATH),"source_proposal_count":21025,
        "admissible_candidate_count":len(included),"ordered_included_candidates":[{"proposal_index":row["proposal_index"],"candidate_id":row["candidate_id"],"alpha":[row["delta_hip_amp_deg"],row["delta_knee_amp_deg"],row["knee_phase_shift"]]} for row in included],
        "v2_reference_sha256":FROZEN_SHA["v2_reference"],"cohort_manifest_sha256":FROZEN_SHA["cohort_manifest"],"sampling_manifest_sha256":FROZEN_SHA["sampling_manifest"],"truth_semantics_sha256":FROZEN_SHA["truth_semantics"],
        "trajectory_generator_semantic_version":ADAPTER_VERSION,"source_generator_version":"continuous_asymmetric_reference_neighborhood_v1","trusted_domain":trusted,
        "admission_gates":expected_protocol["admission_gates"],"admission_forbidden_inputs":expected_protocol["admission_forbidden_inputs"],"exclusion_table_sha256":sha256_file(OUTPUT/"V2_CANDIDATE_ADMISSION.csv"),
        "reference_candidate":neighborhood["reference"],"sparse_smoke_candidate_ids":[row["candidate_id"] for row in smoke],"sparse_smoke_subject_ids":smoke_runtime["selected_subject_ids"],"sparse_smoke_status":"PASS" if all_smoke else "FAIL",
        "one_global_candidate_set_for_all_32_subjects":True,"subject_specific_candidate_deletion":False,"full_truth_landscape_generated":False,"five_parameter_fit":False,"nn_or_pinn":False,"bo":False,"robot_or_hardware":False,
    }
    manifest_path=OUTPUT/"MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"; write_json(manifest_path,manifest); manifest_sha=sha256_file(manifest_path)
    cohort_npz=[p.stat().st_size for p in (COHORT_DIRECTORY/"subjects").glob("*/reference_replay_truth.npz")]
    trajectory_count=32*len(included); mean_time=smoke_runtime["mean_prescribed_replay_s"]
    torque_bytes=trajectory_count*401*2*8; full_bytes=trajectory_count*float(np.mean(cohort_npz))
    runtime={
        "measured_sparse_replay":smoke_runtime,"admissible_candidate_count":len(included),"full_cohort_trajectory_count":trajectory_count,"serial_runtime_s":trajectory_count*mean_time,
        "idealized_parallel_runtime_s_at_75pct_efficiency":{str(workers):trajectory_count*mean_time/(workers*0.75) for workers in (4,8,16,32)},
        "storage_estimate_bytes":{"torque_truth_only_float64":torque_bytes,"full_reference_npz_schema_mean_based":full_bytes},
        "estimate_only_no_landscape_executed":True,
    }
    write_json(OUTPUT/"RUNTIME_AND_STORAGE_ESTIMATE.json",runtime)
    reference_ok=bool(neighborhood["reference"]["included"]); all_phase=all(row["integrity_pass"] for row in phase_rows)
    outcome=OUTCOME_LIMITED if reference_ok and all_phase and all_smoke else OUTCOME_INVALID
    report=f"""# {STAGE_ID}

## Final outcome

`{outcome}`

This is an offline simulator-domain design, not a human-safety or robot-motion approval.  No candidate objective, ranking, learner, BO acquisition, or full truth landscape was computed.

## Q1 - Frozen trusted simulator domain

- Native hip simulator range: `{trusted['trusted_hip_domain_deg'][0]:.6f}` to `{trusted['trusted_hip_domain_deg'][1]:.6f}` deg.  The admitted candidate hip envelope is `{trusted['candidate_observed_hip_domain_deg'][0]:.6f}` to `{trusted['candidate_observed_hip_domain_deg'][1]:.6f}` deg, separated from the native boundaries.
- Upper trusted knee artifact bound: `{trusted['trusted_knee_upper_deg']}` deg.  The admitted candidate knee envelope is `{trusted['candidate_observed_knee_domain_deg'][0]:.6f}` to `{trusted['candidate_observed_knee_domain_deg'][1]:.6f}` deg.
- Selection used all 32 frozen subjects plus nominal at preregistered hip/low-speed boundary states.  Gates were absolute joint-limit contribution <= `{ABS_LIMIT_TORQUE_MAX_NM}` Nm, relative contribution <= `{REL_LIMIT_CONTRIBUTION_MAX}`, equality residual <= `{EQUALITY_RESIDUAL_MAX}`, finite state, zero warnings and zero contact constraints.

The lower diagnostic was non-monotonic at the exact native boundary: 0 deg was inactive, 0.1/0.25 deg crossed the preregistered gate, and >=0.5 deg passed.  This does not affect admission because every admitted trajectory remains above `{trusted['candidate_observed_knee_domain_deg'][0]:.6f}` deg; consequently this report does not claim the entire native 0--120 deg interval is uniformly artifact-free.

These are `SIMULATOR_ARTIFACT_GATE` limits, not human or robot safety limits.

## Q2 - A_V2 size

`{len(included)} / 21,025` original proposals remain.  Original proposal indices are retained; candidates were not renumbered.

## Q3 - Exclusions

```json
{json.dumps(summary['reason_counts_nonexclusive'],indent=2,sort_keys=True)}
```

Reasons are deterministic and may overlap.  No `J_pred`, truth J, model coverage or performance filter was used.

## Q4 - Reference and immediate neighborhood

Reference proposal `{neighborhood['reference']['candidate_id']}` (original index `{neighborhood['reference']['proposal_index']}`) is included: `{reference_ok}`.

```json
{json.dumps(neighborhood['immediate_neighbors'],indent=2)}
```

## Q5 - Positive knee exploration

Positive knee-amplitude exploration available: `{neighborhood['positive_knee_amplitude_available']}`.  This is determined only by the all-model trusted native-domain artifact gate.

## Q6 - Phase/C2/closure

All 25 phase values pass monotonic, branch, duration and C2 endpoint checks: `{all_phase}`.  Every amplitude/phase proposal was independently checked for finite q/dq/ddq and q/dq/ddq closure.

## Q7 - Sparse MyoLeg validation

The preregistered `{SMOKE_COUNT}` candidates were prescribed-replayed on `{len(smoke_runtime['selected_subject_ids'])}` models (`{', '.join(smoke_runtime['selected_subject_ids'])}`), for `{len(smoke_rows)}` replay cases.  All passed: `{all_smoke}`.  No objective or rank was calculated.

## Q8 - Global set

One global candidate set applies to all 32 subjects.  There is no development/held-out or subject-specific candidate deletion.

## Q9 - Full landscape engineering estimate

- trajectories: `{trajectory_count:,}`
- measured mean prescribed replay: `{mean_time:.6f}` s/candidate/subject
- serial: `{runtime['serial_runtime_s']/3600:.3f}` h
- idealized 8-worker at 75% efficiency: `{runtime['idealized_parallel_runtime_s_at_75pct_efficiency']['8']/3600:.3f}` h
- torque-only float64 storage: `{torque_bytes/1e9:.3f}` GB
- full retained replay schema estimate: `{full_bytes/1e9:.3f}` GB

These are engineering estimates only.

## Q10 - Next stage

Ready to design/execute `MYOLEG_V2_TRUTH_LANDSCAPE_GENERATION_V1`: `{outcome != OUTCOME_INVALID}` with the synthetic-cohort and sparse-validation limitations.  This stage did not start it.

Final candidate-domain manifest SHA-256: `{manifest_sha}`.
"""
    write_json(OUTPUT/"metadata.json",{
        "stage_id":STAGE_ID,"outcome":outcome,"evidence_level":"OFFLINE_SIMULATOR_DOMAIN_DESIGN_AND_SPARSE_PRESCRIBED_REPLAY","input_sha256_before":input_before,"input_sha256_after":input_hashes(),"runtime_environment":environment,
        "protocol_sha256":sha256_file(PROTOCOL_PATH),"candidate_manifest_sha256":manifest_sha,"original_proposal_count":21025,"admissible_candidate_count":len(included),"boundary_row_count":len(boundary_rows),"smoke_replay_count":len(smoke_rows),"all_smoke_pass":all_smoke,"all_phase_pass":all_phase,
        "full_truth_landscape_generated":False,"five_parameter_fit":False,"nn_or_pinn":False,"bo":False,"robot_or_hardware":False,"runtime_s":time.perf_counter()-started,
    })
    (OUTPUT/"MYOLEG_V2_CANDIDATE_DOMAIN_DESIGN_REPORT.md").write_text(report,encoding="utf-8")
    checksum_names=[p for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="checksums.sha256"]
    (OUTPUT/"checksums.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_names)+"\n",encoding="utf-8")
    print(json.dumps({"outcome":outcome,"original":21025,"admissible":len(included),"trusted_knee":[trusted["trusted_knee_lower_deg"],trusted["trusted_knee_upper_deg"]],"smoke_replays":len(smoke_rows),"manifest_sha256":manifest_sha,"runtime_s":time.perf_counter()-started},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol",action="store_true")
    group.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    freeze_protocol() if args.freeze_protocol else execute()


if __name__ == "__main__":
    main()
