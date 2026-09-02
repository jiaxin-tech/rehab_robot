"""Build the preregistered MyoLeg structural-heterogeneity design audit.

This module is intentionally importable without the isolated MyoSuite runtime.
MuJoCo is imported only by :func:`build`, which performs a small nominal-model
derivative smoke test over a geometry-selected V3 trajectory subset.  It never
loads cohort truth arrays, held-out subjects, or a personalization objective.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


STAGE_ID = "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_AUDIT_V1"
PROTOCOL_ID = "MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_PROTOCOL_V1"
OUTCOME = "MYOLEG_HETEROGENEITY_EXPANSION_DESIGN_READY_WITH_EVIDENCE_GAPS"
COHORT_V1 = "MYOLEG_VIRTUAL_PATIENT_COHORT_V1"
FUTURE_COHORT = "MYOLEG_VIRTUAL_PATIENT_COHORT_V2"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1"
PROTOCOL = OUTPUT / "MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_PROTOCOL.json"
HELDOUT_AUDIT = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"
MODEL = ROOT / "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml"
V3_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
PRIOR = ROOT / "external_simulation_audits/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1"
CANDIDATE_BUILDER = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
PARAMETERIZATION = ROOT / "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"
FORMAL_REFERENCE = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"

FROZEN_SHA = {
    "protocol": "527763494905ae55b1bb672b1e4f23594c5eeb1461552c5f358c3ffcb1db6bf0",
    "model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v3_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "v3_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "cohort_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "prior_final": "76d3878e278cd2bc79fa56a67c13a3e95142dbb165c7ee40810f197c987624c1",
    "prior_objective": "90efa25073f51fb31c682cedcd7fc69a11b68945ffecd17f4e66fb7626447798",
    "prior_heterogeneity": "0798ac84b25ba3fe6ff0f12a1f93a3d682dbf9e5bb939d6eb85f4c54a6a62e95",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
}

FROZEN_PATHS = {
    "protocol": PROTOCOL,
    "model": MODEL,
    "v3_table": V3_TABLE,
    "v3_manifest": V3_MANIFEST,
    "cohort_manifest": COHORT_MANIFEST,
    "prior_final": PRIOR / "FINAL_BRANCH_DECISION.json",
    "prior_objective": PRIOR / "OBJECTIVE_ADEQUACY_DECISION.json",
    "prior_heterogeneity": PRIOR / "HETEROGENEITY_ADEQUACY_DECISION.json",
    "formal_manifest": FORMAL_MANIFEST,
    "formal_reference": FORMAL_REFERENCE,
}

BIARTICULAR = ("bflh_r", "grac_r", "recfem_r", "sart_r", "semimem_r", "semiten_r", "tfl_r")
HAMSTRING_FAMILY = ("bflh_r", "semimem_r", "semiten_r")
RECTUS_FAMILY = ("recfem_r",)
TRAJECTORIES = (
    "MYOLEG_V3_K0312", "MYOLEG_V3_K0000", "MYOLEG_V3_K0024",
    "MYOLEG_V3_K0600", "MYOLEG_V3_K0624", "MYOLEG_V3_K0012",
    "MYOLEG_V3_K0612", "MYOLEG_V3_K0300", "MYOLEG_V3_K0324",
)
EPS = 1.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise RuntimeError(f"cannot infer CSV schema for {path}")
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_inputs() -> dict[str, str]:
    actual = {name: sha256(path) for name, path in FROZEN_PATHS.items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_AUDIT.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    if protocol["frozen_before_structural_model_inventory_or_local_sensitivity_results"] is not True:
        raise RuntimeError("audit protocol was not preregistered")
    if heldout["held_out_scientific_access_count"] != 0:
        raise RuntimeError("held-out seal changed")
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA["formal_reference"]
    ):
        raise RuntimeError("formal ROM/reference convention changed")
    return actual


def object_name(mujoco: Any, model: Any, object_type: Any, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def actuator_row(mujoco: Any, model: Any, name: str) -> dict[str, Any]:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    tendon_id = int(model.actuator_trnid[identifier, 0])
    return {
        "muscle": name,
        "actuator_id": identifier,
        "tendon": object_name(mujoco, model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id),
        "actuator_lengthrange_m": "|".join(f"{value:.9g}" for value in model.actuator_lengthrange[identifier]),
        "muscle_range_normalized": f"{model.actuator_gainprm[identifier, 0]:.9g}|{model.actuator_gainprm[identifier, 1]:.9g}",
        "force_parameter_n": f"{model.actuator_gainprm[identifier, 2]:.9g}",
        "scale_parameter": f"{model.actuator_gainprm[identifier, 3]:.9g}",
        "lmin_normalized": f"{model.actuator_gainprm[identifier, 4]:.9g}",
        "lmax_normalized": f"{model.actuator_gainprm[identifier, 5]:.9g}",
        "vmax_normalized_per_s": f"{model.actuator_gainprm[identifier, 6]:.9g}",
        "fpmax_dimensionless": f"{model.actuator_gainprm[identifier, 7]:.9g}",
        "fvmax_dimensionless": f"{model.actuator_gainprm[identifier, 8]:.9g}",
    }


def inventory_rows(mujoco: Any, model: Any) -> list[dict[str, Any]]:
    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    rows = [
        ("A01", "muscle", "actuator_gainprm[:,0:2]", "muscle.range", "normalized muscle operating interval used to map transmission length", "dimensionless", "muscle-specific", "no", "no", "yes", "yes", "E3", "candidate only with explicit mapping; not physiological optimal fiber length"),
        ("A02", "muscle", "actuator_gainprm[:,4:6]", "muscle.lmin/lmax", "normalized length support of built-in muscle curve", "dimensionless", "muscle-specific", "no", "no", "yes", "yes", "E3", "candidate for coherent local/pilot variation; population range absent"),
        ("A03", "muscle", "actuator_gainprm[:,7]", "muscle.fpmax", "maximum normalized passive force parameter", "dimensionless", "muscle-specific", "no", "no", "no", "yes", "E3", "V1 common group scaling retained as negative result; relative families are separate candidate"),
        ("A04", "muscle", "actuator_gainprm[:,2]", "muscle.force", "force scaling F0 for built-in muscle", "N", "muscle-specific", "no", "no", "no", "yes", "E2/E3", "relative anatomical family balance candidate; not direct tissue strength"),
        ("A05", "muscle", "actuator_lengthrange", "general.lengthrange", "feasible transmission length range used by muscle normalization", "m", "muscle-specific", "compile-derived/source-explicit", "no", "yes", "yes", "E4", "protected; not a physiological fiber-length parameter"),
        ("B01", "tendon path", "site_pos", "site.pos", "attachment/via point in parent-body coordinates", "m", "site-specific", "source recompile for coherent change", "yes", "indirect", "conditional", "E2", "single-site mutation prohibited; path and calibration must be rebuilt together"),
        ("B02", "tendon path", "geom_pos/geom_size", "wrap geom pos/size", "spatial wrap surface controlling routing", "m", "wrap-specific", "yes", "yes", "indirect", "conditional", "E2", "requires subject-specific geometry and muscle-length calibration"),
        ("C01", "segment", "body_pos/body_ipos", "body.pos/inertial.pos", "joint placement and COM location", "m", "body-specific", "usually yes", "yes", "no", "conditional", "E1/E2", "segment change requires coupled geometry rebuild"),
        ("C02", "segment", "body_mass/body_inertia", "inertial mass/diaginertia", "rigid-body mass and inertia", "kg / kg m2", "body-specific", "no", "no", "no", "yes", "E1/E2", "V1 magnitude-dominant; retain as background/secondary"),
        ("D01", "joint", "dof_damping", "joint.damping", "linear viscous generalized force", "N m s/rad", "DOF-specific", "no", "no", "no", "yes", "E4", "actual non-limit field but overlaps gray-box Bknee/Bhip; stress/secondary only"),
        ("D02", "joint", "jnt_stiffness/qpos_spring", "joint.stiffness/springref", "linear joint spring", "N m/rad", "joint-specific", "no", "no", "no", "yes", "E4", "nominal hip and knee stiffness are zero"),
        ("D03", "joint limit", "jnt_range/jnt_solref/jnt_solimp", "joint range/solver impedance", "constraint limit mechanics", "rad / solver", "joint-specific", "source or runtime", "no", "no", "conditional", "E4", "simulator limit artifact; never subject physiology"),
        ("E01", "tendon", "tendon_stiffness/tendon_damping", "spatial tendon stiffness/damping", "N/m / N s/m", "tendon-specific", "no", "no", "indirect", "yes", "E4", "both zero in current compiled model; no calibrated compliance"),
        ("E02", "tendon", "tendon_lengthspring", "spatial tendon springlength", "spring rest length", "m", "tendon-specific", "no", "no", "indirect", "yes", "E4", "nonzero rest length does not create elasticity when stiffness is zero"),
        ("F01", "constraint", "eq_data", "joint equality polycoef", "knee/patella dependent-coordinate mapping", "mixed", "constraint-specific", "yes for coherent change", "yes", "no", "itself", "E2", "frozen; geometry changes must be compatible with 14 knee/patella equalities"),
    ]
    records = [dict(zip(
        ("parameter_id", "class", "compiled_field", "xml_semantic", "model_semantic", "units", "specificity", "requires_recompile", "changes_geometry", "changes_normalized_muscle_coordinate", "knee_patella_equality_compatible", "evidence_level", "audit_decision"), row
    )) for row in rows]
    nominal = {
        "A01": f"range0={np.min(model.actuator_gainprm[:, 0]):.6g}..{np.max(model.actuator_gainprm[:, 0]):.6g}; range1={np.min(model.actuator_gainprm[:, 1]):.6g}..{np.max(model.actuator_gainprm[:, 1]):.6g}",
        "A02": f"lmin={np.min(model.actuator_gainprm[:, 4]):.6g}..{np.max(model.actuator_gainprm[:, 4]):.6g}; lmax={np.min(model.actuator_gainprm[:, 5]):.6g}..{np.max(model.actuator_gainprm[:, 5]):.6g}",
        "A03": f"fpmax={np.min(model.actuator_gainprm[:, 7]):.6g}..{np.max(model.actuator_gainprm[:, 7]):.6g}",
        "A04": f"force={np.min(model.actuator_gainprm[:, 2]):.6g}..{np.max(model.actuator_gainprm[:, 2]):.6g}",
        "A05": f"lengthrange={np.min(model.actuator_lengthrange):.6g}..{np.max(model.actuator_lengthrange):.6g} m",
        "B01": f"{model.nsite} compiled sites", "B02": f"{model.ngeom} compiled geoms",
        "C01": "body-specific compiled coordinates", "C02": f"mass={np.min(model.body_mass):.6g}..{np.max(model.body_mass):.6g} kg",
        "D01": f"hip={model.dof_damping[int(model.jnt_dofadr[hip])]:.6g}; knee={model.dof_damping[int(model.jnt_dofadr[knee])]:.6g}",
        "D02": f"hip={model.jnt_stiffness[hip]:.6g}; knee={model.jnt_stiffness[knee]:.6g}",
        "D03": f"hip={model.jnt_range[hip].tolist()}; knee={model.jnt_range[knee].tolist()}",
        "E01": f"stiffness={np.unique(model.tendon_stiffness).tolist()}; damping={np.unique(model.tendon_damping).tolist()}",
        "E02": f"springlength={np.min(model.tendon_lengthspring):.6g}..{np.max(model.tendon_lengthspring):.6g} m",
        "F01": "14 knee/patella polynomial equalities; 27 total model equalities",
    }
    for record in records:
        record["nominal_value"] = nominal[record["parameter_id"]]
        record["source_model_path"] = str(MODEL.relative_to(ROOT))
    return records + [{
        "parameter_id": "MODEL_SUMMARY", "class": "compiled model", "compiled_field": "dimensions",
        "xml_semantic": "compiled MyoLeg supine model", "model_semantic": f"{model.nu} muscle actuators; {model.ntendon} tendons; {model.nsite} sites; {model.ngeom} geoms; {model.neq} equalities",
        "units": "count", "specificity": "model", "requires_recompile": "no", "changes_geometry": "no",
        "changes_normalized_muscle_coordinate": "no", "knee_patella_equality_compatible": "yes",
        "evidence_level": "implementation", "audit_decision": f"hip joint {hip}; knee joint {knee}; inventory reflects actual compiled model",
        "nominal_value": "compiled values", "source_model_path": str(MODEL.relative_to(ROOT)),
    }]


def fit_metrics(nominal: np.ndarray, changed: np.ndarray) -> tuple[float, float, float]:
    slope = float(np.dot(nominal, changed) / np.dot(nominal, nominal))
    proportional_residual = changed - slope * nominal
    proportional_nrmse = float(np.sqrt(np.mean(proportional_residual**2)) / np.sqrt(np.mean(changed**2)))
    design = np.column_stack((nominal, np.ones_like(nominal)))
    fitted = design @ np.linalg.lstsq(design, changed, rcond=None)[0]
    denominator = float(np.sum((changed - np.mean(changed)) ** 2))
    affine_r2 = 1.0 if denominator == 0.0 else 1.0 - float(np.sum((changed - fitted) ** 2)) / denominator
    return slope, proportional_nrmse, affine_r2


def mutate_lmax(mujoco: Any, model: Any, sign: int) -> list[str]:
    for name in BIARTICULAR:
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        model.actuator_gainprm[identifier, 5] += sign * EPS
        model.actuator_biasprm[identifier, 5] += sign * EPS
    return [f"{name}:gainprm/biasprm[5]" for name in BIARTICULAR]


def mutate_relative_fpmax(mujoco: Any, model: Any, sign: int) -> list[str]:
    changes = []
    for names, direction in ((RECTUS_FAMILY, sign), (HAMSTRING_FAMILY, -sign)):
        for name in names:
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            scale = math.exp(direction * EPS)
            model.actuator_gainprm[identifier, 7] *= scale
            model.actuator_biasprm[identifier, 7] *= scale
            changes.append(f"{name}:gainprm/biasprm[7]")
    return changes


def mutate_knee_damping(mujoco: Any, model: Any, sign: int) -> list[str]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    dof = int(model.jnt_dofadr[joint])
    model.dof_damping[dof] *= math.exp(sign * EPS)
    return ["knee_angle_r:dof_damping"]


def local_sensitivity(mujoco: Any, model_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_builder = load_module(CANDIDATE_BUILDER, "_heterogeneity_candidate_builder")
    replay_builder = load_module(REPLAY_BUILDER, "_heterogeneity_replay_builder")
    parameterization = load_module(PARAMETERIZATION, "_heterogeneity_parameterization")
    reference = candidate_builder.load_reference_adapter()
    with V3_TABLE.open(newline="", encoding="utf-8") as stream:
        table = {row["candidate_id"]: row for row in csv.DictReader(stream)}
    if set(TRAJECTORIES) - set(table):
        raise RuntimeError("preregistered geometry-only trajectory subset changed")

    generated: dict[str, dict[str, Any]] = {}
    for candidate_id in TRAJECTORIES:
        row = table[candidate_id]
        trajectory = parameterization.generate_v3_trajectory(reference, float(row["beta_flex"]), float(row["beta_extend"]))
        generated[candidate_id] = {
            "time_s": reference["time_s"], "q": trajectory.q, "dq": trajectory.dq,
            "ddq": trajectory.ddq, "phases": reference["phases"], "rows": [],
        }

    def replay_model(model: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        responses: dict[str, np.ndarray] = {}
        warnings = 0
        residual = 0.0
        for candidate_id in TRAJECTORIES:
            arrays, runtime = replay_builder.prescribed_truth(model, generated[candidate_id])
            tau = np.asarray(arrays["tau_truth_nm"], dtype=float)
            if not np.isfinite(tau).all():
                raise RuntimeError("nonfinite local sensitivity replay")
            responses[candidate_id] = tau
            warnings = max(warnings, int(np.max(arrays["warning_count"])))
            residual = max(residual, float(np.max(np.abs(arrays["decomposition_residual_nm"]))))
        return responses, {"warning_count_max": warnings, "decomposition_residual_max_nm": residual}

    nominal_model = mujoco.MjModel.from_xml_path(str(model_path))
    nominal, nominal_integrity = replay_model(nominal_model)
    probes: list[tuple[str, str, str, Callable[[Any, Any, int], list[str]]]] = [
        ("P1_BIARTICULAR_LMAX_COHERENT", "native normalized-coordinate +/-1e-4", "A/B", mutate_lmax),
        ("P2_RECTUS_VS_HAMSTRING_FPMAX_BALANCE", "log scale +/-1e-4", "B/G", mutate_relative_fpmax),
        ("P3_KNEE_DAMPING", "log scale +/-1e-4", "E", mutate_knee_damping),
    ]
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"nominal_integrity": nominal_integrity, "trajectory_ids": list(TRAJECTORIES)}
    for probe_id, magnitude, factor_class, mutator in probes:
        level_responses: dict[int, dict[str, np.ndarray]] = {}
        level_integrity: dict[int, dict[str, Any]] = {}
        modified_fields: list[str] = []
        for sign in (-1, 1):
            model = mujoco.MjModel.from_xml_path(str(model_path))
            modified_fields = mutator(mujoco, model, sign)
            level_responses[sign], level_integrity[sign] = replay_model(model)
        flat_nominal = np.concatenate([nominal[candidate_id].ravel() for candidate_id in TRAJECTORIES])
        fit_by_level = {}
        for sign in (-1, 1):
            changed = np.concatenate([level_responses[sign][candidate_id].ravel() for candidate_id in TRAJECTORIES])
            fit_by_level[sign] = fit_metrics(flat_nominal, changed)
        sensitivities = []
        joint_sensitivities = [[], []]
        rms_by_level: dict[int, dict[str, float]] = {-1: {}, 0: {}, 1: {}}
        for candidate_id in TRAJECTORIES:
            base = nominal[candidate_id]
            minus, plus = level_responses[-1][candidate_id], level_responses[1][candidate_id]
            denominator = max(float(np.linalg.norm(base)), np.finfo(float).eps)
            sensitivities.append(float(np.linalg.norm((plus - minus) / (2.0 * EPS))) / denominator)
            for joint in (0, 1):
                joint_denominator = max(float(np.linalg.norm(base[:, joint])), np.finfo(float).eps)
                joint_sensitivities[joint].append(float(np.linalg.norm((plus[:, joint] - minus[:, joint]) / (2.0 * EPS))) / joint_denominator)
            for sign, response in ((-1, minus), (0, base), (1, plus)):
                rms_by_level[sign][candidate_id] = float(np.sqrt(np.mean(response**2)))
        sensitivity_mean = float(np.mean(sensitivities))
        sensitivity_cv = float(np.std(sensitivities) / abs(sensitivity_mean)) if sensitivity_mean else 0.0

        def gradient(values: dict[str, float]) -> np.ndarray:
            return np.asarray([
                (values["MYOLEG_V3_K0612"] - values["MYOLEG_V3_K0012"]) / 0.06,
                (values["MYOLEG_V3_K0324"] - values["MYOLEG_V3_K0300"]) / 0.06,
            ])

        gradient_nominal = gradient(rms_by_level[0])
        cosines = []
        sign_change = False
        for sign in (-1, 1):
            changed_gradient = gradient(rms_by_level[sign])
            denominator = float(np.linalg.norm(gradient_nominal) * np.linalg.norm(changed_gradient))
            cosines.append(1.0 if denominator == 0 else float(np.dot(gradient_nominal, changed_gradient) / denominator))
            sign_change = sign_change or bool(np.any(np.signbit(gradient_nominal) != np.signbit(changed_gradient)))
        affine_r2 = min(fit_by_level[-1][2], fit_by_level[1][2])
        proportional_nrmse = max(fit_by_level[-1][1], fit_by_level[1][1])
        minimum_cosine = min(cosines)
        nonproportional = bool(
            affine_r2 < 0.9999 or proportional_nrmse > 0.0001
            or sensitivity_cv > 0.01 or minimum_cosine < 0.995 or sign_change
        )
        row = {
            "probe_id": probe_id, "factor_class": factor_class, "status": "RUN",
            "scientific_role": "LOCAL_NUMERICAL_SENSITIVITY_ONLY", "perturbation": magnitude,
            "population_range_inferred": False, "trajectory_count": len(TRAJECTORIES),
            "modified_fields": "|".join(modified_fields),
            "proportional_nrmse_max": f"{proportional_nrmse:.12g}",
            "affine_r2_min": f"{affine_r2:.12g}",
            "local_sensitivity_mean": f"{sensitivity_mean:.12g}",
            "local_sensitivity_cv": f"{sensitivity_cv:.12g}",
            "hip_sensitivity_cv": f"{np.std(joint_sensitivities[0]) / max(abs(np.mean(joint_sensitivities[0])), np.finfo(float).eps):.12g}",
            "knee_sensitivity_cv": f"{np.std(joint_sensitivities[1]) / max(abs(np.mean(joint_sensitivities[1])), np.finfo(float).eps):.12g}",
            "gradient_direction_cosine_min": f"{minimum_cosine:.12g}",
            "gradient_sign_change": sign_change,
            "mechanistic_nonproportionality_threshold_met": nonproportional,
            "warning_count_max": max(nominal_integrity["warning_count_max"], level_integrity[-1]["warning_count_max"], level_integrity[1]["warning_count_max"]),
            "decomposition_residual_max_nm": f"{max(nominal_integrity['decomposition_residual_max_nm'], level_integrity[-1]['decomposition_residual_max_nm'], level_integrity[1]['decomposition_residual_max_nm']):.12g}",
            "interpretation": "local derivative/shape evidence only; not a subject range and not personalization evidence",
        }
        rows.append(row)
        detail[probe_id] = {"fit_by_level": fit_by_level, "level_integrity": level_integrity, "rms_by_level": rms_by_level}
    rows.append({
        "probe_id": "P4_TENDON_ELASTICITY", "factor_class": "F", "status": "NOT_RUN_WITH_REASON",
        "scientific_role": "LOCAL_NUMERICAL_SENSITIVITY_ONLY", "perturbation": "none",
        "population_range_inferred": False, "trajectory_count": 0,
        "modified_fields": "tendon_stiffness/tendon_damping",
        "proportional_nrmse_max": "", "affine_r2_min": "", "local_sensitivity_mean": "",
        "local_sensitivity_cv": "", "hip_sensitivity_cv": "", "knee_sensitivity_cv": "",
        "gradient_direction_cosine_min": "", "gradient_sign_change": "",
        "mechanistic_nonproportionality_threshold_met": False, "warning_count_max": "",
        "decomposition_residual_max_nm": "",
        "interpretation": "not run: current compiled tendon stiffness and damping are zero and no independent calibration exists",
    })
    return rows, detail


def sources() -> list[dict[str, Any]]:
    return [
        {"source_id": "SRC01", "evidence_level": "implementation", "source": "MuJoCo XML Reference: muscle/general actuator", "url": "https://mujoco.readthedocs.io/en/stable/XMLreference.html", "parameter_definition": "range, force, scale, lmin, lmax, vmax, fpmax, fvmax and lengthrange", "population_or_model": "MuJoCo built-in muscle", "reported_variability": "not a population source", "mapping_to_myoleg": "direct compiled field semantics", "limitations": "does not establish physiological one-to-one mapping or population bounds"},
        {"source_id": "SRC02", "evidence_level": "implementation", "source": "MuJoCo Modeling: muscle model", "url": "https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators", "parameter_definition": "transmission length normalization and FLV force model", "population_or_model": "MuJoCo built-in muscle", "reported_variability": "not a population source", "mapping_to_myoleg": "direct runtime semantics", "limitations": "abstract muscle/tendon model"},
        {"source_id": "SRC03", "evidence_level": "E2", "source": "Caggiano et al. 2022 MyoSuite", "url": "https://proceedings.mlr.press/v168/caggiano22a/caggiano22a.pdf", "parameter_definition": "MyoSuite musculoskeletal simulation suite", "population_or_model": "MyoSuite", "reported_variability": "not used for bounds", "mapping_to_myoleg": "software/model provenance", "limitations": "does not validate this proposed cohort parameterization"},
        {"source_id": "SRC04", "evidence_level": "E2", "source": "Blemker and Delp 2005", "url": "https://pubmed.ncbi.nlm.nih.gov/10862133/", "parameter_definition": "subject-specific paths, wrap surfaces, and moment arms from MRI", "population_or_model": "human lower limb models", "reported_variability": "subject-specific geometry demonstrated", "mapping_to_myoleg": "supports geometry as consequential but requires coupled reconstruction", "limitations": "not a direct range for current XML sites"},
        {"source_id": "SRC05", "evidence_level": "E1/E2", "source": "Rajagopal et al. 2016", "url": "https://nmbl.stanford.edu/wp-content/uploads/Rajagopal2016.pdf", "parameter_definition": "lower-limb muscle architecture and geometry in a full-body model", "population_or_model": "cadaver/MRI-informed generic model", "reported_variability": "architecture sources from multiple human datasets", "mapping_to_myoleg": "context for muscle architecture and generic-model limitations", "limitations": "not a MyoLeg parameter-range prescription"},
        {"source_id": "SRC06", "evidence_level": "E1", "source": "Ward et al. 2009", "url": "https://muscle.ucsd.edu/pubs/pdf/Ward_CORR_2009.pdf", "parameter_definition": "optimal fiber length and PCSA", "population_or_model": "21 human lower extremities; 27 muscles", "reported_variability": "human muscle architecture measured", "mapping_to_myoleg": "range evidence may inform a future calibrated mapping", "limitations": "current built-in parameters are not direct optimal fiber length/PCSA fields"},
        {"source_id": "SRC07", "evidence_level": "E2", "source": "Experiment-guided musculoskeletal model tuning 2024", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11199655/", "parameter_definition": "geometry, fiber/slack parameters and passive curves calibrated to experiments", "population_or_model": "subject-specific musculoskeletal modeling", "reported_variability": "calibration procedure, not adopted range", "mapping_to_myoleg": "supports coupled calibration requirement", "limitations": "not validated for this MyoLeg supine setup"},
        {"source_id": "SRC08", "evidence_level": "E2/E3", "source": "Musculotendon parameter uncertainty review 2023", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10172227/", "parameter_definition": "optimal fiber length, maximum force, pennation, slack length and passive curve uncertainty", "population_or_model": "Hill-type musculoskeletal models", "reported_variability": "qualitative uncertainty evidence", "mapping_to_myoleg": "motivates range-evidence program", "limitations": "mapping to MuJoCo normalized fields is indirect"},
    ]


def static_audits(mujoco: Any, model: Any) -> dict[str, list[dict[str, Any]]]:
    muscles = [actuator_row(mujoco, model, name) for name in BIARTICULAR]
    operating = []
    for row in muscles:
        operating.append({
            **row, "exact_semantics": "MuJoCo normalized built-in-muscle fields; not direct physiological fiber length",
            "actuator_lengthrange_protected": True, "candidate_field": "lmax and/or range only after explicit mapping",
            "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE", "taxonomy": "A/B",
        })
    coupling = []
    for row in muscles:
        tendon = row["tendon"]
        tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon)
        coupling.append({
            "muscle": row["muscle"], "tendon": tendon,
            "tendon_path_element_count": int(model.tendon_num[tendon_id]),
            "force_parameter_n": row["force_parameter_n"], "fpmax_dimensionless": row["fpmax_dimensionless"],
            "actual_coupling_mechanism": "muscle-specific force-length-velocity x spatial tendon moment arms at hip and knee",
            "artificial_coupling_coefficient_needed": False,
            "candidate": "relative family force/fpmax or normalized-curve structure",
            "geometry_change": False, "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        })
    geometry = []
    for row in muscles:
        tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, row["tendon"])
        geometry.append({
            "muscle": row["muscle"], "tendon": row["tendon"],
            "compiled_path_element_count": int(model.tendon_num[tendon_id]),
            "candidate_fields": "site_pos; wrap geom_pos/geom_size",
            "configuration_dependent_mapping": True,
            "consistency_class": "REQUIRES_REBUILD/CALIBRATION",
            "synchronized_changes": "neighbor sites|wrap objects|muscle normalized-length calibration|segment/joint geometry",
            "single_site_move_allowed": False, "primary_scheme": False,
        })
    segment = [
        {"factor": "femur length/proportion", "compiled_fields": "body_pos|joint pos|site_pos|geom_pos", "configuration_effect": "joint and muscle paths", "required_coupled_changes": "joint position|attachments|wraps|muscle length calibration|inertia", "rtb3_or_strap_effect": "indirect through distal chain", "decision": "DO_NOT_PERTURB_IN_PRIMARY_EXPANDED_COHORT"},
        {"factor": "tibia length/proportion", "compiled_fields": "body_pos|joint pos|site_pos|geom_pos", "configuration_effect": "joint and muscle paths", "required_coupled_changes": "joint position|attachments|wraps|muscle length calibration|inertia|RTB3", "rtb3_or_strap_effect": "direct", "decision": "DO_NOT_PERTURB_IN_PRIMARY_EXPANDED_COHORT"},
        {"factor": "body COM", "compiled_fields": "body_ipos", "configuration_effect": "gravity/inertial torque", "required_coupled_changes": "consistent segment inertial identification", "rtb3_or_strap_effect": "none if geometry unchanged", "decision": "SECONDARY"},
        {"factor": "mass and inertia", "compiled_fields": "body_mass|body_inertia", "configuration_effect": "mostly magnitude in V1", "required_coupled_changes": "physically consistent inertia tensor", "rtb3_or_strap_effect": "none", "decision": "BACKGROUND_SECONDARY"},
    ]
    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    passive = []
    for name, joint in (("hip_flexion_r", hip), ("knee_angle_r", knee)):
        dof = int(model.jnt_dofadr[joint])
        passive.extend([
            {"joint": name, "field": "dof_damping", "nominal": float(model.dof_damping[dof]), "units": "N m s/rad", "normal_or_limit": "non-limit model field", "truth_learner_independent": False, "decision": "SECONDARY/STRESS_ONLY"},
            {"joint": name, "field": "jnt_stiffness", "nominal": float(model.jnt_stiffness[joint]), "units": "N m/rad", "normal_or_limit": "non-limit model field", "truth_learner_independent": False, "decision": "DO_NOT_USE_AS_PRIMARY_CURRENTLY_ZERO"},
            {"joint": name, "field": "jnt_range/solver impedance", "nominal": "|".join(map(str, model.jnt_range[joint])), "units": "rad/solver", "normal_or_limit": "simulator joint-limit artifact", "truth_learner_independent": True, "decision": "DO_NOT_PERTURB"},
        ])
    tendon = [
        {"field": "tendon_stiffness", "nominal_unique": "|".join(map(str, np.unique(model.tendon_stiffness))), "units": "N/m", "actual_support": True, "calibrated_in_current_model": False, "trajectory_dependent_potential": True, "decision": "SECONDARY/STRESS_ONLY"},
        {"field": "tendon_damping", "nominal_unique": "|".join(map(str, np.unique(model.tendon_damping))), "units": "N s/m", "actual_support": True, "calibrated_in_current_model": False, "trajectory_dependent_potential": True, "decision": "SECONDARY/STRESS_ONLY"},
        {"field": "tendon_lengthspring", "nominal_min": float(np.min(model.tendon_lengthspring)), "nominal_max": float(np.max(model.tendon_lengthspring)), "units": "m", "actual_support": True, "calibrated_in_current_model": False, "trajectory_dependent_potential": False, "decision": "DO_NOT_PERTURB_WITHOUT_STIFFNESS_AND_CALIBRATION"},
    ]
    groups = [
        {"factor": "V1 hip-only common fpmax", "model_fields": "muscle-specific fpmax changed by one common group multiplier", "relative_pattern_changes": False, "evidence": "V1 negative result", "taxonomy": "B", "future_role": "secondary/background; not primary signal source"},
        {"factor": "V1 knee-only common fpmax", "model_fields": "muscle-specific fpmax changed by one common group multiplier", "relative_pattern_changes": False, "evidence": "V1 negative result", "taxonomy": "B", "future_role": "secondary/background; not primary signal source"},
        {"factor": "V1 biarticular common fpmax", "model_fields": "seven muscle-specific fpmax fields changed uniformly", "relative_pattern_changes": False, "evidence": "V1 negative result", "taxonomy": "B", "future_role": "replace in personalization-focused V2 design"},
        {"factor": "rectus-femoris vs hamstring-family relative fpmax", "model_fields": "recfem_r vs bflh_r/semimem_r/semiten_r gainprm/biasprm[7]", "relative_pattern_changes": True, "evidence": "E3 mapping; range absent", "taxonomy": "A", "future_role": "pilot candidate"},
        {"factor": "relative family F0", "model_fields": "muscle.force gainprm/biasprm[2] by anatomical family", "relative_pattern_changes": True, "evidence": "E2/E3; range mapping required", "taxonomy": "A", "future_role": "pilot candidate"},
        {"factor": "80 independent random muscles", "model_fields": "all muscle fields independently", "relative_pattern_changes": True, "evidence": "no parsimonious subject semantics", "taxonomy": "E", "future_role": "prohibited"},
    ]
    return {"operating": operating, "coupling": coupling, "geometry": geometry, "segment": segment, "passive": passive, "tendon": tendon, "groups": groups}


def taxonomy_rows() -> list[dict[str, Any]]:
    rows = [
        ("biarticular normalized-curve lmax profile", "actuator_gainprm/biasprm[5]", "A", "E3", "yes", "yes", "no", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("anatomical-family relative F0 balance", "actuator_gainprm/biasprm[2]", "A", "E2/E3", "yes", "yes", "no", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("rectus-vs-hamstring relative fpmax", "actuator_gainprm/biasprm[7]", "A", "E3", "yes", "yes", "no", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("mass/inertia", "body_mass/body_inertia", "B", "E1/E2", "mostly magnitude", "yes", "no", "EXISTING_V1_RANGE_ONLY"),
        ("COM location", "body_ipos", "B", "E1/E2", "yes", "conditional", "no", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("background activation", "act/ctrl initial condition", "C", "E3", "yes", "yes", "no", "EPISODE_NOT_SUBJECT_RANGE"),
        ("joint damping", "dof_damping", "D", "E4", "yes", "yes", "overlaps Bhip/Bknee", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("tendon stiffness/damping", "tendon_stiffness/tendon_damping", "D", "E4", "potential", "no calibration", "no", "RANGE_REQUIRES_EXTERNAL_EVIDENCE"),
        ("segment length", "body/joint/site/geom coupled geometry", "E", "E1/E2", "yes", "requires rebuild", "no", "NOT_A_SCALAR_RANGE_PROBLEM"),
        ("single-site attachment move", "site_pos", "E", "E4", "yes", "no", "no", "DO_NOT_PERTURB"),
        ("actuator lengthrange", "actuator_lengthrange", "E", "implementation", "yes", "no physiological mapping", "no", "DO_NOT_PERTURB"),
        ("joint-limit curve", "jnt_range/solref/solimp", "E", "E4", "yes", "simulator artifact", "no", "DO_NOT_PERTURB"),
    ]
    keys = ("factor", "actual_fields", "taxonomy", "evidence_level", "configuration_dependent_potential", "geometry_consistency", "truth_learner_parameterization_independence", "range_status")
    return [dict(zip(keys, row)) for row in rows]


def schemes() -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID, "selection_uses_personalization_outcome": False,
        "population_bounds_frozen": False, "new_subjects_generated": False,
        "recommended_scheme_for_pilot": "S1_MINIMAL_STRUCTURAL",
        "schemes": [
            {"scheme_id": "S1_MINIMAL_STRUCTURAL", "scientific_role": "primary pilot candidate", "dimensionality": 4,
             "factors": ["biarticular normalized-curve lmax profile", "rectus-vs-hamstring relative fpmax balance", "hip monoarticular antagonist relative F0", "knee monoarticular antagonist relative F0"],
             "actual_model_fields": ["gainprm/biasprm[5]", "gainprm/biasprm[7]", "gainprm/biasprm[2]", "gainprm/biasprm[2]"],
             "mechanistic_rationale": "curve position plus anatomical-family balance acts through distinct configuration-dependent paths rather than common group scale",
             "evidence_level": "E2/E3", "implementation_risk": "moderate", "geometry_consistency": "no geometry mutation; knee equalities unchanged",
             "range_evidence_exists": False, "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
            {"scheme_id": "S2_MODERATE_STRUCTURAL", "scientific_role": "secondary pilot candidate", "dimensionality": 7,
             "factors": ["hip-flexor/extensor curve profile", "knee-flexor/extensor curve profile", "biarticular curve profile", "rectus-vs-hamstring fpmax", "hip relative F0", "knee relative F0", "background mass/COM profile"],
             "actual_model_fields": ["gainprm/biasprm[5]", "gainprm/biasprm[5]", "gainprm/biasprm[5]", "gainprm/biasprm[7]", "gainprm/biasprm[2]", "gainprm/biasprm[2]", "body_mass/body_inertia/body_ipos"],
             "mechanistic_rationale": "separates curve and family-balance mechanisms while remaining below eight dimensions",
             "evidence_level": "E1-E3 mixed", "implementation_risk": "moderate-high", "geometry_consistency": "conditional; COM requires inertial consistency",
             "range_evidence_exists": False, "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
            {"scheme_id": "S3_STRESS_EXPLORATORY", "scientific_role": "stress only; excluded from primary paper validation", "dimensionality": 6,
             "factors": ["joint damping", "synthetic tendon stiffness", "tendon damping", "passive curve stress", "coupled geometry profile", "activation nuisance"],
             "actual_model_fields": ["dof_damping", "tendon_stiffness", "tendon_damping", "gainprm/biasprm[4:8]", "site/geom/body coupled rebuild", "act/ctrl"],
             "mechanistic_rationale": "robustness exploration only", "evidence_level": "E3/E4", "implementation_risk": "high",
             "geometry_consistency": "not established", "range_evidence_exists": False, "range_status": "STRESS_ONLY_NO_POPULATION_BOUNDS"},
        ],
    }


def range_requirements() -> list[dict[str, Any]]:
    return [
        {"factor": "normalized muscle range/lmin/lmax mapping", "required_evidence": "model-specific calibration linking measured fiber/tendon behavior to MuJoCo normalized fields", "minimum_level": "E2", "acceptable_source": "subject-specific imaging/dynamometry plus validated conversion", "current_gap": "no one-to-one physiological mapping", "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
        {"factor": "relative muscle-family F0", "required_evidence": "human architecture/PCSA or force-capacity distribution with conversion uncertainty", "minimum_level": "E1/E2", "acceptable_source": "human measurements or validated subject-specific model", "current_gap": "current force field is model force scale, not direct measured strength", "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
        {"factor": "relative family fpmax", "required_evidence": "passive force-length calibration by muscle family", "minimum_level": "E1/E2", "acceptable_source": "passive torque/fiber imaging calibration", "current_gap": "literature does not directly specify current normalized fpmax bounds", "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
        {"factor": "COM profile", "required_evidence": "segment-specific human COM distribution and coherent inertia reconstruction", "minimum_level": "E1/E2", "acceptable_source": "anthropometric/imaging study", "current_gap": "V1 fixed COM and approximate inertia scaling", "status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE"},
        {"factor": "moment-arm geometry", "required_evidence": "subject-specific geometry plus coupled path/wrap and muscle-length calibration", "minimum_level": "E2", "acceptable_source": "MRI-derived model validation", "current_gap": "no safe scalar site range", "status": "REQUIRES_REBUILD_CALIBRATION"},
        {"factor": "tendon compliance", "required_evidence": "tendon-specific stiffness/slack calibration in compatible model", "minimum_level": "E1/E2", "acceptable_source": "in-vivo/validated model calibration", "current_gap": "current stiffness/damping zero", "status": "SECONDARY_STRESS_ONLY"},
    ]


def report_text(sensitivity: list[dict[str, Any]], model: Any) -> str:
    ran = [row for row in sensitivity if row["status"] == "RUN"]
    nonprop = [row["probe_id"] for row in ran if row["mechanistic_nonproportionality_threshold_met"]]
    return f"""# MyoLeg Musculoskeletal Heterogeneity Expansion Design Audit V1

## Formal outcome

**{OUTCOME}**

This is an offline design/model-semantics/evidence audit. It generated no subject, cohort, truth landscape, learner, optimizer result, or robot action. The V1 negative results remain formal: `V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED` and `HETEROGENEITY_LIMITATION_DOMINANT`.

Protocol SHA-256: `{FROZEN_SHA['protocol']}`. Held-out scientific access: **0**.

## What the model actually contains

The compiled supine MyoLeg model contains {model.nu} muscle actuators, {model.ntendon} spatial tendons, {model.nsite} sites, {model.ngeom} geoms, and {model.neq} equality constraints, including 14 bilateral knee/patella polynomial equalities. The built-in muscle fields provide normalized operating/curve parameters and muscle-specific force/fpmax values. They do **not** provide a defensible one-to-one label for physiological optimal fiber length or tendon slack length.

`actuator_lengthrange` remains protected. It is the feasible transmission-length range used by MuJoCo normalization, not a subject fiber-length measurement. Current spatial tendon stiffness and damping are both zero; nonzero `springlength` alone is not calibrated tendon elasticity. Hip/knee joint stiffness is zero, while default damping is 0.5 N m s/rad. Joint-limit solver behavior remains a simulator artifact, not subject physiology.

## Local numerical smoke test

The preregistered geometry-only subset used 9 V3 trajectories and a local magnitude of 1e-4 only. This is `LOCAL_NUMERICAL_SENSITIVITY_ONLY`, never a population bound. Three eligible probes ran; tendon elasticity was not substituted after being found ineligible. Non-proportionality threshold met by: **{', '.join(nonprop) if nonprop else 'none'}**. These results establish implementation/derivative behavior only and do not demonstrate subject-specific oracles.

## Q1. Which actual parameters can change configuration-dependent mechanics?

Normalized muscle curve fields (`range`, `lmin/lmax`), muscle-family-relative force/fpmax fields, and spatial path/wrap geometry can do so. Joint damping is trajectory dependent but overlaps the future gray-box damping parameters and is therefore not a primary independent truth factor. Geometry is mechanistically strong but cannot be safely varied as isolated sites.

## Q2. Can operating-length/passive-curve structure be varied defensibly?

**At model-field level, yes; at population-range/physiological-label level, not yet.** A coherent pilot can vary exact normalized curve fields without touching geometry. It must retain their MuJoCo names and cannot rename them optimal fiber length or slack length. Bounds require external calibration evidence.

## Q3. Can biarticular coupling be represented without an artificial coefficient?

Yes. Use the seven real muscle actuators, their muscle-specific curve/force fields, and their spatial tendon paths. A low-dimensional anatomical-family balance changes existing model fields; no new coupling coefficient is needed.

## Q4. Which moment-arm geometry factors are safe?

No single attachment/via/wrap coordinate is safe as an independent subject factor. Coupled subject geometry is scientifically plausible but classified `REQUIRES_REBUILD/CALIBRATION`: neighboring sites, wrap objects, muscle-length calibration, body/joint geometry, and knee/patella consistency must be updated together.

## Q5. Can segment geometry/COM be introduced safely?

COM can be a secondary factor after coherent inertial identification. Femur/tibia length cannot enter the primary expanded cohort through a scalar edit: joints, attachments, paths, wraps, muscle calibration, inertia, and RTB3/strap geometry must all be rebuilt.

## Q6. Are tendon/joint passive mechanics primary factors?

Not now. Current tendon elasticity is uncalibrated and joint stiffness is zero. Joint damping is an actual field but is E4/secondary and not truth-learner independent. Joint-limit curves are excluded.

## Q7. What happens to the existing six V1 factors?

Mass/inertia remain background/secondary anthropometry. The three common fpmax group factors remain valid V1 synthetic factors but are magnitude-dominant and should not be the primary signal source in a personalization-focused V2. They are preserved, not retroactively relabeled.

## Q8. Which 4-8D schemes are defensible?

S1 is a 4-D minimal field-consistent pilot candidate; S2 is a 7-D moderate candidate with more calibration burden; S3 is stress-only. None has frozen population bounds. Scheme ranking used model validity, geometry consistency, semantics, evidence, and parsimony—never oracle diversity.

## Q9. Is a new cohort and split required?

Yes: `NEW_VERSION_REQUIRED = true`. Any adopted structural scheme must become `{FUTURE_COHORT}` with a newly preregistered development/held-out split. The existing 32-subject V1 and its 24/8 split remain immutable; its sealed eight subjects are not automatically confirmatory for a new structural space.

## Q10. Is a small preregistered pilot ready?

**Ready with evidence gaps.** The exact fields and geometry-consistency rules are sufficiently clear for a small preregistered structural-integrity/non-proportionality pilot, but population bounds and some field-to-physiology mappings remain unresolved. The pilot should use nominal plus a few pre-frozen profiles and the same small geometry-selected V3 subset. It must not claim personalization and was not executed here.

## Stop state

- Current objective and normalization: unchanged.
- V3 parameterization/domain: unchanged.
- V1 cohort/32 subjects: unchanged.
- New subjects or truth: none.
- Held-out access: 0.
- Hardware/control/safety: untouched.
- Next stage `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1`: **not executed**.
"""


def build() -> None:
    import mujoco  # isolated MyoSuite environment only

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    audits = static_audits(mujoco, model)
    write_csv(OUTPUT / "MYOLEG_STRUCTURAL_PARAMETER_INVENTORY.csv", inventory_rows(mujoco, model))
    write_csv(OUTPUT / "MUSCLE_OPERATING_LENGTH_PARAMETER_AUDIT.csv", audits["operating"])
    write_csv(OUTPUT / "BIARTICULAR_COUPLING_PARAMETER_AUDIT.csv", audits["coupling"])
    write_csv(OUTPUT / "MOMENT_ARM_GEOMETRY_AUDIT.csv", audits["geometry"])
    write_csv(OUTPUT / "SEGMENT_GEOMETRY_ANTHROPOMETRY_AUDIT.csv", audits["segment"])
    write_csv(OUTPUT / "JOINT_PASSIVE_MECHANICS_AUDIT.csv", audits["passive"])
    write_csv(OUTPUT / "TENDON_PROPERTY_AUDIT.csv", audits["tendon"])
    write_csv(OUTPUT / "MUSCLE_GROUP_HETEROGENEITY_AUDIT.csv", audits["groups"])
    sensitivity, detail = local_sensitivity(mujoco, MODEL)
    write_csv(OUTPUT / "STRUCTURAL_LOCAL_SENSITIVITY_RESULTS.csv", sensitivity)
    write_json(OUTPUT / "STRUCTURAL_LOCAL_SENSITIVITY_DETAIL.json", detail)
    write_csv(OUTPUT / "HETEROGENEITY_FACTOR_TAXONOMY.csv", taxonomy_rows())
    write_json(OUTPUT / "PROPOSED_STRUCTURAL_HETEROGENEITY_SCHEMES.json", schemes())
    write_csv(OUTPUT / "FUTURE_RANGE_EVIDENCE_REQUIREMENTS.csv", range_requirements())
    write_csv(OUTPUT / "EVIDENCE_SOURCES.csv", sources())
    (OUTPUT / "COHORT_VERSIONING_AND_SPLIT_PLAN.md").write_text(
        "# Cohort Versioning and Split Plan\n\n"
        "`NEW_VERSION_REQUIRED = true`. Preserve `MYOLEG_VIRTUAL_PATIENT_COHORT_V1`, all 32 subject identities, and its 24/8 split. A scientifically changed structural parameterization must use `MYOLEG_VIRTUAL_PATIENT_COHORT_V2`.\n\n"
        "Before any V2 generation, freeze: factor semantics; evidence-backed bounds; joint distribution and dependence assumptions; deterministic seed; sample count; development/held-out identities; and all feasibility/integrity gates. Generate both new development and new held-out subjects from the same preregistered V2 design, while keeping held-out truth inaccessible until a separate confirmatory authorization. Do not recycle V1 development outcomes as V2 confirmation, and do not assume the V1 held-out set covers the V2 structural space.\n\n"
        "Recommended sequence: evidence closure -> `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1` -> V2 cohort design protocol -> V2 generation -> development-only truth work -> separately authorized confirmatory held-out stage. The pilot is not executed here.\n",
        encoding="utf-8",
    )
    (OUTPUT / "MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_REPORT.md").write_text(report_text(sensitivity, model), encoding="utf-8")
    metadata = {
        "stage_id": STAGE_ID, "protocol_id": PROTOCOL_ID, "outcome": OUTCOME,
        "protocol_sha256": FROZEN_SHA["protocol"], "frozen_inputs": frozen,
        "preserved_negative_results": ["V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "HETEROGENEITY_LIMITATION_DOMINANT"],
        "current_objective_information_retention_adequate_preserved": True,
        "current_heterogeneity_trajectory_interaction_limited_preserved": True,
        "cohort_v1_identity": COHORT_V1, "new_version_required": True,
        "future_cohort_identity": FUTURE_COHORT, "new_cohort_generated": False,
        "new_subjects_generated": 0, "new_truth_landscape_generated": False,
        "personalization_experiment_run": False, "objective_or_normalization_modified": False,
        "v3_parameterization_or_domain_modified": False, "fpmax_range_expanded": False,
        "held_out_scientific_access_count": 0, "held_out_truth_opened": False,
        "local_sensitivity_role": "LOCAL_NUMERICAL_SENSITIVITY_ONLY",
        "local_sensitivity_population_bounds": False,
        "local_sensitivity_trajectory_selection_uses_geometry_only": True,
        "local_sensitivity_uses_oracle_or_personalization_outcome": False,
        "truth_learner_parameterization_independence_audited": True,
        "robot_or_hardware": False, "next_stage_executed": False,
        "analysis_code_sha256": sha256(Path(__file__)),
        "formal_artifact_sha256_definition": "SHA-256 values in checksums.sha256; manifest excludes itself",
    }
    write_json(OUTPUT / "metadata.json", metadata)
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUTPUT / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files), encoding="utf-8")
    print(json.dumps({"stage_id": STAGE_ID, "outcome": OUTCOME, "protocol_sha256": FROZEN_SHA["protocol"], "held_out_scientific_access_count": 0, "artifact_count": len(files)}, indent=2))


if __name__ == "__main__":
    build()
