"""Build the mechanics-derived, outcome-independent S1 definition amendment."""

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


STAGE_ID = "MYOLEG_S1_STRUCTURAL_FACTOR_DEFINITION_AMENDMENT_V1"
AMENDED_ID = "S1_STRUCTURAL_DEFINITION_AMENDED_V1"
OUTCOME = "S1_STRUCTURAL_FACTOR_DEFINITION_AMENDED_READY_WITH_LIMITATIONS"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1"
PROTOCOL = OUTPUT / "S1_FACTOR_DEFINITION_AMENDMENT_PROTOCOL.json"
OLD_S1 = ROOT / "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1/PROPOSED_STRUCTURAL_HETEROGENEITY_SCHEMES.json"
EVIDENCE = ROOT / "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1/EVIDENCE_SOURCES.csv"
PILOT_DESIGN_METADATA = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1/metadata.json"
PILOT_DESIGN_REPORT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1/MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_REPORT.md"
MODEL = ROOT / "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml"
COORDINATE_MAPPING = ROOT / "external_simulation_audits/myoleg_supine_hip_knee_rehab_feasibility_v1/PROJECT_MYOLEG_COORDINATE_MAPPING.json"
V2_REFERENCE = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"
V3_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_BUILDER = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"
FORMAL_REFERENCE = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
COHORT_V1 = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"

FROZEN_SHA = {
    "protocol": "e09b7a47c6538dd91fe5f044514a2372d20d23a55b1dd490c67437b318223056",
    "old_s1": "47ebf27c43ccca9621e315c1322946bb7b8687e098b243ee8d92f0f66d578394",
    "evidence": "bc5c67abb6a9c8955f60b80fc4da31e736b731820fbd8b4ffffb3ddbd79426f3",
    "pilot_design_metadata": "e53d28c1f526615c7fb7976fce0e26c36bcf76ca7e529f23a6cf9f47bd213c23",
    "pilot_design_report": "b2ddf4290d42f98089e3e6629123d2a1127c1862c38e8fa04ccc16157fa642dc",
    "model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "coordinate_mapping": "83798958fd0b12f5c5314bc32df898f1d6e56d8e224f7673d8aca5c457ce713c",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "v3_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "v3_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "candidate_builder": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "replay_builder": "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "cohort_v1": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
}

FROZEN_PATHS = {
    "protocol": PROTOCOL, "old_s1": OLD_S1, "evidence": EVIDENCE,
    "pilot_design_metadata": PILOT_DESIGN_METADATA, "pilot_design_report": PILOT_DESIGN_REPORT,
    "model": MODEL, "coordinate_mapping": COORDINATE_MAPPING, "v2_reference": V2_REFERENCE,
    "v3_table": V3_TABLE, "v3_manifest": V3_MANIFEST,
    "candidate_builder": CANDIDATE_BUILDER, "replay_builder": REPLAY_BUILDER,
    "formal_manifest": FORMAL_MANIFEST, "formal_reference": FORMAL_REFERENCE, "cohort_v1": COHORT_V1,
}

EFFECTIVE_M = 1.0e-5
NEGLIGIBLE_OTHER_M = 1.0e-6
MIN_EFFECTIVE_FRACTION = 0.80
MIN_SIGN_FRACTION = 0.95
UNIT_Z = 1.0e-8

BIARTICULAR_EXPECTED = ("bflh_r", "grac_r", "recfem_r", "sart_r", "semimem_r", "semiten_r", "tfl_r")
RECTUS_ANATOMICAL = ("recfem_r",)
HAMSTRING_ANATOMICAL = ("bflh_r", "semimem_r", "semiten_r")
HIP_ANATOMY = {
    "addmagDist_r": "ANATOMICALLY_CONSISTENT_HIP_EXTENSOR_COMPONENT",
    "addmagIsch_r": "ANATOMICALLY_CONSISTENT_HIP_EXTENSOR_COMPONENT",
    "addmagMid_r": "ANATOMICALLY_CONSISTENT_HIP_EXTENSOR_COMPONENT",
    "glmax2_r": "ANATOMICALLY_CONSISTENT_HIP_EXTENSOR",
    "glmax3_r": "ANATOMICALLY_CONSISTENT_HIP_EXTENSOR",
    "glmed3_r": "ANATOMICALLY_AMBIGUOUS_MULTI_ACTION_BUT_MODEL_MECHANICS_CLEAR",
    "piri_r": "ANATOMICALLY_AMBIGUOUS_CONFIGURATION_DEPENDENT_ACTION_BUT_MODEL_MECHANICS_CLEAR",
}
KNEE_ANATOMY = {
    "vasint_r": "ANATOMICALLY_CONSISTENT_KNEE_EXTENSOR",
    "vaslat_r": "ANATOMICALLY_CONSISTENT_KNEE_EXTENSOR",
    "vasmed_r": "ANATOMICALLY_CONSISTENT_KNEE_EXTENSOR",
}


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def verify_inputs() -> dict[str, str]:
    actual = {name: sha256(path) for name, path in FROZEN_PATHS.items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")
    old = json.loads(OLD_S1.read_text(encoding="utf-8"))
    s1 = next(row for row in old["schemes"] if row["scheme_id"] == "S1_MINIMAL_STRUCTURAL")
    if s1["dimensionality"] != 4 or len(s1["factors"]) != 4:
        raise RuntimeError("old S1 identity changed")
    pilot = json.loads(PILOT_DESIGN_METADATA.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    mapping = json.loads(COORDINATE_MAPPING.read_text(encoding="utf-8"))
    if pilot["outcome"] != "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY" or pilot["blocker"] != "S1_DEFINITION_INCOMPLETE":
        raise RuntimeError("pilot design blocker changed")
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA["formal_reference"]
        and mapping["joint_mapping"]["signs"] == {"hip": 1.0, "knee": 1.0}
    ):
        raise RuntimeError("coordinate or formal convention changed")
    return actual


def actuator_name(mujoco: Any, model: Any, index: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"unnamed_{index}"


def membership_audit(mujoco: Any, model: Any) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    candidate = load_module(CANDIDATE_BUILDER, "_s1_amendment_candidate_builder")
    replay = load_module(REPLAY_BUILDER, "_s1_amendment_replay_builder")
    reference = candidate.load_reference_adapter()
    if len(reference["q"]) != 401:
        raise RuntimeError("frozen reference state count changed")
    data = mujoco.MjData(model)
    effects = np.zeros((401, model.nu, 2), dtype=float)
    muscle_force_sign_checks = []
    for state, q in enumerate(reference["q"]):
        replay.reset_to_target_state(model, data, q, np.zeros(2), np.zeros(2))
        tangent = replay.independent_coordinate_tangent(model, data)
        mujoco.mj_forward(model, data)
        # MuJoCo muscle actuators produce negative scalar tensile force for
        # positive activation.  Unit positive muscle tension therefore maps
        # through the negative projected actuator moment.
        effects[state] = -replay.sparse_actuator_moment_times_tangent(model, data, tangent)
        if state in (0, 200, 400):
            data.act[:] = 0.5
            mujoco.mj_forward(model, data)
            muscle_force_sign_checks.append(bool(np.all(np.asarray(data.actuator_force) < 0.0)))
    if not np.isfinite(effects).all() or not all(muscle_force_sign_checks):
        raise RuntimeError("moment-arm classification sign/integrity failure")

    rows = []
    groups = {"biarticular": [], "hip_antagonist": [], "knee_antagonist": [], "rectus": [], "hamstring": [], "balance_unchanged": []}
    for actuator in range(model.nu):
        name = actuator_name(mujoco, model, actuator)
        if not name.endswith("_r"):
            continue
        values = effects[:, actuator]
        effective = np.abs(values) >= EFFECTIVE_M
        effective_fraction = np.mean(effective, axis=0)
        maximum = np.max(np.abs(values), axis=0)
        positive = [float(np.mean(values[effective[:, j], j] > 0.0)) if np.any(effective[:, j]) else 0.0 for j in (0, 1)]
        negative = [float(np.mean(values[effective[:, j], j] < 0.0)) if np.any(effective[:, j]) else 0.0 for j in (0, 1)]
        hip_mono = bool(effective_fraction[0] >= MIN_EFFECTIVE_FRACTION and maximum[1] <= NEGLIGIBLE_OTHER_M)
        knee_mono = bool(effective_fraction[1] >= MIN_EFFECTIVE_FRACTION and maximum[0] <= NEGLIGIBLE_OTHER_M)
        biarticular = bool(np.all(effective_fraction >= MIN_EFFECTIVE_FRACTION))
        hip_antagonist = bool(hip_mono and negative[0] >= MIN_SIGN_FRACTION)
        knee_antagonist = bool(knee_mono and negative[1] >= MIN_SIGN_FRACTION)
        rectus_like = bool(biarticular and positive[0] >= MIN_SIGN_FRACTION and negative[1] >= MIN_SIGN_FRACTION)
        hamstring_like = bool(biarticular and negative[0] >= MIN_SIGN_FRACTION and positive[1] >= MIN_SIGN_FRACTION)
        if biarticular:
            groups["biarticular"].append(name)
        if hip_antagonist:
            groups["hip_antagonist"].append(name)
        if knee_antagonist:
            groups["knee_antagonist"].append(name)
        if rectus_like and name in RECTUS_ANATOMICAL:
            groups["rectus"].append(name)
        elif hamstring_like and name in HAMSTRING_ANATOMICAL:
            groups["hamstring"].append(name)
        elif biarticular:
            groups["balance_unchanged"].append(name)
        structural = "BIARTICULAR" if biarticular else "HIP_MONOARTICULAR" if hip_mono else "KNEE_MONOARTICULAR" if knee_mono else "OTHER_OR_AMBIGUOUS"
        family_pattern = "RECTUS_LIKE" if rectus_like else "HAMSTRING_LIKE" if hamstring_like else "OTHER"
        if name in RECTUS_ANATOMICAL:
            anatomy = "ANATOMICALLY_CONSISTENT_RECTUS_FEMORIS"
        elif name in HAMSTRING_ANATOMICAL:
            anatomy = "ANATOMICALLY_CONSISTENT_HAMSTRING"
        elif name in HIP_ANATOMY:
            anatomy = HIP_ANATOMY[name]
        elif name in KNEE_ANATOMY:
            anatomy = KNEE_ANATOMY[name]
        elif rectus_like or hamstring_like:
            anatomy = "MECHANICAL_FAMILY_PATTERN_BUT_NOT_NAMED_ANATOMICAL_FAMILY"
        else:
            anatomy = "NOT_APPLICABLE_OR_NOT_MECHANICALLY_INCLUDED"
        final_balance = "RECTUS_GROUP" if name in groups["rectus"] else "HAMSTRING_GROUP" if name in groups["hamstring"] else "UNCHANGED_BY_BALANCE_FACTOR"
        rows.append({
            "actuator_id": actuator, "muscle": name, "state_count": 401,
            "hip_effective_fraction": f"{effective_fraction[0]:.12g}", "knee_effective_fraction": f"{effective_fraction[1]:.12g}",
            "hip_max_abs_effect_m": f"{maximum[0]:.12g}", "knee_max_abs_effect_m": f"{maximum[1]:.12g}",
            "hip_positive_fraction": f"{positive[0]:.12g}", "hip_negative_fraction": f"{negative[0]:.12g}",
            "knee_positive_fraction": f"{positive[1]:.12g}", "knee_negative_fraction": f"{negative[1]:.12g}",
            "structural_class": structural, "biarticular_included": biarticular,
            "hip_antagonist_included": hip_antagonist, "knee_antagonist_included": knee_antagonist,
            "mechanical_family_pattern": family_pattern, "anatomical_review": anatomy,
            "final_balance_group": final_balance,
            "classification_basis": "unit positive muscle tension effect=-projected actuator moment; no J/oracle/rank",
        })
    if tuple(groups["biarticular"]) != BIARTICULAR_EXPECTED:
        raise RuntimeError(f"frozen seven-muscle biarticular group changed: {groups['biarticular']}")
    if groups["rectus"] != ["recfem_r"] or groups["hamstring"] != ["bflh_r", "semimem_r", "semiten_r"]:
        raise RuntimeError(f"rectus/hamstring family is not well-defined: {groups}")
    evidence = {
        "reference_state_count": 401, "moment_effect_finite": True,
        "positive_activation_muscle_force_negative_at_states_0_200_400": all(muscle_force_sign_checks),
        "effect_sign_definition": "unit positive muscle tension generalized effect = -projected actuator moment",
        "thresholds": {"effective_m": EFFECTIVE_M, "negligible_other_m": NEGLIGIBLE_OTHER_M,
                       "minimum_effective_fraction": MIN_EFFECTIVE_FRACTION, "minimum_sign_fraction": MIN_SIGN_FRACTION},
    }
    return rows, groups, evidence


def group_membership_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hip = []
    knee = []
    for row in rows:
        if row["structural_class"] == "HIP_MONOARTICULAR":
            hip.append({
                "actuator_id": row["actuator_id"], "muscle": row["muscle"],
                "mechanically_included": row["hip_antagonist_included"],
                "target_effective_fraction": row["hip_effective_fraction"],
                "target_negative_fraction": row["hip_negative_fraction"],
                "other_joint_max_abs_m": row["knee_max_abs_effect_m"],
                "anatomical_review": row["anatomical_review"],
                "exclusion_reason": "" if row["hip_antagonist_included"] else "target sign consistency below frozen 0.95 antagonist threshold",
            })
        if row["structural_class"] == "KNEE_MONOARTICULAR":
            knee.append({
                "actuator_id": row["actuator_id"], "muscle": row["muscle"],
                "mechanically_included": row["knee_antagonist_included"],
                "target_effective_fraction": row["knee_effective_fraction"],
                "target_negative_fraction": row["knee_negative_fraction"],
                "other_joint_max_abs_m": row["hip_max_abs_effect_m"],
                "anatomical_review": row["anatomical_review"],
                "exclusion_reason": "" if row["knee_antagonist_included"] else "target sign consistency below frozen 0.95 antagonist threshold",
            })
    return hip, knee


def topology_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name in ("actuator_trnid", "actuator_lengthrange", "tendon_adr", "tendon_num", "wrap_type", "wrap_objid",
                 "site_bodyid", "site_pos", "jnt_type", "jnt_range", "eq_type", "eq_obj1id", "eq_obj2id", "eq_data"):
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def ids(mujoco: Any, model: Any, names: list[str] | tuple[str, ...]) -> list[int]:
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in names]


def apply_lmax(mujoco: Any, model: Any, groups: dict[str, list[str]], z: float) -> None:
    for identifier in ids(mujoco, model, groups["biarticular"]):
        value = float(model.actuator_gainprm[identifier, 5]) * math.exp(z)
        model.actuator_gainprm[identifier, 5] = value
        model.actuator_biasprm[identifier, 5] = value


def apply_balance(mujoco: Any, model: Any, groups: dict[str, list[str]], z: float) -> None:
    for key, sign in (("rectus", 1.0), ("hamstring", -1.0)):
        for identifier in ids(mujoco, model, groups[key]):
            value = float(model.actuator_gainprm[identifier, 7]) * math.exp(sign * z)
            model.actuator_gainprm[identifier, 7] = value
            model.actuator_biasprm[identifier, 7] = value


def apply_f0(mujoco: Any, model: Any, members: list[str], z: float) -> None:
    for identifier in ids(mujoco, model, members):
        value = float(model.actuator_gainprm[identifier, 2]) * math.exp(z)
        model.actuator_gainprm[identifier, 2] = value
        model.actuator_biasprm[identifier, 2] = value


def operator_integrity(mujoco: Any, groups: dict[str, list[str]], reference_q: np.ndarray) -> list[dict[str, Any]]:
    specs: list[tuple[str, int, list[str], Callable[[Any, Any, float], None]]] = [
        ("S1F1_BIARTICULAR_LMAX", 5, groups["biarticular"], lambda mj, m, z: apply_lmax(mj, m, groups, z)),
        ("S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE", 7, groups["rectus"] + groups["hamstring"], lambda mj, m, z: apply_balance(mj, m, groups, z)),
        ("S1F3_HIP_MONO_ANTAGONIST_F0", 2, groups["hip_antagonist"], lambda mj, m, z: apply_f0(mj, m, groups["hip_antagonist"], z)),
        ("S1F4_KNEE_MONO_ANTAGONIST_F0", 2, groups["knee_antagonist"], lambda mj, m, z: apply_f0(mj, m, groups["knee_antagonist"], z)),
    ]
    replay = load_module(REPLAY_BUILDER, "_s1_amendment_replay_integrity")
    base = mujoco.MjModel.from_xml_path(str(MODEL))
    base_gain = np.asarray(base.actuator_gainprm).copy()
    base_bias = np.asarray(base.actuator_biasprm).copy()
    base_topology = topology_fingerprint(base)
    rows = []
    for factor_id, field, members, operator in specs:
        nominal = mujoco.MjModel.from_xml_path(str(MODEL))
        operator(mujoco, nominal, 0.0)
        identity = bool(np.array_equal(nominal.actuator_gainprm, base_gain) and np.array_equal(nominal.actuator_biasprm, base_bias))
        perturbed = mujoco.MjModel.from_xml_path(str(MODEL))
        operator(mujoco, perturbed, UNIT_Z)
        target_ids = ids(mujoco, perturbed, members)
        expected_gain = base_gain.copy()
        expected_bias = base_bias.copy()
        if factor_id == "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE":
            for key, sign in (("rectus", 1.0), ("hamstring", -1.0)):
                target = ids(mujoco, perturbed, groups[key])
                expected_gain[target, field] *= math.exp(sign * UNIT_Z)
                expected_bias[target, field] *= math.exp(sign * UNIT_Z)
        else:
            expected_gain[target_ids, field] *= math.exp(UNIT_Z)
            expected_bias[target_ids, field] *= math.exp(UNIT_Z)
        only_declared = bool(np.array_equal(perturbed.actuator_gainprm, expected_gain) and np.array_equal(perturbed.actuator_biasprm, expected_bias))
        synchronized = bool(np.array_equal(perturbed.actuator_gainprm[:, field], perturbed.actuator_biasprm[:, field]))
        positive = bool(np.all(perturbed.actuator_gainprm[target_ids, field] > 0.0))
        lmin_lt_lmax = True
        if field == 5:
            lmin_lt_lmax = bool(np.all(perturbed.actuator_gainprm[target_ids, 4] < perturbed.actuator_gainprm[target_ids, 5]))
        data = mujoco.MjData(perturbed)
        replay.reset_to_target_state(perturbed, data, reference_q, np.zeros(2), np.zeros(2))
        mujoco.mj_forward(perturbed, data)
        finite = bool(np.isfinite(data.qfrc_actuator).all() and np.isfinite(data.actuator_length).all())
        warnings = int(np.asarray(data.warning.number, dtype=np.int64).sum())
        topology = topology_fingerprint(perturbed) == base_topology
        passed = all((identity, only_declared, synchronized, positive, lmin_lt_lmax, finite, warnings == 0, topology))
        rows.append({
            "factor_id": factor_id, "member_count": len(members), "unit_z": UNIT_Z,
            "nominal_identity_bitwise": identity, "only_declared_members_and_field_changed": only_declared,
            "gain_bias_synchronized": synchronized, "positive_domain": positive,
            "lmin_lt_lmax": lmin_lt_lmax, "single_state_forward_finite": finite,
            "solver_warning_count": warnings, "topology_fingerprint_unchanged": topology,
            "scientific_trajectory_replay": False, "J_or_oracle_read": False, "pass": passed,
        })
    return rows


def operator_audits(groups: dict[str, list[str]], model: Any, mujoco: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lmax_values = {name: float(model.actuator_gainprm[ids(mujoco, model, [name])[0], 5]) for name in groups["biarticular"]}
    lmin_values = {name: float(model.actuator_gainprm[ids(mujoco, model, [name])[0], 4]) for name in groups["biarticular"]}
    ratios = [lmin_values[name] / lmax_values[name] for name in groups["biarticular"] if lmin_values[name] > 0.0]
    lmax = {
        "factor_id": "S1F1_BIARTICULAR_LMAX", "chosen_operator": "L1_LOG_MULTIPLICATIVE_LMAX_ONLY",
        "members": groups["biarticular"], "fields": ["actuator_gainprm[:,5]", "actuator_biasprm[:,5]"],
        "nominal_lmin": lmin_values, "nominal_lmax": lmax_values,
        "operator": "for each member i: lmax_i(z)=lmax_i0*exp(z); set gainprm[i,5]=biasprm[i,5]=lmax_i(z)",
        "nominal_identity": "z=0", "hard_mathematical_domain": f"z > max_i log(lmin_i0/lmax_i0) = {max(math.log(value) for value in ratios):.12g}; no population or pilot bound",
        "unchanged_fields": ["lmin", "range", "actuator_lengthrange", "force", "fpmax", "transmission", "geometry"],
        "semantic": "changes MuJoCo built-in normalized lmax curve parameter; may affect active/passive curve evaluation; not optimal fiber length or tendon slack length",
        "limitations": ["physiological one-to-one mapping unavailable", "population range unavailable"],
        "alternatives": [
            {"id": "L1", "decision": "CHOSEN", "reason": "only inherited field, positive/invertible, relative to each nominal"},
            {"id": "L2", "decision": "REJECTED", "reason": "changes lmin outside inherited exact field and remaps curve width"},
            {"id": "L3", "decision": "REJECTED", "reason": "additive rule lacks global positivity and is not relative to heterogeneous nominals"},
        ],
        "selection_used_scientific_outcome": False,
    }
    balance = {
        "factor_id": "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE", "chosen_operator": "B2_LOG_SYMMETRIC_FAMILY_CENTERS",
        "rectus_group": groups["rectus"], "hamstring_group": groups["hamstring"],
        "unchanged_biarticular": groups["balance_unchanged"],
        "fields": ["actuator_gainprm[:,7]", "actuator_biasprm[:,7]"],
        "operator": "rectus fpmax_i(z)=fpmax_i0*exp(z); hamstring fpmax_i(z)=fpmax_i0*exp(-z); synchronize gain/bias; other biarticular unchanged",
        "nominal_identity": "z=0", "hard_mathematical_domain": "all finite real z preserves positivity; no population or pilot bound",
        "balance_invariant": "unweighted mean of the rectus-family and hamstring-family log-scale centers remains zero; family definition is group-size invariant",
        "global_scale_policy": "V1 common biarticular fpmax is removed from the future personalization-focused scheme; no duplicated global-plus-balance degree of freedom",
        "alternatives": [
            {"id": "B1", "decision": "REJECTED", "reason": "positivity requires abs(delta)<1 and inverse symmetry is limited"},
            {"id": "B2", "decision": "CHOSEN", "reason": "positive, invertible, zero-symmetric, and family-center interpretation is independent of group size"},
            {"id": "B3", "decision": "REJECTED", "reason": "individual-count weighting makes family-center displacement asymmetric"},
        ],
        "selection_used_scientific_outcome": False,
    }
    f0 = {
        "factor_ids": ["S1F3_HIP_MONO_ANTAGONIST_F0", "S1F4_KNEE_MONO_ANTAGONIST_F0"],
        "chosen_operator": "F1_LOG_GROUP_SCALE", "hip_members": groups["hip_antagonist"], "knee_members": groups["knee_antagonist"],
        "fields": ["actuator_gainprm[:,2]", "actuator_biasprm[:,2]"],
        "operator": "for each included member i: F0_i(z)=F0_i0*exp(z); synchronize gain/bias; all nonmembers unchanged",
        "nominal_identity": "z=0", "hard_mathematical_domain": "all finite real z preserves positivity; no population or pilot bound",
        "semantic": "MuJoCo muscle force scale F0 multiplies the built-in muscle active-plus-passive force expression; not pure active strength",
        "member_weighting": "same relative scalar applied to each member's own nominal F0; no ad hoc muscle weights",
        "alternatives": [
            {"id": "F1", "decision": "CHOSEN", "reason": "positive, invertible, relative, exact nominal identity"},
            {"id": "F2", "decision": "REJECTED", "reason": "linear scaling requires z>-1 and is less symmetric under inverse perturbation"},
        ],
        "selection_used_scientific_outcome": False,
    }
    return lmax, balance, f0


def audit_markdown(title: str, audit: dict[str, Any]) -> str:
    return f"# {title}\n\n```json\n{json.dumps(audit, indent=2, sort_keys=True)}\n```\n"


def relationship_rows() -> list[dict[str, Any]]:
    return [
        {"entity": "FEMUR_MASS_INERTIA_SCALE", "entity_type": "V1", "field": "body_mass/body_inertia", "relationship": "RETAIN_AS_BACKGROUND", "relative_to": "amended S1", "reason": "anthropometric background; no exact structural-field overlap"},
        {"entity": "TIBIA_PATELLA_MASS_INERTIA_SCALE", "entity_type": "V1", "field": "body_mass/body_inertia", "relationship": "RETAIN_AS_BACKGROUND", "relative_to": "amended S1", "reason": "anthropometric background; no exact structural-field overlap"},
        {"entity": "FOOT_COMPLEX_MASS_INERTIA_SCALE", "entity_type": "V1", "field": "body_mass/body_inertia", "relationship": "RETAIN_AS_BACKGROUND", "relative_to": "amended S1", "reason": "anthropometric background; no exact structural-field overlap"},
        {"entity": "HIP_ONLY_PASSIVE_FP_MAX_SCALE", "entity_type": "V1", "field": "gainprm/biasprm[7] hip-only", "relationship": "SECONDARY_ONLY", "relative_to": "S1F3 F0", "reason": "different field but prior magnitude-dominant synthetic factor; not primary personalization signal"},
        {"entity": "KNEE_ONLY_PASSIVE_FP_MAX_SCALE", "entity_type": "V1", "field": "gainprm/biasprm[7] knee-only", "relationship": "SECONDARY_ONLY", "relative_to": "S1F4 F0", "reason": "different field but prior magnitude-dominant synthetic factor; not primary personalization signal"},
        {"entity": "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE", "entity_type": "V1", "field": "gainprm/biasprm[7] all biarticular", "relationship": "REMOVE_FROM_PERSONALIZATION_FOCUSED_COHORT", "relative_to": "S1F1/S1F2", "reason": "avoid duplicated global fpmax magnitude with within-family balance; future global-plus-balance decomposition requires a new version"},
        {"entity": "S1F1_BIARTICULAR_LMAX", "entity_type": "AMENDED_S1", "field": "gainprm/biasprm[5]", "relationship": "REPLACED_BY_STRUCTURAL_FACTOR", "relative_to": "V1 biarticular common fpmax role", "reason": "future design relationship only; not physiological equivalence"},
        {"entity": "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE", "entity_type": "AMENDED_S1", "field": "gainprm/biasprm[7]", "relationship": "REPLACED_BY_STRUCTURAL_FACTOR", "relative_to": "V1 biarticular common fpmax", "reason": "within-family balance replaces common magnitude as primary structural role"},
        {"entity": "S1F3_HIP_MONO_ANTAGONIST_F0", "entity_type": "AMENDED_S1", "field": "gainprm/biasprm[2]", "relationship": "REPLACED_BY_STRUCTURAL_FACTOR", "relative_to": "V1 hip-only fpmax primary role", "reason": "future design relationship only; distinct actual field semantics"},
        {"entity": "S1F4_KNEE_MONO_ANTAGONIST_F0", "entity_type": "AMENDED_S1", "field": "gainprm/biasprm[2]", "relationship": "REPLACED_BY_STRUCTURAL_FACTOR", "relative_to": "V1 knee-only fpmax primary role", "reason": "future design relationship only; distinct actual field semantics"},
    ]


def report(groups: dict[str, list[str]], integrity: list[dict[str, Any]], amended_sha: str) -> str:
    return f"""# MyoLeg S1 Structural Factor Definition Amendment V1

## Formal outcome

**{OUTCOME}**

Amended identity: `{AMENDED_ID}`  
Old S1 preserved SHA: `{FROZEN_SHA['old_s1']}`  
Amended definition SHA: `{amended_sha}`

This stage used 401 frozen reference states only for mechanics-based projected moment-arm classification. It did not compute torque truth, J, oracle, rank, personalization, or held-out results. No pilot, subject, cohort, landscape, learner or optimizer was executed.

## Q1. Exact groups

- Biarticular lmax: `{', '.join(groups['biarticular'])}`.
- `RECTUS_GROUP`: `{', '.join(groups['rectus'])}`.
- `HAMSTRING_GROUP`: `{', '.join(groups['hamstring'])}`.
- Balance-unchanged biarticular: `{', '.join(groups['balance_unchanged'])}`.
- Hip monoarticular antagonist F0: `{', '.join(groups['hip_antagonist'])}`.
- Knee monoarticular antagonist F0: `{', '.join(groups['knee_antagonist'])}`.

## Q2. Membership rule

Membership came from the projected actuator moment matrix in project hip/knee coordinates. Near-zero values below 1e-5 m were excluded from sign counts; target coverage had to reach 80%, sign consistency 95%, and a monoarticular non-target joint maximum had to remain at or below 1e-6 m. MuJoCo positive activation produces negative scalar muscle force, so unit positive muscle-tension effect is `-projected actuator moment`. Anatomy was reviewed afterward and did not override mechanical measurements.

`grac_r` is mechanically hamstring-like and `tfl_r` rectus-like, but neither belongs to the named anatomical family; both remain explicitly unchanged. `sart_r` has neither reciprocal family sign pattern. Hip members `glmed3_r` and `piri_r` remain mechanically clear but anatomically multi-action/ambiguous; this is a declared limitation rather than a hidden exclusion.

## Q3. Biarticular lmax operator

For every seven-muscle member, `lmax_i(z)=lmax_i0*exp(z)` and both gain/bias index 5 are assigned the same value. Each muscle uses its own nominal. `lmin`, `range`, `actuator_lengthrange`, geometry and transmission remain unchanged. This is a normalized MuJoCo curve parameter, not optimal fiber length or tendon slack length.

## Q4. Rectus-hamstring fpmax balance

`recfem_r` uses `fpmax_i(z)=fpmax_i0*exp(z)`; `bflh_r`, `semimem_r`, and `semiten_r` use `fpmax_i(z)=fpmax_i0*exp(-z)`. The family log-centers move symmetrically; `grac_r`, `sart_r`, and `tfl_r` remain unchanged. The V1 global biarticular fpmax factor is removed from a future personalization-focused scheme to avoid duplicated magnitude/balance degrees of freedom.

## Q5. Hip/knee F0 operators

Each included group member uses `F0_i(z)=F0_i0*exp(z)` relative to its own nominal, synchronized at gain/bias index 2. No muscle-specific weights are used. F0 scales the built-in active-plus-passive force expression; it is not labeled pure active strength.

## Q6. Nominal identity

All four operators recover the base targeted arrays bitwise at `z=0`.

## Q7. Field synchronization and integrity

At the unit-only `z=1e-8` check, only declared members/fields changed, gain/bias stayed synchronized, positivity and lmin/lmax domains passed, topology remained exact, forward state was finite, and warnings were zero. Integrity rows passed: **{sum(bool(row['pass']) for row in integrity)}/{len(integrity)}**.

## Q8. V1 relationship

Femur, tibia/patella and foot mass/inertia are retained as background. Hip-only and knee-only common fpmax become secondary only. Common biarticular fpmax is removed from a future personalization-focused cohort. The amended factors replace those old factors' primary structural role, not their physiological meaning. Final Cohort V2 composition is not frozen here.

## Q9. Ambiguity and outcome independence

All four factors now have exact members, fields, mathematical operators, nominal identity, domain invariants and V1 relationships. Selection used field semantics, mechanics, invertibility and consistency only; no personalization outcome was read. Remaining limitations are range calibration, indirect physiological mapping, and two anatomically multi-action hip members.

## Q10. Pilot-design V2 readiness

**Yes, with limitations.** The amended SHA may be used as the authoritative input to `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2`. That stage may freeze pilot diagnostic levels, trajectory subset and numeric scientific gates. It was not executed automatically.

## Stop state

- Scientific pilot/replay: **0 / 0**.
- New subjects/cohort/landscape: **none**.
- Held-out scientific access: **0**.
- Population range or pilot diagnostic level frozen: **no**.
- V1/V3/objective/normalization: **unchanged**.
- Robot/hardware: **untouched**.
"""


def build() -> None:
    import mujoco

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = verify_inputs()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    rows, groups, membership_evidence = membership_audit(mujoco, model)
    hip, knee = group_membership_tables(rows)
    write_csv(OUTPUT / "S1_GROUP_MEMBERSHIP_AUDIT.csv", rows)
    write_csv(OUTPUT / "HIP_ANTAGONIST_MEMBERSHIP.csv", hip)
    write_csv(OUTPUT / "KNEE_ANTAGONIST_MEMBERSHIP.csv", knee)

    lmax, balance, f0 = operator_audits(groups, model, mujoco)
    write_json(OUTPUT / "BIARTICULAR_LMAX_OPERATOR_AUDIT.json", lmax)
    write_json(OUTPUT / "RECTUS_HAMSTRING_BALANCE_OPERATOR_AUDIT.json", balance)
    write_json(OUTPUT / "F0_OPERATOR_AUDIT.json", f0)
    (OUTPUT / "BIARTICULAR_LMAX_OPERATOR_AUDIT.md").write_text(audit_markdown("Biarticular lmax Operator Audit", lmax), encoding="utf-8")
    (OUTPUT / "RECTUS_HAMSTRING_BALANCE_OPERATOR_AUDIT.md").write_text(audit_markdown("Rectus-Hamstring Balance Operator Audit", balance), encoding="utf-8")
    (OUTPUT / "F0_OPERATOR_AUDIT.md").write_text(audit_markdown("F0 Operator Audit", f0), encoding="utf-8")
    write_csv(OUTPUT / "V1_S1_FACTOR_RELATIONSHIP.csv", relationship_rows())

    candidate = load_module(CANDIDATE_BUILDER, "_s1_amendment_candidate_integrity")
    reference = candidate.load_reference_adapter()
    integrity = operator_integrity(mujoco, groups, np.asarray(reference["q"][200], dtype=float))
    if not all(row["pass"] for row in integrity):
        raise RuntimeError("operator identity/integrity failure")
    write_csv(OUTPUT / "OPERATOR_IDENTITY_AND_INTEGRITY_CHECKS.csv", integrity)

    amended = {
        "definition_id": AMENDED_ID, "stage_id": STAGE_ID, "old_s1_identity": "S1_MINIMAL_STRUCTURAL",
        "old_s1_sha256": FROZEN_SHA["old_s1"], "old_s1_overwritten": False,
        "declared_dimensionality": 4, "all_factors_resolved": True,
        "membership_rule": json.loads(PROTOCOL.read_text(encoding="utf-8"))["membership_derivation"],
        "membership_evidence": membership_evidence,
        "factors": [
            {"factor_id": "S1F1_BIARTICULAR_LMAX", "inherited_name": "biarticular normalized-curve lmax profile",
             "exact_members": groups["biarticular"], "exact_fields": lmax["fields"], "operator": lmax["operator"],
             "nominal_identity": lmax["nominal_identity"], "invariants": ["gain/bias synchronized", "lmin<lmax", "range/lengthrange/geometry unchanged"],
             "semantics": lmax["semantic"], "known_limitations": lmax["limitations"],
             "v1_relationship": "collectively replaces V1 common biarticular fpmax as primary structural role",
             "evidence_provenance": ["MuJoCo official muscle semantics", "compiled MyoLeg fields", "401-state moment classification"]},
            {"factor_id": "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE", "inherited_name": "rectus-vs-hamstring relative fpmax balance",
             "exact_members": {"RECTUS_GROUP": groups["rectus"], "HAMSTRING_GROUP": groups["hamstring"], "UNCHANGED": groups["balance_unchanged"]},
             "exact_fields": balance["fields"], "operator": balance["operator"], "nominal_identity": balance["nominal_identity"],
             "invariants": ["gain/bias synchronized", "positive fpmax", "family log-center balance", "nonfamily unchanged"],
             "semantics": "within-biarticular passive-force curve family balance; no global common scale",
             "known_limitations": ["population range unavailable", "fpmax is not measured tissue stiffness"],
             "v1_relationship": "replaces V1 common biarticular fpmax primary role; V1 global factor removed",
             "evidence_provenance": ["compiled transmission and 401-state torque signs", "post-mechanical anatomical review"]},
            {"factor_id": "S1F3_HIP_MONO_ANTAGONIST_F0", "inherited_name": "hip monoarticular antagonist relative F0",
             "exact_members": groups["hip_antagonist"], "exact_fields": f0["fields"], "operator": f0["operator"],
             "nominal_identity": f0["nominal_identity"], "invariants": ["gain/bias synchronized", "positive F0", "nonmembers unchanged"],
             "semantics": f0["semantic"], "known_limitations": ["population range unavailable", "glmed3_r and piri_r anatomical action is multi-action/ambiguous"],
             "v1_relationship": "replaces V1 hip-only fpmax primary role; V1 factor secondary only",
             "evidence_provenance": ["compiled monoarticular transmission", "401-state negative flexion-effect consistency"]},
            {"factor_id": "S1F4_KNEE_MONO_ANTAGONIST_F0", "inherited_name": "knee monoarticular antagonist relative F0",
             "exact_members": groups["knee_antagonist"], "exact_fields": f0["fields"], "operator": f0["operator"],
             "nominal_identity": f0["nominal_identity"], "invariants": ["gain/bias synchronized", "positive F0", "nonmembers unchanged"],
             "semantics": f0["semantic"], "known_limitations": ["population range unavailable"],
             "v1_relationship": "replaces V1 knee-only fpmax primary role; V1 factor secondary only",
             "evidence_provenance": ["compiled monoarticular transmission", "401-state negative flexion-effect consistency"]},
        ],
        "population_ranges_frozen": False, "pilot_diagnostic_levels_frozen": False,
        "scientific_outcome_used": False, "J_or_oracle_or_rank_used": False,
        "operator_unit_checks_all_pass": True, "next_authoritative_use": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2",
    }
    write_json(OUTPUT / "S1_STRUCTURAL_DEFINITION_AMENDED_V1.json", amended)
    amended_sha = sha256(OUTPUT / "S1_STRUCTURAL_DEFINITION_AMENDED_V1.json")
    write_json(OUTPUT / "SOURCE_AND_EVIDENCE_METADATA.json", {
        "frozen_inputs": frozen, "protocol_sha256": frozen["protocol"],
        "amended_definition_sha256": amended_sha, "reference_states_used": 401,
        "reference_role": "mechanical membership classification only",
        "muscle_force_sign_semantic": membership_evidence["effect_sign_definition"],
        "inherited_evidence_source_count": sum(1 for _ in csv.DictReader(EVIDENCE.open(newline="", encoding="utf-8"))),
        "anatomical_review_source_basis": ["Rajagopal et al. lower-limb model context in inherited evidence", "compiled MyoLeg paths and moment signs"],
        "scientific_trajectory_truth_replay": False, "J_or_oracle_or_rank_read": False,
        "held_out_scientific_access_count": 0, "new_external_range_bound": False,
    })
    (OUTPUT / "MYOLEG_S1_STRUCTURAL_FACTOR_DEFINITION_AMENDMENT_REPORT.md").write_text(report(groups, integrity, amended_sha), encoding="utf-8")
    write_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID, "outcome": OUTCOME, "amended_definition_id": AMENDED_ID,
        "old_s1_sha256": FROZEN_SHA["old_s1"], "old_s1_overwritten": False,
        "amended_definition_sha256": amended_sha, "all_four_factors_resolved": True,
        "operator_integrity_pass_count": sum(bool(row["pass"]) for row in integrity),
        "operator_integrity_check_count": len(integrity), "population_ranges_frozen": False,
        "pilot_diagnostic_levels_frozen": False, "scientific_pilot_executed": False,
        "scientific_trajectory_replays": 0, "J_or_oracle_or_rank_read": False,
        "held_out_scientific_access_count": 0, "new_virtual_subjects": 0,
        "cohort_v2_generated": False, "truth_landscape_generated": False,
        "v1_cohort_modified": False, "v3_parameterization_or_domain_modified": False,
        "objective_or_normalization_modified": False, "five_parameter_or_ml_training": False,
        "bo_run": False, "robot_or_hardware": False,
        "next_stage": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2",
        "next_stage_executed": False, "analysis_code_sha256": sha256(Path(__file__)),
    })
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUTPUT / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"stage_id": STAGE_ID, "outcome": OUTCOME,
                      "amended_definition_sha256": amended_sha,
                      "groups": groups, "operator_checks": f"{len(integrity)}/{len(integrity)}",
                      "held_out_scientific_access_count": 0}, indent=2))


if __name__ == "__main__":
    build()
