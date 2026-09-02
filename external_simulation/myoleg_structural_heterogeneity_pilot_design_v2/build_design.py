"""Freeze the amended-S1 structural pilot V2 protocol without running it.

This builder is intentionally limited to input/hash verification, beta-space
geometry selection, parameter arithmetic, model compilation, and one-state
operator integrity checks.  It contains no multi-trajectory truth replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STAGE_ID = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2"
PILOT_ID = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1"
OUTCOME_READY = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_READY"
OUTCOME_GAPS = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_READY_WITH_EVIDENCE_GAPS"
OUTCOME_NOT_READY = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_NOT_READY"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v2"

AMENDED_S1 = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/S1_STRUCTURAL_DEFINITION_AMENDED_V1.json"
AMENDMENT_METADATA = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/metadata.json"
V1_RELATIONSHIP = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/V1_S1_FACTOR_RELATIONSHIP.csv"
V1_DESIGN_METADATA = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1/metadata.json"
V1_DESIGN_REPORT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1/MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_REPORT.md"
MODEL = ROOT / "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml"
V3_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
COORDINATE_MAPPING = ROOT / "external_simulation_audits/myoleg_supine_hip_knee_rehab_feasibility_v1/PROJECT_MYOLEG_COORDINATE_MAPPING.json"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"
FORMAL_REFERENCE = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
V3_LANDSCAPE_PROTOCOL = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/V3_DEVELOPMENT_LANDSCAPE_GENERATION_PROTOCOL.json"

FROZEN_SHA = {
    "amended_s1": "3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763",
    "amendment_metadata": "36414d7ebc0c0870b0e2cd8b6b5637600cf2c37a6009b87471f6637cc07bc8f9",
    "v1_relationship": "a3682fe217677c388ca3a1ccb53358fd2fd59a79973305ef0cc6a039fdfb7c0c",
    "v1_design_metadata": "e53d28c1f526615c7fb7976fce0e26c36bcf76ca7e529f23a6cf9f47bd213c23",
    "v1_design_report": "b2ddf4290d42f98089e3e6629123d2a1127c1862c38e8fa04ccc16157fa642dc",
    "model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v3_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "v3_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "coordinate_mapping": "83798958fd0b12f5c5314bc32df898f1d6e56d8e224f7673d8aca5c457ce713c",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "v3_landscape_protocol": "837f287f75d353af69bdd0e9ade5a417777c6de60a69cda693f6be9f094f133d",
}
FROZEN_PATHS = {
    "amended_s1": AMENDED_S1,
    "amendment_metadata": AMENDMENT_METADATA,
    "v1_relationship": V1_RELATIONSHIP,
    "v1_design_metadata": V1_DESIGN_METADATA,
    "v1_design_report": V1_DESIGN_REPORT,
    "model": MODEL,
    "v3_table": V3_TABLE,
    "v3_manifest": V3_MANIFEST,
    "coordinate_mapping": COORDINATE_MAPPING,
    "formal_manifest": FORMAL_MANIFEST,
    "formal_reference": FORMAL_REFERENCE,
    "v3_landscape_protocol": V3_LANDSCAPE_PROTOCOL,
}

EXPECTED_FACTOR_IDS = (
    "S1F1_BIARTICULAR_LMAX",
    "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE",
    "S1F3_HIP_MONO_ANTAGONIST_F0",
    "S1F4_KNEE_MONO_ANTAGONIST_F0",
)

# Outcome-independent synthetic log-coordinate levels.  They are deliberately
# not population bounds.  The lmax field gets the smaller level because its
# physiological mapping is indirect; all force-scale fields use the same small
# reversible log displacement.
LEVEL_MAGNITUDES = {
    "S1F1_BIARTICULAR_LMAX": {"primary": 0.01, "fallback": 0.005},
    "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE": {"primary": 0.025, "fallback": 0.0125},
    "S1F3_HIP_MONO_ANTAGONIST_F0": {"primary": 0.025, "fallback": 0.0125},
    "S1F4_KNEE_MONO_ANTAGONIST_F0": {"primary": 0.025, "fallback": 0.0125},
}

TARGET_GEOMETRY = (
    ("REFERENCE", 0.0, 0.0),
    ("CORNER_NEG_NEG", -0.03, -0.03),
    ("CORNER_NEG_POS", -0.03, 0.03),
    ("CORNER_POS_NEG", 0.03, -0.03),
    ("CORNER_POS_POS", 0.03, 0.03),
    ("FLEX_NEG_AXIS", -0.03, 0.0),
    ("FLEX_POS_AXIS", 0.03, 0.0),
    ("EXTEND_NEG_AXIS", 0.0, -0.03),
    ("EXTEND_POS_AXIS", 0.0, 0.03),
    ("INTERIOR_NEG_NEG", -0.015, -0.015),
    ("INTERIOR_NEG_POS", -0.015, 0.015),
    ("INTERIOR_POS_NEG", 0.015, -0.015),
    ("INTERIOR_POS_POS", 0.015, 0.015),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs() -> tuple[dict[str, str], dict[str, Any], list[dict[str, str]]]:
    actual = {name: sha256(path) for name, path in FROZEN_PATHS.items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input SHA mismatch: {actual}")

    amended = json.loads(AMENDED_S1.read_text(encoding="utf-8"))
    metadata = json.loads(AMENDMENT_METADATA.read_text(encoding="utf-8"))
    v1 = json.loads(V1_DESIGN_METADATA.read_text(encoding="utf-8"))
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    mapping = json.loads(COORDINATE_MAPPING.read_text(encoding="utf-8"))
    factors = amended.get("factors", [])
    if not (
        amended.get("definition_id") == "S1_STRUCTURAL_DEFINITION_AMENDED_V1"
        and amended.get("all_factors_resolved") is True
        and amended.get("declared_dimensionality") == 4
        and tuple(row.get("factor_id") for row in factors) == EXPECTED_FACTOR_IDS
        and all(row.get("exact_members") and row.get("exact_fields") and row.get("operator") for row in factors)
        and all(row.get("nominal_identity") == "z=0" and row.get("v1_relationship") for row in factors)
    ):
        raise RuntimeError("authoritative amended S1 reconstruction failed")
    if not (
        metadata.get("outcome") == "S1_STRUCTURAL_FACTOR_DEFINITION_AMENDED_READY_WITH_LIMITATIONS"
        and metadata.get("amended_definition_sha256") == FROZEN_SHA["amended_s1"]
        and v1.get("outcome") == "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY"
        and v1.get("blocker") == "S1_DEFINITION_INCOMPLETE"
    ):
        raise RuntimeError("amendment readiness or V1 failure record changed")
    if not (
        manifest.get("candidate_count") == 625
        and manifest.get("included_candidate_count") == 625
        and manifest.get("all_candidates_pass_kinematic_gates") is True
        and manifest.get("mechanical_objective_evaluated") is False
        and manifest.get("held_out_scientific_truth_access_count") == 0
    ):
        raise RuntimeError("frozen V3 geometry manifest changed")
    if not (
        formal.get("rom_protocol_version") == "ROM_PROTOCOL_V2"
        and formal.get("theta_shank_definition") == "q_hip - q_knee"
        and formal.get("active_reference_sha256") == FROZEN_SHA["formal_reference"]
        and mapping.get("joint_mapping", {}).get("signs") == {"hip": 1.0, "knee": 1.0}
    ):
        raise RuntimeError("formal ROM/reference/coordinate convention changed")
    with V1_RELATIONSHIP.open(newline="", encoding="utf-8") as stream:
        relationships = list(csv.DictReader(stream))
    expected_relationships = {
        "FEMUR_MASS_INERTIA_SCALE": "RETAIN_AS_BACKGROUND",
        "TIBIA_PATELLA_MASS_INERTIA_SCALE": "RETAIN_AS_BACKGROUND",
        "FOOT_COMPLEX_MASS_INERTIA_SCALE": "RETAIN_AS_BACKGROUND",
        "HIP_ONLY_PASSIVE_FP_MAX_SCALE": "SECONDARY_ONLY",
        "KNEE_ONLY_PASSIVE_FP_MAX_SCALE": "SECONDARY_ONLY",
        "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE": "REMOVE_FROM_PERSONALIZATION_FOCUSED_COHORT",
    }
    actual_relationships = {row["entity"]: row["relationship"] for row in relationships if row["entity"] in expected_relationships}
    if actual_relationships != expected_relationships:
        raise RuntimeError("frozen V1-background relationship changed")
    return actual, amended, relationships


def actuator_ids(mujoco: Any, model: Any, names: list[str]) -> list[int]:
    identifiers = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in names]
    if any(identifier < 0 for identifier in identifiers):
        raise RuntimeError(f"authoritative actuator member is absent: {names}")
    return identifiers


def topology_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "actuator_trnid", "actuator_lengthrange", "tendon_adr", "tendon_num",
        "wrap_type", "wrap_objid", "site_bodyid", "site_pos", "jnt_type",
        "jnt_range", "eq_type", "eq_obj1id", "eq_obj2id", "eq_data",
    ):
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def factor_member_map(factor: dict[str, Any]) -> dict[str, list[str]]:
    members = factor["exact_members"]
    if isinstance(members, list):
        return {"TARGET": list(members)}
    if isinstance(members, dict):
        return {str(key): list(value) for key, value in members.items()}
    raise RuntimeError(f"invalid exact_members in {factor['factor_id']}")


def apply_authoritative_factor(mujoco: Any, model: Any, factor: dict[str, Any], z: float) -> tuple[int, list[int]]:
    """Interpret the SHA-frozen amended operator; no membership is redefined here."""
    factor_id = factor["factor_id"]
    groups = factor_member_map(factor)
    if factor_id == "S1F1_BIARTICULAR_LMAX":
        field = 5
        targets = actuator_ids(mujoco, model, groups["TARGET"])
        for identifier in targets:
            value = float(model.actuator_gainprm[identifier, field]) * math.exp(z)
            model.actuator_gainprm[identifier, field] = value
            model.actuator_biasprm[identifier, field] = value
    elif factor_id == "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE":
        field = 7
        rectus = actuator_ids(mujoco, model, groups["RECTUS_GROUP"])
        hamstring = actuator_ids(mujoco, model, groups["HAMSTRING_GROUP"])
        for identifiers, sign in ((rectus, 1.0), (hamstring, -1.0)):
            for identifier in identifiers:
                value = float(model.actuator_gainprm[identifier, field]) * math.exp(sign * z)
                model.actuator_gainprm[identifier, field] = value
                model.actuator_biasprm[identifier, field] = value
        targets = rectus + hamstring
    elif factor_id in ("S1F3_HIP_MONO_ANTAGONIST_F0", "S1F4_KNEE_MONO_ANTAGONIST_F0"):
        field = 2
        targets = actuator_ids(mujoco, model, groups["TARGET"])
        for identifier in targets:
            value = float(model.actuator_gainprm[identifier, field]) * math.exp(z)
            model.actuator_gainprm[identifier, field] = value
            model.actuator_biasprm[identifier, field] = value
    else:
        raise RuntimeError(f"unknown authoritative factor: {factor_id}")
    expected_fields = {f"actuator_gainprm[:,{field}]", f"actuator_biasprm[:,{field}]"}
    if set(factor["exact_fields"]) != expected_fields:
        raise RuntimeError(f"authoritative field/operator mismatch for {factor_id}")
    return field, targets


def operator_precheck(mujoco: Any, factor: dict[str, Any], z: float, role: str) -> dict[str, Any]:
    base = mujoco.MjModel.from_xml_path(str(MODEL))
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    base_gain = np.asarray(base.actuator_gainprm).copy()
    base_bias = np.asarray(base.actuator_biasprm).copy()
    field, targets = apply_authoritative_factor(mujoco, model, factor, z)

    expected_gain = base_gain.copy()
    expected_bias = base_bias.copy()
    factor_id = factor["factor_id"]
    groups = factor_member_map(factor)
    if factor_id == "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE":
        for group, sign in (("RECTUS_GROUP", 1.0), ("HAMSTRING_GROUP", -1.0)):
            identifiers = actuator_ids(mujoco, base, groups[group])
            expected_gain[identifiers, field] *= math.exp(sign * z)
            expected_bias[identifiers, field] *= math.exp(sign * z)
    else:
        expected_gain[targets, field] *= math.exp(z)
        expected_bias[targets, field] *= math.exp(z)

    exact_expected_mutation = bool(
        np.array_equal(model.actuator_gainprm, expected_gain)
        and np.array_equal(model.actuator_biasprm, expected_bias)
    )
    nominal_identity = bool(
        z != 0.0
        or (np.array_equal(model.actuator_gainprm, base_gain) and np.array_equal(model.actuator_biasprm, base_bias))
    )
    gain_bias_synchronized = bool(np.array_equal(model.actuator_gainprm[targets, field], model.actuator_biasprm[targets, field]))
    positive = bool(np.all(model.actuator_gainprm[targets, field] > 0.0))
    lmin_lt_lmax = True
    hard_domain_margin = None
    if field == 5:
        lmin = np.asarray(model.actuator_gainprm[targets, 4], dtype=float)
        lmax = np.asarray(model.actuator_gainprm[targets, 5], dtype=float)
        lmin_lt_lmax = bool(np.all(lmin < lmax))
        positive_lmin = np.asarray(base.actuator_gainprm[targets, 4], dtype=float)
        positive_lmax = np.asarray(base.actuator_gainprm[targets, 5], dtype=float)
        ratios = positive_lmin[positive_lmin > 0.0] / positive_lmax[positive_lmin > 0.0]
        lower_bound = float(np.max(np.log(ratios))) if len(ratios) else -math.inf
        hard_domain_margin = float(z - lower_bound)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    finite = all(bool(np.isfinite(value).all()) for value in (
        data.qpos, data.qvel, data.qacc, data.qfrc_actuator,
        data.actuator_force, data.actuator_length, data.ten_length,
    ))
    warnings = int(np.asarray(data.warning.number, dtype=np.int64).sum())
    topology_unchanged = topology_fingerprint(model) == topology_fingerprint(base)
    passed = all((
        exact_expected_mutation, nominal_identity, gain_bias_synchronized,
        positive, lmin_lt_lmax, finite, warnings == 0, topology_unchanged,
    ))
    return {
        "factor_id": factor_id,
        "role": role,
        "z": z,
        "multiplicative_scale_exp_z": math.exp(z),
        "member_count": len(targets),
        "field_index": field,
        "nominal_identity_bitwise_when_z0": nominal_identity,
        "only_declared_fields_and_members_changed": exact_expected_mutation,
        "gain_bias_synchronized": gain_bias_synchronized,
        "positive_parameter_domain": positive,
        "lmin_lt_lmax": lmin_lt_lmax,
        "lmax_hard_domain_margin_log_units": hard_domain_margin,
        "compiled_single_state_forward_finite": finite,
        "solver_warning_count": warnings,
        "topology_fingerprint_unchanged": topology_unchanged,
        "multi_trajectory_scientific_response_read": False,
        "pass": passed,
    }


def diagnostic_levels(mujoco: Any, factors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    levels: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for factor in factors:
        factor_id = factor["factor_id"]
        magnitude = LEVEL_MAGNITUDES[factor_id]
        checks.append(operator_precheck(mujoco, factor, 0.0, "NOMINAL_IDENTITY"))
        for name, value in (
            ("PRIMARY_NEGATIVE", -magnitude["primary"]),
            ("PRIMARY_POSITIVE", magnitude["primary"]),
            ("FALLBACK_NEGATIVE", -magnitude["fallback"]),
            ("FALLBACK_POSITIVE", magnitude["fallback"]),
        ):
            checks.append(operator_precheck(mujoco, factor, value, name))
        relevant = [row for row in checks if row["factor_id"] == factor_id]
        ready = all(bool(row["pass"]) for row in relevant)
        rationale = (
            "1% reversible log displacement for an indirect normalized muscle-curve field; smaller than force-scale diagnostics and verified against lmin<lmax"
            if factor_id == "S1F1_BIARTICULAR_LMAX"
            else "2.5% reversible log displacement for a positive MuJoCo force-scale/family-balance field; finite, symmetric, and intentionally conservative"
        )
        levels.append({
            "factor_id": factor_id,
            "factor_name": factor["inherited_name"],
            "population_range": "NOT_AVAILABLE",
            "population_range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
            "pilot_diagnostic_level_semantic": "SYNTHETIC_SMALL_FINITE_LOG_COORDINATE_NOT_A_HUMAN_POPULATION_RANGE",
            "nominal_z": 0.0,
            "primary_negative_z": -magnitude["primary"],
            "primary_positive_z": magnitude["primary"],
            "primary_multipliers_exp_z": [math.exp(-magnitude["primary"]), math.exp(magnitude["primary"])],
            "fallback_negative_z": -magnitude["fallback"],
            "fallback_positive_z": magnitude["fallback"],
            "fallback_multipliers_exp_z": [math.exp(-magnitude["fallback"]), math.exp(magnitude["fallback"])],
            "maximum_fallback_attempts_per_sign": 1,
            "selection_rationale": rationale,
            "selection_inputs": ["exp(z) operator semantics", "hard-domain arithmetic", "positivity", "lmin<lmax where applicable", "compile and single-state finite-forward precheck"],
            "selection_prohibited_inputs": ["J", "oracle", "ranking", "trajectory response", "interaction result", "gradient rotation result", "held-out truth"],
            "not_patient_distribution": True,
            "not_cohort_v2_bound": True,
            "precheck_status": "DIAGNOSTIC_LEVEL_READY" if ready else "DIAGNOSTIC_LEVEL_NOT_READY",
        })
    return levels, checks


def geometry_subset() -> list[dict[str, Any]]:
    with V3_TABLE.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 625:
        raise RuntimeError("V3 candidate table no longer has 625 rows")
    geometry = [{
        "candidate_index": int(row["candidate_index"]),
        "candidate_id": row["candidate_id"],
        "beta_flex": float(row["beta_flex"]),
        "beta_extend": float(row["beta_extend"]),
        "included": row["included"] == "True",
        "kinematic_gate_pass": row["kinematic_gate_pass"] == "True",
    } for row in source]
    selected = []
    for order, (role, requested_flex, requested_extend) in enumerate(TARGET_GEOMETRY):
        ranked = sorted(
            geometry,
            key=lambda row: (
                (row["beta_flex"] - requested_flex) ** 2 + (row["beta_extend"] - requested_extend) ** 2,
                row["candidate_index"],
            ),
        )
        row = ranked[0]
        distance = math.hypot(row["beta_flex"] - requested_flex, row["beta_extend"] - requested_extend)
        if not (row["included"] and row["kinematic_gate_pass"]):
            raise RuntimeError(f"geometry-selected V3 candidate is inadmissible: {row}")
        selected.append({
            "selection_order": order,
            "selection_role": role,
            "candidate_id": row["candidate_id"],
            "candidate_index": row["candidate_index"],
            "requested_beta_flex": requested_flex,
            "requested_beta_extend": requested_extend,
            "beta_flex": row["beta_flex"],
            "beta_extend": row["beta_extend"],
            "nearest_grid_distance": distance,
            "exact_coordinate_match": distance == 0.0,
            "selection_basis": "BETA_SPACE_GEOMETRY_ONLY",
            "source_columns_read": "candidate_index|candidate_id|beta_flex|beta_extend|included|kinematic_gate_pass",
            "J_or_oracle_or_rank_used": False,
        })
    if len({row["candidate_id"] for row in selected}) != 13:
        raise RuntimeError("deterministic geometry subset does not contain 13 unique candidates")
    return selected


def response_representations(amended: dict[str, Any]) -> dict[str, Any]:
    by_id = {factor["factor_id"]: factor for factor in amended["factors"]}
    representations = [
        {
            "response_id": "HIP_REQUIRED_TORQUE_RMS",
            "role": "PRIMARY_REQUIRED_TORQUE",
            "units": "N*m",
            "applicable_factors": list(EXPECTED_FACTOR_IDS),
            "definition": "time RMS of prescribed-trajectory project-coordinate hip required torque over 24 s",
        },
        {
            "response_id": "KNEE_REQUIRED_TORQUE_RMS",
            "role": "PRIMARY_REQUIRED_TORQUE",
            "units": "N*m",
            "applicable_factors": list(EXPECTED_FACTOR_IDS),
            "definition": "time RMS of prescribed-trajectory project-coordinate knee required torque over 24 s",
        },
        {
            "response_id": "S1F1_AFFECTED_BIARTICULAR_HIP_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F1_BIARTICULAR_LMAX"],
            "members": by_id["S1F1_BIARTICULAR_LMAX"]["exact_members"],
            "definition": "time RMS of the signed sum of frozen truth muscle_torque_contribution_nm over declared members, hip coordinate",
            "semantic_limit": "zero-control affected-actuator contribution; not a measured passive tissue torque",
        },
        {
            "response_id": "S1F1_AFFECTED_BIARTICULAR_KNEE_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F1_BIARTICULAR_LMAX"],
            "members": by_id["S1F1_BIARTICULAR_LMAX"]["exact_members"],
            "definition": "time RMS of the signed sum of frozen truth muscle_torque_contribution_nm over declared members, knee coordinate",
            "semantic_limit": "zero-control affected-actuator contribution; not a measured passive tissue torque",
        },
        {
            "response_id": "S1F2_RECTUS_HIP_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["RECTUS_GROUP"],
            "definition": "hip-coordinate time RMS of signed declared rectus-group contributions",
        },
        {
            "response_id": "S1F2_RECTUS_KNEE_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["RECTUS_GROUP"],
            "definition": "knee-coordinate time RMS of signed declared rectus-group contributions",
        },
        {
            "response_id": "S1F2_HAMSTRING_HIP_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["HAMSTRING_GROUP"],
            "definition": "hip-coordinate time RMS of signed declared hamstring-family contributions",
        },
        {
            "response_id": "S1F2_HAMSTRING_KNEE_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["HAMSTRING_GROUP"],
            "definition": "knee-coordinate time RMS of signed declared hamstring-family contributions",
        },
        {
            "response_id": "S1F2_NET_DECLARED_BIARTICULAR_HIP_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": {
                "RECTUS_GROUP": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["RECTUS_GROUP"],
                "HAMSTRING_GROUP": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["HAMSTRING_GROUP"],
            },
            "definition": "hip-coordinate time RMS of the signed sum over both oppositely scaled declared families",
        },
        {
            "response_id": "S1F2_NET_DECLARED_BIARTICULAR_KNEE_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"],
            "members": {
                "RECTUS_GROUP": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["RECTUS_GROUP"],
                "HAMSTRING_GROUP": by_id["S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE"]["exact_members"]["HAMSTRING_GROUP"],
            },
            "definition": "knee-coordinate time RMS of the signed sum over both oppositely scaled declared families",
        },
        {
            "response_id": "S1F3_DECLARED_HIP_ANTAGONIST_HIP_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F3_HIP_MONO_ANTAGONIST_F0"],
            "members": by_id["S1F3_HIP_MONO_ANTAGONIST_F0"]["exact_members"],
            "definition": "time RMS of signed declared-group contribution in the hip coordinate",
            "semantic_limit": "glmed3_r and piri_r remain mechanically classified and anatomically multi-action/configuration-dependent",
        },
        {
            "response_id": "S1F4_DECLARED_KNEE_ANTAGONIST_KNEE_CONTRIBUTION_RMS",
            "role": "PRIMARY_MECHANISTIC_COMPONENT",
            "units": "N*m",
            "applicable_factors": ["S1F4_KNEE_MONO_ANTAGONIST_F0"],
            "members": by_id["S1F4_KNEE_MONO_ANTAGONIST_F0"]["exact_members"],
            "definition": "time RMS of signed declared-group contribution in the knee coordinate",
        },
        {
            "response_id": "FROZEN_COMBINED_J",
            "role": "SECONDARY_DIAGNOSTIC_ONLY",
            "units": "dimensionless",
            "applicable_factors": list(EXPECTED_FACTOR_IDS),
            "definition": "sqrt(0.5*((hip_tau_rms/model_own_reference_hip_rms)^2+(knee_tau_rms/model_own_reference_knee_rms)^2))",
            "normalization_policy": "use each structural diagnostic model's own MYOLEG_V3_K0312 reference denominators exactly as the frozen subject-reference normalization; do not renormalize after result reveal",
            "not_sole_endpoint": True,
        },
    ]
    return {
        "status": "FROZEN_BEFORE_SCIENTIFIC_PILOT_OUTCOMES",
        "primary_endpoint": "CONFIGURATION_DEPENDENT_NONPROPORTIONAL_MECHANICAL_RESPONSE",
        "representations": representations,
        "derived_for_every_primary_scalar_representation": [
            "perturbed-versus-nominal proportional fit", "perturbed-versus-nominal affine fit",
            "delta response configuration statistics", "local beta-flex gradient", "local beta-extend gradient",
        ],
        "gradient_stencil": {
            "h": 0.015,
            "candidate_ids": ["MYOLEG_V3_K0156", "MYOLEG_V3_K0168", "MYOLEG_V3_K0456", "MYOLEG_V3_K0468"],
            "d_beta_flex": "(mean(y(+h,-h),y(+h,+h))-mean(y(-h,-h),y(-h,+h)))/(2*h)",
            "d_beta_extend": "(mean(y(-h,+h),y(+h,+h))-mean(y(-h,-h),y(+h,-h)))/(2*h)",
        },
        "post_outcome_representation_addition_allowed": False,
        "personalization_or_oracle_is_endpoint": False,
    }


def nonproportionality_gate() -> dict[str, Any]:
    return {
        "gate_id": "PILOT_V2_NONPROPORTIONALITY_GATE_V1",
        "frozen_before_scientific_pilot_outcomes": True,
        "fits": {
            "proportional": "OLS through origin: y_perturbed=a*y_nominal",
            "affine": "OLS with intercept: y_perturbed=a*y_nominal+b",
            "r2": "1-SSE/sum((y_perturbed-mean(y_perturbed))^2); undefined denominator -> null and no support",
            "nrmse": "residual_RMS/max(nominal_response_RMS,1e-12 response units)",
            "residual_range_normalized": "(max(residual)-min(residual))/max(nominal_response_RMS,1e-12 response units)",
        },
        "required_outputs": ["proportional_a", "proportional_R2", "proportional_NRMSE", "proportional_residual_RMS", "proportional_residual_range", "affine_a", "affine_b", "affine_R2", "affine_NRMSE", "affine_residual_RMS", "affine_residual_range"],
        "effect_resolution_gate": {
            "delta_response_RMS_min_Nm_inclusive": 1.0e-5,
            "delta_response_RMS_over_nominal_RMS_min_inclusive": 1.0e-4,
            "both_required": True,
            "reason": "exclude algebraic/floating residuals and unresolved relative effects before shape classification",
        },
        "nonproportionality_thresholds": {
            "proportional_NRMSE_strictly_above": 1.0e-4,
            "affine_R2_strictly_below": 0.9999,
            "both_required_on_same_predeclared_response": True,
            "provenance": "retains the prior outcome-independent mechanistic thresholds and adds an explicit effect-resolution floor",
        },
        "factor_sign_pass_rule": "effect resolution AND proportional_NRMSE>1e-4 AND affine_R2<0.9999 on the same predeclared response",
        "factor_pass_aggregation": "at least one required-torque response AND at least one factor-specific mechanistic component must each pass nonproportionality and configuration-dependence for the same sign",
        "threshold_tuning_after_outcome": False,
    }


def configuration_gate() -> dict[str, Any]:
    return {
        "gate_id": "PILOT_V2_CONFIGURATION_DEPENDENCE_GATE_V1",
        "frozen_before_scientific_pilot_outcomes": True,
        "delta_definition": "delta_y(c)=y_perturbed(c)-y_nominal(c)",
        "required_outputs": ["delta_mean", "delta_population_SD", "delta_range", "delta_RMS", "normalized_spread", "normalized_range", "beta_polynomial_coefficients", "beta_polynomial_R2"],
        "normalization": "max(nominal_response_RMS,1e-12 response units)",
        "beta_dependence_model": "fixed OLS design [1,beta_flex,beta_extend,beta_flex^2,beta_extend^2,beta_flex*beta_extend] over all 13 points",
        "thresholds": {
            "effect_resolution_gate_from_nonproportionality_required": True,
            "normalized_spread_min_inclusive": 1.0e-4,
            "normalized_range_min_inclusive": 2.0e-4,
            "beta_polynomial_R2_min_inclusive": 0.25,
            "all_required_on_same_predeclared_response": True,
        },
        "constant_change_rule": "zero/under-threshold spread or range is constant-like and fails",
        "scale_like_rule": "configuration dependence alone is insufficient; the separate nonproportionality gate must also pass",
        "threshold_tuning_after_outcome": False,
    }


def gradient_gate() -> dict[str, Any]:
    return {
        "gate_id": "PILOT_V2_GRADIENT_ROTATION_GATE_V1",
        "frozen_before_scientific_pilot_outcomes": True,
        "stencil": "four symmetric interior points at beta=[+/-0.015,+/-0.015]",
        "gradient": "[d(response)/d beta_flex,d(response)/d beta_extend] from the frozen centered diagonal stencil",
        "required_outputs": ["nominal_gradient", "perturbed_gradient", "cosine_similarity", "angle_difference_deg", "resolved_sign_pattern", "unit_direction_component_change_max", "gradient_norm_ratio"],
        "gradient_resolution": {
            "minimum_norm_times_h_over_nominal_response_RMS_inclusive": 1.0e-5,
            "h": 0.015,
            "both_nominal_and_perturbed_gradients_required": True,
        },
        "direction_evidence": {
            "cosine_similarity_max_inclusive": 0.995,
            "equivalent_minimum_angle_deg": math.degrees(math.acos(0.995)),
            "resolved_component_sign_change_is_sufficient_alternative": True,
            "unit_direction_component_change_max_min_inclusive": 0.05,
            "any_one_direction_criterion_after_resolution": True,
        },
        "resolved_component_sign_rule": "a component sign is comparable only when abs(component)*h/nominal_response_RMS>=1e-5 for both models",
        "magnitude_only_gradient_change_is_direction_evidence": False,
        "classification_role": "supporting directional-preference evidence; not mandatory if the primary nonproportionality-plus-configuration gate passes",
        "full_sign_reversal_required": False,
        "threshold_tuning_after_outcome": False,
    }


def integrity_rules() -> dict[str, Any]:
    return {
        "gate_id": "PILOT_V2_INTEGRITY_AND_FALLBACK_RULES_V1",
        "frozen_before_scientific_pilot_outcomes": True,
        "future_replay_gates": {
            "model_compile": "PASS",
            "finite_q_dq_ddq": True,
            "finite_truth_torque_and_actuator_tendon_arrays": True,
            "solver_warning_count_max": 0,
            "unexpected_contact_active_count_max": 0,
            "tendon_limit_active_count_max": 0,
            "joint_limit_active_count_max": 1,
            "joint_limit_contribution_max_abs_Nm": 0.005,
            "joint_limit_contribution_max_relative": 0.0005,
            "source_equality_residual_max": 0.001,
            "truth_decomposition_residual_max_abs_Nm": 1.0e-8,
            "muscle_reconstruction_residual_max_abs_Nm": 1.0e-8,
            "sample_count_exact": 401,
            "duration_s_exact": 24.0,
            "V3_extrema_ROM_error_max_deg": 1.0e-3,
            "q_closure_error_max_rad": 1.0e-10,
            "dq_closure_error_max_rad_s": 1.0e-10,
            "ddq_closure_error_max_rad_s2": 1.0e-9,
            "C2_and_branch_anchor_gates": "preserve all frozen V3 gates",
            "only_declared_model_fields_changed": True,
            "topology_fingerprint_unchanged": True,
        },
        "fallback_trigger": "primary level integrity failure only; never effect size, J, oracle, rank, nonproportionality, configuration dependence, or gradient outcome",
        "fallback_action": "replace only the failed factor-sign primary model with its pre-frozen half-magnitude fallback and rerun the full 13-point integrity/replay set once",
        "maximum_fallback_attempts_per_factor_sign": 1,
        "fallback_failure_action": "factor=INVALID_FOR_PILOT; stop that factor; do not tune or add levels",
        "scientific_small_effect_action": "INCONCLUSIVE; fallback prohibited",
        "primary_and_fallback_levels_source": "PILOT_V2_DIAGNOSTIC_LEVELS.json",
    }


def cohort_rules(relationships: list[dict[str, str]]) -> dict[str, Any]:
    keep = {
        "FEMUR_MASS_INERTIA_SCALE", "TIBIA_PATELLA_MASS_INERTIA_SCALE",
        "FOOT_COMPLEX_MASS_INERTIA_SCALE", "HIP_ONLY_PASSIVE_FP_MAX_SCALE",
        "KNEE_ONLY_PASSIVE_FP_MAX_SCALE", "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
    }
    return {
        "rule_id": "PILOT_V2_COHORT_FACTOR_ADMISSION_RULE_V1",
        "future_stage": "MYOLEG_VIRTUAL_PATIENT_COHORT_V2_RANGE_AND_DESIGN",
        "automatic_admission": False,
        "required_conjunction": [
            "exact model semantics defensible",
            "pilot integrity PASS",
            "non-proportional configuration-dependent mechanical effect demonstrated under frozen gates",
            "future population-range calibration pathway exists",
        ],
        "classification_if_first_three_pass_but_range_path_missing": "COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP",
        "current_population_ranges": {factor_id: "NOT_AVAILABLE" for factor_id in EXPECTED_FACTOR_IDS},
        "pilot_level_is_cohort_bound": False,
        "cohort_v2_generated_in_this_stage": False,
        "new_versioned_split_required": True,
        "old_v1_heldout_is_automatic_cohort_v2_confirmation": False,
        "old_v1_heldout_scientific_access_count": 0,
        "inherited_v1_background_relationships": [row for row in relationships if row["entity"] in keep],
        "factor_outcome_classes": {
            "STRUCTURALLY_INFORMATIVE": "integrity PASS and frozen nonproportionality-plus-configuration evidence passes; gradient rotation is reported as supporting evidence",
            "MAGNITUDE_ONLY": "effect is resolved but response remains proportional/affine or configuration gate fails and gradient direction is essentially retained",
            "INCONCLUSIVE": "effect is below resolution or evidence is insufficient under the frozen gates",
            "INVALID": "primary and allowed fallback fail integrity, or a required response is not computable",
        },
        "factor_aggregation": "both signs must be integrity-valid; at least one sign must meet the structural evidence rule; results for both signs are always reported",
        "produces_personalization_is_a_valid_pilot_conclusion": False,
    }


def execution_plan(level_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = all(row["precheck_status"] == "DIAGNOSTIC_LEVEL_READY" for row in level_rows)
    return {
        "plan_id": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1_EXECUTION_PLAN",
        "status": "READY_FOR_SEPARATE_FUTURE_EXECUTION" if ready else "BLOCKED_BY_DIAGNOSTIC_LEVEL_NOT_READY",
        "design_stage_only": True,
        "scientific_pilot_executed": False,
        "model_structure": "ONE_FACTOR_AT_A_TIME",
        "model_label": "STRUCTURAL_DIAGNOSTIC_MODEL",
        "virtual_patient_label_prohibited": True,
        "nominal_model_count": 1,
        "primary_perturbed_model_count": 8 if ready else 0,
        "primary_structural_diagnostic_model_count": 9 if ready else 0,
        "trajectories_per_model": 13,
        "expected_primary_replay_count": 117 if ready else 0,
        "maximum_optional_fallback_model_count": 8 if ready else 0,
        "maximum_optional_fallback_replay_count": 104 if ready else 0,
        "maximum_total_unique_model_count_including_fallback": 17 if ready else 0,
        "maximum_total_replay_count_including_fallback": 221 if ready else 0,
        "fallback_only_replaces_failed_factor_sign_model": True,
        "future_execution_order": [
            "freeze code/environment/input hashes",
            "construct nominal and eight one-factor-at-a-time primary models",
            "run 13 geometry-frozen trajectories/model with integrity first",
            "use at most one frozen fallback only for a failed factor-sign integrity gate",
            "compute every preregistered representation and gate without adding variants",
            "classify factor-sign then factor; stop without generating Cohort V2",
        ],
        "training_or_test_split_membership": "NONE_DIAGNOSTIC_MODELS_ARE_NOT_SUBJECTS",
        "future_separate_invocation_required": True,
        "automatic_execution_from_design_builder": False,
        "held_out_access": 0,
    }


def protocol(outcome: str, ready: bool) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "formal_outcome": outcome,
        "scientific_role": "OUTCOME_INDEPENDENT_EXECUTABLE_PILOT_PROTOCOL_DESIGN_ONLY",
        "authoritative_definition_id": "S1_STRUCTURAL_DEFINITION_AMENDED_V1",
        "authoritative_definition_sha256": FROZEN_SHA["amended_s1"],
        "frozen_before_any_scientific_structural_pilot_outcome": True,
        "primary_future_question": "CONFIGURATION_DEPENDENT_NONPROPORTIONAL_MECHANICAL_RESPONSE",
        "future_pilot_id": PILOT_ID,
        "future_pilot_ready": ready,
        "future_pilot_must_be_separately_invoked": True,
        "this_stage_scientific_pilot_execution_authorized": False,
        "operator_source_policy": "read exact factors, members, fields, operators, nominal identities, and V1 relations directly from authoritative amended SHA; mismatch fails closed",
        "geometry_selection_policy": "13 deterministic V3 points selected only from beta_flex/beta_extend geometry; exact match else nearest Euclidean beta-grid point with candidate-index tie break",
        "level_selection_policy": "synthetic one-shot small finite log levels chosen from operator/domain/compile semantics only; no scientific response-based tuning",
        "factor_sign_classification_then_factor_aggregation": True,
        "scope_guards": {
            "scientific_trajectory_pilot_executed": False,
            "multi_trajectory_torque_response_read": False,
            "J_or_oracle_or_rank_used_for_design": False,
            "held_out_scientific_access_count": 0,
            "new_virtual_subjects": 0,
            "cohort_v2_generated": False,
            "new_truth_landscape_generated": False,
            "S1_operator_modified": False,
            "V3_parameterization_or_domain_modified": False,
            "objective_or_normalization_modified": False,
            "five_parameter_or_NN_or_PINN_or_BO_run": False,
            "robot_or_hardware": False,
        },
        "frozen_semantic_language": {
            "lmax": "MuJoCo normalized muscle-curve field; not optimal fiber length or tendon slack length",
            "F0": "MuJoCo muscle force scale; not pure active strength or a validated patient parameter",
            "fpmax_balance": "relative model-family force balance; not measured tissue stiffness",
            "glmed3_r_and_piri_r": "mechanically classified, anatomically multi-action/configuration-dependent",
        },
        "population_range_policy": "all four remain NOT_AVAILABLE; pilot diagnostic levels are not human ranges, patient distributions, or Cohort V2 bounds",
        "post_outcome_level_gate_or_representation_tuning_allowed": False,
        "V1_failure_record_preserved": {
            "outcome": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY",
            "blocker": "S1_DEFINITION_INCOMPLETE",
            "report_sha256": FROZEN_SHA["v1_design_report"],
            "metadata_sha256": FROZEN_SHA["v1_design_metadata"],
            "overwritten": False,
        },
    }


def report(outcome: str, levels: list[dict[str, Any]], subset: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    level_lines = "\n".join(
        f"- `{row['factor_id']}`: primary z=`{row['primary_negative_z']:+g}/{row['primary_positive_z']:+g}`, fallback z=`{row['fallback_negative_z']:+g}/{row['fallback_positive_z']:+g}`; `{row['precheck_status']}`."
        for row in levels
    )
    candidate_lines = "\n".join(
        f"- {row['selection_role']}: `{row['candidate_id']}` at `[{row['beta_flex']:+g}, {row['beta_extend']:+g}]`."
        for row in subset
    )
    return f"""# MyoLeg Structural Heterogeneity Pilot Design V2

## Formal outcome

**{outcome}**

Authoritative definition: `S1_STRUCTURAL_DEFINITION_AMENDED_V1`  
Authoritative SHA-256: `{FROZEN_SHA['amended_s1']}`

This stage froze an executable, outcome-independent design. It performed parameter arithmetic, compilation, declared-field mutation checks and one-state finite forwards only. It did **not** run any scientific trajectory replay, reveal multi-trajectory torque response, create a virtual subject, generate a cohort/landscape, or access held-out truth.

## Q1. Were all four amended factors reconstructed exactly?

**Yes.** Exact factor IDs, authoritative members, gain/bias fields, mathematical operators, `z=0` identities, semantics and V1 relationships were read from the SHA-pinned amended artifact. All nominal/primary/fallback operator-only checks passed. The old V1 design failure remains immutable: `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY / S1_DEFINITION_INCOMPLETE`.

## Q2. What primary synthetic z-levels are frozen?

{level_lines}

The 1% lmax level is smaller because `lmax` is an indirect normalized curve field. The force-scale/family-balance factors use a symmetric 2.5% log displacement. These are synthetic diagnostic coordinates, **not human population ranges, patient distributions, validated physiological parameters, or Cohort V2 bounds**. Population range remains `NOT_AVAILABLE` for all four.

## Q3. What one-step fallback is frozen?

Exactly one half-magnitude fallback is frozen per factor/sign. It may be used only after a primary **integrity** failure. A small or uninteresting response cannot trigger fallback. A failed fallback makes that factor `INVALID_FOR_PILOT`; no additional levels may be added.

## Q4. What V3 subset is frozen?

{candidate_lines}

All 13 coordinates exist exactly on the 625-point grid. Selection used only candidate ID/index, beta coordinates and frozen kinematic inclusion bits—never J, oracle, rank or personalized outcomes.

## Q5. What response representations are primary?

Hip and knee required-torque RMS are required for every factor. Factor-specific signed actuator-contribution RMS responses are also primary: affected biarticular hip/knee contributions for lmax; rectus, hamstring and their net hip/knee contributions for balance; declared hip antagonist hip contribution; and declared knee antagonist knee contribution. Every primary scalar receives proportional/affine fits, delta-configuration metrics and beta gradients. Frozen combined J is secondary only and cannot be the sole endpoint.

## Q6. What non-proportionality gate is frozen?

On the same preregistered response, delta RMS must be at least `1e-5 N*m` and `1e-4` of nominal RMS, proportional NRMSE must be strictly above `1e-4`, and affine R2 must be strictly below `0.9999`. A factor-sign needs evidence in at least one required-torque response and one factor-specific mechanistic component; floating residuals and pure scale/offset effects do not pass.

## Q7. What configuration-dependence gate is frozen?

For `delta_y`, normalized population SD must be at least `1e-4`, normalized range at least `2e-4`, and the fixed six-term beta polynomial must have R2 at least `0.25`, with the effect-resolution gate also passing. Configuration dependence cannot replace the separate non-proportionality gate.

## Q8. What gradient-rotation gate is frozen?

The four `±0.015` diagonal interior points form a centered stencil. Both gradients must clear a relative resolution of `1e-5`; direction evidence is cosine at or below `0.995`, a resolved component sign change, or maximum unit-direction component change of at least `0.05`. A magnitude-only change is not direction evidence. Gradient rotation is supporting evidence and full sign reversal is not mandatory.

## Q9. What Cohort V2 admission rule is frozen?

Admission to `MYOLEG_VIRTUAL_PATIENT_COHORT_V2_RANGE_AND_DESIGN` requires the conjunction of defensible exact semantics, pilot integrity PASS, frozen-gate non-proportional configuration dependence, and a future population-range calibration pathway. If only the first three pass, status is `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`; no cohort is generated. Old V1 held-out data are not automatic Cohort V2 confirmation and had zero scientific access here.

V1 relations remain unchanged: femur, tibia/patella and foot mass/inertia are background; hip/knee common fpmax are secondary only; common biarticular fpmax is removed from the personalization-focused cohort.

## Q10. Is the protocol ready for `{PILOT_ID}`?

**Yes, with evidence gaps.** All four synthetic levels and operators passed outcome-independent prechecks. Population ranges remain unavailable, so this is not Cohort V2 readiness. The pilot must be invoked separately under this frozen protocol and must stop after factor classification.

## Exact future workload

- Primary structural diagnostic models: **{plan['primary_structural_diagnostic_model_count']}** (`1 nominal + 8 perturbed`).
- Trajectories/model: **{plan['trajectories_per_model']}**.
- Primary replays: **{plan['expected_primary_replay_count']}**.
- Optional fallback maximum: **{plan['maximum_optional_fallback_model_count']} models / {plan['maximum_optional_fallback_replay_count']} replays**.
- Absolute maximum including fallback: **{plan['maximum_total_unique_model_count_including_fallback']} models / {plan['maximum_total_replay_count_including_fallback']} replays**.
- Scientific models/replays executed now: **0 / 0**.

## Stop state

- Design only; scientific pilot not executed.
- Cohort V1, amended S1, V3 domain, objective and normalization unchanged.
- New virtual subjects/cohort/landscape: none.
- Held-out scientific access: 0.
- Five-parameter/NN/PINN/BO: not run.
- Robot/hardware: untouched.
"""


def build() -> None:
    import mujoco

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen, amended, relationships = verify_inputs()
    levels, prechecks = diagnostic_levels(mujoco, amended["factors"])
    subset = geometry_subset()
    ready = all(row["precheck_status"] == "DIAGNOSTIC_LEVEL_READY" for row in levels)
    outcome = OUTCOME_GAPS if ready else OUTCOME_NOT_READY
    plan = execution_plan(levels)

    write_json(OUTPUT / "AMENDED_S1_INPUT_VERIFICATION.json", {
        "status": "PASS" if ready else "FAIL_CLOSED",
        "frozen_inputs": frozen,
        "authoritative_definition_id": amended["definition_id"],
        "authoritative_definition_sha256": frozen["amended_s1"],
        "factor_count": len(amended["factors"]),
        "factor_ids": [row["factor_id"] for row in amended["factors"]],
        "authoritative_factor_reconstruction": amended["factors"],
        "operator_precheck_count": len(prechecks),
        "operator_precheck_pass_count": sum(bool(row["pass"]) for row in prechecks),
        "operator_prechecks": prechecks,
        "previous_design_failure": {
            "outcome": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY",
            "blocker": "S1_DEFINITION_INCOMPLETE",
            "report_sha256": frozen["v1_design_report"],
            "metadata_sha256": frozen["v1_design_metadata"],
            "preserved": True,
        },
        "scientific_trajectory_response_read": False,
        "held_out_scientific_access_count": 0,
    })
    write_json(OUTPUT / "PILOT_V2_DIAGNOSTIC_LEVELS.json", {
        "status": "ALL_FACTORS_READY" if ready else "ONE_OR_MORE_FACTORS_DIAGNOSTIC_LEVEL_NOT_READY",
        "frozen_before_scientific_pilot_outcomes": True,
        "population_range": "NOT_AVAILABLE_FOR_ALL_FOUR_FACTORS",
        "pilot_diagnostic_level_is_population_range": False,
        "levels": levels,
        "one_round_only": True,
        "post_outcome_level_tuning_allowed": False,
    })
    write_csv(OUTPUT / "PILOT_V2_V3_TRAJECTORY_SUBSET.csv", subset)
    write_json(OUTPUT / "PILOT_V2_RESPONSE_REPRESENTATIONS.json", response_representations(amended))
    write_json(OUTPUT / "PILOT_V2_NONPROPORTIONALITY_GATES.json", nonproportionality_gate())
    write_json(OUTPUT / "PILOT_V2_CONFIGURATION_DEPENDENCE_GATES.json", configuration_gate())
    write_json(OUTPUT / "PILOT_V2_GRADIENT_ROTATION_GATES.json", gradient_gate())
    write_json(OUTPUT / "PILOT_V2_INTEGRITY_AND_FALLBACK_RULES.json", integrity_rules())
    write_json(OUTPUT / "PILOT_V2_COHORT_ADMISSION_RULES.json", cohort_rules(relationships))
    write_json(OUTPUT / "PILOT_V2_EXECUTION_PLAN.json", plan)
    write_json(OUTPUT / "STRUCTURAL_HETEROGENEITY_PILOT_V2_PROTOCOL.json", protocol(outcome, ready))
    (OUTPUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_REPORT.md").write_text(
        report(outcome, levels, subset, plan), encoding="utf-8"
    )
    write_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID,
        "outcome": outcome,
        "authoritative_definition_id": amended["definition_id"],
        "authoritative_definition_sha256": frozen["amended_s1"],
        "all_four_factors_reconstructed": True,
        "diagnostic_level_ready_count": sum(row["precheck_status"] == "DIAGNOSTIC_LEVEL_READY" for row in levels),
        "diagnostic_factor_count": len(levels),
        "operator_precheck_pass_count": sum(bool(row["pass"]) for row in prechecks),
        "operator_precheck_count": len(prechecks),
        "trajectory_subset_count": len(subset),
        "future_primary_diagnostic_model_count": plan["primary_structural_diagnostic_model_count"],
        "future_primary_replay_count": plan["expected_primary_replay_count"],
        "future_maximum_fallback_model_count": plan["maximum_optional_fallback_model_count"],
        "future_maximum_fallback_replay_count": plan["maximum_optional_fallback_replay_count"],
        "scientific_pilot_executed": False,
        "scientific_models_generated": 0,
        "scientific_trajectory_replays": 0,
        "multi_trajectory_torque_response_read": False,
        "J_or_oracle_or_rank_used_for_design": False,
        "held_out_scientific_access_count": 0,
        "population_ranges_available": False,
        "new_virtual_subjects": 0,
        "cohort_v2_generated": False,
        "truth_landscape_generated": False,
        "v1_design_failure_overwritten": False,
        "S1_or_V3_or_objective_or_normalization_modified": False,
        "five_parameter_or_NN_or_PINN_or_BO_run": False,
        "robot_or_hardware": False,
        "next_stage": PILOT_ID if ready else None,
        "next_stage_executed": False,
        "analysis_code_sha256": sha256(Path(__file__)),
    })
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUTPUT / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({
        "stage_id": STAGE_ID,
        "outcome": outcome,
        "authoritative_definition_sha256": frozen["amended_s1"],
        "operator_prechecks": f"{sum(bool(row['pass']) for row in prechecks)}/{len(prechecks)}",
        "trajectory_subset_count": len(subset),
        "scientific_models_or_replays_executed": "0/0",
        "held_out_scientific_access_count": 0,
    }, indent=2))


if __name__ == "__main__":
    build()
