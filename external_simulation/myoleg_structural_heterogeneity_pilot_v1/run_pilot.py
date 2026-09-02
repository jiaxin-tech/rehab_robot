"""Execute the frozen amended-S1 structural diagnostic pilot.

The two-step command is intentional:

1. ``--prepare`` freezes input verification and the execution protocol.
2. ``--execute`` verifies that protocol SHA and performs only its 13-point,
   one-factor-at-a-time prescribed-state diagnostic replays.

No cohort subject, oracle, candidate ranking, learner, optimizer, or robot path
is available from this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from external_simulation.myoleg_structural_heterogeneity_pilot_design_v2 import build_design as design_v2


STAGE_ID = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1"
OUTPUT = Path(__file__).resolve().parents[2] / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_v1"
FIGURES = OUTPUT / "figures"
ROOT = Path(__file__).resolve().parents[2]

DESIGN_DIR = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v2"
AMENDED_S1 = ROOT / "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/S1_STRUCTURAL_DEFINITION_AMENDED_V1.json"
MODEL = ROOT / "external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml"
V3_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V3_PARAMETERIZATION = ROOT / "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/parameterization.py"
V3_DESIGN_BUILDER = ROOT / "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/build_design.py"
CANDIDATE_BUILDER = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
TRUTH_SEMANTICS = ROOT / "external_simulation_audits/myoleg_reference_trajectory_replay_v1/MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
V2_DOMAIN_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V2_REFERENCE = ROOT / "external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"
FORMAL_REFERENCE = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
COHORT_V1_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
DESIGN_BUILDER = ROOT / "external_simulation/myoleg_structural_heterogeneity_pilot_design_v2/build_design.py"

DESIGN_FILES = {
    "design_input_verification": DESIGN_DIR / "AMENDED_S1_INPUT_VERIFICATION.json",
    "design_report": DESIGN_DIR / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_REPORT.md",
    "design_cohort_rules": DESIGN_DIR / "PILOT_V2_COHORT_ADMISSION_RULES.json",
    "design_configuration_gates": DESIGN_DIR / "PILOT_V2_CONFIGURATION_DEPENDENCE_GATES.json",
    "design_levels": DESIGN_DIR / "PILOT_V2_DIAGNOSTIC_LEVELS.json",
    "design_execution_plan": DESIGN_DIR / "PILOT_V2_EXECUTION_PLAN.json",
    "design_gradient_gates": DESIGN_DIR / "PILOT_V2_GRADIENT_ROTATION_GATES.json",
    "design_integrity_rules": DESIGN_DIR / "PILOT_V2_INTEGRITY_AND_FALLBACK_RULES.json",
    "design_nonproportionality_gates": DESIGN_DIR / "PILOT_V2_NONPROPORTIONALITY_GATES.json",
    "design_response_representations": DESIGN_DIR / "PILOT_V2_RESPONSE_REPRESENTATIONS.json",
    "design_subset": DESIGN_DIR / "PILOT_V2_V3_TRAJECTORY_SUBSET.csv",
    "design_protocol": DESIGN_DIR / "STRUCTURAL_HETEROGENEITY_PILOT_V2_PROTOCOL.json",
    "design_checksums": DESIGN_DIR / "checksums.sha256",
    "design_metadata": DESIGN_DIR / "metadata.json",
}

FROZEN_PATHS = {
    **DESIGN_FILES,
    "amended_s1": AMENDED_S1,
    "model": MODEL,
    "v3_table": V3_TABLE,
    "v3_manifest": V3_MANIFEST,
    "v3_parameterization": V3_PARAMETERIZATION,
    "v3_design_builder": V3_DESIGN_BUILDER,
    "candidate_builder": CANDIDATE_BUILDER,
    "replay_builder": REPLAY_BUILDER,
    "truth_semantics": TRUTH_SEMANTICS,
    "v2_domain_manifest": V2_DOMAIN_MANIFEST,
    "v2_reference": V2_REFERENCE,
    "formal_manifest": FORMAL_MANIFEST,
    "formal_reference": FORMAL_REFERENCE,
    "cohort_v1_manifest": COHORT_V1_MANIFEST,
    "design_builder": DESIGN_BUILDER,
}

FROZEN_SHA = {
    "design_input_verification": "8a6af9380140e8b206b2fc0104f6100a8e5038a91f936f0db64b784ad9eb1bb4",
    "design_report": "bcb5ba8c83f0c3fb1091b82770ad6d7aa5d589a9b01a85b21a020893287b3fb7",
    "design_cohort_rules": "f866fa7668492fecf3890a40fd47e4d3f748edb6835ee4df8e41c1efa2518d07",
    "design_configuration_gates": "7d7af49f951f7ae970d64a65f68094c921b6be16ddf2917f2aa30bcff328bd1c",
    "design_levels": "285d7cb108f0579774a4ce3b825de23cd6aba5987f0079315682b20ed176728d",
    "design_execution_plan": "7afd5c99244bc64ffb1c4466b5c8e8834d4c8c172800c6a28f8246621941b494",
    "design_gradient_gates": "f6508a405f59847f5105273df0a48e3e19470ca123819825a6eba8faf1394410",
    "design_integrity_rules": "d03acf452f34d5965b5d6bf8c2bb0c877ee63b6873c3a6155ff23721cf92a3e3",
    "design_nonproportionality_gates": "0ce4968e1bd9d215fa008678f3c3a6e463c5575e77ea7582f7b6f9f744a09ebd",
    "design_response_representations": "254ea17c2f05d3c2e771660f2eb49a363596a9f876a2313367735dfd159d62d3",
    "design_subset": "d005ddc4590c4c58db81a4ee625609b0a29db02c681b67238b9790e044c658e4",
    "design_protocol": "b3d1f7acc427e5a592a8045ba3e0bab921df1834596d6291c5e4cb3368b69d00",
    "design_checksums": "ac2fad3b2f5b5180ac9d2a138db91235aef3a373a1c591bb391bc28e09795d44",
    "design_metadata": "cbfc657917f67ed9885799312db717285ba95f1f5c227ee769d19e1a16376b4f",
    "amended_s1": "3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763",
    "model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v3_table": "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774",
    "v3_manifest": "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745",
    "v3_parameterization": "e830b5cadd6d970107e59eb9b346650af5ab254b42beecdfaf6b70a5985957ef",
    "v3_design_builder": "5902c24970418d45671ca307d1f403592ed8d83544391caf26476a3bc6df3eae",
    "candidate_builder": "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e",
    "replay_builder": "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "v2_domain_manifest": "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "cohort_v1_manifest": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "design_builder": "4eefeb992dd65d7e4e45e3462fc43f975f0c1f31de7b7e989f3727b65565356e",
}

PROTOCOL_PATH = OUTPUT / "STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL.json"
INPUT_VERIFICATION_PATH = OUTPUT / "PILOT_INPUT_VERIFICATION.json"
HELD_OUT_AUDIT_PATH = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"

FACTOR_IDS = design_v2.EXPECTED_FACTOR_IDS
FACTOR_SHORT = {
    "S1F1_BIARTICULAR_LMAX": "F1 lmax",
    "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE": "F2 fpmax balance",
    "S1F3_HIP_MONO_ANTAGONIST_F0": "F3 hip F0",
    "S1F4_KNEE_MONO_ANTAGONIST_F0": "F4 knee F0",
}
VISUAL_COMPONENT = {
    "S1F1_BIARTICULAR_LMAX": "S1F1_AFFECTED_BIARTICULAR_HIP_CONTRIBUTION_RMS",
    "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE": "S1F2_NET_DECLARED_BIARTICULAR_HIP_CONTRIBUTION_RMS",
    "S1F3_HIP_MONO_ANTAGONIST_F0": "S1F3_DECLARED_HIP_ANTAGONIST_HIP_CONTRIBUTION_RMS",
    "S1F4_KNEE_MONO_ANTAGONIST_F0": "S1F4_DECLARED_KNEE_ANTAGONIST_KNEE_CONTRIBUTION_RMS",
}
RANGE_PATHWAY_CANDIDATES = {
    "S1F1_BIARTICULAR_LMAX": "independent model-specific mapping study from fiber/tendon measurements and passive mechanics to the normalized MuJoCo lmax field",
    "S1F2_RECTUS_HAMSTRING_FPMAX_BALANCE": "independent multi-configuration passive-mechanics calibration separating rectus and hamstring-family fpmax contributions",
    "S1F3_HIP_MONO_ANTAGONIST_F0": "independent architecture/force-capacity and multi-configuration mechanics calibration for the declared hip group",
    "S1F4_KNEE_MONO_ANTAGONIST_F0": "independent architecture/force-capacity and multi-configuration mechanics calibration for the declared knee group",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify_checksum_manifest(directory: Path, manifest: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = directory / relative.strip()
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"checksum manifest mismatch: {target}")
        count += 1
    return count


def verify_inputs() -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in FROZEN_PATHS.items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"FAIL-CLOSED frozen input mismatch: {actual}")
    if verify_checksum_manifest(DESIGN_DIR, DESIGN_FILES["design_checksums"]) != 13:
        raise RuntimeError("Pilot Design V2 checksum coverage changed")

    amended = read_json(AMENDED_S1)
    protocol = read_json(DESIGN_FILES["design_protocol"])
    metadata = read_json(DESIGN_FILES["design_metadata"])
    levels = read_json(DESIGN_FILES["design_levels"])
    responses = read_json(DESIGN_FILES["design_response_representations"])
    nonprop = read_json(DESIGN_FILES["design_nonproportionality_gates"])
    configuration = read_json(DESIGN_FILES["design_configuration_gates"])
    gradient = read_json(DESIGN_FILES["design_gradient_gates"])
    integrity = read_json(DESIGN_FILES["design_integrity_rules"])
    cohort = read_json(DESIGN_FILES["design_cohort_rules"])
    subset = read_csv(DESIGN_FILES["design_subset"])
    formal = read_json(FORMAL_MANIFEST)
    v3_manifest = read_json(V3_MANIFEST)
    truth = read_json(TRUTH_SEMANTICS)

    if not (
        amended.get("definition_id") == "S1_STRUCTURAL_DEFINITION_AMENDED_V1"
        and tuple(row["factor_id"] for row in amended["factors"]) == FACTOR_IDS
        and all(row.get("exact_members") and row.get("exact_fields") and row.get("operator") for row in amended["factors"])
        and protocol.get("formal_outcome") == "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_READY_WITH_EVIDENCE_GAPS"
        and protocol.get("future_pilot_id") == STAGE_ID
        and metadata.get("scientific_pilot_executed") is False
        and metadata.get("held_out_scientific_access_count") == 0
    ):
        raise RuntimeError("FAIL-CLOSED authoritative definition or design status mismatch")
    level_rows = levels.get("levels", [])
    if not (
        levels.get("status") == "ALL_FACTORS_READY"
        and tuple(row["factor_id"] for row in level_rows) == FACTOR_IDS
        and all(row["precheck_status"] == "DIAGNOSTIC_LEVEL_READY" for row in level_rows)
        and [row["primary_positive_z"] for row in level_rows] == [0.01, 0.025, 0.025, 0.025]
        and [row["fallback_positive_z"] for row in level_rows] == [0.005, 0.0125, 0.0125, 0.0125]
        and all(row["population_range"] == "NOT_AVAILABLE" for row in level_rows)
    ):
        raise RuntimeError("FAIL-CLOSED diagnostic level mismatch")
    if not (
        len(subset) == 13
        and [int(row["selection_order"]) for row in subset] == list(range(13))
        and all(row["selection_basis"] == "BETA_SPACE_GEOMETRY_ONLY" for row in subset)
        and all(row["J_or_oracle_or_rank_used"] == "False" for row in subset)
        and len({row["candidate_id"] for row in subset}) == 13
    ):
        raise RuntimeError("FAIL-CLOSED 13-candidate subset mismatch")
    if not (
        responses.get("primary_endpoint") == "CONFIGURATION_DEPENDENT_NONPROPORTIONAL_MECHANICAL_RESPONSE"
        and responses.get("post_outcome_representation_addition_allowed") is False
        and nonprop["effect_resolution_gate"]["delta_response_RMS_min_Nm_inclusive"] == 1.0e-5
        and nonprop["effect_resolution_gate"]["delta_response_RMS_over_nominal_RMS_min_inclusive"] == 1.0e-4
        and nonprop["nonproportionality_thresholds"]["proportional_NRMSE_strictly_above"] == 1.0e-4
        and nonprop["nonproportionality_thresholds"]["affine_R2_strictly_below"] == 0.9999
        and configuration["thresholds"]["normalized_spread_min_inclusive"] == 1.0e-4
        and configuration["thresholds"]["normalized_range_min_inclusive"] == 2.0e-4
        and configuration["thresholds"]["beta_polynomial_R2_min_inclusive"] == 0.25
        and gradient["direction_evidence"]["cosine_similarity_max_inclusive"] == 0.995
        and gradient["direction_evidence"]["unit_direction_component_change_max_min_inclusive"] == 0.05
        and gradient["magnitude_only_gradient_change_is_direction_evidence"] is False
    ):
        raise RuntimeError("FAIL-CLOSED scientific metric/gate mismatch")
    if not (
        integrity["maximum_fallback_attempts_per_factor_sign"] == 1
        and "integrity failure only" in integrity["fallback_trigger"]
        and cohort["automatic_admission"] is False
        and cohort["old_v1_heldout_scientific_access_count"] == 0
        and v3_manifest["candidate_count"] == 625
        and v3_manifest["mechanical_objective_evaluated"] is False
        and formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA["formal_reference"]
        and truth["semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    ):
        raise RuntimeError("FAIL-CLOSED integrity/cohort/formal semantic mismatch")
    return {
        "actual_sha256": actual,
        "amended": amended,
        "design_protocol": protocol,
        "levels": levels,
        "responses": responses,
        "nonprop": nonprop,
        "configuration": configuration,
        "gradient": gradient,
        "integrity": integrity,
        "cohort": cohort,
        "subset": subset,
    }


def execution_protocol(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": "STRUCTURAL_HETEROGENEITY_PILOT_EXECUTION_PROTOCOL_V1",
        "stage_id": STAGE_ID,
        "frozen_before_first_scientific_structural_replay": True,
        "authoritative_definition_sha256": FROZEN_SHA["amended_s1"],
        "pilot_design_protocol_sha256": FROZEN_SHA["design_protocol"],
        "input_sha256": FROZEN_SHA,
        "primary_question": "Do the four preregistered S1 structural factors produce configuration-dependent non-proportional mechanical response across the frozen V3 coordination trajectories rather than magnitude scaling?",
        "model_plan": {
            "label": "STRUCTURAL_DIAGNOSTIC_MODEL",
            "structure": "ONE_FACTOR_AT_A_TIME",
            "nominal_count": 1,
            "primary_perturbed_count": 8,
            "primary_model_count": 9,
            "trajectories_per_model": 13,
            "primary_replay_count": 117,
            "maximum_fallback_models": 8,
            "maximum_total_models": 17,
            "maximum_total_replays": 221,
            "virtual_patient_label_prohibited": True,
        },
        "factor_levels": inputs["levels"]["levels"],
        "candidate_subset": inputs["subset"],
        "response_representations": inputs["responses"],
        "nonproportionality_gate": inputs["nonprop"],
        "configuration_dependence_gate": inputs["configuration"],
        "gradient_rotation_gate": inputs["gradient"],
        "integrity_and_fallback_rules": inputs["integrity"],
        "cohort_admission_rules": inputs["cohort"],
        "classification_logic": {
            "factor_sign_structural": "same sign has >=1 required-torque response AND >=1 factor-specific component, each passing frozen effect-resolution, nonproportionality, and configuration gates",
            "STRUCTURALLY_INFORMATIVE": "both signs integrity-valid and at least one sign is factor_sign_structural",
            "MAGNITUDE_ONLY": "integrity-valid, at least one resolved response, and no nonproportional-plus-configuration or gradient-direction evidence",
            "INCONCLUSIVE": "integrity-valid but effects unresolved or evidence mixed without the frozen factor-sign conjunction",
            "INVALID": "either sign remains integrity-invalid after the single allowed fallback",
            "gradient_rotation_is_supporting_not_mandatory": True,
            "produces_personalization_prohibited": True,
        },
        "primary_decision_logic": {
            "STRUCTURAL_HETEROGENEITY_PILOT_SUPPORTED": "one or more factors are STRUCTURALLY_INFORMATIVE",
            "STRUCTURAL_HETEROGENEITY_PILOT_PARTIALLY_SUPPORTED": "zero factors are STRUCTURALLY_INFORMATIVE but at least one predeclared response passes both nonproportionality and configuration gates",
            "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED": "neither condition above is met",
        },
        "sign_symmetry": {
            "metrics": ["cosine(delta_positive,-delta_negative)", "RMS(delta_positive+delta_negative)/max(RMS(delta_positive),RMS(delta_negative))"],
            "role": "DESCRIPTIVE_ONLY_NO_PREREGISTERED_SYMMETRY_THRESHOLD",
            "may_change_factor_semantics": False,
        },
        "range_pathway_policy": {
            "population_ranges": "NOT_AVAILABLE",
            "candidate_pathways": RANGE_PATHWAY_CANDIDATES,
            "pathways_defensible_for_admission_without_independent_evidence_audit": False,
            "mechanism_supported_does_not_equal_population_variability_calibrated": True,
        },
        "visualization_plan": {
            "factor_curve_files": [f"{factor_id}_nominal_vs_perturbed.png" for factor_id in FACTOR_IDS],
            "cross_factor_files": ["delta_response_across_13_beta_locations.png", "proportional_fit_residuals.png", "gradient_vectors.png", "factor_summary.png"],
            "fixed_component_per_factor": VISUAL_COMPONENT,
            "dramatic_factor_selection": False,
        },
        "scope_guards": {
            "candidate_scientific_replay_count": 13,
            "625_grid_scientific_search": False,
            "cohort_subject_models": False,
            "new_virtual_subjects": 0,
            "cohort_v2_generation": False,
            "oracle_or_rank_or_regret": False,
            "objective_or_normalization_change": False,
            "five_parameter_or_NN_or_PINN_training": False,
            "BO": False,
            "held_out_scientific_access": 0,
            "robot_or_hardware": False,
        },
        "next_stage_auto_execution": False,
    }


def prepare() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing pilot directory: {OUTPUT}")
    inputs = verify_inputs()
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir()
    protocol = execution_protocol(inputs)
    write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256(PROTOCOL_PATH)
    write_json(INPUT_VERIFICATION_PATH, {
        "status": "PASS",
        "stage_id": STAGE_ID,
        "verified_before_scientific_replay": True,
        "frozen_input_sha256": inputs["actual_sha256"],
        "execution_protocol_sha256": protocol_sha,
        "authoritative_factor_ids": list(FACTOR_IDS),
        "authoritative_factor_reconstruction": inputs["amended"]["factors"],
        "design_artifact_count": len(DESIGN_FILES),
        "design_checksum_entry_count": 13,
        "primary_model_count": 9,
        "candidate_subset_count": 13,
        "expected_primary_replay_count": 117,
        "scientific_replay_started": False,
        "J_or_oracle_or_rank_used_for_design_or_selection": False,
        "held_out_scientific_access_count": 0,
    })
    write_json(HELD_OUT_AUDIT_PATH, {
        "classification": "SEALED_V1_HELD_OUT_NOT_APPLICABLE_TO_STRUCTURAL_DIAGNOSTIC_PILOT",
        "cohort_v1_manifest_path": str(COHORT_V1_MANIFEST.relative_to(ROOT)),
        "cohort_v1_manifest_sha256": FROZEN_SHA["cohort_v1_manifest"],
        "allowed_operation_performed": "streaming SHA-256 of manifest only",
        "held_out_metadata_or_model_or_truth_loaded": False,
        "held_out_replay_count": 0,
        "held_out_J_tau_oracle_rank_access_count": 0,
        "held_out_scientific_access_count": 0,
        "old_v1_heldout_is_cohort_v2_confirmation": False,
        "execution_protocol_sha256": protocol_sha,
    })
    print(json.dumps({
        "stage_id": STAGE_ID,
        "phase": "PREPARED_BEFORE_SCIENTIFIC_REPLAY",
        "execution_protocol_sha256": protocol_sha,
        "primary_models": 9,
        "expected_primary_replays": 117,
        "held_out_scientific_access_count": 0,
    }, indent=2))


def verify_prepared_protocol(inputs: Mapping[str, Any]) -> str:
    if not (PROTOCOL_PATH.is_file() and INPUT_VERIFICATION_PATH.is_file() and HELD_OUT_AUDIT_PATH.is_file()):
        raise RuntimeError("pilot must be --prepare'd before --execute")
    verification = read_json(INPUT_VERIFICATION_PATH)
    actual = sha256(PROTOCOL_PATH)
    if not (
        verification.get("status") == "PASS"
        and verification.get("verified_before_scientific_replay") is True
        and verification.get("execution_protocol_sha256") == actual
        and read_json(PROTOCOL_PATH) == execution_protocol(inputs)
        and read_json(HELD_OUT_AUDIT_PATH).get("held_out_scientific_access_count") == 0
    ):
        raise RuntimeError("FAIL-CLOSED prepared execution protocol mismatch")
    result_names = {
        "PILOT_MODEL_INTEGRITY_RESULTS.csv", "PILOT_REPLAY_RESULTS.csv",
        "PILOT_NONPROPORTIONALITY_RESULTS.csv", "PILOT_CONFIGURATION_DEPENDENCE_RESULTS.csv",
        "PILOT_GRADIENT_ROTATION_RESULTS.csv", "PILOT_FACTOR_MECHANISTIC_RESULTS.csv",
        "PILOT_FACTOR_CLASSIFICATION.csv", "PILOT_COHORT_V2_ADMISSION_RESULTS.csv",
        "PILOT_FALLBACK_USAGE_AUDIT.json", "FINAL_PILOT_DECISION.json", "metadata.json",
        "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_REPORT.md", "checksums.sha256",
    }
    existing = result_names & {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if existing:
        raise RuntimeError(f"refusing to overwrite prior pilot results: {sorted(existing)}")
    return actual


def flatten_members(value: Any) -> list[str]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        result: list[str] = []
        for members in value.values():
            result.extend(members)
        return list(dict.fromkeys(result))
    raise RuntimeError(f"invalid frozen member definition: {value}")


def time_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=float) ** 2, time_s) / duration))


def trajectory_integrity(reference: Mapping[str, Any], generated: Any, source_row: Mapping[str, str], gates: Mapping[str, Any]) -> dict[str, Any]:
    q_ref = np.asarray(reference["q"], dtype=float)
    dq_ref = np.asarray(reference["dq"], dtype=float)
    ddq_ref = np.asarray(reference["ddq"], dtype=float)
    q = np.asarray(generated.q, dtype=float)
    dq = np.asarray(generated.dq, dtype=float)
    ddq = np.asarray(generated.ddq, dtype=float)
    time_s = np.asarray(reference["time_s"], dtype=float)
    q_deg, ref_deg = np.degrees(q), np.degrees(q_ref)
    extrema = max(
        [abs(float(np.min(q_deg[:, j]) - np.min(ref_deg[:, j]))) for j in (0, 1)]
        + [abs(float(np.max(q_deg[:, j]) - np.max(ref_deg[:, j]))) for j in (0, 1)]
        + [abs(float(np.ptp(q_deg[:, j]) - np.ptp(ref_deg[:, j]))) for j in (0, 1)]
    )
    anchors = np.isclose(reference["segment_phase"], 0.0, atol=1.0e-15) | np.isclose(reference["segment_phase"], 1.0, atol=1.0e-15)
    metrics = {
        "finite_q_dq_ddq": bool(np.isfinite(np.column_stack((q, dq, ddq))).all()),
        "sample_count": len(q),
        "duration_s": float(time_s[-1] - time_s[0]),
        "max_extrema_rom_error_deg": extrema,
        "hip_q_dq_ddq_array_exact": bool(np.array_equal(q[:, 0], q_ref[:, 0]) and np.array_equal(dq[:, 0], dq_ref[:, 0]) and np.array_equal(ddq[:, 0], ddq_ref[:, 0])),
        "q_closure_error_rad": float(np.max(np.abs(q[-1] - q[0]))),
        "dq_closure_error_rad_s": float(np.max(np.abs(dq[-1] - dq[0]))),
        "ddq_closure_error_rad_s2": float(np.max(np.abs(ddq[-1] - ddq[0]))),
        "branch_anchor_q_error_rad": float(np.max(np.abs(q[anchors] - q_ref[anchors]))),
        "branch_anchor_dq_error_rad_s": float(np.max(np.abs(dq[anchors] - dq_ref[anchors]))),
        "branch_anchor_ddq_error_rad_s2": float(np.max(np.abs(ddq[anchors] - ddq_ref[anchors]))),
        "minimum_warp_derivative": float(np.min(generated.warp_first_derivative)),
        "frozen_candidate_kinematic_gate_pass": source_row["kinematic_gate_pass"] == "True",
    }
    metrics["pass"] = bool(
        metrics["finite_q_dq_ddq"]
        and metrics["sample_count"] == gates["sample_count_exact"]
        and metrics["duration_s"] == gates["duration_s_exact"]
        and metrics["max_extrema_rom_error_deg"] <= gates["V3_extrema_ROM_error_max_deg"]
        and metrics["hip_q_dq_ddq_array_exact"]
        and metrics["q_closure_error_rad"] <= gates["q_closure_error_max_rad"]
        and metrics["dq_closure_error_rad_s"] <= gates["dq_closure_error_max_rad_s"]
        and metrics["ddq_closure_error_rad_s2"] <= gates["ddq_closure_error_max_rad_s2"]
        and metrics["branch_anchor_q_error_rad"] == 0.0
        and metrics["branch_anchor_dq_error_rad_s"] == 0.0
        and metrics["branch_anchor_ddq_error_rad_s2"] == 0.0
        and metrics["frozen_candidate_kinematic_gate_pass"]
    )
    return metrics


def build_models(mujoco: Any, amended: Mapping[str, Any], levels: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    level_by_factor = {row["factor_id"]: row for row in levels["levels"]}
    factor_by_id = {row["factor_id"]: row for row in amended["factors"]}
    base = mujoco.MjModel.from_xml_path(str(MODEL))
    models: dict[str, Any] = {"STRUCTURAL_DIAGNOSTIC_NOMINAL": base}
    specs: list[dict[str, Any]] = [{
        "model_id": "STRUCTURAL_DIAGNOSTIC_NOMINAL", "model_label": "STRUCTURAL_DIAGNOSTIC_MODEL",
        "factor_id": "NOMINAL", "sign": 0, "sign_label": "NOMINAL", "z": 0.0,
        "level_role": "NOMINAL", "fallback": False,
    }]
    primary: dict[tuple[str, int], dict[str, Any]] = {}
    for factor_id in FACTOR_IDS:
        for sign, sign_label in ((-1, "NEGATIVE"), (1, "POSITIVE")):
            z = sign * float(level_by_factor[factor_id]["primary_positive_z"])
            model_id = f"STRUCTURAL_DIAGNOSTIC_{factor_id}_{sign_label}_PRIMARY"
            model = mujoco.MjModel.from_xml_path(str(MODEL))
            design_v2.apply_authoritative_factor(mujoco, model, factor_by_id[factor_id], z)
            spec = {
                "model_id": model_id, "model_label": "STRUCTURAL_DIAGNOSTIC_MODEL",
                "factor_id": factor_id, "sign": sign, "sign_label": sign_label, "z": z,
                "level_role": "PRIMARY", "fallback": False,
            }
            models[model_id] = model
            specs.append(spec)
            primary[(factor_id, sign)] = spec
    return models, specs, primary


def model_precheck(mujoco: Any, model: Any, spec: Mapping[str, Any], amended: Mapping[str, Any]) -> dict[str, Any]:
    base = mujoco.MjModel.from_xml_path(str(MODEL))
    if spec["factor_id"] == "NOMINAL":
        exact = bool(
            np.array_equal(model.actuator_gainprm, base.actuator_gainprm)
            and np.array_equal(model.actuator_biasprm, base.actuator_biasprm)
        )
        check = {
            "only_declared_fields_and_members_changed": exact,
            "gain_bias_synchronized": True,
            "positive_parameter_domain": True,
            "lmin_lt_lmax": True,
            "compiled_single_state_forward_finite": True,
            "solver_warning_count": 0,
            "topology_fingerprint_unchanged": True,
            "pass": exact,
        }
    else:
        factor = next(row for row in amended["factors"] if row["factor_id"] == spec["factor_id"])
        check = design_v2.operator_precheck(mujoco, factor, float(spec["z"]), spec["level_role"])
    return {
        **dict(spec),
        "compile_pass": True,
        "only_declared_fields_changed": bool(check["only_declared_fields_and_members_changed"]),
        "gain_bias_synchronized": bool(check["gain_bias_synchronized"]),
        "valid_parameter_domain": bool(check["positive_parameter_domain"]),
        "lmin_lt_lmax": bool(check["lmin_lt_lmax"]),
        "single_state_forward_finite": bool(check["compiled_single_state_forward_finite"]),
        "single_state_solver_warning_count": int(check["solver_warning_count"]),
        "topology_unchanged": bool(check["topology_fingerprint_unchanged"]),
        "parameter_array_sha256": array_sha(np.asarray(model.actuator_gainprm), np.asarray(model.actuator_biasprm)),
        "operator_precheck_pass": bool(check["pass"]),
    }


def response_values(arrays: Mapping[str, np.ndarray], time_s: np.ndarray, representations: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    tau = np.asarray(arrays["tau_truth_nm"], dtype=float)
    names = [str(value) for value in np.asarray(arrays["actuator_names"])]
    contribution = np.asarray(arrays["muscle_torque_contribution_nm"], dtype=float)
    for response in representations["representations"]:
        response_id = response["response_id"]
        if response_id == "FROZEN_COMBINED_J":
            continue
        if response_id == "HIP_REQUIRED_TORQUE_RMS":
            values[response_id] = time_rms(tau[:, 0], time_s)
            continue
        if response_id == "KNEE_REQUIRED_TORQUE_RMS":
            values[response_id] = time_rms(tau[:, 1], time_s)
            continue
        members = flatten_members(response["members"])
        indices = [names.index(member) for member in members]
        joint = 0 if "_HIP_" in response_id else 1
        signed_group = np.sum(contribution[:, indices, joint], axis=1)
        values[response_id] = time_rms(signed_group, time_s)
    return values


def replay_one(
    replay_builder: Any,
    model: Any,
    spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
    generated: Any,
    trajectory_metrics: Mapping[str, Any],
    reference: Mapping[str, Any],
    representations: Mapping[str, Any],
    integrity_gates: Mapping[str, Any],
    denominators: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    replay_reference = {
        "time_s": reference["time_s"], "q": generated.q, "dq": generated.dq,
        "ddq": generated.ddq, "phases": reference["phases"], "rows": [],
    }
    arrays, runtime = replay_builder.prescribed_truth(model, replay_reference)
    time_s = np.asarray(reference["time_s"], dtype=float)
    values = response_values(arrays, time_s, representations)
    if denominators is None:
        denominators = np.asarray([values["HIP_REQUIRED_TORQUE_RMS"], values["KNEE_REQUIRED_TORQUE_RMS"]], dtype=float)
    values["FROZEN_COMBINED_J"] = float(np.sqrt(0.5 * (
        (values["HIP_REQUIRED_TORQUE_RMS"] / denominators[0]) ** 2
        + (values["KNEE_REQUIRED_TORQUE_RMS"] / denominators[1]) ** 2
    )))

    tau = np.asarray(arrays["tau_truth_nm"], dtype=float)
    joint_limit = np.asarray(arrays["constraint_joint_limit_internal_nm"], dtype=float)
    joint_limit_abs = float(np.max(np.abs(joint_limit)))
    relative = np.abs(joint_limit) / np.maximum(np.abs(tau), denominators[None, :])
    joint_limit_relative = float(np.max(relative))
    decomposition = max(
        float(np.max(np.abs(np.asarray(arrays[key], dtype=float))))
        for key in ("inverse_formula_residual_nm", "decomposition_residual_nm", "muscle_reconstruction_residual_nm")
    )
    numeric_arrays = [value for value in arrays.values() if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)]
    all_finite = all(bool(np.isfinite(value).all()) for value in numeric_arrays)
    warnings = int(np.max(arrays["warning_count"]))
    equality = float(np.max(np.abs(arrays["source_equality_residual"])))
    joint_count = int(np.max(arrays["constraint_joint_limit_active_count"]))
    tendon_count = int(np.max(arrays["constraint_tendon_limit_active_count"]))
    contact_count = int(np.max(arrays["constraint_contact_active_count"]))
    gates = integrity_gates["future_replay_gates"]
    integrity_pass = bool(
        trajectory_metrics["pass"] and all_finite
        and warnings <= gates["solver_warning_count_max"]
        and equality <= gates["source_equality_residual_max"]
        and joint_count <= gates["joint_limit_active_count_max"]
        and tendon_count <= gates["tendon_limit_active_count_max"]
        and contact_count <= gates["unexpected_contact_active_count_max"]
        and joint_limit_abs <= gates["joint_limit_contribution_max_abs_Nm"]
        and joint_limit_relative <= gates["joint_limit_contribution_max_relative"]
        and decomposition <= gates["truth_decomposition_residual_max_abs_Nm"]
    )
    row = {
        "model_id": spec["model_id"], "model_label": spec["model_label"],
        "factor_id": spec["factor_id"], "sign": spec["sign"], "sign_label": spec["sign_label"],
        "z": spec["z"], "level_role": spec["level_role"], "fallback": spec["fallback"],
        "candidate_id": candidate["candidate_id"], "candidate_index": candidate["candidate_index"],
        "selection_order": candidate["selection_order"], "selection_role": candidate["selection_role"],
        "beta_flex": candidate["beta_flex"], "beta_extend": candidate["beta_extend"],
        **values,
        "sample_count": trajectory_metrics["sample_count"], "duration_s": trajectory_metrics["duration_s"],
        "finite_q_dq_ddq_and_truth": bool(trajectory_metrics["finite_q_dq_ddq"] and all_finite),
        "solver_warning_count": warnings, "source_equality_residual_max": equality,
        "joint_limit_active_count_max": joint_count, "tendon_limit_active_count_max": tendon_count,
        "contact_active_count_max": contact_count,
        "joint_limit_contribution_max_abs_Nm": joint_limit_abs,
        "joint_limit_contribution_max_relative": joint_limit_relative,
        "truth_decomposition_residual_max_abs_Nm": decomposition,
        "V3_extrema_ROM_error_max_deg": trajectory_metrics["max_extrema_rom_error_deg"],
        "q_closure_error_rad": trajectory_metrics["q_closure_error_rad"],
        "dq_closure_error_rad_s": trajectory_metrics["dq_closure_error_rad_s"],
        "ddq_closure_error_rad_s2": trajectory_metrics["ddq_closure_error_rad_s2"],
        "branch_anchor_C2_preserved": bool(
            trajectory_metrics["branch_anchor_q_error_rad"] == 0.0
            and trajectory_metrics["branch_anchor_dq_error_rad_s"] == 0.0
            and trajectory_metrics["branch_anchor_ddq_error_rad_s2"] == 0.0
        ),
        "frozen_V3_kinematic_gate_pass": trajectory_metrics["frozen_candidate_kinematic_gate_pass"],
        "replay_wall_time_s": float(runtime["wall_time_s"]),
        "integrity_pass": integrity_pass,
    }
    return row, denominators


def run_model_replays(
    replay_builder: Any,
    model: Any,
    spec: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    generated: Mapping[str, Any],
    trajectory_checks: Mapping[str, Any],
    reference: Mapping[str, Any],
    representations: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    denominators: np.ndarray | None = None
    for candidate in candidates:
        row, denominators = replay_one(
            replay_builder, model, spec, candidate, generated[candidate["candidate_id"]],
            trajectory_checks[candidate["candidate_id"]], reference, representations, integrity, denominators,
        )
        rows.append(row)
    if rows[0]["candidate_id"] != "MYOLEG_V3_K0312" or abs(float(rows[0]["FROZEN_COMBINED_J"]) - 1.0) > 1.0e-12:
        raise RuntimeError(f"model-own frozen reference normalization failed: {spec['model_id']}")
    return rows


def failed_integrity_reasons(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    reasons: set[str] = set()
    for row in rows:
        if not row["finite_q_dq_ddq_and_truth"]:
            reasons.add("nonfinite")
        if row["solver_warning_count"]:
            reasons.add("solver_warning")
        if not row["branch_anchor_C2_preserved"] or not row["frozen_V3_kinematic_gate_pass"]:
            reasons.add("V3_trajectory_invariant")
        if not row["integrity_pass"]:
            reasons.add("one_or_more_frozen_numeric_integrity_gates")
    return sorted(reasons)


def build_fallback_model(mujoco: Any, factor: Mapping[str, Any], level: Mapping[str, Any], sign: int) -> tuple[Any, dict[str, Any]]:
    sign_label = "NEGATIVE" if sign < 0 else "POSITIVE"
    z = sign * float(level["fallback_positive_z"])
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    design_v2.apply_authoritative_factor(mujoco, model, factor, z)
    spec = {
        "model_id": f"STRUCTURAL_DIAGNOSTIC_{factor['factor_id']}_{sign_label}_FALLBACK",
        "model_label": "STRUCTURAL_DIAGNOSTIC_MODEL", "factor_id": factor["factor_id"],
        "sign": sign, "sign_label": sign_label, "z": z, "level_role": "FALLBACK",
        "fallback": True,
    }
    return model, spec


def r2_score(observed: np.ndarray, fitted: np.ndarray) -> float | None:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= np.finfo(float).eps:
        return None
    return float(1.0 - np.sum((observed - fitted) ** 2) / denominator)


def fit_metrics(nominal: np.ndarray, perturbed: np.ndarray, gate: Mapping[str, Any]) -> dict[str, Any]:
    scale = max(float(np.sqrt(np.mean(nominal**2))), 1.0e-12)
    delta = perturbed - nominal
    delta_rms = float(np.sqrt(np.mean(delta**2)))
    denominator = float(np.dot(nominal, nominal))
    proportional_a = float(np.dot(nominal, perturbed) / denominator) if denominator > 0.0 else 0.0
    proportional_fitted = proportional_a * nominal
    proportional_residual = perturbed - proportional_fitted
    prop_rms = float(np.sqrt(np.mean(proportional_residual**2)))
    prop_range = float(np.ptp(proportional_residual))
    affine_design = np.column_stack((nominal, np.ones(len(nominal))))
    affine_a, affine_b = np.linalg.lstsq(affine_design, perturbed, rcond=None)[0]
    affine_fitted = affine_design @ np.asarray([affine_a, affine_b])
    affine_residual = perturbed - affine_fitted
    affine_rms = float(np.sqrt(np.mean(affine_residual**2)))
    affine_range = float(np.ptp(affine_residual))
    effect = gate["effect_resolution_gate"]
    thresholds = gate["nonproportionality_thresholds"]
    absolute_pass = delta_rms >= effect["delta_response_RMS_min_Nm_inclusive"]
    relative = delta_rms / scale
    relative_pass = relative >= effect["delta_response_RMS_over_nominal_RMS_min_inclusive"]
    prop_nrmse = prop_rms / scale
    affine_r2 = r2_score(perturbed, affine_fitted)
    resolved = bool(absolute_pass and relative_pass)
    passed = bool(
        resolved and prop_nrmse > thresholds["proportional_NRMSE_strictly_above"]
        and affine_r2 is not None and affine_r2 < thresholds["affine_R2_strictly_below"]
    )
    return {
        "nominal_response_RMS_across_candidates": scale,
        "delta_response_RMS": delta_rms,
        "delta_response_RMS_over_nominal_RMS": relative,
        "absolute_effect_gate_pass": absolute_pass,
        "relative_effect_gate_pass": relative_pass,
        "effect_resolution_gate_pass": resolved,
        "proportional_a": proportional_a,
        "proportional_R2": r2_score(perturbed, proportional_fitted),
        "proportional_NRMSE": prop_nrmse,
        "proportional_residual_RMS": prop_rms,
        "proportional_residual_range": prop_range,
        "affine_a": float(affine_a), "affine_b": float(affine_b),
        "affine_R2": affine_r2,
        "affine_NRMSE": affine_rms / scale,
        "affine_residual_RMS": affine_rms,
        "affine_residual_range": affine_range,
        "nonproportionality_gate_pass": passed,
    }


def configuration_metrics(nominal: np.ndarray, perturbed: np.ndarray, beta: np.ndarray, fit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    delta = perturbed - nominal
    scale = float(fit["nominal_response_RMS_across_candidates"])
    matrix = np.column_stack((
        np.ones(len(beta)), beta[:, 0], beta[:, 1], beta[:, 0] ** 2,
        beta[:, 1] ** 2, beta[:, 0] * beta[:, 1],
    ))
    coefficients = np.linalg.lstsq(matrix, delta, rcond=None)[0]
    fitted = matrix @ coefficients
    polynomial_r2 = r2_score(delta, fitted)
    spread = float(np.std(delta, ddof=0))
    delta_range = float(np.ptp(delta))
    thresholds = gate["thresholds"]
    normalized_spread = spread / scale
    normalized_range = delta_range / scale
    passed = bool(
        fit["effect_resolution_gate_pass"]
        and normalized_spread >= thresholds["normalized_spread_min_inclusive"]
        and normalized_range >= thresholds["normalized_range_min_inclusive"]
        and polynomial_r2 is not None and polynomial_r2 >= thresholds["beta_polynomial_R2_min_inclusive"]
    )
    return {
        "delta_mean": float(np.mean(delta)),
        "delta_population_SD": spread,
        "delta_range": delta_range,
        "delta_RMS": float(np.sqrt(np.mean(delta**2))),
        "normalized_spread": normalized_spread,
        "normalized_range": normalized_range,
        "beta_polynomial_coefficients": json.dumps([float(value) for value in coefficients], separators=(",", ":")),
        "beta_polynomial_R2": polynomial_r2,
        "effect_resolution_gate_pass": bool(fit["effect_resolution_gate_pass"]),
        "configuration_dependence_gate_pass": passed,
    }


def frozen_gradient(values: Mapping[str, float], stencil: Mapping[str, Any]) -> np.ndarray:
    h = float(stencil["h"])
    return np.asarray([
        ((values["MYOLEG_V3_K0456"] + values["MYOLEG_V3_K0468"]) / 2.0 - (values["MYOLEG_V3_K0156"] + values["MYOLEG_V3_K0168"]) / 2.0) / (2.0 * h),
        ((values["MYOLEG_V3_K0168"] + values["MYOLEG_V3_K0468"]) / 2.0 - (values["MYOLEG_V3_K0156"] + values["MYOLEG_V3_K0456"]) / 2.0) / (2.0 * h),
    ])


def gradient_metrics(nominal_values: Mapping[str, float], perturbed_values: Mapping[str, float], scale: float, response_meta: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    nominal = frozen_gradient(nominal_values, response_meta["gradient_stencil"])
    perturbed = frozen_gradient(perturbed_values, response_meta["gradient_stencil"])
    nominal_norm = float(np.linalg.norm(nominal))
    perturbed_norm = float(np.linalg.norm(perturbed))
    h = float(gate["gradient_resolution"]["h"])
    resolution_threshold = gate["gradient_resolution"]["minimum_norm_times_h_over_nominal_response_RMS_inclusive"]
    nominal_resolved = nominal_norm * h / scale >= resolution_threshold
    perturbed_resolved = perturbed_norm * h / scale >= resolution_threshold
    both = bool(nominal_resolved and perturbed_resolved)
    cosine = float(np.dot(nominal, perturbed) / (nominal_norm * perturbed_norm)) if nominal_norm and perturbed_norm else None
    if cosine is not None:
        cosine = min(1.0, max(-1.0, cosine))
    angle = math.degrees(math.acos(cosine)) if cosine is not None else None
    nominal_unit = nominal / nominal_norm if nominal_norm else np.zeros(2)
    perturbed_unit = perturbed / perturbed_norm if perturbed_norm else np.zeros(2)
    unit_change = float(np.max(np.abs(perturbed_unit - nominal_unit)))
    resolved_components = [
        bool(abs(nominal[index]) * h / scale >= resolution_threshold and abs(perturbed[index]) * h / scale >= resolution_threshold)
        for index in (0, 1)
    ]
    sign_change = any(
        resolved_components[index] and np.signbit(nominal[index]) != np.signbit(perturbed[index])
        for index in (0, 1)
    )
    direction = gate["direction_evidence"]
    passed = bool(
        both and (
            (cosine is not None and cosine <= direction["cosine_similarity_max_inclusive"])
            or sign_change
            or unit_change >= direction["unit_direction_component_change_max_min_inclusive"]
        )
    )
    norm_ratio = perturbed_norm / nominal_norm if nominal_norm else None
    if passed:
        change_type = "DIRECTION_CHANGE"
    elif both and norm_ratio is not None and abs(norm_ratio - 1.0) >= 1.0e-4:
        change_type = "MAGNITUDE_CHANGE_WITHOUT_DIRECTION_GATE"
    else:
        change_type = "DIRECTION_RETAINED_OR_UNRESOLVED"
    return {
        "nominal_gradient_beta_flex": float(nominal[0]),
        "nominal_gradient_beta_extend": float(nominal[1]),
        "perturbed_gradient_beta_flex": float(perturbed[0]),
        "perturbed_gradient_beta_extend": float(perturbed[1]),
        "nominal_gradient_norm": nominal_norm,
        "perturbed_gradient_norm": perturbed_norm,
        "nominal_gradient_resolved": nominal_resolved,
        "perturbed_gradient_resolved": perturbed_resolved,
        "both_gradients_resolved": both,
        "cosine_similarity": cosine,
        "angle_difference_deg": angle,
        "nominal_sign_pattern": f"{int(np.sign(nominal[0]))},{int(np.sign(nominal[1]))}",
        "perturbed_sign_pattern": f"{int(np.sign(perturbed[0]))},{int(np.sign(perturbed[1]))}",
        "resolved_component_sign_change": sign_change,
        "unit_direction_component_change_max": unit_change,
        "gradient_norm_ratio": norm_ratio,
        "gradient_change_type": change_type,
        "gradient_rotation_gate_pass": passed,
    }


def scientific_metrics(
    replay_rows: list[dict[str, Any]],
    actual_models: Mapping[tuple[str, int], Mapping[str, Any]],
    response_meta: Mapping[str, Any],
    nonprop_gate: Mapping[str, Any],
    configuration_gate: Mapping[str, Any],
    gradient_gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_model_candidate = {(row["model_id"], row["candidate_id"]): row for row in replay_rows}
    nominal_id = "STRUCTURAL_DIAGNOSTIC_NOMINAL"
    candidates = sorted(
        [row for row in replay_rows if row["model_id"] == nominal_id],
        key=lambda row: int(row["selection_order"]),
    )
    beta = np.asarray([[row["beta_flex"], row["beta_extend"]] for row in candidates], dtype=float)
    primary_responses = [row for row in response_meta["representations"] if row["role"].startswith("PRIMARY")]
    nonprop_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    mechanistic_rows: list[dict[str, Any]] = []
    vector_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for factor_id in FACTOR_IDS:
        applicable = [row for row in primary_responses if factor_id in row["applicable_factors"]]
        for sign in (-1, 1):
            spec = actual_models[(factor_id, sign)]
            for response in applicable:
                response_id = response["response_id"]
                nominal = np.asarray([row[response_id] for row in candidates], dtype=float)
                perturbed = np.asarray([
                    by_model_candidate[(spec["model_id"], row["candidate_id"])][response_id]
                    for row in candidates
                ], dtype=float)
                vector_cache[(factor_id, sign, response_id)] = (nominal, perturbed)
                identity = {
                    "factor_id": factor_id, "factor_name": FACTOR_SHORT[factor_id],
                    "sign": sign, "sign_label": "NEGATIVE" if sign < 0 else "POSITIVE",
                    "model_id": spec["model_id"], "z": spec["z"], "level_role": spec["level_role"],
                    "response_id": response_id, "response_role": response["role"], "units": response["units"],
                    "candidate_count": 13,
                }
                fit = fit_metrics(nominal, perturbed, nonprop_gate)
                config = configuration_metrics(nominal, perturbed, beta, fit, configuration_gate)
                nominal_map = {row["candidate_id"]: float(row[response_id]) for row in candidates}
                perturbed_map = {
                    row["candidate_id"]: float(by_model_candidate[(spec["model_id"], row["candidate_id"])][response_id])
                    for row in candidates
                }
                gradient = gradient_metrics(nominal_map, perturbed_map, fit["nominal_response_RMS_across_candidates"], response_meta, gradient_gate)
                nonprop_rows.append({**identity, **fit})
                config_rows.append({**identity, **config})
                gradient_rows.append({**identity, **gradient})
                mechanistic_rows.append({
                    **identity,
                    "delta_response_RMS": fit["delta_response_RMS"],
                    "relative_effect": fit["delta_response_RMS_over_nominal_RMS"],
                    "effect_resolution_gate_pass": fit["effect_resolution_gate_pass"],
                    "proportional_NRMSE": fit["proportional_NRMSE"],
                    "affine_R2": fit["affine_R2"],
                    "nonproportionality_gate_pass": fit["nonproportionality_gate_pass"],
                    "normalized_spread": config["normalized_spread"],
                    "normalized_range": config["normalized_range"],
                    "beta_polynomial_R2": config["beta_polynomial_R2"],
                    "configuration_dependence_gate_pass": config["configuration_dependence_gate_pass"],
                    "gradient_cosine_similarity": gradient["cosine_similarity"],
                    "gradient_angle_difference_deg": gradient["angle_difference_deg"],
                    "gradient_change_type": gradient["gradient_change_type"],
                    "gradient_rotation_gate_pass": gradient["gradient_rotation_gate_pass"],
                    "structural_response_gate_pass": bool(fit["nonproportionality_gate_pass"] and config["configuration_dependence_gate_pass"]),
                })

    for row in mechanistic_rows:
        factor_id, sign, response_id = row["factor_id"], int(row["sign"]), row["response_id"]
        negative = vector_cache[(factor_id, -1, response_id)]
        positive = vector_cache[(factor_id, 1, response_id)]
        delta_negative = negative[1] - negative[0]
        delta_positive = positive[1] - positive[0]
        denominator = float(np.linalg.norm(delta_positive) * np.linalg.norm(delta_negative))
        opposite_cosine = float(np.dot(delta_positive, -delta_negative) / denominator) if denominator else None
        inversion_residual = float(
            np.sqrt(np.mean((delta_positive + delta_negative) ** 2))
            / max(np.sqrt(np.mean(delta_positive**2)), np.sqrt(np.mean(delta_negative**2)), 1.0e-12)
        )
        row["positive_vs_negative_opposite_delta_cosine"] = opposite_cosine
        row["positive_plus_negative_delta_relative_RMS"] = inversion_residual
        row["sign_symmetry_role"] = "DESCRIPTIVE_ONLY_NO_PREREGISTERED_SYMMETRY_THRESHOLD"
        row["sign_result_used_to_change_factor_semantics"] = False
    return nonprop_rows, config_rows, gradient_rows, mechanistic_rows


def classify_factors(
    mechanistic: list[dict[str, Any]],
    actual_models: Mapping[tuple[str, int], Mapping[str, Any]],
    model_integrity: Mapping[str, bool],
) -> list[dict[str, Any]]:
    rows = []
    for factor_id in FACTOR_IDS:
        factor_rows = [row for row in mechanistic if row["factor_id"] == factor_id]
        sign_passes = {}
        for sign in (-1, 1):
            sign_rows = [row for row in factor_rows if int(row["sign"]) == sign]
            required = any(row["response_role"] == "PRIMARY_REQUIRED_TORQUE" and row["structural_response_gate_pass"] for row in sign_rows)
            component = any(row["response_role"] == "PRIMARY_MECHANISTIC_COMPONENT" and row["structural_response_gate_pass"] for row in sign_rows)
            sign_passes[sign] = bool(required and component)
            for row in sign_rows:
                row["sign_required_torque_structural_gate_any"] = required
                row["sign_mechanistic_component_structural_gate_any"] = component
                row["factor_sign_structural_gate_pass"] = sign_passes[sign]
        integrity_by_sign = {
            sign: bool(model_integrity[actual_models[(factor_id, sign)]["model_id"]]) for sign in (-1, 1)
        }
        both_integrity = all(integrity_by_sign.values())
        resolved = any(bool(row["effect_resolution_gate_pass"]) for row in factor_rows)
        joint_shape = any(bool(row["structural_response_gate_pass"]) for row in factor_rows)
        gradient = any(bool(row["gradient_rotation_gate_pass"]) for row in factor_rows)
        if not both_integrity:
            classification = "INVALID"
        elif any(sign_passes.values()):
            classification = "STRUCTURALLY_INFORMATIVE"
        elif resolved and not joint_shape and not gradient:
            classification = "MAGNITUDE_ONLY"
        else:
            classification = "INCONCLUSIVE"
        rows.append({
            "factor_id": factor_id, "factor_name": FACTOR_SHORT[factor_id],
            "negative_model_id": actual_models[(factor_id, -1)]["model_id"],
            "positive_model_id": actual_models[(factor_id, 1)]["model_id"],
            "negative_level_role": actual_models[(factor_id, -1)]["level_role"],
            "positive_level_role": actual_models[(factor_id, 1)]["level_role"],
            "negative_integrity_pass": integrity_by_sign[-1], "positive_integrity_pass": integrity_by_sign[1],
            "integrity": "PASS" if both_integrity else "FAIL",
            "maximum_absolute_effect_RMS_Nm": max(float(row["delta_response_RMS"]) for row in factor_rows),
            "maximum_relative_effect": max(float(row["relative_effect"]) for row in factor_rows),
            "nonproportionality_any": any(bool(row["nonproportionality_gate_pass"]) for row in factor_rows),
            "configuration_dependence_any": any(bool(row["configuration_dependence_gate_pass"]) for row in factor_rows),
            "gradient_rotation_any": gradient,
            "negative_factor_sign_structural_pass": sign_passes[-1],
            "positive_factor_sign_structural_pass": sign_passes[1],
            "classification": classification,
            "range_evidence": "NOT_AVAILABLE",
            "population_variability_calibrated": False,
            "range_calibration_pathway_candidate": RANGE_PATHWAY_CANDIDATES[factor_id],
            "range_calibration_pathway_defensible_without_new_audit": False,
            "sign_symmetry_opposite_delta_cosine_min": min(
                float(row["positive_vs_negative_opposite_delta_cosine"])
                for row in factor_rows if row["positive_vs_negative_opposite_delta_cosine"] is not None
            ),
            "sign_inversion_residual_relative_RMS_max": max(float(row["positive_plus_negative_delta_relative_RMS"]) for row in factor_rows),
            "sign_symmetry_formal_gate": "NONE_DESCRIPTIVE_ONLY",
        })
    return rows


def cohort_admission(classifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in classifications:
        classification = row["classification"]
        mechanics = classification == "STRUCTURALLY_INFORMATIVE"
        pathway = bool(row["range_calibration_pathway_defensible_without_new_audit"])
        if mechanics and pathway:
            status = "ELIGIBLE_FOR_COHORT_V2_RANGE_AND_DESIGN"
        elif mechanics:
            status = "COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP"
        elif classification == "MAGNITUDE_ONLY":
            status = "NOT_ELIGIBLE_MAGNITUDE_ONLY"
        elif classification == "INVALID":
            status = "NOT_ELIGIBLE_INVALID"
        else:
            status = "NOT_ELIGIBLE_INCONCLUSIVE"
        rows.append({
            "factor_id": row["factor_id"], "factor_name": row["factor_name"],
            "model_semantics_defensible": True,
            "integrity_pass": row["integrity"] == "PASS",
            "configuration_dependent_nonproportional_mechanics_demonstrated": mechanics,
            "population_range_calibration_pathway_defensible_now": pathway,
            "population_range": "NOT_AVAILABLE",
            "calibration_pathway_candidate": row["range_calibration_pathway_candidate"],
            "admission_result": status,
            "cohort_v2_generated": False,
            "synthetic_diagnostic_z_promoted_to_population_bound": False,
        })
    return rows


def final_decision(classifications: list[dict[str, Any]], mechanistic: list[dict[str, Any]]) -> dict[str, Any]:
    structural = [row for row in classifications if row["classification"] == "STRUCTURALLY_INFORMATIVE"]
    partial_response = any(bool(row["structural_response_gate_pass"]) for row in mechanistic)
    if structural:
        decision = "STRUCTURAL_HETEROGENEITY_PILOT_SUPPORTED"
    elif partial_response:
        decision = "STRUCTURAL_HETEROGENEITY_PILOT_PARTIALLY_SUPPORTED"
    else:
        decision = "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED"
    defensible_path = [row for row in structural if row["range_calibration_pathway_defensible_without_new_audit"]]
    if structural and defensible_path:
        next_stage = "MYOLEG_VIRTUAL_PATIENT_COHORT_V2_RANGE_AND_DESIGN_V1"
    elif structural:
        next_stage = "MYOLEG_STRUCTURAL_FACTOR_RANGE_EVIDENCE_AUDIT_V1"
    elif decision == "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED":
        next_stage = "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_AUDIT_V1"
    else:
        next_stage = "MYOLEG_STRUCTURAL_FACTOR_RANGE_EVIDENCE_AUDIT_V1"
    return {
        "stage_id": STAGE_ID,
        "primary_decision": decision,
        "structurally_informative_factor_count": len(structural),
        "structurally_informative_factor_ids": [row["factor_id"] for row in structural],
        "classification_counts": {
            name: sum(row["classification"] == name for row in classifications)
            for name in ("STRUCTURALLY_INFORMATIVE", "MAGNITUDE_ONLY", "INCONCLUSIVE", "INVALID")
        },
        "population_range_calibrated_factor_count": 0,
        "structural_mechanism_supported_does_not_equal_population_variability_calibrated": True,
        "recommended_independent_next_stage": next_stage,
        "next_stage_executed": False,
        "larger_z_retry_recommended": False,
        "cohort_v2_generated": False,
    }


COLORS = {"nominal": "#222222", "negative": "#2f6fed", "positive": "#d64545", "grid": "#d9dee8", "text": "#1f2937"}


def font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def canvas(title: str, width: int = 1600, height: int = 1000) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), title, fill=COLORS["text"], font=font(28))
    return image, draw


def panel_box(index: int, rows: int, cols: int, width: int = 1600, height: int = 1000) -> tuple[int, int, int, int]:
    margin_x, top, bottom, gap = 70, 90, 55, 30
    usable_w = width - 2 * margin_x - gap * (cols - 1)
    usable_h = height - top - bottom - gap * (rows - 1)
    w, h = usable_w / cols, usable_h / rows
    row, col = divmod(index, cols)
    x0 = int(margin_x + col * (w + gap))
    y0 = int(top + row * (h + gap))
    return x0, y0, int(x0 + w), int(y0 + h)


def line_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, series: list[tuple[str, list[float], str]]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="#9aa4b2", width=1)
    draw.text((x0 + 10, y0 + 8), title, fill=COLORS["text"], font=font(17))
    left, right, top, bottom = x0 + 65, x1 - 20, y0 + 48, y1 - 45
    all_values = [value for _, values, _ in series for value in values]
    lo, hi = min(all_values), max(all_values)
    padding = max((hi - lo) * 0.12, max(abs(lo), abs(hi), 1.0) * 1.0e-6)
    lo, hi = lo - padding, hi + padding
    draw.line((left, top, left, bottom), fill="#7b8794", width=1)
    draw.line((left, bottom, right, bottom), fill="#7b8794", width=1)
    for tick in range(5):
        y = top + (bottom - top) * tick / 4
        value = hi - (hi - lo) * tick / 4
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((x0 + 4, y - 8), f"{value:.3g}", fill="#64748b", font=font(12))
    count = len(series[0][1])
    for label, values, color in series:
        points = []
        for index, value in enumerate(values):
            x = left + (right - left) * index / max(count - 1, 1)
            y = bottom - (bottom - top) * (value - lo) / (hi - lo)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
    legend_x = left
    for label, _, color in series:
        draw.line((legend_x, y1 - 20, legend_x + 24, y1 - 20), fill=color, width=3)
        draw.text((legend_x + 30, y1 - 29), label, fill=COLORS["text"], font=font(12))
        legend_x += 150


def save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=True)


def figures(
    replay_rows: list[dict[str, Any]],
    actual_models: Mapping[tuple[str, int], Mapping[str, Any]],
    nonprop: list[dict[str, Any]],
    gradient: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> list[Path]:
    by_model = {}
    for row in replay_rows:
        by_model.setdefault(row["model_id"], []).append(row)
    for values in by_model.values():
        values.sort(key=lambda row: int(row["selection_order"]))
    nominal = by_model["STRUCTURAL_DIAGNOSTIC_NOMINAL"]
    paths = []
    for factor_id in FACTOR_IDS:
        image, draw = canvas(f"{FACTOR_SHORT[factor_id]}: nominal vs preregistered perturbations")
        neg = by_model[actual_models[(factor_id, -1)]["model_id"]]
        pos = by_model[actual_models[(factor_id, 1)]["model_id"]]
        for panel, response in enumerate(("HIP_REQUIRED_TORQUE_RMS", "KNEE_REQUIRED_TORQUE_RMS")):
            line_panel(draw, panel_box(panel, 1, 2), response, [
                ("negative", [float(row[response]) for row in neg], COLORS["negative"]),
                ("nominal", [float(row[response]) for row in nominal], COLORS["nominal"]),
                ("positive", [float(row[response]) for row in pos], COLORS["positive"]),
            ])
        path = FIGURES / f"{factor_id}_nominal_vs_perturbed.png"
        save_png(image, path)
        paths.append(path)

    image, draw = canvas("Delta required-torque response across the 13 frozen beta locations", height=1500)
    for factor_index, factor_id in enumerate(FACTOR_IDS):
        neg = by_model[actual_models[(factor_id, -1)]["model_id"]]
        pos = by_model[actual_models[(factor_id, 1)]["model_id"]]
        for joint_index, response in enumerate(("HIP_REQUIRED_TORQUE_RMS", "KNEE_REQUIRED_TORQUE_RMS")):
            panel = factor_index * 2 + joint_index
            line_panel(draw, panel_box(panel, 4, 2, height=1500), f"{FACTOR_SHORT[factor_id]} / {response.split('_')[0]}", [
                ("negative delta", [float(a[response]) - float(b[response]) for a, b in zip(neg, nominal)], COLORS["negative"]),
                ("positive delta", [float(a[response]) - float(b[response]) for a, b in zip(pos, nominal)], COLORS["positive"]),
            ])
    path = FIGURES / "delta_response_across_13_beta_locations.png"
    save_png(image, path)
    paths.append(path)

    fit_index = {(row["factor_id"], int(row["sign"]), row["response_id"]): row for row in nonprop}
    image, draw = canvas("Proportional-fit residuals for the predeclared mechanistic representation", height=1400)
    for index, factor_id in enumerate(FACTOR_IDS):
        response = VISUAL_COMPONENT[factor_id]
        series = []
        for sign, label, color in ((-1, "negative residual", COLORS["negative"]), (1, "positive residual", COLORS["positive"])):
            changed = by_model[actual_models[(factor_id, sign)]["model_id"]]
            a = float(fit_index[(factor_id, sign, response)]["proportional_a"])
            residual = [float(row[response]) - a * float(base[response]) for row, base in zip(changed, nominal)]
            series.append((label, residual, color))
        line_panel(draw, panel_box(index, 4, 1, height=1400), f"{FACTOR_SHORT[factor_id]} / {response}", series)
    path = FIGURES / "proportional_fit_residuals.png"
    save_png(image, path)
    paths.append(path)

    gradient_index = {(row["factor_id"], int(row["sign"]), row["response_id"]): row for row in gradient}
    image, draw = canvas("Frozen-stencil gradient vectors: nominal vs perturbed", height=1500)
    for factor_index, factor_id in enumerate(FACTOR_IDS):
        for joint_index, response in enumerate(("HIP_REQUIRED_TORQUE_RMS", "KNEE_REQUIRED_TORQUE_RMS")):
            box = panel_box(factor_index * 2 + joint_index, 4, 2, height=1500)
            x0, y0, x1, y1 = box
            draw.rectangle(box, outline="#9aa4b2")
            draw.text((x0 + 10, y0 + 8), f"{FACTOR_SHORT[factor_id]} / {response.split('_')[0]}", fill=COLORS["text"], font=font(15))
            center = ((x0 + x1) / 2, (y0 + y1) / 2 + 12)
            vectors = []
            nominal_row = gradient_index[(factor_id, -1, response)]
            vectors.append(("nominal", np.asarray([nominal_row["nominal_gradient_beta_flex"], nominal_row["nominal_gradient_beta_extend"]]), COLORS["nominal"]))
            for sign, label, color in ((-1, "negative", COLORS["negative"]), (1, "positive", COLORS["positive"])):
                row = gradient_index[(factor_id, sign, response)]
                vectors.append((label, np.asarray([row["perturbed_gradient_beta_flex"], row["perturbed_gradient_beta_extend"]]), color))
            maximum = max(float(np.linalg.norm(vector)) for _, vector, _ in vectors) or 1.0
            draw.line((x0 + 25, center[1], x1 - 25, center[1]), fill=COLORS["grid"])
            draw.line((center[0], y0 + 40, center[0], y1 - 25), fill=COLORS["grid"])
            legend_x = x0 + 15
            for label, vector, color in vectors:
                scale = min(x1 - x0, y1 - y0) * 0.30 / maximum
                endpoint = (center[0] + vector[0] * scale, center[1] - vector[1] * scale)
                draw.line((center[0], center[1], endpoint[0], endpoint[1]), fill=color, width=4)
                draw.ellipse((endpoint[0] - 4, endpoint[1] - 4, endpoint[0] + 4, endpoint[1] + 4), fill=color)
                draw.text((legend_x, y1 - 20), label, fill=color, font=font(11))
                legend_x += 100
    path = FIGURES / "gradient_vectors.png"
    save_png(image, path)
    paths.append(path)

    image, draw = canvas("Structural-factor gate and classification summary", height=620)
    columns = ["Factor", "Integrity", "Non-prop", "Config", "Gradient", "Classification", "Range"]
    widths = [330, 120, 140, 120, 120, 300, 180]
    x, y = 45, 105
    for label, width in zip(columns, widths):
        draw.rectangle((x, y, x + width, y + 48), fill="#e8eef7", outline="#9aa4b2")
        draw.text((x + 8, y + 13), label, fill=COLORS["text"], font=font(15))
        x += width
    for row_index, row in enumerate(classifications):
        x, row_y = 45, y + 48 + row_index * 82
        values = [
            row["factor_name"], row["integrity"], "PASS" if row["nonproportionality_any"] else "FAIL",
            "PASS" if row["configuration_dependence_any"] else "FAIL",
            "PASS" if row["gradient_rotation_any"] else "FAIL", row["classification"], row["range_evidence"],
        ]
        for label, width in zip(values, widths):
            fill = "#ecfdf3" if label == "PASS" else "#fff7ed" if label in {"FAIL", "NOT_AVAILABLE"} else "white"
            draw.rectangle((x, row_y, x + width, row_y + 82), fill=fill, outline="#c4ccd6")
            draw.text((x + 8, row_y + 28), str(label), fill=COLORS["text"], font=font(13))
            x += width
    path = FIGURES / "factor_summary.png"
    save_png(image, path)
    paths.append(path)
    return paths


def report_text(
    decision: Mapping[str, Any], classifications: list[dict[str, Any]], admissions: list[dict[str, Any]],
    model_rows: list[dict[str, Any]], fallback: Mapping[str, Any], replay_count: int, protocol_sha: str,
) -> str:
    by_class = {row["factor_id"]: row for row in classifications}
    by_admission = {row["factor_id"]: row for row in admissions}
    all_primary = [row for row in model_rows if row["level_role"] in {"NOMINAL", "PRIMARY"}]
    integrity_pass = sum(bool(row["model_integrity_pass"]) for row in all_primary)
    factor_lines = "\n".join(
        f"- `{factor_id}`: `{by_class[factor_id]['classification']}`; non-proportional any={by_class[factor_id]['nonproportionality_any']}, configuration any={by_class[factor_id]['configuration_dependence_any']}, gradient rotation any={by_class[factor_id]['gradient_rotation_any']}; admission `{by_admission[factor_id]['admission_result']}`."
        for factor_id in FACTOR_IDS
    )
    structural = decision["structurally_informative_factor_ids"]
    next_reason = (
        "Mechanistic evidence passed, but every population range and direct field-to-human calibration remains unavailable; an independent range-evidence audit is required before Cohort V2 range/design."
        if structural else
        "No factor met the frozen structural-informativeness logic; increasing z is prohibited and a formulation stop/pivot audit is the independent next step."
    )
    return f"""# MyoLeg Structural Heterogeneity Pilot V1

## Formal decision

**{decision['primary_decision']}**

Execution protocol SHA-256: `{protocol_sha}`  
Authoritative amended S1 SHA-256: `{FROZEN_SHA['amended_s1']}`

This was a mechanistic offline structural diagnostic, not personalization. It executed the frozen 13-trajectory geometry subset for one nominal and eight one-factor-at-a-time models. It computed no oracle, rank, regret, personalization necessity or 625-point search.

## Q1. Did all nine primary diagnostic models pass integrity?

**{integrity_pass == 9}.** Primary integrity PASS: **{integrity_pass}/9**. Every model was compiled from the same nominal model, only authoritative members/fields changed, gain/bias remained synchronized, topology was unchanged, and every replay was checked against the frozen numerical and V3 trajectory gates.

## Q2. Was fallback used?

Fallback models used: **{fallback['fallback_model_count']}**. It was permitted only for preregistered integrity failures. No small effect, failed scientific gate or unfavorable result could trigger it. Actual total replay count: **{replay_count}**.

## Q3-Q7. Factor findings

{factor_lines}

Positive and negative signs are reported separately in all result tables. Opposite-delta cosine and inversion residual are descriptive only because no sign-symmetry threshold was preregistered; neither was used to change factor semantics.

## Q8. Is a defensible population-range calibration pathway already available?

**No factor has an admission-ready population calibration pathway yet.** Candidate measurement/calibration programs are specified, but the normalized `lmax`, family `fpmax`, and active-plus-passive F0 fields do not have validated human mappings or population bounds in the frozen evidence. `STRUCTURAL_MECHANISM_SUPPORTED` therefore does not mean `POPULATION_VARIABILITY_CALIBRATED`.

## Q9. Is Cohort V2 generation justified now?

**No.** Structurally informative factors, if any, are only `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`. No diagnostic z value was promoted to a population bound and no Cohort V2 was generated.

## Q10. Exact independent next stage

`{decision['recommended_independent_next_stage']}`

{next_reason} The next stage was not executed automatically.

## Frozen-gate interpretation

- Non-proportionality required absolute delta RMS >= `1e-5 N*m`, relative delta RMS >= `1e-4`, proportional NRMSE > `1e-4`, and affine R2 < `0.9999` on the same response.
- Configuration dependence additionally required normalized SD >= `1e-4`, normalized range >= `2e-4`, and fixed beta-polynomial R2 >= `0.25`.
- A factor-sign required at least one required-torque response and one preregistered factor component passing both gates.
- Gradient rotation was supporting evidence only; magnitude-only gradient changes never counted as rotation.

## Stop state

- Primary models/replays planned: `9 / 117`; actual replays: `{replay_count}`.
- New virtual subjects or Cohort V2: `0`.
- Held-out scientific access: `0`.
- Objective, normalization, amended S1, V3 parameterization/domain and V1 cohort: unchanged.
- Oracle/Five-parameter/NN/PINN/BO: not run.
- Robot/hardware: untouched.
"""


def execute() -> None:
    started = time.perf_counter()
    inputs = verify_inputs()
    protocol_sha = verify_prepared_protocol(inputs)

    import mujoco
    from external_simulation.myoleg_reference_trajectory_replay_v1 import build_and_replay as replay_builder
    from external_simulation.myoleg_v2_candidate_domain_design_v1 import build_candidate_domain as candidate_builder
    from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization

    reference = candidate_builder.load_reference_adapter()
    subset_rows = sorted(inputs["subset"], key=lambda row: int(row["selection_order"]))
    candidates = [{
        "selection_order": int(row["selection_order"]), "selection_role": row["selection_role"],
        "candidate_id": row["candidate_id"], "candidate_index": int(row["candidate_index"]),
        "beta_flex": float(row["beta_flex"]), "beta_extend": float(row["beta_extend"]),
    } for row in subset_rows]
    if candidates[0]["candidate_id"] != "MYOLEG_V3_K0312":
        raise RuntimeError("frozen reference is not first")
    table = {row["candidate_id"]: row for row in read_csv(V3_TABLE)}
    generated = {
        row["candidate_id"]: parameterization.generate_v3_trajectory(reference, row["beta_flex"], row["beta_extend"])
        for row in candidates
    }
    integrity_gates = inputs["integrity"]["future_replay_gates"]
    trajectory_checks = {
        row["candidate_id"]: trajectory_integrity(reference, generated[row["candidate_id"]], table[row["candidate_id"]], integrity_gates)
        for row in candidates
    }
    if not all(row["pass"] for row in trajectory_checks.values()):
        raise RuntimeError("FAIL-CLOSED frozen V3 trajectory reconstruction integrity failure")

    models, specs, primary_specs = build_models(mujoco, inputs["amended"], inputs["levels"])
    model_prechecks = {spec["model_id"]: model_precheck(mujoco, models[spec["model_id"]], spec, inputs["amended"]) for spec in specs}
    if not all(row["operator_precheck_pass"] for row in model_prechecks.values()):
        raise RuntimeError("FAIL-CLOSED primary model operator precheck failure")
    replay_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    model_integrity: dict[str, bool] = {}
    fallback_events: list[dict[str, Any]] = []
    actual_specs: dict[tuple[str, int], dict[str, Any]] = {}
    factor_by_id = {row["factor_id"]: row for row in inputs["amended"]["factors"]}
    level_by_factor = {row["factor_id"]: row for row in inputs["levels"]["levels"]}

    for spec in specs:
        rows = run_model_replays(
            replay_builder, models[spec["model_id"]], spec, candidates, generated,
            trajectory_checks, reference, inputs["responses"], inputs["integrity"],
        )
        replay_rows.extend(rows)
        replay_pass = all(bool(row["integrity_pass"]) for row in rows)
        model_pass = bool(model_prechecks[spec["model_id"]]["operator_precheck_pass"] and replay_pass)
        model_integrity[spec["model_id"]] = model_pass
        model_rows.append({
            **model_prechecks[spec["model_id"]],
            "trajectory_replay_count": len(rows),
            "trajectory_integrity_pass_count": sum(bool(row["integrity_pass"]) for row in rows),
            "all_replay_integrity_pass": replay_pass,
            "failed_integrity_reasons": "|".join(failed_integrity_reasons(rows)),
            "fallback_triggered": False,
            "model_integrity_pass": model_pass,
        })
        if spec["factor_id"] != "NOMINAL":
            actual_specs[(spec["factor_id"], int(spec["sign"]))] = dict(spec)
    if not model_integrity["STRUCTURAL_DIAGNOSTIC_NOMINAL"]:
        raise RuntimeError("FAIL-CLOSED nominal diagnostic model integrity failure; no fallback allowed")

    for key, primary in primary_specs.items():
        if model_integrity[primary["model_id"]]:
            continue
        factor_id, sign = key
        failed = [row for row in replay_rows if row["model_id"] == primary["model_id"]]
        reasons = failed_integrity_reasons(failed)
        fallback_model, fallback_spec = build_fallback_model(
            mujoco, factor_by_id[factor_id], level_by_factor[factor_id], sign
        )
        check = model_precheck(mujoco, fallback_model, fallback_spec, inputs["amended"])
        fallback_rows = run_model_replays(
            replay_builder, fallback_model, fallback_spec, candidates, generated,
            trajectory_checks, reference, inputs["responses"], inputs["integrity"],
        )
        replay_rows.extend(fallback_rows)
        passed = bool(check["operator_precheck_pass"] and all(row["integrity_pass"] for row in fallback_rows))
        model_integrity[fallback_spec["model_id"]] = passed
        model_rows.append({
            **check, "trajectory_replay_count": len(fallback_rows),
            "trajectory_integrity_pass_count": sum(bool(row["integrity_pass"]) for row in fallback_rows),
            "all_replay_integrity_pass": all(row["integrity_pass"] for row in fallback_rows),
            "failed_integrity_reasons": "|".join(failed_integrity_reasons(fallback_rows)),
            "fallback_triggered": True, "model_integrity_pass": passed,
        })
        actual_specs[key] = fallback_spec
        fallback_events.append({
            "factor_id": factor_id, "sign": sign, "primary_model_id": primary["model_id"],
            "primary_integrity_failure_reasons": reasons,
            "fallback_model_id": fallback_spec["model_id"], "fallback_z": fallback_spec["z"],
            "trigger_was_integrity_only": True, "effect_or_scientific_gate_read_before_trigger": False,
            "fallback_integrity_pass": passed, "additional_level_attempted": False,
        })

    nonprop, configuration, gradient, mechanistic = scientific_metrics(
        replay_rows, actual_specs, inputs["responses"], inputs["nonprop"],
        inputs["configuration"], inputs["gradient"],
    )
    classifications = classify_factors(mechanistic, actual_specs, model_integrity)
    admissions = cohort_admission(classifications)
    decision = final_decision(classifications, mechanistic)
    fallback_audit = {
        "rule": "fallback only after preregistered primary model/numerical integrity failure",
        "primary_model_count": 9,
        "primary_replay_count": 117,
        "fallback_model_count": len(fallback_events),
        "fallback_replay_count": len(fallback_events) * 13,
        "actual_total_model_count": len(model_rows),
        "actual_total_replay_count": len(replay_rows),
        "events": fallback_events,
        "fallback_triggered_by_small_effect_or_failed_scientific_gate": False,
        "larger_perturbation_attempted": False,
        "additional_levels_added": False,
    }

    write_csv(OUTPUT / "PILOT_MODEL_INTEGRITY_RESULTS.csv", model_rows)
    write_csv(OUTPUT / "PILOT_REPLAY_RESULTS.csv", replay_rows)
    write_csv(OUTPUT / "PILOT_NONPROPORTIONALITY_RESULTS.csv", nonprop)
    write_csv(OUTPUT / "PILOT_CONFIGURATION_DEPENDENCE_RESULTS.csv", configuration)
    write_csv(OUTPUT / "PILOT_GRADIENT_ROTATION_RESULTS.csv", gradient)
    write_csv(OUTPUT / "PILOT_FACTOR_MECHANISTIC_RESULTS.csv", mechanistic)
    write_csv(OUTPUT / "PILOT_FACTOR_CLASSIFICATION.csv", classifications)
    write_csv(OUTPUT / "PILOT_COHORT_V2_ADMISSION_RESULTS.csv", admissions)
    write_json(OUTPUT / "PILOT_FALLBACK_USAGE_AUDIT.json", fallback_audit)
    write_json(OUTPUT / "FINAL_PILOT_DECISION.json", decision)
    figure_paths = figures(replay_rows, actual_specs, nonprop, gradient, classifications)
    report = report_text(decision, classifications, admissions, model_rows, fallback_audit, len(replay_rows), protocol_sha)
    atomic_text(OUTPUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_REPORT.md", report)
    elapsed = time.perf_counter() - started
    write_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID,
        "formal_decision": decision["primary_decision"],
        "execution_protocol_sha256": protocol_sha,
        "authoritative_definition_sha256": FROZEN_SHA["amended_s1"],
        "primary_model_count": 9,
        "primary_model_integrity_pass_count": sum(row["model_integrity_pass"] for row in model_rows if row["level_role"] in {"NOMINAL", "PRIMARY"}),
        "primary_replay_count": 117,
        "fallback_model_count": len(fallback_events),
        "fallback_replay_count": len(fallback_events) * 13,
        "actual_model_count": len(model_rows),
        "actual_replay_count": len(replay_rows),
        "candidate_scientific_replay_count": 13,
        "625_grid_scientific_search": False,
        "structurally_informative_factor_ids": decision["structurally_informative_factor_ids"],
        "classification_counts": decision["classification_counts"],
        "population_ranges_available": False,
        "new_virtual_subjects": 0,
        "cohort_v2_generated": False,
        "oracle_or_rank_or_regret_computed": False,
        "objective_or_normalization_modified": False,
        "S1_or_V3_or_V1_cohort_modified": False,
        "five_parameter_or_NN_or_PINN_or_BO_run": False,
        "held_out_scientific_access_count": 0,
        "robot_or_hardware": False,
        "figure_count": len(figure_paths),
        "runtime_seconds": elapsed,
        "recommended_independent_next_stage": decision["recommended_independent_next_stage"],
        "next_stage_executed": False,
        "analysis_code_sha256": sha256(Path(__file__)),
    })
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    atomic_text(
        OUTPUT / "checksums.sha256",
        "".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files),
    )
    print(json.dumps({
        "stage_id": STAGE_ID,
        "formal_decision": decision["primary_decision"],
        "primary_model_integrity": f"{sum(row['model_integrity_pass'] for row in model_rows if row['level_role'] in {'NOMINAL', 'PRIMARY'})}/9",
        "fallback_models": len(fallback_events),
        "actual_replays": len(replay_rows),
        "classifications": {row["factor_id"]: row["classification"] for row in classifications},
        "recommended_next_stage": decision["recommended_independent_next_stage"],
        "held_out_scientific_access_count": 0,
        "runtime_seconds": elapsed,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute()


if __name__ == "__main__":
    main()
