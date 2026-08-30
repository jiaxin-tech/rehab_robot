"""Artifact-level tests for MYOLEG_REFERENCE_TRAJECTORY_REPLAY_V1.

These tests intentionally do not import MuJoCo.  The expensive, frozen-runtime
simulation performs its own numerical invariant tests; normal repository pytest
validates the retained data, hashes, schema and non-interference contract.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
)
BUILDER = (
    ROOT
    / "external_simulation"
    / "myoleg_reference_trajectory_replay_v1"
    / "build_and_replay.py"
)
PRIMARY_REFERENCE = (
    ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
SENSITIVITY_REFERENCE = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def source_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    time_s = np.asarray([float(row["time_s"]) for row in rows])
    q = np.asarray([[float(row["q_hip_rad"]), float(row["q_knee_rad"])] for row in rows])
    dq = np.asarray([[float(row["dq_hip_rad_s"]), float(row["dq_knee_rad_s"])] for row in rows])
    ddq = np.asarray([[float(row["ddq_hip_rad_s2"]), float(row["ddq_knee_rad_s2"])] for row in rows])
    return time_s, q, dq, ddq


def test_required_formal_artifacts_and_checksums() -> None:
    required = {
        "MYOLEG_REFERENCE_TRAJECTORY_REPLAY_REPORT.md",
        "REPLAY_PROTOCOL.json",
        "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json",
        "PRIMARY_REFERENCE_REPLAY.npz",
        "PRIMARY_REFERENCE_REPLAY_SUMMARY.csv",
        "SENSITIVITY_REFERENCE_REPLAY.npz",
        "SENSITIVITY_REFERENCE_REPLAY_SUMMARY.csv",
        "FORCE_SEMANTICS_VALIDATION.csv",
        "DYNAMICS_BALANCE_AUDIT.csv",
        "PRIMARY_VS_SENSITIVITY.csv",
        "RUNTIME_BENCHMARK.json",
        "DATASET_SCHEMA.json",
        "TEST_RESULTS.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert required.issubset({path.name for path in ARTIFACTS.iterdir()})
    for line in (ARTIFACTS / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(ARTIFACTS / relative.strip()) == expected


def test_frozen_inputs_and_protocol_are_unchanged() -> None:
    protocol = read_json("REPLAY_PROTOCOL.json")
    assert sha256(PRIMARY_REFERENCE) == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert protocol["primary_condition"]["model_sha256"] == "c652424679308411fb73a211ad1fc770002fd760c8339c1ed9553888c14e0d41"
    assert protocol["sensitivity_condition"]["model_sha256"] == "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d"
    assert sha256(SENSITIVITY_REFERENCE) == "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678"
    assert protocol["coordinate_convention"]["theta_shank"] == "q_hip - q_knee"
    assert protocol["p0_condition"]["muscle_control"] == 0.0
    assert protocol["p0_condition"]["initial_activation"] == 0.0
    assert protocol["p0_condition"]["warmup_duration_s"] == 0.0
    assert all(not audit["pointwise_clipping_used"] for audit in protocol["input_trajectory_audit"].values())
    assert all(audit["duration_s"] == 24.0 and audit["sample_count"] == 401 for audit in protocol["input_trajectory_audit"].values())
    assert all(value["nu"] == value["ntendon"] == 80 for value in protocol["model_inventory"].values())
    assert all(value["source_knee_equality_count"] == 14 for value in protocol["model_inventory"].values())


def test_truth_semantics_are_explicit_and_passed() -> None:
    semantic = read_json("MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json")
    assert semantic["semantic_version"] == "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
    assert semantic["truth_field"] == "TAU_MY0LEG_REQUIRED_DRIVE"
    assert semantic["generalized_force_truth_semantics"] == "PASS"
    assert semantic["torque_sign_mapping"] == "PASS"
    assert semantic["reduced_coordinate_equation"] == "tau_truth = T(q)^T*r"
    assert "qfrc_actuator(P0)" in semantic["full_coordinate_equation"]
    assert semantic["determinism_fingerprints"]["PRIMARY"]["first"] == semantic["determinism_fingerprints"]["PRIMARY"]["repeat"]
    assert semantic["determinism_fingerprints"]["SENSITIVITY"]["first"] == semantic["determinism_fingerprints"]["SENSITIVITY"]["repeat"]


def test_npz_matches_unmodified_source_trajectories_and_is_finite() -> None:
    for condition, source in (
        ("PRIMARY", PRIMARY_REFERENCE),
        ("SENSITIVITY", SENSITIVITY_REFERENCE),
    ):
        expected_time, expected_q, expected_dq, expected_ddq = source_arrays(source)
        with np.load(ARTIFACTS / f"{condition}_REFERENCE_REPLAY.npz") as dataset:
            assert np.array_equal(dataset["time_s"], expected_time)
            assert np.array_equal(dataset["target_q_rad"], expected_q)
            assert np.array_equal(dataset["target_dq_rad_s"], expected_dq)
            assert np.array_equal(dataset["target_ddq_rad_s2"], expected_ddq)
            assert dataset["tau_truth_nm"].shape == (401, 2)
            assert dataset["actuator_force_n"].shape == (401, 80)
            assert dataset["tendon_length_m"].shape == (401, 80)
            assert dataset["constraint_joint_limit_internal_nm"].shape == (401, 2)
            assert np.array_equal(dataset["joint_names"], np.asarray(["hip", "knee"]))
            for key in dataset.files:
                if dataset[key].dtype.kind not in "USO":
                    assert np.isfinite(dataset[key]).all(), key
            assert np.max(np.abs(dataset["dynamics_balance_residual_nm"])) < 1e-8
            assert np.max(np.abs(dataset["decomposition_residual_nm"])) < 1e-9
            assert np.max(np.abs(dataset["ctrl_max"])) == 0.0
            assert np.max(np.abs(dataset["activation_max"])) == 0.0


def test_limited_extension_caveat_is_quantified_not_hidden() -> None:
    metadata = read_json("metadata.json")
    assert metadata["outcome"] == "MYOLEG_REFERENCE_REPLAY_VALID_WITH_LIMITATIONS"
    assert metadata["extension_sensitivity_assessment"]["assessment"] == "MATERIAL_HIGH_FLEXION_TORQUE_AMPLIFICATION_DETECTED"
    with (ARTIFACTS / "PRIMARY_VS_SENSITIVITY.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row["joint"]: row for row in csv.DictReader(stream)}
    knee = rows["knee"]
    assert float(knee["primary_to_sensitivity_rms_ratio"]) > 2.0
    assert float(knee["high_flexion_constraint_joint_limit_internal_primary_rms_nm"]) > 40.0
    assert float(knee["high_flexion_constraint_joint_limit_internal_sensitivity_rms_nm"]) < 0.01


def test_no_forbidden_research_or_robot_dependency() -> None:
    protocol = read_json("REPLAY_PROTOCOL.json")
    assert not any(protocol["forbidden_operations"].values())
    metadata = read_json("metadata.json")
    assert metadata["five_parameter_fit"] is False
    assert metadata["pinn_trained"] is False
    assert metadata["bo_run"] is False
    assert metadata["landscape_generated"] is False
    assert metadata["robot_connected"] is False
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint({"lower_limb_sim", "hardware", "control", "collection", "safety"})


def test_prior_stage_integrity_and_builder_provenance() -> None:
    metadata = read_json("metadata.json")
    assert metadata["builder_script_sha256"] == sha256(BUILDER)
    assert metadata["tests"]["status"] == "PASS"
    assert metadata["tests"]["failed"] == 0
    assert metadata["source_identity_before"]["hashes"] == metadata["source_identity_after"]["hashes"]
    assert metadata["source_identity_before"]["upstream_asset_sha256"] == metadata["source_identity_after"]["upstream_asset_sha256"]
    assert all(
        value["status"] == "PASS"
        for value in metadata["source_identity_after"]["prior_checksum_verification"].values()
    )
