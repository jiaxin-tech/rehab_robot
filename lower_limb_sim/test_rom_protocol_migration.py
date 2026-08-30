"""Formal ROM protocol migration regression tests (offline only)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from control.start_anchored_relative_trajectory import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.config import hip_range_deg, knee_range_deg
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_EXPERIMENT_MANIFEST,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    SOURCE_ACTIVE_REFERENCE_PATH,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from lower_limb_sim.kinematics import forward_kinematics, inverse_kinematics
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from lower_limb_sim.reference_release import RELEASE_ACTIVE_REFERENCE_PATH
from lower_limb_sim.workspace_atlas import build_workspace_atlas


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SOURCE_SCOPE_MANIFEST = (
    PROJECT_ROOT / "config" / "python_source_scope_manifest.json"
)
FROZEN_STRESS_AUDIT_SCRIPT = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_knee_rom_compatibility_v1"
    / "build_and_audit.py"
)
FROZEN_STRESS_AUDIT_METADATA = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "metadata.json"
)
FROZEN_STRESS_PROTOCOL = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "ROM_EXTENSION_PROTOCOL.json"
)
FROZEN_STRESS_130_XML = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_knee_rom_compatibility_v1"
    / "myoleg_supine_right_knee130_stress_only_v1.xml"
)
FROZEN_STRESS_AUDIT_SCRIPT_SHA256 = (
    "e40b1e9938f60e40ad1464c4ed219bef9fb0111a690f2fdd67843360042745b1"
)
FROZEN_STRESS_130_XML_SHA256 = (
    "d8007d0a65c1d49a988a181c1fd251d0766f6f25cecc89e20f7030aefa444151"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_source_scope_manifest(path: Path = PYTHON_SOURCE_SCOPE_MANIFEST) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["default_active_policy_gate_scan"] is True
    prefixes = [
        entry["path_prefix"].strip("/")
        for entry in manifest["non_production_path_prefixes"]
    ]
    assert all(prefix and ".." not in Path(prefix).parts for prefix in prefixes)
    assert len(prefixes) == len(set(prefixes))
    return manifest


def _is_active_policy_gate_source(relative: Path, scope_manifest: dict) -> bool:
    relative_posix = relative.as_posix()
    for entry in scope_manifest["non_production_path_prefixes"]:
        prefix = entry["path_prefix"].strip("/")
        if relative_posix == prefix or relative_posix.startswith(f"{prefix}/"):
            return False
    return bool(scope_manifest["default_active_policy_gate_scan"])


def _numeric_130_degree_gate_offenders(
    project_root: Path,
    scope_manifest: dict,
) -> list[str]:
    excluded_names = {"test_rom_protocol_migration.py"}
    offenders: list[str] = []
    for path in project_root.rglob("*.py"):
        relative = path.relative_to(project_root)
        if (
            path.name.startswith("test_")
            or path.name in excluded_names
            or ".venv" in relative.parts
            or "hardware/xcoresdk_python" in relative.as_posix()
            or not _is_active_policy_gate_source(relative, scope_manifest)
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if float(node.value) == 130.0:
                    offenders.append(f"{relative}:{node.lineno}")
    return offenders


def test_formal_runtime_and_manifest_are_one_rom_protocol() -> None:
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == hip_range_deg == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == knee_range_deg == (5.0, 145.0)
    assert APPROVED_HIP_ROM_DEG == FORMAL_HIP_ROM_DEG
    assert APPROVED_KNEE_ROM_DEG == FORMAL_KNEE_ROM_DEG
    assert FORMAL_EXPERIMENT_MANIFEST["hip_rom_deg"] == [0.0, 120.0]
    assert FORMAL_EXPERIMENT_MANIFEST["knee_rom_deg"] == [5.0, 145.0]
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert FORMAL_EXPERIMENT_MANIFEST["real_robot_safety_thresholds_reviewed"] is False


def test_workspace_contains_new_130_to_145_degree_region_and_jacobian_audit() -> None:
    atlas = build_workspace_atlas(step_deg=1.0)
    new_region = atlas.loc[atlas["q_knee_deg"].gt(130.0)]
    assert not new_region.empty
    assert new_region["q_knee_deg"].max() == 145.0
    assert new_region["rom_protocol_version"].eq(ROM_PROTOCOL_VERSION).all()
    assert new_region["theta_shank_definition"].eq(THETA_SHANK_DEFINITION).all()
    np.testing.assert_allclose(
        new_region["theta_shank_rad"],
        new_region["q_hip_rad"] - new_region["q_knee_rad"],
        atol=0.0,
        rtol=0.0,
    )
    assert np.isfinite(new_region["jacobian_determinant"]).all()
    assert np.isfinite(new_region["jacobian_condition_number"]).all()


def test_inverse_kinematics_accepts_143_and_rejects_above_145() -> None:
    q_hip = np.deg2rad(105.0)
    for knee_deg, expected_valid in ((143.0, True), (146.0, False)):
        _, _, x_pull, z_pull = forward_kinematics(
            q_hip, np.deg2rad(knee_deg), 0.42, 0.30
        )
        _, recovered_knee, reachable = inverse_kinematics(
            x_pull, z_pull, 0.42, 0.30
        )
        assert reachable is expected_valid
        if expected_valid:
            assert np.rad2deg(recovered_knee) == pytest.approx(knee_deg, abs=1e-10)
        else:
            assert np.isnan(recovered_knee)


def test_active_reference_is_pinned_valid_and_not_clipped() -> None:
    assert ACTIVE_REFERENCE_ID == "reference_measured_asymmetric_closed_slow"
    assert DEFAULT_REFERENCE_PATH.resolve() == RELEASE_ACTIVE_REFERENCE_PATH.resolve()
    assert DEFAULT_REFERENCE_PATH.resolve() == ACTIVE_REFERENCE_PATH.resolve()
    assert DEFAULT_REFERENCE_PATH.read_bytes() == SOURCE_ACTIVE_REFERENCE_PATH.read_bytes()
    assert sha256_file(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256
    reference = pd.read_csv(ACTIVE_REFERENCE_PATH)
    knee_deg = np.rad2deg(reference["q_knee_rad"].to_numpy(float))
    hip_deg = np.rad2deg(reference["q_hip_rad"].to_numpy(float))
    assert knee_deg.max() <= FORMAL_KNEE_ROM_DEG[1]
    assert knee_deg.min() >= FORMAL_KNEE_ROM_DEG[0]
    assert hip_deg.min() >= FORMAL_HIP_ROM_DEG[0]
    assert hip_deg.max() <= FORMAL_HIP_ROM_DEG[1]
    assert reference["joint_limit_valid"].astype(bool).all()
    assert reference["approved_knee_min_deg"].eq(5.0).all()
    assert reference["approved_knee_max_deg"].eq(145.0).all()
    assert reference["source_approved_knee_max_deg"].eq(145.0).all()
    np.testing.assert_allclose(
        reference["theta_shank_rad"],
        reference["q_hip_rad"] - reference["q_knee_rad"],
        atol=1e-14,
        rtol=0.0,
    )


def test_production_python_has_no_numeric_130_degree_gate() -> None:
    assert _numeric_130_degree_gate_offenders(
        PROJECT_ROOT,
        _python_source_scope_manifest(),
    ) == []


def test_actual_production_130_degree_gate_still_fails_scan(tmp_path: Path) -> None:
    active_source = tmp_path / "control" / "active_rom_gate.py"
    active_source.parent.mkdir(parents=True)
    active_source.write_text("FORMAL_KNEE_MAX_DEG = 130.0\n", encoding="utf-8")
    assert _numeric_130_degree_gate_offenders(
        tmp_path,
        _python_source_scope_manifest(),
    ) == ["control/active_rom_gate.py:1"]


def test_historical_external_simulation_is_not_an_active_gate(tmp_path: Path) -> None:
    historical_source = (
        tmp_path / "external_simulation" / "historical_stress" / "build_and_audit.py"
    )
    historical_source.parent.mkdir(parents=True)
    historical_source.write_text("STRESS_ONLY_LIMIT_DEG = 130.0\n", encoding="utf-8")
    assert _numeric_130_degree_gate_offenders(
        tmp_path,
        _python_source_scope_manifest(),
    ) == []


def test_external_simulation_scope_has_offline_metadata_evidence() -> None:
    manifest = _python_source_scope_manifest()
    external_entry = next(
        entry
        for entry in manifest["non_production_path_prefixes"]
        if entry["path_prefix"] == "external_simulation"
    )
    metadata_paths = sorted(PROJECT_ROOT.glob(external_entry["metadata_evidence_glob"]))
    assert metadata_paths
    referenced_builders = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        builder = metadata.get("builder_script_path")
        evidence_level = str(metadata.get("evidence_level", ""))
        if builder and "OFFLINE" in evidence_level:
            referenced_builders.append(Path(builder).resolve())
    assert FROZEN_STRESS_AUDIT_SCRIPT.resolve() in referenced_builders


def test_frozen_stress_only_audit_and_xml_sha_remain_unchanged() -> None:
    metadata = json.loads(FROZEN_STRESS_AUDIT_METADATA.read_text(encoding="utf-8"))
    protocol = json.loads(FROZEN_STRESS_PROTOCOL.read_text(encoding="utf-8"))
    assert _sha256(FROZEN_STRESS_AUDIT_SCRIPT) == FROZEN_STRESS_AUDIT_SCRIPT_SHA256
    assert metadata["builder_script_sha256"] == FROZEN_STRESS_AUDIT_SCRIPT_SHA256
    assert _sha256(FROZEN_STRESS_130_XML) == FROZEN_STRESS_130_XML_SHA256
    assert metadata["stress_130_xml_sha256"] == FROZEN_STRESS_130_XML_SHA256
    assert protocol["stress_only_not_formal_reference_eligible"] is True


def test_legacy_workspace_is_not_the_default_active_loader() -> None:
    legacy = PROJECT_ROOT / "lower_limb_sim" / "data" / "workspace" / "workspace_atlas.csv"
    from lower_limb_sim.config import workspace_csv_path

    assert workspace_csv_path.resolve() != legacy.resolve()
    assert "formal_artifacts/rom_protocol_v2" in workspace_csv_path.as_posix()


def test_hardware_sources_are_not_rom_protocol_consumers() -> None:
    for path in (PROJECT_ROOT / "hardware").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "ROM_PROTOCOL_V2" not in source
        assert "FORMAL_KNEE_ROM_DEG" not in source


def test_manifest_json_is_strict_json_and_runtime_consistent() -> None:
    path = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == FORMAL_EXPERIMENT_MANIFEST
