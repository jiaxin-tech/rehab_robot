"""Formal ROM protocol migration regression tests (offline only)."""

from __future__ import annotations

import ast
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
    excluded_names = {"test_rom_protocol_migration.py"}
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if (
            path.name.startswith("test_")
            or path.name in excluded_names
            or ".venv" in relative.parts
            or "hardware/xcoresdk_python" in relative.as_posix()
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if float(node.value) == 130.0:
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


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
