"""Final frozen-reference release tests; offline and SDK-free."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    RehabFrameConfig,
)
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
)
from lower_limb_sim.reference_release import (
    RELEASE_ACTIVE_REFERENCE_PATH,
    RELEASE_VERSION_MANIFEST_PATH,
    load_final_result_metadata,
    load_frozen_active_reference,
    load_reference_release_manifest,
    require_parent_reference,
    verify_reference_sha256,
)
from lower_limb_sim.run_reference_local_active_asymmetric import _provenance_frame
from lower_limb_sim.run_reference_candidate_evaluation import (
    run_reference_candidate_evaluation,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_one_active_reference_id_and_strict_sha() -> None:
    bundle = load_frozen_active_reference()
    versions = pd.read_csv(RELEASE_VERSION_MANIFEST_PATH)
    active = versions["active_reference"].astype(str).str.lower().eq("true")
    assert active.sum() == 1
    assert versions.loc[active, "trajectory_id"].item() == ACTIVE_REFERENCE_ID
    assert DEFAULT_REFERENCE_PATH.resolve() == RELEASE_ACTIVE_REFERENCE_PATH.resolve()
    assert bundle.audit.valid
    assert bundle.audit.sha256 == ACTIVE_REFERENCE_SHA256
    assert verify_reference_sha256(DEFAULT_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256


def test_one_value_change_triggers_hash_gate(tmp_path: Path) -> None:
    changed = tmp_path / "changed_reference.csv"
    data = RELEASE_ACTIVE_REFERENCE_PATH.read_bytes()
    changed.write_bytes(data.replace(b"0.0,flexion", b"0.1,flexion", 1))
    assert changed.read_bytes() != data
    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        verify_reference_sha256(changed)


def test_release_cycle_duration_rom_and_robot_no_go() -> None:
    manifest = load_reference_release_manifest()
    assert ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES == (5844, 5895, 5934)
    assert (
        manifest["selected_cycle_start_frame"],
        manifest["selected_cycle_peak_frame"],
        manifest["selected_cycle_end_frame"],
    ) == ACTIVE_REFERENCE_PARENT_CYCLE_FRAMES
    assert manifest["hip_rom_deg"] == list(FORMAL_HIP_ROM_DEG)
    assert manifest["knee_rom_deg"] == list(FORMAL_KNEE_ROM_DEG)
    assert manifest["flexion_duration_s"] == pytest.approx(13.6)
    assert manifest["extension_duration_s"] == pytest.approx(10.4)
    assert manifest["total_duration_s"] == pytest.approx(24.0)
    assert manifest["approved_for_offline_personalization"] is True
    assert manifest["approved_for_first_robot_trial"] is False
    assert manifest["robot_execution_status"] == "NO_GO"


def test_geometry_c2_theta_asymmetry_and_extension_invariants() -> None:
    bundle = load_frozen_active_reference()
    trajectory = bundle.trajectory
    audit = bundle.audit
    assert audit.joint_closure_valid
    assert audit.pull_closure_valid
    assert audit.theta_shank_valid
    assert audit.rom_valid
    assert audit.c2_continuity_valid
    assert audit.asymmetry_valid
    assert audit.all_finite
    assert bundle.manifest["trajectory_continuity"] == "C2"
    assert bundle.manifest["measured_extension_is_reversed_flexion"] is False
    assert not trajectory["measured_extension_is_reversed_flexion"].astype(bool).any()
    np.testing.assert_allclose(
        trajectory["theta_shank_rad"],
        trajectory["q_hip_rad"] - trajectory["q_knee_rad"],
        atol=1e-14,
        rtol=0.0,
    )
    assert bundle.manifest["hip_flexion_extension_asymmetry_rmse_deg"] > 1.0
    assert bundle.manifest["knee_flexion_extension_asymmetry_rmse_deg"] > 1.0
    assert bundle.manifest["pull_path_asymmetry_rmse_mm"] > 1.0


def test_symmetric_legacy_rows_cannot_be_active_or_final_inputs() -> None:
    versions = pd.read_csv(RELEASE_VERSION_MANIFEST_PATH)
    legacy = versions["trajectory_id"].astype(str).str.contains(
        "reference_closed_symmetric|reference_closed_c2", regex=True
    )
    assert legacy.any()
    for field in (
        "legacy",
        "not_used_for_final_personalization",
        "not_used_for_robot_execution",
    ):
        assert versions.loc[legacy, field].astype(str).str.lower().eq("true").all()
    assert versions.loc[legacy, "active_reference"].astype(str).str.lower().eq("false").all()
    with pytest.raises(PermissionError, match="canonical release CSV"):
        load_frozen_active_reference(
            PROJECT_ROOT / "lower_limb_sim" / "data" / "reference_closed_c2_slow.csv"
        )


def test_downstream_metadata_requires_and_propagates_parent_binding(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="parent_reference_sha256"):
        require_parent_reference({"parent_reference_id": ACTIVE_REFERENCE_ID})
    missing_parent = tmp_path / "legacy_result.json"
    missing_parent.write_text('{"status": "FORMAL"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent_reference_id"):
        load_final_result_metadata(missing_parent)
    bound_result = tmp_path / "bound_result.json"
    bound_result.write_text(
        json.dumps(
            {
                "parent_reference_id": ACTIVE_REFERENCE_ID,
                "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
            }
        ),
        encoding="utf-8",
    )
    require_parent_reference(load_final_result_metadata(bound_result))
    reference, source_metadata = load_closed_reference_trajectory()
    require_parent_reference(source_metadata)
    _, _, robot_metadata = build_start_anchored_relative_trajectory(
        DEFAULT_REFERENCE_PATH,
        current_tcp_start_pose=(0.4, 0.0, 0.3, 0.1, 0.2, 0.3),
        rehab_frame=RehabFrameConfig(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            False,
        ),
    )
    require_parent_reference(robot_metadata)
    assert robot_metadata["robot_execution_approved"] is False

    provenance = _provenance_frame(
        pd.DataFrame({"value": [1.0]}),
        generated_at_utc="2026-08-13T00:00:00+00:00",
        git_commit=None,
        reference_sha256=ACTIVE_REFERENCE_SHA256,
    )
    assert provenance["parent_reference_id"].eq(ACTIVE_REFERENCE_ID).all()
    assert provenance["parent_reference_sha256"].eq(ACTIVE_REFERENCE_SHA256).all()


def test_legacy_candidate_metadata_cannot_enter_final_result_loader(
    tmp_path: Path,
) -> None:
    result = run_reference_candidate_evaluation(
        output_directory=tmp_path,
        save_outputs=False,
        generate_plots=False,
        samples_per_segment=31,
    )
    assert result.metadata["final_result_eligible"] is False
    with pytest.raises(RuntimeError, match="parent_reference_id"):
        require_parent_reference(result.metadata)


def test_freeze_modules_do_not_import_hardware_sdk() -> None:
    paths = (
        PROJECT_ROOT / "lower_limb_sim" / "reference_release.py",
        PROJECT_ROOT / "lower_limb_sim" / "test_reference_release_freeze.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not any(
            name.startswith(("hardware", "xCoreSDK_python")) for name in imports
        )


def test_formal_manifest_points_to_the_one_release_manifest() -> None:
    formal = json.loads(
        (PROJECT_ROOT / "config" / "formal_experiment_manifest.json").read_text()
    )
    assert formal["reference_release_manifest"] == (
        "reference_release/reference_release_manifest.json"
    )
