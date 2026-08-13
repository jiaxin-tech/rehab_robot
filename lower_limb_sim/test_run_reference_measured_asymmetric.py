"""End-to-end offline regression for the measured-asymmetric release runner."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lower_limb_sim.run_reference_measured_asymmetric import (
    DEFAULT_DETECTED_CYCLES_PATH,
    DEFAULT_FROZEN_LOCAL_DATASET_PATH,
    DEFAULT_FULL_ANGLES_PATH,
    DEFAULT_STAGE5A_METADATA_PATH,
    OUTPUT_FILENAMES,
    run_reference_measured_asymmetric,
)
from lower_limb_sim.visualize_reference_measured_asymmetric import (
    FIGURE_FILENAMES,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
LEGACY_REFERENCE_PATHS = (
    MODULE_DIRECTORY
    / "data"
    / "reference_candidates"
    / "reference_execution_versions.csv",
    MODULE_DIRECTORY
    / "data"
    / "reference_candidates"
    / "reference_closed_c2_phase.csv",
    MODULE_DIRECTORY
    / "data"
    / "reference_candidates"
    / "reference_closed_c2_slow.csv",
    MODULE_DIRECTORY
    / "data"
    / "reference_candidates"
    / "reference_closed_c2_nominal.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_all_true(series: pd.Series) -> None:
    assert series.astype(bool).all()


def _assert_all_false(series: pd.Series) -> None:
    assert not series.astype(bool).any()


def test_offline_runner_persists_complete_measured_asymmetric_release(
    tmp_path: Path,
    monkeypatch,
):
    output_directory = tmp_path / "measured-asymmetric-release"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))

    immutable_inputs = (
        DEFAULT_FULL_ANGLES_PATH,
        DEFAULT_DETECTED_CYCLES_PATH,
        DEFAULT_STAGE5A_METADATA_PATH,
        DEFAULT_FROZEN_LOCAL_DATASET_PATH,
        *LEGACY_REFERENCE_PATHS,
    )
    before_sha256 = {path: _sha256(path) for path in immutable_inputs}

    result = run_reference_measured_asymmetric(
        output_directory=output_directory,
        save_outputs=True,
        generate_plots=True,
    )

    # The complete release is written under tmp_path, never over Stage-5A or
    # either legacy symmetric/C2 product.
    assert output_directory.is_relative_to(tmp_path)
    assert all(path.is_file() for path in result.output_paths.values())
    assert before_sha256 == {path: _sha256(path) for path in immutable_inputs}
    assert result.output_paths["measured_raw"] != result.output_paths["periodic_phase"]

    required_tables = {
        OUTPUT_FILENAMES[key]
        for key in (
            "cycle_closure_audit",
            "measured_raw",
            "periodic_phase",
            "slow",
            "nominal",
            "domain_coverage",
            "manifest",
        )
    }
    assert required_tables.issubset(
        {path.name for path in output_directory.glob("*.csv")}
    )
    assert (output_directory / OUTPUT_FILENAMES["metadata"]).is_file()
    assert set(result.visualization_paths) == set(FIGURE_FILENAMES)
    for filename in FIGURE_FILENAMES:
        path = output_directory / filename
        assert path.is_file()
        assert path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    closure_audit = pd.read_csv(result.output_paths["cycle_closure_audit"])
    assert len(closure_audit) == 4
    assert closure_audit["cycle_candidate_id"].tolist() == [0, 1, 2, 3]
    selected = closure_audit.loc[closure_audit["selected"].astype(bool)]
    assert len(selected) == 1
    assert selected[
        ["cycle_candidate_id", "start_frame", "peak_frame", "end_frame"]
    ].iloc[0].astype(int).tolist() == [2, 5844, 5895, 5934]

    # Persisted reference_measured_raw keeps the selected Stage-5A samples;
    # the separately persisted periodic curve therefore cannot overwrite it.
    full_angles = pd.read_csv(DEFAULT_FULL_ANGLES_PATH)
    source_slice = full_angles.loc[
        full_angles["Frame"].between(5844, 5934)
    ].reset_index(drop=True)
    measured_raw = pd.read_csv(result.output_paths["measured_raw"])
    assert measured_raw["Frame"].astype(int).tolist() == source_slice[
        "Frame"
    ].astype(int).tolist()
    for column in source_slice.select_dtypes(include=[np.number]).columns:
        np.testing.assert_allclose(
            measured_raw[column].to_numpy(dtype=float),
            source_slice[column].to_numpy(dtype=float),
            atol=1e-15,
            rtol=0.0,
            equal_nan=True,
        )
    for column in source_slice.select_dtypes(exclude=[np.number]).columns:
        actual = measured_raw[column].reset_index(drop=True)
        expected = source_slice[column].reset_index(drop=True)
        np.testing.assert_array_equal(actual.isna(), expected.isna())
        populated = ~expected.isna()
        assert actual.loc[populated].astype(str).tolist() == expected.loc[
            populated
        ].astype(str).tolist()
    _assert_all_false(measured_raw["source_values_modified"])
    _assert_all_true(measured_raw["extension_source_is_measured"])
    _assert_all_false(measured_raw["measured_extension_is_reversed_flexion"])

    coverage = pd.read_csv(result.output_paths["domain_coverage"]).set_index(
        "profile"
    )
    assert np.isclose(coverage.loc["slow", "in_domain_percent"], 100.0)
    assert bool(coverage.loc["slow", "coverage_gate_passed"])
    assert np.isclose(
        coverage.loc["nominal", "in_domain_percent"],
        66.33416459,
        atol=1e-8,
        rtol=0.0,
    )
    assert not bool(coverage.loc["nominal", "coverage_gate_passed"])

    slow = pd.read_csv(result.output_paths["slow"])
    nominal = pd.read_csv(result.output_paths["nominal"])
    _assert_all_true(slow["active_reference"])
    _assert_all_true(slow["formal_execution_allowed"])
    _assert_all_true(slow["frozen_local_domain_coverage_valid"])
    _assert_all_false(nominal["active_reference"])
    _assert_all_false(nominal["formal_execution_allowed"])
    _assert_all_false(nominal["frozen_local_domain_coverage_valid"])
    assert nominal["invalid_reason"].str.contains(
        "outside_frozen_local_identification_domain", regex=False
    ).all()

    manifest = pd.read_csv(result.output_paths["manifest"])
    slow_manifest = manifest.loc[manifest["role"].eq("retimed_slow")]
    nominal_manifest = manifest.loc[manifest["role"].eq("retimed_nominal")]
    assert len(slow_manifest) == len(nominal_manifest) == 1
    _assert_all_true(slow_manifest["active_reference"])
    _assert_all_false(slow_manifest["allowed_for_first_robot_trial"])
    _assert_all_true(slow_manifest["active"])
    _assert_all_true(slow_manifest["not_used_for_robot_execution"])
    _assert_all_false(nominal_manifest["active_reference"])
    _assert_all_false(nominal_manifest["allowed_for_first_robot_trial"])
    legacy = manifest.loc[
        manifest["reference_version"].isin(
            ("reference_closed_symmetric", "reference_closed_c2")
        )
    ]
    assert len(legacy) == 3
    _assert_all_false(legacy["active_reference"])
    _assert_all_false(legacy["allowed_for_first_robot_trial"])
    _assert_all_true(legacy["legacy_software_comparison"])
    _assert_all_true(legacy["legacy"])
    _assert_all_true(legacy["not_used_for_final_personalization"])
    _assert_all_true(legacy["not_used_for_robot_execution"])

    metadata_path = result.output_paths["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["detected_natural_cycle_count"] == 4
    assert metadata["selected_cycle_candidate"]["start_frame"] == 5844
    assert metadata["selected_cycle_candidate"]["peak_frame"] == 5895
    assert metadata["selected_cycle_candidate"]["end_frame"] == 5934
    legacy = metadata["legacy_stage5a_selected_cycle"]
    assert legacy["cycle_index"] == 3
    assert [legacy["start_frame"], legacy["peak_frame"], legacy["end_frame"]] == [
        5937,
        5997,
        6040,
    ]
    assert np.isclose(legacy["pull_closure_error_mm"], 199.975638, atol=1e-6)
    assert legacy["legacy_selection_used_closure"] is False
    assert metadata["profiles"]["slow"]["active_reference"] is True
    assert metadata["profiles"]["slow"]["formal_execution_allowed"] is True
    assert metadata["profiles"]["nominal"]["active_reference"] is False
    assert metadata["profiles"]["nominal"]["formal_execution_allowed"] is False
    assert metadata["active_reference_sha256"] == _sha256(
        result.output_paths["slow"]
    )
    for key, expected_digest in metadata["generated_file_sha256"].items():
        assert _sha256(result.output_paths[key]) == expected_digest
    assert metadata["reference_measured_raw_values_modified"] is False
    assert metadata["source_reference_overwritten"] is False
    assert metadata["legacy_reference_overwritten"] is False
    assert metadata["real_robot_sdk_imported"] is False
    assert metadata["real_robot_connected"] is False
    assert metadata["robot_command_sent"] is False


def test_offline_reference_release_has_no_hardware_or_robot_import_path():
    module_paths = (
        MODULE_DIRECTORY / "run_reference_measured_asymmetric.py",
        MODULE_DIRECTORY / "reference_cycle_closure.py",
        MODULE_DIRECTORY / "reference_measured_asymmetric.py",
        MODULE_DIRECTORY / "visualize_reference_measured_asymmetric.py",
    )
    forbidden_roots = ("hardware", "control", "safety", "xCoreSDK_python")
    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == root or name.startswith(root + ".")
            for name in imported
            for root in forbidden_roots
        ), f"forbidden runtime dependency found in {path.name}: {imported}"
