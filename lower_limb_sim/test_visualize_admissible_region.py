from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from .admissible_personalization_region import DEFAULT_REGION_DIRECTORY
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_EXPERIMENT_MANIFEST_PATH,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    PROJECT_ROOT,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from .visualize_admissible_region import (
    DEFAULT_OUTPUT_DIRECTORY,
    GIF_FILENAME,
    METADATA_FILENAME,
    PNG_FILENAMES,
    SUMMARY_FILENAME,
    VISUALIZATION_VERSION,
    generate_admissible_region_visualizations,
)


FROZEN_GENERATOR_PATH = PROJECT_ROOT / "lower_limb_sim" / "continuous_reference_neighborhood.py"
EXPECTED_ARTIFACT_FILENAMES = set(PNG_FILENAMES) | {
    GIF_FILENAME,
    SUMMARY_FILENAME,
    METADATA_FILENAME,
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    destination = tmp_path_factory.mktemp("admissible_region_visualization")
    protected = {
        "reference": _sha256(ACTIVE_REFERENCE_PATH),
        "formal_manifest": _sha256(FORMAL_EXPERIMENT_MANIFEST_PATH),
        "generator": _sha256(FROZEN_GENERATOR_PATH),
        "region_manifest": _sha256(DEFAULT_REGION_DIRECTORY / "admissible_region_manifest.json"),
        "joint_corridor": _sha256(DEFAULT_REGION_DIRECTORY / "joint_corridor_by_phase.csv"),
        "pull_corridor": _sha256(DEFAULT_REGION_DIRECTORY / "pull_corridor_by_phase.csv"),
        "parameter_map": _sha256(DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv"),
    }
    paths = generate_admissible_region_visualizations(
        destination,
        animation_frame_count=16,
        animation_writer_fps=8.0,
    )
    after = {
        "reference": _sha256(ACTIVE_REFERENCE_PATH),
        "formal_manifest": _sha256(FORMAL_EXPERIMENT_MANIFEST_PATH),
        "generator": _sha256(FROZEN_GENERATOR_PATH),
        "region_manifest": _sha256(DEFAULT_REGION_DIRECTORY / "admissible_region_manifest.json"),
        "joint_corridor": _sha256(DEFAULT_REGION_DIRECTORY / "joint_corridor_by_phase.csv"),
        "pull_corridor": _sha256(DEFAULT_REGION_DIRECTORY / "pull_corridor_by_phase.csv"),
        "parameter_map": _sha256(DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv"),
    }
    assert protected == after
    return destination, paths


def test_all_frozen_inputs_exist():
    for path in (
        ACTIVE_REFERENCE_PATH,
        FORMAL_EXPERIMENT_MANIFEST_PATH,
        FROZEN_GENERATOR_PATH,
        DEFAULT_REGION_DIRECTORY / "admissible_region_manifest.json",
        DEFAULT_REGION_DIRECTORY / "joint_corridor_by_phase.csv",
        DEFAULT_REGION_DIRECTORY / "pull_corridor_by_phase.csv",
        DEFAULT_REGION_DIRECTORY / "parameter_space_admissibility.csv",
    ):
        assert Path(path).is_file(), path


def test_all_requested_outputs_are_generated(generated):
    destination, paths = generated
    assert set(paths) == EXPECTED_ARTIFACT_FILENAMES
    assert {path.name for path in destination.iterdir()} == EXPECTED_ARTIFACT_FILENAMES
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


@pytest.mark.parametrize("filename", PNG_FILENAMES)
def test_png_is_readable_and_paper_sized(generated, filename):
    destination, _ = generated
    with Image.open(destination / filename) as image:
        width, height = image.size
        assert image.format == "PNG"
        assert width >= 1200
        assert height >= 700


def test_dynamic_gif_is_readable_and_contains_complete_cycle_frames(generated):
    destination, _ = generated
    with Image.open(destination / GIF_FILENAME) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames >= 10
        animation.seek(0)
        first = animation.convert("RGB").tobytes()
        animation.seek(animation.n_frames - 1)
        final = animation.convert("RGB").tobytes()
        assert first != final


def test_summary_is_caption_ready_and_preserves_evidence_boundary(generated):
    destination, _ = generated
    summary = (destination / SUMMARY_FILENAME).read_text(encoding="utf-8")
    for phrase in (
        "GLOBAL ROM / WORKSPACE",
        "REFERENCE-CENTERED CORRIDOR",
        "IDENTIFICATION REGION",
        "identification region != safety region",
        "identification region != ROM",
        "identification region != workspace",
        "45.857",
        "hip corridor",
        "knee corridor",
        "dynamic_rehabilitation_process.gif",
    ):
        assert phrase in summary


def test_metadata_reads_the_frozen_reference_and_region(generated):
    destination, _ = generated
    metadata = json.loads((destination / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["visualization_version"] == VISUALIZATION_VERSION
    assert metadata["parent_reference_id"] == ACTIVE_REFERENCE_ID
    assert metadata["parent_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["source_inputs"]["active_reference"]["sha256"] == ACTIVE_REFERENCE_SHA256
    assert metadata["source_region_version"] == "REFERENCE_CENTERED_ADMISSIBLE_REGION_V1"
    assert metadata["trajectory_sample_count"] == 401
    assert metadata["parameter_sample_count"] == 21025
    assert len(metadata["representative_candidates"]) == 6
    assert all(row["trajectory_admissible"] for row in metadata["representative_candidates"])


def test_metadata_artifact_checksums_match_generated_bytes(generated):
    destination, _ = generated
    metadata = json.loads((destination / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert set(metadata["artifact_sha256"]) == EXPECTED_ARTIFACT_FILENAMES - {METADATA_FILENAME}
    for filename, expected in metadata["artifact_sha256"].items():
        assert _sha256(destination / filename) == expected


def test_directional_support_and_initial_neighbors_are_truthful(generated):
    destination, _ = generated
    metadata = json.loads((destination / METADATA_FILENAME).read_text(encoding="utf-8"))
    directional = {
        row["direction"]: row for row in metadata["directional_identification_support"]
    }
    assert directional["hip -"]["supported_fraction"] == pytest.approx(1.0)
    assert directional["hip +"]["supported_fraction"] == pytest.approx(0.0)
    assert directional["hip +"]["nearest_rejected_distance"] == pytest.approx(0.25)
    assert directional["knee +"]["farthest_supported_distance"] == pytest.approx(0.25)
    assert directional["knee +"]["nearest_rejected_distance"] == pytest.approx(0.5)
    assert directional["phase -"]["supported_fraction"] == pytest.approx(1.0)
    assert directional["phase +"]["supported_fraction"] == pytest.approx(1.0)
    neighbors = {
        row["label"]: row for row in metadata["initial_trust_region_neighbors"]
    }
    assert neighbors["hip +"]["trajectory_admissible"] is False
    assert neighbors["knee +"]["trajectory_admissible"] is False
    assert neighbors["hip -"]["trajectory_admissible"] is True
    assert neighbors["knee -"]["trajectory_admissible"] is True
    assert neighbors["phase +"]["trajectory_admissible"] is True
    assert neighbors["phase -"]["trajectory_admissible"] is True


def test_rom_reference_formula_and_safety_boundary_are_unchanged(generated):
    destination, _ = generated
    metadata = json.loads((destination / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert sha256_file(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256
    boundaries = metadata["scientific_boundaries"]
    assert boundaries["identification_region_is_safety_region"] is False
    assert boundaries["identification_region_is_rom"] is False
    assert boundaries["identification_region_is_workspace"] is False
    assert boundaries["real_robot_safety_region_status"] == "NOT_DEFINED_NOT_APPROVED"
    assert boundaries["formal_sequential_personalization_rerun"] is False
    assert boundaries["reliability_threshold_resolved"] is False
    assert boundaries["robot_connection_performed"] is False
    assert boundaries["robot_command_sent"] is False


def test_visualization_module_has_no_robot_or_execution_imports():
    source_path = PROJECT_ROOT / "lower_limb_sim" / "visualize_admissible_region.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.lstrip(".").split(".")[0])
    assert imported_roots.isdisjoint({"hardware", "safety", "control", "collection"})


def test_frozen_generator_hardware_and_safety_have_no_worktree_change():
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "lower_limb_sim/continuous_reference_neighborhood.py",
            "lower_limb_sim/run_continuous_reference_neighborhood.py",
            "config/formal_experiment_manifest.json",
            "reference_release/reference_measured_asymmetric_closed_slow.csv",
            "hardware",
            "safety",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_formal_visualization_directory_is_complete_and_self_verifying():
    metadata_path = DEFAULT_OUTPUT_DIRECTORY / METADATA_FILENAME
    assert metadata_path.is_file()
    assert {path.name for path in DEFAULT_OUTPUT_DIRECTORY.iterdir()} == EXPECTED_ARTIFACT_FILENAMES
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for filename, expected in metadata["artifact_sha256"].items():
        assert _sha256(DEFAULT_OUTPUT_DIRECTORY / filename) == expected
