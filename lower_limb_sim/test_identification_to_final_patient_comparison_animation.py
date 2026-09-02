from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .generate_identification_to_final_patient_comparison_animation import (
    _load_animation_data,
    generate_identification_to_final_comparison,
)
from .generate_single_subject_end_to_end_animation import SOURCE_ARTIFACT_DIRECTORY


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_frozen_subjects_and_identification_responses_are_real() -> None:
    data = _load_animation_data(SOURCE_ARTIFACT_DIRECTORY)
    assert [patient["subject_id"] for patient in data["patients"]] == [
        "baseline",
        "hip_stiff",
        "knee_stiff",
    ]
    assert [patient["parameters"]["k_hip_nm_per_rad"] for patient in data["patients"]] == [
        14.99999999980913,
        30.0,
        15.0,
    ]
    assert [patient["parameters"]["k_knee_nm_per_rad"] for patient in data["patients"]] == [
        12.000000000127748,
        11.999999999999893,
        30.0,
    ]
    for patient in data["patients"]:
        assert len(patient["excitation_1"]) == 401
        assert len(patient["excitation_2"]) == 401
        assert patient["shortlist"]["candidate_id"].tolist() == ["C1", "C2", "C3"]
        assert not patient["shortlist"]["truth_read_before_freeze"].astype(bool).any()


def test_comparison_gif_is_hd_multiframe_and_deterministic(tmp_path: Path) -> None:
    first, first_metadata = generate_identification_to_final_comparison(
        output_path=tmp_path / "first.gif", samples_per_motion=3
    )
    second, second_metadata = generate_identification_to_final_comparison(
        output_path=tmp_path / "second.gif", samples_per_motion=3
    )
    assert _sha256(first) == _sha256(second)
    assert first_metadata["gif_sha256"] == second_metadata["gif_sha256"]
    assert first_metadata["visualization_id"] == (
        "NORMAL_VS_STIFFNESS_IDENTIFICATION_TO_FINAL_V1"
    )
    assert first_metadata["frame_count"] == 19
    assert first_metadata["theta_shank_definition"] == "q_hip - q_knee"
    assert first_metadata["truth_used_to_create_or_rank_shortlist"] is False
    assert first_metadata["held_out_final_test_read"] is False
    assert first_metadata["robot_connected"] is False
    with Image.open(first) as image:
        assert image.size == (1600, 900)
        assert image.n_frames == 19
        assert image.info["loop"] == 0
