from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .generate_single_subject_end_to_end_animation import (
    DEFAULT_CASE_ID,
    SOURCE_ARTIFACT_DIRECTORY,
    _load_case,
    generate_single_subject_animation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_animation_uses_the_frozen_real_simulation_case() -> None:
    data = _load_case(SOURCE_ARTIFACT_DIRECTORY, DEFAULT_CASE_ID)
    assert data["subject"].case_id == "baseline__combined_mild"
    assert data["subject"].initial_identification_trial_count == 5
    assert data["shortlist"]["candidate_id"].tolist() == ["C1", "C2", "C3"]
    assert data["execution"]["actual_J"].round(6).tolist() == [
        0.965493,
        0.970224,
        0.976270,
    ]
    assert not data["shortlist"]["truth_read_before_freeze"].astype(bool).any()
    assert not data["execution"]["truth_used_for_shortlist_or_ranking"].astype(bool).any()


def test_animation_is_hd_multiframe_and_deterministic(tmp_path: Path) -> None:
    first, first_metadata = generate_single_subject_animation(
        output_path=tmp_path / "first.gif"
    )
    second, second_metadata = generate_single_subject_animation(
        output_path=tmp_path / "second.gif"
    )
    assert _sha256(first) == _sha256(second)
    assert first_metadata["gif_sha256"] == second_metadata["gif_sha256"]
    assert first_metadata["frame_count"] == 15
    assert first_metadata["truth_used_to_create_or_rank_shortlist"] is False
    assert first_metadata["held_out_final_test_read"] is False
    assert first_metadata["robot_connected"] is False
    with Image.open(first) as image:
        assert image.size == (1280, 720)
        assert image.n_frames == 15
        assert image.info["loop"] == 0
