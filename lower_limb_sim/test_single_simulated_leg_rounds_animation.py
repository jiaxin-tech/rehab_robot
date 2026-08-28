from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .generate_single_simulated_leg_rounds_animation import (
    generate_leg_rounds_animation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_leg_round_animation_is_real_data_hd_and_deterministic(tmp_path: Path) -> None:
    first, first_metadata = generate_leg_rounds_animation(
        output_path=tmp_path / "first.gif", samples_per_round=5
    )
    second, second_metadata = generate_leg_rounds_animation(
        output_path=tmp_path / "second.gif", samples_per_round=5
    )
    assert _sha256(first) == _sha256(second)
    assert first_metadata["gif_sha256"] == second_metadata["gif_sha256"]
    assert first_metadata["case_id"] == "baseline__combined_mild"
    assert first_metadata["visualization_id"] == (
        "SUPINE_PATIENT_HIP_KNEE_FLEXION_ROUNDS_V1"
    )
    assert first_metadata["round_order"] == ["Reference", "C1", "C2", "C3"]
    assert first_metadata["round_actual_J"] == [
        1.0,
        0.965492695909,
        0.970223767367,
        0.976270020461,
    ]
    assert first_metadata["theta_shank_definition"] == "q_hip - q_knee"
    assert first_metadata["L2_is_anatomical_ankle"] is False
    assert first_metadata["truth_used_to_create_or_rank_shortlist"] is False
    assert first_metadata["robot_connected"] is False
    with Image.open(first) as image:
        assert image.size == (1280, 720)
        assert image.n_frames == 21
        assert image.info["loop"] == 0
