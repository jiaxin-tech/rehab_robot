"""Real identification interface must never fabricate missing evidence."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.identify_real_episode import identify_real_episode


def _empty_episode(path):
    path.mkdir()
    (path / "metadata.json").write_text("{}\n", encoding="utf-8")
    pd.DataFrame(
        columns=[
            "host_time_s", "tcp_x", "tcp_y", "tcp_z", "valid", "invalid_reason"
        ]
    ).to_csv(path / "robot_state.csv", index=False)
    pd.DataFrame(
        columns=[
            "query_start_s", "query_end_s", "fx", "fy", "fz", "frame_type", "valid"
        ]
    ).to_csv(path / "robot_wrench.csv", index=False)


def test_missing_reviewed_config_creates_no_fake_outputs(tmp_path):
    episode = tmp_path / "episode"
    _empty_episode(episode)
    with pytest.raises(FileNotFoundError, match="reviewed real identification config"):
        identify_real_episode(episode)
    assert not (episode / "identified_parameters.json").exists()
    assert not (episode / "prediction_metrics.csv").exists()


def test_unreviewed_template_is_rejected_before_reading_empty_data(tmp_path):
    episode = tmp_path / "episode"
    _empty_episode(episode)
    template = {
        "schema_version": 1,
        "reviewed": False,
        "raw_wrench_frame": None,
        "R_rehab_from_raw_wrench": None,
        "force_sign_robot_on_leg": None,
        "assumed_wrench_delay_s": None,
        "baseline_subject_template": {
            "mass_thigh_kg": None,
            "mass_shank_kg": None,
            "com_thigh_m": None,
            "com_shank_m": None,
            "inertia_thigh_kg_m2": None,
            "inertia_shank_kg_m2": None,
            "q0_hip_rad": None,
            "q0_knee_rad": None,
            "gravity_m_s2": None,
        },
        "notes": "draft",
    }
    config = episode / "identification_config.json"
    config.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(PermissionError, match="not reviewed"):
        identify_real_episode(episode)
    assert not (episode / "identified_parameters.json").exists()
    assert not (episode / "prediction_metrics.csv").exists()
