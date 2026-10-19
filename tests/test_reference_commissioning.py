from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from control.reference_commissioning import (
    build_reference_commissioning_artifact,
    load_reference_commissioning_policy,
    retime_formal_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_commissioning_policy_is_narrow_and_non_executable():
    policy = load_reference_commissioning_policy(
        ROOT / "config/reference_commissioning.json"
    )
    assert policy.controller_speed_ratio_percent == 3
    assert policy.max_tcp_speed_m_s == pytest.approx(0.03)
    assert policy.cycles_per_session == 1
    assert policy.allow_missing_sdk_collision_state


def test_retime_preserves_geometry_and_caps_speed():
    source = pd.read_csv(
        ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
    )
    retimed, timing = retime_formal_reference(source, max_tcp_speed_m_s=0.03)
    geometry = ["q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"]
    np.testing.assert_array_equal(source[geometry].to_numpy(), retimed[geometry].to_numpy())
    assert timing["uniform_time_scale"] > 1.0
    assert timing["retimed_max_pull_speed_m_s"] <= 0.03 + 1e-12
    assert timing["retimed_duration_s"] > timing["source_duration_s"]


def test_artifact_never_authorizes_motion(tmp_path):
    policy = load_reference_commissioning_policy(
        ROOT / "config/reference_commissioning.json"
    )
    manifest = build_reference_commissioning_artifact(
        policy, tmp_path / "session", blockers=["timing_not_reviewed"]
    )
    assert manifest["motion_authorized"] is False
    assert manifest["status"] == "PREPARED_NOT_AUTHORIZED"
    assert manifest["remaining_motion_blockers"] == ["timing_not_reviewed"]
    saved = json.loads(
        (tmp_path / "session/commissioning_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["artifact"]["rows"] == 401


def test_policy_rejects_self_authorized_execution(tmp_path):
    source = json.loads(
        (ROOT / "config/reference_commissioning.json").read_text(encoding="utf-8")
    )
    source["execution_enabled"] = True
    source["trajectory"]["path"] = str(
        ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(PermissionError, match="must not self-authorize"):
        load_reference_commissioning_policy(path)
