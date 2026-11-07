from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.restore_session_tcp import (
    MAX_CAPTURE_TO_PATH_M,
    MAX_SEGMENT_M,
    RESTORE_SPEED_RATIO,
    _build_direct_waypoints,
    _build_reverse_waypoints,
)


def _preview() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tcp_x_base": np.linspace(0.7, 0.3, 41),
            "tcp_y_base": np.zeros(41),
            "tcp_z_base": 0.5 + 0.05 * np.sin(np.linspace(0.0, np.pi, 41)),
        }
    )


def test_reverse_waypoints_follow_frozen_path_to_source_start():
    preview = _preview()
    target = (0.7, 0.0, 0.5, 3.1, -0.06, 0.72)
    current = (0.3005, 0.0, 0.5, 3.101, -0.06, 0.72)

    waypoints, summary = _build_reverse_waypoints(current, preview, target)

    poses = waypoints[
        ["tcp_x_base", "tcp_y_base", "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz"]
    ].to_numpy(float)
    assert np.allclose(poses[0], current)
    assert np.allclose(poses[-1, :3], target[:3])
    assert np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).max() <= MAX_SEGMENT_M + 1e-12
    assert summary["capture_to_source_path_distance_m"] < MAX_CAPTURE_TO_PATH_M
    assert summary["path_length_m"] > summary["straight_line_distance_m"]


def test_reverse_waypoints_reject_current_pose_off_previous_path():
    with pytest.raises(PermissionError, match="not close enough"):
        _build_reverse_waypoints(
            (0.3, 0.1, 0.5, 3.1, -0.06, 0.72),
            _preview(),
            (0.7, 0.0, 0.5, 3.1, -0.06, 0.72),
        )


def test_reverse_waypoints_reject_orientation_change():
    with pytest.raises(PermissionError, match="orientation differs"):
        _build_reverse_waypoints(
            (0.3, 0.0, 0.5, 3.2, -0.06, 0.72),
            _preview(),
            (0.7, 0.0, 0.5, 3.1, -0.06, 0.72),
        )


def test_direct_waypoints_return_from_workspace_position_in_bounded_segments():
    target = (0.7, 0.0, 0.5, 3.1, -0.06, 0.72)
    current = (0.25, 0.2, 0.8, 3.101, -0.06, 0.72)

    waypoints, summary = _build_direct_waypoints(current, target)

    poses = waypoints[
        ["tcp_x_base", "tcp_y_base", "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz"]
    ].to_numpy(float)
    assert np.allclose(poses[0], current)
    assert np.allclose(poses[-1, :3], target[:3])
    assert np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).max() <= MAX_SEGMENT_M + 1e-12
    assert summary["route_mode"] == "direct_reviewed_workspace"
    assert RESTORE_SPEED_RATIO == 10


def test_direct_waypoints_reject_unreviewed_orientation_change():
    with pytest.raises(PermissionError, match="orientation differs"):
        _build_direct_waypoints(
            (0.25, 0.2, 0.8, 2.5, -0.06, 0.72),
            (0.7, 0.0, 0.5, 3.1, -0.06, 0.72),
        )
