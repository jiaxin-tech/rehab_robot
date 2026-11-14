"""Replay the returned empty-load evidence without a robot or invented inputs."""
import hashlib
from pathlib import Path
import socket

import pytest

from scripts.review_empty_recording import review


def test_returned_geometry_and_force_gaps_are_reproduced_offline(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    session = root / "results/manual_lab/20261114_174300/empty_full_tcp_20261114_174525"
    original = hashlib.sha256((session / "robot/robot_state.csv").read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail("closed-recording review must not use the network")
    monkeypatch.setattr(socket, "socket", forbidden)
    output = tmp_path / "review"
    result = review(session, output)
    assert result["hardware_accessed"] is False
    assert result["commanded_pose_matches_plan"] is True
    assert result["planned_excursion_m"] == pytest.approx([0., .2, .6133301422621429])
    assert result["tracking_error_mm"]["max"] == pytest.approx(1.5390462558)
    assert len(result["wrench_gaps_exceeding_limit"]) == 14
    assert result["offline_execution_audit"]["passed"] is False
    assert "wrench gap" in result["offline_execution_audit"]["error"]
    assert (output / "plan_vs_measured.png").is_file()
    assert hashlib.sha256((session / "robot/robot_state.csv").read_bytes()).hexdigest() == original
    with pytest.raises(FileExistsError):
        review(session, output)
