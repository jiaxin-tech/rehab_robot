"""Exact-cycle proposals cannot stretch time or change physical limits."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from config.bundle import ROOT, configuration
from control.taught_tcp_timing import review_fixed_cycle
from scripts import preview_taught_timing as timing
from scripts import lab_session as lab

SOURCE = ROOT / "results/manual_lab/20261115_taught_design_v1/settings.json"
CONFIG = ROOT / "config/taught_tcp_empty.template.json"


def seventy():
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    value.update(lift_duration_s=22., reach_duration_s=17., return_duration_s=25.)
    return value


def test_seventy_is_preserved_and_conflict_is_reported(tmp_path):
    original = CONFIG.read_bytes()
    output = tmp_path/"seventy"; config = tmp_path/"review.json"
    result = timing.preview(SOURCE, CONFIG, output, output_config=config)
    assert result["status"] == "OFFLINE_TIMING_REVIEW_REQUIRED"
    assert result["shape_changed"] is False and result["physical_limits_changed"] is False
    assert result["execution_review"]["planned_duration_s"] == 70
    assert set(result["execution_review"]["reasons"]) == {
        "fixed_cycle_speed_limit_conflict", "fixed_cycle_acceleration_limit_conflict"}
    assert result["execution_review"]["maximum_speed_m_s"] == pytest.approx(.12382927, rel=1e-5)
    assert result["execution_review"]["maximum_acceleration_m_s2"] == pytest.approx(.43031401, rel=1e-5)
    actual = json.loads(config.read_text())
    previous = configuration(CONFIG)
    assert actual["empty_load"] == previous["empty_load"] and actual["motion"] == previous["motion"]
    assert actual["trajectory"]["fixed_cycle_duration_s"] == 70
    assert CONFIG.read_bytes() == original
    assert all(timing.sha(output/name) == digest for name,digest in result["output_files_sha256"].items())
    problems = lab.check_config(config, empty_load=True)
    assert any("FIXED_CYCLE_LIMIT_CONFLICT" in problem for problem in problems)
    calls=[]
    with pytest.raises(ValueError, match="FIXED_CYCLE_LIMIT_CONFLICT"):
        lab.prepare(config, tmp_path/"blocked_session", empty_load=True,
                    capture_fn=lambda **kw: calls.append("hardware"))
    assert calls == []


@pytest.mark.parametrize("value",[None, True, 0, float("nan")])
def test_invalid_fixed_duration(value):
    with pytest.raises(ValueError):
        review_fixed_cycle(seventy(), value, speed_limit_m_s=.03, acceleration_limit_m_s2=.01)


def test_duration_mismatch_and_runtime_duration_cannot_be_stretched():
    report = review_fixed_cycle(seventy(), 70, speed_limit_m_s=1., acceleration_limit_m_s2=2.)
    assert report["allowed"]
    assert not review_fixed_cycle(seventy(), 71, speed_limit_m_s=1., acceleration_limit_m_s2=2.)["allowed"]
    changed = review_fixed_cycle(seventy(),70,speed_limit_m_s=1.,acceleration_limit_m_s2=2.,actual_duration_s=425.)
    assert changed["reasons"] == ["fixed_cycle_actual_duration_mismatch"]


def test_output_paths_and_cycle_mismatch_are_rejected_before_write(tmp_path):
    output=tmp_path/"mismatch"
    with pytest.raises(ValueError,match="must equal"):
        timing.preview(SOURCE, CONFIG, output, cycle_duration_s=69.)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):timing.preview(SOURCE,CONFIG,output)


def test_approved_70s_config_prepares_and_dispatches_exact_cycle_with_fake_devices(tmp_path):
    from control.start_anchor import StartAnchor, FixedTcpOrientation
    from control.taught_tcp_trajectory import TRAJECTORY_ID
    approved = ROOT / "config/taught_tcp_70s.empty.json"
    config = configuration(approved)
    expected = configuration(ROOT / "config/taught_tcp_70s.review.json")
    expected["empty_load"].update(max_tcp_speed_m_s=.13, max_tcp_acceleration_m_s2=.45)
    assert config == expected
    assert lab.check_config(approved, empty_load=True) == []
    geometry = json.loads((ROOT / config["trajectory"]["geometry_path"]).read_text(encoding="utf-8"))
    pose = tuple(geometry["recorded_anchor_pose"])
    anchor = StartAnchor(capture_host_time_s=1., tcp_pose_base=pose,
        tcp_position_base_m=pose[:3], tcp_orientation=FixedTcpOrientation(pose[3:]),
        robot_joint_positions=(0.,)*6, trajectory_id=TRAJECTORY_ID,
        reference_start_q_hip=0., reference_start_q_knee=0., anchor_id="offline-approved-test")
    directory = tmp_path / "approved_fake_session"
    def capture(**kwargs):
        assert kwargs["tcp_trajectory_id"] == TRAJECTORY_ID
        anchor.save_json(kwargs["output_path"])
    state = lab.prepare(approved, directory, empty_load=True, capture_fn=capture)
    settings = {k:v for k,v in config["trajectory"].items() if k not in ("mode", "fixed_cycle_duration_s")}
    assert state["status"] == "PREPARED" and state["blockers"] == []
    assert state["tcp_plan"]["duration_s"] == 70.
    assert state["tcp_plan"]["tcp_settings"] == settings
    safety = json.loads((directory/"tcp_safety.json").read_text())
    assert safety["max_tcp_speed_m_s"] == .13
    assert safety["max_tcp_acceleration_m_s2"] == .45
    calls=[]
    def execute(**kwargs):
        assert kwargs["taught_tcp_settings"] == settings
        assert "execution_guard" not in kwargs
        calls.append("fake_dispatch")
        kwargs["dispatch_observer"](0., 1, 2)
        return {"completed":True,"same_session_realtime_cleanup_completed":True}
    result = lab.run(directory, live=True, empty_load=True, execute_fn=execute)
    assert result["status"] == "EMPTY_LOAD_COMPLETE" and calls == ["fake_dispatch"]
