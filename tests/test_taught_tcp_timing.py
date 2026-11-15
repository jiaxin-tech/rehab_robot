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
