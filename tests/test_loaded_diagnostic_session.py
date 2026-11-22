"""Protocol integration and passive-leg evidence; entirely synthetic hardware."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from collection.twincat_leg import atomic_json
from config.bundle import ROOT, configuration
from measurement_validation.loaded_diagnostic import audit_loaded_leg
from scripts import lab_session as lab


CONFIG = ROOT / "config/continuous_taught_90s.loaded_diagnostic.json"
SOURCE = ROOT / "results/manual_lab/20261122_143700/continuous_empty_01"


def test_actual_rigid_setup_cannot_reach_capture_or_motion(tmp_path):
    def forbidden(**kw):
        pytest.fail("rigid setup reached a hardware call")
    assert any("rigid shank" in reason for reason in lab.check_config(CONFIG))
    with pytest.raises(ValueError, match="rigid shank"):
        lab.prepare(CONFIG, tmp_path / "blocked", capture_fn=forbidden)
    c = configuration(CONFIG)
    state = {"config_path": str(CONFIG), "empty_load": False, "trajectory_mode": "continuous_taught_tcp"}
    with pytest.raises(ValueError, match="rigid shank"):
        lab._run_tcp(tmp_path, state, empty_load=False, capture_factory=forbidden, execute_fn=forbidden)
    assert not (tmp_path / "blocked/anchor.json").exists()
    assert c["loaded_trial"]["reviewed"] is False


def synthetic_review(tmp_path):
    """Only a test of a hypothetical rotationally free fixture, not the lab rig."""
    c = configuration(CONFIG)
    c["acquisition"].update(pre_roll_s=.00001, post_roll_s=.00001)
    anchor = json.loads((SOURCE / "anchor.json").read_text(encoding="utf-8"))
    c["loaded_trial"].update(setup_id="SYNTHETIC_TEST_ONLY", connection_type="passive_joint",
        tool_description="Synthetic swivel", support_description="Synthetic fixed hip support",
        review_notes="Synthetic fixture; no actual laboratory review", reviewed=True,
        manual_path_checked=True, reviewed_anchor_pose_base=anchor["tcp_pose_base"],
        reviewed_at="2026-11-22T15:00:00+08:00")
    path = tmp_path / "synthetic.json"
    atomic_json(path, c)
    return c, path, anchor


def test_generic_diagnostic_passes_strict_flag_and_never_trains(tmp_path, monkeypatch):
    c, path, anchor = synthetic_review(tmp_path)
    monkeypatch.setattr(lab, "_check_config_for_operation", lambda *a, **k: [])
    monkeypatch.setattr(lab, "preview_trajectory", lambda **kw: None)
    real_train = lab.train_session
    monkeypatch.setattr(lab, "train_session", lambda *a, **k: pytest.fail("diagnostic trained a policy"))
    def capture(**kw):
        atomic_json(kw["output_path"], anchor)
    def origin_capture(config, output):
        origin = {"usable_for_motion": True, "axes": {n: {"physical_angle_deg": 0.} for n in ("hip", "knee")}}
        atomic_json(output, origin)
        return origin
    folder = tmp_path / "session"
    prepared = lab.prepare(path, folder, capture_fn=capture, leg_origin_capture_fn=origin_capture)
    assert prepared["learning_mode"] == "loaded_diagnostic"
    assert prepared["training_status"] == "NOT_APPLICABLE_DIAGNOSTIC"
    assert prepared["loaded_trial_binding"]["status"] == "BOUND"
    assert "original_training_binding" not in prepared and "force_training_binding" not in prepared
    class Leg:
        def __init__(self, *a): pass
        def start(self): pass
        def stop(self): pass
        def health_reason(self): return None
        def snapshot(self):
            row = {}
            for name, axis in (("hip", 1), ("knee", 4)):
                row.update({f"axis{axis}_fault": False, f"axis{axis}_error_code_hex": "0x0000",
                    f"axis{axis}_operation_enabled": False,
                    f"axis{axis}_position_raw": c["calibration"][name]["zero_counts"]})
            return {"last_sample_ns": time.perf_counter_ns(), "latest": row}
    def execute(**kw):
        assert kw["strict_wrench_freshness"] is True
        assert kw["continuous_taught_settings"]["cycle_duration_s"] == 90
        assert kw["execution_guard"]() is None
        kw["dispatch_observer"](0., 100, 101)
        return {"completed": True, "same_session_realtime_cleanup_completed": True}
    calls = []
    def analyzed(directory, state, config):
        calls.append(directory)
        assert state["learning_mode"] == "loaded_diagnostic"
    monkeypatch.setattr(lab, "_analyze_loaded_diagnostic", analyzed)
    result = lab.run(folder, live=True, capture_factory=Leg, execute_fn=execute)
    assert result["status"] == "CAPTURE_COMPLETE" and calls == [folder]
    with pytest.raises(ValueError, match="does not train"):
        real_train(folder)
    with pytest.raises(ValueError, match="exact empty-load repeats only"):
        lab.repeat(folder, tmp_path / "repeat")


def test_invalid_protocol_cannot_use_empty_route(tmp_path):
    c = configuration(CONFIG)
    c["loaded_trial"]["protocol"] = "unknown"
    path = tmp_path / "invalid.json"
    atomic_json(path, c)
    problems = lab.check_config(path, empty_load=True)
    assert any("exact passive diagnostic protocol" in reason for reason in problems)
    with pytest.raises(ValueError, match="exact passive diagnostic protocol"):
        lab.prepare(path, tmp_path / "invalid", empty_load=True,
                    capture_fn=lambda **k: pytest.fail("invalid protocol reached hardware"))


@pytest.fixture
def leg_record(tmp_path):
    c = configuration(CONFIG)
    clock = np.arange(99.99, 100.212, .002)
    ns = np.rint(clock * 1e9).astype(np.int64)
    frame = pd.DataFrame({"host_monotonic_ns": ns, "query_start_ns": ns - 200000,
                          "query_end_ns": ns, "receive_elapsed_ms": (clock-clock[0])*1000,
                          "sync_quality": "atomic_single_process_image"})
    for name, axis in (("hip", 1), ("knee", 4)):
        for key, value in {"position_raw": c["calibration"][name]["zero_counts"],
            "torque_3c1a_nm": 1., "statusword_hex": "0x0231", "error_code_hex": "0x0000",
            "fault": False, "operation_enabled": False}.items():
            frame[f"axis{axis}_{key}"] = value
    directory = tmp_path / "leg"
    directory.mkdir()
    path = directory / "leg.csv"
    frame.to_csv(path, index=False)
    return tmp_path, c, frame, path, {"schedule_origin_host_time_s": 100., "duration_s": .2}


def test_entire_leg_motion_has_independent_raw_evidence(leg_record):
    folder, c, _, _, robot = leg_record
    result = audit_loaded_leg(folder, c, robot)
    assert result["motion_rows"] >= 99
    assert result["max_motion_receive_gap_s"] < .003
    assert result["quality"]["usable_for_dynamics"] is True
    assert result["training_authorized"] is False and result["comfort_claim_authorized"] is False
    assert result["interpolated_samples"] == 0


@pytest.mark.parametrize("damage", ["gap", "late_start", "early_end", "fault", "error", "enabled",
    "status_fault", "status_enabled", "range", "torque", "missing", "query", "query_clock"])
def test_bad_leg_record_cannot_be_completed(leg_record, damage):
    folder, c, frame, path, robot = leg_record
    if damage == "gap": frame = frame.drop(index=range(30, 70))
    elif damage == "late_start": frame = frame.iloc[15:]
    elif damage == "early_end": frame = frame.iloc[:-15]
    elif damage == "fault": frame.loc[50, "axis1_fault"] = True
    elif damage == "error": frame.loc[50, "axis4_error_code_hex"] = "0x0022"
    elif damage == "enabled": frame.loc[50, "axis1_operation_enabled"] = True
    elif damage == "status_fault": frame.loc[50, "axis1_statusword_hex"] = "0x0239"
    elif damage == "status_enabled": frame.loc[50, "axis4_statusword_hex"] = "0x0237"
    elif damage == "range": frame.loc[50, "axis1_position_raw"] = 1e8
    elif damage == "torque": frame.loc[50, "axis4_torque_3c1a_nm"] = 101.
    elif damage == "missing": frame = frame.drop(columns="axis1_fault")
    elif damage == "query": frame.loc[50, "query_start_ns"] -= 100000000
    elif damage == "query_clock": frame.loc[50, "query_end_ns"] -= 100000
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, KeyError)):
        audit_loaded_leg(folder, c, robot)


def test_full_synthetic_production_shaped_session_completes_without_learning(tmp_path, monkeypatch):
    """Exercise real robot-audit -> ADS-audit -> completion contracts, no device."""
    c, path, anchor = synthetic_review(tmp_path)
    monkeypatch.setattr(lab, "_check_config_for_operation", lambda *a, **k: [])
    monkeypatch.setattr(lab, "preview_trajectory", lambda **kw: None)
    def capture(**kw): atomic_json(kw["output_path"], anchor)
    def origin_capture(config, output):
        origin = {"usable_for_motion": True, "axes": {n: {"physical_angle_deg": 0.} for n in ("hip", "knee")}}
        atomic_json(output, origin)
        return origin
    folder = tmp_path / "synthetic_complete"
    state = lab.prepare(path, folder, capture_fn=capture, leg_origin_capture_fn=origin_capture)
    table = pd.read_csv(folder / "tcp_trajectory.csv")
    clock = table.time_s.to_numpy(float) + 100.
    pd.DataFrame({"trajectory_time_s": table.time_s,
        "dispatch_start_ns": np.rint(clock*1e9).astype(np.int64),
        "dispatch_end_ns": np.rint(clock*1e9).astype(np.int64)+1000}).to_csv(folder / "dispatch.csv", index=False)
    robot_dir = folder / "robot"
    robot_dir.mkdir()
    commands = pd.DataFrame({"trajectory_time_s": table.time_s, "command_valid": True})
    for axis in "xyz":
        commands[f"tcp_target_{axis}"] = table[f"tcp_{axis}_base"]
        commands[f"tcp_target_r{axis}"] = table[f"tcp_r{axis}"]
    commands.to_csv(robot_dir / "trajectory_command.csv", index=False)
    robot_state = pd.DataFrame({"host_time_s": clock, "valid": True})
    for axis in "xyz":
        robot_state[f"tcp_{axis}"] = table[f"tcp_{axis}_base"]
        robot_state[f"tcp_r{axis}"] = table[f"tcp_r{axis}"]
    robot_state.to_csv(robot_dir / "robot_state.csv", index=False)
    wrench = pd.DataFrame({"query_start_s": clock-.001, "query_end_s": clock+.001,
                           "valid": True, "frame_type": "world"})
    for key in ("fx", "fy", "fz", "tx", "ty", "tz"): wrench[key] = 1.
    wrench.to_csv(robot_dir / "robot_wrench.csv", index=False)
    execution = {"completed": True, "same_session_realtime_cleanup_completed": True,
                 "commands_dispatched": len(table), "schedule_origin_host_time_s": 100.}
    atomic_json(robot_dir / "metadata.json", {"logger": {"status": "completed"},
        "execution_result": execution, "strict_wrench_freshness": True})
    state.update(status="CAPTURE_COMPLETE", execution=execution)
    lab.report(folder, state)
    # Production collector publishes both relative elapsed and absolute clocks.
    leg_clock = np.arange(99.99, 190.012, .002)
    ns = np.rint(leg_clock*1e9).astype(np.int64)
    frame = pd.DataFrame({"host_monotonic_ns": ns, "query_start_ns": ns-200000,
        "query_end_ns": ns, "receive_elapsed_ms": (leg_clock-leg_clock[0])*1000,
        "sync_quality": "atomic_single_process_image"})
    for name, axis in (("hip", 1), ("knee", 4)):
        for key, value in {"position_raw": c["calibration"][name]["zero_counts"],
            "torque_3c1a_nm": 1., "statusword_hex": "0x0231", "error_code_hex": "0x0000",
            "fault": False, "operation_enabled": False}.items(): frame[f"axis{axis}_{key}"] = value
    (folder / "leg").mkdir()
    frame.to_csv(folder / "leg/leg.csv", index=False)
    result = lab.complete(folder)
    assert result["status"] == "COMPLETE"
    assert result["completion"]["training_authorized"] is False
    assert result["completion_trajectory_audit"]["duration_s"] == 90
    assert result["loaded_leg_audit"]["duration_s"] == 90
    assert result["passive_response"]["mode"] == "continuous_taught_tcp"
    assert result["training_status"] == "NOT_APPLICABLE_DIAGNOSTIC"
    assert not (folder / "original_training.json").exists()
    assert not (folder / "tcp_force_training.json").exists()
    metadata = json.loads((robot_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["execution_result"].pop("schedule_origin_host_time_s")
    atomic_json(robot_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="recorded finite schedule origin"):
        lab.complete(folder)
    metadata["execution_result"]["schedule_origin_host_time_s"] = 100.
    metadata["strict_wrench_freshness"] = False
    atomic_json(robot_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="recorded strict wrench freshness"):
        lab.complete(folder)
