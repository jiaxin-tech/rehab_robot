"""Offline integration: synthetic hardware data, real validation and frozen learner."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from collection.twincat_leg import (AdsInputImage, AmsAddr, AmsNetId, capture, decode_image,
    LegCaptureProcess, validate_config, atomic_json)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from measurement_validation.robot_leg_round import aligned_measurement, endpoints, validate_analysis
from safety.robot_trial_release import read_json, sha256_file, EVIDENCE_KINDS, LOADED_EVIDENCE
import scripts.run_robot_leg_round as rounds

ROOT = Path(__file__).resolve().parents[1]


def portable_test_config():
    """Relocate checkout files only, always retaining frozen expected hashes."""
    c = rounds.paths(ROOT / "config/robot_leg_round.json")
    if os.environ.get("REHAB_TEST_ROUND_CONFIG"):
        portable = rounds.paths(os.environ["REHAB_TEST_ROUND_CONFIG"])
        assert portable["algorithm_sha256"] == c["algorithm_sha256"]
        assert sorted(portable["algorithm_dependencies"].values()) == sorted(c["algorithm_dependencies"].values())
        c = portable
    def relocate(name):
        parts = name.replace("\\", "/").split("/rehab_robot/", 1)
        return str(ROOT / parts[1]) if len(parts) == 2 else name
    c["algorithm_dependencies"] = {relocate(p): h for p, h in c["algorithm_dependencies"].items()}
    for item in c["candidates"]:
        item["path"] = relocate(item["path"])
    return c


def capture_config():
    c = read_json(ROOT / "config/twincat_leg_capture.json")
    c.update(mapping_reviewed=True, ads_dll_path=__file__, ads_timeout_ms=100,
        startup_timeout_s=.2, max_progress_age_s=.2, shutdown_timeout_s=.2,
        heartbeat_s=.005, no_new_sample_timeout_s=.02, poll_interval_s=.0001)
    c["guards"] = {f"axis{axis}": dict(max_abs_torque_nm=50., expected_operation_enabled=False) for axis in (1, 4)}
    return c


def image(toggle=0, status=0, torque=2.):
    data = bytearray(1366)
    for axis, offset in enumerate((0, 23, 46, 69), 1):
        struct.pack_into("<HiihiHbf", data, offset, status, -1000*axis, -20, -12, 9, 0, 8, torque)
    data[1365] = toggle << 4
    return data


def analysis_config():
    return dict(reviewed=True, alignment_reviewed=True, sensor_receive_delay_s=0.,
        command_application_delay_s=0., max_gap_s=.01, max_read_duration_s=.01,
        max_dispatch_time_error_s=.02, derivative_rate_hz=500., derivative_window_s=.1,
        max_start_angle_error_deg=1., max_tracking_error_deg=2.)


def calibration_config():
    return dict(reviewed=True, independent_validation_passed=True,
        joints={name: dict(counts_per_degree=1000., zero_counts=0., flexion_sign=1,
            safe_min_deg=-10., safe_max_deg=140., torque_sign_to_inverse_dynamics=1) for name in ("hip", "knee")})


def measured_fixture(directory, reference=DEFAULT_REFERENCE_PATH):
    """Explicit synthetic data for software tests only; no hardware is contacted."""
    ref = pd.read_csv(reference)
    t = np.arange(-.1, float(ref.time_s.iloc[-1])+.102, .002)
    origin = 123456789000000
    ticks = origin + np.rint(t*1e9).astype(np.int64)
    data = dict(sample_index=np.arange(len(t)), receive_elapsed_ms=(t-t[0])*1000,
        host_monotonic_ns=ticks, query_start_ns=ticks-100000, query_end_ns=ticks)
    for name, axis in (("hip", 1), ("knee", 4)):
        q = np.interp(t, ref.time_s, ref[f"q_{name}_rad"])
        data.update({f"axis{axis}_position_raw": np.rint(np.rad2deg(q)*1000),
            f"axis{axis}_torque_3c1a_nm": 3.+axis+np.sin(t), f"axis{axis}_statusword_hex": "0x0000",
            f"axis{axis}_error_code_hex": "0x0000", f"axis{axis}_fault": False,
            f"axis{axis}_operation_enabled": False})
    csv = directory / "leg.csv"
    pd.DataFrame(data).to_csv(csv, index=False)
    dispatch = directory / "dispatch.csv"
    d = origin + np.rint(ref.time_s.to_numpy()*1e9).astype(np.int64)
    pd.DataFrame(dict(trajectory_time_s=ref.time_s, dispatch_start_ns=d-100000, dispatch_end_ns=d)).to_csv(dispatch, index=False)
    return csv, dispatch


def training_request(directory, measured):
    c = portable_test_config()
    return dict(algorithm_root=c["algorithm_root"], algorithm_sha256=c["algorithm_sha256"],
        algorithm_dependencies=c["algorithm_dependencies"], geometry_m=c["geometry_m"],
        candidates=c["candidates"], reference_candidate_id=c["reference_candidate_id"],
        setup_id="SYNTHETIC_SOFTWARE_TEST_ONLY", setup_sha256="0"*64,
        reference_sha256=sha256_file(DEFAULT_REFERENCE_PATH), history=[dict(episode_id="synthetic-test",
            candidate_id=c["reference_candidate_id"], measured_path=str(measured),
            measured_sha256=sha256_file(measured), endpoints=dict(E3=1., E2=1., peak_ratio=1.))])


def test_decoder_axis_offsets_raw_signedness_and_native_structs():
    c = capture_config()
    row = decode_image(image(1), c)
    assert ctypes.sizeof(AmsNetId) == 6 and ctypes.sizeof(AmsAddr) == 8
    assert row["axis4_position_raw"] == -4000
    assert row["axis1_torque_6077_raw_bits"] == 65524
    assert row["axis1_torque_6077_pct_rated"] is None
    assert row["axis4_torque_3c1a_nm"] == 2. and row["input_toggle"] == 1
    c["torque_6077_signedness_verified"] = True
    assert decode_image(image(), c)["axis1_torque_6077_pct_rated"] == pytest.approx(-1.2)
    with pytest.raises(ValueError, match="incomplete"):
        decode_image(image()[:-1], c)


def test_polling_cannot_exceed_source_cycle():
    c = capture_config()
    c["poll_interval_s"] = .002
    with pytest.raises(ValueError, match="shorter than"):
        validate_config(c)


def test_capture_ready_and_completed_only_after_close(tmp_path):
    stop = threading.Event()
    stop.wait = lambda *a: pytest.fail("Event.wait is too coarse for Windows 2 ms capture")
    messages = []
    class Source:
        closed = False
        n = 0
        def __init__(self, config): pass
        def read(self):
            self.n += 1
            return image(self.n % 2)
        def close(self): self.closed = True
    source = Source(None)
    def notify(record):
        messages.append(record)
        if record["state"] == "recording":
            assert not source.closed
            stop.set()
        else:
            assert source.closed
            assert len(pd.read_csv(tmp_path / "leg.csv")) == 2
    assert capture(capture_config(), tmp_path / "leg.csv", stop_event=stop,
        notify=notify, source_factory=lambda c: source) == 2
    assert [m["state"] for m in messages] == ["recording", "completed"]


@pytest.mark.parametrize("kind", ["fault", "torque", "stuck", "short"])
def test_capture_failure_never_acknowledges_complete(tmp_path, kind):
    closed, messages = [], []
    class Source:
        def __init__(self, config): pass
        def read(self):
            if kind == "short": return b""
            return image(status=8 if kind == "fault" else 0, torque=51. if kind == "torque" else 2.)
        def close(self): closed.append(True)
    with pytest.raises((RuntimeError, TimeoutError, ValueError)):
        capture(capture_config(), tmp_path / "leg.csv", stop_event=threading.Event(),
            notify=messages.append, source_factory=Source)
    assert closed and not any(m["state"] == "completed" for m in messages)


def test_supervisor_rejects_dead_stale_and_bad_protocol(tmp_path):
    p = tmp_path / "capture.json"
    atomic_json(p, capture_config())
    supervisor = LegCaptureProcess(p, tmp_path)
    assert supervisor.health_reason() == "leg_capture_process_not_running"
    supervisor.process = SimpleNamespace(poll=lambda: None)
    supervisor.status = dict(state="recording", last_sample_ns=time.perf_counter_ns())
    assert supervisor.health_reason() is None
    snapshot = supervisor.snapshot()
    snapshot["state"] = "completed"
    assert supervisor.snapshot()["state"] == "recording"
    supervisor.status["last_sample_ns"] = time.perf_counter_ns()-1_000_000_000
    assert supervisor.health_reason() == "leg_capture_stale"
    supervisor._error = "bad json"
    assert "protocol_error" in supervisor.health_reason()


@pytest.mark.parametrize("code, length", [(0, 1365), (1797, 1366)])
def test_native_ads_short_or_failed_read_is_rejected(code, length):
    reader = AdsInputImage.__new__(AdsInputImage)
    reader.config, reader.port, reader.address = capture_config(), 1, AmsAddr()
    def read(port, address, group, offset, size, buffer, actual):
        ctypes.cast(actual, ctypes.POINTER(ctypes.c_uint32))[0] = length
        return code
    reader.dll = SimpleNamespace(AdsSyncReadReqEx2=read)
    with pytest.raises(OSError, match="ADS input read failed"):
        reader.read()


@pytest.mark.parametrize("hung", [True, False])
def test_supervisor_requires_close_ack_and_bounds_shutdown(tmp_path, hung):
    config = tmp_path / "capture.json"
    atomic_json(config, capture_config())
    supervisor = LegCaptureProcess(config, tmp_path)
    calls = []
    class Process:
        returncode = None if hung else 0
        stdin = SimpleNamespace(write=lambda x: None, flush=lambda: None)
        def poll(self): return self.returncode
        def wait(self, timeout):
            calls.append(timeout)
            if self.returncode is None: raise subprocess.TimeoutExpired("test", timeout)
        def kill(self): self.returncode = -1
    supervisor.process = Process()
    supervisor._reader = SimpleNamespace(join=lambda timeout: None)
    supervisor._stderr = SimpleNamespace(close=lambda: calls.append("closed"))
    supervisor.status = dict(state="recording")
    with pytest.raises(TimeoutError if hung else RuntimeError): supervisor.stop()
    assert calls[-1] == "closed"
    if hung: assert calls[:-1] == [.2, 2]


def test_measured_alignment_and_reference_normalization(tmp_path):
    csv, dispatch = measured_fixture(tmp_path)
    measured, metrics = aligned_measurement(csv, dispatch, DEFAULT_REFERENCE_PATH, analysis_config(), calibration_config())
    ref = pd.read_csv(DEFAULT_REFERENCE_PATH)
    assert np.max(np.abs(measured.q_hip_rad-ref.q_hip_rad)) < .001
    assert np.isfinite(measured.select_dtypes(include="number")).all().all()
    assert endpoints(metrics, metrics) == dict(E3=1., E2=1., peak_ratio=1.)
    assert not metrics["automatically_tared"]


@pytest.mark.parametrize("kind", ["gap", "duplicate", "fault", "missing", "tracking", "dispatch", "short_read"])
def test_invalid_data_cannot_enter_training(tmp_path, kind):
    csv, dispatch = measured_fixture(tmp_path)
    f = pd.read_csv(csv)
    if kind == "gap": f = f.drop(index=range(200, 230))
    if kind == "duplicate": f.loc[200, "host_monotonic_ns"] = f.loc[199, "host_monotonic_ns"]
    if kind == "fault": f.loc[200, "axis4_fault"] = True
    if kind == "missing": f.loc[200, "axis4_torque_3c1a_nm"] = np.nan
    if kind == "tracking": f["axis4_position_raw"] += 10000
    if kind == "short_read": f.loc[200, "query_start_ns"] -= 100_000_000
    if kind == "dispatch":
        d = pd.read_csv(dispatch).iloc[:-1]
        d.to_csv(dispatch, index=False)
    f.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        aligned_measurement(csv, dispatch, DEFAULT_REFERENCE_PATH, analysis_config(), calibration_config())


def test_missing_knee_is_not_substituted():
    c = calibration_config()
    c["joints"]["knee"] = None
    with pytest.raises(ValueError, match="missing knee"):
        validate_analysis(analysis_config(), c)


def test_real_frozen_training_worker(tmp_path):
    csv, dispatch = measured_fixture(tmp_path)
    measured, _ = aligned_measurement(csv, dispatch, DEFAULT_REFERENCE_PATH, analysis_config(), calibration_config())
    target = tmp_path / "measured.csv"
    measured.to_csv(target, index=False)
    request = tmp_path / "request.json"
    atomic_json(request, training_request(tmp_path, target))
    output = tmp_path / "update.json"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round", str(request),
        "--output", str(output)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    update = read_json(output)
    assert update["method"] == "FROZEN_RESIDUAL_GP_MEAN_GREEDY"
    assert len(update["theta"]) == 6
    assert update["action"]["role"] == "STOP"
    assert update["action"]["stop_reason"] == "DOMAIN_EXHAUSTED"


def test_unreviewed_setup_blocks_before_hardware(tmp_path):
    def forbidden(**kwargs): pytest.fail("must not connect")
    result = rounds.prepare(ROOT / "config/robot_leg_round.json", tmp_path / "round", capture_fn=forbidden)
    assert result["status"] == "BLOCKED_BEFORE_CONNECTION"
    assert result["blockers"] and (tmp_path / "round/report.md").is_file()


def test_malformed_setup_still_produces_report(tmp_path):
    p = tmp_path / "bad.json"
    atomic_json(p, {})
    result = rounds.prepare(p, tmp_path / "round")
    assert result["status"] == "BLOCKED_BEFORE_CONNECTION"


def test_round_attempt_is_exclusive_and_persistent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rounds, "_run_once", lambda *a, **k: calls.append(True))
    rounds.run(tmp_path, confirmation=rounds.CONFIRMATION)
    with pytest.raises(FileExistsError): rounds.run(tmp_path, confirmation=rounds.CONFIRMATION)
    assert calls == [True]
    with pytest.raises(PermissionError): rounds.run(tmp_path, confirmation="no")


def prepared_round(tmp_path, *, empty=False, capture_only=False):
    from tests.test_run_rehab_experiment import _reviewed_files
    from tests.test_robot_trial_release import _wrench_config
    anchor, frame, safety = _reviewed_files(tmp_path)
    wrench = _wrench_config(tmp_path / "wrench.json", allowed_operation_states=["idle", "rtcontrolling"])
    configs = {}
    for key, value in (("leg_config", capture_config()), ("calibration", calibration_config()), ("analysis", analysis_config())):
        p = tmp_path / (key+".json")
        atomic_json(p, value)
        configs[key] = str(p)
    c = portable_test_config()
    c.update(configs, reviewed=True, setup_id="SYNTHETIC_TEST_ONLY", reviewer="test", operator="test",
        safety_config=str(safety), frame_config=str(frame), wrench_config=str(wrench),
        robot_ip="192.0.2.1", local_ip="192.0.2.2", robot_class="xMateRobot",
        tool_name="rehab_cuff", workpiece_name="rehab_fixture", wrench_hz=50.,
        collision_event_max_age_s=.5, expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        pre_roll_s=.001, post_roll_s=.001, max_leg_start_age_s=.5, realtime_preparation="same_session",
        robot_identity=dict(robot_model="xMate6-Pro", robot_serial_number="SN-RUNNER",
            controller_version="controller-test", sdk_version="0.7.0"), site_evidence={})
    if empty and capture_only:
        raise ValueError("test fixture cannot be both empty and loaded capture-only")
    c["trial_kind"] = ("empty_load" if empty else
        "dummy_leg_capture_only" if capture_only else "dummy_leg_reference")
    for kind in (*EVIDENCE_KINDS, *(() if empty else LOADED_EVIDENCE)):
        p = tmp_path / (kind+".txt")
        p.write_text("SYNTHETIC TEST EVIDENCE ONLY", encoding="ascii")
        c["site_evidence"][kind] = dict(passed=True, path=str(p), sha256=sha256_file(p))
    p = tmp_path / "round.json"
    atomic_json(p, c)
    assert rounds.check_config(p) == []
    def capture_anchor(**kwargs):
        Path(kwargs["output_path"]).write_bytes(anchor.read_bytes())
    directory = tmp_path / "session"
    assert rounds.prepare(p, directory, capture_fn=capture_anchor)["status"] == "AWAITING_ONSITE_CONFIRMATION"
    return directory


class SyntheticCapture:
    """Test-only source; never exposed by the production CLI."""
    def __init__(self, config, directory): self.directory = directory
    def start(self):
        self.directory.mkdir()
        measured_fixture(self.directory)
    def health_reason(self): return None
    def snapshot(self):
        ref = pd.read_csv(DEFAULT_REFERENCE_PATH).iloc[0]
        return dict(last_sample_ns=time.perf_counter_ns(), latest={f"axis{axis}_position_raw":
            np.rad2deg(ref[f"q_{name}_rad"])*1000 for name, axis in (("hip", 1), ("knee", 4))})
    def stop(self): pass


def synthetic_execute(**kwargs):
    assert kwargs["execution_guard"]() is None
    d = pd.read_csv(kwargs["episode_dir"].parent / "leg/dispatch.csv")
    for r in d.itertuples():
        kwargs["dispatch_observer"](r.trajectory_time_s, r.dispatch_start_ns, r.dispatch_end_ns)
    return dict(completed=True, commands_dispatched=len(d), same_session_realtime_cleanup_completed=True)


def test_full_round_with_real_release_analysis_and_training(tmp_path):
    directory = prepared_round(tmp_path)
    state = rounds.run(directory, confirmation=rounds.CONFIRMATION,
        capture_factory=SyntheticCapture, execute_fn=synthetic_execute)
    assert state["status"] == "ROUND_COMPLETE"
    assert state["training_status"] == "MODEL_UPDATED"
    assert state["endpoints"]["E3"] == 1.
    assert read_json(directory / "algorithm_ready.json")["status"] == "ALGORITHM_INITIALIZED"
    assert (directory / "report.md").is_file() and (directory / "quality.json").is_file()
    assert read_json(directory / "release_approval.json")["scope"] == rounds.LOADED_SCOPE
    # Changing a bound calibration invalidates the existing release.
    from safety.robot_trial_release import load_and_validate_robot_trial_release
    c = rounds.paths(tmp_path / "round.json")
    Path(c["calibration"]).write_text("{}", encoding="ascii")
    with pytest.raises(PermissionError):
        load_and_validate_robot_trial_release(directory / "release_approval.json", reference_path=DEFAULT_REFERENCE_PATH,
            anchor_path=directory / "reviewed_anchor.json", frame_config_path=c["frame_config"],
            safety_config_path=c["safety_config"], wrench_config_path=c["wrench_config"])


def test_empty_load_is_never_trained(tmp_path):
    directory = prepared_round(tmp_path, empty=True)
    state = rounds.run(directory, confirmation=rounds.CONFIRMATION,
        capture_factory=SyntheticCapture, execute_fn=synthetic_execute)
    assert state["training_status"] == "EXCLUDED_EMPTY_LOAD"
    assert not (directory / "training_request.json").exists()


def test_loaded_capture_only_keeps_guards_and_never_trains(tmp_path):
    directory = prepared_round(tmp_path, capture_only=True)
    state = rounds.run(directory, confirmation=rounds.CONFIRMATION,
        capture_factory=SyntheticCapture, execute_fn=synthetic_execute)
    assert state["status"] == "CAPTURE_ONLY_COMPLETE"
    assert state["training_status"] == "SKIPPED_CAPTURE_ONLY"
    assert (directory / "leg/leg.csv").is_file()
    assert (directory / "quality.json").is_file()
    assert (directory / "measured.csv").is_file()
    assert (directory / "measured_metrics.json").is_file()
    assert not (directory / "algorithm_initialization_request.json").exists()
    assert not (directory / "algorithm_ready.json").exists()
    assert not (directory / "training_request.json").exists()
    assert not (directory / "model_update.json").exists()


@pytest.mark.parametrize("kind", ["unready", "incomplete", "unclosed"])
def test_failed_round_keeps_partial_data_without_training(tmp_path, kind):
    directory = prepared_round(tmp_path)
    class Capture(SyntheticCapture):
        def health_reason(self): return "stale" if kind == "unready" else None
        def stop(self):
            if kind == "unclosed": raise RuntimeError("CSV not closed")
    def execute(**kwargs):
        assert kind != "unready", "robot must not execute before capture is ready"
        if kind == "incomplete": return dict(completed=False)
        return synthetic_execute(**kwargs)
    with pytest.raises(RuntimeError):
        rounds.run(directory, confirmation=rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
    state = read_json(directory / "round_status.json")
    assert state["status"] == "ROUND_FAILED" and state["training_status"] == "NOT_UPDATED"
    assert (directory / "leg/leg.csv").is_file()
    assert not (directory / "model_update.json").exists()


def test_external_capture_failure_stops_real_executor(tmp_path):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    from tests.test_robot_trajectory_executor import Motion, Acquisition, _trajectory, _preflight, _safety
    motion = Motion()
    observed = []
    def observer(*args): observed.append(args)
    logger = EpisodeLogger(tmp_path / "episode").start()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety(),
        execution_guard=lambda: "leg_capture_stale" if observed else None, dispatch_observer=observer)
    try:
        with pytest.raises(RuntimeError, match="leg_capture_stale"):
            executor.execute(_trajectory(), _preflight())
    finally:
        logger.close(completed=False)
    assert len(observed) == 1
    assert any(c[0] == "stop" for c in motion.calls)
    assert not any(c[0] == "send" for c in motion.calls)


def test_unavailable_algorithm_blocks_before_capture_and_motion(tmp_path, monkeypatch):
    directory = prepared_round(tmp_path)
    def failed_worker(*args, **kwargs): raise RuntimeError("missing model resources")
    monkeypatch.setattr(rounds, "run_algorithm_worker", failed_worker)
    def forbidden(*a, **k): pytest.fail("hardware must not be used")
    with pytest.raises(RuntimeError, match="missing model resources"):
        rounds.run(directory, confirmation=rounds.CONFIRMATION, capture_factory=forbidden, execute_fn=forbidden)
    assert read_json(directory / "round_status.json")["status"] == "ROUND_FAILED"


def test_connection_class_mismatch_blocks_readonly_preparation(tmp_path):
    prepared_round(tmp_path)
    p = tmp_path / "round.json"
    c = read_json(p)
    c["robot_class"] = "WRONG_TEST_CLASS"
    atomic_json(p, c)
    assert any("connection/class must match" in x for x in rounds.check_config(p))
