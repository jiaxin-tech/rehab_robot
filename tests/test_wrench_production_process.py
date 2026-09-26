"""Offline OS-process tests. All budgets are NOT_SAFETY_THRESHOLDS."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from collection.wrench_process import DurableAudit, Slot, WrenchBudgets, WrenchProcessProvider


@pytest.fixture
def budgets():
    return WrenchBudgets(startup_s=4, query_s=.25, progress_s=2,
        sample_age_s=1, ack_s=2, log_s=.25, shutdown_s=.2,
        terminate_s=.5, state_age_s=1, skew_s=1,
        provenance="NOT_SAFETY_THRESHOLDS")


@pytest.fixture
def make_provider(tmp_path, request, budgets):
    providers = []
    root = Path(os.environ.get("WRENCH_TEST_OUTPUT", str(tmp_path))) / request.node.name.replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)

    def make(*, overrides=None, writer=None, limits=None, **config):
        directory = root / str(len(providers))
        directory.mkdir(exist_ok=False)
        args = dict(mode="offline", rate_hz=50, **config)
        provider = WrenchProcessProvider(args, limits or budgets, directory / "query_events.jsonl",
                                        **({"audit_factory": writer} if writer else {}))
        provider.trace = []
        provider.output = directory
        provider.start()
        providers.append(provider)
        return provider

    yield make
    for provider in providers:
        outcome = provider.cleanup or provider.stop()
        (provider.output / "cleanup_outcome.json").write_text(json.dumps(outcome, indent=2))
        (provider.output / "health_timeline.json").write_text(json.dumps(provider.trace, indent=2))
        (provider.output / "gap_drop_diagnostics.json").write_text(json.dumps({
            "gaps": provider.gaps, "failures": provider.failure_latch,
            "budget_provenance": provider.budgets.provenance}, indent=2))
        events = [e for e in provider.events if e["type"].startswith(("PROCESS", "DISCONNECT", "FORCED", "CLEANUP"))]
        (provider.output / "process_lifecycle.json").write_text(json.dumps(events, indent=2))
        assert not provider.process.is_alive(), "offline child leaked"


def until(provider, predicate, seconds=6):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        state = provider.poll()
        provider.trace.append(state)
        if predicate(state):
            return state
        time.sleep(.002)
    raise AssertionError(f"condition not reached: {provider.health()}")


def test_normal_durable_protocol(make_provider):
    p = make_provider()
    until(p, lambda h: p.query_id >= 3 and h["STREAM_HEALTH"])
    assert p.last_good.valid
    assert p.last_good.host_monotonic_time_s < time.perf_counter()
    outcome = p.stop()
    assert outcome["NORMAL_CLEANUP"]
    assert outcome["OFFLINE_DISCONNECT_COMPLETED"]
    assert outcome["GRACEFUL_SDK_DISCONNECT"] is None  # no SDK used
    events = [json.loads(x) for x in p.audit_path.read_text().splitlines()]
    for event in events:
        if event["type"] == "QUERY_SUCCESS":
            starts = [x for x in events if x["type"] == "QUERY_START_INTENT" and x["query_id"] == event["query_id"]]
            assert len(starts) == 1
            acks = [x for x in events if x["type"] == "DURABLE_ACK" and x["child_event_id"] == starts[0]["event_id"]]
            assert len(acks) == 1
            assert acks[0]["durable_ns"] <= event["native_start_ns"]


@pytest.mark.parametrize("behavior", ["block", "never", "gil_block"])
def test_blocking_parent_progress_and_forced_cleanup(make_provider, behavior):
    p = make_provider(behavior=behavior, delay_s=10)
    until(p, lambda h: h["CURRENT_QUERY_STATE"] == "IN_PROGRESS")
    ticks = 0
    simulated_rt = 0
    deadline = time.perf_counter() + 1
    while time.perf_counter() < deadline and not p.failure_latch:
        simulated_rt += 1  # parent RT surrogate, no native SDK
        p.trace.append(p.poll())
        ticks += 1
        time.sleep(.002)
    assert ticks > 10 and simulated_rt == ticks
    assert "query_pending_too_long" in p.failure_latch
    assert p.health()["completion"] is None
    assert not any(e["type"] in ("QUERY_SUCCESS", "QUERY_ERROR") for e in p.events)
    result = p.stop()
    assert result["FORCED_WORKER_TERMINATION"]
    assert result["GRACEFUL_SDK_DISCONNECT"] is False
    assert result["CONTROLLER_SESSION_RECOVERY"] is None


def test_delayed_263_preserves_last_good(make_provider):
    p = make_provider(behavior="error263", delay_s=.04, good_queries=1)
    until(p, lambda h: h["LAST_ERROR"] is not None)
    h = p.health()
    assert h["LATEST_GOOD_SAMPLE_VALID"]
    assert not h["STREAM_HEALTH"]
    assert h["LAST_ERROR"]["code"] == 263
    assert p.last_good.sequence_id == 1
    assert len([e for e in p.events if e["type"] == "QUERY_ERROR"]) == 1
    assert p.query_id == 2  # no retry


def test_crash_pending_no_fabricated_completion(make_provider):
    p = make_provider(behavior="crash")
    until(p, lambda h: "worker_exited" in h["FAILURE_LATCH"])
    assert p.process.exitcode == 73
    assert p.health()["completion"] is None


def test_stale_last_good_and_failure_stays_latched(make_provider, budgets):
    p = make_provider(behavior="block", good_queries=1, delay_s=.3,
                      limits=replace(budgets, query_s=1, sample_age_s=.08, shutdown_s=1))
    until(p, lambda h: "wrench_sample_stale_or_clock_invalid" in h["FAILURE_LATCH"])
    stamp = p.last_good.host_monotonic_time_s
    assert p.health()["LATEST_GOOD_SAMPLE_VALID"] and not p.health()["STREAM_HEALTH"]
    until(p, lambda h: p.last_completed_ns is not None and p.query_state == "SUCCEEDED" and p.last_good.sequence_id == 2)
    assert p.last_good.host_monotonic_time_s > stamp
    assert not p.health()["STREAM_HEALTH"]
    assert p.failure_latch


@pytest.mark.parametrize("phase", ["startup", "ack", "query", "completion", "disconnect"])
def test_shutdown_phases(make_provider, phase):
    config = {}
    if phase == "startup": config["startup_delay_s"] = 10
    if phase == "query": config.update(behavior="never")
    if phase == "disconnect": config["disconnect_delay_s"] = 10
    p = make_provider(**config)
    if phase == "startup": until(p, lambda h: h["CURRENT_QUERY_STATE"] == "STARTING")
    elif phase == "query":
        until(p, lambda h: bool((p.latest_slot.read() or {}).get("query_in_progress")))
    elif phase == "completion": until(p, lambda h: h["LAST_COMPLETED_QUERY_TIME"] is not None)
    elif phase == "disconnect": until(p, lambda h: h["STREAM_HEALTH"])
    # ack case stops before polling, while child awaits its first durable ACK.
    result = p.stop()
    assert result["process_exited"]
    if phase in ("startup", "query", "disconnect"):
        assert result["FORCED_WORKER_TERMINATION"]
        assert result["GRACEFUL_SDK_DISCONNECT"] is False


def test_missing_first_heartbeat(make_provider, budgets):
    p = make_provider(pre_start_delay_s=10, limits=replace(budgets, startup_s=.15))
    until(p, lambda h: "missing_first_heartbeat" in h["FAILURE_LATCH"])


def test_lost_ack(make_provider, budgets):
    p = make_provider(limits=replace(budgets, ack_s=.1))
    deadline=time.perf_counter()+4
    while p.audit_slot.read() is None and time.perf_counter()<deadline:
        time.sleep(.002)
    time.sleep(.2)  # no poll/ACK after observing first publication
    until(p, lambda h: any("ack_timeout" in x for x in h["FAILURE_LATCH"]))
    assert p.last_good is None


@pytest.mark.parametrize("mode", ["backpressure", "failure"])
def test_logger_faults(make_provider, mode, budgets):
    release = threading.Event()
    def hook(event):
        if mode == "failure": raise OSError("offline disk failure")
        release.wait(2)
    p = make_provider(writer=lambda path: DurableAudit(path, write_hook=hook))
    try:
        until(p, lambda h: any("logger" in x for x in h["FAILURE_LATCH"]))
        assert not p.health()["STREAM_HEALTH"]
        assert p.last_good is None  # no start ACK while fsync is outstanding
    finally:
        release.set()


@pytest.mark.parametrize("fault", ["gap", "duplicate", "order", "pid", "run", "clock", "overflow", "disconnect_order", "stopped_order"])
def test_protocol_faults_are_latched(tmp_path, budgets, fault):
    p = WrenchProcessProvider(dict(mode="offline", rate_hz=50), budgets, tmp_path/'audit.jsonl')
    # Protocol unit test, no process/native provider required.
    p.process = SimpleNamespace(pid=123, is_alive=lambda: True)
    p.started_ns = time.perf_counter_ns()
    p.writer = DurableAudit(p.audit_path)
    now = time.perf_counter_ns()
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
                 event_id=1, query_id=0, type="PROCESS_START", timestamp_ns=now)
    if fault == "gap": event['event_id'] = 3
    if fault == "duplicate": p.expected_event = 2
    if fault == "order": event.update(type="QUERY_SUCCESS", query_id=1)
    if fault == "pid": event['pid'] = 999
    if fault == "run": event['run_id'] = 'wrong'
    if fault == "clock": event['timestamp_ns'] = -1
    if fault == "disconnect_order": event['type'] = 'DISCONNECT_COMPLETE'
    if fault == "stopped_order": event['type'] = 'PROCESS_STOPPED'
    if fault == "overflow":
        with pytest.raises(ValueError, match="overflow"):
            p.audit_slot.put(dict(event, padding='x'*70000))
        p.fatal_slot.put({'reason':'audit_overflow'})
    else:
        p.audit_slot.put(event)
    h = p.poll()
    assert h['FAILURE_LATCH'] and not h['STREAM_HEALTH']
    p.writer.close(1)


def test_invalid_supervisor_clock(make_provider):
    p = make_provider()
    until(p, lambda h: h['STREAM_HEALTH'])
    p.poll(-1)
    assert 'invalid_supervisor_clock' in p.failure_latch


def test_audit_queue_overflow_is_explicit(tmp_path):
    release = threading.Event()
    writer = DurableAudit(tmp_path/'overflow.jsonl', write_hook=lambda _:release.wait(2))
    try:
        with pytest.raises(queue.Full):
            for i in range(1000): writer.submit({'event_id':i})
    finally:
        release.set()
        writer.close(2)


def test_republished_acknowledged_event_is_rejected(make_provider):
    p=make_provider(behavior='never')
    until(p,lambda h:h['CURRENT_QUERY_STATE']=='IN_PROGRESS')
    # Offline protocol injection only; a conforming child is now in the stub.
    event=p.audit_slot.read()
    p.audit_slot.put(event)
    p.poll()
    assert any('sequence' in x for x in p.failure_latch)


def test_state_freshness_failure_is_latched(make_provider,tmp_path):
    from tests.test_real_robot_acquisition import FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    p=make_provider()
    until(p,lambda h:h['STREAM_HEALTH'])
    acquisition=RealRobotAcquisition(FakeAdapter(),EpisodeLogger(tmp_path/'unused'),wrench_provider=p)
    stale=FakeAdapter().read_state_frame()
    acquisition._latest_state=replace(stale,host_monotonic_time_s=time.perf_counter()-10)
    acquisition._latest_wrench=p.last_good
    h=acquisition.latest_health()
    assert not h.valid
    assert 'state_freshness_failure' in p.failure_latch


def test_stationary_preparation_no_live_entry(tmp_path):
    from scripts.prepare_wrench_stationary_validation import prepare
    manifest=prepare(tmp_path/'prepared',50)
    assert manifest['duration_s']==900 and manifest['intended_wrench_hz']==50
    assert not manifest['live_runner_ready'] and not manifest['robot_connected']
    with pytest.raises(ValueError): prepare(tmp_path/'invalid',float('nan'))


def test_existing_control_rejects_historical_good_unhealthy_stream():
    import pandas as pd
    from control.execution_preflight import evaluate_execution_preflight
    from control.robot_trajectory_executor import RokaeMotionExecutor
    from tests.test_robot_trajectory_executor import _safety, _health
    from safety.experiment_safety import ExperimentSafetyConfig
    h = replace(_health(), valid=False, invalid_reason='sdk_error',
                wrench_valid=True, wrench_stream_healthy=False, wrench_failure_latch=('sdk_error',))
    # Deliberately unapproved fake request; evaluate the real health gate without
    # loading or mutating an official trajectory release or calling any SDK.
    audit = SimpleNamespace(trajectory_id='fake',sample_count=0,**{k:True for k in (
        'trajectory_valid','first_target_equals_anchor','final_target_equals_anchor',
        'first_relative_displacement_zero','final_relative_displacement_zero',
        'approved_rom_valid','theta_shank_definition_valid','all_samples_finite','time_strictly_increasing')})
    result = evaluate_execution_preflight(mode='offline',enable_motion=False,operator_confirmation='',
        requested_anchor_id='fake',frame=SimpleNamespace(reviewed=False),
        anchor=SimpleNamespace(reviewed=False,anchor_id='fake',trajectory_id='fake',tcp_pose_base=(0.,)*6),
        safety=ExperimentSafetyConfig(),trajectory=pd.DataFrame(),audit=audit,acquisition_health=h,
        logger=SimpleNamespace(ready=True,healthy=True),robot_adapter=SimpleNamespace(is_connected=False))
    assert not result.allowed
    assert 'acquisition_streams_unhealthy' in result.reasons
    with pytest.raises(PermissionError): result.require_allowed()
    executor = object.__new__(RokaeMotionExecutor)
    executor.logger = SimpleNamespace(healthy_signal=True)
    executor.acquisition = SimpleNamespace(latest_health=lambda:h)
    executor.safety = _safety()
    assert 'sdk_error' in executor._runtime_reasons()
    # No motion object exists in this test; no execute() or motion API called.


def test_acquisition_process_integration(make_provider, tmp_path):
    from tests.test_real_robot_acquisition import FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    p = make_provider(behavior='error263', good_queries=2, delay_s=.05)
    # Integration owns start; use a fresh provider with the same protocol config.
    p.stop()
    provider = WrenchProcessProvider(dict(mode='offline',rate_hz=50,behavior='error263',good_queries=2,delay_s=.05),
                                    p.budgets,tmp_path/'integration_events.jsonl')
    logger = EpisodeLogger(tmp_path/'episode').start()
    adapter = FakeAdapter()
    def forbidden(): raise AssertionError('parent native wrench called')
    adapter.read_internal_wrench = forbidden
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    acquisition.start()
    try:
        deadline=time.perf_counter()+6
        while time.perf_counter()<deadline and 'sdk_error' not in provider.failure_latch: time.sleep(.005)
        h=acquisition.latest_health()
        assert 'sdk_error' in provider.failure_latch
        assert h.wrench_valid and not h.valid
        assert h.wrench_thread_alive  # actual consumer
        assert h.wrench_failure_latch
        assert logger.row_counts['robot_wrench'] == 2
    finally:
        acquisition.stop()
        logger.close(completed=False)
    assert not provider.process.is_alive()


def test_missing_provider_refuses_before_connect(tmp_path):
    from tests.test_real_robot_acquisition import FakeAdapter
    from collection.episode_logger import EpisodeLogger
    from collection.real_robot_acquisition import RealRobotAcquisition
    adapter=FakeAdapter()
    acquisition=RealRobotAcquisition(adapter,EpisodeLogger(tmp_path/'not_started'))
    with pytest.raises(PermissionError): acquisition.start()
    assert adapter.calls == []
