"""Deterministic offline protocol clocks; no SDK or hardware connection."""
from dataclasses import asdict
import json
import threading
import time
from types import SimpleNamespace

import pytest

from collection.real_robot_acquisition import RealRobotAcquisition
from collection.wrench_process import DurableAudit, WrenchBudgets, WrenchProcessProvider
from hardware.rokae_adapter import RobotWrenchFrame


class AuditDouble:
    failure = None

    def __init__(self, delay_ns=0, closed_ok=True):
        self.delay_ns = delay_ns
        self.closed_ok = closed_ok

    def submit(self, event):
        done = threading.Event()
        done.set()
        return dict(done=done, error=None,
                    write_complete_ns=event.get("parent_receive_ns", event["timestamp_ns"]) + self.delay_ns)

    def close(self, budget):
        return self.closed_ok


def provider(tmp_path):
    limits = WrenchBudgets(startup_s=4, query_s=.05, progress_s=2,
        sample_age_s=1, ack_s=.1, log_s=.1, shutdown_s=.2,
        terminate_s=.5, state_age_s=1, skew_s=1,
        provenance="NOT_SAFETY_THRESHOLDS")
    p = WrenchProcessProvider(dict(mode="offline", rate_hz=50), limits, tmp_path / "audit.jsonl")
    p.process = SimpleNamespace(pid=123, is_alive=lambda: True, exitcode=0)
    p.started_ns = 1_000_000_000
    p.writer = AuditDouble()
    return p


@pytest.mark.parametrize("kind", ["QUERY_SUCCESS", "QUERY_ERROR"])
@pytest.mark.parametrize("duration_ns,exceeded", [(40_000_000, False), (60_000_000, True)])
def test_completed_query_cannot_hide_overrun_between_polls(tmp_path, kind, duration_ns, exceeded):
    p = provider(tmp_path)
    p.query_id = 1
    p.query_state = "IN_PROGRESS"
    p.query_start_ns = 1_100_000_000
    start = 1_200_000_000
    end = start + duration_ns
    frame = RobotWrenchFrame(1, start / 1e9, end / 1e9, end / 1e9,
        (start + end) / 2e9, "", "OFFLINE_HOST_QUERY", True, "", "world",
        (1., 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6)
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
        event_id=1, query_id=1, type=kind, timestamp_ns=end,
        native_start_ns=start, completion_ns=end,
        frame=asdict(frame) if kind == "QUERY_SUCCESS" else None,
        error={"code": 263, "message": "offline"} if kind == "QUERY_ERROR" else None)
    p._accept(event, end + 1_000_000)
    assert ("query_pending_too_long" in p.failure_latch) is exceeded
    assert p.events[-1]["completion_ns"] == end
    assert p.pending_audits[0][0]["completion_ns"] == end
    if kind == "QUERY_ERROR":
        assert "sdk_error" in p.failure_latch


@pytest.mark.parametrize("delay_ns,exceeded", [(30_000_000, False), (110_000_000, True)])
def test_finished_audit_write_uses_actual_completion_deadline(tmp_path, delay_ns, exceeded):
    p = provider(tmp_path)
    p.writer = AuditDouble(delay_ns)
    event = dict(version=1, run_id=p.run_id, session_id=p.session_id, pid=123,
        event_id=1, query_id=0, type="PROCESS_START", timestamp_ns=1_100_000_000)
    p._accept(event, 1_100_000_000)
    # A delayed poll must neither hide a late fsync nor penalize a timely one.
    p.poll(now_ns=1_300_000_000)
    assert ("logger_backpressure" in p.failure_latch) is exceeded


@pytest.mark.parametrize("native_age_ns,exceeded", [(10_000_000, False), (60_000_000, True)])
def test_native_deadline_excludes_boundary_ack_wait(tmp_path, native_age_ns, exceeded):
    p = provider(tmp_path)
    now = 1_400_000_000
    p.progress_ns = 1_100_000_000
    p.query_id = 1
    p.query_state = "IN_PROGRESS"
    p.query_start_ns = 1_100_000_000
    p.native_boundary_ns = 1_200_000_000
    p.latest_slot.put(dict(run_id=p.run_id, pid=123, query_id=1,
        query_in_progress=True, native_start_ns=now-native_age_ns))
    p.poll(now_ns=now)
    assert ("query_pending_too_long" in p.failure_latch) is exceeded


@pytest.mark.parametrize("disconnected,fault,closed_ok,normal", [
    (True, False, True, True), (True, True, True, False),
    (False, False, True, False), (True, False, False, False),
])
def test_cleanup_requires_confirmed_disconnect_and_reports_all_faults(
        tmp_path, disconnected, fault, closed_ok, normal):
    p = provider(tmp_path)
    p.started_ns = time.perf_counter_ns()
    p.process.is_alive = lambda: False
    p.offline_disconnect_completed = disconnected
    p.writer = AuditDouble(closed_ok=closed_ok)
    if fault:
        p.fail("offline_test_fault")
    result = p.stop()
    assert result["NORMAL_CLEANUP"] is normal
    assert result["failure_latch"] == p.failure_latch
    logged = next(e["outcome"] for e in p.events if e["type"] == "CLEANUP_OUTCOME")
    if fault or not disconnected:
        assert logged["NORMAL_CLEANUP"] is False
    if not closed_ok:
        assert "audit_cleanup_unconfirmed" in result["failure_latch"]
    assert p.stop() == result


def test_startup_wait_returns_immediately_after_latched_failure():
    acquisition = object.__new__(RealRobotAcquisition)
    acquisition._stop_event = threading.Event()
    acquisition._background_error = None
    ticks = iter([0., 0., 5.])
    acquisition.clock = SimpleNamespace(now_s=lambda: next(ticks))
    calls = []

    def health():
        calls.append(True)
        return SimpleNamespace(valid=False, wrench_failure_latch=("logger_backpressure",))

    acquisition.latest_health = health
    assert acquisition.wait_until_healthy(timeout_s=10) is False
    assert calls == [True]


@pytest.mark.parametrize("write_fails", [False, True])
def test_queued_audit_batch_is_ordered_and_finally_fsynced(tmp_path, monkeypatch, write_fails):
    import collection.wrench_process as module
    entered = threading.Event()
    release = threading.Event()
    fsync_calls = []
    original_fsync = module.os.fsync

    def hook(event):
        if event["index"] == 0:
            entered.set()
            assert release.wait(2)
        if write_fails and event["index"] == 3:
            raise OSError("offline batch write failure")

    def fsync(fd):
        fsync_calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fsync)
    path = tmp_path / "batch.jsonl"
    audit = DurableAudit(path, write_hook=hook)
    tickets = []
    try:
        tickets.append(audit.submit(dict(index=0)))
        assert entered.wait(2)
        tickets.extend(audit.submit(dict(index=i)) for i in range(1, 6))
        assert all(not t["done"].is_set() for t in tickets)
        release.set()
        assert all(t["done"].wait(2) for t in tickets)
    finally:
        release.set()
        closed = audit.close(2)
    assert closed is (not write_fails)
    assert tickets[0]["error"] is None
    if write_fails:
        assert all(t["error"] and t["write_complete_ns"] is None for t in tickets[1:])
        assert len(fsync_calls) == 0
    else:
        assert [e["index"] for e in map(json.loads, path.read_text().splitlines())] == list(range(6))
        assert len(fsync_calls) == 1
        assert len({t["write_complete_ns"] for t in tickets[1:]}) == 1
