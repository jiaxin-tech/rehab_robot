from __future__ import annotations

import argparse
import json
import time

import pytest

from scripts.audit_state_wrench_timing import (
    NOT_DEFINED,
    UNDEFINED,
    build_gates,
    consecutive_error_runs,
    error_histogram,
    error_timeline,
    metric_summary,
    run_compare,
)
from scripts.wrench_process_isolation import RtProcessSupervisor, WrenchProcessSupervisor


def test_metric_summary_has_exact_median_and_percentiles() -> None:
    result = metric_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == {
        "count": 5,
        "mean": 3.0,
        "median": 3.0,
        "p95": pytest.approx(4.8),
        "p99": pytest.approx(4.96),
        "max": 5.0,
    }


def test_error_263_timeline_and_consecutive_runs_are_not_suppressed() -> None:
    events = [
        {"sequence_id": 1, "call_start_ns": 0, "call_end_ns": 1_000_000, "query_latency_ms": 1.0, "success": True, "error_code": None},
        {"sequence_id": 2, "call_start_ns": 20_000_000, "call_end_ns": 30_000_000, "query_latency_ms": 10.0, "success": False, "error_code": 263, "error_message": "failure", "operation_state": "IDLE"},
        {"sequence_id": 3, "call_start_ns": 40_000_000, "call_end_ns": 45_000_000, "query_latency_ms": 5.0, "success": False, "error_code": 263, "error_message": "failure", "operation_state": "IDLE"},
        {"sequence_id": 4, "call_start_ns": 60_000_000, "call_end_ns": 61_000_000, "query_latency_ms": 1.0, "success": True, "error_code": None},
    ]
    assert error_histogram(events) == {"263": 2}
    timeline = error_timeline(events)
    assert [item["error_code"] for item in timeline] == [263, 263]
    assert all(item["recovered"] for item in timeline)
    runs = consecutive_error_runs(events)
    assert runs == [{
        "first_sequence_id": 2,
        "last_sequence_id": 3,
        "count": 2,
        "onset_ns": 20_000_000,
        "end_ns": 45_000_000,
        "duration_ms": 25.0,
        "codes": [263, 263],
        "recovered": True,
        "recovery_sequence_id": 4,
        "recovery_timestamp_ns": 61_000_000,
    }]


def test_missing_formal_timing_thresholds_remain_undefined() -> None:
    summary = {
        "completion": {"completed_requested_duration": True},
        "rt_source": {"ring_overwrite_count": 0, "sequence_gap_count": 0},
        "wrench": {"event_drop_count": 0, "formal_stale_count": NOT_DEFINED},
        "state_transitions": {"before": "idle", "after": "idle", "non_idle_transitions": []},
        "workers": {
            "rt_hung_event_count": 0, "rt_crash_event_count": 0,
            "wrench_hung_event_count": 0, "wrench_crash_event_count": 0,
            "rt_cleanup": {"worker_exited": True, "graceful_disconnect_confirmed": True},
            "wrench_cleanup": {"worker_exited": True, "graceful_disconnect_confirmed": True},
        },
        "rt_ipc": {"formal_stale_count": NOT_DEFINED},
        "supervisor": {"formal_late_cycle_count": NOT_DEFINED},
    }
    gates = build_gates(
        case="test_b",
        summary=summary,
        safety={"max_state_age_s": None, "max_wrench_age_s": None, "max_command_lateness_s": None},
    )
    assert gates["data_integrity"]["status"] == "PASS"
    assert gates["operation_state_stability"]["status"] == "PASS"
    assert gates["rt_source_timing"]["status"] == UNDEFINED
    assert gates["rt_ipc_freshness"]["evidence"]["threshold"] == NOT_DEFINED
    assert gates["wrench_error_reliability"]["status"] == UNDEFINED


def test_rt_source_ring_is_contiguous_and_nonblocking() -> None:
    rt = RtProcessSupervisor(stale_age_ms=None)
    try:
        rt.start({"offline": True, "publish_hz": 125.0})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not rt.poll().rt_valid:
            time.sleep(0.01)
        time.sleep(0.15)
        events = rt.drain_source_events()
        assert len(events) >= 10
        sequences = [event["rt_sequence"] for event in events]
        assert sequences == list(range(sequences[0], sequences[-1] + 1))
        assert rt.source_ring_overwrite_count == 0
        cleanup = rt.stop_normally(3.0)
        assert cleanup["worker_exited"]
    finally:
        if rt.alive:
            rt.terminate()
        rt.close()


def test_wrench_event_stream_preserves_every_error_263_result() -> None:
    wrench = WrenchProcessSupervisor(stale_age_ms=None)
    events = []
    try:
        wrench.start(mode="offline", config={
            "behavior": "error263", "delay_s": 0.001, "target_hz": 50.0,
            "event_queue_size": 256,
        })
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            wrench.poll()
            events.extend(wrench.drain_events())
            time.sleep(0.005)
        cleanup = wrench.stop_normally(3.0)
        events.extend(wrench.drain_events())
        assert cleanup["worker_exited"]
        assert len(events) >= 5
        assert all(event["error_code"] == 263 for event in events)
        assert [event["sequence_id"] for event in events] == list(range(1, len(events) + 1))
        assert wrench.metadata["event_drop_count"] == 0
        assert wrench.metadata["failure_count"] == len(events)
    finally:
        if wrench.alive:
            wrench.terminate()
        wrench.close()


def test_comparison_report_generation_is_deterministic(tmp_path) -> None:
    def payload(case: str, offset: float) -> dict:
        return {
            "case": case,
            "completion": {
                "completed_requested_duration": True,
                "observed_duration_s": 10.0,
                "requested_duration_s": 10.0,
                "fatal_error": None,
            },
            "rt_source": {"interval_ms": {"p99": 8.0 + offset, "max": 9.0 + offset}},
            "rt_ipc": {"age_ms": {"p95": 7.0 + offset, "p99": 8.0 + offset, "max": 10.0 + offset}},
            "supervisor": {"loop_interval_ms": {"p99": 11.0 + offset, "max": 20.0 + offset}},
            "wrench": {"request_count": 0 if case == "test_a" else 100, "failure_count": 0, "error_code_histogram": {}, "error_count_263": 0},
        }

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(payload("test_a", 0.0)), encoding="utf-8")
    b_path.write_text(json.dumps(payload("test_b", 1.0)), encoding="utf-8")
    result = run_compare(argparse.Namespace(
        test_a_summary=str(a_path), test_b_summary=str(b_path), output_dir=str(tmp_path)
    ))
    assert result["comparison_gate"]["status"] == UNDEFINED
    assert result["descriptive_delta"]["rt_ipc_p99_ms"] == 1.0
    assert all((tmp_path / name).exists() for name in (
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.json")),
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.md")),
        next(path.name for path in tmp_path.glob("state_wrench_timing_comparison_*.png")),
    ))
