import pytest

from scripts.characterize_wrench_longrun import concurrency_summary, event_windows, latency_summary


def _row(index: int, latency_ms: float, *, success: bool = True, code=None):
    return {
        "latency_ms": latency_ms,
        "success": success,
        "error_code": code,
        "host_call_start_monotonic_ns": index * 20_000_000,
        "rt_updates_during_call": int(latency_ms / 8),
    }


def test_latency_buckets_deadlines_and_recovery_are_distinct() -> None:
    rows = [
        _row(0, 1.0),
        _row(1, 7.0),
        _row(2, 15.0),
        _row(3, 36.0),
        _row(4, 75.0),
        _row(5, 250.0),
        _row(6, 750.0),
        _row(7, 10_000.0, success=False, code=263),
        _row(8, 1.0),
    ]
    result = latency_summary(rows, 50.0)
    assert result["latency_buckets"] == {
        "le_5_ms": 2,
        "gt_5_le_10_ms": 1,
        "gt_10_le_20_ms": 1,
        "gt_20_le_50_ms": 1,
        "gt_50_le_100_ms": 1,
        "gt_100_le_500_ms": 1,
        "gt_500_le_1000_ms": 1,
        "gt_1000_ms": 1,
    }
    assert result["deadline_miss_count"] == 5
    assert result["error_263_count"] == 1
    assert result["error_263_recovered_on_next_query_count"] == 1
    assert result["query_start_period_ms"]["mean"] == 20.0


def test_concurrency_summary_divides_timestamp_gap_by_sequence_delta() -> None:
    ticks = [
        {
            "main_loop_timestamp_ns": 0,
            "loop_period_ms": None,
            "rt_sequence": 10,
            "rt_timestamp_s": 1.000,
            "rt_age_ms": 2.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_age_ms": 5.0,
            "wrench_stale": False,
            "wrench_valid": True,
            "wrench_worker_alive": True,
            "wrench_query_in_flight": False,
        },
        {
            "main_loop_timestamp_ns": 10_000_000,
            "loop_period_ms": 10.0,
            "rt_sequence": 12,
            "rt_timestamp_s": 1.016,
            "rt_age_ms": 4.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_age_ms": 7.0,
            "wrench_stale": False,
            "wrench_valid": True,
            "wrench_worker_alive": True,
            "wrench_query_in_flight": True,
        },
    ]
    result = concurrency_summary(ticks)
    assert result["rt_source_period_ms"]["mean"] == pytest.approx(8.0)
    assert result["consumer_skipped_rt_sequence_count"] == 1
    assert result["rt_sequence_advance"] == 2


def test_event_windows_keep_previous_current_and_next_calls() -> None:
    rows = []
    for index in range(25):
        row = _row(index, 1.0)
        row.update({
            "sequence_id": index + 1,
            "scheduled_time_ns": index,
            "call_start_ns": index,
            "call_end_ns": index + 1,
            "query_latency_ms": 1.0,
            "host_call_end_monotonic_ns": index + 1,
            "error_message": "",
            "latest_rt_sequence": index,
            "rt_sequence_after_call": index + 1,
            "time_since_last_success_ms": 20.0,
        })
        rows.append(row)
    rows[12]["query_latency_ms"] = 1_500.0
    rows[12]["latency_ms"] = 1_500.0
    windows = event_windows(rows)
    assert len(windows) == 1
    assert len(windows[0]["previous_successful"]) == 10
    assert windows[0]["current"]["sequence_id"] == 13
    assert len(windows[0]["next"]) == 10
