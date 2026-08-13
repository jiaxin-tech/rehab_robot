from scripts.validate_wrench_process_live import summarize


def test_live_summary_distinguishes_rt_period_from_consumer_gap() -> None:
    rows = [
        {
            "main_loop_timestamp_ns": 1_000_000_000,
            "loop_period_ms": None,
            "rt_sequence": 10,
            "rt_timestamp_ns": 990_000_000,
            "rt_age_ms": 10.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_sequence": 2,
            "wrench_age_ms": 15.0,
            "wrench_valid": True,
            "wrench_stale": False,
            "worker_alive": True,
            "worker_hung": False,
            "heartbeat_age_ms": 5.0,
            "last_error_code": None,
        },
        {
            "main_loop_timestamp_ns": 1_010_000_000,
            "loop_period_ms": 10.0,
            "rt_sequence": 12,
            "rt_timestamp_ns": 1_006_000_000,
            "rt_age_ms": 4.0,
            "rt_valid": True,
            "operation_state": "IDLE",
            "wrench_sequence": 3,
            "wrench_age_ms": 5.0,
            "wrench_valid": True,
            "wrench_stale": False,
            "worker_alive": True,
            "worker_hung": False,
            "heartbeat_age_ms": 3.0,
            "last_error_code": None,
        },
    ]
    result = summarize(rows)
    assert result["rt_source_period_ms"]["mean"] == 8.0
    assert result["rt_sequence_advance"] == 2
    assert result["wrench_sequence_advance"] == 1
    assert result["main_loop_rate_hz"] == 100.0
