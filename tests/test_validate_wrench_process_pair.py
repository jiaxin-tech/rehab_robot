import csv

from scripts.validate_wrench_process_live import summarize
from scripts.validate_wrench_process_pair import CSV_FIELDS, _row_for_summary, _run_loop
from scripts.wrench_process_isolation import RtProcessSupervisor, WrenchProcessSupervisor


def test_pair_summary_projection_keeps_rt_and_wrench_fault_domains_distinct() -> None:
    row = {
        "main_loop_timestamp_ns": 10,
        "loop_period_ms": 10.0,
        "rt_sequence": 4,
        "rt_timestamp_ns": 8,
        "rt_age_ms": 2.0,
        "rt_valid": True,
        "operation_state": "IDLE",
        "wrench_sequence": 3,
        "wrench_age_ms": 20.0,
        "wrench_valid": True,
        "wrench_stale": False,
        "wrench_worker_alive": True,
        "wrench_worker_hung": False,
        "wrench_heartbeat_age_ms": 5.0,
        "last_error_code": 263,
    }
    projected = _row_for_summary(row)
    assert projected["worker_hung"] is False
    assert projected["last_error_code"] == 263
    assert projected["rt_sequence"] == 4


def test_pure_supervisor_keeps_100hz_with_two_offline_processes(tmp_path) -> None:
    rt = RtProcessSupervisor(stale_age_ms=50, worker_hung_ms=500)
    wrench = WrenchProcessSupervisor(stale_age_ms=150, worker_hung_ms=500)
    rt.start({"offline": True, "publish_hz": 125.0})
    wrench.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.001, "target_hz": 20.0,
    })
    rows = []
    path = tmp_path / "pair.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        _run_loop(
            stage="overlap",
            duration_s=1.0,
            rt=rt,
            wrench=wrench,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_fault=True,
        )
    wrench_cleanup = wrench.stop_normally()
    rt_cleanup = rt.stop_normally()
    wrench.close()
    rt.close()
    projected = [_row_for_summary(row) for row in rows]
    result = summarize(projected)
    assert result["main_loop_rate_hz"] >= 80.0
    assert result["rt_sequence_advance"] > 20
    assert result["wrench_sequence_advance"] > 5
    assert wrench_cleanup["worker_exited"]
    assert rt_cleanup["worker_exited"]
