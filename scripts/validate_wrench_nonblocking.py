"""Pure-software fault injection for the diagnostic non-blocking wrench path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from scripts.characterize_wrench_longrun import percentile
from scripts.wrench_nonblocking_diagnostic import NonBlockingWrenchWorker


class SyntheticStateProducer:
    def __init__(self, hz: float = 125.0) -> None:
        self.period_s = 1.0 / hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence_id = 0
        self._timestamp_ns: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="synthetic-rt-state", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now_ns = time.perf_counter_ns()
            with self._lock:
                self._sequence_id += 1
                self._timestamp_ns = now_ns
            next_tick += self.period_s
            self._stop.wait(max(0.0, next_tick - time.perf_counter()))

    def snapshot(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        with self._lock:
            sequence_id = self._sequence_id
            timestamp_ns = self._timestamp_ns
        return {
            "sequence_id": sequence_id,
            "timestamp_s": None if timestamp_ns is None else timestamp_ns / 1e9,
            "age_ms": None if timestamp_ns is None else (now_ns - timestamp_ns) / 1e6,
            "operation_state": "IDLE",
            "valid": timestamp_ns is not None,
        }

    def stop(self) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return bool(self._thread is None or not self._thread.is_alive())


class ScriptedQuery:
    def __init__(self) -> None:
        self.calls = 0
        self.scenarios = (
            ("normal_1ms", 0.001, None),
            ("delayed_40ms", 0.040, None),
            ("freeze_500ms", 0.500, None),
            ("long_block_10s", 10.0, None),
            ("sdk_error_mock", 0.001, RuntimeError("xCoreSDK getEndTorque failed (263): synthetic timeout")),
        )

    def __call__(self) -> dict[str, Any]:
        if self.calls < len(self.scenarios):
            name, delay_s, error = self.scenarios[self.calls]
        else:
            name, delay_s, error = ("recovered_normal_1ms", 0.001, None)
        self.calls += 1
        time.sleep(delay_s)
        if error is not None:
            raise error
        return {"scenario": name, "cartesian_force_raw_n": [1.0, 2.0, 3.0]}


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def run(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    state = SyntheticStateProducer()
    query = ScriptedQuery()
    result_rows: list[dict[str, Any]] = []
    result_lock = threading.Lock()

    def record(row: dict[str, Any]) -> None:
        with result_lock:
            result_rows.append(dict(row))

    state.start()
    worker = NonBlockingWrenchWorker(
        query,
        target_hz=20.0,
        stale_threshold_ms=100.0,
        on_result=record,
        state_snapshot=state.snapshot,
    )
    worker.start()
    ticks: list[dict[str, Any]] = []
    start_ns = time.perf_counter_ns()
    previous_ns: int | None = None
    deadline_ns = start_ns + int(12.0e9)
    next_tick_ns = start_ns
    while time.perf_counter_ns() < deadline_ns:
        now_ns = time.perf_counter_ns()
        snapshot = worker.cache.snapshot(now_ns)
        state_snapshot = state.snapshot()
        ticks.append({
            "timestamp_ns": now_ns,
            "period_ms": None if previous_ns is None else (now_ns - previous_ns) / 1e6,
            "wrench_age_ms": snapshot.age_ms,
            "wrench_valid": snapshot.valid,
            "wrench_stale": snapshot.stale,
            "source_alive": snapshot.source_alive,
            "query_in_flight": snapshot.query_in_flight,
            "in_flight_age_ms": snapshot.in_flight_age_ms,
            "last_error": snapshot.last_error,
            "last_error_code": snapshot.last_error_code,
            "state_sequence_id": state_snapshot["sequence_id"],
            "state_age_ms": state_snapshot["age_ms"],
        })
        previous_ns = now_ns
        next_tick_ns += 10_000_000
        remaining_ns = next_tick_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        else:
            next_tick_ns = time.perf_counter_ns()

    worker.request_stop()
    worker_joined = worker.join(timeout=2.0)
    state_joined = state.stop()
    with result_lock:
        rows = list(result_rows)
    long_rows = [row for row in rows if float(row["latency_ms"]) >= 9000.0]
    long_row = long_rows[0] if long_rows else None
    if long_row:
        during_long = [
            tick for tick in ticks
            if int(long_row["host_call_start_monotonic_ns"]) <= int(tick["timestamp_ns"]) <= int(long_row["host_call_end_monotonic_ns"])
        ]
    else:
        during_long = []
    periods = [float(tick["period_ms"]) for tick in ticks if tick["period_ms"] is not None]
    stale_during_long = any(bool(tick["wrench_stale"]) for tick in during_long)
    main_continued = len(during_long) >= 900 and (max(float(tick["period_ms"] or 0.0) for tick in during_long) < 100.0)
    state_updates = int(long_row.get("rt_updates_during_call") or 0) if long_row else 0
    state_continued = state_updates >= 1000
    error_rows = [row for row in rows if not bool(row["success"])]
    error_detected = any(tick.get("last_error_code") == 263 for tick in ticks)
    passed = all((
        bool(long_row), main_continued, state_continued, stale_during_long,
        bool(error_rows), error_detected, worker_joined, state_joined,
    ))
    payload = {
        "schema_version": 1,
        "diagnostic": "wrench_nonblocking_fault_injection",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "real_robot_connected": False,
        "architecture": {
            "isolation": "thread worker plus immutable latest snapshot cache",
            "main_loop_calls_getEndTorque": False,
            "native_call_cancellation_claimed": False,
            "process_recommendation": (
                "Thread isolation is sufficient to keep the main loop responsive when the call releases the GIL, "
                "but a permanently blocked native call cannot be killed safely. Process isolation is required only "
                "if restart/termination of a permanently stuck worker becomes a hard requirement; SDK connection "
                "ownership and concurrent-controller-session support must be validated first."
            ),
        },
        "injections": [
            {"name": name, "delay_ms": delay_s * 1000.0, "exception": None if error is None else str(error)}
            for name, delay_s, error in query.scenarios
        ],
        "query_results": [
            {key: value for key, value in row.items() if key != "value"}
            for row in rows
        ],
        "main_loop_period_ms": summary(periods),
        "long_block": {
            "observed": bool(long_row),
            "latency_ms": None if long_row is None else long_row["latency_ms"],
            "main_ticks_during_block": len(during_long),
            "main_loop_continued": main_continued,
            "state_updates_during_block": state_updates,
            "state_acquisition_continued": state_continued,
            "stale_detected": stale_during_long,
        },
        "worker_error": {
            "error_row_count": len(error_rows),
            "error_263_visible_in_snapshot": error_detected,
            "source_failure_detectable": bool(error_rows and error_detected),
        },
        "cleanup": {
            "worker_joined": worker_joined,
            "state_producer_joined": state_joined,
        },
        "thresholds": {
            "expected_update_period_ms": 50.0,
            "warning_age_ms": 50.0,
            "stale_threshold_ms": 100.0,
            "fatal_unavailable_threshold_ms": None,
            "note": "Diagnostic-only values; no formal safety policy was changed.",
        },
        "result": "PASS" if passed else "BLOCKED",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"wrench_nonblocking_validation_{stamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-software non-blocking wrench fault injection")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    path, payload = run(Path(args.output_dir))
    print(json.dumps({
        "result": payload["result"],
        "long_block": payload["long_block"],
        "worker_error": payload["worker_error"],
        "cleanup": payload["cleanup"],
    }, ensure_ascii=False, indent=2))
    print(f"JSON: {path}")


if __name__ == "__main__":
    main()
