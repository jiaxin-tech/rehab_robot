"""Supervised, strictly read-only long-duration ROKAE wrench characterization.

The child process owns the SDK connection.  A dedicated thread owns every
``getEndTorque`` call while the diagnostic main loop only reads cache/state.
The parent is an outer watchdog; terminating a child detects a timeout but is
not described as cancellation of the underlying native SDK call.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from hardware.windows.rokae_xcore import RokaeRobot
from scripts.wrench_nonblocking_diagnostic import NonBlockingWrenchWorker


CSV_FIELDS = (
    "sequence_id", "cache_sequence_id", "scheduled_time_ns", "call_start_ns",
    "call_end_ns", "query_latency_ms", "host_call_start_monotonic_ns",
    "host_call_end_monotonic_ns", "latency_ms", "success", "error_code",
    "error_message", "robot_state_at_call", "robot_operation_state",
    "latest_rt_sequence", "latest_rt_timestamp", "latest_rt_timestamp_ns",
    "latest_rt_age_ms", "rt_sequence_after_call",
    "rt_updates_during_call", "previous_success_latency_ms",
    "time_since_last_success_ms", "cartesian_force_raw_n",
    "cartesian_torque_raw_nm", "joint_measured_torque_nm",
    "joint_external_torque_nm",
)

CONCURRENCY_CSV_FIELDS = (
    "main_loop_timestamp_ns", "loop_period_ms", "rt_sequence", "rt_timestamp_s",
    "rt_age_ms", "rt_valid", "operation_state", "wrench_sequence",
    "wrench_age_ms", "wrench_valid", "wrench_stale", "wrench_worker_alive",
    "wrench_query_in_flight", "wrench_in_flight_age_ms", "last_wrench_error",
    "last_wrench_error_code",
)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return None
    offset = (len(data) - 1) * fraction
    lower, upper = math.floor(offset), math.ceil(offset)
    if lower == upper:
        return data[lower]
    return data[lower] + (data[upper] - data[lower]) * (offset - lower)


def latency_summary(rows: list[dict[str, Any]], target_hz: float) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    start_periods = [
        (int(rows[index]["host_call_start_monotonic_ns"]) - int(rows[index - 1]["host_call_start_monotonic_ns"])) / 1e6
        for index in range(1, len(rows))
    ]
    deadline_ms = 1000.0 / target_hz
    buckets = {
        "le_5_ms": sum(value <= 5 for value in latencies),
        "gt_5_le_10_ms": sum(5 < value <= 10 for value in latencies),
        "gt_10_le_20_ms": sum(10 < value <= 20 for value in latencies),
        "gt_20_le_50_ms": sum(20 < value <= 50 for value in latencies),
        "gt_50_le_100_ms": sum(50 < value <= 100 for value in latencies),
        "gt_100_le_500_ms": sum(100 < value <= 500 for value in latencies),
        "gt_500_le_1000_ms": sum(500 < value <= 1000 for value in latencies),
        "gt_1000_ms": sum(value > 1000 for value in latencies),
    }
    misses = sum(value > deadline_ms for value in latencies)
    errors_263 = sum(int(row.get("error_code") or -1) == 263 for row in rows)
    recovered_after_263 = 0
    for index, row in enumerate(rows[:-1]):
        if int(row.get("error_code") or -1) == 263 and bool(rows[index + 1]["success"]):
            recovered_after_263 += 1
    return {
        "call_count": len(rows),
        "actual_calls": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "error_count": sum(not bool(row["success"]) for row in rows),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "p99_9": percentile(latencies, 0.999),
            "p99_9_interpretation": (
                "descriptive only; fewer than 1000 calls" if len(latencies) < 1000 else "at least 1000 calls"
            ),
            "max": max(latencies) if latencies else None,
        },
        "latency_buckets": buckets,
        "count_gt_20_ms": sum(value > 20 for value in latencies),
        "count_gt_50_ms": sum(value > 50 for value in latencies),
        "count_gt_100_ms": sum(value > 100 for value in latencies),
        "count_gt_1_s": sum(value > 1000 for value in latencies),
        "latency_gt_20ms_count": sum(value > 20 for value in latencies),
        "latency_gt_50ms_count": sum(value > 50 for value in latencies),
        "latency_gt_100ms_count": sum(value > 100 for value in latencies),
        "latency_gt_1s_count": sum(value > 1000 for value in latencies),
        "deadline_ms": deadline_ms,
        "deadline_miss_count": misses,
        "deadline_miss_ratio": misses / len(latencies) if latencies else None,
        "error_263_count": errors_263,
        "SDK_error_263_count": errors_263,
        "error_263_recovered_on_next_query_count": recovered_after_263,
        "long_blocking_count": sum(value > 1000 for value in latencies),
        "query_start_period_ms": {
            "mean": statistics.fmean(start_periods) if start_periods else None,
            "p50": percentile(start_periods, 0.50),
            "p95": percentile(start_periods, 0.95),
            "p99": percentile(start_periods, 0.99),
            "max": max(start_periods) if start_periods else None,
        },
        "maximum_rt_updates_during_one_query": max(
            (int(row.get("rt_updates_during_call") or 0) for row in rows),
            default=0,
        ),
        "query_latency_note": "query latency is distinct from query start period and data age",
    }


def concurrency_summary(
    ticks: list[dict[str, Any]],
    *,
    freeze_threshold_ms: float = 24.0,
) -> dict[str, Any]:
    loop_periods = [float(row["loop_period_ms"]) for row in ticks if row.get("loop_period_ms") is not None]
    rt_ages = [float(row["rt_age_ms"]) for row in ticks if row.get("rt_age_ms") is not None]
    wrench_ages = [float(row["wrench_age_ms"]) for row in ticks if row.get("wrench_age_ms") is not None]
    source_periods: list[float] = []
    consumer_skipped = 0
    non_monotonic = 0
    frozen_events = 0
    maximum_unchanged_ms = 0.0
    previous_changed_row: dict[str, Any] | None = None
    last_change_tick_ns: int | None = None
    freeze_latched = False
    for row in ticks:
        sequence = row.get("rt_sequence")
        timestamp_s = row.get("rt_timestamp_s")
        tick_ns = int(row["main_loop_timestamp_ns"])
        if sequence is None:
            continue
        if previous_changed_row is None or int(sequence) != int(previous_changed_row["rt_sequence"]):
            if previous_changed_row is not None:
                delta_sequence = int(sequence) - int(previous_changed_row["rt_sequence"])
                previous_timestamp = previous_changed_row.get("rt_timestamp_s")
                if delta_sequence <= 0:
                    non_monotonic += 1
                elif timestamp_s is not None and previous_timestamp is not None:
                    source_periods.append(
                        (float(timestamp_s) - float(previous_timestamp)) * 1000.0 / delta_sequence
                    )
                    consumer_skipped += max(0, delta_sequence - 1)
            previous_changed_row = row
            last_change_tick_ns = tick_ns
            freeze_latched = False
        elif last_change_tick_ns is not None:
            unchanged_ms = (tick_ns - last_change_tick_ns) / 1e6
            maximum_unchanged_ms = max(maximum_unchanged_ms, unchanged_ms)
            if unchanged_ms > freeze_threshold_ms and not freeze_latched:
                frozen_events += 1
                freeze_latched = True

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    first_sequence = next((row.get("rt_sequence") for row in ticks if row.get("rt_sequence") is not None), None)
    last_sequence = next((row.get("rt_sequence") for row in reversed(ticks) if row.get("rt_sequence") is not None), None)
    first_wrench_index = next(
        (index for index, row in enumerate(ticks) if int(row.get("wrench_sequence") or 0) > 0),
        len(ticks),
    )
    first_wrench_sequence = next(
        (row.get("wrench_sequence") for row in ticks if row.get("wrench_sequence") is not None),
        None,
    )
    last_wrench_sequence = next(
        (row.get("wrench_sequence") for row in reversed(ticks) if row.get("wrench_sequence") is not None),
        None,
    )
    loop_mean_ms = statistics.fmean(loop_periods) if loop_periods else None
    return {
        "main_loop_tick_count": len(ticks),
        "main_loop_rate_hz": None if not loop_mean_ms or loop_mean_ms <= 0 else 1000.0 / loop_mean_ms,
        "main_loop_period_ms": stats(loop_periods),
        "rt_sequence_first": first_sequence,
        "rt_sequence_last": last_sequence,
        "rt_sequence_advance": (
            None if first_sequence is None or last_sequence is None else int(last_sequence) - int(first_sequence)
        ),
        "rt_source_period_ms": stats(source_periods),
        "rt_age_ms": stats(rt_ages),
        "consumer_skipped_rt_sequence_count": consumer_skipped,
        "rt_non_monotonic_event_count": non_monotonic,
        "rt_frozen_event_count": frozen_events,
        "rt_freeze_threshold_ms": freeze_threshold_ms,
        "maximum_observed_unchanged_rt_sequence_ms": maximum_unchanged_ms,
        "rt_invalid_tick_count": sum(not bool(row.get("rt_valid")) for row in ticks),
        "non_idle_tick_count": sum(row.get("operation_state") != "IDLE" for row in ticks),
        "wrench_age_ms": stats(wrench_ages),
        "wrench_sequence_first": first_wrench_sequence,
        "wrench_sequence_last": last_wrench_sequence,
        "wrench_sequence_advance": (
            None
            if first_wrench_sequence is None or last_wrench_sequence is None
            else int(last_wrench_sequence) - int(first_wrench_sequence)
        ),
        "wrench_stale_tick_count": sum(bool(row.get("wrench_stale")) for row in ticks),
        "wrench_stale_tick_count_after_first_update": sum(
            bool(row.get("wrench_stale")) for row in ticks[first_wrench_index:]
        ),
        "wrench_invalid_tick_count": sum(not bool(row.get("wrench_valid")) for row in ticks),
        "wrench_invalid_tick_count_after_first_update": sum(
            not bool(row.get("wrench_valid")) for row in ticks[first_wrench_index:]
        ),
        "wrench_worker_dead_tick_count": sum(not bool(row.get("wrench_worker_alive")) for row in ticks),
        "wrench_query_in_flight_tick_count": sum(bool(row.get("wrench_query_in_flight")) for row in ticks),
    }


def _event_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "sequence_id", "scheduled_time_ns", "call_start_ns", "call_end_ns",
            "query_latency_ms", "success", "error_code", "error_message",
            "latest_rt_sequence", "latest_rt_timestamp_ns", "latest_rt_age_ms",
            "rt_sequence_after_call", "rt_updates_during_call",
            "time_since_last_success_ms",
        )
    }


def event_windows(rows: list[dict[str, Any]], radius: int = 10) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        latency_ms = float(row.get("query_latency_ms", row.get("latency_ms", 0.0)))
        error_code = int(row.get("error_code") or -1)
        if latency_ms <= 1000.0 and error_code != 263:
            continue
        result.append({
            "trigger_index": index,
            "trigger_sequence_id": row.get("sequence_id"),
            "trigger": "error_263" if error_code == 263 else "latency_gt_1_s",
            "previous_successful": [
                _event_row(item)
                for item in [candidate for candidate in rows[:index] if bool(candidate.get("success"))][-radius:]
            ],
            "current": _event_row(row),
            "next": [_event_row(item) for item in rows[index + 1:index + radius + 1]],
        })
    return result


def slow_call_concurrency_analysis(
    rows: list[dict[str, Any]],
    ticks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        latency_ms = float(row.get("query_latency_ms", row.get("latency_ms", 0.0)))
        if latency_ms <= 20.0:
            continue
        start_ns = int(row.get("call_start_ns", row["host_call_start_monotonic_ns"]))
        end_ns = int(row.get("call_end_ns", row["host_call_end_monotonic_ns"]))
        during = [tick for tick in ticks if start_ns <= int(tick["main_loop_timestamp_ns"]) <= end_ns]
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        rt_sequences = [int(tick["rt_sequence"]) for tick in during if tick.get("rt_sequence") is not None]
        periods = [float(tick["loop_period_ms"]) for tick in during if tick.get("loop_period_ms") is not None]
        analyses.append({
            "sequence_id": row.get("sequence_id"),
            "latency_ms": latency_ms,
            "success": bool(row.get("success")),
            "error_code": row.get("error_code"),
            "main_loop_ticks_during_call": len(during),
            "main_loop_period_ms_during_call": {
                "p99": percentile(periods, 0.99),
                "max": max(periods) if periods else None,
            },
            "rt_sequence_advance_during_main_loop_window": (
                max(rt_sequences) - min(rt_sequences) if rt_sequences else None
            ),
            "stale_tick_count_during_call": sum(bool(tick.get("wrench_stale")) for tick in during),
            "worker_dead_tick_count_during_call": sum(not bool(tick.get("wrench_worker_alive")) for tick in during),
            "next_query": None if next_row is None else {
                "sequence_id": next_row.get("sequence_id"),
                "success": bool(next_row.get("success")),
                "error_code": next_row.get("error_code"),
                "latency_ms": next_row.get("query_latency_ms", next_row.get("latency_ms")),
            },
            "reconnect_used": False,
        })
    return analyses


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    value = row.pop("value", None) or {}
    output = dict(row)
    for field in (
        "cartesian_force_raw_n", "cartesian_torque_raw_nm",
        "joint_measured_torque_nm", "joint_external_torque_nm",
    ):
        output[field] = json.dumps(value.get(field), ensure_ascii=False) if isinstance(value, dict) else ""
    return output


def _state_snapshot(robot: RokaeRobot) -> dict[str, Any]:
    frame = robot.get_state_frame()
    now_ns = time.perf_counter_ns()
    timestamp_s = frame.host_monotonic_time_s
    age_ms = None if timestamp_s is None else max(0.0, now_ns / 1e9 - timestamp_s) * 1000.0
    return {
        "sequence_id": frame.sequence_id,
        "timestamp_s": timestamp_s,
        "timestamp_ns": None if timestamp_s is None else int(timestamp_s * 1e9),
        "age_ms": age_ms,
        "operation_state": frame.operation_state,
        "valid": frame.valid,
    }


def worker_main(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    json_path = Path(args.json_path)
    concurrency_csv_path = Path(args.concurrency_csv_path)
    concurrency_json_path = Path(args.concurrency_json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    concurrency_ticks: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    watchdog_events: list[dict[str, Any]] = []
    robot = RokaeRobot(args.robot_ip, local_ip=args.local_ip)
    connected = False
    disconnected = False
    worker: NonBlockingWrenchWorker | None = None
    fatal_error: str | None = None
    started_ns = time.perf_counter_ns()
    metadata: dict[str, Any] = {}
    cleanup_error: str | None = None
    acquisition_started_ns: int | None = None
    main_loop_periods_ms: list[float] = []
    previous_main_tick_ns: int | None = None
    maximum_wrench_age_ms: float | None = None
    stale_tick_count = 0
    wrench_rows_since_sync = 0
    concurrency_rows_since_sync = 0
    postcheck: dict[str, Any] = {}
    print(
        "STRICT READ-ONLY APIs: connectToRobot, robotInfo, startReceiveRobotState/update/"
        "getStateData, operationState, powerState, operateMode, getEndTorque(world), "
        "stopReceiveRobotState, disconnectFromRobot",
        flush=True,
    )
    print(f"Robot IP={args.robot_ip}; Local RT IP={args.local_ip}; target={args.target_hz} Hz; duration={args.duration_s} s", flush=True)
    stream = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    stream.flush()
    concurrency_stream = concurrency_csv_path.open("w", newline="", encoding="utf-8")
    concurrency_writer = csv.DictWriter(
        concurrency_stream,
        fieldnames=CONCURRENCY_CSV_FIELDS,
        extrasaction="ignore",
    )
    concurrency_writer.writeheader()
    concurrency_stream.flush()

    def record(raw_row: dict[str, Any]) -> None:
        nonlocal wrench_rows_since_sync
        row = _csv_row(dict(raw_row))
        with rows_lock:
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            wrench_rows_since_sync += 1
            if wrench_rows_since_sync >= max(1, int(args.target_hz)):
                os.fsync(stream.fileno())
                wrench_rows_since_sync = 0

    try:
        robot.connect()
        connected = True
        native = robot._robot
        sdk = robot._sdk
        operation = robot._call("operationState", native.operationState)
        power = robot._call("powerState", native.powerState)
        operate_mode = robot._call("operateMode", native.operateMode)
        info = robot._robot_info
        metadata = {
            "sdk_version": robot._sdk_version,
            "controller_version": str(info.version),
            "robot_model": str(info.type),
            "robot_serial": str(info.id),
            "operation_state": operation.name,
            "power_state": power.name,
            "operate_mode": operate_mode.name,
        }
        print(
            f"SDK={robot._sdk_version}; controller={info.version}; model={info.type}; "
            f"operation={operation.name}; power={power.name}; operateMode={operate_mode.name}",
            flush=True,
        )
        if operation != sdk.OperationState.idle:
            raise RuntimeError(f"requires IDLE, observed {operation.name}")
        if power != sdk.PowerState.on:
            raise RuntimeError(f"requires powerState=on for this approved protocol, observed {power.name}")
        if operate_mode != sdk.OperateMode.automatic:
            raise RuntimeError(f"requires operateMode=automatic, observed {operate_mode.name}")
        worker = NonBlockingWrenchWorker(
            lambda: robot.get_end_wrench("world"),
            target_hz=args.target_hz,
            stale_threshold_ms=args.stale_threshold_ms,
            on_result=record,
            state_snapshot=lambda: _state_snapshot(robot),
        )
        worker.start()
        acquisition_started_ns = time.perf_counter_ns()
        deadline_ns = acquisition_started_ns + int(args.duration_s * 1e9)
        next_tick_ns = acquisition_started_ns
        watchdog_latched = False
        while time.perf_counter_ns() < deadline_ns:
            now_ns = time.perf_counter_ns()
            if previous_main_tick_ns is not None:
                main_loop_periods_ms.append((now_ns - previous_main_tick_ns) / 1e6)
            previous_main_tick_ns = now_ns
            state = _state_snapshot(robot)
            snapshot = worker.cache.snapshot(now_ns)
            loop_period_ms = (
                None if len(main_loop_periods_ms) == 0 else main_loop_periods_ms[-1]
            )
            concurrency_row = {
                "main_loop_timestamp_ns": now_ns,
                "loop_period_ms": loop_period_ms,
                "rt_sequence": state.get("sequence_id"),
                "rt_timestamp_s": state.get("timestamp_s"),
                "rt_age_ms": state.get("age_ms"),
                "rt_valid": state.get("valid"),
                "operation_state": state.get("operation_state"),
                "wrench_sequence": snapshot.sequence_id,
                "wrench_age_ms": snapshot.age_ms,
                "wrench_valid": snapshot.valid,
                "wrench_stale": snapshot.stale,
                "wrench_worker_alive": worker.alive,
                "wrench_query_in_flight": snapshot.query_in_flight,
                "wrench_in_flight_age_ms": snapshot.in_flight_age_ms,
                "last_wrench_error": snapshot.last_error,
                "last_wrench_error_code": snapshot.last_error_code,
            }
            concurrency_ticks.append(concurrency_row)
            concurrency_writer.writerow(concurrency_row)
            concurrency_rows_since_sync += 1
            if concurrency_rows_since_sync >= 100:
                concurrency_stream.flush()
                os.fsync(concurrency_stream.fileno())
                concurrency_rows_since_sync = 0
            if snapshot.age_ms is not None:
                maximum_wrench_age_ms = (
                    snapshot.age_ms
                    if maximum_wrench_age_ms is None
                    else max(maximum_wrench_age_ms, snapshot.age_ms)
                )
            if snapshot.stale:
                stale_tick_count += 1
            if state.get("operation_state") != "IDLE":
                raise RuntimeError(f"robot left IDLE: {state.get('operation_state')}")
            timed_out = bool(
                snapshot.query_in_flight
                and snapshot.in_flight_age_ms is not None
                and snapshot.in_flight_age_ms > args.call_watchdog_s * 1000.0
            )
            if timed_out and not watchdog_latched:
                watchdog_events.append({
                    "detected_ns": now_ns,
                    "in_flight_age_ms": snapshot.in_flight_age_ms,
                    "rt_sequence": state.get("sequence_id"),
                    "rt_age_ms": state.get("age_ms"),
                    "meaning": "timeout detected; underlying native call not claimed cancelled",
                })
                watchdog_latched = True
            if not snapshot.query_in_flight:
                watchdog_latched = False
            next_tick_ns += 10_000_000
            remaining_ns = next_tick_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1e9)
            else:
                next_tick_ns = time.perf_counter_ns()
        worker.request_stop()
        worker.join(timeout=args.call_watchdog_s + 2.0)
        if worker.alive:
            raise RuntimeError("wrench worker remained blocked after stop; native call cancellation unconfirmed")
        post_operation = robot._call("operationState", native.operationState)
        post_power = robot._call("powerState", native.powerState)
        post_operate_mode = robot._call("operateMode", native.operateMode)
        postcheck = {
            "operation_state": post_operation.name,
            "power_state": post_power.name,
            "operate_mode": post_operate_mode.name,
        }
        if post_operation != sdk.OperationState.idle:
            raise RuntimeError(f"postcheck requires IDLE, observed {post_operation.name}")
        if post_power != sdk.PowerState.on:
            raise RuntimeError(f"postcheck requires powerState=on, observed {post_power.name}")
        if post_operate_mode != sdk.OperateMode.automatic:
            raise RuntimeError(f"postcheck requires operateMode=automatic, observed {post_operate_mode.name}")
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}:{exc}"
    finally:
        if worker is not None:
            worker.request_stop()
        safe_to_disconnect = worker is None or not worker.alive
        if connected and safe_to_disconnect:
            try:
                robot.disconnect()
                disconnected = not robot.is_connected
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}:{exc}"
        elif connected:
            cleanup_error = "disconnect_not_called_concurrently_with_blocked_native_wrench_call"
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        concurrency_stream.flush()
        os.fsync(concurrency_stream.fileno())
        concurrency_stream.close()

    finished_ns = time.perf_counter_ns()
    with rows_lock:
        summary = latency_summary(rows, args.target_hz)
        critical_windows = event_windows(rows)
        slow_calls = slow_call_concurrency_analysis(rows, concurrency_ticks)
    concurrency_result = concurrency_summary(concurrency_ticks)
    scheduled_calls = int(math.floor(args.duration_s * args.target_hz))
    summary["scheduled_calls"] = scheduled_calls
    payload = {
        "schema_version": 1,
        "diagnostic": "wrench_longrun",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "robot_ip": args.robot_ip,
        "local_ip": args.local_ip,
        "target_hz": args.target_hz,
        "requested_duration_s": args.duration_s,
        "duration_s": args.duration_s,
        "scheduled_calls": scheduled_calls,
        "actual_calls": len(rows),
        "observed_duration_s": (finished_ns - started_ns) / 1e9,
        "acquisition_duration_s": (
            None if acquisition_started_ns is None else max(0.0, finished_ns - acquisition_started_ns) / 1e9
        ),
        "warning_age_ms": args.warning_age_ms,
        "stale_threshold_ms": args.stale_threshold_ms,
        "call_watchdog_s": args.call_watchdog_s,
        "metadata": metadata,
        "postcheck": postcheck,
        "summary": summary,
        "cache_observation": {
            "maximum_wrench_age_ms": maximum_wrench_age_ms,
            "stale_main_loop_tick_count": stale_tick_count,
            "main_loop_period_ms": {
                "mean": statistics.fmean(main_loop_periods_ms) if main_loop_periods_ms else None,
                "p95": percentile(main_loop_periods_ms, 0.95),
                "p99": percentile(main_loop_periods_ms, 0.99),
                "max": max(main_loop_periods_ms) if main_loop_periods_ms else None,
            },
        },
        "watchdog_events": watchdog_events,
        "critical_event_windows_previous_10_current_next_10": critical_windows,
        "slow_call_concurrency_analysis": slow_calls,
        "concurrency_output": {
            "csv_path": str(concurrency_csv_path),
            "json_path": str(concurrency_json_path),
        },
        "fatal_error": fatal_error,
        "cleanup": {
            "disconnect_attempted": bool(connected and (worker is None or not worker.alive)),
            "disconnected_confirmed": disconnected,
            "error": cleanup_error,
        },
        "read_only": True,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    concurrency_payload = {
        "schema_version": 1,
        "diagnostic": "wrench_realtime_concurrency",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "robot_ip": args.robot_ip,
        "local_ip": args.local_ip,
        "target_wrench_hz": args.target_hz,
        "target_main_loop_hz": 100.0,
        "duration_s": args.duration_s,
        "metadata": metadata,
        "postcheck": postcheck,
        "summary": concurrency_result,
        "slow_call_concurrency_analysis": slow_calls,
        "fatal_error": fatal_error,
        "cleanup": payload["cleanup"],
        "read_only": True,
    }
    concurrency_json_path.write_text(
        json.dumps(concurrency_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "fatal_error": fatal_error, "cleanup": payload["cleanup"]}, ensure_ascii=False), flush=True)
    return 0 if fatal_error is None and disconnected else 2


def supervisor_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rate_name = f"{args.target_hz:g}".replace(".", "p")
    csv_path = output_dir / f"wrench_longrun_{rate_name}hz_{stamp}.csv"
    json_path = output_dir / f"wrench_longrun_{rate_name}hz_{stamp}.json"
    concurrency_csv_path = output_dir / f"wrench_realtime_concurrency_{stamp}.csv"
    concurrency_json_path = output_dir / f"wrench_realtime_concurrency_{stamp}.json"
    command = [
        sys.executable, "-u", "-B", "-m", "scripts.characterize_wrench_longrun",
        "--worker", "--robot-ip", args.robot_ip, "--local-ip", args.local_ip,
        "--target-hz", str(args.target_hz), "--duration-s", str(args.duration_s),
        "--warning-age-ms", str(args.warning_age_ms),
        "--stale-threshold-ms", str(args.stale_threshold_ms),
        "--call-watchdog-s", str(args.call_watchdog_s),
        "--csv-path", str(csv_path), "--json-path", str(json_path),
        "--concurrency-csv-path", str(concurrency_csv_path),
        "--concurrency-json-path", str(concurrency_json_path),
    ]
    print("Outer watchdog launching child process; timeout detection does not imply native-call cancellation.", flush=True)
    child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    hard_timeout_s = args.duration_s + args.call_watchdog_s + 60.0
    try:
        return child.wait(timeout=hard_timeout_s)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5.0)
        watchdog_path = output_dir / f"wrench_longrun_{rate_name}hz_{stamp}_supervisor.json"
        watchdog_path.write_text(json.dumps({
            "diagnostic": "wrench_longrun_outer_watchdog",
            "timeout_detected": True,
            "underlying_sdk_call_cancelled": False,
            "child_terminated": True,
            "hard_timeout_s": hard_timeout_s,
            "csv_path": str(csv_path),
            "child_json_path": str(json_path),
            "concurrency_csv_path": str(concurrency_csv_path),
            "concurrency_json_path": str(concurrency_json_path),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Outer watchdog terminated child; SDK cancellation and disconnect are unconfirmed. {watchdog_path}")
        return 124


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Supervised read-only ROKAE wrench long-run")
    result.add_argument("--robot-ip", required=True)
    result.add_argument("--local-ip", required=True)
    result.add_argument("--target-hz", type=float, choices=(20.0, 50.0), required=True)
    result.add_argument("--duration-s", type=float, default=900.0)
    result.add_argument("--warning-age-ms", type=float, default=50.0)
    result.add_argument("--stale-threshold-ms", type=float, default=100.0)
    result.add_argument("--call-watchdog-s", type=float, default=12.0)
    result.add_argument("--output-dir", default="diagnostics")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--csv-path", help=argparse.SUPPRESS)
    result.add_argument("--json-path", help=argparse.SUPPRESS)
    result.add_argument("--concurrency-csv-path", help=argparse.SUPPRESS)
    result.add_argument("--concurrency-json-path", help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.duration_s <= 0 or args.call_watchdog_s <= 0:
        raise SystemExit("duration and watchdog must be positive")
    if args.worker:
        if not all((args.csv_path, args.json_path, args.concurrency_csv_path, args.concurrency_json_path)):
            raise SystemExit("worker requires output paths")
        raise SystemExit(worker_main(args))
    raise SystemExit(supervisor_main(args))


if __name__ == "__main__":
    main()
