"""Strict read-only RT latest-snapshot IPC freshness diagnostic.

This script never issues motion, mode, power, reset, or clear commands.  The
live RT child exclusively performs connect, read-only identity/status calls,
startReceiveRobotState, updateRobotState/getStateData, stopReceiveRobotState,
and disconnect.  Concurrent mode adds one separately owned, read-only 20 Hz
getEndTorque(world) process.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

from scripts.wrench_process_isolation import RtProcessSupervisor, WrenchProcessSupervisor


CSV_FIELDS = (
    "case",
    "sample_index",
    "main_loop_timestamp_ns",
    "supervisor_loop_period_ms",
    "rt_sequence",
    "source_or_receive_timestamp_ns",
    "publish_timestamp_ns",
    "supervisor_receive_timestamp_ns",
    "observed_source_period_per_sequence_ms",
    "observed_publish_interval_per_sequence_ms",
    "rt_age_ms",
    "rt_ipc_age_ms",
    "rt_publish_to_receive_age_ms",
    "new_snapshot_received",
    "publish_count",
    "publish_success_count",
    "receive_count",
    "overwrite_count",
    "publish_drop_count",
    "rt_valid",
    "rt_stale",
    "operation_state",
    "rt_worker_state",
    "rt_worker_alive",
    "rt_worker_hung",
    "rt_heartbeat_age_ms",
    "rt_last_error_code",
    "rt_last_error",
    "wrench_sequence",
    "wrench_last_success_ns",
    "wrench_age_ms",
    "wrench_valid",
    "wrench_stale",
    "wrench_worker_state",
    "wrench_worker_alive",
    "wrench_worker_hung",
    "wrench_heartbeat_age_ms",
    "wrench_last_error_code",
    "wrench_last_error",
)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize_values(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "p95": None,
            "p99": None,
            "max": None,
            "count_gt_20ms": 0,
            "count_gt_30ms": 0,
            "count_gt_50ms": 0,
            "count_gt_100ms": 0,
        }
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p95": _percentile(clean, 0.95),
        "p99": _percentile(clean, 0.99),
        "max": max(clean),
        "count_gt_20ms": sum(value > 20.0 for value in clean),
        "count_gt_30ms": sum(value > 30.0 for value in clean),
        "count_gt_50ms": sum(value > 50.0 for value in clean),
        "count_gt_100ms": sum(value > 100.0 for value in clean),
    }


def _safe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _source_summary(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    child = dict(metadata.get("source_period_ms") or {})
    if int(child.get("count") or 0) > 0:
        result = summarize_values([])
        result.update(child)
        result["count_gt_10ms"] = int(metadata.get("source_period_gt_10ms_count", 0))
        result["count_gt_20ms"] = int(metadata.get("source_period_gt_20ms_count", 0))
        result["count_gt_30ms"] = int(metadata.get("source_period_gt_30ms_count", 0))
        result["count_gt_50ms"] = int(metadata.get("source_period_gt_50ms_count", 0))
        result["count_gt_100ms"] = int(metadata.get("source_period_gt_100ms_count", 0))
        return result
    return summarize_values(row.get("observed_source_period_per_sequence_ms") for row in rows)


def source_acceptance(summary: dict[str, Any]) -> bool:
    """Apply the requested mean/P99 target while reporting rare freezes separately."""

    source = summary["source_period_ms"]
    return bool(
        int(source.get("count") or 0) >= 100
        and 6.0 <= float(source.get("mean") or 0.0) <= 10.0
        and float(source.get("p99") or 1e9) < 10.0
        and float(source.get("max") or 1e9) < 100.0
        and int(summary.get("sequence_advance") or 0) > 0
    )


def summarize_rt(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    fresh = [row for row in rows if row.get("new_snapshot_received")]
    source = _source_summary(metadata, rows)
    publish = dict(metadata.get("publish_interval_ms") or {})
    if int(publish.get("count") or 0) == 0:
        publish = summarize_values(
            row.get("observed_publish_interval_per_sequence_ms") for row in rows
        )
    ipc = summarize_values(row.get("rt_ipc_age_ms") for row in fresh)
    publish_to_receive = summarize_values(
        row.get("rt_publish_to_receive_age_ms") for row in fresh
    )
    supervisor = summarize_values(row.get("supervisor_loop_period_ms") for row in rows)
    duration_s = 0.0
    if len(rows) > 1:
        duration_s = (
            int(rows[-1]["main_loop_timestamp_ns"])
            - int(rows[0]["main_loop_timestamp_ns"])
        ) / 1e9
    last = rows[-1] if rows else {}
    sequence_advance = 0
    if rows:
        sequence_advance = int(rows[-1]["rt_sequence"]) - int(rows[0]["rt_sequence"])
    source_reliable = source_acceptance(
        {"source_period_ms": source, "sequence_advance": sequence_advance}
    )
    ipc_fresh = bool(
        int(ipc["count"]) >= 100
        and float(ipc["p95"] or 1e9) < 20.0
        and float(ipc["p99"] or 1e9) < 30.0
        and int(ipc["count_gt_50ms"]) <= 1
        and int(ipc["count_gt_100ms"]) == 0
    )
    supervisor_reliable = bool(
        duration_s > 0.0
        and len(rows) / duration_s >= 90.0
        and float(supervisor.get("p99") or 1e9) < 30.0
        and float(supervisor.get("max") or 1e9) < 100.0
    )
    return {
        "duration_s": duration_s,
        "supervisor_loop_rate_hz": None if duration_s <= 0 else len(rows) / duration_s,
        "sequence_advance": sequence_advance,
        "source_period_ms": source,
        "publish_interval_ms": publish,
        "ipc_age_ms": ipc,
        "publish_to_receive_age_ms": publish_to_receive,
        "supervisor_loop_period_ms": supervisor,
        "publish_count": int(last.get("publish_count") or 0),
        "publish_success_count": int(last.get("publish_success_count") or 0),
        "receive_count": int(last.get("receive_count") or 0),
        "overwrite_count": int(last.get("overwrite_count") or 0),
        "publish_drop_count": int(last.get("publish_drop_count") or 0),
        "rt_invalid_tick_count": sum(not bool(row.get("rt_valid")) for row in rows),
        "rt_worker_hung_tick_count": sum(bool(row.get("rt_worker_hung")) for row in rows),
        "source_freeze_count_gt_20ms": int(metadata.get("source_period_gt_20ms_count", 0)),
        "source_reliable": source_reliable,
        "ipc_fresh": ipc_fresh,
        "supervisor_reliable": supervisor_reliable,
        "no_obvious_freshness_tail": bool(ipc_fresh and sequence_advance > 0),
    }


def summarize_wrench(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"tested": False}
    sequence_advance = int(rows[-1]["wrench_sequence"]) - int(rows[0]["wrench_sequence"])
    return {
        "tested": True,
        "sequence_advance": sequence_advance,
        "age_ms": summarize_values(row.get("wrench_age_ms") for row in rows),
        "heartbeat_age_ms": summarize_values(
            row.get("wrench_heartbeat_age_ms") for row in rows
        ),
        "valid_tick_count": sum(bool(row.get("wrench_valid")) for row in rows),
        "stale_tick_count": sum(bool(row.get("wrench_stale")) for row in rows),
        "worker_hung_tick_count": sum(bool(row.get("wrench_worker_hung")) for row in rows),
        "natural_block_observed": any(bool(row.get("wrench_worker_hung")) for row in rows),
        "error_codes_seen": sorted(
            {
                int(row["wrench_last_error_code"])
                for row in rows
                if row.get("wrench_last_error_code") is not None
            }
        ),
    }


def _row(
    *,
    case: str,
    sample_index: int,
    tick_ns: int,
    loop_period_ms: float | None,
    rt: RtProcessSupervisor,
    wrench: WrenchProcessSupervisor | None,
    previous_fresh: dict[str, int | None],
) -> dict[str, Any]:
    rt_observation = rt.poll(tick_ns)
    wrench_observation = None if wrench is None else wrench.poll(tick_ns)
    source_period = None
    publish_interval = None
    if rt_observation.new_snapshot_received:
        previous_sequence = previous_fresh.get("sequence")
        sequence_delta = (
            None
            if previous_sequence is None
            else rt_observation.rt_sequence - int(previous_sequence)
        )
        if sequence_delta is not None and sequence_delta > 0:
            previous_source = previous_fresh.get("source_ns")
            previous_publish = previous_fresh.get("publish_ns")
            if previous_source is not None and rt_observation.source_or_receive_timestamp_ns is not None:
                source_period = (
                    rt_observation.source_or_receive_timestamp_ns - int(previous_source)
                ) / sequence_delta / 1e6
            if previous_publish is not None and rt_observation.publish_timestamp_ns is not None:
                publish_interval = (
                    rt_observation.publish_timestamp_ns - int(previous_publish)
                ) / sequence_delta / 1e6
        previous_fresh["sequence"] = rt_observation.rt_sequence
        previous_fresh["source_ns"] = rt_observation.source_or_receive_timestamp_ns
        previous_fresh["publish_ns"] = rt_observation.publish_timestamp_ns
    return {
        "case": case,
        "sample_index": sample_index,
        "main_loop_timestamp_ns": tick_ns,
        "supervisor_loop_period_ms": loop_period_ms,
        "rt_sequence": rt_observation.rt_sequence,
        "source_or_receive_timestamp_ns": rt_observation.source_or_receive_timestamp_ns,
        "publish_timestamp_ns": rt_observation.publish_timestamp_ns,
        "supervisor_receive_timestamp_ns": rt_observation.supervisor_receive_timestamp_ns,
        "observed_source_period_per_sequence_ms": source_period,
        "observed_publish_interval_per_sequence_ms": publish_interval,
        "rt_age_ms": rt_observation.rt_age_ms,
        "rt_ipc_age_ms": rt_observation.rt_ipc_age_ms,
        "rt_publish_to_receive_age_ms": rt_observation.rt_publish_to_receive_age_ms,
        "new_snapshot_received": rt_observation.new_snapshot_received,
        "publish_count": rt_observation.publish_count,
        "publish_success_count": rt_observation.publish_success_count,
        "receive_count": rt_observation.receive_count,
        "overwrite_count": rt_observation.overwrite_count,
        "publish_drop_count": rt_observation.publish_drop_count,
        "rt_valid": rt_observation.rt_valid,
        "rt_stale": rt_observation.rt_stale,
        "operation_state": rt_observation.operation_state,
        "rt_worker_state": rt_observation.worker_state,
        "rt_worker_alive": rt_observation.worker_alive,
        "rt_worker_hung": rt_observation.worker_hung,
        "rt_heartbeat_age_ms": rt_observation.heartbeat_age_ms,
        "rt_last_error_code": rt_observation.last_error_code,
        "rt_last_error": rt_observation.last_error,
        "wrench_sequence": 0 if wrench_observation is None else wrench_observation.wrench_sequence,
        "wrench_last_success_ns": None if wrench_observation is None else wrench_observation.last_wrench_success_ns,
        "wrench_age_ms": None if wrench_observation is None else wrench_observation.wrench_age_ms,
        "wrench_valid": False if wrench_observation is None else wrench_observation.wrench_valid,
        "wrench_stale": True if wrench_observation is None else wrench_observation.wrench_stale,
        "wrench_worker_state": "not_tested" if wrench_observation is None else wrench_observation.worker_state,
        "wrench_worker_alive": False if wrench_observation is None else wrench_observation.worker_alive,
        "wrench_worker_hung": False if wrench_observation is None else wrench_observation.worker_hung,
        "wrench_heartbeat_age_ms": None if wrench_observation is None else wrench_observation.heartbeat_age_ms,
        "wrench_last_error_code": None if wrench_observation is None else wrench_observation.last_error_code,
        "wrench_last_error": None if wrench_observation is None else wrench_observation.last_error,
    }


def _wait_for_rt(rt: RtProcessSupervisor, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = rt.poll()
        if observation.rt_valid and observation.operation_state == "IDLE":
            return
        if observation.worker_exitcode is not None or observation.worker_hung:
            break
        time.sleep(0.01)
    observation = rt.poll()
    raise RuntimeError(
        "RT did not become valid/read-only-idle: "
        f"state={observation.worker_state}, error={observation.last_error}"
    )


def _collect(
    *,
    case: str,
    duration_s: float,
    rt: RtProcessSupervisor,
    wrench: WrenchProcessSupervisor | None,
    writer: csv.DictWriter,
    stream: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(duration_s * 1e9)
    next_tick_ns = started_ns
    previous_tick_ns: int | None = None
    previous_fresh: dict[str, int | None] = {
        "sequence": None,
        "source_ns": None,
        "publish_ns": None,
    }
    next_progress_ns = started_ns + 30_000_000_000
    stop_reason: str | None = None
    while time.perf_counter_ns() < deadline_ns:
        tick_ns = time.perf_counter_ns()
        loop_period_ms = (
            None if previous_tick_ns is None else (tick_ns - previous_tick_ns) / 1e6
        )
        previous_tick_ns = tick_ns
        row = _row(
            case=case,
            sample_index=len(rows),
            tick_ns=tick_ns,
            loop_period_ms=loop_period_ms,
            rt=rt,
            wrench=wrench,
            previous_fresh=previous_fresh,
        )
        rows.append(row)
        writer.writerow(row)
        if len(rows) % 100 == 0:
            stream.flush()
            os.fsync(stream.fileno())
        if not row["rt_worker_alive"] and row["rt_worker_state"] != "not_started":
            stop_reason = "rt_worker_died"
            break
        if row["rt_worker_hung"]:
            stop_reason = "rt_worker_hung"
            break
        if tick_ns >= next_progress_ns:
            elapsed_s = (tick_ns - started_ns) / 1e9
            print(
                f"{case}: {elapsed_s:.0f}/{duration_s:.0f}s, "
                f"rt_seq={row['rt_sequence']}, ipc_age_ms={row['rt_ipc_age_ms']}, "
                f"wrench_state={row['wrench_worker_state']}",
                flush=True,
            )
            next_progress_ns += 30_000_000_000
        next_tick_ns += 10_000_000
        remaining_ns = next_tick_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        else:
            next_tick_ns = time.perf_counter_ns() + 10_000_000
    return rows, stop_reason


def run_offline(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rt_ipc_freshness_offline_{_stamp()}.json"
    rt = RtProcessSupervisor(stale_age_ms=50.0)
    normal_rows: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {}
    try:
        rt.start({"offline": True, "publish_hz": 125.0})
        _wait_for_rt(rt, 10.0)
        previous_tick: int | None = None
        previous_fresh: dict[str, int | None] = {
            "sequence": None,
            "source_ns": None,
            "publish_ns": None,
        }
        started = time.perf_counter_ns()
        next_tick = started
        while time.perf_counter_ns() - started < 3_000_000_000:
            tick = time.perf_counter_ns()
            normal_rows.append(
                _row(
                    case="offline_normal",
                    sample_index=len(normal_rows),
                    tick_ns=tick,
                    loop_period_ms=None if previous_tick is None else (tick - previous_tick) / 1e6,
                    rt=rt,
                    wrench=None,
                    previous_fresh=previous_fresh,
                )
            )
            previous_tick = tick
            next_tick += 10_000_000
            remaining = next_tick - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
        before_pause = rt.poll().to_dict()
        time.sleep(0.35)
        after_pause = rt.poll().to_dict()
        time.sleep(0.02)
        recovered = rt.poll().to_dict()
        cleanup = rt.stop_normally(5.0)
        if not cleanup.get("worker_exited"):
            cleanup = {**cleanup, **rt.terminate()}
        metadata = rt.metadata
    finally:
        if rt.alive:
            cleanup = rt.terminate()
        rt.poll()
        metadata = rt.metadata
        rt.close()
    normal_summary = summarize_rt(normal_rows, metadata)
    pause_sequence_advance = int(after_pause["rt_sequence"]) - int(before_pause["rt_sequence"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline",
        "architecture": {
            "rt_process_single_thread": bool(metadata.get("rt_process_single_thread")),
            "rt_data_path": "update/get/timestamp -> fixed shared-memory latest snapshot",
            "rt_ipc_transport": metadata.get("rt_ipc_transport"),
            "timestamp_source": metadata.get("timestamp_source"),
            "controller_timestamp_used": False,
        },
        "normal_summary": normal_summary,
        "paused_consumer_injection": {
            "pause_s": 0.35,
            "sequence_advance": pause_sequence_advance,
            "before": before_pause,
            "after": after_pause,
            "recovered": recovered,
            "latest_snapshot_recovered": bool(
                pause_sequence_advance >= 30
                and after_pause["new_snapshot_received"]
                and after_pause["rt_ipc_age_ms"] is not None
                and float(after_pause["rt_ipc_age_ms"]) < 30.0
            ),
            "overwrite_is_ipc_replacement_not_network_loss": True,
        },
        "cleanup": cleanup,
    }
    payload["offline_pass"] = bool(
        payload["architecture"]["rt_process_single_thread"]
        and payload["architecture"]["rt_ipc_transport"]
        == "fixed_shared_memory_latest_snapshot"
        and normal_summary["ipc_fresh"]
        and payload["paused_consumer_injection"]["latest_snapshot_recovered"]
        and int(after_pause["overwrite_count"]) > int(before_pause["overwrite_count"])
        and int(after_pause["publish_drop_count"]) == 0
        and cleanup.get("worker_exited")
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"offline_pass={payload['offline_pass']} output={path}", flush=True)
    return path


def run_live(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    case = "rtonly" if args.mode == "rt-only" else "concurrent"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    csv_path = output_dir / f"rt_ipc_freshness_{case}_{stamp}.csv"
    json_path = output_dir / f"rt_ipc_freshness_{case}_{stamp}.json"
    print(
        "STRICT READ-ONLY: connect/identity/status/RT state/disconnect only; "
        "no motion, Servo/Move/Jog, trajectory, mode/power, reset, or clear calls.",
        flush=True,
    )
    rt = RtProcessSupervisor(
        stale_age_ms=50.0,
        worker_hung_ms=750.0,
        worker_startup_hung_ms=20_000.0,
    )
    wrench = None
    if args.mode == "concurrent":
        wrench = WrenchProcessSupervisor(
            stale_age_ms=150.0,
            worker_hung_ms=750.0,
            worker_startup_hung_ms=20_000.0,
        )
    rt_cleanup: dict[str, Any] = {}
    wrench_cleanup: dict[str, Any] = {}
    fatal_error: str | None = None
    stop_reason: str | None = None
    rows: list[dict[str, Any]] = []
    stream = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    try:
        rt.start(
            {
                "robot_ip": args.robot_ip,
                "local_ip": args.local_ip,
                "robot_class": args.robot_class,
                "state_interval_ms": 8,
            }
        )
        _wait_for_rt(rt, 20.0)
        time.sleep(1.0)
        if wrench is not None:
            wrench.start(
                mode="live",
                config={
                    "robot_ip": args.robot_ip,
                    "local_ip": "",
                    "robot_class": args.robot_class,
                    "target_hz": 20.0,
                },
            )
        rows, stop_reason = _collect(
            case=case,
            duration_s=args.duration,
            rt=rt,
            wrench=wrench,
            writer=writer,
            stream=stream,
        )
        if wrench is not None:
            wrench_observation = wrench.poll()
            if wrench_observation.worker_hung:
                wrench_cleanup = wrench.terminate()
            else:
                wrench_cleanup = wrench.stop_normally(5.0)
                if not wrench_cleanup.get("worker_exited"):
                    wrench_cleanup = {**wrench_cleanup, **wrench.terminate()}
        rt_cleanup = rt.stop_normally(5.0)
        if not rt_cleanup.get("worker_exited"):
            rt_cleanup = {**rt_cleanup, **rt.terminate()}
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}:{exc}"
    finally:
        if wrench is not None and wrench.alive:
            wrench_cleanup = wrench.terminate()
        if rt.alive:
            rt_cleanup = rt.terminate()
        if wrench is not None:
            wrench.poll()
        rt.poll()
        metadata = rt.metadata
        if wrench is not None:
            wrench.close()
        rt.close()
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
    rt_summary = summarize_rt(rows, metadata)
    wrench_summary = summarize_wrench(rows) if wrench is not None else {"tested": False}
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "strict_read_only": True,
        "requested_duration_s": args.duration,
        "robot_ip": args.robot_ip,
        "local_ip": args.local_ip,
        "architecture": {
            "rt_process_single_thread": bool(metadata.get("rt_process_single_thread")),
            "rt_ipc_transport": metadata.get("rt_ipc_transport"),
            "timestamp_source": metadata.get("timestamp_source"),
            "controller_timestamp_used": False,
            "sdk_objects_shared_across_processes": False,
        },
        "rt_worker_metadata": metadata,
        "rt_summary": rt_summary,
        "wrench_summary": wrench_summary,
        "stop_reason": stop_reason,
        "fatal_error": fatal_error,
        "rt_cleanup": rt_cleanup,
        "wrench_cleanup": wrench_cleanup,
    }
    payload["case_pass"] = bool(
        fatal_error is None
        and stop_reason is None
        and payload["architecture"]["rt_process_single_thread"]
        and rt_summary["source_reliable"]
        and rt_summary["ipc_fresh"]
        and rt_summary["supervisor_reliable"]
        and rt_cleanup.get("worker_exited")
        and rt_cleanup.get("graceful_disconnect_confirmed")
    )
    payload["eligible_for_concurrent_test"] = bool(
        args.mode == "rt-only"
        and fatal_error is None
        and stop_reason is None
        and rt_summary["no_obvious_freshness_tail"]
        and rt_summary["sequence_advance"] > 0
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"case_pass={payload['case_pass']} eligible_for_concurrent_test="
        f"{payload['eligible_for_concurrent_test']} csv={csv_path} json={json_path}",
        flush=True,
    )
    return csv_path, json_path, payload


def _metric(summary: dict[str, Any], family: str, name: str) -> str:
    value = summary.get(family, {}).get(name)
    return "n/a" if value is None else f"{float(value):.3f}"


def run_analysis(args: argparse.Namespace) -> Path:
    rt_only = json.loads(Path(args.rt_only_json).read_text(encoding="utf-8"))
    concurrent = None
    if args.concurrent_json:
        concurrent = json.loads(Path(args.concurrent_json).read_text(encoding="utf-8"))
    a = rt_only["rt_summary"]
    b = None if concurrent is None else concurrent["rt_summary"]
    a_source_reliable = source_acceptance(a)
    b_source_reliable = None if b is None else source_acceptance(b)
    single_thread = bool(rt_only["architecture"]["rt_process_single_thread"])
    if concurrent is not None:
        single_thread = single_thread and bool(
            concurrent["architecture"]["rt_process_single_thread"]
        )
    if b is None:
        degrades: bool | str = "not_enough_evidence"
    else:
        a_p99 = float(a["ipc_age_ms"]["p99"] or 0.0)
        b_p99 = float(b["ipc_age_ms"]["p99"] or 0.0)
        degrades = bool(
            (a["ipc_fresh"] and not b["ipc_fresh"])
            or b_p99 > max(30.0, a_p99 * 1.5)
            or int(b["ipc_age_ms"]["count_gt_50ms"])
            > int(a["ipc_age_ms"]["count_gt_50ms"]) + 2
        )
    if a_source_reliable and a["ipc_fresh"]:
        root_cause = "rt_process"
    elif a_source_reliable and not a["ipc_fresh"]:
        root_cause = "windows_scheduling" if not a["supervisor_reliable"] else "ipc"
    elif b is not None and a_source_reliable and not b_source_reliable:
        root_cause = "concurrent_sdk_sessions"
    else:
        root_cause = "unknown"
    if b is not None and a_source_reliable and a["ipc_fresh"] and b_source_reliable and b["ipc_fresh"]:
        status = "PASS"
    elif a_source_reliable or a["ipc_fresh"]:
        status = "PARTIAL"
    else:
        status = "FAIL"
    concurrent_source = "not_tested" if b is None else str(bool(b_source_reliable)).lower()
    concurrent_ipc = "not_tested" if b is None else str(bool(b["ipc_fresh"])).lower()
    degradation_value = degrades if isinstance(degrades, str) else str(degrades).lower()
    lines = [
        "# RT IPC freshness analysis",
        "",
        "All timestamps below are host monotonic timestamps. The SDK exposes no controller timestamp for these RT fields, and none was synthesized.",
        "",
        "| Case | Source P99 / max ms | IPC age P95 / P99 / max ms | IPC >50 ms | Supervisor P99 / max ms |",
        "|---|---:|---:|---:|---:|",
        (
            f"| RT only | {_metric(a, 'source_period_ms', 'p99')} / {_metric(a, 'source_period_ms', 'max')} | "
            f"{_metric(a, 'ipc_age_ms', 'p95')} / {_metric(a, 'ipc_age_ms', 'p99')} / {_metric(a, 'ipc_age_ms', 'max')} | "
            f"{a['ipc_age_ms']['count_gt_50ms']} | {_metric(a, 'supervisor_loop_period_ms', 'p99')} / {_metric(a, 'supervisor_loop_period_ms', 'max')} |"
        ),
    ]
    if b is not None:
        lines.append(
            f"| RT + 20 Hz wrench | {_metric(b, 'source_period_ms', 'p99')} / {_metric(b, 'source_period_ms', 'max')} | "
            f"{_metric(b, 'ipc_age_ms', 'p95')} / {_metric(b, 'ipc_age_ms', 'p99')} / {_metric(b, 'ipc_age_ms', 'max')} | "
            f"{b['ipc_age_ms']['count_gt_50ms']} | {_metric(b, 'supervisor_loop_period_ms', 'p99')} / {_metric(b, 'supervisor_loop_period_ms', 'max')} |"
        )
    lines.extend(
        [
            "",
            (
                "RT-only source freezes >20 ms / max: "
                f"{a['source_freeze_count_gt_20ms']} / {_metric(a, 'source_period_ms', 'max')} ms; "
                f"publish/receive/overwrite/drop: {a['publish_count']} / {a['receive_count']} / "
                f"{a['overwrite_count']} / {a['publish_drop_count']}."
            ),
        ]
    )
    if b is not None:
        wrench = concurrent.get("wrench_summary", {})
        lines.extend(
            [
                (
                    "Concurrent source freezes >20 ms / max: "
                    f"{b['source_freeze_count_gt_20ms']} / {_metric(b, 'source_period_ms', 'max')} ms; "
                    f"publish/receive/overwrite/drop: {b['publish_count']} / {b['receive_count']} / "
                    f"{b['overwrite_count']} / {b['publish_drop_count']}."
                ),
                (
                    "Wrench sequence advance / age P99 / max / heartbeat max: "
                    f"{wrench.get('sequence_advance')} / "
                    f"{_metric(wrench, 'age_ms', 'p99')} / {_metric(wrench, 'age_ms', 'max')} / "
                    f"{_metric(wrench, 'heartbeat_age_ms', 'max')} ms; "
                    f"natural block: {str(bool(wrench.get('natural_block_observed'))).lower()}; "
                    f"errors: {wrench.get('error_codes_seen', [])}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            f"RT_PROCESS_SINGLE_THREAD = {str(single_thread).lower()}",
            f"RT_ONLY_SOURCE_RELIABLE = {str(bool(a_source_reliable)).lower()}",
            f"RT_ONLY_IPC_FRESH = {str(bool(a['ipc_fresh'])).lower()}",
            f"RT_CONCURRENT_SOURCE_RELIABLE = {concurrent_source}",
            f"RT_CONCURRENT_IPC_FRESH = {concurrent_ipc}",
            f"WRENCH_CONCURRENCY_DEGRADES_RT_IPC = {degradation_value}",
            f"ROOT_CAUSE_LAYER = {root_cause}",
            f"PROCESS_ARCHITECTURE_STATUS = {status}",
            "READY_FOR_FIRST_MOTION_TEST = false",
            "",
            "Recommended next step only: preserve this diagnostic architecture and review the captured tails before designing any separately authorized first-motion test; no motion test was executed here.",
            "",
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rt_ipc_freshness_analysis_{_stamp()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"analysis={path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("offline", "rt-only", "concurrent", "analyze"),
    )
    parser.add_argument("--robot-ip", default="192.168.50.103")
    parser.add_argument("--local-ip", default="192.168.50.209")
    parser.add_argument("--robot-class", default="xMateRobot")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument("--rt-only-json")
    parser.add_argument("--concurrent-json")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.mode == "analyze" and not args.rt_only_json:
        parser.error("--rt-only-json is required for analyze mode")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "offline":
        run_offline(args)
    elif args.mode in {"rt-only", "concurrent"}:
        run_live(args)
    else:
        run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
