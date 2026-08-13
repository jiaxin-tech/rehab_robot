"""Strictly read-only live validation of process-isolated ROKAE wrench reads."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from hardware.windows.rokae_xcore import RokaeRobot
from scripts.wrench_process_isolation import WrenchProcessSupervisor


CSV_FIELDS = (
    "stage", "main_loop_timestamp_ns", "loop_period_ms", "rt_sequence",
    "rt_timestamp_ns", "rt_age_ms", "rt_valid", "operation_state",
    "wrench_sequence", "wrench_age_ms", "wrench_valid", "wrench_stale",
    "worker_pid", "worker_start_time_ns", "worker_alive", "worker_exitcode",
    "worker_state", "heartbeat_age_ms", "worker_hung", "last_error_code",
    "last_error", "graceful_disconnect_confirmed",
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


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    periods = [float(row["loop_period_ms"]) for row in rows if row.get("loop_period_ms") is not None]
    rt_ages = [float(row["rt_age_ms"]) for row in rows if row.get("rt_age_ms") is not None]
    wrench_ages = [float(row["wrench_age_ms"]) for row in rows if row.get("wrench_age_ms") is not None]
    heartbeat_ages = [float(row["heartbeat_age_ms"]) for row in rows if row.get("heartbeat_age_ms") is not None]
    rt_periods: list[float] = []
    previous: dict[str, Any] | None = None
    frozen_events = 0
    last_change_ns: int | None = None
    freeze_latched = False
    max_unchanged_ms = 0.0
    for row in rows:
        sequence = row.get("rt_sequence")
        timestamp_ns = row.get("rt_timestamp_ns")
        now_ns = int(row["main_loop_timestamp_ns"])
        if sequence is None:
            continue
        if previous is None or int(sequence) != int(previous["rt_sequence"]):
            if previous is not None and timestamp_ns is not None and previous.get("rt_timestamp_ns") is not None:
                sequence_delta = int(sequence) - int(previous["rt_sequence"])
                if sequence_delta > 0:
                    rt_periods.append(
                        (int(timestamp_ns) - int(previous["rt_timestamp_ns"])) / 1e6 / sequence_delta
                    )
            previous = row
            last_change_ns = now_ns
            freeze_latched = False
        elif last_change_ns is not None:
            unchanged_ms = (now_ns - last_change_ns) / 1e6
            max_unchanged_ms = max(max_unchanged_ms, unchanged_ms)
            if unchanged_ms > 24.0 and not freeze_latched:
                frozen_events += 1
                freeze_latched = True
    first_rt = next((row.get("rt_sequence") for row in rows if row.get("rt_sequence") is not None), None)
    last_rt = next((row.get("rt_sequence") for row in reversed(rows) if row.get("rt_sequence") is not None), None)
    first_wrench = next((row.get("wrench_sequence") for row in rows if int(row.get("wrench_sequence") or 0) > 0), None)
    last_wrench = next((row.get("wrench_sequence") for row in reversed(rows) if int(row.get("wrench_sequence") or 0) > 0), None)
    period_mean = statistics.fmean(periods) if periods else None
    return {
        "tick_count": len(rows),
        "main_loop_rate_hz": None if not period_mean else 1000.0 / period_mean,
        "main_loop_period_ms": stats(periods),
        "rt_sequence_first": first_rt,
        "rt_sequence_last": last_rt,
        "rt_sequence_advance": None if first_rt is None or last_rt is None else int(last_rt) - int(first_rt),
        "rt_source_period_ms": stats(rt_periods),
        "rt_age_ms": stats(rt_ages),
        "rt_frozen_event_count": frozen_events,
        "maximum_unchanged_rt_sequence_ms": max_unchanged_ms,
        "rt_invalid_tick_count": sum(not bool(row.get("rt_valid")) for row in rows),
        "non_idle_tick_count": sum(
            row.get("operation_state") not in (None, "IDLE") for row in rows
        ),
        "wrench_sequence_first": first_wrench,
        "wrench_sequence_last": last_wrench,
        "wrench_sequence_advance": (
            None if first_wrench is None or last_wrench is None else int(last_wrench) - int(first_wrench)
        ),
        "wrench_age_ms": stats(wrench_ages),
        "wrench_stale_tick_count": sum(bool(row.get("wrench_stale")) for row in rows),
        "wrench_invalid_tick_count": sum(not bool(row.get("wrench_valid")) for row in rows),
        "worker_hung_tick_count": sum(bool(row.get("worker_hung")) for row in rows),
        "worker_dead_tick_count": sum(not bool(row.get("worker_alive")) for row in rows),
        "heartbeat_age_ms": stats(heartbeat_ages),
        "error_codes_seen": sorted({
            int(row["last_error_code"])
            for row in rows
            if row.get("last_error_code") is not None
        }),
    }


def _state(robot: RokaeRobot, now_ns: int) -> dict[str, Any]:
    frame = robot.get_state_frame()
    timestamp_ns = (
        None
        if frame.host_monotonic_time_s is None
        else int(frame.host_monotonic_time_s * 1e9)
    )
    age_ms = None if timestamp_ns is None else max(0.0, (now_ns - timestamp_ns) / 1e6)
    return {
        "rt_sequence": frame.sequence_id,
        "rt_timestamp_ns": timestamp_ns,
        "rt_age_ms": age_ms,
        "rt_valid": frame.valid,
        "operation_state": frame.operation_state,
    }


def _preflight(robot: RokaeRobot) -> dict[str, Any]:
    native = robot._robot
    sdk = robot._sdk
    operation = robot._call("operationState", native.operationState)
    power = robot._call("powerState", native.powerState)
    mode = robot._call("operateMode", native.operateMode)
    info = robot._robot_info
    metadata = {
        "sdk_version": robot._sdk_version,
        "controller_version": str(info.version),
        "robot_model": str(info.type),
        "robot_serial": str(info.id),
        "operation_state": operation.name,
        "power_state": power.name,
        "operate_mode": mode.name,
    }
    if operation != sdk.OperationState.idle:
        raise RuntimeError(f"requires operationState=idle, observed {operation.name}")
    if power != sdk.PowerState.on:
        raise RuntimeError(f"requires powerState=on, observed {power.name}")
    if mode != sdk.OperateMode.automatic:
        raise RuntimeError(f"requires operateMode=automatic, observed {mode.name}")
    return metadata


def _loop(
    *,
    stage: str,
    duration_s: float,
    robot: RokaeRobot | None,
    supervisor: WrenchProcessSupervisor,
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    stream: Any,
    stop_on_hung: bool,
) -> str | None:
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(duration_s * 1e9)
    next_tick_ns = started_ns
    previous_tick_ns: int | None = None
    rows_since_sync = 0
    while time.perf_counter_ns() < deadline_ns:
        now_ns = time.perf_counter_ns()
        loop_period_ms = None if previous_tick_ns is None else (now_ns - previous_tick_ns) / 1e6
        previous_tick_ns = now_ns
        state = (
            {
                "rt_sequence": None, "rt_timestamp_ns": None, "rt_age_ms": None,
                "rt_valid": None, "operation_state": None,
            }
            if robot is None
            else _state(robot, now_ns)
        )
        observation = supervisor.poll(now_ns).to_dict()
        row = {
            "stage": stage,
            "main_loop_timestamp_ns": now_ns,
            "loop_period_ms": loop_period_ms,
            **state,
            **{
                "wrench_sequence": observation["wrench_sequence"],
                "wrench_age_ms": observation["wrench_age_ms"],
                "wrench_valid": observation["wrench_valid"],
                "wrench_stale": observation["wrench_stale"],
                "worker_pid": observation["worker_pid"],
                "worker_start_time_ns": observation["worker_start_time_ns"],
                "worker_alive": observation["worker_alive"],
                "worker_exitcode": observation["worker_exitcode"],
                "worker_state": observation["worker_state"],
                "heartbeat_age_ms": observation["heartbeat_age_ms"],
                "worker_hung": observation["worker_hung"],
                "last_error_code": observation["last_error_code"],
                "last_error": observation["last_error"],
                "graceful_disconnect_confirmed": observation["graceful_disconnect_confirmed"],
            },
        }
        rows.append(row)
        writer.writerow(row)
        rows_since_sync += 1
        if rows_since_sync >= 100:
            stream.flush()
            os.fsync(stream.fileno())
            rows_since_sync = 0
        if stop_on_hung and observation["worker_hung"]:
            return "worker_hung"
        if not observation["worker_alive"] and observation["worker_exitcode"] is not None:
            return "worker_died"
        next_tick_ns += 10_000_000
        remaining_ns = next_tick_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        else:
            next_tick_ns = time.perf_counter_ns() + 10_000_000
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join([
        f"# {payload['diagnostic']}",
        "",
        f"- Result: `{payload['result']}`",
        f"- Requested overlap duration: `{payload['requested_duration_s']}` s",
        f"- RT/main session: parent PID `{payload['parent_pid']}`",
        f"- Wrench session: child PID `{payload.get('worker_pid')}`",
        f"- Disconnect order: `{payload['disconnect_order']}`",
        f"- Native block observed: `{str(payload['native_block_observed']).lower()}`",
        f"- Main loop rate: `{summary['main_loop_rate_hz']}` Hz",
        f"- Main loop P99/max: `{summary['main_loop_period_ms']['p99']}` / `{summary['main_loop_period_ms']['max']}` ms",
        f"- RT advance: `{summary['rt_sequence_advance']}`; RT P99/max: `{summary['rt_source_period_ms']['p99']}` / `{summary['rt_source_period_ms']['max']}` ms",
        f"- Wrench advance: `{summary['wrench_sequence_advance']}`; max age: `{summary['wrench_age_ms']['max']}` ms",
        f"- Error codes: `{summary['error_codes_seen']}`",
        f"- Child cleanup: `{json.dumps(payload['child_cleanup'], ensure_ascii=False)}`",
        f"- Parent disconnect confirmed: `{str(payload['parent_disconnect_confirmed']).lower()}`",
        "",
        "Forced child termination, if present, is not a graceful SDK disconnect.",
        "",
    ])


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "wrench_process_multisession" if args.phase == "multi-session" else "wrench_process_isolation_live"
    csv_path = output_dir / f"{prefix}_{stamp}.csv"
    json_path = output_dir / f"{prefix}_{stamp}.json"
    md_path = output_dir / f"{prefix}_{stamp}.md"
    print(
        "STRICT READ-ONLY APIs: Session A connectToRobot, robotInfo, start/update/get RT state, "
        "operationState, powerState, operateMode, stopReceiveRobotState, disconnectFromRobot; "
        "Session B connectToRobot, robotInfo, operationState, powerState, operateMode, "
        "getEndTorque(world), disconnectFromRobot",
        flush=True,
    )
    print(
        "Session ownership: current parent process creates/owns A; spawned child creates/owns B; "
        "no SDK object crosses IPC.",
        flush=True,
    )
    robot = RokaeRobot(args.robot_ip, local_ip=args.local_ip)
    supervisor = WrenchProcessSupervisor(
        stale_age_ms=args.stale_age_ms,
        worker_hung_ms=args.worker_hung_ms,
        worker_startup_hung_ms=args.worker_startup_hung_ms,
    )
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    fatal_error: str | None = None
    stop_reason: str | None = None
    child_cleanup: dict[str, Any] = {}
    parent_disconnect_confirmed = False
    parent_postcheck: dict[str, Any] = {}
    child_pid: int | None = None
    disconnect_order = "rt_first" if args.phase == "multi-session" else "wrench_first"
    stream = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    stream.flush()
    try:
        robot.connect()
        metadata = _preflight(robot)
        child_pid = supervisor.start(mode="live", config={
            "robot_ip": args.robot_ip,
            # Session B does not request the RT local-IP channel.
            "local_ip": "",
            "robot_class": args.robot_class,
            "target_hz": 20.0,
        })
        stop_reason = _loop(
            stage="overlap",
            duration_s=args.duration_s,
            robot=robot,
            supervisor=supervisor,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_hung=True,
        )
        if stop_reason in {"worker_hung", "worker_died"}:
            child_cleanup = supervisor.terminate()
            _loop(
                stage="rt_after_forced_child_stop",
                duration_s=args.post_disconnect_observation_s,
                robot=robot,
                supervisor=supervisor,
                rows=rows,
                writer=writer,
                stream=stream,
                stop_on_hung=False,
            )
            parent_postcheck = _preflight(robot)
            robot.disconnect()
            parent_disconnect_confirmed = not robot.is_connected
        elif disconnect_order == "rt_first":
            parent_postcheck = _preflight(robot)
            robot.disconnect()
            parent_disconnect_confirmed = not robot.is_connected
            _loop(
                stage="wrench_after_rt_disconnect",
                duration_s=args.post_disconnect_observation_s,
                robot=None,
                supervisor=supervisor,
                rows=rows,
                writer=writer,
                stream=stream,
                stop_on_hung=True,
            )
            child_cleanup = supervisor.stop_normally(timeout_s=5.0)
            if not child_cleanup["worker_exited"]:
                child_cleanup = {**child_cleanup, **supervisor.terminate()}
        else:
            child_cleanup = supervisor.stop_normally(timeout_s=5.0)
            if not child_cleanup["worker_exited"]:
                child_cleanup = {**child_cleanup, **supervisor.terminate()}
            _loop(
                stage="rt_after_wrench_disconnect",
                duration_s=args.post_disconnect_observation_s,
                robot=robot,
                supervisor=supervisor,
                rows=rows,
                writer=writer,
                stream=stream,
                stop_on_hung=False,
            )
            parent_postcheck = _preflight(robot)
            robot.disconnect()
            parent_disconnect_confirmed = not robot.is_connected
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}:{exc}"
    finally:
        if supervisor.alive:
            child_cleanup = supervisor.terminate()
        supervisor.poll()
        supervisor.close()
        if robot.is_connected:
            try:
                robot.disconnect()
                parent_disconnect_confirmed = not robot.is_connected
            except BaseException as exc:
                if fatal_error is None:
                    fatal_error = f"cleanup:{type(exc).__name__}:{exc}"
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()

    overlap_rows = [row for row in rows if row["stage"] == "overlap"]
    summary = summarize(overlap_rows)
    post_rows = [row for row in rows if row["stage"] != "overlap"]
    post_summary = summarize(post_rows)
    child_graceful = bool(child_cleanup.get("graceful_disconnect_confirmed"))
    normal_observation = (
        stop_reason is None
        and fatal_error is None
        and summary["wrench_sequence_advance"] is not None
        and int(summary["wrench_sequence_advance"]) > 0
        and summary["rt_sequence_advance"] is not None
        and int(summary["rt_sequence_advance"]) > 0
        and summary["error_codes_seen"] == []
    )
    post_other_side_survived = (
        post_summary["wrench_sequence_advance"] is not None
        and int(post_summary["wrench_sequence_advance"]) > 0
        if disconnect_order == "rt_first"
        else post_summary["rt_sequence_advance"] is not None
        and int(post_summary["rt_sequence_advance"]) > 0
    )
    passed = bool(
        normal_observation
        and post_other_side_survived
        and child_graceful
        and parent_disconnect_confirmed
    )
    result = "pass" if passed else ("native_block_isolated" if stop_reason == "worker_hung" else "fail")
    payload = {
        "schema_version": 1,
        "diagnostic": prefix,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "result": result,
        "pass": passed,
        "robot_ip": args.robot_ip,
        "local_rt_ip": args.local_ip,
        "requested_duration_s": args.duration_s,
        "post_disconnect_observation_s": args.post_disconnect_observation_s,
        "parent_pid": os.getpid(),
        "worker_pid": child_pid,
        "session_ownership": {
            "session_a": "parent process: read-only RT/state",
            "session_b": "spawned wrench child: read-only getEndTorque",
            "sdk_objects_cross_process": False,
            "session_b_local_rt_ip": "",
        },
        "disconnect_order": disconnect_order,
        "metadata": metadata,
        "parent_postcheck": parent_postcheck,
        "summary": summary,
        "post_disconnect_summary": post_summary,
        "stop_reason": stop_reason,
        "native_block_observed": stop_reason == "worker_hung",
        "child_cleanup": child_cleanup,
        "parent_disconnect_confirmed": parent_disconnect_confirmed,
        "other_session_survived_first_disconnect": post_other_side_survived,
        "fatal_error": fatal_error,
        "read_only": True,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return csv_path, json_path, md_path, payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Read-only live wrench process isolation validation")
    result.add_argument("--phase", choices=("multi-session", "live"), required=True)
    result.add_argument("--robot-ip", required=True)
    result.add_argument("--local-ip", required=True)
    result.add_argument("--robot-class", default="xMateRobot")
    result.add_argument("--duration-s", type=float, required=True)
    result.add_argument("--post-disconnect-observation-s", type=float, default=3.0)
    result.add_argument("--stale-age-ms", type=float, default=150.0)
    result.add_argument("--worker-hung-ms", type=float, default=750.0)
    result.add_argument("--worker-startup-hung-ms", type=float, default=15_000.0)
    result.add_argument("--output-dir", default="diagnostics")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.duration_s <= 0 or args.post_disconnect_observation_s <= 0:
        raise SystemExit("durations must be positive")
    csv_path, json_path, md_path, payload = run(args)
    print(json.dumps({
        "result": payload["result"],
        "pass": payload["pass"],
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0 if payload["pass"] or payload["result"] == "native_block_isolated" else 2)


if __name__ == "__main__":
    main()
