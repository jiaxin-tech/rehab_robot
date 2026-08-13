"""Option-B live prototype: pure supervisor + RT process + wrench process."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from scripts.validate_wrench_process_live import summarize
from scripts.wrench_process_isolation import RtProcessSupervisor, WrenchProcessSupervisor


CSV_FIELDS = (
    "stage", "main_loop_timestamp_ns", "loop_period_ms",
    "rt_sequence", "rt_timestamp_ns", "rt_age_ms", "rt_valid", "rt_stale",
    "operation_state", "rt_worker_pid", "rt_worker_start_time_ns",
    "rt_worker_alive", "rt_worker_exitcode", "rt_worker_state",
    "rt_heartbeat_age_ms", "rt_worker_hung", "rt_last_error_code", "rt_last_error",
    "wrench_sequence", "wrench_age_ms", "wrench_valid", "wrench_stale",
    "wrench_worker_pid", "wrench_worker_start_time_ns", "wrench_worker_alive",
    "wrench_worker_exitcode", "wrench_worker_state", "wrench_heartbeat_age_ms",
    "wrench_worker_hung", "last_error_code", "last_error",
)


def _row_for_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_loop_timestamp_ns": row["main_loop_timestamp_ns"],
        "loop_period_ms": row["loop_period_ms"],
        "rt_sequence": row["rt_sequence"],
        "rt_timestamp_ns": row["rt_timestamp_ns"],
        "rt_age_ms": row["rt_age_ms"],
        "rt_valid": row["rt_valid"],
        "operation_state": row["operation_state"],
        "wrench_sequence": row["wrench_sequence"],
        "wrench_age_ms": row["wrench_age_ms"],
        "wrench_valid": row["wrench_valid"],
        "wrench_stale": row["wrench_stale"],
        "worker_alive": row["wrench_worker_alive"],
        "worker_hung": row["wrench_worker_hung"],
        "heartbeat_age_ms": row["wrench_heartbeat_age_ms"],
        "last_error_code": row["last_error_code"],
    }


def _poll_row(
    *,
    stage: str,
    now_ns: int,
    loop_period_ms: float | None,
    rt: RtProcessSupervisor,
    wrench: WrenchProcessSupervisor | None,
) -> dict[str, Any]:
    rt_observation = rt.poll(now_ns)
    wrench_observation = None if wrench is None else wrench.poll(now_ns)
    return {
        "stage": stage,
        "main_loop_timestamp_ns": now_ns,
        "loop_period_ms": loop_period_ms,
        "rt_sequence": rt_observation.rt_sequence,
        "rt_timestamp_ns": rt_observation.rt_timestamp_ns,
        "rt_age_ms": rt_observation.rt_age_ms,
        "rt_valid": rt_observation.rt_valid,
        "rt_stale": rt_observation.rt_stale,
        "operation_state": rt_observation.operation_state,
        "rt_worker_pid": rt_observation.worker_pid,
        "rt_worker_start_time_ns": rt_observation.worker_start_time_ns,
        "rt_worker_alive": rt_observation.worker_alive,
        "rt_worker_exitcode": rt_observation.worker_exitcode,
        "rt_worker_state": rt_observation.worker_state,
        "rt_heartbeat_age_ms": rt_observation.heartbeat_age_ms,
        "rt_worker_hung": rt_observation.worker_hung,
        "rt_last_error_code": rt_observation.last_error_code,
        "rt_last_error": rt_observation.last_error,
        "wrench_sequence": 0 if wrench_observation is None else wrench_observation.wrench_sequence,
        "wrench_age_ms": None if wrench_observation is None else wrench_observation.wrench_age_ms,
        "wrench_valid": False if wrench_observation is None else wrench_observation.wrench_valid,
        "wrench_stale": True if wrench_observation is None else wrench_observation.wrench_stale,
        "wrench_worker_pid": None if wrench_observation is None else wrench_observation.worker_pid,
        "wrench_worker_start_time_ns": None if wrench_observation is None else wrench_observation.worker_start_time_ns,
        "wrench_worker_alive": False if wrench_observation is None else wrench_observation.worker_alive,
        "wrench_worker_exitcode": None if wrench_observation is None else wrench_observation.worker_exitcode,
        "wrench_worker_state": "not_started" if wrench_observation is None else wrench_observation.worker_state,
        "wrench_heartbeat_age_ms": None if wrench_observation is None else wrench_observation.heartbeat_age_ms,
        "wrench_worker_hung": False if wrench_observation is None else wrench_observation.worker_hung,
        "last_error_code": None if wrench_observation is None else wrench_observation.last_error_code,
        "last_error": None if wrench_observation is None else wrench_observation.last_error,
    }


def _run_loop(
    *,
    stage: str,
    duration_s: float,
    rt: RtProcessSupervisor,
    wrench: WrenchProcessSupervisor | None,
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    stream: Any,
    stop_on_fault: bool,
) -> str | None:
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(duration_s * 1e9)
    next_tick_ns = started_ns
    previous_ns: int | None = None
    rows_since_sync = 0
    while time.perf_counter_ns() < deadline_ns:
        now_ns = time.perf_counter_ns()
        period_ms = None if previous_ns is None else (now_ns - previous_ns) / 1e6
        previous_ns = now_ns
        row = _poll_row(
            stage=stage,
            now_ns=now_ns,
            loop_period_ms=period_ms,
            rt=rt,
            wrench=wrench,
        )
        rows.append(row)
        writer.writerow(row)
        rows_since_sync += 1
        if rows_since_sync >= 100:
            stream.flush()
            os.fsync(stream.fileno())
            rows_since_sync = 0
        if stop_on_fault:
            if row["rt_worker_hung"]:
                return "rt_worker_hung"
            if not row["rt_worker_alive"] and row["rt_worker_exitcode"] is not None:
                return "rt_worker_died"
            if wrench is not None and row["wrench_worker_hung"]:
                return "wrench_worker_hung"
            if wrench is not None and not row["wrench_worker_alive"] and row["wrench_worker_exitcode"] is not None:
                return "wrench_worker_died"
        next_tick_ns += 10_000_000
        remaining_ns = next_tick_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        else:
            next_tick_ns = time.perf_counter_ns() + 10_000_000
    return None


def _wait_for_rt(
    rt: RtProcessSupervisor,
    *,
    timeout_s: float,
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    stream: Any,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _run_loop(
            stage="rt_startup",
            duration_s=0.1,
            rt=rt,
            wrench=None,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_fault=True,
        )
        observation = rt.poll()
        if observation.rt_valid and observation.operation_state == "IDLE":
            return
        if observation.worker_hung or (
            not observation.worker_alive and observation.worker_exitcode is not None
        ):
            break
    observation = rt.poll()
    raise RuntimeError(
        "RT process did not become valid: "
        f"state={observation.worker_state}, error={observation.last_error}"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["overlap_summary"]
    return "\n".join([
        "# ROKAE Option-B process isolation live validation",
        "",
        f"- Result: `{payload['result']}`",
        f"- Supervisor / RT / wrench PIDs: `{payload['supervisor_pid']}` / `{payload['rt_worker_pid']}` / `{payload['wrench_worker_pid']}`",
        "- SDK ownership: RT process exclusively owns Session A; wrench process exclusively owns Session B; supervisor owns no SDK object.",
        f"- Requested overlap duration: `{payload['requested_duration_s']}` s",
        f"- Main loop rate: `{summary['main_loop_rate_hz']}` Hz",
        f"- Main loop P99/max: `{summary['main_loop_period_ms']['p99']}` / `{summary['main_loop_period_ms']['max']}` ms",
        f"- RT sequence advance: `{summary['rt_sequence_advance']}`; source P99/max: `{summary['rt_source_period_ms']['p99']}` / `{summary['rt_source_period_ms']['max']}` ms",
        f"- Wrench sequence advance: `{summary['wrench_sequence_advance']}`; max age: `{summary['wrench_age_ms']['max']}` ms",
        f"- Stop reason: `{payload['stop_reason']}`",
        f"- Wrench cleanup: `{json.dumps(payload['wrench_cleanup'], ensure_ascii=False)}`",
        f"- RT cleanup: `{json.dumps(payload['rt_cleanup'], ensure_ascii=False)}`",
        f"- RT source survived wrench stop: `{str(payload['rt_source_survived_wrench_stop']).lower()}`",
        f"- RT IPC freshness pass: `{str(payload['rt_ipc_freshness_pass']).lower()}`",
        "",
        "A forced process termination is not a graceful SDK disconnect.",
        "",
    ])


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"wrench_process_isolation_{stamp}.csv"
    json_path = output_dir / f"wrench_process_isolation_{stamp}.json"
    md_path = output_dir / f"wrench_process_isolation_{stamp}.md"
    print(
        "STRICT READ-ONLY Option B: RT child owns connect/robotInfo/state/status/disconnect; "
        "wrench child owns connect/robotInfo/status/getEndTorque(world)/disconnect; "
        "supervisor owns no SDK object.",
        flush=True,
    )
    rt = RtProcessSupervisor(
        stale_age_ms=args.rt_stale_age_ms,
        worker_hung_ms=args.worker_hung_ms,
        worker_startup_hung_ms=args.worker_startup_hung_ms,
    )
    wrench = WrenchProcessSupervisor(
        stale_age_ms=args.wrench_stale_age_ms,
        worker_hung_ms=args.worker_hung_ms,
        worker_startup_hung_ms=args.worker_startup_hung_ms,
    )
    rows: list[dict[str, Any]] = []
    stop_reason: str | None = None
    fatal_error: str | None = None
    wrench_cleanup: dict[str, Any] = {}
    rt_cleanup: dict[str, Any] = {}
    rt_pid: int | None = None
    wrench_pid: int | None = None
    stream = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    stream.flush()
    try:
        rt_pid = rt.start({
            "robot_ip": args.robot_ip,
            "local_ip": args.local_ip,
            "robot_class": args.robot_class,
            "publish_hz": 100.0,
        })
        _wait_for_rt(
            rt,
            timeout_s=args.worker_startup_hung_ms / 1000.0,
            rows=rows,
            writer=writer,
            stream=stream,
        )
        wrench_pid = wrench.start(mode="live", config={
            "robot_ip": args.robot_ip,
            "local_ip": "",
            "robot_class": args.robot_class,
            "target_hz": 20.0,
        })
        stop_reason = _run_loop(
            stage="overlap",
            duration_s=args.duration_s,
            rt=rt,
            wrench=wrench,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_fault=True,
        )
        if stop_reason in {"wrench_worker_hung", "wrench_worker_died"}:
            wrench_cleanup = wrench.terminate()
        else:
            wrench_cleanup = wrench.stop_normally(timeout_s=5.0)
            if not wrench_cleanup["worker_exited"]:
                wrench_cleanup = {**wrench_cleanup, **wrench.terminate()}
        _run_loop(
            stage="rt_after_wrench_stop",
            duration_s=args.post_wrench_observation_s,
            rt=rt,
            wrench=wrench,
            rows=rows,
            writer=writer,
            stream=stream,
            stop_on_fault=False,
        )
        rt_cleanup = rt.stop_normally(timeout_s=5.0)
        if not rt_cleanup["worker_exited"]:
            rt_cleanup = {**rt_cleanup, **rt.terminate()}
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}:{exc}"
    finally:
        if wrench.alive:
            wrench_cleanup = wrench.terminate()
        if rt.alive:
            rt_cleanup = rt.terminate()
        wrench.poll()
        rt.poll()
        wrench.close()
        rt.close()
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()

    overlap_rows = [_row_for_summary(row) for row in rows if row["stage"] == "overlap"]
    post_rows = [_row_for_summary(row) for row in rows if row["stage"] == "rt_after_wrench_stop"]
    overlap_summary = summarize(overlap_rows)
    post_summary = summarize(post_rows)
    rt_source_survived = bool(
        post_summary["rt_sequence_advance"] is not None
        and int(post_summary["rt_sequence_advance"]) > 0
        and all(bool(row["rt_worker_alive"]) for row in rows if row["stage"] == "rt_after_wrench_stop")
        and not any(bool(row["rt_worker_hung"]) for row in rows if row["stage"] == "rt_after_wrench_stop")
    )
    rt_ipc_freshness_pass = post_summary["rt_invalid_tick_count"] == 0
    main_ok = bool(
        overlap_summary["main_loop_rate_hz"] is not None
        and float(overlap_summary["main_loop_rate_hz"]) >= 80.0
    )
    rt_ok = bool(
        overlap_summary["rt_sequence_advance"] is not None
        and int(overlap_summary["rt_sequence_advance"]) > 0
        and overlap_summary["rt_invalid_tick_count"] == 0
    )
    wrench_progress = bool(
        overlap_summary["wrench_sequence_advance"] is not None
        and int(overlap_summary["wrench_sequence_advance"]) > 0
    )
    wrench_termination_ok = bool(
        wrench_cleanup.get("worker_exited") or wrench_cleanup.get("worker_terminated")
    )
    rt_cleanup_ok = bool(
        rt_cleanup.get("worker_exited")
        and rt_cleanup.get("graceful_disconnect_confirmed")
    )
    if stop_reason == "wrench_worker_hung":
        isolation_core = bool(
            main_ok and rt_source_survived and wrench_termination_ok and rt_cleanup_ok
        )
        passed = bool(isolation_core and rt_ipc_freshness_pass)
        if passed:
            result = "native_block_isolated"
        elif isolation_core:
            result = "native_block_isolated_rt_ipc_freshness_failed"
        else:
            result = "native_block_isolation_failed"
    else:
        normal = bool(
            stop_reason is None
            and fatal_error is None
            and main_ok
            and rt_ok
            and wrench_progress
            and overlap_summary["error_codes_seen"] == []
            and wrench_cleanup.get("graceful_disconnect_confirmed")
            and rt_source_survived
            and rt_ipc_freshness_pass
            and rt_cleanup_ok
        )
        result = "normal_operation_pass" if normal else "fail"
        passed = normal
    payload = {
        "schema_version": 1,
        "diagnostic": "wrench_process_isolation_option_b_live",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "pass": passed,
        "robot_ip": args.robot_ip,
        "local_rt_ip": args.local_ip,
        "requested_duration_s": args.duration_s,
        "post_wrench_observation_s": args.post_wrench_observation_s,
        "supervisor_pid": os.getpid(),
        "rt_worker_pid": rt_pid,
        "wrench_worker_pid": wrench_pid,
        "session_ownership": {
            "supervisor": "no SDK object; bounded non-blocking IPC readers only",
            "session_a": "RT child process exclusively owns RT/state SDK connection",
            "session_b": "wrench child process exclusively owns getEndTorque SDK connection",
            "sdk_objects_cross_process": False,
        },
        "rt_metadata": rt.metadata,
        "wrench_metadata": wrench.metadata,
        "overlap_summary": overlap_summary,
        "post_wrench_summary": post_summary,
        "stop_reason": stop_reason,
        "native_block_observed": stop_reason == "wrench_worker_hung",
        "wrench_cleanup": wrench_cleanup,
        "rt_cleanup": rt_cleanup,
        "process_isolation_core_pass": bool(
            main_ok and rt_source_survived and wrench_termination_ok and rt_cleanup_ok
        ),
        "rt_source_survived_wrench_stop": rt_source_survived,
        "rt_ipc_freshness_pass": rt_ipc_freshness_pass,
        "fatal_error": fatal_error,
        "read_only": True,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return csv_path, json_path, md_path, payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Option-B read-only process isolation live validator")
    result.add_argument("--robot-ip", required=True)
    result.add_argument("--local-ip", required=True)
    result.add_argument("--robot-class", default="xMateRobot")
    result.add_argument("--duration-s", type=float, default=180.0)
    result.add_argument("--post-wrench-observation-s", type=float, default=3.0)
    result.add_argument("--wrench-stale-age-ms", type=float, default=150.0)
    result.add_argument("--rt-stale-age-ms", type=float, default=50.0)
    result.add_argument("--worker-hung-ms", type=float, default=750.0)
    result.add_argument("--worker-startup-hung-ms", type=float, default=15_000.0)
    result.add_argument("--output-dir", default="diagnostics")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.duration_s < 180.0 or args.duration_s > 300.0:
        raise SystemExit("live duration must be within the approved 180-300 second range")
    csv_path, json_path, md_path, payload = run(args)
    print(json.dumps({
        "result": payload["result"],
        "pass": payload["pass"],
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0 if payload["pass"] else 2)


if __name__ == "__main__":
    main()
