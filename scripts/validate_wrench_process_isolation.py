"""Offline-first validator for the diagnostic wrench process prototype."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from scripts.characterize_wrench_longrun import percentile
from scripts.wrench_process_isolation import WrenchProcessSupervisor, run_fresh_sanity


CSV_FIELDS = (
    "scenario", "main_loop_timestamp_ns", "elapsed_ms", "main_loop_period_ms",
    "worker_pid", "worker_start_time_ns", "worker_alive", "worker_exitcode",
    "worker_state", "last_heartbeat_ns", "heartbeat_age_ms", "worker_hung",
    "wrench_sequence", "last_wrench_success_ns", "wrench_age_ms",
    "wrench_valid", "wrench_stale", "last_error_code", "last_error",
    "graceful_disconnect_confirmed",
)


OFFLINE_CASES: tuple[dict[str, Any], ...] = (
    {"name": "normal", "config": {"behavior": "normal", "delay_s": 0.001}, "duration_s": 0.8},
    {"name": "slow_40ms", "config": {"behavior": "normal", "delay_s": 0.040}, "duration_s": 0.8},
    {
        "name": "stale_500ms", "config": {"behavior": "normal", "delay_s": 0.500},
        "duration_s": 1.2, "hung_ms": 1500.0,
    },
    {
        "name": "long_block_10s", "config": {"behavior": "normal", "delay_s": 10.0},
        "duration_s": 10.5, "hung_ms": 300.0,
    },
    {
        "name": "permanent_block", "config": {"behavior": "permanent", "permanent_sleep_s": 3600.0},
        "duration_s": 0.8, "hung_ms": 250.0, "force_terminate": True,
    },
    {"name": "error_263", "config": {"behavior": "error263", "delay_s": 0.001}, "duration_s": 0.8},
    {"name": "worker_exception", "config": {"behavior": "exception", "delay_s": 0.001}, "duration_s": 0.8},
    {
        "name": "worker_crash", "config": {"behavior": "crash", "delay_s": 0.001, "crash_after": 1},
        "duration_s": 0.8,
    },
    {
        "name": "ipc_saturation", "config": {"behavior": "normal", "delay_s": 0.0001, "target_hz": 2000.0},
        "duration_s": 0.8,
    },
)


def _wait_until_ns(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_000_000) / 1e9)


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def run_offline_case(case: dict[str, Any], csv_writer: csv.DictWriter | None = None) -> dict[str, Any]:
    name = str(case["name"])
    config = {"target_hz": 20.0, **dict(case["config"])}
    supervisor = WrenchProcessSupervisor(
        stale_age_ms=150.0,
        worker_hung_ms=float(case.get("hung_ms", 300.0)),
    )
    pid = supervisor.start(mode="offline", config=config)
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(float(case["duration_s"]) * 1e9)
    next_tick_ns = started_ns
    previous_tick_ns: int | None = None
    periods: list[float] = []
    rows: list[dict[str, Any]] = []
    parent_work_counter = 0
    while time.perf_counter_ns() < deadline_ns:
        now_ns = time.perf_counter_ns()
        period_ms = None if previous_tick_ns is None else (now_ns - previous_tick_ns) / 1e6
        if period_ms is not None:
            periods.append(period_ms)
        previous_tick_ns = now_ns
        observation = supervisor.poll(now_ns).to_dict()
        row = {
            "scenario": name,
            "main_loop_timestamp_ns": now_ns,
            "elapsed_ms": (now_ns - started_ns) / 1e6,
            "main_loop_period_ms": period_ms,
            **observation,
        }
        rows.append(row)
        if csv_writer is not None:
            csv_writer.writerow(row)
        # Observable independent parent-side work that must progress even when
        # the child is inside a native-like block.
        parent_work_counter += 1
        if name == "worker_crash" and not observation["worker_alive"] and len(rows) > 5:
            break
        next_tick_ns += 10_000_000
        if next_tick_ns <= time.perf_counter_ns():
            next_tick_ns = time.perf_counter_ns() + 10_000_000
        _wait_until_ns(next_tick_ns)

    if bool(case.get("force_terminate")):
        cleanup = supervisor.terminate()
    else:
        cleanup = supervisor.stop_normally(timeout_s=1.0)
        if not cleanup["worker_exited"]:
            cleanup = {**cleanup, "normal_stop_failed": True, **supervisor.terminate()}
    final = supervisor.poll().to_dict()
    supervisor.close()
    hung_seen = any(bool(row["worker_hung"]) for row in rows)
    stale_seen = any(bool(row["wrench_stale"]) for row in rows)
    alive_seen = any(bool(row["worker_alive"]) for row in rows)
    death_seen = any(not bool(row["worker_alive"]) for row in rows[1:]) or not final["worker_alive"]
    max_sequence = max((int(row["wrench_sequence"]) for row in rows), default=0)
    main_rate_hz = None if not periods else 1000.0 / statistics.fmean(periods)
    result = {
        "scenario": name,
        "worker_pid": pid,
        "requested_duration_s": float(case["duration_s"]),
        "observed_duration_s": (time.perf_counter_ns() - started_ns) / 1e9,
        "main_loop_tick_count": len(rows),
        "main_loop_rate_hz": main_rate_hz,
        "main_loop_period_ms": _stats(periods),
        "parent_work_counter": parent_work_counter,
        "worker_alive_seen": alive_seen,
        "worker_death_visible": death_seen,
        "wrench_stale_seen": stale_seen,
        "worker_hung_seen": hung_seen,
        "maximum_wrench_sequence_seen": max_sequence,
        "last_error_code": final["last_error_code"],
        "last_error": final["last_error"],
        "cleanup": cleanup,
    }
    result["pass"] = offline_case_passes(result)
    return result


def offline_case_passes(result: dict[str, Any]) -> bool:
    scenario = result["scenario"]
    rate_ok = float(result.get("main_loop_rate_hz") or 0.0) >= 80.0
    cleanup = result["cleanup"]
    terminated = bool(cleanup.get("worker_terminated") or cleanup.get("worker_exited"))
    common = rate_ok and result["parent_work_counter"] == result["main_loop_tick_count"] and terminated
    if scenario in {"normal", "slow_40ms"}:
        return common and result["maximum_wrench_sequence_seen"] > 0
    if scenario == "stale_500ms":
        return common and result["wrench_stale_seen"] and result["maximum_wrench_sequence_seen"] > 0
    if scenario == "long_block_10s":
        return common and result["worker_hung_seen"] and result["wrench_stale_seen"] and result["maximum_wrench_sequence_seen"] > 0
    if scenario == "permanent_block":
        return common and result["worker_hung_seen"] and bool(cleanup.get("forced"))
    if scenario == "error_263":
        return common and result["last_error_code"] == 263
    if scenario == "worker_exception":
        return common and "synthetic worker exception" in str(result.get("last_error"))
    if scenario == "worker_crash":
        return common and result["worker_death_visible"]
    if scenario == "ipc_saturation":
        return common and result["maximum_wrench_sequence_seen"] > result["main_loop_tick_count"]
    return False


def render_offline_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Wrench process isolation offline validation",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "- Architecture: parent/supervisor and spawned child process; bounded latest-snapshot IPC; no SDK object crosses IPC.",
        "- Thresholds are diagnostic-only: wrench stale 150 ms; per-case worker hung thresholds are recorded in JSON.",
        "",
        "| Scenario | Pass | Main Hz | P99 period ms | Stale | Hung | Error | Cleanup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        cleanup_ok = bool(result["cleanup"].get("worker_exited") or result["cleanup"].get("worker_terminated"))
        lines.append(
            f"| {result['scenario']} | {str(result['pass']).lower()} | "
            f"{float(result['main_loop_rate_hz'] or 0):.3f} | "
            f"{float(result['main_loop_period_ms']['p99'] or 0):.3f} | "
            f"{str(result['wrench_stale_seen']).lower()} | "
            f"{str(result['worker_hung_seen']).lower()} | "
            f"{result['last_error_code'] or ''} | {str(cleanup_ok).lower()} |"
        )
    lines.extend([
        "",
        "Forced termination is intentionally reported separately from graceful SDK disconnect; offline mock cleanup does not make any claim about a native SDK session.",
        "",
    ])
    return "\n".join(lines)


def run_offline(output_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"wrench_process_isolation_offline_{stamp}.csv"
    json_path = output_dir / f"wrench_process_isolation_offline_{stamp}.json"
    md_path = output_dir / f"wrench_process_isolation_offline_{stamp}.md"
    results: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in OFFLINE_CASES:
            result = run_offline_case(case, writer)
            results.append(result)
            stream.flush()
    payload = {
        "schema_version": 1,
        "diagnostic": "wrench_process_isolation_offline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(bool(result["pass"]) for result in results),
        "results": results,
        "acceptance": {
            "parent_never_blocks_on_worker": all((result["main_loop_rate_hz"] or 0) >= 80 for result in results),
            "ten_second_block_main_loop_continues": next(result["pass"] for result in results if result["scenario"] == "long_block_10s"),
            "permanent_block_detected_and_terminated": next(result["pass"] for result in results if result["scenario"] == "permanent_block"),
            "stale_visible": next(result["wrench_stale_seen"] for result in results if result["scenario"] == "stale_500ms"),
            "worker_death_visible": next(result["worker_death_visible"] for result in results if result["scenario"] == "worker_crash"),
            "bounded_latest_snapshot_survives_saturation": next(result["pass"] for result in results if result["scenario"] == "ipc_saturation"),
        },
        "read_only": True,
        "robot_contacted": False,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_offline_markdown(payload), encoding="utf-8")
    return csv_path, json_path, md_path, payload


def run_sanity(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"wrench_process_fresh_sanity_{stamp}.json"
    print(
        "STRICT READ-ONLY APIs in fresh spawned process: connectToRobot, robotInfo, "
        "operationState, powerState, operateMode, disconnectFromRobot",
        flush=True,
    )
    result = run_fresh_sanity({
        "robot_ip": args.robot_ip,
        "local_ip": "",
        "robot_class": args.robot_class,
    }, timeout_s=args.timeout_s)
    payload = {
        "schema_version": 1,
        "diagnostic": "wrench_process_fresh_session_sanity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "robot_ip": args.robot_ip,
        "local_ip_passed_to_status_only_session": "",
        "result": result,
        "pass": bool(
            result.get("fresh_connection_success")
            and result.get("status_read_success")
            and result.get("disconnect_success")
            and not result.get("process_terminated_by_parent")
        ),
        "read_only": True,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Offline-first wrench process isolation validator")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--fresh-sanity", action="store_true")
    result.add_argument("--robot-ip", default="192.168.50.103")
    result.add_argument("--local-ip", default="192.168.50.209")
    result.add_argument("--robot-class", default="xMateRobot")
    result.add_argument("--timeout-s", type=float, default=30.0)
    result.add_argument("--output-dir", default="diagnostics")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.offline:
        csv_path, json_path, md_path, payload = run_offline(Path(args.output_dir))
        print(json.dumps({
            "overall_pass": payload["overall_pass"],
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        }, ensure_ascii=False), flush=True)
        raise SystemExit(0 if payload["overall_pass"] else 2)
    path, payload = run_sanity(args)
    print(json.dumps({"pass": payload["pass"], "json": str(path)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if payload["pass"] else 2)


if __name__ == "__main__":
    main()
