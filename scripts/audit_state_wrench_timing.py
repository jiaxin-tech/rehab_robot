"""Formal W1 state/wrench timing audit (strictly read-only, no robot motion)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

_matplotlib_cache = Path(tempfile.gettempdir()) / "rehab_robot_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.wrench_process_isolation import RtProcessSupervisor, WrenchProcessSupervisor


UNDEFINED = "UNDEFINED"
NOT_DEFINED = "NOT_FORMALLY_DEFINED"
SUPERVISOR_FIELDS = (
    "case", "sample_index", "main_loop_timestamp_ns", "loop_interval_ms",
    "rt_sequence", "source_or_receive_timestamp_ns", "publish_timestamp_ns",
    "supervisor_receive_timestamp_ns", "rt_ipc_age_ms", "rt_age_ms",
    "rt_new_snapshot", "rt_valid", "rt_stale", "rt_worker_alive",
    "rt_worker_state", "rt_worker_hung", "rt_heartbeat_age_ms",
    "rt_publish_count", "rt_receive_count", "rt_overwrite_count",
    "rt_publish_drop_count", "operation_state", "wrench_sequence",
    "wrench_last_success_ns", "wrench_age_ms", "wrench_valid", "wrench_stale",
    "wrench_worker_alive", "wrench_worker_state", "wrench_worker_hung",
    "wrench_heartbeat_age_ms", "wrench_last_error_code", "wrench_last_error",
)
RT_SOURCE_FIELDS = (
    "rt_sequence", "source_or_receive_timestamp_ns", "publish_timestamp_ns",
    "source_interval_ms", "publish_interval_ms", "source_to_publish_ms",
)
WRENCH_FIELDS = (
    "sequence_id", "host_timestamp_ns", "call_start_ns", "call_end_ns",
    "query_latency_ms", "request_inter_arrival_ms", "success", "error_code",
    "error_message", "operation_state", "joint_measured_torque_nm", "joint_external_torque_nm",
    "cartesian_force_raw_n", "cartesian_torque_raw_nm",
)


def artifact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def metric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean) if clean else None,
    }


def threshold_counts(values: Iterable[float], thresholds_ms: Iterable[float]) -> dict[str, int]:
    clean = [float(value) for value in values]
    return {f"gt_{float(limit):g}_ms": sum(value > limit for value in clean) for limit in thresholds_ms}


def error_histogram(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for event in events:
        if bool(event.get("success")):
            continue
        code = event.get("error_code")
        key = "NO_NUMERIC_CODE" if code is None else str(int(code))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def consecutive_error_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    for index in range(len(events) + 1):
        failed = index < len(events) and not bool(events[index].get("success"))
        if failed and start is None:
            start = index
        if not failed and start is not None:
            group = events[start:index]
            runs.append({
                "first_sequence_id": group[0].get("sequence_id"),
                "last_sequence_id": group[-1].get("sequence_id"),
                "count": len(group),
                "onset_ns": group[0].get("call_start_ns"),
                "end_ns": group[-1].get("call_end_ns"),
                "duration_ms": (
                    int(group[-1]["call_end_ns"]) - int(group[0]["call_start_ns"])
                ) / 1e6,
                "codes": [item.get("error_code") for item in group],
                "recovered": index < len(events) and bool(events[index].get("success")),
                "recovery_sequence_id": None if index >= len(events) else events[index].get("sequence_id"),
                "recovery_timestamp_ns": None if index >= len(events) else events[index].get("call_end_ns"),
            })
            start = None
    return runs


def error_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if bool(event.get("success")):
            continue
        next_success = next(
            (candidate for candidate in events[index + 1:] if bool(candidate.get("success"))),
            None,
        )
        timeline.append({
            "sequence_id": event.get("sequence_id"),
            "onset_ns": event.get("call_start_ns"),
            "end_ns": event.get("call_end_ns"),
            "duration_ms": event.get("query_latency_ms"),
            "error_code": event.get("error_code"),
            "error_message": event.get("error_message"),
            "operation_state": event.get("operation_state"),
            "recovered": next_success is not None,
            "recovery_sequence_id": None if next_success is None else next_success.get("sequence_id"),
            "recovery_timestamp_ns": None if next_success is None else next_success.get("call_end_ns"),
        })
    return timeline


def gate(status: str, reason: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "evidence": evidence}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        return {
            "commit_sha": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit_sha": None, "branch": None, "dirty": None, "error": str(exc)}


def load_formal_context(repo: Path) -> dict[str, Any]:
    manifest_path = repo / "config" / "formal_experiment_manifest.json"
    safety_path = repo / "config" / "experiment_safety.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    return {
        "formal_manifest_path": str(manifest_path.resolve()),
        "formal_manifest_sha256": sha256_file(manifest_path),
        "formal_manifest": manifest,
        "safety_config_path": str(safety_path.resolve()),
        "safety_config_sha256": sha256_file(safety_path),
        "safety_config": safety,
    }


def resolve_protocol(args: argparse.Namespace, formal: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(formal["formal_manifest"].get("state_wrench_timing_audit") or {})
    duration = args.duration if args.duration is not None else configured.get("duration_s")
    duration_source = "cli" if args.duration is not None else "formal_manifest"
    if duration is None:
        duration = 900.0 if args.mode != "offline" else 3.0
        duration_source = "recommended_default_not_frozen_in_manifest"
    wrench_hz = args.wrench_hz if args.wrench_hz is not None else configured.get("wrench_hz")
    wrench_rate_source = "cli" if args.wrench_hz is not None else "formal_manifest"
    if wrench_hz is None:
        wrench_hz = 20.0
        wrench_rate_source = "current_experiment_default_not_frozen_in_manifest"
    return {
        "duration_s": float(duration),
        "duration_source": duration_source,
        "supervisor_hz": 100.0,
        "rt_requested_interval_ms": 8,
        "wrench_hz": float(wrench_hz),
        "wrench_rate_source": wrench_rate_source,
    }


def _json_cell(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def _wait_for_rt(rt: RtProcessSupervisor, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = rt.poll()
        rt.drain_source_events()
        if observation.rt_valid and observation.operation_state == "IDLE":
            return
        if observation.worker_hung or observation.worker_exitcode is not None:
            break
        time.sleep(0.01)
    observation = rt.poll()
    raise RuntimeError(
        f"RT startup failed: state={observation.worker_state}, error={observation.last_error}"
    )


def _wait_for_wrench(wrench: WrenchProcessSupervisor, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = wrench.poll()
        if observation.worker_alive and observation.worker_state in {"ready", "querying", "idle"}:
            return
        if observation.worker_exitcode is not None or observation.worker_hung:
            break
        time.sleep(0.01)
    observation = wrench.poll()
    raise RuntimeError(
        f"wrench startup failed: state={observation.worker_state}, error={observation.last_error}"
    )


def build_gates(
    *,
    case: str,
    summary: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    integrity_ok = bool(
        summary["completion"]["completed_requested_duration"]
        and summary["rt_source"]["ring_overwrite_count"] == 0
        and summary["rt_source"]["sequence_gap_count"] == 0
        and summary["wrench"]["event_drop_count"] == 0
    )
    operation_ok = bool(
        str(summary["state_transitions"].get("before", "")).lower() == "idle"
        and str(summary["state_transitions"].get("after", "")).lower() == "idle"
        and not summary["state_transitions"].get("non_idle_transitions")
    )
    process_ok = bool(
        summary["workers"]["rt_hung_event_count"] == 0
        and summary["workers"]["rt_crash_event_count"] == 0
        and summary["workers"]["wrench_crash_event_count"] == 0
        and summary["workers"]["wrench_hung_event_count"] == 0
    )
    cleanup_ok = bool(
        summary["workers"]["rt_cleanup"].get("worker_exited")
        and summary["workers"]["rt_cleanup"].get("graceful_disconnect_confirmed")
        and (
            case == "test_a"
            or (
                summary["workers"]["wrench_cleanup"].get("worker_exited")
                and summary["workers"]["wrench_cleanup"].get("graceful_disconnect_confirmed")
            )
        )
    )
    gates = {
        "data_integrity": gate("PASS" if integrity_ok else "FAIL", "lossless audit telemetry and requested duration"),
        "operation_state_stability": gate("PASS" if operation_ok else "FAIL", "operationState must remain idle"),
        "process_stability": gate("PASS" if process_ok else "FAIL", "no worker crash or hung event"),
        "cleanup": gate("PASS" if cleanup_ok else "FAIL", "graceful worker exit and SDK disconnect"),
        "rt_source_timing": gate(UNDEFINED, "no formal RT source interval threshold exists", threshold=NOT_DEFINED),
    }
    state_limit = safety.get("max_state_age_s")
    if state_limit is None:
        gates["rt_ipc_freshness"] = gate(UNDEFINED, "max_state_age_s is unset", threshold=NOT_DEFINED)
    else:
        stale = summary["rt_ipc"]["formal_stale_count"]
        gates["rt_ipc_freshness"] = gate("PASS" if stale == 0 else "FAIL", "configured max_state_age_s", threshold_s=state_limit, stale_count=stale)
    lateness = safety.get("max_command_lateness_s")
    if lateness is None:
        gates["supervisor_timing"] = gate(UNDEFINED, "max_command_lateness_s is unset", threshold=NOT_DEFINED)
    else:
        late = summary["supervisor"]["formal_late_cycle_count"]
        gates["supervisor_timing"] = gate("PASS" if late == 0 else "FAIL", "configured max_command_lateness_s", threshold_s=lateness, late_count=late)
    if case == "test_a":
        gates["wrench_freshness"] = gate(UNDEFINED, "wrench is intentionally off in Test A")
        gates["wrench_error_reliability"] = gate(UNDEFINED, "wrench is intentionally off in Test A")
    else:
        wrench_limit = safety.get("max_wrench_age_s")
        if wrench_limit is None:
            gates["wrench_freshness"] = gate(UNDEFINED, "max_wrench_age_s is unset", threshold=NOT_DEFINED)
        else:
            stale = summary["wrench"]["formal_stale_count"]
            gates["wrench_freshness"] = gate("PASS" if stale == 0 else "FAIL", "configured max_wrench_age_s", threshold_s=wrench_limit, stale_count=stale)
        gates["wrench_error_reliability"] = gate(UNDEFINED, "no formal wrench failure/error-263 acceptance threshold exists", threshold=NOT_DEFINED)
    return gates


def render_audit(summary: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    source = summary["rt_source"]["interval_ms"]
    ipc = summary["rt_ipc"]["age_ms"]
    current_age = summary["rt_ipc"].get("current_snapshot_age_ms", {})
    loop = summary["supervisor"]["loop_interval_ms"]
    wrench = summary["wrench"]
    lines = [
        f"# Formal W1 state–wrench timing audit: {summary['case']}", "",
        "Strictly read-only. No Servo/Move/Jog/trajectory/mode/power/reset/clear command was issued.", "",
        f"- Result integrity: `{summary['gates']['data_integrity']['status']}`",
        f"- Requested/observed duration: `{summary['protocol']['duration_s']}` / `{fmt(summary['completion']['observed_duration_s'])}` s",
        f"- Git: `{summary['metadata']['git']['branch']}` @ `{summary['metadata']['git']['commit_sha']}` (dirty={summary['metadata']['git']['dirty']})",
        f"- Robot/controller/SDK: `{summary['metadata']['robot'].get('robot_model')}` / `{summary['metadata']['robot'].get('controller_version')}` / `{summary['metadata']['robot'].get('sdk_version')}`",
        "", "## Timing layers", "",
        "| Layer | mean | median | P95 | P99 | max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| RT source interval ms | {fmt(source['mean'])} | {fmt(source['median'])} | {fmt(source['p95'])} | {fmt(source['p99'])} | {fmt(source['max'])} |",
        f"| RT IPC age ms | {fmt(ipc['mean'])} | {fmt(ipc['median'])} | {fmt(ipc['p95'])} | {fmt(ipc['p99'])} | {fmt(ipc['max'])} |",
        f"| RT current snapshot age ms | {fmt(current_age.get('mean'))} | {fmt(current_age.get('median'))} | {fmt(current_age.get('p95'))} | {fmt(current_age.get('p99'))} | {fmt(current_age.get('max'))} |",
        f"| Supervisor loop ms | {fmt(loop['mean'])} | {fmt(loop['median'])} | {fmt(loop['p95'])} | {fmt(loop['p99'])} | {fmt(loop['max'])} |",
        f"| Wrench inter-arrival ms | {fmt(wrench['inter_arrival_ms']['mean'])} | {fmt(wrench['inter_arrival_ms']['median'])} | {fmt(wrench['inter_arrival_ms']['p95'])} | {fmt(wrench['inter_arrival_ms']['p99'])} | {fmt(wrench['inter_arrival_ms']['max'])} |",
        "",
        f"RT IPC delivery-age counts >20/50/100/1000 ms: `{summary['rt_ipc']['descriptive_age_counts']}`.",
        f"RT current-snapshot-age counts >20/50/100/1000 ms: `{summary['rt_ipc'].get('descriptive_current_age_counts')}`.",
        f"Wrench requests/success/failure: `{wrench['request_count']}/{wrench['success_count']}/{wrench['failure_count']}`; errors: `{wrench['error_code_histogram']}`.",
        f"OperationState before/after: `{summary['state_transitions']['before']}` / `{summary['state_transitions']['after']}`; non-idle transitions: `{len(summary['state_transitions']['non_idle_transitions'])}`.",
        "", "## Formal gates", "",
    ]
    for name, result in summary["gates"].items():
        lines.append(f"- `{name}` = `{result['status']}` — {result['reason']}")
    lines.extend([
        "", "## Artifacts", "",
        *[f"- `{name}`: `{path}`" for name, path in summary["artifacts"].items()],
        "", "`SAFE_TO_PROCEED_MANUAL_PUSH = false` (W1 artifacts require review before W2).",
        "", "`READY_FOR_FIRST_MOTION_TEST = false`", "",
    ])
    return "\n".join(lines)


def plot_case(path: Path, series: Mapping[str, list[float]]) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(12, 13), constrained_layout=True)
    plots = (
        ("RT source interval", "rt_source_interval_ms", "ms"),
        ("RT IPC age", "rt_ipc_age_ms", "ms"),
        ("RT current snapshot age", "rt_current_age_ms", "ms"),
        ("Supervisor loop interval", "supervisor_loop_ms", "ms"),
        ("Wrench inter-arrival", "wrench_inter_arrival_ms", "ms"),
    )
    for axis, (title, key, unit) in zip(axes, plots):
        values = series.get(key, [])
        axis.plot(range(len(values)), values, linewidth=0.55)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("sample index")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    formal = load_formal_context(repo)
    protocol = resolve_protocol(args, formal)
    case = "test_a" if args.mode == "test-a" else "test_b"
    offline = args.mode == "offline"
    if offline:
        case = "offline_test_b"
    stamp = artifact_stamp()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"state_wrench_timing_{case}_{stamp}"
    paths = {
        "supervisor_csv": str((Path(str(prefix) + "_supervisor.csv")).resolve()),
        "rt_source_csv": str((Path(str(prefix) + "_rt_source.csv")).resolve()),
        "wrench_csv": str((Path(str(prefix) + "_wrench.csv")).resolve()),
        "events_json": str((Path(str(prefix) + "_events.json")).resolve()),
        "summary_json": str((Path(str(prefix) + "_summary.json")).resolve()),
        "audit_markdown": str((Path(str(prefix) + "_audit.md")).resolve()),
        "timing_plot": str((Path(str(prefix) + "_timing.png")).resolve()),
    }
    print(
        "STRICT READ-ONLY W1: connect/identity/status/RT state/getEndTorque/disconnect only; "
        "no motion, Servo/Move/Jog, trajectory, mode/power, reset, or clear.", flush=True,
    )
    rt = RtProcessSupervisor(stale_age_ms=None, worker_startup_hung_ms=20_000.0)
    wrench = WrenchProcessSupervisor(stale_age_ms=None, worker_startup_hung_ms=20_000.0)
    use_wrench = case != "test_a"
    supervisor_periods: list[float] = []
    rt_ipc_ages: list[float] = []
    rt_current_ages: list[float] = []
    wrench_ages: list[float] = []
    rt_source_periods: list[float] = []
    rt_publish_periods: list[float] = []
    wrench_inter_arrivals: list[float] = []
    wrench_events: list[dict[str, Any]] = []
    worker_events: list[dict[str, Any]] = []
    source_sequence_gap_count = 0
    rt_invalid_count = 0
    rt_formal_stale_count: int | str = NOT_DEFINED
    wrench_formal_stale_count: int | str = NOT_DEFINED
    supervisor_formal_late_count: int | str = NOT_DEFINED
    rt_cleanup: dict[str, Any] = {}
    wrench_cleanup: dict[str, Any] = {}
    fatal_error: str | None = None
    completed = False
    previous_tick_ns: int | None = None
    previous_source: dict[str, int] | None = None
    previous_wrench_start_ns: int | None = None
    last_wrench_alive: bool | None = None
    wrench_hung_latched = False
    rt_hung_latched = False
    rt_crash_count = 0
    wrench_crash_count = 0
    wrench_hung_count = 0
    rt_hung_count = 0
    sample_index = 0
    started_ns: int | None = None
    finished_ns: int | None = None
    supervisor_path = Path(paths["supervisor_csv"])
    source_path = Path(paths["rt_source_csv"])
    wrench_path = Path(paths["wrench_csv"])
    with (
        supervisor_path.open("w", newline="", encoding="utf-8") as supervisor_stream,
        source_path.open("w", newline="", encoding="utf-8") as source_stream,
        wrench_path.open("w", newline="", encoding="utf-8") as wrench_stream,
    ):
        supervisor_writer = csv.DictWriter(supervisor_stream, fieldnames=SUPERVISOR_FIELDS)
        source_writer = csv.DictWriter(source_stream, fieldnames=RT_SOURCE_FIELDS)
        wrench_writer = csv.DictWriter(wrench_stream, fieldnames=WRENCH_FIELDS)
        supervisor_writer.writeheader()
        source_writer.writeheader()
        wrench_writer.writeheader()
        try:
            rt_config: dict[str, Any] = {"offline": True, "publish_hz": 125.0} if offline else {
                "robot_ip": args.robot_ip,
                "local_ip": args.local_ip,
                "robot_class": args.robot_class,
                "state_interval_ms": 8,
                "operation_poll_interval_s": 1.0,
                "allowed_power_states": ["on", "off"],
            }
            rt.start(rt_config)
            _wait_for_rt(rt)
            if use_wrench:
                wrench_config: dict[str, Any] = {
                    "target_hz": protocol["wrench_hz"], "event_queue_size": 4096,
                }
                if offline:
                    wrench_config.update({"behavior": args.offline_wrench_behavior, "delay_s": args.offline_delay_s})
                    wrench.start(mode="offline", config=wrench_config)
                else:
                    wrench_config.update({
                        "robot_ip": args.robot_ip, "local_ip": "",
                        "robot_class": args.robot_class,
                        "allowed_power_states": ["on", "off"],
                    })
                    wrench.start(mode="live", config=wrench_config)
                _wait_for_wrench(wrench)
            started_ns = time.perf_counter_ns()
            deadline_ns = started_ns + int(protocol["duration_s"] * 1e9)
            next_tick_ns = started_ns
            next_progress_ns = started_ns + 60_000_000_000
            while time.perf_counter_ns() < deadline_ns:
                tick_ns = time.perf_counter_ns()
                loop_ms = None if previous_tick_ns is None else (tick_ns - previous_tick_ns) / 1e6
                previous_tick_ns = tick_ns
                if loop_ms is not None:
                    supervisor_periods.append(loop_ms)
                rt_observation = rt.poll(tick_ns)
                wrench_observation = wrench.poll(tick_ns) if use_wrench else None
                if rt_observation.new_snapshot_received and rt_observation.rt_ipc_age_ms is not None:
                    rt_ipc_ages.append(rt_observation.rt_ipc_age_ms)
                if rt_observation.rt_age_ms is not None:
                    rt_current_ages.append(rt_observation.rt_age_ms)
                if not rt_observation.rt_valid:
                    rt_invalid_count += 1
                if wrench_observation is not None and wrench_observation.wrench_age_ms is not None:
                    wrench_ages.append(wrench_observation.wrench_age_ms)
                supervisor_writer.writerow({
                    "case": case, "sample_index": sample_index,
                    "main_loop_timestamp_ns": tick_ns, "loop_interval_ms": loop_ms,
                    "rt_sequence": rt_observation.rt_sequence,
                    "source_or_receive_timestamp_ns": rt_observation.source_or_receive_timestamp_ns,
                    "publish_timestamp_ns": rt_observation.publish_timestamp_ns,
                    "supervisor_receive_timestamp_ns": rt_observation.supervisor_receive_timestamp_ns,
                    "rt_ipc_age_ms": rt_observation.rt_ipc_age_ms,
                    "rt_age_ms": rt_observation.rt_age_ms,
                    "rt_new_snapshot": rt_observation.new_snapshot_received,
                    "rt_valid": rt_observation.rt_valid, "rt_stale": rt_observation.rt_stale,
                    "rt_worker_alive": rt_observation.worker_alive,
                    "rt_worker_state": rt_observation.worker_state,
                    "rt_worker_hung": rt_observation.worker_hung,
                    "rt_heartbeat_age_ms": rt_observation.heartbeat_age_ms,
                    "rt_publish_count": rt_observation.publish_count,
                    "rt_receive_count": rt_observation.receive_count,
                    "rt_overwrite_count": rt_observation.overwrite_count,
                    "rt_publish_drop_count": rt_observation.publish_drop_count,
                    "operation_state": rt_observation.operation_state,
                    "wrench_sequence": 0 if wrench_observation is None else wrench_observation.wrench_sequence,
                    "wrench_last_success_ns": None if wrench_observation is None else wrench_observation.last_wrench_success_ns,
                    "wrench_age_ms": None if wrench_observation is None else wrench_observation.wrench_age_ms,
                    "wrench_valid": None if wrench_observation is None else wrench_observation.wrench_valid,
                    "wrench_stale": None if wrench_observation is None else wrench_observation.wrench_stale,
                    "wrench_worker_alive": None if wrench_observation is None else wrench_observation.worker_alive,
                    "wrench_worker_state": None if wrench_observation is None else wrench_observation.worker_state,
                    "wrench_worker_hung": None if wrench_observation is None else wrench_observation.worker_hung,
                    "wrench_heartbeat_age_ms": None if wrench_observation is None else wrench_observation.heartbeat_age_ms,
                    "wrench_last_error_code": None if wrench_observation is None else wrench_observation.last_error_code,
                    "wrench_last_error": None if wrench_observation is None else wrench_observation.last_error,
                })
                for source_event in rt.drain_source_events():
                    source_interval = None
                    publish_interval = None
                    if previous_source is not None:
                        sequence_delta = int(source_event["rt_sequence"]) - int(previous_source["rt_sequence"])
                        if sequence_delta != 1:
                            source_sequence_gap_count += max(0, sequence_delta - 1)
                        source_interval = (int(source_event["source_or_receive_timestamp_ns"]) - int(previous_source["source_or_receive_timestamp_ns"])) / 1e6
                        publish_interval = (int(source_event["publish_timestamp_ns"]) - int(previous_source["publish_timestamp_ns"])) / 1e6
                        rt_source_periods.append(source_interval)
                        rt_publish_periods.append(publish_interval)
                    source_writer.writerow({
                        **source_event, "source_interval_ms": source_interval,
                        "publish_interval_ms": publish_interval,
                        "source_to_publish_ms": (int(source_event["publish_timestamp_ns"]) - int(source_event["source_or_receive_timestamp_ns"])) / 1e6,
                    })
                    previous_source = source_event
                if use_wrench:
                    for event in wrench.drain_events():
                        interarrival = None
                        if previous_wrench_start_ns is not None:
                            interarrival = (int(event["call_start_ns"]) - previous_wrench_start_ns) / 1e6
                            wrench_inter_arrivals.append(interarrival)
                        previous_wrench_start_ns = int(event["call_start_ns"])
                        stored = dict(event)
                        stored["request_inter_arrival_ms"] = interarrival
                        stored["operation_state"] = rt_observation.operation_state
                        wrench_events.append(stored)
                        wrench_writer.writerow({key: _json_cell(stored.get(key)) for key in WRENCH_FIELDS})
                    assert wrench_observation is not None
                    if last_wrench_alive is True and not wrench_observation.worker_alive:
                        wrench_crash_count += 1
                        worker_events.append({"timestamp_ns": tick_ns, "worker": "wrench", "event": "crash_or_exit", "exitcode": wrench_observation.worker_exitcode})
                    last_wrench_alive = wrench_observation.worker_alive
                    if wrench_observation.worker_hung and not wrench_hung_latched:
                        wrench_hung_count += 1
                        wrench_hung_latched = True
                        worker_events.append({"timestamp_ns": tick_ns, "worker": "wrench", "event": "hung", "heartbeat_age_ms": wrench_observation.heartbeat_age_ms})
                    if not wrench_observation.worker_hung:
                        wrench_hung_latched = False
                if rt_observation.worker_hung and not rt_hung_latched:
                    rt_hung_count += 1
                    rt_hung_latched = True
                    worker_events.append({"timestamp_ns": tick_ns, "worker": "rt", "event": "hung", "heartbeat_age_ms": rt_observation.heartbeat_age_ms})
                    raise RuntimeError("RT worker hung")
                if not rt_observation.worker_hung:
                    rt_hung_latched = False
                if not rt_observation.worker_alive:
                    rt_crash_count += 1
                    worker_events.append({"timestamp_ns": tick_ns, "worker": "rt", "event": "crash_or_exit", "exitcode": rt_observation.worker_exitcode})
                    raise RuntimeError("RT worker exited during acquisition")
                sample_index += 1
                if sample_index % 100 == 0:
                    for stream in (supervisor_stream, source_stream, wrench_stream):
                        stream.flush()
                    if sample_index % 1000 == 0:
                        for stream in (supervisor_stream, source_stream, wrench_stream):
                            os.fsync(stream.fileno())
                if tick_ns >= next_progress_ns:
                    print(f"{case}: {(tick_ns-started_ns)/1e9:.0f}/{protocol['duration_s']:.0f}s rt_seq={rt_observation.rt_sequence} wrench_seq={0 if wrench_observation is None else wrench_observation.wrench_sequence}", flush=True)
                    next_progress_ns += 60_000_000_000
                next_tick_ns += 10_000_000
                remaining_ns = next_tick_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
                else:
                    next_tick_ns = time.perf_counter_ns() + 10_000_000
            completed = True
            finished_ns = time.perf_counter_ns()
        except BaseException as exc:
            fatal_error = f"{type(exc).__name__}:{exc}"
            finished_ns = time.perf_counter_ns()
        finally:
            if use_wrench:
                for event in wrench.drain_events():
                    interarrival = None if previous_wrench_start_ns is None else (int(event["call_start_ns"]) - previous_wrench_start_ns) / 1e6
                    if interarrival is not None:
                        wrench_inter_arrivals.append(interarrival)
                    previous_wrench_start_ns = int(event["call_start_ns"])
                    stored = dict(event)
                    stored["request_inter_arrival_ms"] = interarrival
                    wrench_events.append(stored)
                    wrench_writer.writerow({key: _json_cell(stored.get(key)) for key in WRENCH_FIELDS})
                if wrench.alive:
                    wrench_cleanup = wrench.stop_normally(5.0)
                    if not wrench_cleanup.get("worker_exited"):
                        wrench_cleanup = {**wrench_cleanup, **wrench.terminate()}
                wrench.poll()
                for event in wrench.drain_events():
                    stored = dict(event)
                    stored["request_inter_arrival_ms"] = None
                    wrench_events.append(stored)
                    wrench_writer.writerow({key: _json_cell(stored.get(key)) for key in WRENCH_FIELDS})
            if rt.alive:
                rt_cleanup = rt.stop_normally(5.0)
                if not rt_cleanup.get("worker_exited"):
                    rt_cleanup = {**rt_cleanup, **rt.terminate()}
            rt.poll()
            for source_event in rt.drain_source_events():
                source_interval = None if previous_source is None else (int(source_event["source_or_receive_timestamp_ns"]) - int(previous_source["source_or_receive_timestamp_ns"])) / 1e6
                publish_interval = None if previous_source is None else (int(source_event["publish_timestamp_ns"]) - int(previous_source["publish_timestamp_ns"])) / 1e6
                if source_interval is not None:
                    rt_source_periods.append(source_interval)
                    rt_publish_periods.append(publish_interval)
                source_writer.writerow({**source_event, "source_interval_ms": source_interval, "publish_interval_ms": publish_interval, "source_to_publish_ms": (int(source_event["publish_timestamp_ns"]) - int(source_event["source_or_receive_timestamp_ns"])) / 1e6})
                previous_source = source_event
            for stream in (supervisor_stream, source_stream, wrench_stream):
                stream.flush()
                os.fsync(stream.fileno())
            rt_metadata = rt.metadata
            wrench_metadata = wrench.metadata if use_wrench else {}
            if use_wrench:
                wrench.close()
            rt.close()
    observed_duration_s = 0.0 if started_ns is None or finished_ns is None else (finished_ns - started_ns) / 1e9
    safety = formal["safety_config"]
    if safety.get("max_state_age_s") is not None:
        rt_formal_stale_count = sum(value > float(safety["max_state_age_s"]) * 1000.0 for value in rt_current_ages)
    if safety.get("max_wrench_age_s") is not None:
        wrench_formal_stale_count = sum(value > float(safety["max_wrench_age_s"]) * 1000.0 for value in wrench_ages)
    if safety.get("max_command_lateness_s") is not None:
        expected_ms = 1000.0 / protocol["supervisor_hz"]
        supervisor_formal_late_count = sum(value - expected_ms > float(safety["max_command_lateness_s"]) * 1000.0 for value in supervisor_periods)
    failures = [event for event in wrench_events if not bool(event.get("success"))]
    transitions = list(rt_metadata.get("operation_state_transitions") or [])
    non_idle = [event for event in transitions if event.get("from") is not None and str(event.get("to", "")).lower() != "idle"]
    summary: dict[str, Any] = {
        "schema_version": 1, "audit": "formal_state_wrench_timing_W1", "case": case,
        "strict_read_only": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "metadata": {
            "git": git_metadata(repo), "windows_version": platform.platform(),
            "windows_release": platform.win32_ver(), "python_version": sys.version,
            "clock": "time.perf_counter_ns host monotonic high-resolution",
            "wall_clock_used_for_latency": False,
            "formal_context": formal,
            "robot": rt_metadata,
            "wrench_worker": wrench_metadata,
            "requested_rates": {"supervisor_hz": 100.0, "rt_interval_ms": 8, "wrench_hz": 0.0 if not use_wrench else protocol["wrench_hz"]},
            "actual_sample_counts": {"supervisor": sample_index, "rt_source": len(rt_source_periods) + (1 if previous_source else 0), "wrench_requests_logged": len(wrench_events)},
        },
        "completion": {"completed_requested_duration": completed, "requested_duration_s": protocol["duration_s"], "observed_duration_s": observed_duration_s, "fatal_error": fatal_error},
        "rt_source": {
            "interval_ms": metric_summary(rt_source_periods),
            "publish_interval_ms": metric_summary(rt_publish_periods),
            "configured_interval_threshold": NOT_DEFINED,
            "count_above_configured_threshold": NOT_DEFINED,
            "descriptive_interval_counts": threshold_counts(rt_source_periods, (10, 20, 50, 100, 1000)),
            "sequence_gap_count": source_sequence_gap_count,
            "ring_overwrite_count": rt.source_ring_overwrite_count,
            "update_timeout_count": int(rt_metadata.get("update_timeout_count", 0)),
        },
        "rt_ipc": {
            "age_ms": metric_summary(rt_ipc_ages),
            "current_snapshot_age_ms": metric_summary(rt_current_ages),
            "descriptive_age_counts": threshold_counts(rt_ipc_ages, (20, 50, 100, 1000)),
            "descriptive_current_age_counts": threshold_counts(rt_current_ages, (20, 50, 100, 1000)),
            "formal_stale_threshold_s": safety.get("max_state_age_s") if safety.get("max_state_age_s") is not None else NOT_DEFINED,
            "formal_stale_count": rt_formal_stale_count,
            "invalid_tick_count": rt_invalid_count,
        },
        "supervisor": {
            "loop_interval_ms": metric_summary(supervisor_periods),
            "formal_lateness_threshold_s": safety.get("max_command_lateness_s") if safety.get("max_command_lateness_s") is not None else NOT_DEFINED,
            "formal_late_cycle_count": supervisor_formal_late_count,
            "missed_cycle_count": NOT_DEFINED if safety.get("max_command_lateness_s") is None else supervisor_formal_late_count,
        },
        "wrench": {
            "requested_rate_hz": 0.0 if not use_wrench else protocol["wrench_hz"],
            "request_count": int(wrench_metadata.get("request_count", len(wrench_events))) if use_wrench else 0,
            "success_count": int(wrench_metadata.get("success_count", sum(bool(event.get("success")) for event in wrench_events))) if use_wrench else 0,
            "failure_count": int(wrench_metadata.get("failure_count", len(failures))) if use_wrench else 0,
            "inter_arrival_ms": metric_summary(wrench_inter_arrivals),
            "query_latency_ms": metric_summary(event.get("query_latency_ms") for event in wrench_events),
            "host_observed_data_age_ms": metric_summary(wrench_ages),
            "formal_stale_threshold_s": safety.get("max_wrench_age_s") if safety.get("max_wrench_age_s") is not None else NOT_DEFINED,
            "formal_stale_count": wrench_formal_stale_count,
            "error_code_histogram": error_histogram(wrench_events),
            "error_count_263": sum(int(event.get("error_code") or -1) == 263 for event in failures),
            "consecutive_error_runs": consecutive_error_runs(wrench_events),
            "event_publish_count": int(wrench_metadata.get("event_publish_count", len(wrench_events))) if use_wrench else 0,
            "event_drop_count": int(wrench_metadata.get("event_drop_count", 0)) if use_wrench else 0,
        },
        "state_transitions": {"before": rt_metadata.get("operation_state_before", rt_metadata.get("operation_state")), "after": rt_metadata.get("operation_state_after"), "poll_interval_s": rt_metadata.get("operation_poll_interval_s"), "timeline": transitions, "non_idle_transitions": non_idle},
        "workers": {
            "rt_hung_event_count": rt_hung_count, "rt_crash_event_count": rt_crash_count,
            "wrench_hung_event_count": wrench_hung_count, "wrench_crash_event_count": wrench_crash_count,
            "wrench_restart_count": 0, "events": worker_events,
            "rt_cleanup": rt_cleanup, "wrench_cleanup": wrench_cleanup,
        },
        "artifacts": paths,
        "SAFE_TO_PROCEED_MANUAL_PUSH": False,
        "READY_FOR_FIRST_MOTION_TEST": False,
    }
    summary["gates"] = build_gates(case="test_a" if case == "test_a" else "test_b", summary=summary, safety=safety)
    events_payload = {
        "error_timeline": error_timeline(wrench_events),
        "consecutive_error_runs": summary["wrench"]["consecutive_error_runs"],
        "worker_events": worker_events,
        "operation_state_transitions": transitions,
        "error_263_explicitly_preserved": True,
    }
    Path(paths["events_json"]).write_text(json.dumps(events_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["summary_json"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["audit_markdown"]).write_text(render_audit(summary), encoding="utf-8")
    plot_case(Path(paths["timing_plot"]), {
        "rt_source_interval_ms": rt_source_periods, "rt_ipc_age_ms": rt_ipc_ages,
        "rt_current_age_ms": rt_current_ages,
        "supervisor_loop_ms": supervisor_periods, "wrench_inter_arrival_ms": wrench_inter_arrivals,
    })
    print(f"summary={paths['summary_json']} completed={completed} fatal_error={fatal_error}", flush=True)
    return summary


def render_comparison(a: Mapping[str, Any], b: Mapping[str, Any], artifacts: Mapping[str, str]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    rows = []
    for label, summary in (("Test A", a), ("Test B", b)):
        rows.append(
            f"| {label} | {fmt(summary['rt_source']['interval_ms']['p99'])} / {fmt(summary['rt_source']['interval_ms']['max'])} | "
            f"{fmt(summary['rt_ipc']['age_ms']['p95'])} / {fmt(summary['rt_ipc']['age_ms']['p99'])} / {fmt(summary['rt_ipc']['age_ms']['max'])} | "
            f"{fmt(summary['rt_ipc'].get('current_snapshot_age_ms', {}).get('max'))} | "
            f"{fmt(summary['supervisor']['loop_interval_ms']['p99'])} / {fmt(summary['supervisor']['loop_interval_ms']['max'])} | "
            f"{summary['wrench']['request_count']} / {summary['wrench']['failure_count']} |"
        )
    return "\n".join([
        "# Formal W1 Test A vs Test B comparison", "",
        "No A-vs-B degradation PASS threshold exists in the formal manifest; comparison gate is `UNDEFINED`.", "",
        "| Case | RT source P99/max ms | RT IPC P95/P99/max ms | Current age max ms | Supervisor P99/max ms | Wrench requests/failures |",
        "|---|---:|---:|---:|---:|---:|", *rows, "",
        f"- Test A completed: `{a['completion']['completed_requested_duration']}` ({a['completion']['observed_duration_s']:.3f}/{a['completion']['requested_duration_s']:.3f} s)",
        f"- Test B completed: `{b['completion']['completed_requested_duration']}` ({b['completion']['observed_duration_s']:.3f}/{b['completion']['requested_duration_s']:.3f} s); fatal error: `{b['completion']['fatal_error']}`",
        f"- Test B error histogram: `{b['wrench']['error_code_histogram']}`",
        f"- Test B error 263 count: `{b['wrench']['error_count_263']}`",
        f"- Comparison gate: `{UNDEFINED}` (`{NOT_DEFINED}`)",
        "- SAFE_TO_PROCEED_MANUAL_PUSH = false (requires W1 review)",
        "- READY_FOR_FIRST_MOTION_TEST = false", "", "## Artifacts", "",
        *[f"- `{key}`: `{value}`" for key, value in artifacts.items()], "",
    ])


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    a = json.loads(Path(args.test_a_summary).read_text(encoding="utf-8"))
    b = json.loads(Path(args.test_b_summary).read_text(encoding="utf-8"))
    stamp = artifact_stamp()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"state_wrench_timing_comparison_{stamp}"
    artifacts = {
        "comparison_json": str(Path(str(prefix) + ".json").resolve()),
        "comparison_markdown": str(Path(str(prefix) + ".md").resolve()),
        "comparison_plot": str(Path(str(prefix) + ".png").resolve()),
    }
    comparison = {
        "schema_version": 1, "audit": "formal_state_wrench_timing_W1_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_a_summary": str(Path(args.test_a_summary).resolve()),
        "test_b_summary": str(Path(args.test_b_summary).resolve()),
        "descriptive_delta": {
            "rt_source_p99_ms": b["rt_source"]["interval_ms"]["p99"] - a["rt_source"]["interval_ms"]["p99"],
            "rt_ipc_p99_ms": b["rt_ipc"]["age_ms"]["p99"] - a["rt_ipc"]["age_ms"]["p99"],
            "supervisor_p99_ms": b["supervisor"]["loop_interval_ms"]["p99"] - a["supervisor"]["loop_interval_ms"]["p99"],
        },
        "comparison_gate": {"status": UNDEFINED, "reason": "no formal A-vs-B degradation threshold", "threshold": NOT_DEFINED},
        "artifacts": artifacts, "SAFE_TO_PROCEED_MANUAL_PUSH": False,
        "READY_FOR_FIRST_MOTION_TEST": False,
    }
    Path(artifacts["comparison_json"]).write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(artifacts["comparison_markdown"]).write_text(render_comparison(a, b, artifacts), encoding="utf-8")
    labels = ["source P99", "IPC P99", "supervisor P99"]
    a_values = [a["rt_source"]["interval_ms"]["p99"], a["rt_ipc"]["age_ms"]["p99"], a["supervisor"]["loop_interval_ms"]["p99"]]
    b_values = [b["rt_source"]["interval_ms"]["p99"], b["rt_ipc"]["age_ms"]["p99"], b["supervisor"]["loop_interval_ms"]["p99"]]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = range(len(labels))
    axis.bar([value - 0.2 for value in positions], a_values, width=0.4, label="Test A")
    axis.bar([value + 0.2 for value in positions], b_values, width=0.4, label="Test B")
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("ms")
    axis.set_title("Formal W1 descriptive P99 comparison")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(artifacts["comparison_plot"], dpi=140)
    plt.close(figure)
    print(f"comparison={artifacts['comparison_markdown']}", flush=True)
    return comparison


def _csv_floats(path: str, field: str, *, require_new: bool = False) -> list[float]:
    values: list[float] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if require_new and str(row.get("rt_new_snapshot", "")).lower() != "true":
                continue
            value = row.get(field)
            if value not in (None, ""):
                values.append(float(value))
    return values


def refresh_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Re-derive freshness fields from immutable raw CSV logs."""

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    supervisor_csv = summary["artifacts"]["supervisor_csv"]
    current_ages = _csv_floats(supervisor_csv, "rt_age_ms")
    delivery_ages = _csv_floats(supervisor_csv, "rt_ipc_age_ms", require_new=True)
    summary["rt_ipc"]["age_ms"] = metric_summary(delivery_ages)
    summary["rt_ipc"]["current_snapshot_age_ms"] = metric_summary(current_ages)
    summary["rt_ipc"]["descriptive_age_counts"] = threshold_counts(delivery_ages, (20, 50, 100, 1000))
    summary["rt_ipc"]["descriptive_current_age_counts"] = threshold_counts(current_ages, (20, 50, 100, 1000))
    safety = summary["metadata"]["formal_context"]["safety_config"]
    if safety.get("max_state_age_s") is not None:
        summary["rt_ipc"]["formal_stale_count"] = sum(
            value > float(safety["max_state_age_s"]) * 1000.0 for value in current_ages
        )
    summary["gates"] = build_gates(
        case=summary["case"], summary=summary, safety=safety
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(summary["artifacts"]["audit_markdown"]).write_text(render_audit(summary), encoding="utf-8")
    plot_case(Path(summary["artifacts"]["timing_plot"]), {
        "rt_source_interval_ms": _csv_floats(summary["artifacts"]["rt_source_csv"], "source_interval_ms"),
        "rt_ipc_age_ms": delivery_ages,
        "rt_current_age_ms": current_ages,
        "supervisor_loop_ms": _csv_floats(supervisor_csv, "loop_interval_ms"),
        "wrench_inter_arrival_ms": _csv_floats(summary["artifacts"]["wrench_csv"], "request_inter_arrival_ms"),
    })
    print(f"refreshed={summary_path.resolve()}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("offline", "test-a", "test-b", "compare", "refresh"))
    parser.add_argument("--robot-ip", default="192.168.50.103")
    parser.add_argument("--local-ip", default="192.168.50.209")
    parser.add_argument("--robot-class", default="xMateRobot")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--wrench-hz", type=float)
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument("--offline-wrench-behavior", default="normal", choices=("normal", "error263", "exception"))
    parser.add_argument("--offline-delay-s", type=float, default=0.001)
    parser.add_argument("--test-a-summary")
    parser.add_argument("--test-b-summary")
    parser.add_argument("--summary")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.wrench_hz is not None and args.wrench_hz <= 0:
        parser.error("--wrench-hz must be positive")
    if args.mode == "compare" and (not args.test_a_summary or not args.test_b_summary):
        parser.error("compare requires --test-a-summary and --test-b-summary")
    if args.mode == "refresh" and not args.summary:
        parser.error("refresh requires --summary")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "compare":
        run_compare(args)
    elif args.mode == "refresh":
        refresh_summary(args)
    else:
        run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
