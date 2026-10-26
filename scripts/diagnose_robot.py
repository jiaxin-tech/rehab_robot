"""Read-only state, wrench-query and alignment diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Iterator, Sequence

from config import settings
import time
from collection.state import state_age_ms
from collection.snapshot import read_live_robot_state_sample
from hardware.windows.rokae_internal_wrench import RokaeInternalWrenchSource
from hardware.windows.rokae_xcore import RokaeRobot


def make_robot(robot_ip: str, local_ip: str | None = None) -> RokaeRobot:
    """Build the normal project adapter without changing robot state."""
    return RokaeRobot(
        ip_address=robot_ip,
        local_ip=settings.ROBOT_LOCAL_IP if local_ip is None else local_ip,
        robot_class=settings.ROBOT_CLASS,
        state_interval_ms=settings.ROBOT_STATE_MS,
        max_linear_speed_m_s=settings.ROBOT_MAX_LINEAR_SPEED_M_S,
        command_cache_size=settings.ROBOT_CMD_CACHE,
        rt_network_tolerance_percent=settings.ROBOT_RT_NETWORK_TOLERANCE,
        rt_filter_hz=settings.ROBOT_RT_FILTER_HZ,
    )


@contextmanager
def readonly_connection(
    robot: Any,
    *,
    use_wrench_stream: bool = False,
) -> Iterator[tuple[Any, RokaeInternalWrenchSource | None]]:
    """Connect and always release read-only diagnostic resources.

    ``disconnect`` only tears down local SDK/session resources.  This context
    intentionally does not call ``enable``, motion APIs, force calibration,
    drag mode, or ``SafetyGuard`` (which can issue a stop command).
    """
    source: RokaeInternalWrenchSource | None = None
    try:
        robot.connect()
        if use_wrench_stream:
            source = RokaeInternalWrenchSource(robot)
            source.connect()
            source.start_streaming()
        yield robot, source
    finally:
        if source is not None:
            try:
                source.disconnect()
            except Exception as exc:
                print(f"Warning: wrench source cleanup failed: {type(exc).__name__}: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Warning: robot cleanup failed: {type(exc).__name__}: {exc}")


def require_confirmed_bias(source: RokaeInternalWrenchSource, confirmed: bool) -> tuple[float, ...]:
    """Apply existing session-local software bias only after an explicit ack."""
    if not confirmed:
        raise ValueError(
            "Software bias requires --confirm-unloaded; keep the configured tool/load "
            "stationary with no human contact. This does not calibrate the controller."
        )
    return source.set_bias()


def percentile(values: Sequence[float], fraction: float) -> float | None:
    usable = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not usable:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    offset = (len(usable) - 1) * fraction
    lower = math.floor(offset)
    upper = math.ceil(offset)
    if lower == upper:
        return usable[lower]
    return usable[lower] + (usable[upper] - usable[lower]) * (offset - lower)


def numeric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "count": len(usable),
        "mean": fmean(usable) if usable else None,
        "p50": percentile(usable, 0.50),
        "p95": percentile(usable, 0.95),
        "p99": percentile(usable, 0.99),
        "min": min(usable) if usable else None,
        "max": max(usable) if usable else None,
    }


def invalid_reason_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("invalid_reason") or "").split(";"):
            if reason.strip():
                counts[reason.strip()] += 1
    return dict(sorted(counts.items()))


def summarize_recording(session: Path) -> dict:
    """Summarize actual mainline snapshots, including startup and failed runs."""
    path = Path(session) / "robot/aligned_snapshot.csv"
    fields = {"query_duration_ms": 1., "state_age_s": 1000.,
              "wrench_age_s": 1000., "state_wrench_skew_s": 1000.}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not set(fields).union({"valid", "host_time_s", "invalid_reason"}).issubset(reader.fieldnames or []):
                raise ValueError("aligned snapshot columns missing")
            rows = list(reader)
        times = [float(row["host_time_s"]) for row in rows]
        if not all(math.isfinite(value) and value >= 0 for value in times):
            raise ValueError("invalid snapshot timestamps")
        metrics, invalid_numeric = {}, 0
        for field, scale in fields.items():
            values = []
            for row in rows:
                raw = row[field]
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    invalid_numeric += 1
                    continue
                if not math.isfinite(value) or value < 0:
                    invalid_numeric += 1
                    continue
                values.append(value * scale)
            metrics[field.removesuffix("_s").removesuffix("_ms") + "_ms"] = numeric_summary(values)
        valid = [str(row["valid"]).lower() == "true" for row in rows]
        first = next((i for i, good in enumerate(valid) if good), None)
        return dict(available=True, samples=len(rows), valid_samples=sum(valid),
            recorded_span_s=times[-1] - times[0] if times else 0.,
            timestamp_order_valid=all(b > a for a, b in zip(times, times[1:])),
            maximum_snapshot_gap_ms=max(((b-a)*1000. for a, b in zip(times, times[1:])), default=None),
            includes_startup=True, ever_ready=first is not None,
            invalid_after_ready=None if first is None else valid[first:].count(False),
            invalid_numeric_values=invalid_numeric, metrics=metrics,
            invalid_reasons=invalid_reason_counts(rows))
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        return dict(available=False, reason=f"{type(exc).__name__}:{exc}")


def sequence_drops(sequence_ids: Sequence[int | None]) -> int:
    """Count missing frames between increasing source sequence identifiers."""
    dropped = 0
    previous: int | None = None
    for value in sequence_ids:
        if value is None:
            continue
        current = int(value)
        if previous is not None and current > previous + 1:
            dropped += current - previous - 1
        previous = current
    return dropped


def vec_mean(vectors: Iterable[Sequence[float] | None]) -> tuple[float, float, float] | None:
    usable = [tuple(float(value) for value in vector) for vector in vectors if vector is not None and len(vector) == 3 and all(math.isfinite(float(value)) for value in vector)]
    if not usable:
        return None
    return tuple(fmean(vector[index] for vector in usable) for index in range(3))  # type: ignore[return-value]


def vec_subtract(left: Sequence[float] | None, right: Sequence[float] | None) -> tuple[float, float, float] | None:
    if left is None or right is None or len(left) != 3 or len(right) != 3:
        return None
    result = tuple(float(left[index]) - float(right[index]) for index in range(3))
    return result if all(math.isfinite(value) for value in result) else None  # type: ignore[return-value]


def vector_norm(vector: Sequence[float] | None) -> float | None:
    if vector is None or len(vector) != 3 or not all(math.isfinite(float(value)) for value in vector):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def rotation_push_analysis(
    baseline_base_force_n: Iterable[Sequence[float] | None],
    pushed_base_force_n: Iterable[Sequence[float] | None],
    expected_axis: str,
) -> dict[str, Any]:
    """Describe a manual positive-axis push without declaring it verified."""
    axis = expected_axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("expected_axis must be X, Y, or Z")
    baseline = vec_mean(baseline_base_force_n)
    pushed = vec_mean(pushed_base_force_n)
    delta = vec_subtract(pushed, baseline)
    if delta is None:
        return {
            "expected_axis": axis,
            "baseline_mean_base_force_n": baseline,
            "push_mean_base_force_n": pushed,
            "delta_base_force_n": None,
            "principal_axis": None,
            "principal_sign": None,
            "cross_axis_ratio": None,
            "expected_axis_positive": None,
        }
    principal_index = max(range(3), key=lambda index: abs(delta[index]))
    magnitude = abs(delta[principal_index])
    cross = math.sqrt(sum(delta[index] ** 2 for index in range(3) if index != principal_index))
    expected_index = "XYZ".index(axis)
    return {
        "expected_axis": axis,
        "baseline_mean_base_force_n": baseline,
        "push_mean_base_force_n": pushed,
        "delta_base_force_n": delta,
        "principal_axis": "XYZ"[principal_index],
        "principal_sign": "+" if delta[principal_index] >= 0.0 else "-",
        "cross_axis_ratio": cross / magnitude if magnitude > 0.0 else None,
        "expected_axis_positive": delta[expected_index] > 0.0,
    }


def pose_dependence_analysis(
    pose_rows: Sequence[dict[str, Any]],
    *,
    force_threshold_n: float,
    torque_threshold_nm: float,
) -> dict[str, Any]:
    """Assess residual change across manually selected static poses.

    This reports whether the *software-bias residual* changes with pose.  It
    intentionally makes no statement about SDK gravity compensation.
    """
    def maximum_delta(field: str) -> float | None:
        vectors = [row.get(field) for row in pose_rows]
        baseline = next((value for value in vectors if vector_norm(value) is not None), None)
        if baseline is None:
            return None
        deltas = [vector_norm(vec_subtract(value, baseline)) for value in vectors]
        usable = [value for value in deltas if value is not None]
        return max(usable) if usable else None

    force_delta = maximum_delta("corrected_force_mean_n")
    torque_delta = maximum_delta("corrected_torque_mean_nm")
    observed = (
        (force_delta is not None and force_delta > force_threshold_n)
        or (torque_delta is not None and torque_delta > torque_threshold_nm)
    )
    return {
        "pose_count": len(pose_rows),
        "max_corrected_force_delta_n": force_delta,
        "max_corrected_torque_delta_nm": torque_delta,
        "force_threshold_n": force_threshold_n,
        "torque_threshold_nm": torque_threshold_nm,
        "software_bias_pose_dependence_observed": observed,
        "interpretation": (
            "This is an empirical residual-bias result only; it does not establish "
            "whether xCoreSDK applies gravity compensation."
        ),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def write_report(output_dir: str | Path, name: str, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    """Keep samples in CSV and write only summary metrics to JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    csv_path = destination / f"{stem}.csv"
    json_path = destination / f"{stem}.json"
    fieldnames = sorted({key for row in rows for key in row}) or ["sample_index"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(value) if not isinstance(value, (list, tuple, dict)) else json.dumps(_json_safe(value), ensure_ascii=False) for key, value in row.items()})
    payload = {"summary": _json_safe(summary), "samples_csv": csv_path.name}
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return csv_path, json_path

def measure_state(robot: Any, *, duration_s: float, poll_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect cached RT frames; dependency injection keeps this fake-testable."""
    period_ns = int(1_000_000_000 / poll_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    previous_perf_ns: int | None = None
    previous_state_time_s: float | None = None
    previous_sequence_id: int | None = None
    rows: list[dict[str, Any]] = []
    exceptions = 0
    while time.perf_counter_ns() < deadline_ns:
        started_ns = time.perf_counter_ns()
        try:
            frame = robot.get_state_frame()
            # RokaeRobot timestamps accepted RT frames with perf_counter_ns;
            # use the identical host clock here to avoid cross-clock epoch
            # ambiguity on Windows.
            now_monotonic_s = time.perf_counter_ns() / 1_000_000_000.0
            age_ms = state_age_ms(now_monotonic_s, frame.host_monotonic_time_s)
            update_period_ms = None
            if (
                previous_sequence_id is None or frame.sequence_id != previous_sequence_id
            ) and frame.host_monotonic_time_s is not None and previous_state_time_s is not None:
                update_period_ms = (frame.host_monotonic_time_s - previous_state_time_s) * 1000.0
            if frame.sequence_id != previous_sequence_id and frame.host_monotonic_time_s is not None:
                previous_state_time_s = frame.host_monotonic_time_s
            previous_sequence_id = frame.sequence_id
            row = {
                "sample_index": len(rows), "perf_counter_ns": started_ns,
                "sequence_id": frame.sequence_id, "valid": frame.valid,
                "invalid_reason": frame.invalid_reason, "operation_state": frame.operation_state,
                "state_host_time_s": frame.host_monotonic_time_s,
                "state_age_ms": age_ms,
                "rt_update_period_ms": update_period_ms,
                "sample_period_ms": None if previous_perf_ns is None else (started_ns - previous_perf_ns) / 1e6,
            }
            previous_perf_ns = started_ns
        except Exception as exc:
            exceptions += 1
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "valid": False,
                   "invalid_reason": f"state_read_error:{type(exc).__name__}:{exc}", "sequence_id": None,
                   "state_age_ms": None, "sample_period_ms": None, "operation_state": None}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    update_summary = numeric_summary(row.get("rt_update_period_ms") for row in rows)
    summary = {
        "diagnostic": "rt_state_timing", "requested_poll_hz": poll_hz, "sample_count": len(rows),
        "valid_count": sum(bool(row["valid"]) for row in rows), "exception_count": exceptions,
        "dropped_frames": sequence_drops([row.get("sequence_id") for row in rows]),
        "state_age_ms": numeric_summary(row.get("state_age_ms") for row in rows),
        "rt_update_period_ms": update_summary,
        "sample_period_ms": numeric_summary(row.get("sample_period_ms") for row in rows),
        "estimated_rt_update_hz": (
            1000.0 / float(update_summary["mean"])
            if update_summary["mean"] else None
        ),
        "unique_state_frames": len({row.get("sequence_id") for row in rows if row.get("sequence_id") is not None}),
        "invalid_reasons": invalid_reason_counts(rows),
    }
    return rows, summary


def measure_wrench(robot: Any, *, duration_s: float, target_hz: float, reference_frame: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    period_ns = int(1_000_000_000 / target_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    while time.perf_counter_ns() < deadline_ns:
        scheduled_ns = next_tick_ns
        started_ns = time.perf_counter_ns()
        try:
            raw = robot.get_end_wrench(reference_frame)
            finished_ns = time.perf_counter_ns()
            row = {"sample_index": len(rows), "scheduled_perf_ns": scheduled_ns, "started_perf_ns": started_ns,
                   "finished_perf_ns": finished_ns, "query_duration_ms": (finished_ns - started_ns) / 1e6,
                   "start_lateness_ms": max(0, started_ns - scheduled_ns) / 1e6, "valid": True,
                   "invalid_reason": "", "adapter_query_started_s": raw.get("force_query_started_s"),
                   "adapter_query_finished_s": raw.get("force_query_finished_s"), "raw_force_frame": raw.get("raw_force_frame")}
        except Exception as exc:
            finished_ns = time.perf_counter_ns()
            row = {"sample_index": len(rows), "scheduled_perf_ns": scheduled_ns, "started_perf_ns": started_ns,
                   "finished_perf_ns": finished_ns, "query_duration_ms": (finished_ns - started_ns) / 1e6,
                   "start_lateness_ms": max(0, started_ns - scheduled_ns) / 1e6, "valid": False,
                   "invalid_reason": f"get_end_torque_error:{type(exc).__name__}:{exc}"}
        # A query misses the acquisition deadline when it alone consumes the
        # whole period. Scheduler lateness is retained separately so ordinary
        # host wake-up jitter is not misreported as an SDK-query failure.
        row["deadline_miss"] = row["query_duration_ms"] > period_ns / 1e6
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    durations = numeric_summary(row["query_duration_ms"] for row in rows)
    summary = {"diagnostic": "wrench_query_timing", "target_hz": target_hz, "deadline_ms": period_ns / 1e6,
               "sample_count": len(rows), "valid_count": sum(bool(row["valid"]) for row in rows),
               "deadline_miss_count": sum(bool(row["deadline_miss"]) for row in rows),
               "query_duration_ms": durations,
               "stable_50hz_supported": target_hz == 50.0 and sum(bool(row["deadline_miss"]) for row in rows) == 0 and sum(not bool(row["valid"]) for row in rows) == 0}
    return rows, summary


def measure_alignment(robot: Any, source: Any, *, duration_s: float, sample_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    period_ns = int(1_000_000_000 / sample_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    previous = None
    rows: list[dict[str, Any]] = []
    while time.perf_counter_ns() < deadline_ns:
        started_ns = time.perf_counter_ns()
        try:
            sample = read_live_robot_state_sample(robot, source, previous_sample=previous)
            previous = sample
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "sequence_id": sample.sequence_id,
                   "valid": sample.valid, "invalid_reason": sample.invalid_reason,
                   "robot_state_age_ms": sample.robot_state_age_ms, "force_sample_age_ms": sample.force_sample_age_ms,
                   "state_internal_skew_ms": sample.state_internal_skew_ms, "force_query_duration_ms": sample.force_query_duration_ms,
                   "force_estimate_valid": sample.force_estimate_valid, "base_transform_kind": sample.base_transform_kind}
        except Exception as exc:
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "sequence_id": None, "valid": False,
                   "invalid_reason": f"snapshot_read_error:{type(exc).__name__}:{exc}", "robot_state_age_ms": None,
                   "force_sample_age_ms": None, "state_internal_skew_ms": None, "force_query_duration_ms": None,
                   "force_estimate_valid": False, "base_transform_kind": None}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    summary = {"diagnostic": "snapshot_alignment", "sample_hz": sample_hz, "sample_count": len(rows),
               "valid_count": sum(bool(row["valid"]) for row in rows),
               "robot_state_age_ms": numeric_summary(row["robot_state_age_ms"] for row in rows),
               "force_sample_age_ms": numeric_summary(row["force_sample_age_ms"] for row in rows),
               "state_internal_skew_ms": numeric_summary(row["state_internal_skew_ms"] for row in rows),
               "force_query_duration_ms": numeric_summary(row["force_query_duration_ms"] for row in rows),
               "invalid_reasons": invalid_reason_counts(rows)}
    return rows, summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Read-only robot timing diagnostics")
    parser.add_argument("kind", choices=("state", "wrench", "alignment"))
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--local-ip", default=settings.ROBOT_LOCAL_IP)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--software-bias", action="store_true")
    parser.add_argument("--confirm-unloaded", action="store_true")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.duration, args.hz)):
        parser.error("--duration and --hz must be finite and positive")
    if args.software_bias and (args.kind != "alignment" or not args.confirm_unloaded):
        parser.error("software bias requires alignment and --confirm-unloaded")
    rows, summary = [], {"diagnostic": args.kind}
    try:
        with readonly_connection(make_robot(args.robot_ip, args.local_ip),
                                 use_wrench_stream=args.kind == "alignment") as (robot, source):
            if args.kind == "state":
                rows, summary = measure_state(robot, duration_s=args.duration, poll_hz=args.hz)
            elif args.kind == "wrench":
                rows, summary = measure_wrench(robot, duration_s=args.duration, target_hz=args.hz,
                                              reference_frame=settings.ROBOT_FORCE_RAW_FRAME)
            else:
                if args.software_bias:
                    require_confirmed_bias(source, args.confirm_unloaded)
                rows, summary = measure_alignment(robot, source, duration_s=args.duration, sample_hz=args.hz)
                summary["software_bias_used"] = args.software_bias
                summary["base_rotation_config_verified"] = settings.BASE_WRENCH_ROTATION_VERIFIED
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, args.kind, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    return int(bool(summary.get("fatal_error")) or not summary.get("valid_count"))


def sleep_until(next_tick_ns: int, clock_ns: Callable[[], int], sleep: Callable[[float], None]) -> None:
    remaining_ns = next_tick_ns - clock_ns()
    if remaining_ns > 0:
        sleep(remaining_ns / 1_000_000_000.0)


if __name__ == "__main__":
    raise SystemExit(main())
