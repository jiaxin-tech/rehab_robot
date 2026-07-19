"""Shared, read-only helpers for ROKAE xCoreSDK diagnostic scripts.

The helpers deliberately live under ``scripts`` so they remain tooling rather
than part of the collection/control contract.  They create the same adapter
and internal-wrench source used by collection, but never enable, move, drag,
calibrate, or stop the robot.
"""

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
from hardware.windows import RokaeInternalWrenchSource, RokaeRobot


def make_robot(robot_ip: str) -> RokaeRobot:
    """Build the normal project adapter without changing robot state."""
    return RokaeRobot(
        ip_address=robot_ip,
        local_ip=settings.ROBOT_LOCAL_IP,
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
    """Persist both machine-readable forms with a stable, flat CSV record."""
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
    payload = {"summary": _json_safe(summary), "rows": _json_safe(list(rows))}
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return csv_path, json_path


def sleep_until(next_tick_ns: int, clock_ns: Callable[[], int], sleep: Callable[[float], None]) -> None:
    remaining_ns = next_tick_ns - clock_ns()
    if remaining_ns > 0:
        sleep(remaining_ns / 1_000_000_000.0)
