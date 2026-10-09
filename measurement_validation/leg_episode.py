"""TwinCAT dummy-leg episode importer and data-quality checker.

The 2026-10-07 acquisition wrote four CSVs per episode.  Those files cannot be
used as identification data as-is, and this module is what makes that explicit
instead of letting a downstream script quietly misinterpret them:

* four axes were read one after another (``not_atomic_axes_read_sequentially``),
  so a single row is not a single instant;
* the observed loop rate was about 2.6 Hz, far below any dynamics requirement;
* ``position_motor_deg`` is a motor-side encoder value, not a joint angle;
* the static torque offset moves with pose, gravity, brake and support.

Design rules
------------

1. A joint-angle conversion is impossible without a real calibration.  Missing
   gear ratio, zero or direction raises ``CalibrationError``; the importer never
   falls back to 1:1 counts.
2. Nothing is tared by default.  ``raw_torque_nm`` is always preserved and any
   offset summary is reported as descriptive only.
3. Sample rate, jitter, gaps, per-axis channel skew and non-atomicity are
   reported as measured values.  A rate that is too low is a reported quality
   failure, not a silently accepted input.
4. Column names are matched through an explicit, overridable alias map because
   the export naming was not frozen in the handover record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class LegImportError(ValueError):
    """Raised when an episode cannot be interpreted without guessing."""


class CalibrationError(LegImportError):
    """Raised when a joint angle is requested without a real calibration."""


# Motor-side scale quoted in the handover record (0.0004577637 deg per count).
# It is recorded as evidence, never applied as a joint conversion.
MOTOR_SIDE_DEG_PER_COUNT = 0.0004577637

AXIS_JOINTS = {1: "hip", 4: "knee"}

# Flexion direction confirmed on 2026-10-07: hip decreases, knee increases.
FLEXION_SIGN = {"hip": -1, "knee": +1}

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    "position": ("position", "position_actual", "position_motor_deg", "pos", "6064"),
    "velocity": ("velocity", "velocity_actual", "vel", "606c"),
    "torque_nm": ("torqueclose", "torque_close", "torque_nm", "actual_torque", "3c1a"),
    "torque_pct": ("torque_6077", "torque_pct", "torque_percent", "motor_torque", "6077"),
    "statusword": ("statusword", "status_word", "status", "6041"),
    "error_code": ("error_code", "error", "603f"),
    "mode": ("mode", "6061"),
    "brake_override": ("brake_override", "brake", "381e"),
}
# Alias lists are normalised once, so an underscore spelling such as
# ``host_time_s`` matches a ``Host Time [s]`` column instead of silently failing.
_SIGNAL_ALIASES = {key: tuple(_normalise(alias) for alias in aliases)
                   for key, aliases in _SIGNAL_ALIASES.items()}

_TIME_ALIASES = tuple(_normalise(alias) for alias in
    ("host_time_s", "timestamp_s", "time_s", "timestamp", "hosttime", "t", "time"))
_SYNC_ALIASES = tuple(_normalise(alias) for alias in
    ("sync_quality", "synchronization", "sync", "data_quality", "quality"))


@dataclass(frozen=True)
class JointCalibration:
    """Motor counts to flexion angle.  Every field is required.

    Attributes
    ----------
    counts_per_degree
        Motor-side counts per one degree of joint rotation.
    zero_counts
        Encoder count that corresponds to the calibrated mechanical zero.
    flexion_sign
        ``+1`` if the count grows during flexion, ``-1`` if it falls.
    safe_min_deg, safe_max_deg
        Reviewed flexion limits for this joint.
    """

    counts_per_degree: float
    zero_counts: float
    flexion_sign: int
    safe_min_deg: float
    safe_max_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.counts_per_degree) or self.counts_per_degree <= 0:
            raise CalibrationError("counts_per_degree must be finite and positive")
        if not math.isfinite(self.zero_counts):
            raise CalibrationError("zero_counts must be finite")
        if self.flexion_sign not in (-1, 1):
            raise CalibrationError("flexion_sign must be exactly -1 or 1")
        if not (math.isfinite(self.safe_min_deg) and math.isfinite(self.safe_max_deg)):
            raise CalibrationError("safe range must be finite")
        if self.safe_min_deg >= self.safe_max_deg:
            raise CalibrationError("safe_min_deg must be below safe_max_deg")

    def to_degrees(self, counts: Any) -> np.ndarray:
        values = np.asarray(counts, dtype=float)
        return self.flexion_sign * (values - self.zero_counts) / self.counts_per_degree


@dataclass
class LegJoint:
    axis: int
    joint: str
    host_time_s: np.ndarray
    position_counts: np.ndarray | None = None
    motor_position_deg: np.ndarray | None = None
    flexion_deg: np.ndarray | None = None
    velocity: np.ndarray | None = None
    torque_nm: np.ndarray | None = None
    torque_pct: np.ndarray | None = None
    statusword: np.ndarray | None = None
    error_code: np.ndarray | None = None
    mode: np.ndarray | None = None
    brake_override: np.ndarray | None = None
    signals: dict[str, str] = field(default_factory=dict)


@dataclass
class LegEpisode:
    path: Path
    host_time_s: np.ndarray
    joints: dict[str, LegJoint]
    sync_quality: str | None
    sync_quality_values: tuple[str, ...]
    column_map: dict[str, str]
    unmapped_columns: tuple[str, ...]
    calibration: dict[str, JointCalibration] | None

    @property
    def rows(self) -> int:
        return int(len(self.host_time_s))


def _build_alias_index(columns: Iterable[str]) -> dict[str, str]:
    return {_normalise(column): column for column in columns}


def _match_signal(index: Mapping[str, str], axis: int, signal: str) -> str | None:
    """Find the column for one axis signal using the declared aliases.

    A match must bind to the intended axis, but the axis check ignores the
    characters consumed by the signal alias.  Otherwise ``axis4_torque_3c1a``
    carries a bare ``1`` (from ``3c1a``) and would wrongly bind to axis 1.
    """

    aliases = _SIGNAL_ALIASES[signal]
    candidates = []
    for normalised, original in index.items():
        best_rank = None
        for alias in aliases:
            start = 0
            while True:
                found = normalised.find(alias, start)
                if found < 0:
                    break
                rank = _axis_rank(normalised, found, found + len(alias), axis)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                start = found + 1
        if best_rank is None:
            continue
        candidates.append((best_rank, len(normalised), original))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _axis_rank(normalised: str, start: int, stop: int, axis: int) -> int | None:
    """Rank how strongly the text outside ``[start, stop)`` names ``axis``.

    Rank 0-3 is an explicit token (``axis1``, ``ax1``, ``j1``, ``drive1``);
    rank 4 is a bare digit that is not embedded in a numeric run.  ``None``
    means there is no evidence for this axis.
    """

    outside = normalised[:start] + "\x00" + normalised[stop:]
    for rank, token in enumerate((f"axis{axis}", f"ax{axis}", f"j{axis}", f"drive{axis}")):
        if token in outside:
            return rank
    if re.search(r"(?:^|[^0-9])" + f"{axis}" + r"(?=[^0-9]|$)", outside):
        return 4
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise LegImportError(f"missing episode file: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas message passthrough
        raise LegImportError(f"could not read {path.name}: {exc}") from exc
    if frame.empty:
        raise LegImportError(f"{path.name} contains no rows")
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values


def _resolve_time(frame: pd.DataFrame, column_map: Mapping[str, str] | None) -> tuple[np.ndarray, str]:
    if column_map and "time" in column_map:
        column = column_map["time"]
        if column not in frame.columns:
            raise LegImportError(f"declared time column not present: {column}")
        return _numeric(frame, column), column
    index = _build_alias_index(frame.columns)
    for alias in _TIME_ALIASES:
        if alias in index:
            column = index[alias]
            return _numeric(frame, column), column
    raise LegImportError(
        "no time column found; declare one explicitly. Missing a clock would make "
        "every derived rate meaningless, so the importer refuses to invent one."
    )


def read_leg_episode(
    path: str | Path,
    *,
    calibration: Mapping[str, JointCalibration] | None = None,
    column_map: Mapping[str, str] | None = None,
    axes: Sequence[int] = (1, 4),
) -> LegEpisode:
    """Read one dummy-leg CSV into a checked episode object.

    ``column_map`` may override any discovered column, keyed by ``time``,
    ``sync_quality`` or ``axis{axis}_{signal}`` (for example
    ``axis1_torque_nm``).
    """

    csv_path = Path(path)
    frame = _read_csv(csv_path)
    overrides = dict(column_map or {})
    host_time_s, time_column = _resolve_time(frame, overrides)
    if not np.isfinite(host_time_s).all():
        raise LegImportError("time column contains non-finite values")
    if len(host_time_s) > 1 and np.any(np.diff(host_time_s) < 0):
        raise LegImportError("time column is not monotonically non-decreasing")

    index = _build_alias_index(frame.columns)
    sync_column = overrides.get("sync_quality")
    if sync_column is None:
        for alias in _SYNC_ALIASES:
            if alias in index:
                sync_column = index[alias]
                break
    sync_values: tuple[str, ...] = ()
    sync_quality: str | None = None
    if sync_column is not None:
        if sync_column not in frame.columns:
            raise LegImportError(f"declared sync-quality column not present: {sync_column}")
        sync_values = tuple(sorted({str(value) for value in frame[sync_column].dropna().unique()}))
        sync_quality = ";".join(sync_values) if sync_values else None

    resolved: dict[str, str] = {"time": time_column}
    if sync_column is not None:
        resolved["sync_quality"] = sync_column
    joints: dict[str, LegJoint] = {}
    for axis in axes:
        joint_name = AXIS_JOINTS.get(axis, f"axis{axis}")
        joint = LegJoint(axis=axis, joint=joint_name, host_time_s=host_time_s)
        for signal in _SIGNAL_ALIASES:
            key = f"axis{axis}_{signal}"
            column = overrides.get(key)
            if column is None:
                column = _match_signal(index, axis, signal)
            else:
                if column not in frame.columns:
                    raise LegImportError(f"declared column not present: {column}")
            if column is None:
                continue
            joint.signals[signal] = column
            resolved[key] = column
            values = _numeric(frame, column)
            if signal == "position":
                joint.position_counts = values
                joint.motor_position_deg = values * MOTOR_SIDE_DEG_PER_COUNT
            elif signal == "velocity":
                joint.velocity = values
            elif signal == "torque_nm":
                joint.torque_nm = values
            elif signal == "torque_pct":
                joint.torque_pct = values
            elif signal == "statusword":
                joint.statusword = values
            elif signal == "error_code":
                joint.error_code = values
            elif signal == "mode":
                joint.mode = values
            elif signal == "brake_override":
                joint.brake_override = values
        if joint.position_counts is None and joint.torque_nm is None:
            raise LegImportError(
                f"axis {axis} resolved neither a position nor a joint-torque column; "
                "supply an explicit column_map rather than guessing names"
            )
        if calibration is not None:
            if joint_name not in calibration:
                raise CalibrationError(f"no calibration supplied for joint {joint_name}")
            joint.flexion_deg = calibration[joint_name].to_degrees(joint.position_counts)
        joints[joint_name] = joint

    mapped_columns = set(resolved.values())
    unmapped = tuple(column for column in frame.columns if column not in mapped_columns)
    return LegEpisode(
        path=csv_path,
        host_time_s=host_time_s,
        joints=joints,
        sync_quality=sync_quality,
        sync_quality_values=sync_values,
        column_map=resolved,
        unmapped_columns=unmapped,
        calibration=dict(calibration) if calibration is not None else None,
    )


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "std": None, "minimum": None, "maximum": None}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def quality_report(
    episode: LegEpisode,
    *,
    minimum_rate_hz: float = 100.0,
    expected_period_s: float | None = None,
) -> dict[str, Any]:
    """Measure what the episode actually is, and say whether it is usable.

    ``usable_for_dynamics`` is deliberately conservative: non-atomic axis reads
    or a rate below ``minimum_rate_hz`` fail it, because either one invalidates
    identification, phase analysis and high-rate control validation.
    """

    times = np.asarray(episode.host_time_s, dtype=float)
    report: dict[str, Any] = {"file": episode.path.name, "rows": episode.rows}
    if episode.rows < 2:
        report.update(
            sample_rate_hz=None,
            mean_period_s=None,
            jitter_s=None,
            duration_s=0.0,
            gap_count=None,
            usable_for_dynamics=False,
            findings=["fewer_than_two_rows"],
        )
        report["joints"] = {
            name: {"torque_nm": _distribution(joint.torque_nm) if joint.torque_nm is not None else None}
            for name, joint in episode.joints.items()
        }
        return report

    span = float(times[-1] - times[0])
    diffs = np.diff(times)
    report["duration_s"] = span
    report["sample_rate_hz"] = float((episode.rows - 1) / span) if span > 0 else None
    report["mean_period_s"] = float(diffs.mean())
    report["jitter_s"] = {
        "std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "max_abs_deviation_from_median": float(np.max(np.abs(diffs - np.median(diffs)))),
    }
    period = expected_period_s if expected_period_s is not None else float(np.median(diffs))
    report["expected_period_s"] = period
    report["gap_count"] = int(np.sum(diffs > period * 1.5)) if period > 0 else None

    findings: list[str] = []
    rate = report["sample_rate_hz"]
    if rate is None:
        findings.append("undefined_sample_rate")
    elif rate < minimum_rate_hz:
        findings.append(f"sample_rate_below_{minimum_rate_hz:g}hz:{rate:.3f}")

    non_atomic = any(
        "atomic" in value.lower() and "not" in value.lower()
        for value in episode.sync_quality_values
    )
    report["non_atomic_axis_reads"] = bool(non_atomic or not episode.sync_quality_values)
    if report["non_atomic_axis_reads"]:
        findings.append("axis_channels_not_a_single_instant")
    if not episode.sync_quality_values:
        findings.append("no_sync_quality_field")

    joint_report: dict[str, Any] = {}
    for name, joint in episode.joints.items():
        entry: dict[str, Any] = {"axis": joint.axis, "signals": dict(joint.signals)}
        if joint.torque_nm is not None:
            entry["raw_torque_nm"] = _distribution(joint.torque_nm)
            entry["torque_offset_note"] = (
                "descriptive only; never subtracted by default because the offset moves "
                "with pose, gravity, brake and support"
            )
        if joint.torque_pct is not None:
            entry["motor_side_torque_pct"] = _distribution(joint.torque_pct)
        if joint.position_counts is not None:
            entry["position_counts"] = _distribution(joint.position_counts)
            entry["motor_side_deg_range"] = [
                float(np.nanmin(joint.motor_position_deg)),
                float(np.nanmax(joint.motor_position_deg)),
            ]
            entry["joint_angle_available"] = joint.flexion_deg is not None
            if joint.flexion_deg is not None:
                entry["flexion_deg_range"] = [
                    float(np.nanmin(joint.flexion_deg)),
                    float(np.nanmax(joint.flexion_deg)),
                ]
            else:
                findings.append(f"{name}_has_no_joint_angle_calibration")
        if joint.brake_override is not None:
            unique = np.unique(joint.brake_override[np.isfinite(joint.brake_override)])
            entry["brake_override_values"] = [float(value) for value in unique]
            entry["brake_state_not_end_of_run"] = (
                "the CSV shows the override during acquisition, not the restored state"
            )
        if joint.statusword is not None:
            unique = np.unique(joint.statusword[np.isfinite(joint.statusword)])
            entry["statusword_values"] = [float(value) for value in unique]
        if joint.error_code is not None:
            nonzero = joint.error_code[np.isfinite(joint.error_code) & (joint.error_code != 0)]
            entry["nonzero_error_count"] = int(len(nonzero))
        joint_report[name] = entry
    report["joints"] = joint_report

    usable = not any(
        finding.startswith("sample_rate_below") or finding == "axis_channels_not_a_single_instant"
        for finding in findings
    )
    report["usable_for_dynamics"] = bool(usable)
    report["findings"] = findings
    report["interpretation"] = (
        "usable for identification-scale analysis"
        if usable
        else "trial/acquisition evidence only; not identification, phase or high-rate control data"
    )
    return report


def _calibration_from_json(path: str | Path) -> dict[str, JointCalibration]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise CalibrationError("calibration file must be a non-empty JSON object")
    calibration: dict[str, JointCalibration] = {}
    for joint, values in payload.items():
        if not isinstance(values, dict):
            raise CalibrationError(f"calibration for {joint} must be an object")
        missing = [
            key
            for key in ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")
            if key not in values
        ]
        if missing:
            raise CalibrationError(f"calibration for {joint} is missing: {','.join(missing)}")
        calibration[joint] = JointCalibration(
            counts_per_degree=values["counts_per_degree"],
            zero_counts=values["zero_counts"],
            flexion_sign=int(values["flexion_sign"]),
            safe_min_deg=values["safe_min_deg"],
            safe_max_deg=values["safe_max_deg"],
        )
    return calibration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path, help="dummy-leg episode CSV")
    parser.add_argument("--calibration", type=Path, default=None, help="joint calibration JSON")
    parser.add_argument("--column-map", type=Path, default=None, help="column override JSON")
    parser.add_argument("--minimum-rate-hz", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None, help="write the report to this JSON file")
    args = parser.parse_args(argv)

    calibration = _calibration_from_json(args.calibration) if args.calibration else None
    column_map = (
        json.loads(args.column_map.read_text(encoding="utf-8")) if args.column_map else None
    )
    episode = read_leg_episode(args.episode, calibration=calibration, column_map=column_map)
    report = quality_report(episode, minimum_rate_hz=args.minimum_rate_hz)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
