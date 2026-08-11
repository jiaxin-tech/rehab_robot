"""Reference-trajectory flexion/extension cycle detection.

A complete lower-limb cycle is defined from a knee-extension minimum, through
one knee-flexion maximum, to the following knee-extension minimum.  Extrema
are found from a Savitzky-Golay-smoothed ``q_knee`` signal; the original signal
is retained for continuity and smoothing-residual audits.

The default representative-cycle policy is deliberately conservative: only a
complete, continuous, smooth cycle may be selected automatically.  A trailing
low-to-high partial motion is reported as an incomplete cycle and is never the
default representative.  Explicit frame bounds and cycle-index selections are
supported and are clearly labelled as manual choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


CYCLE_COLUMNS = (
    "cycle_index",
    "start_frame",
    "peak_flexion_frame",
    "end_frame",
    "duration_frames",
    "duration_intervals",
    "q_hip_min_rad",
    "q_hip_max_rad",
    "q_hip_range_rad",
    "q_knee_min_rad",
    "q_knee_max_rad",
    "q_knee_range_rad",
    "q_hip_range_deg",
    "q_knee_range_deg",
    "cycle_complete",
    "cycle_continuous",
    "cycle_smooth",
    "cycle_quality_score",
    "knee_monotonic_fraction",
    "smoothing_residual_fraction",
    "segment_index",
    "cycle_invalid_reason",
)


@dataclass(frozen=True)
class CycleDetectionConfig:
    """Validated settings for smooth-extrema cycle detection."""

    smoothing_window_frames: int = 11
    smoothing_polynomial_order: int = 3
    minimum_extrema_distance_frames: int = 5
    minimum_cycle_duration_frames: int = 7
    minimum_peak_prominence_rad: float = float(np.deg2rad(2.0))
    minimum_knee_excursion_rad: float = float(np.deg2rad(5.0))
    maximum_smoothing_residual_fraction: float = 0.20
    minimum_monotonic_fraction: float = 0.85
    minimum_quality_score: float = 0.60
    maximum_time_gap_factor: float = 2.5
    include_incomplete_tail: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "smoothing_window_frames": self.smoothing_window_frames,
            "smoothing_polynomial_order": self.smoothing_polynomial_order,
            "minimum_extrema_distance_frames": (
                self.minimum_extrema_distance_frames
            ),
            "minimum_cycle_duration_frames": self.minimum_cycle_duration_frames,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
        if self.smoothing_window_frames < 3:
            raise ValueError("smoothing_window_frames must be at least 3.")
        if self.smoothing_window_frames % 2 == 0:
            raise ValueError("smoothing_window_frames must be odd.")
        if not 0 <= self.smoothing_polynomial_order < self.smoothing_window_frames:
            raise ValueError(
                "smoothing_polynomial_order must be non-negative and smaller "
                "than smoothing_window_frames."
            )
        if self.minimum_extrema_distance_frames < 1:
            raise ValueError("minimum_extrema_distance_frames must be positive.")
        if self.minimum_cycle_duration_frames < 3:
            raise ValueError("minimum_cycle_duration_frames must be at least 3.")
        for name in (
            "minimum_peak_prominence_rad",
            "minimum_knee_excursion_rad",
            "maximum_time_gap_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.maximum_time_gap_factor <= 1.0:
            raise ValueError("maximum_time_gap_factor must be greater than 1.")
        for name in (
            "maximum_smoothing_residual_fraction",
            "minimum_monotonic_fraction",
            "minimum_quality_score",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if not isinstance(self.include_incomplete_tail, (bool, np.bool_)):
            raise TypeError("include_incomplete_tail must be boolean.")


@dataclass(frozen=True)
class ReferenceCycleSelection:
    """One selected reference cycle and an auditable selection explanation."""

    cycle_index: int | None
    start_frame: int
    peak_flexion_frame: int | None
    end_frame: int
    duration_frames: int
    q_hip_range_rad: float
    q_knee_range_rad: float
    cycle_complete: bool
    cycle_continuous: bool
    cycle_smooth: bool
    cycle_quality_score: float
    selection_reason: str
    manual_selection: bool
    selection_mode: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_trajectory(
    full_angles: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray,
    q_hip_column: str,
    q_knee_column: str,
) -> pd.DataFrame:
    if isinstance(full_angles, pd.DataFrame):
        dataframe = full_angles.copy(deep=False)
    elif isinstance(full_angles, Mapping):
        dataframe = pd.DataFrame(full_angles)
    else:
        values = np.asarray(full_angles, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise TypeError(
                "full_angles must be a DataFrame/mapping or an (N, 2) array "
                "ordered as [q_hip, q_knee]."
            )
        dataframe = pd.DataFrame(
            {q_hip_column: values[:, 0], q_knee_column: values[:, 1]}
        )
    missing = {q_hip_column, q_knee_column}.difference(dataframe.columns)
    if missing:
        raise ValueError(f"full_angles is missing columns: {sorted(missing)}")
    if len(dataframe) < 3:
        raise ValueError("full_angles must contain at least three frames.")
    return dataframe


def _effective_savgol_parameters(
    sample_count: int,
    config: CycleDetectionConfig,
) -> tuple[int, int] | None:
    if sample_count < 3:
        return None
    window = min(config.smoothing_window_frames, sample_count)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return None
    polynomial_order = min(config.smoothing_polynomial_order, window - 1)
    return window, polynomial_order


def _smooth(values: np.ndarray, config: CycleDetectionConfig) -> np.ndarray:
    parameters = _effective_savgol_parameters(len(values), config)
    if parameters is None:
        return values.astype(float, copy=True)
    window, polynomial_order = parameters
    return np.asarray(
        savgol_filter(
            values,
            window_length=window,
            polyorder=polynomial_order,
            mode="interp",
        ),
        dtype=float,
    )


def _contiguous_segments(
    valid: np.ndarray,
    continuous_edge: np.ndarray,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for frame, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = frame
        ends_here = start is not None and (
            not is_valid
            or frame == len(valid) - 1
            or not continuous_edge[frame]
        )
        if not ends_here:
            continue
        end = frame if is_valid else frame - 1
        if end >= start:
            segments.append((start, end))
        start = None
        if is_valid and frame < len(valid) - 1 and not continuous_edge[frame]:
            start = frame + 1 if valid[frame + 1] else None
    return segments


def _prepare_signals(
    full_angles: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray,
    *,
    config: CycleDetectionConfig,
    q_hip_column: str,
    q_knee_column: str,
    time_column: str | None,
    valid_column: str | None,
) -> dict[str, object]:
    dataframe = _coerce_trajectory(full_angles, q_hip_column, q_knee_column)
    q_hip = dataframe[q_hip_column].to_numpy(dtype=float)
    q_knee = dataframe[q_knee_column].to_numpy(dtype=float)
    valid = np.isfinite(q_hip) & np.isfinite(q_knee)

    effective_valid_column = valid_column
    if effective_valid_column is None and "sample_valid" in dataframe:
        effective_valid_column = "sample_valid"
    if effective_valid_column is not None:
        if effective_valid_column not in dataframe:
            raise ValueError(
                f"valid_column {effective_valid_column!r} is not present."
            )
        validity = dataframe[effective_valid_column]
        if validity.isna().any():
            raise ValueError("valid_column must not contain missing values.")
        valid &= validity.astype(bool).to_numpy()

    continuous_edge = np.ones(len(dataframe), dtype=bool)
    effective_time_column = time_column
    if effective_time_column is None and "time_s" in dataframe:
        effective_time_column = "time_s"
    if effective_time_column is not None:
        if effective_time_column not in dataframe:
            raise ValueError(
                f"time_column {effective_time_column!r} is not present."
            )
        time_s = dataframe[effective_time_column].to_numpy(dtype=float)
        if not np.isfinite(time_s).all():
            raise ValueError("time values must all be finite.")
        differences = np.diff(time_s)
        if np.any(differences <= 0.0):
            raise ValueError("time values must be strictly increasing.")
        nominal_step = float(np.median(differences))
        continuous_edge[:-1] = (
            differences <= config.maximum_time_gap_factor * nominal_step
        )
    continuous_edge[-1] = False

    segments = _contiguous_segments(valid, continuous_edge)
    if not any(end - start + 1 >= 3 for start, end in segments):
        raise ValueError("full_angles contains no continuous finite segment of 3 frames.")
    q_hip_smoothed = np.full(len(dataframe), np.nan, dtype=float)
    q_knee_smoothed = np.full(len(dataframe), np.nan, dtype=float)
    for start, end in segments:
        positions = slice(start, end + 1)
        q_hip_smoothed[positions] = _smooth(q_hip[positions], config)
        q_knee_smoothed[positions] = _smooth(q_knee[positions], config)
    return {
        "dataframe": dataframe,
        "q_hip": q_hip,
        "q_knee": q_knee,
        "q_hip_smoothed": q_hip_smoothed,
        "q_knee_smoothed": q_knee_smoothed,
        "valid": valid,
        "continuous_edge": continuous_edge,
        "segments": segments,
        "config": config,
    }


def _endpoint_is_minimum(
    values: np.ndarray,
    *,
    at_start: bool,
    config: CycleDetectionConfig,
) -> bool:
    span = min(max(config.minimum_extrema_distance_frames, 2), len(values) - 1)
    tolerance = 0.02 * config.minimum_knee_excursion_rad
    if at_start:
        return bool(values[0] <= np.min(values[1 : span + 1]) + tolerance)
    return bool(values[-1] <= np.min(values[-span - 1 : -1]) + tolerance)


def _manual_completeness(
    q_knee_smoothed: np.ndarray,
    start: int,
    peak: int,
    end: int,
    continuous: bool,
    config: CycleDetectionConfig,
) -> bool:
    if not continuous or not start < peak < end:
        return False
    values = q_knee_smoothed[start : end + 1]
    if not np.isfinite(values).all():
        return False
    knee_range = float(np.ptp(values))
    if knee_range < config.minimum_knee_excursion_rad:
        return False
    peak_value = float(q_knee_smoothed[peak])
    endpoint_drop = min(
        peak_value - float(q_knee_smoothed[start]),
        peak_value - float(q_knee_smoothed[end]),
    )
    return bool(
        endpoint_drop >= 0.75 * knee_range
        and end - start + 1 >= config.minimum_cycle_duration_frames
    )


def _summarize_cycle(
    prepared: Mapping[str, object],
    *,
    start_frame: int,
    peak_flexion_frame: int,
    end_frame: int,
    cycle_complete: bool | None,
    segment_index: int,
    incomplete_reason: str = "",
) -> dict[str, object]:
    config = prepared["config"]
    assert isinstance(config, CycleDetectionConfig)
    q_hip = np.asarray(prepared["q_hip"], dtype=float)
    q_knee = np.asarray(prepared["q_knee"], dtype=float)
    q_hip_smoothed = np.asarray(prepared["q_hip_smoothed"], dtype=float)
    q_knee_smoothed = np.asarray(prepared["q_knee_smoothed"], dtype=float)
    valid = np.asarray(prepared["valid"], dtype=bool)
    continuous_edge = np.asarray(prepared["continuous_edge"], dtype=bool)

    if not 0 <= start_frame <= peak_flexion_frame <= end_frame < len(q_knee):
        raise ValueError("cycle frame bounds are out of order or outside the data.")
    selected = slice(start_frame, end_frame + 1)
    continuous = bool(
        valid[selected].all()
        and (
            end_frame == start_frame
            or continuous_edge[start_frame:end_frame].all()
        )
    )
    if cycle_complete is None:
        complete = _manual_completeness(
            q_knee_smoothed,
            start_frame,
            peak_flexion_frame,
            end_frame,
            continuous,
            config,
        )
    else:
        complete = bool(cycle_complete and continuous)

    hip_values = q_hip_smoothed[selected]
    knee_values = q_knee_smoothed[selected]
    q_hip_min = float(np.nanmin(hip_values))
    q_hip_max = float(np.nanmax(hip_values))
    q_knee_min = float(np.nanmin(knee_values))
    q_knee_max = float(np.nanmax(knee_values))
    q_hip_range = q_hip_max - q_hip_min
    q_knee_range = q_knee_max - q_knee_min

    raw_residual = q_knee[selected] - knee_values
    residual_rms = float(np.sqrt(np.nanmean(raw_residual**2)))
    residual_fraction = residual_rms / max(
        q_knee_range,
        config.minimum_knee_excursion_rad,
    )
    tolerance = max(0.005 * q_knee_range, np.deg2rad(0.02))
    flexion_difference = np.diff(
        q_knee_smoothed[start_frame : peak_flexion_frame + 1]
    )
    extension_difference = np.diff(
        q_knee_smoothed[peak_flexion_frame : end_frame + 1]
    )
    monotonic_components: list[float] = []
    if flexion_difference.size:
        monotonic_components.append(float(np.mean(flexion_difference >= -tolerance)))
    if extension_difference.size:
        monotonic_components.append(float(np.mean(extension_difference <= tolerance)))
    monotonic_fraction = (
        float(np.mean(monotonic_components)) if monotonic_components else 0.0
    )
    smooth = bool(
        residual_fraction <= config.maximum_smoothing_residual_fraction
        and monotonic_fraction >= config.minimum_monotonic_fraction
    )
    residual_score = float(
        np.clip(
            1.0
            - residual_fraction
            / max(config.maximum_smoothing_residual_fraction, 1e-12),
            0.0,
            1.0,
        )
    )
    amplitude_score = float(
        np.clip(q_knee_range / config.minimum_knee_excursion_rad, 0.0, 1.0)
    )
    duration_frames = end_frame - start_frame + 1
    duration_score = float(
        np.clip(
            duration_frames / config.minimum_cycle_duration_frames,
            0.0,
            1.0,
        )
    )
    quality_score = (
        0.35 * residual_score
        + 0.35 * monotonic_fraction
        + 0.20 * amplitude_score
        + 0.10 * duration_score
    )
    if not continuous:
        quality_score *= 0.25
    if not complete:
        quality_score *= 0.50

    invalid_reasons = []
    if not complete:
        invalid_reasons.append(incomplete_reason or "incomplete_cycle")
    if not continuous:
        invalid_reasons.append("discontinuous_samples")
    if not smooth:
        invalid_reasons.append("insufficient_smoothness")
    if q_knee_range < config.minimum_knee_excursion_rad:
        invalid_reasons.append("insufficient_knee_excursion")
    return {
        "start_frame": int(start_frame),
        "peak_flexion_frame": int(peak_flexion_frame),
        "end_frame": int(end_frame),
        "duration_frames": int(duration_frames),
        "duration_intervals": int(end_frame - start_frame),
        "q_hip_min_rad": q_hip_min,
        "q_hip_max_rad": q_hip_max,
        "q_hip_range_rad": q_hip_range,
        "q_knee_min_rad": q_knee_min,
        "q_knee_max_rad": q_knee_max,
        "q_knee_range_rad": q_knee_range,
        "q_hip_range_deg": float(np.rad2deg(q_hip_range)),
        "q_knee_range_deg": float(np.rad2deg(q_knee_range)),
        "cycle_complete": complete,
        "cycle_continuous": continuous,
        "cycle_smooth": smooth,
        "cycle_quality_score": float(np.clip(quality_score, 0.0, 1.0)),
        "knee_monotonic_fraction": monotonic_fraction,
        "smoothing_residual_fraction": residual_fraction,
        "segment_index": int(segment_index),
        "cycle_invalid_reason": ";".join(dict.fromkeys(invalid_reasons)),
    }


def _segment_cycles(
    prepared: Mapping[str, object],
    start: int,
    end: int,
    segment_index: int,
) -> list[dict[str, object]]:
    config = prepared["config"]
    assert isinstance(config, CycleDetectionConfig)
    smoothed = np.asarray(prepared["q_knee_smoothed"], dtype=float)
    values = smoothed[start : end + 1]
    if len(values) < 3:
        return []
    prominence = config.minimum_peak_prominence_rad
    distance = config.minimum_extrema_distance_frames
    local_peaks = find_peaks(values, prominence=prominence, distance=distance)[0]
    local_valleys = find_peaks(-values, prominence=prominence, distance=distance)[0]
    valleys = list(int(value) for value in local_valleys)
    if _endpoint_is_minimum(values, at_start=True, config=config):
        valleys.append(0)
    if _endpoint_is_minimum(values, at_start=False, config=config):
        valleys.append(len(values) - 1)
    valleys = sorted(set(valleys))

    cycles: list[dict[str, object]] = []
    for local_start, local_end in zip(valleys[:-1], valleys[1:]):
        peaks_between = local_peaks[
            (local_peaks > local_start) & (local_peaks < local_end)
        ]
        if not len(peaks_between):
            continue
        local_peak = int(peaks_between[np.argmax(values[peaks_between])])
        excursion = float(
            values[local_peak]
            - min(values[local_start], values[local_end])
        )
        if excursion < config.minimum_knee_excursion_rad:
            continue
        cycles.append(
            _summarize_cycle(
                prepared,
                start_frame=start + local_start,
                peak_flexion_frame=start + local_peak,
                end_frame=start + local_end,
                cycle_complete=True,
                segment_index=segment_index,
            )
        )

    if not config.include_incomplete_tail:
        return cycles
    tail_start_local: int | None = valleys[-1] if valleys else None
    if tail_start_local is None and _endpoint_is_minimum(
        values, at_start=True, config=config
    ):
        tail_start_local = 0
    if tail_start_local is None or tail_start_local >= len(values) - 1:
        return cycles
    tail_values = values[tail_start_local:]
    tail_peak_offset = int(np.argmax(tail_values))
    tail_peak_local = tail_start_local + tail_peak_offset
    tail_excursion = float(tail_values[tail_peak_offset] - tail_values[0])
    already_complete_end = any(
        int(cycle["end_frame"]) == start + len(values) - 1 for cycle in cycles
    )
    if (
        not already_complete_end
        and tail_peak_local > tail_start_local
        and tail_excursion >= config.minimum_knee_excursion_rad
    ):
        reason = (
            "incomplete_end_of_recording"
            if end == len(smoothed) - 1
            else "incomplete_before_gap"
        )
        cycles.append(
            _summarize_cycle(
                prepared,
                start_frame=start + tail_start_local,
                peak_flexion_frame=start + tail_peak_local,
                end_frame=end,
                cycle_complete=False,
                segment_index=segment_index,
                incomplete_reason=reason,
            )
        )
    return cycles


def detect_flexion_extension_cycles(
    full_angles: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray,
    *,
    config: CycleDetectionConfig | None = None,
    q_hip_column: str = "q_hip_rad",
    q_knee_column: str = "q_knee_rad",
    time_column: str | None = None,
    valid_column: str | None = None,
) -> pd.DataFrame:
    """Detect complete ``minimum -> maximum -> minimum`` knee cycles.

    Frame fields use zero-based row positions and ``end_frame`` is inclusive.
    ``duration_frames`` is therefore ``end_frame - start_frame + 1``.  Invalid
    samples and long timestamp gaps split the signal; cycles never cross them.
    A qualifying trailing partial movement is returned with
    ``cycle_complete=False``.
    """

    settings = config if config is not None else CycleDetectionConfig()
    if not isinstance(settings, CycleDetectionConfig):
        raise TypeError("config must be a CycleDetectionConfig instance.")
    prepared = _prepare_signals(
        full_angles,
        config=settings,
        q_hip_column=q_hip_column,
        q_knee_column=q_knee_column,
        time_column=time_column,
        valid_column=valid_column,
    )
    rows: list[dict[str, object]] = []
    for segment_index, (start, end) in enumerate(prepared["segments"]):
        rows.extend(_segment_cycles(prepared, start, end, segment_index))
    rows.sort(key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))
    for cycle_index, row in enumerate(rows):
        row["cycle_index"] = cycle_index
    cycles = pd.DataFrame(rows, columns=CYCLE_COLUMNS)
    if not cycles.empty:
        integer_columns = (
            "cycle_index",
            "start_frame",
            "peak_flexion_frame",
            "end_frame",
            "duration_frames",
            "duration_intervals",
            "segment_index",
        )
        cycles.loc[:, integer_columns] = cycles.loc[:, integer_columns].astype(int)
        for column in ("cycle_complete", "cycle_continuous", "cycle_smooth"):
            cycles[column] = cycles[column].astype(bool)

    # These attrs let an immediate explicit-frame selection audit a range that
    # is not identical to an automatically detected cycle.  They are never
    # needed for automatic selection and are intentionally not serialized.
    cycles.attrs["_prepared_signals"] = prepared
    cycles.attrs["detection_config"] = asdict(settings)
    cycles.attrs["frame_convention"] = "zero_based_inclusive_end"
    cycles.attrs["representative_policy"] = (
        "complete_and_continuous_and_smooth_then_highest_quality"
    )
    return cycles


def _selection_from_row(
    row: pd.Series | Mapping[str, object],
    *,
    selection_reason: str,
    manual_selection: bool,
    selection_mode: str,
) -> ReferenceCycleSelection:
    values = dict(row)
    return ReferenceCycleSelection(
        cycle_index=int(values["cycle_index"]),
        start_frame=int(values["start_frame"]),
        peak_flexion_frame=int(values["peak_flexion_frame"]),
        end_frame=int(values["end_frame"]),
        duration_frames=int(values["duration_frames"]),
        q_hip_range_rad=float(values["q_hip_range_rad"]),
        q_knee_range_rad=float(values["q_knee_range_rad"]),
        cycle_complete=bool(values["cycle_complete"]),
        cycle_continuous=bool(values["cycle_continuous"]),
        cycle_smooth=bool(values["cycle_smooth"]),
        cycle_quality_score=float(values["cycle_quality_score"]),
        selection_reason=selection_reason,
        manual_selection=manual_selection,
        selection_mode=selection_mode,
    )


def _validate_explicit_frame(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer frame position.")
    return int(value)


def select_representative_cycle(
    cycles: pd.DataFrame,
    *,
    cycle_index: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    minimum_quality_score: float | None = None,
) -> ReferenceCycleSelection:
    """Select one detected cycle or an explicitly bounded manual range.

    Explicit ``cycle_index`` and ``start_frame/end_frame`` override the default
    policy and set ``manual_selection=True``.  A manual frame range need not
    match an automatic cycle; when detector attrs are available its
    completeness and quality are recomputed, otherwise it is conservatively
    marked incomplete.  The default never selects an incomplete tail.
    """

    if not isinstance(cycles, pd.DataFrame):
        raise TypeError("cycles must be the DataFrame returned by cycle detection.")
    missing = set(CYCLE_COLUMNS).difference(cycles.columns)
    if missing:
        raise ValueError(f"cycles is missing columns: {sorted(missing)}")
    explicit_bounds = start_frame is not None or end_frame is not None
    if cycle_index is not None and explicit_bounds:
        raise ValueError("choose either cycle_index or start_frame/end_frame, not both.")
    if explicit_bounds and (start_frame is None or end_frame is None):
        raise ValueError("start_frame and end_frame must be provided together.")

    if cycle_index is not None:
        selected_index = _validate_explicit_frame(cycle_index, "cycle_index")
        selected = cycles.loc[cycles["cycle_index"].eq(selected_index)]
        if len(selected) != 1:
            raise ValueError(f"cycle_index {selected_index} was not detected.")
        row = selected.iloc[0]
        status = "complete" if bool(row["cycle_complete"]) else "incomplete"
        return _selection_from_row(
            row,
            selection_reason=(
                f"explicit cycle_index={selected_index} requested; detected "
                f"cycle audit status is {status}"
            ),
            manual_selection=True,
            selection_mode="explicit_cycle_index",
        )

    if explicit_bounds:
        selected_start = _validate_explicit_frame(start_frame, "start_frame")
        selected_end = _validate_explicit_frame(end_frame, "end_frame")
        if selected_start < 0 or selected_end <= selected_start:
            raise ValueError(
                "manual frame range requires 0 <= start_frame < end_frame."
            )
        matching = cycles.loc[
            cycles["start_frame"].eq(selected_start)
            & cycles["end_frame"].eq(selected_end)
        ]
        if len(matching) == 1:
            row = matching.iloc[0]
            selection = _selection_from_row(
                row,
                selection_reason=(
                    "manual_frame_range matched an automatically detected cycle; "
                    "automatic completeness audit retained"
                ),
                manual_selection=True,
                selection_mode="manual_frame_range",
            )
            return selection

        prepared = cycles.attrs.get("_prepared_signals")
        if isinstance(prepared, Mapping):
            sample_count = len(np.asarray(prepared["q_knee"], dtype=float))
            if selected_end >= sample_count:
                raise ValueError(
                    f"end_frame {selected_end} is outside {sample_count} frames."
                )
            smoothed = np.asarray(prepared["q_knee_smoothed"], dtype=float)
            interval = smoothed[selected_start : selected_end + 1]
            if not np.isfinite(interval).any():
                raise ValueError("manual frame range contains no finite knee angle.")
            peak = selected_start + int(np.nanargmax(interval))
            summary = _summarize_cycle(
                prepared,
                start_frame=selected_start,
                peak_flexion_frame=peak,
                end_frame=selected_end,
                cycle_complete=None,
                segment_index=-1,
                incomplete_reason="manual_frame_range_incomplete",
            )
            summary["cycle_index"] = -1
            selection = _selection_from_row(
                summary,
                selection_reason=(
                    "manual_frame_range did not match an automatic cycle; "
                    "completeness was re-audited from continuity, endpoint, peak, "
                    "duration, and knee-excursion checks"
                ),
                manual_selection=True,
                selection_mode="manual_frame_range",
            )
            return ReferenceCycleSelection(
                **{
                    **selection.as_dict(),
                    "cycle_index": None,
                }
            )
        return ReferenceCycleSelection(
            cycle_index=None,
            start_frame=selected_start,
            peak_flexion_frame=None,
            end_frame=selected_end,
            duration_frames=selected_end - selected_start + 1,
            q_hip_range_rad=np.nan,
            q_knee_range_rad=np.nan,
            cycle_complete=False,
            cycle_continuous=False,
            cycle_smooth=False,
            cycle_quality_score=0.0,
            selection_reason=(
                "manual_frame_range accepted without detector source attrs; "
                "completeness cannot be established and is conservatively false"
            ),
            manual_selection=True,
            selection_mode="manual_frame_range",
        )

    if cycles.empty:
        raise ValueError("no flexion/extension cycle was detected.")
    if minimum_quality_score is None:
        config_values = cycles.attrs.get("detection_config", {})
        threshold = float(config_values.get("minimum_quality_score", 0.60))
    else:
        threshold = float(minimum_quality_score)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_quality_score must be finite and in [0, 1].")
    eligible = cycles.loc[
        cycles["cycle_complete"].astype(bool)
        & cycles["cycle_continuous"].astype(bool)
        & cycles["cycle_smooth"].astype(bool)
        & cycles["cycle_quality_score"].ge(threshold)
    ].copy()
    if eligible.empty:
        raise ValueError(
            "no complete, continuous, smooth cycle satisfies the default "
            f"quality threshold {threshold:.3f}; use an explicit audited "
            "cycle_index or frame range if manual review justifies it."
        )
    eligible = eligible.sort_values(
        ["cycle_quality_score", "q_knee_range_rad", "cycle_index"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return _selection_from_row(
        eligible.iloc[0],
        selection_reason=(
            "automatically selected the highest-quality cycle among complete, "
            f"continuous, smooth candidates with quality >= {threshold:.3f}; "
            "stable ties prefer larger knee range then smaller cycle_index"
        ),
        manual_selection=False,
        selection_mode="automatic_representative",
    )


def select_reference_trajectory_cycle(
    full_angles: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray,
    *,
    cycle_index: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    config: CycleDetectionConfig | None = None,
    q_hip_column: str = "q_hip_rad",
    q_knee_column: str = "q_knee_rad",
    time_column: str | None = None,
    valid_column: str | None = None,
) -> ReferenceCycleSelection:
    """Detect cycles and select one using the same explicit/default API."""

    cycles = detect_flexion_extension_cycles(
        full_angles,
        config=config,
        q_hip_column=q_hip_column,
        q_knee_column=q_knee_column,
        time_column=time_column,
        valid_column=valid_column,
    )
    return select_representative_cycle(
        cycles,
        cycle_index=cycle_index,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def extract_selected_cycle(
    full_angles: pd.DataFrame,
    selection: ReferenceCycleSelection,
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Extract the inclusive selected frame range from the original table."""

    if not isinstance(full_angles, pd.DataFrame):
        raise TypeError("full_angles must be a pandas DataFrame for extraction.")
    if not isinstance(selection, ReferenceCycleSelection):
        raise TypeError("selection must be a ReferenceCycleSelection.")
    if selection.end_frame >= len(full_angles):
        raise ValueError("selection end_frame is outside full_angles.")
    selected = full_angles.iloc[
        selection.start_frame : selection.end_frame + 1
    ].copy()
    selected.insert(0, "source_frame", np.arange(
        selection.start_frame,
        selection.end_frame + 1,
        dtype=int,
    ))
    if reset_index:
        selected.reset_index(drop=True, inplace=True)
    return selected


# Clear convenience aliases for downstream callers.
detect_trajectory_cycles = detect_flexion_extension_cycles
choose_reference_cycle = select_reference_trajectory_cycle


__all__ = [
    "CYCLE_COLUMNS",
    "CycleDetectionConfig",
    "ReferenceCycleSelection",
    "choose_reference_cycle",
    "detect_flexion_extension_cycles",
    "detect_trajectory_cycles",
    "extract_selected_cycle",
    "select_reference_trajectory_cycle",
    "select_representative_cycle",
]
