"""Measured-angle velocity and acceleration estimation for stage 4.5D.

The functions in this module deliberately estimate derivatives from angle and
timestamp samples only.  Simulated ground-truth velocity or acceleration is
neither an argument nor an input dependency.  Offline algorithms are labelled
as using future samples; causal algorithms process each contiguous trajectory
segment using current and historical samples only.

All four public algorithms return the same schema through
:class:`DerivativeEstimationResult`::

    dq_hip_est_rad_s, dq_knee_est_rad_s,
    ddq_hip_est_rad_s2, ddq_knee_est_rad_s2,
    derivative_valid, derivative_reason,
    filter_delay_s, uses_future_samples

The input may be a ``pandas.DataFrame`` or three arrays ``time, q_hip,
q_knee``.  DataFrame callers may explicitly select measured/reconstructed
angle columns, but derivative-like or ``*_true_*`` angle column names are
rejected.  Extra ground-truth derivative columns in a larger audit table are
ignored and cannot affect the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


DERIVATIVE_METHODS = (
    "central_difference_offline",
    "savitzky_golay_offline",
    "causal_backward_difference",
    "causal_filter_and_difference",
)

DERIVATIVE_OUTPUT_COLUMNS = (
    "dq_hip_est_rad_s",
    "dq_knee_est_rad_s",
    "ddq_hip_est_rad_s2",
    "ddq_knee_est_rad_s2",
    "derivative_valid",
    "derivative_reason",
    "filter_delay_s",
    "uses_future_samples",
)

DEFAULT_GROUP_COLUMN_CANDIDATES = (
    "subject_id",
    "trajectory_id",
    "trajectory_family",
    "generalization_family",
    "speed_profile",
    "dataset_split",
    "phase",
)

ANGLE_COLUMN_CANDIDATES = (
    ("q_hip_est_rad", "q_knee_est_rad"),
    ("q_hip_measured_rad", "q_knee_measured_rad"),
    ("q_hip_observed_rad", "q_knee_observed_rad"),
    ("q_hip_rad", "q_knee_rad"),
)

DEFAULT_VALID_COLUMN_CANDIDATES = (
    "angle_valid",
    "ik_valid",
    "joint_continuity_valid",
    "sample_valid",
)

_TIME_EPSILON_S = 1e-12


@dataclass(frozen=True)
class DerivativeEstimationConfig:
    """Fixed derivative settings.

    ``maximum_time_gap_s=None`` chooses a per-group limit equal to
    ``maximum_gap_factor`` times the median positive sampling interval.  This
    makes the default work at different sampling frequencies while still
    splitting dropped blocks.  Set an explicit positive value for a protocol
    with a fixed maximum gap.
    """

    savgol_window_length: int = 21
    savgol_polynomial_order: int = 3
    causal_filter_window_length: int = 5
    maximum_time_gap_s: float | None = None
    maximum_gap_factor: float = 2.5
    uniform_time_relative_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.savgol_window_length < 5:
            raise ValueError("savgol_window_length must be at least 5.")
        if self.savgol_window_length % 2 == 0:
            raise ValueError("savgol_window_length must be odd.")
        if not 2 <= self.savgol_polynomial_order < self.savgol_window_length:
            raise ValueError(
                "savgol_polynomial_order must be at least 2 and smaller "
                "than savgol_window_length."
            )
        if self.causal_filter_window_length < 3:
            raise ValueError(
                "causal_filter_window_length must be at least 3."
            )
        if (
            self.maximum_time_gap_s is not None
            and (
                not np.isfinite(self.maximum_time_gap_s)
                or self.maximum_time_gap_s <= 0.0
            )
        ):
            raise ValueError("maximum_time_gap_s must be finite and positive.")
        if (
            not np.isfinite(self.maximum_gap_factor)
            or self.maximum_gap_factor <= 1.0
        ):
            raise ValueError("maximum_gap_factor must be finite and > 1.")
        if (
            not np.isfinite(self.uniform_time_relative_tolerance)
            or self.uniform_time_relative_tolerance < 0.0
        ):
            raise ValueError(
                "uniform_time_relative_tolerance must be finite and >= 0."
            )


@dataclass(frozen=True)
class DerivativeEstimationResult:
    """Estimated derivatives and auditable algorithm metadata."""

    dataframe: pd.DataFrame
    metadata: dict[str, object]

    def __getitem__(self, key: str) -> pd.Series:
        """Convenience proxy for callers that only need an output column."""

        return self.dataframe[key]

    def __len__(self) -> int:
        return len(self.dataframe)


@dataclass(frozen=True)
class _PreparedInput:
    dataframe: pd.DataFrame
    time_column: str
    hip_angle_column: str
    knee_angle_column: str
    valid_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    array_input: bool


def _normalise_config(
    config: DerivativeEstimationConfig | None,
    *,
    maximum_time_gap_s: float | None,
) -> DerivativeEstimationConfig:
    resolved = config or DerivativeEstimationConfig()
    if maximum_time_gap_s is not None:
        resolved = replace(resolved, maximum_time_gap_s=maximum_time_gap_s)
    return resolved


def _validate_angle_column_name(name: str) -> None:
    lowered = str(name).lower()
    forbidden = (
        "true",
        "ground_truth",
        "dq_",
        "ddq_",
        "velocity",
        "acceleration",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError(
            f"Angle source column {name!r} is not an allowed measured-angle "
            "column; true or derivative columns are prohibited."
        )


def _resolve_angle_columns(
    dataframe: pd.DataFrame,
    angle_columns: tuple[str, str] | None,
) -> tuple[str, str]:
    if angle_columns is not None:
        if len(angle_columns) != 2:
            raise ValueError("angle_columns must contain hip and knee columns.")
        hip, knee = (str(angle_columns[0]), str(angle_columns[1]))
        _validate_angle_column_name(hip)
        _validate_angle_column_name(knee)
        missing = {hip, knee}.difference(dataframe.columns)
        if missing:
            raise ValueError(
                "Missing selected angle columns: " + ", ".join(sorted(missing))
            )
        return hip, knee

    for hip, knee in ANGLE_COLUMN_CANDIDATES:
        if hip in dataframe.columns and knee in dataframe.columns:
            return hip, knee
    candidates = " or ".join(f"({h}, {k})" for h, k in ANGLE_COLUMN_CANDIDATES)
    raise ValueError(f"No measured-angle pair found; expected {candidates}.")


def _resolve_group_columns(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if group_columns is None:
        return tuple(
            column
            for column in DEFAULT_GROUP_COLUMN_CANDIDATES
            if column in dataframe.columns
        )
    resolved = tuple(str(column) for column in group_columns)
    missing = set(resolved).difference(dataframe.columns)
    if missing:
        raise ValueError(
            "Missing grouping columns: " + ", ".join(sorted(missing))
        )
    return resolved


def _resolve_valid_columns(
    dataframe: pd.DataFrame,
    valid_column: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if valid_column is None:
        return tuple(
            column
            for column in DEFAULT_VALID_COLUMN_CANDIDATES
            if column in dataframe.columns
        )
    if isinstance(valid_column, str):
        resolved = (valid_column,)
    else:
        resolved = tuple(str(column) for column in valid_column)
    missing = set(resolved).difference(dataframe.columns)
    if missing:
        raise ValueError(
            "Missing validity columns: " + ", ".join(sorted(missing))
        )
    return resolved


def _prepare_input(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None,
    q_knee_rad: Sequence[float] | np.ndarray | None,
    *,
    valid_mask: Sequence[bool] | np.ndarray | None,
    time_column: str,
    angle_columns: tuple[str, str] | None,
    valid_column: str | Sequence[str] | None,
    group_columns: Sequence[str] | None,
) -> _PreparedInput:
    if isinstance(data_or_time, pd.DataFrame):
        if q_hip_rad is not None or q_knee_rad is not None or valid_mask is not None:
            raise ValueError(
                "q arrays and valid_mask must not accompany a DataFrame input."
            )
        dataframe = data_or_time.copy(deep=True).reset_index(drop=True)
        if time_column not in dataframe.columns:
            raise ValueError(f"Missing time column {time_column!r}.")
        hip, knee = _resolve_angle_columns(dataframe, angle_columns)
        return _PreparedInput(
            dataframe=dataframe,
            time_column=time_column,
            hip_angle_column=hip,
            knee_angle_column=knee,
            valid_columns=_resolve_valid_columns(dataframe, valid_column),
            group_columns=_resolve_group_columns(dataframe, group_columns),
            array_input=False,
        )

    if q_hip_rad is None or q_knee_rad is None:
        raise ValueError("Array input requires time, q_hip_rad, and q_knee_rad.")
    time = np.asarray(data_or_time, dtype=float)
    hip = np.asarray(q_hip_rad, dtype=float)
    knee = np.asarray(q_knee_rad, dtype=float)
    if time.ndim != 1 or hip.ndim != 1 or knee.ndim != 1:
        raise ValueError("time and angle arrays must be one-dimensional.")
    if not (len(time) == len(hip) == len(knee)):
        raise ValueError("time and angle arrays must have equal lengths.")
    if valid_mask is None:
        valid = np.ones(len(time), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or len(valid) != len(time):
            raise ValueError("valid_mask must be one-dimensional and aligned.")
    dataframe = pd.DataFrame(
        {
            "time_s": time,
            "q_hip_est_rad": hip,
            "q_knee_est_rad": knee,
            "angle_valid": valid,
        }
    )
    return _PreparedInput(
        dataframe=dataframe,
        time_column="time_s",
        hip_angle_column="q_hip_est_rad",
        knee_angle_column="q_knee_est_rad",
        valid_columns=("angle_valid",),
        group_columns=(),
        array_input=True,
    )


def _append_reason(reasons: np.ndarray, positions: Iterable[int], text: str) -> None:
    positions_array = np.asarray(list(positions), dtype=int)
    if not len(positions_array):
        return
    current = reasons[positions_array].astype(str)
    reasons[positions_array] = np.where(
        current == "",
        text,
        np.char.add(np.char.add(current, ";"), text),
    )


def _group_positions(
    dataframe: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> list[np.ndarray]:
    if not group_columns:
        return [np.arange(len(dataframe), dtype=int)]
    # Use an explicit position column so arbitrary/duplicate input indices can
    # never alter assignment. ``dropna=False`` keeps auditable missing labels.
    grouping = dataframe.loc[:, list(group_columns)].copy()
    grouping["__row_position"] = np.arange(len(dataframe), dtype=int)
    return [
        group["__row_position"].to_numpy(dtype=int)
        for _, group in grouping.groupby(
            list(group_columns), sort=False, dropna=False
        )
    ]


def _effective_maximum_gap(
    group_time: np.ndarray,
    config: DerivativeEstimationConfig,
) -> float:
    if config.maximum_time_gap_s is not None:
        return float(config.maximum_time_gap_s)
    positive_differences = np.diff(group_time)
    positive_differences = positive_differences[
        np.isfinite(positive_differences) & (positive_differences > 0.0)
    ]
    if not len(positive_differences):
        return np.inf
    # A minority of large gaps should not inflate the nominal sample period.
    median_dt = float(np.median(positive_differences))
    return config.maximum_gap_factor * median_dt


def _contiguous_valid_runs(
    positions: np.ndarray,
    time: np.ndarray,
    base_valid: np.ndarray,
    maximum_gap_s: float,
) -> tuple[list[np.ndarray], set[int]]:
    runs: list[np.ndarray] = []
    current: list[int] = []
    starts_after_long_gap: set[int] = set()
    previous_position: int | None = None
    for position in positions:
        p = int(position)
        if not base_valid[p]:
            if current:
                runs.append(np.asarray(current, dtype=int))
                current = []
            previous_position = None
            continue
        if previous_position is not None:
            dt = float(time[p] - time[previous_position])
            if dt <= _TIME_EPSILON_S:
                raise ValueError(
                    "Timestamps must be strictly increasing within each "
                    "trajectory/phase group."
                )
            if dt > maximum_gap_s + _TIME_EPSILON_S:
                if current:
                    runs.append(np.asarray(current, dtype=int))
                current = []
                starts_after_long_gap.add(p)
        current.append(p)
        previous_position = p
    if current:
        runs.append(np.asarray(current, dtype=int))
    return runs, starts_after_long_gap


def _adaptive_savgol_window(
    segment_length: int,
    config: DerivativeEstimationConfig,
) -> int | None:
    window = min(config.savgol_window_length, segment_length)
    if window % 2 == 0:
        window -= 1
    minimum = max(5, config.savgol_polynomial_order + 1)
    if minimum % 2 == 0:
        minimum += 1
    return window if window >= minimum else None


def _central_difference(
    segment_time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.gradient(values, segment_time, edge_order=2)
    second = np.gradient(first, segment_time, edge_order=2)
    return np.asarray(first, dtype=float), np.asarray(second, dtype=float)


def _savgol_derivatives(
    segment_time: np.ndarray,
    values: np.ndarray,
    config: DerivativeEstimationConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    window = _adaptive_savgol_window(len(values), config)
    if window is None:
        return None
    differences = np.diff(segment_time)
    dt = float(np.median(differences))
    tolerance = max(
        _TIME_EPSILON_S,
        config.uniform_time_relative_tolerance * abs(dt),
    )
    if not np.all(np.abs(differences - dt) <= tolerance):
        return None
    first = savgol_filter(
        values,
        window_length=window,
        polyorder=config.savgol_polynomial_order,
        deriv=1,
        delta=dt,
        mode="interp",
    )
    second = savgol_filter(
        values,
        window_length=window,
        polyorder=config.savgol_polynomial_order,
        deriv=2,
        delta=dt,
        mode="interp",
    )
    return np.asarray(first, dtype=float), np.asarray(second, dtype=float)


def _backward_difference(
    segment_time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    first = np.full(count, np.nan)
    second = np.full(count, np.nan)
    if count >= 2:
        dt = np.diff(segment_time)
        first[1:] = np.diff(values) / dt
    if count >= 3:
        velocity_center_dt = 0.5 * (
            np.diff(segment_time)[1:] + np.diff(segment_time)[:-1]
        )
        second[2:] = np.diff(first[1:]) / velocity_center_dt
    return first, second


def _causal_filtered_difference(
    segment_time: np.ndarray,
    values: np.ndarray,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing moving-average filter followed by backward differences."""

    count = len(values)
    filtered = np.full(count, np.nan)
    if count >= window_length:
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
        filtered[window_length - 1 :] = (
            cumulative[window_length:] - cumulative[:-window_length]
        ) / float(window_length)
    first = np.full(count, np.nan)
    second = np.full(count, np.nan)
    first_start = window_length
    if count > first_start:
        dt = segment_time[first_start:] - segment_time[first_start - 1 : -1]
        first[first_start:] = (
            filtered[first_start:] - filtered[first_start - 1 : -1]
        ) / dt
    second_start = window_length + 1
    if count > second_start:
        center_dt = 0.5 * (
            segment_time[second_start:] - segment_time[second_start - 1 : -1]
            + segment_time[second_start - 1 : -1]
            - segment_time[second_start - 2 : -2]
        )
        second[second_start:] = (
            first[second_start:] - first[second_start - 1 : -1]
        ) / center_dt
    return first, second


def estimate_joint_derivatives(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None = None,
    q_knee_rad: Sequence[float] | np.ndarray | None = None,
    *,
    method: str = "savitzky_golay_offline",
    config: DerivativeEstimationConfig | None = None,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    time_column: str = "time_s",
    angle_columns: tuple[str, str] | None = None,
    valid_column: str | Sequence[str] | None = None,
    group_columns: Sequence[str] | None = None,
    maximum_time_gap_s: float | None = None,
) -> DerivativeEstimationResult:
    """Estimate hip/knee derivatives from angle samples only.

    Parameters
    ----------
    data_or_time:
        A DataFrame containing timestamps and measured/reconstructed angles, or
        a one-dimensional timestamp array.  For array input, pass hip and knee
        angle arrays as the next two arguments.
    method:
        One of :data:`DERIVATIVE_METHODS`.
    angle_columns:
        Optional explicit measured-angle pair.  Names containing ``true`` or
        derivative terminology are rejected.  No true velocity/acceleration
        argument exists.
    group_columns:
        Boundaries are never crossed.  By default all available trajectory,
        speed, split, and phase identity columns are used.
    """

    if method not in DERIVATIVE_METHODS:
        raise ValueError(
            f"Unknown derivative method {method!r}; choose one of "
            f"{', '.join(DERIVATIVE_METHODS)}."
        )
    resolved_config = _normalise_config(
        config,
        maximum_time_gap_s=maximum_time_gap_s,
    )
    prepared = _prepare_input(
        data_or_time,
        q_hip_rad,
        q_knee_rad,
        valid_mask=valid_mask,
        time_column=time_column,
        angle_columns=angle_columns,
        valid_column=valid_column,
        group_columns=group_columns,
    )
    dataframe = prepared.dataframe
    count = len(dataframe)
    time = dataframe[prepared.time_column].to_numpy(dtype=float)
    hip = dataframe[prepared.hip_angle_column].to_numpy(dtype=float)
    knee = dataframe[prepared.knee_angle_column].to_numpy(dtype=float)

    base_valid = np.isfinite(time) & np.isfinite(hip) & np.isfinite(knee)
    validity_rejected = np.zeros(count, dtype=bool)
    for column in prepared.valid_columns:
        column_valid = dataframe[column].fillna(False).astype(bool).to_numpy()
        validity_rejected |= ~column_valid
        base_valid &= column_valid

    dq_hip = np.full(count, np.nan)
    dq_knee = np.full(count, np.nan)
    ddq_hip = np.full(count, np.nan)
    ddq_knee = np.full(count, np.nan)
    derivative_valid = np.zeros(count, dtype=bool)
    reasons = np.full(count, "", dtype=object)
    offline = method.endswith("_offline")
    uses_future = np.full(count, offline, dtype=bool)
    filter_delay = np.full(count, np.nan)

    _append_reason(reasons, np.flatnonzero(validity_rejected), "source_angle_invalid")
    _append_reason(
        reasons,
        np.flatnonzero(~np.isfinite(time)),
        "nonfinite_timestamp",
    )
    _append_reason(
        reasons,
        np.flatnonzero(~np.isfinite(hip) | ~np.isfinite(knee)),
        "nonfinite_angle",
    )

    long_gap_starts: set[int] = set()
    segment_count = 0
    group_maximum_gaps: list[float] = []
    for positions in _group_positions(dataframe, prepared.group_columns):
        group_time = time[positions]
        finite_group_time = group_time[np.isfinite(group_time)]
        maximum_gap = _effective_maximum_gap(
            finite_group_time,
            resolved_config,
        )
        group_maximum_gaps.append(float(maximum_gap))
        runs, group_gap_starts = _contiguous_valid_runs(
            positions,
            time,
            base_valid,
            maximum_gap,
        )
        long_gap_starts.update(group_gap_starts)
        for run in runs:
            segment_count += 1
            segment_time = time[run]
            if len(run) > 1 and np.any(np.diff(segment_time) <= 0.0):
                raise ValueError(
                    "Timestamps must be strictly increasing within each "
                    "contiguous segment."
                )
            median_dt = (
                float(np.median(np.diff(segment_time)))
                if len(run) > 1
                else np.nan
            )

            if method == "central_difference_offline":
                filter_delay[run] = 0.0
                if len(run) < 3:
                    _append_reason(
                        reasons, run, "insufficient_segment_samples"
                    )
                    continue
                hip_first, hip_second = _central_difference(
                    segment_time, hip[run]
                )
                knee_first, knee_second = _central_difference(
                    segment_time, knee[run]
                )
            elif method == "savitzky_golay_offline":
                filter_delay[run] = 0.0
                hip_derivatives = _savgol_derivatives(
                    segment_time, hip[run], resolved_config
                )
                knee_derivatives = _savgol_derivatives(
                    segment_time, knee[run], resolved_config
                )
                if hip_derivatives is None or knee_derivatives is None:
                    reason = (
                        "insufficient_savgol_segment_samples"
                        if _adaptive_savgol_window(len(run), resolved_config)
                        is None
                        else "nonuniform_time_for_savgol"
                    )
                    _append_reason(reasons, run, reason)
                    continue
                hip_first, hip_second = hip_derivatives
                knee_first, knee_second = knee_derivatives
            elif method == "causal_backward_difference":
                filter_delay[run] = median_dt
                hip_first, hip_second = _backward_difference(
                    segment_time, hip[run]
                )
                knee_first, knee_second = _backward_difference(
                    segment_time, knee[run]
                )
            else:
                filter_delay[run] = (
                    0.5
                    * (resolved_config.causal_filter_window_length - 1)
                    * median_dt
                )
                hip_first, hip_second = _causal_filtered_difference(
                    segment_time,
                    hip[run],
                    resolved_config.causal_filter_window_length,
                )
                knee_first, knee_second = _causal_filtered_difference(
                    segment_time,
                    knee[run],
                    resolved_config.causal_filter_window_length,
                )

            dq_hip[run] = hip_first
            dq_knee[run] = knee_first
            ddq_hip[run] = hip_second
            ddq_knee[run] = knee_second
            run_valid = (
                np.isfinite(hip_first)
                & np.isfinite(knee_first)
                & np.isfinite(hip_second)
                & np.isfinite(knee_second)
            )
            derivative_valid[run] = run_valid
            if not offline:
                _append_reason(
                    reasons,
                    run[~run_valid],
                    "insufficient_causal_history",
                )

    for start in sorted(long_gap_starts):
        # The first samples after a gap are independently initialised.  Add an
        # explicit audit reason where the derivative is not yet available.
        if not derivative_valid[start]:
            _append_reason(reasons, [start], "long_time_gap_boundary")

    finite_derivatives = (
        np.isfinite(dq_hip)
        & np.isfinite(dq_knee)
        & np.isfinite(ddq_hip)
        & np.isfinite(ddq_knee)
    )
    derivative_valid &= base_valid & finite_derivatives
    unexpected_nonfinite = base_valid & ~finite_derivatives & (reasons == "")
    _append_reason(
        reasons,
        np.flatnonzero(unexpected_nonfinite),
        "derivative_unavailable",
    )
    reasons[derivative_valid] = ""

    dataframe["dq_hip_est_rad_s"] = dq_hip
    dataframe["dq_knee_est_rad_s"] = dq_knee
    dataframe["ddq_hip_est_rad_s2"] = ddq_hip
    dataframe["ddq_knee_est_rad_s2"] = ddq_knee
    dataframe["derivative_valid"] = derivative_valid
    dataframe["derivative_reason"] = reasons.astype(str)
    dataframe["filter_delay_s"] = filter_delay
    dataframe["uses_future_samples"] = uses_future

    if derivative_valid.any():
        finite_valid = np.isfinite(
            dataframe.loc[
                derivative_valid,
                [
                    "dq_hip_est_rad_s",
                    "dq_knee_est_rad_s",
                    "ddq_hip_est_rad_s2",
                    "ddq_knee_est_rad_s2",
                    "filter_delay_s",
                ],
            ].to_numpy(dtype=float)
        ).all()
        if not finite_valid:
            raise RuntimeError("A valid derivative sample is non-finite.")

    finite_maximum_gaps = [gap for gap in group_maximum_gaps if np.isfinite(gap)]
    metadata: dict[str, object] = {
        "method": method,
        "offline_only": offline,
        "causal": not offline,
        "uses_future_samples": offline,
        "ground_truth_dq_ddq_used": False,
        "input_columns_read": [
            prepared.time_column,
            prepared.hip_angle_column,
            prepared.knee_angle_column,
            *prepared.valid_columns,
            *prepared.group_columns,
        ],
        "angle_columns": [
            prepared.hip_angle_column,
            prepared.knee_angle_column,
        ],
        "valid_columns": list(prepared.valid_columns),
        "group_columns": list(prepared.group_columns),
        "segment_count": segment_count,
        "long_gap_boundary_count": len(long_gap_starts),
        "valid_sample_count": int(derivative_valid.sum()),
        "invalid_sample_count": int((~derivative_valid).sum()),
        "maximum_time_gap_s": (
            float(resolved_config.maximum_time_gap_s)
            if resolved_config.maximum_time_gap_s is not None
            else None
        ),
        "effective_maximum_time_gap_s_min": (
            float(min(finite_maximum_gaps)) if finite_maximum_gaps else None
        ),
        "effective_maximum_time_gap_s_max": (
            float(max(finite_maximum_gaps)) if finite_maximum_gaps else None
        ),
        "savgol_window_length": resolved_config.savgol_window_length,
        "savgol_polynomial_order": resolved_config.savgol_polynomial_order,
        "causal_filter_window_length": (
            resolved_config.causal_filter_window_length
        ),
        "array_input": prepared.array_input,
        "deterministic": True,
    }
    return DerivativeEstimationResult(dataframe=dataframe, metadata=metadata)


def central_difference_offline(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None = None,
    q_knee_rad: Sequence[float] | np.ndarray | None = None,
    **kwargs: object,
) -> DerivativeEstimationResult:
    """Symmetric offline finite differences (uses future samples)."""

    return estimate_joint_derivatives(
        data_or_time,
        q_hip_rad,
        q_knee_rad,
        method="central_difference_offline",
        **kwargs,
    )


def savitzky_golay_offline(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None = None,
    q_knee_rad: Sequence[float] | np.ndarray | None = None,
    **kwargs: object,
) -> DerivativeEstimationResult:
    """Symmetric Savitzky-Golay smoothing/differentiation (offline only)."""

    return estimate_joint_derivatives(
        data_or_time,
        q_hip_rad,
        q_knee_rad,
        method="savitzky_golay_offline",
        **kwargs,
    )


def causal_backward_difference(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None = None,
    q_knee_rad: Sequence[float] | np.ndarray | None = None,
    **kwargs: object,
) -> DerivativeEstimationResult:
    """Backward finite differences using current and historical angles only."""

    return estimate_joint_derivatives(
        data_or_time,
        q_hip_rad,
        q_knee_rad,
        method="causal_backward_difference",
        **kwargs,
    )


def causal_filter_and_difference(
    data_or_time: pd.DataFrame | Sequence[float] | np.ndarray,
    q_hip_rad: Sequence[float] | np.ndarray | None = None,
    q_knee_rad: Sequence[float] | np.ndarray | None = None,
    **kwargs: object,
) -> DerivativeEstimationResult:
    """Trailing moving-average filter plus causal backward differences."""

    return estimate_joint_derivatives(
        data_or_time,
        q_hip_rad,
        q_knee_rad,
        method="causal_filter_and_difference",
        **kwargs,
    )


__all__ = [
    "ANGLE_COLUMN_CANDIDATES",
    "DEFAULT_GROUP_COLUMN_CANDIDATES",
    "DERIVATIVE_METHODS",
    "DERIVATIVE_OUTPUT_COLUMNS",
    "DerivativeEstimationConfig",
    "DerivativeEstimationResult",
    "causal_backward_difference",
    "causal_filter_and_difference",
    "central_difference_offline",
    "estimate_joint_derivatives",
    "savitzky_golay_offline",
]
