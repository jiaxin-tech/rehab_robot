"""Load metrics on complete raw intervals with explicit phase boundaries."""
from __future__ import annotations

import numpy as np


METHOD = "RAW_WRENCH_EXACT_INTERVAL_TRAPEZOID_RMS_V1"


def load_window(clock, values, valid, begin, end, max_gap_s):
    """Retain raw peaks; interpolate only the two interval boundaries.

    Values are force/torque magnitudes, not a resampled command/model table.
    Invalid source samples and excessive source gaps are never interpolated.
    """
    clock = np.asarray(clock, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if (clock.ndim != 1 or len(clock) < 2 or values.shape != (len(clock), 2)
            or valid.shape != clock.shape or not np.isfinite(clock).all()
            or np.any(np.diff(clock) <= 0)):
        raise ValueError("load endpoints require strictly ordered raw samples")
    rounding = 8 * np.spacing(max(abs(float(clock[0])), abs(float(clock[-1])), 1.))
    if (not np.isfinite([begin, end, max_gap_s]).all() or begin >= end
            or max_gap_s <= 0 or begin < clock[0] - rounding or end > clock[-1] + rounding):
        raise ValueError("load interval requires covered finite boundaries; no extrapolation")
    # CSV and host-time addition can differ by a few floating-point ULPs.
    begin, end = max(begin, clock[0]), min(end, clock[-1])
    left = int(np.searchsorted(clock, begin, side="right") - 1)
    right = int(np.searchsorted(clock, end, side="left"))
    indices = np.arange(left, right + 1)
    if not valid[indices].all() or not np.isfinite(values[indices]).all():
        raise ValueError("invalid raw wrench in load interval or boundary bracket")
    if np.diff(clock[indices]).max() > max_gap_s + rounding:
        raise ValueError("raw wrench gap exceeds load interval limit")
    points = np.r_[begin, clock[(clock > begin) & (clock < end)], end]
    clipped = np.column_stack([np.interp(points, clock[indices], values[indices, column])
                               for column in range(2)])
    duration = float(end - begin)
    return {"rms": np.sqrt(np.trapezoid(clipped ** 2, points, axis=0) / duration),
            "peaks": np.max(np.abs(clipped), axis=0), "source_indices": indices,
            "duration_s": duration, "raw_sample_count": int(np.sum((clock >= begin) & (clock <= end))),
            "method": METHOD}
