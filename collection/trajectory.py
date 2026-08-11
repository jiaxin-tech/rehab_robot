"""Generic Cartesian path geometry used by collection/alignment.

The former single-joint sinusoidal sweep and drag-based circle calibration
belonged to the retired experiment branch.  Current rehabilitation references
come from ``lower_limb_sim`` and are transformed explicitly by ``control``.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryProjection:
    """Nearest valid path projection in the robot base frame."""

    trajectory_s: float
    arc_length_m: float
    tangent_base: tuple[float, float, float]


class TrajectoryGeometry:
    """Arc-length and tangent helper for a Cartesian waypoint path.

    Consecutive duplicate waypoints are skipped for projection rather than
    normalized into a bogus direction.  A path containing no non-zero segment
    remains representable but all projections return an explicit invalid reason.
    """

    def __init__(self, waypoints: np.ndarray):
        values = np.asarray(waypoints, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("Trajectory must be an (N, >=3) finite waypoint array")
        if len(values) == 0 or not np.all(np.isfinite(values[:, :3])):
            raise ValueError("Trajectory positions must be non-empty and finite")
        self.waypoints = values
        self.positions_m = values[:, :3].copy()
        deltas = np.diff(self.positions_m, axis=0)
        self.segment_length_m = np.linalg.norm(deltas, axis=1)
        self._valid_segment = self.segment_length_m > 1e-12
        self.arc_at_waypoint_m = np.concatenate(
            ([0.0], np.cumsum(self.segment_length_m))
        )
        self.total_arc_length_m = float(self.arc_at_waypoint_m[-1])

    def project(
        self,
        position_m,
        *,
        reference_arc_length_m: float | None = None,
        continuity_tolerance_m: float = 0.003,
    ) -> tuple[TrajectoryProjection | None, str]:
        """Project a base-frame position, optionally preserving path phase.

        A rehabilitation trajectory can intentionally retrace the same physical
        arc.  Geometry alone then has more than one valid tangent/arc result.
        Supplying the preceding chronological arc length resolves candidates
        within a small spatial tolerance by continuity instead of silently
        mapping every return pass to the first segment.
        """
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None, "trajectory_projection_invalid_position"
        if self.total_arc_length_m <= 1e-12 or not np.any(self._valid_segment):
            return None, "trajectory_zero_length"
        if not np.isfinite(continuity_tolerance_m) or continuity_tolerance_m < 0.0:
            return None, "trajectory_projection_continuity_tolerance_invalid"
        reference_arc: float | None = None
        if reference_arc_length_m is not None:
            reference_arc = float(reference_arc_length_m)
            if not np.isfinite(reference_arc):
                return None, "trajectory_reference_arc_invalid"
            if reference_arc < -1e-12 or reference_arc > self.total_arc_length_m + 1e-12:
                return None, "trajectory_reference_arc_out_of_range"
            reference_arc = min(self.total_arc_length_m, max(0.0, reference_arc))

        candidates: list[tuple[float, int, float, float]] = []
        for index, length in enumerate(self.segment_length_m):
            if not self._valid_segment[index]:
                continue
            start = self.positions_m[index]
            delta = self.positions_m[index + 1] - start
            alpha = float(np.dot(position - start, delta) / (length * length))
            alpha = min(1.0, max(0.0, alpha))
            candidate = start + alpha * delta
            distance_sq = float(np.dot(position - candidate, position - candidate))
            arc_length_m = float(
                self.arc_at_waypoint_m[index] + alpha * self.segment_length_m[index]
            )
            candidates.append((distance_sq, index, alpha, arc_length_m))

        if not candidates:
            return None, "trajectory_zero_length"
        min_distance_sq = min(item[0] for item in candidates)
        if reference_arc is None:
            # Stable ordering makes the no-phase case deterministic, while the
            # caller can choose the continuity-aware path whenever it has a
            # previous projection/command phase.
            _, best_index, best_alpha, arc_length_m = min(
                candidates, key=lambda item: (item[0], item[1])
            )
        else:
            tolerance_sq = float(continuity_tolerance_m) ** 2
            phase_candidates = [
                item for item in candidates if item[0] <= min_distance_sq + tolerance_sq
            ]
            _, best_index, best_alpha, arc_length_m = min(
                phase_candidates,
                key=lambda item: (abs(item[3] - reference_arc), item[0], item[1]),
            )
        delta = self.positions_m[best_index + 1] - self.positions_m[best_index]
        tangent = delta / self.segment_length_m[best_index]
        return (
            TrajectoryProjection(
                trajectory_s=arc_length_m / self.total_arc_length_m,
                arc_length_m=arc_length_m,
                tangent_base=tuple(float(value) for value in tangent),
            ),
            "",
        )

    def pose_at_normalized_s(self, trajectory_s: float) -> np.ndarray:
        """Interpolate a full waypoint pose at a clipped arc-length parameter."""
        if not np.isfinite(trajectory_s):
            raise ValueError("trajectory_s must be finite")
        if self.total_arc_length_m <= 1e-12 or not np.any(self._valid_segment):
            raise ValueError("Cannot sample a zero-length trajectory")
        arc_m = min(1.0, max(0.0, float(trajectory_s))) * self.total_arc_length_m
        index = int(np.searchsorted(self.arc_at_waypoint_m, arc_m, side="right") - 1)
        index = min(max(0, index), len(self.segment_length_m) - 1)
        if not self._valid_segment[index]:
            candidates = np.flatnonzero(self._valid_segment)
            index = int(candidates[np.argmin(np.abs(candidates - index))])
        segment_start_m = self.arc_at_waypoint_m[index]
        alpha = (arc_m - segment_start_m) / self.segment_length_m[index]
        alpha = min(1.0, max(0.0, float(alpha)))
        return (1.0 - alpha) * self.waypoints[index] + alpha * self.waypoints[index + 1]


def project_along_tangent(vector_base, tangent_base) -> float | None:
    """Project a finite base-frame Cartesian vector onto a unit tangent."""
    vector = np.asarray(vector_base, dtype=float)
    tangent = np.asarray(tangent_base, dtype=float)
    if vector.shape != (3,) or tangent.shape != (3,):
        return None
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(tangent)):
        return None
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return None
    value = float(np.dot(vector, tangent / norm))
    return value if np.isfinite(value) else None
