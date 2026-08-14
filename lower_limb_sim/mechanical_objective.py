"""Mechanical-only objective for offline reference personalization.

The objective is deliberately limited to modeled joint torque.  It is not a
comfort, clinical-outcome, safety, or robot-execution metric.  All ratios are
normalized to the same subject/scenario's frozen active reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


MECHANICAL_OBJECTIVE_VERSION = "mechanical_joint_torque_objective_v1"
OBJECTIVE_EQUIVALENCE_TOLERANCE = 0.005


@dataclass(frozen=True)
class MechanicalTorqueMetrics:
    hip_rms_torque_nm: float
    knee_rms_torque_nm: float
    hip_peak_abs_torque_nm: float
    knee_peak_abs_torque_nm: float
    hip_rms_torque_rate_nm_s: float
    knee_rms_torque_rate_nm_s: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicalObjectiveResult:
    trajectory_id: str
    metrics: MechanicalTorqueMetrics
    reference_metrics: MechanicalTorqueMetrics
    hip_rms_ratio: float
    knee_rms_ratio: float
    mechanical_cost_j_rms: float
    hip_peak_ratio: float
    knee_peak_ratio: float
    combined_peak_ratio: float
    hip_torque_rate_ratio: float
    knee_torque_rate_ratio: float
    combined_torque_rate_ratio: float
    reference_deviation: float
    reference_deviation_definition: str

    def as_dict(self, *, prefix: str = "") -> dict[str, Any]:
        values: dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "hip_rms_ratio": self.hip_rms_ratio,
            "knee_rms_ratio": self.knee_rms_ratio,
            "mechanical_cost_j_rms": self.mechanical_cost_j_rms,
            "hip_peak_ratio": self.hip_peak_ratio,
            "knee_peak_ratio": self.knee_peak_ratio,
            "combined_peak_ratio": self.combined_peak_ratio,
            "hip_torque_rate_ratio": self.hip_torque_rate_ratio,
            "knee_torque_rate_ratio": self.knee_torque_rate_ratio,
            "combined_torque_rate_ratio": self.combined_torque_rate_ratio,
            "reference_deviation": self.reference_deviation,
            "reference_deviation_definition": self.reference_deviation_definition,
        }
        values.update(self.metrics.as_dict())
        values.update(
            {
                f"reference_{key}": value
                for key, value in self.reference_metrics.as_dict().items()
            }
        )
        return {f"{prefix}{key}": value for key, value in values.items()}


def _finite_vector(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) < 3:
        raise ValueError(f"{name} must be a one-dimensional vector with >=3 samples")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _time_weighted_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0.0:
        raise ValueError("time_s duration must be positive")
    return float(np.sqrt(np.trapezoid(values**2, time_s) / duration))


def compute_torque_metrics(
    time_s: Iterable[float],
    tau_hip_nm: Iterable[float],
    tau_knee_nm: Iterable[float],
) -> MechanicalTorqueMetrics:
    """Compute duration-aware joint-torque and torque-rate summaries."""

    time = _finite_vector(time_s, "time_s")
    hip = _finite_vector(tau_hip_nm, "tau_hip_nm")
    knee = _finite_vector(tau_knee_nm, "tau_knee_nm")
    if not (len(time) == len(hip) == len(knee)):
        raise ValueError("time and joint torque vectors must have equal length")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    hip_rate = np.gradient(hip, time, edge_order=2)
    knee_rate = np.gradient(knee, time, edge_order=2)
    return MechanicalTorqueMetrics(
        hip_rms_torque_nm=_time_weighted_rms(hip, time),
        knee_rms_torque_nm=_time_weighted_rms(knee, time),
        hip_peak_abs_torque_nm=float(np.max(np.abs(hip))),
        knee_peak_abs_torque_nm=float(np.max(np.abs(knee))),
        hip_rms_torque_rate_nm_s=_time_weighted_rms(hip_rate, time),
        knee_rms_torque_rate_nm_s=_time_weighted_rms(knee_rate, time),
    )


def _positive_ratio(numerator: float, denominator: float, name: str) -> float:
    if not math.isfinite(numerator) or numerator < 0.0:
        raise ValueError(f"{name} numerator must be finite and non-negative")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"{name} reference denominator must be finite and positive")
    return float(numerator / denominator)


def evaluate_mechanical_objective(
    *,
    trajectory_id: str,
    metrics: MechanicalTorqueMetrics,
    reference_metrics: MechanicalTorqueMetrics,
    hip_rms_deviation_deg: float,
    knee_rms_deviation_deg: float,
) -> MechanicalObjectiveResult:
    """Normalize one candidate against the same model's frozen reference."""

    if not str(trajectory_id).strip():
        raise ValueError("trajectory_id must not be empty")
    hip_deviation = float(hip_rms_deviation_deg)
    knee_deviation = float(knee_rms_deviation_deg)
    if (
        not math.isfinite(hip_deviation)
        or not math.isfinite(knee_deviation)
        or hip_deviation < 0.0
        or knee_deviation < 0.0
    ):
        raise ValueError("joint RMS reference deviations must be finite and non-negative")

    hip_rms_ratio = _positive_ratio(
        metrics.hip_rms_torque_nm,
        reference_metrics.hip_rms_torque_nm,
        "hip RMS torque",
    )
    knee_rms_ratio = _positive_ratio(
        metrics.knee_rms_torque_nm,
        reference_metrics.knee_rms_torque_nm,
        "knee RMS torque",
    )
    hip_peak_ratio = _positive_ratio(
        metrics.hip_peak_abs_torque_nm,
        reference_metrics.hip_peak_abs_torque_nm,
        "hip peak torque",
    )
    knee_peak_ratio = _positive_ratio(
        metrics.knee_peak_abs_torque_nm,
        reference_metrics.knee_peak_abs_torque_nm,
        "knee peak torque",
    )
    hip_rate_ratio = _positive_ratio(
        metrics.hip_rms_torque_rate_nm_s,
        reference_metrics.hip_rms_torque_rate_nm_s,
        "hip torque rate",
    )
    knee_rate_ratio = _positive_ratio(
        metrics.knee_rms_torque_rate_nm_s,
        reference_metrics.knee_rms_torque_rate_nm_s,
        "knee torque rate",
    )
    return MechanicalObjectiveResult(
        trajectory_id=str(trajectory_id),
        metrics=metrics,
        reference_metrics=reference_metrics,
        hip_rms_ratio=hip_rms_ratio,
        knee_rms_ratio=knee_rms_ratio,
        mechanical_cost_j_rms=float(
            math.sqrt((hip_rms_ratio**2 + knee_rms_ratio**2) / 2.0)
        ),
        hip_peak_ratio=hip_peak_ratio,
        knee_peak_ratio=knee_peak_ratio,
        combined_peak_ratio=float(
            math.sqrt((hip_peak_ratio**2 + knee_peak_ratio**2) / 2.0)
        ),
        hip_torque_rate_ratio=hip_rate_ratio,
        knee_torque_rate_ratio=knee_rate_ratio,
        combined_torque_rate_ratio=float(
            math.sqrt((hip_rate_ratio**2 + knee_rate_ratio**2) / 2.0)
        ),
        reference_deviation=float(
            math.sqrt((hip_deviation**2 + knee_deviation**2) / 2.0)
        ),
        reference_deviation_definition=(
            "root_mean_square_of_hip_and_knee_rms_angle_deviation_deg"
        ),
    )


def rank_feasible_candidates(
    candidates: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    equivalence_tolerance: float = OBJECTIVE_EQUIVALENCE_TOLERANCE,
) -> pd.DataFrame:
    """Return feasible candidates in deterministic selection order.

    Candidates within ``equivalence_tolerance`` of the minimum mechanical cost
    are treated as mechanically equivalent and ordered by deviation, peak
    ratio, torque-rate ratio, and finally lexical trajectory identity.
    """

    table = (
        candidates.copy(deep=True)
        if isinstance(candidates, pd.DataFrame)
        else pd.DataFrame(list(candidates))
    )
    required = {
        "trajectory_id",
        "trajectory_feasible",
        "mechanical_cost_j_rms",
        "reference_deviation",
        "combined_peak_ratio",
        "combined_torque_rate_ratio",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ranking table missing columns: {sorted(missing)}")
    tolerance = float(equivalence_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("equivalence_tolerance must be finite and non-negative")
    table = table.loc[table["trajectory_feasible"].astype(bool)].copy()
    if table.empty:
        return table
    numeric_columns = (
        "mechanical_cost_j_rms",
        "reference_deviation",
        "combined_peak_ratio",
        "combined_torque_rate_ratio",
    )
    numeric = table.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("ranking metrics must be finite")
    minimum = float(table["mechanical_cost_j_rms"].min())
    table["mechanically_equivalent_to_minimum"] = (
        table["mechanical_cost_j_rms"] <= minimum + tolerance + 1e-15
    )
    equivalent = table.loc[table["mechanically_equivalent_to_minimum"]].sort_values(
        [
            "reference_deviation",
            "combined_peak_ratio",
            "combined_torque_rate_ratio",
            "trajectory_id",
        ],
        kind="mergesort",
    )
    remaining = table.loc[~table["mechanically_equivalent_to_minimum"]].sort_values(
        [
            "mechanical_cost_j_rms",
            "reference_deviation",
            "combined_peak_ratio",
            "combined_torque_rate_ratio",
            "trajectory_id",
        ],
        kind="mergesort",
    )
    ranked = pd.concat((equivalent, remaining), ignore_index=True)
    ranked["deterministic_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["objective_equivalence_tolerance"] = tolerance
    return ranked


__all__ = [
    "MECHANICAL_OBJECTIVE_VERSION",
    "OBJECTIVE_EQUIVALENCE_TOLERANCE",
    "MechanicalObjectiveResult",
    "MechanicalTorqueMetrics",
    "compute_torque_metrics",
    "evaluate_mechanical_objective",
    "rank_feasible_candidates",
]
