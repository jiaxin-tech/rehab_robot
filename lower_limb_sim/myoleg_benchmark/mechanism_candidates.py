"""A small, predeclared V2 candidate family for mechanism screening.

The family changes cycle duration and the timing/shape of a smooth knee
coordination excursion. It is intentionally separate from the frozen V1
``Domain`` and does not claim to model assistance force: the native MyoLeg
prescribed-state interface has no actuator-assistance control input.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np

from lower_limb_sim.visualization.myoleg_robot_scene import native_domain


DURATION_SCALES = (0.8, 1.0, 1.2)
COORDINATION_LAGS = (-0.12, 0.0, 0.12)
COORDINATION_AMPLITUDES = (-0.04, 0.0, 0.04)
MECHANISM_ROM_LIMIT_RAD = 0.02


@dataclass(frozen=True)
class MechanismPoint:
    candidate_id: str
    candidate_index: int
    duration_scale: float
    coordination_lag: float
    coordination_amplitude: float
    trajectory: Any
    time_s: np.ndarray
    kinematic: dict[str, float]

    @property
    def features(self) -> tuple[float, float, float]:
        return (self.duration_scale, self.coordination_lag, self.coordination_amplitude)

    @property
    def beta(self) -> tuple[float, float]:
        """Legacy endpoint payload slots; V2 factors live in ``features``."""
        return (0.0, 0.0)


def _bump(s: np.ndarray, lag: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C2 compact bump and its first two derivatives with respect to phase."""
    start = 0.25 + lag
    t = np.clip((s - start) / 0.5, 0.0, 1.0)
    active = ((s >= start) & (s <= start + 0.5)).astype(float)
    b = t**3 * (1.0 - t)**3
    db = 3.0 * t**2 * (1.0 - t)**2 * (1.0 - 2.0 * t)
    d2b = 6.0 * t * (1.0 - t) * (1.0 - 3.0 * t + 3.0 * t**2)
    return active * b, active * db / 0.5, active * d2b / 0.5**2


def make_mechanism_trajectory(reference: Any, *, duration_scale: float,
                              coordination_lag: float,
                              coordination_amplitude: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Construct one bounded V2 candidate around the reference trajectory.

    The knee excursion is checked against the reference range, with at most
    ``MECHANISM_ROM_LIMIT_RAD`` of explicitly declared envelope expansion. This
    is a diagnostic mechanism domain; it does not mean that the native interface
    preserves the original ROM exactly or exposes an assistance controller.
    """
    if duration_scale <= 0 or abs(coordination_lag) > 0.12 or abs(coordination_amplitude) > 0.08:
        raise ValueError("MECHANISM_PARAMETER_OUT_OF_DECLARED_RANGE")
    old_time = np.asarray(reference.time_s, dtype=float)
    phase = (old_time - old_time[0]) / (old_time[-1] - old_time[0])
    q = np.asarray(reference.q, dtype=float).copy()
    dq = np.asarray(reference.dq, dtype=float).copy()
    ddq = np.asarray(reference.ddq, dtype=float).copy()
    span = float(np.ptp(q[:, 1]))
    bump, dbump, d2bump = _bump(phase, coordination_lag)
    # A signed, smooth knee excursion is a coordination perturbation, not a
    # claim that the simulator exposes a physiological assistance controller.
    q[:, 1] += coordination_amplitude * span * bump
    phase_rate = 1.0 / (old_time[-1] - old_time[0])
    dq[:, 1] += coordination_amplitude * span * dbump * phase_rate
    ddq[:, 1] += coordination_amplitude * span * d2bump * phase_rate**2
    new_time = old_time[0] + (old_time - old_time[0]) * duration_scale
    dq /= duration_scale
    ddq /= duration_scale**2
    reference_lo, reference_hi = float(np.min(reference.q[:, 1])), float(np.max(reference.q[:, 1]))
    rom_excess = max(float(np.max(reference_lo - q[:, 1])), float(np.max(q[:, 1] - reference_hi)), 0.0)
    if rom_excess > MECHANISM_ROM_LIMIT_RAD:
        raise ValueError("MECHANISM_ROM_LIMIT")
    kin = {
        "duration_scale": float(duration_scale),
        "coordination_lag": float(coordination_lag),
        "coordination_amplitude": float(coordination_amplitude),
        "hip_speed_ratio": float(np.max(np.abs(dq[:, 0])) / np.max(np.abs(reference.dq[:, 0]))),
        "knee_speed_ratio": float(np.max(np.abs(dq[:, 1])) / np.max(np.abs(reference.dq[:, 1]))),
        "hip_accel_ratio": float(np.max(np.abs(ddq[:, 0])) / np.max(np.abs(reference.ddq[:, 0]))),
        "knee_accel_ratio": float(np.max(np.abs(ddq[:, 1])) / np.max(np.abs(reference.ddq[:, 1]))),
        "rom_excess_rad": float(rom_excess),
    }
    if not np.isfinite(np.concatenate([q.ravel(), dq.ravel(), ddq.ravel(), new_time])).all():
        raise ValueError("NONFINITE_MECHANISM_TRAJECTORY")
    if kin["hip_speed_ratio"] > 1.5 or kin["knee_speed_ratio"] > 1.5 or kin["hip_accel_ratio"] > 2.0 or kin["knee_accel_ratio"] > 2.0:
        raise ValueError("KINEMATIC_COMPARISON_LIMIT")
    return {"time_s": new_time, "q": q, "dq": dq, "ddq": ddq,
            "phases": np.asarray(reference.phases)}, kin


class MechanismDomain:
    """The fixed 27-point speed/coordination screening domain.

    This domain tests whether native dynamics contain a subject-dependent
    response signal. It is not itself a personalization policy or evidence that
    a learner can improve regret.
    """

    family = "DURATION_COORDINATION_V2"

    def __init__(self, source: Any | None = None):
        source = native_domain() if source is None else source
        self.profile = source.profile
        self.subject_reference = source.subject_reference
        self.points: list[MechanismPoint] = []
        self.rejections: list[dict[str, object]] = []
        combinations = list(product(DURATION_SCALES, COORDINATION_LAGS, COORDINATION_AMPLITUDES))
        combinations.sort(key=lambda x: (x != (1.0, 0.0, 0.0), x))
        for parameters in combinations:
            try:
                trajectory, kin = make_mechanism_trajectory(self.subject_reference,
                                                            duration_scale=parameters[0],
                                                            coordination_lag=parameters[1],
                                                            coordination_amplitude=parameters[2])
                index = len(self.points)
                self.points.append(MechanismPoint(
                    f"{self.family}:{index}", index, *parameters,
                    SimpleNamespace(q=trajectory["q"], dq=trajectory["dq"], ddq=trajectory["ddq"]),
                    trajectory["time_s"], kin))
            except ValueError as error:
                self.rejections.append({"parameters": parameters, "reason": str(error)})
        self.lookup = {point.candidate_id: point for point in self.points}
        self.reference = next(point for point in self.points
                              if point.features == (1.0, 0.0, 0.0))

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def by_id(self, candidate_id: str) -> MechanismPoint:
        return self.lookup[candidate_id]


__all__ = ["DURATION_SCALES", "COORDINATION_LAGS", "COORDINATION_AMPLITUDES",
           "MechanismPoint", "MechanismDomain", "make_mechanism_trajectory"]
