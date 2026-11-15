"""Offline fixed-cycle review; never change a physical execution limit."""
from numbers import Real
import numpy as np
from control.taught_tcp_trajectory import DURATIONS, HOLDS, stage_kinematic_extrema


def review_fixed_cycle(settings, cycle_duration_s, *, speed_limit_m_s,
                       acceleration_limit_m_s2, margin=1.01, actual_duration_s=None):
    values = (cycle_duration_s, speed_limit_m_s, acceleration_limit_m_s2, margin)
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("fixed cycle and physical limits require finite positive numbers")
    if cycle_duration_s > 3600 or margin < 1:
        raise ValueError("fixed cycle must be at most 3600 seconds and margin at least 1")
    extrema = stage_kinematic_extrema(settings)
    total = sum(settings[key] for key in (*DURATIONS, *HOLDS))
    speed = max(v for v, a in extrema)
    acceleration = max(a for v, a in extrema)
    reasons = []
    if not np.isclose(total, cycle_duration_s, atol=1e-9, rtol=0):
        reasons.append("fixed_cycle_duration_mismatch")
    if actual_duration_s is not None and (isinstance(actual_duration_s, bool)
            or not isinstance(actual_duration_s, Real) or not np.isfinite(actual_duration_s)
            or not np.isclose(actual_duration_s, cycle_duration_s, atol=1e-9, rtol=0)):
        reasons.append("fixed_cycle_actual_duration_mismatch")
    if margin*speed > speed_limit_m_s + 1e-12:
        reasons.append("fixed_cycle_speed_limit_conflict")
    if margin*acceleration > acceleration_limit_m_s2 + 1e-12:
        reasons.append("fixed_cycle_acceleration_limit_conflict")
    return dict(allowed=not reasons, reasons=reasons, requested_duration_s=float(cycle_duration_s),
        planned_duration_s=float(total), maximum_speed_m_s=float(speed),
        maximum_acceleration_m_s2=float(acceleration), speed_limit_m_s=float(speed_limit_m_s),
        acceleration_limit_m_s2=float(acceleration_limit_m_s2), retiming_margin=float(margin),
        automatic_retiming=False, physical_limits_changed=False)
