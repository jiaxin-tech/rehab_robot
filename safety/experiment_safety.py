"""Physical limits required for robot execution; no approval workflow."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from config.bundle import resolve_config_path

SCALAR_LIMITS = (
    "max_tcp_speed_m_s", "max_tcp_acceleration_m_s2",
    "max_start_anchor_position_error_m", "max_start_anchor_orientation_error_rad",
    "max_command_lateness_s", "max_force_n", "max_torque_nm",
    "max_state_age_s", "max_wrench_age_s", "max_state_wrench_skew_s",
    "rt_filter_hz",
)


@dataclass(frozen=True)
class ExperimentSafetyConfig:
    max_tcp_speed_m_s: float | None = None
    max_tcp_acceleration_m_s2: float | None = None
    max_start_anchor_position_error_m: float | None = None
    max_start_anchor_orientation_error_rad: float | None = None
    max_command_lateness_s: float | None = None
    max_force_n: float | None = None
    max_torque_nm: float | None = None
    max_state_age_s: float | None = None
    max_wrench_age_s: float | None = None
    max_state_wrench_skew_s: float | None = None
    workspace_min_base_m: tuple | None = None
    workspace_max_base_m: tuple | None = None
    rt_filter_hz: float | None = None
    rt_network_tolerance_percent: float | None = None

    def __post_init__(self):
        for name in SCALAR_LIMITS:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError(name + " must be a finite positive number")
        for name in ("workspace_min_base_m", "workspace_max_base_m"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (tuple, list)) or len(value) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
                    raise ValueError(name + " must contain three finite numbers")
                object.__setattr__(self, name, tuple(value))
        lower, upper = self.workspace_min_base_m, self.workspace_max_base_m
        if (lower is None) != (upper is None) or (lower is not None and any(a >= b for a, b in zip(lower, upper))):
            raise ValueError("workspace bounds must be paired and increasing")
        value = self.rt_network_tolerance_percent
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not float(value).is_integer() or not 0 <= value <= 100):
            raise ValueError("rt_network_tolerance_percent must be an integer in [0,100]")
        if self.rt_filter_hz is not None and not 1 <= self.rt_filter_hz <= 1000:
            raise ValueError("rt_filter_hz must be in [1,1000]")

    def execution_block_reasons(self):
        return tuple("missing " + name for name in (*SCALAR_LIMITS, "workspace_min_base_m", "workspace_max_base_m", "rt_network_tolerance_percent") if getattr(self, name) is None)

    def require_execute_allowed(self):
        reasons = self.execution_block_reasons()
        if reasons:
            raise ValueError("motion limits: " + "; ".join(reasons))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load_json(cls, path):
        source = resolve_config_path(path, section="experiment_safety")
        return cls(**json.loads(source.read_text(encoding="utf-8")))


def load_experiment_safety_config(path):
    return ExperimentSafetyConfig.load_json(path)
