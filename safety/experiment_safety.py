"""Reviewed, fail-closed safety configuration for real motion execution.

No physical threshold is chosen here.  The repository default keeps every
limit unset, and execution remains blocked until a human reviewer supplies all
limits and explicitly marks the configuration reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


EXPERIMENT_SAFETY_SCHEMA_VERSION = 3
_SCALAR_LIMIT_FIELDS = (
    "max_tcp_speed_m_s",
    "max_tcp_acceleration_m_s2",
    "max_start_anchor_position_error_m",
    "max_start_anchor_orientation_error_rad",
    "max_command_lateness_s",
    "max_force_n",
    "max_torque_nm",
    "max_state_age_s",
    "max_wrench_age_s",
    "max_state_wrench_skew_s",
)
_IDENTITY_AND_NAME_FIELDS = (
    "expected_robot_model",
    "expected_robot_serial_number",
    "expected_controller_version",
    "reviewed_tool_name",
    "reviewed_workpiece_name",
)
_REVIEW_FLAG_FIELDS = (
    "robot_identity_reviewed",
    "tool_workpiece_reviewed",
    "payload_configuration_reviewed",
    "collision_configuration_reviewed",
    "joint_soft_limits_reviewed",
    "realtime_configuration_reviewed",
)


def _positive_optional(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be null or a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be null or a finite positive number")
    return parsed


def _nonnegative_optional(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be null or a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be null or a finite non-negative number")
    return parsed


def _optional_nonempty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be null or a non-empty string")
    return value.strip()


def _finite_vector_optional(
    value: object,
    size: int,
    field_name: str,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != size:
        raise ValueError(f"{field_name} must be null or contain {size} finite values")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} must be null or contain {size} finite values")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be null or contain {size} finite values")
        parsed.append(number)
    return tuple(parsed)


def _joint_soft_limits_optional(
    value: object,
    field_name: str,
) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 6:
        raise ValueError(f"{field_name} must be null or contain six [lower, upper] pairs")
    parsed: list[tuple[float, float]] = []
    for index, pair in enumerate(value, start=1):
        values = _finite_vector_optional(pair, 2, f"{field_name}[{index - 1}]")
        assert values is not None
        lower, upper = values
        if lower >= upper:
            raise ValueError(f"{field_name}[{index - 1}] lower must be below upper")
        parsed.append((lower, upper))
    return tuple(parsed)


def _workspace_vector(value: object, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{field_name} must be null or contain three finite values")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} must be null or contain three finite values")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be null or contain three finite values")
        parsed.append(number)
    return tuple(parsed)  # type: ignore[return-value]


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class ExperimentSafetyConfig:
    """Site-specific limits required before any real trajectory execution."""

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
    workspace_min_base_m: tuple[float, float, float] | None = None
    workspace_max_base_m: tuple[float, float, float] | None = None
    expected_robot_model: str | None = None
    expected_robot_serial_number: str | None = None
    expected_controller_version: str | None = None
    reviewed_tool_name: str | None = None
    reviewed_workpiece_name: str | None = None
    reviewed_payload_mass_kg: float | None = None
    reviewed_payload_cog_m: tuple[float, float, float] | None = None
    reviewed_payload_inertia_kg_m2: tuple[float, float, float] | None = None
    reviewed_joint_soft_limits_rad: tuple[tuple[float, float], ...] | None = None
    reviewed_rt_filter_hz: float | None = None
    reviewed_rt_network_tolerance_percent: float | None = None
    robot_identity_reviewed: bool = False
    tool_workpiece_reviewed: bool = False
    payload_configuration_reviewed: bool = False
    collision_configuration_reviewed: bool = False
    joint_soft_limits_reviewed: bool = False
    realtime_configuration_reviewed: bool = False
    reviewed: bool = False
    notes: str = ""
    schema_version: int = EXPERIMENT_SAFETY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != EXPERIMENT_SAFETY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be the integer {EXPERIMENT_SAFETY_SCHEMA_VERSION}"
            )
        if type(self.reviewed) is not bool:
            raise ValueError("reviewed must be a JSON/Python boolean")
        for field_name in _REVIEW_FLAG_FIELDS:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a JSON/Python boolean")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        for field_name in _SCALAR_LIMIT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _positive_optional(getattr(self, field_name), field_name),
            )
        workspace_min = _workspace_vector(
            self.workspace_min_base_m,
            "workspace_min_base_m",
        )
        workspace_max = _workspace_vector(
            self.workspace_max_base_m,
            "workspace_max_base_m",
        )
        if (workspace_min is None) != (workspace_max is None):
            raise ValueError(
                "workspace_min_base_m and workspace_max_base_m must both be null "
                "or both be configured"
            )
        if workspace_min is not None and workspace_max is not None:
            if any(lower >= upper for lower, upper in zip(workspace_min, workspace_max)):
                raise ValueError(
                    "workspace_min_base_m must be strictly below "
                    "workspace_max_base_m on every axis"
                )
        object.__setattr__(self, "workspace_min_base_m", workspace_min)
        object.__setattr__(self, "workspace_max_base_m", workspace_max)
        for field_name in _IDENTITY_AND_NAME_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _optional_nonempty_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "reviewed_payload_mass_kg",
            _nonnegative_optional(
                self.reviewed_payload_mass_kg,
                "reviewed_payload_mass_kg",
            ),
        )
        payload_cog = _finite_vector_optional(
            self.reviewed_payload_cog_m,
            3,
            "reviewed_payload_cog_m",
        )
        payload_inertia = _finite_vector_optional(
            self.reviewed_payload_inertia_kg_m2,
            3,
            "reviewed_payload_inertia_kg_m2",
        )
        if payload_inertia is not None and any(value < 0.0 for value in payload_inertia):
            raise ValueError(
                "reviewed_payload_inertia_kg_m2 must contain non-negative values"
            )
        object.__setattr__(self, "reviewed_payload_cog_m", payload_cog)
        object.__setattr__(
            self,
            "reviewed_payload_inertia_kg_m2",
            payload_inertia,
        )
        object.__setattr__(
            self,
            "reviewed_joint_soft_limits_rad",
            _joint_soft_limits_optional(
                self.reviewed_joint_soft_limits_rad,
                "reviewed_joint_soft_limits_rad",
            ),
        )
        reviewed_filter = _positive_optional(
            self.reviewed_rt_filter_hz,
            "reviewed_rt_filter_hz",
        )
        if reviewed_filter is not None and not 1.0 <= reviewed_filter <= 1000.0:
            raise ValueError("reviewed_rt_filter_hz must be in [1, 1000]")
        object.__setattr__(self, "reviewed_rt_filter_hz", reviewed_filter)
        reviewed_network_tolerance = _nonnegative_optional(
            self.reviewed_rt_network_tolerance_percent,
            "reviewed_rt_network_tolerance_percent",
        )
        if reviewed_network_tolerance is not None and reviewed_network_tolerance > 100.0:
            raise ValueError(
                "reviewed_rt_network_tolerance_percent must be in [0, 100]"
            )
        object.__setattr__(
            self,
            "reviewed_rt_network_tolerance_percent",
            reviewed_network_tolerance,
        )

    def execution_block_reasons(self) -> tuple[str, ...]:
        """Return every reason real execution must remain disabled."""

        reasons: list[str] = []
        if not self.reviewed:
            reasons.append("experiment_safety_not_reviewed")
        for field_name in _SCALAR_LIMIT_FIELDS:
            if getattr(self, field_name) is None:
                reasons.append(f"{field_name}_not_configured")
        if self.workspace_min_base_m is None or self.workspace_max_base_m is None:
            reasons.append("workspace_bounds_not_configured")
        for field_name in _IDENTITY_AND_NAME_FIELDS:
            if getattr(self, field_name) is None:
                reasons.append(f"{field_name}_not_configured")
        for field_name in (
            "reviewed_payload_mass_kg",
            "reviewed_payload_cog_m",
            "reviewed_payload_inertia_kg_m2",
            "reviewed_joint_soft_limits_rad",
            "reviewed_rt_filter_hz",
            "reviewed_rt_network_tolerance_percent",
        ):
            if getattr(self, field_name) is None:
                reasons.append(f"{field_name}_not_configured")
        for field_name in _REVIEW_FLAG_FIELDS:
            if not getattr(self, field_name):
                reasons.append(
                    f"{field_name.removesuffix('_reviewed')}_not_reviewed"
                )
        return tuple(reasons)

    def validate_for_execution(self) -> tuple[bool, tuple[str, ...]]:
        reasons = self.execution_block_reasons()
        return not reasons, reasons

    @property
    def execution_allowed(self) -> bool:
        return not self.execution_block_reasons()

    def require_execute_allowed(self) -> None:
        """Raise unless a human-reviewed, complete configuration is present."""

        reasons = self.execution_block_reasons()
        if reasons:
            raise PermissionError(
                "real robot execution blocked by experiment safety: "
                + ";".join(reasons)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "max_tcp_speed_m_s": self.max_tcp_speed_m_s,
            "max_tcp_acceleration_m_s2": self.max_tcp_acceleration_m_s2,
            "max_start_anchor_position_error_m": self.max_start_anchor_position_error_m,
            "max_start_anchor_orientation_error_rad": self.max_start_anchor_orientation_error_rad,
            "max_command_lateness_s": self.max_command_lateness_s,
            "max_force_n": self.max_force_n,
            "max_torque_nm": self.max_torque_nm,
            "max_state_age_s": self.max_state_age_s,
            "max_wrench_age_s": self.max_wrench_age_s,
            "max_state_wrench_skew_s": self.max_state_wrench_skew_s,
            "workspace_min_base_m": (
                None
                if self.workspace_min_base_m is None
                else list(self.workspace_min_base_m)
            ),
            "workspace_max_base_m": (
                None
                if self.workspace_max_base_m is None
                else list(self.workspace_max_base_m)
            ),
            "expected_robot_model": self.expected_robot_model,
            "expected_robot_serial_number": self.expected_robot_serial_number,
            "expected_controller_version": self.expected_controller_version,
            "reviewed_tool_name": self.reviewed_tool_name,
            "reviewed_workpiece_name": self.reviewed_workpiece_name,
            "reviewed_payload_mass_kg": self.reviewed_payload_mass_kg,
            "reviewed_payload_cog_m": (
                None
                if self.reviewed_payload_cog_m is None
                else list(self.reviewed_payload_cog_m)
            ),
            "reviewed_payload_inertia_kg_m2": (
                None
                if self.reviewed_payload_inertia_kg_m2 is None
                else list(self.reviewed_payload_inertia_kg_m2)
            ),
            "reviewed_joint_soft_limits_rad": (
                None
                if self.reviewed_joint_soft_limits_rad is None
                else [list(pair) for pair in self.reviewed_joint_soft_limits_rad]
            ),
            "reviewed_rt_filter_hz": self.reviewed_rt_filter_hz,
            "reviewed_rt_network_tolerance_percent": (
                self.reviewed_rt_network_tolerance_percent
            ),
            "robot_identity_reviewed": self.robot_identity_reviewed,
            "tool_workpiece_reviewed": self.tool_workpiece_reviewed,
            "payload_configuration_reviewed": self.payload_configuration_reviewed,
            "collision_configuration_reviewed": self.collision_configuration_reviewed,
            "joint_soft_limits_reviewed": self.joint_soft_limits_reviewed,
            "realtime_configuration_reviewed": (
                self.realtime_configuration_reviewed
            ),
            "reviewed": self.reviewed,
            "notes": self.notes,
        }

    def save_json(self, path: str | Path) -> Path:
        return _atomic_write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentSafetyConfig":
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"experiment safety JSON not found: {source}")
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object_pairs,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid experiment safety JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("experiment safety JSON must contain one object")
        expected = {
            "schema_version",
            *_SCALAR_LIMIT_FIELDS,
            "workspace_min_base_m",
            "workspace_max_base_m",
            *_IDENTITY_AND_NAME_FIELDS,
            "reviewed_payload_mass_kg",
            "reviewed_payload_cog_m",
            "reviewed_payload_inertia_kg_m2",
            "reviewed_joint_soft_limits_rad",
            "reviewed_rt_filter_hz",
            "reviewed_rt_network_tolerance_percent",
            *_REVIEW_FLAG_FIELDS,
            "reviewed",
            "notes",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                f"experiment safety fields must be exactly {sorted(expected)}; "
                f"got {sorted(actual)}"
            )
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            max_tcp_speed_m_s=payload["max_tcp_speed_m_s"],  # type: ignore[arg-type]
            max_tcp_acceleration_m_s2=payload["max_tcp_acceleration_m_s2"],  # type: ignore[arg-type]
            max_start_anchor_position_error_m=payload["max_start_anchor_position_error_m"],  # type: ignore[arg-type]
            max_start_anchor_orientation_error_rad=payload["max_start_anchor_orientation_error_rad"],  # type: ignore[arg-type]
            max_command_lateness_s=payload["max_command_lateness_s"],  # type: ignore[arg-type]
            max_force_n=payload["max_force_n"],  # type: ignore[arg-type]
            max_torque_nm=payload["max_torque_nm"],  # type: ignore[arg-type]
            max_state_age_s=payload["max_state_age_s"],  # type: ignore[arg-type]
            max_wrench_age_s=payload["max_wrench_age_s"],  # type: ignore[arg-type]
            max_state_wrench_skew_s=payload["max_state_wrench_skew_s"],  # type: ignore[arg-type]
            workspace_min_base_m=payload["workspace_min_base_m"],  # type: ignore[arg-type]
            workspace_max_base_m=payload["workspace_max_base_m"],  # type: ignore[arg-type]
            expected_robot_model=payload["expected_robot_model"],  # type: ignore[arg-type]
            expected_robot_serial_number=payload["expected_robot_serial_number"],  # type: ignore[arg-type]
            expected_controller_version=payload["expected_controller_version"],  # type: ignore[arg-type]
            reviewed_tool_name=payload["reviewed_tool_name"],  # type: ignore[arg-type]
            reviewed_workpiece_name=payload["reviewed_workpiece_name"],  # type: ignore[arg-type]
            reviewed_payload_mass_kg=payload["reviewed_payload_mass_kg"],  # type: ignore[arg-type]
            reviewed_payload_cog_m=payload["reviewed_payload_cog_m"],  # type: ignore[arg-type]
            reviewed_payload_inertia_kg_m2=payload["reviewed_payload_inertia_kg_m2"],  # type: ignore[arg-type]
            reviewed_joint_soft_limits_rad=payload["reviewed_joint_soft_limits_rad"],  # type: ignore[arg-type]
            reviewed_rt_filter_hz=payload["reviewed_rt_filter_hz"],  # type: ignore[arg-type]
            reviewed_rt_network_tolerance_percent=payload["reviewed_rt_network_tolerance_percent"],  # type: ignore[arg-type]
            robot_identity_reviewed=payload["robot_identity_reviewed"],  # type: ignore[arg-type]
            tool_workpiece_reviewed=payload["tool_workpiece_reviewed"],  # type: ignore[arg-type]
            payload_configuration_reviewed=payload["payload_configuration_reviewed"],  # type: ignore[arg-type]
            collision_configuration_reviewed=payload["collision_configuration_reviewed"],  # type: ignore[arg-type]
            joint_soft_limits_reviewed=payload["joint_soft_limits_reviewed"],  # type: ignore[arg-type]
            realtime_configuration_reviewed=payload["realtime_configuration_reviewed"],  # type: ignore[arg-type]
            reviewed=payload["reviewed"],  # type: ignore[arg-type]
            notes=payload["notes"],  # type: ignore[arg-type]
        )


def load_experiment_safety_config(path: str | Path) -> ExperimentSafetyConfig:
    return ExperimentSafetyConfig.load_json(path)


def save_experiment_safety_config(
    config: ExperimentSafetyConfig,
    path: str | Path,
) -> Path:
    if not isinstance(config, ExperimentSafetyConfig):
        raise TypeError("config must be an ExperimentSafetyConfig")
    return config.save_json(path)


def require_execute_safety(config: ExperimentSafetyConfig) -> None:
    if not isinstance(config, ExperimentSafetyConfig):
        raise TypeError("config must be an ExperimentSafetyConfig")
    config.require_execute_allowed()


__all__ = [
    "EXPERIMENT_SAFETY_SCHEMA_VERSION",
    "ExperimentSafetyConfig",
    "load_experiment_safety_config",
    "require_execute_safety",
    "save_experiment_safety_config",
]
