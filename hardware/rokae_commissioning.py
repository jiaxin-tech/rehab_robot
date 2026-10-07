"""Supervised commissioning controls for a ROKAE xCoreSDK robot.

This module is the supervised bench-commissioning layer: connectivity
self-test, status readout, explicit power/reset, bounded jog and an operator
stop.  It is intentionally separate from the frozen rehabilitation experiment
executor in ``control/robot_trajectory_executor.py``, which keeps its own
reviewed preflight gate and its own pinned slow trajectory.

Design rules enforced here:

* no command that can change servo power or move the robot is issued without
  an explicit operator confirmation string supplied by the caller;
* every motion command is bounded by a per-command step limit and by the
  robot's own live joint soft limits or the configured Cartesian workspace;
* a motion command is refused when the state stream is invalid/stale, when
  joint soft limits are unavailable, or when the controller collision query is
  unavailable or reports a collision;
* every attempted command is appended to an operator audit log.

This layer does not certify safety and does not replace the site risk
assessment, the vendor manual, a physical emergency stop, or a trained
operator standing at that emergency stop.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

COMMISSIONING_CONFIRMATION = "I CONFIRM SUPERVISED COMMISSIONING MOTION"

DEFAULT_MAX_JOINT_STEP_RAD = 0.05
DEFAULT_MAX_CARTESIAN_STEP_M = 0.01
DEFAULT_MAX_SPEED_RATIO = 10
DEFAULT_JOINT_SOFT_LIMIT_MARGIN_RAD = 0.05
DEFAULT_MAX_STATE_AGE_S = 0.5

_CONFIRMATION_ACTIONS = frozenset(
    {
        "power_on",
        "power_off",
        "reset",
        "jog_joint",
        "jog_cartesian",
        "hold_pose",
        "prepare_realtime",
        "end_realtime",
    }
)


class CommissioningError(RuntimeError):
    """Raised when a commissioning precondition is not satisfied."""


def requires_confirmation(action: str) -> bool:
    """True when ``action`` is power/motion and needs the exact text."""
    return action in _CONFIRMATION_ACTIONS


def check_confirmation(action: str, confirmation: str | None) -> None:
    """Refuse a power/motion action locally, before any robot connection.

    Raising for a bad confirmation before opening an SDK session means a typo
    cannot leave a half-open controller session behind, and the refusal is
    instant instead of waiting on a network timeout.
    """
    if not requires_confirmation(action):
        return
    if confirmation != COMMISSIONING_CONFIRMATION:
        raise CommissioningError(
            f"{action} requires --confirm with the exact text: {COMMISSIONING_CONFIRMATION}"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(values: Sequence[float], size: int, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of {size} finite numbers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of {size} finite numbers") from exc
    if len(items) != size:
        raise ValueError(f"{field_name} must contain exactly {size} values")
    out = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} must contain only numbers")
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        out.append(value)
    return tuple(out)


@dataclass(frozen=True)
class CommissioningLimits:
    """Per-command bounds for supervised commissioning."""

    max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD
    max_cartesian_step_m: float = DEFAULT_MAX_CARTESIAN_STEP_M
    max_speed_ratio: int = DEFAULT_MAX_SPEED_RATIO
    joint_soft_limit_margin_rad: float = DEFAULT_JOINT_SOFT_LIMIT_MARGIN_RAD
    workspace_min_base_m: tuple[float, float, float] | None = None
    workspace_max_base_m: tuple[float, float, float] | None = None
    max_state_age_s: float = DEFAULT_MAX_STATE_AGE_S
    max_network_tolerance_percent: int = 20
    require_collision_query: bool = True
    require_joint_soft_limits: bool = True

    def __post_init__(self) -> None:
        for field_name in ("max_joint_step_rad", "max_cartesian_step_m", "max_state_age_s"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a positive number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{field_name} must be a positive number")
        if isinstance(self.max_speed_ratio, bool) or not isinstance(self.max_speed_ratio, int):
            raise ValueError("max_speed_ratio must be an integer percentage")
        if not 1 <= self.max_speed_ratio <= 100:
            raise ValueError("max_speed_ratio must be between 1 and 100")
        if isinstance(self.max_network_tolerance_percent, bool) or not isinstance(
            self.max_network_tolerance_percent, int
        ):
            raise ValueError("max_network_tolerance_percent must be an integer percentage")
        if not 0 <= self.max_network_tolerance_percent <= 100:
            raise ValueError("max_network_tolerance_percent must be between 0 and 100")
        margin = self.joint_soft_limit_margin_rad
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError("joint_soft_limit_margin_rad must be a non-negative number")
        if not math.isfinite(float(margin)) or float(margin) < 0.0:
            raise ValueError("joint_soft_limit_margin_rad must be a non-negative number")
        lower = self.workspace_min_base_m
        upper = self.workspace_max_base_m
        if (lower is None) != (upper is None):
            raise ValueError("workspace_min_base_m and workspace_max_base_m must both be set or both null")
        if lower is not None and upper is not None:
            low = _finite(lower, 3, "workspace_min_base_m")
            high = _finite(upper, 3, "workspace_max_base_m")
            if any(a >= b for a, b in zip(low, high)):
                raise ValueError("workspace_min_base_m must be strictly below workspace_max_base_m")
            object.__setattr__(self, "workspace_min_base_m", low)
            object.__setattr__(self, "workspace_max_base_m", high)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommandAuditLog:
    """Append-only JSONL record of every attempted commissioning action."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).resolve() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def append(self, **payload: Any) -> dict[str, Any]:
        record = {"timestamp_utc": _utc_now_iso(), "monotonic_s": time.monotonic()}
        record.update(payload)
        self._records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
                handle.write("\n")
        return record

class RokaeCommissioningConsole:
    """Supervised connect/status/power/jog/stop console over a ROKAE adapter."""

    def __init__(
        self,
        adapter: Any,
        *,
        limits: CommissioningLimits | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        native = getattr(adapter, "native_robot", None)
        if native is None:
            raise TypeError("commissioning console requires an adapter exposing native_robot")
        if not hasattr(native, "enable") or not hasattr(native, "stop"):
            raise TypeError(
                "commissioning console requires the native SDK wrapper with enable/stop support"
            )
        self.adapter = adapter
        self.native = native
        self.limits = limits or CommissioningLimits()
        self.audit = CommandAuditLog(audit_path)
        self._connected = False
        self._powered = False
        self._realtime_prepared = False
        self._state_stream_started = False

    # ---------------------------------------------------------------- session

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def powered(self) -> bool:
        return self._powered

    @property
    def realtime_prepared(self) -> bool:
        return self._realtime_prepared

    def _require_confirmation(self, action: str, confirmation: str | None) -> None:
        check_confirmation(action, confirmation)

    def connect(self) -> dict[str, Any]:
        """Connect and start the observation state stream. No power or motion."""
        if self._connected:
            return {"connected": True, "already_connected": True}
        self.adapter.connect()
        self._connected = True
        try:
            self.adapter.start_state_stream()
            self._state_stream_started = True
        except Exception:
            self._safe_disconnect()
            raise
        self.audit.append(action="connect", ok=True)
        return {"connected": True, "already_connected": False}

    def _safe_disconnect(self) -> list[str]:
        errors: list[str] = []
        if self._state_stream_started:
            try:
                self.adapter.stop_state_stream()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                errors.append(f"stop_state_stream:{type(exc).__name__}:{exc}")
            self._state_stream_started = False
        if self._connected:
            try:
                self.adapter.disconnect()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                errors.append(f"disconnect:{type(exc).__name__}:{exc}")
        self._connected = False
        return errors

    def disconnect(self) -> dict[str, Any]:
        """Stop realtime motion if active, power down, then disconnect."""
        warnings: list[str] = []
        if self._realtime_prepared:
            try:
                self.native.stop_realtime(switch_to_nrt=True)
            except Exception as exc:
                warnings.append(f"stop_realtime:{type(exc).__name__}:{exc}")
            self._realtime_prepared = False
        if self._powered:
            try:
                self.stop_motion("disconnect")
            except Exception as exc:
                warnings.append(f"stop_motion:{type(exc).__name__}:{exc}")
            try:
                self.native.disable()
            except Exception as exc:
                warnings.append(f"disable:{type(exc).__name__}:{exc}")
            self._powered = False
        errors = self._safe_disconnect()
        self.audit.append(action="disconnect", ok=not errors, warnings=warnings, errors=errors)
        return {"disconnected": not self._connected, "warnings": warnings, "errors": errors}

    # ----------------------------------------------------------------- status

    def _live_summary(self) -> dict[str, Any]:
        return dict(self.adapter.get_robot_state_summary())

    def _require_healthy_state(self, summary: Mapping[str, Any]) -> None:
        if not summary.get("connected"):
            raise CommissioningError("adapter reports not connected")
        if not summary.get("state_stream_thread_alive"):
            raise CommissioningError("state stream thread is not alive")
        if not summary.get("state_valid"):
            raise CommissioningError(
                "robot state is invalid: " + str(summary.get("state_invalid_reason") or "unknown")
            )
        age_ms = summary.get("state_age_ms")
        if age_ms is None or float(age_ms) > self.limits.max_state_age_s * 1000.0:
            raise CommissioningError(
                f"robot state is stale: age_ms={age_ms!r}, limit_s={self.limits.max_state_age_s}"
            )
        if self.limits.require_collision_query:
            if not summary.get("collision_state_query_valid"):
                raise CommissioningError(
                    "controller collision query is unavailable: "
                    + str(summary.get("collision_state_invalid_reason") or "unknown")
                )
            if summary.get("collision_state"):
                raise CommissioningError("controller reports an active collision state")
        if self.limits.require_joint_soft_limits and not summary.get("joint_soft_limits_valid"):
            raise CommissioningError("joint soft limits are unavailable; refusing motion")

    def status(self) -> dict[str, Any]:
        """Read-only status snapshot; safe to call at any time after connect."""
        summary = self._live_summary()
        summary["operator_powered"] = self._powered
        summary["operator_realtime_prepared"] = self._realtime_prepared
        summary["commissioning_limits"] = self.limits.to_dict()
        return summary

    def self_test(self) -> dict[str, Any]:
        """Connectivity self-test: read-only checks plus explicit cleanup result."""
        checks: dict[str, Any] = {}
        cleanup: dict[str, Any] = {}
        started = time.perf_counter()
        opened_here = False
        if not self._connected:
            self.connect()
            opened_here = True

        def run(name: str, method) -> Any:
            began = time.perf_counter_ns()
            try:
                value = method()
            except Exception as exc:
                checks[name] = {
                    "ok": False,
                    "duration_ms": (time.perf_counter_ns() - began) / 1e6,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                return None
            checks[name] = {
                "ok": True,
                "duration_ms": (time.perf_counter_ns() - began) / 1e6,
                "value": value if isinstance(value, (str, int, float, bool, list, dict)) else str(value),
            }
            return value

        try:
            run("sdk_metadata", lambda: self.native.get_robot_metadata())
            summary = run("robot_state_summary", lambda: self.adapter.get_robot_state_summary())
            run("tcp_pose", lambda: list(self.adapter.read_tcp_pose()))
            run("joint_positions", lambda: list(self.adapter.read_joint_positions()))
            wrench = run("internal_wrench", lambda: self.adapter.read_internal_wrench())
            if isinstance(wrench, object) and hasattr(wrench, "valid"):
                checks["internal_wrench"]["semantic_valid"] = bool(wrench.valid)
            if isinstance(summary, Mapping):
                checks["health"] = {
                    "ok": bool(
                        summary.get("state_valid")
                        and summary.get("state_stream_thread_alive")
                        and summary.get("collision_state_query_valid")
                        and summary.get("joint_soft_limits_valid")
                    ),
                    "state_valid": summary.get("state_valid"),
                    "state_age_ms": summary.get("state_age_ms"),
                    "collision_state": summary.get("collision_state"),
                    "collision_query_valid": summary.get("collision_state_query_valid"),
                    "joint_soft_limits_valid": summary.get("joint_soft_limits_valid"),
                    "operation_state": summary.get("operation_state"),
                }
        finally:
            if opened_here:
                errors = self._safe_disconnect()
                cleanup["disconnect"] = {"ok": not self._connected, "errors": list(errors)}
            else:
                errors = []
                cleanup["disconnect"] = {
                    "ok": True,
                    "skipped": "session was already open before self_test",
                }
            cleanup["opened_by_self_test"] = opened_here
            cleanup["errors"] = errors
            cleanup["ok"] = not errors
        result = {
            "schema_version": 1,
            "self_test": "rokae_commissioning_connectivity",
            "robot_ip": getattr(self.native, "ip_address", None),
            "connected_during_test": True,
            "duration_ms": (time.perf_counter() - started) * 1e3,
            "checks": checks,
            "cleanup": cleanup,
            "motion_commanded": False,
            "power_commanded": False,
        }
        result["success"] = bool(checks) and all(
            item.get("ok") for item in checks.values()
        ) and cleanup["ok"]
        self.audit.append(action="self_test", ok=result["success"])
        return result

    # ------------------------------------------------------------------ power

    def power_on(self, *, confirmation: str | None = None, speed_ratio: int | None = None) -> dict[str, Any]:
        self._require_confirmation("power_on", confirmation)
        if not self._connected:
            raise CommissioningError("connect before power_on")
        summary = self._live_summary()
        self._require_healthy_state(summary)
        ratio = self.limits.max_speed_ratio if speed_ratio is None else int(speed_ratio)
        if not 1 <= ratio <= self.limits.max_speed_ratio:
            raise CommissioningError(
                f"speed_ratio must be between 1 and {self.limits.max_speed_ratio}"
            )
        self.native.enable(load=0.0)
        self.native.set_speed(ratio)
        self._powered = True
        self.audit.append(action="power_on", speed_ratio=ratio, ok=True)
        return {"powered": True, "speed_ratio": ratio, "mode": self.native.get_robot_mode()}

    def power_off(self, *, confirmation: str | None = None) -> dict[str, Any]:
        self._require_confirmation("power_off", confirmation)
        if not self._connected:
            raise CommissioningError("connect before power_off")
        try:
            self.native.stop()
        except Exception:
            pass
        self.native.disable()
        self._powered = False
        self.audit.append(action="power_off", ok=True)
        return {"powered": False}

    def reset(self, *, confirmation: str | None = None) -> dict[str, Any]:
        self._require_confirmation("reset", confirmation)
        if not self._connected:
            raise CommissioningError("connect before reset")
        self.native.clear_error()
        self.native.reset()
        self.audit.append(action="reset", ok=True)
        return {"reset": True, "mode": self.native.get_robot_mode()}

    def _issue_and_confirm(
        self,
        action: str,
        issue_command,
        *,
        context: dict[str, Any],
        after=None,
    ) -> None:
        """Issue one already-validated motion command and audit the real outcome.

        The audit entry is written whether the SDK call fails, the robot never
        confirms idle, or the command completes.  A failed wait is re-raised so
        the operator sees the motion still in flight instead of a false success.
        """
        issued = False
        idle_ok = False
        failure: str | None = None
        try:
            issue_command()
            issued = True
            idle_ok = bool(self.native.wait_idle(timeout=30.0))
            if not idle_ok:
                failure = "wait_idle returned false"
            elif after is not None:
                after()
        except Exception as exc:
            failure = f"{type(exc).__name__}:{exc}"
            raise
        finally:
            self.audit.append(
                action=action,
                command_issued=issued,
                ok=issued and idle_ok and failure is None,
                failure=failure,
                **context,
            )

    # --------------------------------------------------------------- realtime

    def prepare_realtime(
        self,
        *,
        confirmation: str | None = None,
        network_tolerance_percent: int | None = None,
    ) -> dict[str, Any]:
        """Select realtime command mode and power on, without starting motion.

        The frozen experiment executor attaches to an already prepared realtime
        controller and never changes mode or power itself.  This is the missing
        operator-side preparation step for that contract.  It selects the mode,
        enables servo power, and obtains the SDK realtime controller, but it
        does not start the 1 ms Cartesian loop and never sends a target.
        """
        self._require_confirmation("prepare_realtime", confirmation)
        if not self._connected:
            raise CommissioningError("connect before prepare_realtime")
        if self._powered:
            raise CommissioningError(
                "servo power is already on from this console; power_off before "
                "switching to realtime command mode"
            )
        if not getattr(self.native, "local_ip", ""):
            raise CommissioningError(
                "realtime control needs this PC's address on the robot subnet; "
                "pass --local-ip (xCoreSDK uses a second network channel)"
            )
        summary = self._live_summary()
        self._require_healthy_state(summary)
        tolerance = (
            self.limits.max_network_tolerance_percent
            if network_tolerance_percent is None
            else int(network_tolerance_percent)
        )
        if not 0 <= tolerance <= self.limits.max_network_tolerance_percent:
            raise CommissioningError(
                "network_tolerance_percent must be between 0 and "
                f"{self.limits.max_network_tolerance_percent}"
            )
        self.native.rt_network_tolerance_percent = tolerance
        self.native.enable_realtime(load=0.0)
        self._powered = True
        self._realtime_prepared = True
        state = self._realtime_state()
        self.audit.append(
            action="prepare_realtime",
            network_tolerance_percent=tolerance,
            ok=True,
            **state,
        )
        return {
            "realtime_prepared": True,
            "network_tolerance_percent": tolerance,
            "motion_loop_started": False,
            **state,
        }

    def end_realtime(self, *, confirmation: str | None = None) -> dict[str, Any]:
        """Return to non-realtime command mode and power down the servos."""
        self._require_confirmation("end_realtime", confirmation)
        if not self._connected:
            raise CommissioningError("connect before end_realtime")
        errors: list[str] = []
        try:
            self.native.stop()
        except Exception as exc:
            errors.append(f"stop:{type(exc).__name__}:{exc}")
        try:
            self.native.stop_realtime(switch_to_nrt=True)
        except Exception as exc:
            errors.append(f"stop_realtime:{type(exc).__name__}:{exc}")
        try:
            self.native.disable()
        except Exception as exc:
            errors.append(f"disable:{type(exc).__name__}:{exc}")
        self._powered = False
        self._realtime_prepared = False
        state = self._realtime_state()
        self.audit.append(action="end_realtime", ok=not errors, errors=errors, **state)
        return {"realtime_prepared": False, "errors": errors, **state}

    def _realtime_state(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        reader = getattr(self.native, "_realtime_prepared_state", None)
        if callable(reader):
            try:
                result.update(reader())
            except Exception as exc:  # pragma: no cover - readback best effort
                result["realtime_readback_error"] = f"{type(exc).__name__}:{exc}"
        return result

    # ----------------------------------------------------------------- motion

    def stop_motion(self, reason: str = "operator_stop") -> dict[str, Any]:
        """Operator stop. Never gated by confirmation and always audited."""
        errors: list[str] = []
        try:
            self.native.stop()
        except Exception as exc:
            errors.append(f"stop:{type(exc).__name__}:{exc}")
        if hasattr(self.native, "stop_realtime"):
            try:
                self.native.stop_realtime(switch_to_nrt=True)
                # A confirmed realtime stop returns the controller to
                # non-realtime mode, so the prepared-realtime state is over.
                self._realtime_prepared = False
            except Exception as exc:
                errors.append(f"stop_realtime:{type(exc).__name__}:{exc}")
        self.audit.append(action="stop_motion", reason=reason, ok=not errors, errors=errors)
        return {"stopped": not errors, "reason": reason, "errors": errors}

    def _joint_soft_limits(self, summary: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
        metadata = summary.get("robot_metadata")
        limits = metadata.get("joint_soft_limits_rad") if isinstance(metadata, Mapping) else None
        if not limits:
            raise CommissioningError("joint soft limits are unavailable; refusing joint motion")
        parsed: list[tuple[float, float]] = []
        for pair in limits:
            parsed.append((float(pair[0]), float(pair[1])))
        if len(parsed) != 6:
            raise CommissioningError("joint soft limits must contain six pairs")
        return tuple(parsed)

    def jog_joint(
        self,
        joint_index: int,
        delta_rad: float,
        *,
        confirmation: str | None = None,
    ) -> dict[str, Any]:
        """Relative joint jog of one axis, bounded by step and live soft limits."""
        self._require_confirmation("jog_joint", confirmation)
        if not self._connected or not self._powered:
            raise CommissioningError("power_on before jog_joint")
        if isinstance(joint_index, bool) or not isinstance(joint_index, int) or not 0 <= joint_index <= 5:
            raise CommissioningError("joint_index must be an integer 0..5")
        delta = float(delta_rad)
        if not math.isfinite(delta) or delta == 0.0:
            raise CommissioningError("delta_rad must be a nonzero finite number")
        if abs(delta) > self.limits.max_joint_step_rad + 1e-12:
            raise CommissioningError(
                f"delta_rad {delta} exceeds per-command limit {self.limits.max_joint_step_rad}"
            )
        summary = self._live_summary()
        self._require_healthy_state(summary)
        current = _finite(summary.get("joint_position_rad") or (), 6, "joint_position_rad")
        limits = self._joint_soft_limits(summary)
        target = list(current)
        target[joint_index] += delta
        low, high = limits[joint_index]
        margin = self.limits.joint_soft_limit_margin_rad
        if not (low + margin <= target[joint_index] <= high - margin):
            raise CommissioningError(
                f"target joint {joint_index} = {target[joint_index]:.6f} rad is outside "
                f"soft limits [{low}, {high}] with margin {margin}"
            )
        self._issue_and_confirm(
            "jog_joint",
            lambda: self.native.move_j(target),
            context={
                "joint_index": joint_index,
                "delta_rad": delta,
                "target_rad": list(target),
                "from_rad": list(current),
            },
        )
        return {
            "joint_index": joint_index,
            "from_rad": list(current),
            "target_rad": list(target),
            "delta_rad": delta,
        }

    def jog_cartesian(
        self,
        delta_xyz_m: Sequence[float],
        *,
        confirmation: str | None = None,
    ) -> dict[str, Any]:
        """Relative Cartesian jog in the robot base frame, orientation held."""
        self._require_confirmation("jog_cartesian", confirmation)
        if not self._connected or not self._powered:
            raise CommissioningError("power_on before jog_cartesian")
        delta = _finite(delta_xyz_m, 3, "delta_xyz_m")
        if all(component == 0.0 for component in delta):
            raise CommissioningError("delta_xyz_m must not be all zero")
        if any(abs(component) > self.limits.max_cartesian_step_m + 1e-12 for component in delta):
            raise CommissioningError(
                f"each delta_xyz_m component must not exceed {self.limits.max_cartesian_step_m} m"
            )
        summary = self._live_summary()
        self._require_healthy_state(summary)
        pose = list(_finite(summary.get("tcp_pose_base_m_rad") or (), 6, "tcp_pose_base_m_rad"))
        target = list(pose)
        for index, component in enumerate(delta):
            target[index] += component
        if self.limits.workspace_min_base_m is not None and self.limits.workspace_max_base_m is not None:
            low = self.limits.workspace_min_base_m
            high = self.limits.workspace_max_base_m
            for index in range(3):
                if not (low[index] <= target[index] <= high[index]):
                    raise CommissioningError(
                        f"target axis {index} = {target[index]:.6f} m is outside the reviewed "
                        f"workspace [{low[index]}, {high[index]}]"
                    )
        self._issue_and_confirm(
            "jog_cartesian",
            lambda: self.native.move_l(target),
            context={
                "delta_xyz_m": list(delta),
                "from_pose": list(pose),
                "target_pose": list(target),
            },
        )
        return {
            "from_pose_base_m_rad": list(pose),
            "target_pose_base_m_rad": list(target),
            "delta_xyz_m": list(delta),
        }

    def hold_pose(self, *, confirmation: str | None = None, seconds: float = 1.0) -> dict[str, Any]:
        """Command the current pose as the target, useful as a motion smoke test."""
        self._require_confirmation("hold_pose", confirmation)
        if not self._connected or not self._powered:
            raise CommissioningError("power_on before hold_pose")
        hold_s = float(seconds)
        if not math.isfinite(hold_s) or not 0.0 < hold_s <= 10.0:
            raise CommissioningError("seconds must be in (0, 10]")
        summary = self._live_summary()
        self._require_healthy_state(summary)
        pose = list(_finite(summary.get("tcp_pose_base_m_rad") or (), 6, "tcp_pose_base_m_rad"))
        self._issue_and_confirm(
            "hold_pose",
            lambda: self.native.move_l(pose),
            context={"pose": list(pose), "seconds": hold_s},
            after=lambda: time.sleep(hold_s),
        )
        return {"held_pose_base_m_rad": pose, "seconds": hold_s}

__all__ = [
    "COMMISSIONING_CONFIRMATION",
    "CommandAuditLog",
    "CommissioningError",
    "CommissioningLimits",
    "RokaeCommissioningConsole",
    "check_confirmation",
    "requires_confirmation",
]