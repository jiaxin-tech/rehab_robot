"""Build an audited, non-executable temporary commissioning trajectory.

The source geometry is the frozen formal reference.  This module may only
retime that geometry to an explicit Cartesian speed cap; it does not authorize
robot motion and it never changes the frozen reference artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from config.bundle import resolve_config_path
from typing import Any, Mapping

import numpy as np
import pandas as pd


MODE = "FORMAL_REFERENCE_TEMPORARY_COMMISSIONING"
SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _aware_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReferenceCommissioningPolicy:
    config_path: Path
    trajectory_id: str
    reference_path: Path
    reference_sha256: str
    controller_speed_ratio_percent: int
    max_tcp_speed_m_s: float
    expires_at: str
    allow_missing_sdk_collision_state: bool
    require_controller_collision_protection: bool
    require_operator_at_emergency_stop: bool
    cycles_per_session: int

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= _aware_time(self.expires_at, "expires_at")

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "config_path": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "trajectory_id": self.trajectory_id,
            "reference_path": str(self.reference_path),
            "reference_sha256": self.reference_sha256,
            "controller_speed_ratio_percent": self.controller_speed_ratio_percent,
            "max_tcp_speed_m_s": self.max_tcp_speed_m_s,
            "expires_at": self.expires_at,
            "allow_missing_sdk_collision_state": self.allow_missing_sdk_collision_state,
            "require_controller_collision_protection": self.require_controller_collision_protection,
            "require_operator_at_emergency_stop": self.require_operator_at_emergency_stop,
            "cycles_per_session": self.cycles_per_session,
            "capture_only": True,
            "training_allowed": False,
            "patient_present_allowed": False,
        }


def load_reference_commissioning_policy(path: str | Path) -> ReferenceCommissioningPolicy:
    config_path = resolve_config_path(path, section="reference_commissioning")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "mode", "enabled", "execution_enabled", "trajectory",
        "speed", "scope", "safety", "expires_at", "note",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("reference commissioning configuration schema mismatch")
    if payload["schema_version"] != SCHEMA_VERSION or payload["mode"] != MODE:
        raise ValueError("unsupported reference commissioning mode")
    if payload["enabled"] is not True:
        raise PermissionError("reference commissioning mode is disabled")
    # Deliberate two-key rule: producing an artifact is allowed, but this file
    # can never turn motion on by itself.
    if payload["execution_enabled"] is not False:
        raise PermissionError("commissioning policy must not self-authorize execution")

    trajectory = payload["trajectory"]
    if not isinstance(trajectory, dict) or set(trajectory) != {"id", "path", "sha256"}:
        raise ValueError("trajectory binding schema mismatch")
    reference = (config_path.parent / trajectory["path"]).resolve()
    if not reference.is_file() or sha256_file(reference) != trajectory["sha256"]:
        raise PermissionError("frozen formal reference is missing or changed")

    speed = payload["speed"]
    if not isinstance(speed, dict) or set(speed) != {
        "controller_speed_ratio_percent", "max_tcp_speed_m_s"
    }:
        raise ValueError("commissioning speed schema mismatch")
    ratio = speed["controller_speed_ratio_percent"]
    cap = speed["max_tcp_speed_m_s"]
    if type(ratio) is not int or ratio != 3:
        raise ValueError("temporary commissioning controller speed ratio must be exactly 3%")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap):
        raise ValueError("max_tcp_speed_m_s must be finite")
    if not 0.0 < float(cap) <= 0.03:
        raise ValueError("temporary commissioning TCP speed cap must be at most 0.03 m/s")

    scope = payload["scope"]
    expected_scope = {
        "dummy_leg_only": True,
        "patient_present_allowed": False,
        "capture_only": True,
        "training_allowed": False,
        "cycles_per_session": 1,
    }
    if scope != expected_scope:
        raise PermissionError("temporary commissioning scope must be dummy-leg, one-cycle, capture-only")

    safety = payload["safety"]
    expected_safety_keys = {
        "allow_missing_sdk_collision_state",
        "require_controller_collision_protection",
        "require_operator_at_emergency_stop",
        "fail_stop_on_state_or_wrench_fault",
    }
    if not isinstance(safety, dict) or set(safety) != expected_safety_keys:
        raise ValueError("temporary commissioning safety schema mismatch")
    if safety["allow_missing_sdk_collision_state"] is not True:
        raise ValueError("this policy is only for the documented unavailable SDK collision state")
    for key in (
        "require_controller_collision_protection",
        "require_operator_at_emergency_stop",
        "fail_stop_on_state_or_wrench_fault",
    ):
        if safety[key] is not True:
            raise PermissionError(f"temporary commissioning requires {key}=true")
    if _aware_time(payload["expires_at"], "expires_at") <= datetime.now(timezone.utc):
        raise PermissionError("temporary commissioning policy is expired")
    if not isinstance(payload["note"], str):
        raise ValueError("note must be a string")

    return ReferenceCommissioningPolicy(
        config_path=config_path,
        trajectory_id=str(trajectory["id"]),
        reference_path=reference,
        reference_sha256=str(trajectory["sha256"]),
        controller_speed_ratio_percent=ratio,
        max_tcp_speed_m_s=float(cap),
        expires_at=str(payload["expires_at"]),
        allow_missing_sdk_collision_state=True,
        require_controller_collision_protection=True,
        require_operator_at_emergency_stop=True,
        cycles_per_session=1,
    )


def retime_formal_reference(
    reference: pd.DataFrame,
    *,
    max_tcp_speed_m_s: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Uniformly slow a reference without changing any geometric sample."""
    required = {"time_s", "x_pull_m", "z_pull_m", "trajectory_id"}
    missing = required.difference(reference.columns)
    if missing or len(reference) < 3:
        raise ValueError("formal reference is incomplete: " + ",".join(sorted(missing)))
    result = reference.copy(deep=True)
    time_s = result["time_s"].to_numpy(dtype=float)
    pull = result[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
    if not np.isfinite(time_s).all() or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("formal reference time is invalid")
    if not np.isfinite(pull).all():
        raise ValueError("formal reference pull path is invalid")
    speed = np.linalg.norm(np.gradient(pull, time_s, axis=0, edge_order=2), axis=1)
    source_max = float(speed.max())
    scale = max(1.0, source_max / float(max_tcp_speed_m_s))
    result["time_s"] = time_s[0] + (time_s - time_s[0]) * scale

    velocity_columns = [
        "dq_hip_rad_s", "dq_knee_rad_s", "minimum_jerk_phase_rate_s_inv"
    ]
    acceleration_columns = [
        "ddq_hip_rad_s2", "ddq_knee_rad_s2",
        "minimum_jerk_phase_acceleration_s_inv2",
    ]
    for name in velocity_columns:
        if name in result:
            result[name] = result[name].astype(float) / scale
    for name in acceleration_columns:
        if name in result:
            result[name] = result[name].astype(float) / (scale * scale)
    if "retimed_timing_is_original" in result:
        result["retimed_timing_is_original"] = False

    retimed_time = result["time_s"].to_numpy(dtype=float)
    retimed_speed = np.linalg.norm(
        np.gradient(pull, retimed_time, axis=0, edge_order=2), axis=1
    )
    actual_max = float(retimed_speed.max())
    if actual_max > max_tcp_speed_m_s + 1e-12:
        raise RuntimeError("retimed trajectory exceeds commissioning speed cap")
    return result, {
        "source_duration_s": float(time_s[-1] - time_s[0]),
        "retimed_duration_s": float(retimed_time[-1] - retimed_time[0]),
        "uniform_time_scale": float(scale),
        "source_max_pull_speed_m_s": source_max,
        "retimed_max_pull_speed_m_s": actual_max,
    }


def build_reference_commissioning_artifact(
    policy: ReferenceCommissioningPolicy,
    output_dir: str | Path,
    *,
    blockers: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = pd.read_csv(policy.reference_path)
    ids = source["trajectory_id"].astype(str).unique().tolist()
    if ids != [policy.trajectory_id]:
        raise PermissionError("formal reference trajectory id does not match policy")
    retimed, timing = retime_formal_reference(
        source,
        max_tcp_speed_m_s=policy.max_tcp_speed_m_s,
    )
    csv_path = destination / "commissioning_reference.csv"
    retimed.to_csv(csv_path, index=False)
    reasons = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PREPARED_NOT_AUTHORIZED" if reasons else "PREPARED_AWAITING_SEPARATE_MOTION_RELEASE",
        "motion_authorized": False,
        "reason": (
            "repository motion gates remain unresolved"
            if reasons
            else "artifact preparation never grants robot motion"
        ),
        "policy": policy.metadata(),
        "geometry_unchanged": True,
        "timing": timing,
        "artifact": {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
            "rows": int(len(retimed)),
        },
        "remaining_motion_blockers": reasons,
        "collision_waiver_scope": (
            "SDK collision-state observability only; controller collision protection "
            "and the operator-held emergency stop remain mandatory"
        ),
    }
    manifest_path = destination / "commissioning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "MODE",
    "ReferenceCommissioningPolicy",
    "build_reference_commissioning_artifact",
    "load_reference_commissioning_policy",
    "retime_formal_reference",
    "sha256_file",
]
