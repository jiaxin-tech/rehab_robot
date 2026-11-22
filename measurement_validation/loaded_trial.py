"""Review and freeze a small passive dummy-leg diagnostic; no hardware access.

An empty-motion completion is provenance only, never a force-quality approval.
The existing continuous taught curve and runtime guards remain authoritative.
"""
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from collection.twincat_leg import atomic_json
from config.bundle import ROOT
from control.continuous_taught_trajectory import _load_geometry, _settings

PROTOCOL = "passive_dummy_leg_diagnostic_v1"
FIELDS = {"protocol", "setup_id", "connection_type", "tool_description",
          "support_description", "reviewed", "manual_path_checked",
          "reviewed_anchor_pose_base", "reviewed_geometry_sha256",
          "reviewed_scale_xyz", "review_notes", "reviewed_at",
          "strict_wrench_freshness", "source_empty_session",
          "source_empty_status_sha256"}
SOURCE_FILES = ("round_status.json", "tcp_plan.json", "prepared_config.json", "dispatch.csv")
SOURCE_CODE = ("measurement_validation/loaded_trial.py",
               "control/continuous_taught_trajectory.py",
               "measurement_validation/robot_execution.py",
               "measurement_validation/loaded_diagnostic.py",
               "collection/twincat_leg.py", "scripts/lab_session.py",
               "collection/leg_capture_diagnostics.py", "collection/leg_progress_publisher.py",
               "scripts/run_rehab_experiment.py", "control/robot_trajectory_executor.py")


def active(c):
    return isinstance(c, dict) and isinstance(c.get("loaded_trial"), dict) and c["loaded_trial"].get("protocol") == PROTOCOL


def _path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("loaded diagnostic requires a nonempty source path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"missing loaded diagnostic evidence: {path}") from exc


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid loaded diagnostic evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"loaded diagnostic evidence must be an object: {path}")
    return value


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("loaded diagnostic requires finite JSON values") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the loaded diagnostic limit {maximum}")
    return float(value)


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size
        or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} requires {size} finite numbers")
    return np.asarray(value, float)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} requires a lowercase SHA-256 digest")


def _spec(c):
    trajectory = c.get("trajectory", {})
    if not isinstance(trajectory, dict) or trajectory.get("mode") != "continuous_taught_tcp":
        raise ValueError("loaded diagnostic requires continuous_taught_tcp")
    spec = _settings({k: v for k, v in trajectory.items() if k not in ("mode", "fixed_cycle_duration_s")})
    for key, required in (("cycle_duration_s", 90.), ("ramp_duration_s", 5.), ("sample_period_s", .02)):
        if spec[key] != required:
            raise ValueError(f"loaded diagnostic requires {key}={required}")
    fixed = trajectory.get("fixed_cycle_duration_s")
    if isinstance(fixed, bool) or not isinstance(fixed, Real) or fixed != 90.:
        raise ValueError("loaded diagnostic requires fixed_cycle_duration_s=90")
    for axis in "xyz":
        _number(spec[f"scale_{axis}"], f"scale_{axis}", .03)
    try:
        _load_geometry(spec)  # Actual file SHA, finite geometry and curve validation.
    except OSError as exc:
        raise ValueError("loaded diagnostic geometry file is unavailable") from exc
    return spec


def _source(c, spec):
    protocol = c["loaded_trial"]
    source = _path(protocol["source_empty_session"])
    hashes = {name: _hash(source / name) for name in SOURCE_FILES}
    if hashes["round_status.json"] != protocol["source_empty_status_sha256"]:
        raise ValueError("source empty status SHA-256 changed")
    state, plan, prepared = (_read(source / name) for name in SOURCE_FILES[:3])
    execution = state.get("execution", {})
    if (state.get("status") not in ("EMPTY_LOAD_COMPLETE", "COMPLETE")
        or state.get("empty_load") is not True
        or state.get("trajectory_mode") != "continuous_taught_tcp"
        or execution.get("completed") is not True
        or execution.get("same_session_realtime_cleanup_completed") is not True
        or type(execution.get("commands_dispatched")) is not int
        or execution["commands_dispatched"] != 4501):
        raise ValueError("source must be a completed 90-second empty motion with realtime cleanup")
    source_spec = {k: v for k, v in prepared.get("trajectory", {}).items() if k not in ("mode", "fixed_cycle_duration_s")}
    if (plan != state.get("tcp_plan") or plan.get("mode") != "continuous_taught_tcp"
        or prepared.get("trajectory", {}).get("mode") != "continuous_taught_tcp"
        or plan.get("tcp_settings") != source_spec
        or plan.get("geometry_sha256") != spec["geometry_sha256"]
        or source_spec.get("geometry_sha256") != spec["geometry_sha256"]
        or any(source_spec.get(key) != spec[key] for key in ("cycle_duration_s", "ramp_duration_s", "sample_period_s"))
        or plan.get("duration_s") != 90.):
        raise ValueError("source empty plan/configuration/geometry mismatch")
    # Existing physical and freshness limits can only be retained or tightened.
    for key, value in prepared.get("motion", {}).items():
        if key.startswith("max_") and isinstance(value, Real) and not isinstance(value, bool):
            _number(c.get("motion", {}).get(key), key, value)
    for key, direction in (("workspace_min_base_m", 1), ("workspace_max_base_m", -1)):
        old = _vector(prepared.get("motion", {}).get(key), 3, "source " + key)
        new = _vector(c.get("motion", {}).get(key), 3, key)
        if np.any(direction * (new - old) < 0):
            raise ValueError("loaded diagnostic cannot expand the source workspace")
    return hashes


def validate_config(c, *, require_review=True):
    if not active(c) or set(c["loaded_trial"]) != FIELDS:
        raise ValueError("loaded_trial requires the exact passive diagnostic protocol fields")
    p = c["loaded_trial"]
    for key in ("reviewed", "manual_path_checked", "strict_wrench_freshness"):
        if type(p[key]) is not bool:
            raise ValueError(f"loaded_trial.{key} must be a boolean")
    if p["strict_wrench_freshness"] is not True:
        raise ValueError("loaded diagnostic requires strict wrench freshness")
    if require_review and p["connection_type"] == "rigid_fixture":
        raise ValueError("rigid shank connection requires a coupled position/orientation trajectory; fixed-orientation path not supported")
    if c.get("training", {}).get("enabled") is not False or c.get("training", {}).get("mode") != "disabled":
        raise ValueError("loaded diagnostic requires training disabled")
    if c.get("validation", {}).get("capture_dummy_leg") is not True:
        raise ValueError("loaded diagnostic requires read-only dummy-leg capture")
    spec = _spec(c)
    _sha(p["reviewed_geometry_sha256"], "reviewed_geometry_sha256")
    _sha(p["source_empty_status_sha256"], "source_empty_status_sha256")
    if p["reviewed_geometry_sha256"] != spec["geometry_sha256"]:
        raise ValueError("reviewed geometry differs from the configured geometry")
    scales = _vector(p["reviewed_scale_xyz"], 3, "reviewed_scale_xyz")
    if not np.array_equal(scales, [spec[f"scale_{a}"] for a in "xyz"]):
        raise ValueError("reviewed scales differ from the configured scales")
    for section in ("motion", "empty_load"):
        for key, maximum in (("max_tcp_speed_m_s", .005), ("max_tcp_acceleration_m_s2", .01)):
            _number(c.get(section, {}).get(key), f"{section}.{key}", maximum)
    for key, maximum in (("max_force_n", 100.), ("max_torque_nm", 200.)):
        _number(c.get("motion", {}).get(key), key, maximum)
    for key in ("setup_id", "tool_description", "support_description", "review_notes"):
        if (p[key] is not None or require_review) and (not isinstance(p[key], str) or not p[key].strip()):
            raise ValueError(f"manual review requires nonempty {key}")
    if (p["connection_type"] is not None or require_review) and p["connection_type"] not in ("rope_or_strap", "passive_joint", "rigid_fixture"):
        raise ValueError("manual review requires a supported connection_type")
    if p["reviewed_anchor_pose_base"] is not None or require_review:
        _vector(p["reviewed_anchor_pose_base"], 6, "reviewed_anchor_pose_base")
    if p["reviewed_at"] is not None or require_review:
        try:
            stamp = datetime.fromisoformat(p["reviewed_at"])
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("timezone missing")
        except (TypeError, ValueError) as exc:
            raise ValueError("reviewed_at requires an ISO timestamp with timezone") from exc
    if require_review and (p["reviewed"] is not True or p["manual_path_checked"] is not True):
        raise ValueError("loaded diagnostic requires completed manual setup/path review")
    _source(c, spec)
    _canonical(c)


def _binding(c, plan):
    validate_config(c)
    spec = _spec(c)
    if (not isinstance(plan, dict) or plan.get("mode") != "continuous_taught_tcp"
        or plan.get("tcp_settings") != spec or plan.get("geometry_sha256") != spec["geometry_sha256"]
        or plan.get("duration_s") != 90.):
        raise ValueError("loaded diagnostic plan differs from the reviewed configuration")
    anchor = plan.get("anchor", {})
    actual = _vector(anchor.get("tcp_pose_base"), 6, "prepared anchor pose")
    orientation = anchor.get("tcp_orientation", {})
    if (not np.array_equal(_vector(anchor.get("tcp_position_base_m"), 3, "anchor position"), actual[:3])
        or orientation.get("strategy") != "fixed" or orientation.get("representation") != "euler_xyz_rad"
        or not np.array_equal(_vector(orientation.get("values_rad"), 3, "anchor orientation"), actual[3:])):
        raise ValueError("loaded diagnostic requires a consistent fixed-orientation anchor")
    reviewed = _vector(c["loaded_trial"]["reviewed_anchor_pose_base"], 6, "reviewed_anchor_pose_base")
    angle = (Rotation.from_euler("xyz", reviewed[3:]).inv() * Rotation.from_euler("xyz", actual[3:])).magnitude()
    if (np.linalg.norm(actual[:3] - reviewed[:3]) > c["motion"]["max_start_anchor_position_error_m"]
        or angle > c["motion"]["max_start_anchor_orientation_error_rad"]):
        raise ValueError("prepared anchor is outside the manually reviewed pose tolerance")
    return {"schema_version": 1, "status": "BOUND", "protocol": PROTOCOL,
            "protocol_fields": json.loads(_canonical(c["loaded_trial"])),
            "config_sha256": _digest(c), "plan_sha256": _digest(plan),
            "source_empty_sha256": _source(c, spec),
            "geometry_sha256": _hash(_path(spec["geometry_path"])),
            "source_code_sha256": {name: _hash(ROOT / name) for name in SOURCE_CODE},
            "source_empty_motion_completed": True,
            "source_empty_quality": "not asserted; source completion proves motion only"}


def freeze_binding(directory, c, plan):
    binding = _binding(c, plan)
    directory = Path(directory)
    origin, digest = _leg_origin(directory)
    binding["leg_start_origin_sha256"] = digest
    atomic_json(directory / "loaded_trial_binding.json", binding)
    return binding


def _leg_origin(directory):
    path = Path(directory) / "leg_start_origin.json"
    origin = _read(path)
    if origin.get("usable_for_motion") is not True:
        raise ValueError("loaded diagnostic requires a usable captured leg start origin")
    axes = origin.get("axes", {})
    for name in ("hip", "knee"):
        if not isinstance(axes.get(name), dict):
            raise ValueError("loaded diagnostic requires captured hip/knee start angles")
        _vector([axes[name].get("physical_angle_deg")], 1, name + " start angle")
    return origin, _hash(path)


def require_binding(directory, c, state):
    saved = _read(Path(directory) / "loaded_trial_binding.json")
    expected = _binding(c, state.get("tcp_plan"))
    origin, digest = _leg_origin(directory)
    expected["leg_start_origin_sha256"] = digest
    if (_canonical(origin) != _canonical(state.get("leg_start_origin"))
        or saved != state.get("loaded_trial_binding") or saved != expected):
        raise ValueError("loaded diagnostic binding/configuration/plan/evidence changed; prepare again")
    return saved
