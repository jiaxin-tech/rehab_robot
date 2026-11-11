"""Robot-driven Cartesian lift; joint angles are measured responses only."""
from dataclasses import dataclass, asdict
import json
import hashlib
import numpy as np
import pandas as pd

PROFILE = "independent_tcp"
TRAJECTORY_ID = "independent_tcp_lift_v1"


def trajectory_digest(table):
    return hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TcpAudit:
    trajectory_id: str
    sample_count: int
    trajectory_valid: bool
    invalid_reasons: tuple[str, ...] = ()

    def as_dict(self):
        return asdict(self)


def build_tcp_trajectory(settings, *, current_tcp_start_pose, rehab_frame):
    """Seventh-order rest-to-rest lift and retrace with optional holds."""
    pose = np.asarray(current_tcp_start_pose, float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("TCP anchor must have six finite values")
    keys = ("target_peak_tcp_z_base_m", "horizontal_displacement_m",
            "lift_duration_s", "return_duration_s", "start_hold_s",
            "peak_hold_s", "sample_period_s")
    spec = {key: float(settings[key]) for key in keys}
    if not np.isfinite(list(spec.values())).all():
        raise ValueError("TCP trajectory settings must be finite")
    for key in ("lift_duration_s", "return_duration_s", "sample_period_s"):
        if spec[key] <= 0:
            raise ValueError(key + " must be positive")
    if spec["sample_period_s"] < .001 or max(spec.values()) > 3600:
        raise ValueError("TCP trajectory sampling or duration is outside supported bounds")
    if min(spec["start_hold_s"], spec["peak_hold_s"]) < 0:
        raise ValueError("TCP hold durations must be nonnegative")
    rotation = rehab_frame.rotation_base_from_rehab
    if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("TCP peak height requires rehab Z aligned with base +Z")
    height = spec["target_peak_tcp_z_base_m"] - pose[2]
    if height <= 0:
        raise ValueError("TCP peak height must exceed captured start height")
    displacement = np.array([spec["horizontal_displacement_m"], 0., height])
    times, phases, progress = [0.], ["start_hold"], [0.]
    for name, duration, start, end in (
        ("start_hold", spec["start_hold_s"], 0., 0.),
        ("lift", spec["lift_duration_s"], 0., 1.),
        ("peak_hold", spec["peak_hold_s"], 1., 1.),
        ("return", spec["return_duration_s"], 1., 0.),
    ):
        if duration == 0:
            continue
        count = max(8, int(np.ceil(duration / spec["sample_period_s"])))
        if count > 200000:
            raise ValueError("TCP trajectory has too many samples")
        u = np.linspace(0., 1., count + 1)[1:]
        blend = u**4 * (35 - 84*u + 70*u**2 - 20*u**3)
        times.extend((times[-1] + duration * u).tolist())
        progress.extend((start + (end-start)*blend).tolist())
        phases.extend([name]*count)
    t = np.asarray(times)
    delta = np.asarray(progress)[:, None] * displacement
    xyz = pose[:3] + delta @ rotation.T
    xyz[0] = xyz[-1] = pose[:3]
    velocity = np.gradient(xyz, t, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    table = pd.DataFrame({"time_s": t, "trajectory_id": TRAJECTORY_ID,
                          "trajectory_profile": PROFILE, "trajectory_phase": phases,
                          "trajectory_valid": True,
                          "tcp_plan_json": json.dumps(spec, sort_keys=True)})
    for i, axis in enumerate("xyz"):
        table[f"delta_{axis}_R"] = delta[:, i]
        table[f"tcp_{axis}_base"] = xyz[:, i]
        table[f"tcp_v{axis}_base"] = velocity[:, i]
        table[f"tcp_a{axis}_base"] = acceleration[:, i]
        table[f"tcp_r{axis}"] = pose[3+i]
    return table, TcpAudit(TRAJECTORY_ID, len(table), True), {
        "mode": PROFILE, "tcp_settings": spec,
        "joint_reference_used": False, "joint_tracking_required": False,
        "actual_peak_tcp_z_base_m": float(xyz[:, 2].max()),
        "duration_s": float(t[-1]),
    }


def audit_tcp_trajectory(table, safety, *, expected_trajectory_id=None,
                         frame=None, anchor=None):
    """Reconstruct the declared Cartesian plan at both execution gates."""
    from control.start_anchored_relative_trajectory import RehabFrameConfig
    reasons = []
    try:
        if expected_trajectory_id not in (None, TRAJECTORY_ID):
            reasons.append("tcp_trajectory_id_binding_mismatch")
        if table["trajectory_id"].tolist() != [TRAJECTORY_ID]*len(table):
            reasons.append("tcp_trajectory_id_invalid")
        if table["trajectory_profile"].tolist() != [PROFILE]*len(table):
            reasons.append("tcp_profile_invalid")
        if not all(type(v) in (bool, np.bool_) and v for v in table["trajectory_valid"]):
            reasons.append("tcp_trajectory_invalid_rows")
        plans = table["tcp_plan_json"].unique()
        if len(plans) != 1:
            raise ValueError("inconsistent TCP plans")
        pose_cols = [f"tcp_{a}_base" for a in "xyz"] + [f"tcp_r{a}" for a in "xyz"]
        pose = table.iloc[0][pose_cols].to_numpy(float)
        if frame is None:
            peak = table.loc[table["tcp_z_base"].idxmax()]
            dx = float(json.loads(plans[0])["horizontal_displacement_m"])
            direction = np.array([1., 0., 0.])
            if abs(dx) > 1e-12:
                direction = (peak[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)-pose[:3])/dx
                direction[2] = 0.
            frame = RehabFrameConfig(tuple(direction), (0., 0., 1.))
        expected, _, _ = build_tcp_trajectory(json.loads(plans[0]),
            current_tcp_start_pose=pose if anchor is None else anchor.tcp_pose_base,
            rehab_frame=frame)
        if table["trajectory_phase"].tolist() != expected["trajectory_phase"].tolist():
            reasons.append("tcp_phase_mismatch")
        numeric = ["time_s"] + pose_cols + [f"delta_{a}_R" for a in "xyz"] + [
            f"tcp_{d}{a}_base" for d in ("v", "a") for a in "xyz"]
        if len(table) != len(expected) or not np.allclose(
                table[numeric].to_numpy(float), expected[numeric].to_numpy(float),
                atol=1e-10, rtol=0):
            reasons.append("tcp_plan_content_mismatch")
        t = table["time_s"].to_numpy(float)
        xyz = table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
        velocity = np.gradient(xyz, t, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
        speed = float(np.linalg.norm(velocity, axis=1).max())
        accel = float(np.linalg.norm(acceleration, axis=1).max())
        if not np.isfinite(xyz).all() or not np.isfinite([speed, accel]).all():
            raise ValueError("nonfinite TCP data")
        if speed > safety.max_tcp_speed_m_s + 1e-12:
            reasons.append("trajectory_speed_limit_exceeded")
        if accel > safety.max_tcp_acceleration_m_s2 + 1e-12:
            reasons.append("trajectory_acceleration_limit_exceeded")
        if ((xyz < safety.workspace_min_base_m) | (xyz > safety.workspace_max_base_m)).any():
            reasons.append("trajectory_outside_reviewed_workspace")
        return tuple(reasons), speed, accel
    except (KeyError, ValueError, TypeError, IndexError):
        return ("tcp_plan_invalid",), None, None
