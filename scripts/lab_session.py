"""Single entry point for recording and analyzing dummy-leg experiments."""
from __future__ import annotations
import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from config.bundle import BUNDLE_PATH, ROOT, configuration, load_section
from collection.twincat_leg import (
    AdsInputImage,
    LegCaptureProcess,
    atomic_json,
    decode_image,
    positive,
    validate_config,
)
from control.execution_preflight import OPERATOR_CONFIRMATION
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from hardware.safety_events import validate_collision_source
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)
from measurement_validation.leg_episode import JointCalibration
from measurement_validation.offline_leg import inspect_file
from measurement_validation.robot_leg_round import aligned_measurement, endpoints, validate_analysis
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.acquire_robot_data import _configured_adapter_factory, load_live_wrench_config, run_acquisition
from scripts.capture_start_anchor import run_capture
from scripts.diagnose_robot import summarize_recording
from scripts.preview_rehab_trajectory import preview_trajectory
from scripts.run_rehab_experiment import run_execute


def _empty_load_settings(c):
    settings = c.get("empty_load")
    if not isinstance(settings, dict):
        raise ValueError("empty_load settings required")
    speed = positive(settings.get("max_tcp_speed_m_s"), "max_tcp_speed_m_s")
    acceleration = positive(
        settings.get("max_tcp_acceleration_m_s2"),
        "max_tcp_acceleration_m_s2",
    )
    margin = positive(settings.get("retiming_margin"), "retiming_margin")
    if margin < 1.0:
        raise ValueError("empty_load retiming_margin must be at least 1")
    return speed, acceleration, margin


def _trajectory_target_peak_z(c):
    settings = c.get("trajectory")
    if settings is None:
        settings = c.get("empty_load", {})
    if not isinstance(settings, dict):
        raise ValueError("trajectory settings must be an object")
    value = settings.get("target_peak_tcp_z_base_m")
    if value is None:
        return None
    return positive(value, "target_peak_tcp_z_base_m")


def _empty_load_motion(c):
    speed, acceleration, _ = _empty_load_settings(c)
    motion = dict(c["motion"])
    motion.update(
        max_tcp_speed_m_s=speed,
        max_tcp_acceleration_m_s2=acceleration,
    )
    return motion


def _target_peak_adjustment(
    c,
    anchor,
    frame,
    reference=DEFAULT_REFERENCE_PATH,
    *,
    enforce_motion_limits,
    allow_anisotropic_z_scale=False,
):
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        reference,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    if not audit.trajectory_valid:
        raise ValueError("source trajectory is invalid")
    target_peak_z = _trajectory_target_peak_z(c)
    rehab_z_scale = 1.0
    if target_peak_z is not None:
        frame_z = frame.rotation_base_from_rehab[:, 2]
        if not np.allclose(
            frame_z, np.array([0.0, 0.0, 1.0]), atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "target_peak_tcp_z_base_m requires rehab Z aligned with base +Z"
            )
        source_peak_delta_z = float(trajectory["delta_z_R"].max())
        requested_delta_z = float(target_peak_z - anchor.tcp_pose_base[2])
        if source_peak_delta_z <= 0.0 or requested_delta_z <= 0.0:
            raise ValueError("target peak Z must be above the captured TCP origin")
        rehab_z_scale = requested_delta_z / source_peak_delta_z
        if (
            not allow_anisotropic_z_scale
            and not np.isclose(rehab_z_scale, 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError(
                "leg-linked target peak requires a geometry-consistent joint "
                "trajectory; anisotropic rehab Z scaling is empty-load only"
            )
        trajectory, audit, _ = build_start_anchored_relative_trajectory(
            reference,
            current_tcp_start_pose=anchor.tcp_pose_base,
            rehab_frame=frame,
            rehab_z_scale=rehab_z_scale,
        )
    position_columns = ["tcp_x_base", "tcp_y_base", "tcp_z_base"]
    minimum = trajectory[position_columns].min().to_numpy(float)
    maximum = trajectory[position_columns].max().to_numpy(float)
    workspace_min = np.asarray(c["motion"]["workspace_min_base_m"], dtype=float)
    workspace_max = np.asarray(c["motion"]["workspace_max_base_m"], dtype=float)
    if np.any(minimum < workspace_min) or np.any(maximum > workspace_max):
        raise ValueError("scaled trajectory exceeds configured XYZ workspace")
    actual_peak_z = float(trajectory["tcp_z_base"].max())
    if target_peak_z is not None and not np.isclose(
        actual_peak_z, target_peak_z, atol=1e-12, rtol=0.0
    ):
        raise ValueError("scaled trajectory does not reach target peak Z")
    velocity = trajectory[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ].to_numpy(float)
    acceleration = trajectory[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ].to_numpy(float)
    peak_speed = float(np.linalg.norm(velocity, axis=1).max())
    peak_acceleration = float(np.linalg.norm(acceleration, axis=1).max())
    if enforce_motion_limits and (
        peak_speed > float(c["motion"]["max_tcp_speed_m_s"])
        or peak_acceleration > float(c["motion"]["max_tcp_acceleration_m_s2"])
    ):
        raise ValueError("scaled trajectory exceeds configured motion limits")
    metadata = {
        "target_peak_tcp_z_base_m": target_peak_z,
        "actual_peak_tcp_z_base_m": actual_peak_z,
        "minimum_tcp_base_m": minimum.tolist(),
        "maximum_tcp_base_m": maximum.tolist(),
        "rehab_z_scale": float(rehab_z_scale),
        "peak_tcp_speed_m_s": peak_speed,
        "peak_tcp_acceleration_m_s2": peak_acceleration,
    }
    return trajectory, audit, metadata


def _joint_calibration(values):
    return JointCalibration(
        **{
            key: values[key]
            for key in (
                "counts_per_degree",
                "zero_counts",
                "flexion_sign",
                "safe_min_deg",
                "safe_max_deg",
            )
        }
    )


def _validate_leg_start_freshness(c):
    heartbeat = positive(c["leg"].get("heartbeat_s"), "leg.heartbeat_s")
    max_age = positive(
        c["acquisition"].get("max_leg_start_age_s"),
        "acquisition.max_leg_start_age_s",
    )
    if heartbeat > max_age / 2.0:
        raise ValueError(
            "leg.heartbeat_s must be at most half max_leg_start_age_s"
        )


def _tracking_error_reason(
    reference,
    conversions,
    latest,
    trajectory_time_s,
    max_tracking_error_deg,
):
    """Return a fail-closed reason when the passive leg leaves its reference."""

    time_s = reference["time_s"].to_numpy(dtype=float)
    target_time_s = float(trajectory_time_s)
    limit = positive(max_tracking_error_deg, "max_tracking_error_deg")
    for name, axis in (("hip", 1), ("knee", 4)):
        raw = latest[f"axis{axis}_position_raw"]
        actual_deg = float(conversions[name].to_degrees(raw))
        expected_deg = float(
            np.rad2deg(
                np.interp(
                    target_time_s,
                    time_s,
                    reference[f"q_{name}_rad"].to_numpy(dtype=float),
                )
            )
        )
        error_deg = abs(actual_deg - expected_deg)
        if not np.isfinite(error_deg) or error_deg > limit:
            return (
                f"{name} tracking error {error_deg:.3f}deg exceeds "
                f"{limit:.3f}deg"
            )
    return None


def _build_leg_start_origin(c, rows, reference=DEFAULT_REFERENCE_PATH):
    if not rows:
        raise ValueError("leg start origin requires sampled input images")
    frame = pd.read_csv(reference) if not isinstance(reference, pd.DataFrame) else reference
    payload = {
        "schema_version": 1,
        "origin_role": "session_trajectory_start",
        "calibration_zero_counts_changed": False,
        "sample_count": len(rows),
        "axes": {},
        "session_calibration": {},
        "blocking_reasons": [],
    }
    for name, axis in (("hip", 1), ("knee", 4)):
        values = c["calibration"][name]
        physical = _joint_calibration(values)
        raw_values = np.asarray(
            [row[f"axis{axis}_position_raw"] for row in rows], dtype=float
        )
        if not np.isfinite(raw_values).all():
            raise ValueError(f"{name} leg start origin is non-finite")
        origin_raw = float(np.median(raw_values))
        source_reference_start_deg = float(
            np.rad2deg(frame[f"q_{name}_rad"].to_numpy(float)[0])
        )
        session_values = dict(values)
        trajectory_start_deg = float(physical.to_degrees(origin_raw))
        last = rows[-1]
        span_raw = float(np.ptp(raw_values))
        span_deg = span_raw / float(values["counts_per_degree"])
        reasons = payload["blocking_reasons"]
        if last[f"axis{axis}_operation_enabled"]:
            reasons.append(f"axis{axis}_operation_enabled")
        if last[f"axis{axis}_fault"]:
            reasons.append(f"axis{axis}_fault")
        if last[f"axis{axis}_error_code_hex"] != "0x0000":
            reasons.append(f"axis{axis}_error_code:{last[f'axis{axis}_error_code_hex']}")
        if span_deg > 0.1:
            reasons.append(f"axis{axis}_position_unstable")
        payload["axes"][name] = {
            "axis": axis,
            "position_origin_raw": origin_raw,
            "position_span_raw": span_raw,
            "position_span_deg": span_deg,
            "physical_angle_deg": float(physical.to_degrees(origin_raw)),
            "trajectory_start_deg": trajectory_start_deg,
            "source_reference_start_deg": source_reference_start_deg,
            "statusword": last[f"axis{axis}_statusword_hex"],
            "operation_enabled": last[f"axis{axis}_operation_enabled"],
            "fault": last[f"axis{axis}_fault"],
            "error_code": last[f"axis{axis}_error_code_hex"],
        }
        payload["session_calibration"][name] = session_values
    payload["usable_for_motion"] = not payload["blocking_reasons"]
    return payload


def _linked_trajectory_adjustment(c, anchor, frame, leg_origin,
                                  reference=DEFAULT_REFERENCE_PATH):
    """Build a geometry-consistent linked path from measured joint angles."""

    settings = c.get("trajectory")
    if not isinstance(settings, dict):
        raise ValueError("trajectory settings must be an object")
    target_peak_z = _trajectory_target_peak_z(c)
    if target_peak_z is None:
        raise ValueError("linked trajectory requires target_peak_tcp_z_base_m")
    phase_shift_min = positive(
        settings.get("linked_knee_phase_shift_min_s"),
        "linked_knee_phase_shift_min_s",
    )
    phase_shift_max = positive(
        settings.get("linked_knee_phase_shift_max_s"),
        "linked_knee_phase_shift_max_s",
    )
    if phase_shift_max <= phase_shift_min:
        raise ValueError(
            "linked_knee_phase_shift_max_s must exceed its minimum"
        )
    extra_limit_margin = positive(
        settings.get("linked_physical_limit_extra_margin_deg"),
        "linked_physical_limit_extra_margin_deg",
    )
    max_hip_scale = positive(
        settings.get("linked_hip_delta_scale_max"),
        "linked_hip_delta_scale_max",
    )
    joint_start_deg = np.asarray(
        [
            leg_origin["axes"]["hip"]["physical_angle_deg"],
            leg_origin["axes"]["knee"]["physical_angle_deg"],
        ],
        dtype=float,
    )
    if not np.isfinite(joint_start_deg).all():
        raise ValueError("linked trajectory joint start is non-finite")
    joint_start_rad = np.deg2rad(joint_start_deg)

    def build(scale, phase_shift):
        return build_start_anchored_relative_trajectory(
            reference,
            current_tcp_start_pose=anchor.tcp_pose_base,
            rehab_frame=frame,
            rehab_z_scale=1.0,
            joint_start_rad=joint_start_rad,
            hip_delta_scale=scale,
            knee_phase_shift_s=phase_shift,
        )

    required_knee_min = (
        float(c["calibration"]["knee"]["safe_min_deg"])
        + positive(
            c["acquisition"]["max_tracking_error_deg"],
            "max_tracking_error_deg",
        )
        + extra_limit_margin
    )

    def planned_knee_min(phase_shift):
        candidate, _, _ = build(1.0, phase_shift)
        return float(
            np.rad2deg(candidate["q_knee_ref"].to_numpy(float)).min()
        )

    if planned_knee_min(phase_shift_min) >= required_knee_min:
        phase_shift = phase_shift_min
    elif planned_knee_min(phase_shift_max) < required_knee_min:
        raise ValueError(
            "linked knee phase range cannot preserve the physical-limit margin"
        )
    else:
        low_phase = phase_shift_min
        high_phase = phase_shift_max
        for _ in range(80):
            midpoint = (low_phase + high_phase) / 2.0
            if planned_knee_min(midpoint) < required_knee_min:
                low_phase = midpoint
            else:
                high_phase = midpoint
        phase_shift = high_phase

    requested_delta_z = float(target_peak_z - anchor.tcp_pose_base[2])
    if requested_delta_z <= 0.0:
        raise ValueError("target peak Z must be above the captured TCP origin")

    def residual(scale):
        trajectory, _, _ = build(scale, phase_shift)
        return float(trajectory["delta_z_R"].max()) - requested_delta_z

    low_scale = 0.1
    low_residual = residual(low_scale)
    high_residual = residual(max_hip_scale)
    if low_residual > 0.0 or high_residual < 0.0:
        raise ValueError(
            "target peak Z is not bracketed by linked hip scale limits"
        )
    for _ in range(80):
        midpoint = (low_scale + max_hip_scale) / 2.0
        if residual(midpoint) < 0.0:
            low_scale = midpoint
        else:
            max_hip_scale = midpoint
    hip_scale = (low_scale + max_hip_scale) / 2.0
    trajectory, audit, metadata = build(hip_scale, phase_shift)
    if not audit.trajectory_valid:
        raise ValueError("linked source trajectory is invalid")

    ranges = {}
    for name in ("hip", "knee"):
        angles = np.rad2deg(trajectory[f"q_{name}_ref"].to_numpy(float))
        limits = c["calibration"][name]
        if (
            not np.isfinite(angles).all()
            or angles.min() < float(limits["safe_min_deg"])
            or angles.max() > float(limits["safe_max_deg"])
        ):
            raise ValueError(name + " linked trajectory exceeds physical range")
        ranges[name] = [float(angles.min()), float(angles.max())]
    source_knee = np.rad2deg(
        trajectory["source_q_knee_ref"].to_numpy(float)
    )
    planned_knee = np.rad2deg(trajectory["q_knee_ref"].to_numpy(float))
    excursion_error = abs(
        float(np.ptp(planned_knee)) - float(np.ptp(source_knee))
    )
    if excursion_error > 0.5:
        raise ValueError("linked knee excursion was not preserved")

    position_columns = ["tcp_x_base", "tcp_y_base", "tcp_z_base"]
    minimum = trajectory[position_columns].min().to_numpy(float)
    maximum = trajectory[position_columns].max().to_numpy(float)
    workspace_min = np.asarray(c["motion"]["workspace_min_base_m"], dtype=float)
    workspace_max = np.asarray(c["motion"]["workspace_max_base_m"], dtype=float)
    if np.any(minimum < workspace_min) or np.any(maximum > workspace_max):
        raise ValueError("linked trajectory exceeds configured XYZ workspace")
    actual_peak_z = float(trajectory["tcp_z_base"].max())
    if not np.isclose(actual_peak_z, target_peak_z, atol=1e-9, rtol=0.0):
        raise ValueError("linked trajectory does not reach target peak Z")
    velocity = trajectory[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ].to_numpy(float)
    acceleration = trajectory[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ].to_numpy(float)
    peak_speed = float(np.linalg.norm(velocity, axis=1).max())
    peak_acceleration = float(np.linalg.norm(acceleration, axis=1).max())
    if (
        peak_speed > float(c["motion"]["max_tcp_speed_m_s"])
        or peak_acceleration > float(c["motion"]["max_tcp_acceleration_m_s2"])
    ):
        raise ValueError("linked trajectory exceeds configured motion limits")
    return trajectory, audit, {
        "mode": "geometry_consistent_linked",
        "target_peak_tcp_z_base_m": target_peak_z,
        "actual_peak_tcp_z_base_m": actual_peak_z,
        "minimum_tcp_base_m": minimum.tolist(),
        "maximum_tcp_base_m": maximum.tolist(),
        "rehab_z_scale": 1.0,
        "peak_tcp_speed_m_s": peak_speed,
        "peak_tcp_acceleration_m_s2": peak_acceleration,
        "planned_joint_range_deg": ranges,
        "preserved_knee_excursion_deg": float(np.ptp(planned_knee)),
        "knee_excursion_error_deg": excursion_error,
        "required_planned_knee_min_deg": required_knee_min,
        "joint_profile_transform": metadata["joint_profile_transform"],
    }


def _capture_leg_start_origin(config_path, output_path, *, sample_count=20):
    c = configuration(config_path)
    source = AdsInputImage(load_section("twincat_leg_capture", config_path=config_path))
    rows = []
    try:
        for _ in range(sample_count):
            rows.append(decode_image(source.read(), source.config))
            time.sleep(0.01)
    finally:
        source.close()
    if c.get("trajectory", {}).get("mode") in ("independent_tcp", "original_candidate"):
        payload = {"origin_role": "passive_leg_start_check", "axes": {},
                   "session_calibration": c["calibration"],
                   "calibration_zero_counts_changed": False, "blocking_reasons": []}
        for name, axis in (("hip", 1), ("knee", 4)):
            conversion = _joint_calibration(c["calibration"][name])
            raw = np.asarray([row[f"axis{axis}_position_raw"] for row in rows], float)
            angle = float(conversion.to_degrees(float(np.median(raw))))
            payload["axes"][name] = {"physical_angle_deg": angle,
                "position_origin_raw": float(np.median(raw))}
            if (not np.isfinite(raw).all() or not np.isfinite(angle)
                or not conversion.safe_min_deg <= angle <= conversion.safe_max_deg
                or np.ptp(raw)/conversion.counts_per_degree > 0.1):
                payload["blocking_reasons"].append(name + " invalid or unstable start")
            for row in rows:
                if (row[f"axis{axis}_fault"] or row[f"axis{axis}_operation_enabled"]
                    or row[f"axis{axis}_error_code_hex"] != "0x0000"):
                    payload["blocking_reasons"].append(name + " not passive or drive fault")
                    break
        payload["usable_for_motion"] = not payload["blocking_reasons"]
    else:
        payload = _build_leg_start_origin(c, rows)
    atomic_json(output_path, payload)
    if not payload["usable_for_motion"]:
        raise RuntimeError(
            "leg start origin capture blocked: "
            + ";".join(payload["blocking_reasons"])
        )
    return payload


def check_config(config_path=BUNDLE_PATH, *, empty_load=False, require_physical_files=False):
    """Validate configuration; optionally require files on the lab computer."""
    problems = []
    try:
        c = configuration(config_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [str(exc)]
    def check(name, action):
        try:
            action()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(name + ": " + str(exc))
    check("motion", lambda: ExperimentSafetyConfig(
        **(_empty_load_motion(c) if empty_load else c["motion"])
    ).require_execute_allowed())
    check("collision", lambda: validate_collision_source(c["robot"]["collision_source"], c["robot"]["collision_event_max_age_s"]))
    capture_leg = not empty_load and c.get("validation", {}).get("capture_dummy_leg", True)
    if capture_leg:
        check("ADS", lambda: validate_config(load_section("twincat_leg_capture", config_path=config_path)))
        check("leg start freshness", lambda: _validate_leg_start_freshness(c))
        if require_physical_files and not Path(load_section("twincat_leg_capture", config_path=config_path)["ads_dll_path"]).is_file():
            problems.append("leg.ads_dll_path: DLL not found")
    acquisition_fields = (("wrench_hz",) if not capture_leg else
        ("pre_roll_s", "post_roll_s", "wrench_hz", "max_leg_start_age_s",
         "max_start_angle_error_deg", "max_tracking_error_deg"))
    independent = c.get("trajectory", {}).get("mode") in ("independent_tcp", "original_candidate")
    if independent:
        acquisition_fields = tuple(key for key in acquisition_fields if key != "max_tracking_error_deg")
        # Candidate association is checked against the frozen session plan.
    for name in acquisition_fields:
        check("acquisition." + name, lambda name=name: positive(c["acquisition"].get(name), name))
    check("wrench", lambda: load_live_wrench_config(config_path))
    def calibration():
        for name, values in c["calibration"].items():
            JointCalibration(**{k: values[k] for k in ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")})
        if set(c["calibration"]) != {"hip", "knee"}:
            raise ValueError("hip and knee calibration required")
    if capture_leg:
        check("calibration", calibration)
    from scripts.tcp_force_training import active as force_active, validate_config as validate_force
    if force_active(c):
        check("TCP force task", lambda: validate_force(c, measurement=not empty_load))
    if not empty_load and c.get("trajectory", {}).get("mode") == "original_candidate":
        from measurement_validation.robot_observation import validate_robot_observation
        check("robot-only observation", lambda: validate_robot_observation(c))
    if c["training"]["enabled"] and not independent:
        check("analysis", lambda: validate_analysis(c["analysis"], {"joints": c["calibration"]}))
        check("training.timeout_s", lambda: positive(c["training"]["timeout_s"], "timeout_s"))
    return problems


def _check_config_for_operation(config_path, *, empty_load=False):
    """Run the physical check while keeping simple test doubles compatible."""

    parameters = inspect.signature(check_config).parameters
    if "require_physical_files" in parameters:
        return check_config(
            config_path,
            empty_load=empty_load,
            require_physical_files=True,
        )
    return check_config(config_path)


def report(directory, state):
    if (directory / "robot/aligned_snapshot.csv").is_file():
        state["robot_timing"] = summarize_recording(directory)
    atomic_json(directory / "round_status.json", state)
    lines = ["# 假腿实验", "", "状态：" + state["status"],
             "训练：" + state.get("training_status", "DISABLED"),
             "原因：" + state.get("reason", "无"), ""]
    lines += ["- " + x for x in state.get("blockers", [])]
    timing = state.get("robot_timing", {})
    if timing:
        if timing.get("available"):
            lines += [f"机器人快照（含启动）：有效 {timing['valid_samples']}/{timing['samples']}；"
                      f"曾就绪 {timing['ever_ready']}；就绪后失效 {timing['invalid_after_ready']}。"]
            for field, label in (("query_duration_ms", "力查询"), ("state_age_ms", "状态时龄"),
                                 ("wrench_age_ms", "力时龄"), ("state_wrench_skew_ms", "状态/力时差")):
                metric = timing["metrics"][field]
                values = ["无数据" if metric[k] is None else f"{metric[k]:.3f}" for k in ("p95", "max")]
                lines += [f"- {label} P95/最大：{' / '.join(values)} 毫秒"]
            lines += ["机器人失效原因：" + json.dumps(timing["invalid_reasons"], ensure_ascii=False)]
        else:
            lines += ["机器人时序统计不可用：" + timing["reason"]]
    quality = state.get("quality", {})
    if quality:
        lines += ["采集：" + json.dumps({k: quality.get(k) for k in
            ("rows", "duration_s", "sample_rate_hz", "usable_for_dynamics")}, ensure_ascii=False)]
        lines += ["数据问题：" + "; ".join(quality["findings"])] if quality.get("findings") else []
    if "metrics" in state:
        metrics = state["metrics"]
        lines += ["实测力矩：" + json.dumps({k: metrics.get(k) for k in
            ("components_nm", "peaks_nm")}, ensure_ascii=False)]
    if "model" in state:
        lines += ["模型建议：" + json.dumps(state["model"].get("action"), ensure_ascii=False)]
        lines += ["参考自身归一化为 1；单候选 DOMAIN_EXHAUSTED 不代表优化收益。"]
    if "force_training" in state:
        lines += ["TCP 受力决策：" + json.dumps(state["force_training"]["action"], ensure_ascii=False),
                  "仅使用机器人实测运动与受力；未辨识人体关节参数，预测不等于临床安全保证。"]
    if (directory / "leg/leg.csv").is_file():
        lines += ["", "原始数据：leg/leg.csv 和 robot/。"]
    if (directory / "dispatch.csv").is_file():
        lines += ["发送记录：dispatch.csv。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def observe(config_path, directory, *, duration_s=10., capture_factory=LegCaptureProcess,
            acquire_fn=run_acquisition):
    """Record robot and leg together without sending power, mode or motion commands."""
    positive(duration_s, "duration_s")
    config_path = Path(config_path).resolve()
    c = configuration(config_path)
    capture_leg = c.get("validation", {}).get("capture_dummy_leg", True)
    if capture_leg:
        leg_config = validate_config(load_section("leg", config_path=config_path))
        if not Path(leg_config["ads_dll_path"]).is_file():
            raise FileNotFoundError("leg.ads_dll_path: DLL not found")
    wrench = load_live_wrench_config(config_path)
    positive(c["acquisition"]["wrench_hz"], "wrench_hz")
    robot = c["robot"]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(status="OBSERVING", config_path=str(config_path), motion_commanded=False)
    capture = None
    primary_failure = None
    report(directory, state)
    try:
        if capture_leg:
            capture = capture_factory(config_path, directory / "leg")
            capture.start()
        result = acquire_fn(robot_ip=robot["ip"], episode_dir=directory / "robot",
            duration_s=duration_s, mode="live", live_wrench_config=wrench,
            wrench_hz=c["acquisition"]["wrench_hz"],
            adapter_factory=_configured_adapter_factory(local_ip=robot["local_ip"],
                robot_class=robot["robot_class"], state_interval_ms=None,
                collision_source=robot["collision_source"],
                collision_event_max_age_s=robot["collision_event_max_age_s"]),
            observation_guard=capture.health_reason if capture is not None else None)
        if result.get("completed") is not True:
            raise RuntimeError("robot observation incomplete")
        reason = capture.health_reason() if capture is not None else None
        if reason:
            raise RuntimeError(reason)
        state["observation"] = result
    except BaseException as exc:
        primary_failure = exc
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        try:
            if capture is not None:
                capture.stop()
        except BaseException as exc:
            cleanup_reason = "capture cleanup: " + str(exc)
            if primary_failure is None:
                state.update(status="FAILED", reason=cleanup_reason)
                raise
            state["capture_cleanup_error"] = cleanup_reason
        finally:
            report(directory, state)
    try:
        if capture_leg:
            state["quality"] = inspect_file(directory / "leg/leg.csv", expected_hz=c["leg"]["expected_hz"])
        state["status"] = "OBSERVATION_COMPLETE"
    except Exception as exc:
        state.update(status="FAILED", reason="quality analysis: " + str(exc))
        raise
    finally:
        report(directory, state)
    return state


def _prepare_empty_load_artifacts(config_path, directory):
    c = configuration(config_path)
    speed_limit, acceleration_limit, margin = _empty_load_settings(c)
    anchor = load_start_anchor(directory / "anchor.json")
    frame = load_rehab_frame_config(config_path)
    reference, source_metadata = load_closed_reference_trajectory(DEFAULT_REFERENCE_PATH)
    baseline, audit, adjustment = _target_peak_adjustment(
        c,
        anchor,
        frame,
        reference,
        enforce_motion_limits=False,
        allow_anisotropic_z_scale=True,
    )
    target_peak_z = adjustment["target_peak_tcp_z_base_m"]
    rehab_z_scale = adjustment["rehab_z_scale"]
    velocity = baseline[["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]].to_numpy(float)
    acceleration = baseline[["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]].to_numpy(float)
    source_peak_speed = float(np.linalg.norm(velocity, axis=1).max())
    source_peak_acceleration = float(np.linalg.norm(acceleration, axis=1).max())
    time_scale = margin * max(
        1.0,
        source_peak_speed / speed_limit,
        np.sqrt(source_peak_acceleration / acceleration_limit),
    )
    retimed = reference.copy(deep=True)
    source_time = retimed["time_s"].to_numpy(float)
    retimed["time_s"] = (source_time - source_time[0]) * time_scale
    for name in (
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "minimum_jerk_phase_rate_s_inv",
    ):
        if name in retimed:
            retimed[name] = retimed[name].to_numpy(float) / time_scale
    for name in (
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "minimum_jerk_phase_acceleration_s_inv2",
    ):
        if name in retimed:
            retimed[name] = retimed[name].to_numpy(float) / time_scale ** 2
    if "profile" in retimed:
        retimed["profile"] = "empty_load_commissioning_slow"
    if "retimed_timing_is_original" in retimed:
        retimed["retimed_timing_is_original"] = False
    reference_path = directory / "empty_load_reference.csv"
    retimed.to_csv(reference_path, index=False)
    safety_path = directory / "empty_load_safety.json"
    atomic_json(safety_path, _empty_load_motion(c))
    retimed_trajectory, retimed_audit, _ = build_start_anchored_relative_trajectory(
        reference_path,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
        rehab_z_scale=rehab_z_scale,
    )
    retimed_velocity = retimed_trajectory[
        ["tcp_vx_base", "tcp_vy_base", "tcp_vz_base"]
    ].to_numpy(float)
    retimed_acceleration = retimed_trajectory[
        ["tcp_ax_base", "tcp_ay_base", "tcp_az_base"]
    ].to_numpy(float)
    retimed_peak_speed = float(np.linalg.norm(retimed_velocity, axis=1).max())
    retimed_peak_acceleration = float(np.linalg.norm(retimed_acceleration, axis=1).max())
    if (
        not retimed_audit.trajectory_valid
        or retimed_peak_speed > speed_limit
        or retimed_peak_acceleration > acceleration_limit
    ):
        raise ValueError("empty-load retimed trajectory exceeds its independent limits")
    metadata = {
        "mode": "empty_load",
        "source_reference": source_metadata,
        "reference_path": str(reference_path.resolve()),
        "safety_path": str(safety_path.resolve()),
        "time_scale": float(time_scale),
        "source_duration_s": float(source_time[-1] - source_time[0]),
        "retimed_duration_s": float(retimed["time_s"].iloc[-1]),
        "source_peak_speed_m_s": source_peak_speed,
        "source_peak_acceleration_m_s2": source_peak_acceleration,
        "retimed_peak_speed_m_s": retimed_peak_speed,
        "retimed_peak_acceleration_m_s2": retimed_peak_acceleration,
        "max_tcp_speed_m_s": speed_limit,
        "max_tcp_acceleration_m_s2": acceleration_limit,
        "retiming_margin": margin,
        "target_peak_tcp_z_base_m": target_peak_z,
        "actual_peak_tcp_z_base_m": float(retimed_trajectory["tcp_z_base"].max()),
        "minimum_tcp_z_base_m": float(retimed_trajectory["tcp_z_base"].min()),
        "rehab_z_scale": float(rehab_z_scale),
        "dummy_leg_required": False,
        "training_allowed": False,
    }
    atomic_json(directory / "empty_load_retiming.json", metadata)
    return metadata


def _prepare_original_candidate(config_path, directory, c, state, empty_load,
                                leg_origin_capture_fn, selected, parent_session, repeat_index):
    from control.execution_preflight import audit_execution_trajectory_content
    from scripts.original_training_adapter import freeze_binding, commit_original_prediction, file_hash
    anchor = load_start_anchor(directory / "anchor.json")
    frame = load_rehab_frame_config(config_path)
    frozen = directory / "original_candidate.csv"
    shutil.copyfile(selected["path"], frozen)
    table, audit, metadata = build_start_anchored_relative_trajectory(frozen,
        current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    metadata.update(mode="original_candidate", candidate_id=selected["candidate_id"],
        reference_path=str(frozen), reference_sha256=file_hash(frozen), anchor=anchor.to_dict())
    motion = _empty_load_motion(c)
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        motion[key] = min(motion[key], c["motion"][key])
    reasons, _, _ = audit_execution_trajectory_content(table, ExperimentSafetyConfig(**motion),
        expected_trajectory_id=anchor.trajectory_id, frame=frame, anchor=anchor)
    blockers = list(reasons)
    if not audit.trajectory_valid:
        blockers.extend(audit.invalid_reasons)
    requested_z = _trajectory_target_peak_z(c)
    actual_z = float(table.tcp_z_base.max())
    metadata["actual_peak_tcp_z_base_m"] = actual_z
    if requested_z is not None and not np.isclose(actual_z, requested_z, atol=1e-9, rtol=0):
        blockers.append(f"original candidate peak Z {actual_z:.6f}m differs from requested {requested_z:.6f}m; candidate is not reshaped")
    table.to_csv(directory / "tcp_trajectory.csv", index=False)
    atomic_json(directory / "tcp_plan.json", metadata)
    atomic_json(directory / "tcp_safety.json", motion)
    atomic_json(directory / "prepared_config.json", c)
    if not empty_load:
        binding = freeze_binding(directory, c, metadata)
        state["original_training_binding"] = binding
        if binding["status"] != "BOUND":
            blockers.append("original training: " + binding["reason"])
        if parent_session is not None and binding["status"] == "BOUND":
            state["pretrial_prediction"] = commit_original_prediction(parent_session, directory, binding)
        leg_path = directory / "leg_start_origin.json"
        state["leg_start_origin"] = leg_origin_capture_fn(config_path, leg_path)
        state["leg_start_origin_path"] = str(leg_path)
    preview_trajectory(anchor=anchor, frame_config=config_path, reference=frozen,
                       output_dir=directory / "preview")
    state.update(status="PREPARED", trajectory_mode="original_candidate", tcp_plan=metadata,
        reference_path=str(frozen), safety_config_path=str(directory / "tcp_safety.json"),
        training_status="NOT_APPLICABLE" if empty_load else "PENDING",
        blockers=blockers + _check_config_for_operation(config_path, empty_load=empty_load))
    if parent_session is not None:
        state.update(parent_session=str(parent_session), repeat_index=int(repeat_index))


def _prepare_tcp_artifacts(config_path, directory, c, state, empty_load,
                           leg_origin_capture_fn, trajectory_settings=None,
                           parent_session=None, repeat_index=None):
    from control.independent_tcp_trajectory import build_tcp_trajectory, audit_tcp_trajectory
    anchor = load_start_anchor(directory / "anchor.json")
    frame = load_rehab_frame_config(config_path)
    settings = dict(trajectory_settings or c["trajectory"])
    settings.pop("mode", None)
    # Both modes use the same conservative limits and Cartesian geometry.
    motion = dict(c["motion"])
    speed, acceleration, margin = _empty_load_settings(c)
    motion["max_tcp_speed_m_s"] = min(speed, motion["max_tcp_speed_m_s"])
    motion["max_tcp_acceleration_m_s2"] = min(acceleration, motion["max_tcp_acceleration_m_s2"])
    table, _, _ = build_tcp_trajectory(settings,
        current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    v = np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]], axis=1).max()
    a = np.linalg.norm(table[[f"tcp_a{axis}_base" for axis in "xyz"]], axis=1).max()
    scale = max(1., margin*v/motion["max_tcp_speed_m_s"],
                np.sqrt(margin*a/motion["max_tcp_acceleration_m_s2"]))
    for key in ("lift_duration_s", "return_duration_s"):
        settings[key] *= scale
    table, audit, metadata = build_tcp_trajectory(settings,
        current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    metadata["anchor"] = anchor.to_dict()
    safety = ExperimentSafetyConfig(**motion)
    safety.require_execute_allowed()
    reasons, _, _ = audit_tcp_trajectory(table, safety,
        expected_trajectory_id=anchor.trajectory_id, frame=frame, anchor=anchor)
    if reasons:
        raise ValueError("TCP trajectory blocked: " + ";".join(reasons))
    table.to_csv(directory / "tcp_trajectory.csv", index=False)
    atomic_json(directory / "tcp_plan.json", metadata)
    atomic_json(directory / "tcp_safety.json", motion)
    atomic_json(directory / "prepared_config.json", c)
    from scripts import tcp_force_training as force
    force_mode = force.active(c)
    if force_mode:
        state["learning_mode"] = "tcp_force"
        state["force_training_binding"] = force.freeze_binding(directory, c, metadata, measurement=False)
        if not empty_load and parent_session is not None:
            state["pretrial_prediction"] = force.commit_prediction(parent_session, directory, state["force_training_binding"])
    if not empty_load and not force_mode:
        from scripts.original_training_adapter import freeze_binding, commit_original_prediction
        state["original_training_binding"] = freeze_binding(directory, c, metadata)
        if parent_session is not None and state["original_training_binding"]["status"] == "BOUND":
            state["pretrial_prediction"] = commit_original_prediction(
                parent_session, directory, state["original_training_binding"])
    if not empty_load and c.get("validation", {}).get("capture_dummy_leg", True):
        leg_path = directory / "leg_start_origin.json"
        origin = leg_origin_capture_fn(config_path, leg_path)
        state.update(leg_start_origin_path=str(leg_path), leg_start_origin=origin)
    preview_trajectory(anchor=anchor, frame_config=config_path,
        output_dir=directory / "preview", tcp_trajectory_settings=metadata["tcp_settings"])
    state.update(trajectory_mode="independent_tcp", tcp_plan=metadata,
        reference_path=str(directory / "tcp_trajectory.csv"),
        safety_config_path=str(directory / "tcp_safety.json"),
        **({"parent_session": str(parent_session), "repeat_index": int(repeat_index)}
           if parent_session is not None else {}),
        status="PREPARED", blockers=_check_config_for_operation(config_path, empty_load=empty_load))
    state["training_status"] = "NOT_APPLICABLE" if empty_load else "PENDING"
    if not empty_load and not force_mode and state["original_training_binding"]["status"] != "BOUND":
        state["training_status"] = "PENDING_INPUT_BINDING"
        state["blockers"].append("original training: " + state["original_training_binding"]["reason"])


def _passive_leg_guard(capture, c, origin, *, before_motion=False):
    reason = capture.health_reason()
    if reason:
        return reason
    snapshot = capture.snapshot()
    age = (time.perf_counter_ns() - snapshot["last_sample_ns"]) / 1e9
    if not np.isfinite(age) or not 0 <= age <= c["acquisition"]["max_leg_start_age_s"]:
        return "leg sample stale"
    for name, axis in (("hip", 1), ("knee", 4)):
        row = snapshot["latest"]
        if row[f"axis{axis}_fault"] or row[f"axis{axis}_error_code_hex"] != "0x0000":
            return name + " drive fault"
        if row[f"axis{axis}_operation_enabled"]:
            return name + " drive enabled during passive experiment"
        conversion = _joint_calibration(c["calibration"][name])
        angle = conversion.to_degrees(row[f"axis{axis}_position_raw"])
        if not np.isfinite(angle) or not conversion.safe_min_deg <= angle <= conversion.safe_max_deg:
            return name + " outside physical calibrated range"
        if before_motion and abs(angle-origin["axes"][name]["physical_angle_deg"]) > c["acquisition"]["max_start_angle_error_deg"]:
            return name + " moved since prepare; prepare a new session"
    return None


def _run_tcp(directory, state, *, empty_load, capture_factory, execute_fn):
    if bool(state.get("empty_load")) != bool(empty_load):
        raise ValueError("prepared session mode does not match requested run mode")
    config_path = state["config_path"]
    c = configuration(config_path)
    if c != json.loads((directory / "prepared_config.json").read_text(encoding="utf-8")):
        raise ValueError("configuration changed; prepare a new TCP session")
    blockers = _check_config_for_operation(config_path, empty_load=empty_load)
    if blockers:
        raise ValueError("complete physical parameters: " + "; ".join(blockers))
    if state.get("blockers"):
        raise ValueError("prepared trajectory blocked: " + "; ".join(state["blockers"]))
    anchor = load_start_anchor(directory / "anchor.json")
    if not empty_load:
        if state.get("learning_mode") == "tcp_force":
            from scripts.tcp_force_training import require_binding
        else:
            from scripts.original_training_adapter import require_binding
        require_binding(directory, c, state)
    plan = json.loads((directory / "tcp_plan.json").read_text(encoding="utf-8"))
    if plan != state["tcp_plan"] or plan["anchor"] != anchor.to_dict():
        raise ValueError("prepared TCP plan or anchor changed; prepare a new session")
    if state.get("pretrial_prediction") is not None:
        prediction = json.loads((directory / "pretrial_prediction.json").read_text(encoding="utf-8"))
        force_mode = state.get("learning_mode") == "tcp_force"
        binding = state["force_training_binding"] if force_mode else state["original_training_binding"]
        if prediction != state["pretrial_prediction"] or prediction["candidate_id"] != binding["candidate_id"]:
            raise ValueError("prepared pretrial prediction changed")
        if force_mode:
            from scripts.tcp_force_training import verify_training as verify_source, file_hash
            source_name = "tcp_force_training.json"
        else:
            from scripts.original_training_adapter import verify_original_training as verify_source, file_hash
            source_name = "original_training.json"
        verify_source(state["parent_session"])
        if file_hash(Path(state["parent_session"]) / source_name) != prediction["source_training_sha256"]:
            raise ValueError("original source prediction changed after prepare")
    frame = load_rehab_frame_config(config_path)
    from control.prepared_trajectory import build_prepared_trajectory
    table, _, _ = build_prepared_trajectory(plan,
        current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    from control.prepared_trajectory import assert_frozen_trajectory
    assert_frozen_trajectory(table, directory / "tcp_trajectory.csv")
    expected_motion = dict(c["motion"])
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        expected_motion[key] = min(c["empty_load"][key], expected_motion[key])
    actual_motion = json.loads((directory / "tcp_safety.json").read_text(encoding="utf-8"))
    if actual_motion != expected_motion:
        raise ValueError("prepared TCP safety settings changed")
    from control.execution_preflight import audit_execution_trajectory_content
    reasons, _, _ = audit_execution_trajectory_content(table, ExperimentSafetyConfig(**actual_motion),
        expected_trajectory_id=anchor.trajectory_id, anchor=anchor, frame=frame)
    if reasons:
        raise ValueError("prepared TCP trajectory invalid: " + ";".join(reasons))
    if not empty_load and state.get("learning_mode") == "tcp_force":
        if state.get("trial_purpose") == "evaluation":
            from scripts.tcp_force_training import verify_training, verify_evaluation
            source = Path(state["evaluation_source"])
            source_state = json.loads((source / "round_status.json").read_text(encoding="utf-8"))
            (verify_evaluation if source_state.get("trial_purpose") == "evaluation" else verify_training)(source)
        from measurement_validation.force_trial_ledger import claim
        state["force_attempt"] = claim(directory, c, state)
    capture, dispatches, failure = None, [], None
    robot, acquisition = c["robot"], c["acquisition"]
    state.update(status="RUNNING", blockers=[])
    report(directory, state)
    try:
        guard = None
        if not empty_load and c.get("validation", {}).get("capture_dummy_leg", True):
            origin = json.loads((directory / "leg_start_origin.json").read_text(encoding="utf-8"))
            if not origin.get("usable_for_motion"):
                raise ValueError("passive leg start check failed")
            capture = capture_factory(config_path, directory / "leg")
            capture.start()
            def guard():
                return _passive_leg_guard(capture, c, origin, before_motion=not dispatches)
            deadline = time.perf_counter() + acquisition["pre_roll_s"]
            while time.perf_counter() < deadline:
                reason = guard()
                if reason:
                    raise RuntimeError(reason)
                time.sleep(.01)
        def dispatched(t, start, end):
            dispatches.append(dict(trajectory_time_s=t, dispatch_start_ns=start, dispatch_end_ns=end))
        kwargs = dict(robot_ip=robot["ip"], local_ip=robot["local_ip"],
            episode_dir=directory / "robot", anchor_path=directory / "anchor.json",
            requested_anchor_id=anchor.anchor_id, frame_config_path=config_path,
            safety_config_path=directory / "tcp_safety.json", trajectory_id=anchor.trajectory_id,
            enable_motion=True,
            operator_confirmation=OPERATOR_CONFIRMATION,
            live_wrench_config=load_live_wrench_config(config_path),
            adapter_factory=_configured_adapter_factory(local_ip=robot["local_ip"], robot_class=robot["robot_class"],
                state_interval_ms=None, collision_source=robot["collision_source"], collision_event_max_age_s=robot["collision_event_max_age_s"]),
            wrench_hz=acquisition["wrench_hz"], collision_source=robot["collision_source"],
            collision_event_max_age_s=robot["collision_event_max_age_s"],
            prepare_realtime=True, dispatch_observer=dispatched)
        if state["trajectory_mode"] == "original_candidate":
            kwargs["reference_path_override"] = Path(plan["reference_path"])
        else:
            kwargs["tcp_trajectory_settings"] = plan["tcp_settings"]
        if guard is not None:
            kwargs["execution_guard"] = guard
        result = execute_fn(**kwargs)
        if result.get("completed") is not True or not dispatches:
            raise RuntimeError("robot did not complete the TCP trajectory")
        if capture is not None:
            deadline = time.perf_counter() + acquisition["post_roll_s"]
            while time.perf_counter() < deadline:
                reason = guard()
                if reason:
                    raise RuntimeError(reason)
                time.sleep(.01)
            capture.stop()
            capture = None
        state.update(status="EMPTY_LOAD_COMPLETE" if empty_load else "CAPTURE_COMPLETE", execution=result)
    except BaseException as exc:
        failure = exc
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        try:
            if capture is not None:
                capture.stop()
        except BaseException as exc:
            state["capture_cleanup_error"] = str(exc)
            if failure is None:
                state.update(status="FAILED", reason="capture cleanup: " + str(exc))
                raise
        finally:
            if dispatches:
                pd.DataFrame(dispatches).to_csv(directory / "dispatch.csv", index=False)
            report(directory, state)
    if not empty_load:
        analysis_error = None
        try:
            if c.get("validation", {}).get("capture_dummy_leg", True):
                state["quality"] = inspect_file(directory / "leg/leg.csv", expected_hz=c["leg"]["expected_hz"])
                from measurement_validation.passive_response import analyze_passive_response
                state["passive_response"] = analyze_passive_response(directory, c)
            state["analysis_status"] = "COMPLETE"
            state.pop("response_analysis_error", None)
        except Exception as exc:
            state["response_analysis_error"] = str(exc)
            state["analysis_status"] = "FAILED"
            analysis_error = exc
        finally:
            report(directory, state)
        # Adapt passive measurements to the existing algorithm; never substitute a learner.
        trained = train_session(directory, training_mode="passive")
        if analysis_error is not None:
            raise RuntimeError("passive response analysis failed: " + str(analysis_error)) from analysis_error
        return trained
    return state


def prepare(
    config_path,
    directory,
    *,
    capture_fn=run_capture,
    leg_origin_capture_fn=_capture_leg_start_origin,
    empty_load=False,
    trajectory_settings=None,
    parent_session=None,
    repeat_index=None,
):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config_path = Path(config_path).resolve()
    c = configuration(config_path)
    state = dict(status="PREPARING", config_path=str(config_path), empty_load=bool(empty_load))
    report(directory, state)
    try:
        r = c["robot"]
        independent = c.get("trajectory", {}).get("mode") == "independent_tcp"
        original = c.get("trajectory", {}).get("mode") == "original_candidate"
        if independent:
            from scripts.tcp_force_training import active, selected_settings
            if active(c):
                _, trajectory_settings = selected_settings(c, trajectory_settings)
        selected = None
        if original:
            from scripts.original_training_adapter import catalog
            cid = ((trajectory_settings or {}).get("candidate_id")
                   or c["training"]["reference_candidate_id"])
            matches = [item for item in catalog(c) if item["candidate_id"] == cid]
            if len(matches) != 1:
                raise ValueError("original selected candidate is not in the catalog")
            selected = matches[0]
        capture_fn(robot_ip=r["ip"], output_path=directory / "anchor.json",
                   **({"independent_tcp": True} if independent else {}),
                   **({"reference": selected["path"]} if original else {}),
                   adapter_factory=_configured_adapter_factory(local_ip=r["local_ip"], robot_class=r["robot_class"], state_interval_ms=None,
                       collision_source=r["collision_source"], collision_event_max_age_s=r["collision_event_max_age_s"]))
        if original:
            _prepare_original_candidate(config_path, directory, c, state, empty_load,
                leg_origin_capture_fn, selected, parent_session, repeat_index)
            return state
        if independent:
            _prepare_tcp_artifacts(config_path, directory, c, state, empty_load,
                                   leg_origin_capture_fn, trajectory_settings,
                                   parent_session, repeat_index)
            return state
        reference = DEFAULT_REFERENCE_PATH
        if empty_load:
            empty_load_metadata = _prepare_empty_load_artifacts(config_path, directory)
            reference = Path(empty_load_metadata["reference_path"])
            state["empty_load_retiming"] = empty_load_metadata
            state["reference_path"] = str(reference.resolve())
            state["safety_config_path"] = empty_load_metadata["safety_path"]
        else:
            leg_origin_path = directory / "leg_start_origin.json"
            leg_origin = leg_origin_capture_fn(config_path, leg_origin_path)
            state["leg_start_origin_path"] = str(leg_origin_path.resolve())
            state["leg_start_origin"] = leg_origin
            anchor = load_start_anchor(directory / "anchor.json")
            frame = load_rehab_frame_config(config_path)
            linked_trajectory, _, trajectory_adjustment = _linked_trajectory_adjustment(
                c,
                anchor,
                frame,
                leg_origin,
                reference,
            )
            adjustment_path = directory / "trajectory_adjustment.json"
            atomic_json(adjustment_path, trajectory_adjustment)
            state["trajectory_adjustment_path"] = str(adjustment_path.resolve())
            state["trajectory_adjustment"] = trajectory_adjustment
        preview_trajectory(
            anchor=directory / "anchor.json",
            frame_config=config_path,
            reference=reference,
            output_dir=directory / "preview",
            rehab_z_scale=(
                float(empty_load_metadata["rehab_z_scale"])
                if empty_load
                else float(trajectory_adjustment["rehab_z_scale"])
            ),
            joint_start_rad=(
                None
                if empty_load
                else trajectory_adjustment["joint_profile_transform"][
                    "joint_start_rad"
                ]
            ),
            hip_delta_scale=(
                1.0
                if empty_load
                else trajectory_adjustment["joint_profile_transform"][
                    "hip_delta_scale"
                ]
            ),
            knee_phase_shift_s=(
                0.0
                if empty_load
                else trajectory_adjustment["joint_profile_transform"][
                    "knee_phase_shift_s"
                ]
            ),
        )
        if not empty_load:
            analysis_reference = pd.read_csv(reference).copy(deep=True)
            analysis_reference["q_hip_rad"] = linked_trajectory["q_hip_ref"]
            analysis_reference["q_knee_rad"] = linked_trajectory["q_knee_ref"]
            analysis_reference["theta_shank_rad"] = (
                analysis_reference["q_hip_rad"]
                - analysis_reference["q_knee_rad"]
            )
            analysis_time = analysis_reference["time_s"].to_numpy(float)
            for name in ("hip", "knee"):
                angle = analysis_reference[f"q_{name}_rad"].to_numpy(float)
                velocity = np.gradient(angle, analysis_time, edge_order=2)
                analysis_reference[f"dq_{name}_rad_s"] = velocity
                analysis_reference[f"ddq_{name}_rad_s2"] = np.gradient(
                    velocity, analysis_time, edge_order=2
                )
            analysis_reference_path = directory / "linked_joint_reference.csv"
            analysis_reference.to_csv(analysis_reference_path, index=False)
            state["analysis_reference_path"] = str(
                analysis_reference_path.resolve()
            )
        blockers = _check_config_for_operation(config_path, empty_load=empty_load)
        state.update(status="PREPARED", blockers=blockers)
    except BaseException as exc:
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        report(directory, state)
    return state


def run_algorithm_worker(directory, request, timeout):
    work = ROOT / ".cache/training" / directory.name
    work.mkdir(parents=True, exist_ok=True)
    request_path, output_path = work / "request.json", work / "result.json"
    atomic_json(request_path, request)
    command = [sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round", str(request_path), "--output", str(output_path)]
    with (directory / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return json.loads(output_path.read_text(encoding="utf-8"))


def train_session(directory, *, training_mode="auto"):
    directory = Path(directory).resolve()
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if training_mode not in ("auto", "passive", "joint"):
        raise ValueError("unknown training mode")
    independent = state.get("trajectory_mode") in ("independent_tcp", "original_candidate")
    if training_mode == "joint" and independent:
        raise ValueError("joint-reference training cannot train independent TCP motion")
    if training_mode == "passive" and not independent:
        raise ValueError("passive measurement training requires a passive TCP session")
    if not independent:
        return analyze(directory)
    if state.get("empty_load") or state["status"] not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE", "COMPLETE"):
        raise ValueError("passive training requires a completed with-leg motion")
    state["training_status"] = "RUNNING"
    state.pop("training_error", None)
    report(directory, state)
    try:
        if state.get("learning_mode") == "tcp_force":
            from scripts.tcp_force_training import train, evaluate
            if state.get("trial_purpose") == "evaluation":
                state["force_evaluation"] = evaluate(directory)
                (directory / "training.log").write_text("TCP_FORCE_EVALUATION_ONLY; no GP update\n", encoding="utf-8")
            else:
                state["force_training"] = train(directory)
                (directory / "training.log").write_text("TCP_FORCE_GP_EI completed; see tcp_force_training.json\n", encoding="utf-8")
        else:
            from scripts.original_training_adapter import train_original
            state["original_training"] = train_original(directory, run_algorithm_worker)
        state["training_status"] = "COMPLETE"
    except Exception as exc:
        state.update(training_status="FAILED", training_error=str(exc))
        state.pop("original_training", None)
        state.pop("force_training", None)
        state.pop("force_evaluation", None)
        raise
    finally:
        report(directory, state)
    return state


def analyze(directory):
    directory = Path(directory).resolve()
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if state["status"] not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE", "COMPLETE"):
        raise ValueError("a completed recording is required")
    if state.get("trajectory_mode") in ("independent_tcp", "original_candidate"):
        if state.get("learning_mode") == "tcp_force":
            from scripts.tcp_force_training import measure
            state["force_response"] = measure(directory)
            state["analysis_status"] = "COMPLETE"
            report(directory, state)
            return state
        from measurement_validation.passive_response import analyze_passive_response
        state["passive_response"] = analyze_passive_response(directory, configuration(directory / "prepared_config.json"))
        state["analysis_status"] = "COMPLETE"
        state.pop("response_analysis_error", None)
        report(directory, state)
        return state
    try:
        c = configuration(state["config_path"])
        if c["training"].get("observation_source") == "robot_only":
            raise ValueError("historical joint-sensor training is validation only; prepare an original_candidate robot-only session")
        positive(c["training"]["timeout_s"], "training.timeout_s")
        analysis_calibration = c["calibration"]
        leg_origin_path = state.get("leg_start_origin_path")
        if leg_origin_path:
            leg_origin = json.loads(Path(leg_origin_path).read_text(encoding="utf-8"))
            analysis_calibration = leg_origin["session_calibration"]
        analysis_reference_path = Path(
            state.get("analysis_reference_path", DEFAULT_REFERENCE_PATH)
        )
        measured, metrics = aligned_measurement(directory / "leg/leg.csv", directory / "dispatch.csv",
            analysis_reference_path, c["analysis"], {"joints": analysis_calibration})
        measured.to_csv(directory / "measured.csv", index=False)
        state["metrics"] = metrics
        candidates = []
        for item in c["training"]["candidates"]:
            item = dict(item)
            from config.bundle import resolve_config_path
            item["path"] = str(resolve_config_path(ROOT / item["path"]))
            candidates.append(item)
        reference_id = pd.read_csv(DEFAULT_REFERENCE_PATH).trajectory_id.iloc[0]
        request = dict(candidates=candidates, reference_candidate_id=reference_id,
                       experiment_scope="single_reference_calibration",
                       history=[dict(episode_id=directory.name, candidate_id=reference_id,
                                     measured_path=str(directory / "measured.csv"), endpoints=endpoints(metrics, metrics))])
        state["model"] = run_algorithm_worker(directory, request, c["training"]["timeout_s"])
        state["training_scope"] = "SINGLE_REFERENCE_CALIBRATION_NOT_CROSS_TRIAL_OPTIMIZATION"
        state.update(status="ROUND_COMPLETE", training_status="COMPLETE")
        state.pop("reason", None)
    except Exception as exc:
        state.update(training_status="FAILED", reason=str(exc))
        raise
    finally:
        report(directory, state)
    return state


def _run(directory, *, live=False, empty_load=False, capture_factory=LegCaptureProcess,
         execute_fn=run_execute):
    if live is not True:
        raise PermissionError("run --live explicitly starts robot motion")
    directory = Path(directory).resolve()
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if state["status"] != "PREPARED":
        raise ValueError("prepare a new session before running")
    if state.get("trajectory_mode") == "original_candidate" and not empty_load:
        raise ValueError("six-parameter route is an offline comparison only; prepare an independent TCP force session")
    if state.get("trajectory_mode") in ("independent_tcp", "original_candidate"):
        return _run_tcp(directory, state, empty_load=empty_load,
                        capture_factory=capture_factory, execute_fn=execute_fn)
    if bool(state.get("empty_load", False)) != bool(empty_load):
        raise ValueError("prepared session mode does not match requested run mode")
    config_path = state["config_path"]
    c = configuration(config_path)
    if c.get("trajectory", {}).get("mode") in ("independent_tcp", "original_candidate"):
        raise ValueError("legacy session cannot run in independent TCP mode; prepare a new session")
    blockers = _check_config_for_operation(
        state["config_path"], empty_load=empty_load
    )
    if blockers:
        state["blockers"] = blockers
        report(directory, state)
        raise ValueError("complete physical parameters: " + "; ".join(blockers))
    acquisition, robot = c["acquisition"], c["robot"]
    reference_path = Path(state["reference_path"]) if empty_load else DEFAULT_REFERENCE_PATH
    guard_reference_path = (
        reference_path
        if empty_load
        else Path(state.get("analysis_reference_path", DEFAULT_REFERENCE_PATH))
    )
    reference = pd.read_csv(guard_reference_path)
    empty_load_retiming = None
    trajectory_adjustment = None
    if empty_load:
        empty_load_retiming = state.get("empty_load_retiming")
        if not isinstance(empty_load_retiming, dict):
            retiming_path = directory / "empty_load_retiming.json"
            if not retiming_path.is_file():
                raise ValueError("prepared empty-load retiming metadata is missing")
            empty_load_retiming = json.loads(retiming_path.read_text(encoding="utf-8"))
    else:
        trajectory_adjustment = state.get("trajectory_adjustment")
        if not isinstance(trajectory_adjustment, dict):
            adjustment_path = directory / "trajectory_adjustment.json"
            if adjustment_path.is_file():
                trajectory_adjustment = json.loads(
                    adjustment_path.read_text(encoding="utf-8")
                )
            elif _trajectory_target_peak_z(c) is not None:
                raise ValueError(
                    "prepared trajectory adjustment is missing; prepare a new session"
                )
            else:
                trajectory_adjustment = {"rehab_z_scale": 1.0}
    conversions = {}
    physical_conversions = {}
    if not empty_load:
        calibration = c["calibration"]
        physical_conversions = {
            name: _joint_calibration(values)
            for name, values in calibration.items()
        }
        session_calibration = calibration
        leg_origin_path = state.get("leg_start_origin_path")
        if leg_origin_path:
            leg_origin = json.loads(Path(leg_origin_path).read_text(encoding="utf-8"))
            if not leg_origin.get("usable_for_motion"):
                raise ValueError("prepared leg start origin is not usable for motion")
            session_calibration = leg_origin["session_calibration"]
        conversions = {
            name: _joint_calibration(values)
            for name, values in session_calibration.items()
        }
        for name, conversion in conversions.items():
            angles = np.rad2deg(reference["q_" + name + "_rad"].to_numpy(float))
            if not np.isfinite(angles).all() or np.any(angles < conversion.safe_min_deg) or np.any(angles > conversion.safe_max_deg):
                raise ValueError(name + " reference exceeds leg range")
    capture, dispatches = None, []
    primary_failure = None
    state.update(status="RUNNING", blockers=[])
    report(directory, state)
    try:
        guard = None
        if not empty_load:
            capture = capture_factory(config_path, directory / "leg")
            capture.start()
            def guard():
                reason = capture.health_reason()
                if reason:
                    return reason
                snapshot = capture.snapshot()
                age = (time.perf_counter_ns() - snapshot["last_sample_ns"]) / 1e9
                if age < 0 or age > acquisition["max_leg_start_age_s"]:
                    return "leg sample stale"
                for name, axis in (("hip", 1), ("knee", 4)):
                    raw = snapshot["latest"][f"axis{axis}_position_raw"]
                    physical_angle = physical_conversions[name].to_degrees(raw)
                    if (
                        not np.isfinite(physical_angle)
                        or not physical_conversions[name].safe_min_deg
                        <= physical_angle
                        <= physical_conversions[name].safe_max_deg
                    ):
                        return name + " outside physical calibrated range"
                    angle = conversions[name].to_degrees(raw)
                    if not np.isfinite(angle) or not conversions[name].safe_min_deg <= angle <= conversions[name].safe_max_deg:
                        return name + " outside session trajectory range"
                    if not dispatches and abs(angle - np.rad2deg(reference["q_" + name + "_rad"].iloc[0])) > acquisition["max_start_angle_error_deg"]:
                        return name + " not at trajectory starting angle"
                if dispatches:
                    reason = _tracking_error_reason(
                        reference,
                        conversions,
                        snapshot["latest"],
                        dispatches[-1]["trajectory_time_s"],
                        acquisition["max_tracking_error_deg"],
                    )
                    if reason:
                        return reason
                return None
        def record_for(duration):
            deadline = time.perf_counter() + duration
            while time.perf_counter() < deadline:
                reason = guard()
                if reason:
                    raise RuntimeError(reason)
                time.sleep(.01)
        if not empty_load:
            record_for(acquisition["pre_roll_s"])
        anchor = load_start_anchor(directory / "anchor.json")
        def dispatched(t, start, end):
            dispatches.append(dict(trajectory_time_s=t, dispatch_start_ns=start, dispatch_end_ns=end))
        execute_kwargs = dict(robot_ip=robot["ip"], local_ip=robot["local_ip"], episode_dir=directory / "robot",
            anchor_path=directory / "anchor.json", requested_anchor_id=anchor.anchor_id,
            frame_config_path=config_path,
            safety_config_path=state.get("safety_config_path", config_path),
            trajectory_id=reference.trajectory_id.iloc[0], enable_motion=True,
            operator_confirmation=OPERATOR_CONFIRMATION,
            reference_path_override=reference_path if empty_load else None,
            rehab_z_scale=(
                float(empty_load_retiming["rehab_z_scale"])
                if empty_load
                else float(trajectory_adjustment["rehab_z_scale"])
            ),
            joint_profile_transform=(
                None
                if empty_load
                else trajectory_adjustment.get("joint_profile_transform")
            ),
            joint_limits_deg=(
                None
                if empty_load
                else {
                    name: [
                        values["safe_min_deg"],
                        values["safe_max_deg"],
                    ]
                    for name, values in c["calibration"].items()
                }
            ),
            live_wrench_config=load_live_wrench_config(config_path),
            adapter_factory=_configured_adapter_factory(local_ip=robot["local_ip"], robot_class=robot["robot_class"],
                state_interval_ms=None, collision_source=robot["collision_source"], collision_event_max_age_s=robot["collision_event_max_age_s"]),
            wrench_hz=acquisition["wrench_hz"], collision_source=robot["collision_source"],
            collision_event_max_age_s=robot["collision_event_max_age_s"], prepare_realtime=True,
            dispatch_observer=dispatched)
        if guard is not None:
            execute_kwargs["execution_guard"] = guard
        result = execute_fn(**execute_kwargs)
        if result.get("completed") is not True or not dispatches:
            raise RuntimeError("robot did not complete the trajectory")
        if not empty_load:
            record_for(acquisition["post_roll_s"])
            capture.stop()
            capture = None
        state.update(status="EMPTY_LOAD_COMPLETE" if empty_load else "CAPTURE_COMPLETE",
                     execution=result)
    except BaseException as exc:
        primary_failure = exc
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        try:
            if capture is not None:
                capture.stop()
        except BaseException as exc:
            cleanup_reason = "capture cleanup: " + str(exc)
            if primary_failure is None:
                state.update(status="FAILED", reason=cleanup_reason)
                raise
            state["capture_cleanup_error"] = cleanup_reason
        finally:
            if dispatches:
                pd.DataFrame(dispatches).to_csv(directory / "dispatch.csv", index=False)
            report(directory, state)
    if not empty_load:
        state["quality"] = inspect_file(directory / "leg/leg.csv", expected_hz=c["leg"]["expected_hz"])
        report(directory, state)
    if not empty_load and c["training"]["enabled"]:
        try:
            return analyze(directory)
        except Exception as exc:
            state.update(training_status="FAILED", reason=str(exc))
            report(directory, state)
            raise
    return state


def run(directory, *, live=False, empty_load=False, capture_factory=LegCaptureProcess,
        execute_fn=run_execute):
    if live is not True:
        raise PermissionError("run --live explicitly starts robot motion")
    lock = Path(directory).resolve() / "running.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        try:
            result = _run(directory, live=live, empty_load=empty_load,
                          capture_factory=capture_factory, execute_fn=execute_fn)
        except BaseException as exc:
            from measurement_validation.force_trial_ledger import finish
            try:
                finish(directory, ok=False, error=f"{type(exc).__name__}: {exc}")
            except BaseException as ledger_error:
                exc.add_note("force attempt ledger completion failed: " + str(ledger_error))
            raise
        from measurement_validation.force_trial_ledger import finish
        finish(directory, ok=True)
        return result
    finally:
        lock.unlink(missing_ok=True)


def run_from_current(config_path, directory, *, live=False, empty_load=False):
    """Capture this run's current pose, then execute its relative trajectory."""
    if live is not True:
        raise PermissionError("run --from-current --live explicitly starts robot motion")
    blockers = _check_config_for_operation(config_path, empty_load=empty_load)
    if blockers:
        raise ValueError("complete physical parameters: " + "; ".join(blockers))
    if empty_load:
        prepare(config_path, directory, empty_load=True)
        return run(directory, live=True, empty_load=True)
    prepare(config_path, directory)
    return run(directory, live=True)


def repeat(source_directory, directory, *, live=False, empty_load=None,
           use_trained=False, evaluation=False,
           capture_fn=run_capture, leg_origin_capture_fn=_capture_leg_start_origin,
           capture_factory=LegCaptureProcess, execute_fn=run_execute):
    """Create a new anchored session with the source TCP plan and optionally run it."""
    source_directory = Path(source_directory).resolve()
    target_directory = Path(directory).resolve()
    source_state = json.loads((source_directory / "round_status.json").read_text(encoding="utf-8"))
    completed = {"EMPTY_LOAD_COMPLETE", "CAPTURE_COMPLETE", "ROUND_COMPLETE", "COMPLETE"}
    if source_state.get("status") not in completed:
        raise ValueError("repeat source must be a completed session")
    if source_state.get("trajectory_mode") not in ("independent_tcp", "original_candidate"):
        raise ValueError("repeat currently requires an independent TCP source session")
    source_plan = source_state.get("tcp_plan")
    if not isinstance(source_plan, dict):
        raise ValueError("repeat source is missing frozen TCP plan")
    if source_state["trajectory_mode"] == "original_candidate":
        if source_plan.get("mode") != "original_candidate" or not source_plan.get("candidate_id"):
            raise ValueError("repeat source is missing its original candidate")
    elif not isinstance(source_plan.get("tcp_settings"), dict):
        raise ValueError("repeat source is missing frozen TCP settings")
    source_config = Path(source_state["config_path"])
    config = configuration(source_config)
    prepared = source_directory / "prepared_config.json"
    if not prepared.is_file() or configuration(prepared) != config:
        raise ValueError("repeat source configuration changed; use a reviewed source session")
    mode_empty = bool(source_state.get("empty_load", False)) if empty_load is None else bool(empty_load)
    if mode_empty != bool(source_state.get("empty_load", False)):
        raise ValueError("repeat mode must match the source session")
    force_mode = source_state.get("learning_mode") == "tcp_force"
    study_root = None
    if evaluation and (mode_empty or not force_mode or use_trained):
        raise ValueError("--evaluation is a separate exact loaded force repeat, without --trained")
    if not mode_empty and force_mode:
        from measurement_validation.force_trial_ledger import continuation
        if not use_trained and not evaluation:
            raise ValueError("exact force repeats require explicit --evaluation; they are not adaptive search trials")
        study_root = continuation(source_directory, config, evaluation=evaluation)
        if evaluation:
            from scripts.tcp_force_training import verify_training, verify_evaluation
            (verify_evaluation if source_state.get("trial_purpose") == "evaluation" else verify_training)(source_directory)
            settings_evaluation = {"candidate_id": source_state["force_training_binding"]["candidate_id"]}
    settings = ({"candidate_id": source_plan["candidate_id"]}
        if source_state["trajectory_mode"] == "original_candidate" else source_plan["tcp_settings"])
    if evaluation:
        settings = settings_evaluation
    if mode_empty and source_state.get("learning_mode") == "tcp_force":
        settings = {"candidate_id": source_state["force_training_binding"]["candidate_id"]}
    if (not mode_empty or use_trained) and not evaluation:
        force_mode = source_state.get("learning_mode") == "tcp_force"
        if force_mode:
            from scripts.tcp_force_training import selected_candidate as selected_tcp_settings, verify_training as verify_original_training
        else:
            from scripts.original_training_adapter import selected_tcp_settings, verify_original_training
        # Even exact repeats require the preceding with-leg trial to be trained.
        suggestion = selected_tcp_settings(source_directory) if use_trained else None
        if not mode_empty and source_state.get("training_status") != "COMPLETE":
            raise ValueError("train the source with-leg motion before repeating")
        if not mode_empty and not use_trained:
            verify_original_training(source_directory)
            if force_mode:
                settings = {"candidate_id": source_state["force_training_binding"]["candidate_id"]}
            else:
                raise ValueError("DUPLICATE_CANDIDATE_NO_REPEAT_NOISE_MODEL: exact repeats cannot be appended to the unchanged original algorithm; use an independent reference recording for validation")
        if use_trained:
            settings = suggestion
    repeat_index = int(source_state.get("repeat_index", 0)) + 1
    prepare(source_config, target_directory, empty_load=mode_empty,
            trajectory_settings=settings,
            parent_session=None if evaluation else source_directory,
            repeat_index=None if evaluation else repeat_index,
            capture_fn=capture_fn, leg_origin_capture_fn=leg_origin_capture_fn)
    prepared_state = json.loads((target_directory / "round_status.json").read_text(encoding="utf-8"))
    if study_root is not None:
        prepared_state.update(force_study_root=str(study_root), trial_purpose="evaluation" if evaluation else "adaptive")
    if evaluation:
        prepared_state.update(evaluation_source=str(source_directory), enters_adaptive_history=False)
    prepared_state["repeat_plan_source"] = source_state.get("learning_mode", "original_algorithm") if use_trained else "exact_parameters"
    report(target_directory, prepared_state)
    if live:
        return run(target_directory, live=True, empty_load=mode_empty,
                   capture_factory=capture_factory, execute_fn=execute_fn)
    return json.loads((target_directory / "round_status.json").read_text(encoding="utf-8"))


def complete(directory):
    """Promote a successful recording to a final, reviewable COMPLETE state."""
    directory = Path(directory).resolve()
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if state.get("status") == "COMPLETE" and state.get("trajectory_mode") not in ("independent_tcp", "original_candidate"):
        return state
    if state.get("status") not in {"EMPTY_LOAD_COMPLETE", "CAPTURE_COMPLETE", "ROUND_COMPLETE", "COMPLETE"}:
        raise ValueError("complete requires a successful executed or analyzed session")
    execution = state.get("execution", {})
    if (state.get("trajectory_mode") in ("independent_tcp", "original_candidate") or execution) and (execution.get("completed") is not True or
                      execution.get("same_session_realtime_cleanup_completed") is not True):
        raise ValueError("execution completion and realtime cleanup evidence are required")
    if state.get("trajectory_mode") in ("independent_tcp", "original_candidate"):
        if not state.get("empty_load"):
            if state.get("learning_mode") == "tcp_force":
                if state.get("trial_purpose") == "evaluation":
                    from scripts.tcp_force_training import verify_evaluation as verify_original_training
                else:
                    from scripts.tcp_force_training import verify_training as verify_original_training
            else:
                from scripts.original_training_adapter import verify_original_training
            if state.get("training_status") != "COMPLETE":
                raise ValueError("passive training must complete before final completion")
            verify_original_training(directory)
        if state.get("empty_load"):
            from measurement_validation.robot_execution import EVIDENCE, audit_robot_execution
            required = tuple(directory / name for name in EVIDENCE)
            missing = [str(path.relative_to(directory)) for path in required if not path.is_file()]
            if missing:
                raise ValueError("completion evidence missing: " + ", ".join(missing))
            state["completion_trajectory_audit"] = audit_robot_execution(directory, allow_empty=True)
        else:
            if state.get("learning_mode") == "tcp_force" and not configuration(directory / "prepared_config.json").get("validation", {}).get("capture_dummy_leg", True):
                required = ()
            else:
                required = (directory / "leg/leg.csv", directory / "passive_response.csv",
                            directory / "passive_response_summary.json")
            if required and not (directory / "passive_response_summary.json").is_file():
                if state.get("learning_mode") == "tcp_force":
                    from measurement_validation.passive_response import analyze_passive_response
                    state["passive_response"] = analyze_passive_response(directory, configuration(directory / "prepared_config.json"))
                    state.pop("response_analysis_error", None)
                    report(directory, state)
                else:
                    analyze(directory)
                state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
            if state.get("response_analysis_error"):
                raise ValueError("resolve passive response analysis before completion")
        missing = [str(path.relative_to(directory)) for path in required if not path.is_file()]
        if missing:
            raise ValueError("completion evidence missing: " + ", ".join(missing))
    state["status"] = "COMPLETE"
    state.setdefault("completion", {"reviewed": False, "completed_at_host_time_s": time.perf_counter()})
    report(directory, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "observe", "prepare", "run", "repeat", "analyze", "train", "complete"))
    parser.add_argument("--config", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--source", type=Path, help="completed session used as a repeat source")
    parser.add_argument("--trained", action="store_true", help="prepare the trained policy's next TCP candidate")
    parser.add_argument("--evaluation", action="store_true", help="separate exact force repeat; never append to adaptive GP history")
    parser.add_argument("--training-mode", choices=("auto", "passive", "joint"), default="auto")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--from-current", action="store_true", help="run from a newly captured current pose in a new session")
    parser.add_argument("--empty-load", action="store_true",
                        help="prepare, run, or repeat a slow trajectory without dummy-leg capture")
    parser.add_argument("--duration", type=float, default=10., help="observation duration in seconds")
    args = parser.parse_args(argv)
    if args.from_current and args.action != "run":
        parser.error("--from-current requires run")
    if args.empty_load and args.action not in ("prepare", "run", "repeat"):
        parser.error("--empty-load requires prepare or run")
    if args.action == "repeat" and args.source is None:
        parser.error("--source is required for repeat")
    if args.trained and args.action != "repeat":
        parser.error("--trained requires repeat")
    if args.evaluation and (args.action != "repeat" or args.trained):
        parser.error("--evaluation requires repeat without --trained")
    if args.training_mode != "auto" and args.action != "train":
        parser.error("--training-mode requires train")
    if args.action == "complete" and args.live:
        parser.error("complete is offline and does not accept --live")
    if args.action == "check":
        problems = check_config(args.config, require_physical_files=True)
        print("\n".join(problems) if problems else "READY")
        return int(bool(problems))
    if args.session is None:
        parser.error("--session is required")
    if args.action == "observe":
        state = observe(args.config, args.session, duration_s=args.duration)
    elif args.action == "prepare":
        state = prepare(args.config, args.session, empty_load=args.empty_load)
    elif args.action == "run":
        run_options = dict(live=args.live)
        if args.empty_load:
            run_options["empty_load"] = True
        state = (run_from_current(args.config, args.session, **run_options)
                 if args.from_current else run(args.session, **run_options))
    elif args.action == "repeat":
        state = repeat(args.source, args.session, live=args.live,
                       use_trained=args.trained,
                       evaluation=args.evaluation,
                       empty_load=(True if args.empty_load else None))
    elif args.action == "complete":
        state = complete(args.session)
    elif args.action == "train":
        state = train_session(args.session, training_mode=args.training_mode)
    else:
        state = analyze(args.session)
    print(state["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
