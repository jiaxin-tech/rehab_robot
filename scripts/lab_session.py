"""Single entry point for recording and analyzing dummy-leg experiments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from config.bundle import BUNDLE_PATH, ROOT, configuration, load_section
from collection.twincat_leg import LegCaptureProcess, atomic_json, positive, validate_config
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


def _empty_load_motion(c):
    speed, acceleration, _ = _empty_load_settings(c)
    motion = dict(c["motion"])
    motion.update(
        max_tcp_speed_m_s=speed,
        max_tcp_acceleration_m_s2=acceleration,
    )
    return motion


def check_config(config_path=BUNDLE_PATH, *, empty_load=False):
    """Validate actual device settings and limits without connecting hardware."""
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
    if not empty_load:
        check("ADS", lambda: validate_config(load_section("twincat_leg_capture", config_path=config_path)))
        if not Path(load_section("twincat_leg_capture", config_path=config_path)["ads_dll_path"]).is_file():
            problems.append("leg.ads_dll_path: DLL not found")
    acquisition_fields = (("wrench_hz",) if empty_load else
        ("pre_roll_s", "post_roll_s", "wrench_hz", "max_leg_start_age_s", "max_start_angle_error_deg"))
    for name in acquisition_fields:
        check("acquisition." + name, lambda name=name: positive(c["acquisition"].get(name), name))
    check("wrench", lambda: load_live_wrench_config(config_path))
    def calibration():
        for name, values in c["calibration"].items():
            JointCalibration(**{k: values[k] for k in ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")})
        if set(c["calibration"]) != {"hip", "knee"}:
            raise ValueError("hip and knee calibration required")
    if not empty_load:
        check("calibration", calibration)
    if c["training"]["enabled"]:
        check("analysis", lambda: validate_analysis(c["analysis"], {"joints": c["calibration"]}))
        check("training.timeout_s", lambda: positive(c["training"]["timeout_s"], "timeout_s"))
    return problems


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
    report(directory, state)
    try:
        capture = capture_factory(config_path, directory / "leg")
        capture.start()
        result = acquire_fn(robot_ip=robot["ip"], episode_dir=directory / "robot",
            duration_s=duration_s, mode="live", live_wrench_config=wrench,
            wrench_hz=c["acquisition"]["wrench_hz"],
            adapter_factory=_configured_adapter_factory(local_ip=robot["local_ip"],
                robot_class=robot["robot_class"], state_interval_ms=None,
                collision_source=robot["collision_source"],
                collision_event_max_age_s=robot["collision_event_max_age_s"]),
            observation_guard=capture.health_reason)
        if result.get("completed") is not True:
            raise RuntimeError("robot observation incomplete")
        reason = capture.health_reason()
        if reason:
            raise RuntimeError(reason)
        state["observation"] = result
    except BaseException as exc:
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        try:
            if capture is not None:
                capture.stop()
        except BaseException as exc:
            state.update(status="FAILED", reason="capture cleanup: " + str(exc))
            raise
        finally:
            report(directory, state)
    try:
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
    baseline, audit, _ = build_start_anchored_relative_trajectory(
        reference,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    if not audit.trajectory_valid:
        raise ValueError("empty-load source trajectory is invalid")
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
        "dummy_leg_required": False,
        "training_allowed": False,
    }
    atomic_json(directory / "empty_load_retiming.json", metadata)
    return metadata


def prepare(config_path, directory, *, capture_fn=run_capture, empty_load=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config_path = Path(config_path).resolve()
    c = configuration(config_path)
    state = dict(status="PREPARING", config_path=str(config_path), empty_load=bool(empty_load))
    report(directory, state)
    try:
        r = c["robot"]
        capture_fn(robot_ip=r["ip"], output_path=directory / "anchor.json",
                   adapter_factory=_configured_adapter_factory(local_ip=r["local_ip"], robot_class=r["robot_class"], state_interval_ms=None,
                       collision_source=r["collision_source"], collision_event_max_age_s=r["collision_event_max_age_s"]))
        reference = DEFAULT_REFERENCE_PATH
        if empty_load:
            empty_load_metadata = _prepare_empty_load_artifacts(config_path, directory)
            reference = Path(empty_load_metadata["reference_path"])
            state["empty_load_retiming"] = empty_load_metadata
            state["reference_path"] = str(reference.resolve())
            state["safety_config_path"] = empty_load_metadata["safety_path"]
        preview_trajectory(anchor=directory / "anchor.json", frame_config=config_path,
                           reference=reference, output_dir=directory / "preview")
        blockers = (check_config(config_path, empty_load=True) if empty_load
                    else check_config(config_path))
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


def analyze(directory):
    directory = Path(directory).resolve()
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if state["status"] not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE"):
        raise ValueError("a completed recording is required")
    try:
        c = configuration(state["config_path"])
        positive(c["training"]["timeout_s"], "training.timeout_s")
        measured, metrics = aligned_measurement(directory / "leg/leg.csv", directory / "dispatch.csv",
            DEFAULT_REFERENCE_PATH, c["analysis"], {"joints": c["calibration"]})
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
                       history=[dict(episode_id=directory.name, candidate_id=reference_id,
                                     measured_path=str(directory / "measured.csv"), endpoints=endpoints(metrics, metrics))])
        state["model"] = run_algorithm_worker(directory, request, c["training"]["timeout_s"])
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
    if bool(state.get("empty_load", False)) != bool(empty_load):
        raise ValueError("prepared session mode does not match requested run mode")
    config_path = state["config_path"]
    c = configuration(config_path)
    blockers = (check_config(state["config_path"], empty_load=True) if empty_load
                else check_config(state["config_path"]))
    if blockers:
        state["blockers"] = blockers
        report(directory, state)
        raise ValueError("complete physical parameters: " + "; ".join(blockers))
    acquisition, robot = c["acquisition"], c["robot"]
    reference_path = Path(state["reference_path"]) if empty_load else DEFAULT_REFERENCE_PATH
    reference = pd.read_csv(reference_path)
    conversions = {}
    if not empty_load:
        calibration = c["calibration"]
        conversions = {name: JointCalibration(**{k: v[k] for k in ("counts_per_degree", "zero_counts", "flexion_sign", "safe_min_deg", "safe_max_deg")}) for name, v in calibration.items()}
        for name, conversion in conversions.items():
            angles = np.rad2deg(reference["q_" + name + "_rad"].to_numpy(float))
            if not np.isfinite(angles).all() or np.any(angles < conversion.safe_min_deg) or np.any(angles > conversion.safe_max_deg):
                raise ValueError(name + " reference exceeds leg range")
    capture, dispatches = None, []
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
                    angle = conversions[name].to_degrees(snapshot["latest"][f"axis{axis}_position_raw"])
                    if not np.isfinite(angle) or not conversions[name].safe_min_deg <= angle <= conversions[name].safe_max_deg:
                        return name + " outside calibrated range"
                    if not dispatches and abs(angle - np.rad2deg(reference["q_" + name + "_rad"].iloc[0])) > acquisition["max_start_angle_error_deg"]:
                        return name + " not at trajectory starting angle"
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
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        try:
            if capture is not None:
                capture.stop()
        except BaseException as exc:
            state.update(status="FAILED", reason="capture cleanup: " + str(exc))
            raise
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
        return _run(directory, live=live, empty_load=empty_load,
                    capture_factory=capture_factory, execute_fn=execute_fn)
    finally:
        lock.unlink(missing_ok=True)


def run_from_current(config_path, directory, *, live=False, empty_load=False):
    """Capture this run's current pose, then execute its relative trajectory."""
    if live is not True:
        raise PermissionError("run --from-current --live explicitly starts robot motion")
    blockers = (check_config(config_path, empty_load=True) if empty_load
                else check_config(config_path))
    if blockers:
        raise ValueError("complete physical parameters: " + "; ".join(blockers))
    if empty_load:
        prepare(config_path, directory, empty_load=True)
        return run(directory, live=True, empty_load=True)
    prepare(config_path, directory)
    return run(directory, live=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "observe", "prepare", "run", "train"))
    parser.add_argument("--config", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--from-current", action="store_true", help="run from a newly captured current pose in a new session")
    parser.add_argument("--empty-load", action="store_true",
                        help="prepare or run a separately retimed slow trajectory without dummy-leg capture")
    parser.add_argument("--duration", type=float, default=10., help="observation duration in seconds")
    args = parser.parse_args(argv)
    if args.from_current and args.action != "run":
        parser.error("--from-current requires run")
    if args.empty_load and args.action not in ("prepare", "run"):
        parser.error("--empty-load requires prepare or run")
    if args.action == "check":
        problems = check_config(args.config)
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
    else:
        state = analyze(args.session)
    print(state["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
