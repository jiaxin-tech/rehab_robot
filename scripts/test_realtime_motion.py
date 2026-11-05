"""Prepare and run a feedback-verified millimetre-scale ROKAE RT smoke test.

``prepare`` is observation-only.  ``run`` is single-use, requires explicit
live/unloaded/operator confirmations, and commands a 0.5--2.0 mm base +Z
minimum-jerk move followed by a return to the captured start pose.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from collection.twincat_leg import atomic_json
from config.bundle import BUNDLE_PATH, configuration
from control.execution_preflight import OPERATOR_CONFIRMATION
from control.realtime_link_diagnostic import (
    MAX_STEP_M,
    MIN_STEP_M,
    REALTIME_LINK_PROFILE,
    REALTIME_LINK_TRAJECTORY_ID,
    RealtimeLinkTrackingGuard,
    audit_realtime_link_trajectory,
    build_realtime_link_trajectory,
    evaluate_realtime_link_live,
    evaluate_realtime_link_offline,
)
from control.robot_trajectory_executor import RokaeMotionExecutor
from control.start_anchor import load_start_anchor
from hardware.safety_events import validate_collision_source
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.acquire_robot_data import (
    _configured_adapter_factory,
    build_live_wrench_provider,
    load_live_wrench_config,
)
from scripts.capture_start_anchor import run_capture
from scripts.lab_session import _empty_load_motion, check_config
from scripts.run_rehab_experiment import (
    _cleanup_realtime_session,
    _default_motion_factory,
    _prepare_realtime_session,
)
from utils.clock import TIMESTAMP_SOURCE
from utils.provenance import current_git_commit


AdapterFactory = Callable[[str], Any]
MotionFactory = Callable[[Any], Any]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _adapter_factory(config: dict[str, Any]) -> AdapterFactory:
    robot = config["robot"]
    return _configured_adapter_factory(
        local_ip=robot["local_ip"],
        robot_class=robot["robot_class"],
        state_interval_ms=None,
        collision_source=robot["collision_source"],
        collision_event_max_age_s=robot["collision_event_max_age_s"],
    )


def prepare(
    session: str | Path,
    *,
    config_path: str | Path = BUNDLE_PATH,
    step_mm: float = 1.0,
    capture_fn=run_capture,
) -> dict[str, Any]:
    """Capture a start pose and freeze the diagnostic trajectory; no motion."""

    directory = Path(session).resolve()
    resolved_config = Path(config_path).resolve()
    step_m = float(step_mm) / 1000.0
    if not MIN_STEP_M <= step_m <= MAX_STEP_M:
        raise ValueError(
            f"step_mm must be between {MIN_STEP_M * 1000:g} and "
            f"{MAX_STEP_M * 1000:g}"
        )
    blockers = check_config(resolved_config, empty_load=True)
    if blockers:
        raise ValueError("configuration blocked: " + "; ".join(blockers))
    config = configuration(resolved_config)
    directory.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "PREPARING",
        "motion_commanded": False,
        "config_path": str(resolved_config),
        "step_mm": float(step_mm),
    }
    atomic_json(directory / "status.json", state)
    try:
        robot = config["robot"]
        capture_fn(
            robot_ip=robot["ip"],
            output_path=directory / "anchor.json",
            adapter_factory=_adapter_factory(config),
            notes="realtime link diagnostic start anchor",
        )
        anchor = load_start_anchor(directory / "anchor.json")
        trajectory = build_realtime_link_trajectory(
            anchor.tcp_pose_base,
            step_m=step_m,
            q_hip_ref=anchor.reference_start_q_hip,
            q_knee_ref=anchor.reference_start_q_knee,
        )
        safety = ExperimentSafetyConfig(**_empty_load_motion(config))
        reasons, speed, acceleration = audit_realtime_link_trajectory(
            trajectory,
            safety,
            expected_trajectory_id=REALTIME_LINK_TRAJECTORY_ID,
        )
        if reasons:
            raise PermissionError("diagnostic trajectory audit: " + ";".join(reasons))
        trajectory_path = directory / "trajectory.csv"
        trajectory.to_csv(trajectory_path, index=False)
        manifest = {
            "schema_version": 1,
            "trajectory_id": REALTIME_LINK_TRAJECTORY_ID,
            "trajectory_profile": REALTIME_LINK_PROFILE,
            "anchor_id": anchor.anchor_id,
            "step_m": step_m,
            "maximum_tcp_speed_m_s": speed,
            "maximum_tcp_acceleration_m_s2": acceleration,
            "trajectory_sha256": _sha256(trajectory_path),
            "anchor_sha256": _sha256(directory / "anchor.json"),
            "config_sha256": _sha256(resolved_config),
        }
        atomic_json(directory / "manifest.json", manifest)
        state.update(
            status="PREPARED",
            anchor_id=anchor.anchor_id,
            manifest=manifest,
        )
    except BaseException as exc:
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        atomic_json(directory / "status.json", state)
    return state


def execute_realtime_link(
    *,
    episode_dir: str | Path,
    config_path: str | Path,
    anchor_path: str | Path,
    trajectory: pd.DataFrame,
    requested_anchor_id: str,
    step_m: float,
    enable_motion: bool,
    confirm_unloaded: bool,
    operator_confirmation: str,
    adapter_factory: AdapterFactory | None = None,
    motion_factory: MotionFactory | None = None,
) -> dict[str, Any]:
    """Execute the already frozen diagnostic path with all live safety gates."""

    config = configuration(config_path)
    robot = config["robot"]
    acquisition_config = config["acquisition"]
    validate_collision_source(
        robot["collision_source"],
        robot["collision_event_max_age_s"],
    )
    anchor = load_start_anchor(anchor_path)
    safety = ExperimentSafetyConfig(**_empty_load_motion(config))
    offline = evaluate_realtime_link_offline(
        enable_motion=enable_motion,
        confirm_unloaded=confirm_unloaded,
        operator_confirmation=operator_confirmation,
        requested_anchor_id=requested_anchor_id,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
    )
    # Invalid confirmation or content is rejected before constructing an
    # adapter, starting a logger, or touching a controller session.
    offline.require_allowed()
    trajectory_step_m = float(
        trajectory["tcp_z_base"].max() - trajectory["tcp_z_base"].iloc[0]
    )
    if not abs(trajectory_step_m - float(step_m)) <= 1e-10:
        raise PermissionError("prepared step does not match diagnostic trajectory")
    live_wrench_config = load_live_wrench_config(config_path)
    if "rtcontrolling" not in live_wrench_config.get("allowed_operation_states", []):
        raise ValueError("RT acquisition configuration required")
    wrench_hz = float(acquisition_config["wrench_hz"])
    if wrench_hz <= 0.0:
        raise ValueError("positive acquisition.wrench_hz required")

    logger = EpisodeLogger(
        episode_dir,
        {
            "git_commit": current_git_commit(),
            "robot_ip": robot["ip"],
            "robot_local_ip": robot["local_ip"],
            "trajectory_id": REALTIME_LINK_TRAJECTORY_ID,
            "trajectory_profile": REALTIME_LINK_PROFILE,
            "start_anchor": anchor.to_dict(),
            "experiment_safety": safety.to_dict(),
            "operator_confirmation": operator_confirmation,
            "confirm_unloaded": confirm_unloaded,
            "timestamp_source": TIMESTAMP_SOURCE,
            "robot_execution_approved": False,
            "motion_api_physical_validation": False,
        },
    )
    adapter: Any | None = None
    acquisition: RealRobotAcquisition | None = None
    wrench_provider: Any | None = None
    executor: RokaeMotionExecutor | None = None
    tracker: RealtimeLinkTrackingGuard | None = None
    completed = False
    realtime_prepared = False
    result = None
    try:
        logger.start()
        factory = adapter_factory or _adapter_factory(config)
        adapter = factory(robot["ip"])
        realtime_prepared = True
        if safety.rt_filter_hz is None or safety.rt_network_tolerance_percent is None:
            raise PermissionError("reviewed realtime parameters unavailable")
        _prepare_realtime_session(
            adapter,
            filter_hz=safety.rt_filter_hz,
            network_tolerance_percent=safety.rt_network_tolerance_percent,
        )
        wrench_provider = build_live_wrench_provider(
            Path(episode_dir),
            config=live_wrench_config,
            wrench_hz=wrench_hz,
        )
        acquisition = RealRobotAcquisition(
            adapter,
            logger,
            wrench_provider=wrench_provider,
            wrench_hz=wrench_hz,
        )
        acquisition.start(manage_connection=False)
        startup_timeout_s = float(live_wrench_config["budgets"]["startup_s"]) + 0.5
        if not acquisition.wait_until_healthy(
            timeout_s=startup_timeout_s,
            arm_runtime_grace=True,
        ):
            raise RuntimeError(
                "state/wrench streams did not become healthy: "
                + acquisition.latest_health().invalid_reason
            )
        preflight = evaluate_realtime_link_live(
            offline=offline,
            acquisition_health=acquisition.latest_health(),
            logger=logger,
            robot_adapter=adapter,
            current_tcp_pose_base=adapter.read_tcp_pose(),
            anchor=anchor,
            safety=safety,
        )
        preflight.require_allowed()
        logger.update_metadata(
            {
                "runtime_preflight": preflight.as_metadata(),
                "robot_state_summary_before_motion": preflight.runtime_robot_summary,
                "robot_execution_approved": False,
            }
        )
        tracker = RealtimeLinkTrackingGuard(adapter, anchor.tcp_pose_base, step_m)
        motion_adapter = (motion_factory or _default_motion_factory)(adapter)
        executor = RokaeMotionExecutor(
            motion_adapter,
            acquisition,
            logger,
            safety,
            execution_guard=tracker.check,
            dispatch_observer=tracker.observe_dispatch,
            use_windows_scheduler_priority=False,
            trajectory_profile=REALTIME_LINK_PROFILE,
        )
        result = executor.execute(trajectory, preflight)
        tracker.require_complete()
        logger.update_metadata(
            {
                "execution_result": asdict(result),
                "tracking_verification": tracker.as_metadata(),
                "motion_api_physical_validation": True,
            }
        )
        completed = True
    except KeyboardInterrupt as exc:
        if executor is not None:
            try:
                executor.request_stop("operator_keyboard_interrupt")
            except BaseException as stop_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"additional software-stop failure: {stop_exc}")
        raise
    except BaseException as exc:
        if executor is not None:
            try:
                executor.request_stop(f"diagnostic_exception:{type(exc).__name__}:{exc}")
            except BaseException as stop_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"additional software-stop failure: {stop_exc}")
        if not logger.write_timeout_event.is_set() and logger.healthy:
            try:
                logger.mark_failed(f"diagnostic_exception:{type(exc).__name__}:{exc}")
            except BaseException:
                pass
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        logger_may_close = not logger.write_timeout_event.is_set()
        if acquisition is not None:
            try:
                acquisition.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
                logger_may_close = bool(
                    logger_may_close and not acquisition.live_producer_names
                )
        if realtime_prepared and adapter is not None:
            if acquisition is not None and acquisition.live_producer_names:
                native = getattr(adapter, "native_robot", None)
                for action in (
                    lambda: native.stop_realtime(switch_to_nrt=True),
                    lambda: native.disable(),
                ):
                    try:
                        action()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                cleanup_errors.append(
                    RuntimeError("live producers remain; disconnect unconfirmed")
                )
            else:
                cleanup_errors.extend(_cleanup_realtime_session(adapter))
        if logger.pending_write_count:
            logger_may_close = False
        if logger_may_close:
            try:
                logger.close(
                    completed=completed and not cleanup_errors,
                    stop_reason=(
                        None
                        if completed and not cleanup_errors
                        else "diagnostic_failed_or_stopped"
                    ),
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            import sys

            active_exception = sys.exc_info()[1]
            detail = ";".join(
                f"{type(item).__name__}:{item}" for item in cleanup_errors
            )
            if active_exception is not None and hasattr(active_exception, "add_note"):
                active_exception.add_note("diagnostic cleanup failure: " + detail)
            elif active_exception is None:
                raise RuntimeError("diagnostic cleanup failure: " + detail)
    return {
        "completed": completed,
        "trajectory_id": REALTIME_LINK_TRAJECTORY_ID,
        "commands_dispatched": result.commands_dispatched if result else 0,
        "stop_reason": result.stop_reason if result else None,
        "tracking_verification": tracker.as_metadata() if tracker else None,
        "same_session_realtime_cleanup_completed": realtime_prepared,
    }


def run(
    session: str | Path,
    *,
    live: bool,
    confirm_unloaded: bool,
    operator_confirmation: str,
    adapter_factory: AdapterFactory | None = None,
    motion_factory: MotionFactory | None = None,
) -> dict[str, Any]:
    directory = Path(session).resolve()
    status_path = directory / "status.json"
    state = _load_json(status_path)
    if state.get("status") != "PREPARED":
        raise ValueError("prepare a new diagnostic session before running")
    manifest = _load_json(directory / "manifest.json")
    config_path = Path(state["config_path"]).resolve()
    checks = (
        (directory / "trajectory.csv", manifest["trajectory_sha256"], "trajectory"),
        (directory / "anchor.json", manifest["anchor_sha256"], "anchor"),
        (config_path, manifest["config_sha256"], "configuration"),
    )
    for path, expected_hash, label in checks:
        if _sha256(path) != expected_hash:
            raise PermissionError(f"prepared {label} changed; prepare a new session")
    trajectory = pd.read_csv(directory / "trajectory.csv")
    # Confirmation and frozen-content failures leave the prepared session
    # untouched and happen before constructing any hardware object.
    config = configuration(config_path)
    anchor = load_start_anchor(directory / "anchor.json")
    safety = ExperimentSafetyConfig(**_empty_load_motion(config))
    evaluate_realtime_link_offline(
        enable_motion=live,
        confirm_unloaded=confirm_unloaded,
        operator_confirmation=operator_confirmation,
        requested_anchor_id=manifest["anchor_id"],
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
    ).require_allowed()
    state.update(status="RUNNING", motion_commanded=True)
    atomic_json(status_path, state)
    try:
        result = execute_realtime_link(
            episode_dir=directory / "robot",
            config_path=config_path,
            anchor_path=directory / "anchor.json",
            trajectory=trajectory,
            requested_anchor_id=manifest["anchor_id"],
            step_m=float(manifest["step_m"]),
            enable_motion=live,
            confirm_unloaded=confirm_unloaded,
            operator_confirmation=operator_confirmation,
            adapter_factory=adapter_factory,
            motion_factory=motion_factory,
        )
        if result.get("completed") is not True:
            raise RuntimeError("realtime link diagnostic did not complete")
        state.update(status="COMPLETE", result=result)
        state.pop("reason", None)
        return state
    except BaseException as exc:
        state.update(status="FAILED", reason=str(exc))
        raise
    finally:
        atomic_json(status_path, state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="feedback-verified 0.5--2 mm ROKAE realtime link test"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="capture a start anchor and trajectory without commanding motion",
    )
    prepare_parser.add_argument("--session", required=True)
    prepare_parser.add_argument("--config", default=str(BUNDLE_PATH))
    prepare_parser.add_argument("--step-mm", type=float, default=1.0)
    run_parser = subparsers.add_parser("run", help="run one prepared test")
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument("--live", action="store_true")
    run_parser.add_argument("--confirm-unloaded", action="store_true")
    run_parser.add_argument("--operator-confirmation", default="")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.session, config_path=args.config, step_mm=args.step_mm)
    else:
        result = run(
            args.session,
            live=args.live,
            confirm_unloaded=args.confirm_unloaded,
            operator_confirmation=args.operator_confirmation,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
