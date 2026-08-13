"""Unified preview, observation-only acquire, and gated execute runner.

Only ``--mode execute`` together with ``--enable-motion`` can reach the motion
adapter.  All offline review gates are evaluated before the adapter is created
or a robot connection is attempted.  The execute path assumes the operator has
prepared mode and servo power externally; it never powers on or moves to start.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from config import settings
from control.execution_preflight import (
    OPERATOR_CONFIRMATION,
    evaluate_execution_preflight,
    evaluate_offline_execution_request,
)
from control.robot_trajectory_executor import RokaeMotionExecutor
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import (
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from lower_limb_sim.formal_protocol import (
    FORMAL_HIP_ROM_DEG as APPROVED_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG as APPROVED_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
    MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from safety.experiment_safety import load_experiment_safety_config
from scripts.acquire_robot_data import run_acquisition
from scripts.preview_rehab_trajectory import preview_trajectory
from utils.clock import TIMESTAMP_SOURCE
from utils.provenance import current_git_commit


AdapterFactory = Callable[[str], Any]
MotionFactory = Callable[[Any], Any]
DEFAULT_FRAME_CONFIG = Path(__file__).resolve().parents[1] / "config" / "rehab_frame_config.json"
DEFAULT_SAFETY_CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment_safety.json"


def _reference_path(trajectory_id: str) -> Path:
    source_candidate_directory = (
        Path(__file__).resolve().parents[1]
        / "lower_limb_sim"
        / "data"
        / "reference_candidates"
    )
    paths = {
        FIRST_ROBOT_TRIAL_TRAJECTORY_ID: DEFAULT_REFERENCE_PATH,
        MEASURED_ASYMMETRIC_NOMINAL_ID: source_candidate_directory
        / "reference_measured_asymmetric_closed_nominal.csv",
    }
    try:
        return paths[trajectory_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown measured-asymmetric trajectory: {trajectory_id}"
        ) from exc


def _default_adapter_factory(robot_ip: str, *, local_ip: str):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip, local_ip=local_ip)


def _default_motion_factory(adapter: Any):
    from hardware.rokae_motion import RokaeCartesianMotionAdapter

    return RokaeCartesianMotionAdapter(adapter)


def run_execute(
    *,
    robot_ip: str,
    episode_dir: str | Path,
    anchor_path: str | Path,
    requested_anchor_id: str,
    frame_config_path: str | Path,
    safety_config_path: str | Path,
    trajectory_id: str,
    enable_motion: bool,
    operator_confirmation: str,
    local_ip: str | None = None,
    adapter_factory: AdapterFactory | None = None,
    motion_factory: MotionFactory | None = None,
) -> dict[str, Any]:
    """Run one real episode only after static and live gates both pass."""
    anchor = load_start_anchor(anchor_path)
    frame = load_rehab_frame_config(frame_config_path)
    safety = load_experiment_safety_config(safety_config_path)
    reference_path = _reference_path(trajectory_id)
    trajectory, audit, trajectory_metadata = build_start_anchored_relative_trajectory(
        reference_path,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    offline = evaluate_offline_execution_request(
        mode="execute",
        enable_motion=enable_motion,
        operator_confirmation=operator_confirmation,
        requested_anchor_id=requested_anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
    )
    # This is deliberately before adapter construction and before logging: a
    # bad flag/review/reference request cannot even attempt a robot connection.
    offline.require_allowed()
    resolved_local_ip = str(
        settings.ROBOT_LOCAL_IP if local_ip is None else local_ip
    ).strip()
    if adapter_factory is None and not resolved_local_ip:
        raise PermissionError(
            "real robot execution blocked: reviewed local xCoreSDK interface IP "
            "is required via --local-ip or config.settings.ROBOT_LOCAL_IP"
        )

    logger = EpisodeLogger(
        episode_dir,
        {
            "git_commit": current_git_commit(),
            "robot_ip": robot_ip,
            "robot_local_ip": resolved_local_ip or None,
            "sdk_version_if_available": None,
            "trajectory_id": audit.trajectory_id,
            "parent_reference_id": trajectory_metadata.get(
                "parent_reference_id"
            ),
            "parent_reference_sha256": trajectory_metadata.get(
                "parent_reference_sha256"
            ),
            "reference_version": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
            "reference_path": str(reference_path.resolve()),
            "approved_hip_rom": list(APPROVED_HIP_ROM_DEG),
            "approved_knee_rom": list(APPROVED_KNEE_ROM_DEG),
            "rom_protocol_version": ROM_PROTOCOL_VERSION,
            "rehab_frame": frame.as_metadata(),
            "rehab_frame_config_path": str(Path(frame_config_path).resolve()),
            "start_anchor": anchor.to_dict(),
            "start_anchor_path": str(Path(anchor_path).resolve()),
            "experiment_safety": safety.to_dict(),
            "experiment_safety_config_path": str(
                Path(safety_config_path).resolve()
            ),
            "trajectory_generation": trajectory_metadata,
            "tcp_orientation_strategy": "fixed",
            "experiment_mode": "start_anchored_relative",
            "robot_execution_approved": False,
            "operator_confirmation": operator_confirmation,
            "timestamp_source": TIMESTAMP_SOURCE,
            "trajectory_audit": audit.as_dict(),
            "motion_api_static_evidence": (
                "xCoreSDK_0.7.0_realtime_cartesian_pyi_and_vendor_examples"
            ),
            "motion_api_physical_validation": False,
        },
    )
    adapter: Any | None = None
    acquisition: RealRobotAcquisition | None = None
    executor: RokaeMotionExecutor | None = None
    completed = False
    result = None
    try:
        logger.start()
        adapter = (
            adapter_factory(robot_ip)
            if adapter_factory is not None
            else _default_adapter_factory(robot_ip, local_ip=resolved_local_ip)
        )
        acquisition = RealRobotAcquisition(adapter, logger)
        acquisition.start(manage_connection=True)
        if not acquisition.wait_until_healthy(timeout_s=3.0):
            raise RuntimeError(
                "state/wrench streams did not become healthy: "
                + acquisition.latest_health().invalid_reason
            )
        health = acquisition.latest_health()
        preflight = evaluate_execution_preflight(
            mode="execute",
            enable_motion=enable_motion,
            operator_confirmation=operator_confirmation,
            requested_anchor_id=requested_anchor_id,
            frame=frame,
            anchor=anchor,
            safety=safety,
            trajectory=trajectory,
            audit=audit,
            acquisition_health=health,
            logger=logger,
            robot_adapter=adapter,
            current_tcp_pose_base=adapter.read_tcp_pose(),
        )
        preflight.require_allowed()
        robot_summary = preflight.runtime_robot_summary
        sdk_version = (
            robot_summary.get("robot_metadata", {}).get("xcore_sdk_version")
            if isinstance(robot_summary, dict)
            else None
        )
        logger.update_metadata(
            {
                "sdk_version_if_available": sdk_version,
                "robot_state_summary_before_motion": robot_summary,
                "runtime_preflight": preflight.as_metadata(),
                # Attachment, a second live-health check, and the final anchor
                # check still remain inside the single-use executor.
                "robot_execution_approved": False,
            }
        )
        motion_adapter = (motion_factory or _default_motion_factory)(adapter)
        executor = RokaeMotionExecutor(
            motion_adapter,
            acquisition,
            logger,
            safety,
        )
        result = executor.execute(trajectory, preflight)
        logger.update_metadata({"execution_result": asdict(result)})
        completed = True
    except KeyboardInterrupt as exc:
        if executor is not None:
            try:
                executor.request_stop("operator_keyboard_interrupt")
            except BaseException as stop_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "software stop failed during KeyboardInterrupt: "
                        f"{type(stop_exc).__name__}:{stop_exc}"
                    )
        if not logger.write_timeout_event.is_set() and logger.healthy:
            try:
                logger.mark_failed("operator_keyboard_interrupt")
            except BaseException:
                pass
        raise
    except BaseException as exc:
        if executor is not None:
            try:
                executor.request_stop(
                    f"runner_exception:{type(exc).__name__}:{exc}"
                )
            except BaseException as stop_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "additional software-stop failure: "
                        f"{type(stop_exc).__name__}:{stop_exc}"
                    )
        if not logger.write_timeout_event.is_set() and logger.healthy:
            try:
                logger.mark_failed(
                    f"execute_exception:{type(exc).__name__}:{exc}"
                )
            except BaseException as logger_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "additional logger failure: "
                        f"{type(logger_exc).__name__}:{logger_exc}"
                    )
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        logger_may_close = not logger.write_timeout_event.is_set()
        if acquisition is not None:
            try:
                acquisition.stop()
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
                # A producer still blocked inside the native SDK may publish to
                # the logger later.  Do not race-close its files or disconnect.
                logger_may_close = bool(
                    logger_may_close
                    and not acquisition.live_producer_names
                )
        if logger.pending_write_count:
            logger_may_close = False
        if logger_may_close:
            try:
                logger.close(
                    completed=completed and not cleanup_errors,
                    stop_reason=(
                        None
                        if completed and not cleanup_errors
                        else "execute_failed_or_stopped"
                    ),
                )
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if cleanup_errors:
            import sys

            active_exception = sys.exc_info()[1]
            detail = ";".join(
                f"{type(item).__name__}:{item}" for item in cleanup_errors
            )
            if active_exception is not None:
                if hasattr(active_exception, "add_note"):
                    active_exception.add_note("execute cleanup failure: " + detail)
            else:
                raise RuntimeError("execute cleanup failure: " + detail) from cleanup_errors[0]
    return {
        "episode_dir": str(Path(episode_dir).resolve()),
        "completed": completed,
        "trajectory_id": trajectory_id,
        "commands_dispatched": result.commands_dispatched if result else 0,
        "stop_reason": result.stop_reason if result else None,
        "motion_was_explicitly_enabled": enable_motion,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROKAE rehabilitation experiment runner")
    parser.add_argument("--mode", choices=("preview", "acquire", "execute"), default="preview")
    parser.add_argument("--ip")
    parser.add_argument(
        "--local-ip",
        help=(
            "local Windows NIC address for reviewed xCoreSDK realtime mode; "
            "falls back to config.settings.ROBOT_LOCAL_IP"
        ),
    )
    parser.add_argument("--episode-dir")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--anchor")
    parser.add_argument("--anchor-id")
    parser.add_argument("--frame-config", default=str(DEFAULT_FRAME_CONFIG))
    parser.add_argument("--safety-config", default=str(DEFAULT_SAFETY_CONFIG))
    parser.add_argument(
        "--trajectory",
        default=FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
        choices=(
            FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
            MEASURED_ASYMMETRIC_NOMINAL_ID,
        ),
    )
    parser.add_argument("--preview-output-dir")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--operator-confirmation",
        default="",
        help=f"execute requires the exact text: {OPERATOR_CONFIRMATION}",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: AdapterFactory | None = None,
    motion_factory: MotionFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "preview":
        if not args.anchor or not args.preview_output_dir:
            raise ValueError("preview requires --anchor and --preview-output-dir")
        result = preview_trajectory(
            anchor=args.anchor,
            frame_config=args.frame_config,
            reference=_reference_path(args.trajectory),
            output_dir=args.preview_output_dir,
        )
    elif args.mode == "acquire":
        if not args.ip or not args.episode_dir or args.duration_s is None:
            raise ValueError("acquire requires --ip, --episode-dir, and --duration-s")
        result = run_acquisition(
            robot_ip=args.ip,
            episode_dir=args.episode_dir,
            duration_s=args.duration_s,
            adapter_factory=adapter_factory,
        )
    else:
        required = {
            "--ip": args.ip,
            "--episode-dir": args.episode_dir,
            "--anchor": args.anchor,
            "--anchor-id": args.anchor_id,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("execute missing required arguments: " + ", ".join(missing))
        result = run_execute(
            robot_ip=args.ip,
            episode_dir=args.episode_dir,
            anchor_path=args.anchor,
            requested_anchor_id=args.anchor_id,
            frame_config_path=args.frame_config,
            safety_config_path=args.safety_config,
            trajectory_id=args.trajectory,
            enable_motion=args.enable_motion,
            operator_confirmation=args.operator_confirmation,
            local_ip=args.local_ip,
            adapter_factory=adapter_factory,
            motion_factory=motion_factory,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
