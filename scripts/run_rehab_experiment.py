"""Unified preview, observation-only acquire, and gated execute runner.

Only ``--mode execute`` together with ``--enable-motion`` can reach the motion
adapter.  All offline review gates are evaluated before the adapter is created
or a robot connection is attempted. Reviewed releases may opt into same-session
RT preparation. Loaded trials require an explicit bound setup and valid collision
state; the empty-load commissioning exception never applies to loaded trials.
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
from safety.empty_load_commissioning_release import (
    SCOPE as EMPTY_LOAD_COMMISSIONING_SCOPE,
    load_and_validate_empty_load_commissioning_release,
)
from safety.robot_trial_release import (
    LOADED_SCOPE, load_and_validate_robot_trial_release, read_json,
)
from scripts.acquire_robot_data import (
    build_live_wrench_provider,
    load_live_wrench_config,
    run_acquisition,
)
from scripts.preview_rehab_trajectory import preview_trajectory
from utils.clock import TIMESTAMP_SOURCE
from utils.provenance import current_git_commit
from hardware.safety_events import validate_collision_source


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


def _default_adapter_factory(robot_ip: str, *, local_ip: str,
                             collision_source: str = "query",
                             collision_event_max_age_s: float | None = None):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip, local_ip=local_ip,
                             collision_source=collision_source,
                             collision_event_max_age_s=collision_event_max_age_s)


def _default_motion_factory(adapter: Any):
    from hardware.rokae_motion import RokaeCartesianMotionAdapter

    return RokaeCartesianMotionAdapter(adapter)


def _prepare_empty_load_realtime_session(
    adapter: Any,
    *,
    filter_hz: float,
    network_tolerance_percent: float,
) -> None:
    """Connect and prepare RT mode without starting a motion loop.

    This is intentionally reachable only after the caller has validated the
    hash-bound empty-load commissioning release.  Keeping preparation and
    execution on the same native session avoids a prepare CLI disconnect from
    undoing realtime mode before the executor attaches.
    """

    if not float(network_tolerance_percent).is_integer():
        raise ValueError("reviewed RT network tolerance must be an integer percent")
    adapter.connect()
    native = getattr(adapter, "native_robot", None)
    if native is None or not callable(getattr(native, "enable_realtime", None)):
        raise TypeError("empty-load realtime preparation requires the native SDK wrapper")
    native.rt_network_tolerance_percent = int(network_tolerance_percent)
    native.rt_filter_hz = float(filter_hz)
    native.enable_realtime(load=0.0)


def _cleanup_empty_load_realtime_session(adapter: Any) -> list[BaseException]:
    """Best-effort stop, power-down and disconnect for a prepared session."""

    errors: list[BaseException] = []
    native = getattr(adapter, "native_robot", None)
    if native is not None:
        try:
            native.stop_realtime(switch_to_nrt=True)
        except BaseException as exc:
            errors.append(exc)
        try:
            native.disable()
        except BaseException as exc:
            errors.append(exc)
    try:
        adapter.disconnect()
    except BaseException as exc:
        errors.append(exc)
    return errors


def _require_loaded_realtime_setup(release: Any) -> None:
    payload = release.require_current(check_files=True)
    if payload.get("scope") != LOADED_SCOPE:
        raise PermissionError("same-session RT requires a reviewed loaded or commissioning release")
    setup_path = (Path(release.approval_path).parent / payload["trial_setup"]["path"]).resolve()
    setup = read_json(setup_path)
    config = read_json(setup_path.parent / setup["inputs"]["round_config"]["path"])
    if config.get("realtime_preparation") != "same_session":
        raise PermissionError("bound round setup must explicitly review same_session RT preparation")


def _prepare_loaded_realtime_session(adapter: Any, *, release: Any,
                                     filter_hz: float, network_tolerance_percent: float) -> None:
    # load=0 is the SDK interface sentinel; it does not configure payload mass.
    # The controller toolset is preserved and checked again by live preflight.
    if not float(network_tolerance_percent).is_integer():
        raise ValueError("reviewed RT network tolerance must be an integer percent")
    adapter.connect()
    release.require_current(check_files=True)
    summary = adapter.get_robot_state_summary()
    release.require_runtime(summary)
    if (summary.get("collision_state_query_valid") is not True
            or summary.get("collision_state") is not False):
        raise PermissionError("loaded RT preparation requires valid clear collision state")
    native = getattr(adapter, "native_robot", None)
    if native is None or not callable(getattr(native, "enable_realtime", None)):
        raise TypeError("loaded realtime preparation requires the native SDK wrapper")
    if native.get_robot_mode() != "IDLE":
        raise PermissionError("loaded RT preparation requires initial IDLE state")
    native.rt_network_tolerance_percent = int(network_tolerance_percent)
    native.rt_filter_hz = float(filter_hz)
    native.enable_realtime(load=0.0)


def run_execute(
    *,
    robot_ip: str,
    episode_dir: str | Path,
    anchor_path: str | Path,
    requested_anchor_id: str,
    frame_config_path: str | Path,
    safety_config_path: str | Path,
    trajectory_id: str,
    reference_path_override: str | Path | None = None,
    enable_motion: bool,
    operator_confirmation: str,
    local_ip: str | None = None,
    live_wrench_config: dict[str, Any] | None = None,
    wrench_hz: float | None = None,
    adapter_factory: AdapterFactory | None = None,
    motion_factory: MotionFactory | None = None,
    collision_source: str = "query",
    collision_event_max_age_s: float | None = None,
    release_approval_path: str | Path | None = None,
    wrench_config_path: str | Path | None = None,
    execution_guard=None,
    dispatch_observer=None,
    prepare_realtime: bool = False,
) -> dict[str, Any]:
    """Run one real episode only after static and live gates both pass."""
    validate_collision_source(collision_source, collision_event_max_age_s)
    anchor = load_start_anchor(anchor_path)
    frame = load_rehab_frame_config(frame_config_path)
    safety = load_experiment_safety_config(safety_config_path)
    reference_path = (
        Path(reference_path_override).resolve()
        if reference_path_override is not None
        else _reference_path(trajectory_id)
    )
    if reference_path_override is not None and release_approval_path is None:
        raise PermissionError(
            "real robot execution blocked: reference override requires a bound release"
        )
    release_approval = None
    if release_approval_path is not None:
        if wrench_config_path is None:
            raise ValueError("release approval requires --wrench-config")
        release_payload = json.loads(
            Path(release_approval_path).read_text(encoding="utf-8")
        )
        release_loader = (
            load_and_validate_empty_load_commissioning_release
            if release_payload.get("scope") == EMPTY_LOAD_COMMISSIONING_SCOPE
            else load_and_validate_robot_trial_release
        )
        release_approval = release_loader(
            release_approval_path,
            reference_path=reference_path,
            safety_config_path=safety_config_path,
            frame_config_path=frame_config_path,
            anchor_path=anchor_path,
            wrench_config_path=wrench_config_path,
        )
    loaded_preparation = False
    if prepare_realtime:
        if release_approval is None:
            raise PermissionError("same-session realtime preparation requires a reviewed release")
        if release_approval.payload.get("scope") != EMPTY_LOAD_COMMISSIONING_SCOPE:
            _require_loaded_realtime_setup(release_approval)
            loaded_preparation = True
        if (live_wrench_config is None
                or "rtcontrolling" not in live_wrench_config.get("allowed_operation_states", [])):
            raise PermissionError("same-session RT wrench config must review rtcontrolling")
    trajectory, audit, trajectory_metadata = build_start_anchored_relative_trajectory(
        reference_path,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    if release_approval is not None:
        release_approval.require_acquisition(
            config=live_wrench_config, wrench_hz=wrench_hz, collision_source=collision_source,
            collision_event_max_age_s=collision_event_max_age_s)
        release_approval.require_objects(
            safety=safety, frame=frame, anchor=anchor, trajectory=trajectory
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
        release_approval=release_approval,
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
    # Execution needs a healthy state *and* wrench stream, so the isolated
    # wrench provider has to be wired here exactly as in observation
    # acquisition. Without it the connection gate in
    # ``RealRobotAcquisition.start`` refuses to proceed, which is the intended
    # fail-closed behaviour rather than a path to reopen.
    if live_wrench_config is None:
        raise PermissionError(
            "real robot execution blocked: reviewed live wrench configuration "
            "is required (--wrench-config)"
        )
    if wrench_hz is None or wrench_hz <= 0.0:
        raise PermissionError(
            "real robot execution blocked: explicit positive --wrench-hz is required"
        )
    if live_wrench_config["connection"]["robot_ip"] != robot_ip:
        raise PermissionError(
            "real robot execution blocked: wrench connection robot_ip must match"
        )
    if live_wrench_config["connection"]["local_ip"] != resolved_local_ip:
        raise PermissionError(
            "real robot execution blocked: wrench connection local_ip must match"
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
            "collision_source": collision_source,
            "collision_event_max_age_s": collision_event_max_age_s,
            "trial_release": release_approval.as_metadata() if release_approval is not None else None,
        },
    )
    adapter: Any | None = None
    acquisition: RealRobotAcquisition | None = None
    wrench_provider: Any | None = None
    executor: RokaeMotionExecutor | None = None
    completed = False
    result = None
    realtime_prepared_here = False
    try:
        logger.start()
        adapter = (
            adapter_factory(robot_ip)
            if adapter_factory is not None
            else _default_adapter_factory(robot_ip, local_ip=resolved_local_ip,
                                          collision_source=collision_source,
                                          collision_event_max_age_s=collision_event_max_age_s)
        )
        if prepare_realtime:
            if safety.reviewed_rt_filter_hz is None:
                raise PermissionError("reviewed RT filter frequency is unavailable")
            if safety.reviewed_rt_network_tolerance_percent is None:
                raise PermissionError("reviewed RT network tolerance is unavailable")
            # Own cleanup before the first connection attempt so a partial
            # enable failure still reaches stop/power-down/disconnect.
            realtime_prepared_here = True
            preparation = (_prepare_loaded_realtime_session if loaded_preparation
                           else _prepare_empty_load_realtime_session)
            preparation(adapter, filter_hz=safety.reviewed_rt_filter_hz,
                        network_tolerance_percent=safety.reviewed_rt_network_tolerance_percent,
                        **({"release": release_approval} if loaded_preparation else {}))
        wrench_provider = build_live_wrench_provider(
            Path(episode_dir),
            config=live_wrench_config,
            wrench_hz=float(wrench_hz),
        )
        acquisition = RealRobotAcquisition(
            adapter,
            logger,
            wrench_provider=wrench_provider,
            wrench_hz=float(wrench_hz),
        )
        acquisition.start(manage_connection=not realtime_prepared_here)
        startup_timeout_s = float(live_wrench_config["budgets"]["startup_s"]) + 0.5
        if not acquisition.wait_until_healthy(timeout_s=startup_timeout_s):
            raise RuntimeError(
                "state/wrench streams did not become healthy: "
                + acquisition.latest_health().invalid_reason
            )
        health = acquisition.latest_health()
        if release_approval is not None:
            release_approval.require_runtime(adapter.get_robot_state_summary())
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
            release_approval=release_approval,
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
        extra = {}
        if execution_guard is not None:
            extra["execution_guard"] = execution_guard
        if dispatch_observer is not None:
            extra["dispatch_observer"] = dispatch_observer
        executor = RokaeMotionExecutor(
            motion_adapter,
            acquisition,
            logger,
            safety,
            **extra,
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
        if (
            realtime_prepared_here
            and adapter is not None
        ):
            if acquisition is not None and acquisition.live_producer_names:
                # Stop/power-down must still be attempted; avoid disconnecting a
                # session while a native producer may still be using it.
                native = getattr(adapter, "native_robot", None)
                for action in (lambda: native.stop_realtime(switch_to_nrt=True),
                               lambda: native.disable()):
                    try:
                        action()
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                cleanup_errors.append(RuntimeError("live producers remain; disconnect unconfirmed"))
            else:
                cleanup_errors.extend(_cleanup_empty_load_realtime_session(adapter))
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
        "same_session_realtime_cleanup_completed": realtime_prepared_here,
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
    parser.add_argument(
        "--reference-path",
        default=None,
        help="release-bound reference override; execute only",
    )
    parser.add_argument("--preview-output-dir")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--prepare-realtime",
        action="store_true",
        help=(
            "same-session RT mode preparation; requires a hash-bound commissioning "
            "release or an explicitly reviewed loaded same_session setup"
        ),
    )
    parser.add_argument("--collision-source", choices=("query", "events"), default="query")
    parser.add_argument("--collision-event-max-age-s", type=float, default=None,
                        help="required reviewed event freshness budget for events mode")
    parser.add_argument(
        "--wrench-config",
        default=None,
        help=(
            "reviewed live wrench configuration JSON; required for execute so the "
            "isolated wrench producer is wired explicitly"
        ),
    )
    parser.add_argument(
        "--wrench-hz",
        type=float,
        default=None,
        help="explicit wrench rate for execute; must match the provider rate",
    )
    parser.add_argument(
        "--release-approval",
        default=None,
        help="hash-bound approved first-robot-trial release JSON",
    )
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
    validate_collision_source(args.collision_source, args.collision_event_max_age_s)
    if args.mode == "preview":
        if args.collision_source != "query":
            raise ValueError("safety event subscription is not available in offline preview")
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
        if adapter_factory is None:
            from scripts.acquire_robot_data import _configured_adapter_factory
            adapter_factory = _configured_adapter_factory(
                local_ip=args.local_ip, robot_class=None, state_interval_ms=None,
                collision_source=args.collision_source,
                collision_event_max_age_s=args.collision_event_max_age_s,
            )
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
        live_wrench_config = None
        if args.wrench_config is not None:
            live_wrench_config = load_live_wrench_config(args.wrench_config)
        result = run_execute(
            robot_ip=args.ip,
            episode_dir=args.episode_dir,
            anchor_path=args.anchor,
            requested_anchor_id=args.anchor_id,
            frame_config_path=args.frame_config,
            safety_config_path=args.safety_config,
            trajectory_id=args.trajectory,
            reference_path_override=args.reference_path,
            enable_motion=args.enable_motion,
            operator_confirmation=args.operator_confirmation,
            local_ip=args.local_ip,
            live_wrench_config=live_wrench_config,
            wrench_hz=args.wrench_hz,
            adapter_factory=adapter_factory,
            motion_factory=motion_factory,
            collision_source=args.collision_source,
            collision_event_max_age_s=args.collision_event_max_age_s,
            release_approval_path=args.release_approval,
            wrench_config_path=args.wrench_config,
            prepare_realtime=args.prepare_realtime,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
