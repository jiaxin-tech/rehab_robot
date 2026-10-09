"""Observation-only state/wrench acquisition into the five-file schema.

No project motion, power, or mode command is issued.  Vendor construction,
connect, and disconnect can still have controller-session side effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from utils.clock import TIMESTAMP_SOURCE
from utils.provenance import current_git_commit


AdapterFactory = Callable[[str], Any]
WrenchProviderFactory = Callable[[Path], Any]

# ``offline`` budgets are structurally valid but carry no engineering meaning.
# They exist so the wiring and its regressions can run without hardware; the
# provenance string is what stops them from ever reaching a live session.
WRENCH_MODES = ("state-only", "offline", "live")
OFFLINE_WRENCH_BUDGETS = {
    "startup_s": 5.0,
    "query_s": 0.5,
    "progress_s": 2.0,
    "sample_age_s": 2.0,
    "ack_s": 2.0,
    "log_s": 2.0,
    "shutdown_s": 0.5,
    "terminate_s": 0.5,
    "state_age_s": 2.0,
    "skew_s": 2.0,
}
LIVE_WRENCH_CONFIG_KEYS = (
    "schema_version",
    "scope",
    "reviewed",
    "connection",
    "expected_identity",
    "allowed_power_states",
    "budgets",
)
LIVE_CONNECTION_KEYS = ("robot_ip", "local_ip", "robot_class")
LIVE_IDENTITY_KEYS = ("robot_model", "robot_serial", "controller_version", "sdk_version")


def _default_adapter_factory(robot_ip: str):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip)


def _configured_adapter_factory(
    *,
    local_ip: str | None,
    robot_class: str | None,
    state_interval_ms: int | None,
    collision_source: str = "query",
    collision_event_max_age_s: float | None = None,
) -> AdapterFactory:
    """Build an adapter factory that overrides only the supplied settings.

    ``ROBOT_LOCAL_IP`` is empty by default, so a live observation session needs
    an explicit local interface.  Passing ``None`` keeps the repository default
    instead of silently substituting one.
    """

    from hardware.safety_events import validate_collision_source
    validate_collision_source(collision_source, collision_event_max_age_s)

    def factory(robot_ip: str):
        from hardware.rokae_adapter import RokaeRobotAdapter

        kwargs: dict[str, Any] = {}
        if local_ip is not None:
            kwargs["local_ip"] = local_ip
        if robot_class is not None:
            kwargs["robot_class"] = robot_class
        if state_interval_ms is not None:
            kwargs["state_interval_ms"] = state_interval_ms
        if collision_source == "events":
            kwargs["collision_source"] = collision_source
            kwargs["collision_event_max_age_s"] = collision_event_max_age_s
        return RokaeRobotAdapter(robot_ip, **kwargs)

    return factory


def build_offline_wrench_provider(episode_dir: Path, *, wrench_hz: float) -> Any:
    """Explicit OFFLINE provider used by wiring tests and dry runs."""

    from collection.wrench_process import WrenchBudgets, WrenchProcessProvider

    budgets = WrenchBudgets(**OFFLINE_WRENCH_BUDGETS, provenance="NOT_SAFETY_THRESHOLDS")
    return WrenchProcessProvider(
        {"mode": "offline", "rate_hz": wrench_hz},
        budgets,
        Path(episode_dir) / "wrench_query_events.jsonl",
    )


def load_live_wrench_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a reviewed live observation configuration.

    This validates structure and provenance only.  It is not an approval
    service: the values still have to come from a site review, and the
    ``reviewed`` flag records that the review happened outside this process.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("live wrench configuration must be a JSON object")
    missing = [key for key in LIVE_WRENCH_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError("missing live wrench configuration keys: " + ",".join(missing))
    if payload["schema_version"] != 1:
        raise ValueError("unsupported live wrench configuration schema_version")
    if payload["scope"] != "LIVE_OBSERVATION_ACQUISITION":
        raise ValueError("live wrench configuration scope must be LIVE_OBSERVATION_ACQUISITION")
    if payload["reviewed"] is not True:
        raise ValueError("live wrench configuration must record reviewed=true")
    connection = payload["connection"]
    if not isinstance(connection, dict) or set(connection) != set(LIVE_CONNECTION_KEYS):
        raise ValueError("live wrench connection must contain exactly " + ",".join(LIVE_CONNECTION_KEYS))
    if any(not isinstance(connection[key], str) or not connection[key].strip() for key in LIVE_CONNECTION_KEYS):
        raise ValueError("live wrench connection fields must be non-empty strings")
    identity = payload["expected_identity"]
    if not isinstance(identity, dict) or set(identity) != set(LIVE_IDENTITY_KEYS):
        raise ValueError("live wrench expected_identity must contain exactly " + ",".join(LIVE_IDENTITY_KEYS))
    if any(not isinstance(identity[key], str) or not identity[key].strip() for key in LIVE_IDENTITY_KEYS):
        raise ValueError("live wrench identity fields must be non-empty strings")
    power_states = payload["allowed_power_states"]
    if not isinstance(power_states, list) or not power_states or any(
        state not in ("on", "off") for state in power_states
    ):
        raise ValueError("live wrench allowed_power_states must be a non-empty subset of on/off")
    budgets = payload["budgets"]
    if not isinstance(budgets, dict):
        raise ValueError("live wrench budgets must be an object")
    if budgets.get("provenance") == "NOT_SAFETY_THRESHOLDS":
        raise ValueError("offline budgets are forbidden for a live observation session")
    return payload


def build_live_wrench_provider(
    episode_dir: Path,
    *,
    config: dict[str, Any],
    wrench_hz: float,
) -> Any:
    """Explicit LIVE provider. Requires a reviewed configuration file."""

    from collection.wrench_process import WrenchBudgets, WrenchProcessProvider

    budgets = WrenchBudgets(**config["budgets"])
    if budgets.provenance == "NOT_SAFETY_THRESHOLDS":
        raise ValueError("offline budgets are forbidden for a live observation session")
    provider_config = dict(config["connection"])
    provider_config.update(
        mode="live",
        rate_hz=wrench_hz,
        expected_identity=dict(config["expected_identity"]),
        allowed_power_states=list(config["allowed_power_states"]),
    )
    return WrenchProcessProvider(
        provider_config,
        budgets,
        Path(episode_dir) / "wrench_query_events.jsonl",
    )


def _stop_refused_sdk_cleanup(acquisition: Any, exc: BaseException) -> bool:
    """Return true when acquisition deliberately kept a live SDK query attached."""

    message = str(exc)
    if (
        "refusing SDK disconnect" in message
        or "acquisition_threads_did_not_stop" in message
    ):
        return True
    try:
        health = acquisition.latest_health()
    except Exception:
        health = None
    if health is not None and any(
        getattr(health, name, False) is True
        for name in (
            "state_thread_alive",
            "wrench_thread_alive",
            "alignment_thread_alive",
        )
    ):
        return True
    threads = getattr(acquisition, "_threads", None)
    if isinstance(threads, dict):
        return any(
            bool(getattr(thread, "is_alive", lambda: False)())
            for thread in threads.values()
        )
    return False


def _record_logger_failure(logger: EpisodeLogger, reason: str) -> None:
    if not logger.healthy:
        return
    try:
        logger.mark_failed(reason)
    except BaseException:
        # Cleanup continues independently even when metadata/failure recording
        # itself is unavailable.  logger.close() is still attempted below.
        pass


def _wait_until_ready(acquisition: Any, mode: str, timeout_s: float) -> tuple[bool, str]:
    """Wait for the readiness that the selected mode can actually reach.

    A state-only episode intentionally has no wrench producer, so the full
    ``AcquisitionHealth.valid`` flag can never become true.  Requiring it would
    make the diagnostic mode unusable.  Wrench modes keep the strict gate.
    """

    deadline = time.perf_counter() + timeout_s
    last_reason = "not_ready"
    while time.perf_counter() < deadline:
        health = acquisition.latest_health()
        last_reason = health.invalid_reason or "not_ready"
        if mode == "state-only":
            if (
                health.state_valid
                and health.state_thread_alive
                and health.alignment_thread_alive
            ):
                return True, ""
        elif health.valid:
            return True, ""
        if acquisition.background_error:
            return False, acquisition.background_error
        time.sleep(0.005)
    return False, last_reason


def run_acquisition(
    *,
    robot_ip: str,
    episode_dir: str | Path,
    duration_s: float,
    adapter_factory: AdapterFactory | None = None,
    mode: str = "state-only",
    wrench_provider_factory: WrenchProviderFactory | None = None,
    live_wrench_config: dict[str, Any] | None = None,
    wrench_hz: float | None = None,
) -> dict[str, Any]:
    """Acquire one observation-only episode.

    ``mode`` selects the wrench path and is deliberately explicit:

    * ``state-only``  no wrench producer; a state-only diagnostic episode.
    * ``offline``     an explicit OFFLINE process-isolated stub provider.
    * ``live``        a reviewed :class:`WrenchProcessProvider` from a site
                      configuration file.  Budgets and identity must be
                      reviewed; this runner never invents thresholds.

    A state-only episode is not the reviewed observation acquisition described
    in the real-robot procedure and is marked as such in the episode metadata.
    """

    if mode not in WRENCH_MODES:
        raise ValueError("mode must be one of " + ",".join(WRENCH_MODES))
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if wrench_hz is not None and (not isinstance(wrench_hz, (int, float)) or isinstance(wrench_hz, bool) or wrench_hz <= 0):
        raise ValueError("wrench_hz must be a positive number when supplied")
    episode_path = Path(episode_dir)
    if mode == "state-only" and wrench_provider_factory is not None:
        raise ValueError("wrench_provider_factory is incompatible with state-only mode")
    if mode == "offline" and live_wrench_config is not None:
        raise ValueError("live_wrench_config is incompatible with offline mode")
    if mode == "live":
        if wrench_provider_factory is not None:
            raise ValueError("wrench_provider_factory is incompatible with live mode")
        if live_wrench_config is None:
            raise ValueError("live mode requires a reviewed live_wrench_config")

    adapter_local_ip = None
    adapter_robot_class = None
    if live_wrench_config is not None:
        connection = live_wrench_config["connection"]
        adapter_local_ip = connection["local_ip"]
        adapter_robot_class = connection["robot_class"]
        configured_ip = connection["robot_ip"]
        if configured_ip != robot_ip:
            raise ValueError("live wrench configuration robot_ip must match the acquisition robot_ip")
    effective_wrench_hz = float(wrench_hz) if wrench_hz is not None else None
    base_metadata = {
        "git_commit": current_git_commit(),
        "robot_ip": robot_ip,
        "experiment_mode": "observation_only_acquire",
        "robot_execution_approved": False,
        "timestamp_source": TIMESTAMP_SOURCE,
        "wrench_mode": mode,
        "reviewed_observation_acquisition": bool(mode == "live"),
    }
    logger = EpisodeLogger(
        episode_path,
        base_metadata,
    ).start()
    adapter: Any | None = None
    acquisition: RealRobotAcquisition | None = None
    wrench_provider: Any | None = None
    acquisition_start_attempted = False
    completed = False
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    refuse_adapter_cleanup = False
    try:
        if mode == "offline":
            rate = effective_wrench_hz if effective_wrench_hz is not None else 50.0
            effective_wrench_hz = rate
            if wrench_provider_factory is not None:
                wrench_provider = wrench_provider_factory(episode_path, rate)
            else:
                wrench_provider = build_offline_wrench_provider(episode_path, wrench_hz=rate)
        elif mode == "live":
            rate = effective_wrench_hz if effective_wrench_hz is not None else 50.0
            effective_wrench_hz = rate
            wrench_provider = build_live_wrench_provider(
                episode_path,
                config=live_wrench_config,
                wrench_hz=rate,
            )
        adapter = (adapter_factory or _default_adapter_factory)(robot_ip)
        acquisition = RealRobotAcquisition(
            adapter,
            logger,
            wrench_provider=wrench_provider,
            diagnostic_state_only=(mode == "state-only"),
            **({"wrench_hz": effective_wrench_hz} if effective_wrench_hz is not None else {}),
        )
        # This runner owns connect/disconnect so every lifecycle stage can be
        # attempted independently.  Acquisition still owns its producer
        # threads and state-stream stop ordering.
        adapter.connect()
        acquisition_start_attempted = True
        acquisition.start(manage_connection=False)
        ready, reason = _wait_until_ready(acquisition, mode, timeout_s=3.0)
        if not ready:
            raise RuntimeError(f"acquisition did not become ready: {reason}")
        logger.update_metadata({"robot_state_summary": adapter.get_robot_state_summary()})
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            if acquisition.background_error:
                raise RuntimeError(acquisition.background_error)
            time.sleep(min(0.02, max(0.0, deadline - time.perf_counter())))
        completed = True
    except BaseException as exc:
        primary_error = exc
        _record_logger_failure(
            logger,
            f"acquire_exception:{type(exc).__name__}:{exc}",
        )
    finally:
        if acquisition is not None and acquisition_start_attempted:
            try:
                acquisition.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
                refuse_adapter_cleanup = _stop_refused_sdk_cleanup(
                    acquisition,
                    exc,
                )

        if adapter is not None and not refuse_adapter_cleanup:
            # acquisition.stop() already stops the state stream when it returns
            # normally.  Before acquisition.start(), or after a non-thread
            # cleanup failure, perform/retry both adapter cleanup stages and do
            # not let one prevent the other.
            should_stop_stream_directly = (
                not acquisition_start_attempted or bool(cleanup_errors)
            )
            if should_stop_stream_directly:
                try:
                    adapter.stop_state_stream()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                adapter.disconnect()
            except BaseException as exc:
                cleanup_errors.append(exc)

        if cleanup_errors:
            details = ";".join(
                f"{type(exc).__name__}:{exc}" for exc in cleanup_errors
            )
            _record_logger_failure(logger, f"acquire_cleanup:{details}")

        terminal_completed = bool(
            completed and primary_error is None and not cleanup_errors
        )
        if not refuse_adapter_cleanup:
            try:
                logger.close(
                    completed=terminal_completed,
                    stop_reason=None if terminal_completed else "acquire_failed",
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        elif wrench_provider is not None and wrench_provider.cleanup is None:
            # A provider that still owns a native child keeps its cleanup record
            # open on purpose; do not pretend the episode was closed.
            pass
        # When a producer is still inside a native SDK call, it retains the
        # logger and may yet return through its failure path.  Closing the CSV
        # handles here would race that producer just as disconnecting would.

    if cleanup_errors:
        cleanup_error = cleanup_errors[0]
        if primary_error is not None:
            raise cleanup_error from primary_error
        raise cleanup_error
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    return {
        "episode_dir": str(episode_path.resolve()),
        "completed": True,
        "duration_s": duration_s,
        "wrench_mode": mode,
        "row_counts": logger.row_counts,
        "motion_commanded": False,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Observation-only ROKAE episode acquisition; vendor session "
            "side effects still require supervision"
        )
    )
    parser.add_argument("--ip", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=WRENCH_MODES,
        default="state-only",
        help=(
            "state-only: diagnostic episode without a wrench producer; "
            "offline: process-isolated offline stub; live: reviewed provider"
        ),
    )
    parser.add_argument(
        "--wrench-config",
        default=None,
        help="reviewed live observation configuration JSON (required for --mode live)",
    )
    parser.add_argument(
        "--wrench-hz",
        type=float,
        default=None,
        help="explicit wrench rate; must match the provider rate",
    )
    parser.add_argument(
        "--local-ip",
        default=None,
        help="local interface for the SDK session; required in practice for live observation",
    )
    parser.add_argument("--robot-class", default=None, help="override ROBOT_CLASS")
    parser.add_argument("--state-interval-ms", type=int, default=None, help="override ROBOT_STATE_MS")
    parser.add_argument("--collision-source", choices=("query", "events"), default="query")
    parser.add_argument("--collision-event-max-age-s", type=float, default=None,
                        help="required reviewed event freshness budget for events mode")
    args = parser.parse_args(argv)
    from hardware.safety_events import validate_collision_source
    validate_collision_source(args.collision_source, args.collision_event_max_age_s)
    live_wrench_config = None
    if args.wrench_config is not None:
        live_wrench_config = load_live_wrench_config(args.wrench_config)
    if args.mode == "live" and live_wrench_config is None:
        parser.error("--wrench-config is required when --mode live")
    if args.mode == "offline" and (args.local_ip or args.robot_class or args.state_interval_ms
                                   or args.collision_source == "events"):
        parser.error("live connection and safety event options are incompatible with offline wrench mode")
    if adapter_factory is None:
        adapter_factory = _configured_adapter_factory(
            local_ip=args.local_ip,
            robot_class=args.robot_class,
            state_interval_ms=args.state_interval_ms,
            collision_source=args.collision_source,
            collision_event_max_age_s=args.collision_event_max_age_s,
        )
    result = run_acquisition(
        robot_ip=args.ip,
        episode_dir=args.episode_dir,
        duration_s=args.duration_s,
        adapter_factory=adapter_factory,
        mode=args.mode,
        live_wrench_config=live_wrench_config,
        wrench_hz=args.wrench_hz,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
