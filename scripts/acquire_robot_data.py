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


def _default_adapter_factory(robot_ip: str):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip)


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


def run_acquisition(
    *,
    robot_ip: str,
    episode_dir: str | Path,
    duration_s: float,
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    logger = EpisodeLogger(
        episode_dir,
        {
            "git_commit": current_git_commit(),
            "robot_ip": robot_ip,
            "experiment_mode": "observation_only_acquire",
            "robot_execution_approved": False,
            "timestamp_source": TIMESTAMP_SOURCE,
        },
    ).start()
    adapter: Any | None = None
    acquisition: RealRobotAcquisition | None = None
    acquisition_start_attempted = False
    completed = False
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    refuse_adapter_cleanup = False
    try:
        adapter = (adapter_factory or _default_adapter_factory)(robot_ip)
        acquisition = RealRobotAcquisition(adapter, logger)
        # This runner owns connect/disconnect so every lifecycle stage can be
        # attempted independently.  Acquisition still owns its producer
        # threads and state-stream stop ordering.
        adapter.connect()
        acquisition_start_attempted = True
        acquisition.start(manage_connection=False)
        if not acquisition.wait_until_healthy(timeout_s=3.0):
            raise RuntimeError(
                "state/wrench acquisition did not become healthy: "
                + acquisition.latest_health().invalid_reason
            )
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
        "episode_dir": str(Path(episode_dir).resolve()),
        "completed": True,
        "duration_s": duration_s,
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
    args = parser.parse_args(argv)
    result = run_acquisition(
        robot_ip=args.ip,
        episode_dir=args.episode_dir,
        duration_s=args.duration_s,
        adapter_factory=adapter_factory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
