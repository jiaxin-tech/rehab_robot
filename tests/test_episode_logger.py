"""Offline tests for the hardware-independent real-episode data logger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from collection.episode_logger import (
    EpisodeLogger,
    EpisodeLoggerError,
    STREAM_FILENAMES,
)
from utils.clock import PerfCounterClock, TIMESTAMP_SOURCE


class _DeterministicClock:
    def __init__(self, first_ns: int = 10_000_000_000) -> None:
        self.value_ns = first_ns

    def now_ns(self) -> int:
        result = self.value_ns
        self.value_ns += 1_000_000
        return result

    def now_s(self) -> float:
        return self.now_ns() / 1_000_000_000


def _read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise AssertionError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


class _WriterCleanupGate:
    """Block only the durable writer's pending-set cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.writer_entered = threading.Event()
        self.release_writer = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "episode-command-durable-write":
            self.writer_entered.set()
            if not self.release_writer.wait(timeout=2.0):
                raise TimeoutError("test did not release writer cleanup gate")
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()
        return False


class ClockTests(unittest.TestCase):
    def test_clock_uses_perf_counter_ns_as_the_only_source(self) -> None:
        with patch("utils.clock.time.perf_counter_ns", return_value=1_234_567_890):
            clock = PerfCounterClock()
            self.assertEqual(clock.now_ns(), 1_234_567_890)
            self.assertAlmostEqual(clock.now_s(), 1.23456789)
        self.assertEqual(clock.timestamp_source, TIMESTAMP_SOURCE)


class EpisodeLoggerTests(unittest.TestCase):
    def test_ready_barrier_independent_streams_and_sparse_values(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0001"
            logger = EpisodeLogger(
                episode,
                {
                    "git_commit": None,
                    "trajectory_id": "reference_measured_asymmetric_closed_slow",
                },
                clock=_DeterministicClock(),
            )
            self.assertFalse(episode.exists())
            self.assertFalse(logger.wait_until_ready(timeout=0.0))

            logger.start()
            self.assertTrue(logger.ready_event.is_set())
            self.assertTrue(logger.wait_until_ready(timeout=0.0))
            self.assertTrue(logger.healthy)
            for filename in (*STREAM_FILENAMES.values(), "metadata.json"):
                self.assertTrue((episode / filename).is_file(), filename)

            logger.append_robot_state(
                host_time_s=1.0,
                q1=None,
                tcp_x=0.31,
                valid=False,
                invalid_reason="joint_positions_unavailable",
            )
            logger.append_robot_wrench(
                query_start_s=1.001,
                query_end_s=1.004,
                publish_time_s=1.004,
                fx=None,
                frame_type="world",
                query_duration_ms=3.0,
                valid=False,
                invalid_reason="cartesian_force_unavailable",
            )
            logger.append_trajectory_command(
                host_time_s=1.005,
                trajectory_time_s=0.0,
                trajectory_phase=0.0,
                delta_x_R=0.0,
                delta_y_R=0.0,
                delta_z_R=0.0,
                tcp_target_x=0.31,
                tcp_target_y=None,
                tcp_target_z=0.42,
                command_valid=True,
                invalid_reason="",
            )
            logger.append_aligned_snapshot(
                host_time_s=1.006,
                state_time_s=1.0,
                wrench_time_s=1.004,
                state_age_s=0.006,
                wrench_age_s=0.002,
                state_wrench_skew_s=0.004,
                state_valid=False,
                wrench_valid=False,
                state_thread_alive=True,
                wrench_thread_alive=True,
                query_duration_ms=3.0,
                valid=False,
                invalid_reason="joint_positions_unavailable;cartesian_force_unavailable",
            )
            logger.update_metadata({"operator_confirmation": None})
            logger.close(completed=True)

            state = _read_one(episode / "robot_state.csv")
            self.assertEqual(state["q1"], "")
            self.assertEqual(state["q2"], "")
            self.assertNotEqual(state["valid"], "")
            self.assertEqual(state["invalid_reason"], "joint_positions_unavailable")

            wrench = _read_one(episode / "robot_wrench.csv")
            self.assertEqual(wrench["fx"], "")
            self.assertEqual(wrench["fy"], "")
            self.assertEqual(wrench["query_duration_ms"], "3.0")

            command = _read_one(episode / "trajectory_command.csv")
            self.assertEqual(command["tcp_target_y"], "")
            self.assertEqual(command["delta_x_R"], "0.0")

            snapshot = _read_one(episode / "aligned_snapshot.csv")
            self.assertEqual(snapshot["state_wrench_skew_s"], "0.004")
            self.assertEqual(snapshot["invalid_reason"], (
                "joint_positions_unavailable;cartesian_force_unavailable"
            ))

            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertIsNone(metadata["git_commit"])
            self.assertIsNone(metadata["operator_confirmation"])
            self.assertEqual(metadata["logger"]["status"], "completed")
            self.assertEqual(metadata["logger"]["timestamp_source"], TIMESTAMP_SOURCE)
            self.assertEqual(
                metadata["logger"]["row_counts"],
                {
                    "aligned_snapshot": 1,
                    "robot_state": 1,
                    "robot_wrench": 1,
                    "trajectory_command": 1,
                },
            )
            self.assertGreater(metadata["logger"]["episode_duration_s"], 0.0)
            self.assertGreater(
                metadata["logger"]["average_stream_publish_rate_hz"]["robot_state"],
                0.0,
            )
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_append_error_marks_logger_failed_and_blocks_future_writes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0002"
            logger = EpisodeLogger(episode, clock=_DeterministicClock()).start()

            with self.assertRaisesRegex(EpisodeLoggerError, "unexpected_field"):
                logger.append_robot_state(
                    host_time_s=2.0,
                    valid=True,
                    invalid_reason="",
                    unexpected_field=123,
                )

            self.assertEqual(logger.status, "failed")
            self.assertFalse(logger.healthy)
            self.assertTrue(logger.failure_event.is_set())
            self.assertFalse(logger.wait_until_ready(timeout=0.0))
            self.assertIn("unexpected_field", logger.failure_reason or "")
            with self.assertRaises(EpisodeLoggerError):
                logger.append_robot_wrench(valid=False, invalid_reason="blocked")

            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failed_stream"], "robot_state")
            self.assertEqual(metadata["logger"]["row_counts"]["robot_state"], 0)
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_startup_collision_fails_without_overwriting_existing_data(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_existing"
            episode.mkdir()
            state_path = episode / "robot_state.csv"
            metadata_path = episode / "metadata.json"
            state_path.write_text("do-not-overwrite\n", encoding="utf-8")
            metadata_path.write_text('{"owner":"existing"}\n', encoding="utf-8")
            logger = EpisodeLogger(episode, clock=_DeterministicClock())

            with self.assertRaisesRegex(EpisodeLoggerError, "already exist"):
                logger.start()

            self.assertEqual(state_path.read_text(encoding="utf-8"), "do-not-overwrite\n")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"owner":"existing"}\n',
            )
            self.assertEqual(logger.status, "failed")
            self.assertTrue(logger.failed)
            self.assertFalse(logger.wait_until_ready(timeout=0.0))

            logger.close(completed=False, stop_reason="startup cleanup")
            self.assertEqual(state_path.read_text(encoding="utf-8"), "do-not-overwrite\n")
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"owner":"existing"}\n',
            )
            self.assertEqual(list(episode.glob(".metadata.*.tmp")), [])

    def test_csv_only_collision_close_does_not_create_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_csv_collision"
            episode.mkdir()
            state_path = episode / "robot_state.csv"
            state_path.write_text("foreign-state\n", encoding="utf-8")
            logger = EpisodeLogger(episode, clock=_DeterministicClock())

            with self.assertRaisesRegex(EpisodeLoggerError, "already exist"):
                logger.start()
            logger.close(completed=False)

            self.assertEqual(state_path.read_text(encoding="utf-8"), "foreign-state\n")
            self.assertFalse((episode / "metadata.json").exists())

    def test_completed_writer_ack_is_not_delayed_by_pending_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            logger = EpisodeLogger(
                Path(root) / "episode_cleanup_gate",
                clock=_DeterministicClock(),
            ).start()
            cleanup_gate = _WriterCleanupGate()
            logger._pending_write_lock = cleanup_gate
            caller_done = threading.Event()
            errors: list[BaseException] = []

            def append_bounded() -> None:
                try:
                    logger.append_trajectory_command_bounded(
                        timeout_s=1.0,
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    caller_done.set()

            caller = threading.Thread(target=append_bounded, name="bounded-caller")
            caller.start()
            try:
                self.assertTrue(cleanup_gate.writer_entered.wait(timeout=1.0))
                # Durable completion wins the handshake before the worker can
                # block in pending-set bookkeeping.
                self.assertTrue(caller_done.wait(timeout=0.2))
                self.assertEqual(errors, [])
                self.assertFalse(logger.failure_event.is_set())
            finally:
                cleanup_gate.release_writer.set()
                caller.join(timeout=1.0)
            self.assertFalse(caller.is_alive())
            self.assertTrue(logger.wait_for_pending_writes(1.0))
            logger.close(completed=True)

    def test_bounded_timeout_signals_without_logger_lock_and_blocks_later_append(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_bounded_timeout"
            logger = EpisodeLogger(episode, clock=_DeterministicClock()).start()
            fsync_entered = threading.Event()
            release_fsync = threading.Event()
            caller_done = threading.Event()
            errors: list[BaseException] = []

            def blocking_fsync(_descriptor: int) -> None:
                fsync_entered.set()
                if not release_fsync.wait(timeout=2.0):
                    raise TimeoutError("test did not release fsync")

            def append_bounded() -> None:
                try:
                    logger.append_trajectory_command_bounded(
                        timeout_s=0.05,
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    caller_done.set()

            with patch("collection.episode_logger.os.fsync", side_effect=blocking_fsync):
                caller = threading.Thread(target=append_bounded, name="bounded-caller")
                caller.start()
                self.assertTrue(fsync_entered.wait(timeout=1.0))
                self.assertTrue(caller_done.wait(timeout=1.0))
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], EpisodeLoggerError)
                self.assertTrue(logger.write_timeout_event.is_set())
                self.assertTrue(logger.failure_event.is_set())
                self.assertFalse(logger.healthy_signal)

                later_done = threading.Event()
                later_errors: list[BaseException] = []

                def append_after_timeout() -> None:
                    try:
                        logger.append_robot_state(valid=True, invalid_reason="")
                    except BaseException as exc:
                        later_errors.append(exc)
                    finally:
                        later_done.set()

                later = threading.Thread(target=append_after_timeout)
                later.start()
                # This must finish while fsync still owns the logger I/O lock.
                self.assertTrue(later_done.wait(timeout=0.2))
                self.assertEqual(len(later_errors), 1)
                self.assertIsInstance(later_errors[0], EpisodeLoggerError)

                release_fsync.set()
                caller.join(timeout=1.0)
                later.join(timeout=1.0)
                self.assertTrue(logger.wait_for_pending_writes(1.0))

            self.assertEqual(logger.status, "failed")
            logger.close(completed=False)
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failed_stream"], "trajectory_command")

    def test_signal_failure_returns_while_fsync_holds_logger_lock(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_signal_failure"
            logger = EpisodeLogger(episode, clock=_DeterministicClock()).start()
            fsync_entered = threading.Event()
            release_fsync = threading.Event()
            append_done = threading.Event()
            signal_done = threading.Event()

            def blocking_fsync(_descriptor: int) -> None:
                fsync_entered.set()
                if not release_fsync.wait(timeout=2.0):
                    raise TimeoutError("test did not release fsync")

            def append_command() -> None:
                try:
                    logger.append_trajectory_command(
                        host_time_s=1.0,
                        command_valid=True,
                        invalid_reason="",
                    )
                finally:
                    append_done.set()

            def publish_failure() -> None:
                logger.signal_failure("sensor_dead", stream="wrench")
                signal_done.set()

            with patch("collection.episode_logger.os.fsync", side_effect=blocking_fsync):
                writer = threading.Thread(target=append_command)
                writer.start()
                self.assertTrue(fsync_entered.wait(timeout=1.0))
                signaler = threading.Thread(target=publish_failure)
                signaler.start()
                self.assertTrue(signal_done.wait(timeout=0.2))
                self.assertTrue(logger.failure_event.is_set())
                self.assertFalse(logger.healthy_signal)
                release_fsync.set()
                writer.join(timeout=1.0)
                signaler.join(timeout=1.0)
                self.assertTrue(append_done.is_set())

            logger.close(completed=False)
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["logger"]["status"], "failed")
            self.assertEqual(metadata["logger"]["failure_reason"], "sensor_dead")
            self.assertEqual(metadata["logger"]["failed_stream"], "wrench")


if __name__ == "__main__":
    unittest.main()
