"""Offline command-schedule pre-recording tests; no robot or SDK access."""
import csv
import math
from pathlib import Path
import threading
import time
from unittest.mock import patch

import pytest

from collection.episode_logger import EpisodeLogger, EpisodeLoggerError


def row(index=0):
    return {
        "host_time_s": float(index),
        "trajectory_time_s": index * .02,
        "trajectory_phase": "lift",
        "tcp_target_x": .7,
        "tcp_target_y": -.1,
        "tcp_target_z": .3 + index * .001,
        "tcp_target_rx": 3.0,
        "tcp_target_ry": 0.0,
        "tcp_target_rz": .7,
        "command_valid": True,
        "invalid_reason": "",
    }


def read_commands(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_whole_schedule_is_written_once_with_receipt_and_sparse_cells(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    rows = [row(0), row(1), {"trajectory_time_s": .04, "command_valid": True}]
    receipt = logger.append_trajectory_commands_bounded(rows, timeout_s=1.0)
    assert receipt["method"] == "whole_schedule_before_motion"
    assert receipt["intent_kind"] == "pre_recorded_schedule"
    assert receipt["row_count"] == 3
    assert receipt["writer_thread_count"] == 1
    assert receipt["flush_completed"] is True
    assert receipt["fsync_completed"] is True
    written = read_commands(tmp_path / "episode/trajectory_command.csv")
    assert len(written) == 3
    assert written[2]["tcp_target_x"] == ""
    assert logger.row_counts["trajectory_command"] == 3
    logger.close(completed=True)


@pytest.mark.parametrize("bad", [
    [{"unknown": 1}],
    [{"trajectory_time_s": float("nan")}],
    [{"trajectory_time_s": True}],
    [{"command_valid": 1}],
    ["row"],
    [],
])
def test_batch_schema_validation_fails_before_file_write(tmp_path, bad):
    logger = EpisodeLogger(tmp_path / "episode").start()
    with pytest.raises((TypeError, ValueError)):
        logger.append_trajectory_commands_bounded(bad, timeout_s=1.0)
    assert logger.row_counts["trajectory_command"] == 0
    logger.close(completed=True)


def test_batch_size_limit_is_fail_closed_before_writer_start(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    with pytest.raises(ValueError, match="limit"):
        logger.append_trajectory_commands_bounded([row(0)] * 200001, timeout_s=1.0)
    assert logger.row_counts["trajectory_command"] == 0
    logger.close(completed=True)


def test_batch_timeout_signals_without_waiting_for_flush_and_late_writer_stays_failed(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    entered = threading.Event()
    release = threading.Event()

    def blocked_flush():
        entered.set()
        assert release.wait(2.0)

    errors = []

    def write():
        try:
            logger.append_trajectory_commands_bounded([row(0), row(1)], timeout_s=.03)
        except BaseException as exc:
            errors.append(exc)

    with patch.object(logger._handles["trajectory_command"], "flush", side_effect=blocked_flush):
        worker = threading.Thread(target=write)
        worker.start()
        assert entered.wait(1.0)
        worker.join(1.0)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], EpisodeLoggerError)
        assert logger.failure_event.is_set()
        assert not logger.healthy_signal
        release.set()
    assert logger.wait_for_pending_writes(1.0)
    assert logger.status == "failed"
    with pytest.raises(EpisodeLoggerError):
        logger.append_trajectory_commands_bounded([row(2)], timeout_s=1.0)
    logger.close(completed=False)


def test_batch_does_not_block_independent_state_append_before_failure(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    receipt = logger.append_trajectory_commands_bounded([row(0)], timeout_s=1.0)
    assert receipt["row_count"] == 1
    logger.append_robot_state(valid=True, invalid_reason="")
    assert logger.row_counts["robot_state"] == 1
    logger.close(completed=True)
