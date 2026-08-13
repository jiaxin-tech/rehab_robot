from __future__ import annotations

import queue
import time

import pytest

from scripts.validate_wrench_process_isolation import offline_case_passes, run_offline_case
from scripts.wrench_process_isolation import (
    RtProcessSupervisor,
    WrenchProcessSupervisor,
    drain_latest,
    publish_latest,
)


def _wait_for(supervisor: WrenchProcessSupervisor, predicate, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    latest = supervisor.poll()
    while time.monotonic() < deadline:
        latest = supervisor.poll()
        if predicate(latest):
            return latest
        time.sleep(0.005)
    return latest


def test_bounded_latest_snapshot_replaces_old_value_without_blocking() -> None:
    channel: queue.Queue = queue.Queue(maxsize=1)
    assert publish_latest(channel, {"sequence_id": 1})
    started = time.perf_counter()
    assert publish_latest(channel, {"sequence_id": 2})
    assert time.perf_counter() - started < 0.05
    assert drain_latest(channel) == {"sequence_id": 2}


@pytest.mark.parametrize(
    ("name", "behavior", "delay_s", "expect_code"),
    [
        ("normal", "normal", 0.001, None),
        ("slow_40ms", "normal", 0.040, None),
        ("error_263", "error263", 0.001, 263),
        ("worker_exception", "exception", 0.001, None),
    ],
)
def test_normal_slow_and_error_ipc(name, behavior, delay_s, expect_code) -> None:
    supervisor = WrenchProcessSupervisor(stale_age_ms=150, worker_hung_ms=500)
    supervisor.start(mode="offline", config={
        "behavior": behavior,
        "delay_s": delay_s,
        "target_hz": 20.0,
    })
    observation = _wait_for(
        supervisor,
        lambda item: item.wrench_sequence >= 1 or item.last_error is not None,
    )
    cleanup = supervisor.stop_normally()
    supervisor.close()
    assert cleanup["worker_exited"]
    assert observation.wrench_sequence >= 1
    if expect_code is not None:
        assert observation.last_error_code == expect_code
    if name == "worker_exception":
        assert "synthetic worker exception" in str(observation.last_error)


def test_stale_and_heartbeat_timeout_are_independent() -> None:
    supervisor = WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=200)
    supervisor.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.5, "target_hz": 20.0,
    })
    stale = _wait_for(supervisor, lambda item: item.wrench_stale, timeout_s=0.15)
    hung = _wait_for(supervisor, lambda item: item.worker_hung, timeout_s=0.5)
    cleanup = supervisor.terminate()
    supervisor.close()
    assert stale.wrench_stale
    assert hung.worker_hung
    assert cleanup["worker_terminated"]
    assert cleanup["graceful_disconnect_confirmed"] is False


def test_starting_worker_uses_separate_startup_timeout() -> None:
    supervisor = WrenchProcessSupervisor(
        stale_age_ms=50,
        worker_hung_ms=10,
        worker_startup_hung_ms=500,
    )
    supervisor.start(mode="offline", config={
        "behavior": "normal", "delay_s": 0.001, "target_hz": 20.0,
    })
    first = supervisor.poll()
    assert not first.worker_hung
    cleanup = supervisor.stop_normally()
    supervisor.close()
    assert cleanup["worker_exited"]


def test_permanent_hung_worker_is_terminatable_and_parent_survives() -> None:
    supervisor = WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=100)
    supervisor.start(mode="offline", config={
        "behavior": "permanent", "permanent_sleep_s": 3600.0,
    })
    hung = _wait_for(supervisor, lambda item: item.worker_hung, timeout_s=1.0)
    parent_counter = 0
    for _ in range(20):
        supervisor.poll()
        parent_counter += 1
    cleanup = supervisor.terminate()
    death = supervisor.poll()
    supervisor.close()
    assert hung.worker_hung
    assert parent_counter == 20
    assert cleanup["worker_terminated"]
    assert not death.worker_alive


def test_worker_crash_becomes_visible_without_killing_parent() -> None:
    supervisor = WrenchProcessSupervisor(stale_age_ms=50, worker_hung_ms=500)
    supervisor.start(mode="offline", config={
        "behavior": "crash", "delay_s": 0.001, "crash_after": 1,
    })
    dead = _wait_for(supervisor, lambda item: not item.worker_alive and item.worker_exitcode is not None)
    supervisor.join(0.1)
    supervisor.close()
    assert not dead.worker_alive
    assert dead.worker_exitcode == 17


def test_offline_case_acceptance_rejects_low_parent_rate() -> None:
    result = {
        "scenario": "normal",
        "main_loop_rate_hz": 5.0,
        "parent_work_counter": 1,
        "main_loop_tick_count": 1,
        "maximum_wrench_sequence_seen": 1,
        "cleanup": {"worker_exited": True},
    }
    assert not offline_case_passes(result)


def test_real_ten_second_offline_block_does_not_block_parent_loop() -> None:
    result = run_offline_case({
        "name": "long_block_10s",
        "config": {"behavior": "normal", "delay_s": 10.0},
        "duration_s": 10.4,
        "hung_ms": 250.0,
    })
    assert result["pass"]
    assert result["main_loop_tick_count"] >= 800
    assert result["worker_hung_seen"]
    assert result["wrench_stale_seen"]


def test_saturated_process_ipc_keeps_worker_and_parent_progressing() -> None:
    result = run_offline_case({
        "name": "ipc_saturation",
        "config": {"behavior": "normal", "delay_s": 0.0001, "target_hz": 2000.0},
        "duration_s": 0.8,
    })
    assert result["pass"]
    assert result["maximum_wrench_sequence_seen"] > result["main_loop_tick_count"]


def test_rt_process_publishes_plain_state_and_stops_cleanly() -> None:
    supervisor = RtProcessSupervisor(stale_age_ms=50, worker_hung_ms=200)
    supervisor.start({"offline": True, "publish_hz": 125.0})
    observation = _wait_for(
        supervisor,
        lambda item: item.rt_valid and item.rt_sequence > 2,
    )
    cleanup = supervisor.stop_normally()
    final = supervisor.poll()
    supervisor.close()
    assert observation.rt_age_ms is not None and observation.rt_age_ms < 50
    assert observation.operation_state == "IDLE"
    assert cleanup["worker_exited"]
    assert cleanup["graceful_disconnect_confirmed"] is True
    assert not final.worker_alive
