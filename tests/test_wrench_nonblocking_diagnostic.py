from __future__ import annotations

import threading
import time

from scripts.wrench_nonblocking_diagnostic import LatestWrenchCache, NonBlockingWrenchWorker


def test_cache_becomes_stale_without_blocking_snapshot() -> None:
    cache = LatestWrenchCache(stale_threshold_ms=20.0)
    cache.set_source_alive(True)
    cache.publish_success({"force": 1}, 1_000_000_000)
    fresh = cache.snapshot(1_010_000_000)
    stale = cache.snapshot(1_030_000_000)
    assert fresh.valid and not fresh.stale and fresh.age_ms == 10.0
    assert not stale.valid and stale.stale and stale.age_ms == 30.0
    assert stale.source_alive


def test_blocked_worker_does_not_block_main_snapshot() -> None:
    entered = threading.Event()
    release = threading.Event()

    def query():
        entered.set()
        release.wait(timeout=1.0)
        return {"force": 1}

    worker = NonBlockingWrenchWorker(query, target_hz=50, stale_threshold_ms=20)
    worker.start()
    assert entered.wait(timeout=0.5)
    before = time.perf_counter()
    snapshot = worker.cache.snapshot()
    assert time.perf_counter() - before < 0.05
    assert snapshot.query_in_flight and snapshot.source_alive
    release.set()
    worker.request_stop()
    assert worker.join(timeout=1.0)


def test_worker_exception_is_visible_and_recoverable() -> None:
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("xCoreSDK getEndTorque failed (263): synthetic")
        return {"force": 2}

    worker = NonBlockingWrenchWorker(query, target_hz=200, stale_threshold_ms=100)
    worker.start()
    deadline = time.monotonic() + 1.0
    saw_error = False
    saw_recovery = False
    while time.monotonic() < deadline:
        snapshot = worker.cache.snapshot()
        saw_error |= snapshot.last_error_code == 263
        saw_recovery |= snapshot.valid and snapshot.sequence_id >= 2
        if saw_error and saw_recovery:
            break
        time.sleep(0.001)
    worker.request_stop()
    assert worker.join(timeout=1.0)
    assert saw_error and saw_recovery
    final = worker.cache.snapshot()
    assert final.valid
    assert final.last_error_code == 263


def test_worker_reports_scheduled_and_exact_call_timestamps() -> None:
    rows = []
    worker = NonBlockingWrenchWorker(
        lambda: {"force": 1},
        target_hz=200,
        stale_threshold_ms=100,
        on_result=rows.append,
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while len(rows) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    worker.request_stop()
    assert worker.join(timeout=1.0)
    assert len(rows) >= 2
    first = rows[0]
    assert first["scheduled_time_ns"] <= first["call_start_ns"] <= first["call_end_ns"]
    assert first["query_latency_ms"] == first["latency_ms"]
