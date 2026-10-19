"""Local single-use ledger tests; approvals and hardware are test doubles."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing as mp

import pytest

import safety.empty_load_commissioning_release as releases


def _claim_in_spawned_process(release, root, results):
    releases._ROOT = root
    restored = replace(release, _token=releases._VALIDATED)
    try:
        restored.claim_execution()
    except PermissionError:
        results.put(False)
    else:
        results.put(True)


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "_ROOT", tmp_path)
    now = datetime.now(timezone.utc)
    payload = dict(issued_at=(now-timedelta(minutes=1)).isoformat(),
                   expires_at=(now+timedelta(hours=1)).isoformat())
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return releases.ValidatedEmptyLoadCommissioningRelease(
        payload_json=json.dumps(payload), approval_path=str(path),
        approval_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        files=(), _token=releases._VALIDATED)


def test_concurrent_claims_have_one_winner_and_survive_reconstruction(release):
    def claim(_):
        try:
            return release.claim_execution()
        except PermissionError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert sum(r is not None for r in results) == 1
    restored = replace(release)
    with pytest.raises(PermissionError, match="already claimed"):
        restored.require_unused()
    with pytest.raises(PermissionError, match="already claimed"):
        restored.claim_execution()
    # Runtime validity remains checkable by the process that owns the claim.
    restored.require_current(check_files=True)


def test_spawned_processes_cannot_both_claim_same_release(release, tmp_path):
    context = mp.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=_claim_in_spawned_process,
        args=(release, tmp_path, results)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert sorted(results.get(timeout=2) for _ in processes) == [False, True]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        results.close()


def test_failed_claim_flush_stays_consumed(release, monkeypatch):
    def fail_fsync(_):
        raise OSError("offline storage fault")
    monkeypatch.setattr(releases.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="storage fault"):
        release.claim_execution()
    with pytest.raises(PermissionError, match="already claimed"):
        release.require_unused()


def test_changed_approval_is_rejected_before_claim(release):
    from pathlib import Path
    Path(release.approval_path).write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="reviewed file changed"):
        release.claim_execution()
    assert not release._claim_path().exists()


@pytest.mark.parametrize("already_claimed", [False, True])
def test_executor_claims_before_first_intent_and_never_reuses(release, tmp_path, monkeypatch, already_claimed):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    from tests.test_robot_trajectory_executor import Acquisition, Motion, _preflight, _safety, _trajectory

    # Runtime identity checks have separate coverage; isolate the real claim
    # integration using a synthetic approval and the offline motion adapter.
    monkeypatch.setattr(releases.ValidatedEmptyLoadCommissioningRelease, "require_runtime", lambda *_: None)
    if already_claimed:
        release.claim_execution()
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    original_start = motion.start_cartesian_hold

    def start(pose):
        assert release._claim_path().is_file()
        assert logger.row_counts["trajectory_command"] == 1
        return original_start(pose)

    motion.start_cartesian_hold = start
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    preflight = replace(_preflight(), release_approval=release)
    try:
        if already_claimed:
            with pytest.raises(PermissionError, match="already claimed"):
                executor.execute(_trajectory(), preflight)
            assert not any(c[0] == "start" for c in motion.calls)
            assert logger.row_counts["trajectory_command"] == 0
        else:
            assert executor.execute(_trajectory(), preflight).completed
    finally:
        logger.close(completed=False)


def test_preflight_failure_does_not_consume_release(release, tmp_path):
    from collection.episode_logger import EpisodeLogger
    from control.robot_trajectory_executor import RokaeMotionExecutor
    from tests.test_robot_trajectory_executor import Acquisition, Motion, _preflight, _safety, _trajectory
    logger = EpisodeLogger(tmp_path / "episode").start()
    try:
        executor = RokaeMotionExecutor(Motion(), Acquisition(), logger, _safety())
        with pytest.raises(PermissionError, match="blocked_test"):
            executor.execute(_trajectory(), replace(_preflight(False), release_approval=release))
        assert not release._claim_path().exists()
    finally:
        logger.close(completed=False)
