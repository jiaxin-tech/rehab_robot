"""Concurrent cache publication must not overwrite or hide disk failures."""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark import simulation


def write_trace(path, **changes):
    values = dict(tau_nm=np.arange(12, dtype=float).reshape(6, 2),
                  decomposition_residual_nm=1e-14,
                  simulation_identity="model", trajectory_sha256="trajectory")
    values.update(changes)
    np.savez_compressed(path, **values)
    return path


def test_concurrent_publishers_keep_one_complete_trace(tmp_path):
    staging = [write_trace(tmp_path / f"worker_{i}.npz") for i in range(6)]
    expected = staging[0].read_bytes()
    destination = tmp_path / "trace.npz"
    start = threading.Barrier(len(staging))

    def publish(path):
        start.wait()
        simulation._publish_cache(path, destination)

    with ThreadPoolExecutor(max_workers=len(staging)) as pool:
        list(pool.map(publish, staging))
    assert destination.read_bytes() == expected
    assert all(not path.exists() for path in staging)


def test_existing_open_reader_is_not_replaced(tmp_path, monkeypatch):
    destination = write_trace(tmp_path / "trace.npz")
    original = destination.stat()
    staging = write_trace(tmp_path / "worker.npz")
    with np.load(destination, allow_pickle=False) as reader:
        simulation._publish_cache(staging, destination)
        np.testing.assert_array_equal(reader["tau_nm"], np.arange(12).reshape(6, 2))
    assert destination.stat().st_ino == original.st_ino
    assert destination.stat().st_mtime_ns == original.st_mtime_ns
    assert not staging.exists()


def test_reader_opened_after_install_does_not_block_cleanup(tmp_path, monkeypatch):
    destination = tmp_path / "trace.npz"
    staging = write_trace(tmp_path / "worker.npz")
    real_link = simulation.os.link
    readers = []

    def link_and_open(source, target):
        real_link(source, target)
        readers.append(np.load(target, allow_pickle=False))

    monkeypatch.setattr(simulation.os, "link", link_and_open)
    try:
        simulation._publish_cache(staging, destination)
        np.testing.assert_array_equal(readers[0]["tau_nm"], np.arange(12).reshape(6, 2))
    finally:
        for reader in readers:
            reader.close()
    assert not staging.exists()


@pytest.mark.parametrize("change", [
    {"simulation_identity": "wrong-model"},
    {"trajectory_sha256": "wrong-trajectory"},
    {"tau_nm": np.zeros((6, 2))},
    {"tau_nm": np.zeros((6, 1))},
    {"tau_nm": np.full((6, 2), np.nan)},
    {"decomposition_residual_nm": 1e-2},
])
def test_conflicting_existing_trace_is_rejected_without_overwrite(tmp_path, change):
    destination = write_trace(tmp_path / "trace.npz", **change)
    original = destination.read_bytes()
    staging = write_trace(tmp_path / "worker.npz")
    with pytest.raises(OSError, match="CONCURRENT_CACHE_CONTENT_MISMATCH"):
        simulation._publish_cache(staging, destination)
    assert destination.read_bytes() == original
    assert not staging.exists()


def test_corrupt_existing_file_is_rejected(tmp_path):
    destination = tmp_path / "trace.npz"
    destination.write_bytes(b"not an NPZ")
    staging = write_trace(tmp_path / "worker.npz")
    with pytest.raises(OSError, match="CONCURRENT_CACHE_CONTENT_MISMATCH"):
        simulation._publish_cache(staging, destination)
    assert destination.read_bytes() == b"not an NPZ"
    assert not staging.exists()


def test_other_io_failure_propagates(tmp_path, monkeypatch):
    destination = tmp_path / "trace.npz"
    staging = write_trace(tmp_path / "worker.npz")

    def denied(source, target):
        raise PermissionError("synthetic filesystem denial")

    monkeypatch.setattr(simulation.os, "link", denied)
    with pytest.raises(PermissionError, match="synthetic filesystem denial"):
        simulation._publish_cache(staging, destination)
    assert not destination.exists()
    assert not staging.exists()
