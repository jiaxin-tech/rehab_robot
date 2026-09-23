"""Platform compatibility for replay diagnostics, without running a replay."""

import builtins
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


BUILDER = (
    Path(__file__).resolve().parents[1]
    / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"
)


def test_replay_module_imports_without_resource(monkeypatch):
    original_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ModuleNotFoundError("No module named 'resource'", name="resource")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    module = runpy.run_path(str(BUILDER), run_name="_replay_import_test")
    assert module["memory_peak_mib"]() is None


@pytest.mark.parametrize(
    ("platform", "peak_rss"),
    [("darwin", 12 * 1024**2), ("linux", 12 * 1024)],
)
def test_memory_peak_retains_platform_units(monkeypatch, platform, peak_rss):
    from external_simulation.myoleg_reference_trajectory_replay_v1 import build_and_replay

    resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=peak_rss),
    )
    monkeypatch.setattr(build_and_replay, "resource", resource)
    monkeypatch.setattr(build_and_replay.sys, "platform", platform)
    assert build_and_replay.memory_peak_mib() == pytest.approx(12.0)
