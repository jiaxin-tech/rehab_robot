"""Loaded diagnostics must stop on a wrench gap despite empty-load grace."""
from types import SimpleNamespace

import pytest

from config.bundle import configuration
from control.robot_trajectory_executor import RokaeMotionExecutor
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.run_rehab_experiment import run_execute


def context(*, strict=False, stale=True, missing_force=True):
    health = SimpleNamespace(
        valid=not stale,
        invalid_reason=("wrench_stale_or_unavailable;state_wrench_skew_unavailable_or_exceeded" if stale else ""),
        wrench_stale_grace_active=stale, wrench_stale_grace_remaining_s=.2,
        state_thread_alive=True, wrench_thread_alive=True, alignment_thread_alive=True,
        state_age_s=.001, wrench_age_s=.2 if stale else .001,
        state_wrench_skew_s=.1 if stale else .001,
        force_magnitude_n=None if missing_force and stale else 1.,
        torque_magnitude_nm=None if missing_force and stale else 1.)
    stops = []
    adapter = SimpleNamespace()
    acquisition = SimpleNamespace(adapter=adapter, latest_health=lambda: health)
    logger = SimpleNamespace(healthy_signal=True)
    motion = SimpleNamespace(request_stop=stops.append)
    safety = ExperimentSafetyConfig(**configuration()["motion"])
    executor = RokaeMotionExecutor(motion, acquisition, logger, safety,
                                  strict_wrench_freshness=strict)
    return executor, health, stops


@pytest.mark.parametrize("missing_force", [False, True])
def test_strict_rejects_stale_age_and_skew_that_default_grace_allows(missing_force):
    ordinary, _, _ = context(strict=False, missing_force=missing_force)
    assert ordinary._runtime_reasons() == ()
    strict, _, _ = context(strict=True, missing_force=missing_force)
    reasons = strict._runtime_reasons()
    assert "wrench_stale" in reasons
    assert "state_wrench_skew_limit_exceeded" in reasons
    assert "wrench_stale_or_unavailable" in reasons[0]
    if missing_force:
        assert "force_magnitude_n_invalid" in reasons
        assert "force_limit_or_value_unavailable" in reasons
        assert "torque_magnitude_nm_invalid" in reasons
        assert "torque_limit_or_value_unavailable" in reasons


def test_strict_watchdog_requests_stop_on_grace_active_gap():
    executor, _, stops = context(strict=True)
    executor._runtime_watchdog_loop()
    assert len(stops) == 1
    assert stops[0].startswith("runtime_watchdog:runtime safety gate:")
    assert "wrench_stale" in stops[0]
    assert executor._stop_event.is_set() and executor._stop_confirmed


@pytest.mark.parametrize("strict", [False, True])
def test_fresh_data_passes_and_other_limits_stay_enforced(strict):
    executor, health, _ = context(strict=strict, stale=False)
    assert executor._runtime_reasons() == ()
    health.force_magnitude_n = executor.safety.max_force_n + 1
    health.state_age_s = executor.safety.max_state_age_s + .01
    reasons = executor._runtime_reasons()
    assert "force_limit_exceeded" in reasons and "state_stale" in reasons


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_strict_flag_is_boolean_and_runner_rejects_it_before_file_or_device_io(invalid):
    with pytest.raises(TypeError, match="strict_wrench_freshness must be bool"):
        context(strict=invalid)
    with pytest.raises(TypeError, match="strict_wrench_freshness must be bool"):
        run_execute(robot_ip="192.0.2.1", episode_dir="nonexistent", anchor_path="nonexistent",
            requested_anchor_id="offline", frame_config_path="nonexistent",
            safety_config_path="nonexistent", trajectory_id="offline", enable_motion=False,
            operator_confirmation="", strict_wrench_freshness=invalid)
