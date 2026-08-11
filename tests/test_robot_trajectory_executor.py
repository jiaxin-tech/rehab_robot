"""Offline scheduler and unified-stop tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pandas as pd
import pytest

from collection.episode_logger import EpisodeLogger, EpisodeLoggerError
from collection.real_robot_acquisition import AcquisitionHealth
from control.execution_preflight import (
    ExecutionPreflight,
    _LIVE_PREFLIGHT_TOKEN,
    audit_execution_trajectory_content,
    experiment_safety_digest,
    trajectory_execution_digest,
)
from control.robot_trajectory_executor import RokaeMotionExecutor
from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from safety.experiment_safety import ExperimentSafetyConfig


def _safety():
    return ExperimentSafetyConfig(
        max_tcp_speed_m_s=1_000_000.0,
        max_tcp_acceleration_m_s2=1_000_000_000.0,
        max_start_anchor_position_error_m=0.01,
        max_start_anchor_orientation_error_rad=0.01,
        max_command_lateness_s=0.1,
        max_force_n=20.0,
        max_torque_nm=5.0,
        max_state_age_s=1.0,
        max_wrench_age_s=1.0,
        max_state_wrench_skew_s=1.0,
        workspace_min_base_m=(-1.0, -1.0, -1.0),
        workspace_max_base_m=(1.0, 1.0, 1.0),
        expected_robot_model="xMate6-Pro",
        expected_robot_serial_number="SN-EXECUTOR",
        expected_controller_version="controller-test",
        reviewed_tool_name="rehab_cuff",
        reviewed_workpiece_name="rehab_fixture",
        reviewed_payload_mass_kg=1.25,
        reviewed_payload_cog_m=(0.0, 0.0, 0.08),
        reviewed_payload_inertia_kg_m2=(0.01, 0.02, 0.03),
        reviewed_joint_soft_limits_rad=tuple((-2.0, 2.0) for _ in range(6)),
        reviewed_rt_filter_hz=25.0,
        reviewed_rt_network_tolerance_percent=20.0,
        robot_identity_reviewed=True,
        tool_workpiece_reviewed=True,
        payload_configuration_reviewed=True,
        collision_configuration_reviewed=True,
        joint_soft_limits_reviewed=True,
        realtime_configuration_reviewed=True,
        reviewed=True,
        notes="synthetic test values",
    )


def _health():
    return AcquisitionHealth(
        host_time_s=1.0,
        state_time_s=1.0,
        wrench_time_s=1.0,
        state_age_s=0.0,
        wrench_age_s=0.0,
        state_wrench_skew_s=0.0,
        state_valid=True,
        wrench_valid=True,
        state_thread_alive=True,
        wrench_thread_alive=True,
        alignment_thread_alive=True,
        query_duration_ms=1.0,
        force_magnitude_n=1.0,
        torque_magnitude_nm=0.1,
        valid=True,
        invalid_reason="",
    )


class Acquisition:
    def __init__(self):
        self.health = _health()
        self.summary = {
            "connected": True,
            "state_valid": True,
            "state_stream_thread_alive": True,
            "operation_state": "IDLE",
            "collision_state_query_valid": True,
            "collision_state": False,
            "joint_position_rad": [0.0] * 6,
            "robot_metadata": {
                "robot_model": "xMate6-Pro",
                "robot_serial_number": "SN-EXECUTOR",
                "controller_version": "controller-test",
                "joint_soft_limits_rad": [[-2.0, 2.0]] * 6,
                "sdk_tool_payload": {
                    "toolset_load_mass_kg": 1.25,
                    "toolset_load_cog_m": [0.0, 0.0, 0.08],
                    "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
                    "sdk_available_tool_names": ["rehab_cuff"],
                    "sdk_available_workobject_names": ["rehab_fixture"],
                    "active_hmi_tool_workobject_verified": False,
                },
            },
        }
        self.adapter = SimpleNamespace(
            get_robot_state_summary=lambda: dict(self.summary)
        )

    def latest_health(self):
        return self.health

    def latest_state_frame(self):
        return SimpleNamespace(
            valid=True,
            tcp_position_m=(0.3, 0.0, 0.4),
            tcp_orientation_rad=(0.0, 0.0, 0.0),
        )


class Motion:
    def __init__(self):
        self.calls = []

    def attach_externally_prepared(self, *, reviewed_filter_hz):
        self.calls.append(("attach", reviewed_filter_hz))

    def start_cartesian_hold(self, pose):
        self.calls.append(("start", tuple(pose)))

    def send_cartesian_target(self, pose):
        self.calls.append(("send", tuple(pose)))

    def has_motion_error(self):
        return False

    def request_stop(self, reason):
        self.calls.append(("stop", reason))


def _trajectory():
    time_s = np.asarray([0.0, 0.001, 0.002])
    q_hip = np.asarray([0.1, 0.1001, 0.1])
    q_knee = np.asarray([0.2, 0.2001, 0.2])
    _, _, pull_x, pull_z = forward_kinematics(q_hip, q_knee, L1, L2)
    delta = np.column_stack(
        (
            pull_x - pull_x[0],
            np.zeros(3),
            pull_z - pull_z[0],
        )
    )
    delta[-1] = 0.0
    xyz = np.asarray((0.3, 0.0, 0.4)) + delta
    velocity = np.gradient(xyz, time_s, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time_s, axis=0, edge_order=2)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "trajectory_id": ["reference_measured_asymmetric_closed_slow"] * 3,
            "trajectory_phase": ["start", "middle", "end"],
            "delta_x_R": delta[:, 0],
            "delta_y_R": delta[:, 1],
            "delta_z_R": delta[:, 2],
            "tcp_x_base": xyz[:, 0],
            "tcp_y_base": xyz[:, 1],
            "tcp_z_base": xyz[:, 2],
            "tcp_rx": [0.0, 0.0, 0.0],
            "tcp_ry": [0.0, 0.0, 0.0],
            "tcp_rz": [0.0, 0.0, 0.0],
            "tcp_vx_base": velocity[:, 0],
            "tcp_vy_base": velocity[:, 1],
            "tcp_vz_base": velocity[:, 2],
            "tcp_ax_base": acceleration[:, 0],
            "tcp_ay_base": acceleration[:, 1],
            "tcp_az_base": acceleration[:, 2],
            "q_hip_ref": q_hip,
            "q_knee_ref": q_knee,
            "theta_shank_ref": q_hip - q_knee,
            "trajectory_valid": [True, True, True],
            "experiment_mode": ["start_anchored_relative"] * 3,
            "tcp_orientation_strategy": ["fixed_at_start_anchor"] * 3,
        }
    )


def _preflight(allowed=True, trajectory=None, safety=None):
    bound_trajectory = _trajectory() if trajectory is None else trajectory
    bound_safety = _safety() if safety is None else safety
    content_reasons, max_speed, max_acceleration = audit_execution_trajectory_content(
        bound_trajectory,
        bound_safety,
        expected_trajectory_id="reference_measured_asymmetric_closed_slow",
    )
    assert content_reasons == ()
    return ExecutionPreflight(
        allowed=allowed,
        reasons=() if allowed else ("blocked_test",),
        trajectory_id="reference_measured_asymmetric_closed_slow",
        anchor_id="anchor",
        maximum_tcp_speed_m_s=max_speed,
        maximum_tcp_acceleration_m_s2=max_acceleration,
        runtime_robot_summary={"synthetic_test": True},
        evaluation_phase="live",
        trajectory_sha256=trajectory_execution_digest(bound_trajectory),
        experiment_safety_sha256=experiment_safety_digest(bound_safety),
        _live_token=_LIVE_PREFLIGHT_TOKEN,
    )


def test_executor_logs_before_each_send_and_stops_once(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    result = executor.execute(_trajectory(), _preflight())
    logger.close(completed=True)
    assert result.completed
    assert result.commands_dispatched == 3
    assert logger.row_counts["trajectory_command"] == 3
    assert motion.calls[0] == ("attach", 25.0)
    assert motion.calls[1][0] == "start"
    assert [call[0] for call in motion.calls if isinstance(call, tuple)].count("send") == 2
    assert [call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"] == [
        ("stop", "trajectory_completed")
    ]


def test_preflight_failure_sends_no_motion(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    with pytest.raises(PermissionError, match="blocked_test"):
        executor.execute(_trajectory(), _preflight(False))
    logger.close(completed=False)
    assert motion.calls == []


def test_runtime_stale_state_enters_unified_stop(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()
    acquisition.health = replace(acquisition.health, state_age_s=2.0)
    motion = Motion()
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="state_stale"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert len([call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]) == 1


def test_logging_failure_prevents_target_dispatch(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    original_append = logger._append

    def fail_append(stream_name, values):
        if stream_name == "trajectory_command":
            raise EpisodeLoggerError("simulated disk failure")
        original_append(stream_name, values)

    logger._append = fail_append
    with pytest.raises(EpisodeLoggerError, match="disk failure"):
        executor.execute(_trajectory(), _preflight())
    logger._append = original_append
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] in {"start", "send"}
        for call in motion.calls
    )
    assert len([call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]) == 1


def test_operator_stop_is_idempotent(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    executor.request_stop("operator stop")
    executor.request_stop("second stop")
    logger.close(completed=False)
    assert motion.calls == [("stop", "operator stop")]


def test_prestart_stop_cannot_be_cleared_by_execute(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    executor.request_stop("operator prestart stop")
    with pytest.raises(RuntimeError, match="stopped before start"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert motion.calls == [("stop", "operator prestart stop")]


def test_executor_cannot_be_reused_after_completion(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    executor.execute(_trajectory(), _preflight())
    with pytest.raises(RuntimeError, match="single-use"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=True)


def test_attach_is_followed_by_fresh_health_gate_before_start(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()

    class HealthChangingMotion(Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.health = replace(acquisition.health, state_age_s=2.0)

    motion = HealthChangingMotion()
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="post-attach.*state_stale"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert motion.calls[0] == ("attach", 25.0)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_collision_change_during_attach_blocks_realtime_start(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()

    class CollisionChangingMotion(Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.summary["collision_state"] = True

    motion = CollisionChangingMotion()
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="collision_detected_after_attachment"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_payload_change_during_attach_blocks_realtime_start(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()

    class PayloadChangingMotion(Motion):
        def attach_externally_prepared(self, *, reviewed_filter_hz):
            super().attach_externally_prepared(
                reviewed_filter_hz=reviewed_filter_hz
            )
            acquisition.summary["robot_metadata"]["sdk_tool_payload"][
                "toolset_load_mass_kg"
            ] = 2.0

    motion = PayloadChangingMotion()
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="payload_mass_changed_after_preflight"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_slow_intent_logging_misses_deadline_without_dispatch(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    original_append = logger._append

    def slow_append(stream_name, values):
        if stream_name == "trajectory_command":
            time.sleep(0.02)
        original_append(stream_name, values)

    logger._append = slow_append
    motion = Motion()
    safety = replace(_safety(), max_command_lateness_s=0.005)
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, safety)
    with pytest.raises((RuntimeError, EpisodeLoggerError), match="durability|deadline"):
        executor.execute(_trajectory(), _preflight(safety=safety))
    assert logger.wait_for_pending_writes(1.0)
    logger._append = original_append
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] in {"start", "send"}
        for call in motion.calls
    )


def test_stop_failure_never_returns_completed_result(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()

    class StopFailingMotion(Motion):
        def __init__(self):
            super().__init__()
            self.stop_attempts = 0

        def request_stop(self, reason):
            super().request_stop(reason)
            self.stop_attempts += 1
            if self.stop_attempts == 1:
                raise RuntimeError("simulated stopLoop failure")

    motion = StopFailingMotion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    with pytest.raises(RuntimeError, match="stopLoop failure"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert len(
        [call for call in motion.calls if isinstance(call, tuple) and call[0] == "stop"]
    ) == 2


def test_unbound_preflight_never_reaches_motion(tmp_path):
    trajectory = _trajectory()
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    manual = replace(
        _preflight(trajectory=trajectory),
        evaluation_phase="offline",
        _live_token=None,
    )
    with pytest.raises(PermissionError, match="not_live_bound"):
        executor.execute(trajectory, manual)
    logger.close(completed=False)
    assert motion.calls == []


def test_preflight_cannot_be_reused_with_a_different_safety_snapshot(tmp_path):
    trajectory = _trajectory()
    approved = _preflight(trajectory=trajectory)
    changed_safety = replace(_safety(), reviewed_rt_filter_hz=30.0)
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(
        motion,
        Acquisition(),
        logger,
        changed_safety,
    )
    with pytest.raises(PermissionError, match="safety_binding_mismatch"):
        executor.execute(trajectory, approved)
    logger.close(completed=False)
    assert motion.calls == []


def test_mutated_trajectory_never_reaches_motion(tmp_path):
    trajectory = _trajectory()
    logger = EpisodeLogger(tmp_path / "episode-mutated").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    bound = _preflight(trajectory=trajectory)
    trajectory.loc[1, "tcp_x_base"] += 1e-6
    with pytest.raises(PermissionError, match="binding_mismatch"):
        executor.execute(trajectory, bound)
    logger.close(completed=False)
    assert motion.calls == []


def test_string_false_is_not_accepted_as_trajectory_valid(tmp_path):
    trajectory = _trajectory()
    bound = _preflight(trajectory=trajectory)
    trajectory["trajectory_valid"] = ["True", "False", "True"]
    # Rebind only the digest to exercise the scheduler's independent content
    # audit instead of stopping at mutation detection.
    bound = replace(
        bound,
        trajectory_sha256=trajectory_execution_digest(trajectory),
    )
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    with pytest.raises(PermissionError, match="not_strict_boolean"):
        executor.execute(trajectory, bound)
    logger.close(completed=False)
    assert motion.calls == []


def test_final_metadata_persistence_is_followed_by_fresh_collision_gate(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()
    motion = Motion()
    original_update = logger.update_metadata

    def collision_after_approval_metadata(updates):
        original_update(updates)
        if updates.get("initial_hold_intent_persisted") is True:
            acquisition.summary["collision_state"] = True

    logger.update_metadata = collision_after_approval_metadata
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="collision_detected_after_attachment"):
        executor.execute(_trajectory(), _preflight())
    logger.update_metadata = original_update
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_metadata_delay_expires_initial_hold_deadline_before_start(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    original_update = logger.update_metadata

    def slow_final_metadata(updates):
        original_update(updates)
        if updates.get("initial_hold_intent_persisted") is True:
            time.sleep(0.01)

    logger.update_metadata = slow_final_metadata
    safety = replace(_safety(), max_command_lateness_s=0.005)
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, safety)
    with pytest.raises(RuntimeError, match="initial hold deadline expired"):
        executor.execute(_trajectory(), _preflight(safety=safety))
    logger.update_metadata = original_update
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "start" for call in motion.calls
    )


def test_health_change_during_command_logging_blocks_that_target(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = Acquisition()
    motion = Motion()
    original_bounded = logger.append_trajectory_command_bounded
    calls = 0

    def change_health_after_logging(*args, **kwargs):
        nonlocal calls
        original_bounded(*args, **kwargs)
        calls += 1
        if calls == 2:
            acquisition.health = replace(acquisition.health, wrench_age_s=2.0)

    logger.append_trajectory_command_bounded = change_health_after_logging
    executor = RokaeMotionExecutor(motion, acquisition, logger, _safety())
    with pytest.raises(RuntimeError, match="post-log.*wrench_stale"):
        executor.execute(_trajectory(), _preflight())
    logger.append_trajectory_command_bounded = original_bounded
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "send" for call in motion.calls
    )


def test_operator_stop_published_during_logging_prevents_later_send(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    motion = Motion()
    executor = RokaeMotionExecutor(motion, Acquisition(), logger, _safety())
    original_bounded = logger.append_trajectory_command_bounded
    second_log_entered = threading.Event()
    release_second_log = threading.Event()
    calls = 0

    def block_second_log(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_log_entered.set()
            assert release_second_log.wait(timeout=1.0)
        return original_bounded(*args, **kwargs)

    logger.append_trajectory_command_bounded = block_second_log
    errors: list[BaseException] = []

    def run_execution():
        try:
            executor.execute(_trajectory(), _preflight())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_execution)
    worker.start()
    assert second_log_entered.wait(timeout=1.0)
    executor.request_stop("operator_stop_during_logging")
    release_second_log.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors
    logger.append_trajectory_command_bounded = original_bounded
    logger.close(completed=False)
    assert not any(
        isinstance(call, tuple) and call[0] == "send" for call in motion.calls
    )
    assert executor.stop_reason == "operator_stop_during_logging"


def test_operator_reason_can_never_be_reported_as_normal_completion(tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()

    class OperatorWinsAtCompletion(RokaeMotionExecutor):
        def request_stop(self, reason):
            if reason == "trajectory_completed" and self.stop_reason is None:
                super().request_stop("operator_stop_at_completion_boundary")
            return super().request_stop(reason)

    motion = Motion()
    executor = OperatorWinsAtCompletion(
        motion,
        Acquisition(),
        logger,
        _safety(),
    )
    with pytest.raises(RuntimeError, match="did not reach confirmed normal completion"):
        executor.execute(_trajectory(), _preflight())
    logger.close(completed=False)
    assert executor.stop_reason == "operator_stop_at_completion_boundary"
