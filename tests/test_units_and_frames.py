"""Offline coverage for the xCoreSDK-only snapshot collection contract."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from collection.collector import DataCollector
from collection.safety_guard import SafetyGuard
from collection.snapshot import compose_robot_state_sample
from collection.state import (
    InternalWrenchFrame,
    KinematicStateFrame,
    calculate_software_bias,
    rpy_euler_xyz_rotation_matrix,
    rotate_vector,
    transform_wrench,
)
from collection.trajectory import (
    TrajectoryGeometry,
    calibrate_joint_center,
    project_along_tangent,
)
from config import settings
from control.mpc_controller import MPCController
from hardware.windows.rokae_internal_wrench import RokaeInternalWrenchSource
from hardware.windows.rokae_xcore import RokaeRobot
from models.comfort_net import load_dataset
from scripts.train_pinn import load_tangential_episode


def _state_frame(now_s: float, *, sequence_id: int = 1, position=(0.2, 0.0, 0.0),
                 velocity=(0.1, 0.0, 0.0), valid: bool = True,
                 reason: str = "", collision: bool | None = None) -> KinematicStateFrame:
    return KinematicStateFrame(
        sequence_id=sequence_id,
        host_monotonic_time_s=now_s,
        wall_time_iso="2026-01-01T00:00:00.000+00:00",
        robot_device_time_s=None,
        valid=valid,
        invalid_reason=reason,
        tcp_position_m=tuple(float(value) for value in position),
        tcp_orientation_rad=(0.0, 0.0, 0.0),
        tcp_linear_velocity_mps=tuple(float(value) for value in velocity),
        tcp_angular_velocity_radps=(0.0, 0.0, 0.0),
        velocity_source="numerical_difference_realtime_frame_rpy",
        joint_position_rad=(0.0,) * 6,
        joint_velocity_radps=(0.0,) * 6,
        pose_time_s=now_s,
        joint_time_s=now_s,
        velocity_time_s=now_s,
        operation_state="IDLE",
        collision_state=collision,
        controller_error=None,
        keypad_state=(),
    )


def _wrench_frame(now_s: float, *, sequence_id: int = 1, valid: bool = True,
                  reason: str = "", force=(1.0, 2.0, 3.0),
                  torque=(0.1, 0.2, 0.3),
                  transform_kind: str = "sdk_base") -> InternalWrenchFrame:
    return InternalWrenchFrame(
        sequence_id=sequence_id,
        host_monotonic_time_s=now_s,
        wall_time_iso="2026-01-01T00:00:00.000+00:00",
        valid=valid,
        invalid_reason=reason,
        source=settings.ROBOT_FORCE_SOURCE,
        joint_measured_torque_nm=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        joint_external_torque_nm=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        cartesian_force_raw_n=force,
        cartesian_torque_raw_nm=torque,
        raw_force_frame="world",
        cartesian_force_bias_n=(0.1, 0.2, 0.3),
        cartesian_torque_bias_nm=(0.01, 0.02, 0.03),
        cartesian_force_corrected_n=(force[0] - 0.1, force[1] - 0.2, force[2] - 0.3),
        cartesian_torque_corrected_nm=(torque[0] - 0.01, torque[1] - 0.02, torque[2] - 0.03),
        cartesian_force_base_n=(force[0] - 0.1, force[1] - 0.2, force[2] - 0.3),
        cartesian_torque_base_nm=(torque[0] - 0.01, torque[1] - 0.02, torque[2] - 0.03),
        base_transform_kind=transform_kind,
        force_time_s=now_s,
        torque_time_s=now_s,
        force_query_started_s=now_s - 0.001,
        force_query_finished_s=now_s + 0.001,
    )


class _LiveRobot:
    def __init__(self) -> None:
        self.index = 0
        self.stopped = False
        self.collision = False

    def get_state_frame(self) -> KinematicStateFrame:
        self.index += 1
        now_s = time.monotonic()
        return _state_frame(
            now_s,
            sequence_id=self.index,
            position=(min(0.9, 0.05 * self.index), 0.0, 0.0),
            collision=self.collision,
        )

    def get_robot_metadata(self):
        return {"robot_model": "fake-rokae", "xcore_sdk_version": "0.7.0"}

    def get_joint_soft_limits_rad(self):
        return tuple((-2.0, 2.0) for _ in range(6))

    def get_collision_state(self):
        return self.collision

    def stop(self):
        self.stopped = True


class _LiveWrench:
    @property
    def bias(self):
        return (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)

    def snapshot(self, now_s: float) -> InternalWrenchFrame:
        return _wrench_frame(now_s, sequence_id=1)


class _FakeNativeSpeedRobot:
    def __init__(self) -> None:
        self.speed_mm_s = None

    def setDefaultSpeed(self, speed_mm_s, ec):
        self.speed_mm_s = speed_mm_s


class _CalibrationRobot:
    def __init__(self, points):
        self._points = iter(points)

    def get_cartesian_pose(self):
        return next(self._points)


class UnitAndFrameTests(unittest.TestCase):
    def test_sdk_boundary_uses_si_and_only_converts_vendor_speed(self):
        pose = [0.3, -0.2, 0.35, 0.0, np.pi / 2, 0.0]
        self.assertEqual(RokaeRobot._pose_to_sdk(pose), pose)
        self.assertEqual(RokaeRobot._pose_from_sdk(pose), pose)
        robot = RokaeRobot("192.0.2.1", max_linear_speed_m_s=0.8)
        native = _FakeNativeSpeedRobot()
        robot._robot = native
        robot.is_connected = True
        robot.set_speed(25)
        self.assertAlmostEqual(native.speed_mm_s, 200.0)

    def test_snapshot_keeps_sources_times_and_distinct_joint_torques(self):
        sample = compose_robot_state_sample(
            _state_frame(10.0), _wrench_frame(10.0), sample_time_s=10.01
        )
        self.assertTrue(sample.valid)
        self.assertEqual(sample.joint_measured_torque_nm[0], 1.0)
        self.assertEqual(sample.joint_external_torque_nm[0], 10.0)
        self.assertEqual(sample.cartesian_force_raw_n, (1.0, 2.0, 3.0))
        self.assertEqual(sample.cartesian_force_corrected_n, (0.9, 1.8, 2.7))
        self.assertAlmostEqual(sample.force_query_duration_ms, 2.0)
        self.assertLess(sample.state_internal_skew_ms, 0.001)
        pending = compose_robot_state_sample(
            _state_frame(11.0),
            _wrench_frame(11.0, transform_kind="rotation_only_pending_robot_validation"),
            sample_time_s=11.01,
        )
        self.assertFalse(pending.valid)
        self.assertIn("base_wrench_rotation_requires_robot_validation", pending.invalid_reason)

        previous = compose_robot_state_sample(
            _state_frame(12.0, velocity=(0.1, 0.0, 0.0)),
            _wrench_frame(12.0), sample_time_s=12.001,
        )
        accelerated = compose_robot_state_sample(
            _state_frame(12.02, velocity=(0.2, 0.0, 0.0)),
            _wrench_frame(12.02), sample_time_s=12.021,
            previous_sample=previous,
        )
        self.assertEqual(accelerated.acceleration_source, "numerical_difference")
        self.assertAlmostEqual(accelerated.tcp_linear_acceleration_est_mps2[0], 5.0)

    def test_stale_skew_and_future_timestamps_are_invalid(self):
        stale = compose_robot_state_sample(
            _state_frame(1.0), _wrench_frame(1.0), sample_time_s=1.2
        )
        self.assertFalse(stale.valid)
        self.assertIn("robot_state_stale", stale.invalid_reason)

        skewed = compose_robot_state_sample(
            _state_frame(10.0), _wrench_frame(10.03), sample_time_s=10.035,
            max_robot_state_age_s=1.0, max_force_sample_age_s=1.0,
            max_internal_skew_s=0.02,
        )
        self.assertFalse(skewed.valid)
        self.assertIn("robot_state_internal_skew", skewed.invalid_reason)

        future = compose_robot_state_sample(
            _state_frame(20.1), _wrench_frame(20.1), sample_time_s=20.0
        )
        self.assertFalse(future.valid)
        self.assertIn("robot_state_time_in_future", future.invalid_reason)

    def test_wrench_rotation_and_full_point_transform(self):
        rotation = rpy_euler_xyz_rotation_matrix((0.0, 0.0, math.pi / 2.0))
        self.assertTrue(np.allclose(rotate_vector(rotation, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)))
        force_b, torque_b = transform_wrench(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), rotation, (0.0, 0.0, 1.0)
        )
        self.assertTrue(np.allclose(force_b, (0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(torque_b, (-1.0, 0.0, 0.0)))

    def test_tangent_projection_and_retraced_path_continuity(self):
        geometry = TrajectoryGeometry(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        outbound, _ = geometry.project((0.5, 0.0, 0.0))
        inbound, _ = geometry.project((0.5, 0.0, 0.0), reference_arc_length_m=1.5)
        self.assertEqual(outbound.tangent_base, (1.0, 0.0, 0.0))
        self.assertEqual(inbound.tangent_base, (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(project_along_tangent((3.0, 0.0, 0.0), outbound.tangent_base), 3.0)
        self.assertAlmostEqual(project_along_tangent((-2.0, 0.0, 0.0), inbound.tangent_base), 2.0)
        zero = TrajectoryGeometry(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        projection, reason = zero.project((0.0, 0.0, 0.0))
        self.assertIsNone(projection)
        self.assertEqual(reason, "trajectory_zero_length")

    def test_software_bias_and_internal_wrench_raw_corrected_fields(self):
        self.assertEqual(
            calculate_software_bias([(1, 2, 3, 4, 5, 6), (3, 4, 5, 6, 7, 8)]),
            (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        )
        source = RokaeInternalWrenchSource(object())
        source._bias = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
        frame = source._frame_from_raw({
            "host_monotonic_time_s": 3.0,
            "cartesian_force_raw_n": [4.0, 5.0, 6.0],
            "cartesian_torque_raw_nm": [0.4, 0.5, 0.6],
            "joint_measured_torque_nm": [1, 2, 3, 4, 5, 6],
            "joint_external_torque_nm": [11, 12, 13, 14, 15, 16],
            "raw_force_frame": "world",
        })
        self.assertEqual(frame.cartesian_force_raw_n, (4.0, 5.0, 6.0))
        self.assertEqual(frame.cartesian_force_corrected_n, (3.0, 4.0, 5.0))
        self.assertEqual(frame.joint_measured_torque_nm[0], 1.0)
        self.assertEqual(frame.joint_external_torque_nm[0], 11.0)

    def test_circle_center_regression(self):
        robot = _CalibrationRobot([
            [4.0, 0.4, 2.0], [1.0, 0.4, 5.0], [-2.0, 0.4, 2.0],
        ])
        with patch("builtins.input", side_effect=["", "", ""]):
            center, radius = calibrate_joint_center(robot, use_drag=False)
        self.assertTrue(np.allclose(center, (1.0, 0.4, 2.0)))
        self.assertAlmostEqual(radius, 3.0)

    def test_mpc_prediction_and_full_trajectory_target_use_same_integrator(self):
        controller = MPCController(horizon=1, dt=0.1)
        states = controller._predict_states(np.array([0.0, 1.0]), np.array([2.0]))
        geometry = TrajectoryGeometry(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
        target, next_velocity, next_arc = controller.acceleration_to_trajectory_pose(
            geometry, 0.0, 1.0, 2.0
        )
        self.assertAlmostEqual(states[1, 0], next_arc)
        self.assertAlmostEqual(states[1, 1], next_velocity)
        self.assertGreater(target[0], 0.0)
        self.assertGreater(target[1], 0.0)

    def test_collector_csv_metadata_and_schema_readers(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with patch.object(settings, "DATA_DIR", data_dir):
                collector = DataCollector(_LiveRobot(), _LiveWrench(), "subject", "session")
                collector.start_episode(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
                for _ in range(12):
                    self.assertTrue(collector.record_sample())
                    time.sleep(0.001)
                output = collector.end_episode(comfort_label=0, pain_label=1)
                features, labels, _ = load_dataset(data_dir)
                t, arc, force = load_tangential_episode(output)

            with open(output, newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            for name in (
                "sample_time_s", "state_internal_skew_ms", "joint_measured_torque_1_nm",
                "joint_external_torque_1_nm", "fx_raw_n", "fx_corrected_n", "fx_base_n",
                "force_tangent_n", "force_query_duration_ms", "force_estimate_valid",
            ):
                self.assertIn(name, row)
            self.assertEqual(row["schema_version"], "3")
            self.assertEqual(row["frame"], "base")
            self.assertEqual(features.shape, (12, 9))
            self.assertTrue(np.all(labels == 1.0))
            self.assertEqual(t.shape, (12,))
            self.assertEqual(arc.shape, (12,))
            self.assertEqual(force.shape, (12,))
            with open(output.replace(".csv", ".json"), encoding="utf-8") as stream:
                metadata = json.load(stream)
            self.assertIn("started_at", metadata)
            self.assertIn("finished_at", metadata)
            self.assertEqual(metadata["comfort_label"], 0)
            self.assertEqual(metadata["pain_label"], 1)

    def test_invalid_episode_is_saved_and_sample_index_does_not_repeat(self):
        class OneInvalidWrench(_LiveWrench):
            def __init__(self):
                self.calls = 0

            def snapshot(self, now_s):
                self.calls += 1
                return _wrench_frame(now_s, valid=self.calls != 1,
                                     reason="simulated_wrench_gap" if self.calls == 1 else "")

        with tempfile.TemporaryDirectory() as data_dir:
            with patch.object(settings, "DATA_DIR", data_dir), patch.object(
                settings, "WRITE_INVALID_SAMPLES", False
            ):
                collector = DataCollector(_LiveRobot(), OneInvalidWrench(), "subject", "session")
                collector.start_episode()
                self.assertFalse(collector.record_sample())
                self.assertTrue(collector.record_sample())
                output = collector.abort_episode("simulated_abort")
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], "1")
            with open(output.replace(".csv", ".json"), encoding="utf-8") as stream:
                metadata = json.load(stream)
            self.assertFalse(metadata["completed"])
            self.assertEqual(metadata["stop_reason"], "simulated_abort")

    def test_background_sampler_runs_independently_at_configured_cadence(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with patch.object(settings, "DATA_DIR", data_dir):
                collector = DataCollector(_LiveRobot(), _LiveWrench(), "subject", "session")
                collector.start_episode()
                collector.start_background_sampling(sample_hz=50.0)
                time.sleep(0.14)
                output = collector.end_episode()
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 4)

    def test_stream_failure_and_collision_are_visible_to_safety(self):
        adapter = RokaeRobot("192.0.2.1")
        adapter.is_connected = True
        adapter.robot_mode = "IDLE"
        adapter._accept_state([0.3, 0.0, 0.4, 0.0, 0.0, 0.0], [0.0] * 6, 1.0)
        adapter._mark_state_error(RuntimeError("simulated stream failure"))
        state = adapter.get_state_frame()
        self.assertFalse(state.valid)
        self.assertIn("robot_state_stream_error", state.invalid_reason)

        robot = _LiveRobot()
        robot.collision = True
        guard = SafetyGuard(robot=robot, wrench_source=_LiveWrench())
        with patch.object(settings, "CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED", True):
            with self.assertRaisesRegex(RuntimeError, "robot_collision"):
                guard.check()
        self.assertTrue(robot.stopped)


if __name__ == "__main__":
    unittest.main()
