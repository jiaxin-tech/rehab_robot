"""Episode collector built around explicit robot state snapshots.

No external force-device sample is maintained here.  A row combines the latest
xCore realtime state packet with a separately time-stamped internal
``getEndTorque`` query, then records their age/skew and validity rather than
pretending that sequential reads are atomic.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Sequence

from collection.snapshot import read_live_robot_state_sample
from collection.state import RobotStateSample, merge_invalid_reasons
from collection.trajectory import TrajectoryGeometry, project_along_tangent
from config import settings
from utils.logger import get_logger


logger = get_logger("Collector")


class DataCollector:
    """Persist traceable, SI-unit robot state snapshots one episode at a time."""

    FIELDNAMES = [
        "schema_version", "frame", "units", "episode_id", "mode",
        "sample_index", "sequence_id", "sample_time_s", "host_monotonic_time_s",
        "wall_time_iso", "robot_device_time_s", "robot_state_time_s",
        "pose_time_s", "joint_time_s", "velocity_time_s", "torque_time_s", "force_time_s",
        "force_query_started_s", "force_query_finished_s", "force_query_duration_ms",
        "robot_state_age_ms", "force_sample_age_ms", "state_internal_skew_ms",
        "valid", "invalid_reason",
        "trajectory_s", "trajectory_arc_length_m", "tangent_x", "tangent_y", "tangent_z",
        "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad",
        "vx_mps", "vy_mps", "vz_mps", "wx_radps", "wy_radps", "wz_radps",
        "ax_est_mps2", "ay_est_mps2", "az_est_mps2", "velocity_source", "acceleration_source",
        "q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad",
        "dq1_radps", "dq2_radps", "dq3_radps", "dq4_radps", "dq5_radps", "dq6_radps",
        "joint_measured_torque_1_nm", "joint_measured_torque_2_nm",
        "joint_measured_torque_3_nm", "joint_measured_torque_4_nm",
        "joint_measured_torque_5_nm", "joint_measured_torque_6_nm",
        "joint_external_torque_1_nm", "joint_external_torque_2_nm",
        "joint_external_torque_3_nm", "joint_external_torque_4_nm",
        "joint_external_torque_5_nm", "joint_external_torque_6_nm",
        "fx_raw_n", "fy_raw_n", "fz_raw_n", "tx_raw_nm", "ty_raw_nm", "tz_raw_nm",
        "raw_force_frame",
        "fx_bias_n", "fy_bias_n", "fz_bias_n", "tx_bias_nm", "ty_bias_nm", "tz_bias_nm",
        "fx_corrected_n", "fy_corrected_n", "fz_corrected_n",
        "tx_corrected_nm", "ty_corrected_nm", "tz_corrected_nm",
        "fx_base_n", "fy_base_n", "fz_base_n", "tx_base_nm", "ty_base_nm", "tz_base_nm",
        "base_wrench_transform_kind",
        "force_tangent_n", "velocity_tangent_mps", "acceleration_tangent_mps2",
        "robot_operation_state", "robot_collision", "controller_error",
        "force_estimate_valid", "force_source",
    ]

    def __init__(
        self,
        robot: Any,
        force_sensor: Any,
        subject_id: str,
        session_id: str,
        mode: str = "passive",
        *,
        wrench_source: Any | None = None,
    ) -> None:
        self.robot = robot
        # ``force_sensor`` remains the positional compatibility slot.  The new
        # contract requires a robot-internal source exposing snapshot().
        self.wrench_source = wrench_source if wrench_source is not None else force_sensor
        self.mode = mode
        self.subject_id = subject_id
        self.session_id = session_id
        self._buf: list[dict[str, Any]] = []
        self._previous_sample: RobotStateSample | None = None
        self._trajectory: TrajectoryGeometry | None = None
        self._previous_trajectory_arc_length_m: float | None = None
        self._episode_id: str | None = None
        self._episode_started_at: str | None = None
        self._active = False
        self._sample_attempt_index = 0
        self._sample_errors = 0
        self._invalid_samples = 0
        self._sample_attempt_index = 0
        self._dropped_state_frames = 0
        self._last_state_sequence_id: int | None = None

        self._sampling_stop = threading.Event()
        self._sampling_thread: threading.Thread | None = None
        self._sampling_exception: BaseException | None = None

        self.out_dir = os.path.join(settings.DATA_DIR, subject_id, session_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self._ep_count = self._count_existing()
        self._write_session_metadata()
        logger.info("采集会话路径: %s，已有 %d 个 episode", self.out_dir, self._ep_count)

    def _count_existing(self) -> int:
        max_idx = 0
        for fpath in glob.glob(os.path.join(self.out_dir, "episode_*.csv")):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            try:
                max_idx = max(max_idx, int(stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max_idx

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (list, tuple)):
            return [DataCollector._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): DataCollector._json_safe(item) for key, item in value.items()}
        return str(value)

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        value = result.stdout.strip()
        return value or None

    def _settings_snapshot(self) -> dict[str, Any]:
        values = {
            name: getattr(settings, name)
            for name in dir(settings)
            if name.isupper()
        }
        return self._json_safe(values)

    def _robot_metadata(self) -> dict[str, Any]:
        if not hasattr(self.robot, "get_robot_metadata"):
            return {}
        try:
            return self._json_safe(self.robot.get_robot_metadata())
        except Exception as exc:
            return {"metadata_read_error": f"{type(exc).__name__}:{exc}"}

    @staticmethod
    def _atomic_json(path: str, payload: dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp_path, path)

    def _write_session_metadata(self) -> None:
        payload = {
            "schema_version": settings.DATA_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "robot": self._robot_metadata(),
            "sample_hz": settings.COLLECT_HZ,
            "trajectory_source": "runtime_supplied_or_none",
            "tool_name_expected": settings.TOOL_NAME,
            "workpiece_name_expected": settings.WORKPIECE_NAME,
            "payload_mass_kg_expected": settings.PAYLOAD_MASS_KG,
            "payload_com_m_expected": settings.PAYLOAD_COM_M,
            "force_data_source": settings.ROBOT_FORCE_SOURCE,
            "raw_force_frame": settings.ROBOT_FORCE_RAW_FRAME,
            "control_frame": settings.CONTROL_FRAME,
            "base_wrench_transform_kind": settings.BASE_WRENCH_TRANSFORM_KIND,
            "base_wrench_rotation_verified": settings.BASE_WRENCH_ROTATION_VERIFIED,
            "safety_preconditions": {
                "workspace_min_m": settings.WORKSPACE_MIN_M,
                "workspace_max_m": settings.WORKSPACE_MAX_M,
                "require_workspace_limits": settings.REQUIRE_WORKSPACE_LIMITS,
                "joint_soft_limit_margin_rad": settings.JOINT_SOFT_LIMIT_MARGIN_RAD,
                "require_joint_soft_limits": settings.REQUIRE_JOINT_SOFT_LIMITS,
                "controller_collision_configuration_confirmed": (
                    settings.CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED
                ),
                "require_collision_state_query": settings.REQUIRE_COLLISION_STATE_QUERY,
            },
            "units": settings.SI_UNITS,
            "git_commit": self._git_commit(),
            "config_snapshot": self._settings_snapshot(),
            "limitations": [
                "xCoreSDK v0.7.0 RT stream has no device timestamp, velocity, torque, wrench, or collision field.",
                "getEndTorque is a separate controller query; CSV state_internal_skew_ms must be checked.",
                "Cartesian compensation and wrench reference-point semantics require real-robot/vendor validation.",
            ],
        }
        self._atomic_json(os.path.join(self.out_dir, "metadata.json"), payload)

    def start_episode(
        self,
        trajectory: Sequence[Sequence[float]] | None = None,
        *,
        episode_id: str | None = None,
    ) -> str:
        if self._active:
            raise RuntimeError("An episode is already active")
        self._buf = []
        self._previous_sample = None
        self._sample_errors = 0
        self._invalid_samples = 0
        self._dropped_state_frames = 0
        self._last_state_sequence_id = None
        self._sampling_exception = None
        self._trajectory = (
            TrajectoryGeometry(trajectory) if trajectory is not None else None
        )
        self._previous_trajectory_arc_length_m = None
        self._episode_id = episode_id or f"episode_{self._ep_count + 1:04d}"
        self._episode_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._active = True
        logger.info("%s 开始", self._episode_id)
        return self._episode_id

    @staticmethod
    def _component(values: Sequence[float] | None, index: int) -> float | None:
        if values is None or len(values) <= index:
            return None
        value = float(values[index])
        return value if math.isfinite(value) else None

    def _empty_invalid_row(self, sample_index: int, reason: str) -> dict[str, Any]:
        row = {name: None for name in self.FIELDNAMES}
        now_s = time.monotonic()
        row.update(
            {
                "schema_version": settings.DATA_SCHEMA_VERSION,
                "frame": settings.CONTROL_FRAME,
                "units": settings.SI_UNITS,
                "episode_id": self._episode_id,
                "mode": self.mode,
                "sample_index": sample_index,
                "sample_time_s": now_s,
                "host_monotonic_time_s": now_s,
                "wall_time_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "valid": 0,
                "invalid_reason": reason,
                "force_source": settings.ROBOT_FORCE_SOURCE,
                "force_estimate_valid": 0,
            }
        )
        return row

    def _apply_trajectory(self, sample: RobotStateSample) -> RobotStateSample:
        if self._trajectory is None:
            return sample
        projection, reason = self._trajectory.project(
            sample.tcp_position_m,
            reference_arc_length_m=self._previous_trajectory_arc_length_m,
        )
        if projection is None:
            return replace(
                sample,
                valid=False,
                invalid_reason=merge_invalid_reasons(sample.invalid_reason, reason),
            )
        force_tangent = project_along_tangent(
            sample.cartesian_force_base_n, projection.tangent_base
        )
        velocity_tangent = project_along_tangent(
            sample.tcp_linear_velocity_mps, projection.tangent_base
        )
        acceleration_tangent = project_along_tangent(
            sample.tcp_linear_acceleration_est_mps2, projection.tangent_base
        )
        invalid_reason = sample.invalid_reason
        valid = sample.valid
        if force_tangent is None:
            valid = False
            invalid_reason = merge_invalid_reasons(
                invalid_reason, "force_tangent_unavailable"
            )
        self._previous_trajectory_arc_length_m = projection.arc_length_m
        return replace(
            sample,
            valid=valid,
            invalid_reason=invalid_reason,
            trajectory_s=projection.trajectory_s,
            trajectory_arc_length_m=projection.arc_length_m,
            trajectory_tangent=projection.tangent_base,
            force_tangent_n=force_tangent,
            velocity_tangent_mps=velocity_tangent,
            acceleration_tangent_mps2=acceleration_tangent,
        )

    def _row_from_sample(self, sample: RobotStateSample, sample_index: int) -> dict[str, Any]:
        row = {name: None for name in self.FIELDNAMES}
        row.update(
            {
                "schema_version": settings.DATA_SCHEMA_VERSION,
                "frame": settings.CONTROL_FRAME,
                "units": settings.SI_UNITS,
                "episode_id": self._episode_id,
                "mode": self.mode,
                "sample_index": sample_index,
                "sequence_id": sample.sequence_id,
                "sample_time_s": sample.sample_time_s,
                "host_monotonic_time_s": sample.host_monotonic_time_s,
                "wall_time_iso": sample.wall_time_iso,
                "robot_device_time_s": sample.robot_device_time_s,
                "robot_state_time_s": sample.robot_state_time_s,
                "pose_time_s": sample.pose_time_s,
                "joint_time_s": sample.joint_time_s,
                "velocity_time_s": sample.velocity_time_s,
                "torque_time_s": sample.torque_time_s,
                "force_time_s": sample.force_time_s,
                "force_query_started_s": sample.force_query_started_s,
                "force_query_finished_s": sample.force_query_finished_s,
                "force_query_duration_ms": sample.force_query_duration_ms,
                "robot_state_age_ms": sample.robot_state_age_ms,
                "force_sample_age_ms": sample.force_sample_age_ms,
                "state_internal_skew_ms": sample.state_internal_skew_ms,
                "valid": int(sample.valid),
                "invalid_reason": sample.invalid_reason,
                "trajectory_s": sample.trajectory_s,
                "trajectory_arc_length_m": sample.trajectory_arc_length_m,
                "tangent_x": self._component(sample.trajectory_tangent, 0),
                "tangent_y": self._component(sample.trajectory_tangent, 1),
                "tangent_z": self._component(sample.trajectory_tangent, 2),
                "x_m": self._component(sample.tcp_position_m, 0),
                "y_m": self._component(sample.tcp_position_m, 1),
                "z_m": self._component(sample.tcp_position_m, 2),
                "rx_rad": self._component(sample.tcp_orientation_rad, 0),
                "ry_rad": self._component(sample.tcp_orientation_rad, 1),
                "rz_rad": self._component(sample.tcp_orientation_rad, 2),
                "vx_mps": self._component(sample.tcp_linear_velocity_mps, 0),
                "vy_mps": self._component(sample.tcp_linear_velocity_mps, 1),
                "vz_mps": self._component(sample.tcp_linear_velocity_mps, 2),
                "wx_radps": self._component(sample.tcp_angular_velocity_radps, 0),
                "wy_radps": self._component(sample.tcp_angular_velocity_radps, 1),
                "wz_radps": self._component(sample.tcp_angular_velocity_radps, 2),
                "ax_est_mps2": self._component(sample.tcp_linear_acceleration_est_mps2, 0),
                "ay_est_mps2": self._component(sample.tcp_linear_acceleration_est_mps2, 1),
                "az_est_mps2": self._component(sample.tcp_linear_acceleration_est_mps2, 2),
                "velocity_source": sample.velocity_source,
                "acceleration_source": sample.acceleration_source,
                "raw_force_frame": sample.raw_force_frame,
                "base_wrench_transform_kind": sample.base_transform_kind,
                "force_tangent_n": sample.force_tangent_n,
                "velocity_tangent_mps": sample.velocity_tangent_mps,
                "acceleration_tangent_mps2": sample.acceleration_tangent_mps2,
                "robot_operation_state": sample.operation_state,
                "robot_collision": (
                    None if sample.collision_state is None else int(sample.collision_state)
                ),
                "controller_error": sample.controller_error,
                "force_estimate_valid": int(sample.force_estimate_valid),
                "force_source": settings.ROBOT_FORCE_SOURCE,
            }
        )
        for prefix, values, suffix in (
            ("q", sample.joint_position_rad, "_rad"),
            ("dq", sample.joint_velocity_radps, "_radps"),
            ("joint_measured_torque_", sample.joint_measured_torque_nm, "_nm"),
            ("joint_external_torque_", sample.joint_external_torque_nm, "_nm"),
        ):
            for index in range(6):
                name = f"{prefix}{index + 1}{suffix}"
                row[name] = self._component(values, index)
        for names, values in (
            (("fx_raw_n", "fy_raw_n", "fz_raw_n"), sample.cartesian_force_raw_n),
            (("tx_raw_nm", "ty_raw_nm", "tz_raw_nm"), sample.cartesian_torque_raw_nm),
            (("fx_bias_n", "fy_bias_n", "fz_bias_n"), sample.cartesian_force_bias_n),
            (("tx_bias_nm", "ty_bias_nm", "tz_bias_nm"), sample.cartesian_torque_bias_nm),
            (("fx_corrected_n", "fy_corrected_n", "fz_corrected_n"), sample.cartesian_force_corrected_n),
            (("tx_corrected_nm", "ty_corrected_nm", "tz_corrected_nm"), sample.cartesian_torque_corrected_nm),
            (("fx_base_n", "fy_base_n", "fz_base_n"), sample.cartesian_force_base_n),
            (("tx_base_nm", "ty_base_nm", "tz_base_nm"), sample.cartesian_torque_base_nm),
        ):
            for index, name in enumerate(names):
                row[name] = self._component(values, index)
        return row

    def record_sample(self) -> bool:
        """Capture one cached state/wrench snapshot; invalid rows remain traceable."""
        if not self._active:
            raise RuntimeError("Call start_episode() before record_sample()")
        # Keep sample indices monotonic even when WRITE_INVALID_SAMPLES=False;
        # a dropped invalid row must not cause the next valid sample to reuse it.
        sample_index = self._sample_attempt_index
        self._sample_attempt_index += 1
        try:
            sample = read_live_robot_state_sample(
                self.robot,
                self.wrench_source,
                previous_sample=self._previous_sample,
            )
            sample = self._apply_trajectory(sample)
            if self._last_state_sequence_id is not None:
                self._dropped_state_frames += max(
                    0, sample.sequence_id - self._last_state_sequence_id - 1
                )
            self._last_state_sequence_id = sample.sequence_id
            self._previous_sample = sample
            row = self._row_from_sample(sample, sample_index)
            success = sample.valid
        except Exception as exc:
            self._sample_errors += 1
            row = self._empty_invalid_row(
                sample_index, f"collector_snapshot_error:{type(exc).__name__}:{exc}"
            )
            success = False

        if not success:
            self._invalid_samples += 1
        if success or settings.WRITE_INVALID_SAMPLES:
            self._buf.append(row)
        return success

    def start_background_sampling(self, sample_hz: float = settings.COLLECT_HZ) -> None:
        """Run collection independently from any trajectory-command scheduler."""
        if not self._active:
            raise RuntimeError("Start an episode before background sampling")
        if self._sampling_thread is not None and self._sampling_thread.is_alive():
            raise RuntimeError("Background sampling is already running")
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self._sampling_stop.clear()
        self._sampling_exception = None

        def loop() -> None:
            period_s = 1.0 / sample_hz
            next_tick = time.monotonic()
            while not self._sampling_stop.is_set():
                try:
                    self.record_sample()
                except BaseException as exc:  # thread boundary: preserve diagnostic
                    self._sampling_exception = exc
                    self._sampling_stop.set()
                    break
                next_tick += period_s
                wait_s = next_tick - time.monotonic()
                if wait_s > 0:
                    self._sampling_stop.wait(wait_s)
                elif -wait_s > period_s:
                    next_tick = time.monotonic()

        self._sampling_thread = threading.Thread(
            target=loop, name="robot-state-collection", daemon=True
        )
        self._sampling_thread.start()

    def stop_background_sampling(self) -> None:
        self._sampling_stop.set()
        thread = self._sampling_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._sampling_thread = None
        if self._sampling_exception is not None:
            exc = self._sampling_exception
            self._sampling_exception = None
            raise RuntimeError(f"Collection thread failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _numeric_max(rows: list[dict[str, Any]], field: str) -> float | None:
        values: list[float] = []
        for row in rows:
            value = row.get(field)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        return max(values) if values else None

    def end_episode(
        self,
        *,
        completed: bool = True,
        stop_reason: str | None = None,
        operator_note: str | None = None,
    ) -> str:
        """Atomically save CSV and episode JSON even after an abnormal finish."""
        if not self._active:
            raise RuntimeError("No active episode to finish")
        sampling_failure: RuntimeError | None = None
        try:
            self.stop_background_sampling()
        except RuntimeError as exc:
            # Persist the partial episode before surfacing the collection-thread
            # failure to the caller/safety path.
            sampling_failure = exc
            completed = False
            stop_reason = stop_reason or f"collection_thread_error:{exc}"
        self._ep_count += 1
        episode_id = self._episode_id or f"episode_{self._ep_count:04d}"
        csv_path = os.path.join(self.out_dir, f"episode_{self._ep_count:04d}.csv")
        tmp_path = f"{csv_path}.tmp"
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.FIELDNAMES, extrasaction="raise")
                writer.writeheader()
                writer.writerows(self._buf)
            os.replace(tmp_path, csv_path)
        except Exception:
            self._active = False
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metadata = {
            "schema_version": settings.DATA_SCHEMA_VERSION,
            "episode_id": episode_id,
            "csv_file": os.path.basename(csv_path),
            "started_at": self._episode_started_at,
            "finished_at": finished_at,
            "completed": bool(completed),
            "stop_reason": stop_reason,
            "operator_note": operator_note,
            "sample_count": len(self._buf),
            "valid_sample_count": sum(int(row.get("valid") == 1) for row in self._buf),
            "invalid_sample_count": sum(int(row.get("valid") != 1) for row in self._buf),
            "dropped_state_frame_count": self._dropped_state_frames,
            "collector_snapshot_error_count": self._sample_errors,
            "sampling_thread_error": str(sampling_failure) if sampling_failure else None,
            "max_robot_state_age_ms": self._numeric_max(self._buf, "robot_state_age_ms"),
            "max_force_sample_age_ms": self._numeric_max(self._buf, "force_sample_age_ms"),
            "max_internal_state_skew_ms": self._numeric_max(self._buf, "state_internal_skew_ms"),
            "force_reference_bias": (
                list(getattr(self.wrench_source, "bias", None) or ()) or None
            ),
            "force_data_source": settings.ROBOT_FORCE_SOURCE,
            "raw_force_frame": settings.ROBOT_FORCE_RAW_FRAME,
            "control_frame": settings.CONTROL_FRAME,
            "base_wrench_rotation_verified": settings.BASE_WRENCH_ROTATION_VERIFIED,
        }
        self._atomic_json(
            os.path.join(self.out_dir, f"episode_{self._ep_count:04d}.json"),
            self._json_safe(metadata),
        )
        self._active = False
        self._episode_id = None
        self._episode_started_at = None
        logger.info(
            "%s 已保存: %s (%d rows, valid=%d, invalid=%d)",
            episode_id,
            csv_path,
            len(self._buf),
            metadata["valid_sample_count"],
            metadata["invalid_sample_count"],
        )
        if sampling_failure is not None:
            raise sampling_failure
        return csv_path

    def abort_episode(self, reason: str, operator_note: str | None = None) -> str | None:
        """Persist the partial episode/metadata during any controlled exception path."""
        if not self._active:
            return None
        try:
            return self.end_episode(
                completed=False,
                stop_reason=reason,
                operator_note=operator_note,
            )
        except RuntimeError as exc:
            # end_episode has already atomically persisted the abnormal metadata
            # before raising a background-thread failure. Do not mask the caller's
            # original robot exception or skip its emergency stop path.
            logger.error("异常 episode 已保存，但采集线程也失败: %s", exc)
            return None
