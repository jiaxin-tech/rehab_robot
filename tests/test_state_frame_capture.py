"""Offline state-capture contracts: accepted frames, not native wire timing.

The controlled source publishes independently of reads. Consumer pauses and
overflow are event-controlled, so a latest-only implementation fails reliably.
No SDK load, network connection, real robot, or wall-time throughput claim.
"""
from dataclasses import replace
import csv
import threading
import time

import pytest

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from collection.state import KinematicStateFrame
from collection.state_buffer import StateBufferOverflow, StateFrameBuffer
from hardware.rokae_adapter import RokaeRobotAdapter
from hardware.windows.rokae_xcore import RokaeRobot
from scripts.run_wrench_stationary_validation import ObservedAdapter


def state_frame(sequence, stamp=None):
    stamp = time.perf_counter() if stamp is None else stamp
    return KinematicStateFrame(
        sequence_id=sequence, host_monotonic_time_s=stamp, wall_time_iso=None,
        robot_device_time_s=None, valid=True, invalid_reason="",
        tcp_position_m=(float(sequence), 0., 0.), tcp_orientation_rad=(0., 0., 0.),
        tcp_linear_velocity_mps=None, tcp_angular_velocity_radps=None,
        velocity_source="unavailable", joint_position_rad=(float(sequence),) * 6,
        joint_velocity_radps=None, pose_time_s=stamp, joint_time_s=stamp,
        velocity_time_s=None, operation_state="IDLE", collision_state=None,
        controller_error=None,
    )


def wait_for(predicate, timeout=3.):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.001)
    raise AssertionError("offline state-capture condition did not arrive")


def read_state_rows(directory):
    with (directory / "robot_state.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ControlledSource:
    """Independent publication with a deliberate, bounded consumer pause."""
    def __init__(self, capacity=32, tail_sequence=None):
        self.capacity = capacity
        self.tail_sequence = tail_sequence
        self.connected = False
        self.latest = None
        self.buffer = None
        self.source_error = None
        self.generated = []
        self.calls = []
        self.pause_requested = threading.Event()
        self.drain_entered = threading.Event()
        self.resume_drain = threading.Event()
        self.lock = threading.Lock()

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def is_connected(self):
        return self.connected

    def start_state_stream(self):
        self.calls.append("start_state_stream")
        self.publish(1)

    def begin_state_capture(self):
        self.calls.append("begin_state_capture")
        with self.lock:
            self.buffer = StateFrameBuffer(capacity=self.capacity)
            if self.latest is not None:
                self.buffer.publish(self.latest)
        return True

    def publish(self, sequence):
        item = state_frame(sequence)
        with self.lock:
            self.latest = item
            self.generated.append(item)
            if self.buffer is not None:
                self.buffer.publish(item)
        return item

    def read_state_frame(self):
        with self.lock:
            item = self.latest
            error = self.source_error
        if item is None:
            raise RuntimeError("controlled_source_not_ready")
        return replace(item, valid=False, invalid_reason=error, controller_error=error) if error else item

    def drain_state_frames(self):
        if self.pause_requested.is_set():
            self.drain_entered.set()
            if not self.resume_drain.wait(3.):
                raise TimeoutError("test did not release state consumer")
        with self.lock:
            buffer = self.buffer
        return () if buffer is None else buffer.drain()

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        if self.tail_sequence is not None:
            sequence, self.tail_sequence = self.tail_sequence, None
            self.publish(sequence)

    def end_state_capture(self):
        self.calls.append("end_state_capture")
        with self.lock:
            self.buffer = None

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False


def acquisition_for(source, tmp_path):
    logger = EpisodeLogger(tmp_path / "episode").start()
    acquisition = RealRobotAcquisition(source, logger, diagnostic_state_only=True)
    acquisition.start()
    return acquisition, logger


def close_fixture(acquisition, logger, source):
    source.resume_drain.set()
    try:
        acquisition.stop()
    except StateBufferOverflow:
        # Overflow is expected in its failure test; it must never become a pass.
        pass
    finally:
        logger.close(completed=False, stop_reason="OFFLINE_STATE_CAPTURE_REGRESSION")


def test_buffer_drains_every_accepted_frame_in_order_once():
    buffer = StateFrameBuffer(capacity=4)
    frames = tuple(state_frame(sequence) for sequence in range(1, 5))
    for item in frames:
        buffer.publish(item)
    assert buffer.drain() == frames
    assert buffer.drain() == ()
    following = state_frame(5)
    buffer.publish(following)
    assert buffer.drain() == (following,)


def test_overflow_is_explicit_and_stays_latched_on_publish_and_drain():
    buffer = StateFrameBuffer(capacity=2)
    buffer.publish(state_frame(1))
    buffer.publish(state_frame(2))
    with pytest.raises(StateBufferOverflow):
        buffer.publish(state_frame(3))
    with pytest.raises(StateBufferOverflow):
        buffer.drain()
    with pytest.raises(StateBufferOverflow):
        buffer.publish(state_frame(4))
    # The failed buffer never silently becomes healthy because somebody drained it.
    with pytest.raises(StateBufferOverflow):
        buffer.drain()


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_buffer_rejects_nonpositive_or_nonintegral_capacity(capacity):
    with pytest.raises((TypeError, ValueError)):
        StateFrameBuffer(capacity=capacity)


def test_consumer_pause_preserves_all_published_frames_and_latest_reads_do_not_consume(tmp_path):
    source = ControlledSource(capacity=32)
    acquisition, logger = acquisition_for(source, tmp_path)
    try:
        wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.pause_requested.set()
        assert source.drain_entered.wait(2.)
        for sequence in range(2, 13):
            source.publish(sequence)
        assert source.read_state_frame().sequence_id == 12
        assert source.read_state_frame().sequence_id == 12
        source.resume_drain.set()
        wait_for(lambda: logger.row_counts["robot_state"] == 12)
        assert acquisition.latest_state_frame().sequence_id == 12
        assert acquisition.latest_health().state_valid
    finally:
        close_fixture(acquisition, logger, source)
    rows = read_state_rows(tmp_path / "episode")
    assert [int(float(row["q1"])) for row in rows] == list(range(1, 13))
    assert [float(row["host_time_s"]) for row in rows] == [item.host_monotonic_time_s for item in source.generated]


def test_capture_overflow_latches_background_failure_instead_of_using_old_latest_as_success(tmp_path):
    source = ControlledSource(capacity=2)
    acquisition, logger = acquisition_for(source, tmp_path)
    try:
        wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.pause_requested.set()
        assert source.drain_entered.wait(2.)
        source.publish(2)
        source.publish(3)
        with pytest.raises(StateBufferOverflow):
            source.publish(4)
        source.resume_drain.set()
        wait_for(lambda: acquisition.background_error is not None)
        assert "overflow" in acquisition.background_error.lower()
        assert logger.failed
        assert not acquisition.running
    finally:
        close_fixture(acquisition, logger, source)


def test_empty_batch_still_refreshes_latest_source_error_without_rewriting_same_frame(tmp_path):
    source = ControlledSource()
    acquisition, logger = acquisition_for(source, tmp_path)
    try:
        wait_for(lambda: logger.row_counts["robot_state"] == 1)
        assert source.buffer.drain() == ()
        source.source_error = "injected_source_error_without_new_frame"
        wait_for(lambda: "injected_source_error_without_new_frame" in acquisition.latest_health().invalid_reason
                 and not acquisition.latest_health().state_valid)
        assert "injected_source_error_without_new_frame" in acquisition.latest_health().invalid_reason
        assert logger.row_counts["robot_state"] == 1
    finally:
        close_fixture(acquisition, logger, source)
    assert len(read_state_rows(tmp_path / "episode")) == 1


def test_source_is_stopped_then_tail_is_drained_before_capture_end_and_disconnect(tmp_path):
    source = ControlledSource(tail_sequence=3)
    acquisition, logger = acquisition_for(source, tmp_path)
    try:
        wait_for(lambda: logger.row_counts["robot_state"] == 1)
        source.publish(2)
        wait_for(lambda: logger.row_counts["robot_state"] == 2)
    finally:
        close_fixture(acquisition, logger, source)
    assert [int(float(row["q1"])) for row in read_state_rows(tmp_path / "episode")] == [1, 2, 3]
    assert source.calls.index("stop_state_stream") < source.calls.index("end_state_capture") < source.calls.index("disconnect")
    assert not source.connected


def make_native():
    native = RokaeRobot("192.0.2.1")
    native.is_connected = True
    native.robot_mode = "IDLE"
    return native


def accept_native(native, sequence):
    native._accept_state([float(sequence), 0., .4, 0., 0., 0.], [float(sequence)] * 6,
                         1000. + sequence * .008, (False, False))


def test_native_latest_only_remains_nonconsuming_without_optional_capture():
    native = make_native()
    for sequence in range(1, 1101):
        accept_native(native, sequence)
    latest = native.get_state_frame()
    assert latest.valid
    assert latest.sequence_id == 1100
    assert native.get_state_frame() == latest


def test_native_capture_preserves_accepted_frames_and_latest_health_error():
    native = make_native()
    assert native.begin_state_capture() is True
    for sequence in (1, 2, 3):
        accept_native(native, sequence)
    assert native.get_state_frame().sequence_id == 3
    assert native.get_state_frame().sequence_id == 3
    batch = native.drain_state_frames()
    assert tuple(item.sequence_id for item in batch) == (1, 2, 3)
    assert all(item.valid for item in batch)
    assert native.drain_state_frames() == ()
    native._mark_state_error(RuntimeError("injected_native_state_error"))
    invalid_latest = native.get_state_frame()
    assert not invalid_latest.valid
    assert "injected_native_state_error" in invalid_latest.invalid_reason
    assert invalid_latest.sequence_id == 3
    assert native.drain_state_frames() == ()
    assert all(item.valid for item in batch), "historical accepted frames must remain immutable"
    native.end_state_capture()
    assert native.get_state_frame() == invalid_latest


class LegacyNative:
    def __init__(self):
        self.latest = state_frame(7)

    def get_state_frame(self):
        return self.latest


def test_adapter_keeps_legacy_latest_native_compatible_without_claiming_capture():
    native = LegacyNative()
    adapter = RokaeRobotAdapter(native_robot=native)
    assert adapter.begin_state_capture() is False
    assert adapter.read_state_frame() is native.latest
    adapter.end_state_capture()
    assert adapter.read_state_frame() is native.latest


def test_adapter_delegates_native_capture_and_preserves_nondestructive_latest():
    native = make_native()
    adapter = RokaeRobotAdapter(native_robot=native)
    assert adapter.begin_state_capture() is True
    accept_native(native, 1)
    accept_native(native, 2)
    assert adapter.read_state_frame().sequence_id == 2
    assert tuple(item.sequence_id for item in adapter.drain_state_frames()) == (1, 2)
    assert adapter.read_state_frame().sequence_id == 2
    adapter.end_state_capture()
    assert adapter.read_state_frame().sequence_id == 2


class ListEvidence:
    def __init__(self):
        self.rows = []

    def put(self, row):
        self.rows.append(row)


def test_observed_adapter_audits_each_drained_frame_without_latest_ahead_of_batch():
    source = ControlledSource()
    source.connect()
    source.start_state_stream()
    evidence = ListEvidence()
    observed = ObservedAdapter(source, evidence, lambda *args, **kwargs: None,
                               "offline-session", "offline-run", {})
    assert observed.begin_state_capture() is True
    source.publish(2)
    source.publish(3)
    assert observed.read_state_frame().sequence_id == 3
    assert observed.read_state_frame().sequence_id == 3
    assert evidence.rows == [], "a health/latest read must not jump audit ahead of buffered frames"
    assert tuple(item.sequence_id for item in observed.drain_state_frames()) == (1, 2, 3)
    assert [row["sequence_id"] for row in evidence.rows] == [1, 2, 3]
    assert [row["sequence_gap"] for row in evidence.rows] == [0, 0, 0]
    assert observed.drain_state_frames() == ()
    assert observed.read_state_frame().sequence_id == 3
    assert len(evidence.rows) == 3
    observed.end_state_capture()


def test_observed_adapter_legacy_latest_still_deduplicates_audit():
    legacy = RokaeRobotAdapter(native_robot=LegacyNative())
    evidence = ListEvidence()
    observed = ObservedAdapter(legacy, evidence, lambda *args, **kwargs: None,
                               "offline-session", "offline-run", {})
    assert observed.begin_state_capture() is False
    observed.read_state_frame()
    observed.read_state_frame()
    assert [row["sequence_id"] for row in evidence.rows] == [7]
