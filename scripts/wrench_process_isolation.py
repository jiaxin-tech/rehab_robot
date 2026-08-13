"""Diagnostic-only process isolation for blocking ROKAE wrench queries.

Every child process creates and owns its own resources.  Parent/child IPC
contains plain dictionaries only and uses bounded latest-snapshot queues; no
SDK or robot object is inherited, pickled, or shared across processes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import multiprocessing as mp
from multiprocessing.context import BaseContext
import os
import queue
import re
import time
from typing import Any, Callable, Mapping


IPC_SCHEMA_VERSION = 1


def _error_code(error: BaseException) -> int | None:
    match = re.search(r"\((\d+)\)", str(error))
    return int(match.group(1)) if match else None


def _timing_summary(samples_ms: list[float]) -> dict[str, float | int | None]:
    values = sorted(float(value) for value in samples_ms)
    if not values:
        return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}

    def percentile(percent: float) -> float:
        position = (len(values) - 1) * percent
        lower = int(position)
        upper = min(len(values) - 1, lower + 1)
        fraction = position - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[-1],
    }


def publish_latest(channel: Any, payload: Mapping[str, Any]) -> bool:
    """Publish without waiting; replace one queued old snapshot if necessary."""

    item = dict(payload)
    try:
        channel.put_nowait(item)
        return True
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(item)
            return True
        except queue.Full:
            # A concurrent consumer/feeder race is harmless for latest-state
            # telemetry.  The producer must never wait for IPC capacity.
            return False


def publish_latest_counted(
    channel: Any,
    payload: Mapping[str, Any],
    counters: dict[str, int],
) -> str:
    """Publish latest telemetry and expose IPC overwrite/drop accounting.

    ``overwrite_count`` only means a bounded IPC snapshot was replaced.  It is
    not an RT network packet-loss counter.
    """

    counters["publish_count"] = int(counters.get("publish_count", 0)) + 1
    initial = {
        **dict(payload),
        "publish_count": counters["publish_count"],
        "publish_success_count": int(counters.get("publish_success_count", 0)) + 1,
        "overwrite_count": int(counters.get("overwrite_count", 0)),
        "publish_drop_count": int(counters.get("publish_drop_count", 0)),
    }
    try:
        channel.put_nowait(initial)
        counters["publish_success_count"] = initial["publish_success_count"]
        return "published"
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        replacement = {
            **dict(payload),
            "publish_count": counters["publish_count"],
            "publish_success_count": int(counters.get("publish_success_count", 0)) + 1,
            "overwrite_count": int(counters.get("overwrite_count", 0)) + 1,
            "publish_drop_count": int(counters.get("publish_drop_count", 0)),
        }
        try:
            channel.put_nowait(replacement)
            counters["publish_success_count"] = replacement["publish_success_count"]
            counters["overwrite_count"] = replacement["overwrite_count"]
            return "overwritten"
        except queue.Full:
            counters["publish_drop_count"] = int(counters.get("publish_drop_count", 0)) + 1
            return "dropped"


def drain_latest(channel: Any) -> dict[str, Any] | None:
    """Drain currently available messages without blocking and return newest."""

    latest: dict[str, Any] | None = None
    while True:
        try:
            latest = dict(channel.get_nowait())
        except queue.Empty:
            return latest


def drain_latest_counted(channel: Any) -> tuple[dict[str, Any] | None, int]:
    """Return the newest immediately available item and number dequeued."""

    latest: dict[str, Any] | None = None
    count = 0
    while True:
        try:
            latest = dict(channel.get_nowait())
            count += 1
        except queue.Empty:
            return latest, count


@dataclass(frozen=True)
class ProcessObservation:
    host_timestamp_ns: int
    worker_pid: int | None
    worker_start_time_ns: int | None
    worker_alive: bool
    worker_exitcode: int | None
    worker_state: str
    last_heartbeat_ns: int | None
    heartbeat_age_ms: float | None
    worker_hung: bool
    wrench_sequence: int
    last_wrench_success_ns: int | None
    wrench_age_ms: float | None
    wrench_valid: bool
    wrench_stale: bool
    last_error_code: int | None
    last_error: str | None
    graceful_disconnect_confirmed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WrenchProcessSupervisor:
    """Non-blocking parent view of one independently owned worker process."""

    def __init__(
        self,
        *,
        stale_age_ms: float = 150.0,
        worker_hung_ms: float = 750.0,
        worker_startup_hung_ms: float = 15_000.0,
        context: BaseContext | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if stale_age_ms <= 0 or worker_hung_ms <= 0 or worker_startup_hung_ms <= 0:
            raise ValueError("stale, hung, and startup thresholds must be positive")
        self.stale_age_ms = float(stale_age_ms)
        self.worker_hung_ms = float(worker_hung_ms)
        self.worker_startup_hung_ms = float(worker_startup_hung_ms)
        self.context = context or mp.get_context("spawn")
        self.clock_ns = clock_ns
        self._snapshot_queue: Any | None = None
        self._heartbeat_queue: Any | None = None
        self._stop_event: Any | None = None
        self._process: Any | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._last_success_snapshot: dict[str, Any] | None = None
        self._last_heartbeat: dict[str, Any] | None = None
        self._last_error_code: int | None = None
        self._last_error: str | None = None
        self._forced_termination = False

    @property
    def process(self) -> Any | None:
        return self._process

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    @property
    def metadata(self) -> dict[str, Any]:
        if self._last_heartbeat is None:
            return {}
        return dict(self._last_heartbeat.get("metadata") or {})

    def start(self, *, mode: str, config: Mapping[str, Any]) -> int:
        if self._process is not None:
            raise RuntimeError("worker process has already been created")
        self._snapshot_queue = self.context.Queue(maxsize=1)
        # A few bounded slots preserve rapid starting -> connecting -> ready
        # state transitions without creating an unbounded telemetry backlog.
        self._heartbeat_queue = self.context.Queue(maxsize=4)
        self._stop_event = self.context.Event()
        # Only plain configuration and IPC primitives cross the spawn boundary.
        self._process = self.context.Process(
            target=_worker_entry,
            args=(
                str(mode),
                dict(config),
                self._snapshot_queue,
                self._heartbeat_queue,
                self._stop_event,
            ),
            name=f"diagnostic-wrench-process-{mode}",
            daemon=False,
        )
        self._process.start()
        return int(self._process.pid)

    def poll(self, now_ns: int | None = None) -> ProcessObservation:
        current_ns = self.clock_ns() if now_ns is None else int(now_ns)
        if self._snapshot_queue is not None:
            snapshot = drain_latest(self._snapshot_queue)
            if snapshot is not None:
                self._last_snapshot = snapshot
                if bool(snapshot.get("success")):
                    self._last_success_snapshot = snapshot
                else:
                    self._last_error_code = snapshot.get("error_code")
                    self._last_error = str(snapshot.get("error_message") or "") or None
        if self._heartbeat_queue is not None:
            heartbeat = drain_latest(self._heartbeat_queue)
            if heartbeat is not None:
                self._last_heartbeat = heartbeat
                if heartbeat.get("last_error_code") is not None:
                    self._last_error_code = int(heartbeat["last_error_code"])
                if heartbeat.get("last_error"):
                    self._last_error = str(heartbeat["last_error"])

        process = self._process
        alive = bool(process is not None and process.is_alive())
        exitcode = None if process is None else process.exitcode
        heartbeat_ns = (
            None
            if self._last_heartbeat is None
            else self._last_heartbeat.get("last_heartbeat_ns")
        )
        heartbeat_age_ms = (
            None
            if heartbeat_ns is None
            else max(0.0, (current_ns - int(heartbeat_ns)) / 1e6)
        )
        success_ns = (
            None
            if self._last_success_snapshot is None
            else self._last_success_snapshot.get("host_timestamp_ns")
        )
        wrench_age_ms = (
            None
            if success_ns is None
            else max(0.0, (current_ns - int(success_ns)) / 1e6)
        )
        stale = wrench_age_ms is None or wrench_age_ms > self.stale_age_ms
        worker_state = (
            "not_started"
            if self._last_heartbeat is None
            else str(self._last_heartbeat.get("state", "unknown"))
        )
        hung_threshold_ms = (
            self.worker_startup_hung_ms
            if worker_state in {"not_started", "starting", "connecting"}
            else self.worker_hung_ms
        )
        hung = bool(
            alive
            and heartbeat_age_ms is not None
            and heartbeat_age_ms > hung_threshold_ms
        )
        graceful = None
        if self._last_heartbeat is not None:
            graceful = self._last_heartbeat.get("graceful_disconnect_confirmed")
        if self._forced_termination:
            graceful = False
        return ProcessObservation(
            host_timestamp_ns=current_ns,
            worker_pid=(
                self.pid
                if self._last_heartbeat is None
                else self._last_heartbeat.get("worker_pid", self.pid)
            ),
            worker_start_time_ns=(
                None
                if self._last_heartbeat is None
                else self._last_heartbeat.get("worker_start_time_ns")
            ),
            worker_alive=alive,
            worker_exitcode=exitcode,
            worker_state=worker_state,
            last_heartbeat_ns=None if heartbeat_ns is None else int(heartbeat_ns),
            heartbeat_age_ms=heartbeat_age_ms,
            worker_hung=hung,
            wrench_sequence=(
                0 if self._last_snapshot is None else int(self._last_snapshot.get("sequence_id", 0))
            ),
            last_wrench_success_ns=None if success_ns is None else int(success_ns),
            wrench_age_ms=wrench_age_ms,
            wrench_valid=bool(self._last_success_snapshot is not None and not stale),
            wrench_stale=stale,
            last_error_code=self._last_error_code,
            last_error=self._last_error,
            graceful_disconnect_confirmed=(
                None if graceful is None else bool(graceful)
            ),
        )

    def request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def join(self, timeout_s: float) -> bool:
        if self._process is None:
            return True
        self._process.join(timeout=max(0.0, float(timeout_s)))
        self.poll()
        return not self._process.is_alive()

    def stop_normally(self, timeout_s: float = 3.0) -> dict[str, Any]:
        started_ns = self.clock_ns()
        self.request_stop()
        exited = self.join(timeout_s)
        observation = self.poll()
        return {
            "worker_exited": exited,
            "worker_exitcode": observation.worker_exitcode,
            "latency_ms": (self.clock_ns() - started_ns) / 1e6,
            "graceful_disconnect_confirmed": observation.graceful_disconnect_confirmed,
            "forced": False,
        }

    def terminate(self, join_timeout_s: float = 2.0) -> dict[str, Any]:
        started_ns = self.clock_ns()
        process = self._process
        used_kill = False
        if process is not None and process.is_alive():
            self._forced_termination = True
            process.terminate()
            process.join(timeout=max(0.0, float(join_timeout_s)))
            if process.is_alive():
                used_kill = True
                process.kill()
                process.join(timeout=max(0.0, float(join_timeout_s)))
        observation = self.poll()
        return {
            "worker_terminated": bool(process is None or not process.is_alive()),
            "worker_exitcode": observation.worker_exitcode,
            "termination_latency_ms": (self.clock_ns() - started_ns) / 1e6,
            "used_kill": used_kill,
            "graceful_disconnect_confirmed": False,
            "forced": True,
        }

    def close(self) -> None:
        for channel in (self._snapshot_queue, self._heartbeat_queue):
            if channel is not None:
                try:
                    channel.close()
                    channel.cancel_join_thread()
                except (OSError, ValueError):
                    pass


def _heartbeat(
    channel: Any,
    *,
    start_ns: int,
    state: str,
    connected: bool,
    last_success_ns: int | None,
    last_error_code: int | None,
    last_error: str | None,
    graceful_disconnect_confirmed: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    now_ns = time.perf_counter_ns()
    publish_latest(channel, {
        "schema_version": IPC_SCHEMA_VERSION,
        "worker_pid": os.getpid(),
        "worker_start_time_ns": start_ns,
        "last_heartbeat_ns": now_ns,
        "state": state,
        "connected": connected,
        "last_wrench_success_ns": last_success_ns,
        "last_error_code": last_error_code,
        "last_error": last_error,
        "graceful_disconnect_confirmed": graceful_disconnect_confirmed,
        "metadata": dict(metadata or {}),
    })


def _publish_result(
    channel: Any,
    *,
    sequence_id: int,
    started_ns: int,
    finished_ns: int,
    success: bool,
    value: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    value = dict(value or {})
    publish_latest(channel, {
        "schema_version": IPC_SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "host_timestamp_ns": finished_ns,
        "call_start_ns": started_ns,
        "call_end_ns": finished_ns,
        "query_latency_ms": (finished_ns - started_ns) / 1e6,
        "success": bool(success),
        "error_code": None if error is None else _error_code(error),
        "error_message": "" if error is None else f"{type(error).__name__}:{error}",
        "joint_measured_torque_nm": value.get("joint_measured_torque_nm"),
        "joint_external_torque_nm": value.get("joint_external_torque_nm"),
        "cartesian_force_raw_n": value.get("cartesian_force_raw_n"),
        "cartesian_torque_raw_nm": value.get("cartesian_torque_raw_nm"),
    })


def _offline_query(config: Mapping[str, Any], sequence_id: int) -> dict[str, Any]:
    behavior = str(config.get("behavior", "normal"))
    delay_s = float(config.get("delay_s", 0.001))
    if behavior == "permanent":
        time.sleep(float(config.get("permanent_sleep_s", 3600.0)))
    else:
        time.sleep(delay_s)
    if behavior == "crash" and sequence_id >= int(config.get("crash_after", 1)):
        os._exit(int(config.get("crash_exitcode", 17)))
    if behavior == "error263":
        raise RuntimeError("xCoreSDK getEndTorque failed (263): synthetic offline fault")
    if behavior == "exception":
        raise RuntimeError("synthetic worker exception")
    return {
        "joint_measured_torque_nm": [float(sequence_id)] * 6,
        "joint_external_torque_nm": [0.0] * 6,
        "cartesian_force_raw_n": [1.0, 2.0, 3.0],
        "cartesian_torque_raw_nm": [0.1, 0.2, 0.3],
    }


def _load_live_sdk(
    config: Mapping[str, Any],
    *,
    include_wrench: bool = True,
) -> tuple[Any, Any, Any | None, dict[str, Any]]:
    # Import only inside the spawned owner process.  No native object crosses IPC.
    from hardware.windows.rokae_xcore import _load_sdk

    sdk = _load_sdk()
    robot_type = getattr(sdk, str(config.get("robot_class", "xMateRobot")))
    robot = robot_type()
    robot_ip = str(config["robot_ip"])
    local_ip = str(config.get("local_ip", ""))
    if local_ip:
        robot.connectToRobot(robot_ip, local_ip)
    else:
        robot.connectToRobot(robot_ip)

    def call(action: str, method: Callable[..., Any]) -> Any:
        ec: dict[str, Any] = {}
        result = method(ec)
        code = int(ec.get("ec", 0))
        if code:
            raise RuntimeError(
                f"xCoreSDK {action} failed ({code}): {ec.get('message', 'unknown')}"
            )
        return result

    info = call("robotInfo", robot.robotInfo)
    operation = call("operationState", robot.operationState)
    power = call("powerState", robot.powerState)
    operate_mode = call("operateMode", robot.operateMode)
    metadata = {
        "sdk_version": str(sdk.BaseRobot.sdkVersion()),
        "controller_version": str(info.version),
        "robot_model": str(info.type),
        "robot_serial": str(info.id),
        "operation_state": operation.name,
        "power_state": power.name,
        "operate_mode": operate_mode.name,
    }
    if operation != sdk.OperationState.idle:
        raise RuntimeError(f"requires operationState=idle, observed {operation.name}")
    if power != sdk.PowerState.on:
        raise RuntimeError(f"requires powerState=on, observed {power.name}")
    if operate_mode != sdk.OperateMode.automatic:
        raise RuntimeError(f"requires operateMode=automatic, observed {operate_mode.name}")
    return sdk, robot, robot.forceControl() if include_wrench else None, metadata


def _live_query(sdk: Any, force_control: Any) -> dict[str, Any]:
    joint_measured = sdk.PyTypeVectorDouble()
    joint_external = sdk.PyTypeVectorDouble()
    cart_torque = sdk.PyTypeVectorDouble()
    cart_force = sdk.PyTypeVectorDouble()
    ec: dict[str, Any] = {}
    force_control.getEndTorque(
        sdk.FrameType.world,
        joint_measured,
        joint_external,
        cart_torque,
        cart_force,
        ec,
    )
    code = int(ec.get("ec", 0))
    if code:
        raise RuntimeError(
            f"xCoreSDK getEndTorque failed ({code}): {ec.get('message', 'unknown')}"
        )
    return {
        "joint_measured_torque_nm": [float(item) for item in joint_measured.content()[:6]],
        "joint_external_torque_nm": [float(item) for item in joint_external.content()[:6]],
        "cartesian_force_raw_n": [float(item) for item in cart_force.content()[:3]],
        "cartesian_torque_raw_nm": [float(item) for item in cart_torque.content()[:3]],
    }


def _disconnect_live(robot: Any) -> None:
    ec: dict[str, Any] = {}
    robot.disconnectFromRobot(ec)
    code = int(ec.get("ec", 0))
    if code:
        raise RuntimeError(
            f"xCoreSDK disconnectFromRobot failed ({code}): {ec.get('message', 'unknown')}"
        )


def _worker_entry(
    mode: str,
    config: dict[str, Any],
    snapshot_queue: Any,
    heartbeat_queue: Any,
    stop_event: Any,
) -> None:
    start_ns = time.perf_counter_ns()
    connected = False
    robot: Any | None = None
    sdk: Any | None = None
    force_control: Any | None = None
    metadata: dict[str, Any] = {}
    last_success_ns: int | None = None
    last_error_code: int | None = None
    last_error: str | None = None
    graceful: bool | None = None
    sequence_id = 0
    target_hz = float(config.get("target_hz", 20.0))
    period_ns = max(1, int(1e9 / target_hz))
    _heartbeat(
        heartbeat_queue,
        start_ns=start_ns,
        state="starting",
        connected=False,
        last_success_ns=None,
        last_error_code=None,
        last_error=None,
    )
    try:
        if mode == "live":
            _heartbeat(
                heartbeat_queue,
                start_ns=start_ns,
                state="connecting",
                connected=False,
                last_success_ns=None,
                last_error_code=None,
                last_error=None,
            )
            sdk, robot, force_control, metadata = _load_live_sdk(config)
            connected = True
        elif mode != "offline":
            raise ValueError(f"unknown worker mode: {mode}")
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="ready",
            connected=connected,
            last_success_ns=None,
            last_error_code=None,
            last_error=None,
            metadata=metadata,
        )
        next_tick_ns = time.perf_counter_ns()
        while not stop_event.is_set():
            sequence_id += 1
            started_ns = time.perf_counter_ns()
            _heartbeat(
                heartbeat_queue,
                start_ns=start_ns,
                state="querying",
                connected=connected,
                last_success_ns=last_success_ns,
                last_error_code=last_error_code,
                last_error=last_error,
                metadata=metadata,
            )
            try:
                if mode == "live":
                    assert sdk is not None and force_control is not None
                    value = _live_query(sdk, force_control)
                else:
                    value = _offline_query(config, sequence_id)
                finished_ns = time.perf_counter_ns()
                _publish_result(
                    snapshot_queue,
                    sequence_id=sequence_id,
                    started_ns=started_ns,
                    finished_ns=finished_ns,
                    success=True,
                    value=value,
                )
                last_success_ns = finished_ns
            except Exception as exc:
                finished_ns = time.perf_counter_ns()
                last_error_code = _error_code(exc)
                last_error = f"{type(exc).__name__}:{exc}"
                _publish_result(
                    snapshot_queue,
                    sequence_id=sequence_id,
                    started_ns=started_ns,
                    finished_ns=finished_ns,
                    success=False,
                    error=exc,
                )
            _heartbeat(
                heartbeat_queue,
                start_ns=start_ns,
                state="idle",
                connected=connected,
                last_success_ns=last_success_ns,
                last_error_code=last_error_code,
                last_error=last_error,
                metadata=metadata,
            )
            next_tick_ns += period_ns
            remaining_ns = next_tick_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                next_tick_ns = time.perf_counter_ns() + period_ns
    except BaseException as exc:
        last_error_code = _error_code(exc)
        last_error = f"{type(exc).__name__}:{exc}"
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="fatal_error",
            connected=connected,
            last_success_ns=last_success_ns,
            last_error_code=last_error_code,
            last_error=last_error,
            metadata=metadata,
        )
    finally:
        if mode == "live" and robot is not None and connected:
            try:
                _disconnect_live(robot)
                graceful = True
                connected = False
            except BaseException as exc:
                graceful = False
                last_error_code = _error_code(exc)
                last_error = f"{type(exc).__name__}:{exc}"
        elif mode == "offline":
            graceful = True
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="stopped",
            connected=connected,
            last_success_ns=last_success_ns,
            last_error_code=last_error_code,
            last_error=last_error,
            graceful_disconnect_confirmed=graceful,
            metadata=metadata,
        )
        for channel in (snapshot_queue, heartbeat_queue):
            try:
                channel.close()
                channel.join_thread()
            except (OSError, ValueError):
                pass


def _sanity_worker(config: dict[str, Any], result_queue: Any) -> None:
    robot: Any | None = None
    connected = False
    graceful = False
    metadata: dict[str, Any] = {}
    error: str | None = None
    try:
        _sdk, robot, _force_control, metadata = _load_live_sdk(
            config,
            include_wrench=False,
        )
        connected = True
        _disconnect_live(robot)
        connected = False
        graceful = True
    except BaseException as exc:
        error = f"{type(exc).__name__}:{exc}"
        if robot is not None and connected:
            try:
                _disconnect_live(robot)
                connected = False
                graceful = True
            except BaseException as disconnect_exc:
                graceful = False
                error += f";disconnect:{type(disconnect_exc).__name__}:{disconnect_exc}"
    publish_latest(result_queue, {
        "worker_pid": os.getpid(),
        "fresh_connection_success": bool(metadata),
        "status_read_success": bool(metadata),
        "disconnect_success": graceful,
        "metadata": metadata,
        "error": error,
    })
    try:
        result_queue.close()
        result_queue.join_thread()
    except (OSError, ValueError):
        pass


def run_fresh_sanity(
    config: Mapping[str, Any],
    *,
    timeout_s: float = 30.0,
    context: BaseContext | None = None,
) -> dict[str, Any]:
    """Run one fresh-process, read-only connect/status/disconnect check."""

    ctx = context or mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_sanity_worker,
        args=(dict(config), result_queue),
        name="diagnostic-rokae-fresh-sanity",
        daemon=False,
    )
    started_ns = time.perf_counter_ns()
    process.start()
    process.join(timeout=max(0.0, float(timeout_s)))
    terminated = False
    if process.is_alive():
        terminated = True
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    result = drain_latest(result_queue) or {}
    result.update({
        "process_exitcode": process.exitcode,
        "process_terminated_by_parent": terminated,
        "elapsed_ms": (time.perf_counter_ns() - started_ns) / 1e6,
        "graceful_disconnect_confirmed": bool(result.get("disconnect_success")) and not terminated,
    })
    result_queue.close()
    result_queue.cancel_join_thread()
    return result


class RtLatestSharedSnapshot:
    """Single-writer latest RT snapshot backed by fixed shared memory.

    The RT worker never waits for a consumer and does not use a
    ``multiprocessing.Queue`` feeder thread for the RT data path.  A small
    sequence lock lets the supervisor reject a snapshot that was being
    replaced while it was copied.  Replacements are IPC overwrites, not RT
    network packet-loss observations.
    """

    _VERSION = 0
    _RT_SEQUENCE = 1
    _SOURCE_NS = 2
    _PUBLISH_NS = 3
    _PUBLISH_COUNT = 4
    _RT_VALID = 5
    _HEADER_SIZE = 6
    _VECTOR_SIZE = 12

    def __init__(self, context: BaseContext) -> None:
        self._header = context.Array("q", self._HEADER_SIZE, lock=False)
        self._vectors = context.Array("d", self._VECTOR_SIZE, lock=False)

    def publish(self, payload: Mapping[str, Any]) -> int:
        version = int(self._header[self._VERSION])
        if version & 1:
            version += 1
        publish_count = int(self._header[self._PUBLISH_COUNT]) + 1
        self._header[self._VERSION] = version + 1
        self._header[self._RT_SEQUENCE] = int(payload["rt_sequence"])
        self._header[self._SOURCE_NS] = int(payload["source_or_receive_timestamp_ns"])
        self._header[self._PUBLISH_NS] = int(payload["publish_timestamp_ns"])
        self._header[self._PUBLISH_COUNT] = publish_count
        self._header[self._RT_VALID] = 1 if payload.get("rt_valid") else 0
        vector = [
            *list(payload.get("tcp_pose_abc_m_rad") or ())[:6],
            *list(payload.get("joint_position_rad") or ())[:6],
        ]
        vector.extend([0.0] * (self._VECTOR_SIZE - len(vector)))
        for index, value in enumerate(vector[: self._VECTOR_SIZE]):
            self._vectors[index] = float(value)
        self._header[self._VERSION] = version + 2
        return publish_count

    def read(self, attempts: int = 8) -> dict[str, Any] | None:
        for _ in range(max(1, int(attempts))):
            version_before = int(self._header[self._VERSION])
            if version_before == 0:
                return None
            if version_before & 1:
                continue
            header = [int(value) for value in self._header]
            vectors = [float(value) for value in self._vectors]
            version_after = int(self._header[self._VERSION])
            if version_before != version_after or version_after & 1:
                continue
            publish_count = header[self._PUBLISH_COUNT]
            return {
                "schema_version": IPC_SCHEMA_VERSION,
                "rt_sequence": header[self._RT_SEQUENCE],
                "source_or_receive_timestamp_ns": header[self._SOURCE_NS],
                "publish_timestamp_ns": header[self._PUBLISH_NS],
                "rt_timestamp_ns": header[self._SOURCE_NS],
                "rt_valid": bool(header[self._RT_VALID]),
                "operation_state": "IDLE",
                "tcp_pose_abc_m_rad": vectors[:6],
                "joint_position_rad": vectors[6:12],
                "publish_count": publish_count,
                "publish_success_count": publish_count,
                "publish_drop_count": 0,
            }
        return None


@dataclass(frozen=True)
class RtProcessObservation:
    host_timestamp_ns: int
    worker_pid: int | None
    worker_start_time_ns: int | None
    worker_alive: bool
    worker_exitcode: int | None
    worker_state: str
    last_heartbeat_ns: int | None
    heartbeat_age_ms: float | None
    worker_hung: bool
    rt_sequence: int
    source_or_receive_timestamp_ns: int | None
    publish_timestamp_ns: int | None
    supervisor_receive_timestamp_ns: int | None
    rt_timestamp_ns: int | None
    rt_age_ms: float | None
    rt_ipc_age_ms: float | None
    rt_publish_to_receive_age_ms: float | None
    new_snapshot_received: bool
    publish_count: int
    publish_success_count: int
    receive_count: int
    overwrite_count: int
    publish_drop_count: int
    rt_valid: bool
    rt_stale: bool
    operation_state: str | None
    last_error_code: int | None
    last_error: str | None
    graceful_disconnect_confirmed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RtProcessSupervisor:
    """Parent-side non-blocking view of a separately owned RT SDK session."""

    def __init__(
        self,
        *,
        stale_age_ms: float = 50.0,
        worker_hung_ms: float = 750.0,
        worker_startup_hung_ms: float = 15_000.0,
        context: BaseContext | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if min(stale_age_ms, worker_hung_ms, worker_startup_hung_ms) <= 0:
            raise ValueError("RT thresholds must be positive")
        self.stale_age_ms = float(stale_age_ms)
        self.worker_hung_ms = float(worker_hung_ms)
        self.worker_startup_hung_ms = float(worker_startup_hung_ms)
        self.context = context or mp.get_context("spawn")
        self.clock_ns = clock_ns
        self._shared_state: RtLatestSharedSnapshot | None = None
        self._heartbeat_queue: Any | None = None
        self._stop_event: Any | None = None
        self._process: Any | None = None
        self._last_state: dict[str, Any] | None = None
        self._last_heartbeat: dict[str, Any] | None = None
        self._last_supervisor_receive_ns: int | None = None
        self._receive_count = 0
        self._forced_termination = False

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    @property
    def metadata(self) -> dict[str, Any]:
        if self._last_heartbeat is None:
            return {}
        return dict(self._last_heartbeat.get("metadata") or {})

    def start(self, config: Mapping[str, Any]) -> int:
        if self._process is not None:
            raise RuntimeError("RT worker process has already been created")
        self._shared_state = RtLatestSharedSnapshot(self.context)
        self._heartbeat_queue = self.context.Queue(maxsize=4)
        self._stop_event = self.context.Event()
        self._process = self.context.Process(
            target=_rt_worker_entry,
            args=(dict(config), self._shared_state, self._heartbeat_queue, self._stop_event),
            name="diagnostic-rokae-rt-process",
            daemon=False,
        )
        self._process.start()
        return int(self._process.pid)

    def poll(self, now_ns: int | None = None) -> RtProcessObservation:
        poll_start_ns = self.clock_ns() if now_ns is None else int(now_ns)
        new_snapshot_received = False
        if self._shared_state is not None:
            state = self._shared_state.read()
            if state is not None and int(state.get("rt_sequence", 0)) != int(
                0 if self._last_state is None else self._last_state.get("rt_sequence", 0)
            ):
                self._last_state = state
                self._last_supervisor_receive_ns = self.clock_ns()
                self._receive_count += 1
                new_snapshot_received = True
        if self._heartbeat_queue is not None:
            heartbeat = drain_latest(self._heartbeat_queue)
            if heartbeat is not None:
                self._last_heartbeat = heartbeat
        current_ns = max(poll_start_ns, self.clock_ns())
        process = self._process
        alive = bool(process is not None and process.is_alive())
        exitcode = None if process is None else process.exitcode
        heartbeat_ns = None if self._last_heartbeat is None else self._last_heartbeat.get("last_heartbeat_ns")
        heartbeat_age_ms = None if heartbeat_ns is None else max(0.0, (current_ns - int(heartbeat_ns)) / 1e6)
        worker_state = "not_started" if self._last_heartbeat is None else str(self._last_heartbeat.get("state", "unknown"))
        hung_threshold_ms = (
            self.worker_startup_hung_ms
            if worker_state in {"not_started", "starting", "connecting"}
            else self.worker_hung_ms
        )
        hung = bool(alive and heartbeat_age_ms is not None and heartbeat_age_ms > hung_threshold_ms)
        timestamp_ns = (
            None
            if self._last_state is None
            else self._last_state.get(
                "source_or_receive_timestamp_ns",
                self._last_state.get("rt_timestamp_ns"),
            )
        )
        publish_timestamp_ns = (
            None if self._last_state is None else self._last_state.get("publish_timestamp_ns")
        )
        receive_timestamp_ns = self._last_supervisor_receive_ns
        age_ms = None if timestamp_ns is None else max(0.0, (current_ns - int(timestamp_ns)) / 1e6)
        ipc_age_ms = (
            None
            if timestamp_ns is None or receive_timestamp_ns is None
            else max(0.0, (int(receive_timestamp_ns) - int(timestamp_ns)) / 1e6)
        )
        publish_to_receive_age_ms = (
            None
            if publish_timestamp_ns is None or receive_timestamp_ns is None
            else max(0.0, (int(receive_timestamp_ns) - int(publish_timestamp_ns)) / 1e6)
        )
        stale = age_ms is None or age_ms > self.stale_age_ms
        graceful = None if self._last_heartbeat is None else self._last_heartbeat.get("graceful_disconnect_confirmed")
        if self._forced_termination:
            graceful = False
        return RtProcessObservation(
            host_timestamp_ns=current_ns,
            worker_pid=(self.pid if self._last_heartbeat is None else self._last_heartbeat.get("worker_pid", self.pid)),
            worker_start_time_ns=(None if self._last_heartbeat is None else self._last_heartbeat.get("worker_start_time_ns")),
            worker_alive=alive,
            worker_exitcode=exitcode,
            worker_state=worker_state,
            last_heartbeat_ns=None if heartbeat_ns is None else int(heartbeat_ns),
            heartbeat_age_ms=heartbeat_age_ms,
            worker_hung=hung,
            rt_sequence=0 if self._last_state is None else int(self._last_state.get("rt_sequence", 0)),
            source_or_receive_timestamp_ns=None if timestamp_ns is None else int(timestamp_ns),
            publish_timestamp_ns=(
                None if publish_timestamp_ns is None else int(publish_timestamp_ns)
            ),
            supervisor_receive_timestamp_ns=receive_timestamp_ns,
            rt_timestamp_ns=None if timestamp_ns is None else int(timestamp_ns),
            rt_age_ms=age_ms,
            rt_ipc_age_ms=ipc_age_ms,
            rt_publish_to_receive_age_ms=publish_to_receive_age_ms,
            new_snapshot_received=new_snapshot_received,
            publish_count=(
                0 if self._last_state is None else int(self._last_state.get("publish_count", 0))
            ),
            publish_success_count=(
                0
                if self._last_state is None
                else int(self._last_state.get("publish_success_count", 0))
            ),
            receive_count=self._receive_count,
            overwrite_count=(
                0
                if self._last_state is None
                else max(
                    0,
                    int(self._last_state.get("publish_count", 0)) - self._receive_count,
                )
            ),
            publish_drop_count=(
                0 if self._last_state is None else int(self._last_state.get("publish_drop_count", 0))
            ),
            rt_valid=bool(self._last_state is not None and self._last_state.get("rt_valid") and not stale),
            rt_stale=stale,
            operation_state=None if self._last_state is None else self._last_state.get("operation_state"),
            last_error_code=(None if self._last_heartbeat is None else self._last_heartbeat.get("last_error_code")),
            last_error=(None if self._last_heartbeat is None else self._last_heartbeat.get("last_error")),
            graceful_disconnect_confirmed=None if graceful is None else bool(graceful),
        )

    def request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def join(self, timeout_s: float) -> bool:
        if self._process is None:
            return True
        self._process.join(timeout=max(0.0, float(timeout_s)))
        self.poll()
        return not self._process.is_alive()

    def stop_normally(self, timeout_s: float = 5.0) -> dict[str, Any]:
        started_ns = self.clock_ns()
        self.request_stop()
        exited = self.join(timeout_s)
        observation = self.poll()
        return {
            "worker_exited": exited,
            "worker_exitcode": observation.worker_exitcode,
            "latency_ms": (self.clock_ns() - started_ns) / 1e6,
            "graceful_disconnect_confirmed": observation.graceful_disconnect_confirmed,
            "forced": False,
        }

    def terminate(self, join_timeout_s: float = 2.0) -> dict[str, Any]:
        started_ns = self.clock_ns()
        process = self._process
        used_kill = False
        if process is not None and process.is_alive():
            self._forced_termination = True
            process.terminate()
            process.join(timeout=max(0.0, float(join_timeout_s)))
            if process.is_alive():
                used_kill = True
                process.kill()
                process.join(timeout=max(0.0, float(join_timeout_s)))
        observation = self.poll()
        return {
            "worker_terminated": bool(process is None or not process.is_alive()),
            "worker_exitcode": observation.worker_exitcode,
            "termination_latency_ms": (self.clock_ns() - started_ns) / 1e6,
            "used_kill": used_kill,
            "graceful_disconnect_confirmed": False,
            "forced": True,
        }

    def close(self) -> None:
        for channel in (self._heartbeat_queue,):
            if channel is not None:
                try:
                    channel.close()
                    channel.cancel_join_thread()
                except (OSError, ValueError):
                    pass


def _rt_worker_entry(
    config: dict[str, Any],
    shared_state: RtLatestSharedSnapshot,
    heartbeat_queue: Any,
    stop_event: Any,
) -> None:
    start_ns = time.perf_counter_ns()
    native: Any | None = None
    connected = False
    streaming = False
    metadata: dict[str, Any] = {}
    last_error: str | None = None
    last_error_code: int | None = None
    graceful: bool | None = None
    counters = {"publish_count": 0, "publish_success_count": 0, "publish_drop_count": 0}
    source_period_samples_ms: list[float] = []
    publish_interval_samples_ms: list[float] = []
    previous_source_ns: int | None = None
    previous_publish_ns: int | None = None
    update_timeout_count = 0
    _heartbeat(
        heartbeat_queue,
        start_ns=start_ns,
        state="starting",
        connected=False,
        last_success_ns=None,
        last_error_code=None,
        last_error=None,
    )
    try:
        if bool(config.get("offline")):
            metadata = {
                "source": "synthetic_rt_offline",
                "rt_process_single_thread": True,
                "rt_ipc_transport": "fixed_shared_memory_latest_snapshot",
                "timestamp_source": "synthetic_host_monotonic_receive_timestamp",
            }
            _heartbeat(
                heartbeat_queue,
                start_ns=start_ns,
                state="ready",
                connected=False,
                last_success_ns=None,
                last_error_code=None,
                last_error=None,
                metadata=metadata,
            )
            sequence_id = 0
            period_ns = int(1e9 / float(config.get("publish_hz", 125.0)))
            next_tick_ns = time.perf_counter_ns()
            last_heartbeat_publish_ns = 0
            while not stop_event.is_set():
                sequence_id += 1
                receive_ns = time.perf_counter_ns()
                publish_ns = time.perf_counter_ns()
                if previous_source_ns is not None:
                    source_period_samples_ms.append((receive_ns - previous_source_ns) / 1e6)
                if previous_publish_ns is not None:
                    publish_interval_samples_ms.append((publish_ns - previous_publish_ns) / 1e6)
                previous_source_ns = receive_ns
                previous_publish_ns = publish_ns
                counters["publish_count"] = shared_state.publish({
                    "schema_version": IPC_SCHEMA_VERSION,
                    "rt_sequence": sequence_id,
                    "source_or_receive_timestamp_ns": receive_ns,
                    "publish_timestamp_ns": publish_ns,
                    "rt_timestamp_ns": receive_ns,
                    "timestamp_source": "synthetic_host_monotonic_receive_timestamp",
                    "rt_valid": True,
                    "operation_state": "IDLE",
                    "tcp_pose_abc_m_rad": [0.0] * 6,
                    "joint_position_rad": [0.0] * 6,
                })
                counters["publish_success_count"] = counters["publish_count"]
                if publish_ns - last_heartbeat_publish_ns >= 100_000_000:
                    _heartbeat(
                        heartbeat_queue,
                        start_ns=start_ns,
                        state="streaming",
                        connected=False,
                        last_success_ns=receive_ns,
                        last_error_code=None,
                        last_error=None,
                        metadata={**metadata, **counters},
                    )
                    last_heartbeat_publish_ns = publish_ns
                next_tick_ns += period_ns
                remaining_ns = next_tick_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    stop_event.wait(remaining_ns / 1e9)
                else:
                    next_tick_ns = time.perf_counter_ns() + period_ns
            graceful = True
            return
        # This is intentionally a direct, single-threaded diagnostic path.
        # Do not call RokaeRobot.connect(): it creates an internal RT listener
        # thread and would reintroduce listener -> cache -> publisher latency.
        from hardware.windows.rokae_xcore import _load_sdk

        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="connecting",
            connected=False,
            last_success_ns=None,
            last_error_code=None,
            last_error=None,
        )
        sdk = _load_sdk()
        sdk_version = str(sdk.BaseRobot.sdkVersion())
        robot_type = getattr(sdk, str(config.get("robot_class", "xMateRobot")))
        native = robot_type()
        robot_ip = str(config["robot_ip"])
        local_ip = str(config["local_ip"])
        if local_ip:
            native.connectToRobot(robot_ip, local_ip)
        else:
            native.connectToRobot(robot_ip)
        connected = True

        def call(action: str, method: Callable[..., Any]) -> Any:
            ec: dict[str, Any] = {}
            value = method(ec)
            code = int(ec.get("ec", 0))
            if code:
                raise RuntimeError(
                    f"xCoreSDK {action} failed ({code}): "
                    f"{ec.get('message', 'unknown xCoreSDK error')}"
                )
            return value

        info = call("robotInfo", native.robotInfo)
        operation = call("operationState", native.operationState)
        power = call("powerState", native.powerState)
        operate_mode = call("operateMode", native.operateMode)
        metadata = {
            "sdk_version": sdk_version,
            "controller_version": str(info.version),
            "robot_model": str(info.type),
            "robot_serial": str(info.id),
            "operation_state": operation.name,
            "power_state": power.name,
            "operate_mode": operate_mode.name,
            "rt_process_single_thread": True,
            "rt_ipc_transport": "fixed_shared_memory_latest_snapshot",
            "timestamp_source": "host_monotonic_immediately_after_getStateData_no_controller_timestamp",
            "state_interval_ms": int(config.get("state_interval_ms", 8)),
        }
        if operation != sdk.OperationState.idle:
            raise RuntimeError(f"requires operationState=idle, observed {operation.name}")
        if power != sdk.PowerState.on:
            raise RuntimeError(f"requires powerState=on, observed {power.name}")
        if operate_mode != sdk.OperateMode.automatic:
            raise RuntimeError(f"requires operateMode=automatic, observed {operate_mode.name}")
        interval_ms = int(config.get("state_interval_ms", 8))
        fields = [
            sdk.RtSupportedFields.tcpPoseAbc_m,
            sdk.RtSupportedFields.jointPos_m,
            sdk.RtSupportedFields.keypads,
        ]
        native.startReceiveRobotState(timedelta(milliseconds=interval_ms), fields)
        streaming = True
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="ready",
            connected=True,
            last_success_ns=None,
            last_error_code=None,
            last_error=None,
            metadata=metadata,
        )
        sequence_id = 0
        last_heartbeat_publish_ns = 0
        timeout = timedelta(milliseconds=max(1, interval_ms * 2))
        while not stop_event.is_set():
            updated = native.updateRobotState(timeout)
            if not updated:
                update_timeout_count += 1
                continue
            tcp = sdk.PyTypeVectorDouble()
            joints = sdk.PyTypeVectorDouble()
            keypads = sdk.PyTypeVectorBool()
            native.getStateData(sdk.RtSupportedFields.tcpPoseAbc_m, tcp, 6)
            native.getStateData(sdk.RtSupportedFields.jointPos_m, joints, 6)
            native.getStateData(sdk.RtSupportedFields.keypads, keypads)
            # xCoreSDK exposes no controller timestamp for these RT fields.
            # Timestamp immediately after all fields for this frame are read.
            receive_ns = time.perf_counter_ns()
            tcp_values = [float(value) for value in list(tcp.content())[:6]]
            joint_values = [float(value) for value in list(joints.content())[:6]]
            if len(tcp_values) != 6 or len(joint_values) != 6:
                raise RuntimeError("xCoreSDK returned an incomplete RT state frame")
            sequence_id += 1
            publish_ns = time.perf_counter_ns()
            if previous_source_ns is not None:
                source_period_samples_ms.append((receive_ns - previous_source_ns) / 1e6)
            if previous_publish_ns is not None:
                publish_interval_samples_ms.append((publish_ns - previous_publish_ns) / 1e6)
            previous_source_ns = receive_ns
            previous_publish_ns = publish_ns
            counters["publish_count"] = shared_state.publish({
                "schema_version": IPC_SCHEMA_VERSION,
                "rt_sequence": sequence_id,
                "source_or_receive_timestamp_ns": receive_ns,
                "publish_timestamp_ns": publish_ns,
                "rt_timestamp_ns": receive_ns,
                "timestamp_source": metadata["timestamp_source"],
                "rt_valid": True,
                "operation_state": operation.name.upper(),
                "tcp_pose_abc_m_rad": tcp_values,
                "joint_position_rad": joint_values,
                "keypad_state": [bool(value) for value in keypads.content()],
            })
            counters["publish_success_count"] = counters["publish_count"]
            if publish_ns - last_heartbeat_publish_ns >= 100_000_000:
                _heartbeat(
                    heartbeat_queue,
                    start_ns=start_ns,
                    state="streaming",
                    connected=True,
                    last_success_ns=receive_ns,
                    last_error_code=last_error_code,
                    last_error=last_error,
                    metadata={**metadata, **counters},
                )
                last_heartbeat_publish_ns = publish_ns
    except BaseException as exc:
        last_error_code = _error_code(exc)
        last_error = f"{type(exc).__name__}:{exc}"
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="fatal_error",
            connected=connected,
            last_success_ns=None,
            last_error_code=last_error_code,
            last_error=last_error,
            metadata=metadata,
        )
    finally:
        if native is not None and connected:
            try:
                if streaming:
                    native.stopReceiveRobotState()
                    streaming = False
                ec: dict[str, Any] = {}
                native.disconnectFromRobot(ec)
                code = int(ec.get("ec", 0))
                if code:
                    raise RuntimeError(
                        f"xCoreSDK disconnectFromRobot failed ({code}): "
                        f"{ec.get('message', 'unknown xCoreSDK error')}"
                    )
                connected = False
                graceful = True
            except BaseException as exc:
                graceful = False
                last_error_code = _error_code(exc)
                last_error = f"{type(exc).__name__}:{exc}"
        _heartbeat(
            heartbeat_queue,
            start_ns=start_ns,
            state="stopped",
            connected=connected,
            last_success_ns=None,
            last_error_code=last_error_code,
            last_error=last_error,
            graceful_disconnect_confirmed=graceful,
            metadata={
                **metadata,
                **counters,
                "source_period_ms": _timing_summary(source_period_samples_ms),
                "publish_interval_ms": _timing_summary(publish_interval_samples_ms),
                "source_period_gt_10ms_count": sum(
                    value > 10.0 for value in source_period_samples_ms
                ),
                "source_period_gt_20ms_count": sum(
                    value > 20.0 for value in source_period_samples_ms
                ),
                "source_period_gt_30ms_count": sum(
                    value > 30.0 for value in source_period_samples_ms
                ),
                "source_period_gt_50ms_count": sum(
                    value > 50.0 for value in source_period_samples_ms
                ),
                "source_period_gt_100ms_count": sum(
                    value > 100.0 for value in source_period_samples_ms
                ),
                "update_timeout_count": update_timeout_count,
            },
        )
        for channel in (heartbeat_queue,):
            try:
                channel.close()
                channel.join_thread()
            except (OSError, ValueError):
                pass


__all__ = [
    "IPC_SCHEMA_VERSION",
    "ProcessObservation",
    "RtProcessObservation",
    "RtProcessSupervisor",
    "WrenchProcessSupervisor",
    "drain_latest",
    "drain_latest_counted",
    "publish_latest",
    "publish_latest_counted",
    "run_fresh_sanity",
]
