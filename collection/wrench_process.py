"""Spawn-owned wrench queries with acknowledged, asynchronous audit events.

No native import occurs in the supervisor. Budgets have no production defaults.
The audit mailbox is stop-and-wait until the supervisor admits a record to a
bounded queue.  Disk persistence is asynchronous and separately supervised so a
50 Hz publisher never serially waits for one fsync per protocol event. Fixed-size
shared byte slots avoid multiprocessing Queue feeder/join locks that can be
corrupted when a native owner is terminated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from hardware.rokae_adapter import RobotWrenchFrame

VERSION = 1
CAPACITY = 65536


def _frame(payload):
    values = dict(payload)
    for key in ("cartesian_force_raw_n", "cartesian_torque_raw_nm",
                "joint_measured_torque_nm", "joint_external_torque_nm"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return RobotWrenchFrame(**values)


@dataclass(frozen=True)
class WrenchBudgets:
    startup_s: float
    query_s: float
    progress_s: float
    sample_age_s: float
    ack_s: float
    log_s: float
    shutdown_s: float
    terminate_s: float
    state_age_s: float
    skew_s: float
    provenance: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "provenance":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("budget provenance required")
            elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"explicit positive budget required: {key}")


class Slot:
    """One writer, bounded publication; readers never acquire child locks."""
    def __init__(self, context, capacity=CAPACITY):
        self.data = context.RawArray("B", capacity)
        self.length = context.RawValue("i", 0)
        self.version = context.RawValue("q", 0)

    def put(self, value):
        body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        raw = hashlib.sha256(body).digest() + body
        if len(raw) > len(self.data):
            raise ValueError("audit_overflow")
        self.version.value += 1  # odd: write in progress
        self.data[:len(raw)] = raw
        self.length.value = len(raw)
        self.version.value += 1

    def read(self):
        packet = self.read_packet()
        return None if packet is None else packet[1]

    def read_packet(self):
        version = self.version.value
        if version == 0 or version % 2:
            return None
        size = self.length.value
        if not 32 <= size <= len(self.data):
            raise ValueError("invalid_slot_length")
        raw = bytes(self.data[:size])
        if self.version.value != version:
            return None
        if hashlib.sha256(raw[32:]).digest() != raw[:32]:
            raise ValueError("torn_slot")
        return version, json.loads(raw[32:])


class DurableAudit:
    """Bounded batched writer with periodic and close-time fsync.

    A wedged filesystem writer is a daemon with explicit unresolved status;
    supervision never waits on its I/O lock and never spawns replacement writers.
    Tickets acknowledge a successful write+flush, not disk durability. Any
    periodic fsync failure is published globally and latched by the supervisor;
    close succeeds only after a final fsync.
    """
    def __init__(self, path, *, write_hook=None, sync_interval_s=.25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.work = queue.Queue(maxsize=64)
        self.failure = None
        self.closed = False
        self.write_hook = write_hook
        if (isinstance(sync_interval_s, bool)
                or not isinstance(sync_interval_s, (int, float))
                or not math.isfinite(sync_interval_s)
                or sync_interval_s <= 0):
            raise ValueError("sync_interval_s must be finite and positive")
        self.sync_interval_s = float(sync_interval_s)
        self.fsync_count = 0
        self.last_fsync_ns = None
        self.written_count = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="wrench-audit-writer")
        self.thread.start()

    def submit(self, event):
        if self.failure or self.closed:
            raise RuntimeError(self.failure or "audit_closed")
        ticket = {"done": threading.Event(), "error": None, "write_complete_ns": None}
        self.work.put_nowait((dict(event), ticket))
        return ticket

    def _run(self):
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                dirty = False
                last_sync_s = time.perf_counter()
                while True:
                    timeout = max(0.0, self.sync_interval_s - (time.perf_counter() - last_sync_s)) if dirty else None
                    try:
                        item = self.work.get(timeout=timeout)
                    except queue.Empty:
                        try:
                            stream.flush()
                            os.fsync(stream.fileno())
                            self.fsync_count += 1
                            self.last_fsync_ns = time.perf_counter_ns()
                            dirty = False
                            last_sync_s = time.perf_counter()
                        except BaseException as exc:
                            self.failure = f"audit_fsync:{type(exc).__name__}:{exc}"
                            return
                        continue
                    if item is None:
                        if dirty:
                            try:
                                stream.flush()
                                os.fsync(stream.fileno())
                                self.fsync_count += 1
                                self.last_fsync_ns = time.perf_counter_ns()
                            except BaseException as exc:
                                self.failure = f"audit_fsync:{type(exc).__name__}:{exc}"
                        return
                    batch = [item]
                    closing = False
                    # Absorb everything already queued. Records accumulated
                    # during a periodic fsync therefore become one write batch.
                    while not closing and len(batch) < self.work.maxsize:
                        try:
                            item = self.work.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            closing = True
                            break
                        batch.append(item)
                    try:
                        for event, ticket in batch:
                            if self.write_hook:
                                self.write_hook(event)
                            event["log_write_ns"] = time.perf_counter_ns()
                            stream.write(json.dumps(event, allow_nan=False) + "\n")
                        stream.flush()
                        write_complete_ns = time.perf_counter_ns()
                        for _, ticket in batch:
                            ticket["write_complete_ns"] = write_complete_ns
                        self.written_count += len(batch)
                        dirty = True
                    except BaseException as exc:
                        self.failure = f"audit_write:{type(exc).__name__}:{exc}"
                        for _, ticket in batch:
                            ticket["error"] = self.failure
                        return
                    finally:
                        for _, ticket in batch:
                            ticket["done"].set()
                    if closing:
                        if dirty:
                            try:
                                os.fsync(stream.fileno())
                                self.fsync_count += 1
                                self.last_fsync_ns = time.perf_counter_ns()
                            except BaseException as exc:
                                self.failure = f"audit_fsync:{type(exc).__name__}:{exc}"
                        return
        except BaseException as exc:
            self.failure = f"audit_open:{type(exc).__name__}:{exc}"

    def close(self, budget):
        if not self.closed:
            self.closed = True
            try:
                self.work.put_nowait(None)
            except queue.Full:
                self.failure = self.failure or "audit_shutdown_overflow"
        self.thread.join(budget)
        return not self.thread.is_alive() and self.failure is None


def _child(config, budgets, run_id, session_id, audit, latest, ack, stop, fatal):
    """Only this spawned function may construct a live wrench session."""
    event_id = 0
    query_id = 0
    session = None
    pid = os.getpid()

    def emit(kind, *, honor_stop=True, **fields):
        nonlocal event_id
        event_id += 1
        event = dict(version=VERSION, run_id=run_id, session_id=session_id,
                     pid=pid, event_id=event_id, query_id=query_id, type=kind,
                     timestamp_ns=time.perf_counter_ns(), **fields)
        audit.put(event)
        deadline = time.perf_counter() + budgets.ack_s
        while ack.value != event_id:
            if time.perf_counter() >= deadline:
                raise TimeoutError("ack_timeout")
            time.sleep(.001)
        return event

    try:
        if config.get("pre_start_delay_s"):
            time.sleep(config["pre_start_delay_s"])
        emit("PROCESS_START", connection=config, dispatch_ns=config["dispatch_ns"])
        if config.get("startup_delay_s"):
            time.sleep(config["startup_delay_s"])
        if config.get("connection_audit"):
            emit("CONNECT_START", session_kind=config["mode"])
        if config["mode"] == "live":
            from hardware.wrench_session import WrenchSession
            session = WrenchSession(config)
            identity = session.connect()
        else:
            identity = {"provider": "OFFLINE_STUB", "NOT_SAFETY_THRESHOLDS": True}
        if config.get("connection_audit"):
            emit("CONNECT_COMPLETE", identity=identity, session_kind=config["mode"])
        emit("PROCESS_READY", identity=identity)
        next_tick = time.perf_counter()
        while not stop.value:
            query_id += 1
            intent_ns = time.perf_counter_ns()
            scheduled_dispatch_ns = int(next_tick * 1e9)
            dispatch_lateness_ns = max(0, intent_ns - scheduled_dispatch_ns)
            # The in-progress marker is independent of the audit mailbox, so a
            # native hang remains observable even though each successful query
            # now needs only one cross-process audit admission.
            started = time.perf_counter_ns()
            latest.put(dict(query_id=query_id, run_id=run_id, pid=pid,
                            query_in_progress=True, completion=None, native_start_ns=started))
            raw = None
            error = None
            try:
                if session is not None:
                    frame = session.query()
                else:
                    from collection.wrench_test_provider import query
                    frame = query(config, query_id, started)
                raw = asdict(frame)
                if not frame.valid:
                    raise ValueError(frame.invalid_reason)
            except Exception as exc:
                error = {"code": getattr(exc, "code", None), "message": str(exc),
                         "exception_type": type(exc).__name__}
                raw = None
            ended = time.perf_counter_ns()
            payload = dict(query_id=query_id, run_id=run_id, session_id=session_id,
                           pid=pid, frame=raw, error=error, completion_ns=ended)
            latest.put(payload)
            emit("QUERY_ERROR" if error else "QUERY_SUCCESS", honor_stop=False,
                 intent_ns=intent_ns, boundary_ns=started,
                 scheduled_dispatch_ns=scheduled_dispatch_ns,
                 dispatch_lateness_ns=dispatch_lateness_ns,
                 native_start_ns=started, completion_ns=ended, frame=raw, error=error)
            if error:
                break  # no retry after SDK failure
            # Keep requested start-to-start cadence; never add a whole period
            # after query/ACK latency. Missed deadlines remain in intent records.
            next_tick = max(next_tick + 1.0 / config["rate_hz"], time.perf_counter())
            while not stop.value and time.perf_counter() < next_tick:
                time.sleep(.001)
    except InterruptedError:
        pass
    except BaseException as exc:
        fatal.put({"reason": f"child:{type(exc).__name__}:{exc}", "pid": pid})
    finally:
        try:
            if ack.value != event_id:
                raise RuntimeError("unacknowledged_audit_mailbox_cleanup_unconfirmed")
            emit("DISCONNECT_START", honor_stop=False)
            if config.get("disconnect_delay_s"):
                time.sleep(config["disconnect_delay_s"])
            if session is not None:
                session.disconnect()
            emit("DISCONNECT_COMPLETE", honor_stop=False,
                 graceful_sdk_disconnect=(True if session is not None else None),
                 offline_disconnect_completed=session is None)
            emit("PROCESS_STOPPED", honor_stop=False)
        except BaseException as exc:
            previous = fatal.read()
            prior = previous["reason"] + ";" if previous else ""
            fatal.put({"reason": prior + f"cleanup:{type(exc).__name__}:{exc}", "pid": pid})


class WrenchProcessProvider:
    """Single polling owner; health reads are immutable snapshots, no IPC I/O."""
    def __init__(self, config, budgets: WrenchBudgets, audit_path, *, audit_factory=DurableAudit):
        self.config = json.loads(json.dumps(config, allow_nan=False))
        if self.config.get("mode") not in ("offline", "live"):
            raise ValueError("explicit offline/live mode required")
        rate = self.config.get("rate_hz")
        if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("explicit rate_hz required")
        if self.config["mode"] == "offline" and budgets.provenance != "NOT_SAFETY_THRESHOLDS":
            raise ValueError("offline budgets must be labelled NOT_SAFETY_THRESHOLDS")
        if self.config["mode"] == "live":
            required = (
                "robot_ip",
                "local_ip",
                "robot_class",
                "expected_identity",
                "allowed_power_states",
                "allowed_operation_states",
            )
            if any(not self.config.get(k) for k in required) or budgets.provenance == "NOT_SAFETY_THRESHOLDS":
                raise ValueError("live identity/connection/reviewed budgets required")
            if set(self.config["expected_identity"]) != {"robot_model", "robot_serial", "controller_version", "sdk_version"}:
                raise ValueError("complete identity binding required")
            if not set(self.config["allowed_operation_states"]).issubset(
                {"idle", "rtcontrolling"}
            ):
                raise ValueError("invalid live wrench operation-state allowlist")
        self.budgets = budgets
        self.run_id = uuid.uuid4().hex
        self.session_id = uuid.uuid4().hex
        self.context = mp.get_context("spawn")
        self.audit_slot = Slot(self.context)
        self.latest_slot = Slot(self.context)
        self.fatal_slot = Slot(self.context)
        self.ack = self.context.RawValue("q", 0)
        self.stop_flag = self.context.RawValue("b", 0)
        self.audit_path = Path(audit_path)
        self.audit_factory = audit_factory
        self.writer = None
        self.process = None
        self.expected_event = 1
        self.seen_slot_version = 0
        self.pending_audits = deque()
        self.last_good = None
        self.last_error = None
        self.query_id = 0
        self.query_state = "NOT_STARTED"
        self.lifecycle_state = "NOT_STARTED"
        self.query_start_ns = None
        self.native_boundary_ns = None
        self.last_completed_ns = None
        self.progress_ns = None
        self.last_now_ns = None
        self.started_ns = None
        self.failure_latch = []
        self.events = deque(maxlen=1024)  # diagnostics cache; durable log is authoritative
        self.timeline = []
        self.stopping = False
        self.exit_seen = False
        self.disconnect_confirmed = None
        self.offline_disconnect_completed = False
        self.gaps = []
        self.cleanup = None
        self.supervisor_event_id = 0
        self.audit_written_count = 0
        self.audit_max_write_latency_ns = 0
        self.audit_last_write_complete_ns = None
        self.native_query_overrun_count = 0
        self.max_native_query_duration_ns = 0
        self._health = {}
        self._poll_lock = threading.Lock()
        self._record_lock = threading.RLock()
        self._supervisor_stop = threading.Event()
        self._supervisor_thread = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("no restart/new session on provider")
        self.writer = self.audit_factory(self.audit_path)
        self.started_ns = time.perf_counter_ns()
        self.config["dispatch_ns"] = self.started_ns
        self.process = self.context.Process(target=_child, args=(
            self.config, self.budgets, self.run_id, self.session_id,
            self.audit_slot, self.latest_slot, self.ack, self.stop_flag, self.fatal_slot),
            name="production-wrench-owner", daemon=False)
        self.process.start()
        # Publish a complete initial health record before concurrent callers
        # can lose the poll lock to the dedicated supervisor.
        self.poll()
        self._supervisor_stop.clear()
        self._supervisor_thread = threading.Thread(
            target=self._supervise,
            name="production-wrench-supervisor",
            daemon=True,
        )
        self._supervisor_thread.start()
        return self

    def _supervise(self):
        """Continuously service bounded IPC independently of acquisition I/O."""
        while not self._supervisor_stop.is_set():
            self.poll()
            self._supervisor_stop.wait(.002)

    def _cache_record(self, kind, **fields):
        with self._record_lock:
            self.supervisor_event_id += 1
            event = dict(version=VERSION, origin="supervisor", run_id=self.run_id,
                         session_id=self.session_id, pid=os.getpid(), query_id=self.query_id,
                         event_id=self.supervisor_event_id, type=kind,
                         timestamp_ns=time.perf_counter_ns(), **fields)
            self.events.append(event)
            return event

    def _record(self, kind, **fields):
        event = self._cache_record(kind, **fields)
        if self.writer:
            try:
                self.writer.submit(event)
            except queue.Full:
                if "logger_backpressure" not in self.failure_latch:
                    self.failure_latch.append("logger_backpressure")
                self.stop_flag.value = 1
            except Exception as exc:
                reason = f"logger_failure:{type(exc).__name__}:{exc}"
                if reason not in self.failure_latch:
                    self.failure_latch.append(reason)
                self.stop_flag.value = 1

    def fail(self, reason):
        with self._record_lock:
            if reason not in self.failure_latch:
                self.failure_latch.append(reason)
                self.stop_flag.value = 1
                self._record("FAILURE_LATCH", reason=reason)

    def _accept(self, event, now):
        for key, expected in (("version", VERSION), ("run_id", self.run_id),
                              ("session_id", self.session_id), ("pid", self.process.pid)):
            if event.get(key) != expected:
                raise ValueError(f"{key}_mismatch")
        if event.get("event_id") != self.expected_event:
            self.gaps.append({"expected": self.expected_event, "observed": event.get("event_id")})
            raise ValueError("event_sequence_gap_duplicate_or_reorder")
        stamp = event.get("timestamp_ns")
        if not isinstance(stamp, int) or stamp < self.started_ns or stamp > now:
            raise ValueError("invalid_event_clock")
        if self.progress_ns is not None and stamp < self.progress_ns:
            raise ValueError("clock_regression")
        kind = event["type"]
        qid = event.get("query_id")
        if kind in ("QUERY_SUCCESS", "QUERY_ERROR"):
            observed_in_progress = (
                qid == self.query_id
                and self.query_state in ("IN_PROGRESS", "HUNG")
            )
            completed_between_polls = (
                qid == self.query_id + 1
                and self.query_state in ("IDLE", "SUCCEEDED")
            )
            if not (observed_in_progress or completed_between_polls):
                raise ValueError("out_of_order_completion")
            query_start_ns = (
                self.query_start_ns if observed_in_progress else event.get("intent_ns")
            )
            if (not isinstance(query_start_ns, int)
                    or not query_start_ns <= event["native_start_ns"] <= event["completion_ns"] <= stamp):
                raise ValueError("invalid_completion_clock")
            if kind == "QUERY_ERROR" and (event.get("frame") is not None or not event.get("error")):
                raise ValueError("invalid_error_payload")
            if kind == "QUERY_SUCCESS":
                frame = _frame(event["frame"])
                if not frame.valid or frame.host_monotonic_time_s is None:
                    raise ValueError("invalid_success_frame")
                for values in (frame.cartesian_force_raw_n, frame.cartesian_torque_raw_nm):
                    if values is None or len(values) != 3 or not all(math.isfinite(x) for x in values):
                        raise ValueError("invalid_wrench_vector")
                if not event["native_start_ns"] / 1e9 <= frame.host_monotonic_time_s <= event["completion_ns"] / 1e9:
                    raise ValueError("invalid_sample_clock")
        elif kind == "PROCESS_START":
            if self.query_state != "NOT_STARTED" or qid != 0:
                raise ValueError("duplicate_process_start")
        elif kind == "PROCESS_READY":
            if self.query_state != "STARTING" or qid != 0:
                raise ValueError("duplicate_process_ready")
            if self.config.get("connection_audit") and self.lifecycle_state != "CONNECTED":
                raise ValueError("connection_evidence_missing")
        elif kind == "CONNECT_START":
            if not self.config.get("connection_audit") or self.lifecycle_state != "STARTING" or qid != 0:
                raise ValueError("out_of_order_connect_start")
        elif kind == "CONNECT_COMPLETE":
            if self.lifecycle_state != "CONNECTING" or qid != 0:
                raise ValueError("out_of_order_connect_complete")
        elif kind == "DISCONNECT_START":
            if qid != self.query_id or self.lifecycle_state not in ("STARTING", "CONNECTING", "CONNECTED", "READY"):
                raise ValueError("duplicate_disconnect_start")
        elif kind == "DISCONNECT_COMPLETE":
            if qid != self.query_id or self.lifecycle_state != "DISCONNECTING":
                raise ValueError("out_of_order_disconnect_complete")
        elif kind == "PROCESS_STOPPED":
            if qid != self.query_id or self.lifecycle_state != "DISCONNECTED":
                raise ValueError("out_of_order_process_stopped")
        else:
            raise ValueError("unknown_event")
        if kind == "QUERY_ERROR":
            self.fail("sdk_error")  # do not wait for fsync to invalidate stream
        if kind in ("QUERY_SUCCESS", "QUERY_ERROR"):
            # A completed query can replace the in-flight IPC marker between
            # polls. Its measured duration must still enforce the same budget.
            duration_ns = event["completion_ns"] - event["native_start_ns"]
            self.max_native_query_duration_ns = max(
                self.max_native_query_duration_ns, duration_ns
            )
            if duration_ns / 1e9 > self.budgets.query_s:
                self.native_query_overrun_count += 1
                self.fail("query_pending_too_long")
                self._record("QUERY_BUDGET_EXCEEDED", native_start_ns=event["native_start_ns"],
                             completion_ns=event["completion_ns"], duration_ns=duration_ns,
                             budget_s=self.budgets.query_s)
        event = dict(event, parent_receive_ns=now, parent_accept_ns=now)
        try:
            ticket = self.writer.submit(event)
        except queue.Full:
            self.fail("logger_backpressure")
            return
        except Exception as exc:
            self.fail(f"logger_failure:{type(exc).__name__}:{exc}")
            return
        self.pending_audits.append((event, ticket, now))
        self._apply_accepted(event)

    def _apply_accepted(self, event):
        """Apply an event only after bounded audit admission, then release child."""
        self.events.append(event)
        self.progress_ns = event["timestamp_ns"]
        kind = event["type"]
        if kind == "PROCESS_START":
            self.query_state = "STARTING"
            self.lifecycle_state = "STARTING"
        elif kind == "PROCESS_READY":
            self.query_state = "IDLE"
            self.lifecycle_state = "READY"
        elif kind == "CONNECT_START":
            self.lifecycle_state = "CONNECTING"
        elif kind == "CONNECT_COMPLETE":
            self.lifecycle_state = "CONNECTED"
        elif kind in ("QUERY_SUCCESS", "QUERY_ERROR"):
            self.query_id = event["query_id"]
            self.query_start_ns = event["native_start_ns"]
            self.native_boundary_ns = event.get("boundary_ns", event["native_start_ns"])
            self.last_completed_ns = event["completion_ns"]
            self.query_state = "ERROR" if kind == "QUERY_ERROR" else "SUCCEEDED"
            if kind == "QUERY_ERROR":
                self.last_error = event["error"]
                self.fail("sdk_error")
            else:
                self.last_good = _frame(event["frame"])
        elif kind == "DISCONNECT_START":
            self.lifecycle_state = "DISCONNECTING"
        elif kind == "DISCONNECT_COMPLETE":
            self.lifecycle_state = "DISCONNECTED"
            self.disconnect_confirmed = event["graceful_sdk_disconnect"]
            self.offline_disconnect_completed = event["offline_disconnect_completed"]
        elif kind == "PROCESS_STOPPED":
            self.lifecycle_state = "STOPPED"
        # The child ACK means bounded admission, never fsync completion. Actual
        # write completion is supervised through pending_audits. Disk sync is
        # periodic and its failures are reported through writer.failure.
        self.ack.value = event["event_id"]
        self.expected_event += 1

    def _drain_audits(self, now, *, record_commits=True):
        """Consume completed fsync tickets and latch any persistence failure."""
        writes = {}
        while self.pending_audits:
            event, ticket, accepted_ns = self.pending_audits[0]
            if not ticket["done"].is_set():
                if (now - accepted_ns) / 1e9 > self.budgets.log_s:
                    self.fail("logger_backpressure")
                break
            self.pending_audits.popleft()
            if ticket["error"]:
                self.fail("logger_failure:" + ticket["error"])
                continue
            write_complete_ns = ticket.get("write_complete_ns")
            if not isinstance(write_complete_ns, int) or write_complete_ns < accepted_ns:
                self.fail("logger_failure:invalid_write_timestamp")
                continue
            if (write_complete_ns - accepted_ns) / 1e9 > self.budgets.log_s:
                self.fail("logger_backpressure")
            latency_ns = write_complete_ns - accepted_ns
            self.audit_written_count += 1
            self.audit_max_write_latency_ns = max(self.audit_max_write_latency_ns, latency_ns)
            self.audit_last_write_complete_ns = write_complete_ns
            writes.setdefault(write_complete_ns, []).append(event["event_id"])
        if record_commits:
            for write_complete_ns, event_ids in writes.items():
                # Runtime provenance only: feeding receipts back into the same
                # persistent queue creates audit write amplification. A single
                # close-time summary is persisted after all tickets settle.
                self._cache_record(
                    "AUDIT_BATCH_WRITTEN",
                    child_event_ids=event_ids,
                    write_complete_ns=write_complete_ns,
                )

    def poll(self, now_ns=None):
        if self.cleanup is not None:
            return self.health()  # terminated child's IPC is quarantined
        if not self._poll_lock.acquire(blocking=False):
            return self.health()
        try:
            return self._poll(now_ns)
        finally:
            self._poll_lock.release()

    def _poll(self, now_ns=None):
        now = time.perf_counter_ns() if now_ns is None else now_ns
        if not isinstance(now, int) or now < (self.last_now_ns or self.started_ns or 0):
            self.fail("invalid_supervisor_clock")
            now = time.perf_counter_ns()
        self.last_now_ns = now
        if self.writer and self.writer.failure:
            self.fail("logger_failure:" + self.writer.failure)
        try:
            fatal = self.fatal_slot.read()
            if fatal:
                self.fail(fatal["reason"])
            self._drain_audits(now)
            packet = self.audit_slot.read_packet()
            if packet and packet[0] != self.seen_slot_version:
                version, event = packet
                self.seen_slot_version = version
                self._accept(event, max(now, time.perf_counter_ns()))
        except Exception as exc:
            self.fail(f"telemetry_failure:{type(exc).__name__}:{exc}")
        if now_ns is None:
            now = time.perf_counter_ns()
        alive = bool(self.process and self.process.is_alive())
        pending = self.query_state in ("START_INTENT", "IN_PROGRESS", "HUNG")
        native_marker = None
        try:
            candidate_marker = self.latest_slot.read()
            if (
                isinstance(candidate_marker, dict)
                and candidate_marker.get("run_id") == self.run_id
                and candidate_marker.get("pid") == getattr(self.process, "pid", None)
                and candidate_marker.get("query_id") in (self.query_id, self.query_id + 1)
            ):
                native_marker = candidate_marker
        except Exception as exc:
            self.fail(f"telemetry_failure:{type(exc).__name__}:{exc}")
        native_pending = bool(
            native_marker is not None
            and native_marker.get("query_in_progress") is True
            and isinstance(native_marker.get("native_start_ns"), int)
        )
        if (native_pending and native_marker["query_id"] == self.query_id + 1
                and self.query_state in ("IDLE", "SUCCEEDED")):
            self.query_id = native_marker["query_id"]
            self.query_start_ns = native_marker["native_start_ns"]
            self.native_boundary_ns = native_marker["native_start_ns"]
            self.query_state = "IN_PROGRESS"
            pending = True
        sample_time = self.last_good.host_monotonic_time_s if self.last_good else None
        age = None if sample_time is None else now / 1e9 - sample_time
        progress_age = None if self.progress_ns is None else (now - self.progress_ns) / 1e9
        if not self.stopping and not self.failure_latch:
            if not self.progress_ns and self.started_ns and (now - self.started_ns) / 1e9 > self.budgets.startup_s:
                self.fail("missing_first_heartbeat")
            if self.query_state in ("STARTING", "NOT_STARTED"):
                if self.started_ns and (now - self.started_ns) / 1e9 > self.budgets.startup_s:
                    self.fail("startup_timeout")
            # query_s starts at the child-published native_start_ns marker.
            # The read-only query result is admitted as one bounded audit event;
            # audit write/sync work is not counted as vendor query time.
            if (
                native_pending
                and (now - int(native_marker["native_start_ns"])) / 1e9
                > self.budgets.query_s
            ):
                if self.query_state != "HUNG":
                    self.query_state = "HUNG"
                    self._record("QUERY_HUNG", completion=None)
                self.fail("query_pending_too_long")
            # A primary fault already requests child shutdown. Do not obscure
            # its root cause with stale/progress cascade faults while the
            # uninterruptible native call returns and disconnects.
            if (not self.failure_latch and progress_age is not None
                    and progress_age > self.budgets.progress_s):
                self.fail("worker_progress_missing")
            if (not self.failure_latch and age is not None
                    and (age < 0 or age > self.budgets.sample_age_s)):
                self.fail("wrench_sample_stale_or_clock_invalid")
        if self.process and not alive and not self.exit_seen:
            self.exit_seen = True
            self._record("PROCESS_EXIT", exitcode=self.process.exitcode, completion=None if pending else self.last_completed_ns)
            if not self.stopping and not self.failure_latch:
                self.fail("worker_exited")
        self._health = dict(LATEST_GOOD_SAMPLE_VALID=bool(self.last_good and self.last_good.valid),
            CURRENT_QUERY_STATE=self.query_state, LAST_ERROR=self.last_error,
            QUERY_IN_PROGRESS=pending, QUERY_START_TIME=self.query_start_ns,
            LAST_COMPLETED_QUERY_TIME=self.last_completed_ns, WRENCH_SAMPLE_AGE=age,
            WORKER_HEARTBEAT_AGE=progress_age, PROCESS_ALIVE=alive,
            STREAM_HEALTH=bool(alive and self.last_good and not self.failure_latch and not self.stopping),
            FAILURE_LATCH=tuple(self.failure_latch), entry_unconfirmed=pending,
            AUDIT_PENDING_COUNT=len(self.pending_audits),
            NATIVE_QUERY_OVERRUN_COUNT=self.native_query_overrun_count,
            MAX_NATIVE_QUERY_DURATION_MS=self.max_native_query_duration_ns / 1e6,
            host_ns=now, query_id=self.query_id, completion=None if pending else self.last_completed_ns)
        # Explicit caller-owned collection; avoid unbounded 900 s in-memory history.
        return dict(self._health)

    def health(self):
        value = dict(self._health)
        # A fault published outside poll must be visible immediately, even when
        # another consumer owns poll or a logger is backpressured.
        if self.failure_latch:
            value.update(STREAM_HEALTH=False, FAILURE_LATCH=tuple(self.failure_latch))
        return value

    def stop(self):
        if self.cleanup is not None:
            return dict(self.cleanup)
        self.stopping = True
        self.stop_flag.value = 1
        self._record("PROCESS_STOP_REQUEST")
        deadline = time.perf_counter() + self.budgets.shutdown_s
        while self.process and self.process.is_alive() and time.perf_counter() < deadline:
            self.poll()
            self.process.join(.002)
        self.poll()
        forced = bool(self.process and self.process.is_alive())
        if forced:
            self.fail("forced_worker_termination")
            self._record("FORCED_WORKER_TERMINATION", completion=None)
            self.process.terminate()
            self.process.join(self.budgets.terminate_s)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(self.budgets.terminate_s)
            self.disconnect_confirmed = False
            # Do not read child-owned memory after forced termination.
        self._supervisor_stop.set()
        supervisor = self._supervisor_thread
        if supervisor is not None and supervisor is not threading.current_thread():
            supervisor.join(self.budgets.shutdown_s)
            if supervisor.is_alive():
                self.fail("supervisor_thread_stop_unconfirmed")
            else:
                self._supervisor_thread = None
        exited = bool(self.process and not self.process.is_alive())
        if exited and not self.exit_seen:
            self.exit_seen = True
            self._record("PROCESS_EXIT", exitcode=self.process.exitcode,
                         completion=None if self.query_state in ("HUNG", "IN_PROGRESS", "START_INTENT") else self.last_completed_ns)
        # Child ACKs mean bounded admission. Before declaring cleanup, settle
        # every admitted write ticket within the reviewed logging budget. The
        # writer close below performs the final fsync for all records.
        audit_deadline = time.perf_counter() + self.budgets.log_s
        while self.pending_audits and time.perf_counter() < audit_deadline:
            self._drain_audits(time.perf_counter_ns())
            if self.pending_audits:
                time.sleep(.001)
        if self.pending_audits:
            self.fail("logger_backpressure")
        disconnect_complete = (
            self.disconnect_confirmed is True if self.config["mode"] == "live"
            else self.offline_disconnect_completed
        )
        self.cleanup = dict(NORMAL_CLEANUP=bool(exited and not forced and disconnect_complete
                                              and not self.failure_latch),
                            GRACEFUL_SDK_DISCONNECT=False if forced else self.disconnect_confirmed,
                            OFFLINE_DISCONNECT_COMPLETED=self.offline_disconnect_completed,
                            FORCED_WORKER_TERMINATION=forced, CONTROLLER_SESSION_RECOVERY=None,
                            process_exited=exited, exitcode=self.process.exitcode if self.process else None,
                            query_state=self.query_state, failure_latch=list(self.failure_latch))
        self._record(
            "AUDIT_WRITE_SUMMARY",
            written_child_event_count=self.audit_written_count,
            pending_child_event_count=len(self.pending_audits),
            max_child_write_latency_ns=self.audit_max_write_latency_ns,
            last_child_write_complete_ns=self.audit_last_write_complete_ns,
            periodic_fsync_count=getattr(self.writer, "fsync_count", None),
            last_periodic_fsync_ns=getattr(self.writer, "last_fsync_ns", None),
            native_query_overrun_count=self.native_query_overrun_count,
            max_native_query_duration_ns=self.max_native_query_duration_ns,
        )
        # Snapshot before queueing: the writer must not race later mutations.
        # audit_closed is confirmed only by close() and the returned outcome.
        self._record("CLEANUP_OUTCOME", outcome=dict(self.cleanup))
        durable = self.writer.close(self.budgets.shutdown_s) if self.writer else False
        self.cleanup["audit_closed"] = durable
        if not durable:
            self.fail("audit_cleanup_unconfirmed")
        self.cleanup["NORMAL_CLEANUP"] &= durable and not bool(self.failure_latch)
        self.cleanup["failure_latch"] = list(self.failure_latch)
        self._health.update(PROCESS_ALIVE=not exited, STREAM_HEALTH=False,
                            FAILURE_LATCH=tuple(self.failure_latch))
        return dict(self.cleanup)
