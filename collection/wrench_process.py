"""Spawn-owned wrench queries with acknowledged, durable audit events.

No native import occurs in the supervisor. Budgets have no production defaults.
The audit mailbox is stop-and-wait: its writer cannot overwrite an unacknowledged
record. Fixed-size shared byte slots avoid multiprocessing Queue feeder/join locks
that can be corrupted when a native owner is terminated.
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
    """Bounded asynchronous fsync. An ACK is issued only after done is set.

    A wedged filesystem writer is a daemon with explicit unresolved status;
    supervision never waits on its I/O lock and never spawns replacement writers.
    """
    def __init__(self, path, *, write_hook=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.work = queue.Queue(maxsize=64)
        self.failure = None
        self.closed = False
        self.write_hook = write_hook
        self.thread = threading.Thread(target=self._run, daemon=True, name="wrench-audit-writer")
        self.thread.start()

    def submit(self, event):
        if self.failure or self.closed:
            raise RuntimeError(self.failure or "audit_closed")
        ticket = {"done": threading.Event(), "error": None, "durable_ns": None}
        self.work.put_nowait((dict(event), ticket))
        return ticket

    def _run(self):
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                while True:
                    item = self.work.get()
                    if item is None:
                        return
                    event, ticket = item
                    try:
                        if self.write_hook:
                            self.write_hook(event)
                        # This field marks write dispatch, NOT fsync completion.
                        event["log_write_ns"] = time.perf_counter_ns()
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        ticket["durable_ns"] = time.perf_counter_ns()
                    except BaseException as exc:
                        ticket["error"] = self.failure = f"audit_write:{type(exc).__name__}:{exc}"
                        return
                    finally:
                        ticket["done"].set()
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
            intent = emit("QUERY_START_INTENT", completion=None,
                          scheduled_dispatch_ns=int(next_tick * 1e9),
                          dispatch_lateness_ns=max(0, time.perf_counter_ns() - int(next_tick * 1e9)))
            if stop.value:
                break
            # This is a confirmed PRE-CALL boundary, not proof that the CPU
            # entered vendor code. The following ACK gap remains explicit.
            boundary = emit("QUERY_NATIVE_ENTER", entry_semantics="PRE_CALL_BOUNDARY")
            if stop.value:
                break
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
                 intent_ns=intent["timestamp_ns"], boundary_ns=boundary["timestamp_ns"],
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
            required = ("robot_ip", "local_ip", "robot_class", "expected_identity", "allowed_power_states")
            if any(not self.config.get(k) for k in required) or budgets.provenance == "NOT_SAFETY_THRESHOLDS":
                raise ValueError("live identity/connection/reviewed budgets required")
            if set(self.config["expected_identity"]) != {"robot_model", "robot_serial", "controller_version", "sdk_version"}:
                raise ValueError("complete identity binding required")
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
        self.pending_ticket = None
        self.pending_event = None
        self.pending_since = None
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
        self._health = {}
        self._poll_lock = threading.Lock()

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
        return self

    def _record(self, kind, **fields):
        self.supervisor_event_id += 1
        event = dict(version=VERSION, origin="supervisor", run_id=self.run_id,
                     session_id=self.session_id, pid=os.getpid(), query_id=self.query_id,
                     event_id=self.supervisor_event_id, type=kind,
                     timestamp_ns=time.perf_counter_ns(), **fields)
        self.events.append(event)
        if self.writer:
            try:
                self.writer.submit(event)
            except Exception:
                if "telemetry_failure" not in self.failure_latch:
                    self.failure_latch.append("telemetry_failure")
                self.stop_flag.value = 1

    def fail(self, reason):
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
        if kind == "QUERY_START_INTENT":
            if qid != self.query_id + 1 or self.query_state not in ("IDLE", "SUCCEEDED") or self.failure_latch:
                raise ValueError("duplicate_query_state")
        elif kind == "QUERY_NATIVE_ENTER":
            if qid != self.query_id or self.query_state != "START_INTENT":
                raise ValueError("out_of_order_native_enter")
        elif kind in ("QUERY_SUCCESS", "QUERY_ERROR"):
            if qid != self.query_id or self.query_state not in ("IN_PROGRESS", "HUNG"):
                raise ValueError("out_of_order_completion")
            if not self.query_start_ns <= event["native_start_ns"] <= event["completion_ns"] <= stamp:
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
        event = dict(event, parent_receive_ns=now)
        self.pending_ticket = self.writer.submit(event)
        self.pending_event = event
        self.pending_since = now

    def _apply(self, event, durable_ns):
        self.events.append(dict(event, durable_ns=durable_ns))
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
        elif kind == "QUERY_START_INTENT":
            self.query_id = event["query_id"]
            self.query_start_ns = event["timestamp_ns"]
            self.native_boundary_ns = None
            self.query_state = "START_INTENT"
            self._record("QUERY_PENDING", completion=None, entry_unconfirmed=True)
        elif kind == "QUERY_NATIVE_ENTER":
            self.native_boundary_ns = event["timestamp_ns"]
            self.query_state = "IN_PROGRESS"
        elif kind in ("QUERY_SUCCESS", "QUERY_ERROR"):
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
        # Persist fsync-completion provenance in a subsequent record. The ACK
        # timestamp is never mislabelled as the original query timestamp.
        self._record("DURABLE_ACK", child_event_id=event["event_id"], durable_ns=durable_ns)
        self.ack.value = event["event_id"]
        self.expected_event += 1

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
            if self.pending_ticket:
                if self.audit_slot.version.value != self.seen_slot_version:
                    raise ValueError("unacknowledged_mailbox_overwrite")
                ticket = self.pending_ticket
                if ticket["done"].is_set():
                    if ticket["error"]:
                        self.fail("logger_failure:" + ticket["error"])
                    else:
                        self._apply(self.pending_event, ticket["durable_ns"])
                    self.pending_ticket = None
                    self.pending_event = None
                elif (now - self.pending_since) / 1e9 > self.budgets.log_s:
                    self.fail("logger_backpressure")
            else:
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
        sample_time = self.last_good.host_monotonic_time_s if self.last_good else None
        age = None if sample_time is None else now / 1e9 - sample_time
        progress_age = None if self.progress_ns is None else (now - self.progress_ns) / 1e9
        if not self.stopping:
            if not self.progress_ns and self.started_ns and (now - self.started_ns) / 1e9 > self.budgets.startup_s:
                self.fail("missing_first_heartbeat")
            if self.query_state in ("STARTING", "NOT_STARTED"):
                if self.started_ns and (now - self.started_ns) / 1e9 > self.budgets.startup_s:
                    self.fail("startup_timeout")
            elif progress_age is not None and progress_age > self.budgets.progress_s:
                self.fail("worker_progress_missing")
            if pending and (now - self.query_start_ns) / 1e9 > self.budgets.query_s:
                if self.query_state != "HUNG":
                    self.query_state = "HUNG"
                    self._record("QUERY_HUNG", completion=None)
                self.fail("query_pending_too_long")
            if age is not None and (age < 0 or age > self.budgets.sample_age_s):
                self.fail("wrench_sample_stale_or_clock_invalid")
        if self.process and not alive and not self.exit_seen:
            self.exit_seen = True
            self._record("PROCESS_EXIT", exitcode=self.process.exitcode, completion=None if pending else self.last_completed_ns)
            if not self.stopping:
                self.fail("worker_exited")
        self._health = dict(LATEST_GOOD_SAMPLE_VALID=bool(self.last_good and self.last_good.valid),
            CURRENT_QUERY_STATE=self.query_state, LAST_ERROR=self.last_error,
            QUERY_IN_PROGRESS=pending, QUERY_START_TIME=self.query_start_ns,
            LAST_COMPLETED_QUERY_TIME=self.last_completed_ns, WRENCH_SAMPLE_AGE=age,
            WORKER_HEARTBEAT_AGE=progress_age, PROCESS_ALIVE=alive,
            STREAM_HEALTH=bool(alive and self.last_good and not self.failure_latch and not self.stopping),
            FAILURE_LATCH=tuple(self.failure_latch), entry_unconfirmed=pending,
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
        exited = bool(self.process and not self.process.is_alive())
        if exited and not self.exit_seen:
            self.exit_seen = True
            self._record("PROCESS_EXIT", exitcode=self.process.exitcode,
                         completion=None if self.query_state in ("HUNG", "IN_PROGRESS", "START_INTENT") else self.last_completed_ns)
        self.cleanup = dict(NORMAL_CLEANUP=exited and not forced,
                            GRACEFUL_SDK_DISCONNECT=False if forced else self.disconnect_confirmed,
                            OFFLINE_DISCONNECT_COMPLETED=self.offline_disconnect_completed,
                            FORCED_WORKER_TERMINATION=forced, CONTROLLER_SESSION_RECOVERY=None,
                            process_exited=exited, exitcode=self.process.exitcode if self.process else None,
                            query_state=self.query_state, failure_latch=list(self.failure_latch))
        self._record("CLEANUP_OUTCOME", outcome=self.cleanup)
        durable = self.writer.close(self.budgets.log_s) if self.writer else False
        self.cleanup["audit_closed"] = durable
        if not durable:
            self.fail("audit_cleanup_unconfirmed")
        self.cleanup["NORMAL_CLEANUP"] &= durable and not bool(self.failure_latch)
        self._health.update(PROCESS_ALIVE=not exited, STREAM_HEALTH=False,
                            FAILURE_LATCH=tuple(self.failure_latch))
        return dict(self.cleanup)
