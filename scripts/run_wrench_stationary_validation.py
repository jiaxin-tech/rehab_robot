"""Stationary-only production acquisition integration. Defaults to OFFLINE_DRY_RUN.

No motion executor is imported. Live admission binds a reviewed artifact to the
exact request and source hashes, operator and case. Parent native calls remain
in-process: a timed-out call is unresolved, never cancelled or retried.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from collection.episode_logger import EpisodeLogger
from collection.real_robot_acquisition import RealRobotAcquisition
from collection.state import KinematicStateFrame
from collection.wrench_process import DurableAudit, WrenchBudgets, WrenchProcessProvider

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/run_wrench_stationary_validation.py", "collection/real_robot_acquisition.py",
           "collection/wrench_process.py", "collection/episode_logger.py", "hardware/wrench_session.py",
           "hardware/rokae_adapter.py", "hardware/windows/rokae_xcore.py")
MODES = ("OFFLINE_DRY_RUN", "LIVE_STATIONARY")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_request(request, case):
    if request.get("schema_version") != 1 or not isinstance(request.get("request_id"), str):
        raise ValueError("versioned stationary request required; regenerate legacy preparation")
    uuid.UUID(request["request_id"])
    if case not in ("A", "B") or request.get("duration_s") != 900:
        raise ValueError("only A/B with frozen 900 s duration supported")
    if request.get("order") != ["A", "B"] or request.get("cases") != {
            "A": "DISABLED_FOR_STATIONARY_A", "B": "isolated_wrench"}:
        raise ValueError("case definition/order mismatch")
    for name in ("intended_wrench_hz", "state_poll_hz", "alignment_hz", "supervisor_hz", "rt_interval_ms"):
        if not positive(request.get(name)):
            raise ValueError("explicit positive rate/interval required: " + name)
    if request.get("retry") is not False or request.get("reference_release") != "NO_GO":
        raise ValueError("stationary request must preserve no retry and NO_GO")
    criteria = request.get("criteria") or {}
    if not isinstance(criteria, dict): raise ValueError("criteria must be an object or null")
    allowed = {"max_supervisor_interval_s", "max_rt_receive_interval_s", "max_delivery_latency_s"}
    if set(criteria) - allowed or any(v is not None and not positive(v) for v in criteria.values()):
        raise ValueError("unsupported or invalid explicit diagnostic criteria")


def authorize(request, case, authorization, acknowledgment, prior_a=None):
    """Runs BEFORE constructing any SDK adapter. This is evidence binding, not a signature service."""
    if not isinstance(authorization, dict):
        raise PermissionError("separately reviewed stationary authorization artifact required")
    a = authorization
    required = {"schema_version": 1, "scope": "LIVE_STATIONARY_NO_MOTION", "reviewed": True,
                "request_sha256": digest(request), "case": case, "source_hashes": source_hashes(),
                "previous_session_recovery_confirmed": True, "operator_present": True,
                "stop_procedure_reviewed": True, "parent_native_lifecycle_reviewed": True,
                "end_state_path": "existing_session_operationState"}
    if any(a.get(k) != v for k, v in required.items()):
        raise PermissionError("authorization/request/source/case/review binding mismatch")
    for key in ("authorization_id", "reviewer", "operator", "stop_procedure", "recovery_record",
                "setup_record", "vendor_session_record"):
        if not isinstance(a.get(key), str) or not a[key].strip():
            raise PermissionError("missing review evidence: " + key)
    if acknowledgment != a["authorization_id"] + ":" + a["operator"]:
        raise PermissionError("exact authorization-id:operator acknowledgement required")
    try:
        expiry = datetime.fromisoformat(a["expires_at"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
            raise ValueError("expired")
    except (ValueError, KeyError, TypeError) as exc:
        raise PermissionError("valid authorization expiry required") from exc
    try:
        budgets = WrenchBudgets(**(request.get("budgets") or {}))
    except (TypeError, ValueError) as exc:
        raise PermissionError("complete reviewed diagnostic budgets required") from exc
    if budgets.provenance == "NOT_SAFETY_THRESHOLDS":
        raise PermissionError("offline budgets forbidden for hardware")
    for key in ("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"):
        if not positive((request.get("lifecycle_budgets") or {}).get(key)):
            raise PermissionError("reviewed lifecycle budget required: " + key)
    if not request.get("expected_identity") or not request.get("connection") or not request.get("allowed_power_states"):
        raise PermissionError("current identity, connection and power observation policy required")
    if set(request["expected_identity"]) != {"robot_model", "robot_serial", "controller_version", "sdk_version"}:
        raise PermissionError("incomplete identity binding")
    if set(request["connection"]) != {"robot_ip", "local_ip", "robot_class"}:
        raise PermissionError("explicit connection fields required")
    if any(not isinstance(v, str) or not v for v in request["connection"].values()) or any(
            not isinstance(v, str) or not v for v in request["expected_identity"].values()):
        raise PermissionError("nonempty connection and identity strings required")
    if not isinstance(request["allowed_power_states"], list) or any(
            v not in ("on", "off") for v in request["allowed_power_states"]):
        raise PermissionError("explicit lowercase observed power states required")
    if not positive((request.get("criteria") or {}).get("max_supervisor_interval_s")):
        raise PermissionError("reviewed supervisor response criterion required before hardware")
    if case == "B":
        if (not prior_a or prior_a.get("case") != "A" or not prior_a.get("completed_900s")
                or prior_a.get("mode") != "LIVE_STATIONARY" or prior_a.get("failures")
                or prior_a.get("request_sha256") != digest(request)
                or prior_a.get("source_hashes") != source_hashes()
                or a.get("prior_a_summary_sha256") != digest(prior_a)):
            raise PermissionError("B requires reviewed successful A from same frozen request")


class OfflineStateAdapter:
    """State source stub only; all acquisition, health and logging stay production."""
    def __init__(self, *, final_state=True, state_failure=False):
        self.connected = False
        self.sequence = 0
        self.final_state = final_state
        self.state_failure = state_failure

    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def is_connected(self): return self.connected
    def start_state_stream(self): pass
    def stop_state_stream(self): pass

    def read_state_frame(self):
        self.sequence += 1
        now = time.perf_counter()
        return KinematicStateFrame(self.sequence, now, None, None, not self.state_failure,
            "injected_state_failure" if self.state_failure else "", (0., 0., 0.), (0., 0., 0.),
            None, None, "unavailable", (0.,) * 6, None, now, now, None, "IDLE", None, None)

    def final_operation_state(self):
        if not self.final_state:
            raise RuntimeError("offline_final_state_unavailable")
        return "IDLE"


class Evidence:
    """Reuse the bounded durable writer; never block supervisor on fsync."""
    def __init__(self, path, budget):
        self.writer = DurableAudit(path)
        self.pending = deque()
        self.budget = budget
        self.failure = None

    def put(self, row):
        try:
            self.pending.append((time.perf_counter(), self.writer.submit(row)))
        except Exception as exc:
            self.failure = f"telemetry_submit:{type(exc).__name__}:{exc}"

    def check(self):
        while self.pending and self.pending[0][1]["done"].is_set():
            _, ticket = self.pending.popleft()
            if ticket["error"]:
                self.failure = str(ticket["error"])
        if self.writer.failure:
            self.failure = str(self.writer.failure)
        if self.pending and time.perf_counter() - self.pending[0][0] > self.budget:
            self.failure = "telemetry_durability_deadline"
        return self.failure

    def close(self):
        closed = self.writer.close(self.budget)
        self.check()
        return closed and not self.failure


class ObservedAdapter:
    """Narrow whitelist facade: no motion/native handle delegated to callers."""
    def __init__(self, adapter, state_log, lifecycle, session_id, run_id, criteria):
        self.adapter, self.state_log, self.lifecycle = adapter, state_log, lifecycle
        self.session_id = session_id
        self.last_sequence = None
        self.previous_receive = None
        self.disconnect_confirmed = False
        self.connected_once = False
        self.run_id, self.criteria = run_id, criteria
        self.failure = None

    # Required ONLY for existing acquisition's live identity/config binding.
    @property
    def native_robot(self): return self.adapter.native_robot

    def connect(self):
        self.lifecycle("CONNECT_START", session_id=self.session_id)
        self.adapter.connect()
        self.connected_once = True
        self.lifecycle("CONNECT_COMPLETE", session_id=self.session_id)

    def disconnect(self):
        self.lifecycle("DISCONNECT_START", session_id=self.session_id)
        self.adapter.disconnect()
        self.disconnect_confirmed = True
        self.lifecycle("DISCONNECT_COMPLETE", session_id=self.session_id)

    def is_connected(self): return self.adapter.is_connected()
    def start_state_stream(self): self.adapter.start_state_stream()
    def stop_state_stream(self): self.adapter.stop_state_stream()
    def read_robot_metadata(self): return self.adapter.read_robot_metadata()

    def read_state_frame(self):
        frame = self.adapter.read_state_frame()
        delivered = time.perf_counter()
        if frame.sequence_id != self.last_sequence:
            receive = frame.host_monotonic_time_s
            interval = receive - self.previous_receive if receive is not None and self.previous_receive is not None else None
            latency = delivered - receive if receive is not None else None
            for key, value in (("max_rt_receive_interval_s", interval), ("max_delivery_latency_s", latency)):
                if value is not None and self.criteria.get(key) is not None and value > self.criteria[key]:
                    self.failure = key + "_exceeded"
            self.state_log.put(dict(asdict(frame), delivery_s=delivered, run_id=self.run_id,
                session_id=self.session_id, pid=os.getpid(),
                source_timestamp_s=frame.robot_device_time_s, source_clock="device_unavailable" if frame.robot_device_time_s is None else "device",
                receive_timestamp_s=receive, ipc_latency_s=None,
                delivery_latency_s=latency, host_receive_interval_s=interval,
                sequence_gap=max(0, frame.sequence_id - self.last_sequence - 1) if self.last_sequence is not None else 0))
            self.last_sequence, self.previous_receive = frame.sequence_id, receive
        return frame


def bounded_call(fn, seconds):
    """Does NOT cancel native calls or guarantee scheduling under parent GIL blockage."""
    result = {}
    def work():
        try: result["value"] = fn()
        except Exception as exc: result["error"] = f"{type(exc).__name__}:{exc}"
    thread = threading.Thread(target=work, daemon=True, name="stationary-lifecycle")
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        result["error"] = "UNRESOLVED_PARENT_CALL_DESIGN_REVIEW_REQUIRED"
    return dict(result), thread.is_alive()


def rows(path, failures=None):
    if not Path(path).exists(): return []
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if failures is None: raise
            failures.append(f"telemetry_corrupt:{Path(path).name}:{number}")
    return result


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return dict(n=len(values), minimum=values[0] if values else None,
                maximum=values[-1] if values else None,
                mean=sum(values) / len(values) if values else None,
                p99=values[max(0, math.ceil(.99 * len(values)) - 1)] if values else None)


def csv_export(path, records, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def run_case(request, case, output_root, *, mode="OFFLINE_DRY_RUN", authorization=None,
             acknowledgment=None, prior_a=None, offline_seconds=1., behavior="normal",
             final_state=True, state_failure=False):
    validate_request(request, case)
    if mode not in MODES: raise ValueError("unknown mode")
    live = mode == "LIVE_STATIONARY"
    if live:
        authorize(request, case, authorization, acknowledgment, prior_a)
        if behavior != "normal" or not final_state or state_failure:
            raise PermissionError("fault injection forbidden on hardware")
        budgets = WrenchBudgets(**request["budgets"])
        lifecycle_budgets = request["lifecycle_budgets"]
    else:
        if not positive(offline_seconds): raise ValueError("positive dry-run wall duration required")
        budgets = WrenchBudgets(startup_s=5., query_s=.5, progress_s=2., sample_age_s=2.,
            ack_s=2., log_s=2., shutdown_s=.5, terminate_s=.5, state_age_s=2., skew_s=2.,
            provenance="NOT_SAFETY_THRESHOLDS")
        lifecycle_budgets = dict.fromkeys(("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"), 5.)
    run_id, parent_session = str(uuid.uuid4()), str(uuid.uuid4())
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    if live:
        # One artifact = one attempted session. Kept even after failure; no auto
        # re-use through another output directory or replacement run.
        claims = ROOT / "results/wrench_stationary_validation_v2/authorization_claims"
        claims.mkdir(parents=True, exist_ok=True)
        claim = hashlib.sha256(authorization["authorization_id"].encode()).hexdigest()
        with (claims / (claim + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(dict(authorization_id=authorization["authorization_id"], run_id=run_id,
                           request_sha256=digest(request), case=case), stream)
            stream.flush()
            os.fsync(stream.fileno())
    snapshot = dict(schema_version=1, request=request, request_sha256=digest(request), case=case,
        mode=mode, run_id=run_id, parent_session_id=parent_session, parent_pid=os.getpid(),
        source_hashes=source_hashes(), authorization=authorization, budgets=asdict(budgets),
        lifecycle_budgets=lifecycle_budgets, offline_wall_duration_s=None if live else offline_seconds,
        timestamp_semantics="host perf_counter; no invented device timestamps",
        reference_release="NO_GO")
    write_json(out / "protocol_snapshot.json", snapshot)
    life = Evidence(out / "process_lifecycle.jsonl", budgets.log_s)
    health_log = Evidence(out / "health_timeline.jsonl", budgets.log_s)
    state_log = Evidence(out / "rt_state_events.jsonl", budgets.log_s)
    event_id = 0
    event_lock = threading.Lock()
    def event(kind, **extra):
        nonlocal event_id
        with event_lock:
            event_id += 1
            life.put(dict(type=kind, event_id=event_id, run_id=run_id, origin="runner",
                session_id=parent_session, pid=os.getpid(), host_ns=time.perf_counter_ns(), **extra))
    # Avoid duplicate session_id in keyword construction.
    def lifecycle(kind, **extra):
        extra.pop("session_id", None)
        event(kind, **extra)
    if live:
        # The only branch that can construct a hardware adapter, after admission.
        from hardware.rokae_adapter import RokaeRobotAdapter
        conn = request["connection"]
        raw_adapter = RokaeRobotAdapter(conn["robot_ip"], local_ip=conn["local_ip"],
            robot_class=conn["robot_class"], state_interval_ms=request["rt_interval_ms"])
    else:
        raw_adapter = OfflineStateAdapter(final_state=final_state, state_failure=state_failure)
    adapter = ObservedAdapter(raw_adapter, state_log, lifecycle, parent_session, run_id, request.get("criteria") or {})
    provider = None
    if case == "B":
        config = dict(mode="live" if live else "offline", rate_hz=request["intended_wrench_hz"], connection_audit=True)
        if live:
            config.update(request["connection"], expected_identity=request["expected_identity"],
                          allowed_power_states=request["allowed_power_states"])
        else:
            config.update(behavior=behavior, delay_s=.1, good_queries=1 if behavior == "error263" else 0)
        provider = WrenchProcessProvider(config, budgets, out / "wrench_query_events.jsonl")
        provider.run_id = run_id
        event("CHILD_SESSION_ASSIGNED", child_session_id=provider.session_id, configuration=config)
    else:
        (out / "wrench_query_events.jsonl").touch()
        event("DISABLED_FOR_STATIONARY_A", wrench_health="N/A_INTENTIONALLY_DISABLED")
    logger = EpisodeLogger(out / "episode", metadata=snapshot)
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider,
        diagnostic_state_only=case == "A", wrench_hz=request["intended_wrench_hz"],
        state_poll_hz=request["state_poll_hz"], alignment_hz=request["alignment_hz"])
    failures = []
    unresolved = False
    started = None
    actual = 0.
    first_detection = None
    end_state = dict(final_operation_state=None, reason="not_attempted", source=None)
    cleanup = {}
    try:
        logger.start()
        result, unresolved = bounded_call(adapter.connect, lifecycle_budgets["connect_s"])
        if result.get("error"): raise RuntimeError(result["error"])
        if live:
            def verify_site():
                actual_identity = raw_adapter.read_robot_metadata()
                aliases = {"robot_serial": "robot_serial_number"}
                for key, value in request["expected_identity"].items():
                    if str(actual_identity.get(aliases.get(key, key))) != value:
                        raise PermissionError("parent_identity_mismatch:" + key)
                native = raw_adapter.native_robot
                if native._rt_controller is not None:
                    raise PermissionError("unexpected_motion_controller")
                if native.get_robot_mode().upper() != "IDLE":
                    raise PermissionError("stationary_idle_required")
                power = native._call("powerState", native._robot.powerState).name.lower()
                if power not in request["allowed_power_states"]:
                    raise PermissionError("power_observation_mismatch")
                event("PARENT_IDENTITY_CONFIRMED", identity=actual_identity, power_state=power)
            result, unresolved = bounded_call(verify_site, lifecycle_budgets["connect_s"])
            if result.get("error"): raise RuntimeError(result["error"])
        result, unresolved = bounded_call(lambda: acquisition.start(manage_connection=False), lifecycle_budgets["connect_s"])
        if result.get("error"): raise RuntimeError(result["error"])
        started = time.perf_counter()
        previous = started
        duration = 900. if live else offline_seconds
        while time.perf_counter() - started < duration:
            now = time.perf_counter()
            health = acquisition.latest_health()
            ph = provider.health() if provider else {}
            reason = list(ph.get("FAILURE_LATCH", ()))
            state = acquisition.latest_state_frame()
            if state is not None and (not health.state_valid or health.state_age_s is None or health.state_age_s > budgets.state_age_s):
                reason.append("state_unhealthy")
            if state is None and now - started > budgets.startup_s: reason.append("state_startup_timeout")
            if acquisition.background_error: reason.append(acquisition.background_error)
            if not health.state_thread_alive or not health.alignment_thread_alive:
                reason.append("acquisition_producer_exited")
            if case == "B" and not health.wrench_thread_alive:
                reason.append("wrench_consumer_exited")
            if adapter.failure: reason.append(adapter.failure)
            supervisor_limit = (request.get("criteria") or {}).get("max_supervisor_interval_s")
            if supervisor_limit is not None and now - previous > supervisor_limit:
                reason.append("supervisor_interval_exceeded")
            if not logger.healthy_signal: reason.append("episode_logger_failure")
            for audit in (life, health_log, state_log):
                audit_failure = audit.check()
                if audit_failure: reason.append(audit_failure)
            if reason and first_detection is None: first_detection = time.perf_counter_ns()
            published = time.perf_counter_ns() if reason else None
            health_log.put(dict(run_id=run_id, tick_ns=time.perf_counter_ns(), loop_interval_s=now - previous,
                **asdict(health), provider=ph, query_elapsed_s=(time.perf_counter_ns() - ph["QUERY_START_TIME"]) / 1e9
                if ph.get("QUERY_IN_PROGRESS") and ph.get("QUERY_START_TIME") else None,
                wrench_status="N/A_INTENTIONALLY_DISABLED" if case == "A" else "ENABLED",
                reasons=reason, first_detection_ns=first_detection, published_failure_ns=published))
            previous = now
            if reason:
                failures.extend(reason)
                break
            time.sleep(1. / request["supervisor_hz"])
        actual = time.perf_counter() - started
    except Exception as exc:
        failures.append(f"runner:{type(exc).__name__}:{exc}")
    finally:
        if started is not None and not actual: actual = time.perf_counter() - started
        event("STOP_REQUEST", failures=failures)
        if not unresolved:
            result, unresolved = bounded_call(acquisition.stop, lifecycle_budgets["stop_s"])
            stop_confirmed = not unresolved and not result.get("error")
            if result.get("error"): failures.append("stop:" + result["error"])
            # Only existing session, after state producers have stopped. No reconnect.
            if not unresolved and not result.get("error") and adapter.is_connected():
                event("FINAL_OPERATION_STATE_QUERY")
                fn = raw_adapter.native_robot.get_robot_mode if live else raw_adapter.final_operation_state
                end_result, unresolved = bounded_call(fn, lifecycle_budgets["end_state_s"])
                end_state = dict(final_operation_state=end_result.get("value"), reason=end_result.get("error"),
                                 source="existing_session_operationState" if live else "offline_stub",
                                 query_end_ns=None if unresolved else time.perf_counter_ns())
            else:
                end_state["reason"] = "existing_session_unavailable_or_stop_unconfirmed"
            if stop_confirmed and not unresolved and adapter.is_connected():
                result, unresolved = bounded_call(adapter.disconnect, lifecycle_budgets["disconnect_s"])
                if result.get("error"): failures.append("disconnect:" + result["error"])
        else:
            end_state["reason"] = "parent_call_unresolved_no_concurrent_query_or_disconnect"
        if end_state["final_operation_state"] is None: failures.append("final_operation_state_missing")
        elif end_state["final_operation_state"].upper() != "IDLE": failures.append("final_operation_state_not_idle")
        child_cleanup = provider.cleanup if provider else None
        if provider and not child_cleanup: failures.append("child_cleanup_unconfirmed")
        if child_cleanup and not child_cleanup["NORMAL_CLEANUP"]: failures.append("child_cleanup_failed")
        if unresolved: failures.append("unresolved_parent_native_call")
        result, logger_unresolved = bounded_call(lambda: logger.close(completed=not failures and live and actual >= 900,
            stop_reason=";".join(failures) or (None if live else "offline_short_run_not_900s")), lifecycle_budgets["logger_s"])
        if result.get("error"): failures.append("logger_close:" + result["error"])
        cleanup = dict(parent=dict(normal_cleanup=adapter.disconnect_confirmed and not unresolved,
            graceful_sdk_disconnect=adapter.disconnect_confirmed if live else None,
            offline_disconnect_completed=adapter.disconnect_confirmed if not live else None,
            unresolved_call=unresolved, process_exitcode=None, process_exit="runner_still_running",
            forced_termination=False, controller_session_recovery=None), child=child_cleanup,
            logger_closed=not logger_unresolved and not result.get("error"))
        event("CLEANUP_OUTCOME", outcome=cleanup)
        for name, audit in (("state", state_log), ("health", health_log), ("lifecycle", life)):
            closed = audit.close()
            cleanup[name + "_audit_closed"] = closed
            if not closed: failures.append(name + "_audit_unconfirmed")
    # Derive exports only from persisted evidence, never from overwritten latest slots.
    state_rows, health_rows, queries = (rows(out / name, failures) for name in
        ("rt_state_events.jsonl", "health_timeline.jsonl", "wrench_query_events.jsonl"))
    csv_export(out / "rt_state.csv", state_rows, list(state_rows[0]) if state_rows else ["sequence_id", "source_timestamp_s", "receive_timestamp_s", "delivery_s"])
    success = [r for r in queries if r.get("type") == "QUERY_SUCCESS"]
    errors = [r for r in queries if r.get("type") == "QUERY_ERROR"]
    samples = [dict(r["frame"], run_id=run_id, session_id=r["session_id"], query_id=r["query_id"],
                    parent_receive_ns=r.get("parent_receive_ns")) for r in success]
    csv_export(out / "wrench_samples.csv", samples, list(samples[0]) if samples else ["sequence_id", "host_monotonic_time_s", "cartesian_force_raw_n", "cartesian_torque_raw_nm"])
    intents = {r["query_id"] for r in queries if r.get("type") == "QUERY_START_INTENT"}
    completed = {r["query_id"] for r in success + errors}
    if case == "B" and not success: failures.append("no_successful_wrench_sample")
    if not state_rows: failures.append("no_state_samples")
    if any(r.get("sequence_gap", 0) for r in state_rows): failures.append("rt_capture_sequence_gap")
    if not cleanup["parent"]["normal_cleanup"]: failures.append("parent_cleanup_unconfirmed")
    # Query audit remains authoritative; selected child lifecycle events are also
    # indexed in the common lifecycle stream without changing origin/IDs.
    lifecycle_types = {"PROCESS_START", "PROCESS_READY", "CONNECT_START", "CONNECT_COMPLETE", "DISCONNECT_START", "DISCONNECT_COMPLETE",
                       "PROCESS_STOPPED", "PROCESS_EXIT", "FORCED_WORKER_TERMINATION", "CLEANUP_OUTCOME"}
    if cleanup["child"] and cleanup["child"].get("audit_closed") and cleanup["lifecycle_audit_closed"]:
        with (out / "process_lifecycle.jsonl").open("a", encoding="utf-8") as stream:
            for row in queries:
                if row.get("type") in lifecycle_types:
                    stream.write(json.dumps(dict(row, source_stream="wrench_query_events.jsonl")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    metrics = {name: distribution(r.get(field) for r in health_rows) for name, field in (
        ("state_age_s", "state_age_s"), ("wrench_age_s", "wrench_age_s"),
        ("state_wrench_skew_s", "state_wrench_skew_s"), ("supervisor_interval_s", "loop_interval_s"))}
    metrics.update(rt_host_receive_interval_s=distribution(r.get("host_receive_interval_s") for r in state_rows),
        rt_delivery_latency_s=distribution(r.get("delivery_latency_s") for r in state_rows),
        query_latency_s=distribution((r["completion_ns"] - r["native_start_ns"]) / 1e9 for r in success + errors))
    summary = dict(schema_version=1, run_id=run_id, request_sha256=digest(request), case=case, mode=mode,
        source_hashes=snapshot["source_hashes"],
        requested_duration_s=900, actual_duration_s=actual, completed_900s=bool(live and actual >= 900 and not failures),
        offline_dry_run_pass=not failures if not live else None,
        intended_rt_rate_hz=1000 / request["rt_interval_ms"], intended_state_poll_hz=request["state_poll_hz"],
        intended_wrench_rate_hz=request["intended_wrench_hz"], wrench_status="N/A_INTENTIONALLY_DISABLED" if case == "A" else "ENABLED",
        rt_sample_count=len(state_rows), wrench_planned_queries=0 if case == "A" else math.floor(900 * request["intended_wrench_hz"]),
        wrench_started_queries=len(intents), wrench_completed_queries=len(completed),
        query_count_semantics="started=start intent; pre-call boundary does not prove native entry",
        wrench_successful_queries=len(success), wrench_error_queries=len(errors), pending_queries=sorted(intents - completed),
        sdk_error_histogram=dict(Counter(str(r["error"].get("code")) for r in errors)), distributions=metrics,
        rt_device_source_timing="UNAVAILABLE_NO_DEVICE_TIMESTAMP", criteria=request.get("criteria"),
        timing_acceptance="UNDEFINED", reliability_acceptance="UNDEFINED", cleanup=cleanup,
        final_robot_operation_state=end_state["final_operation_state"],
        logger_transport_gaps=dict(provider=provider.gaps if provider else [],
            rt_sequence_skips=sum(r.get("sequence_gap", 0) for r in state_rows),
            telemetry_failures=[f for f in failures if "audit" in f or "telemetry" in f or "logger" in f]),
        failures=list(dict.fromkeys(failures)), robot_connected=live and adapter.connected_once,
        robot_motion_executed=False, reference_release="NO_GO", r9_closed=False)
    write_json(out / "cleanup_outcome.json", cleanup)
    write_json(out / "end_state.json", end_state)
    write_json(out / "case_summary.json", summary)
    write_json(out / "sha256_index.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    return out, summary


def compare(a, b):
    if (a.get("case") != "A" or b.get("case") != "B" or a.get("request_sha256") != b.get("request_sha256")
            or a.get("mode") != b.get("mode") or not a.get("source_hashes")
            or a.get("source_hashes") != b.get("source_hashes")):
        raise ValueError("A/B must use same frozen request and mode")
    return dict(status="DESCRIPTIVE_ONLY" if a["completed_900s"] and b["completed_900s"] else "INCOMPLETE",
        acceptance="UNDEFINED", causality="NOT_ESTABLISHED", rt_device_source_timing="UNAVAILABLE",
        cases={"A": a["run_id"], "B": b["run_id"]},
        differences={key: {"A": a["distributions"][key], "B": b["distributions"][key],
            "p99_delta": b["distributions"][key]["p99"] - a["distributions"][key]["p99"]
            if a["distributions"][key]["p99"] is not None and b["distributions"][key]["p99"] is not None else None}
            for key in ("rt_host_receive_interval_s", "rt_delivery_latency_s", "state_age_s", "supervisor_interval_s")},
        process_stability={"A": a["failures"], "B": b["failures"]}, cleanup={"A": a["cleanup"], "B": b["cleanup"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("A", "B"))
    parser.add_argument("--mode", choices=MODES, default="OFFLINE_DRY_RUN")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/wrench_stationary_validation_v2")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--operator-ack")
    parser.add_argument("--prior-a-summary", type=Path)
    parser.add_argument("--offline-seconds", type=float, default=1.)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8")) if path else None
    out, summary = run_case(load(args.request), args.case, args.output_root, mode=args.mode,
        authorization=load(args.authorization), acknowledgment=args.operator_ack,
        prior_a=load(args.prior_a_summary), offline_seconds=args.offline_seconds)
    if args.case == "B" and args.prior_a_summary:
        write_json(out / "ab_comparison.json", compare(load(args.prior_a_summary), summary))
        write_json(out / "sha256_index.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir() if p.is_file() and p.name != "sha256_index.json"})
    print(json.dumps({"output": str(out), "failures": summary["failures"], "completed_900s": summary["completed_900s"]}))
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
