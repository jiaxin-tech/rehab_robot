"""Durable attempt accounting for force-search and separate evaluation repeats.

Claiming happens before acquisition or motion. Failed or unfinished attempts
remain charged and close the study; no record is removed or automatically
retried. A study is rooted at its first prepared session, so sibling sessions
cannot obtain separate budgets by pointing at the same successful ancestor.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

NAME = "force_trial_ledger.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(config):
    declared = config["force_training"].get("study_identity", {})
    if not isinstance(declared, dict) or type(declared.get("strict", False)) is not bool:
        raise ValueError("force study_identity must declare a boolean strict mode")
    values = {key: declared.get(key) for key in ("setup_id", "mechanical_condition_id")}
    present = all(isinstance(value, str) and value.strip() for value in values.values())
    if declared.get("strict") and not present:
        raise ValueError("strict force study requires setup_id and mechanical_condition_id")
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in values.values()):
        raise ValueError("force setup/mechanical condition identities must be nonempty strings")
    return dict(**values, strict=declared.get("strict", False), identity_declared=present)


def root_for(directory, state=None):
    directory = Path(directory).resolve()
    state = read(directory / "round_status.json") if state is None else state
    if state.get("force_study_root"):
        return Path(state["force_study_root"]).resolve()
    seen = set()
    while state.get("parent_session"):
        if directory in seen:
            raise ValueError("cyclic force study history")
        seen.add(directory)
        directory = Path(state["parent_session"]).resolve()
        state = read(directory / "round_status.json")
    return directory


@contextmanager
def locked(root):
    lock = Path(root) / "force_trial_ledger.lock"
    try:
        stream = lock.open("x", encoding="ascii")
    except FileExistsError as exc:
        raise ValueError("force study ledger is in use or interrupted; inspect its records") from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _budgets(config):
    adaptive = config["force_training"]["total_budget"]
    evaluation = config["force_training"].get("evaluation_budget", adaptive)
    if any(type(value) is not int or value < 1 for value in (adaptive, evaluation)):
        raise ValueError("force adaptive/evaluation budgets must be positive integers")
    return dict(adaptive=adaptive, evaluation=evaluation)


def _configuration(config):
    return {key: config.get(key) for key in ("force_training", "robot", "motion", "empty_load", "trajectory", "frame")}


def _load(root, config):
    path = Path(root) / NAME
    if not path.is_file():
        return None
    ledger = read(path)
    if (ledger.get("schema_version") != 1 or ledger.get("study_root") != str(Path(root).resolve())
            or ledger.get("study_identity") != identity(config) or ledger.get("budgets") != _budgets(config)
            or ledger.get("study_configuration") != _configuration(config)):
        raise ValueError("force study identity or budgets changed")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or any(row.get("attempt_index") != i for i, row in enumerate(attempts, 1)):
        raise ValueError("force study attempt ledger is incomplete")
    if len({row.get("session") for row in attempts}) != len(attempts):
        raise ValueError("force study contains duplicate attempted sessions")
    return ledger


def _require_open(ledger):
    if any(row["status"] != "COMPLETE" for row in ledger["attempts"]):
        raise ValueError("force study has a failed or unfinished attempt; no automatic continuation")


def continuation(source, config, *, evaluation=False):
    """Check STOP and available budgets before preparing a child session."""
    source = Path(source).resolve()
    state = read(source / "round_status.json")
    purpose = state.get("trial_purpose", "adaptive")
    if purpose == "evaluation" and not evaluation:
        raise ValueError("evaluation recordings cannot become adaptive-search parents")
    root = root_for(source, state)
    ledger = _load(root, config)
    if ledger is not None:
        _require_open(ledger)
        selected = "evaluation" if evaluation else "adaptive"
        if sum(row["purpose"] == selected for row in ledger["attempts"]) >= ledger["budgets"][selected]:
            raise ValueError(selected.upper() + "_TOTAL_BUDGET_REACHED")
        if not evaluation:
            rows = [row for row in ledger["attempts"] if row["purpose"] == "adaptive"]
            if rows and rows[-1]["session"] != str(source):
                raise ValueError("adaptive parent is not the latest charged attempt; sibling retry refused")
    elif not evaluation:
        # Old captures remain available for offline analysis. They cannot seed
        # new hardware attempts while claiming an unobserved complete budget.
        raise ValueError("historical force recordings have no complete attempt ledger; start an explicit new study")
    if not evaluation:
        result = read(source / "tcp_force_training.json")
        if result["action"]["role"] == "STOP" or result["action"]["candidate_id"] is None:
            raise ValueError("force algorithm stopped: " + str(result["action"]["stop_reason"]))
    return root


def claim(directory, config, state):
    directory = Path(directory).resolve()
    root = root_for(directory, state)
    purpose = state.get("trial_purpose", "adaptive")
    if purpose not in ("adaptive", "evaluation"):
        raise ValueError("force trial_purpose must be adaptive or evaluation")
    with locked(root):
        ledger = _load(root, config)
        if ledger is None:
            if state.get("parent_session"):
                raise ValueError("cannot omit previous hardware attempts from the force study budget")
            ledger = dict(schema_version=1, study_root=str(root), study_identity=identity(config),
                          budgets=_budgets(config), study_configuration=_configuration(config), attempts=[])
        _require_open(ledger)
        if any(row["session"] == str(directory) for row in ledger["attempts"]):
            raise ValueError("this force session was already attempted; no replay")
        used = sum(row["purpose"] == purpose for row in ledger["attempts"])
        if used >= ledger["budgets"][purpose]:
            raise ValueError(purpose.upper() + "_TOTAL_BUDGET_REACHED")
        candidate = state["force_training_binding"]["candidate_id"]
        if purpose == "adaptive":
            prior = [row for row in ledger["attempts"] if row["purpose"] == "adaptive"]
            if prior:
                parent = Path(state.get("parent_session", "")).resolve()
                if str(parent) != prior[-1]["session"]:
                    raise ValueError("adaptive parent is not the latest charged attempt; sibling retry refused")
                result = read(parent / "tcp_force_training.json")
                if result["action"]["candidate_id"] is None:
                    raise ValueError("force algorithm stopped: " + str(result["action"]["stop_reason"]))
                if result["action"]["candidate_id"] != candidate:
                    raise ValueError("adaptive candidate differs from the preceding policy proposal; use explicit evaluation")
            elif candidate != config["force_training"]["reference_candidate_id"]:
                raise ValueError("force reference must be the first adaptive attempt")
            elif state.get("parent_session"):
                raise ValueError("historical parent is absent from the charged adaptive ledger")
        else:
            source = Path(state.get("evaluation_source", "")).resolve()
            source_state = read(source / "round_status.json")
            if source_state.get("force_training_binding", {}).get("candidate_id") != candidate:
                raise ValueError("evaluation must repeat its declared source candidate")
            if root_for(source, source_state) != root:
                raise ValueError("evaluation source belongs to another force study")
        attempt = dict(attempt_index=len(ledger["attempts"])+1, attempt_id=uuid4().hex,
                       session=str(directory), candidate_id=candidate, purpose=purpose, status="RUNNING",
                       started_utc=datetime.now(timezone.utc).isoformat())
        ledger["attempts"].append(attempt)
        write(root / NAME, ledger)
        ticket = dict(study_root=str(root), attempt_id=attempt["attempt_id"], attempt_index=attempt["attempt_index"])
        write(directory / "force_attempt.json", ticket)
        return ticket


def finish(directory, *, ok, error=None):
    directory = Path(directory).resolve()
    ticket_path = directory / "force_attempt.json"
    if not ticket_path.is_file():
        return
    ticket = read(ticket_path)
    root = Path(ticket["study_root"])
    with locked(root):
        ledger = read(root / NAME)
        row = ledger["attempts"][ticket["attempt_index"]-1]
        if row["attempt_id"] != ticket["attempt_id"] or row["session"] != str(directory) or row["status"] != "RUNNING":
            raise ValueError("force attempt completion identity or state mismatch")
        row.update(status="COMPLETE" if ok else "FAILED", error=error,
                   finished_utc=datetime.now(timezone.utc).isoformat())
        write(root / NAME, ledger)


def snapshot(directory, config, state):
    root = root_for(directory, state)
    ledger = _load(root, config)
    if ledger is None:
        return dict(complete=False, study_root=str(root), historical_completed_only=True,
                    study_identity=identity(config))
    if any(row["status"] == "FAILED" or (row["status"] == "RUNNING" and row["session"] != str(Path(directory).resolve()))
           for row in ledger["attempts"]):
        raise ValueError("force study has a failed or another unfinished attempt; training is closed")
    return dict(complete=True, study_root=str(root), historical_completed_only=False,
                study_identity=ledger["study_identity"], total_attempts=len(ledger["attempts"]),
                adaptive_attempts=sum(row["purpose"] == "adaptive" for row in ledger["attempts"]),
                evaluation_attempts=sum(row["purpose"] == "evaluation" for row in ledger["attempts"]),
                failed_attempts=sum(row["status"] == "FAILED" for row in ledger["attempts"]),
                unfinished_attempts=sum(row["status"] == "RUNNING" for row in ledger["attempts"]))
