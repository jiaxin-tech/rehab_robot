"""Attempt accounting is tested without constructing any hardware object."""
from copy import deepcopy
from pathlib import Path

import pytest

from measurement_validation import force_trial_ledger as ledger


def config(*, budget=3, strict=False):
    return {"force_training": {"total_budget": budget, "evaluation_budget": 2,
        "reference_candidate_id": "reference", "study_identity": {
            "setup_id": "rig_a", "mechanical_condition_id": "condition_a", "strict": strict}}}


def session(path, *, parent=None, candidate="reference", purpose="adaptive", source=None, root=None):
    path.mkdir()
    state = dict(status="PREPARED", force_training_binding={"candidate_id": candidate}, trial_purpose=purpose)
    if parent is not None: state["parent_session"] = str(parent)
    if source is not None: state["evaluation_source"] = str(source)
    if root is not None: state["force_study_root"] = str(root)
    ledger.write(path / "round_status.json", state)
    return state


def success(path, *, action="candidate_1"):
    ledger.finish(path, ok=True)
    state = ledger.read(path / "round_status.json")
    state.update(status="CAPTURE_COMPLETE", training_status="COMPLETE")
    ledger.write(path / "round_status.json", state)
    ledger.write(path / "tcp_force_training.json", {"action": {"candidate_id": action,
        "role": "STOP" if action is None else "TCP_SEARCH", "stop_reason": "STOP_TEST" if action is None else None}})


def test_failed_and_unfinished_attempts_close_same_ancestor_budget(tmp_path):
    c = config()
    root = tmp_path / "root"
    state = session(root)
    ticket = ledger.claim(root, c, state)
    assert ticket["attempt_index"] == 1
    sibling = tmp_path / "sibling"
    sibling_state = session(sibling, parent=root, candidate="candidate_1")
    with pytest.raises(ValueError, match="unfinished"):
        ledger.claim(sibling, c, sibling_state)
    ledger.finish(root, ok=False, error="synthetic capture failed")
    with pytest.raises(ValueError, match="failed or unfinished"):
        ledger.claim(sibling, c, sibling_state)
    records = ledger.read(root / ledger.NAME)["attempts"]
    assert len(records) == 1 and records[0]["status"] == "FAILED"


def test_successful_sibling_cannot_retry_from_old_ancestor(tmp_path):
    c = config()
    root = tmp_path / "root"
    ledger.claim(root, c, session(root))
    success(root)
    first = tmp_path / "child_a"
    ledger.claim(first, c, session(first, parent=root, candidate="candidate_1"))
    success(first, action="candidate_2")
    second = tmp_path / "child_b"
    with pytest.raises(ValueError, match="latest charged"):
        ledger.claim(second, c, session(second, parent=root, candidate="candidate_1"))
    with pytest.raises(ValueError, match="latest charged"):
        ledger.continuation(root, c)
    assert len(ledger.read(root / ledger.NAME)["attempts"]) == 2


def test_budget_blocks_before_next_attempt_is_recorded(tmp_path):
    c = config(budget=1)
    root = tmp_path / "root"
    ledger.claim(root, c, session(root))
    success(root)
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="TOTAL_BUDGET_REACHED"):
        ledger.claim(child, c, session(child, parent=root, candidate="candidate_1"))
    assert len(ledger.read(root / ledger.NAME)["attempts"]) == 1


def test_stop_allows_explicit_evaluation_with_independent_accounting(tmp_path):
    c = config(budget=1)
    root = tmp_path / "root"
    ledger.claim(root, c, session(root))
    success(root, action=None)
    evaluation_root = ledger.continuation(root, c, evaluation=True)
    child = tmp_path / "evaluation"
    state = session(child, purpose="evaluation", source=root, root=evaluation_root)
    ledger.claim(child, c, state)
    ledger.finish(child, ok=True)
    counts = ledger.snapshot(child, c, state)
    assert counts["total_attempts"] == 2 and counts["adaptive_attempts"] == counts["evaluation_attempts"] == 1
    with pytest.raises(ValueError, match="evaluation recordings cannot"):
        ledger.continuation(child, c)


@pytest.mark.parametrize("change", ["mechanical", "setup", "configuration"])
def test_changed_physical_identity_or_protocol_refuses_before_claim(tmp_path, change):
    c = config(strict=True)
    root = tmp_path / "root"
    ledger.claim(root, c, session(root))
    success(root)
    changed = deepcopy(c)
    if change == "mechanical": changed["force_training"]["study_identity"]["mechanical_condition_id"] = "other"
    if change == "setup": changed["force_training"]["study_identity"]["setup_id"] = "other"
    if change == "configuration": changed["motion"] = {"different": True}
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="identity or budgets changed"):
        ledger.claim(child, changed, session(child, parent=root, candidate="candidate_1"))


def test_strict_identity_requires_explicit_setup_and_condition():
    c = config(strict=True)
    c["force_training"]["study_identity"]["mechanical_condition_id"] = None
    with pytest.raises(ValueError, match="mechanical_condition_id"):
        ledger.identity(c)
    assert not ledger.identity({"force_training": {}})["identity_declared"]


def test_history_without_attempt_evidence_cannot_seed_new_adaptive_run(tmp_path):
    root = tmp_path / "historical"
    session(root)
    with pytest.raises(ValueError, match="no complete attempt ledger"):
        ledger.continuation(root, config())
