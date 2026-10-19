"""Hash-bound trial release tests; no robot connection or motion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from safety.robot_trial_release import (
    build_robot_trial_release_approval,
    load_and_validate_robot_trial_release,
    sha256_file,
    write_robot_trial_release,
)
from tests.test_run_rehab_experiment import _reviewed_files
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH


def _wrench_config(path: Path, *, allowed_operation_states=None) -> Path:
    payload = {
        "schema_version": 1,
        "scope": "LIVE_OBSERVATION_ACQUISITION",
        "reviewed": True,
        "connection": {"robot_ip": "192.0.2.1", "local_ip": "192.0.2.2", "robot_class": "xMateRobot"},
        "expected_identity": {
            "robot_model": "xMate6-Pro", "robot_serial": "SN-RUNNER",
            "controller_version": "controller-test", "sdk_version": "0.7.0",
        },
        "allowed_power_states": ["off", "on"],
        "budgets": {
            "startup_s": 5.0, "query_s": 0.5, "progress_s": 2.0, "sample_age_s": 1.0,
            "ack_s": 1.0, "log_s": 1.0, "shutdown_s": 1.0, "terminate_s": 1.0,
            "state_age_s": 1.0, "skew_s": 1.0, "provenance": "REVIEWED_SITE_BUDGET",
        },
    }
    if allowed_operation_states is not None:
        payload["allowed_operation_states"] = allowed_operation_states
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _draft(tmp_path: Path):
    anchor, frame, safety = _reviewed_files(tmp_path)
    wrench = _wrench_config(tmp_path / "wrench.json")
    expires = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    payload = build_robot_trial_release_approval(
        reviewer="safety", operator="operator", expires_at=expires,
        robot_identity={"robot_model": "xMate6-Pro", "robot_serial_number": "SN-RUNNER",
                        "controller_version": "controller-test", "sdk_version": "0.7.0"},
        reference_path=DEFAULT_REFERENCE_PATH, safety_config_path=safety,
        frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
        collision_source="events", collision_event_max_age_s=0.5, wrench_hz=50.0,
        notes="synthetic reviewed release test",
    )
    return payload, anchor, frame, safety, wrench


def test_draft_is_explicitly_unapproved_and_written_once(tmp_path):
    payload, *_ = _draft(tmp_path)
    assert payload["approved"] is False
    output = tmp_path / "release.json"
    write_robot_trial_release(output, payload)
    with pytest.raises(FileExistsError):
        write_robot_trial_release(output, payload)


def test_approval_requires_bound_site_review_and_all_evidence(tmp_path):
    payload, anchor, frame, safety, wrench = _draft(tmp_path)
    draft = tmp_path / "draft.json"
    write_robot_trial_release(draft, payload)
    evidence = {}
    for name in ("controller.json", "acquisition.json", "wrench.jsonl", "stop.json"):
        path = tmp_path / name
        path.write_text("validated\n", encoding="utf-8")
        evidence[name] = {"passed": True, "path": name, "sha256": sha256_file(path)}
    site = tmp_path / "site_review.json"
    site.write_text(json.dumps({
        "scope": payload["scope"], "decision": "approved", "reviewer": payload["reviewer"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_sha256": payload["bundle_sha256"], "records": {
            "controller_safety": evidence["controller.json"],
            "acquisition_stability": evidence["acquisition.json"],
            "wrench_semantics": evidence["wrench.jsonl"],
            "setup_and_stop": evidence["stop.json"],
        },
    }), encoding="utf-8")
    approved = dict(payload, approved=True, site_review={"path": site.name, "sha256": sha256_file(site)})
    approved_path = tmp_path / "approved.json"
    write_robot_trial_release(approved_path, approved)
    validated = load_and_validate_robot_trial_release(
        approved_path, reference_path=DEFAULT_REFERENCE_PATH, safety_config_path=safety,
        frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
    )
    assert validated.payload["approved"] is True
    assert validated.payload["collision_source"] == "events"
    validated.require_current(check_files=True)


def test_input_change_invalidates_approved_release(tmp_path):
    payload, anchor, frame, safety, wrench = _draft(tmp_path)
    draft = tmp_path / "draft.json"
    write_robot_trial_release(draft, payload)
    evidence = {}
    for name in ("controller.json", "acquisition.json", "wrench.jsonl", "stop.json"):
        evidence_path = tmp_path / name
        evidence_path.write_text("validated\n", encoding="utf-8")
        evidence[name] = {"passed": True, "path": name, "sha256": sha256_file(evidence_path)}
    site = tmp_path / "site.json"
    site.write_text(json.dumps({
        "scope": payload["scope"], "decision": "approved", "reviewer": payload["reviewer"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_sha256": payload["bundle_sha256"], "records": {
            "controller_safety": evidence["controller.json"],
            "acquisition_stability": evidence["acquisition.json"],
            "wrench_semantics": evidence["wrench.jsonl"],
            "setup_and_stop": evidence["stop.json"],
        },
    }), encoding="utf-8")
    approved = dict(payload, approved=True, site_review={"path": site.name, "sha256": sha256_file(site)})
    approved_path = tmp_path / "approved.json"
    write_robot_trial_release(approved_path, approved)
    safety.write_text(safety.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="input binding mismatch"):
        load_and_validate_robot_trial_release(
            approved_path, reference_path=DEFAULT_REFERENCE_PATH, safety_config_path=safety,
            frame_config_path=frame, anchor_path=anchor, wrench_config_path=wrench,
        )
