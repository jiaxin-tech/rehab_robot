"""Offline, evidence-bound release for one supervised reference trial.

Hashes bind reviewed content, not the identity of a human signer. Review files
are trusted local records; this module is not a digital signature authority.
The frozen reference remains NO_GO and is never rewritten by this workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from config.bundle import resolve_config_path
from typing import Any, Mapping
from uuid import uuid4
import math

from hardware.safety_events import validate_collision_source

SCOPE = "FIRST_ROBOT_TRIAL_EMPTY_LOAD"
LOADED_SCOPE = "ROBOT_DUMMY_LEG_REFERENCE_TRIAL"
LOADED_TRIAL_KINDS = ("dummy_leg_reference", "dummy_leg_capture_only")
LOADED_EVIDENCE = ("dummy_leg_installation", "joint_angle_calibration", "cross_device_timing")
APPROVAL_CONFIRMATION = "I CONFIRM REVIEWED FIRST ROBOT TRIAL RELEASE"
IDENTITY_KEYS = ("robot_model", "robot_serial_number", "controller_version", "sdk_version")
EVIDENCE_KINDS = ("controller_safety", "acquisition_stability", "wrench_semantics", "setup_and_stop")
INPUT_KEYS = ("reference", "safety_config", "frame_config", "anchor", "wrench_config")
_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = (
    "config/bundle.py", "utils/evidence.py", "safety/robot_trial_release.py", "safety/experiment_safety.py",
    "scripts/run_rehab_experiment.py", "scripts/acquire_robot_data.py",
    "control/execution_preflight.py", "control/robot_trajectory_executor.py",
    "control/start_anchor.py", "control/start_anchored_relative_trajectory.py",
    "lower_limb_sim/reference_release.py", "hardware/safety_events.py",
    "hardware/rokae_adapter.py", "hardware/windows/rokae_xcore.py",
    "hardware/rokae_motion.py", "collection/real_robot_acquisition.py",
    "collection/episode_logger.py", "collection/wrench_process.py",
    "scripts/prepare_robot_trial_release.py",
    "collection/twincat_leg.py", "measurement_validation/robot_leg_round.py",
    "measurement_validation/leg_episode.py", "measurement_validation/offline_leg.py",
    "scripts/train_robot_leg_round.py", "scripts/run_robot_leg_round.py",
)
_VALIDATED = object()
_KEYS = {
    "schema_version", "scope", "approval_id", "approved", "reviewer", "operator",
    "issued_at", "expires_at", "robot_identity", "bindings", "object_bindings",
    "code_sha256", "collision_source", "collision_event_max_age_s", "wrench_hz",
    "bundle_sha256", "site_review", "notes",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(resolve_config_path(path).read_bytes()).hexdigest()


def _input_paths(reference, safety, frame, anchor, wrench):
    return dict(reference=resolve_config_path(reference),
                safety_config=resolve_config_path(safety, section="experiment_safety"),
                frame_config=resolve_config_path(frame, section="rehab_frame_config"),
                anchor=resolve_config_path(anchor),
                wrench_config=resolve_config_path(wrench, section="live_observation_acquisition_20261018"))


def object_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    payload = json.loads(resolve_config_path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    return payload


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp string required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"non-empty {name} required")
    return value.strip()


def _code_hashes():
    return {name: sha256_file(_ROOT / name) for name in _SOURCES}


def _bundle(payload):
    values = {key: payload[key] for key in (
        "scope", "robot_identity", "bindings", "object_bindings", "code_sha256",
        "collision_source", "collision_event_max_age_s", "wrench_hz",
    )}
    if "trial_setup" in payload:
        values["trial_setup"] = payload["trial_setup"]
    return object_digest(values)


def _trial_setup(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise PermissionError("bound dummy-leg trial setup required")
    path = Path(reference["path"]).resolve()
    if sha256_file(path) != reference["sha256"]:
        raise PermissionError("trial setup checksum mismatch")
    setup = read_json(path)
    if setup.get("scope") != LOADED_SCOPE or setup.get("reviewed") is not True:
        raise PermissionError("reviewed connected dummy-leg setup required")
    inputs = setup.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"leg_acquisition", "calibration", "analysis", "round_config"}:
        raise ValueError("trial setup inputs incomplete")
    files = [(str(path), reference["sha256"])]
    data = {}
    input_paths = {}
    for name, item in inputs.items():
        p = (path.parent / item["path"]).resolve()
        if sha256_file(p) != item["sha256"]:
            raise PermissionError("trial setup input changed: " + name)
        data[name] = read_json(p)
        input_paths[name] = p
        files.append((str(p), item["sha256"]))
    from collection.twincat_leg import validate_config
    from measurement_validation.robot_leg_round import validate_analysis
    validate_config(data["leg_acquisition"])
    validate_analysis(data["analysis"], data["calibration"])
    if (data["round_config"].get("reviewed") is not True or
            data["round_config"].get("trial_kind") not in LOADED_TRIAL_KINDS):
        raise PermissionError("round config is not reviewed for a connected dummy-leg trial")
    for name, key in (("leg_acquisition", "leg_config"), ("calibration", "calibration"), ("analysis", "analysis")):
        actual = (input_paths["round_config"].parent / data["round_config"][key]).resolve()
        if actual != input_paths[name]:
            raise PermissionError("trial setup and round input path mismatch: " + name)
    return tuple(files)


def _objects(paths):
    # Lazy imports avoid a preflight -> release -> preflight import cycle.
    from control.start_anchor import load_start_anchor
    from control.start_anchored_relative_trajectory import load_rehab_frame_config, build_start_anchored_relative_trajectory
    from control.execution_preflight import trajectory_execution_digest
    from safety.experiment_safety import load_experiment_safety_config
    from lower_limb_sim.reference_release import load_frozen_active_reference

    load_frozen_active_reference(paths["reference"])
    safety = load_experiment_safety_config(paths["safety_config"])
    frame = load_rehab_frame_config(paths["frame_config"])
    anchor = load_start_anchor(paths["anchor"])
    wrench = read_json(paths["wrench_config"])
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        paths["reference"], current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    hashes = {
        "safety": object_digest(safety.to_dict()), "frame": object_digest(frame.as_metadata()),
        "anchor": object_digest(anchor.to_dict()), "wrench": object_digest(wrench),
        "trajectory": trajectory_execution_digest(trajectory),
    }
    return safety, frame, anchor, wrench, trajectory, audit, hashes


def build_robot_trial_release_approval(
    *, reviewer: str, operator: str, expires_at: str, robot_identity: Mapping[str, str],
    reference_path, safety_config_path, frame_config_path, anchor_path, wrench_config_path,
    collision_source: str, collision_event_max_age_s: float | None, wrench_hz: float,
    notes: str = "",
    scope: str = SCOPE, trial_setup_path=None,
) -> dict[str, Any]:
    """Create a draft only. Approval requires a separate, bound site review."""
    validate_collision_source(collision_source, collision_event_max_age_s)
    if scope not in (SCOPE, LOADED_SCOPE):
        raise ValueError("unsupported release scope")
    if scope == SCOPE and trial_setup_path is not None:
        raise ValueError("empty-load release cannot contain a connected-leg setup")
    if isinstance(wrench_hz, bool) or not isinstance(wrench_hz, (int, float)) or not math.isfinite(wrench_hz) or wrench_hz <= 0:
        raise ValueError("finite positive wrench_hz required")
    if set(robot_identity) != set(IDENTITY_KEYS):
        raise ValueError("robot_identity schema mismatch")
    identity = {key: _text(robot_identity[key], key) for key in IDENTITY_KEYS}
    paths = _input_paths(reference_path, safety_config_path, frame_config_path, anchor_path, wrench_config_path)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    *_, objects = _objects(paths)
    if hashes != {key: sha256_file(path) for key, path in paths.items()}:
        raise PermissionError("release inputs changed during draft generation")
    now = datetime.now(timezone.utc)
    if _time(expires_at) <= now:
        raise ValueError("expiry must be in the future")
    payload = dict(
        schema_version=1, scope=scope, approval_id=f"RBT-{uuid4().hex[:12].upper()}", approved=False,
        reviewer=_text(reviewer, "reviewer"), operator=_text(operator, "operator"),
        issued_at=now.isoformat(), expires_at=expires_at, robot_identity=identity,
        bindings=hashes, object_bindings=objects, code_sha256=_code_hashes(),
        collision_source=collision_source, collision_event_max_age_s=collision_event_max_age_s,
        wrench_hz=wrench_hz, notes=str(notes), site_review=None,
    )
    if scope == LOADED_SCOPE:
        if trial_setup_path is None:
            raise ValueError("connected dummy-leg release requires --trial-setup")
        payload["trial_setup"] = dict(path=str(Path(trial_setup_path).resolve()), sha256=sha256_file(trial_setup_path))
        _trial_setup(payload["trial_setup"])
    payload["bundle_sha256"] = _bundle(payload)
    return payload


def _validate_site_review(payload, artifact_dir: Path, now: datetime) -> tuple[tuple[str, str], ...]:
    reference = payload["site_review"]
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise PermissionError("bound site_review is required")
    path = (artifact_dir / _text(reference["path"], "site review path")).resolve()
    digest = sha256_file(path)
    if digest != reference["sha256"]:
        raise PermissionError("site review checksum mismatch")
    review = read_json(path)
    if (review.get("scope") != payload["scope"] or review.get("decision") != "approved"
            or review.get("reviewer") != payload["reviewer"]
            or review.get("bundle_sha256") != payload["bundle_sha256"]):
        raise PermissionError("site review scope/reviewer/bundle/decision mismatch")
    reviewed = _time(review.get("reviewed_at"))
    if reviewed > now or reviewed < _time(payload["issued_at"]):
        raise PermissionError("site review timestamp is outside the review window")
    records = review.get("records")
    kinds = set(EVIDENCE_KINDS) | (set(LOADED_EVIDENCE) if payload["scope"] == LOADED_SCOPE else set())
    if not isinstance(records, dict) or set(records) != kinds:
        raise PermissionError("site review evidence set incomplete")
    files = [(str(path), digest)]
    for kind, record in records.items():
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise PermissionError(f"site evidence not passed: {kind}")
        evidence = resolve_config_path(path.parent / _text(record.get("path"), "evidence path"))
        if not evidence.is_file() or evidence.stat().st_size == 0 or sha256_file(evidence) != record.get("sha256"):
            raise PermissionError(f"site evidence missing/empty/changed: {kind}")
        files.append((str(evidence), record["sha256"]))
    return tuple(files)


@dataclass(frozen=True)
class ValidatedRobotTrialRelease:
    """Immutable validated snapshot, re-bound by preflight to in-memory inputs."""
    payload_json: str
    approval_path: str
    approval_sha256: str
    files: tuple[tuple[str, str], ...]
    _token: object = field(repr=False, compare=False)

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    def as_metadata(self) -> dict:
        return {"approval_path": self.approval_path, "approval_sha256": self.approval_sha256,
                "approval": json.loads(self.payload_json)}

    def require_current(self, *, check_files: bool = False):
        if self._token is not _VALIDATED:
            raise PermissionError("release is not validated")
        payload = json.loads(self.payload_json)
        now = datetime.now(timezone.utc)
        if not _time(payload["issued_at"]) <= now < _time(payload["expires_at"]):
            raise PermissionError("release not yet valid or expired")
        if check_files:
            for path, expected in ((self.approval_path, self.approval_sha256), *self.files):
                if sha256_file(path) != expected:
                    raise PermissionError("release reviewed file changed: " + path)
        return payload

    def require_objects(self, *, safety, frame, anchor, trajectory):
        from control.execution_preflight import trajectory_execution_digest
        payload = self.require_current()
        current = dict(safety=object_digest(safety.to_dict()), frame=object_digest(frame.as_metadata()),
                       anchor=object_digest(anchor.to_dict()), trajectory=trajectory_execution_digest(trajectory))
        for key, digest in current.items():
            if payload["object_bindings"][key] != digest:
                raise PermissionError("release object binding mismatch: " + key)

    def require_acquisition(self, *, config, wrench_hz, collision_source, collision_event_max_age_s):
        payload = self.require_current()
        if object_digest(config) != payload["object_bindings"]["wrench"]:
            raise PermissionError("release wrench object binding mismatch")
        if (isinstance(wrench_hz, bool) or not isinstance(wrench_hz, (int, float))
                or not math.isfinite(wrench_hz) or wrench_hz != payload["wrench_hz"]):
            raise PermissionError("release wrench frequency mismatch")
        if (collision_source != payload["collision_source"]
                or collision_event_max_age_s != payload["collision_event_max_age_s"]):
            raise PermissionError("release collision configuration mismatch")

    def require_runtime(self, summary):
        payload = self.require_current()
        metadata = summary.get("robot_metadata") or {}
        for key in IDENTITY_KEYS:
            actual = metadata.get("xcore_sdk_version" if key == "sdk_version" else key)
            if actual != payload["robot_identity"][key]:
                raise PermissionError("release runtime identity mismatch: " + key)
        if summary.get("collision_state_source") != payload["collision_source"]:
            raise PermissionError("release runtime collision source mismatch")
        if payload["collision_source"] == "events":
            event = summary.get("safety_event_status") or {}
            if event.get("max_age_s") != payload["collision_event_max_age_s"]:
                raise PermissionError("release runtime event budget mismatch")


def load_and_validate_robot_trial_release(
    path, *, reference_path, safety_config_path, frame_config_path, anchor_path, wrench_config_path,
) -> ValidatedRobotTrialRelease:
    path = Path(path).resolve()
    approval_hash = sha256_file(path)
    payload = read_json(path)
    files = validate_robot_trial_release_payload(
        payload, artifact_path=path, reference_path=reference_path,
        safety_config_path=safety_config_path, frame_config_path=frame_config_path,
        anchor_path=anchor_path, wrench_config_path=wrench_config_path)
    result = ValidatedRobotTrialRelease(json.dumps(payload, allow_nan=False), str(path), approval_hash,
                                       files, _VALIDATED)
    result.require_current(check_files=True)
    return result


def validate_robot_trial_release_payload(
    payload, *, artifact_path, reference_path, safety_config_path, frame_config_path,
    anchor_path, wrench_config_path,
) -> tuple[tuple[str, str], ...]:
    """Fully validate proposed publication without writing an approved artifact."""
    path = Path(artifact_path).resolve()
    keys = _KEYS | ({"trial_setup"} if payload.get("scope") == LOADED_SCOPE else set())
    if set(payload) != keys or type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or payload["scope"] not in (SCOPE, LOADED_SCOPE):
        raise ValueError("robot trial release approval schema mismatch")
    if payload["approved"] is not True:
        raise PermissionError("robot trial release approval is not approved")
    for key in ("approval_id", "reviewer", "operator"):
        _text(payload[key], key)
    now = datetime.now(timezone.utc)
    if not _time(payload["issued_at"]) <= now < _time(payload["expires_at"]):
        raise PermissionError("release not yet valid or expired")
    validate_collision_source(payload["collision_source"], payload["collision_event_max_age_s"])
    hz = payload["wrench_hz"]
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("finite positive wrench_hz required")
    if payload["code_sha256"] != _code_hashes() or payload["bundle_sha256"] != _bundle(payload):
        raise PermissionError("release code or bundle binding mismatch")
    paths = _input_paths(reference_path, safety_config_path, frame_config_path, anchor_path, wrench_config_path)
    current_hashes = {key: sha256_file(p) for key, p in paths.items()}
    if payload["bindings"] != current_hashes:
        raise PermissionError("robot trial release input binding mismatch")
    safety, frame, anchor, wrench, trajectory, audit, objects = _objects(paths)
    if objects != payload["object_bindings"]:
        raise PermissionError("release parsed object binding mismatch")
    safety.require_execute_allowed()
    if not frame.reviewed or not anchor.reviewed or not audit.trajectory_valid:
        raise PermissionError("release frame/anchor/trajectory not ready")
    from scripts.acquire_robot_data import load_live_wrench_config
    from collection.wrench_process import WrenchBudgets
    from control.execution_preflight import audit_execution_trajectory_content
    checked_wrench = load_live_wrench_config(paths["wrench_config"])
    WrenchBudgets(**checked_wrench["budgets"])
    if "on" not in wrench["allowed_power_states"]:
        raise PermissionError("release wrench configuration does not allow powered observation")
    identity = payload["robot_identity"]
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY_KEYS):
        raise ValueError("robot_identity schema mismatch")
    for key in IDENTITY_KEYS:
        _text(identity[key], key)
        wrench_key = "robot_serial" if key == "robot_serial_number" else key
        if wrench["expected_identity"][wrench_key] != identity[key]:
            raise PermissionError("release wrench identity mismatch: " + key)
        if key != "sdk_version":
            if getattr(safety, "expected_" + key) != identity[key] or getattr(anchor, key) != identity[key]:
                raise PermissionError("release safety/anchor identity mismatch: " + key)
    reasons, _, _ = audit_execution_trajectory_content(trajectory, safety, frame=frame, anchor=anchor, require_official_reference=True)
    if reasons:
        raise PermissionError("release trajectory blocked: " + ";".join(reasons))
    evidence_files = _validate_site_review(payload, path.parent, now)
    files = tuple((str(p.resolve()), current_hashes[key]) for key, p in paths.items())
    files += tuple((str(_ROOT / name), digest) for name, digest in payload["code_sha256"].items())
    files += evidence_files
    if payload["scope"] == LOADED_SCOPE:
        files += _trial_setup(payload["trial_setup"])
    for filename, expected in files:
        if sha256_file(filename) != expected:
            raise PermissionError("release reviewed file changed during validation: " + filename)
    return files


def write_robot_trial_release(path, payload) -> Path:
    """Publish a new artifact exclusively; never overwrite earlier evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return destination
