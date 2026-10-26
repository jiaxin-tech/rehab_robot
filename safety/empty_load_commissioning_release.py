"""Hash-bound, single-cycle release for the reviewed empty-load commissioning run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from config.bundle import resolve_config_path
from typing import Any, Mapping

import pandas as pd

from control.reference_commissioning import retime_formal_reference
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from safety.experiment_safety import load_experiment_safety_config


SCOPE = "FIRST_ROBOT_TRIAL_EMPTY_LOAD_FIXTURE_ONLY"
SCHEMA_VERSION = 1
EXPECTED_REFERENCE_SHA256 = "892da699b9afd0f57b8e77a5fa1c9b8da6069599515f7e2c24fec6b064b3b945"
EXPECTED_SPEED_CAP_M_S = 0.03
EXPECTED_COLLISION_ERROR_CODE = 259
_VALIDATED = object()
_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "config/bundle.py", "utils/evidence.py",
    "safety/empty_load_commissioning_release.py",
    "control/execution_preflight.py",
    "control/robot_trajectory_executor.py",
    "scripts/run_rehab_experiment.py",
    "scripts/acquire_robot_data.py",
    "collection/real_robot_acquisition.py",
    "collection/episode_logger.py",
    "collection/wrench_process.py",
    "hardware/rokae_adapter.py",
    "hardware/wrench_session.py",
    "hardware/windows/rokae_xcore.py",
    "hardware/rokae_motion.py",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(resolve_config_path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(resolve_config_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")
    return parsed.astimezone(timezone.utc)


def validate_retimed_reference(path: str | Path) -> dict[str, float]:
    """Prove the reviewed CSV is the exact 0.03 m/s uniform retime of the frozen path."""
    reference_path = Path(path).resolve()
    digest = sha256_file(reference_path)
    if digest != EXPECTED_REFERENCE_SHA256:
        raise PermissionError("empty-load commissioning reference checksum mismatch")
    source = pd.read_csv(DEFAULT_REFERENCE_PATH)
    expected, timing = retime_formal_reference(
        source,
        max_tcp_speed_m_s=EXPECTED_SPEED_CAP_M_S,
    )
    actual = pd.read_csv(reference_path)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise PermissionError(
            "empty-load commissioning reference is not the exact reviewed uniform retime"
        ) from exc
    if timing["retimed_max_pull_speed_m_s"] > EXPECTED_SPEED_CAP_M_S + 1e-12:
        raise PermissionError("empty-load commissioning reference exceeds speed cap")
    return timing


@dataclass(frozen=True)
class ValidatedEmptyLoadCommissioningRelease:
    payload_json: str
    approval_path: str
    approval_sha256: str
    files: tuple[tuple[str, str], ...]
    _token: object

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "approval_path": self.approval_path,
            "approval_sha256": self.approval_sha256,
            "approval": self.payload,
        }

    def _claim_path(self) -> Path:
        return _ROOT / "results" / "commissioning_release_uses" / (self.approval_sha256 + ".json")

    def require_unused(self) -> None:
        self.require_current()
        if self._claim_path().exists():
            raise PermissionError("commissioning release already claimed; no automatic retry")

    def claim_execution(self) -> str:
        """Conservatively consume at arming, before the first command intent.

        Exclusive creation arbitrates independent processes on this checkout.
        An incomplete claim remains consumed after a crash or write failure.
        This is a local ledger, not a cross-machine authorization service.
        """
        self.require_current(check_files=True)
        path = self._claim_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(dict(approval_sha256=self.approval_sha256,
                               approval_path=self.approval_path,
                               claimed_at=datetime.now(timezone.utc).isoformat(),
                               pid=os.getpid(), state="CLAIMED_NO_AUTOMATIC_RETRY"), stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise PermissionError("commissioning release already claimed; no automatic retry") from exc
        return str(path)

    def require_current(self, *, check_files: bool = False) -> dict[str, Any]:
        if self._token is not _VALIDATED:
            raise PermissionError("commissioning release is not validated")
        payload = self.payload
        now = datetime.now(timezone.utc)
        if not _aware(payload["issued_at"], "issued_at") <= now < _aware(
            payload["expires_at"], "expires_at"
        ):
            raise PermissionError("commissioning release is not current")
        if check_files:
            for filename, expected in (
                (self.approval_path, self.approval_sha256),
                *self.files,
            ):
                if sha256_file(filename) != expected:
                    raise PermissionError(
                        "commissioning release reviewed file changed: " + filename
                    )
        return payload

    def require_objects(self, *, safety, frame, anchor, trajectory) -> None:
        from control.execution_preflight import (
            experiment_safety_digest,
            trajectory_execution_digest,
        )

        payload = self.require_current()
        expected = payload["object_bindings"]
        actual = {
            "safety": experiment_safety_digest(safety),
            "frame": hashlib.sha256(
                json.dumps(
                    frame.as_metadata(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "anchor": hashlib.sha256(
                json.dumps(
                    anchor.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "trajectory": trajectory_execution_digest(trajectory),
        }
        if actual != expected:
            raise PermissionError("commissioning release parsed object binding mismatch")

    def require_acquisition(
        self,
        *,
        config,
        wrench_hz,
        collision_source,
        collision_event_max_age_s,
    ) -> None:
        payload = self.require_current()
        if config != payload["wrench_config_object"]:
            raise PermissionError("commissioning wrench object binding mismatch")
        if not math.isclose(float(wrench_hz), float(payload["wrench_hz"]), abs_tol=0.0):
            raise PermissionError("commissioning wrench frequency mismatch")
        if collision_source != "query" or collision_event_max_age_s is not None:
            raise PermissionError("commissioning collision query configuration mismatch")

    def require_runtime(self, summary: Mapping[str, Any]) -> None:
        payload = self.require_current()
        metadata = summary.get("robot_metadata")
        if not isinstance(metadata, Mapping):
            raise PermissionError("commissioning runtime robot metadata unavailable")
        for key, expected in payload["robot_identity"].items():
            runtime_key = "xcore_sdk_version" if key == "sdk_version" else key
            if metadata.get(runtime_key) != expected:
                raise PermissionError("commissioning runtime identity mismatch: " + key)
        if summary.get("collision_state_source") != "query":
            raise PermissionError("commissioning runtime collision source mismatch")
        if summary.get("collision_state") is True:
            raise PermissionError("commissioning runtime collision detected")
        if summary.get("collision_state_query_valid") is not True:
            reason = str(summary.get("collision_state_invalid_reason") or "")
            if "259" not in reason:
                raise PermissionError(
                    "commissioning collision waiver applies only to SDK error 259"
                )


def load_and_validate_empty_load_commissioning_release(
    path: str | Path,
    *,
    reference_path: str | Path,
    safety_config_path: str | Path,
    frame_config_path: str | Path,
    anchor_path: str | Path,
    wrench_config_path: str | Path,
) -> ValidatedEmptyLoadCommissioningRelease:
    approval_path = Path(path).resolve()
    payload = _read_json(approval_path)
    required = {
        "schema_version",
        "scope",
        "approval_id",
        "approved",
        "reviewer",
        "operator",
        "issued_at",
        "expires_at",
        "single_cycle",
        "training_allowed",
        "reference_sha256",
        "review_request",
        "bindings",
        "object_bindings",
        "robot_identity",
        "wrench_hz",
        "wrench_config_object",
        "code_sha256",
        "allow_missing_sdk_collision_state_error",
        "controller_collision_protection_required",
        "operator_emergency_stop_required",
        "confirmation",
    }
    if set(payload) != required:
        raise ValueError("empty-load commissioning release schema mismatch")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["scope"] != SCOPE
        or payload["approved"] is not True
        or payload["single_cycle"] is not True
        or payload["training_allowed"] is not False
    ):
        raise PermissionError("empty-load commissioning release is not approved")
    if payload["allow_missing_sdk_collision_state_error"] != EXPECTED_COLLISION_ERROR_CODE:
        raise PermissionError("unexpected collision-state waiver")
    if (
        payload["controller_collision_protection_required"] is not True
        or payload["operator_emergency_stop_required"] is not True
    ):
        raise PermissionError("hardware collision protection and emergency stop are mandatory")
    expected_code = {name: sha256_file(_ROOT / name) for name in CODE_FILES}
    if payload["code_sha256"] != expected_code:
        raise PermissionError("commissioning execution code changed after review")
    now = datetime.now(timezone.utc)
    if not _aware(payload["issued_at"], "issued_at") <= now < _aware(
        payload["expires_at"], "expires_at"
    ):
        raise PermissionError("empty-load commissioning release is not current")

    request_path = (approval_path.parent / payload["review_request"]["path"]).resolve()
    request = _read_json(request_path)
    if sha256_file(request_path) != payload["review_request"]["sha256"]:
        raise PermissionError("commissioning review request changed")
    if payload["confirmation"] != request.get("required_confirmation"):
        raise PermissionError("commissioning reviewer confirmation mismatch")
    if request.get("scope") != SCOPE or request.get("reviewer") != payload["reviewer"]:
        raise PermissionError("commissioning review request scope/reviewer mismatch")
    if request.get("physical_scope") != {
        "human_subject_present": False,
        "dummy_leg_installed": False,
        "flange_fixture_installed": True,
        "cycles": 1,
        "training_allowed": False,
        "capture_required": True,
    }:
        raise PermissionError("commissioning physical scope differs from reviewed request")
    request_reference = request.get("bindings", {}).get("retimed_reference", {})
    if request_reference.get("sha256") != EXPECTED_REFERENCE_SHA256:
        raise PermissionError("review request did not approve the bound retimed reference")
    waiver = request.get("known_exception_requiring_explicit_waiver", {})
    if waiver.get("error_code") != EXPECTED_COLLISION_ERROR_CODE:
        raise PermissionError("review request collision waiver mismatch")

    paths = {
        "reference": resolve_config_path(reference_path),
        "safety": resolve_config_path(safety_config_path, section="experiment_safety"),
        "frame": resolve_config_path(frame_config_path, section="rehab_frame_config"),
        "anchor": Path(anchor_path).resolve(),
        "wrench": resolve_config_path(wrench_config_path, section="live_observation_acquisition_20261018"),
    }
    hashes = {name: sha256_file(value) for name, value in paths.items()}
    if hashes != payload["bindings"]:
        raise PermissionError("commissioning release file binding mismatch")
    if hashes["reference"] != payload["reference_sha256"]:
        raise PermissionError("commissioning reference binding mismatch")
    validate_retimed_reference(paths["reference"])

    safety = load_experiment_safety_config(paths["safety"])
    safety.require_execute_allowed()
    requested_limits = request.get("candidate_limits_requiring_review", {})
    expected_limits = {
        "start_position_tolerance_m": safety.max_start_anchor_position_error_m,
        "start_orientation_tolerance_rad": safety.max_start_anchor_orientation_error_rad,
        "start_orientation_tolerance_deg": 3.0,
        "command_lateness_s": safety.max_command_lateness_s,
        "rt_filter_hz": safety.reviewed_rt_filter_hz,
        "rt_network_tolerance_percent": safety.reviewed_rt_network_tolerance_percent,
    }
    if requested_limits != expected_limits:
        raise PermissionError("reviewed safety limits differ from the signed request")
    frame = load_rehab_frame_config(paths["frame"])
    anchor = load_start_anchor(paths["anchor"])
    if not frame.reviewed or not anchor.reviewed:
        raise PermissionError("commissioning frame/anchor not reviewed")
    from scripts.acquire_robot_data import load_live_wrench_config

    wrench = load_live_wrench_config(paths["wrench"])
    if "on" not in wrench["allowed_power_states"]:
        raise PermissionError("commissioning observation must allow powered state")
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        paths["reference"],
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    if not audit.trajectory_valid:
        raise PermissionError("commissioning trajectory audit failed")

    from control.execution_preflight import (
        audit_execution_trajectory_content,
        experiment_safety_digest,
        trajectory_execution_digest,
    )

    reasons, _, _ = audit_execution_trajectory_content(
        trajectory,
        safety,
        frame=frame,
        anchor=anchor,
        require_official_reference=False,
    )
    if reasons:
        raise PermissionError("commissioning trajectory blocked: " + ";".join(reasons))
    object_bindings = {
        "safety": experiment_safety_digest(safety),
        "frame": hashlib.sha256(
            json.dumps(frame.as_metadata(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "anchor": hashlib.sha256(
            json.dumps(anchor.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "trajectory": trajectory_execution_digest(trajectory),
    }
    if object_bindings != payload["object_bindings"]:
        raise PermissionError("commissioning release object binding mismatch")
    if wrench != payload["wrench_config_object"]:
        raise PermissionError("commissioning release wrench object mismatch")

    files = tuple((str(value), hashes[name]) for name, value in paths.items()) + (
        (str(request_path), payload["review_request"]["sha256"]),
    ) + tuple((str(_ROOT / name), digest) for name, digest in expected_code.items())
    result = ValidatedEmptyLoadCommissioningRelease(
        payload_json=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        approval_path=str(approval_path),
        approval_sha256=sha256_file(approval_path),
        files=files,
        _token=_VALIDATED,
    )
    result.require_current(check_files=True)
    result.require_unused()
    return result


__all__ = [
    "SCOPE",
    "CODE_FILES",
    "ValidatedEmptyLoadCommissioningRelease",
    "load_and_validate_empty_load_commissioning_release",
    "validate_retimed_reference",
]
