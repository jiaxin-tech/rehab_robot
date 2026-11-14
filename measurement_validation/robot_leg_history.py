"""Prepare an ordered, fixed-reference history without hardware or motion APIs.

Completion and setup identities are declarations from upstream measurement
validation. This adapter checks their consistency, file hashes and complete
aligned tables; it does not validate physical calibration or approve motion.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lower_limb_sim.mechanical_endpoints import branch_rms_components
from measurement_validation.torque_endpoints import endpoints, torque_metrics_from_trace, RAW_ENDPOINT_METHOD

STATE_COLUMNS = ("q_hip_rad", "q_knee_rad", "dq_hip_rad_s", "dq_knee_rad_s",
                 "ddq_hip_rad_s2", "ddq_knee_rad_s2")
TORQUE_COLUMNS = ("tau_hip_nm", "tau_knee_nm")
IDENTITY_KEYS = ("setup_id", "setup_sha256", "geometry_m", "reference_candidate_id", "reference_sha256")
CLASSIFICATIONS = ("MEASURED_DATA", "SYNTHETIC_INTERFACE_DEMO")
COMPLETE_STATUSES = ("VALIDATED_COMPLETE", "CAPTURE_ONLY_COMPLETE", "ROUND_COMPLETE")
TIME_ATOL = 1e-9  # Serialization identity tolerance, not a physical timing limit.


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")
    return value


def _digest(value, name):
    _text(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"lowercase SHA-256 required: {name}")
    return value


def _resolve(base, value):
    return (base / _text(value, "path")).resolve(strict=True)


def _verified_file(base, name, digest):
    path = _resolve(base, name)
    if not path.is_file() or sha256_file(path) != _digest(digest, "file hash"):
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return path


def _numeric(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    try:
        values = frame.loc[:, list(columns)].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} requires numeric columns") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values")
    return values


def _series(frame, phase_column, label):
    values = _numeric(frame, ("time_s", *STATE_COLUMNS), label)
    time = values[:, 0]
    if len(time) < 5 or not np.all(np.diff(time) > 0) or abs(time[0]) > TIME_ATOL:
        raise ValueError(f"{label} requires complete increasing time starting at zero")
    if phase_column not in frame:
        raise ValueError(f"{label} missing {phase_column}")
    phase = frame[phase_column].to_numpy()
    transitions = phase[np.r_[True, phase[1:] != phase[:-1]]].tolist()
    if transitions != ["flexion", "extension"] or min(np.sum(phase == p) for p in transitions) < 2:
        raise ValueError(f"{label} requires complete flexion then extension phases")
    return time, phase


def measurement_endpoints(record, classification, candidate_sha256, frame):
    """Use complete raw-rate endpoints for measured data; sparse is demo only."""
    descriptor = record.get("endpoint_source")
    if descriptor is None:
        if classification == "MEASURED_DATA":
            raise ValueError("MEASURED_DATA requires a complete high-rate endpoint_source")
        time, phase = _series(frame, "phase", "endpoint frame")
        torque = _numeric(frame, TORQUE_COLUMNS, "endpoint frame")
        return dict(components_nm=list(branch_rms_components(*torque.T, time, phase)),
                    peaks_nm=np.max(np.abs(torque), axis=0).tolist()), ("SPARSE_INTERFACE_ONLY", "disjoint_phase_labels")
    path = _verified_file(Path.cwd(), descriptor.get("path"), descriptor.get("sha256"))
    source = read_json(path)
    if source.get("measured_sha256") != record.get("measured_sha256"):
        raise ValueError("raw endpoint source belongs to another measured round")
    if source.get("reference_sha256") != candidate_sha256:
        raise ValueError("raw endpoint source candidate identity mismatch")
    if source.get("endpoint_sampling", {}).get("method") != RAW_ENDPOINT_METHOD:
        raise ValueError("complete raw-rate endpoint sampling method required")
    for key in ("source_sha256", "dispatch_sha256", "analysis_config_sha256", "calibration_config_sha256"):
        _digest(source.get(key), key)
    trace = source.get("endpoint_trace")
    metrics = torque_metrics_from_trace(trace)
    for key in ("components_nm", "peaks_nm"):
        declared = np.asarray(source.get(key), float)
        if declared.shape != np.shape(metrics[key]) or not np.allclose(metrics[key], declared, rtol=1e-12, atol=1e-12):
            raise ValueError("declared raw endpoints do not match complete endpoint trace")
    boundary = float(frame.loc[frame.phase.eq("flexion"), "time_s"].iloc[-1])
    if (not np.isclose(trace["time_s"][-1], frame.time_s.iloc[-1], rtol=0, atol=TIME_ATOL)
            or not np.isclose(trace["phase_boundary_time_s"], boundary, rtol=0, atol=TIME_ATOL)):
        raise ValueError("raw endpoint trace does not cover the aligned motion/phase interval")
    return metrics, (RAW_ENDPOINT_METHOD, trace["phase_boundary_convention"],
                     source["analysis_config_sha256"], source["calibration_config_sha256"])


def prepare_history(manifest_path):
    """Return a trainer-compatible request and audit; write nothing on failure."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("history manifest schema_version must be 1 or 2")
    mode = manifest.get("replay_mode", "fit_only")
    if mode not in ("fit_only", "strict_policy"):
        raise ValueError("replay_mode must be fit_only or strict_policy")
    strict = mode == "strict_policy"
    if strict and manifest.get("attempt_ledger_complete") is not True:
        raise ValueError("strict_policy requires an explicit complete attempt ledger")
    classification = manifest.get("data_classification")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"data_classification must be one of {CLASSIFICATIONS}")
    template_descriptor = manifest.get("training_template", {})
    template_path = _verified_file(manifest_path.parent, template_descriptor.get("path"), template_descriptor.get("sha256"))
    request = deepcopy(read_json(template_path))
    if request.get("history") or request.get("attempts"):
        raise ValueError("training template history must be empty; give all ordered rounds in the manifest")
    for key in ("setup_id", "reference_candidate_id"):
        _text(request.get(key), key)
    for key in ("setup_sha256", "reference_sha256"):
        _digest(request.get(key), key)
    geometry = request.get("geometry_m")
    if (not isinstance(geometry, list) or len(geometry) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0 for x in geometry)):
        raise ValueError("two finite positive geometry_m values required")
    root = _resolve(template_path.parent, request.get("algorithm_root"))
    sources = request.get("algorithm_sha256", {})
    if set(sources) != {"config.py", "surrogate.py", "policy.py"}:
        raise ValueError("complete frozen algorithm source hashes required")
    for name, digest in sources.items():
        _verified_file(root, name, digest)
    request["algorithm_root"] = str(root)
    dependencies = request.get("algorithm_dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("frozen algorithm_dependencies required")
    request["algorithm_dependencies"] = {
        str(_verified_file(template_path.parent, name, digest)): digest for name, digest in dependencies.items()
    }
    catalog = request.get("candidates")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("nonempty candidate catalog required")
    curves = {}
    for item in catalog:
        candidate_id = _text(item.get("candidate_id"), "candidate_id")
        if candidate_id in curves:
            raise ValueError("duplicate candidate ID in catalog")
        path = _verified_file(template_path.parent, item.get("path"), item.get("sha256"))
        curve = pd.read_csv(path)
        time, phase = _series(curve, "cycle_phase", f"candidate {candidate_id}")
        for parameter in ("beta_flex", "beta_extend", "share_shift"):
            value = item.get(parameter)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"finite candidate {parameter} required")
        item["path"] = str(path)
        curves[candidate_id] = (time, phase, item)
    reference_id = request["reference_candidate_id"]
    if reference_id not in curves or curves[reference_id][2]["sha256"] != request["reference_sha256"]:
        raise ValueError("reference identity does not match candidate catalog")
    ref_time, ref_phase, _ = curves[reference_id]
    for time, phase, _ in curves.values():
        if (len(time) != len(ref_time) or not np.allclose(time[[0, -1]], ref_time[[0, -1]], rtol=0, atol=TIME_ATOL)
                or not np.array_equal(phase, ref_phase)):
            raise ValueError("candidate catalog must share reference duration, sample count and phase labels")
    rounds = manifest.get("attempts") if strict else manifest.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("nonempty ordered rounds required")
    history, metrics_rows, attempts, episodes, used = [], [], [], set(), set()
    baseline = sampling = None
    failed = False
    for index, record in enumerate(rounds, 1):
        is_failed = strict and record.get("outcome") == "FAILED"
        if failed:
            raise ValueError("attempt after a failed terminal attempt")
        if not is_failed and record.get("status") not in COMPLETE_STATUSES:
            raise ValueError(f"round {index} is not a validated complete measurement")
        if record.get("data_classification") != classification:
            raise ValueError(f"round {index} data_classification mismatch")
        for key in IDENTITY_KEYS:
            if record.get(key) != request[key]:
                raise ValueError(f"round {index} {key} mismatch")
        episode_id = _text(record.get("episode_id"), "episode_id")
        candidate_id = _text(record.get("candidate_id"), "candidate_id")
        if episode_id in episodes or candidate_id in used:
            raise ValueError("duplicate episode or candidate; repetitions need a separate repeatability analysis")
        if candidate_id not in curves:
            raise ValueError("round candidate ID absent from catalog")
        if index == 1 and candidate_id != reference_id:
            raise ValueError("first round must execute the reference candidate")
        if strict:
            if record.get("attempt_index") != index or not isinstance(record.get("role"), str):
                raise ValueError("strict attempt requires contiguous index and proposed role")
            if record.get("outcome") not in ("COMPLETED", "FAILED"):
                raise ValueError("strict attempt outcome must be COMPLETED or FAILED")
        if is_failed:
            if not isinstance(record.get("failure_reason"), str) or not record["failure_reason"].strip():
                raise ValueError("failed attempt reason required")
            if any(key in record for key in ("measured_path", "measured_sha256", "endpoints", "endpoint_source")):
                raise ValueError("failed attempt must not supply training data")
            attempts.append(deepcopy(record))
            episodes.add(episode_id)
            used.add(candidate_id)
            failed = True
            continue
        path = _verified_file(manifest_path.parent, record.get("measured_path"), record.get("measured_sha256"))
        frame = pd.read_csv(path)
        time, phase = _series(frame, "phase", f"round {index}")
        torque = _numeric(frame, TORQUE_COLUMNS, f"round {index}")
        planned_time, planned_phase, _ = curves[candidate_id]
        # aligned_measurement keeps actual dispatch time in time_s and the
        # complete commanded grid separately. Never substitute planned states.
        command_time = (_numeric(frame, ("command_time_s",), f"round {index}")[:, 0]
                        if "command_time_s" in frame else time)
        if (len(time) != len(planned_time) or not np.allclose(command_time, planned_time, rtol=0, atol=TIME_ATOL)
                or not np.array_equal(phase, planned_phase)):
            raise ValueError(f"round {index} time/phase coverage is incomplete or differs from its candidate")
        for key, expected in (("episode_id", episode_id), ("candidate_id", candidate_id), ("data_classification", classification)):
            if key in frame and not frame[key].eq(expected).all():
                raise ValueError(f"round {index} CSV {key} mismatch")
        descriptor = record.get("endpoint_source")
        if descriptor is not None:
            source = _verified_file(manifest_path.parent, descriptor.get("path"), descriptor.get("sha256"))
            record = dict(record, endpoint_source=dict(path=str(source), sha256=descriptor["sha256"]))
        metrics, this_sampling = measurement_endpoints(record, classification, curves[candidate_id][2]["sha256"], frame)
        if baseline is None:
            baseline, sampling = metrics, this_sampling
        elif this_sampling != sampling:
            raise ValueError("mixed reference/candidate endpoint sampling conventions")
        scores = endpoints(metrics, baseline)
        history.append(dict(episode_id=episode_id, candidate_id=candidate_id,
                            measured_path=str(path), measured_sha256=record["measured_sha256"], endpoints=scores))
        if descriptor is not None:
            history[-1]["endpoint_source"] = record["endpoint_source"]
        if strict:
            attempts.append(dict(record, **history[-1]))
        metrics_rows.append(dict(trial=index, episode_id=episode_id, candidate_id=candidate_id,
                                 declared_upstream_status=record["status"], **metrics, **scores))
        episodes.add(episode_id)
        used.add(candidate_id)
    request["history"] = history
    request["replay_mode"] = mode
    request["attempt_ledger_complete"] = strict
    if strict:
        request["attempts"] = attempts
    request["data_classification"] = classification
    request["history_manifest_sha256"] = sha256_file(manifest_path)
    audit = dict(schema_version=1, status="HISTORY_PREPARED", data_classification=classification,
                 manifest_path=str(manifest_path), manifest_sha256=sha256_file(manifest_path),
                 template_path=str(template_path), template_sha256=template_descriptor["sha256"],
                 identity={key: request[key] for key in IDENTITY_KEYS},
                 baseline_episode_id=history[0]["episode_id"] if history else None, baseline_metrics=baseline,
                 rounds=metrics_rows, motion_approved=False, policy_state_replayed=False,
                 replay_mode=mode, attempts=attempts if strict else None, endpoint_sampling=sampling,
                 completed_rounds_only=not strict, total_attempt_budget_accounted=strict,
                 budget_scope=("Explicit complete attempt ledger; failed attempts consume one slot and terminate replay." if strict else
                    "Completed validated rounds only; no failed-attempt budget or historical probe/stop state is reconstructed."),
                 validation_scope="Input-declared upstream completion; file hashes, identities, finite aligned states/torques, complete time/phase coverage and fixed measured reference. No physical calibration or hardware validation.")
    return request, audit
