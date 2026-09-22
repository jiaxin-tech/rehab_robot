"""Pure offline analysis for real-measurement validity and sensitivity.

The module consumes already recorded mappings.  It has no robot, controller,
motion, calibration, safety, BO, or personalization dependency.  Missing and
invalid observations remain missing; they are never replaced with zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_ID = "IMPLEMENT_REAL_MEASUREMENT_VALIDATION_ANALYSIS_V1"
INPUT_SCHEMA_VERSION = "REAL_MEASUREMENT_VALIDATION_INPUT_V1"
STATUSES = {"SUPPORTED", "INSUFFICIENT", "FAILED"}
PHASES = ("PRE", "LOAD", "POST")
BRANCHES = ("FLEXION", "EXTENSION")
FORCE_COMPONENTS = ("fx", "fy", "fz")
TORQUE_COMPONENTS = ("mx", "my", "mz")
DIRECT_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "fx": ("fx", "fx_raw_n"),
    "fy": ("fy", "fy_raw_n"),
    "fz": ("fz", "fz_raw_n"),
    "mx": ("mx", "tx", "mx_raw_nm", "tx_raw_nm"),
    "my": ("my", "ty", "my_raw_nm", "ty_raw_nm"),
    "mz": ("mz", "tz", "mz_raw_nm", "tz_raw_nm"),
}
MODEL_DERIVED_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "hip_response": ("hip_response", "hip_mechanical_response"),
    "knee_response": ("knee_response", "knee_mechanical_response"),
    **{
        f"joint_external_torque_{index}": (f"joint_external_torque_{index}",)
        for index in range(1, 7)
    },
}
DIRECT_CHANNEL_ALIASES.update(
    {
        f"joint_measured_torque_{index}": (f"joint_measured_torque_{index}",)
        for index in range(1, 7)
    }
)
SETUP_METADATA_FIELDS = (
    "robot_model",
    "robot_serial_number",
    "controller_version",
    "active_tool_name",
    "active_workobject_name",
    "tcp_id",
    "payload_id",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "valid"}:
            return True
        if normalized in {"false", "0", "no", "invalid"}:
            return False
    return None


def _first_finite(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for name in aliases:
        value = _finite(row.get(name))
        if value is not None:
            return value
    return None


def _sample_standard_deviation(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else None


def _numeric_summary(values: Iterable[float | None]) -> dict[str, Any]:
    usable = np.asarray([value for value in values if value is not None], dtype=float)
    if usable.size == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "p95": None,
        }
    return {
        "count": int(usable.size),
        "mean": float(np.mean(usable)),
        "standard_deviation": _sample_standard_deviation(usable.tolist()),
        "minimum": float(np.min(usable)),
        "maximum": float(np.max(usable)),
        "p95": float(np.percentile(usable, 95.0)),
    }


def _vector_mean(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    if not vectors:
        return None
    return np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()


def _vector_std(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    if len(vectors) < 2:
        return None
    return np.std(np.asarray(vectors, dtype=float), axis=0, ddof=1).tolist()


def _vector_norm(vector: Sequence[float] | None) -> float | None:
    if vector is None:
        return None
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        return None
    return float(np.linalg.norm(values))


def _force_vector(row: Mapping[str, Any]) -> list[float | None]:
    return [_first_finite(row, DIRECT_CHANNEL_ALIASES[name]) for name in FORCE_COMPONENTS]


def _torque_vector(row: Mapping[str, Any]) -> list[float | None]:
    return [_first_finite(row, DIRECT_CHANNEL_ALIASES[name]) for name in TORQUE_COMPONENTS]


def _reference_vector(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float] | None, bool]:
    vectors: list[list[float]] = []
    for row in rows:
        supplied = row.get("reference_load_vector_n")
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            values = [_finite(value) for value in supplied]
        else:
            values = [_finite(row.get(f"reference_{name}_n")) for name in FORCE_COMPONENTS]
        if len(values) == 3 and all(value is not None for value in values):
            vectors.append([float(value) for value in values])
    if not vectors:
        return None, True
    consistent = all(np.allclose(vector, vectors[0], rtol=0.0, atol=1.0e-12) for vector in vectors[1:])
    return (vectors[0] if consistent else None), consistent


def _query_duration_ms(row: Mapping[str, Any]) -> float | None:
    direct = _first_finite(row, ("query_duration_ms", "query_latency_ms"))
    if direct is not None:
        return direct
    start = _first_finite(row, ("query_start_s",))
    end = _first_finite(row, ("query_end_s",))
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000.0


def _invalid_reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("invalid_reason") or "")
        for item in reason.split(";"):
            if item.strip():
                counts[item.strip()] += 1
    return dict(sorted(counts.items()))


def _criterion_status(checks: list[dict[str, Any]], *, data_available: bool) -> tuple[str, str]:
    if not data_available:
        return "INSUFFICIENT", "required analyzable data are unavailable"
    if not checks:
        return "INSUFFICIENT", "no externally supplied decision criteria"
    if any(check["observed"] is None for check in checks):
        return "INSUFFICIENT", "at least one supplied criterion cannot be evaluated"
    failed = [check["criterion"] for check in checks if not check["passed"]]
    if failed:
        return "FAILED", "failed supplied criteria: " + ", ".join(failed)
    return "SUPPORTED", "all externally supplied criteria passed"


def analyze_static_load(
    records: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze PRE/LOAD/POST cells without inventing missing force values."""

    criteria = dict(criteria or {})
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        identity = (
            row.get("session_id"),
            row.get("pose_id"),
            row.get("direction_id"),
            row.get("load_level_id"),
            row.get("repeat_id"),
            row.get("cell_id"),
        )
        groups[identity].append(row)

    cells: list[dict[str, Any]] = []
    total_usable = 0
    for identity, rows in groups.items():
        phase_raw: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
        phase_vectors: dict[str, list[list[float]]] = {phase: [] for phase in PHASES}
        phase_torque_vectors: dict[str, list[list[float]]] = {
            phase: [] for phase in PHASES
        }
        for row in rows:
            phase = str(row.get("phase") or "").upper()
            if phase not in PHASES:
                continue
            vector = _force_vector(row)
            torque_vector = _torque_vector(row)
            source_valid = _bool(row.get("valid")) is True
            state_valid = _bool(row.get("robot_state_valid"))
            usable = source_valid and state_valid is not False and all(value is not None for value in vector)
            phase_raw[phase].append(
                {
                    "force_n": vector,
                    "torque_nm": torque_vector,
                    "valid": usable,
                    "invalid_reason": row.get("invalid_reason") or None,
                    "query_duration_ms": _query_duration_ms(row),
                }
            )
            if usable:
                phase_vectors[phase].append([float(value) for value in vector])
                total_usable += 1
            if source_valid and state_valid is not False and all(
                value is not None for value in torque_vector
            ):
                phase_torque_vectors[phase].append(
                    [float(value) for value in torque_vector]
                )

        means = {phase: _vector_mean(phase_vectors[phase]) for phase in PHASES}
        complete = all(means[phase] is not None for phase in PHASES)
        baseline_mean: list[float] | None = None
        delta_force: list[float] | None = None
        drift: list[float] | None = None
        if complete:
            pre = np.asarray(means["PRE"], dtype=float)
            load = np.asarray(means["LOAD"], dtype=float)
            post = np.asarray(means["POST"], dtype=float)
            baseline_mean = ((pre + post) / 2.0).tolist()
            delta_force = (load - (pre + post) / 2.0).tolist()
            drift = (post - pre).tolist()
        baseline_vectors = phase_vectors["PRE"] + phase_vectors["POST"]
        baseline_std = _vector_std(baseline_vectors)
        torque_means = {
            phase: _vector_mean(phase_torque_vectors[phase]) for phase in PHASES
        }
        torque_complete = all(torque_means[phase] is not None for phase in PHASES)
        torque_baseline: list[float] | None = None
        torque_delta: list[float] | None = None
        torque_drift: list[float] | None = None
        if torque_complete:
            torque_pre = np.asarray(torque_means["PRE"], dtype=float)
            torque_load = np.asarray(torque_means["LOAD"], dtype=float)
            torque_post = np.asarray(torque_means["POST"], dtype=float)
            torque_baseline = ((torque_pre + torque_post) / 2.0).tolist()
            torque_delta = (torque_load - (torque_pre + torque_post) / 2.0).tolist()
            torque_drift = (torque_post - torque_pre).tolist()
        torque_baseline_vectors = (
            phase_torque_vectors["PRE"] + phase_torque_vectors["POST"]
        )
        reference, reference_consistent = _reference_vector(rows)
        sign_agreement: dict[str, bool | None] | None = None
        direction_agreement: float | None = None
        if delta_force is not None and reference is not None:
            sign_agreement = {}
            for index, component in enumerate(FORCE_COMPONENTS):
                if abs(reference[index]) <= 1.0e-12:
                    sign_agreement[component] = None
                else:
                    sign_agreement[component] = delta_force[index] * reference[index] > 0.0
            measured_norm = _vector_norm(delta_force)
            reference_norm = _vector_norm(reference)
            if measured_norm and reference_norm:
                direction_agreement = float(
                    np.dot(delta_force, reference) / (measured_norm * reference_norm)
                )
        counts = {
            phase: {
                "raw": len(phase_raw[phase]),
                "usable": len(phase_vectors[phase]),
                "missing_or_invalid": len(phase_raw[phase]) - len(phase_vectors[phase]),
            }
            for phase in PHASES
        }
        cells.append(
            {
                "cell_identity": {
                    "session_id": identity[0],
                    "pose_id": identity[1],
                    "direction_id": identity[2],
                    "load_level_id": identity[3],
                    "repeat_id": identity[4],
                    "cell_id": identity[5],
                },
                "complete_pre_load_post": complete,
                "sample_counts": counts,
                "raw_phase_samples": phase_raw,
                "pre_mean_force_n": means["PRE"],
                "load_mean_force_n": means["LOAD"],
                "post_mean_force_n": means["POST"],
                "baseline_bias_force_n": baseline_mean,
                "baseline_standard_deviation_force_n": baseline_std,
                "drift_post_minus_pre_force_n": drift,
                "measured_load_vector_n": delta_force,
                "pre_mean_torque_nm": torque_means["PRE"],
                "load_mean_torque_nm": torque_means["LOAD"],
                "post_mean_torque_nm": torque_means["POST"],
                "baseline_bias_torque_nm": torque_baseline,
                "baseline_standard_deviation_torque_nm": _vector_std(
                    torque_baseline_vectors
                ),
                "drift_post_minus_pre_torque_nm": torque_drift,
                "load_delta_torque_nm": torque_delta,
                "reference_load_vector_n": reference,
                "reference_load_consistent": reference_consistent,
                "sign_agreement_by_component": sign_agreement,
                "direction_cosine_agreement": direction_agreement,
                "query_timing_ms": _numeric_summary(_query_duration_ms(row) for row in rows),
            }
        )

    repeat_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        identity = cell["cell_identity"]
        key = (
            identity["session_id"],
            identity["pose_id"],
            identity["direction_id"],
            identity["load_level_id"],
        )
        if cell["measured_load_vector_n"] is not None:
            repeat_groups[key].append(cell)
    contrast_repeatability = []
    for key, group_cells in repeat_groups.items():
        vectors = [cell["measured_load_vector_n"] for cell in group_cells]
        norms = [_vector_norm(vector) for vector in vectors]
        contrast_repeatability.append(
            {
                "condition": {
                    "session_id": key[0],
                    "pose_id": key[1],
                    "direction_id": key[2],
                    "load_level_id": key[3],
                },
                "repeat_count": len(vectors),
                "mean_measured_load_vector_n": _vector_mean(vectors),
                "component_repeatability_standard_deviation_n": _vector_std(vectors),
                "load_norm_repeatability": _numeric_summary(norms),
            }
        )

    total_records = len(records)
    complete_fraction = (
        sum(cell["complete_pre_load_post"] for cell in cells) / len(cells)
        if cells
        else None
    )
    valid_fraction = total_usable / total_records if total_records else None
    direction_values = [cell["direction_cosine_agreement"] for cell in cells]
    direction_values = [value for value in direction_values if value is not None]
    sign_values = [
        value
        for cell in cells
        for value in (cell["sign_agreement_by_component"] or {}).values()
        if value is not None
    ]
    bias_norms = [_vector_norm(cell["baseline_bias_force_n"]) for cell in cells]
    std_norms = [_vector_norm(cell["baseline_standard_deviation_force_n"]) for cell in cells]
    drift_norms = [_vector_norm(cell["drift_post_minus_pre_force_n"]) for cell in cells]
    query_values = [_query_duration_ms(row) for row in records]

    checks: list[dict[str, Any]] = []

    def add_check(name: str, observed: Any, threshold: Any, passed: bool | None) -> None:
        checks.append(
            {
                "criterion": name,
                "observed": observed,
                "threshold": threshold,
                "passed": passed,
            }
        )

    if "minimum_valid_fraction" in criteria:
        threshold = _finite(criteria["minimum_valid_fraction"])
        add_check(
            "minimum_valid_fraction",
            valid_fraction,
            threshold,
            None if valid_fraction is None or threshold is None else valid_fraction >= threshold,
        )
    if criteria.get("require_complete_phases") is True:
        add_check("require_complete_phases", complete_fraction, 1.0, complete_fraction == 1.0)
    if "minimum_direction_cosine" in criteria:
        threshold = _finite(criteria["minimum_direction_cosine"])
        observed = min(direction_values) if direction_values else None
        add_check(
            "minimum_direction_cosine",
            observed,
            threshold,
            None if observed is None or threshold is None else observed >= threshold,
        )
    if criteria.get("require_sign_agreement") is True:
        observed = all(sign_values) if sign_values else None
        add_check("require_sign_agreement", observed, True, observed is True if observed is not None else None)
    for criterion, values, operator in (
        ("maximum_bias_norm_n", bias_norms, "maximum"),
        ("maximum_baseline_std_norm_n", std_norms, "maximum"),
        ("maximum_drift_norm_n", drift_norms, "maximum"),
        ("maximum_query_duration_ms", query_values, "maximum"),
    ):
        if criterion in criteria:
            threshold = _finite(criteria[criterion])
            usable = [value for value in values if value is not None]
            observed = max(usable) if usable else None
            add_check(
                criterion,
                observed,
                threshold,
                None if observed is None or threshold is None else observed <= threshold,
            )

    status, reason = _criterion_status(checks, data_available=bool(cells and total_usable))
    return {
        "level": "LEVEL_1_MEASUREMENT_VALIDITY",
        "criterion_status": status,
        "criterion_status_reason": reason,
        "criteria_source": "EXTERNALLY_SUPPLIED" if criteria else "NOT_SUPPLIED",
        "criteria": criteria,
        "criterion_checks": checks,
        "raw_record_count": total_records,
        "usable_record_count": total_usable,
        "missing_or_invalid_record_count": total_records - total_usable,
        "valid_fraction": valid_fraction,
        "invalid_reason_counts": _invalid_reason_counts(records),
        "complete_cell_fraction": complete_fraction,
        "query_timing_ms": _numeric_summary(query_values),
        "state_field_coverage": {
            field: sum(_finite(row.get(field)) is not None for row in records)
            for field in (
                "state_host_time_s",
                "tcp_x_m",
                "tcp_y_m",
                "tcp_z_m",
                "q1_rad",
                "q2_rad",
                "q3_rad",
                "q4_rad",
                "q5_rad",
                "q6_rad",
            )
        },
        "cells": cells,
        "known_load_contrast_repeatability": contrast_repeatability,
    }


def _metadata_value(episode: Mapping[str, Any], field: str) -> Any:
    if field in episode:
        return episode[field]
    metadata = episode.get("metadata")
    return metadata.get(field) if isinstance(metadata, Mapping) else None


def _channel_value(sample: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    return _first_finite(sample, aliases)


def _task_direction(task_direction: Mapping[str, Any] | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not task_direction:
        return None, {
            "available": False,
            "reason": "task direction not supplied; no Cartesian axis is assumed",
            "geometry_validated": False,
        }
    vector = task_direction.get("vector")
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        return None, {
            "available": False,
            "reason": "task direction vector is missing or malformed",
            "geometry_validated": bool(task_direction.get("geometry_validated") is True),
        }
    values = np.asarray([_finite(value) for value in vector], dtype=object)
    if values.shape != (3,) or any(value is None for value in values):
        return None, {
            "available": False,
            "reason": "task direction vector must contain three finite values",
            "geometry_validated": bool(task_direction.get("geometry_validated") is True),
        }
    numeric = values.astype(float)
    norm = float(np.linalg.norm(numeric))
    if norm <= 0.0:
        return None, {
            "available": False,
            "reason": "task direction vector has zero norm",
            "geometry_validated": bool(task_direction.get("geometry_validated") is True),
        }
    return numeric / norm, {
        "available": True,
        "reason": None,
        "unit_vector": (numeric / norm).tolist(),
        "frame": task_direction.get("frame"),
        "source": task_direction.get("source"),
        "geometry_validated": bool(task_direction.get("geometry_validated") is True),
        "decision_eligible": bool(task_direction.get("geometry_validated") is True),
    }


def _interpolate_curve(phases: Sequence[float], values: Sequence[float], grid: np.ndarray) -> list[float | None] | None:
    if len(phases) < 2:
        return None
    grouped: dict[float, list[float]] = defaultdict(list)
    for phase, value in zip(phases, values):
        grouped[float(phase)].append(float(value))
    ordered = sorted((phase, float(np.mean(items))) for phase, items in grouped.items())
    if len(ordered) < 2:
        return None
    source_phase = np.asarray([item[0] for item in ordered], dtype=float)
    source_value = np.asarray([item[1] for item in ordered], dtype=float)
    interpolated = np.interp(grid, source_phase, source_value, left=np.nan, right=np.nan)
    return [float(value) if math.isfinite(float(value)) else None for value in interpolated]


def extract_episode_features(
    episode: Mapping[str, Any],
    *,
    task_direction: Mapping[str, Any] | None = None,
    time_local_grid_size: int = 51,
) -> dict[str, Any]:
    """Extract full-cycle, branch, peak, and time-local response features."""

    if time_local_grid_size < 3:
        raise ValueError("time_local_grid_size must be at least 3")
    samples = episode.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        samples = []
    direction, direction_info = _task_direction(task_direction)
    grid = np.linspace(0.0, 1.0, time_local_grid_size)
    aliases = {**DIRECT_CHANNEL_ALIASES, **MODEL_DERIVED_CHANNEL_ALIASES}
    channel_values: dict[str, list[tuple[float | None, str | None, float]]] = defaultdict(list)
    invalid_reasons: Counter[str] = Counter()
    valid_sample_count = 0
    for raw_sample in samples:
        if not isinstance(raw_sample, Mapping):
            invalid_reasons["sample_not_mapping"] += 1
            continue
        valid = _bool(raw_sample.get("valid")) is True
        state_valid = _bool(raw_sample.get("state_valid", raw_sample.get("robot_state_valid")))
        if state_valid is False:
            valid = False
        if not valid:
            reasons = str(raw_sample.get("invalid_reason") or "sample_invalid").split(";")
            for reason in reasons:
                if reason.strip():
                    invalid_reasons[reason.strip()] += 1
            continue
        valid_sample_count += 1
        phase = _first_finite(raw_sample, ("phase", "trajectory_phase"))
        branch_value = str(raw_sample.get("branch") or "").upper()
        branch = branch_value if branch_value in BRANCHES else None
        direct_force = [_channel_value(raw_sample, DIRECT_CHANNEL_ALIASES[name]) for name in FORCE_COMPONENTS]
        for channel, channel_aliases in aliases.items():
            value = _channel_value(raw_sample, channel_aliases)
            if value is not None:
                channel_values[channel].append((phase, branch, value))
        if direction is not None and all(value is not None for value in direct_force):
            projection = float(np.dot(np.asarray(direct_force, dtype=float), direction))
            channel_values["task_direction_projection"].append((phase, branch, projection))

    features: dict[str, dict[str, float | None]] = {}
    curves: dict[str, list[float | None] | None] = {}
    provenance: dict[str, str] = {}
    for channel, observations in channel_values.items():
        values = np.asarray([item[2] for item in observations], dtype=float)
        flexion = [item[2] for item in observations if item[1] == "FLEXION"]
        extension = [item[2] for item in observations if item[1] == "EXTENSION"]
        features[channel] = {
            "full_cycle_rms": float(np.sqrt(np.mean(values**2))) if values.size else None,
            "flexion_rms": float(np.sqrt(np.mean(np.square(flexion)))) if flexion else None,
            "extension_rms": float(np.sqrt(np.mean(np.square(extension)))) if extension else None,
            "peak_abs": float(np.max(np.abs(values))) if values.size else None,
        }
        phases = [item[0] for item in observations if item[0] is not None]
        phased_values = [item[2] for item in observations if item[0] is not None]
        curves[channel] = _interpolate_curve(phases, phased_values, grid)
        if channel in MODEL_DERIVED_CHANNEL_ALIASES:
            provenance[channel] = "MODEL_DERIVED"
        elif channel == "task_direction_projection":
            provenance[channel] = "DERIVED_FROM_EXPLICIT_TASK_DIRECTION"
        else:
            provenance[channel] = "ROBOT_REPORTED_RAW_CHANNEL"

    required_metadata = {
        "episode_id": episode.get("episode_id"),
        "entity_id": episode.get("entity_id"),
        "entity_kind": episode.get("entity_kind"),
        "beta_flex": _finite(episode.get("beta_flex")),
        "beta_extend": _finite(episode.get("beta_extend")),
        "rom_profile_id": episode.get("rom_profile_id"),
        "rom_profile_version": episode.get("rom_profile_version"),
        "hardware_setup_id": episode.get("hardware_setup_id"),
        "measurement_source": episode.get("measurement_source"),
    }
    missing_metadata = [
        field
        for field, value in required_metadata.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    setup_metadata = {field: _metadata_value(episode, field) for field in SETUP_METADATA_FIELDS}
    return {
        **required_metadata,
        "metadata_complete": not missing_metadata,
        "missing_metadata_fields": missing_metadata,
        "setup_metadata": setup_metadata,
        "raw_sample_count": len(samples),
        "valid_sample_count": valid_sample_count,
        "invalid_sample_count": len(samples) - valid_sample_count,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "features": features,
        "channel_provenance": provenance,
        "time_local_phase_grid": grid.tolist(),
        "time_local_curves": curves,
        "task_direction_projection": direction_info,
        "analysis_valid": bool(not missing_metadata and valid_sample_count >= 2 and features),
    }


def _feature_rows(episode_feature: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for channel, metrics in episode_feature.get("features", {}).items():
        for metric, value in metrics.items():
            finite = _finite(value)
            if finite is not None:
                output[f"{channel}.{metric}"] = finite
    return output


def _decision_feature_is_eligible(
    feature_id: str, rows: Sequence[Mapping[str, Any]]
) -> bool:
    if not feature_id.startswith("task_direction_projection."):
        return True
    return bool(rows) and all(
        row.get("task_direction_projection", {}).get("decision_eligible") is True
        for row in rows
    )


def _condition_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("entity_id"),
        row.get("entity_kind"),
        row.get("beta_flex"),
        row.get("beta_extend"),
        row.get("rom_profile_id"),
        row.get("rom_profile_version"),
        row.get("hardware_setup_id"),
        row.get("measurement_source"),
    )


def _context_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    condition = _condition_key(row)
    return condition[:2] + condition[4:]


def _key_mapping(key: tuple[Any, ...], *, include_beta: bool) -> dict[str, Any]:
    fields = (
        "entity_id",
        "entity_kind",
        "beta_flex",
        "beta_extend",
        "rom_profile_id",
        "rom_profile_version",
        "hardware_setup_id",
        "measurement_source",
    ) if include_beta else (
        "entity_id",
        "entity_kind",
        "rom_profile_id",
        "rom_profile_version",
        "hardware_setup_id",
        "measurement_source",
    )
    return dict(zip(fields, key))


def _metadata_consistent(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    inconsistent = []
    for field in SETUP_METADATA_FIELDS:
        values = {
            json.dumps(row.get("setup_metadata", {}).get(field), sort_keys=True)
            for row in rows
        }
        if len(values) > 1:
            inconsistent.append(field)
    return not inconsistent, inconsistent


def _curve_pairs(rows: Sequence[Mapping[str, Any]], channel: str) -> dict[str, Any]:
    correlations: list[float] = []
    rms_differences: list[float] = []
    for left_index, left in enumerate(rows):
        left_curve = left.get("time_local_curves", {}).get(channel)
        if left_curve is None:
            continue
        for right in rows[left_index + 1 :]:
            right_curve = right.get("time_local_curves", {}).get(channel)
            if right_curve is None:
                continue
            left_values = np.asarray([np.nan if value is None else value for value in left_curve], dtype=float)
            right_values = np.asarray([np.nan if value is None else value for value in right_curve], dtype=float)
            mask = np.isfinite(left_values) & np.isfinite(right_values)
            if np.sum(mask) < 2:
                continue
            delta = left_values[mask] - right_values[mask]
            rms_differences.append(float(np.sqrt(np.mean(delta**2))))
            if np.std(left_values[mask]) > 0.0 and np.std(right_values[mask]) > 0.0:
                correlations.append(float(np.corrcoef(left_values[mask], right_values[mask])[0, 1]))
    return {
        "pair_count": len(rms_differences),
        "correlation": _numeric_summary(correlations),
        "rms_difference": _numeric_summary(rms_differences),
    }


def analyze_same_trajectory_repeatability(
    episode_features: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any] | None = None,
    *,
    upstream_status: str = "SUPPORTED",
) -> dict[str, Any]:
    """Group same-beta, same-ROM, same-setup episodes and quantify variation."""

    criteria = dict(criteria or {})
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_features:
        grouped[_condition_key(row)].append(row)
    group_results = []
    for key, rows in grouped.items():
        metadata_consistent, inconsistent_fields = _metadata_consistent(rows)
        valid_rows = [row for row in rows if row.get("analysis_valid") is True]
        feature_ids = sorted({feature for row in valid_rows for feature in _feature_rows(row)})
        variability = []
        for feature_id in feature_ids:
            values = [_feature_rows(row).get(feature_id) for row in valid_rows]
            usable = [value for value in values if value is not None]
            summary = _numeric_summary(usable)
            mean = summary["mean"]
            standard_deviation = summary["standard_deviation"]
            variability.append(
                {
                    "feature_id": feature_id,
                    **summary,
                    "coefficient_of_variation_abs": (
                        abs(standard_deviation / mean)
                        if standard_deviation is not None and mean not in (None, 0.0)
                        else None
                    ),
                }
            )
        channels = sorted(
            {
                channel
                for row in valid_rows
                for channel, curve in row.get("time_local_curves", {}).items()
                if curve is not None
            }
        )
        group_results.append(
            {
                "condition": _key_mapping(key, include_beta=True),
                "episode_ids": [row.get("episode_id") for row in rows],
                "repeat_count": len(rows),
                "valid_episode_count": len(valid_rows),
                "valid_episode_fraction": len(valid_rows) / len(rows) if rows else None,
                "metadata_consistent": metadata_consistent,
                "inconsistent_setup_metadata_fields": inconsistent_fields,
                "feature_variability": variability,
                "time_series_similarity": {
                    channel: _curve_pairs(valid_rows, channel) for channel in channels
                },
            }
        )

    eligible = [group for group in group_results if group["valid_episode_count"] >= 2]
    eligible_episode_rows = [
        row for row in episode_features if row.get("analysis_valid") is True
    ]
    decision_features = set(criteria.get("decision_feature_ids") or [])
    feature_cvs = [
        feature["coefficient_of_variation_abs"]
        for group in eligible
        for feature in group["feature_variability"]
        if feature["coefficient_of_variation_abs"] is not None
        and (not decision_features or feature["feature_id"] in decision_features)
        and _decision_feature_is_eligible(feature["feature_id"], eligible_episode_rows)
    ]
    correlations = [
        summary["correlation"]["minimum"]
        for group in eligible
        for channel, summary in group["time_series_similarity"].items()
        if summary["correlation"]["minimum"] is not None
        and (not decision_features or any(item.startswith(f"{channel}.") for item in decision_features))
    ]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, observed: Any, threshold: Any, passed: bool | None) -> None:
        checks.append({"criterion": name, "observed": observed, "threshold": threshold, "passed": passed})

    if "minimum_repeats_per_group" in criteria:
        threshold = _finite(criteria["minimum_repeats_per_group"])
        observed = min((group["valid_episode_count"] for group in group_results), default=None)
        add_check(name="minimum_repeats_per_group", observed=observed, threshold=threshold, passed=None if observed is None or threshold is None else observed >= threshold)
    if "minimum_valid_episode_fraction" in criteria:
        threshold = _finite(criteria["minimum_valid_episode_fraction"])
        observed = min((group["valid_episode_fraction"] for group in group_results), default=None)
        add_check(name="minimum_valid_episode_fraction", observed=observed, threshold=threshold, passed=None if observed is None or threshold is None else observed >= threshold)
    if criteria.get("require_metadata_consistency") is True:
        observed = all(group["metadata_consistent"] for group in group_results) if group_results else None
        add_check(name="require_metadata_consistency", observed=observed, threshold=True, passed=observed if observed is not None else None)
    if "maximum_feature_cv" in criteria:
        threshold = _finite(criteria["maximum_feature_cv"])
        observed = max(feature_cvs) if feature_cvs else None
        add_check(name="maximum_feature_cv", observed=observed, threshold=threshold, passed=None if observed is None or threshold is None else observed <= threshold)
    if "minimum_time_series_correlation" in criteria:
        threshold = _finite(criteria["minimum_time_series_correlation"])
        observed = min(correlations) if correlations else None
        add_check(name="minimum_time_series_correlation", observed=observed, threshold=threshold, passed=None if observed is None or threshold is None else observed >= threshold)

    if upstream_status != "SUPPORTED":
        status, reason = "INSUFFICIENT", "blocked by Level 1 measurement validity"
    else:
        status, reason = _criterion_status(checks, data_available=bool(eligible))
    return {
        "level": "LEVEL_2_SAME_TRAJECTORY_REPEATABILITY",
        "criterion_status": status,
        "criterion_status_reason": reason,
        "blocked_by": None if upstream_status == "SUPPORTED" else "MEASUREMENT_VALIDITY",
        "criteria_source": "EXTERNALLY_SUPPLIED" if criteria else "NOT_SUPPLIED",
        "criteria": criteria,
        "criterion_checks": checks,
        "same_condition_group_count": len(group_results),
        "repeatable_group_count": len(eligible),
        "groups": group_results,
    }


def _pooled_within_standard_deviation(groups: Sequence[Sequence[float]]) -> float | None:
    usable = [np.asarray(group, dtype=float) for group in groups if len(group) >= 2]
    degrees = sum(len(group) - 1 for group in usable)
    if degrees <= 0:
        return None
    numerator = sum((len(group) - 1) * float(np.var(group, ddof=1)) for group in usable)
    return math.sqrt(numerator / degrees)


def _trajectory_feature_effect(feature_id: str, beta_rows: Mapping[tuple[float, float], Sequence[Mapping[str, Any]]]) -> dict[str, Any] | None:
    values_by_beta: dict[tuple[float, float], list[float]] = {}
    for beta, rows in beta_rows.items():
        values = [_feature_rows(row).get(feature_id) for row in rows]
        usable = [value for value in values if value is not None]
        if usable:
            values_by_beta[beta] = usable
    if len(values_by_beta) < 2:
        return None
    means = {beta: float(np.mean(values)) for beta, values in values_by_beta.items()}
    best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    maximum_delta = -1.0
    beta_items = list(means)
    for index, left in enumerate(beta_items):
        for right in beta_items[index + 1 :]:
            delta = abs(means[left] - means[right])
            if delta > maximum_delta:
                maximum_delta = delta
                best_pair = (left, right)
    within = _pooled_within_standard_deviation(list(values_by_beta.values()))
    snr = maximum_delta / within if within is not None and within > 0.0 else None
    return {
        "feature_id": feature_id,
        "condition_means": [
            {"beta": list(beta), "mean": means[beta], "repeat_count": len(values_by_beta[beta])}
            for beta in beta_items
        ],
        "maximum_between_trajectory_effect": maximum_delta,
        "maximum_effect_beta_pair": [list(best_pair[0]), list(best_pair[1])] if best_pair else None,
        "pooled_within_trajectory_standard_deviation": within,
        "trajectory_snr": snr,
        "snr_unavailable_reason": (
            "zero_or_unavailable_within_trajectory_variability" if snr is None else None
        ),
    }


def _time_local_effect(channel: str, beta_rows: Mapping[tuple[float, float], Sequence[Mapping[str, Any]]]) -> dict[str, Any] | None:
    curves_by_beta: dict[tuple[float, float], list[np.ndarray]] = {}
    grid: np.ndarray | None = None
    for beta, rows in beta_rows.items():
        curves = []
        for row in rows:
            curve = row.get("time_local_curves", {}).get(channel)
            if curve is None:
                continue
            values = np.asarray([np.nan if value is None else value for value in curve], dtype=float)
            curves.append(values)
            if grid is None:
                grid = np.asarray(row["time_local_phase_grid"], dtype=float)
        if curves:
            curves_by_beta[beta] = curves
    if len(curves_by_beta) < 2 or grid is None:
        return None
    means = {beta: np.nanmean(np.vstack(curves), axis=0) for beta, curves in curves_by_beta.items()}
    best: tuple[tuple[float, float], tuple[float, float], int, float] | None = None
    betas = list(means)
    for index, left in enumerate(betas):
        for right in betas[index + 1 :]:
            difference = np.abs(means[left] - means[right])
            finite_indices = np.flatnonzero(np.isfinite(difference))
            if not finite_indices.size:
                continue
            local_index = int(finite_indices[np.argmax(difference[finite_indices])])
            effect = float(difference[local_index])
            if best is None or effect > best[3]:
                best = (left, right, local_index, effect)
    if best is None:
        return None
    within_groups = []
    for curves in curves_by_beta.values():
        values = [curve[best[2]] for curve in curves if math.isfinite(float(curve[best[2]]))]
        within_groups.append(values)
    within = _pooled_within_standard_deviation(within_groups)
    return {
        "channel": channel,
        "phase_grid": grid.tolist(),
        "condition_mean_curves": [
            {
                "beta": list(beta),
                "mean_curve": [float(value) if math.isfinite(float(value)) else None for value in curve],
            }
            for beta, curve in means.items()
        ],
        "maximum_time_local_effect": best[3],
        "phase_at_maximum_effect": float(grid[best[2]]),
        "maximum_effect_beta_pair": [list(best[0]), list(best[1])],
        "pooled_within_standard_deviation_at_maximum": within,
        "time_local_snr": best[3] / within if within is not None and within > 0.0 else None,
    }


def analyze_trajectory_sensitivity(
    episode_features: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any] | None = None,
    *,
    upstream_measurement_status: str = "SUPPORTED",
    upstream_repeatability_status: str = "SUPPORTED",
) -> dict[str, Any]:
    """Compare preselected beta conditions using between/within effect ratios."""

    criteria = dict(criteria or {})
    contexts: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_features:
        contexts[_context_key(row)].append(row)
    results = []
    for key, rows in contexts.items():
        valid_rows = [row for row in rows if row.get("analysis_valid") is True]
        beta_rows: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            beta_rows[(float(row["beta_flex"]), float(row["beta_extend"]))].append(row)
        feature_ids = sorted({feature for row in valid_rows for feature in _feature_rows(row)})
        effects = [
            effect
            for feature_id in feature_ids
            if (effect := _trajectory_feature_effect(feature_id, beta_rows)) is not None
        ]
        channels = sorted(
            {
                channel
                for row in valid_rows
                for channel, curve in row.get("time_local_curves", {}).items()
                if curve is not None
            }
        )
        local_effects = [
            effect
            for channel in channels
            if (effect := _time_local_effect(channel, beta_rows)) is not None
        ]
        metadata_consistent, inconsistent_fields = _metadata_consistent(valid_rows)
        results.append(
            {
                "context": _key_mapping(key, include_beta=False),
                "beta_condition_count": len(beta_rows),
                "repeat_count_by_beta": [
                    {"beta": list(beta), "repeat_count": len(beta_group)}
                    for beta, beta_group in beta_rows.items()
                ],
                "metadata_consistent": metadata_consistent,
                "inconsistent_setup_metadata_fields": inconsistent_fields,
                "feature_effects": effects,
                "time_local_effects": local_effects,
            }
        )

    analyzable = [result for result in results if result["beta_condition_count"] >= 2]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, observed: Any, threshold: Any, passed: bool | None) -> None:
        checks.append({"criterion": name, "observed": observed, "threshold": threshold, "passed": passed})

    if "minimum_beta_conditions" in criteria:
        threshold = _finite(criteria["minimum_beta_conditions"])
        observed = min((result["beta_condition_count"] for result in results), default=None)
        add_check("minimum_beta_conditions", observed, threshold, None if observed is None or threshold is None else observed >= threshold)
    if "minimum_repeats_per_beta" in criteria:
        threshold = _finite(criteria["minimum_repeats_per_beta"])
        observed = min(
            (
                item["repeat_count"]
                for result in analyzable
                for item in result["repeat_count_by_beta"]
            ),
            default=None,
        )
        add_check("minimum_repeats_per_beta", observed, threshold, None if observed is None or threshold is None else observed >= threshold)
    if criteria.get("require_metadata_consistency") is True:
        observed = all(result["metadata_consistent"] for result in analyzable) if analyzable else None
        add_check("require_metadata_consistency", observed, True, observed if observed is not None else None)
    if "minimum_trajectory_snr" in criteria:
        threshold = _finite(criteria["minimum_trajectory_snr"])
        decision_features = list(criteria.get("decision_feature_ids") or [])
        observed_by_feature = {}
        for feature_id in decision_features:
            rows_for_feature = [
                row
                for context_rows in contexts.values()
                for row in context_rows
                if row.get("analysis_valid") is True
            ]
            values = (
                [
                    effect["trajectory_snr"]
                    for result in analyzable
                    for effect in result["feature_effects"]
                    if effect["feature_id"] == feature_id
                    and effect["trajectory_snr"] is not None
                ]
                if _decision_feature_is_eligible(feature_id, rows_for_feature)
                else []
            )
            observed_by_feature[feature_id] = min(values) if values else None
        observed = min(observed_by_feature.values()) if observed_by_feature and all(value is not None for value in observed_by_feature.values()) else None
        add_check(
            "minimum_trajectory_snr_for_predeclared_features",
            {"minimum": observed, "by_feature": observed_by_feature} if observed_by_feature else None,
            threshold,
            None if observed is None or threshold is None else observed >= threshold,
        )

    if upstream_measurement_status != "SUPPORTED":
        status, reason = "INSUFFICIENT", "blocked by Level 1 measurement validity"
        blocked_by = "MEASUREMENT_VALIDITY"
    elif upstream_repeatability_status != "SUPPORTED":
        status, reason = "INSUFFICIENT", "blocked by Level 2 same-trajectory repeatability"
        blocked_by = "SAME_TRAJECTORY_REPEATABILITY"
    else:
        status, reason = _criterion_status(checks, data_available=bool(analyzable))
        blocked_by = None
    return {
        "level": "LEVEL_3_TRAJECTORY_SENSITIVITY",
        "criterion_status": status,
        "criterion_status_reason": reason,
        "blocked_by": blocked_by,
        "criteria_source": "EXTERNALLY_SUPPLIED" if criteria else "NOT_SUPPLIED",
        "criteria": criteria,
        "criterion_checks": checks,
        "context_count": len(results),
        "analyzable_context_count": len(analyzable),
        "contexts": results,
    }


def run_validation_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run all three levels while preserving the evidence hierarchy."""

    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {INPUT_SCHEMA_VERSION}")
    evidence_classification = str(payload.get("evidence_classification") or "UNSPECIFIED")
    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), Mapping) else {}
    static_records = payload.get("static_records")
    episodes = payload.get("episodes")
    if not isinstance(static_records, Sequence) or isinstance(static_records, (str, bytes)):
        static_records = []
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        episodes = []
    task_direction = payload.get("task_direction")
    if not isinstance(task_direction, Mapping):
        task_direction = None

    level_1 = analyze_static_load(static_records, criteria.get("level_1"))
    episode_features = [
        extract_episode_features(episode, task_direction=task_direction)
        for episode in episodes
        if isinstance(episode, Mapping)
    ]
    level_2 = analyze_same_trajectory_repeatability(
        episode_features,
        criteria.get("level_2"),
        upstream_status=level_1["criterion_status"],
    )
    level_3 = analyze_trajectory_sensitivity(
        episode_features,
        criteria.get("level_3"),
        upstream_measurement_status=level_1["criterion_status"],
        upstream_repeatability_status=level_2["criterion_status"],
    )
    criterion_statuses = {
        "MEASUREMENT_VALIDITY": level_1["criterion_status"],
        "SAME_TRAJECTORY_REPEATABILITY": level_2["criterion_status"],
        "TRAJECTORY_SENSITIVITY": level_3["criterion_status"],
    }
    is_real = evidence_classification == "REAL_MEASUREMENT"
    public_statuses = (
        criterion_statuses
        if is_real
        else {name: "INSUFFICIENT" for name in criterion_statuses}
    )
    ready = is_real and all(status == "SUPPORTED" for status in public_statuses.values())
    direction_info = _task_direction(task_direction)[1]
    return {
        "analysis_id": ANALYSIS_ID,
        ANALYSIS_ID: "IMPLEMENTED_WITH_LIMITATIONS",
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "evidence_classification": evidence_classification,
        "not_real_measurement_evidence": not is_real,
        "MEASUREMENT_VALIDITY": public_statuses["MEASUREMENT_VALIDITY"],
        "SAME_TRAJECTORY_REPEATABILITY": public_statuses[
            "SAME_TRAJECTORY_REPEATABILITY"
        ],
        "TRAJECTORY_SENSITIVITY": public_statuses["TRAJECTORY_SENSITIVITY"],
        "criterion_statuses_before_evidence_classification": criterion_statuses,
        "FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY": ready,
        "future_subject_trajectory_gate_reason": (
            "all three real-measurement levels are supported"
            if ready
            else "requires real evidence with Level 1, Level 2, and Level 3 all supported"
        ),
        "personalization_conclusion_emitted": False,
        "robot_connection_attempted": False,
        "robot_action_count": 0,
        "personalization_algorithm_runs": 0,
        "PINN_training_runs": 0,
        "task_direction_projection": direction_info,
        "measurement_provenance": {
            "raw_force_torque_channels": "ROBOT_REPORTED_RAW_CHANNEL",
            "task_direction_projection": "DERIVED_ONLY_WHEN_EXPLICIT_DIRECTION_IS_SUPPLIED",
            "hip_knee_decomposition": "MODEL_DERIVED",
            "physical_semantics_require_real_validation": True,
        },
        "level_1": level_1,
        "episode_features": episode_features,
        "level_2": level_2,
        "level_3": level_3,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_analysis_input(path: str | Path) -> dict[str, Any]:
    """Load a normalized JSON input and any referenced offline CSV/JSON files."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analysis input must be a JSON object")
    base = source.parent
    static_records = list(payload.get("static_records") or [])
    for item in payload.get("static_csv_paths") or []:
        static_records.extend(_read_csv((base / item).resolve()))
    episodes = list(payload.get("episodes") or [])
    for item in payload.get("episode_json_paths") or []:
        episode_payload = json.loads((base / item).resolve().read_text(encoding="utf-8"))
        if isinstance(episode_payload, list):
            episodes.extend(episode_payload)
        else:
            episodes.append(episode_payload)
    payload["static_records"] = static_records
    payload["episodes"] = episodes
    return payload


def _json_cell(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _json_cell(value) if isinstance(value, (dict, list, tuple)) else value
                    for field, value in row.items()
                }
            )


def write_analysis_outputs(result: Mapping[str, Any], output_directory: str | Path) -> dict[str, str]:
    """Write one JSON summary plus compact tables; never modify source logs."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "validation_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    static_rows = [
        {
            **cell["cell_identity"],
            "complete_pre_load_post": cell["complete_pre_load_post"],
            "measured_load_vector_n": cell["measured_load_vector_n"],
            "reference_load_vector_n": cell["reference_load_vector_n"],
            "direction_cosine_agreement": cell["direction_cosine_agreement"],
            "baseline_bias_force_n": cell["baseline_bias_force_n"],
            "baseline_standard_deviation_force_n": cell[
                "baseline_standard_deviation_force_n"
            ],
            "drift_post_minus_pre_force_n": cell["drift_post_minus_pre_force_n"],
            "load_delta_torque_nm": cell["load_delta_torque_nm"],
            "baseline_bias_torque_nm": cell["baseline_bias_torque_nm"],
            "baseline_standard_deviation_torque_nm": cell[
                "baseline_standard_deviation_torque_nm"
            ],
            "drift_post_minus_pre_torque_nm": cell[
                "drift_post_minus_pre_torque_nm"
            ],
        }
        for cell in result["level_1"]["cells"]
    ]
    episode_rows = [
        {
            "episode_id": row["episode_id"],
            "entity_id": row["entity_id"],
            "beta_flex": row["beta_flex"],
            "beta_extend": row["beta_extend"],
            "rom_profile_id": row["rom_profile_id"],
            "rom_profile_version": row["rom_profile_version"],
            "hardware_setup_id": row["hardware_setup_id"],
            "analysis_valid": row["analysis_valid"],
            "features": row["features"],
            "channel_provenance": row["channel_provenance"],
        }
        for row in result["episode_features"]
    ]
    repeatability_rows = [
        {
            **group["condition"],
            "repeat_count": group["repeat_count"],
            "valid_episode_count": group["valid_episode_count"],
            "metadata_consistent": group["metadata_consistent"],
            "feature_variability": group["feature_variability"],
            "time_series_similarity": group["time_series_similarity"],
        }
        for group in result["level_2"]["groups"]
    ]
    effect_rows = [
        {**context["context"], **effect}
        for context in result["level_3"]["contexts"]
        for effect in context["feature_effects"]
    ]
    local_rows = [
        {**context["context"], **effect}
        for context in result["level_3"]["contexts"]
        for effect in context["time_local_effects"]
    ]
    table_paths = {
        "static_cells": output / "level_1_static_cells.csv",
        "episode_features": output / "episode_features.csv",
        "repeatability_groups": output / "level_2_repeatability_groups.csv",
        "trajectory_effects": output / "level_3_trajectory_effects.csv",
        "time_local_effects": output / "level_3_time_local_effects.csv",
    }
    for name, rows in (
        ("static_cells", static_rows),
        ("episode_features", episode_rows),
        ("repeatability_groups", repeatability_rows),
        ("trajectory_effects", effect_rows),
        ("time_local_effects", local_rows),
    ):
        _write_csv(table_paths[name], rows)
    return {
        "summary": str(summary_path),
        **{name: str(path) for name, path in table_paths.items() if path.exists()},
    }


def build_demo_input() -> dict[str, Any]:
    """Create deterministic synthetic data that exercises all analysis levels."""

    static_records = []
    for repeat in range(1, 4):
        for phase_index, phase in enumerate(PHASES):
            for sample_index in range(5):
                baseline = 0.02 * (repeat - 2) + 0.005 * sample_index
                load = 9.8 if phase == "LOAD" else 0.0
                drift = 0.04 if phase == "POST" else 0.0
                static_records.append(
                    {
                        "session_id": "demo_static",
                        "pose_id": "POSE_A",
                        "direction_id": "POSITIVE_Z",
                        "load_level_id": "KNOWN_1KG_EQUIVALENT",
                        "repeat_id": repeat,
                        "cell_id": f"demo_cell_{repeat}",
                        "phase": phase,
                        "phase_sample_index": sample_index,
                        "fx_raw_n": baseline,
                        "fy_raw_n": -0.5 * baseline,
                        "fz_raw_n": baseline + drift + load,
                        "reference_load_vector_n": [0.0, 0.0, 9.8],
                        "query_latency_ms": 3.0 + 0.1 * phase_index,
                        "state_host_time_s": repeat * 100 + phase_index + sample_index / 10,
                        "tcp_x_m": 0.3,
                        "tcp_y_m": 0.0,
                        "tcp_z_m": 0.4,
                        "q1_rad": 0.1,
                        "valid": True,
                        "robot_state_valid": True,
                        "invalid_reason": "",
                    }
                )

    episodes = []
    beta_conditions = ((0.0, 0.0), (0.03, -0.03), (-0.03, 0.03))
    phases = np.linspace(0.0, 1.0, 81)
    for beta_index, beta in enumerate(beta_conditions):
        for repeat in range(3):
            samples = []
            for sample_index, phase in enumerate(phases):
                repeat_noise = 0.015 * math.sin(2.0 * math.pi * phase + repeat)
                contrast = beta_index * 0.35 * math.exp(-((phase - 0.35) / 0.09) ** 2)
                samples.append(
                    {
                        "time_s": float(sample_index * 0.05),
                        "phase": float(phase),
                        "branch": "FLEXION" if phase < 0.5 else "EXTENSION",
                        "fx": 1.0 + 0.1 * math.sin(2.0 * math.pi * phase) + repeat_noise,
                        "fy": -0.2 + 0.03 * math.cos(2.0 * math.pi * phase),
                        "fz": 5.0 + math.sin(math.pi * phase) + contrast + repeat_noise,
                        "mx": 0.10 * math.sin(2.0 * math.pi * phase),
                        "my": 0.08 * math.cos(2.0 * math.pi * phase),
                        "mz": 0.05 * math.sin(4.0 * math.pi * phase),
                        "hip_response": 2.0 + 0.4 * math.sin(math.pi * phase) + contrast,
                        "knee_response": 1.5 + 0.3 * math.sin(math.pi * phase),
                        "valid": True,
                        "state_valid": True,
                        "invalid_reason": "",
                    }
                )
            episodes.append(
                {
                    "episode_id": f"demo_beta_{beta_index}_repeat_{repeat}",
                    "entity_id": "DUMMY_LEG_DEMO",
                    "entity_kind": "DUMMY_LEG",
                    "beta_flex": beta[0],
                    "beta_extend": beta[1],
                    "rom_profile_id": "DEMO_ROM",
                    "rom_profile_version": "V1",
                    "hardware_setup_id": "DEMO_SETUP_A",
                    "measurement_source": "SYNTHETIC_DEMO_GENERATOR",
                    "robot_model": "DEMO_ONLY",
                    "robot_serial_number": "DEMO_ONLY",
                    "controller_version": "DEMO_ONLY",
                    "active_tool_name": "DEMO_TOOL",
                    "active_workobject_name": "DEMO_WORKOBJECT",
                    "tcp_id": "DEMO_TCP",
                    "payload_id": "DEMO_PAYLOAD",
                    "samples": samples,
                }
            )
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "evidence_classification": "OFFLINE_SYNTHETIC_DEMO_ONLY",
        "static_records": static_records,
        "episodes": episodes,
        "task_direction": None,
        "criteria": {
            "level_1": {
                "minimum_valid_fraction": 0.95,
                "require_complete_phases": True,
                "minimum_direction_cosine": 0.99,
                "require_sign_agreement": True,
                "maximum_baseline_std_norm_n": 0.05,
                "maximum_drift_norm_n": 0.10,
                "maximum_query_duration_ms": 10.0,
            },
            "level_2": {
                "minimum_repeats_per_group": 3,
                "minimum_valid_episode_fraction": 1.0,
                "require_metadata_consistency": True,
                "decision_feature_ids": ["fz.full_cycle_rms"],
                "maximum_feature_cv": 0.01,
                "minimum_time_series_correlation": 0.99,
            },
            "level_3": {
                "minimum_beta_conditions": 3,
                "minimum_repeats_per_beta": 3,
                "require_metadata_consistency": True,
                "decision_feature_ids": ["fz.full_cycle_rms"],
                "minimum_trajectory_snr": 2.0,
            },
        },
    }
