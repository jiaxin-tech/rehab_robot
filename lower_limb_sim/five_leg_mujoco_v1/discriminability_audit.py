"""Diagnostic-only V3 trajectory and objective discriminability audit."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from personalization.rom_gated_v2 import SubjectSpecificV3CandidateDomain

from .benchmark import build_leg_domain
from .model import (
    FrozenBenchmarkDefinition,
    MechanicalLegDefinition,
    load_frozen_benchmark_definition,
    make_mujoco_model,
)
from .replay import custom_passive_torque_project_coordinates


PACKAGE_DIR = Path(__file__).resolve().parent
PRIMARY_RESULTS = PACKAGE_DIR / "results" / "benchmark_summary.json"
AUDIT_ID = "V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1"
SELECTED_BETAS = (
    (0.0, 0.0),
    (-0.03, 0.03),
    (0.03, -0.03),
    (-0.03, -0.03),
    (0.03, 0.03),
)
COMPONENT_NAMES = (
    "inertia",
    "gravity",
    "coriolis",
    "passive_stiffness",
    "damping",
    "coupling",
    "joint_limit_constraint",
)
SECONDARY_METRICS = (
    "full_rms_nm",
    "full_peak_nm",
    "flexion_rms_nm",
    "extension_rms_nm",
    "hip_rms_nm",
    "knee_rms_nm",
    "integrated_squared_load_nm2_s",
)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _time_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(values**2, time_s) / duration))


def _trapezoid_weights(time_s: np.ndarray) -> np.ndarray:
    intervals = np.diff(time_s)
    weights = np.empty_like(time_s)
    weights[0] = intervals[0] / 2.0
    weights[-1] = intervals[-1] / 2.0
    weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    return weights / float(time_s[-1] - time_s[0])


def _pairwise_rms_summary(values: np.ndarray) -> dict[str, float]:
    distances = pdist(np.asarray(values, dtype=float), metric="euclidean")
    distances /= math.sqrt(values.shape[1])
    return {
        "pairwise_rms_median": float(np.median(distances)),
        "pairwise_rms_p95": float(np.percentile(distances, 95.0)),
        "pairwise_rms_maximum": float(np.max(distances)),
        "maximum_instantaneous_separation": float(
            np.max(np.ptp(values, axis=0))
        ),
    }


def _candidate_for_beta(
    domain: SubjectSpecificV3CandidateDomain,
    beta: tuple[float, float],
):
    return next(candidate for candidate in domain if candidate.beta == beta)


def trajectory_separability(
    leg: MechanicalLegDefinition,
    domain: SubjectSpecificV3CandidateDomain,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    candidates = tuple(domain)
    q = np.stack([candidate.trajectory.q for candidate in candidates])
    dq = np.stack([candidate.trajectory.dq for candidate in candidates])
    ddq = np.stack([candidate.trajectory.ddq for candidate in candidates])
    time_s = domain.subject_reference.time_s
    summary: dict[str, Any] = {"leg_id": leg.leg_id}
    for name, values in (("q", q), ("dq", dq), ("ddq", ddq)):
        for joint_index, joint_name in enumerate(("hip", "knee")):
            stats = _pairwise_rms_summary(values[:, :, joint_index])
            for key, value in stats.items():
                summary[f"{name}_{joint_name}_{key}"] = value
            angular_unit = {"q": "deg", "dq": "deg_s", "ddq": "deg_s2"}[name]
            for key, value in stats.items():
                summary[f"{name}_{joint_name}_{key}_{angular_unit}"] = math.degrees(
                    value
                )
            reference_values = values[domain.reference.candidate_index, :, joint_index]
            reference_peak = float(np.max(np.abs(reference_values)))
            summary[f"{name}_{joint_name}_reference_peak_abs"] = reference_peak
            summary[f"{name}_{joint_name}_maximum_separation_over_reference_peak"] = (
                stats["maximum_instantaneous_separation"] / reference_peak
                if reference_peak > 0.0
                else 0.0
            )

    weights = _trapezoid_weights(time_s)
    integrated_path_distance = np.zeros(len(candidates) * (len(candidates) - 1) // 2)
    maximum_path_separation = 0.0
    for sample_index, weight in enumerate(weights):
        separation = pdist(q[:, sample_index, :], metric="euclidean")
        integrated_path_distance += weight * separation
        maximum_path_separation = max(
            maximum_path_separation, float(np.max(separation))
        )
    summary.update(
        {
            "joint_path_integrated_distance_mean_rad": float(
                np.mean(integrated_path_distance)
            ),
            "joint_path_integrated_distance_median_rad": float(
                np.median(integrated_path_distance)
            ),
            "joint_path_integrated_distance_p95_rad": float(
                np.percentile(integrated_path_distance, 95.0)
            ),
            "joint_path_maximum_separation_rad": maximum_path_separation,
        }
    )

    reference = _candidate_for_beta(domain, (0.0, 0.0))
    selected_rows = []
    for beta in SELECTED_BETAS[1:]:
        candidate = _candidate_for_beta(domain, beta)
        delta = candidate.trajectory.q - reference.trajectory.q
        pointwise = np.linalg.norm(delta, axis=1)
        selected_rows.append(
            {
                "leg_id": leg.leg_id,
                "comparison": f"reference_to_beta_{beta[0]:+.3f}_{beta[1]:+.3f}",
                "beta_flex": beta[0],
                "beta_extend": beta[1],
                "hip_rms_angle_difference_rad": _rms(delta[:, 0]),
                "knee_rms_angle_difference_rad": _rms(delta[:, 1]),
                "joint_path_mean_distance_rad": float(
                    np.trapezoid(pointwise, time_s)
                    / (time_s[-1] - time_s[0])
                ),
                "joint_path_rms_distance_rad": _time_rms(pointwise, time_s),
                "joint_path_max_distance_rad": float(np.max(pointwise)),
            }
        )
    curves = {
        "time_s": np.asarray(time_s),
        "q_spread": np.ptp(q, axis=0),
        "dq_spread": np.ptp(dq, axis=0),
        "ddq_spread": np.ptp(ddq, axis=0),
    }
    return summary, selected_rows, curves


def _crossing_time(
    time_s: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    if values[-1] >= values[0]:
        xp = values
        fp = time_s
    else:
        xp = values[::-1]
        fp = time_s[::-1]
    return float(np.interp(target, xp, fp))


def timing_effects(
    leg: MechanicalLegDefinition,
    domain: SubjectSpecificV3CandidateDomain,
) -> dict[str, Any]:
    q = np.stack([candidate.trajectory.q for candidate in domain])
    dq = np.stack([candidate.trajectory.dq for candidate in domain])
    ddq = np.stack([candidate.trajectory.ddq for candidate in domain])
    time_s = np.asarray(domain.subject_reference.time_s)
    phases = np.asarray(domain.subject_reference.phases)
    output: dict[str, Any] = {"leg_id": leg.leg_id}
    progression_times: dict[tuple[str, str, float], np.ndarray] = {}
    for branch in ("flexion", "extension"):
        mask = phases == branch
        branch_time = time_s[mask]
        for joint_index, joint_name in enumerate(("hip", "knee")):
            values = q[:, mask, joint_index]
            low = np.min(values, axis=1)
            high = np.max(values, axis=1)
            for progression in (0.25, 0.5, 0.75):
                if branch == "flexion":
                    targets = low + progression * (high - low)
                else:
                    targets = high - progression * (high - low)
                crossings = np.asarray(
                    [
                        _crossing_time(branch_time, item, target)
                        for item, target in zip(values, targets)
                    ]
                )
                progression_times[(branch, joint_name, progression)] = crossings
                output[
                    f"{branch}_{joint_name}_progression_{int(progression * 100)}_timing_range_s"
                ] = float(np.ptp(crossings))
        relative = (
            progression_times[(branch, "knee", 0.5)]
            - progression_times[(branch, "hip", 0.5)]
        )
        output[f"{branch}_hip_knee_half_progression_phase_range_s"] = float(
            np.ptp(relative)
        )
        output[f"{branch}_hip_knee_half_progression_phase_max_abs_s"] = float(
            np.max(np.abs(relative))
        )
        for derivative_name, values in (("velocity", dq), ("acceleration", ddq)):
            for joint_index, joint_name in enumerate(("hip", "knee")):
                branch_values = np.abs(values[:, mask, joint_index])
                peak_times = branch_time[np.argmax(branch_values, axis=1)]
                output[
                    f"{branch}_{joint_name}_peak_{derivative_name}_timing_range_s"
                ] = float(np.ptp(peak_times))

    reference_index = domain.reference.candidate_index
    for label, beta in (
        ("oracle_boundary", (-0.03, 0.03)),
        ("opposite_boundary", (0.03, -0.03)),
    ):
        candidate_index = _candidate_for_beta(domain, beta).candidate_index
        for branch in ("flexion", "extension"):
            for progression in (0.25, 0.5, 0.75):
                times = progression_times[(branch, "knee", progression)]
                output[
                    f"{label}_{branch}_knee_{int(progression * 100)}_timing_shift_vs_reference_s"
                ] = float(times[candidate_index] - times[reference_index])
    return output


def _project_from_mujoco(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray((values[0], -values[1]), dtype=float)


def _branch_time_rms(
    values: np.ndarray,
    time_s: np.ndarray,
    mask: np.ndarray,
) -> float:
    branch_time = time_s[mask]
    branch_values = values[mask]
    return _time_rms(branch_values, branch_time)


def decompose_candidate_response(
    *,
    model: mujoco.MjModel,
    leg: MechanicalLegDefinition,
    time_s: np.ndarray,
    phases: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Strictly decompose the unchanged inverse-dynamics endpoint.

    MuJoCo provides mass, bias, and passive terms.  Linear/cubic stiffness,
    damping, and custom coupling are split using the frozen declared model
    coefficients.  The returned reconstruction residual verifies that no load
    was invented or omitted.
    """

    time = np.asarray(time_s, dtype=float)
    project_q = np.asarray(q, dtype=float)
    project_dq = np.asarray(dq, dtype=float)
    project_ddq = np.asarray(ddq, dtype=float)
    data = mujoco.MjData(model)
    total = np.empty_like(project_q)
    components = {
        name: np.empty_like(project_q) for name in COMPONENT_NAMES
    }
    mass_matrix = np.empty((model.nv, model.nv), dtype=float)

    for index in range(len(time)):
        q_hip, q_knee = project_q[index]
        dq_hip, dq_knee = project_dq[index]
        ddq_hip, ddq_knee = project_ddq[index]
        q_mj = np.asarray((q_hip, -q_knee), dtype=float)
        dq_mj = np.asarray((dq_hip, -dq_knee), dtype=float)
        ddq_mj = np.asarray((ddq_hip, -ddq_knee), dtype=float)

        data.qpos[:] = q_mj
        data.qvel[:] = dq_mj
        data.qacc[:] = ddq_mj
        custom_passive = custom_passive_torque_project_coordinates(
            leg, q_hip, q_knee
        )
        data.qfrc_applied[:] = (custom_passive[0], -custom_passive[1])
        mujoco.mj_inverse(model, data)
        total[index] = _project_from_mujoco(
            np.asarray(data.qfrc_inverse - data.qfrc_applied)
        )
        actual_bias_mj = np.asarray(data.qfrc_bias).copy()
        constraint_mj = np.asarray(data.qfrc_constraint).copy()
        try:
            # MuJoCo >=3.4 exposes M through MjData and accepts the data object.
            mujoco.mj_fullM(model, data, mass_matrix)
        except TypeError:  # pragma: no cover - compatibility with older 3.x
            mujoco.mj_fullM(model, mass_matrix, data.qM)
        components["inertia"][index] = _project_from_mujoco(
            mass_matrix @ ddq_mj
        )

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity_mj = np.asarray(data.qfrc_bias).copy()
        components["gravity"][index] = _project_from_mujoco(gravity_mj)
        components["coriolis"][index] = _project_from_mujoco(
            actual_bias_mj - gravity_mj
        )

        hip_displacement = q_hip - leg.hip_neutral_rad
        knee_displacement = q_knee - leg.knee_neutral_rad
        components["passive_stiffness"][index] = (
            leg.hip_stiffness_nm_per_rad * hip_displacement
            + leg.hip_cubic_stiffness_nm_per_rad3 * hip_displacement**3,
            leg.knee_stiffness_nm_per_rad * knee_displacement
            + leg.knee_cubic_stiffness_nm_per_rad3 * knee_displacement**3,
        )
        components["damping"][index] = (
            leg.hip_damping_nm_s_per_rad * dq_hip,
            leg.knee_damping_nm_s_per_rad * dq_knee,
        )
        coupling_displacement = (
            hip_displacement - leg.coupling_ratio * knee_displacement
        )
        components["coupling"][index] = (
            leg.coupling_stiffness_nm_per_rad * coupling_displacement,
            -leg.coupling_stiffness_nm_per_rad
            * leg.coupling_ratio
            * coupling_displacement,
        )
        # At exact frozen ROM extrema MuJoCo may activate its declared joint
        # limits.  In inverse dynamics this contribution enters with a minus
        # sign relative to qfrc_constraint.
        components["joint_limit_constraint"][index] = _project_from_mujoco(
            -constraint_mj
        )

    reconstructed = sum(components.values(), start=np.zeros_like(total))
    residual = total - reconstructed
    total_norm = np.linalg.norm(total, axis=1)
    metrics: dict[str, Any] = {
        "full_rms_nm": _time_rms(total_norm, time),
        "full_peak_nm": float(np.max(total_norm)),
        "integrated_load_nm_s": float(np.trapezoid(total_norm, time)),
        "integrated_squared_load_nm2_s": float(
            np.trapezoid(total_norm**2, time)
        ),
        "hip_rms_nm": _time_rms(total[:, 0], time),
        "knee_rms_nm": _time_rms(total[:, 1], time),
        "hip_peak_nm": float(np.max(np.abs(total[:, 0]))),
        "knee_peak_nm": float(np.max(np.abs(total[:, 1]))),
        "component_reconstruction_rms_nm": _rms(residual),
        "component_reconstruction_max_abs_nm": float(np.max(np.abs(residual))),
    }
    for name, values in components.items():
        component_norm = np.linalg.norm(values, axis=1)
        metrics[f"{name}_rms_nm"] = _time_rms(component_norm, time)
        metrics[f"{name}_peak_nm"] = float(np.max(component_norm))
        metrics[f"{name}_available"] = True

    for branch in ("flexion", "extension"):
        mask = np.asarray(phases) == branch
        branch_time = time[mask]
        branch_total = total[mask]
        branch_norm = total_norm[mask]
        metrics[f"{branch}_rms_nm"] = _time_rms(branch_norm, branch_time)
        metrics[f"{branch}_peak_nm"] = float(np.max(branch_norm))
        metrics[f"{branch}_integrated_load_nm_s"] = float(
            np.trapezoid(branch_norm, branch_time)
        )
        metrics[f"{branch}_integrated_squared_load_nm2_s"] = float(
            np.trapezoid(branch_norm**2, branch_time)
        )
        for joint_index, joint_name in enumerate(("hip", "knee")):
            metrics[f"{branch}_{joint_name}_rms_nm"] = _time_rms(
                branch_total[:, joint_index], branch_time
            )
            metrics[f"{branch}_{joint_name}_peak_nm"] = float(
                np.max(np.abs(branch_total[:, joint_index]))
            )

    arrays = {"total": total, **components}
    arrays["passive"] = components["passive_stiffness"] + components["coupling"]
    return metrics, arrays


def _relative_range(values: np.ndarray, reference_index: int) -> dict[str, Any]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    reference = float(values[reference_index])
    absolute_range = maximum - minimum
    return {
        "minimum": minimum,
        "maximum": maximum,
        "reference": reference,
        "absolute_range": absolute_range,
        "relative_range_by_reference": (
            absolute_range / reference
            if abs(reference) > 1.0e-12
            else (0.0 if absolute_range <= 1.0e-12 else None)
        ),
        "relative_range_by_reference_defined": abs(reference) > 1.0e-12
        or absolute_range <= 1.0e-12,
        "coefficient_of_variation": (
            float(np.std(values) / np.mean(values))
            if abs(float(np.mean(values))) > 1.0e-12
            else 0.0
        ),
    }


def decompose_leg_landscape(
    definition: FrozenBenchmarkDefinition,
    leg: MechanicalLegDefinition,
    domain: SubjectSpecificV3CandidateDomain,
    primary_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    model = make_mujoco_model(definition, leg)
    time_s = np.asarray(domain.subject_reference.time_s)
    phases = np.asarray(domain.subject_reference.phases)
    truth_by_index = {row["candidate_index"]: row for row in primary_rows}
    rows: list[dict[str, Any]] = []
    response_curves = {
        name: np.empty((len(domain), len(time_s)), dtype=float)
        for name in ("total", "passive", "inertia")
    }
    for candidate in domain:
        metrics, arrays = decompose_candidate_response(
            model=model,
            leg=leg,
            time_s=time_s,
            phases=phases,
            q=candidate.trajectory.q,
            dq=candidate.trajectory.dq,
            ddq=candidate.trajectory.ddq,
        )
        frozen_endpoint = float(
            truth_by_index[candidate.candidate_index]["endpoint_value_nm"]
        )
        rows.append(
            {
                "leg_id": leg.leg_id,
                "candidate_id": candidate.candidate_id,
                "canonical_beta_id": candidate.canonical_beta_id,
                "candidate_index": candidate.candidate_index,
                "beta_flex": candidate.beta_flex,
                "beta_extend": candidate.beta_extend,
                **metrics,
                "frozen_primary_endpoint_nm": frozen_endpoint,
                "endpoint_recompute_abs_error_nm": abs(
                    metrics["full_rms_nm"] - frozen_endpoint
                ),
            }
        )
        for name in response_curves:
            response_curves[name][candidate.candidate_index] = np.linalg.norm(
                arrays[name], axis=1
            )

    reference_index = domain.reference.candidate_index
    component_sensitivity = []
    for name in COMPONENT_NAMES:
        field = f"{name}_rms_nm"
        values = np.asarray([row[field] for row in rows], dtype=float)
        best = rows[int(np.argmin(values))]
        component_sensitivity.append(
            {
                "leg_id": leg.leg_id,
                "component": name,
                "oracle_beta": [best["beta_flex"], best["beta_extend"]],
                **_relative_range(values, reference_index),
            }
        )

    time_local: dict[str, Any] = {"leg_id": leg.leg_id}
    curves: dict[str, np.ndarray] = {"time_s": time_s}
    for name, matrix in response_curves.items():
        spread = np.ptp(matrix, axis=0)
        peak_index = int(np.argmax(spread))
        weights = _trapezoid_weights(time_s)
        weighted_spread = weights * spread
        top_count = max(1, int(math.ceil(0.10 * len(time_s))))
        top_indices = np.argpartition(weighted_spread, -top_count)[-top_count:]
        time_local.update(
            {
                f"{name}_largest_separation_nm": float(spread[peak_index]),
                f"{name}_largest_separation_time_s": float(time_s[peak_index]),
                f"{name}_half_max_duration_fraction": float(
                    np.mean(spread >= 0.5 * spread[peak_index])
                ),
                f"{name}_top_10_percent_time_share_of_integrated_spread": float(
                    np.sum(weighted_spread[top_indices])
                    / np.sum(weighted_spread)
                ),
            }
        )
        curves[f"{name}_spread"] = spread
    time_local["maximum_endpoint_recompute_abs_error_nm"] = max(
        row["endpoint_recompute_abs_error_nm"] for row in rows
    )
    time_local["maximum_component_reconstruction_abs_error_nm"] = max(
        row["component_reconstruction_max_abs_nm"] for row in rows
    )
    return rows, component_sensitivity, time_local, curves


def characterize_landscape_dynamic_range(
    leg_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["candidate_index"])
    values = np.asarray([row["endpoint_value_nm"] for row in ordered], dtype=float)
    reference_index = next(
        index
        for index, row in enumerate(ordered)
        if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
    )
    result: dict[str, Any] = {
        "leg_id": leg_id,
        "J_min_nm": float(np.min(values)),
        "J_max_nm": float(np.max(values)),
        "J_reference_nm": float(values[reference_index]),
        "absolute_range_nm": float(np.ptp(values)),
        "relative_range_by_reference": float(np.ptp(values) / values[reference_index]),
        "coefficient_of_variation": float(np.std(values) / np.mean(values)),
    }
    oracle = float(np.min(values))
    for tolerance in (0.001, 0.005, 0.01, 0.02, 0.05):
        result[f"within_{tolerance * 100:g}_percent_oracle_count"] = int(
            np.sum(values <= oracle * (1.0 + tolerance))
        )
    return result


def _grid(
    rows: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flex = np.asarray(sorted({float(row["beta_flex"]) for row in rows}))
    extend = np.asarray(sorted({float(row["beta_extend"]) for row in rows}))
    values = np.empty((len(flex), len(extend)), dtype=float)
    flex_index = {value: index for index, value in enumerate(flex)}
    extend_index = {value: index for index, value in enumerate(extend)}
    for row in rows:
        values[flex_index[float(row["beta_flex"])]][
            extend_index[float(row["beta_extend"])]
        ] = float(row[field])
    return flex, extend, values


def gradient_and_shape_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    per_leg: list[dict[str, Any]] = []
    fields: dict[str, dict[str, np.ndarray]] = {}
    for leg_id, rows in rows_by_leg.items():
        flex, extend, raw = _grid(rows, "endpoint_value_nm")
        normalized = (raw - np.min(raw)) / np.ptp(raw)
        raw_gf, raw_ge = np.gradient(raw, flex, extend, edge_order=2)
        gf, ge = np.gradient(normalized, flex, extend, edge_order=2)
        hff = np.gradient(gf, flex, axis=0, edge_order=2)
        hee = np.gradient(ge, extend, axis=1, edge_order=2)
        mixed = 0.5 * (
            np.gradient(gf, extend, axis=1, edge_order=2)
            + np.gradient(ge, flex, axis=0, edge_order=2)
        )
        local_minima = []
        for first in range(1, len(flex) - 1):
            for second in range(1, len(extend) - 1):
                neighborhood = normalized[first - 1 : first + 2, second - 1 : second + 2]
                if normalized[first, second] < np.min(np.delete(neighborhood.ravel(), 4)):
                    local_minima.append((float(flex[first]), float(extend[second])))
        reference_i = int(np.flatnonzero(np.isclose(flex, 0.0))[0])
        reference_j = int(np.flatnonzero(np.isclose(extend, 0.0))[0])
        gradient_magnitude = np.hypot(raw_gf, raw_ge)
        normalized_magnitude = np.hypot(gf, ge)
        oracle_flat = int(np.argmin(raw))
        oracle_i, oracle_j = np.unravel_index(oracle_flat, raw.shape)
        per_leg.append(
            {
                "leg_id": leg_id,
                "mean_gradient_magnitude_nm_per_beta": float(np.mean(gradient_magnitude)),
                "median_gradient_magnitude_nm_per_beta": float(np.median(gradient_magnitude)),
                "maximum_gradient_magnitude_nm_per_beta": float(np.max(gradient_magnitude)),
                "reference_gradient_flex_nm_per_beta": float(raw_gf[reference_i, reference_j]),
                "reference_gradient_extend_nm_per_beta": float(raw_ge[reference_i, reference_j]),
                "fraction_gradient_points_descend_toward_negative_flex": float(
                    np.mean(raw_gf > 0.0)
                ),
                "fraction_gradient_points_descend_toward_positive_extend": float(
                    np.mean(raw_ge < 0.0)
                ),
                "fraction_gradient_points_with_joint_boundary_direction": float(
                    np.mean((raw_gf > 0.0) & (raw_ge < 0.0))
                ),
                "normalized_mean_gradient_magnitude_per_beta": float(
                    np.mean(normalized_magnitude)
                ),
                "reference_normalized_curvature_flex_per_beta2": float(
                    hff[reference_i, reference_j]
                ),
                "reference_normalized_curvature_extend_per_beta2": float(
                    hee[reference_i, reference_j]
                ),
                "oracle_normalized_hessian_eigenvalues_per_beta2": np.linalg.eigvalsh(
                    np.asarray(
                        [
                            [hff[oracle_i, oracle_j], mixed[oracle_i, oracle_j]],
                            [mixed[oracle_i, oracle_j], hee[oracle_i, oracle_j]],
                        ]
                    )
                ).tolist(),
                "interior_strict_local_minimum_count": len(local_minima),
                "interior_strict_local_minima": [list(item) for item in local_minima],
                "global_oracle_beta": [float(flex[oracle_i]), float(extend[oracle_j])],
            }
        )
        fields[leg_id] = {
            "beta_flex": flex,
            "beta_extend": extend,
            "normalized": normalized,
            "gradient_flex": gf,
            "gradient_extend": ge,
            "curvature_trace": hff + hee,
        }

    pairwise: list[dict[str, Any]] = []
    for first_id, second_id in combinations(rows_by_leg, 2):
        first = fields[first_id]
        second = fields[second_id]
        first_gradient = np.concatenate(
            (first["gradient_flex"].ravel(), first["gradient_extend"].ravel())
        )
        second_gradient = np.concatenate(
            (second["gradient_flex"].ravel(), second["gradient_extend"].ravel())
        )
        denominator = np.linalg.norm(first_gradient) * np.linalg.norm(second_gradient)
        pairwise.append(
            {
                "leg_a": first_id,
                "leg_b": second_id,
                "normalized_landscape_spearman": float(
                    spearmanr(first["normalized"].ravel(), second["normalized"].ravel()).statistic
                ),
                "normalized_gradient_field_cosine_similarity": float(
                    np.dot(first_gradient, second_gradient) / denominator
                ),
                "normalized_curvature_trace_spearman": float(
                    spearmanr(
                        first["curvature_trace"].ravel(),
                        second["curvature_trace"].ravel(),
                    ).statistic
                ),
            }
        )
    return per_leg, pairwise, fields


def secondary_metric_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oracle_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    leg_ids = list(rows_by_leg)
    for leg_id, rows in rows_by_leg.items():
        primary = np.asarray([row["full_rms_nm"] for row in rows])
        for metric in SECONDARY_METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            best_index = int(np.argmin(values))
            best = rows[best_index]
            reference_index = next(
                index
                for index, row in enumerate(rows)
                if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
            )
            oracle_rows.append(
                {
                    "leg_id": leg_id,
                    "metric": metric,
                    "mechanical_meaning": {
                        "full_rms_nm": "unchanged primary vector RMS",
                        "full_peak_nm": "maximum instantaneous joint-torque-vector norm",
                        "flexion_rms_nm": "branch-only vector RMS during flexion",
                        "extension_rms_nm": "branch-only vector RMS during extension",
                        "hip_rms_nm": "full-cycle hip required-torque RMS",
                        "knee_rms_nm": "full-cycle knee required-torque RMS",
                        "integrated_squared_load_nm2_s": "time integral of squared joint-torque-vector norm",
                    }[metric],
                    "oracle_beta": [best["beta_flex"], best["beta_extend"]],
                    "oracle_candidate_index": best["candidate_index"],
                    "oracle_value": float(values[best_index]),
                    "reference_value": float(values[reference_index]),
                    "relative_range_by_reference": float(
                        np.ptp(values) / values[reference_index]
                    ),
                    "spearman_vs_primary": float(spearmanr(values, primary).statistic),
                    "differs_from_primary_oracle": best_index != int(np.argmin(primary)),
                }
            )

    for metric in SECONDARY_METRICS:
        matrix = np.asarray(
            [[row[metric] for row in rows_by_leg[leg_id]] for leg_id in leg_ids],
            dtype=float,
        )
        normalized = (matrix - np.min(matrix, axis=1, keepdims=True)) / np.ptp(
            matrix, axis=1, keepdims=True
        )
        common = np.mean(normalized, axis=0)
        interaction = normalized - common
        correlations = [
            float(spearmanr(normalized[a], normalized[b]).statistic)
            for a, b in combinations(range(len(leg_ids)), 2)
        ]
        metric_oracles = {
            int(np.argmin(matrix[index])) for index in range(len(leg_ids))
        }
        interaction_rows.append(
            {
                "metric": metric,
                "unique_cross_leg_oracle_count": len(metric_oracles),
                "minimum_pairwise_spearman": min(correlations),
                "median_pairwise_spearman": float(np.median(correlations)),
                "normalized_subject_by_trajectory_interaction_rms": _rms(interaction),
                "normalized_subject_by_trajectory_interaction_max_abs": float(
                    np.max(np.abs(interaction))
                ),
            }
        )
    return oracle_rows, interaction_rows


def joint_branch_ordering_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for leg_id, rows in rows_by_leg.items():
        values = {
            field: np.asarray([row[field] for row in rows], dtype=float)
            for field in (
                "full_rms_nm",
                "hip_rms_nm",
                "knee_rms_nm",
                "flexion_rms_nm",
                "extension_rms_nm",
            )
        }

        def beta_at_minimum(field: str) -> list[float]:
            best = rows[int(np.argmin(values[field]))]
            return [float(best["beta_flex"]), float(best["beta_extend"])]

        output.append(
            {
                "leg_id": leg_id,
                "primary_oracle_beta": beta_at_minimum("full_rms_nm"),
                "hip_oracle_beta": beta_at_minimum("hip_rms_nm"),
                "knee_oracle_beta": beta_at_minimum("knee_rms_nm"),
                "flexion_oracle_beta": beta_at_minimum("flexion_rms_nm"),
                "extension_oracle_beta": beta_at_minimum("extension_rms_nm"),
                "hip_vs_knee_spearman": float(
                    spearmanr(values["hip_rms_nm"], values["knee_rms_nm"]).statistic
                ),
                "flexion_vs_extension_spearman": float(
                    spearmanr(
                        values["flexion_rms_nm"], values["extension_rms_nm"]
                    ).statistic
                ),
                "hip_vs_primary_spearman": float(
                    spearmanr(values["hip_rms_nm"], values["full_rms_nm"]).statistic
                ),
                "knee_vs_primary_spearman": float(
                    spearmanr(values["knee_rms_nm"], values["full_rms_nm"]).statistic
                ),
                "flexion_vs_primary_spearman": float(
                    spearmanr(
                        values["flexion_rms_nm"], values["full_rms_nm"]
                    ).statistic
                ),
                "extension_vs_primary_spearman": float(
                    spearmanr(
                        values["extension_rms_nm"], values["full_rms_nm"]
                    ).statistic
                ),
            }
        )
    return output


def _diagnostic_conclusion(
    trajectory_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    dynamic_range_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    secondary_oracles: list[dict[str, Any]],
    gradient_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    max_knee_angle_deg = max(
        row["q_knee_maximum_instantaneous_separation_deg"]
        for row in trajectory_rows
    )
    max_timing_displacement_s = max(
        max(
            value
            for key, value in row.items()
            if "progression" in key and key.endswith("timing_range_s")
        )
        for row in timing_rows
    )
    all_within_five_percent = all(
        row["within_5_percent_oracle_count"] == 625
        for row in dynamic_range_rows
    )
    median_primary_relative_range = float(
        np.median([row["relative_range_by_reference"] for row in dynamic_range_rows])
    )
    maximum_component_relative_range = max(
        row["relative_range_by_reference"]
        for row in component_rows
        if row["reference"] > 1.0e-12
        and row["component"] != "joint_limit_constraint"
    )
    maximum_secondary_relative_range = max(
        row["relative_range_by_reference"] for row in secondary_oracles
    )
    minimum_gradient_cosine = min(
        row["normalized_gradient_field_cosine_similarity"]
        for row in gradient_pairs
    )

    trajectory_variation_material = (
        max_knee_angle_deg >= 5.0 and max_timing_displacement_s >= 0.1
    )
    primary_endpoint_flat = (
        all_within_five_percent and median_primary_relative_range < 0.02
    )
    decomposed_response_more_sensitive = (
        maximum_component_relative_range >= 0.05
        or maximum_secondary_relative_range >= 0.05
    )
    normalized_gradients_highly_consistent = minimum_gradient_cosine >= 0.9

    if not trajectory_variation_material:
        limitation = "V3_TRAJECTORY_FAMILY_INSUFFICIENTLY_DISCRIMINATIVE"
        direction = "REDESIGN_TRAJECTORY_FAMILY"
    elif primary_endpoint_flat and decomposed_response_more_sensitive:
        limitation = "MECHANICAL_ENDPOINT_INSUFFICIENTLY_DISCRIMINATIVE"
        direction = "REVISIT_MECHANICAL_ENDPOINT"
    elif not primary_endpoint_flat and normalized_gradients_highly_consistent:
        limitation = "CURRENT_MECHANICAL_SYSTEM_SUPPORTS_NEAR_UNIVERSAL_COORDINATION_OPTIMUM"
        direction = "PIVOT_TO_UNIVERSAL_MECHANICAL_TRAJECTORY_OPTIMIZATION"
    else:
        limitation = "MIXED"
        direction = "WAIT_FOR_REAL_MEASURED_SUBJECT_DATA"

    return {
        "trajectory_variation_material": trajectory_variation_material,
        "primary_endpoint_flat": primary_endpoint_flat,
        "decomposed_response_more_sensitive": decomposed_response_more_sensitive,
        "normalized_gradients_highly_consistent": normalized_gradients_highly_consistent,
        "maximum_knee_angle_separation_deg": max_knee_angle_deg,
        "maximum_timing_displacement_s": max_timing_displacement_s,
        "all_625_candidates_within_five_percent_for_every_leg": all_within_five_percent,
        "median_primary_relative_range": median_primary_relative_range,
        "maximum_component_relative_range": maximum_component_relative_range,
        "maximum_secondary_relative_range": maximum_secondary_relative_range,
        "minimum_normalized_gradient_field_cosine_similarity": minimum_gradient_cosine,
        "PRIMARY_LIMITATION": limitation,
        "NEXT_SCIENTIFIC_DIRECTION": direction,
    }


def load_primary_results() -> dict[str, Any]:
    payload = json.loads(PRIMARY_RESULTS.read_text(encoding="utf-8"))
    if payload["FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1"] != "COMPLETE":
        raise RuntimeError("frozen five-leg benchmark is not complete")
    if payload["total_candidate_replays"] != 5 * 625:
        raise RuntimeError("frozen benchmark must contain exactly 5 x 625 rows")
    if payload["coordinate_convention"] != "theta_shank = q_hip - q_knee":
        raise RuntimeError("coordinate convention changed")
    return payload


def run_discriminability_audit() -> dict[str, Any]:
    definition = load_frozen_benchmark_definition()
    primary = load_primary_results()
    primary_rows_by_leg = {
        leg.leg_id: [
            row for row in primary["landscape_rows"] if row["leg_id"] == leg.leg_id
        ]
        for leg in definition.legs
    }
    domains: dict[str, SubjectSpecificV3CandidateDomain] = {}
    trajectory_rows: list[dict[str, Any]] = []
    selected_path_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    trajectory_curves: dict[str, dict[str, np.ndarray]] = {}
    response_rows_by_leg: dict[str, list[dict[str, Any]]] = {}
    component_rows: list[dict[str, Any]] = []
    time_local_rows: list[dict[str, Any]] = []
    time_local_curves: dict[str, dict[str, np.ndarray]] = {}

    for leg in definition.legs:
        domain = build_leg_domain(leg)
        domains[leg.leg_id] = domain
        separation, paths, curves = trajectory_separability(leg, domain)
        trajectory_rows.append(separation)
        selected_path_rows.extend(paths)
        trajectory_curves[leg.leg_id] = curves
        timing_rows.append(timing_effects(leg, domain))
        responses, components, time_local, local_curves = decompose_leg_landscape(
            definition,
            leg,
            domain,
            primary_rows_by_leg[leg.leg_id],
        )
        response_rows_by_leg[leg.leg_id] = responses
        component_rows.extend(components)
        time_local_rows.append(time_local)
        time_local_curves[leg.leg_id] = local_curves

    dynamic_range_rows = [
        characterize_landscape_dynamic_range(leg_id, rows)
        for leg_id, rows in primary_rows_by_leg.items()
    ]
    gradient_rows, gradient_pairs, gradient_fields = gradient_and_shape_analysis(
        primary_rows_by_leg
    )
    secondary_oracles, secondary_interactions = secondary_metric_analysis(
        response_rows_by_leg
    )
    joint_branch = joint_branch_ordering_analysis(response_rows_by_leg)
    conclusion = _diagnostic_conclusion(
        trajectory_rows,
        timing_rows,
        dynamic_range_rows,
        component_rows,
        secondary_oracles,
        gradient_pairs,
    )
    recompute_error = max(
        row["maximum_endpoint_recompute_abs_error_nm"] for row in time_local_rows
    )
    reconstruction_error = max(
        row["maximum_component_reconstruction_abs_error_nm"]
        for row in time_local_rows
    )
    status = (
        "COMPLETE"
        if recompute_error <= 1.0e-10 and reconstruction_error <= 1.0e-10
        else "COMPLETE_WITH_LIMITATIONS"
    )
    return {
        "audit_id": AUDIT_ID,
        AUDIT_ID: status,
        "classification": "OFFLINE_SYNTHETIC_MUJOCO_DIAGNOSTIC_ONLY",
        "source_benchmark_id": definition.benchmark_id,
        "source_benchmark_preserved": True,
        "frozen_leg_count": 5,
        "frozen_candidate_count_per_leg": 625,
        "diagnostic_candidate_replays": 5 * 625,
        "coordinate_convention": "theta_shank = q_hip - q_knee",
        "primary_endpoint_unchanged": True,
        "beta_grid_unchanged": True,
        "V3_operator_unchanged": True,
        "trajectory_separability": trajectory_rows,
        "selected_path_separation": selected_path_rows,
        "timing_effects": timing_rows,
        "primary_landscape_dynamic_range": dynamic_range_rows,
        "component_sensitivity": component_rows,
        "time_local_sensitivity": time_local_rows,
        "gradient_and_shape_by_leg": gradient_rows,
        "cross_leg_normalized_shape": gradient_pairs,
        "joint_branch_ordering": joint_branch,
        "secondary_metric_oracles": secondary_oracles,
        "secondary_metric_cross_leg_interaction": secondary_interactions,
        "component_decomposition": {
            "strictly_available": list(COMPONENT_NAMES),
            "unavailable": [],
            "method": (
                "MuJoCo M(q)qdd plus bias split into gravity and Coriolis, "
                "with frozen declared linear/cubic stiffness, damping, and "
                "custom hip-knee coupling; verified by exact reconstruction"
            ),
            "maximum_endpoint_recompute_abs_error_nm": recompute_error,
            "maximum_reconstruction_abs_error_nm": reconstruction_error,
        },
        "diagnostic_conclusion": conclusion,
        "PRIMARY_LIMITATION": conclusion["PRIMARY_LIMITATION"],
        "NEXT_SCIENTIFIC_DIRECTION": conclusion["NEXT_SCIENTIFIC_DIRECTION"],
        "personalization_algorithm_development_supported": False,
        "algorithm_runs": 0,
        "robot_actions": 0,
        "PINN_training": 0,
        "response_rows": [
            row
            for leg_id in response_rows_by_leg
            for row in response_rows_by_leg[leg_id]
        ],
        "_plot_data": {
            "domains": domains,
            "trajectory_curves": trajectory_curves,
            "time_local_curves": time_local_curves,
            "gradient_fields": gradient_fields,
        },
    }


__all__ = [
    "AUDIT_ID",
    "COMPONENT_NAMES",
    "PRIMARY_RESULTS",
    "SECONDARY_METRICS",
    "characterize_landscape_dynamic_range",
    "decompose_candidate_response",
    "decompose_leg_landscape",
    "gradient_and_shape_analysis",
    "joint_branch_ordering_analysis",
    "load_primary_results",
    "run_discriminability_audit",
    "secondary_metric_analysis",
    "timing_effects",
    "trajectory_separability",
]
