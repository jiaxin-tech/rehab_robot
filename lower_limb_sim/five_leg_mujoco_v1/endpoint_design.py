"""Mechanically predefined endpoint comparison for the frozen five-leg study."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .benchmark import build_leg_domain
from .discriminability_audit import decompose_candidate_response
from .model import load_frozen_benchmark_definition, make_mujoco_model


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_AUDIT = (
    PACKAGE_DIR / "results_discriminability_audit_v1" / "audit_summary.json"
)
STUDY_ID = "DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1"
FEATURE_VECTOR_ID = "MECHANICAL_ENDPOINT_FEATURE_VECTOR_V1"
FEATURE_VECTOR_FIELDS = (
    "relative_hip_flexion_rms",
    "relative_hip_extension_rms",
    "relative_knee_flexion_rms",
    "relative_knee_extension_rms",
    "relative_hip_flexion_peak",
    "relative_hip_extension_peak",
    "relative_knee_flexion_peak",
    "relative_knee_extension_peak",
)
ENDPOINT_IDS = ("E0", "E1", "E2", "E3", "E4")
NEAR_OPTIMAL_TOLERANCES = (0.001, 0.005, 0.01, 0.02, 0.05)
ENDPOINT_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "E0": {
        "name": "FULL_CYCLE_DUAL_JOINT_AGGREGATE_RMS",
        "unit": "N_m",
        "formula": "sqrt(integral(tau_hip^2 + tau_knee^2) dt / duration)",
        "mechanical_interpretation": "historical full-cycle dual-joint aggregate load baseline",
    },
    "E1": {
        "name": "JOINT_BALANCED_RELATIVE_RMS",
        "unit": "reference_ratio",
        "formula": "max(r_hip_RMS, r_knee_RMS)",
        "mechanical_interpretation": "minimize the worse relative joint RMS without hip-knee cancellation",
    },
    "E2": {
        "name": "BRANCH_BALANCED_RELATIVE_RMS",
        "unit": "reference_ratio",
        "formula": "max(r_hip_flex_RMS, r_hip_extend_RMS, r_knee_flex_RMS, r_knee_extend_RMS)",
        "mechanical_interpretation": "minimize the worse joint-by-movement-branch relative RMS",
    },
    "E3": {
        "name": "JOINT_BALANCED_RELATIVE_PEAK",
        "unit": "reference_ratio",
        "formula": "max(r_hip_peak, r_knee_peak)",
        "mechanical_interpretation": "minimize the worse relative full-cycle joint peak",
    },
    "E4": {
        "name": "BRANCH_BALANCED_RELATIVE_PEAK",
        "unit": "reference_ratio",
        "formula": "max(r_hip_flex_peak, r_hip_extend_peak, r_knee_flex_peak, r_knee_extend_peak)",
        "mechanical_interpretation": "minimize the worse relative joint-by-branch peak",
    },
}


def _time_rms(values: np.ndarray, time_s: np.ndarray) -> float:
    duration = float(time_s[-1] - time_s[0])
    return float(np.sqrt(np.trapezoid(np.asarray(values) ** 2, time_s) / duration))


def load_source_audit() -> dict[str, Any]:
    payload = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if payload["V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1"] != "COMPLETE":
        raise RuntimeError("source discriminability audit must be complete")
    if payload["PRIMARY_LIMITATION"] != "MECHANICAL_ENDPOINT_INSUFFICIENTLY_DISCRIMINATIVE":
        raise RuntimeError("source audit does not authorize endpoint redesign")
    if len(payload["response_rows"]) != 5 * 625:
        raise RuntimeError("source response table must contain exactly 5 x 625 rows")
    return payload


def _response_metrics_from_torque(
    tau: np.ndarray,
    time_s: np.ndarray,
    phases: np.ndarray,
) -> dict[str, float]:
    torque = np.asarray(tau, dtype=float)
    time = np.asarray(time_s, dtype=float)
    norm = np.linalg.norm(torque, axis=1)
    metrics = {
        "full_rms_nm": _time_rms(norm, time),
        "full_peak_nm": float(np.max(norm)),
        "hip_rms_nm": _time_rms(torque[:, 0], time),
        "knee_rms_nm": _time_rms(torque[:, 1], time),
        "hip_peak_nm": float(np.max(np.abs(torque[:, 0]))),
        "knee_peak_nm": float(np.max(np.abs(torque[:, 1]))),
    }
    for branch in ("flexion", "extension"):
        mask = np.asarray(phases) == branch
        branch_time = time[mask]
        branch_torque = torque[mask]
        metrics[f"{branch}_rms_nm"] = _time_rms(
            np.linalg.norm(branch_torque, axis=1), branch_time
        )
        for joint_index, joint_name in enumerate(("hip", "knee")):
            values = branch_torque[:, joint_index]
            metrics[f"{branch}_{joint_name}_rms_nm"] = _time_rms(
                values, branch_time
            )
            metrics[f"{branch}_{joint_name}_peak_nm"] = float(
                np.max(np.abs(values))
            )
    return metrics


def _relative_feature_mapping(
    metrics: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    mapping = {
        "relative_hip_flexion_rms": (
            metrics["flexion_hip_rms_nm"] / reference["flexion_hip_rms_nm"]
        ),
        "relative_hip_extension_rms": (
            metrics["extension_hip_rms_nm"] / reference["extension_hip_rms_nm"]
        ),
        "relative_knee_flexion_rms": (
            metrics["flexion_knee_rms_nm"] / reference["flexion_knee_rms_nm"]
        ),
        "relative_knee_extension_rms": (
            metrics["extension_knee_rms_nm"] / reference["extension_knee_rms_nm"]
        ),
        "relative_hip_flexion_peak": (
            metrics["flexion_hip_peak_nm"] / reference["flexion_hip_peak_nm"]
        ),
        "relative_hip_extension_peak": (
            metrics["extension_hip_peak_nm"] / reference["extension_hip_peak_nm"]
        ),
        "relative_knee_flexion_peak": (
            metrics["flexion_knee_peak_nm"] / reference["flexion_knee_peak_nm"]
        ),
        "relative_knee_extension_peak": (
            metrics["extension_knee_peak_nm"] / reference["extension_knee_peak_nm"]
        ),
    }
    mapping.update(
        {
            "relative_full_hip_rms": metrics["hip_rms_nm"] / reference["hip_rms_nm"],
            "relative_full_knee_rms": metrics["knee_rms_nm"] / reference["knee_rms_nm"],
            "relative_full_hip_peak": metrics["hip_peak_nm"] / reference["hip_peak_nm"],
            "relative_full_knee_peak": metrics["knee_peak_nm"] / reference["knee_peak_nm"],
            "relative_flexion_vector_rms": metrics["flexion_rms_nm"]
            / reference["flexion_rms_nm"],
            "relative_extension_vector_rms": metrics["extension_rms_nm"]
            / reference["extension_rms_nm"],
        }
    )
    return mapping


def _endpoint_values(
    metrics: Mapping[str, float], relative: Mapping[str, float]
) -> dict[str, float]:
    return {
        "E0": float(metrics["full_rms_nm"]),
        "E1": max(
            relative["relative_full_hip_rms"],
            relative["relative_full_knee_rms"],
        ),
        "E2": max(
            relative["relative_hip_flexion_rms"],
            relative["relative_hip_extension_rms"],
            relative["relative_knee_flexion_rms"],
            relative["relative_knee_extension_rms"],
        ),
        "E3": max(
            relative["relative_full_hip_peak"],
            relative["relative_full_knee_peak"],
        ),
        "E4": max(
            relative["relative_hip_flexion_peak"],
            relative["relative_hip_extension_peak"],
            relative["relative_knee_flexion_peak"],
            relative["relative_knee_extension_peak"],
        ),
    }


def build_feature_representation(
    response_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in response_rows:
        grouped.setdefault(row["leg_id"], []).append(row)
    output: list[dict[str, Any]] = []
    by_leg: dict[str, list[dict[str, Any]]] = {}
    for leg_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["candidate_index"])
        reference = next(
            row
            for row in ordered
            if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
        )
        built: list[dict[str, Any]] = []
        for row in ordered:
            relative = _relative_feature_mapping(row, reference)
            endpoints = _endpoint_values(row, relative)
            feature_row = {
                "feature_vector_id": FEATURE_VECTOR_ID,
                "leg_id": leg_id,
                "candidate_id": row["candidate_id"],
                "canonical_beta_id": row["canonical_beta_id"],
                "candidate_index": row["candidate_index"],
                "beta_flex": row["beta_flex"],
                "beta_extend": row["beta_extend"],
                **{
                    key: row[key]
                    for key in (
                        "flexion_hip_rms_nm",
                        "extension_hip_rms_nm",
                        "flexion_knee_rms_nm",
                        "extension_knee_rms_nm",
                        "flexion_hip_peak_nm",
                        "extension_hip_peak_nm",
                        "flexion_knee_peak_nm",
                        "extension_knee_peak_nm",
                        "hip_rms_nm",
                        "knee_rms_nm",
                        "hip_peak_nm",
                        "knee_peak_nm",
                    )
                },
                **relative,
                "feature_vector": [relative[field] for field in FEATURE_VECTOR_FIELDS],
                **{f"endpoint_{key}": value for key, value in endpoints.items()},
            }
            built.append(feature_row)
            output.append(feature_row)
        by_leg[leg_id] = built
    return output, by_leg


def _grid(
    rows: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flex = np.asarray(sorted({float(row["beta_flex"]) for row in rows}))
    extend = np.asarray(sorted({float(row["beta_extend"]) for row in rows}))
    values = np.empty((len(flex), len(extend)), dtype=float)
    fi = {value: index for index, value in enumerate(flex)}
    ei = {value: index for index, value in enumerate(extend)}
    for row in rows:
        values[fi[float(row["beta_flex"])], ei[float(row["beta_extend"])]] = float(
            row[field]
        )
    return flex, extend, values


def endpoint_discriminability(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for endpoint_id in ENDPOINT_IDS:
        field = f"endpoint_{endpoint_id}"
        for leg_id, rows in rows_by_leg.items():
            values = np.asarray([row[field] for row in rows], dtype=float)
            reference_index = next(
                index
                for index, row in enumerate(rows)
                if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
            )
            reference = float(values[reference_index])
            minimum = float(np.min(values))
            best = rows[int(np.argmin(values))]
            item: dict[str, Any] = {
                "endpoint_id": endpoint_id,
                "endpoint_name": ENDPOINT_DEFINITIONS[endpoint_id]["name"],
                "leg_id": leg_id,
                "oracle_beta": [best["beta_flex"], best["beta_extend"]],
                "oracle_candidate_index": best["candidate_index"],
                "oracle_value": minimum,
                "reference_value": reference,
                "maximum_value": float(np.max(values)),
                "absolute_range": float(np.ptp(values)),
                "relative_range_by_reference": float(np.ptp(values) / reference),
                "coefficient_of_variation": float(np.std(values) / np.mean(values)),
                "exact_minimum_tie_count_at_1e_12": int(
                    np.sum(np.isclose(values, minimum, rtol=1.0e-12, atol=1.0e-12))
                ),
            }
            for tolerance in NEAR_OPTIMAL_TOLERANCES:
                item[f"within_{tolerance * 100:g}_percent_optimum_count"] = int(
                    np.sum(values <= minimum * (1.0 + tolerance))
                )
            output.append(item)
    return output


def cross_leg_decision_structure(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leg_ids = list(rows_by_leg)
    summaries = []
    pairwise_rows = []
    for endpoint_id in ENDPOINT_IDS:
        field = f"endpoint_{endpoint_id}"
        matrix = np.asarray(
            [[row[field] for row in rows_by_leg[leg_id]] for leg_id in leg_ids],
            dtype=float,
        )
        ranges = np.ptp(matrix, axis=1)
        normalized = (matrix - np.min(matrix, axis=1, keepdims=True)) / ranges[:, None]
        common = np.mean(normalized, axis=0)
        interaction = normalized - common
        total = normalized - float(np.mean(normalized))
        gradients: dict[str, np.ndarray] = {}
        for leg_index, leg_id in enumerate(leg_ids):
            flex, extend, grid = _grid(rows_by_leg[leg_id], field)
            grid = (grid - np.min(grid)) / np.ptp(grid)
            gf, ge = np.gradient(grid, flex, extend, edge_order=2)
            gradients[leg_id] = np.concatenate((gf.ravel(), ge.ravel()))

        endpoint_pairs = []
        for first_index, second_index in combinations(range(len(leg_ids)), 2):
            first_id = leg_ids[first_index]
            second_id = leg_ids[second_index]
            first_gradient = gradients[first_id]
            second_gradient = gradients[second_id]
            cosine = float(
                np.dot(first_gradient, second_gradient)
                / (np.linalg.norm(first_gradient) * np.linalg.norm(second_gradient))
            )
            row = {
                "endpoint_id": endpoint_id,
                "leg_a": first_id,
                "leg_b": second_id,
                "spearman": float(
                    spearmanr(matrix[first_index], matrix[second_index]).statistic
                ),
                "normalized_gradient_cosine_similarity": cosine,
            }
            endpoint_pairs.append(row)
            pairwise_rows.append(row)

        relative_regret = matrix / np.min(matrix, axis=1, keepdims=True) - 1.0
        universal: dict[str, Any] = {}
        for tolerance in NEAR_OPTIMAL_TOLERANCES:
            indices = np.flatnonzero(np.all(relative_regret <= tolerance, axis=0))
            universal[f"universal_within_{tolerance * 100:g}_percent_count"] = len(
                indices
            )
            universal[f"universal_within_{tolerance * 100:g}_percent_indices"] = (
                indices.tolist()
            )
        oracle_indices = [int(np.argmin(row)) for row in matrix]
        summaries.append(
            {
                "endpoint_id": endpoint_id,
                "oracle_indices": oracle_indices,
                "oracle_betas": [
                    [
                        rows_by_leg[leg_id][oracle_indices[index]]["beta_flex"],
                        rows_by_leg[leg_id][oracle_indices[index]]["beta_extend"],
                    ]
                    for index, leg_id in enumerate(leg_ids)
                ],
                "unique_oracle_count": len(set(oracle_indices)),
                "minimum_pairwise_spearman": min(row["spearman"] for row in endpoint_pairs),
                "median_pairwise_spearman": float(
                    np.median([row["spearman"] for row in endpoint_pairs])
                ),
                "minimum_normalized_gradient_cosine_similarity": min(
                    row["normalized_gradient_cosine_similarity"]
                    for row in endpoint_pairs
                ),
                "median_normalized_gradient_cosine_similarity": float(
                    np.median(
                        [
                            row["normalized_gradient_cosine_similarity"]
                            for row in endpoint_pairs
                        ]
                    )
                ),
                "normalized_subject_by_trajectory_interaction_rms": float(
                    np.sqrt(np.mean(interaction**2))
                ),
                "normalized_subject_by_trajectory_interaction_max_abs": float(
                    np.max(np.abs(interaction))
                ),
                "normalized_interaction_energy_fraction": float(
                    np.sum(interaction**2) / np.sum(total**2)
                ),
                **universal,
            }
        )
    return summaries, pairwise_rows


def anti_cancellation_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    representatives = []
    tolerance = 1.0e-12
    for leg_id, rows in rows_by_leg.items():
        reference = next(
            row for row in rows if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
        )
        categorized = []
        for row in rows:
            hip = row["relative_full_hip_rms"]
            knee = row["relative_full_knee_rms"]
            flexion = row["relative_flexion_vector_rms"]
            extension = row["relative_extension_vector_rms"]
            joint_tradeoff = (
                (hip < 1.0 - tolerance and knee > 1.0 + tolerance)
                or (knee < 1.0 - tolerance and hip > 1.0 + tolerance)
            )
            branch_tradeoff = (
                (flexion < 1.0 - tolerance and extension > 1.0 + tolerance)
                or (extension < 1.0 - tolerance and flexion > 1.0 + tolerance)
            )
            e0_relative = row["endpoint_E0"] / reference["endpoint_E0"]
            categorized.append(
                {
                    **row,
                    "e0_relative": e0_relative,
                    "joint_tradeoff": joint_tradeoff,
                    "branch_tradeoff": branch_tradeoff,
                    "e0_within_0_1_percent_of_reference": abs(e0_relative - 1.0)
                    <= 0.001,
                    "joint_anti_cancellation_gap": row["endpoint_E1"] - e0_relative,
                    "branch_anti_cancellation_gap": row["endpoint_E2"] - e0_relative,
                }
            )
        summaries.append(
            {
                "leg_id": leg_id,
                "hip_improves_knee_worsens_count": sum(
                    row["relative_full_hip_rms"] < 1.0 - tolerance
                    and row["relative_full_knee_rms"] > 1.0 + tolerance
                    for row in categorized
                ),
                "knee_improves_hip_worsens_count": sum(
                    row["relative_full_knee_rms"] < 1.0 - tolerance
                    and row["relative_full_hip_rms"] > 1.0 + tolerance
                    for row in categorized
                ),
                "flexion_improves_extension_worsens_count": sum(
                    row["relative_flexion_vector_rms"] < 1.0 - tolerance
                    and row["relative_extension_vector_rms"] > 1.0 + tolerance
                    for row in categorized
                ),
                "extension_improves_flexion_worsens_count": sum(
                    row["relative_extension_vector_rms"] < 1.0 - tolerance
                    and row["relative_flexion_vector_rms"] > 1.0 + tolerance
                    for row in categorized
                ),
                "joint_tradeoff_count": sum(row["joint_tradeoff"] for row in categorized),
                "branch_tradeoff_count": sum(row["branch_tradeoff"] for row in categorized),
                "joint_tradeoff_hidden_inside_E0_0_1_percent_count": sum(
                    row["joint_tradeoff"] and row["e0_within_0_1_percent_of_reference"]
                    for row in categorized
                ),
                "branch_tradeoff_hidden_inside_E0_0_1_percent_count": sum(
                    row["branch_tradeoff"] and row["e0_within_0_1_percent_of_reference"]
                    for row in categorized
                ),
                "maximum_joint_anti_cancellation_gap": max(
                    row["joint_anti_cancellation_gap"] for row in categorized
                ),
                "maximum_branch_anti_cancellation_gap": max(
                    row["branch_anti_cancellation_gap"] for row in categorized
                ),
            }
        )
        for kind, field in (
            ("JOINT_TRADEOFF", "joint_anti_cancellation_gap"),
            ("BRANCH_TRADEOFF", "branch_anti_cancellation_gap"),
        ):
            eligible = [
                row
                for row in categorized
                if row["e0_within_0_1_percent_of_reference"]
                and row["joint_tradeoff" if kind == "JOINT_TRADEOFF" else "branch_tradeoff"]
            ]
            if eligible:
                selected = max(eligible, key=lambda row: row[field])
                representatives.append(
                    {
                        "leg_id": leg_id,
                        "tradeoff_kind": kind,
                        "candidate_index": selected["candidate_index"],
                        "beta": [selected["beta_flex"], selected["beta_extend"]],
                        "E0_relative": selected["e0_relative"],
                        "E1": selected["endpoint_E1"],
                        "E2": selected["endpoint_E2"],
                        "relative_hip_rms": selected["relative_full_hip_rms"],
                        "relative_knee_rms": selected["relative_full_knee_rms"],
                        "relative_flexion_vector_rms": selected[
                            "relative_flexion_vector_rms"
                        ],
                        "relative_extension_vector_rms": selected[
                            "relative_extension_vector_rms"
                        ],
                        "anti_cancellation_gap": selected[field],
                    }
                )
    return summaries, representatives


def scalarization_information_loss(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for leg_id, rows in rows_by_leg.items():
        features = np.asarray([row["feature_vector"] for row in rows], dtype=float)
        feature_distance = pdist(features, metric="euclidean")
        feature_linf = pdist(features, metric="chebyshev")
        for endpoint_id in ENDPOINT_IDS:
            values = np.asarray([row[f"endpoint_{endpoint_id}"] for row in rows])
            endpoint_difference = pdist(values[:, None], metric="cityblock")
            reference = next(
                row[f"endpoint_{endpoint_id}"]
                for row in rows
                if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
            )
            output.append(
                {
                    "leg_id": leg_id,
                    "endpoint_id": endpoint_id,
                    "spearman_pairwise_endpoint_difference_vs_feature_distance": float(
                        spearmanr(endpoint_difference, feature_distance).statistic
                    ),
                    "pair_count_endpoint_within_0_1_percent_but_feature_linf_at_least_0_5_percent": int(
                        np.sum(
                            (endpoint_difference <= 0.001 * reference)
                            & (feature_linf >= 0.005)
                        )
                    ),
                    "total_pair_count": len(feature_distance),
                }
            )
    return output


def _perturb_torque(
    tau: np.ndarray,
    time_s: np.ndarray,
    reference_joint_peaks: np.ndarray,
    phase_pair: tuple[float, float],
) -> np.ndarray:
    normalized_time = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    output = np.asarray(tau, dtype=float).copy()
    for joint_index, phase in enumerate(phase_pair):
        waveform = (
            np.sin(2.0 * np.pi * 5.0 * normalized_time + phase)
            + 0.5 * np.sin(2.0 * np.pi * 11.0 * normalized_time - 0.7 * phase)
        ) / 1.5
        output[:, joint_index] += (
            0.0025 * reference_joint_peaks[joint_index] * waveform
        )
    return output


def robustness_analysis(
    baseline_rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    definition = load_frozen_benchmark_definition()
    scenario_rows: list[dict[str, Any]] = []
    peak_persistence_rows: list[dict[str, Any]] = []
    maximum_baseline_recompute_error = 0.0
    for leg in definition.legs:
        domain = build_leg_domain(leg)
        model = make_mujoco_model(definition, leg)
        time_s = np.asarray(domain.subject_reference.time_s)
        phases = np.asarray(domain.subject_reference.phases)
        raw_by_scenario: dict[str, list[dict[str, float]]] = {
            name: []
            for name in (
                "DETERMINISTIC_TORQUE_WAVE_A",
                "DETERMINISTIC_TORQUE_WAVE_B",
                "HALF_RATE_EVEN_SAMPLES",
                "HALF_RATE_ODD_OFFSET_SAMPLES",
            )
        }
        baseline_metrics = []
        exact_torques = []
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
            baseline_metrics.append(metrics)
            exact_torques.append(arrays["total"])
        reference_index = domain.reference.candidate_index
        reference_joint_peaks = np.max(
            np.abs(exact_torques[reference_index]), axis=0
        )
        even_indices = np.arange(0, len(time_s), 2)
        odd_indices = np.unique(
            np.concatenate(
                (
                    np.asarray([0]),
                    np.arange(1, len(time_s) - 1, 2),
                    np.asarray([len(time_s) - 1]),
                )
            )
        )
        for candidate_index, tau in enumerate(exact_torques):
            raw_by_scenario["DETERMINISTIC_TORQUE_WAVE_A"].append(
                _response_metrics_from_torque(
                    _perturb_torque(tau, time_s, reference_joint_peaks, (0.0, math.pi / 3.0)),
                    time_s,
                    phases,
                )
            )
            raw_by_scenario["DETERMINISTIC_TORQUE_WAVE_B"].append(
                _response_metrics_from_torque(
                    _perturb_torque(tau, time_s, reference_joint_peaks, (math.pi / 2.0, -math.pi / 4.0)),
                    time_s,
                    phases,
                )
            )
            raw_by_scenario["HALF_RATE_EVEN_SAMPLES"].append(
                _response_metrics_from_torque(
                    tau[even_indices], time_s[even_indices], phases[even_indices]
                )
            )
            raw_by_scenario["HALF_RATE_ODD_OFFSET_SAMPLES"].append(
                _response_metrics_from_torque(
                    tau[odd_indices], time_s[odd_indices], phases[odd_indices]
                )
            )
            for branch in ("flexion", "extension"):
                mask = phases == branch
                for joint_index, joint_name in enumerate(("hip", "knee")):
                    values = np.abs(tau[mask, joint_index])
                    peak = float(np.max(values))
                    peak_persistence_rows.append(
                        {
                            "leg_id": leg.leg_id,
                            "candidate_index": candidate_index,
                            "branch": branch,
                            "joint": joint_name,
                            "samples_at_or_above_99_percent_peak": int(
                                np.sum(values >= 0.99 * peak)
                            ),
                            "fraction_samples_at_or_above_99_percent_peak": float(
                                np.mean(values >= 0.99 * peak)
                            ),
                        }
                    )

        baseline_rows = baseline_rows_by_leg[leg.leg_id]
        for index, metrics in enumerate(baseline_metrics):
            maximum_baseline_recompute_error = max(
                maximum_baseline_recompute_error,
                abs(metrics["full_rms_nm"] - baseline_rows[index]["endpoint_E0"]),
            )
        baseline_endpoint = {
            endpoint_id: np.asarray(
                [row[f"endpoint_{endpoint_id}"] for row in baseline_rows], dtype=float
            )
            for endpoint_id in ENDPOINT_IDS
        }
        for scenario_name, scenario_metrics in raw_by_scenario.items():
            scenario_reference = scenario_metrics[reference_index]
            scenario_endpoint = {endpoint_id: [] for endpoint_id in ENDPOINT_IDS}
            for metrics in scenario_metrics:
                relative = _relative_feature_mapping(metrics, scenario_reference)
                endpoints = _endpoint_values(metrics, relative)
                for endpoint_id, value in endpoints.items():
                    scenario_endpoint[endpoint_id].append(value)
            for endpoint_id in ENDPOINT_IDS:
                baseline_values = baseline_endpoint[endpoint_id]
                perturbed_values = np.asarray(scenario_endpoint[endpoint_id], dtype=float)
                baseline_oracle = int(np.argmin(baseline_values))
                perturbed_oracle = int(np.argmin(perturbed_values))
                relative_change = np.abs(
                    (perturbed_values - baseline_values) / baseline_values
                )
                scenario_rows.append(
                    {
                        "leg_id": leg.leg_id,
                        "scenario": scenario_name,
                        "endpoint_id": endpoint_id,
                        "ranking_spearman_vs_baseline": float(
                            spearmanr(baseline_values, perturbed_values).statistic
                        ),
                        "median_absolute_relative_endpoint_change": float(
                            np.median(relative_change)
                        ),
                        "p95_absolute_relative_endpoint_change": float(
                            np.percentile(relative_change, 95.0)
                        ),
                        "maximum_absolute_relative_endpoint_change": float(
                            np.max(relative_change)
                        ),
                        "baseline_oracle_index": baseline_oracle,
                        "perturbed_oracle_index": perturbed_oracle,
                        "exact_oracle_stable": baseline_oracle == perturbed_oracle,
                        "baseline_relative_regret_of_perturbed_oracle": float(
                            baseline_values[perturbed_oracle]
                            / baseline_values[baseline_oracle]
                            - 1.0
                        ),
                    }
                )

    summaries = []
    for endpoint_id in ENDPOINT_IDS:
        rows = [row for row in scenario_rows if row["endpoint_id"] == endpoint_id]
        summaries.append(
            {
                "endpoint_id": endpoint_id,
                "minimum_ranking_spearman": min(
                    row["ranking_spearman_vs_baseline"] for row in rows
                ),
                "median_ranking_spearman": float(
                    np.median([row["ranking_spearman_vs_baseline"] for row in rows])
                ),
                "exact_oracle_stability_fraction": float(
                    np.mean([row["exact_oracle_stable"] for row in rows])
                ),
                "maximum_baseline_relative_regret_of_perturbed_oracle": max(
                    row["baseline_relative_regret_of_perturbed_oracle"] for row in rows
                ),
                "maximum_absolute_relative_endpoint_change": max(
                    row["maximum_absolute_relative_endpoint_change"] for row in rows
                ),
                "perturbation_definition": (
                    "two fixed waveform cases at 0.25% of reference joint peak, "
                    "plus two deterministic half-rate sample grids"
                ),
                "not_a_validated_sensor_noise_model": True,
            }
        )
    summaries.append(
        {
            "endpoint_id": "BASELINE_RECOMPUTE_CHECK",
            "maximum_E0_recompute_abs_error_nm": maximum_baseline_recompute_error,
        }
    )
    return scenario_rows, summaries, peak_persistence_rows


def peak_persistence_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["leg_id"], row["branch"], row["joint"]) for row in rows})
    for leg_id, branch, joint in keys:
        values = np.asarray(
            [
                row["fraction_samples_at_or_above_99_percent_peak"]
                for row in rows
                if row["leg_id"] == leg_id
                and row["branch"] == branch
                and row["joint"] == joint
            ]
        )
        output.append(
            {
                "leg_id": leg_id,
                "branch": branch,
                "joint": joint,
                "median_fraction_samples_at_or_above_99_percent_peak": float(
                    np.median(values)
                ),
                "maximum_fraction_samples_at_or_above_99_percent_peak": float(
                    np.max(values)
                ),
            }
        )
    return output


def measurement_compatibility() -> list[dict[str, Any]]:
    rows = []
    for feature in (
        "hip/knee flexion/extension RMS",
        "hip/knee flexion/extension peak",
        "full hip/knee RMS and peak",
    ):
        rows.append(
            {
                "quantity": feature,
                "current_status": "MODEL_DERIVED",
                "current_source": "frozen MuJoCo inverse-dynamics joint torque",
                "future_measurement_path": "calibrated wrench-to-joint mapping or validated gray-box estimate",
                "real_world_validated": False,
            }
        )
    for endpoint_id in ENDPOINT_IDS:
        rows.append(
            {
                "quantity": endpoint_id,
                "current_status": "MODEL_DERIVED",
                "current_source": "scalarization of frozen simulated joint-load features",
                "future_measurement_path": "same formula after physical signal/frame validation",
                "real_world_validated": False,
            }
        )
    rows.extend(
        (
            {
                "quantity": "robot wrench in a verified task direction",
                "current_status": "DIRECTLY_MEASURABLE",
                "current_source": "future calibrated robot wrench stream",
                "future_measurement_path": "requires reference-point, frame, timing, and task-direction physical validation",
                "real_world_validated": False,
            },
            {
                "quantity": "gray-box-estimated joint load",
                "current_status": "MODEL_DERIVED",
                "current_source": "model estimate from measured inputs",
                "future_measurement_path": "validate against independent physical load evidence",
                "real_world_validated": False,
            },
            {
                "quantity": "contact pressure or distributed interface load",
                "current_status": "FUTURE_SENSOR_REQUIRED",
                "current_source": "unavailable in current benchmark",
                "future_measurement_path": "future tactile or pressure sensing",
                "real_world_validated": False,
            },
        )
    )
    return rows


def endpoint_selection_characterization(
    discriminability: list[dict[str, Any]],
    cross_leg: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cross = {row["endpoint_id"]: row for row in cross_leg}
    robust = {
        row["endpoint_id"]: row
        for row in robustness
        if row["endpoint_id"] in ENDPOINT_IDS
    }
    output = []
    for endpoint_id in ENDPOINT_IDS:
        endpoint_rows = [
            row for row in discriminability if row["endpoint_id"] == endpoint_id
        ]
        output.append(
            {
                "endpoint_id": endpoint_id,
                "mechanical_interpretation": ENDPOINT_DEFINITIONS[endpoint_id][
                    "mechanical_interpretation"
                ],
                "anti_cancellation_property": {
                    "E0": "none across joints, branches, or time",
                    "E1": "prevents hip-knee RMS cancellation",
                    "E2": "prevents joint and branch RMS cancellation",
                    "E3": "prevents hip-knee peak cancellation",
                    "E4": "prevents joint and branch peak cancellation",
                }[endpoint_id],
                "minimum_relative_range_across_legs": min(
                    row["relative_range_by_reference"] for row in endpoint_rows
                ),
                "maximum_relative_range_across_legs": max(
                    row["relative_range_by_reference"] for row in endpoint_rows
                ),
                "normalized_interaction_energy_fraction": cross[endpoint_id][
                    "normalized_interaction_energy_fraction"
                ],
                "minimum_pairwise_spearman": cross[endpoint_id][
                    "minimum_pairwise_spearman"
                ],
                "minimum_robustness_ranking_spearman": robust[endpoint_id][
                    "minimum_ranking_spearman"
                ],
                "exact_oracle_stability_fraction": robust[endpoint_id][
                    "exact_oracle_stability_fraction"
                ],
                "arbitrary_continuous_weights": False,
                "current_measurement_status": "MODEL_DERIVED",
                "recommendation": {
                    "E0": "retain only as historical baseline",
                    "E1": "retain as joint-balanced RMS diagnostic",
                    "E2": "recommended next primary simulated mechanical endpoint",
                    "E3": "not recommended because its beta landscape is extremely flat and perturbation-unstable",
                    "E4": "not recommended because its beta landscape is extremely flat and perturbation-unstable",
                }[endpoint_id],
            }
        )
    return output


def run_endpoint_design_study() -> dict[str, Any]:
    source = load_source_audit()
    feature_rows, rows_by_leg = build_feature_representation(source["response_rows"])
    discriminability = endpoint_discriminability(rows_by_leg)
    cross_leg, pairwise = cross_leg_decision_structure(rows_by_leg)
    anti_cancellation, representatives = anti_cancellation_analysis(rows_by_leg)
    information_loss = scalarization_information_loss(rows_by_leg)
    robustness_rows, robustness_summary, persistence_rows = robustness_analysis(
        rows_by_leg
    )
    selection = endpoint_selection_characterization(
        discriminability, cross_leg, robustness_summary
    )
    return {
        "study_id": STUDY_ID,
        STUDY_ID: "COMPLETE",
        "classification": "OFFLINE_SYNTHETIC_MUJOCO_ENDPOINT_DESIGN_ONLY",
        "source_audit": "V3_TRAJECTORY_OBJECTIVE_DISCRIMINABILITY_AUDIT_V1",
        "source_audit_preserved": True,
        "five_leg_parameters_preserved": True,
        "ROM_preserved": True,
        "V3_preserved": True,
        "E0_historical_definition_preserved": True,
        "continuous_weight_optimization": False,
        "endpoint_definitions_fixed_before_oracle_characterization": True,
        "endpoint_definitions": ENDPOINT_DEFINITIONS,
        "feature_vector": {
            "feature_vector_id": FEATURE_VECTOR_ID,
            "ordered_fields": list(FEATURE_VECTOR_FIELDS),
            "normalization": "each leg's own beta=(0,0) reference",
            "cohort_mean_normalization": False,
        },
        "endpoint_discriminability": discriminability,
        "cross_leg_decision_structure": cross_leg,
        "pairwise_endpoint_structure": pairwise,
        "anti_cancellation_summary": anti_cancellation,
        "anti_cancellation_representatives": representatives,
        "scalarization_information_loss": information_loss,
        "robustness_scenarios": robustness_rows,
        "robustness_summary": robustness_summary,
        "peak_persistence_summary": peak_persistence_summary(persistence_rows),
        "measurement_compatibility": measurement_compatibility(),
        "endpoint_selection_characterization": selection,
        "selection_rationale": (
            "E2 directly addresses both observed hip-knee and flexion-extension "
            "cancellation, uses only within-leg reference ratios and a predefined "
            "worst joint-by-branch maximum, is stable under the fixed perturbation "
            "study, and preserves the most feature-vector separation without "
            "outcome-tuned weights. Its reference optimum in four legs is a valid "
            "conservative no-branch-worsening result, not a failure to manufacture "
            "oracle diversity. E1 remains a simpler joint-balanced diagnostic. Peak "
            "endpoints are too flat and perturbation-sensitive."
        ),
        "selection_not_based_on_oracle_diversity": True,
        "limitations": [
            "offline synthetic MuJoCo systems only",
            "E2 remains a scalar and does not preserve the complete feature vector",
            "E2 still has 500 universal candidates within 1 percent across all five legs",
            "deterministic perturbations are sensitivity probes, not a validated sensor noise model",
            "all current endpoint inputs are model-derived and not physically validated",
        ],
        "MECHANICAL_ENDPOINT_REDESIGN_V1": "SUPPORTED_WITH_LIMITATIONS",
        "RECOMMENDED_PRIMARY_ENDPOINT": "E2",
        "READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY": True,
        "feature_rows": feature_rows,
        "algorithm_runs": 0,
        "robot_actions": 0,
        "PINN_training": 0,
    }


__all__ = [
    "ENDPOINT_DEFINITIONS",
    "ENDPOINT_IDS",
    "FEATURE_VECTOR_FIELDS",
    "FEATURE_VECTOR_ID",
    "STUDY_ID",
    "anti_cancellation_analysis",
    "build_feature_representation",
    "cross_leg_decision_structure",
    "endpoint_discriminability",
    "load_source_audit",
    "measurement_compatibility",
    "endpoint_selection_characterization",
    "robustness_analysis",
    "run_endpoint_design_study",
    "scalarization_information_loss",
]
