"""Necessity evaluation using the frozen branch-balanced E2 endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_STUDY = PACKAGE_DIR / "results_endpoint_design_v1" / "study_summary.json"
STUDY_ID = "REEVALUATE_PERSONALIZATION_NECESSITY_WITH_E2_V1"
E2_ENDPOINT_ID = "BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS"
COMPONENT_FIELDS = (
    "relative_hip_flexion_rms",
    "relative_hip_extension_rms",
    "relative_knee_flexion_rms",
    "relative_knee_extension_rms",
)
COMPONENT_LABELS = {
    "relative_hip_flexion_rms": "hip_flexion",
    "relative_hip_extension_rms": "hip_extension",
    "relative_knee_flexion_rms": "knee_flexion",
    "relative_knee_extension_rms": "knee_extension",
}
NEAR_TOLERANCES = (0.001, 0.005, 0.01, 0.02, 0.05)


def load_endpoint_study() -> dict[str, Any]:
    payload = json.loads(SOURCE_STUDY.read_text(encoding="utf-8"))
    if payload["DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1"] != "COMPLETE":
        raise RuntimeError("endpoint design study must be complete")
    if payload["RECOMMENDED_PRIMARY_ENDPOINT"] != "E2":
        raise RuntimeError("E2 is not the frozen recommended endpoint")
    if not payload["READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY"]:
        raise RuntimeError("endpoint study did not authorize necessity reevaluation")
    if len(payload["feature_rows"]) != 5 * 625:
        raise RuntimeError("endpoint study must contain exactly 5 x 625 feature rows")
    e2_definition = payload["endpoint_definitions"]["E2"]
    if e2_definition["formula"] != (
        "max(r_hip_flex_RMS, r_hip_extend_RMS, "
        "r_knee_flex_RMS, r_knee_extend_RMS)"
    ):
        raise RuntimeError("frozen E2 formula changed")
    if payload["feature_vector"]["normalization"] != (
        "each leg's own beta=(0,0) reference"
    ):
        raise RuntimeError("frozen E2 normalization changed")
    return payload


def _group_rows(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload["feature_rows"]:
        grouped.setdefault(row["leg_id"], []).append(row)
    ordered = {
        leg_id: sorted(rows, key=lambda row: row["candidate_index"])
        for leg_id, rows in grouped.items()
    }
    if len(ordered) != 5 or any(len(rows) != 625 for rows in ordered.values()):
        raise RuntimeError("frozen replay must contain five 625-candidate landscapes")
    candidate_grids = [
        [
            (row["candidate_index"], row["beta_flex"], row["beta_extend"])
            for row in rows
        ]
        for rows in ordered.values()
    ]
    if any(grid != candidate_grids[0] for grid in candidate_grids[1:]):
        raise RuntimeError("frozen V3 candidate grids are not aligned across legs")
    return ordered


def oracle_characterization(
    rows_by_leg: dict[str, list[dict[str, Any]]], endpoint_id: str
) -> list[dict[str, Any]]:
    output = []
    field = f"endpoint_{endpoint_id}"
    for leg_id, rows in rows_by_leg.items():
        values = np.asarray([row[field] for row in rows], dtype=float)
        oracle_index = int(np.argmin(values))
        oracle = rows[oracle_index]
        reference_index = next(
            index
            for index, row in enumerate(rows)
            if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
        )
        reference = float(values[reference_index])
        minimum = float(values[oracle_index])
        result: dict[str, Any] = {
            "endpoint_id": endpoint_id,
            "leg_id": leg_id,
            "oracle_candidate_index": oracle_index,
            "oracle_beta": [oracle["beta_flex"], oracle["beta_extend"]],
            "oracle_value": minimum,
            "reference_candidate_index": reference_index,
            "reference_value": reference,
            "reference_to_oracle_improvement": reference - minimum,
            "reference_to_oracle_improvement_relative": reference / minimum - 1.0,
            "absolute_range": float(np.ptp(values)),
            "relative_range_by_reference": float(np.ptp(values) / reference),
            "coefficient_of_variation": float(np.std(values) / np.mean(values)),
        }
        for tolerance in NEAR_TOLERANCES:
            result[f"within_{tolerance * 100:g}_percent_oracle_count"] = int(
                np.sum(values <= minimum * (1.0 + tolerance))
            )
        output.append(result)
    return output


def oracle_transfer_matrix(
    rows_by_leg: dict[str, list[dict[str, Any]]], endpoint_id: str
) -> list[dict[str, Any]]:
    leg_ids = list(rows_by_leg)
    field = f"endpoint_{endpoint_id}"
    matrix = np.asarray(
        [[row[field] for row in rows_by_leg[leg_id]] for leg_id in leg_ids],
        dtype=float,
    )
    oracle_indices = np.argmin(matrix, axis=1)
    output = []
    for source_index, source_leg in enumerate(leg_ids):
        source_oracle_index = int(oracle_indices[source_index])
        source_row = rows_by_leg[source_leg][source_oracle_index]
        for target_index, target_leg in enumerate(leg_ids):
            target_oracle_index = int(oracle_indices[target_index])
            target_oracle = float(matrix[target_index, target_oracle_index])
            transferred = float(matrix[target_index, source_oracle_index])
            output.append(
                {
                    "endpoint_id": endpoint_id,
                    "source_leg": source_leg,
                    "source_oracle_candidate_index": source_oracle_index,
                    "source_oracle_beta": [
                        source_row["beta_flex"],
                        source_row["beta_extend"],
                    ],
                    "target_leg": target_leg,
                    "target_oracle_candidate_index": target_oracle_index,
                    "target_oracle_value": target_oracle,
                    "transferred_value": transferred,
                    "absolute_regret": transferred - target_oracle,
                    "relative_regret": transferred / target_oracle - 1.0,
                }
            )
    return output


def common_candidate_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]], endpoint_id: str
) -> dict[str, Any]:
    leg_ids = list(rows_by_leg)
    field = f"endpoint_{endpoint_id}"
    matrix = np.asarray(
        [[row[field] for row in rows_by_leg[leg_id]] for leg_id in leg_ids],
        dtype=float,
    )
    oracle_indices = np.argmin(matrix, axis=1)
    oracle_values = matrix[np.arange(len(leg_ids)), oracle_indices]
    mean_landscape = np.mean(matrix, axis=0)
    common_index = int(np.argmin(mean_landscape))
    canonical = rows_by_leg[leg_ids[0]][common_index]
    absolute_regret = matrix[:, common_index] - oracle_values
    relative_regret = matrix[:, common_index] / oracle_values - 1.0
    universal: dict[str, Any] = {}
    per_candidate_relative_regret = matrix / oracle_values[:, None] - 1.0
    for tolerance in NEAR_TOLERANCES:
        indices = np.flatnonzero(
            np.all(per_candidate_relative_regret <= tolerance, axis=0)
        )
        universal[f"universal_within_{tolerance * 100:g}_percent_count"] = len(
            indices
        )
        universal[f"universal_within_{tolerance * 100:g}_percent_indices"] = (
            indices.tolist()
        )
    return {
        "endpoint_id": endpoint_id,
        "common_candidate_index": common_index,
        "common_beta": [canonical["beta_flex"], canonical["beta_extend"]],
        "common_mean_endpoint": float(mean_landscape[common_index]),
        "mean_absolute_common_regret": float(np.mean(absolute_regret)),
        "maximum_absolute_common_regret": float(np.max(absolute_regret)),
        "mean_relative_common_regret": float(np.mean(relative_regret)),
        "maximum_relative_common_regret": float(np.max(relative_regret)),
        "per_leg": [
            {
                "leg_id": leg_id,
                "oracle_candidate_index": int(oracle_indices[index]),
                "oracle_value": float(oracle_values[index]),
                "common_value": float(matrix[index, common_index]),
                "absolute_common_regret": float(absolute_regret[index]),
                "relative_common_regret": float(relative_regret[index]),
            }
            for index, leg_id in enumerate(leg_ids)
        ],
        **universal,
    }


def branch_driver_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows = []
    summaries = []
    for leg_id, rows in rows_by_leg.items():
        values = np.asarray([[row[field] for field in COMPONENT_FIELDS] for row in rows])
        endpoint = np.asarray([row["endpoint_E2"] for row in rows])
        oracle_index = int(np.argmin(endpoint))
        oracle_value = float(endpoint[oracle_index])
        active_by_candidate = []
        for index, row in enumerate(rows):
            maximum = float(np.max(values[index]))
            active_indices = np.flatnonzero(
                np.isclose(values[index], maximum, rtol=1.0e-12, atol=1.0e-12)
            )
            active = [COMPONENT_LABELS[COMPONENT_FIELDS[item]] for item in active_indices]
            active_by_candidate.append(active)
            candidate_rows.append(
                {
                    "leg_id": leg_id,
                    "candidate_index": row["candidate_index"],
                    "beta_flex": row["beta_flex"],
                    "beta_extend": row["beta_extend"],
                    "E2": row["endpoint_E2"],
                    "active_worst_components": active,
                    **{COMPONENT_LABELS[field]: row[field] for field in COMPONENT_FIELDS},
                }
            )
        near_indices = np.flatnonzero(endpoint <= oracle_value * 1.001)

        def counts(indices: np.ndarray) -> dict[str, int]:
            return {
                label: sum(label in active_by_candidate[index] for index in indices)
                for label in COMPONENT_LABELS.values()
            }

        summaries.append(
            {
                "leg_id": leg_id,
                "oracle_candidate_index": oracle_index,
                "oracle_beta": [
                    rows[oracle_index]["beta_flex"],
                    rows[oracle_index]["beta_extend"],
                ],
                "oracle_value": oracle_value,
                "oracle_active_worst_components": active_by_candidate[oracle_index],
                "oracle_component_values": {
                    COMPONENT_LABELS[field]: float(values[oracle_index, field_index])
                    for field_index, field in enumerate(COMPONENT_FIELDS)
                },
                "near_0_1_percent_candidate_count": len(near_indices),
                "near_0_1_percent_active_component_counts": counts(near_indices),
                "full_landscape_active_component_counts": counts(
                    np.arange(len(rows))
                ),
            }
        )
    return candidate_rows, summaries


def _two_way_energy(matrix: np.ndarray) -> dict[str, float]:
    grand = float(np.mean(matrix))
    leg_effect = np.mean(matrix, axis=1, keepdims=True) - grand
    trajectory_effect = np.mean(matrix, axis=0, keepdims=True) - grand
    interaction = matrix - grand - leg_effect - trajectory_effect
    leg_energy = matrix.shape[1] * float(np.sum(leg_effect**2))
    trajectory_energy = matrix.shape[0] * float(np.sum(trajectory_effect**2))
    interaction_energy = float(np.sum(interaction**2))
    total = leg_energy + trajectory_energy + interaction_energy
    return {
        "raw_two_way_leg_main_effect_fraction": leg_energy / total,
        "raw_two_way_trajectory_main_effect_fraction": trajectory_energy / total,
        "raw_two_way_interaction_fraction": interaction_energy / total,
    }


def endpoint_comparison(
    source: dict[str, Any],
    rows_by_leg: dict[str, list[dict[str, Any]]],
    oracle_rows: dict[str, list[dict[str, Any]]],
    transfer_rows: dict[str, list[dict[str, Any]]],
    common: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    structure = {
        row["endpoint_id"]: row
        for row in source["cross_leg_decision_structure"]
        if row["endpoint_id"] in {"E0", "E2"}
    }
    leg_ids = list(rows_by_leg)
    output = []
    for endpoint_id in ("E0", "E2"):
        matrix = np.asarray(
            [
                [row[f"endpoint_{endpoint_id}"] for row in rows_by_leg[leg_id]]
                for leg_id in leg_ids
            ]
        )
        transfer = transfer_rows[endpoint_id]
        nonzero = [row for row in transfer if row["relative_regret"] > 1.0e-12]
        output.append(
            {
                "endpoint_id": endpoint_id,
                "unique_oracle_count": len(
                    {row["oracle_candidate_index"] for row in oracle_rows[endpoint_id]}
                ),
                "minimum_pairwise_spearman": structure[endpoint_id][
                    "minimum_pairwise_spearman"
                ],
                "median_pairwise_spearman": structure[endpoint_id][
                    "median_pairwise_spearman"
                ],
                "minimum_normalized_gradient_cosine_similarity": structure[
                    endpoint_id
                ]["minimum_normalized_gradient_cosine_similarity"],
                "median_normalized_gradient_cosine_similarity": structure[
                    endpoint_id
                ]["median_normalized_gradient_cosine_similarity"],
                "normalized_interaction_energy_fraction": structure[endpoint_id][
                    "normalized_interaction_energy_fraction"
                ],
                "normalized_interaction_rms": structure[endpoint_id][
                    "normalized_subject_by_trajectory_interaction_rms"
                ],
                "maximum_cross_leg_transfer_relative_regret": max(
                    row["relative_regret"] for row in transfer
                ),
                "nonzero_transfer_cell_count": len(nonzero),
                "mean_common_relative_regret": common[endpoint_id][
                    "mean_relative_common_regret"
                ],
                "maximum_common_relative_regret": common[endpoint_id][
                    "maximum_relative_common_regret"
                ],
                "universal_within_0.5_percent_count": common[endpoint_id][
                    "universal_within_0.5_percent_count"
                ],
                "universal_within_1_percent_count": common[endpoint_id][
                    "universal_within_1_percent_count"
                ],
                **_two_way_energy(matrix),
            }
        )
    return output


def robustness_aware_necessity(
    source: dict[str, Any],
    transfer: list[dict[str, Any]],
    common: dict[str, Any],
) -> dict[str, Any]:
    robustness = next(
        row for row in source["robustness_summary"] if row["endpoint_id"] == "E2"
    )
    per_leg_variability = {
        leg_id: max(
            row["maximum_absolute_relative_endpoint_change"]
            for row in source["robustness_scenarios"]
            if row["endpoint_id"] == "E2" and row["leg_id"] == leg_id
        )
        for leg_id in {row["leg_id"] for row in source["robustness_scenarios"]}
    }
    per_leg_common = []
    for row in common["per_leg"]:
        variability = per_leg_variability[row["leg_id"]]
        per_leg_common.append(
            {
                **row,
                "maximum_perturbation_relative_variability": variability,
                "common_regret_to_perturbation_ratio": (
                    row["relative_common_regret"] / variability
                    if variability > 0.0
                    else None
                ),
            }
        )
    maximum_transfer = max(row["relative_regret"] for row in transfer)
    maximum_variability = robustness["maximum_absolute_relative_endpoint_change"]
    return {
        "fixed_perturbation_definition": robustness["perturbation_definition"],
        "not_a_validated_sensor_noise_model": True,
        "minimum_E2_ranking_spearman": robustness["minimum_ranking_spearman"],
        "E2_exact_oracle_stability_fraction": robustness[
            "exact_oracle_stability_fraction"
        ],
        "maximum_E2_relative_endpoint_variability": maximum_variability,
        "maximum_transfer_relative_regret": maximum_transfer,
        "maximum_transfer_regret_to_variability_ratio": maximum_transfer
        / maximum_variability,
        "mean_common_relative_regret": common["mean_relative_common_regret"],
        "maximum_common_relative_regret": common["maximum_relative_common_regret"],
        "mean_common_regret_to_maximum_variability_ratio": common[
            "mean_relative_common_regret"
        ]
        / maximum_variability,
        "maximum_common_regret_to_maximum_variability_ratio": common[
            "maximum_relative_common_regret"
        ]
        / maximum_variability,
        "per_leg_common_regret_vs_variability": per_leg_common,
    }


def run_e2_necessity_study() -> dict[str, Any]:
    source = load_endpoint_study()
    rows_by_leg = _group_rows(source)
    oracles = {
        endpoint_id: oracle_characterization(rows_by_leg, endpoint_id)
        for endpoint_id in ("E0", "E2")
    }
    transfers = {
        endpoint_id: oracle_transfer_matrix(rows_by_leg, endpoint_id)
        for endpoint_id in ("E0", "E2")
    }
    common = {
        endpoint_id: common_candidate_analysis(rows_by_leg, endpoint_id)
        for endpoint_id in ("E0", "E2")
    }
    driver_rows, driver_summary = branch_driver_analysis(rows_by_leg)
    comparison = endpoint_comparison(source, rows_by_leg, oracles, transfers, common)
    robustness = robustness_aware_necessity(source, transfers["E2"], common["E2"])
    pairwise = [
        row
        for row in source["pairwise_endpoint_structure"]
        if row["endpoint_id"] in {"E0", "E2"}
    ]
    e2_oracle_groups: dict[tuple[float, float], list[str]] = {}
    for row in oracles["E2"]:
        beta = tuple(row["oracle_beta"])
        e2_oracle_groups.setdefault(beta, []).append(row["leg_id"])
    e2_landscape_rows = [
        {
            "leg_id": row["leg_id"],
            "candidate_id": row["candidate_id"],
            "canonical_beta_id": row["canonical_beta_id"],
            "candidate_index": row["candidate_index"],
            "beta_flex": row["beta_flex"],
            "beta_extend": row["beta_extend"],
            "E0": row["endpoint_E0"],
            "E2": row["endpoint_E2"],
            **{COMPONENT_LABELS[field]: row[field] for field in COMPONENT_FIELDS},
        }
        for rows in rows_by_leg.values()
        for row in rows
    ]
    return {
        "study_id": STUDY_ID,
        STUDY_ID: "COMPLETE",
        "classification": "OFFLINE_SYNTHETIC_MUJOCO_E2_NECESSITY_EVALUATION_ONLY",
        "source_endpoint_study": "DESIGN_MECHANICALLY_INTERPRETABLE_ENDPOINT_V1",
        "source_endpoint_study_preserved": True,
        "E2_endpoint_id": E2_ENDPOINT_ID,
        "E2_definition": source["endpoint_definitions"]["E2"],
        "E2_source_definition_name": source["endpoint_definitions"]["E2"]["name"],
        "E2_definition_frozen": True,
        "five_leg_parameters_preserved": True,
        "ROM_preserved": True,
        "V3_preserved": True,
        "candidate_count_per_leg": 625,
        "E2_oracle_characterization": oracles["E2"],
        "E2_oracle_diversity": {
            "unique_oracle_count": len(e2_oracle_groups),
            "groups": [
                {"oracle_beta": list(beta), "leg_ids": leg_ids}
                for beta, leg_ids in e2_oracle_groups.items()
            ],
            "diversity_alone_does_not_establish_personalization": True,
        },
        "E0_oracle_characterization": oracles["E0"],
        "E2_oracle_transfer_matrix": transfers["E2"],
        "E0_oracle_transfer_matrix": transfers["E0"],
        "E2_common_candidate": common["E2"],
        "E0_common_candidate": common["E0"],
        "E2_branch_driver_summary": driver_summary,
        "E2_branch_driver_rows": driver_rows,
        "E0_vs_E2_comparison": comparison,
        "E0_E2_pairwise_landscape_similarity": pairwise,
        "robustness_aware_necessity": robustness,
        "decision_rationale": (
            "E2 creates a real simulated ordering interaction and a robust transfer "
            "penalty when the exceptional Leg 4 oracle is transferred to other legs. "
            "However, four of five legs retain the exact reference oracle, the "
            "cohort-optimal common reference has zero regret on those four and only "
            "0.071 percent relative regret on Leg 4, 228 candidates are universally "
            "within 0.5 percent, and mean common regret is no larger than the fixed "
            "perturbation variability. The systems therefore still primarily support "
            "a shared common trajectory rather than a sufficiently strong sequential "
            "personalization target."
        ),
        "E2_PERSONALIZATION_NECESSITY": "NOT_SUPPORTED",
        "READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON": False,
        "algorithm_runs": 0,
        "robot_actions": 0,
        "PINN_training": 0,
        "E2_landscape_rows": e2_landscape_rows,
    }


__all__ = [
    "COMPONENT_FIELDS",
    "STUDY_ID",
    "branch_driver_analysis",
    "common_candidate_analysis",
    "endpoint_comparison",
    "load_endpoint_study",
    "oracle_characterization",
    "oracle_transfer_matrix",
    "robustness_aware_necessity",
    "run_e2_necessity_study",
]
