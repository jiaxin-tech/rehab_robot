"""Full 5 x 625 MuJoCo landscape generation and characterization."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from personalization.observations import EpisodeObservation
from personalization.rom_gated_v2 import (
    SubjectSpecificFullDynamicsGrayBoxEndpointAdapter,
    SubjectSpecificV3CandidateDomain,
)

from .model import (
    FrozenBenchmarkDefinition,
    MechanicalLegDefinition,
    load_frozen_benchmark_definition,
    make_mujoco_model,
)
from .replay import replay_trajectory


def build_leg_domain(
    leg: MechanicalLegDefinition,
) -> SubjectSpecificV3CandidateDomain:
    return SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(
        leg.make_rom_profile()
    )


def generate_leg_landscape(
    definition: FrozenBenchmarkDefinition,
    leg: MechanicalLegDefinition,
) -> tuple[SubjectSpecificV3CandidateDomain, list[dict[str, Any]], dict[str, Any]]:
    """Execute every frozen V3 candidate through one MuJoCo leg."""

    domain = build_leg_domain(leg)
    model = make_mujoco_model(definition, leg)
    rows: list[dict[str, Any]] = []
    for candidate in domain:
        replay = replay_trajectory(
            model=model,
            definition=definition,
            leg=leg,
            time_s=domain.subject_reference.time_s,
            q_project=candidate.trajectory.q,
            dq_project=candidate.trajectory.dq,
            ddq_project=candidate.trajectory.ddq,
        )
        rows.append(
            {
                "leg_id": leg.leg_id,
                "candidate_id": candidate.candidate_id,
                "canonical_beta_id": candidate.canonical_beta_id,
                "candidate_index": candidate.candidate_index,
                "beta_flex": candidate.beta_flex,
                "beta_extend": candidate.beta_extend,
                "endpoint_name": definition.endpoint_name,
                "endpoint_unit": definition.endpoint_unit,
                **replay.as_metrics(),
            }
        )

    reference = domain.reference
    first = replay_trajectory(
        model=model,
        definition=definition,
        leg=leg,
        time_s=domain.subject_reference.time_s,
        q_project=reference.trajectory.q,
        dq_project=reference.trajectory.dq,
        ddq_project=reference.trajectory.ddq,
    )
    second = replay_trajectory(
        model=model,
        definition=definition,
        leg=leg,
        time_s=domain.subject_reference.time_s,
        q_project=reference.trajectory.q,
        dq_project=reference.trajectory.dq,
        ddq_project=reference.trajectory.ddq,
    )
    determinism_difference = abs(
        float(first.endpoint_value_nm) - float(second.endpoint_value_nm)
    )
    audit = {
        **domain.invariant_summary(),
        "leg_id": leg.leg_id,
        "valid_candidate_count": sum(row["valid"] for row in rows),
        "invalid_candidate_count": sum(not row["valid"] for row in rows),
        "maximum_tracking_rms_rad": max(row["tracking_rms_rad"] for row in rows),
        "maximum_tracking_max_abs_rad": max(
            row["tracking_max_abs_rad"] for row in rows
        ),
        "reference_replay_endpoint_absolute_difference_nm": determinism_difference,
        "endpoint_reproducible": determinism_difference <= 1.0e-12,
        "coordinate_convention": "theta_shank = q_hip - q_knee",
        "mujoco_replay": "EXACT_STATE_MJ_INVERSE",
    }
    return domain, rows, audit


def characterize_oracle(
    leg: MechanicalLegDefinition,
    domain: SubjectSpecificV3CandidateDomain,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return {
            "leg_id": leg.leg_id,
            "valid_candidates": 0,
            "oracle_available": False,
        }
    oracle = min(valid, key=lambda row: (row["endpoint_value_nm"], row["candidate_index"]))
    reference = next(
        row for row in valid if row["candidate_id"] == domain.reference.candidate_id
    )
    oracle_value = float(oracle["endpoint_value_nm"])
    reference_value = float(reference["endpoint_value_nm"])
    near_1 = [
        row for row in valid if float(row["endpoint_value_nm"]) <= 1.01 * oracle_value
    ]
    near_5 = [
        row for row in valid if float(row["endpoint_value_nm"]) <= 1.05 * oracle_value
    ]
    return {
        "leg_id": leg.leg_id,
        "rom_profile_id": domain.profile.profile_id,
        "rom_profile_fingerprint": domain.profile.fingerprint,
        "hip_rom_deg": [
            math.degrees(domain.profile.hip_min_rad),
            math.degrees(domain.profile.hip_max_rad),
        ],
        "knee_rom_deg": [
            math.degrees(domain.profile.knee_min_rad),
            math.degrees(domain.profile.knee_max_rad),
        ],
        "valid_candidates": len(valid),
        "invalid_candidates": len(rows) - len(valid),
        "reference_J_nm": reference_value,
        "oracle_candidate_id": oracle["candidate_id"],
        "oracle_canonical_beta_id": oracle["canonical_beta_id"],
        "oracle_beta": [oracle["beta_flex"], oracle["beta_extend"]],
        "oracle_J_nm": oracle_value,
        "reference_to_oracle_improvement_nm": reference_value - oracle_value,
        "reference_to_oracle_improvement_percent": (
            100.0 * (reference_value - oracle_value) / reference_value
        ),
        "within_1_percent_candidate_count": len(near_1),
        "within_5_percent_candidate_count": len(near_5),
        "oracle_available": True,
    }


def compare_gray_box(
    definition: FrozenBenchmarkDefinition,
    leg: MechanicalLegDefinition,
    domain: SubjectSpecificV3CandidateDomain,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit the unchanged five-parameter adapter using reference evidence only."""

    truth_by_id = {row["candidate_id"]: row for row in rows if row["valid"]}
    reference = truth_by_id[domain.reference.candidate_id]
    adapter = SubjectSpecificFullDynamicsGrayBoxEndpointAdapter(domain)
    adapter.fit(
        [
            EpisodeObservation(
                episode_id=f"{leg.leg_id}:REFERENCE_GRAYBOX_FIT",
                trial_index=1,
                candidate_id=domain.reference.candidate_id,
                beta_flex=0.0,
                beta_extend=0.0,
                endpoint_name=definition.endpoint_name,
                endpoint_value=float(reference["endpoint_value_nm"]),
                endpoint_unit=definition.endpoint_unit,
                endpoint_uncertainty=definition.reference_endpoint_uncertainty_nm,
                valid=True,
                metadata={
                    "classification": definition.classification,
                    "evidence_scope": "REFERENCE_CANDIDATE_ONLY",
                },
            )
        ]
    )
    predictions = {
        candidate.candidate_id: float(adapter.predict_value(candidate))
        for candidate in domain
    }
    for row in rows:
        row["gray_box_predicted_J_nm"] = predictions[row["candidate_id"]]

    ordered = sorted(truth_by_id.values(), key=lambda row: row["candidate_index"])
    truth = np.asarray([row["endpoint_value_nm"] for row in ordered], dtype=float)
    predicted = np.asarray(
        [predictions[row["candidate_id"]] for row in ordered], dtype=float
    )
    residual = predicted - truth
    rmse = float(np.sqrt(np.mean(residual**2)))
    truth_range = float(np.ptp(truth))
    normalized_rmse = rmse / truth_range if truth_range > 0.0 else math.nan
    correlation = float(spearmanr(truth, predicted).statistic)
    predicted_best = min(
        ordered,
        key=lambda row: (predictions[row["candidate_id"]], row["candidate_index"]),
    )
    true_best = min(
        ordered, key=lambda row: (row["endpoint_value_nm"], row["candidate_index"])
    )
    predicted_best_truth = float(predicted_best["endpoint_value_nm"])
    true_best_value = float(true_best["endpoint_value_nm"])
    misleading_threshold = float(
        definition.analysis["directionally_misleading_spearman_threshold"]
    )
    weak_threshold = float(definition.analysis["weak_ranking_spearman_threshold"])
    if correlation < misleading_threshold:
        behavior = "DIRECTIONALLY_MISLEADING"
    elif correlation < weak_threshold:
        behavior = "WEAK_RANKING_INFORMATION"
    else:
        behavior = "INFORMATIVE_RANKING"
    metadata = adapter.metadata()
    return {
        "leg_id": leg.leg_id,
        "fit_evidence": "REFERENCE_CANDIDATE_ONLY",
        "fit_valid_episode_count": metadata["fit_valid_episode_count"],
        "estimated_effective_parameters": metadata["estimated_parameters"],
        "rmse_nm": rmse,
        "normalized_rmse_by_truth_range": normalized_rmse,
        "spearman_rank_correlation": correlation,
        "predicted_oracle_candidate_id": predicted_best["candidate_id"],
        "predicted_oracle_canonical_beta_id": predicted_best[
            "canonical_beta_id"
        ],
        "predicted_oracle_beta": [
            predicted_best["beta_flex"],
            predicted_best["beta_extend"],
        ],
        "true_oracle_beta": [true_best["beta_flex"], true_best["beta_extend"]],
        "regret_of_physics_predicted_best_nm": predicted_best_truth - true_best_value,
        "relative_regret_of_physics_predicted_best": (
            predicted_best_truth / true_best_value - 1.0
        ),
        "ranking_behavior": behavior,
        "directionally_misleading": behavior == "DIRECTIONALLY_MISLEADING",
    }


def pairwise_landscape_analysis(
    rows_by_leg: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for first_id, second_id in combinations(rows_by_leg, 2):
        first = np.asarray(
            [
                row["endpoint_value_nm"]
                for row in sorted(
                    rows_by_leg[first_id], key=lambda item: item["candidate_index"]
                )
            ],
            dtype=float,
        )
        second = np.asarray(
            [
                row["endpoint_value_nm"]
                for row in sorted(
                    rows_by_leg[second_id], key=lambda item: item["candidate_index"]
                )
            ],
            dtype=float,
        )
        scale = float(np.dot(first, second) / np.dot(first, first))
        scaling_residual = second - scale * first
        second_range = float(np.ptp(second))
        output.append(
            {
                "leg_a": first_id,
                "leg_b": second_id,
                "spearman_rank_correlation": float(
                    spearmanr(first, second).statistic
                ),
                "best_proportional_scale_b_from_a": scale,
                "proportional_scaling_rmse_nm": float(
                    np.sqrt(np.mean(scaling_residual**2))
                ),
                "proportional_scaling_nrmse_by_leg_b_range": (
                    float(np.sqrt(np.mean(scaling_residual**2))) / second_range
                    if second_range > 0.0
                    else math.nan
                ),
            }
        )
    return output


def personalization_necessity_analysis(
    definition: FrozenBenchmarkDefinition,
    rows_by_leg: dict[str, list[dict[str, Any]]],
    oracle_rows: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_leg_ids = [leg.leg_id for leg in definition.legs]
    landscapes = np.asarray(
        [
            [
                row["endpoint_value_nm"]
                for row in sorted(
                    rows_by_leg[leg_id], key=lambda item: item["candidate_index"]
                )
            ]
            for leg_id in ordered_leg_ids
        ],
        dtype=float,
    )
    reference_index = next(
        row["candidate_index"]
        for row in rows_by_leg[ordered_leg_ids[0]]
        if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
    )
    normalized = landscapes / landscapes[:, reference_index, np.newaxis]
    common_shape = np.mean(normalized, axis=0)
    interaction = normalized - common_shape[np.newaxis, :]
    total = normalized - float(np.mean(normalized))
    interaction_rms = float(np.sqrt(np.mean(interaction**2)))
    interaction_max_abs = float(np.max(np.abs(interaction)))
    interaction_fraction = float(
        np.sum(interaction**2) / np.sum(total**2)
    )
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    rank_one_fraction = float(
        singular_values[0] ** 2 / np.sum(singular_values**2)
    )

    relative_regret = landscapes / np.min(landscapes, axis=1)[:, np.newaxis] - 1.0
    universal_1 = np.flatnonzero(np.all(relative_regret <= 0.01, axis=0))
    universal_5 = np.flatnonzero(np.all(relative_regret <= 0.05, axis=0))
    minimax_index = int(np.argmin(np.max(relative_regret, axis=0)))
    canonical_rows = sorted(
        rows_by_leg[ordered_leg_ids[0]], key=lambda item: item["candidate_index"]
    )
    minimax = canonical_rows[minimax_index]
    unique_oracles = {
        tuple(float(value) for value in row["oracle_beta"]) for row in oracle_rows
    }
    minimum_spearman = min(row["spearman_rank_correlation"] for row in pairwise)
    criteria = definition.analysis["personalization_readiness"]
    all_stable = all(
        audit["valid_candidate_count"] == 625
        and audit["invalid_candidate_count"] == 0
        and audit["endpoint_reproducible"]
        and audit["pointwise_clipping_count"] == 0
        for audit in audits
    )
    distinct_oracles = len(unique_oracles) >= int(criteria["minimum_unique_oracle_count"])
    distinct_ordering = minimum_spearman <= float(
        criteria["maximum_pairwise_spearman_for_distinct_ordering"]
    )
    material_interaction = interaction_fraction >= float(
        criteria["minimum_normalized_interaction_fraction"]
    )
    no_universal_near_oracle = len(universal_5) == 0
    supports_personalization = all(
        (distinct_oracles, distinct_ordering, material_interaction, no_universal_near_oracle)
    )
    ready = all_stable and supports_personalization
    conclusion = (
        "MUJOCO_BENCHMARK_SUPPORTS_PERSONALIZATION_NECESSITY"
        if supports_personalization
        else "MUJOCO_BENCHMARK_DOES_NOT_SUPPORT_PERSONALIZATION_NECESSITY"
    )
    return {
        "unique_oracle_beta_count": len(unique_oracles),
        "oracle_betas": [list(values) for values in sorted(unique_oracles)],
        "pairwise_spearman_minimum": minimum_spearman,
        "pairwise_spearman_median": float(
            np.median([row["spearman_rank_correlation"] for row in pairwise])
        ),
        "normalized_subject_by_trajectory_interaction_fraction": interaction_fraction,
        "normalized_subject_by_trajectory_interaction_rms": interaction_rms,
        "normalized_subject_by_trajectory_interaction_max_abs": interaction_max_abs,
        "normalized_landscape_range_by_leg": {
            leg_id: float(np.ptp(normalized[index]))
            for index, leg_id in enumerate(ordered_leg_ids)
        },
        "normalized_landscape_rank_one_explained_fraction": rank_one_fraction,
        "universal_within_1_percent_candidate_count": len(universal_1),
        "universal_within_5_percent_candidate_count": len(universal_5),
        "universal_within_1_percent_candidate_indices": universal_1.tolist(),
        "universal_within_5_percent_candidate_indices": universal_5.tolist(),
        "minimax_common_candidate": {
            "candidate_index": minimax_index,
            "canonical_beta_id": minimax["canonical_beta_id"],
            "beta": [minimax["beta_flex"], minimax["beta_extend"]],
            "maximum_relative_regret": float(
                np.max(relative_regret[:, minimax_index])
            ),
            "per_leg_relative_regret": {
                leg_id: float(relative_regret[index, minimax_index])
                for index, leg_id in enumerate(ordered_leg_ids)
            },
        },
        "all_legs_stable_625": all_stable,
        "distinct_oracle_criterion": distinct_oracles,
        "distinct_ordering_criterion": distinct_ordering,
        "material_interaction_criterion": material_interaction,
        "no_universal_5_percent_candidate_criterion": no_universal_near_oracle,
        "conclusion": conclusion,
        "READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO": ready,
    }


def run_full_benchmark() -> dict[str, Any]:
    definition = load_frozen_benchmark_definition()
    rows_by_leg: dict[str, list[dict[str, Any]]] = {}
    domains: dict[str, SubjectSpecificV3CandidateDomain] = {}
    audits = []
    oracle_rows = []
    gray_box_rows = []
    for leg in definition.legs:
        domain, rows, audit = generate_leg_landscape(definition, leg)
        domains[leg.leg_id] = domain
        rows_by_leg[leg.leg_id] = rows
        audits.append(audit)
        oracle_rows.append(characterize_oracle(leg, domain, rows))
        gray_box_rows.append(compare_gray_box(definition, leg, domain, rows))

    pairwise = pairwise_landscape_analysis(rows_by_leg)
    necessity = personalization_necessity_analysis(
        definition, rows_by_leg, oracle_rows, pairwise, audits
    )
    ready = necessity["READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO"]
    status = (
        "COMPLETE"
        if all(audit["valid_candidate_count"] == 625 for audit in audits)
        else "COMPLETE_WITH_LIMITATIONS"
    )
    return {
        "benchmark_id": definition.benchmark_id,
        "classification": definition.classification,
        "FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1": status,
        "READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO": ready,
        "personalization_necessity_conclusion": necessity["conclusion"],
        "mujoco_version": __import__("mujoco").__version__,
        "parameters_frozen_before_landscape_generation": True,
        "coordinate_convention": "theta_shank = q_hip - q_knee",
        "endpoint": {
            "name": definition.endpoint_name,
            "unit": definition.endpoint_unit,
            "interpretation": "SIMULATED_MECHANICAL_ENDPOINT",
        },
        "leg_count": len(definition.legs),
        "candidate_count_per_leg": 625,
        "total_candidate_replays": sum(len(rows) for rows in rows_by_leg.values()),
        "leg_parameters_and_rom": [leg.as_table_row() for leg in definition.legs],
        "trajectory_audits": audits,
        "oracle_characterization": oracle_rows,
        "pairwise_landscape_analysis": pairwise,
        "gray_box_prediction_quality": gray_box_rows,
        "personalization_necessity_analysis": necessity,
        "landscape_rows": [
            row for leg_id in rows_by_leg for row in rows_by_leg[leg_id]
        ],
        "algorithm_runs": 0,
        "robot_actions": 0,
        "PINN_training": 0,
    }


__all__ = [
    "build_leg_domain",
    "characterize_oracle",
    "compare_gray_box",
    "generate_leg_landscape",
    "pairwise_landscape_analysis",
    "personalization_necessity_analysis",
    "run_full_benchmark",
]
