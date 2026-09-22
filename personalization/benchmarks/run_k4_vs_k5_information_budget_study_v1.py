"""Paired K=4 versus K=5 prior-identification information-budget study."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from personalization.active_diagnostic_v1 import (
    PHYSICS_MODE,
    STANDARD_MODE,
    ROMGatedActiveDiagnosticEpisode,
)
from personalization.benchmarks.metrics import evaluate_run
from personalization.benchmarks.run_active_diagnostic_v1 import (
    ACTIVE_DIAGNOSTIC_METHOD,
    DEFAULT_SEEDS,
    PAIR_TOLERANCE,
    _aggregate,
    _environment,
    _physics,
    _write_csv,
)
from personalization.benchmarks.run_equal_budget import NOISE_SETTINGS, PRIOR_QUALITIES
from personalization.environment import make_primary_cases
from personalization.repeated_active_diagnostic_k5_v1 import (
    REPEATED_ACTIVE_DIAGNOSTIC_METHOD,
    ROMGatedRepeatedDiagnosticEpisode,
    verify_frozen_rule,
)
from personalization.rom_gated_v2 import (
    ROMGatedPersonalizationEpisode,
    SubjectSpecificV3CandidateDomain,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "personalization" / "benchmarks" / "results_k4_vs_k5_information_budget_v1"
)
K4_ACTIVE_SUMMARY = (
    ROOT
    / "personalization"
    / "benchmarks"
    / "results_active_diagnostic_v1"
    / "benchmark_summary.json"
)
STANDARD_K4 = "Standard BO K=4"
STANDARD_K5 = "Standard BO K=5"
FIXED_K4 = "Fixed Physics BO K=4"
FIXED_K5 = "Fixed Physics BO K=5"
ACTIVE_K4 = "Active Diagnostic Arbitration K=4"
REPEATED_K5 = "Repeated Active Diagnostic Arbitration K=5"
METHODS = (STANDARD_K4, STANDARD_K5, FIXED_K4, FIXED_K5, ACTIVE_K4, REPEATED_K5)
HIGHLY_SIMILAR_BETA_DISTANCE = 0.005


def _group_runs(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    fields = ("case", "seed", "prior_quality", "noise_label")
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(row[field] for field in fields)
        grouped.setdefault(identity, {})[row["method"]] = row
    return grouped


def _aggregate_value(
    aggregate: list[dict[str, Any]],
    quality: str,
    noise: str,
    method: str,
    metric: str,
) -> float:
    return float(
        next(
            row[metric]
            for row in aggregate
            if row["prior_quality"] == quality
            and row["noise_label"] == noise
            and row["method"] == method
        )
    )


def _paired_wtl(
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]],
    *,
    quality: str,
    noise: str,
    method: str,
    baseline: str,
    metric: str = "final_regret",
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            methods[method][metric] - methods[baseline][metric]
            for identity, methods in grouped.items()
            if identity[2] == quality and identity[3] == noise
        ],
        dtype=float,
    )
    return {
        "method": method,
        "baseline": baseline,
        "wins": int(np.sum(deltas < -PAIR_TOLERANCE)),
        "ties": int(np.sum(np.abs(deltas) <= PAIR_TOLERANCE)),
        "losses": int(np.sum(deltas > PAIR_TOLERANCE)),
        "paired_delta_mean": float(np.mean(deltas)),
        "paired_delta_median": float(np.median(deltas)),
    }


def _historical_k4_equivalence(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    historical = json.loads(K4_ACTIVE_SUMMARY.read_text(encoding="utf-8"))
    old = {
        (row["method"], row["prior_quality"], row["noise_label"]): row
        for row in historical["aggregate"]
    }
    method_map = {
        STANDARD_K4: "Standard BO",
        FIXED_K4: "Fixed Physics BO",
        ACTIVE_K4: ACTIVE_DIAGNOSTIC_METHOD,
    }
    maximum_error = 0.0
    compared = 0
    for row in aggregate:
        if row["method"] not in method_map:
            continue
        parent = old[
            (method_map[row["method"]], row["prior_quality"], row["noise_label"])
        ]
        for metric in (
            "final_regret_mean",
            "final_regret_median",
            "final_regret_std",
            "best_seen_regret_mean",
        ):
            maximum_error = max(
                maximum_error, abs(float(row[metric]) - float(parent[metric]))
            )
            compared += 1
    return {
        "historical_summary_path": str(K4_ACTIVE_SUMMARY.relative_to(ROOT)),
        "compared_metric_count": compared,
        "maximum_absolute_error": maximum_error,
        "exact_within_1e_12": maximum_error <= PAIR_TOLERANCE,
    }


def _arbitration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in (ACTIVE_K4, REPEATED_K5):
        for quality in PRIOR_QUALITIES:
            for noise in NOISE_SETTINGS:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["prior_quality"] == quality
                    and row["noise_label"] == noise
                ]
                physics_count = sum(
                    row["selected_expert"] == PHYSICS_MODE for row in selected
                )
                standard_count = len(selected) - physics_count
                informative = quality in {"P0", "P1", "P2"}
                output.append(
                    {
                        "method": method,
                        "prior_quality": quality,
                        "noise_label": noise,
                        "episode_count": len(selected),
                        "physics_selection_count": physics_count,
                        "physics_selection_rate": physics_count / len(selected),
                        "standard_selection_count": standard_count,
                        "standard_selection_rate": standard_count / len(selected),
                        "false_rejection_count": standard_count if informative else None,
                        "missed_rejection_count": physics_count if not informative else None,
                    }
                )
    return output


def _informative_cost(
    aggregate: list[dict[str, Any]],
    arbitration: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for quality in ("P0", "P1", "P2"):
        for noise in NOISE_SETTINGS:
            standard_k5 = _aggregate_value(
                aggregate, quality, noise, STANDARD_K5, "final_regret_mean"
            )
            fixed_k5 = _aggregate_value(
                aggregate, quality, noise, FIXED_K5, "final_regret_mean"
            )
            repeated = _aggregate_value(
                aggregate, quality, noise, REPEATED_K5, "final_regret_mean"
            )
            active_k4 = _aggregate_value(
                aggregate, quality, noise, ACTIVE_K4, "final_regret_mean"
            )
            fixed_benefit = standard_k5 - fixed_k5
            repeated_benefit = standard_k5 - repeated
            arbitration_row = next(
                row
                for row in arbitration
                if row["method"] == REPEATED_K5
                and row["prior_quality"] == quality
                and row["noise_label"] == noise
            )
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "standard_k5_final_regret_mean": standard_k5,
                    "fixed_k5_final_regret_mean": fixed_k5,
                    "repeated_k5_final_regret_mean": repeated,
                    "active_k4_final_regret_mean": active_k4,
                    "repeated_k5_cost_vs_fixed_k5": repeated - fixed_k5,
                    "repeated_k5_difference_vs_active_k4": repeated - active_k4,
                    "repeated_k5_best_seen_difference_vs_active_k4": (
                        _aggregate_value(
                            aggregate,
                            quality,
                            noise,
                            REPEATED_K5,
                            "best_seen_regret_mean",
                        )
                        - _aggregate_value(
                            aggregate,
                            quality,
                            noise,
                            ACTIVE_K4,
                            "best_seen_regret_mean",
                        )
                    ),
                    "fixed_k5_benefit_vs_standard_k5": fixed_benefit,
                    "repeated_k5_benefit_vs_standard_k5": repeated_benefit,
                    "benefit_retention_fraction": (
                        repeated_benefit / fixed_benefit
                        if fixed_benefit > PAIR_TOLERANCE
                        else None
                    ),
                    "physics_selection_rate": arbitration_row[
                        "physics_selection_rate"
                    ],
                    "false_rejection_rate": arbitration_row[
                        "standard_selection_rate"
                    ],
                }
            )
    return output


def _evidence_consistency_summary(
    diagnostic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    categories = (
        "CONSISTENT_FAVORS_PHYSICS",
        "CONSISTENT_FAVORS_STANDARD",
        "CONFLICTING_EVIDENCE",
        "NEUTRAL_OR_PARTIALLY_NEUTRAL_EVIDENCE",
        "INCOMPLETE_INVALID_EVIDENCE",
    )
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            selected = [
                row
                for row in diagnostic_rows
                if row["prior_quality"] == quality and row["noise_label"] == noise
            ]
            row: dict[str, Any] = {
                "prior_quality": quality,
                "noise_label": noise,
                "episode_count": len(selected),
            }
            for category in categories:
                row[category] = sum(
                    item["evidence_consistency"] == category for item in selected
                )
            row["trial_2_favors_physics_count"] = sum(
                item["trial_2_evidence_sign"] == "FAVORS_PHYSICS"
                for item in selected
            )
            row["trial_2_favors_standard_count"] = sum(
                item["trial_2_evidence_sign"] == "FAVORS_STANDARD"
                for item in selected
            )
            row["trial_3_favors_physics_count"] = sum(
                item["trial_3_evidence_sign"] == "FAVORS_PHYSICS"
                for item in selected
            )
            row["trial_3_favors_standard_count"] = sum(
                item["trial_3_evidence_sign"] == "FAVORS_STANDARD"
                for item in selected
            )
            output.append(row)
    return output


def _information_gain_summary(
    diagnostic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            selected = [
                row
                for row in diagnostic_rows
                if row["prior_quality"] == quality and row["noise_label"] == noise
            ]
            distances = np.asarray(
                [row["diagnostic_beta_distance"] for row in selected], dtype=float
            )
            divergence_2 = np.asarray(
                [row["diagnostic_divergence_2"] for row in selected], dtype=float
            )
            divergence_3 = np.asarray(
                [row["diagnostic_divergence_3"] for row in selected], dtype=float
            )
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "episode_count": len(selected),
                    "duplicate_diagnostic_candidate_count": sum(
                        row["diagnostic_candidate_repeated"] for row in selected
                    ),
                    "diagnostic_beta_distance_mean": float(np.mean(distances)),
                    "diagnostic_beta_distance_median": float(np.median(distances)),
                    "highly_similar_distance_threshold": HIGHLY_SIMILAR_BETA_DISTANCE,
                    "highly_similar_candidate_count": int(
                        np.sum(distances <= HIGHLY_SIMILAR_BETA_DISTANCE)
                    ),
                    "diagnostic_divergence_2_mean": float(np.mean(divergence_2)),
                    "diagnostic_divergence_3_mean": float(np.mean(divergence_3)),
                    "divergence_3_minus_2_mean": float(
                        np.mean(divergence_3 - divergence_2)
                    ),
                }
            )
    return output


def _p3_analysis(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    arbitration: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _group_runs(rows)
    output = []
    for noise in NOISE_SETTINGS:
        k4_arbitration = next(
            row
            for row in arbitration
            if row["method"] == ACTIVE_K4
            and row["prior_quality"] == "P3"
            and row["noise_label"] == noise
        )
        k5_arbitration = next(
            row
            for row in arbitration
            if row["method"] == REPEATED_K5
            and row["prior_quality"] == "P3"
            and row["noise_label"] == noise
        )
        standard_k4 = _aggregate_value(
            aggregate, "P3", noise, STANDARD_K4, "final_regret_mean"
        )
        standard_k5 = _aggregate_value(
            aggregate, "P3", noise, STANDARD_K5, "final_regret_mean"
        )
        active_k4 = _aggregate_value(
            aggregate, "P3", noise, ACTIVE_K4, "final_regret_mean"
        )
        repeated_k5 = _aggregate_value(
            aggregate, "P3", noise, REPEATED_K5, "final_regret_mean"
        )
        output.append(
            {
                "noise_label": noise,
                "mean_final_regret": {
                    method: _aggregate_value(
                        aggregate, "P3", noise, method, "final_regret_mean"
                    )
                    for method in METHODS
                },
                "median_final_regret": {
                    method: _aggregate_value(
                        aggregate, "P3", noise, method, "final_regret_median"
                    )
                    for method in METHODS
                },
                "k4_rejection_rate": k4_arbitration["standard_selection_rate"],
                "k5_rejection_rate": k5_arbitration["standard_selection_rate"],
                "k4_missed_rejection_count": k4_arbitration[
                    "missed_rejection_count"
                ],
                "k5_missed_rejection_count": k5_arbitration[
                    "missed_rejection_count"
                ],
                "missed_rejection_reduction": (
                    k4_arbitration["missed_rejection_count"]
                    - k5_arbitration["missed_rejection_count"]
                ),
                "k4_negative_transfer_vs_standard_k4": active_k4 - standard_k4,
                "k5_negative_transfer_vs_standard_k5": repeated_k5 - standard_k5,
                "k5_vs_corresponding_standard_wtl": _paired_wtl(
                    grouped,
                    quality="P3",
                    noise=noise,
                    method=REPEATED_K5,
                    baseline=STANDARD_K5,
                ),
                "k4_vs_corresponding_standard_wtl": _paired_wtl(
                    grouped,
                    quality="P3",
                    noise=noise,
                    method=ACTIVE_K4,
                    baseline=STANDARD_K4,
                ),
                "k5_vs_k4_active_wtl": _paired_wtl(
                    grouped,
                    quality="P3",
                    noise=noise,
                    method=REPEATED_K5,
                    baseline=ACTIVE_K4,
                ),
                "k5_minus_k4_mean_final_regret": repeated_k5 - active_k4,
            }
        )
    return output


def _moderate_missed_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_runs(rows)
    selected = [
        methods
        for identity, methods in grouped.items()
        if identity[2] == "P3" and identity[3] == "moderate"
    ]
    k4_missed = [
        methods for methods in selected if methods[ACTIVE_K4]["selected_expert"] == PHYSICS_MODE
    ]
    k5_missed = [
        methods for methods in selected if methods[REPEATED_K5]["selected_expert"] == PHYSICS_MODE
    ]
    corrected = [
        methods
        for methods in k4_missed
        if methods[REPEATED_K5]["selected_expert"] == STANDARD_MODE
    ]
    newly_missed = [
        methods
        for methods in selected
        if methods[ACTIVE_K4]["selected_expert"] == STANDARD_MODE
        and methods[REPEATED_K5]["selected_expert"] == PHYSICS_MODE
    ]
    return {
        "k4_missed_rejection_count": len(k4_missed),
        "k5_missed_rejection_count": len(k5_missed),
        "k4_missed_corrected_by_k5_count": len(corrected),
        "k4_correct_but_k5_missed_count": len(newly_missed),
        "k4_missed_episode_mean_regret_at_k4": float(
            np.mean([methods[ACTIVE_K4]["final_regret"] for methods in k4_missed])
        ),
        "same_k4_missed_episodes_mean_regret_at_k5": float(
            np.mean([methods[REPEATED_K5]["final_regret"] for methods in k4_missed])
        ),
        "k5_missed_episode_mean_regret": float(
            np.mean([methods[REPEATED_K5]["final_regret"] for methods in k5_missed])
        ),
        "k5_standard_selected_episode_mean_regret": float(
            np.mean(
                [
                    methods[REPEATED_K5]["final_regret"]
                    for methods in selected
                    if methods[REPEATED_K5]["selected_expert"] == STANDARD_MODE
                ]
            )
        ),
    }


def _study_decision(
    informative: list[dict[str, Any]],
    arbitration: list[dict[str, Any]],
    p3: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [
        row
        for row in informative
        if row["fixed_k5_benefit_vs_standard_k5"] > PAIR_TOLERANCE
    ]
    fixed_benefit = sum(row["fixed_k5_benefit_vs_standard_k5"] for row in positive)
    repeated_benefit = sum(
        row["repeated_k5_benefit_vs_standard_k5"] for row in positive
    )
    retention = repeated_benefit / fixed_benefit if fixed_benefit else math.nan
    false_rejection_rates = [
        row["standard_selection_rate"]
        for row in arbitration
        if row["method"] == REPEATED_K5
        and row["prior_quality"] in {"P0", "P1", "P2"}
    ]
    k4_total_missed = sum(row["k4_missed_rejection_count"] for row in p3)
    k5_total_missed = sum(row["k5_missed_rejection_count"] for row in p3)
    moderate = next(row for row in p3 if row["noise_label"] == "moderate")

    informative_retained = retention >= 0.75
    false_rejection_controlled = max(false_rejection_rates) <= 0.25
    clear_identification_improvement = (
        k5_total_missed <= math.floor(0.5 * k4_total_missed)
        and moderate["k5_missed_rejection_count"]
        <= math.floor(0.5 * moderate["k4_missed_rejection_count"])
    )
    moderate_regret_materially_improved = (
        moderate["k5_minus_k4_mean_final_regret"]
        <= -0.25 * moderate["mean_final_regret"][ACTIVE_K4]
    )
    if (
        informative_retained
        and clear_identification_improvement
        and moderate_regret_materially_improved
    ):
        scientific_conclusion = "K4_INFORMATION_LIMITATION_SUPPORTED"
    elif (
        k5_total_missed >= k4_total_missed
        or moderate["k5_minus_k4_mean_final_regret"] >= 0.0
    ):
        scientific_conclusion = (
            "ADDITIONAL_DIAGNOSTIC_TRIAL_DOES_NOT_RESOLVE_PRIOR_IDENTIFICATION"
        )
    else:
        scientific_conclusion = "EVIDENCE_MIXED"

    stable_p3_rejection = all(row["k5_rejection_rate"] >= 0.75 for row in p3)
    acceptable_p3_transfer = all(
        row["k5_negative_transfer_vs_standard_k5"] <= 0.05 for row in p3
    )
    ready = all(
        (
            informative_retained,
            false_rejection_controlled,
            clear_identification_improvement,
            moderate_regret_materially_improved,
            stable_p3_rejection,
            acceptable_p3_transfer,
        )
    )
    return {
        "scientific_conclusion": scientific_conclusion,
        "informative_positive_benefit_retention_fraction": retention,
        "informative_benefit_retained_at_75_percent": informative_retained,
        "maximum_informative_false_rejection_rate": max(false_rejection_rates),
        "informative_false_rejection_rate_at_most_25_percent": false_rejection_controlled,
        "k4_total_p3_missed_rejection_count": k4_total_missed,
        "k5_total_p3_missed_rejection_count": k5_total_missed,
        "clear_identification_improvement": clear_identification_improvement,
        "moderate_regret_materially_improved_by_25_percent": (
            moderate_regret_materially_improved
        ),
        "k5_p3_rejection_at_least_75_percent_each_noise": stable_p3_rejection,
        "k5_p3_negative_transfer_at_most_0_05_each_noise": acceptable_p3_transfer,
        "FREEZE_PRIMARY_PERSONALIZATION_BUDGET": "K5" if ready else "NONE",
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
    }


def run_study(seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    verify_frozen_rule()
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != DEFAULT_SEEDS:
        raise ValueError("formal K4-vs-K5 study seeds are frozen at 0...4")
    rom_controller, profile = determine_synthetic_rom(
        make_offline_rom_development_cases()[1]
    )
    domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for noise_label, noise_std in NOISE_SETTINGS.items():
        for case in make_primary_cases(noise_std):
            for quality in PRIOR_QUALITIES:
                for seed in seed_values:
                    condition_results = {}
                    for method in METHODS:
                        environment = _environment(domain, case, seed)
                        episode_id = f"{case.name}:{quality}:{noise_label}:{seed}:{method}"
                        if method in {STANDARD_K4, STANDARD_K5}:
                            budget = 4 if method == STANDARD_K4 else 5
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=episode_id,
                                profile=profile,
                                domain=domain,
                                adaptation_budget=budget,
                            ).run(environment, method="Standard BO")
                        elif method in {FIXED_K4, FIXED_K5}:
                            budget = 4 if method == FIXED_K4 else 5
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=episode_id,
                                profile=profile,
                                domain=domain,
                                adaptation_budget=budget,
                            ).run(
                                environment,
                                method="Physics-Informed BO",
                                physics_model=_physics(case, quality),
                            )
                        elif method == ACTIVE_K4:
                            result = ROMGatedActiveDiagnosticEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(
                                environment,
                                physics_model=_physics(case, quality),
                            )
                        else:
                            result = ROMGatedRepeatedDiagnosticEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(
                                environment,
                                physics_model=_physics(case, quality),
                            )
                        if environment.oracle_access_count != 0:
                            raise RuntimeError("algorithm accessed evaluator oracle")
                        metrics = evaluate_run(result.sequential_result, environment)
                        metrics.update(
                            {
                                "method": method,
                                "case": case.name,
                                "seed": seed,
                                "prior_quality": quality,
                                "noise_label": noise_label,
                                "noise_std": noise_std,
                                "rom_profile_id": profile.profile_id,
                                "rom_profile_fingerprint": profile.fingerprint,
                                "selected_expert": None,
                            }
                        )
                        if method == ACTIVE_K4:
                            metrics["selected_expert"] = (
                                result.sequential_result.selected_expert
                            )
                        elif method == REPEATED_K5:
                            repeated = result.sequential_result
                            metrics["selected_expert"] = repeated.selected_expert
                            diagnostic_2, diagnostic_3 = repeated.diagnostic_selections
                            evidence_2, evidence_3 = repeated.diagnostic_evidence
                            beta_distance = math.dist(
                                diagnostic_2.candidate.beta,
                                diagnostic_3.candidate.beta,
                            )
                            diagnostic_rows.append(
                                {
                                    "case": case.name,
                                    "seed": seed,
                                    "prior_quality": quality,
                                    "noise_label": noise_label,
                                    "diagnostic_candidate_2": (
                                        diagnostic_2.candidate.candidate_id
                                    ),
                                    "diagnostic_beta_2": list(
                                        diagnostic_2.candidate.beta
                                    ),
                                    "diagnostic_divergence_2": (
                                        diagnostic_2.diagnostic_score
                                    ),
                                    "delta_score_2": evidence_2.score_difference,
                                    "trial_2_evidence_sign": evidence_2.evidence_sign,
                                    "diagnostic_candidate_3": (
                                        diagnostic_3.candidate.candidate_id
                                    ),
                                    "diagnostic_beta_3": list(
                                        diagnostic_3.candidate.beta
                                    ),
                                    "diagnostic_divergence_3": (
                                        diagnostic_3.diagnostic_score
                                    ),
                                    "delta_score_3": evidence_3.score_difference,
                                    "trial_3_evidence_sign": evidence_3.evidence_sign,
                                    "cumulative_evidence": (
                                        repeated.arbitration.cumulative_evidence
                                    ),
                                    "evidence_consistency": (
                                        repeated.arbitration.evidence_consistency
                                    ),
                                    "selected_expert": repeated.selected_expert,
                                    "diagnostic_candidate_repeated": (
                                        diagnostic_2.candidate.candidate_id
                                        == diagnostic_3.candidate.candidate_id
                                    ),
                                    "diagnostic_beta_distance": beta_distance,
                                }
                            )
                        rows.append(metrics)
                        condition_results[method] = result
                    fingerprints = {
                        result.rom_profile_fingerprint
                        for result in condition_results.values()
                    }
                    if fingerprints != {profile.fingerprint}:
                        raise RuntimeError("paired methods used different ROM fingerprints")

    aggregate = _aggregate(rows)
    historical = _historical_k4_equivalence(aggregate)
    if not historical["exact_within_1e_12"]:
        raise RuntimeError("frozen K4 historical behavior drifted")
    arbitration = _arbitration_summary(rows)
    informative = _informative_cost(aggregate, arbitration)
    consistency = _evidence_consistency_summary(diagnostic_rows)
    information_gain = _information_gain_summary(diagnostic_rows)
    p3 = _p3_analysis(rows, aggregate, arbitration)
    moderate_missed = _moderate_missed_analysis(rows)
    decision = _study_decision(informative, arbitration, p3)
    ready = decision["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"]
    return {
        "status": "K4_VS_K5_PRIOR_IDENTIFICATION_INFORMATION_BUDGET_STUDY_V1_COMPLETED",
        "classification": "OFFLINE_ALGORITHM_DEVELOPMENT_EVIDENCE_ONLY",
        "scientific_conclusion": decision["scientific_conclusion"],
        "FREEZE_PRIMARY_PERSONALIZATION_BUDGET": decision[
            "FREEZE_PRIMARY_PERSONALIZATION_BUDGET"
        ],
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
        "primary_budgets_compared": [4, 5],
        "seeds": list(seed_values),
        "prior_qualities": list(PRIOR_QUALITIES),
        "noise_settings": NOISE_SETTINGS,
        "truth_case_count_per_noise": len(make_primary_cases()),
        "candidate_count": len(domain),
        "rom_calibration_observation_count": len(rom_controller.ledger.entries),
        "rom_calibration_excluded_from_both_budgets": True,
        "rom_profile": profile.as_dict(),
        "same_rom_fingerprint_for_all_paired_methods": True,
        "methods": list(METHODS),
        "run_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "arbitration_summary": arbitration,
        "informative_prior_cost": informative,
        "diagnostic_evidence_rows": diagnostic_rows,
        "evidence_consistency_summary": consistency,
        "information_gain_summary": information_gain,
        "p3_analysis": p3,
        "p3_moderate_missed_rejection_analysis": moderate_missed,
        "study_decision": decision,
        "historical_k4_equivalence": historical,
        "causal_audit": {
            "k4_algorithm_modified": False,
            "trial_2_prediction_history": "D1",
            "trial_3_prediction_history": "D2",
            "arbitration_occurs_after_trial_3": True,
            "selected_expert_fixed_for_trials_4_and_5": True,
            "case_label_available_to_algorithm": False,
            "future_truth_available_to_algorithm": False,
        },
        "robot_actions": 0,
        "human_actions": 0,
        "PINN_training": 0,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "study_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "run_metrics.csv", payload["rows"])
    _write_csv(output_dir / "aggregate_metrics.csv", payload["aggregate"])
    _write_csv(output_dir / "arbitration_summary.csv", payload["arbitration_summary"])
    _write_csv(output_dir / "informative_prior_cost.csv", payload["informative_prior_cost"])
    _write_csv(output_dir / "diagnostic_evidence.csv", payload["diagnostic_evidence_rows"])
    _write_csv(
        output_dir / "evidence_consistency_summary.csv",
        payload["evidence_consistency_summary"],
    )
    _write_csv(
        output_dir / "information_gain_summary.csv",
        payload["information_gain_summary"],
    )
    _write_csv(output_dir / "p3_analysis.csv", payload["p3_analysis"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_study()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scientific_conclusion": payload["scientific_conclusion"],
                "run_count": payload["run_count"],
                "historical_k4_exact": payload["historical_k4_equivalence"][
                    "exact_within_1e_12"
                ],
                "FREEZE_PRIMARY_PERSONALIZATION_BUDGET": payload[
                    "FREEZE_PRIMARY_PERSONALIZATION_BUDGET"
                ],
                "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": payload[
                    "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"
                ],
                "robot_actions": payload["robot_actions"],
                "elapsed_ms": elapsed_ms,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
