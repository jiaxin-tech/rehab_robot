"""Five-method paired offline benchmark for active diagnostic arbitration V1."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from personalization.active_diagnostic_v1 import (
    ACTIVE_DIAGNOSTIC_METHOD,
    PHYSICS_MODE,
    STANDARD_MODE,
    ROMGatedActiveDiagnosticEpisode,
    verify_frozen_rule,
)
from personalization.adaptive_trust_v1 import (
    ADAPTIVE_TRUST_METHOD,
    ROMGatedAdaptiveTrustEpisode,
)
from personalization.benchmarks.metrics import evaluate_run
from personalization.benchmarks.run_equal_budget import NOISE_SETTINGS, PRIOR_QUALITIES
from personalization.benchmarks.run_predictive_failover_v1 import (
    DEFAULT_SEEDS,
    FIXED_METHOD,
    PAIR_TOLERANCE,
    STANDARD_METHOD,
    _aggregate,
    _environment,
    _physics,
    _write_csv,
)
from personalization.environment import make_primary_cases
from personalization.predictive_failover_v1 import (
    PREDICTIVE_FAILOVER_METHOD,
    ROMGatedPredictiveFailoverEpisode,
)
from personalization.rom_gated_v2 import (
    ROMGatedPersonalizationEpisode,
    SubjectSpecificV3CandidateDomain,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "personalization" / "benchmarks" / "results_active_diagnostic_v1"
)
PASSIVE_BASELINE_SUMMARY = (
    ROOT
    / "personalization"
    / "benchmarks"
    / "results_predictive_failover_v1"
    / "benchmark_summary.json"
)
METHODS = (
    STANDARD_METHOD,
    FIXED_METHOD,
    ADAPTIVE_TRUST_METHOD,
    PREDICTIVE_FAILOVER_METHOD,
    ACTIVE_DIAGNOSTIC_METHOD,
)
PRIMARY_BUDGET = 4


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


def _pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_runs(rows)
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            conditions = [
                methods
                for identity, methods in grouped.items()
                if identity[2] == quality and identity[3] == noise
            ]
            for baseline in METHODS[:-1]:
                deltas = np.asarray(
                    [
                        methods[ACTIVE_DIAGNOSTIC_METHOD]["final_regret"]
                        - methods[baseline]["final_regret"]
                        for methods in conditions
                    ],
                    dtype=float,
                )
                output.append(
                    {
                        "comparison": f"{ACTIVE_DIAGNOSTIC_METHOD} vs {baseline}",
                        "prior_quality": quality,
                        "noise_label": noise,
                        "paired_run_count": len(deltas),
                        "wins": int(np.sum(deltas < -PAIR_TOLERANCE)),
                        "ties": int(np.sum(np.abs(deltas) <= PAIR_TOLERANCE)),
                        "losses": int(np.sum(deltas > PAIR_TOLERANCE)),
                        "paired_delta_mean": float(np.mean(deltas)),
                        "paired_delta_median": float(np.median(deltas)),
                    }
                )
    return output


def _historical_equivalence(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    historical = json.loads(PASSIVE_BASELINE_SUMMARY.read_text(encoding="utf-8"))
    old = {
        (row["method"], row["prior_quality"], row["noise_label"]): row
        for row in historical["aggregate"]
    }
    maximum_error = 0.0
    compared = 0
    for row in aggregate:
        if row["method"] == ACTIVE_DIAGNOSTIC_METHOD:
            continue
        parent = old[
            (row["method"], row["prior_quality"], row["noise_label"])
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
        "historical_summary_path": str(PASSIVE_BASELINE_SUMMARY.relative_to(ROOT)),
        "compared_metric_count": compared,
        "maximum_absolute_error": maximum_error,
        "exact_within_1e_12": maximum_error <= PAIR_TOLERANCE,
    }


def _arbitration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in rows if row["method"] == ACTIVE_DIAGNOSTIC_METHOD]
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            selected = [
                row
                for row in active
                if row["prior_quality"] == quality and row["noise_label"] == noise
            ]
            physics_count = sum(row["selected_expert"] == PHYSICS_MODE for row in selected)
            standard_count = sum(
                row["selected_expert"] == STANDARD_MODE for row in selected
            )
            inconclusive_count = sum(
                row["arbitration_decision"] == "INCONCLUSIVE_DEFAULT_PHYSICS"
                for row in selected
            )
            informative = quality in {"P0", "P1", "P2"}
            misleading = quality == "P3"
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "episode_count": len(selected),
                    "physics_selection_count": physics_count,
                    "physics_selection_rate": physics_count / len(selected),
                    "standard_selection_count": standard_count,
                    "standard_selection_rate": standard_count / len(selected),
                    "inconclusive_default_physics_count": inconclusive_count,
                    "correct_arbitration_count": (
                        physics_count if informative else standard_count
                    ),
                    "incorrect_arbitration_count": (
                        standard_count if informative else physics_count
                    ),
                    "false_rejection_of_useful_physics_count": (
                        standard_count if informative else None
                    ),
                    "failure_to_reject_misleading_physics_count": (
                        physics_count if misleading else None
                    ),
                }
            )
    return output


def _benefit_retention(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for quality in ("P0", "P1", "P2"):
        for noise in NOISE_SETTINGS:
            standard = _aggregate_value(
                aggregate, quality, noise, STANDARD_METHOD, "final_regret_mean"
            )
            fixed = _aggregate_value(
                aggregate, quality, noise, FIXED_METHOD, "final_regret_mean"
            )
            active = _aggregate_value(
                aggregate, quality, noise, ACTIVE_DIAGNOSTIC_METHOD, "final_regret_mean"
            )
            fixed_benefit = standard - fixed
            active_benefit = standard - active
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "standard_final_regret_mean": standard,
                    "fixed_final_regret_mean": fixed,
                    "active_final_regret_mean": active,
                    "fixed_physics_benefit_vs_standard": fixed_benefit,
                    "active_benefit_vs_standard": active_benefit,
                    "benefit_retention_fraction": (
                        active_benefit / fixed_benefit
                        if fixed_benefit > PAIR_TOLERANCE
                        else None
                    ),
                    "active_regret_cost_vs_fixed": active - fixed,
                    "active_best_seen_cost_vs_fixed": (
                        _aggregate_value(
                            aggregate,
                            quality,
                            noise,
                            ACTIVE_DIAGNOSTIC_METHOD,
                            "best_seen_regret_mean",
                        )
                        - _aggregate_value(
                            aggregate,
                            quality,
                            noise,
                            FIXED_METHOD,
                            "best_seen_regret_mean",
                        )
                    ),
                }
            )
    return output


def _negative_transfer(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            standard = _aggregate_value(
                aggregate, quality, noise, STANDARD_METHOD, "final_regret_mean"
            )
            row: dict[str, Any] = {
                "prior_quality": quality,
                "noise_label": noise,
                "standard_final_regret_mean": standard,
            }
            for method, key in (
                (FIXED_METHOD, "fixed"),
                (ADAPTIVE_TRUST_METHOD, "adaptive"),
                (PREDICTIVE_FAILOVER_METHOD, "passive_failover"),
                (ACTIVE_DIAGNOSTIC_METHOD, "active_diagnostic"),
            ):
                mean = _aggregate_value(
                    aggregate, quality, noise, method, "final_regret_mean"
                )
                row[f"{key}_final_regret_mean"] = mean
                row[f"{key}_negative_transfer_mean"] = mean - standard
            row["active_improvement_vs_fixed"] = (
                row["fixed_negative_transfer_mean"]
                - row["active_diagnostic_negative_transfer_mean"]
            )
            row["active_improvement_vs_adaptive"] = (
                row["adaptive_negative_transfer_mean"]
                - row["active_diagnostic_negative_transfer_mean"]
            )
            row["active_improvement_vs_passive"] = (
                row["passive_failover_negative_transfer_mean"]
                - row["active_diagnostic_negative_transfer_mean"]
            )
            output.append(row)
    return output


def _diagnostic_candidate_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["canonical_beta_id"],
            row["beta_flex"],
            row["beta_extend"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (candidate_id, beta_flex, beta_extend), items in sorted(
        grouped.items(), key=lambda pair: (-len(pair[1]), pair[0][0])
    ):
        scores = np.asarray([item["diagnostic_divergence"] for item in items])
        output.append(
            {
                "canonical_beta_id": candidate_id,
                "beta_flex": beta_flex,
                "beta_extend": beta_extend,
                "selection_count": len(items),
                "selection_fraction": len(items) / len(rows),
                "diagnostic_divergence_mean": float(np.mean(scores)),
                "diagnostic_divergence_min": float(np.min(scores)),
                "diagnostic_divergence_max": float(np.max(scores)),
            }
        )
    return output


def _p3_analysis(
    aggregate: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    arbitration: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    passive_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for noise in NOISE_SETTINGS:
        arbitration_row = next(
            row
            for row in arbitration
            if row["prior_quality"] == "P3" and row["noise_label"] == noise
        )
        negative_row = next(
            row
            for row in negative
            if row["prior_quality"] == "P3" and row["noise_label"] == noise
        )
        passive_row = next(
            row
            for row in passive_summary["failover_summary"]
            if row["prior_quality"] == "P3" and row["noise_label"] == noise
        )
        comparisons = [
            row
            for row in pairwise
            if row["prior_quality"] == "P3" and row["noise_label"] == noise
        ]
        output.append(
            {
                "noise_label": noise,
                "mean_final_regret": {
                    method: _aggregate_value(
                        aggregate, "P3", noise, method, "final_regret_mean"
                    )
                    for method in METHODS
                },
                "paired_win_tie_loss": {
                    row["comparison"].split(" vs ", maxsplit=1)[1]: {
                        "wins": row["wins"],
                        "ties": row["ties"],
                        "losses": row["losses"],
                    }
                    for row in comparisons
                },
                "standard_selection_count": arbitration_row[
                    "standard_selection_count"
                ],
                "misleading_prior_rejection_rate": arbitration_row[
                    "standard_selection_rate"
                ],
                "active_missed_rejection_count": arbitration_row[
                    "failure_to_reject_misleading_physics_count"
                ],
                "passive_missed_failover_count": passive_row[
                    "missed_failover_count"
                ],
                "missed_rejection_reduction_vs_passive": (
                    passive_row["missed_failover_count"]
                    - arbitration_row["failure_to_reject_misleading_physics_count"]
                ),
                "negative_transfer_relative_to_standard": {
                    FIXED_METHOD: negative_row["fixed_negative_transfer_mean"],
                    ADAPTIVE_TRUST_METHOD: negative_row[
                        "adaptive_negative_transfer_mean"
                    ],
                    PREDICTIVE_FAILOVER_METHOD: negative_row[
                        "passive_failover_negative_transfer_mean"
                    ],
                    ACTIVE_DIAGNOSTIC_METHOD: negative_row[
                        "active_diagnostic_negative_transfer_mean"
                    ],
                },
                "active_improvement_vs_fixed": negative_row[
                    "active_improvement_vs_fixed"
                ],
                "active_improvement_vs_adaptive": negative_row[
                    "active_improvement_vs_adaptive"
                ],
                "active_improvement_vs_passive": negative_row[
                    "active_improvement_vs_passive"
                ],
            }
        )
    return output


def _readiness_assessment(
    benefit: list[dict[str, Any]],
    arbitration: list[dict[str, Any]],
    p3: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [
        row
        for row in benefit
        if row["fixed_physics_benefit_vs_standard"] > PAIR_TOLERANCE
    ]
    fixed_benefit = sum(row["fixed_physics_benefit_vs_standard"] for row in positive)
    active_benefit = sum(row["active_benefit_vs_standard"] for row in positive)
    retained_fraction = active_benefit / fixed_benefit if fixed_benefit else math.nan
    informative_false_rejection_rates = [
        row["standard_selection_rate"]
        for row in arbitration
        if row["prior_quality"] in {"P0", "P1", "P2"}
    ]
    benefit_retained = retained_fraction >= 0.75
    false_rejection_controlled = max(informative_false_rejection_rates) <= 0.25
    p3_stable_improvement = all(
        row["active_improvement_vs_fixed"] > PAIR_TOLERANCE
        and row["active_improvement_vs_adaptive"] > PAIR_TOLERANCE
        and row["active_improvement_vs_passive"] > PAIR_TOLERANCE
        for row in p3
    )
    missed_rejection_reduced = all(
        row["missed_rejection_reduction_vs_passive"] > 0 for row in p3
    )
    p3_no_unresolved_catastrophic_transfer = all(
        row["negative_transfer_relative_to_standard"][ACTIVE_DIAGNOSTIC_METHOD]
        <= max(
            row["negative_transfer_relative_to_standard"][ADAPTIVE_TRUST_METHOD],
            row["negative_transfer_relative_to_standard"][
                PREDICTIVE_FAILOVER_METHOD
            ],
        )
        for row in p3
    )
    ready = all(
        (
            benefit_retained,
            false_rejection_controlled,
            p3_stable_improvement,
            missed_rejection_reduced,
            p3_no_unresolved_catastrophic_transfer,
        )
    )
    return {
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
        "informative_positive_benefit_retention_fraction": retained_fraction,
        "benefit_largely_retained_at_75_percent": benefit_retained,
        "maximum_informative_false_rejection_rate": max(
            informative_false_rejection_rates
        ),
        "informative_false_rejection_rate_at_most_25_percent": (
            false_rejection_controlled
        ),
        "p3_improves_over_fixed_adaptive_and_passive_at_all_noise_levels": (
            p3_stable_improvement
        ),
        "p3_missed_rejection_reduced_vs_passive_at_all_noise_levels": (
            missed_rejection_reduced
        ),
        "p3_no_unresolved_catastrophic_negative_transfer": (
            p3_no_unresolved_catastrophic_transfer
        ),
        "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED": (
            not ready
        ),
    }


def run_benchmark(seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    verify_frozen_rule()
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != DEFAULT_SEEDS:
        raise ValueError("formal active-diagnostic seeds are frozen at 0...4")
    passive_summary = json.loads(PASSIVE_BASELINE_SUMMARY.read_text(encoding="utf-8"))
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
                        if method == STANDARD_METHOD:
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(environment, method="Standard BO")
                        elif method == FIXED_METHOD:
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(
                                environment,
                                method="Physics-Informed BO",
                                physics_model=_physics(case, quality),
                            )
                        elif method == ADAPTIVE_TRUST_METHOD:
                            result = ROMGatedAdaptiveTrustEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(
                                environment,
                                physics_model=_physics(case, quality),
                            )
                        elif method == PREDICTIVE_FAILOVER_METHOD:
                            result = ROMGatedPredictiveFailoverEpisode(
                                episode_id=episode_id, profile=profile, domain=domain
                            ).run(
                                environment,
                                physics_model=_physics(case, quality),
                            )
                        else:
                            result = ROMGatedActiveDiagnosticEpisode(
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
                                "diagnostic_candidate_id": None,
                                "diagnostic_canonical_beta_id": None,
                                "diagnostic_beta": None,
                                "diagnostic_divergence": None,
                                "selected_expert": None,
                                "arbitration_decision": None,
                                "arbitration_score_difference": None,
                            }
                        )
                        if method == ACTIVE_DIAGNOSTIC_METHOD:
                            active = result.sequential_result
                            candidate = active.diagnostic_selection.candidate
                            canonical = domain.by_id(candidate.candidate_id)
                            metrics.update(
                                {
                                    "diagnostic_candidate_id": candidate.candidate_id,
                                    "diagnostic_canonical_beta_id": (
                                        canonical.canonical_beta_id
                                    ),
                                    "diagnostic_beta": list(candidate.beta),
                                    "diagnostic_divergence": (
                                        active.diagnostic_selection.diagnostic_score
                                    ),
                                    "selected_expert": active.selected_expert,
                                    "arbitration_decision": active.arbitration.decision,
                                    "arbitration_score_difference": (
                                        active.arbitration.score_difference
                                    ),
                                }
                            )
                            diagnostic_rows.append(
                                {
                                    "case": case.name,
                                    "seed": seed,
                                    "prior_quality": quality,
                                    "noise_label": noise_label,
                                    "candidate_id": candidate.candidate_id,
                                    "canonical_beta_id": canonical.canonical_beta_id,
                                    "beta_flex": candidate.beta_flex,
                                    "beta_extend": candidate.beta_extend,
                                    "diagnostic_divergence": (
                                        active.diagnostic_selection.diagnostic_score
                                    ),
                                    "standard_mean_before": (
                                        active.diagnostic_selection.predictions.standard_bo_mean_before
                                    ),
                                    "standard_std_before": (
                                        active.diagnostic_selection.predictions.standard_bo_std_before
                                    ),
                                    "physics_mean_before": (
                                        active.diagnostic_selection.predictions.physics_bo_mean_before
                                    ),
                                    "physics_std_before": (
                                        active.diagnostic_selection.predictions.physics_bo_std_before
                                    ),
                                    "physics_offset": (
                                        active.diagnostic_selection.physics_offset
                                    ),
                                    "adjusted_physics_mean": (
                                        active.diagnostic_selection.adjusted_physics_mean
                                    ),
                                    "observed_y": (
                                        active.ledger.entries[1].observation.endpoint_value
                                    ),
                                    "standard_log_score": (
                                        active.arbitration.standard_log_score
                                    ),
                                    "physics_log_score": active.arbitration.physics_log_score,
                                    "score_difference": active.arbitration.score_difference,
                                    "selected_expert": active.selected_expert,
                                    "arbitration_decision": active.arbitration.decision,
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
    historical = _historical_equivalence(aggregate)
    if not historical["exact_within_1e_12"]:
        raise RuntimeError("frozen historical method behavior drifted")
    pairwise = _pairwise(rows)
    arbitration = _arbitration_summary(rows)
    benefit = _benefit_retention(aggregate)
    negative = _negative_transfer(aggregate)
    diagnostic_candidates = _diagnostic_candidate_summary(diagnostic_rows)
    p3 = _p3_analysis(
        aggregate, pairwise, arbitration, negative, passive_summary
    )
    readiness = _readiness_assessment(benefit, arbitration, p3)
    ready = readiness["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"]
    return {
        "status": "ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1_BENCHMARK_COMPLETED",
        "implementation_status": (
            "ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1_IMPLEMENTED"
            if ready
            else "ACTIVE_PRIOR_DIAGNOSTIC_ARBITRATION_V1_IMPLEMENTED_WITH_LIMITATIONS"
        ),
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
        "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED": (
            readiness[
                "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED"
            ]
        ),
        "classification": "OFFLINE_ALGORITHM_DEVELOPMENT_EVIDENCE_ONLY",
        "primary_budget": PRIMARY_BUDGET,
        "seeds": list(seed_values),
        "prior_qualities": list(PRIOR_QUALITIES),
        "noise_settings": NOISE_SETTINGS,
        "truth_case_count_per_noise": len(make_primary_cases()),
        "candidate_count": len(domain),
        "rom_calibration_observation_count": len(rom_controller.ledger.entries),
        "rom_calibration_excluded_from_K": True,
        "rom_profile": profile.as_dict(),
        "same_rom_fingerprint_for_all_paired_methods": True,
        "methods": list(METHODS),
        "run_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "pairwise": pairwise,
        "arbitration_summary": arbitration,
        "benefit_retention": benefit,
        "negative_transfer": negative,
        "p3_analysis": p3,
        "diagnostic_selection_rows": diagnostic_rows,
        "diagnostic_candidate_summary": diagnostic_candidates,
        "historical_baseline_equivalence": historical,
        "readiness_assessment": readiness,
        "causal_audit": {
            "trial_1_candidate": "reference_beta_[0,0]",
            "trial_2_selection_history": "D1_only",
            "trial_2_predictions_frozen_before_observation": True,
            "arbitration_inputs": "stored_trial_2_predictions_and_revealed_valid_y2",
            "case_label_available_to_algorithm": False,
            "future_truth_available_to_algorithm": False,
            "selected_expert_fixed_for_trials_3_and_4": True,
        },
        "robot_actions": 0,
        "human_actions": 0,
        "PINN_training": 0,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "run_metrics.csv", payload["rows"])
    _write_csv(output_dir / "aggregate_metrics.csv", payload["aggregate"])
    _write_csv(output_dir / "paired_comparisons.csv", payload["pairwise"])
    _write_csv(output_dir / "arbitration_summary.csv", payload["arbitration_summary"])
    _write_csv(output_dir / "benefit_retention.csv", payload["benefit_retention"])
    _write_csv(output_dir / "negative_transfer_summary.csv", payload["negative_transfer"])
    _write_csv(output_dir / "p3_analysis.csv", payload["p3_analysis"])
    _write_csv(
        output_dir / "diagnostic_selections.csv",
        payload["diagnostic_selection_rows"],
    )
    _write_csv(
        output_dir / "diagnostic_candidate_summary.csv",
        payload["diagnostic_candidate_summary"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_benchmark()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                "status": payload["implementation_status"],
                "run_count": payload["run_count"],
                "historical_baselines_exact": payload[
                    "historical_baseline_equivalence"
                ]["exact_within_1e_12"],
                "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED": payload[
                    "K4_PRIOR_IDENTIFICATION_AND_OPTIMIZATION_MAY_BE_INFORMATION_LIMITED"
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
