"""Formal paired offline benchmark for predictive-evidence failover V1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from personalization.adaptive_trust_v1 import (
    ADAPTIVE_TRUST_METHOD,
    ROMGatedAdaptiveTrustEpisode,
)
from personalization.benchmarks.metrics import evaluate_run
from personalization.benchmarks.run_adaptive_trust_v1 import _environment, _physics
from personalization.benchmarks.run_equal_budget import NOISE_SETTINGS, PRIOR_QUALITIES
from personalization.environment import make_primary_cases
from personalization.predictive_failover_v1 import (
    FAILOVER_THRESHOLD,
    PREDICTIVE_FAILOVER_METHOD,
    ROMGatedPredictiveFailoverEpisode,
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
    ROOT / "personalization" / "benchmarks" / "results_predictive_failover_v1"
)
ADAPTIVE_BASELINE_SUMMARY = (
    ROOT
    / "personalization"
    / "benchmarks"
    / "results_adaptive_trust_v1"
    / "benchmark_summary.json"
)
STANDARD_METHOD = "Standard BO"
FIXED_METHOD = "Fixed Physics BO"
METHODS = (
    STANDARD_METHOD,
    FIXED_METHOD,
    ADAPTIVE_TRUST_METHOD,
    PREDICTIVE_FAILOVER_METHOD,
)
PRIMARY_BUDGET = 4
DEFAULT_SEEDS = tuple(range(5))
PAIR_TOLERANCE = 1.0e-12


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    encoded = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(encoded[0]))
        writer.writeheader()
        writer.writerows(encoded)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["prior_quality"], row["noise_label"])
        groups.setdefault(key, []).append(row)
    output = []
    for (method, quality, noise), items in sorted(groups.items()):
        final = np.asarray([item["final_regret"] for item in items], dtype=float)
        best = np.asarray([item["best_seen_regret"] for item in items], dtype=float)
        output.append(
            {
                "method": method,
                "prior_quality": quality,
                "noise_label": noise,
                "run_count": len(items),
                "final_regret_mean": float(np.mean(final)),
                "final_regret_median": float(np.median(final)),
                "final_regret_std": float(np.std(final, ddof=1)),
                "final_regret_p2_5": float(np.percentile(final, 2.5)),
                "final_regret_p95": float(np.percentile(final, 95)),
                "final_regret_p97_5": float(np.percentile(final, 97.5)),
                "best_seen_regret_mean": float(np.mean(best)),
            }
        )
    return output


def _group_runs(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    identity_fields = ("case", "seed", "prior_quality", "noise_label")
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(row[field] for field in identity_fields)
        grouped.setdefault(identity, {})[row["method"]] = row
    return grouped


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
            for baseline in (STANDARD_METHOD, FIXED_METHOD, ADAPTIVE_TRUST_METHOD):
                deltas = np.asarray(
                    [
                        methods[PREDICTIVE_FAILOVER_METHOD]["final_regret"]
                        - methods[baseline]["final_regret"]
                        for methods in conditions
                    ],
                    dtype=float,
                )
                output.append(
                    {
                        "comparison": f"{PREDICTIVE_FAILOVER_METHOD} vs {baseline}",
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


def _method_mean(
    aggregate: list[dict[str, Any]], quality: str, noise: str, method: str
) -> float:
    return float(
        next(
            row["final_regret_mean"]
            for row in aggregate
            if row["prior_quality"] == quality
            and row["noise_label"] == noise
            and row["method"] == method
        )
    )


def _negative_transfer(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            standard = _method_mean(aggregate, quality, noise, STANDARD_METHOD)
            row: dict[str, Any] = {
                "prior_quality": quality,
                "noise_label": noise,
                "standard_final_regret_mean": standard,
            }
            for method in (FIXED_METHOD, ADAPTIVE_TRUST_METHOD, PREDICTIVE_FAILOVER_METHOD):
                mean = _method_mean(aggregate, quality, noise, method)
                key = (
                    "fixed"
                    if method == FIXED_METHOD
                    else "adaptive"
                    if method == ADAPTIVE_TRUST_METHOD
                    else "failover"
                )
                row[f"{key}_final_regret_mean"] = mean
                row[f"{key}_negative_transfer_mean"] = mean - standard
            row["failover_improvement_vs_adaptive"] = (
                row["adaptive_negative_transfer_mean"]
                - row["failover_negative_transfer_mean"]
            )
            row["failover_improvement_vs_fixed"] = (
                row["fixed_negative_transfer_mean"]
                - row["failover_negative_transfer_mean"]
            )
            output.append(row)
    return output


def _benefit_retention(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for quality in ("P0", "P1", "P2"):
        for noise in NOISE_SETTINGS:
            standard = _method_mean(aggregate, quality, noise, STANDARD_METHOD)
            fixed = _method_mean(aggregate, quality, noise, FIXED_METHOD)
            failover = _method_mean(
                aggregate, quality, noise, PREDICTIVE_FAILOVER_METHOD
            )
            fixed_benefit = standard - fixed
            failover_benefit = standard - failover
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "standard_final_regret_mean": standard,
                    "fixed_final_regret_mean": fixed,
                    "failover_final_regret_mean": failover,
                    "fixed_physics_benefit_vs_standard": fixed_benefit,
                    "failover_benefit_vs_standard": failover_benefit,
                    "benefit_retention_fraction": (
                        failover_benefit / fixed_benefit
                        if fixed_benefit > PAIR_TOLERANCE
                        else None
                    ),
                    "failover_regret_difference_vs_fixed": failover - fixed,
                }
            )
    return output


def _failover_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    failover_rows = [
        row for row in rows if row["method"] == PREDICTIVE_FAILOVER_METHOD
    ]
    for quality in PRIOR_QUALITIES:
        for noise in NOISE_SETTINGS:
            selected = [
                row
                for row in failover_rows
                if row["prior_quality"] == quality and row["noise_label"] == noise
            ]
            trials = [int(row["failover_trial"]) for row in selected if row["failed_over"]]
            informative_label = quality in {"P0", "P1", "P2"}
            misleading_label = quality == "P3"
            output.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise,
                    "episode_count": len(selected),
                    "failover_count": len(trials),
                    "failover_rate": len(trials) / len(selected),
                    "median_failover_trial": (
                        float(np.median(trials)) if trials else None
                    ),
                    "failover_trial_counts": {
                        str(trial): trials.count(trial) for trial in range(2, 5)
                    },
                    "false_failover_evaluator_label": informative_label,
                    "false_failover_count": len(trials) if informative_label else None,
                    "missed_failover_evaluator_label": misleading_label,
                    "missed_failover_count": (
                        len(selected) - len(trials) if misleading_label else None
                    ),
                }
            )
    return output


def _historical_equivalence(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    historical = json.loads(ADAPTIVE_BASELINE_SUMMARY.read_text(encoding="utf-8"))
    old = {
        (row["method"], row["prior_quality"], row["noise_label"]): row
        for row in historical["aggregate"]
    }
    mapping = {
        STANDARD_METHOD: STANDARD_METHOD,
        FIXED_METHOD: "Fixed-Trust Physics BO",
        ADAPTIVE_TRUST_METHOD: ADAPTIVE_TRUST_METHOD,
    }
    maximum_error = 0.0
    compared = 0
    for row in aggregate:
        if row["method"] not in mapping:
            continue
        parent = old[
            (
                mapping[row["method"]],
                row["prior_quality"],
                row["noise_label"],
            )
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
        "historical_summary_path": str(ADAPTIVE_BASELINE_SUMMARY.relative_to(ROOT)),
        "compared_metric_count": compared,
        "maximum_absolute_error": maximum_error,
        "exact_within_1e_12": maximum_error <= PAIR_TOLERANCE,
    }


def _p3_analysis(
    aggregate: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    failover: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for noise in NOISE_SETTINGS:
        fail = next(
            row
            for row in failover
            if row["prior_quality"] == "P3" and row["noise_label"] == noise
        )
        transfer = next(
            row
            for row in negative
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
                    method: _method_mean(aggregate, "P3", noise, method)
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
                "failover_count": fail["failover_count"],
                "failover_rate": fail["failover_rate"],
                "median_failover_trial": fail["median_failover_trial"],
                "missed_failover_count": fail["missed_failover_count"],
                "negative_transfer_relative_to_standard": {
                    "Fixed Physics BO": transfer["fixed_negative_transfer_mean"],
                    ADAPTIVE_TRUST_METHOD: transfer[
                        "adaptive_negative_transfer_mean"
                    ],
                    PREDICTIVE_FAILOVER_METHOD: transfer[
                        "failover_negative_transfer_mean"
                    ],
                },
                "failover_improvement_vs_adaptive": transfer[
                    "failover_improvement_vs_adaptive"
                ],
                "failover_improvement_vs_fixed": transfer[
                    "failover_improvement_vs_fixed"
                ],
            }
        )
    return output


def _readiness_assessment(
    benefit: list[dict[str, Any]],
    failover: list[dict[str, Any]],
    p3: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_benefit_rows = [
        row
        for row in benefit
        if row["fixed_physics_benefit_vs_standard"] > PAIR_TOLERANCE
    ]
    aggregate_fixed_benefit = sum(
        row["fixed_physics_benefit_vs_standard"] for row in positive_benefit_rows
    )
    aggregate_failover_benefit = sum(
        row["failover_benefit_vs_standard"] for row in positive_benefit_rows
    )
    retained_fraction = (
        aggregate_failover_benefit / aggregate_fixed_benefit
        if aggregate_fixed_benefit > 0.0
        else math.nan
    )
    informative_false_rates = [
        row["failover_rate"]
        for row in failover
        if row["prior_quality"] in {"P0", "P1", "P2"}
    ]
    p3_improves_both = all(
        row["failover_improvement_vs_adaptive"] > PAIR_TOLERANCE
        and row["failover_improvement_vs_fixed"] > PAIR_TOLERANCE
        for row in p3
    )
    p3_failover_occurs = all(row["failover_count"] > 0 for row in p3)
    moderate = next(row for row in p3 if row["noise_label"] == "moderate")
    moderate_not_newly_worse = moderate["failover_improvement_vs_adaptive"] >= 0.0
    benefit_largely_retained = retained_fraction >= 0.75
    stable_false_failover = max(informative_false_rates) <= 0.25
    ready = all(
        (
            benefit_largely_retained,
            stable_false_failover,
            p3_improves_both,
            p3_failover_occurs,
            moderate_not_newly_worse,
        )
    )
    return {
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
        "informative_positive_benefit_retention_fraction": retained_fraction,
        "benefit_largely_retained_at_75_percent": benefit_largely_retained,
        "maximum_informative_failover_rate": max(informative_false_rates),
        "informative_failover_rate_at_most_25_percent": stable_false_failover,
        "p3_improves_over_fixed_and_adaptive_at_all_noise_levels": p3_improves_both,
        "p3_failover_occurs_at_all_noise_levels": p3_failover_occurs,
        "p3_moderate_not_worse_than_adaptive": moderate_not_newly_worse,
    }


def run_benchmark(seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    verify_frozen_rule()
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != DEFAULT_SEEDS:
        raise ValueError("formal predictive-failover seeds are frozen at 0...4")
    rom_controller, profile = determine_synthetic_rom(
        make_offline_rom_development_cases()[1]
    )
    domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
    rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

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
                        else:
                            result = ROMGatedPredictiveFailoverEpisode(
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
                                "failed_over": None,
                                "failover_trial": None,
                                "cumulative_evidence_final": None,
                                "cumulative_evidence_trajectory": None,
                            }
                        )
                        if method == PREDICTIVE_FAILOVER_METHOD:
                            failover_result = result.sequential_result
                            metrics["failed_over"] = failover_result.failed_over
                            metrics["failover_trial"] = failover_result.failover_trial
                            metrics["cumulative_evidence_final"] = (
                                failover_result.evidence_states[-1].cumulative_evidence
                            )
                            metrics["cumulative_evidence_trajectory"] = [
                                state.cumulative_evidence
                                for state in failover_result.evidence_states
                            ]
                            for entry in failover_result.ledger.entries:
                                evidence_rows.append(
                                    {
                                        "case": case.name,
                                        "seed": seed,
                                        "prior_quality": quality,
                                        "noise_label": noise_label,
                                        "trial": entry.trial_index,
                                        "candidate_id": entry.candidate.candidate_id,
                                        "beta_flex": entry.candidate.beta_flex,
                                        "beta_extend": entry.candidate.beta_extend,
                                        "standard_bo_mean_before": entry.standard_bo_mean_before,
                                        "standard_bo_std_before": entry.standard_bo_std_before,
                                        "physics_bo_mean_before": entry.physics_bo_mean_before,
                                        "physics_bo_std_before": entry.physics_bo_std_before,
                                        "physics_offset_before": entry.physics_offset_before,
                                        "physics_scoring_mean": entry.physics_scoring_mean,
                                        "observed_y": entry.observation.endpoint_value,
                                        "observation_valid": entry.observation.valid,
                                        "standard_log_score": entry.standard_log_score,
                                        "physics_log_score": entry.physics_log_score,
                                        "score_difference": entry.score_difference,
                                        "cumulative_evidence": entry.cumulative_evidence_after,
                                        "mode_after": entry.mode_after_observation,
                                        "failover_triggered": entry.failover_triggered_this_trial,
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
        raise RuntimeError("frozen Standard/Fixed/Adaptive baseline behavior drifted")
    pairwise = _pairwise(rows)
    negative = _negative_transfer(aggregate)
    benefit = _benefit_retention(aggregate)
    failover = _failover_summary(rows)
    p3 = _p3_analysis(aggregate, pairwise, failover, negative)
    readiness = _readiness_assessment(benefit, failover, p3)
    ready = readiness["READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK"]
    return {
        "status": "PREDICTIVE_EVIDENCE_PHYSICS_PRIOR_FAILOVER_V1_BENCHMARK_COMPLETED",
        "implementation_status": (
            "PREDICTIVE_EVIDENCE_PHYSICS_PRIOR_FAILOVER_V1_IMPLEMENTED"
            if ready
            else "PREDICTIVE_EVIDENCE_PHYSICS_PRIOR_FAILOVER_V1_IMPLEMENTED_WITH_LIMITATIONS"
        ),
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": ready,
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
        "failover_threshold": FAILOVER_THRESHOLD,
        "run_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "pairwise": pairwise,
        "negative_transfer": negative,
        "benefit_retention": benefit,
        "failover_summary": failover,
        "p3_analysis": p3,
        "evidence_trajectory_rows": evidence_rows,
        "historical_baseline_equivalence": historical,
        "readiness_assessment": readiness,
        "causal_audit": {
            "prediction_history_size_at_trial_k": "k-1",
            "score_start_trial": 2,
            "offset_inputs": "valid observations and stored physics forecasts from i<k",
            "case_label_available_to_arbitrator": False,
            "future_truth_available_to_arbitrator": False,
            "continuous_mixture_used_by_primary_method": False,
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
    _write_csv(output_dir / "negative_transfer_summary.csv", payload["negative_transfer"])
    _write_csv(output_dir / "benefit_retention.csv", payload["benefit_retention"])
    _write_csv(output_dir / "failover_summary.csv", payload["failover_summary"])
    _write_csv(output_dir / "p3_analysis.csv", payload["p3_analysis"])
    _write_csv(
        output_dir / "predictive_evidence_trajectories.csv",
        payload["evidence_trajectory_rows"],
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
