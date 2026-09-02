"""Post-run metrics with explicitly isolated oracle access."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..environment import (
    OFFLINE_ALGORITHM_DEVELOPMENT_EVALUATION,
    AnalyticBenchmarkEnvironment,
)
from ..sequential import SequentialRunResult


def evaluate_run(
    result: SequentialRunResult,
    evaluator: AnalyticBenchmarkEnvironment,
) -> dict[str, Any]:
    optimum, optimum_value = evaluator.oracle_optimum()
    executed = [entry.candidate for entry in result.ledger.entries]
    truth_values = [evaluator.oracle_value(candidate) for candidate in executed]
    simple_regret = []
    best_so_far = math.inf
    for value in truth_values:
        best_so_far = min(best_so_far, value)
        simple_regret.append(best_so_far - optimum_value)
    final = result.model_recommended_final_candidate
    if final is None:
        final_value = math.nan
        final_regret = math.nan
        distance = math.nan
    else:
        final_value = evaluator.oracle_value(final)
        final_regret = final_value - optimum_value
        distance = math.hypot(
            final.beta_flex - optimum.beta_flex,
            final.beta_extend - optimum.beta_extend,
        )
    unique_count = len({candidate.candidate_id for candidate in executed})
    return {
        "classification": OFFLINE_ALGORITHM_DEVELOPMENT_EVALUATION,
        "method": result.method,
        "budget": result.budget,
        "final_candidate_id": final.candidate_id if final else None,
        "final_value": final_value,
        "final_regret": final_regret,
        "best_seen_regret": min(truth_values) - optimum_value,
        "simple_regret_per_trial": simple_regret,
        "distance_to_optimum_beta": distance,
        "invalid_observation_count": sum(
            not entry.observation.valid for entry in result.ledger.entries
        ),
        "unique_candidate_count": unique_count,
        "candidate_diversity_fraction": unique_count / result.budget,
        "duplicate_count": result.budget - unique_count,
        "selected_candidate_ids": [item.candidate_id for item in executed],
        "selected_beta_sequence": [list(item.beta) for item in executed],
        "oracle_candidate_id": optimum.candidate_id,
        "oracle_value": optimum_value,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("method", "budget", "prior_quality", "noise_label")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group, items in sorted(groups.items()):
        final = np.asarray([item["final_regret"] for item in items], dtype=float)
        best = np.asarray([item["best_seen_regret"] for item in items], dtype=float)
        result.append(
            {
                **dict(zip(keys, group)),
                "run_count": len(items),
                "final_regret_mean": float(np.mean(final)),
                "final_regret_median": float(np.median(final)),
                "final_regret_p95": float(np.percentile(final, 95)),
                "best_seen_regret_mean": float(np.mean(best)),
                "invalid_mean": float(
                    np.mean([item["invalid_observation_count"] for item in items])
                ),
            }
        )
    return result


def physics_bo_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = ("case", "seed", "budget", "prior_quality", "noise_label")
    grouped: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in identity), {})[row["method"]] = row[
            "final_regret"
        ]
    baselines = sorted({row["method"] for row in rows} - {"Physics-Informed BO"})
    output = []
    for baseline in baselines:
        wins = ties = losses = 0
        for methods in grouped.values():
            if baseline not in methods or "Physics-Informed BO" not in methods:
                continue
            delta = methods["Physics-Informed BO"] - methods[baseline]
            if abs(delta) <= 1.0e-12:
                ties += 1
            elif delta < 0.0:
                wins += 1
            else:
                losses += 1
        output.append(
            {
                "comparison": f"Physics-Informed BO vs {baseline}",
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        )
    return output
