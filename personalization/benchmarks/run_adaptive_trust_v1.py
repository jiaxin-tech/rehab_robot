"""Paired offline benchmark for the frozen adaptive physics-trust rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from personalization.adaptive_trust_v1 import (
    ADAPTIVE_TRUST_METHOD,
    EXPECTED_RULE_SHA256,
    PhysicsPriorTrustEstimator,
    PhysicsPriorTrustEvidence,
    ROMGatedAdaptiveTrustEpisode,
    verify_frozen_rule,
)
from personalization.environment import (
    OFFLINE_ALGORITHM_TEST_CASE,
    AnalyticBenchmarkEnvironment,
    _stable_normal,
    make_primary_cases,
)
from personalization.models.physics_graybox import (
    AnalyticDevelopmentPhysicsAdapter,
    PhysicsSubjectModel,
)
from personalization.observations import EpisodeObservation
from personalization.rom_gated_v2 import (
    ROMGatedPersonalizationEpisode,
    SubjectSpecificCandidate,
    SubjectSpecificV3CandidateDomain,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
)

from .metrics import evaluate_run
from .run_equal_budget import NOISE_SETTINGS, PRIOR_QUALITIES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "personalization" / "benchmarks" / "results_adaptive_trust_v1"
HISTORICAL_V1_SUMMARY = (
    ROOT / "personalization" / "benchmarks" / "results_v1" / "benchmark_summary.json"
)
METHODS = ("Standard BO", "Fixed-Trust Physics BO", ADAPTIVE_TRUST_METHOD)
PRIMARY_BUDGET = 4
DEFAULT_SEEDS = tuple(range(5))
PAIR_TOLERANCE = 1.0e-12


class CanonicalNoiseAnalyticBenchmarkEnvironment(AnalyticBenchmarkEnvironment):
    """Existing frozen truth with noise keyed by canonical V3 identity.

    Subject-specific candidate IDs must differ across ROMs.  Keying the already
    frozen deterministic noise rule by ``canonical_beta_id`` preserves the V1
    P0-P3 noise realization while retaining the V2 subject candidate identity.
    """

    def evaluate(
        self, candidate: SubjectSpecificCandidate, trial_index: int
    ) -> EpisodeObservation:
        canonical_id = candidate.canonical_beta_id
        metadata = {
            "classification": OFFLINE_ALGORITHM_TEST_CASE,
            "case": self.case.name,
            "seed": self.seed,
            "truth_hidden_from_selector": True,
            "canonical_noise_candidate_id": canonical_id,
            "subject_candidate_id": candidate.candidate_id,
        }
        if canonical_id in self.case.invalid_candidate_ids:
            return EpisodeObservation(
                episode_id=f"{self.case.name}:{self.seed}:{trial_index}",
                trial_index=trial_index,
                candidate_id=candidate.candidate_id,
                beta_flex=candidate.beta_flex,
                beta_extend=candidate.beta_extend,
                endpoint_name=self.endpoint_name,
                endpoint_value=None,
                endpoint_unit=self.endpoint_unit,
                endpoint_uncertainty=self.case.noise_std,
                valid=False,
                invalid_reason="INJECTED_INVALID_OFFLINE_EPISODE",
                metadata=metadata,
            )
        noise = self.case.noise_std * _stable_normal(
            self.seed, canonical_id, trial_index
        )
        outlier = (
            self.case.outlier_magnitude
            if canonical_id in self.case.outlier_candidate_ids
            else 0.0
        )
        return EpisodeObservation(
            episode_id=f"{self.case.name}:{self.seed}:{trial_index}",
            trial_index=trial_index,
            candidate_id=candidate.candidate_id,
            beta_flex=candidate.beta_flex,
            beta_extend=candidate.beta_extend,
            endpoint_name=self.endpoint_name,
            endpoint_value=self._truth(candidate) + noise + outlier,
            endpoint_unit=self.endpoint_unit,
            endpoint_uncertainty=self.case.noise_std,
            valid=True,
            metadata={**metadata, "outlier_injected": bool(outlier)},
        )


def _physics(case, quality: str) -> PhysicsSubjectModel:
    return PhysicsSubjectModel(
        AnalyticDevelopmentPhysicsAdapter(
            optimum_beta=case.optimum_beta,
            prior_quality=quality,
            landscape=case.landscape,
        )
    )


def _environment(domain, case, seed: int):
    return CanonicalNoiseAnalyticBenchmarkEnvironment(domain, case, seed=seed)


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


def _extended_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["prior_quality"], row["noise_label"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (method, quality, noise), items in sorted(grouped.items()):
        final = np.asarray([item["final_regret"] for item in items], dtype=float)
        best = np.asarray([item["best_seen_regret"] for item in items], dtype=float)
        trust = np.asarray(
            [item["final_trust"] for item in items if item["final_trust"] is not None],
            dtype=float,
        )
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
                "final_trust_mean": float(np.mean(trust)) if len(trust) else None,
                "final_trust_median": float(np.median(trust)) if len(trust) else None,
                "fallback_dominant_selection_frequency_mean": (
                    float(
                        np.mean(
                            [
                                item["fallback_dominant_selection_frequency"]
                                for item in items
                                if item["fallback_dominant_selection_frequency"]
                                is not None
                            ]
                        )
                    )
                    if len(trust)
                    else None
                ),
            }
        )
    return output


def _paired_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identity_fields = ("case", "seed", "prior_quality", "noise_label")
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(row[field] for field in identity_fields)
        grouped.setdefault(identity, {})[row["method"]] = row
    pairwise = []
    negative = []
    worse_than_both = []
    for quality in PRIOR_QUALITIES:
        for noise_label in NOISE_SETTINGS:
            selected = [
                methods
                for identity, methods in grouped.items()
                if identity[2] == quality and identity[3] == noise_label
            ]
            for baseline in ("Standard BO", "Fixed-Trust Physics BO"):
                deltas = np.asarray(
                    [
                        methods[ADAPTIVE_TRUST_METHOD]["final_regret"]
                        - methods[baseline]["final_regret"]
                        for methods in selected
                    ],
                    dtype=float,
                )
                pairwise.append(
                    {
                        "comparison": f"{ADAPTIVE_TRUST_METHOD} vs {baseline}",
                        "prior_quality": quality,
                        "noise_label": noise_label,
                        "paired_run_count": len(deltas),
                        "wins": int(np.sum(deltas < -PAIR_TOLERANCE)),
                        "ties": int(np.sum(np.abs(deltas) <= PAIR_TOLERANCE)),
                        "losses": int(np.sum(deltas > PAIR_TOLERANCE)),
                        "paired_delta_mean": float(np.mean(deltas)),
                        "paired_delta_median": float(np.median(deltas)),
                        "paired_delta_p2_5": float(np.percentile(deltas, 2.5)),
                        "paired_delta_p97_5": float(np.percentile(deltas, 97.5)),
                    }
                )
            standard = np.asarray(
                [methods["Standard BO"]["final_regret"] for methods in selected]
            )
            fixed = np.asarray(
                [methods["Fixed-Trust Physics BO"]["final_regret"] for methods in selected]
            )
            adaptive = np.asarray(
                [methods[ADAPTIVE_TRUST_METHOD]["final_regret"] for methods in selected]
            )
            fixed_benefit = float(np.mean(standard) - np.mean(fixed))
            adaptive_benefit = float(np.mean(standard) - np.mean(adaptive))
            negative.append(
                {
                    "prior_quality": quality,
                    "noise_label": noise_label,
                    "run_count": len(selected),
                    "standard_final_regret_mean": float(np.mean(standard)),
                    "fixed_final_regret_mean": float(np.mean(fixed)),
                    "adaptive_final_regret_mean": float(np.mean(adaptive)),
                    "fixed_negative_transfer_mean": float(np.mean(fixed - standard)),
                    "adaptive_negative_transfer_mean": float(
                        np.mean(adaptive - standard)
                    ),
                    "negative_transfer_reduction_mean": float(
                        np.mean((fixed - standard) - (adaptive - standard))
                    ),
                    "fixed_physics_benefit_mean": fixed_benefit,
                    "adaptive_physics_benefit_mean": adaptive_benefit,
                    "informative_benefit_retention_fraction": (
                        adaptive_benefit / fixed_benefit
                        if abs(fixed_benefit) > PAIR_TOLERANCE
                        else None
                    ),
                    "adaptive_negative_transfer_positive_fraction": float(
                        np.mean((adaptive - standard) > PAIR_TOLERANCE)
                    ),
                }
            )
    for identity, methods in grouped.items():
        adaptive = methods[ADAPTIVE_TRUST_METHOD]["final_regret"]
        fixed = methods["Fixed-Trust Physics BO"]["final_regret"]
        standard = methods["Standard BO"]["final_regret"]
        if adaptive > fixed + PAIR_TOLERANCE and adaptive > standard + PAIR_TOLERANCE:
            worse_than_both.append(
                {
                    **dict(zip(identity_fields, identity)),
                    "adaptive_final_regret": adaptive,
                    "fixed_final_regret": fixed,
                    "standard_final_regret": standard,
                }
            )
    return pairwise, negative, worse_than_both


def _historical_equivalence(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    historical = json.loads(HISTORICAL_V1_SUMMARY.read_text(encoding="utf-8"))
    historical_lookup = {
        (row["method"], row["prior_quality"], row["noise_label"]): row
        for row in historical["aggregate"]
        if row["budget"] == PRIMARY_BUDGET
        and row["method"] in {"Standard BO", "Physics-Informed BO"}
    }
    maximum_error = 0.0
    compared = 0
    for row in aggregate:
        historical_method = (
            "Physics-Informed BO"
            if row["method"] == "Fixed-Trust Physics BO"
            else row["method"]
        )
        if historical_method not in {"Standard BO", "Physics-Informed BO"}:
            continue
        parent = historical_lookup[
            (historical_method, row["prior_quality"], row["noise_label"])
        ]
        for metric in (
            "final_regret_mean",
            "final_regret_median",
            "final_regret_p95",
            "best_seen_regret_mean",
        ):
            maximum_error = max(maximum_error, abs(float(row[metric]) - float(parent[metric])))
            compared += 1
    return {
        "historical_summary_path": str(HISTORICAL_V1_SUMMARY.relative_to(ROOT)),
        "compared_metric_count": compared,
        "maximum_absolute_error": maximum_error,
        "exact_within_1e_12": maximum_error <= PAIR_TOLERANCE,
    }


def _adversarial_audit() -> dict[str, Any]:
    scenarios = {
        "magnitude_bias_only": [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)],
        "scaling_bias": [(0.0, -0.7), (1.0, 1.1), (2.0, 2.9)],
        "ranking_inversion": [(0.0, 0.0), (1.0, -1.0), (2.0, -2.0)],
        "one_noisy_point": [(100.0, -100.0)],
    }
    output = {}
    for name, pairs in scenarios.items():
        estimator = PhysicsPriorTrustEstimator()
        state = estimator.initial_state()
        evidence = []
        trajectory = []
        for trial_index, (observed, predicted) in enumerate(pairs, start=1):
            evidence.append(
                PhysicsPriorTrustEvidence(
                    trial_index,
                    f"{name}:{trial_index}",
                    0.0025 * trial_index,
                    -0.0025 * trial_index,
                    observed,
                    predicted,
                    10.0 if name == "one_noisy_point" else 0.0,
                )
            )
            state = estimator.update(
                state,
                tuple(evidence),
                trial_index=trial_index,
                current_observation_valid=True,
            )
            trajectory.append(state.as_dict())
        output[name] = {
            "definition_frozen_before_benchmark": True,
            "final_trust": state.trust_score,
            "trust_states": trajectory,
        }
    estimator = PhysicsPriorTrustEstimator()
    initial = estimator.initial_state()
    invalid = estimator.update(
        initial,
        (),
        trial_index=1,
        current_observation_valid=False,
        invalid_reason="GENERIC_FROZEN_INVALID_CASE",
    )
    output["invalid_point"] = {
        "definition_frozen_before_benchmark": True,
        "final_trust": invalid.trust_score,
        "trust_states": [invalid.as_dict()],
    }
    return {
        "classification": "OFFLINE_ALGORITHM_TEST_ONLY",
        "primary_rule_sha256": EXPECTED_RULE_SHA256,
        "scenarios": output,
    }


def run_benchmark(seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    verify_frozen_rule()
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != DEFAULT_SEEDS:
        raise ValueError("formal adaptive-trust benchmark seeds are frozen at 0...4")
    rom_case = make_offline_rom_development_cases()[1]
    rom_controller, profile = determine_synthetic_rom(rom_case)
    domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
    rows: list[dict[str, Any]] = []
    trust_rows: list[dict[str, Any]] = []
    representatives: dict[str, Any] = {}

    for noise_label, noise_std in NOISE_SETTINGS.items():
        for case in make_primary_cases(noise_std):
            for quality in PRIOR_QUALITIES:
                for seed in seed_values:
                    condition_results = {}
                    for method in METHODS:
                        environment = _environment(domain, case, seed)
                        if method == "Standard BO":
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=f"{case.name}:{quality}:{noise_label}:{seed}:standard",
                                profile=profile,
                                domain=domain,
                            ).run(environment, method="Standard BO")
                        elif method == "Fixed-Trust Physics BO":
                            result = ROMGatedPersonalizationEpisode(
                                episode_id=f"{case.name}:{quality}:{noise_label}:{seed}:fixed",
                                profile=profile,
                                domain=domain,
                            ).run(
                                environment,
                                method="Physics-Informed BO",
                                physics_model=_physics(case, quality),
                            )
                        else:
                            result = ROMGatedAdaptiveTrustEpisode(
                                episode_id=f"{case.name}:{quality}:{noise_label}:{seed}:adaptive",
                                profile=profile,
                                domain=domain,
                            ).run(
                                environment,
                                physics_model=_physics(case, quality),
                            )
                        if environment.oracle_access_count != 0:
                            raise RuntimeError("adaptation or trust update accessed oracle")
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
                                "final_trust": None,
                                "fallback_dominant_selection_frequency": None,
                                "trust_after_each_trial": None,
                            }
                        )
                        if method == ADAPTIVE_TRUST_METHOD:
                            adaptive = result.sequential_result
                            metrics["final_trust"] = adaptive.trust_states[-1].trust_score
                            metrics["fallback_dominant_selection_frequency"] = (
                                adaptive.fallback_dominant_selection_frequency
                            )
                            metrics["trust_after_each_trial"] = [
                                state.trust_score for state in adaptive.trust_states
                            ]
                            for trial, (entry, state, regret) in enumerate(
                                zip(
                                    adaptive.ledger.entries,
                                    adaptive.trust_states,
                                    metrics["simple_regret_per_trial"],
                                ),
                                start=1,
                            ):
                                trust_rows.append(
                                    {
                                        "case": case.name,
                                        "seed": seed,
                                        "prior_quality": quality,
                                        "noise_label": noise_label,
                                        "trial": trial,
                                        "candidate_id": entry.candidate.candidate_id,
                                        "canonical_beta_id": domain.by_id(
                                            entry.candidate.candidate_id
                                        ).canonical_beta_id,
                                        "beta_flex": entry.candidate.beta_flex,
                                        "beta_extend": entry.candidate.beta_extend,
                                        "observed_y": entry.observation.endpoint_value,
                                        "physics_prediction": (
                                            entry.physics_prediction_before_observation
                                        ),
                                        "prediction_residual": entry.prediction_residual,
                                        "pairwise_ranking_score": state.ranking_consistency_metrics[
                                            "ranking_score"
                                        ],
                                        "comparable_pair_count": state.ranking_consistency_metrics[
                                            "comparable_pair_count"
                                        ],
                                        "concordant_pair_count": state.ranking_consistency_metrics[
                                            "concordant_pair_count"
                                        ],
                                        "trust_before": entry.physics_prior_trust_before,
                                        "trust_after": entry.physics_prior_trust_after,
                                        "simple_regret": regret,
                                        "next_selected_candidate_id": (
                                            entry.selected_next_candidate.candidate_id
                                            if entry.selected_next_candidate is not None
                                            else None
                                        ),
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
                    representative_key = f"{quality}_{noise_label}_{case.name}_seed{seed}"
                    if representative_key in {
                        "P1_low_smooth_convex_seed0",
                        "P3_low_smooth_convex_seed0",
                    }:
                        representatives[representative_key] = {
                            method: result.as_dict()
                            for method, result in condition_results.items()
                        }

    aggregate = _extended_aggregate(rows)
    pairwise, negative, worse = _paired_analysis(rows)
    historical = _historical_equivalence(aggregate)
    if not historical["exact_within_1e_12"]:
        raise RuntimeError("frozen Standard/Fixed Physics benchmark definitions drifted")
    p3_moderate = next(
        row
        for row in negative
        if row["prior_quality"] == "P3" and row["noise_label"] == "moderate"
    )
    return {
        "status": "ADAPTIVE_PHYSICS_PRIOR_TRUST_PERSONALIZATION_V1_BENCHMARK_COMPLETED",
        "implementation_status": (
            "ADAPTIVE_PHYSICS_PRIOR_TRUST_PERSONALIZATION_V1_"
            "IMPLEMENTED_WITH_LIMITATIONS"
        ),
        "READY_FOR_FIVE_LEG_MUJOCO_BENCHMARK": False,
        "readiness_reason": (
            "The frozen rule does not reliably fall back within K=4: "
            f"P3 moderate-noise adaptive negative transfer is "
            f"{p3_moderate['adaptive_negative_transfer_mean']:.12g}, and "
            f"adaptive trust is worse than both baselines in {len(worse)} of "
            f"{len(rows) // len(METHODS)} paired conditions. A separately "
            "preregistered algorithm revision is required before a five-leg "
            "MuJoCo benchmark."
        ),
        "classification": "OFFLINE_ALGORITHM_DEVELOPMENT_EVIDENCE_ONLY",
        "primary_rule_sha256": EXPECTED_RULE_SHA256,
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
        "trust_trajectory_rows": trust_rows,
        "aggregate": aggregate,
        "pairwise": pairwise,
        "negative_transfer": negative,
        "adaptive_worse_than_both_count": len(worse),
        "adaptive_worse_than_both_cases": worse,
        "historical_V1_baseline_equivalence": historical,
        "adversarial_trust_audit": _adversarial_audit(),
        "representative_ledgers": representatives,
        "causal_audit": {
            "environment_oracle_access_before_post_run_evaluation": 0,
            "trust_rule_inputs": "executed valid observations and pre-observation physics predictions only",
            "case_label_available_to_trust_estimator": False,
            "future_truth_available_to_trust_estimator": False,
        },
        "robot_actions": 0,
        "human_actions": 0,
        "PINN_training": 0,
    }


def _aggregate_lookup(payload: dict[str, Any], quality: str, noise: str, method: str) -> dict[str, Any]:
    return next(
        row
        for row in payload["aggregate"]
        if row["prior_quality"] == quality
        and row["noise_label"] == noise
        and row["method"] == method
    )


def _figure_p1(payload: dict[str, Any], output: Path) -> None:
    rows = [
        row
        for row in payload["rows"]
        if row["prior_quality"] == "P1" and row["noise_label"] == "low"
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method, color in zip(METHODS, ("#4c78a8", "#e45756", "#54a24b")):
        curves = np.asarray(
            [row["simple_regret_per_trial"] for row in rows if row["method"] == method]
        )
        ax.plot(
            range(1, 5), curves.mean(axis=0), marker="o", color=color, label=method
        )
    trust_ax = ax.twinx()
    trust = np.asarray(
        [
            row["trust_after_each_trial"]
            for row in rows
            if row["method"] == ADAPTIVE_TRUST_METHOD
        ]
    )
    trust_ax.plot(
        range(1, 5), trust.mean(axis=0), color="#b279a2", marker="s",
        linestyle="--", label="Adaptive trust after trial"
    )
    ax.set(
        xlabel="adaptation trial",
        ylabel="mean simple regret",
        title="P1 low-noise: regret and causal trust trajectory (offline)",
    )
    trust_ax.set_ylabel("mean algorithmic trust", color="#b279a2")
    trust_ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(1, 5))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = trust_ax.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _figure_p3(payload: dict[str, Any], output: Path) -> None:
    rows = [
        row
        for row in payload["rows"]
        if row["prior_quality"] == "P3" and row["noise_label"] == "low"
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method, color in zip(METHODS, ("#4c78a8", "#e45756", "#54a24b")):
        curves = np.asarray(
            [row["simple_regret_per_trial"] for row in rows if row["method"] == method]
        )
        ax.plot(
            range(1, 5), curves.mean(axis=0), marker="o", color=color, label=method
        )
    ax.set(
        xlabel="adaptation trial",
        ylabel="mean simple regret",
        title="P3 low-noise: fixed, standard, and adaptive trust (offline)",
    )
    ax.set_xticks(range(1, 5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _figure_prior_quality(payload: dict[str, Any], output: Path) -> None:
    qualities = list(PRIOR_QUALITIES)
    x = np.arange(len(qualities))
    fig, (ax_regret, ax_trust) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    width = 0.24
    for index, (method, color) in enumerate(
        zip(METHODS, ("#4c78a8", "#e45756", "#54a24b"))
    ):
        means = [
            _aggregate_lookup(payload, quality, "low", method)["final_regret_mean"]
            for quality in qualities
        ]
        ax_regret.bar(x + (index - 1) * width, means, width, color=color, label=method)
    final_trust = [
        _aggregate_lookup(payload, quality, "low", ADAPTIVE_TRUST_METHOD)[
            "final_trust_mean"
        ]
        for quality in qualities
    ]
    ax_trust.plot(x, final_trust, color="#b279a2", marker="o", linewidth=2)
    ax_regret.set(
        ylabel="mean final regret",
        title="Prior quality: final regret and adaptive trust (low noise, offline)",
    )
    ax_regret.legend(fontsize=8)
    ax_trust.set(ylabel="mean final trust", xlabel="frozen prior quality")
    ax_trust.set_ylim(0.0, 1.05)
    ax_trust.set_xticks(x, qualities)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "run_metrics.csv", payload["rows"])
    _write_csv(output_dir / "aggregate_metrics.csv", payload["aggregate"])
    _write_csv(output_dir / "paired_comparisons.csv", payload["pairwise"])
    _write_csv(output_dir / "negative_transfer_summary.csv", payload["negative_transfer"])
    _write_csv(output_dir / "trust_trajectories.csv", payload["trust_trajectory_rows"])
    (output_dir / "adversarial_trust_audit.json").write_text(
        json.dumps(payload["adversarial_trust_audit"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _figure_p1(payload, output_dir / "figure_1_p1_regret_and_trust.png")
    _figure_p3(payload, output_dir / "figure_2_p3_method_comparison.png")
    _figure_prior_quality(payload, output_dir / "figure_3_prior_quality_regret_trust.png")
    artifact_names = (
        "benchmark_summary.json",
        "run_metrics.csv",
        "aggregate_metrics.csv",
        "paired_comparisons.csv",
        "negative_transfer_summary.csv",
        "trust_trajectories.csv",
        "adversarial_trust_audit.json",
        "figure_1_p1_regret_and_trust.png",
        "figure_2_p3_method_comparison.png",
        "figure_3_prior_quality_regret_trust.png",
    )
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in artifact_names),
        encoding="utf-8",
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
                "status": payload["status"],
                "run_count": payload["run_count"],
                "primary_rule_sha256": payload["primary_rule_sha256"],
                "historical_baselines_exact": payload[
                    "historical_V1_baseline_equivalence"
                ]["exact_within_1e_12"],
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
