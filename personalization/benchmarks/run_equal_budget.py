"""Run and visualize the preregistered equal-budget offline smoke benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..candidates import V3CandidateDomain
from ..environment import AnalyticBenchmarkEnvironment, make_primary_cases
from ..models.physics_graybox import (
    AnalyticDevelopmentPhysicsAdapter,
    PhysicsSubjectModel,
)
from ..models.residual_gp import PhysicsInformedResidualModel
from ..sequential import METHODS, run_sequential_personalization
from .metrics import aggregate_rows, evaluate_run, physics_bo_pairwise


PRIMARY_BUDGET = 4
SENSITIVITY_BUDGETS = (3, 5)
PRIOR_QUALITIES = ("P0", "P1", "P2", "P3")
NOISE_SETTINGS = {"zero": 0.0, "low": 0.03, "moderate": 0.10}


def _physics(case, quality: str) -> PhysicsSubjectModel:
    return PhysicsSubjectModel(
        AnalyticDevelopmentPhysicsAdapter(
            optimum_beta=case.optimum_beta,
            prior_quality=quality,
            landscape=case.landscape,
        )
    )


def run_one(
    domain: V3CandidateDomain,
    *,
    case,
    seed: int,
    budget: int,
    prior_quality: str,
    noise_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    ledgers = {}
    for method in METHODS:
        environment = AnalyticBenchmarkEnvironment(domain, case, seed=seed)
        result = run_sequential_personalization(
            environment,
            domain,
            method=method,
            budget=budget,
            seed=seed,
            physics_model=(
                _physics(case, prior_quality)
                if method in {"Model-Only Greedy", "Physics-Informed BO"}
                else None
            ),
        )
        if environment.oracle_access_count != 0:
            raise RuntimeError("selector/model accessed analytic oracle during adaptation")
        metrics = evaluate_run(result, environment)
        metrics.update(
            {
                "case": case.name,
                "seed": seed,
                "prior_quality": prior_quality,
                "noise_label": noise_label,
                "noise_std": case.noise_std,
            }
        )
        rows.append(metrics)
        ledgers[method] = result.as_dict()
    return rows, ledgers


def run_benchmark(
    *,
    seeds: Iterable[int] = range(5),
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    domain = V3CandidateDomain.from_frozen_artifact()
    rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    for noise_label, noise_std in NOISE_SETTINGS.items():
        for case in make_primary_cases(noise_std):
            for quality in PRIOR_QUALITIES:
                for seed in seeds:
                    new_rows, ledgers = run_one(
                        domain,
                        case=case,
                        seed=int(seed),
                        budget=PRIMARY_BUDGET,
                        prior_quality=quality,
                        noise_label=noise_label,
                    )
                    rows.extend(new_rows)
                    if (
                        representative is None
                        and case.name == "mildly_nonlinear"
                        and quality == "P1"
                        and noise_label == "low"
                    ):
                        representative = {
                            "case": case,
                            "quality": quality,
                            "seed": int(seed),
                            "ledgers": ledgers,
                        }
    sensitivity = []
    if include_sensitivity:
        case = make_primary_cases(NOISE_SETTINGS["low"])[0]
        for budget in SENSITIVITY_BUDGETS:
            for quality in PRIOR_QUALITIES:
                for seed in list(seeds)[:3]:
                    new_rows, _ = run_one(
                        domain,
                        case=case,
                        seed=int(seed),
                        budget=budget,
                        prior_quality=quality,
                        noise_label="low",
                    )
                    sensitivity.extend(new_rows)
    return {
        "status": "PHYSICS_INFORMED_SEQUENTIAL_PERSONALIZATION_V1_IMPLEMENTED_WITH_LIMITATIONS",
        "classification": "OFFLINE_ALGORITHM_EVIDENCE",
        "primary_budget": PRIMARY_BUDGET,
        "sensitivity_budgets": list(SENSITIVITY_BUDGETS),
        "candidate_domain": "P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3",
        "candidate_count": len(domain),
        "pinning": {"PINN_NOT_JUSTIFIED": True, "PINN_TRAINING": 0},
        "rows": rows,
        "aggregate": aggregate_rows(rows),
        "pairwise": physics_bo_pairwise(rows),
        "sensitivity_rows": sensitivity,
        "sensitivity_aggregate": aggregate_rows(sensitivity) if sensitivity else [],
        "representative": representative,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def _figure_landscape(payload: dict[str, Any], output: Path) -> None:
    representative = payload["representative"]
    case = representative["case"]
    domain = V3CandidateDomain.from_frozen_artifact()
    environment = AnalyticBenchmarkEnvironment(domain, case, seed=representative["seed"])
    axis = sorted({item.beta_flex for item in domain})
    values = np.asarray(
        [environment._truth(domain.nearest(x, z)) for x in axis for z in axis]
    ).reshape((len(axis), len(axis)))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    contour = ax.contourf(axis, axis, values.T, levels=24, cmap="viridis")
    for method, result in representative["ledgers"].items():
        points = [entry["candidate"] for entry in result["ledger"]["entries"]]
        ax.plot(
            [point["beta_flex"] for point in points],
            [point["beta_extend"] for point in points],
            marker="o",
            linewidth=1.2,
            label=method,
        )
    ax.set(xlabel="beta_flex", ylabel="beta_extend", title="Offline development landscape and K=4 samples")
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(contour, ax=ax, label="synthetic mechanical cost")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _figure_regret(payload: dict[str, Any], output: Path) -> None:
    rows = [
        row
        for row in payload["rows"]
        if row["prior_quality"] == "P1" and row["noise_label"] == "low"
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        curves = np.asarray([row["simple_regret_per_trial"] for row in selected])
        ax.plot(range(1, PRIMARY_BUDGET + 1), curves.mean(axis=0), marker="o", label=method)
    ax.set(xlabel="adaptation trial", ylabel="mean simple regret", title="Equal-budget regret (P1, low noise)")
    ax.set_xticks(range(1, PRIMARY_BUDGET + 1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _figure_correction(payload: dict[str, Any], output: Path) -> None:
    representative = payload["representative"]
    case = representative["case"]
    domain = V3CandidateDomain.from_frozen_artifact()
    ledger = representative["ledgers"]["Physics-Informed BO"]["ledger"]["entries"]
    history = []
    from ..observations import EpisodeObservation

    for entry in ledger:
        raw = entry["observation"]
        history.append(EpisodeObservation(**raw))
    physics = _physics(case, representative["quality"])
    model = PhysicsInformedResidualModel(physics)
    model.fit(history)
    slice_candidates = [item for item in domain if item.beta_extend == 0.0]
    x = [item.beta_flex for item in slice_candidates]
    prior = [physics.predict(item).mean for item in slice_candidates]
    residual = [model.predict(item).metadata["residual_mean"] for item in slice_candidates]
    corrected = [model.predict(item).mean for item in slice_candidates]
    std = [model.predict(item).std for item in slice_candidates]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True)
    axes[0].plot(x, prior); axes[0].set_ylabel("physics prior")
    axes[1].plot(x, residual); axes[1].fill_between(x, np.asarray(residual)-np.asarray(std), np.asarray(residual)+np.asarray(std), alpha=.2); axes[1].set_ylabel("residual GP")
    axes[2].plot(x, corrected); axes[2].set_ylabel("corrected mean"); axes[2].set_xlabel("beta_flex at beta_extend=0")
    fig.suptitle("Physics prior + residual GP = corrected prediction")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _figure_prior_quality(payload: dict[str, Any], output: Path) -> None:
    rows = [row for row in payload["rows"] if row["noise_label"] == "low"]
    x = np.arange(len(PRIOR_QUALITIES))
    width = 0.13
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for index, method in enumerate(METHODS):
        means = [
            np.mean(
                [
                    row["final_regret"]
                    for row in rows
                    if row["method"] == method and row["prior_quality"] == quality
                ]
            )
            for quality in PRIOR_QUALITIES
        ]
        ax.bar(x + (index - 2.5) * width, means, width, label=method)
    ax.set_xticks(x, PRIOR_QUALITIES)
    ax.set(ylabel="mean final regret", xlabel="physics-prior quality", title="Prior-quality stress test (low noise)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in payload.items() if key != "representative"}
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "run_metrics.csv", payload["rows"])
    _write_csv(output_dir / "aggregate_metrics.csv", payload["aggregate"])
    _write_csv(output_dir / "prior_pairwise_win_tie_loss.csv", payload["pairwise"])
    _write_csv(output_dir / "budget_sensitivity.csv", payload["sensitivity_aggregate"])
    _figure_landscape(payload, output_dir / "figure_1_landscape_samples.png")
    _figure_regret(payload, output_dir / "figure_2_regret_vs_trial.png")
    _figure_correction(payload, output_dir / "figure_3_physics_residual_corrected.png")
    _figure_prior_quality(payload, output_dir / "figure_4_prior_quality.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_v1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    payload = run_benchmark(
        seeds=args.seeds, include_sensitivity=not args.no_sensitivity
    )
    write_outputs(payload, args.output_dir)
    print(json.dumps({
        "status": payload["status"],
        "primary_run_count": len(payload["rows"]),
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
