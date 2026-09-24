"""Native MyoLeg screening for the predeclared V2 mechanism candidate family.

This is a scope diagnostic, not an optimization run. Every subject receives
the same 27 candidates, the reference is evaluated first, and the evaluator
computes E3/E2/peak and an evaluator-only constrained oracle only after all
requested traces are collected. The V1 candidate domain and outputs are
untouched. The result is a mechanism-scope diagnostic, not a personalization
performance evaluation: no learner selects a candidate and no regret reduction
claim is supported by this screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import Environment
from .mechanism_candidates import MechanismDomain
from .simulation import ROOT, SimulatorBackend, development_ids


DEFAULT_OUTPUT = ROOT / "outputs/myoleg_mechanism_screen_v1"
DEFAULT_CACHE = ROOT / ".cache/myoleg-benchmark-v1"


def _subject_rows(subject_id: str, *, cache_dir: Path) -> list[dict[str, object]]:
    domain = MechanismDomain()
    backend = SimulatorBackend(subject_id, domain, cache_dir)
    environment = Environment(domain, backend, subject_id=subject_id, noise_std=0.0, seed=0, tier=0.01, physics=False)
    rows = []
    for index, point in enumerate(domain, 1):
        measurement = environment.evaluate(point, index)
        observation = measurement.observation
        rows.append({
            "subject_id": subject_id,
            "family": domain.family,
            "candidate_id": point.candidate_id,
            "duration_scale": point.duration_scale,
            "coordination_lag": point.coordination_lag,
            "coordination_amplitude": point.coordination_amplitude,
            "kinematic_hip_speed_ratio": point.kinematic["hip_speed_ratio"],
            "kinematic_knee_speed_ratio": point.kinematic["knee_speed_ratio"],
            "kinematic_hip_accel_ratio": point.kinematic["hip_accel_ratio"],
            "kinematic_knee_accel_ratio": point.kinematic["knee_accel_ratio"],
            "E3": float(observation.endpoint_value) if observation.valid else np.nan,
            "E2": measurement.E2,
            "peak_ratio": measurement.peak_ratio,
            "feasible": bool(measurement.feasible),
            "valid": bool(observation.valid),
            "invalid_reason": observation.invalid_reason or "",
            "disk_hit": int(backend.disk_hits),
            "fresh_simulations": int(backend.fresh_simulations),
        })
    return rows


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    oracle_rows = []
    ranks = []
    pivot = rows.pivot(index="candidate_id", columns="subject_id", values="E3")
    subjects = sorted(str(x) for x in rows.subject_id.unique())
    for subject_id, group in rows.groupby("subject_id", sort=True):
        feasible = group[group.feasible & group.E3.notna()]
        pool = feasible if not feasible.empty else group[group.E3.notna()]
        oracle = pool.sort_values(["E3", "candidate_id"], kind="stable").iloc[0]
        reference = group[group.candidate_id == "DURATION_COORDINATION_V2:0"].iloc[0]
        oracle_rows.append({
            "subject_id": subject_id,
            "oracle_candidate_id": oracle.candidate_id,
            "oracle_E3": float(oracle.E3),
            "reference_E3": float(reference.E3),
            "reference_regret": float(reference.E3 - oracle.E3),
            "feasible_count": int(len(feasible)),
            "candidate_count": int(len(group)),
        })
    for i, left in enumerate(subjects):
        for right in subjects[i + 1:]:
            common = pivot[[left, right]].dropna()
            ranks.append(float(common[left].rank().corr(common[right].rank())))
    summary = pd.DataFrame(oracle_rows)
    metrics = {
        "subject_count": int(len(subjects)),
        "candidate_count": int(rows.candidate_id.nunique()),
        "oracle_unique_count": int(summary.oracle_candidate_id.nunique()),
        "oracle_unique_fraction": float(summary.oracle_candidate_id.nunique() / len(summary)) if len(summary) else 0.0,
        "rank_spearman_median": float(np.nanmedian(ranks)) if ranks else 1.0,
        "rank_spearman_min": float(np.nanmin(ranks)) if ranks else 1.0,
        "reference_regret_median": float(summary.reference_regret.median()) if len(summary) else np.nan,
        "reference_regret_max": float(summary.reference_regret.max()) if len(summary) else np.nan,
        "feasible_fraction_median": float((summary.feasible_count / summary.candidate_count).median()) if len(summary) else np.nan,
    }
    return summary, metrics


def run_screen(*, subjects: list[str] | None = None, output_dir: Path = DEFAULT_OUTPUT,
               cache_dir: Path = DEFAULT_CACHE) -> Path:
    subjects = list(development_ids()) if subjects is None else list(subjects)
    allowed = set(development_ids())
    if not set(subjects).issubset(allowed):
        raise ValueError("MECHANISM_SCREEN_ONLY_FROZEN_DEVELOPMENT_SUBJECTS")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = pd.DataFrame([row for subject_id in subjects for row in _subject_rows(subject_id, cache_dir=Path(cache_dir))])
    summary, metrics = summarize(rows)
    rows.to_csv(output_dir / "landscape.csv", index=False)
    summary.to_csv(output_dir / "oracle_summary.csv", index=False)
    protocol = {
        "experiment_id": "MYOLEG_MECHANISM_SCREEN_V1",
        "subject_scope": "frozen development subjects only; no sealed access",
        "candidate_family": "DURATION_COORDINATION_V2",
        "candidate_design": "duration_scale x smooth knee coordination lag x coordination amplitude",
        "candidate_count": int(rows.candidate_id.nunique()),
        "reference": "DURATION_COORDINATION_V2:0 = (1.0, 0.0, 0.0)",
        "cache_dir": str(Path(cache_dir)),
        "learner_access": "none; evaluator-only diagnostic",
        "evaluation_scope": "native prescribed-state mechanism screen; no adaptive policy or personalization performance evaluation",
        "oracle_scope": "evaluator-only after all candidates; not available to a learner and not a personalized recommendation",
        "screen_decision_rule": "provisional_signal iff oracle_unique_count > 1 and cross_subject_E3_rank_spearman_median < 0.99; exploratory screen only, not a confirmatory personalization criterion",
        "assistance_force": "not represented by native prescribed-state interface",
        "metrics": metrics,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# MyoLeg V2 mechanism candidate screen", "",
             "Native prescribed-state mechanism diagnostic; no assistance force or clinical effect is represented.",
             "The constrained oracle is evaluator-only after the complete landscape is collected. No adaptive learner was run, so this report does not establish personalized benefit or regret reduction.", "",
             f"- subjects: {metrics['subject_count']}", f"- candidates: {metrics['candidate_count']}",
             f"- unique constrained oracles: {metrics['oracle_unique_count']}/{metrics['subject_count']}",
             f"- cross-subject E3 rank Spearman median/min: {metrics['rank_spearman_median']:.6f}/{metrics['rank_spearman_min']:.6f}",
             f"- reference regret median/max: {metrics['reference_regret_median']:.6f}/{metrics['reference_regret_max']:.6f}", ""]
    if metrics["oracle_unique_count"] > 1 and metrics["rank_spearman_median"] < 0.99:
        lines.append("Decision: this screen shows a provisional subject-dependent ranking signal; it may justify a separately frozen controlled comparison, but it is not evidence that personalization succeeds.")
    else:
        lines.append("Decision: this mechanism family still does not create enough subject-dependent ranking variation; design a new interaction mechanism before expanding the grid. This native screen does not support a personalization-success claim.")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--subjects", nargs="+", default=None)
    args = parser.parse_args(argv)
    path = run_screen(subjects=args.subjects, output_dir=args.output_dir, cache_dir=args.cache_dir)
    print(json.dumps({"complete": True, "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
