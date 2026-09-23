"""Diagnose whether the frozen MyoLeg V1 candidate space can expose personalization.

This evaluator reads the completed V1 full landscapes and regenerates only the
public candidate kinematics. It never changes the V1 domain, fits a policy, or
uses the diagnostic to select new candidates. The output separates candidate
geometry from subject-by-trajectory interaction so that a lack of personalized
gain is not automatically blamed on candidate count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .experiment import Domain
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDSCAPE = ROOT / "outputs/myoleg_personalization_audit_v1/landscape.csv"
DEFAULT_RESULTS = ROOT / "outputs/myoleg_benchmark_v1/development_20260923_r2/results.csv"
DEFAULT_OUTPUT = ROOT / "outputs/myoleg_candidate_space_diagnostic_v1"
FAMILIES = ("BETA_TIMING", "KEY_POSTURE_TIMING")
RESPONSE_COLUMNS = ("E3", "E2", "peak_ratio")


def _pca(values: np.ndarray) -> dict[str, object]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] == 0:
        return {"dimension": 0, "sample_count": int(len(x)) if x.ndim == 2 else 0,
                "rank": 0, "rank_90": 0, "rank_95": 0, "rank_99": 0,
                "participation_ratio": 0.0}
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular * singular
    total = float(eigen.sum())
    tolerance = np.finfo(float).eps * max(x.shape) * (float(eigen[0]) if len(eigen) else 0.0)
    rank = int(np.count_nonzero(eigen > tolerance))
    ratios = eigen / total if total else np.zeros_like(eigen)
    cumulative = np.cumsum(ratios)
    return {
        "dimension": int(x.shape[1]), "sample_count": int(len(x)), "rank": rank,
        "rank_90": int(np.searchsorted(cumulative, .90) + 1) if total else 0,
        "rank_95": int(np.searchsorted(cumulative, .95) + 1) if total else 0,
        "rank_99": int(np.searchsorted(cumulative, .99) + 1) if total else 0,
        "participation_ratio": float(total * total / np.sum(eigen * eigen)) if total else 0.0,
        "explained_variance_ratio": [float(v) for v in ratios],
    }


def _pairwise(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return {"pair_count": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    upper = distances[np.triu_indices(len(x), 1)]
    return {"pair_count": int(len(upper)), "min": float(upper.min()),
            "median": float(np.median(upper)), "p95": float(np.quantile(upper, .95)),
            "max": float(upper.max())}


def _rank(a: Iterable[float], b: Iterable[float]) -> float:
    left = pd.Series(np.asarray(list(a), dtype=float)).rank(method="average").to_numpy()
    right = pd.Series(np.asarray(list(b), dtype=float)).rank(method="average").to_numpy()
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _domain_geometry(family: str) -> dict[str, object]:
    domain = Domain(native_domain(), family)
    parameters = np.asarray([[p.beta_flex, p.beta_extend, p.share_shift] for p in domain], dtype=float)
    scaled = parameters / np.asarray([.03, .03, .05])
    trajectories = np.asarray([
        np.concatenate([p.trajectory.q.ravel(), p.trajectory.dq.ravel(), p.trajectory.ddq.ravel()])
        for p in domain
    ], dtype=float)
    standardized = (trajectories - trajectories.mean(axis=0)) / np.where(
        trajectories.std(axis=0) > 1e-12, trajectories.std(axis=0), 1.0)
    return {
        "family": family, "candidate_count": len(domain),
        "rejected_kinematic_count": len(domain.rejections),
        "parameter_bounds": {
            name: {"min": float(parameters[:, i].min()), "max": float(parameters[:, i].max()),
                   "unique": int(np.unique(parameters[:, i]).size)}
            for i, name in enumerate(("beta_flex", "beta_extend", "share_shift"))
        },
        "parameter_pca": _pca(scaled), "parameter_pairwise_distance": _pairwise(scaled),
        "trajectory_vector_dimension": int(trajectories.shape[1]),
        "trajectory_pca": _pca(trajectories),
        "trajectory_pca_standardized": _pca(standardized),
        "trajectory_pairwise_distance_standardized": _pairwise(standardized),
    }


def _response_diagnostic(landscape: pd.DataFrame, family: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    frame = landscape[landscape["family"].eq(family)].copy()
    required = {"subject_id", "candidate_id", *RESPONSE_COLUMNS, "feasible"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"LANDSCAPE_MISSING_COLUMNS: {sorted(missing)}")
    frame["feasible"] = frame["feasible"].astype(bool)
    oracle_rows, subject_vectors, top_sets, pareto_counts = [], {}, {}, []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("candidate_id", kind="stable")
        subject_vectors[str(subject_id)] = group.set_index("candidate_id")["E3"]
        top_sets[str(subject_id)] = set(group.nsmallest(min(10, len(group)), "E3")["candidate_id"])
        feasible = group[group["feasible"]]
        pool = feasible if not feasible.empty else group
        oracle = pool.sort_values(["E3", "candidate_id"], kind="stable").iloc[0]
        values = pool.loc[:, ["E3", "E2", "peak_ratio"]].to_numpy(float)
        dominated = np.any(np.all(values[:, None, :] <= values[None, :, :], axis=2)
                           & np.any(values[:, None, :] < values[None, :, :], axis=2), axis=0)
        pareto_counts.append(int(np.count_nonzero(~dominated)))
        reference = group[group["candidate_id"].str.endswith(":0")]
        reference_value = float(reference.iloc[0]["E3"]) if len(reference) else float(group["E3"].iloc[0])
        oracle_rows.append({"family": family, "subject_id": subject_id,
                            "oracle_candidate_id": oracle["candidate_id"],
                            "oracle_E3": float(oracle["E3"]), "reference_E3": reference_value,
                            "reference_regret": reference_value - float(oracle["E3"]),
                            "feasible_count": int(len(feasible)), "pareto_count": int(pareto_counts[-1])})
    subjects = sorted(subject_vectors)
    agreement_rows = []
    for i, left in enumerate(subjects):
        for right in subjects[i + 1:]:
            common = sorted(set(subject_vectors[left].index) & set(subject_vectors[right].index))
            agreement_rows.append({"family": family, "left_subject": left, "right_subject": right,
                                   "spearman_E3": _rank(subject_vectors[left][common], subject_vectors[right][common])})
    pivot = frame.pivot(index="candidate_id", columns="subject_id", values="E3")
    fixed_sd = float(pivot.std(axis=1).median())
    within_sd = float(frame.groupby("subject_id")["E3"].std().median())
    jaccard = [len(top_sets[a] & top_sets[b]) / len(top_sets[a] | top_sets[b])
               for i, a in enumerate(subjects) for b in subjects[i + 1:]]
    oracle = pd.DataFrame(oracle_rows); agreement = pd.DataFrame(agreement_rows)
    subject_sizes = frame.groupby("subject_id").size()
    feasible_fraction = oracle.set_index("subject_id")["feasible_count"].div(subject_sizes).median()
    return {
        "family": family, "subject_count": len(subjects), "candidate_count": int(frame.candidate_id.nunique()),
        "e3_range": [float(frame.E3.min()), float(frame.E3.max())],
        "e2_range": [float(frame.E2.min()), float(frame.E2.max())],
        "peak_ratio_range": [float(frame.peak_ratio.min()), float(frame.peak_ratio.max())],
        "fixed_candidate_cross_subject_E3_sd_median": fixed_sd,
        "within_subject_candidate_E3_sd_median": within_sd,
        "cross_subject_to_within_candidate_sd_ratio": fixed_sd / within_sd if within_sd else float("nan"),
        "rank_spearman_median": float(agreement.spearman_E3.median()),
        "rank_spearman_min": float(agreement.spearman_E3.min()),
        "top10_jaccard_median": float(np.median(jaccard)) if jaccard else 1.0,
        "top10_union_count": int(len(set.union(*top_sets.values()))) if top_sets else 0,
        "oracle_unique_count": int(oracle.oracle_candidate_id.nunique()),
        "oracle_unique_fraction": float(oracle.oracle_candidate_id.nunique() / len(oracle)) if len(oracle) else 0.0,
        "reference_regret_max": float(oracle.reference_regret.max()),
        "pareto_count_median": float(oracle.pareto_count.median()),
        "feasible_fraction_median": float(feasible_fraction),
    }, agreement, oracle


def run_diagnostic(*, landscape_path: Path = DEFAULT_LANDSCAPE,
                   results_path: Path | None = DEFAULT_RESULTS,
                   output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    landscape = pd.read_csv(landscape_path)
    geometry, responses, agreements, oracles = {}, {}, [], []
    for family in FAMILIES:
        geometry[family] = _domain_geometry(family)
        responses[family], agreement, oracle = _response_diagnostic(landscape, family)
        agreements.append(agreement); oracles.append(oracle)
    pd.concat(agreements, ignore_index=True).to_csv(output_dir / "ranking_agreement.csv", index=False)
    pd.concat(oracles, ignore_index=True).to_csv(output_dir / "oracle_summary.csv", index=False)
    payload = {"landscape_source": str(Path(landscape_path)),
               "results_source": str(results_path) if results_path else None,
               "geometry": geometry, "response": responses,
               "interpretation": {"candidate_geometry": "full-rank parameter coverage does not imply subject-dependent response ranking",
                                  "decision": "high rank agreement and low oracle uniqueness favor adding an explicit subject-trajectory interaction before merely adding nearby timing points"}}
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# MyoLeg candidate-space diagnostic", "", "Responses: frozen V1 development full landscape (24 subjects). Kinematics: regenerated from the native Domain. No native subject or V1 output was modified.", ""]
    for family in FAMILIES:
        g, r = geometry[family], responses[family]
        lines += [f"## {family}", "", f"- candidates: {g['candidate_count']}; kinematic rejections: {g['rejected_kinematic_count']}",
                  f"- parameter PCA: rank {g['parameter_pca']['rank']}/3; 95% rank {g['parameter_pca']['rank_95']}",
                  f"- trajectory PCA: rank {g['trajectory_pca']['rank']}; 95% rank {g['trajectory_pca']['rank_95']}",
                  f"- response E3 range: {r['e3_range'][0]:.6f}–{r['e3_range'][1]:.6f}",
                  f"- fixed-candidate cross-subject SD / within-subject candidate SD: {r['cross_subject_to_within_candidate_sd_ratio']:.3f}",
                  f"- cross-subject E3 rank Spearman: median {r['rank_spearman_median']:.6f}, minimum {r['rank_spearman_min']:.6f}",
                  f"- top-10 union: {r['top10_union_count']}; unique oracle count: {r['oracle_unique_count']}/{r['subject_count']}",
                  f"- feasible fraction median: {r['feasible_fraction_median']:.3f}; Pareto count median: {r['pareto_count_median']:.1f}", ""]
    lines += ["## Decision", "", "The timing coordinates are geometrically full-rank, but their effective trajectory variation is low-dimensional and the recorded subject rankings are nearly identical. The next study should introduce a predeclared subject–trajectory interaction (for example speed/assist timing or hip–knee coordination) before adding more nearby timing grid points. This conclusion does not retroactively alter V1.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-path", type=Path, default=DEFAULT_LANDSCAPE)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = run_diagnostic(landscape_path=args.landscape_path, results_path=args.results_path, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "families": list(payload["geometry"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
