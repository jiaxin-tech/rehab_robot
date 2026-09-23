"""Causal held-out validation of the five-parameter MyoLeg gray-box model.

This evaluator is intentionally separate from the frozen V1 runner.  It reads
only the candidate IDs in a causal trial prefix, reconstructs the corresponding
measurement payloads through :class:`SimulatorBackend`, fits the existing
five-parameter model, and then scores predictions on candidates that were not
in that prefix.  The full candidate response is used only by this evaluator to
measure prediction error; no truth is passed to a learner or selection rule.

Typical smoke run (one development subject and one random seed)::

    python -m lower_limb_sim.myoleg_benchmark.graybox_cv \
      --subjects MYOLEG_VP_001 --methods RANDOM --seeds 0 \
      --families BETA_TIMING --budgets 1 2 4 8 --max-groups 4

The resulting ``summary.csv`` contains fit/boundary/mismatch diagnostics and
top-regret metrics; ``predictions.csv`` contains torque and E3/E2/peak errors
for every held-out candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lower_limb_sim.config import L1, L2, identification_lower_bounds, identification_upper_bounds
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
    predict_joint_torque,
)
from lower_limb_sim.mechanical_endpoints import branch_rms_components
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain

from .experiment import Domain, FAMILIES, is_feasible, paired_noise, response_metrics
from .simulation import NOMINAL, ROOT, SimulatorBackend, development_ids


DEFAULT_EXECUTED_RESULTS = ROOT / "outputs/myoleg_benchmark_v1/development_20260923_r2"
COMPONENT_NAMES = ("hip_flex", "hip_extend", "knee_flex", "knee_extend")
_TOL = 1e-8


@dataclass(frozen=True)
class FitResult:
    """Serializable fit diagnostics used by the evaluator and its tests."""

    parameters: dict[str, float]
    success: bool
    status: str
    cost: float
    train_torque_rmse_nm: float
    train_samples: int
    boundary_hits: tuple[str, ...]
    optimizer_message: str


def _point_frame(point, tau: np.ndarray, *, episode_id: str) -> pd.DataFrame:
    """Map a torque trace to the estimator's declared planar-force schema."""
    tau = np.asarray(tau, dtype=float)
    if tau.shape != (len(point.time_s), 2) or not np.isfinite(tau).all():
        raise ValueError("INVALID_TORQUE_TRACE")
    q = np.asarray(point.trajectory.q, dtype=float)
    dq = np.asarray(point.trajectory.dq, dtype=float)
    ddq = np.asarray(point.trajectory.ddq, dtype=float)
    jacobian_t = leg_jacobian(q[:, 0], q[:, 1], L1, L2).swapaxes(-1, -2)
    force = np.linalg.solve(jacobian_t, tau[..., None])[..., 0]
    return pd.DataFrame({
        "time_s": np.asarray(point.time_s, dtype=float),
        "episode_id": episode_id,
        "q_hip_rad": q[:, 0], "q_knee_rad": q[:, 1],
        "dq_hip_rad_s": dq[:, 0], "dq_knee_rad_s": dq[:, 1],
        "ddq_hip_rad_s2": ddq[:, 0], "ddq_knee_rad_s2": ddq[:, 1],
        "fx_observed_n": force[:, 0], "fz_observed_n": force[:, 1],
        "sample_valid": np.ones(len(point.time_s), dtype=bool),
    })


def fit_graybox(
    traces: Sequence[tuple[object, np.ndarray]],
    *,
    baseline_template=None,
) -> FitResult:
    """Fit the existing five-parameter model using only ``traces``.

    ``traces`` are ordered causal observations.  No candidate outcome table is
    accepted by this function, which makes accidental full-landscape fitting
    difficult.  A failed/singular fit is reported explicitly and falls back to
    the declared initial guess solely so that held-out diagnostics remain
    inspectable.
    """
    template = baseline_template or baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS["baseline"])
    if not traces:
        raise ValueError("GRAYBOX_REQUIRES_AT_LEAST_ONE_TRACE")
    frames = [_point_frame(point, tau, episode_id=f"graybox:{i}") for i, (point, tau) in enumerate(traces)]
    training = pd.concat(frames, ignore_index=True)
    try:
        result = estimate_subject_parameters(training, template, L1=L1, L2=L2)
        params = {name: float(result.estimated_parameters[name]) for name in PARAMETER_NAMES}
        boundary = tuple(
            name for name in PARAMETER_NAMES
            if abs(params[name] - float(identification_lower_bounds[name])) <= _TOL
            or abs(params[name] - float(identification_upper_bounds[name])) <= _TOL
        )
        return FitResult(
            parameters=params,
            success=bool(result.optimizer_success),
            status="OK" if result.optimizer_success else "OPTIMIZER_FAILED",
            cost=float(result.cost),
            train_torque_rmse_nm=float(result.residual_statistics["torque_rmse_combined_nm"]),
            train_samples=int(result.valid_training_samples),
            boundary_hits=boundary,
            optimizer_message=str(result.optimizer_message),
        )
    except Exception as error:  # diagnostics must preserve a failed prefix
        initial = {name: float({
            "mass_scale": 1.0,
            "k_hip_nm_per_rad": 10.0,
            "k_knee_nm_per_rad": 10.0,
            "b_hip_nm_s_per_rad": 1.0,
            "b_knee_nm_s_per_rad": 1.0,
        }[name]) for name in PARAMETER_NAMES}
        return FitResult(
            parameters=initial, success=False, status="FIT_FAILED",
            cost=float("nan"), train_torque_rmse_nm=float("nan"),
            train_samples=int(len(training)), boundary_hits=(),
            optimizer_message=f"{type(error).__name__}: {error}",
        )


def _measured_trace(
    backend,
    point,
    *,
    subject_id: str,
    family: str,
    seed: int,
    noise_std: float,
    joint_scale: np.ndarray,
) -> np.ndarray:
    tau = np.asarray(backend.requested(point), dtype=float)
    if noise_std:
        tau = paired_noise(
            tau, subject_id=subject_id, family=family,
            candidate_id=point.candidate_id, seed=seed,
            relative_std=noise_std, joint_scale=joint_scale,
        )
    if tau.shape != (len(point.time_s), 2) or not np.isfinite(tau).all():
        raise ValueError("INVALID_MEASURED_TRACE")
    return tau


def evaluate_prefix(
    domain: Domain,
    backend,
    *,
    subject_id: str,
    family: str,
    method: str,
    seed: int,
    noise_std: float,
    prefix_candidate_ids: Sequence[str],
    budget: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit one causal prefix and score every candidate outside that prefix."""
    prefix_ids = list(dict.fromkeys(str(x) for x in prefix_candidate_ids))
    if not prefix_ids or prefix_ids[0] != domain.reference.candidate_id:
        raise ValueError("GRAYBOX_PREFIX_MUST_START_WITH_REFERENCE")
    points = [domain.by_id(candidate_id) for candidate_id in prefix_ids]
    raw_reference = np.asarray(backend.requested(domain.reference), dtype=float)
    if raw_reference.shape != (len(domain.reference.time_s), 2):
        raise ValueError("INVALID_REFERENCE_TRACE")
    joint_scale = np.sqrt(np.mean(raw_reference ** 2, axis=0))
    reference = raw_reference if not noise_std else paired_noise(
        raw_reference, subject_id=subject_id, family=family,
        candidate_id=domain.reference.candidate_id, seed=seed,
        relative_std=noise_std, joint_scale=joint_scale,
    )
    traces: dict[str, np.ndarray] = {domain.reference.candidate_id: reference}
    for point in points[1:]:
        traces[point.candidate_id] = _measured_trace(
            backend, point, subject_id=subject_id, family=family,
            seed=seed, noise_std=noise_std, joint_scale=joint_scale,
        )
    fit = fit_graybox([(point, traces[point.candidate_id]) for point in points])
    template = baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS["baseline"])
    reference = traces[domain.reference.candidate_id]
    phases = domain.subject_reference.phases
    reference_components = np.asarray(
        branch_rms_components(*reference.T, domain.reference.time_s, phases), dtype=float,
    )
    reference_peaks = np.max(np.abs(reference), axis=0)

    predictions: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for point in domain:
        measured = traces.get(point.candidate_id)
        if measured is None:
            measured = _measured_trace(
                backend, point, subject_id=subject_id, family=family,
                seed=seed, noise_std=noise_std, joint_scale=joint_scale,
            )
            traces[point.candidate_id] = measured
        frame = _point_frame(point, measured, episode_id=f"score:{point.candidate_id}")
        pred_hip, pred_knee = predict_joint_torque(frame, template, fit.parameters, L1)
        pred_tau = np.column_stack((pred_hip, pred_knee))
        predicted_metrics = response_metrics(pred_tau, point, phases, reference_components, reference_peaks)
        true_metrics = response_metrics(measured, point, phases, reference_components, reference_peaks)
        pred_feasible = is_feasible(predicted_metrics, .01)
        true_feasible = is_feasible(true_metrics, .01)
        pred_row = {
            "candidate_id": point.candidate_id,
            "candidate_index": point.candidate_index,
            "pred_E3": float(predicted_metrics["E3"]),
            "pred_E2": float(predicted_metrics["E2"]),
            "pred_peak_ratio": float(predicted_metrics["peak_ratio"]),
            "true_E3": float(true_metrics["E3"]),
            "true_E2": float(true_metrics["E2"]),
            "true_peak_ratio": float(true_metrics["peak_ratio"]),
            "pred_feasible": bool(pred_feasible),
            "true_feasible": bool(true_feasible),
            "torque_rmse_hip_nm": float(np.sqrt(np.mean((pred_tau[:, 0] - measured[:, 0]) ** 2))),
            "torque_rmse_knee_nm": float(np.sqrt(np.mean((pred_tau[:, 1] - measured[:, 1]) ** 2))),
            "torque_rmse_combined_nm": float(np.sqrt(np.mean((pred_tau - measured) ** 2))),
        }
        all_rows.append(pred_row)
        if point.candidate_id not in prefix_ids:
            predictions.append(pred_row)
    pred_frame = pd.DataFrame(predictions)
    all_frame = pd.DataFrame(all_rows)
    if pred_frame.empty:
        raise ValueError("GRAYBOX_NO_HELDOUT_CANDIDATES")
    pred_frame["pred_rank"] = pred_frame["pred_E3"].rank(method="first").astype(int)
    pred_frame["true_rank"] = pred_frame["true_E3"].rank(method="first").astype(int)
    pred_frame["rank_error"] = pred_frame["pred_rank"] - pred_frame["true_rank"]
    if len(pred_frame) > 1 and pred_frame["pred_E3"].nunique() > 1 and pred_frame["true_E3"].nunique() > 1:
        rank_corr = float(spearmanr(pred_frame["pred_E3"], pred_frame["true_E3"]).statistic)
    else:
        rank_corr = float("nan")
    candidate_pool = all_frame.copy()
    predicted_eligible = candidate_pool[candidate_pool["pred_feasible"]]
    predicted_best = (predicted_eligible.sort_values(["pred_E3", "candidate_index"], kind="stable").iloc[0]
                      if not predicted_eligible.empty else candidate_pool.sort_values(["pred_E3", "candidate_index"], kind="stable").iloc[0])
    true_eligible = candidate_pool[candidate_pool["true_feasible"]]
    true_best = (true_eligible.sort_values(["true_E3", "candidate_index"], kind="stable").iloc[0]
                 if not true_eligible.empty else candidate_pool.sort_values(["true_E3", "candidate_index"], kind="stable").iloc[0])
    pred_score = float(predicted_best["true_E3"] if bool(predicted_best["true_feasible"]) else max(1., predicted_best["true_E3"]))
    true_score = float(true_best["true_E3"] if bool(true_best["true_feasible"]) else max(1., true_best["true_E3"]))
    heldout = pred_frame
    summary = {
        "subject_id": subject_id, "family": family, "method": method,
        "seed": int(seed), "noise_std": float(noise_std), "budget": int(budget),
        "prefix_size": len(prefix_ids), "heldout_count": len(heldout),
        "fit_success": fit.success, "fit_status": fit.status,
        "fit_cost": fit.cost, "train_torque_rmse_nm": fit.train_torque_rmse_nm,
        "train_samples": fit.train_samples,
        "boundary_hits": ";".join(fit.boundary_hits),
        "mismatch_flag": ("BOUNDARY_LIMITED" if fit.boundary_hits else
                           "HIGH_TORQUE_MISMATCH" if float(heldout["torque_rmse_combined_nm"].mean()) > 3. else "OK"),
        "heldout_torque_rmse_mean_nm": float(heldout["torque_rmse_combined_nm"].mean()),
        "heldout_E3_abs_error_mean": float(np.abs(heldout["pred_E3"] - heldout["true_E3"]).mean()),
        "heldout_E2_abs_error_mean": float(np.abs(heldout["pred_E2"] - heldout["true_E2"]).mean()),
        "heldout_peak_abs_error_mean": float(np.abs(heldout["pred_peak_ratio"] - heldout["true_peak_ratio"]).mean()),
        "rank_spearman": rank_corr,
        "predicted_best_candidate": str(predicted_best["candidate_id"]),
        "predicted_best_true_E3": float(predicted_best["true_E3"]),
        "true_oracle_candidate": str(true_best["candidate_id"]),
        "true_oracle_E3": float(true_best["true_E3"]),
        "top_regret": float(max(0., pred_score - true_score)),
        "predicted_best_true_feasible": bool(predicted_best["true_feasible"]),
        "predicted_best_pred_feasible": bool(predicted_best["pred_feasible"]),
        "optimizer_message": fit.optimizer_message,
    }
    pred_frame.insert(0, "subject_id", subject_id)
    pred_frame.insert(1, "family", family)
    pred_frame.insert(2, "method", method)
    pred_frame.insert(3, "seed", int(seed))
    pred_frame.insert(4, "noise_std", float(noise_std))
    pred_frame.insert(5, "budget", int(budget))
    pred_frame.insert(6, "prefix_size", len(prefix_ids))
    pred_frame["abs_E3_error"] = np.abs(pred_frame["pred_E3"] - pred_frame["true_E3"])
    pred_frame["abs_E2_error"] = np.abs(pred_frame["pred_E2"] - pred_frame["true_E2"])
    pred_frame["abs_peak_error"] = np.abs(pred_frame["pred_peak_ratio"] - pred_frame["true_peak_ratio"])
    return pred_frame, summary


def _expand_subjects(values: Sequence[str]) -> list[str]:
    allowed = set(development_ids())
    result: list[str] = []
    for value in values:
        if value == "development":
            result.extend(sorted(allowed))
        elif value == "native":
            result.append(NOMINAL)
        elif value in allowed:
            result.append(value)
        else:
            raise ValueError("ONLY_NATIVE_AND_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED")
    return list(dict.fromkeys(result))


def run_evaluator(
    executed_results: Path,
    output_dir: Path,
    *,
    subjects: Sequence[str],
    families: Sequence[str] = FAMILIES,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budgets: Sequence[int] = (1, 2, 4, 8),
    noise_levels: Sequence[float] = (0.,),
    cache_dir: Path = ROOT / ".cache/myoleg-benchmark-v1",
    max_groups: int | None = None,
) -> dict[str, object]:
    """Run the evaluator from an existing frozen V1 trial history."""
    history_path = Path(executed_results) / "trial_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)
    required = {"subject_id", "family", "noise_std", "seed", "method", "trial", "candidate_id", "measurement_valid"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"EXECUTED_HISTORY_MISSING_COLUMNS: {sorted(missing)}")
    subject_list = _expand_subjects(subjects)
    family_list = list(families)
    method_set = set(methods) if methods is not None else set(history["method"].astype(str))
    seed_set = set(int(x) for x in seeds) if seeds is not None else None
    budget_set = sorted(set(int(x) for x in budgets))
    noise_set = {float(x) for x in noise_levels}
    selected = history[
        history["subject_id"].astype(str).isin(subject_list)
        & history["family"].astype(str).isin(family_list)
        & history["method"].astype(str).isin(method_set)
        & history["noise_std"].astype(float).isin(noise_set)
    ].copy()
    if seed_set is not None:
        selected = selected[selected["seed"].astype(int).isin(seed_set)]
    groups = list(selected.groupby(["subject_id", "family", "noise_std", "method", "seed"], sort=True))
    if max_groups is not None:
        groups = groups[:int(max_groups)]
    output_dir.mkdir(parents=True, exist_ok=False)
    domain_cache: dict[str, Domain] = {}
    backend_cache: dict[tuple[str, str], SimulatorBackend] = {}
    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for (subject_id, family, noise_std, method, seed), frame in groups:
        if family not in FAMILIES:
            continue
        if family not in domain_cache:
            domain_cache[family] = Domain(native_domain(), family)
        domain = domain_cache[family]
        key = (str(subject_id), str(family))
        if key not in backend_cache:
            backend_cache[key] = SimulatorBackend(str(subject_id), domain, cache_dir)
        ordered = frame.sort_values("trial", kind="stable")
        for budget in budget_set:
            prefix = ordered[ordered["trial"].astype(int) <= budget]
            if len(prefix) < budget or not bool(prefix["measurement_valid"].astype(bool).all()):
                skipped.append({"subject_id": subject_id, "family": family, "method": method, "seed": int(seed), "budget": budget, "reason": "INCOMPLETE_OR_INVALID_PREFIX"})
                continue
            ids = [str(x) for x in prefix["candidate_id"].tolist()]
            try:
                predictions, summary = evaluate_prefix(
                    domain, backend_cache[key], subject_id=str(subject_id), family=str(family),
                    method=str(method), seed=int(seed), noise_std=float(noise_std),
                    prefix_candidate_ids=ids, budget=budget,
                )
            except Exception as error:
                skipped.append({"subject_id": subject_id, "family": family, "method": method, "seed": int(seed), "budget": budget, "reason": f"{type(error).__name__}: {error}"})
                continue
            summary_rows.append(summary)
            prediction_frames.append(predictions)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False) if prediction_frames else pd.DataFrame().to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")
    protocol = {
        "experiment_id": "MYOLEG_GRAYBOX_HELDOUT_CV_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executed_results": str(Path(executed_results).resolve()),
        "subjects": subject_list, "families": family_list,
        "methods": sorted(method_set), "seeds": sorted(seed_set) if seed_set is not None else "all",
        "budgets": budget_set, "noise_levels": sorted(noise_set),
        "causal_rule": "fit uses candidate IDs with trial <= budget; held-out candidates are never used for fitting",
        "sealed_access": "disabled; native and frozen development IDs only",
        "model": "mass_scale, k_hip, k_knee, b_hip, b_knee; existing parameter_estimator bounds",
        "constraints": {"E2_max": 1.01, "peak_ratio_max": 1.1},
        "rows": {"summary": len(summary_rows), "predictions": int(sum(len(x) for x in prediction_frames)), "skipped": len(skipped)},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# MyoLeg gray-box held-out validation",
        "",
        "This evaluator fits the five-parameter model from causal V1 prefixes and scores unexecuted candidates.",
        "It is a diagnostic and does not modify the frozen V1 artifacts.",
        "",
        f"- summary rows: {len(summary_rows)}",
        f"- prediction rows: {sum(len(x) for x in prediction_frames)}",
        f"- skipped prefixes: {len(skipped)}",
        "",
        "See `summary.csv` for boundary/mismatch flags, ranking correlation and top regret; `predictions.csv` contains torque and endpoint errors.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed-results", type=Path, default=DEFAULT_EXECUTED_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["development"])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.])
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/myoleg-benchmark-v1")
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if any(int(x) < 1 or int(x) > 8 for x in args.budgets):
        parser.error("budgets must be in 1..8")
    run_evaluator(
        args.executed_results, args.output_dir, subjects=args.subjects,
        families=args.families, methods=args.methods, seeds=args.seeds,
        budgets=args.budgets, noise_levels=args.noise_levels,
        cache_dir=args.cache_dir, max_groups=args.max_groups,
    )
    print(f"COMPLETE: {args.output_dir}")


if __name__ == "__main__":
    main()
