"""Development-only full-domain personalization identifiability audit.

This evaluator is deliberately independent from the benchmark learner.  It
replays every kinematically valid candidate for native/development subjects at
zero measurement noise, then joins those evaluator-only values to an existing
benchmark's executed trial pool.  It never accepts sealed subject IDs and it
never exposes the full landscape to a learner.

Example::

    python -m lower_limb_sim.myoleg_benchmark.personalization_audit \
      --subjects development \
      --executed-results outputs/myoleg_benchmark_v1/development_20260923_r2 \
      --output-dir outputs/myoleg_personalization_audit_v1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from lower_limb_sim.mechanical_endpoints import branch_rms_components
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain

from .experiment import Domain, FAMILIES, is_feasible, response_metrics
from .simulation import SimulatorBackend, NOMINAL, ROOT, development_ids, file_sha


TIE_TOLERANCE = 1e-12
NEAR_ORACLE_EPSILONS = (1e-4, 5e-4, 1e-3)


def _resolve_subjects(names: list[str]) -> list[str]:
    """Expand only native/development selectors; reject sealed IDs early."""
    allowed = set(development_ids())
    subjects: list[str] = []
    for name in names:
        if name == "native":
            subjects.append(NOMINAL)
        elif name == "development":
            subjects.extend(sorted(allowed))
        elif name in allowed:
            subjects.append(name)
        else:
            raise ValueError(
                "SEALED_OR_UNKNOWN_SUBJECT_REJECTED: "
                f"{name!r}; only native and frozen development subjects are allowed"
            )
    return list(dict.fromkeys(subjects))


def _domain(family: str) -> Domain:
    return Domain(native_domain(), family)


def _landscape_job(job: tuple[str, str, str, float]) -> dict[str, Any]:
    """Evaluate one subject/family full domain in a child process."""
    subject_id, family, cache_dir, tier = job
    domain = _domain(family)
    backend = SimulatorBackend(subject_id, domain, cache_dir)
    phases = domain.subject_reference.phases
    reference = backend.requested(domain.reference)
    reference_components = np.asarray(
        branch_rms_components(*reference.T, domain.reference.time_s, phases), dtype=float
    )
    reference_peaks = np.max(np.abs(reference), axis=0)
    scope = "NATIVE" if subject_id == NOMINAL else "DEVELOPMENT"
    rows: list[dict[str, Any]] = []
    for point in domain:
        tau = backend.requested(point)
        metrics = response_metrics(
            tau, point, phases, reference_components, reference_peaks
        )
        component_names = ("hip_flex", "hip_extend", "knee_flex", "knee_extend")
        components = branch_rms_components(*tau.T, point.time_s, phases)
        rows.append(
            {
                "scope": scope,
                "subject_id": subject_id,
                "family": family,
                "candidate_id": point.candidate_id,
                "candidate_index": point.candidate_index,
                "beta_flex": point.beta_flex,
                "beta_extend": point.beta_extend,
                "share_shift": point.share_shift,
                "E3": metrics["E3"],
                "E2": metrics["E2"],
                "peak_ratio": metrics["peak_ratio"],
                "feasible": is_feasible(metrics, tier),
                **{
                    f"{name}_rms_nm": float(value)
                    for name, value in zip(component_names, components)
                },
            }
        )
    return {
        "subject_id": subject_id,
        "family": family,
        "rows": rows,
        "domain_size": len(domain),
        "kinematic_rejections": domain.rejections,
        "provenance": backend.provenance,
        "cache_diagnostics": {
            "fresh_simulations": backend.fresh_simulations,
            "disk_hits": backend.disk_hits,
            "max_decomposition_residual_nm": backend.max_decomposition_residual_nm,
        },
    }


def _oracle_table(landscape: pd.DataFrame, tier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-subject oracle summaries and development common baselines."""
    if landscape.empty:
        return pd.DataFrame(), pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for (scope, family), frame in landscape.groupby(["scope", "family"], sort=True):
        # Native is reported as a diagnostic only and never participates in
        # development common-trajectory selection.
        dev = frame[frame["scope"] == "DEVELOPMENT"]
        if not dev.empty:
            # A common recommendation must pay the same explicit infeasibility
            # penalty used by the benchmark scorer.  Sorting raw E3 here could
            # otherwise select a mechanically infeasible candidate for one or
            # more subjects and make the population baseline look artificially
            # strong.
            dev = dev.copy()
            dev["selection_score"] = np.where(
                dev["feasible"].astype(bool), dev["E3"], np.maximum(1.0, dev["E3"])
            )
            by_candidate = dev.groupby("candidate_id", sort=False)
            common = by_candidate["selection_score"].mean().sort_values(kind="stable")
            worst = by_candidate["selection_score"].max().sort_values(kind="stable")
            feasible_counts = by_candidate["feasible"].sum()
            common_candidate = str(common.index[0])
            worst_candidate = str(worst.index[0])
            common_all_feasible = bool(feasible_counts.get(common_candidate, 0) == len(dev["subject_id"].unique()))
            common_rows.append(
                {
                    "scope": "DEVELOPMENT",
                    "family": family,
                    "development_subject_count": int(dev["subject_id"].nunique()),
                    "common_candidate_id": common_candidate,
                    "common_mean_selection_score": float(common.iloc[0]),
                    "worst_case_candidate_id": worst_candidate,
                    "worst_case_selection_score": float(worst.iloc[0]),
                    "common_all_subjects_feasible": common_all_feasible,
                    "common_feasible_intersection_count": int((feasible_counts == len(dev["subject_id"].unique())).sum()),
                    "selection_rule": "mean benchmark selection score over development subjects; evaluator-only",
                }
            )
        else:
            common_candidate = None
            worst_candidate = None
        for subject_id, subject in frame.groupby("subject_id", sort=True):
            feasible = subject[subject["feasible"]].sort_values(["E3", "candidate_index"], kind="stable")
            if feasible.empty:
                oracle = subject.sort_values(["E3", "candidate_index"], kind="stable").iloc[0]
                oracle_status = "NO_FEASIBLE_CANDIDATE_PENALIZED"
            else:
                oracle = feasible.iloc[0]
                oracle_status = "FEASIBLE_ORACLE"
            oracle_e3 = float(oracle["E3"])
            oracle_score = float(oracle_e3 if oracle["feasible"] else max(1.0, oracle_e3))
            eligible = feasible
            tie_count = int(np.sum(np.abs(eligible["E3"].to_numpy() - oracle_e3) <= TIE_TOLERANCE)) if not eligible.empty else 0
            record: dict[str, Any] = {
                "scope": scope,
                "subject_id": subject_id,
                "family": family,
                "domain_size": int(len(subject)),
                "feasible_count": int(subject["feasible"].sum()),
                "oracle_candidate_id": str(oracle["candidate_id"]),
                "oracle_beta_flex": float(oracle["beta_flex"]),
                "oracle_beta_extend": float(oracle["beta_extend"]),
                "oracle_share_shift": float(oracle["share_shift"]),
                "oracle_E3": oracle_e3,
                "oracle_selection_score": oracle_score,
                "oracle_E2": float(oracle["E2"]),
                "oracle_peak_ratio": float(oracle["peak_ratio"]),
                "oracle_status": oracle_status,
                "tie_equivalent_count": tie_count,
            }
            for epsilon in NEAR_ORACLE_EPSILONS:
                near = eligible[eligible["E3"] <= oracle_e3 * (1.0 + epsilon)] if not eligible.empty else eligible
                record[f"near_oracle_count_eps_{epsilon:g}"] = int(len(near))
                record[f"near_oracle_fraction_eps_{epsilon:g}"] = float(len(near) / len(eligible)) if not eligible.empty else np.nan
            if common_candidate is not None and scope == "DEVELOPMENT":
                common = subject[subject["candidate_id"] == common_candidate]
                if len(common) == 1:
                    common_row = common.iloc[0]
                    common_score = float(common_row["E3"] if common_row["feasible"] else max(1.0, common_row["E3"]))
                    record.update(
                        common_candidate_id=common_candidate,
                        common_E3=float(common_row["E3"]),
                        common_feasible=bool(common_row["feasible"]),
                        common_selection_score=common_score,
                        relative_common_regret=(
                            float(common_score - oracle_score) / common_score
                            if common_score != 0 else np.nan
                        ),
                    )
            summaries.append(record)
    return pd.DataFrame(summaries), pd.DataFrame(common_rows)


def _candidate_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    # Normalize by the frozen candidate parameter spans: beta -0.12..0.12,
    # timing share -0.10..0.10.  This is geometry, never an outcome weight.
    scales = np.asarray([0.24, 0.24, 0.20], dtype=float)
    a = np.asarray([row_a["beta_flex"], row_a["beta_extend"], row_a["share_shift"]], dtype=float)
    b = np.asarray([row_b["beta_flex"], row_b["beta_extend"], row_b["share_shift"]], dtype=float)
    return float(np.sqrt(np.mean(((a - b) / scales) ** 2)))


def _pool_comparison(
    landscape: pd.DataFrame,
    oracle: pd.DataFrame,
    executed_results: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each causal no-noise executed prefix with full-domain truth."""
    columns = [
        "scope", "subject_id", "family", "method", "seed", "budget", "status",
        "executed_count", "domain_size", "coverage_fraction", "recommendation_id",
        "recommendation_E3", "recommendation_feasible", "full_oracle_id", "full_oracle_E3",
        "recommendation_selection_score",
        "pool_oracle_id", "pool_oracle_E3", "pool_has_full_oracle", "pool_has_near_oracle",
        "exploration_regret_relative_pct", "recommendation_regret_relative_pct",
        "distance_to_full_oracle", "no_feasible_candidate_in_pool",
    ]
    if executed_results is None:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    history_path = executed_results / "trial_history.csv"
    results_path = executed_results / "results.csv"
    if not history_path.exists() or not results_path.exists():
        raise FileNotFoundError("executed-results requires trial_history.csv and results.csv")
    history = pd.read_csv(history_path)
    results = pd.read_csv(results_path)
    for frame in (history, results):
        if "noise_std" not in frame.columns:
            raise ValueError("EXECUTED_RESULTS_MISSING_NOISE_COLUMN")
    # The frozen benchmark predates the explicit scope column.  Derive it from
    # the already validated subject IDs instead of changing or re-running V1.
    # This keeps the join compatible with both the old artifact and future
    # reports that may persist scope directly.
    allowed_development = set(development_ids())
    for frame in (history, results):
        derived_scope = np.where(
            frame["subject_id"].astype(str).eq(NOMINAL), "NATIVE",
            np.where(frame["subject_id"].astype(str).isin(allowed_development),
                     "DEVELOPMENT", "UNKNOWN"),
        )
        if (derived_scope == "UNKNOWN").any():
            raise ValueError("EXECUTED_RESULTS_SCOPE_CONTAINS_UNKNOWN_SUBJECT")
        if "scope" in frame.columns and not np.array_equal(frame["scope"].astype(str).to_numpy(), derived_scope):
            raise ValueError("EXECUTED_RESULTS_SCOPE_ID_MISMATCH")
        frame["scope"] = derived_scope
    history = history[np.isclose(history["noise_std"].astype(float), 0.0)]
    results = results[np.isclose(results["noise_std"].astype(float), 0.0)]
    if history.empty or results.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    oracle_idx = oracle.set_index(["subject_id", "family"], drop=False)
    out: list[dict[str, Any]] = []
    for key, result in results.groupby(["scope", "subject_id", "family", "method", "seed"], sort=True):
        scope, subject_id, family, method, seed = key
        if (subject_id, family) not in oracle_idx.index:
            continue
        candidates = history[
            (history["scope"] == scope)
            & (history["subject_id"] == subject_id)
            & (history["family"] == family)
            & (history["method"] == method)
            & (history["seed"] == seed)
        ]
        if candidates.empty:
            continue
        for _, result_row in result.sort_values("budget").iterrows():
            budget = int(result_row["budget"])
            prefix = candidates[candidates["trial"].astype(int) <= budget]
            ids = [str(x) for x in prefix["candidate_id"].dropna().unique() if str(x) != "nan"]
            subject_truth = landscape[(landscape["subject_id"] == subject_id) & (landscape["family"] == family)]
            pool = subject_truth[subject_truth["candidate_id"].isin(ids)]
            feasible_pool = pool[pool["feasible"]].sort_values(["E3", "candidate_index"], kind="stable")
            full = oracle_idx.loc[(subject_id, family)]
            selected = str(result_row["recommendation_id"]) if pd.notna(result_row["recommendation_id"]) else ""
            if selected and selected not in ids:
                raise ValueError("RECOMMENDATION_NOT_IN_EXECUTED_PREFIX")
            selected_rows = subject_truth[subject_truth["candidate_id"] == selected]
            selected_row = selected_rows.iloc[0] if len(selected_rows) else None
            pool_row = feasible_pool.iloc[0] if len(feasible_pool) else None
            full_row = subject_truth[subject_truth["candidate_id"] == full["oracle_candidate_id"]].iloc[0]
            pool_e3 = float(pool_row["E3"]) if pool_row is not None else np.nan
            recommendation_e3 = float(selected_row["E3"]) if selected_row is not None else np.nan
            recommendation_feasible = bool(selected_row is not None and selected_row["feasible"])
            # Match the runner's declared loss, including failed/incomplete
            # runs.  A retained earlier recommendation does not erase failure.
            complete = str(result_row.get("status", "unknown")).lower() == "complete"
            recommendation_score = (float(recommendation_e3 if recommendation_feasible else max(1.0, recommendation_e3))
                                    if complete and np.isfinite(recommendation_e3) else 1.0)
            full_e3 = float(full["oracle_E3"])
            full_score = float(full["oracle_selection_score"])
            # Capture is defined over candidates that could actually be
            # recommended under the frozen load gate; an infeasible point in
            # the executed pool does not count as a near-oracle opportunity.
            pool_has_near = bool((feasible_pool["E3"] <= full_score * (1.0 + 1e-3)).any())
            distance = _candidate_distance(selected_row, full_row) if selected_row is not None else np.nan
            out.append(
                {
                    "scope": scope, "subject_id": subject_id, "family": family,
                    "method": method, "seed": int(seed), "budget": budget,
                    "status": result_row.get("status", "unknown"),
                    "executed_count": int(len(ids)), "domain_size": int(len(subject_truth)),
                    "coverage_fraction": float(len(ids) / len(subject_truth)),
                    "recommendation_id": selected or None,
                    "recommendation_E3": recommendation_e3,
                    "recommendation_feasible": bool(recommendation_feasible and complete),
                    "recommendation_selection_score": recommendation_score,
                    "full_oracle_id": str(full["oracle_candidate_id"]), "full_oracle_E3": full_e3,
                    "pool_oracle_id": str(pool_row["candidate_id"]) if pool_row is not None else None,
                    "pool_oracle_E3": pool_e3,
                    "pool_has_full_oracle": bool(full["oracle_candidate_id"] in ids),
                    "pool_has_near_oracle": pool_has_near,
                    "exploration_regret_relative_pct": (100.0 * (pool_e3 - full_score) / full_score) if np.isfinite(pool_e3) else np.nan,
                    "recommendation_regret_relative_pct": (100.0 * (recommendation_score - full_score) / full_score) if np.isfinite(recommendation_score) else np.nan,
                    "distance_to_full_oracle": distance,
                    "no_feasible_candidate_in_pool": bool(pool_row is None),
                }
            )
    pool_df = pd.DataFrame(out, columns=columns)
    if pool_df.empty:
        return pool_df, pd.DataFrame()
    grouped = []
    for key, frame in pool_df.groupby(["scope", "family", "method", "budget"], sort=True):
        scope, family, method, budget = key
        grouped.append(
            {
                "scope": scope, "family": family, "method": method, "budget": int(budget),
                "runs": int(len(frame)), "subject_count": int(frame["subject_id"].nunique()),
                "mean_exploration_regret_pct": float(frame["exploration_regret_relative_pct"].mean()),
                "median_exploration_regret_pct": float(frame["exploration_regret_relative_pct"].median()),
                "mean_recommendation_regret_pct": float(frame["recommendation_regret_relative_pct"].mean()),
                "median_recommendation_regret_pct": float(frame["recommendation_regret_relative_pct"].median()),
                "pool_full_oracle_capture_rate": float(frame["pool_has_full_oracle"].mean()),
                "pool_near_oracle_capture_rate": float(frame["pool_has_near_oracle"].mean()),
                "recommendation_feasible_rate": float(frame["recommendation_feasible"].mean()),
                "no_feasible_pool_rate": float(frame["no_feasible_candidate_in_pool"].mean()),
                "mean_executed_count": float(frame["executed_count"].mean()),
                "mean_domain_coverage_fraction": float(frame["coverage_fraction"].mean()),
            }
        )
    return pool_df, pd.DataFrame(grouped)


def _interaction_audit(
    landscape: pd.DataFrame, oracle: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize subject-by-candidate interaction and pairwise transfer.

    These are evaluator-only descriptive statistics.  They are never passed
    to a learner and are intentionally computed after the full landscape has
    been generated.
    """
    pairwise: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for family, frame in landscape.groupby("family", sort=True):
        dev = frame[frame["scope"] == "DEVELOPMENT"].copy()
        if dev.empty:
            continue
        pivot = dev.pivot(index="subject_id", columns="candidate_id", values="E3")
        if not np.isfinite(pivot.to_numpy()).all():
            raise ValueError("INTERACTION_REQUIRES_COMPLETE_FINITE_CANDIDATE_MATRIX")
        subject_ids = sorted(pivot.index.astype(str))
        candidate_ids = list(pivot.columns.astype(str))
        # Two-way decomposition: Y_ic = grand + subject_i + candidate_c +
        # interaction_ic.  Raw E3 (including infeasible candidates) describes
        # the response surface; transfer scores separately apply feasibility.
        total = float(np.sum((pivot.to_numpy() - float(pivot.to_numpy().mean())) ** 2))
        candidate_mean = pivot.mean(axis=0)
        subject_mean = pivot.mean(axis=1)
        grand = float(pivot.to_numpy().mean())
        residual = pivot.sub(candidate_mean, axis=1).sub(subject_mean, axis=0) + grand
        interaction_fraction = float(np.sum(residual.to_numpy() ** 2) / total) if total else 0.0
        family_oracle = oracle[(oracle["scope"] == "DEVELOPMENT") & (oracle["family"] == family)]
        oracle_lookup = family_oracle.set_index("subject_id", drop=False)
        scores = pivot.copy()
        for subject_id in subject_ids:
            rows = dev[dev["subject_id"] == subject_id]
            scores.loc[subject_id] = np.where(rows.set_index("candidate_id").loc[candidate_ids]["feasible"].to_numpy(),
                                              rows.set_index("candidate_id").loc[candidate_ids]["E3"].to_numpy(),
                                              np.maximum(1.0, rows.set_index("candidate_id").loc[candidate_ids]["E3"].to_numpy()))
        for i, source_id in enumerate(subject_ids):
            source_oracle = str(oracle_lookup.loc[source_id, "oracle_candidate_id"])
            source_row = dev[(dev["subject_id"] == source_id) & (dev["candidate_id"] == source_oracle)].iloc[0]
            for target_id in subject_ids[i + 1:]:
                target_oracle = str(oracle_lookup.loc[target_id, "oracle_candidate_id"])
                target_row = dev[(dev["subject_id"] == target_id) & (dev["candidate_id"] == target_oracle)].iloc[0]
                source_in_target = scores.loc[target_id, source_oracle]
                target_score = scores.loc[target_id, target_oracle]
                target_in_source = scores.loc[source_id, target_oracle]
                source_score = scores.loc[source_id, source_oracle]
                pairwise.append({
                    "family": family,
                    "subject_a": source_id,
                    "subject_b": target_id,
                    "spearman_rank": float(pivot.loc[source_id].corr(pivot.loc[target_id], method="spearman")),
                    "oracle_a": source_oracle,
                    "oracle_b": target_oracle,
                    "oracle_same": bool(source_oracle == target_oracle),
                    "oracle_distance": _candidate_distance(source_row, target_row),
                    "transfer_a_to_b_regret_pct": float(100.0 * (source_in_target - target_score) / source_in_target) if source_in_target else 0.0,
                    "transfer_b_to_a_regret_pct": float(100.0 * (target_in_source - source_score) / target_in_source) if target_in_source else 0.0,
                })
        pair = pd.DataFrame([row for row in pairwise if row["family"] == family])
        summaries.append({
            "scope": "DEVELOPMENT",
            "family": family,
            "subject_count": int(len(subject_ids)),
            "candidate_count": int(len(candidate_ids)),
            "unique_oracle_count": int(family_oracle["oracle_candidate_id"].nunique()),
            "interaction_fraction": interaction_fraction,
            "pair_count": int(len(pair)),
            "median_spearman_rank": float(pair["spearman_rank"].median()) if not pair.empty else np.nan,
            "median_transfer_regret_pct": float(pd.concat([pair["transfer_a_to_b_regret_pct"], pair["transfer_b_to_a_regret_pct"]]).median()) if not pair.empty else np.nan,
            "p75_transfer_regret_pct": float(pd.concat([pair["transfer_a_to_b_regret_pct"], pair["transfer_b_to_a_regret_pct"]]).quantile(.75)) if not pair.empty else np.nan,
            "oracle_same_fraction": float(pair["oracle_same"].mean()) if not pair.empty else np.nan,
        })
    return pd.DataFrame(pairwise), pd.DataFrame(summaries)


def _run(args: argparse.Namespace) -> Path:
    subjects = _resolve_subjects(args.subjects)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=False, exist_ok=False)
    cache_dir = str(Path(args.cache_dir).resolve())
    jobs = [(subject, family, cache_dir, float(args.tier)) for subject in subjects for family in args.families]
    protocol = {
        "audit_id": "MYOLEG_PERSONALIZATION_IDENTIFIABILITY_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": subjects,
        "families": args.families,
        "tier": float(args.tier),
        "noise_std": 0.0,
        "candidate_domain_source": "Domain(native_domain(), family), kinematic constraints only",
        "oracle_rule": "minimum feasible float64 E3, candidate_index tie break; no-feasible pool uses max(1,E3) penalty",
        "tie_tolerance": TIE_TOLERANCE,
        "near_oracle_epsilons": list(NEAR_ORACLE_EPSILONS),
        "executed_results": str(Path(args.executed_results).resolve()) if args.executed_results else None,
        "scope": "native diagnostic plus 24 frozen development subjects; sealed subject IDs rejected before model access",
        "sealed_scientific_access_count": 0,
        "code_change_scope": "independent evaluator; learner and benchmark source unchanged",
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            "personalization_audit.py": file_sha(Path(__file__)),
            "experiment.py": file_sha(Path(__file__).with_name("experiment.py")),
            "simulation.py": file_sha(Path(__file__).with_name("simulation.py")),
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    # Keep the access boundary as a separate, easy-to-audit artifact.  The
    # cohort manifest is read only for its development split; sealed model
    # deltas, arrays, and truth landscapes are never opened by this CLI.
    scope_audit = {
        "audit_id": protocol["audit_id"],
        "requested_subject_selectors": args.subjects,
        "resolved_subject_ids": subjects,
        "development_ids_used": sorted(development_ids()),
        "sealed_subject_ids_requested": [],
        "sealed_scientific_access_count": 0,
        "sealed_model_delta_reads": 0,
        "sealed_truth_array_reads": 0,
        "manifest_read_for_split_only": True,
        "learner_received_full_landscape": False,
        "oracle_used_by_learner": False,
        "scope_status": "DEVELOPMENT_ONLY_NO_SEALED_ACCESS",
    }
    (output / "scope_audit.json").write_text(json.dumps(scope_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    if args.workers == 1:
        futures = [(job, None) for job in jobs]
        for completed, (job, _) in enumerate(futures, 1):
            try:
                result = _landscape_job(job)
            except Exception as error:  # pragma: no cover - runtime diagnostic
                failures.append({"job": job, "type": type(error).__name__, "error": str(error)})
                print(f"failed {completed}/{len(jobs)}: {job[0]}:{job[1]}: {type(error).__name__}: {error}", flush=True)
                continue
            all_rows.extend(result["rows"])
            provenance[f"{result['subject_id']}:{result['family']}"] = {k: result[k] for k in ("domain_size", "kinematic_rejections", "provenance", "cache_diagnostics")}
            print(f"completed {completed}/{len(jobs)}: {job[0]}:{job[1]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            submitted = {pool.submit(_landscape_job, job): job for job in jobs}
            for completed, future in enumerate(as_completed(submitted), 1):
                job = submitted[future]
                try:
                    result = future.result()
                except Exception as error:  # pragma: no cover - runtime diagnostic
                    failures.append({"job": job, "type": type(error).__name__, "error": str(error)})
                    print(f"failed {completed}/{len(jobs)}: {job[0]}:{job[1]}: {type(error).__name__}: {error}", flush=True)
                    continue
                all_rows.extend(result["rows"])
                provenance[f"{result['subject_id']}:{result['family']}"] = {k: result[k] for k in ("domain_size", "kinematic_rejections", "provenance", "cache_diagnostics")}
                print(f"completed {completed}/{len(jobs)}: {job[0]}:{job[1]}", flush=True)
    if failures:
        (output / "job_errors.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"FULL_DOMAIN_AUDIT_INCOMPLETE: {len(failures)} job(s) failed")
    landscape = pd.DataFrame(all_rows)
    landscape = landscape.sort_values(["scope", "family", "subject_id", "candidate_index"], kind="stable")
    landscape.to_csv(output / "landscape.csv", index=False)
    oracle, common = _oracle_table(landscape, float(args.tier))
    oracle.to_csv(output / "oracle_summary.csv", index=False)
    common.to_csv(output / "common_summary.csv", index=False)
    pool, pool_summary = _pool_comparison(landscape, oracle, Path(args.executed_results).resolve() if args.executed_results else None)
    pool.to_csv(output / "pool_vs_full.csv", index=False)
    pool_summary.to_csv(output / "pool_vs_full_summary.csv", index=False)
    pairwise, interaction = _interaction_audit(landscape, oracle)
    pairwise.to_csv(output / "pairwise_interaction.csv", index=False)
    interaction.to_csv(output / "interaction_summary.csv", index=False)
    (output / "simulation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    completion = {
        "complete": True,
        "subjects": len(subjects),
        "families": len(args.families),
        "jobs": len(jobs),
        "landscape_rows": int(len(landscape)),
        "oracle_rows": int(len(oracle)),
        "pool_rows": int(len(pool)),
        "pairwise_rows": int(len(pairwise)),
        "sealed_scientific_access_count": 0,
        "scope_audit": str(output / "scope_audit.json"),
        "output": str(output),
    }
    (output / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["development"], help="native, development, or explicit development IDs")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--executed-results", type=Path, help="completed benchmark directory containing results.csv/trial_history.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/myoleg-benchmark-v1")
    parser.add_argument("--tier", type=float, default=.01, help="E2 feasibility slack; peak limit remains 1.10")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not np.isfinite(args.tier) or args.tier < 0:
        parser.error("--tier must be finite and nonnegative")
    try:
        output = _run(args)
    except (FileExistsError, ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"complete": True, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
