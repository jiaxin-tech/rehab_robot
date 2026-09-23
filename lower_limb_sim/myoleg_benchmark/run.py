"""Run a separately recorded MyoLeg development benchmark, never sealed subjects.

python -m lower_limb_sim.myoleg_benchmark.run --subjects native development --workers 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import traceback

import numpy as np
import pandas as pd

from lower_limb_sim.mechanical_endpoints import branch_rms_components
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain
from .experiment import Domain, FAMILIES, METHODS, response_metrics, is_feasible, run_sequence
from .simulation import SimulatorBackend, NOMINAL, ROOT, development_ids, file_sha

_DOMAINS = {}
_BACKENDS = {}


def _domain(family):
    if family not in _DOMAINS:
        _DOMAINS[family] = Domain(native_domain(), family)
    return _DOMAINS[family]


def _backend(subject_id, family, cache_dir):
    key = (subject_id, family, cache_dir)
    if key not in _BACKENDS:
        _BACKENDS[key] = SimulatorBackend(subject_id, _domain(family), cache_dir)
    return _BACKENDS[key]


def _evaluate_results(run, domain, backend, identity, budgets, tier):
    # Called only after selection ends. No result here is passed back to learner.
    phases = domain.subject_reference.phases
    reference_failed = not run["rows"][0]["measurement_valid"]
    if not reference_failed:
        ref = backend.requested(domain.reference)
        components = np.array(branch_rms_components(*ref.T, domain.reference.time_s, phases))
        peaks = np.max(np.abs(ref), axis=0)
    evaluated = []
    for budget in budgets:
        prefix = run["rows"][:budget]
        last = prefix[-1]
        complete = not reference_failed and (identity["method"] == "REFERENCE" or len(prefix) == budget)
        status = "complete" if complete else (run["failure"] or "incomplete")
        selected = last["recommendation_id"]
        metrics = dict(E3=np.nan, E2=np.nan, peak_ratio=np.nan)
        feasible = False
        component_names = ("hip_flex", "hip_extend", "knee_flex", "knee_extend")
        raw_components = np.full(4, np.nan)
        if selected is not None and not reference_failed:
            point = domain.by_id(selected)
            selected_tau = backend.requested(point)
            metrics = response_metrics(selected_tau, point, phases, components, peaks)
            raw_components = branch_rms_components(*selected_tau.T, point.time_s, phases)
            feasible = is_feasible(metrics, tier)
        if not complete or selected is None:
            loss = 1.
        else:
            loss = metrics["E3"] if feasible else max(1., metrics["E3"])
        evaluated.append(dict(
            **identity, budget=budget, executed_trials=len(prefix), status=status,
            recommendation_id=selected, true_E3=metrics["E3"], true_E2=metrics["E2"],
            true_peak_ratio=metrics["peak_ratio"], true_feasible=bool(feasible and complete),
            **{f"true_{name}_rms_nm": float(value) for name, value in zip(component_names, raw_components)},
            **{f"reference_{name}_rms_nm": float(value) for name, value in zip(
                component_names, components if not reference_failed else np.full(4, np.nan))},
            selection_loss=loss, improvement_pct=100 * (1 - loss),
            constraint_violations=sum(r["measurement_valid"] and not r["observed_feasible"] for r in prefix),
            invalid_trials=sum(not r["measurement_valid"] for r in prefix),
            elapsed_s=run["elapsed_s"],
        ))
    return evaluated


def _job(job):
    subject, family, noise, seed, methods, budgets, tier, cache_dir = job
    domain = _domain(family)
    backend = _backend(subject, family, cache_dir)
    results, histories, diagnostics = [], [], []
    fresh_before, disk_before = backend.fresh_simulations, backend.disk_hits
    for method in methods:
        identity = dict(subject_id=subject, family=family, noise_std=noise, seed=seed, method=method)
        outcome = run_sequence(domain, backend, subject_id=subject, method=method,
                               budget=max(budgets), noise_std=noise, seed=seed, tier=tier)
        results.extend(_evaluate_results(outcome, domain, backend, identity, budgets, tier))
        histories.extend(dict(**identity, **r) for r in outcome["rows"])
        diagnostics.extend(dict(**identity, **r) for r in outcome["diagnostics"])
    return dict(results=results, histories=histories, diagnostics=diagnostics,
                provenance=backend.provenance, domain_size=len(domain),
                kinematic_rejections=domain.rejections,
                cache_diagnostics=dict(fresh_simulations=backend.fresh_simulations-fresh_before,
                                       disk_hits=backend.disk_hits-disk_before,
                                       max_decomposition_residual_nm=backend.max_decomposition_residual_nm))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["native"], help="native, development, or explicit development IDs")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--noise-seeds", type=int, default=10)
    parser.add_argument("--noise-levels", nargs="*", type=float, default=[.01, .03])
    parser.add_argument("--noise-subjects", choices=["native", "all"], default="native")
    parser.add_argument("--tier", type=float, default=.01)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/myoleg-benchmark-v1")
    args = parser.parse_args(argv)
    allowed = development_ids()
    subjects = []
    for name in args.subjects:
        subjects.extend([NOMINAL] if name == "native" else allowed if name == "development" else [name])
    subjects = list(dict.fromkeys(subjects))
    if any(s != NOMINAL and s not in allowed for s in subjects):
        parser.error("only native and frozen development subjects are allowed; sealed subjects are disabled")
    budgets = sorted(set(args.budgets))
    if min(budgets) < 1 or max(budgets) > 8 or min(args.random_seeds, args.noise_seeds, args.workers) < 1:
        parser.error("budgets must be 1..8; seed and worker counts must be positive")
    if not np.isfinite(args.tier) or args.tier < 0 or any(not np.isfinite(n) or n <= 0 for n in args.noise_levels):
        parser.error("tier must be finite nonnegative; noise levels must be positive finite")
    output = args.output_dir or ROOT / "outputs/myoleg_benchmark_v1" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    tracked_sources = sorted(Path(__file__).parent.glob("*.py"))
    protocol = dict(
        experiment_id="MYOLEG_DEVELOPMENT_BENCHMARK_V1", created_utc=datetime.now(timezone.utc).isoformat(),
        subjects=subjects, families=args.families, methods=args.methods, budgets=budgets, primary_budget=4,
        primary_noise_std=0., noise_levels=args.noise_levels, noise_subjects=args.noise_subjects,
        seeds=dict(random=list(range(args.random_seeds)), paired_noise=list(range(args.noise_seeds))),
        endpoint="E3 mean of four same-subject reference-normalized branch RMS ratios",
        constraints=dict(E2_max=1+args.tier, joint_peak_ratio_max=1.1, kinematic_speed_ratio_max=1.5, kinematic_acceleration_ratio_max=2),
        noise="Independent additive Gaussian torque-sample noise, std=level times noiseless reference joint RMS. Same trace for subject/family/candidate/seed across methods. Reference itself also noisy; q/dq/ddq exact. Not a validated sensor model.",
        uncertainty="Diagonal delta-method approximation level*sqrt(2/N); shared reference denominator correlation is not modeled",
        selection="All valid requested responses including load violations train models; incumbent/final recommendation only observed feasible, executed candidates. No truth filtering.",
        selection_loss="True E3 for a truly feasible recommendation; max(1,true E3) if infeasible. Incomplete/failed/no-recommendation runs score 1 and are flagged separately. This is a declared failure penalty, not a simulated fallback.",
        development_ranking="K=4, noise=0: true-feasible recommendation rate first, then subject-equal mean selection_loss. Paired intervals within 1e-12 of zero do not establish unique superiority; this is numerical tolerance, not a practical-effect threshold.",
        infrastructure_failures="Job exceptions are recorded immediately and other jobs are collected; any missing job marks completion false and prevents report publication. Never impute a physical outcome for an I/O failure.",
        algorithm_change="New versioned harness: infeasible measurements retained; residual GP mean greedy added; 3D space filling; noise passed to GP. Frozen old results unchanged.",
        kernel=dict(name="fixed Matern52", length_scale=.7, signal_std=.6, feature_scales=[.03,.03,.05], xi=0.),
        comparison_scope="development only; native nominal separated from 24-subject cohort; no sealed truth or oracle access; no tuning on final results",
        truth="Fresh prescribed-state MyoLeg generalized required-drive torque with private content-addressed cache; not forward tracking or real cuff force",
        python=platform.python_version(), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        dependency_lock_sha256=file_sha(ROOT / "requirements-win-py312.lock.txt"),
        elapsed_s_semantics="Full max-budget run wall time including simulation/cache; repeated on budget-prefix rows; NOT comparable algorithm compute time",
        source_sha256={str(p.relative_to(ROOT)): file_sha(p) for p in tracked_sources},
        cache_dir=str(args.cache_dir.resolve()), workers=args.workers,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    jobs = []
    for subject in subjects:
        for family in args.families:
            for seed in range(args.random_seeds):
                methods = args.methods if seed == 0 else [m for m in args.methods if m == "RANDOM"]
                if methods:
                    jobs.append((subject, family, 0., seed, methods, budgets, args.tier, str(args.cache_dir.resolve())))
            if args.noise_subjects == "all" or subject == NOMINAL:
                for noise in args.noise_levels:
                    for seed in range(args.noise_seeds):
                        jobs.append((subject, family, noise, seed, args.methods, budgets, args.tier, str(args.cache_dir.resolve())))
    print(json.dumps(dict(output=str(output), jobs=len(jobs), subjects=len(subjects)), ensure_ascii=False), flush=True)
    all_results, all_histories, all_diagnostics, provenance = [], [], [], {}
    failed_jobs = []

    def record_failure(job, error):
        subject, family, noise, seed, methods, _, _, _ = job
        failed_jobs.append(dict(subject_id=subject, family=family, noise_std=noise,
                                seed=seed, methods=methods, error_type=type(error).__name__,
                                error=str(error), traceback=traceback.format_exc()))
        (output / "job_errors.json").write_text(json.dumps(failed_jobs, indent=2), encoding="utf-8")
        print(f"JOB FAILED: {subject}:{family}:noise={noise}:seed={seed}: {type(error).__name__}: {error}", flush=True)

    def record(result, completed):
        all_results.extend(result["results"])
        all_histories.extend(result["histories"])
        all_diagnostics.extend(result["diagnostics"])
        key = result["results"][0]["subject_id"] + ":" + result["results"][0]["family"]
        old_counts = provenance.get(key, {}).get("cache_diagnostics", {})
        provenance[key] = {k: result[k] for k in ("provenance", "domain_size", "kinematic_rejections", "cache_diagnostics")}
        for counter in ("fresh_simulations", "disk_hits"):
            provenance[key]["cache_diagnostics"][counter] += old_counts.get(counter, 0)
        provenance[key]["cache_diagnostics"]["max_decomposition_residual_nm"] = max(
            old_counts.get("max_decomposition_residual_nm", 0), result["cache_diagnostics"]["max_decomposition_residual_nm"])
        pd.DataFrame(all_results).to_csv(output / "results.csv", index=False)
        pd.DataFrame(all_histories).to_csv(output / "trial_history.csv", index=False)
        (output / "identification_diagnostics.json").write_text(json.dumps(all_diagnostics, indent=2, default=str), encoding="utf-8")
        (output / "simulation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"completed {completed}/{len(jobs)}: {key}; results={len(all_results)}", flush=True)

    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            try:
                result = _job(job)
            except Exception as error:
                record_failure(job, error)
                continue
            record(result, i)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_jobs = {pool.submit(_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_jobs), 1):
                try:
                    result = future.result()
                except Exception as error:
                    record_failure(future_jobs[future], error)
                    continue
                record(result, i)
    summary = dict(completed=not failed_jobs, jobs=len(jobs), failed_jobs=len(failed_jobs),
                   result_rows=len(all_results), trial_rows=len(all_histories),
                   protocol_sha256=file_sha(output / "protocol.json"),
                   source_sha256=protocol["source_sha256"])
    (output / "completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failed_jobs:
        raise RuntimeError(f"INCOMPLETE_BENCHMARK: {len(failed_jobs)} job(s) failed; see {output / 'job_errors.json'}")
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
