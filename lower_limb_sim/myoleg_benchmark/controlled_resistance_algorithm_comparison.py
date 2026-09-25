"""Equal-budget algorithm comparison on the frozen resistance stress field."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .controlled_resistance_cohort import DEFAULT_MANIFEST, load_profiles
from .controlled_resistance_experiment import GRID, PROBES, build_candidates, common_value
from .personalized_v2 import CandidateView, GateConfig, LocalResidualModel, ObservationRecord, SASTBO, run_sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/myoleg_controlled_resistance_algorithm_v1"
METHODS = ("COMMON_POLICY", "RANDOM", "RESIDUAL_BO", "SAST_BO")
SEEDS = (0, 1, 2, 3, 4)


def _observation(profile, candidate, trial):
    return ObservationRecord(candidate_id=candidate.candidate_id, features=candidate.features,
                             value=profile.value(candidate.features), uncertainty=0.0, feasible=True,
                             constraint_values={k: v[0] for k, v in candidate.constraint_predictions.items()},
                             trial_index=trial)


def _truth(profile, candidates):
    values = {c.candidate_id: profile.value(c.features) for c in candidates}
    oracle = min(values, key=lambda x: (values[x], x))
    return values, oracle


def run_profile(profile, method, *, budget=8, seed=0):
    if method not in METHODS or profile.split != "DEVELOPMENT" or budget < 5:
        raise ValueError("RESISTANCE_ALGORITHM_SCOPE_OR_BUDGET")
    candidates = build_candidates()
    by_id = {c.candidate_id: c for c in candidates}
    by_features = {c.features: c for c in candidates}
    reference = by_features[(1.0, 0.5, 0.5)]
    probes = tuple(by_features[x] for x in PROBES)
    if method == "SAST_BO":
        controller = SASTBO(
            candidates, reference_id=reference.candidate_id,
            calibration_probe_ids=tuple(p.candidate_id for p in probes),
            common_policy_id=reference.candidate_id,
            common_predictor=lambda c: common_value(c.features),
            preapproved_candidate_ids=(reference.candidate_id, *(p.candidate_id for p in probes)),
            config=GateConfig(interaction_threshold=0.0025, posterior_threshold=0.95),
        )
        def evaluator(candidate, trial):
            return _observation(profile, candidate, trial)
        sequence = run_sequence(controller, evaluator, budget=budget)
        observations = sequence["observations"]
        recommendation = controller.recommend()
        gate_active = bool(controller.last_gate.active)
        termination = sequence["termination"] or "BUDGET_EXHAUSTED"
    else:
        rng = np.random.default_rng(seed)
        observations = []
        used = set()
        for trial in range(1, budget + 1):
            if trial == 1:
                candidate = reference
            elif trial <= 1 + len(probes):
                candidate = probes[trial - 2]
            elif method == "COMMON_POLICY":
                break
            elif method == "RANDOM":
                available = [c for c in candidates if c.candidate_id not in used]
                candidate = available[int(rng.integers(len(available)))]
            else:
                available = [c for c in candidates if c.candidate_id not in used]
                valid = list(observations)
                model = LocalResidualModel(length_scale=0.7, signal_std=0.6)
                xs = [by_id[o.candidate_id].features for o in valid]
                ys = [o.value - common_value(o.features) for o in valid]
                model.fit(xs, ys, [1e-6] * len(ys))
                candidate = min(available, key=lambda c: (common_value(c.features) + model.predict(c.features)[0], c.candidate_id))
            if candidate.candidate_id in used:
                break
            used.add(candidate.candidate_id)
            observations.append(_observation(profile, candidate, trial))
        recommendation = min(observations, key=lambda o: (o.value, o.trial_index)).candidate_id if observations else None
        gate_active = False
        termination = "BUDGET_EXHAUSTED" if len(observations) == budget else "COMMON_POLICY_FIXED"
    values, oracle = _truth(profile, candidates)
    common = values[reference.candidate_id]
    recommended = None if recommendation is None else values[recommendation]
    return {
        "profile_id": profile.profile_id, "scenario": profile.scenario, "arm": profile.arm,
        "method": method, "seed": seed, "budget": budget,
        "gate_active": gate_active, "recommendation_id": recommendation,
        "recommendation_value": recommended, "common_value": common,
        "oracle_id": oracle, "oracle_value": values[oracle],
        "common_regret": common - values[oracle],
        "policy_regret": None if recommended is None else recommended - values[oracle],
        "improvement_over_common": None if recommended is None else common - recommended,
        "executed_trials": len(observations), "termination": termination,
    }


def run_benchmark(*, output_dir=DEFAULT_OUTPUT, budget=8):
    profiles = load_profiles("DEVELOPMENT")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "experiment_id": "CONTROLLED_RESISTANCE_ALGORITHM_V1",
        "scope": "development-only controlled synthetic resistance mechanism",
        "methods": METHODS, "seeds": SEEDS, "budget": budget,
        "profile_ids": [p.profile_id for p in profiles], "candidate_count": len(build_candidates()),
        "probes": PROBES, "confirmatory_access": False, "oracle_access": "evaluator-only after run",
        "cohort_manifest_sha256": manifest["manifest_fingerprint_sha256"],
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name("controlled_resistance_experiment.py"),
            Path(__file__).with_name("controlled_resistance_cohort.py"), Path(__file__).with_name("personalized_v2.py"))},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = [run_profile(p, method, budget=budget, seed=seed)
            for p in profiles for method in METHODS for seed in (SEEDS if method in ("RANDOM",) else (0,))]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "results.csv", index=False)
    summary = frame.groupby(["scenario", "method"], sort=True).agg(
        runs=("profile_id", "count"), gate_active_rate=("gate_active", "mean"),
        common_regret_mean=("common_regret", "mean"), policy_regret_mean=("policy_regret", "mean"),
        improvement_mean=("improvement_over_common", "mean"), executed_trials_mean=("executed_trials", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "completion.json").write_text(json.dumps({"complete": True, "runs": len(frame), "confirmatory_access": False}, indent=2) + "\n")
    lines = ["# Controlled resistance algorithm comparison", "",
             "Development-only controlled synthetic mechanism test; no native or patient claim.",
             "Random has five seeds; other methods are deterministic under the frozen protocol.", "",
             "| Scenario | Method | Runs | Gate active | Common regret | Policy regret | Improvement | Trials |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['scenario']} | {row['method']} | {row['runs']} | {row['gate_active_rate']:.0%} | "
                     f"{row['common_regret_mean']:.6f} | {row['policy_regret_mean']:.6f} | {row['improvement_mean']:.6f} | {row['executed_trials_mean']:.1f} |")
    lines += ["", "No confirmatory profiles were loaded. Results are software mechanism evidence only."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps({"complete": True, "output": str(run_benchmark(output_dir=args.output_dir, budget=args.budget))}))


if __name__ == "__main__":
    main()
