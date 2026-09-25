"""Equal-budget algorithm comparison on the frozen resistance stress field."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd

from .controlled_resistance_cohort import DEFAULT_MANIFEST, load_profiles
from .controlled_resistance_experiment import (
    GRID,
    PROBES,
    build_candidates,
    common_value,
    constraint_values_for,
)
from .personalized_v2 import (
    CandidateView,
    GateConfig,
    LocalResidualModel,
    ObservationRecord,
    SASTBO,
    run_sequence,
)
from personalization.selectors.bo import expected_improvement


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/myoleg_controlled_resistance_algorithm_v2"
METHODS = ("COMMON_POLICY", "RANDOM", "RESIDUAL_GREEDY", "PURE_EI", "SAST_BO")
SEEDS = (0, 1, 2, 3, 4)
K4_PROBES = PROBES[:2]
K8_PROBES = PROBES

# The three candidate coordinates have different physical ranges.  All local
# surrogate comparisons use this frozen dimensionless scaling so that the
# duration coordinate (roughly one unit) cannot dominate timing/share (roughly
# one quarter unit) simply because of its numeric representation.
FEATURE_REFERENCE = np.asarray((1.0, 0.5, 0.5), dtype=float)
FEATURE_SCALE = np.asarray((0.1, 0.25, 0.25), dtype=float)
SAFETY_PROBABILITY = 0.95
E2_LIMIT = 1.01
PEAK_LIMIT = 1.10
SAST_CONFIG = GateConfig(interaction_threshold=0.0025, posterior_threshold=0.95, length_scale=1.0)


def _scaled_features(features):
    values = np.asarray(features, dtype=float)
    if values.shape != FEATURE_REFERENCE.shape or not np.isfinite(values).all():
        raise ValueError("INVALID_RESISTANCE_FEATURES")
    return (values - FEATURE_REFERENCE) / FEATURE_SCALE


def _prior_safe(candidate: CandidateView) -> bool:
    """Apply the same conservative prior admissibility to every baseline."""

    prediction = candidate.constraint_predictions
    z = NormalDist().inv_cdf(1.0 - (1.0 - SAFETY_PROBABILITY) / 2.0)
    for name, limit in (("E2", E2_LIMIT), ("peak_ratio", PEAK_LIMIT)):
        if name not in prediction:
            return False
        mean, std = map(float, prediction[name])
        if not np.isfinite((mean, std)).all() or mean < 0.0 or std < 0.0:
            return False
        if mean + z * std > limit:
            return False
    return True


def _observation(profile, candidate, trial):
    constraints = constraint_values_for(candidate.features)
    feasible = constraints["E2"] <= 1.01 and constraints["peak_ratio"] <= 1.10
    return ObservationRecord(candidate_id=candidate.candidate_id, features=candidate.features,
                             value=profile.value(candidate.features), uncertainty=0.0, feasible=feasible,
                             constraint_values=constraints,
                             trial_index=trial)


def _truth(profile, candidates):
    values = {c.candidate_id: profile.value(c.features) for c in candidates}
    feasible = [
        c.candidate_id for c in candidates
        if constraint_values_for(c.features)["E2"] <= 1.01
        and constraint_values_for(c.features)["peak_ratio"] <= 1.10
    ]
    if not feasible:
        raise RuntimeError("RESISTANCE_STRESS_FIELD_HAS_NO_FEASIBLE_CANDIDATE")
    oracle = min(feasible, key=lambda x: (values[x], x))
    return values, oracle


def _fit_residual_model(observations, by_id):
    """Fit the comparison residual model from queried observations only."""

    model = LocalResidualModel(length_scale=1.0, signal_std=0.6)
    xs = [_scaled_features(by_id[o.candidate_id].features) for o in observations]
    ys = [float(o.value) - common_value(o.features) for o in observations]
    model.fit(xs, ys, [max(float(o.uncertainty), 1.0e-6) for o in observations])
    return model


def _select_residual_greedy(available, observations, by_id):
    model = _fit_residual_model(observations, by_id)
    return min(
        available,
        key=lambda c: (
            common_value(c.features) + model.predict(_scaled_features(c.features))[0],
            c.candidate_id,
        ),
    )


def _select_pure_ei(available, observations, by_id):
    """Select by EI without evidence gating (the acquisition ablation)."""

    model = _fit_residual_model(observations, by_id)
    incumbent = min(float(o.value) for o in observations if o.valid and o.value is not None)
    scored = []
    for candidate in available:
        residual_mean, residual_std = model.predict(_scaled_features(candidate.features))
        mean = common_value(candidate.features) + residual_mean
        acquisition = expected_improvement(mean, residual_std, incumbent)
        scored.append((float(acquisition), candidate))
    # EI is maximized; candidate_id is a frozen deterministic tie-breaker.
    scored.sort(key=lambda item: (-item[0], item[1].candidate_id))
    return scored[0][1]


def _as_decision_record(trial, candidate, mode):
    return {"trial_index": int(trial), "candidate_id": candidate.candidate_id, "mode": mode}


def _probes_for_budget(budget: int):
    if budget == 4:
        return K4_PROBES
    if budget >= 5:
        return K8_PROBES
    raise ValueError("RESISTANCE_BUDGET_REQUIRES_AT_LEAST_FOUR_TRIALS")


def run_profile(profile, method, *, budget=8, seed=0):
    if method not in METHODS or profile.split != "DEVELOPMENT" or budget < 4:
        raise ValueError("RESISTANCE_ALGORITHM_SCOPE_OR_BUDGET")
    candidates = build_candidates()
    by_id = {c.candidate_id: c for c in candidates}
    by_features = {c.features: c for c in candidates}
    reference = by_features[(1.0, 0.5, 0.5)]
    probe_features = _probes_for_budget(budget)
    probes = tuple(by_features[x] for x in probe_features)
    if method == "SAST_BO":
        controller = SASTBO(
            candidates, reference_id=reference.candidate_id,
            calibration_probe_ids=tuple(p.candidate_id for p in probes),
            common_policy_id=reference.candidate_id,
            common_predictor=lambda c: common_value(c.features),
            preapproved_candidate_ids=(reference.candidate_id, *(p.candidate_id for p in probes)),
            feature_transform=lambda c: _scaled_features(c.features),
            config=SAST_CONFIG,
        )
        def evaluator(candidate, trial):
            return _observation(profile, candidate, trial)
        sequence = run_sequence(controller, evaluator, budget=budget)
        observations = sequence["observations"]
        decisions = [asdict(decision) for decision in sequence["decisions"]]
        recommendation = controller.recommend()
        gate_active = bool(controller.last_gate.active)
        termination = sequence["termination"] or "BUDGET_EXHAUSTED"
    else:
        rng = np.random.default_rng(seed)
        observations = []
        decisions = []
        used = set()
        for trial in range(1, budget + 1):
            if trial == 1:
                candidate = reference
            elif trial <= 1 + len(probes):
                candidate = probes[trial - 2]
            elif method == "COMMON_POLICY":
                break
            elif method == "RANDOM":
                available = [c for c in candidates if c.candidate_id not in used and _prior_safe(c)]
                candidate = available[int(rng.integers(len(available)))]
            elif method == "RESIDUAL_GREEDY":
                available = [c for c in candidates if c.candidate_id not in used and _prior_safe(c)]
                candidate = _select_residual_greedy(available, observations, by_id)
            elif method == "PURE_EI":
                available = [c for c in candidates if c.candidate_id not in used and _prior_safe(c)]
                candidate = _select_pure_ei(available, observations, by_id)
            else:
                raise ValueError(f"UNKNOWN_RESISTANCE_METHOD: {method}")
            if candidate.candidate_id in used:
                break
            mode = "CALIBRATION" if trial <= 1 + len(probes) else method
            decisions.append(_as_decision_record(trial, candidate, mode))
            used.add(candidate.candidate_id)
            observations.append(_observation(profile, candidate, trial))
        # The common arm is genuinely fixed: probes are consumed only to keep
        # the calibration cost comparable; they never replace the declared
        # reference recommendation.
        if method == "COMMON_POLICY":
            recommendation = reference.candidate_id if observations else None
        else:
            eligible = [o for o in observations if o.valid and o.feasible]
            recommendation = min(eligible, key=lambda o: (o.value, o.trial_index)).candidate_id if eligible else None
        gate_active = False
        termination = (
            "COMMON_POLICY_FIXED"
            if method == "COMMON_POLICY"
            else ("BUDGET_EXHAUSTED" if len(observations) == budget else "SEQUENCE_TERMINATED")
        )
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
        "observations": [asdict(observation) for observation in observations],
        "decisions": decisions,
    }


def run_benchmark(*, output_dir=DEFAULT_OUTPUT, budget=8):
    profiles = load_profiles("DEVELOPMENT")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "experiment_id": "CONTROLLED_RESISTANCE_ALGORITHM_V2",
        "scope": "development-only controlled synthetic resistance mechanism",
        "methods": METHODS, "seeds": SEEDS, "budget": budget,
        "budget_definition": "maximum_trials; fixed common and inactive-gate fallback may stop once the observed reference is the declared recommendation",
        "early_stop_is_reported": True,
        "profile_ids": [p.profile_id for p in profiles], "candidate_count": len(build_candidates()),
        "probes": {"K4": K4_PROBES, "K8": K8_PROBES},
        "confirmatory_access": False, "oracle_access": "evaluator-only after run",
        "feature_scaling": {"reference": FEATURE_REFERENCE.tolist(), "scale": FEATURE_SCALE.tolist()},
        "sast_config": asdict(SAST_CONFIG),
        "cohort_manifest_sha256": manifest["manifest_fingerprint_sha256"],
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name("controlled_resistance_experiment.py"),
            Path(__file__).with_name("controlled_resistance_cohort.py"), Path(__file__).with_name("personalized_v2.py"))},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = [run_profile(p, method, budget=budget, seed=seed)
            for p in profiles for method in METHODS for seed in (SEEDS if method in ("RANDOM",) else (0,))]
    frame = pd.DataFrame(rows)
    scalar = frame.drop(columns=["observations", "decisions"])
    scalar.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "decision_logs.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    summary = scalar.groupby(["scenario", "method"], sort=True).agg(
        runs=("profile_id", "count"), gate_active_rate=("gate_active", "mean"),
        common_regret_mean=("common_regret", "mean"), policy_regret_mean=("policy_regret", "mean"),
        improvement_mean=("improvement_over_common", "mean"), executed_trials_mean=("executed_trials", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "completion.json").write_text(json.dumps({"complete": True, "runs": len(frame), "confirmatory_access": False}, indent=2) + "\n")
    lines = ["# Controlled resistance algorithm comparison", "",
             "Development-only controlled synthetic mechanism test; no native or patient claim.",
             "All methods use the same reference plus frozen calibration probes and the same maximum budget. COMMON_POLICY always recommends the reference; fixed common and inactive-gate fallback may stop after calibration because the recommendation is already observed, and the actual trial cost is reported. RANDOM uses five seeds and the other methods are deterministic.", "",
             "| Scenario | Method | Runs | Gate active | Common regret | Policy regret | Improvement | Trials |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['scenario']} | {row['method']} | {row['runs']} | {row['gate_active_rate']:.0%} | "
                     f"{row['common_regret_mean']:.6f} | {row['policy_regret_mean']:.6f} | {row['improvement_mean']:.6f} | {row['executed_trials_mean']:.1f} |")
    lines += ["", "No confirmatory profiles were loaded. Results are software mechanism evidence only."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps({"complete": True, "output": str(run_benchmark(output_dir=args.output_dir, budget=args.budget))}))


if __name__ == "__main__":
    main()
