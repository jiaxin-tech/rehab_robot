"""Small development pilot for the resistance-interaction stress test.

The only learner-visible data are requested scalar responses and declared
constraint predictions.  Full profile parameters and the oracle are used only
after the policy run by this evaluator.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .controlled_resistance_cohort import (
    DEFAULT_MANIFEST, FORMULA_COEFFICIENTS, ResistanceProfile, build_profiles, load_profiles,
)
from .personalized_v2 import CandidateView, GateConfig, ObservationRecord, SASTBO, run_sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/myoleg_controlled_resistance_pilot_v6"
GRID = tuple((d, t, h) for d in (0.9, 1.0, 1.1) for t in (0.25, 0.5, 0.75) for h in (0.25, 0.5, 0.75))
PROBES = ((0.9, 0.5, 0.5), (1.1, 0.5, 0.5), (1.0, 0.25, 0.5), (1.0, 0.5, 0.75))
GATE_CONFIG = GateConfig(interaction_threshold=0.0025, posterior_threshold=0.95, length_scale=1.0)
PRACTICAL_GAP = 0.005
FEATURE_REFERENCE = np.asarray((1.0, 0.5, 0.5), dtype=float)
FEATURE_SCALE = np.asarray((0.1, 0.25, 0.25), dtype=float)


def _scaled_features(features):
    values = np.asarray(features, dtype=float)
    if values.shape != FEATURE_REFERENCE.shape or not np.isfinite(values).all():
        raise ValueError("INVALID_RESISTANCE_FEATURES")
    return (values - FEATURE_REFERENCE) / FEATURE_SCALE


def common_value(features: tuple[float, ...]) -> float:
    d, t, h = map(float, features)
    return float(1.0 + FORMULA_COEFFICIENTS["duration_common_cost"] * ((d - 1.0) / 0.1) ** 2 +
                 FORMULA_COEFFICIENTS["timing_common_cost"] * ((t - 0.5) / 0.25) ** 2 +
                 FORMULA_COEFFICIENTS["hip_share_common_cost"] * ((h - 0.5) / 0.25) ** 2)


def constraint_values_for(features: tuple[float, ...]) -> dict[str, float]:
    """Return the declared synthetic constraint outcome for one candidate.

    This is still an analytical stress-field constraint, not a native or
    clinical safety limit.  Unlike the original pilot, the field contains
    genuinely infeasible corners and the evaluator returns the outcome
    separately from the candidate's conservative prior prediction.
    """

    duration, timing, hip_share = map(float, features)
    timing_offset = (timing - 0.5) / 0.25
    share_offset = (hip_share - 0.5) / 0.25
    duration_offset = (duration - 1.0) / 0.1
    e2 = 0.99 + 0.012 * (timing_offset**2 + share_offset**2)
    peak_ratio = 1.00 + 0.008 * duration_offset**2 + 0.004 * abs(timing_offset - share_offset)
    return {"E2": float(e2), "peak_ratio": float(peak_ratio)}


def build_candidates() -> tuple[CandidateView, ...]:
    rows = []
    for index, features in enumerate(GRID):
        # Candidate-side values are only a conservative prior.  The evaluator
        # returns the actual declared stress-field outcome after execution.
        truth = constraint_values_for(features)
        rows.append(CandidateView(f"resist:{index:02d}", features,
                                  {"E2": (min(truth["E2"], 1.01), 0.004),
                                   "peak_ratio": (min(truth["peak_ratio"], 1.05), 0.005)}))
    return tuple(rows)


def run_profile(profile: ResistanceProfile, *, budget: int = 8) -> dict:
    if budget < 5 or profile.split != "DEVELOPMENT":
        raise ValueError("RESISTANCE_PILOT_DEVELOPMENT_BUDGET_OR_SPLIT")
    candidates = build_candidates()
    by_features = {c.features: c for c in candidates}
    reference = by_features[(1.0, 0.5, 0.5)]
    probes = tuple(by_features[x] for x in PROBES)
    controller = SASTBO(
        candidates, reference_id=reference.candidate_id,
        calibration_probe_ids=tuple(p.candidate_id for p in probes),
        common_policy_id=reference.candidate_id,
        common_predictor=lambda candidate: common_value(candidate.features),
        preapproved_candidate_ids=(reference.candidate_id, *(p.candidate_id for p in probes)),
        feature_transform=lambda c: _scaled_features(c.features),
        config=GATE_CONFIG,
    )

    def evaluator(candidate: CandidateView, trial: int) -> ObservationRecord:
        constraints = constraint_values_for(candidate.features)
        feasible = constraints["E2"] <= 1.01 and constraints["peak_ratio"] <= 1.10
        return ObservationRecord(
            candidate_id=candidate.candidate_id, features=candidate.features,
            value=profile.value(candidate.features), uncertainty=0.0, feasible=feasible,
            constraint_values=constraints,
            trial_index=trial,
        )

    sequence = run_sequence(controller, evaluator, budget=budget)
    recommendation_id = controller.recommend()
    truth = {c.candidate_id: profile.value(c.features) for c in candidates}
    feasible_ids = [
        c.candidate_id for c in candidates
        if constraint_values_for(c.features)["E2"] <= 1.01
        and constraint_values_for(c.features)["peak_ratio"] <= 1.10
    ]
    if not feasible_ids:
        raise RuntimeError("RESISTANCE_STRESS_FIELD_HAS_NO_FEASIBLE_CANDIDATE")
    oracle_id = min(feasible_ids, key=lambda cid: (truth[cid], cid))
    common_id = reference.candidate_id
    rec_value = None if recommendation_id is None else truth[recommendation_id]
    oracle_value, common_at = truth[oracle_id], truth[common_id]
    return {
        "profile_id": profile.profile_id, "scenario": profile.scenario,
        "arm": profile.arm, "budget": budget,
        "gate_status": controller.last_gate.status,
        "gate_interaction_score": controller.last_gate.interaction_score,
        "gate_posterior_score": controller.last_gate.posterior_probability,
        "recommendation_id": recommendation_id,
        "recommendation_value": rec_value,
        "common_id": common_id, "common_value": common_at,
        "oracle_id": oracle_id, "oracle_value": oracle_value,
        "common_regret": common_at - oracle_value,
        "policy_regret": None if rec_value is None else rec_value - oracle_value,
        "policy_improvement_over_common": None if rec_value is None else common_at - rec_value,
        "gate_active": bool(controller.last_gate.active),
        "executed_trials": len(sequence["observations"]),
        "termination": sequence["termination"] or "BUDGET_EXHAUSTED",
        "observations": [asdict(o) for o in sequence["observations"]],
    }


def run_benchmark(*, output_dir: Path = DEFAULT_OUTPUT, budget: int = 8) -> Path:
    profiles = load_profiles("DEVELOPMENT")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "experiment_id": "CONTROLLED_RESISTANCE_INTERACTION_V2_DEVELOPMENT",
        "truth_scope": "controlled synthetic resistance mechanism; not native or physiological evidence",
        "profile_ids": [p.profile_id for p in profiles], "budget": budget,
        "candidate_count": len(build_candidates()), "probe_features": PROBES,
        "gate_config": asdict(GATE_CONFIG), "confirmatory_access": False,
        "feature_scaling": {"reference": FEATURE_REFERENCE.tolist(), "scale": FEATURE_SCALE.tolist()},
        "cohort_manifest_sha256": manifest["manifest_fingerprint_sha256"],
        "oracle_access": "evaluator-only after run",
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("controlled_resistance_cohort.py"),
            Path(__file__).with_name("personalized_v2.py"))},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = [run_profile(profile, budget=budget) for profile in profiles]
    scalar = [{k: v for k, v in row.items() if k not in ("observations",)} for row in rows]
    pd.DataFrame(scalar).to_csv(output_dir / "results.csv", index=False)
    (output_dir / "decision_logs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = pd.DataFrame(scalar).groupby(["scenario", "arm"], sort=True).agg(
        profiles=("profile_id", "count"), gate_active_rate=("gate_active", "mean"),
        common_regret_mean=("common_regret", "mean"), policy_regret_mean=("policy_regret", "mean"),
        policy_improvement_mean=("policy_improvement_over_common", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    positive = pd.DataFrame([row for row in scalar if row["scenario"] == "DIVERGENT_RESISTANCE"])
    null = pd.DataFrame([row for row in scalar if row["scenario"] == "COMMON_RESISTANCE"])
    detectable = positive["common_regret"] >= PRACTICAL_GAP
    useful = (positive["policy_improvement_over_common"] >= PRACTICAL_GAP)
    decision = {
        "null_gate_active_rate": float(null["gate_active"].mean()),
        "positive_profile_count": int(len(positive)),
        "positive_profiles_with_practical_common_regret": int(detectable.sum()),
        "positive_profiles_with_practical_policy_improvement": int(useful.sum()),
        "positive_detection_fraction": float(detectable.mean()) if len(positive) else 0.0,
        "positive_policy_improvement_fraction": float(useful.mean()) if len(positive) else 0.0,
        "positive_mean_policy_improvement": float(positive["policy_improvement_over_common"].mean()) if len(positive) else 0.0,
        "decision": "MECHANISM_PILOT_ELIGIBLE_FOR_ALGORITHM_COMPARISON" if (
            null["gate_active"].sum() == 0 and detectable.mean() >= 0.75 and
            positive["policy_improvement_over_common"].mean() >= PRACTICAL_GAP
        ) else "HOLD_MECHANISM_OR_GATE_REQUIRES_REVISION",
        "threshold": PRACTICAL_GAP,
        "confirmatory_ready": False,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = ["# Controlled resistance interaction pilot", "",
             "Analytical mechanism stress test; not native MyoLeg physiology or patient evidence.",
             "The learner receives only queried values; profile parameters and oracle scoring are evaluator-only.", "",
             "| Scenario / arm | Profiles | Gate active | Common regret | Policy regret | Improvement |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['scenario']} / {row['arm']} | {row['profiles']} | {row['gate_active_rate']:.0%} | "
                     f"{row['common_regret_mean']:.6f} | {row['policy_regret_mean']:.6f} | {row['policy_improvement_mean']:.6f} |")
    lines += ["", f"Decision: {decision['decision']}.",
              f"Null gate active rate: {decision['null_gate_active_rate']:.0%}; practical divergent common-regret fraction: {decision['positive_detection_fraction']:.0%}; practical policy-improvement fraction: {decision['positive_policy_improvement_fraction']:.0%}.",
              "A profile can trigger the gate without benefiting from personalization (the FAST arm is an explicit example); detection and useful regret reduction are reported separately.",
              "This pilot is a go/no-go mechanism check, not a confirmatory experiment.",
              "No confirmatory profiles were loaded; no claim about patient physiology or clinical safety is supported."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "completion.json").write_text(json.dumps({"complete": True, "profiles": len(rows), "confirmatory_access": False}, indent=2) + "\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps({"complete": True, "output": str(run_benchmark(output_dir=args.output_dir, budget=args.budget))}))


if __name__ == "__main__":
    main()
