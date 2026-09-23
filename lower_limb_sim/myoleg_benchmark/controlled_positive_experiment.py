"""Policy-level controlled stress test for evidence-gated personalization.

This runner intentionally uses a fixed analytical response surface. It is a
software stress test of the gate and recommendation contract, not a native
MyoLeg physiological or safety validation. The learner receives only queried
observations; oracle scoring happens after the policy run. Profile IDs repeat
three deterministic response fields (common, arm A, arm B), so they are not
independent subject replicates.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .controlled_positive_cohort import DEFAULT_MANIFEST, ControlledProfile, load_profiles
from .personalized_v2 import CandidateView, GateConfig, ObservationRecord, SASTBO, run_sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/myoleg_controlled_positive_policy_v2"
GRID_VALUES = tuple(float(x) for x in np.linspace(-4.0, 4.0, 9))
SHIFT_VALUES = (-2.0, -1.0, 0.0, 1.0, 2.0)
COMMON_CURVATURE = 0.002
CONSTRAINT_STD = 0.001
PROBE_FEATURES = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
GATE_CONFIG = GateConfig(interaction_threshold=0.0025, posterior_threshold=0.95)


def build_grid() -> tuple[CandidateView, ...]:
    rows = []
    for index, features in enumerate((a, b, c) for a in GRID_VALUES for b in GRID_VALUES for c in SHIFT_VALUES):
        amplitude = float(np.sum(np.abs(features)))
        constraints = {"E2": (0.95 + 0.002 * amplitude, CONSTRAINT_STD),
                       "peak_ratio": (0.95 + 0.002 * amplitude, CONSTRAINT_STD)}
        rows.append(CandidateView(f"grid:{index:03d}", features, constraints))
    return tuple(rows)


def common_value(features: tuple[float, ...]) -> float:
    x = np.asarray(features, dtype=float)
    return float(1.0 + COMMON_CURVATURE * np.sum(x * x))


def profile_value(profile: ControlledProfile, features: tuple[float, ...]) -> float:
    return float(common_value(features) * profile.gain(features))


def _validate_budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 4:
        raise ValueError("budget must include reference and three calibration probes (>=4)")


def run_profile(profile: ControlledProfile, *, budget: int = 8) -> dict[str, Any]:
    _validate_budget(budget)
    if profile.split != "DEVELOPMENT":
        raise ValueError("CONFIRMATORY_RESERVED: this runner only validates development")
    candidates = build_grid()
    by_features = {candidate.features: candidate for candidate in candidates}
    reference = by_features[(0.0, 0.0, 0.0)]
    probes = tuple(by_features[features] for features in PROBE_FEATURES)
    controller = SASTBO(
        candidates,
        reference_id=reference.candidate_id,
        calibration_probe_ids=tuple(p.candidate_id for p in probes),
        common_policy_id=reference.candidate_id,
        common_predictor=lambda candidate: common_value(candidate.features),
        preapproved_candidate_ids=(reference.candidate_id, *(p.candidate_id for p in probes)),
        config=GATE_CONFIG,
    )

    def evaluator(candidate: CandidateView, trial: int) -> ObservationRecord:
        return ObservationRecord(
            candidate_id=candidate.candidate_id,
            features=candidate.features,
            value=profile_value(profile, candidate.features),
            uncertainty=0.0,
            feasible=True,
            constraint_values={name: pair[0] for name, pair in candidate.constraint_predictions.items()},
            trial_index=trial,
        )

    sequence = run_sequence(controller, evaluator, budget=budget)
    recommendation_id = controller.recommend()
    observed = {o.candidate_id: o for o in sequence["observations"]}
    if recommendation_id is not None and recommendation_id not in observed:
        raise RuntimeError("RECOMMENDATION_OUTSIDE_EXECUTED_BUDGET")
    truth = {candidate.candidate_id: profile_value(profile, candidate.features) for candidate in candidates}
    oracle_id = min(truth, key=lambda key: (truth[key], key))
    oracle_value = truth[oracle_id]
    common_value_at = truth[reference.candidate_id]
    recommendation_value = None if recommendation_id is None else truth[recommendation_id]
    decisions = []
    for trial, decision in enumerate(sequence["decisions"], 1):
        decisions.append({
            "trial": trial,
            "candidate_id": None if decision.candidate is None else decision.candidate.candidate_id,
            "mode": decision.mode,
            "reason": decision.reason,
            "gate_status_before": decision.gate.status,
            "gate_evidence_score_before": decision.gate.posterior_probability,
            "safe_candidate_count": decision.safe_candidate_count,
            "executed": trial <= len(sequence["observations"]),
        })
    return {
        "profile_id": profile.profile_id, "scenario": profile.scenario,
        "split": profile.split, "arm": profile.arm, "budget": budget,
        "gate_status": controller.last_gate.status,
        "gate_evidence_score": controller.last_gate.posterior_probability,
        "gate_interaction_score": controller.last_gate.interaction_score,
        "recommendation_mode": ("NO_SAFE_RECOMMENDATION" if recommendation_id is None
                                 else "PERSONALIZED_OBSERVED" if controller.last_gate.active
                                 else "COMMON_POLICY"),
        "recommendation_id": recommendation_id, "recommendation_value": recommendation_value,
        "common_id": reference.candidate_id, "common_value": common_value_at,
        "oracle_id": oracle_id, "oracle_value": oracle_value,
        "common_regret": common_value_at - oracle_value,
        "personalized_regret": None if recommendation_value is None else recommendation_value - oracle_value,
        "improvement_over_common": None if recommendation_value is None else common_value_at - recommendation_value,
        "executed_trials": len(observed), "termination": sequence["termination"] or "BUDGET_EXHAUSTED",
        "decisions": decisions, "observations": [asdict(o) for o in sequence["observations"]],
    }


def _scalar(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in ("decisions", "observations")}


def run_benchmark(*, output_dir: Path = DEFAULT_OUTPUT, budget: int = 8, split: str = "DEVELOPMENT") -> Path:
    _validate_budget(budget)
    if split != "DEVELOPMENT":
        raise ValueError("CONFIRMATORY_RESERVED: only DEVELOPMENT is allowed")
    profiles = load_profiles(split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    protocol = {
        "experiment_id": "CONTROLLED_POSITIVE_POLICY_V2",
        "truth_scope": "analytical policy stress test; not native MyoLeg or physiological validation",
        "split": split, "budget": budget, "profile_ids": [p.profile_id for p in profiles],
        "cohort_manifest_sha256": manifest["manifest_fingerprint_sha256"],
        "unique_response_fields": len({(p.field_strength, p.center_features) for p in profiles}),
        "replication_limit": "IDs repeat the same common/A/B fields; no independent n=12 claim",
        "confirmatory_access": False, "common_policy": "fixed analytical-surface reference",
        "common_curvature": COMMON_CURVATURE, "probe_features": PROBE_FEATURES,
        "candidate_grid": len(build_grid()), "noise_std": 0.0,
        "config": asdict(GATE_CONFIG),
        "gate_interpretation": "heuristic residual evidence score, not a calibrated interaction posterior",
        "constraint_scope": "known analytical constraints; not native safety validation",
        "final_recommendation": "best actually observed feasible candidate when gate active; common otherwise",
        "oracle_access": "evaluator-only after policy run",
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("personalized_v2.py"),
            Path(__file__).with_name("controlled_positive_cohort.py"))},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = [run_profile(profile, budget=budget) for profile in profiles]
    summary = pd.DataFrame([_scalar(row) for row in rows])
    summary.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "decision_logs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = ["# Controlled positive policy validation", "",
             "Synthetic analytical stress test, not native MyoLeg physiology or patient evidence.", "",
             f"Budget ceiling: {budget}; profiles: {len(profiles)}; unique response fields: 3.",
             "IDs repeat identical deterministic fields; do not treat them as independent subjects.",
             "No confirmatory responses were evaluated. No stochastic confidence interval is claimed.",
             "Final personalized recommendations were observed within the budget.", "",
             "| Response field | Gate active | Executed trials | Common regret | SAST regret | Reduction |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario, arm), group in summary.groupby(["scenario", "arm"], sort=True):
        common = float(group.common_regret.mean())
        regret = float(group.personalized_regret.mean())
        reduction = "n/a" if common == 0 else f"{100 * (common - regret) / common:.1f}%"
        lines.append(f"| {scenario}/{arm} | {(group.gate_status == 'PERSONALIZATION_ACTIVE').mean():.0%} | "
                     f"{group.executed_trials.mean():g} | {common:.6f} | {regret:.6f} | {reduction} |")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "completion.json").write_text(json.dumps({"complete": True, "profiles": len(rows), "confirmatory_access": False}, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--split", choices=("DEVELOPMENT",), default="DEVELOPMENT")
    args = parser.parse_args(argv)
    if args.budget < 4:
        parser.error("budget must include reference and three calibration probes")
    path = run_benchmark(output_dir=args.output_dir, budget=args.budget, split=args.split)
    print(json.dumps({"complete": True, "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
