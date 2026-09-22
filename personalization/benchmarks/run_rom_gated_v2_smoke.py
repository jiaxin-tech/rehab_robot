"""Run the offline ROM -> subject V3 -> K=4 architecture smoke benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from personalization.rom_gated_v2 import (
    ROMGatedPersonalizationEpisode,
    SubjectSpecificOfflineMechanicalEnvironment,
    SubjectSpecificV3CandidateDomain,
    assert_same_frozen_rom_comparison,
    determine_synthetic_rom,
    make_offline_rom_development_cases,
    make_subject_physics_model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "personalization" / "benchmarks" / "results_v2_rom_gated"
METHODS = ("Reference", "Space Filling", "Standard BO", "Physics-Informed BO")
PRIMARY_K = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_value(result) -> float | None:
    values = [
        float(entry.observation.endpoint_value)
        for entry in result.sequential_result.ledger.entries
        if entry.observation.valid and entry.observation.endpoint_value is not None
    ]
    return min(values) if values else None


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cases_payload = []

    for case in make_offline_rom_development_cases():
        controller, profile = determine_synthetic_rom(case)
        domain = SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)
        case_rows = []
        case_results = {}
        for method in METHODS:
            environment = SubjectSpecificOfflineMechanicalEnvironment(domain, case)
            episode = ROMGatedPersonalizationEpisode(
                episode_id=f"ROM_GATED_V2:{case.case_id}:{method}",
                profile=profile,
                domain=domain,
                adaptation_budget=PRIMARY_K,
            )
            result = episode.run(
                environment,
                method=method,
                seed=0,
                physics_model=(
                    make_subject_physics_model(domain)
                    if method == "Physics-Informed BO"
                    else None
                ),
            )
            ledger = result.sequential_result.ledger
            row = {
                "case_id": case.case_id,
                "method": method,
                "rom_profile_id": profile.profile_id,
                "rom_profile_fingerprint": profile.fingerprint,
                "rom_calibration_observation_count": len(controller.ledger.entries),
                "personalization_adaptation_budget": PRIMARY_K,
                "personalization_trial_count": len(ledger.entries),
                "first_candidate_is_subject_reference": (
                    ledger.entries[0].candidate.candidate_id
                    == domain.reference.candidate_id
                ),
                "best_observed_endpoint": _best_value(result),
                "executed_subject_candidate_ids": "|".join(
                    ledger.executed_candidate_ids
                ),
                "executed_canonical_beta_ids": "|".join(
                    domain.by_id(candidate_id).canonical_beta_id
                    for candidate_id in ledger.executed_candidate_ids
                ),
                "candidate_count": len(domain),
                "robot_actions": 0,
                "human_actions": 0,
            }
            rows.append(row)
            case_rows.append(row)
            case_results[method] = result
        assert_same_frozen_rom_comparison(case_results)
        fingerprints = {row["rom_profile_fingerprint"] for row in case_rows}
        if fingerprints != {profile.fingerprint}:
            raise RuntimeError("baseline ROM mismatch inside development case")
        cases_payload.append(
            {
                "case_id": case.case_id,
                "classification": "ALGORITHM_DEVELOPMENT_ONLY",
                "profile": profile.as_dict(),
                "rom_calibration_ledger": controller.ledger.as_dict(),
                "rom_calibration_gate_sequence": [
                    entry.gate_result.status.value for entry in controller.ledger.entries
                ],
                "candidate_domain_invariants": domain.invariant_summary(),
                "same_frozen_rom_for_all_baselines": True,
                "methods": [
                    {
                        "method": row["method"],
                        "best_observed_endpoint": row["best_observed_endpoint"],
                        "executed_canonical_beta_ids": row[
                            "executed_canonical_beta_ids"
                        ].split("|"),
                    }
                    for row in case_rows
                ],
            }
        )

    csv_path = output_dir / "ROM_GATED_V2_SMOKE_RUNS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "implementation_status": (
            "SUBJECT_SPECIFIC_ROM_GATED_SEQUENTIAL_PERSONALIZATION_V2_IMPLEMENTED_WITH_LIMITATIONS"
        ),
        "classification": "OFFLINE_ALGORITHM_DEVELOPMENT_ONLY",
        "human_readiness": "NOT_HUMAN_READY",
        "robot_approval": "NOT_ROBOT_APPROVED",
        "stage_separation": {
            "stage_0": "task-specific ROM determination",
            "stage_1": "fixed-ROM beta_flex/beta_extend coordination personalization",
            "rom_determination_budget_is_personalization_budget": False,
            "primary_personalization_K": PRIMARY_K,
        },
        "case_count": len(cases_payload),
        "method_count_per_case": len(METHODS),
        "run_count": len(rows),
        "methods": list(METHODS),
        "cases": cases_payload,
        "all_runs_completed": all(
            row["personalization_trial_count"] == PRIMARY_K for row in rows
        ),
        "all_stage1_runs_started_at_subject_reference": all(
            row["first_candidate_is_subject_reference"] for row in rows
        ),
        "all_candidate_domains_have_625_members": all(
            case["candidate_domain_invariants"]["candidate_count"] == 625
            for case in cases_payload
        ),
        "all_subject_rom_invariants_pass": all(
            case["candidate_domain_invariants"]["all_kinematic_gates_pass"]
            and case["candidate_domain_invariants"]["pointwise_clipping_count"] == 0
            for case in cases_payload
        ),
        "V1_core_reused_without_rewrite": True,
        "frozen_V3_operator_reused_without_rewrite": True,
        "pressure_limit": "NOT_AVAILABLE",
        "validated_force_threshold": "NOT_AVAILABLE",
        "safety_margin_policy": None,
        "robot_actions": 0,
        "human_actions": 0,
        "PINN_training_runs": 0,
        "interpretation": (
            "Architecture compatibility smoke only; method values are not human, "
            "clinical, comfort, safety, or robot-effectiveness evidence."
        ),
        "runs_csv": csv_path.name,
    }
    summary_path = output_dir / "ROM_GATED_V2_SMOKE_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums_path = output_dir / "checksums.sha256"
    checksums_path.write_text(
        f"{_sha256(summary_path)}  {summary_path.name}\n"
        f"{_sha256(csv_path)}  {csv_path.name}\n",
        encoding="utf-8",
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        "ROM-gated V2 offline smoke: "
        f"{len(cases_payload)} cases, {len(rows)} runs, K={PRIMARY_K}, "
        f"robot actions=0, elapsed={elapsed_ms:.1f} ms"
    )
    print(f"summary: {summary_path}")
    return summary


if __name__ == "__main__":
    run()
