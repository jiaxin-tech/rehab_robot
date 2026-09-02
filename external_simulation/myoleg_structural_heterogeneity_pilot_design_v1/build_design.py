"""Build the design-only structural-heterogeneity pilot preregistration.

The authoritative S1 artifact is deliberately treated as immutable.  If it
does not name exact target members and factor direction conventions, this
builder emits a complete fail-closed audit package and no executable pilot.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE_ID = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V1"
OUTCOME = "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_NOT_READY"
BLOCKER = "S1_DEFINITION_INCOMPLETE"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v1"
EXPANSION = ROOT / "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1"
SCHEMES = EXPANSION / "PROPOSED_STRUCTURAL_HETEROGENEITY_SCHEMES.json"
BIARTICULAR = EXPANSION / "BIARTICULAR_COUPLING_PARAMETER_AUDIT.csv"
MUSCLE_GROUPS = EXPANSION / "MUSCLE_GROUP_HETEROGENEITY_AUDIT.csv"
OPERATING = EXPANSION / "MUSCLE_OPERATING_LENGTH_PARAMETER_AUDIT.csv"
EVIDENCE = EXPANSION / "EVIDENCE_SOURCES.csv"
EXPANSION_METADATA = EXPANSION / "metadata.json"
EXPANSION_CHECKSUMS = EXPANSION / "checksums.sha256"
COHORT_V1 = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
PRIOR_FINAL = ROOT / "external_simulation_audits/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1/FINAL_BRANCH_DECISION.json"
FORMAL_MANIFEST = ROOT / "config/formal_experiment_manifest.json"
FORMAL_REFERENCE = ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"

FROZEN_SHA = {
    "authoritative_schemes": "47ebf27c43ccca9621e315c1322946bb7b8687e098b243ee8d92f0f66d578394",
    "biarticular_audit": "cfec07329d8bba93773838703a9baa71bae161e49ff8cfebf4b25d96703b951c",
    "muscle_group_audit": "e151ca561c1554151567048a3e62f64c3f25e5807ad1664d0eddce1802f78531",
    "operating_audit": "274717a81faa295369046f2592c9030c1ac7abd7a80a7423617c941adda507d9",
    "evidence_sources": "bc5c67abb6a9c8955f60b80fc4da31e736b731820fbd8b4ffffb3ddbd79426f3",
    "expansion_metadata": "98922d00a5d03c8f1c576471aa933ffffaa3ac58a38445ff739b50c807b1340f",
    "expansion_checksums": "4f1572d2a406ac8a7ae9c6419a8885c42bacac62339a449aad4a7260801190c4",
    "cohort_v1": "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057",
    "prior_final": "76d3878e278cd2bc79fa56a67c13a3e95142dbb165c7ee40810f197c987624c1",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
}

FROZEN_PATHS = {
    "authoritative_schemes": SCHEMES,
    "biarticular_audit": BIARTICULAR,
    "muscle_group_audit": MUSCLE_GROUPS,
    "operating_audit": OPERATING,
    "evidence_sources": EVIDENCE,
    "expansion_metadata": EXPANSION_METADATA,
    "expansion_checksums": EXPANSION_CHECKSUMS,
    "cohort_v1": COHORT_V1,
    "prior_final": PRIOR_FINAL,
    "formal_manifest": FORMAL_MANIFEST,
    "formal_reference": FORMAL_REFERENCE,
}

EXPECTED_S1_FACTORS = [
    "biarticular normalized-curve lmax profile",
    "rectus-vs-hamstring relative fpmax balance",
    "hip monoarticular antagonist relative F0",
    "knee monoarticular antagonist relative F0",
]
EXPECTED_S1_FIELDS = [
    "gainprm/biasprm[5]", "gainprm/biasprm[7]",
    "gainprm/biasprm[2]", "gainprm/biasprm[2]",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_frozen_inputs() -> dict[str, str]:
    actual = {name: sha256(path) for name, path in FROZEN_PATHS.items()}
    if actual != FROZEN_SHA:
        raise RuntimeError(f"frozen input changed: {actual}")
    schemes = json.loads(SCHEMES.read_text(encoding="utf-8"))
    s1 = next((row for row in schemes["schemes"] if row["scheme_id"] == "S1_MINIMAL_STRUCTURAL"), None)
    if s1 is None:
        raise RuntimeError("authoritative S1 missing")
    if s1["factors"] != EXPECTED_S1_FACTORS or s1["actual_model_fields"] != EXPECTED_S1_FIELDS or s1["dimensionality"] != 4:
        raise RuntimeError("authoritative S1 identity changed")
    expansion = json.loads(EXPANSION_METADATA.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    if expansion["outcome"] != "MYOLEG_HETEROGENEITY_EXPANSION_DESIGN_READY_WITH_EVIDENCE_GAPS":
        raise RuntimeError("frozen expansion outcome changed")
    if not (
        formal["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and formal["theta_shank_definition"] == "q_hip - q_knee"
        and formal["active_reference_sha256"] == FROZEN_SHA["formal_reference"]
    ):
        raise RuntimeError("formal conventions changed")
    return actual


def exact_factor_rows() -> list[dict[str, Any]]:
    biarticular = [row["muscle"] for row in read_csv(BIARTICULAR)]
    group_rows = {row["factor"]: row for row in read_csv(MUSCLE_GROUPS)}
    relative = group_rows["rectus-femoris vs hamstring-family relative fpmax"]
    return [
        {
            "factor_index": 1, "factor_name": EXPECTED_S1_FACTORS[0], "actual_fields": EXPECTED_S1_FIELDS[0],
            "affected_objects": "|".join(biarticular), "target_source": "frozen BIARTICULAR_COUPLING_PARAMETER_AUDIT.csv",
            "target_members_complete": True, "direction_and_coherent_update_rule_complete": False,
            "replaces_or_supplements_v1": "NOT_SPECIFIED_IN_AUTHORITATIVE_S1",
            "definition_status": "INCOMPLETE_DIRECTION_AND_UPDATE_RULE",
        },
        {
            "factor_index": 2, "factor_name": EXPECTED_S1_FACTORS[1], "actual_fields": EXPECTED_S1_FIELDS[1],
            "affected_objects": relative["model_fields"], "target_source": "frozen MUSCLE_GROUP_HETEROGENEITY_AUDIT.csv",
            "target_members_complete": True, "direction_and_coherent_update_rule_complete": False,
            "replaces_or_supplements_v1": "candidate replacement for V1 biarticular common fpmax; exact relation not frozen in S1",
            "definition_status": "INCOMPLETE_DIRECTION_AND_NORMALIZATION_RULE",
        },
        {
            "factor_index": 3, "factor_name": EXPECTED_S1_FACTORS[2], "actual_fields": EXPECTED_S1_FIELDS[2],
            "affected_objects": "NOT_SPECIFIED", "target_source": "authoritative S1 contains no muscle members",
            "target_members_complete": False, "direction_and_coherent_update_rule_complete": False,
            "replaces_or_supplements_v1": "NOT_SPECIFIED_IN_AUTHORITATIVE_S1",
            "definition_status": BLOCKER,
        },
        {
            "factor_index": 4, "factor_name": EXPECTED_S1_FACTORS[3], "actual_fields": EXPECTED_S1_FIELDS[3],
            "affected_objects": "NOT_SPECIFIED", "target_source": "authoritative S1 contains no muscle members",
            "target_members_complete": False, "direction_and_coherent_update_rule_complete": False,
            "replaces_or_supplements_v1": "NOT_SPECIFIED_IN_AUTHORITATIVE_S1",
            "definition_status": BLOCKER,
        },
    ]


def semantics_rows() -> list[dict[str, Any]]:
    operating = {row["muscle"]: row for row in read_csv(OPERATING)}
    lmax = "|".join(f"{name}={operating[name]['lmax_normalized']}" for name in operating)
    fpmax_names = ("recfem_r", "bflh_r", "semimem_r", "semiten_r")
    fpmax = "|".join(f"{name}={operating[name]['fpmax_dimensionless']}" for name in fpmax_names)
    return [
        {
            "factor_index": 1, "factor_name": EXPECTED_S1_FACTORS[0], "actual_fields": EXPECTED_S1_FIELDS[0],
            "nominal_values": lmax, "unit_or_normalized_semantic": "dimensionless MuJoCo built-in-muscle lmax; not physiological optimal fiber length",
            "muscle_body_grouping": "seven audited right biarticular muscles", "behavior_changed": "normalized muscle curve support/shape mapping",
            "expected_effect": "potentially configuration-dependent", "compilation_or_recalibration": "runtime field mutation possible; coherent target/update convention must be frozen",
            "evidence_level": "E3", "population_range": "NOT AVAILABLE", "semantic_status": "MODEL_FIELD_DEFENSIBLE_FACTOR_OPERATION_INCOMPLETE",
        },
        {
            "factor_index": 2, "factor_name": EXPECTED_S1_FACTORS[1], "actual_fields": EXPECTED_S1_FIELDS[1],
            "nominal_values": fpmax, "unit_or_normalized_semantic": "dimensionless maximum normalized passive-force parameter; not tissue stiffness",
            "muscle_body_grouping": "recfem_r versus bflh_r/semimem_r/semiten_r", "behavior_changed": "relative passive-force contribution through different biarticular paths",
            "expected_effect": "potentially configuration-dependent", "compilation_or_recalibration": "runtime field mutation possible; balance normalization/sign convention missing",
            "evidence_level": "E3", "population_range": "NOT AVAILABLE", "semantic_status": "MODEL_FIELD_DEFENSIBLE_FACTOR_OPERATION_INCOMPLETE",
        },
        {
            "factor_index": 3, "factor_name": EXPECTED_S1_FACTORS[2], "actual_fields": EXPECTED_S1_FIELDS[2],
            "nominal_values": "NOT_RESOLVABLE_WITHOUT_TARGET_MEMBERS", "unit_or_normalized_semantic": "MuJoCo muscle force scale in N; not directly measured patient strength",
            "muscle_body_grouping": "NOT SPECIFIED", "behavior_changed": "relative hip monoarticular antagonist force mapping",
            "expected_effect": "potentially configuration-dependent", "compilation_or_recalibration": "runtime field mutation possible only after exact group/update rule",
            "evidence_level": "E2/E3", "population_range": "NOT AVAILABLE", "semantic_status": BLOCKER,
        },
        {
            "factor_index": 4, "factor_name": EXPECTED_S1_FACTORS[3], "actual_fields": EXPECTED_S1_FIELDS[3],
            "nominal_values": "NOT_RESOLVABLE_WITHOUT_TARGET_MEMBERS", "unit_or_normalized_semantic": "MuJoCo muscle force scale in N; not directly measured patient strength",
            "muscle_body_grouping": "NOT SPECIFIED", "behavior_changed": "relative knee monoarticular antagonist force mapping",
            "expected_effect": "potentially configuration-dependent", "compilation_or_recalibration": "runtime field mutation possible only after exact group/update rule",
            "evidence_level": "E2/E3", "population_range": "NOT AVAILABLE", "semantic_status": BLOCKER,
        },
    ]


def range_rows() -> list[dict[str, Any]]:
    requirements = {
        1: "model-specific calibration from measured fiber/tendon behavior to MuJoCo normalized lmax",
        2: "muscle-family passive force-length calibration and mapping to fpmax",
        3: "human architecture/PCSA or validated force-capacity distribution plus exact group definition",
        4: "human architecture/PCSA or validated force-capacity distribution plus exact group definition",
    }
    return [{
        "factor_index": index, "factor_name": name, "population_range": "NOT AVAILABLE",
        "pilot_diagnostic_level": "NOT FROZEN", "range_status": "RANGE_REQUIRES_EXTERNAL_EVIDENCE",
        "required_evidence_or_calibration": requirements[index],
        "pilot_level_is_population_bound": False, "reason_not_frozen": BLOCKER,
    } for index, name in enumerate(EXPECTED_S1_FACTORS, 1)]


def protocol(frozen: dict[str, str]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID, "scientific_role": "PREREGISTRATION_PILOT_DESIGN_ONLY",
        "formal_outcome": OUTCOME, "blocker": BLOCKER,
        "authoritative_source": str(SCHEMES.relative_to(ROOT)),
        "authoritative_source_sha256": frozen["authoritative_schemes"],
        "frozen_before_any_structural_pilot_scientific_outcome": True,
        "s1_scheme_id": "S1_MINIMAL_STRUCTURAL", "s1_declared_dimensionality": 4,
        "exact_reconstruction_gate": {
            "factor_names_complete": True, "model_fields_complete": True,
            "affected_target_members_complete": False, "factor_direction_update_rules_complete": False,
            "status": BLOCKER, "fail_closed_action": "stop pilot design; do not guess targets, levels, or execution plan",
        },
        "preserved_results": [
            "V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "HETEROGENEITY_LIMITATION_DOMINANT",
            "CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE", "CURRENT_HETEROGENEITY_TRAJECTORY_INTERACTION_LIMITED",
        ],
        "scope_guards": {
            "cohort_v1_modified": False, "new_virtual_subjects": 0, "cohort_v2_generated": False,
            "new_landscape_generated": False, "five_parameter_or_ml_training": False,
            "bo_run": False, "held_out_scientific_access": 0, "robot_or_hardware": False,
            "pilot_scientific_outcome_executed": False,
        },
        "population_range_vs_diagnostic_level": {
            "population_ranges_frozen": False, "diagnostic_levels_frozen": False,
            "reason": BLOCKER, "pilot_levels_must_never_be_labeled_population_or_patient_ranges": True,
        },
        "blocked_design_elements": [
            "diagnostic perturbation levels", "deterministic executable V3 subset",
            "numeric non-proportionality gates", "numeric gradient-rotation gates",
            "numeric configuration-dependence gates", "fallback levels", "diagnostic model/replay count",
        ],
        "next_action": "create a new versioned S1 definition amendment that explicitly freezes every muscle member, positive/negative group, coherent scaling/normalization rule, and V1 replace/supplement relation; do not overwrite the frozen source",
        "pilot_execution_authorized": False,
    }


def report() -> str:
    return f"""# MyoLeg Structural Heterogeneity Pilot Design V1

## Formal outcome

**{OUTCOME}**

Blocking status: **{BLOCKER}**.

This is a scientifically required fail-closed result, not a failed simulation. No pilot truth was executed. The authoritative S1 artifact supplies four factor names, four field families and a declared dimensionality of four, but it does not fully specify the target members and signed/coherent update operation needed to create deterministic models.

## Why the design stopped

- Factor 1 identifies the biarticular `lmax` field and companion audit identifies seven muscles, but the authoritative S1 does not freeze how their heterogeneous nominal values are shifted or normalized coherently.
- Factor 2 identifies rectus-versus-hamstring `fpmax` members through a companion audit, but the positive/negative balance convention and normalization constraint are not frozen.
- Factors 3 and 4 name hip/knee monoarticular antagonist relative F0, but no exact muscle membership, agonist/antagonist side, or update convention exists in the frozen artifacts.
- The frozen S1 does not explicitly say which factors replace or supplement V1 factors.

Guessing these definitions from anatomy would create a new S1 after seeing prior evidence, contrary to the authoritative-source and no-redefinition rules. Therefore no diagnostic level, trajectory subset, numeric scientific gate, fallback or replay count was made executable.

## Q1. Exact frozen S1 factors and fields

The four exact names and field families are preserved in `S1_EXACT_FACTOR_DEFINITION.csv`. Exact target/update definitions are incomplete, so S1 cannot be reconstructed deterministically.

## Q2. Which factors have defensible model semantics?

All four point to real MuJoCo fields. Factors 1 and 2 have partially resolved muscle groups. Field semantics alone are insufficient: factors 3/4 lack targets, and factors 1/2 lack complete coherent operations.

## Q3. Population-range evidence versus diagnostic perturbations

No S1 factor has an evidence-backed population range. All remain `RANGE_REQUIRES_EXTERNAL_EVIDENCE`. Synthetic diagnostic perturbations were not frozen because the exact factor operators are incomplete.

## Q4. Exact perturbation levels

**Not frozen.** Any numeric levels now would attach to ambiguous operators and violate `S1_DEFINITION_INCOMPLETE` fail-closed handling.

## Q5. Deterministic V3 subset

**Not frozen as executable.** The geometry-only selection rule can be designed later, after S1 is made deterministic; no candidate ID was selected from J, oracle or rank here.

## Q6. Scientific gates

**Numeric gates not frozen.** Required endpoint families are documented in the blocked gate artifact, but they are deliberately non-executable until exact factor reconstruction succeeds.

## Q7. Integrity and fallback

Required integrity categories are documented. No fallback level exists because neither primary level nor exact factor operation is frozen. The fail-closed action is zero execution.

## Q8. Model and replay count

`0` structural diagnostic models and `0` replays are authorized or executed in this stage.

## Q9. Cohort V2 admission rule

A future factor must have defensible semantics, integrity PASS, demonstrated configuration-dependent non-proportional response, and a range-calibration pathway. A missing direct range may yield `COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP`, never automatic cohort admission. `NEW_VERSION_REQUIRED = true` remains frozen.

## Q10. Is the pilot ready?

**No.** A new versioned S1-definition amendment must first freeze exact target membership, sign/direction, coherent normalization and V1 relationship for all four factors. The original S1/V1 artifacts must not be overwritten.

## Stop state

- Pilot executed: **no**.
- Structural models/replays: **0 / 0**.
- New subjects/cohort/landscape: **none**.
- Held-out scientific access: **0**.
- V1 negative evidence: preserved.
- Robot/hardware: untouched.
"""


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs()
    factors = exact_factor_rows()
    if all(row["target_members_complete"] and row["direction_and_coherent_update_rule_complete"] for row in factors):
        raise RuntimeError("expected fail-closed S1 incompleteness was not observed")

    write_json(OUTPUT / "STRUCTURAL_HETEROGENEITY_PILOT_PROTOCOL.json", protocol(frozen))
    write_csv(OUTPUT / "S1_EXACT_FACTOR_DEFINITION.csv", factors)
    write_csv(OUTPUT / "PILOT_FACTOR_SEMANTICS.csv", semantics_rows())
    write_csv(OUTPUT / "PILOT_RANGE_EVIDENCE.csv", range_rows())
    write_json(OUTPUT / "PILOT_DIAGNOSTIC_LEVELS.json", {
        "status": "NOT_FROZEN", "blocker": BLOCKER, "levels": [],
        "population_ranges": [], "pilot_level_is_population_range": False,
        "finite_fallback_level": None, "execution_authorized": False,
    })
    write_csv(OUTPUT / "PILOT_V3_TRAJECTORY_SUBSET.csv", [{
        "status": "NOT_FROZEN", "candidate_id": "", "beta_flex": "", "beta_extend": "",
        "selection_basis": "geometry-only rule required after S1 completion; no J/oracle/rank used",
        "blocker": BLOCKER, "execution_authorized": False,
    }])
    write_json(OUTPUT / "NONPROPORTIONALITY_GATES.json", {
        "status": "NOT_NUMERICALLY_FROZEN", "blocker": BLOCKER, "execution_authorized": False,
        "required_metric_families": {
            "proportional_and_affine": ["hip torque RMS", "knee torque RMS", "combined diagnostic if mathematically frozen", "predeclared physical component"],
            "fit_statistics": ["R2", "NRMSE", "residual range", "residual RMS"],
            "gradient_rotation": ["cosine similarity", "angle change", "sign change", "relative component change"],
            "configuration_dependence": ["delta response range", "SD", "normalized variation", "beta dependence"],
        },
        "factor_outcomes": ["STRUCTURALLY_INFORMATIVE", "MAGNITUDE_ONLY", "INCONCLUSIVE", "INVALID"],
        "personalization_is_not_a_success_metric": True,
    })
    write_json(OUTPUT / "PILOT_INTEGRITY_GATES.json", {
        "status": "CATEGORIES_FROZEN_NUMERIC_EXECUTION_BLOCKED", "blocker": BLOCKER,
        "required_gates": ["compilation", "finite state", "no solver warning", "no unexpected contact",
                           "no unintended joint-limit activation", "equality residual", "tendon/actuator finite",
                           "frozen V3 task invariants", "exact truth decomposition consistency"],
        "primary_level": None, "fallback_level": None, "maximum_fallback_attempts": 0,
        "failure_action": "INVALID_AND_STOP", "execution_authorized": False,
    })
    write_json(OUTPUT / "COHORT_V2_FACTOR_ADMISSION_RULES.json", {
        "new_version_required": True, "future_identity": "MYOLEG_VIRTUAL_PATIENT_COHORT_V2",
        "automatic_admission": False,
        "required_conjunction": ["model semantics defensible", "integrity PASS",
                                 "configuration-dependent non-proportional response demonstrated",
                                 "future range calibration pathway exists"],
        "range_gap_status": "COHORT_V2_CANDIDATE_WITH_RANGE_EVIDENCE_GAP",
        "additional_review": ["evidence", "interpretability", "reproducibility",
                              "truth-learner independence", "dimensionality", "calibration feasibility"],
        "old_v1_heldout_is_automatic_confirmation": False,
    })
    write_json(OUTPUT / "PILOT_EXECUTION_PLAN.json", {
        "status": "BLOCKED", "blocker": BLOCKER, "execution_authorized": False,
        "structural_diagnostic_model_count": 0, "trajectory_replay_count": 0,
        "new_virtual_subject_count": 0, "cohort_v2_generated": False,
        "scientific_pilot_outcome_executed": False,
        "required_resolution": [
            "freeze exact members for hip monoarticular antagonist F0",
            "freeze exact members for knee monoarticular antagonist F0",
            "freeze positive/negative direction and invariant normalization for every balance/profile",
            "freeze whether each factor replaces or supplements V1",
        ],
        "next_stage": None, "pilot_stage_not_authorized": "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_V1",
    })
    write_json(OUTPUT / "SOURCE_AND_EVIDENCE_METADATA.json", {
        "authoritative_source": str(SCHEMES.relative_to(ROOT)),
        "authoritative_source_sha256": frozen["authoritative_schemes"],
        "companion_frozen_sources": {name: {"path": str(FROZEN_PATHS[name].relative_to(ROOT)), "sha256": frozen[name]}
                                     for name in ("biarticular_audit", "muscle_group_audit", "operating_audit", "evidence_sources")},
        "inherited_evidence_source_count": len(read_csv(EVIDENCE)),
        "new_external_evidence_used_to_redefine_s1": False,
        "personalization_outcome_used": False, "held_out_scientific_access_count": 0,
    })
    (OUTPUT / "MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_REPORT.md").write_text(report(), encoding="utf-8")
    write_json(OUTPUT / "metadata.json", {
        "stage_id": STAGE_ID, "outcome": OUTCOME, "blocker": BLOCKER,
        "authoritative_s1_sha256": frozen["authoritative_schemes"], "frozen_inputs": frozen,
        "s1_factor_count_declared": 4, "s1_exactly_reconstructable": False,
        "pilot_design_ready": False, "pilot_execution_authorized": False,
        "diagnostic_levels_frozen": False, "trajectory_subset_frozen": False,
        "numeric_scientific_gates_frozen": False, "diagnostic_models_generated": 0,
        "trajectory_replays_executed": 0, "new_virtual_subjects": 0,
        "cohort_v2_generated": False, "new_landscape_generated": False,
        "held_out_scientific_access_count": 0, "objective_or_normalization_modified": False,
        "v3_parameterization_or_domain_modified": False, "cohort_v1_modified": False,
        "five_parameter_or_ml_training": False, "bo_run": False, "robot_or_hardware": False,
        "preserved_negative_results": ["V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "HETEROGENEITY_LIMITATION_DOMINANT"],
        "analysis_code_sha256": sha256(Path(__file__)), "automatic_commit": False,
    })
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUTPUT / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"stage_id": STAGE_ID, "outcome": OUTCOME, "blocker": BLOCKER,
                      "pilot_execution_authorized": False, "model_count": 0,
                      "replay_count": 0, "held_out_scientific_access_count": 0}, indent=2))


if __name__ == "__main__":
    build()
