"""Build the evidence-only MyoLeg stop-or-pivot formulation audit.

This module reads frozen reports and decisions only.  It does not import a
simulator, load truth arrays, compute an oracle, train a model, optimize a
trajectory, or access robot/hardware code.  ``--prepare`` freezes the decision
protocol and exact evidence-file hashes.  ``--execute`` verifies that freeze
before writing the route assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


STAGE_ID = "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_AUDIT_V1"
PROTOCOL_ID = "STOP_OR_PIVOT_DECISION_PROTOCOL_V1"
PRIMARY_DECISION = "PIVOT_TO_MEASUREMENT_DRIVEN_PERSONALIZATION"
NEXT_STAGE = "PERSONALIZED_REHABILITATION_RESEARCH_FORMULATION_V2"
SYNTHETIC_OPTION_DECISION = "STOP"
UNIVERSAL_OPTION_DECISION = "RETAIN_AS_LIMITED_NONPERSONALIZED_SECONDARY_BRANCH"
PINN_DECISION = "PINN_NOT_YET_JUSTIFIED"
BO_DECISION = "PERSONALIZED_BO_NOT_YET_JUSTIFIED_WITHOUT_SUBJECT_FEEDBACK"
FROZEN_S1_SHA256 = "3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763"

# Patched exactly once after --prepare.  Execution fails closed until frozen.
FROZEN_PROTOCOL_SHA256 = "029ffe5bcc91ca4ffc8d9db3216b7df0c92c73f0aa253a26a2d2d9b24f173503"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_personalization_formulation_stop_or_pivot_audit_v1"
PROTOCOL_PATH = OUTPUT / "STOP_OR_PIVOT_DECISION_PROTOCOL.json"
INPUT_VERIFICATION_PATH = OUTPUT / "EVIDENCE_INPUT_VERIFICATION.json"
ACCESS_AUDIT_PATH = OUTPUT / "HELD_OUT_ACCESS_AUDIT.json"


EVIDENCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "evidence_id": "E01_MYOLEG_COHORT_V1",
        "path": "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_REPORT.md",
        "required_markers": ("MYOLEG_VIRTUAL_PATIENT_COHORT_V1_VALID_WITH_LIMITATIONS", "Reference-response heterogeneity"),
        "role": "independent simulator cohort has descriptive absolute mechanical heterogeneity, not patient preference truth",
    },
    {
        "evidence_id": "E02_V2_NECESSITY",
        "path": "external_simulation_audits/myoleg_v2_personalization_necessity_audit_v1/MYOLEG_V2_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md",
        "required_markers": ("PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "Relative common regret"),
        "role": "V2 development oracle upper-bound audit",
    },
    {
        "evidence_id": "E03_V2_ROOT_CAUSE",
        "path": "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1/MYOLEG_V2_PERSONALIZATION_SIGNAL_ROOT_CAUSE_REPORT.md",
        "required_markers": ("PERSONALIZATION_SIGNAL_ROOT_CAUSE_IDENTIFIED", "99.939077%", "0.033114%"),
        "role": "dominant common candidate effect and weak interaction diagnosis",
    },
    {
        "evidence_id": "E04_V2_PARAMETERIZATION",
        "path": "external_simulation_audits/myoleg_v2_trajectory_parameterization_boundary_audit_v1/MYOLEG_V2_TRAJECTORY_PARAMETERIZATION_BOUNDARY_AUDIT_REPORT.md",
        "required_markers": ("TRAJECTORY_PARAMETERIZATION_ROOT_CAUSE_SUPPORTED", "0.997742", "NOT_ADEQUATE_FOR_CURRENT_PERSONALIZATION_QUESTION"),
        "role": "amplitude/ROM confounding and fixed-task design rationale",
    },
    {
        "evidence_id": "E05_V3_PARAMETERIZATION",
        "path": "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_TRAJECTORY_PARAMETERIZATION_DESIGN_REPORT.md",
        "required_markers": ("beta_flex", "beta_extend", "625"),
        "role": "validated fixed-ROM branch-aware V3 coordination family",
    },
    {
        "evidence_id": "E06_V3_LANDSCAPE",
        "path": "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/MYOLEG_V3_DEVELOPMENT_TRUTH_LANDSCAPE_GENERATION_REPORT.md",
        "required_markers": ("24", "625", "held-out"),
        "role": "complete valid development landscape and sealed held-out boundary",
    },
    {
        "evidence_id": "E07_V3_NECESSITY",
        "path": "external_simulation_audits/myoleg_v3_personalization_necessity_audit_v1/MYOLEG_V3_PERSONALIZATION_NECESSITY_AUDIT_REPORT.md",
        "required_markers": ("V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED", "0.135074%", "24/24", "0.03, -0.03"),
        "role": "fixed-task mechanical personalization necessity upper bound",
    },
    {
        "evidence_id": "E08_OBJECTIVE_HETEROGENEITY",
        "path": "external_simulation_audits/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1/MYOLEG_OBJECTIVE_AND_HETEROGENEITY_DECISION_REPORT.md",
        "required_markers": ("HETEROGENEITY_LIMITATION_DOMINANT", "CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE"),
        "role": "objective versus heterogeneity diagnosis",
    },
    {
        "evidence_id": "E09_HETEROGENEITY_EXPANSION",
        "path": "external_simulation_audits/myoleg_musculoskeletal_heterogeneity_expansion_design_audit_v1/MYOLEG_MUSCULOSKELETAL_HETEROGENEITY_EXPANSION_DESIGN_REPORT.md",
        "required_markers": ("offline design/model-semantics/evidence audit", "V3_PERSONALIZATION_NECESSITY_NOT_SUPPORTED"),
        "role": "structural factor design and population-range evidence gaps",
    },
    {
        "evidence_id": "E10_AMENDED_S1",
        "path": "external_simulation_audits/myoleg_s1_structural_factor_definition_amendment_v1/S1_STRUCTURAL_DEFINITION_AMENDED_V1.json",
        "required_markers": ("S1F1_BIARTICULAR_LMAX", "S1F4_KNEE_MONO_ANTAGONIST_F0"),
        "role": "authoritative structural factor semantics",
        "exact_sha256": FROZEN_S1_SHA256,
    },
    {
        "evidence_id": "E11_PILOT_DESIGN_V2",
        "path": "external_simulation_audits/myoleg_structural_heterogeneity_pilot_design_v2/MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2_REPORT.md",
        "required_markers": ("READY_WITH_EVIDENCE_GAPS", "DIAGNOSTIC_LEVEL_READY"),
        "role": "preregistered structural pilot design and gates",
    },
    {
        "evidence_id": "E12_STRUCTURAL_PILOT_REPORT",
        "path": "external_simulation_audits/myoleg_structural_heterogeneity_pilot_v1/MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_REPORT.md",
        "required_markers": ("STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED", "MAGNITUDE_ONLY", "Fallback models used: **0**"),
        "role": "latest structural mechanism result",
    },
    {
        "evidence_id": "E13_STRUCTURAL_PILOT_DECISION",
        "path": "external_simulation_audits/myoleg_structural_heterogeneity_pilot_v1/FINAL_PILOT_DECISION.json",
        "required_markers": ("STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED", '"MAGNITUDE_ONLY": 4'),
        "role": "machine-readable latest stop condition",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_checksum_path(path: Path) -> Path | None:
    candidate = path.parent / "checksums.sha256"
    return candidate if candidate.is_file() else None


def verify_evidence_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in EVIDENCE_SPECS:
        path = ROOT / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing frozen evidence: {spec['path']}")
        content = path.read_text(encoding="utf-8")
        missing = [marker for marker in spec["required_markers"] if marker not in content]
        if missing:
            raise RuntimeError(f"evidence semantic marker mismatch for {spec['evidence_id']}: {missing}")
        digest = sha256_file(path)
        if spec.get("exact_sha256") and digest != spec["exact_sha256"]:
            raise RuntimeError(f"authoritative SHA mismatch for {spec['evidence_id']}")
        checksum_path = source_checksum_path(path)
        rows.append(
            {
                "evidence_id": spec["evidence_id"],
                "path": spec["path"],
                "sha256": digest,
                "role": spec["role"],
                "required_markers": list(spec["required_markers"]),
                "semantic_markers_pass": True,
                "source_checksums_path": str(checksum_path.relative_to(ROOT)) if checksum_path else None,
                "source_checksums_sha256": sha256_file(checksum_path) if checksum_path else None,
            }
        )
    return rows


def build_protocol(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "audit_type": "SCIENTIFIC_FORMULATION_PROJECT_DECISION_AUDIT",
        "frozen_latest_conclusion": "STRUCTURAL_HETEROGENEITY_PILOT_NOT_SUPPORTED",
        "core_question": "Is MyoLeg-generated parameter heterogeneity plus a mechanical torque objective scientifically justified as the primary source of subject-specific trajectory preference?",
        "evidence_files": evidence,
        "route_candidates": {
            "OPTION_A_CONTINUE_SYNTHETIC_MYOLEG_PERSONALIZATION": ["CONTINUE", "LIMITED_CONTINUATION", "STOP"],
            "OPTION_B_UNIVERSAL_MECHANICAL_TRAJECTORY_OPTIMIZATION": "retain V3 and common mechanical objective without a strong personalization claim",
            "OPTION_C_MEASUREMENT_DRIVEN_PERSONALIZATION": "obtain subject-specific signal from executed-trial measurements or explicit feedback",
        },
        "decision_rule": {
            "continue_synthetic_only_if": [
                "independent frozen evidence demonstrates material subject-by-trajectory ordering",
                "heterogeneity ranges are independently defensible",
                "continuation does not depend on outcome-driven factor/range redesign",
            ],
            "universal_option_is_secondary_if": [
                "a common mechanical ordering is supported",
                "personalization necessity is not supported",
                "claims remain offline/common and do not imply comfort or human benefit",
            ],
            "measurement_pivot_is_primary_if": [
                "repeated independent synthetic audits do not support a personalized mechanical ordering",
                "the original multi-round individual-improvement goal still requires subject-specific observations",
                "a falsifiable measured-data formulation can keep physics as prior rather than preference truth",
            ],
        },
        "comfort_semantics": {
            "mechanical_measurements_are_comfort_truth": False,
            "comfort_or_preference_claim_requires_direct_subject_feedback": True,
            "tactile_pressure_role": "MEASURED_INTERACTION_FEATURE_OR_POSSIBLE_COMFORT_CORRELATE",
        },
        "prohibited": [
            "increase_S1_z", "design_outcome_driven_S2_or_S3", "generate_Cohort_V2",
            "modify_objective_to_induce_oracle_diversity", "modify_V3_candidate_domain",
            "train_five_parameter_or_NN_or_PINN", "run_BO", "read_held_out_scientific_truth",
            "access_robot_or_hardware", "start_next_stage",
        ],
        "allowed_operations": [
            "read_and_hash_named_frozen_reports_and_decisions",
            "compare_route_semantics_and evidence limitations",
            "write_conceptual_architecture_and_project_decision",
        ],
        "protocol_frozen_before_route_assessment_artifacts_written": True,
        "does_not_claim_frozen_before_historical_evidence": True,
    }


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"prepare requires an empty output directory: {OUTPUT}")
    evidence = verify_evidence_files()
    protocol = build_protocol(evidence)
    atomic_json(PROTOCOL_PATH, protocol)
    atomic_json(
        INPUT_VERIFICATION_PATH,
        {
            "stage_id": STAGE_ID,
            "all_required_inputs_present": True,
            "all_semantic_markers_pass": True,
            "authoritative_s1_sha256": FROZEN_S1_SHA256,
            "authoritative_s1_sha_pass": True,
            "evidence_file_count": len(evidence),
            "evidence_files": evidence,
            "scientific_arrays_read": 0,
            "simulator_replays_run": 0,
        },
    )
    atomic_json(
        ACCESS_AUDIT_PATH,
        {
            "stage_id": STAGE_ID,
            "held_out_file_access_count": 0,
            "held_out_scientific_access_count": 0,
            "held_out_arrays_loaded": 0,
            "held_out_outcomes_read": 0,
            "oracle_or_rank_or_regret_computed": False,
            "operation": "NO_HELD_OUT_ACCESS_REQUIRED_FOR_EVIDENCE_ONLY_FORMULATION_AUDIT",
        },
    )
    print(json.dumps({"protocol_sha256": sha256_file(PROTOCOL_PATH), "evidence_files": len(evidence)}, indent=2))


def verify_frozen_protocol_and_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if FROZEN_PROTOCOL_SHA256 == "TO_BE_FROZEN_AFTER_PREPARE":
        raise RuntimeError("protocol SHA has not been frozen into the implementation")
    actual_protocol_sha = sha256_file(PROTOCOL_PATH)
    if actual_protocol_sha != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol SHA mismatch: {actual_protocol_sha}")
    protocol = read_json(PROTOCOL_PATH)
    current = verify_evidence_files()
    frozen = protocol["evidence_files"]
    if current != frozen:
        raise RuntimeError("frozen evidence files changed after protocol preparation")
    access = read_json(ACCESS_AUDIT_PATH)
    if access["held_out_scientific_access_count"] != 0:
        raise RuntimeError("held-out scientific access must remain zero")
    return protocol, current


def cumulative_evidence_chain(evidence: list[dict[str, Any]]) -> str:
    source_lines = "\n".join(
        f"- `{row['evidence_id']}` — `{row['path']}` — SHA `{row['sha256']}`" for row in evidence
    )
    return f"""# Cumulative Evidence Chain

This audit uses frozen reports/decisions only. It reads no held-out scientific arrays and performs no new simulation or optimization.

## Evidence progression

1. The original same-family five-parameter truth raised circular-validation concern; direct damping/stiffness-like MyoLeg factors were therefore excluded from the independent primary cohort design.
2. MyoLeg Cohort V1 retained real simulator-level absolute response variation (for example reference hip torque RMS CV 4.305%), but its ranges were structured synthetic ranges, not a patient population distribution or comfort truth.
3. V2 necessity: `PERSONALIZATION_NECESSITY_NOT_SUPPORTED`; 24 development subjects shared one exact oracle and common regret was zero.
4. V2 root cause: candidate-main variance 99.939077%, subject-by-candidate interaction 0.033114%, with common monotonic direction dominant.
5. V2 parameterization: ROM/extrema described 0.997742 (99.7742%) of the common candidate effect; two amplitude coordinates changed the rehabilitation task.
6. V3 removed that confound using fixed-ROM branch-aware `beta_flex, beta_extend`; the complete development landscape was 24 subjects x 625 candidates.
7. V3 necessity remained negative: 24/24 shared `beta=[+0.03,-0.03]`, common and cross-oracle-transfer regret were zero, and interaction was only 0.135074%.
8. Objective diagnostics retained ordering across raw/normalized/time/peak/component views. Formal result: `CURRENT_OBJECTIVE_INFORMATION_RETENTION_ADEQUATE` and `HETEROGENEITY_LIMITATION_DOMINANT`.
9. More structural S1 semantics were independently amended and preregistered.
10. The 9-model/117-replay structural pilot passed integrity with fallback 0, yet all four factors were `MAGNITUDE_ONLY`; none passed the frozen structural-informativeness logic.

## Convergent interpretation

The evidence does not say that humans lack individual differences. It says the current synthetic MyoLeg factorization plus frozen torque objective has repeatedly failed to provide a material subject-specific coordination-path ordering. Continuing to redesign synthetic factors until oracle diversity appears would be outcome-driven and scientifically weak.

## Frozen source inventory

{source_lines}
"""


def synthetic_assessment() -> str:
    return """# Option A — Continue Synthetic MyoLeg Personalization

## Decision

`STOP`

This stops the use of synthetic MyoLeg parameter heterogeneity as the **primary source of subject-specific preference truth**. It does not discard MyoLeg as an engineering simulator.

## Evidence supporting limited scientific utility

- MyoLeg is independent of the simplified five-parameter learner and provides a valuable mechanics stress environment.
- Cohort V1 contains absolute mechanical response heterogeneity.
- V3 provides a valid fixed-task coordination family and the negative audits are reproducible.

## Evidence against continuation as primary personalization truth

- V2 and V3 both fail their frozen personalization-necessity criteria.
- V3 has one oracle for 24/24 subjects, zero common regret, zero transfer regret, and near-identical rankings.
- The objective audit did not uncover a large subject-specific ordering hidden by RMS or normalization.
- Four more structural preregistered S1 factors all remained magnitude-only.
- S1 diagnostic levels lack population calibration and cannot be promoted to virtual-patient bounds.

## Risk and resource assessment

- Outcome-driven factor expansion risk: **HIGH**.
- Repeatedly redesigning subjects until oracle diversity appears: **NOT ACCEPTABLE**.
- Interpretability if continued: **LOW**, because a positive result would be conditional on post-negative redesign choices.
- Publishability: strongest as a transparent negative/formulation result, weak as manufactured synthetic personalization evidence.
- Resource cost: high relative to the evidence value of another arbitrary factor/range search.

Increasing z, inventing S2/S3 solely for different oracles, Cohort V2 generation with uncalibrated ranges, and objective-weight search solely to create diversity are explicitly stopped.
"""


def universal_option() -> str:
    return """# Option B — Universal Mechanical Trajectory Optimization

## Assessment

`RETAIN_AS_LIMITED_NONPERSONALIZED_SECONDARY_BRANCH`

The current development simulations support a common mechanical ordering within the frozen V3 domain: all 24 development models select `beta=[+0.03,-0.03]`, and the common-candidate regret is zero. This can motivate a population/common offline mechanical trajectory-design question.

Important limits:

- The common optimum is on the frozen candidate boundary; it is not an unconstrained physical optimum.
- The result is simulator-development evidence, not robot, human, comfort, safety, or clinical validation.
- A shared torque-objective trajectory does not answer the advisor's original multi-round individual-improvement goal.
- Novelty is limited if presented only as a common two-parameter grid optimum.

Useful contribution: V3 offers a clean fixed-ROM coordination family, and the evidence shows that task-amplitude confounding can be removed. Universal mechanical optimization may remain a baseline or secondary engineering result, not the primary personalization claim.
"""


def measurement_option() -> str:
    return f"""# Option C — Measurement-Driven Personalization

## Recommendation

`{PRIMARY_DECISION}`

This is the primary pivot because it preserves the original individual, multi-round research goal without claiming that synthetic MyoLeg parameters are patient preference truth. The pivot is scientifically plausible, not already validated: actual measured data may still show no learnable personalization signal.

## Mechanical-only primary question

Given a fixed rehabilitation task and the offline-feasibility-screened low-dimensional V3 coordination-path family, can a physics-informed subject model and low-budget adaptive exploration use only that subject's executed-trial force, torque, pressure, and tracking measurements to select a trajectory with a better prespecified **mechanical interaction metric** than fixed-reference, common-trajectory, random, and non-adaptive baselines under equal trial budget?

Required data: synchronized executed-trial interaction mechanics and state/tracking data. No comfort claim is permitted.

## Preference/comfort primary question

Given the same fixed task and feasible V3 family, can a low-budget preference-learning method use explicit per-subject ratings or pairwise trajectory choices, with mechanical feasibility constraints, to identify a trajectory the subject reports as preferable under equal trial budget?

Required data: direct rating, pairwise choice, or equivalent explicit human response, plus separate mechanical/safety monitoring. Pressure or torque alone is not preference truth.

## Conceptual architecture

1. **Fixed task/family:** retain validated `beta_flex, beta_extend`; do not change ROM to manufacture benefit.
2. **Physics prior:** analytical dynamics, MyoLeg, or a gray-box model provides feasibility and an initial mechanical prediction.
3. **Subject adaptation:** start with the existing five-parameter gray-box identification concept, refit only from executed trials and revalidate its adequacy for measured data. Consider a residual NN or physics-informed residual NN only if measured residual structure and sufficient data justify it.
4. **Low-budget selection:** compare random exploration, BO without a subject model, model-informed BO, and preference-based BO under the same trial budget.
5. **Feedback target:** mechanical measurements for mechanical personalization; direct feedback for preference/comfort; both remain constrained by independent feasibility and safety gates.

## Fair future validation philosophy

Replace “does a synthetic cohort contain different oracle trajectories?” with: “given actual subject-specific observations, does the adaptive method predict or select better than non-personalized and non-adaptive baselines under equal trial budget?”

Conceptual baselines: fixed reference, population/common trajectory, random exploration, model-only prediction, BO without subject model, gray-box plus BO, and—only when justified—residual/PINN plus BO.

This audit does not define a human protocol, trial count, safety threshold, or robot release gate. Those remain separate prerequisites.
"""


def pinn_role() -> str:
    return f"""# PINN Role Reassessment

## Decision

`{PINN_DECISION}`

There is currently no subject-specific measured dataset that gives a PINN a distinct scientific task. Training one now would add complexity without evidence that a physics residual is learnable or needed.

A PINN or physics-informed residual network becomes scientifically testable only after executed subject trials provide:

- synchronized inputs and mechanical targets;
- a prespecified train/validation separation;
- evidence that the gray-box physics baseline leaves systematic, repeatable residual structure;
- enough independent observations to compare against simpler residual and gray-box baselines.

Its future task would be `physics baseline + subject-specific residual`, not generation of synthetic patient preference truth. It must demonstrate equal-budget predictive or selection benefit over the simpler model.
"""


def bo_role() -> str:
    return f"""# BO Role Reassessment

## Current decision

`{BO_DECISION}`

BO is a trajectory-selection mechanism; it does not create a personalized objective.

- **Mechanical BO:** optimize a prespecified measured force/torque/pressure interaction metric within fixed task and feasibility constraints.
- **Model-informed BO:** use a physics/gray-box prediction as prior or mean function, then update only from the subject's executed trials.
- **Preference-based BO:** optimize a latent preference inferred from direct ratings or pairwise choices, while mechanical quantities act as constraints/features rather than comfort labels.

Without real subject feedback, BO can optimize only a simulator/common mechanical objective. That is an offline method exercise, not evidence of personalized trajectory benefit.
"""


def tactile_role() -> str:
    return """# Tactile and Feedback Role

Tactile-array signals can provide local pressure magnitude, distribution, center-of-pressure movement, peak pressure, and pressure concentration. Their valid present role is:

`MEASURED_INTERACTION_FEATURE / POSSIBLE_COMFORT_CORRELATE`

Pressure is not automatically comfort, preference, tissue safety, or clinical effectiveness. A mechanical-load study may use pressure as a primary or secondary mechanical outcome if the metric and sensor validity are prespecified. A comfort/preference study additionally requires direct subject feedback such as a rating or pairwise choice.

Future human feedback implies independent ethics, safety, calibration, and robot-motion approval prerequisites. Current status remains `NOT_HUMAN_READY` and `NOT_ROBOT_APPROVED`.

If the target claim is comfort or preference, the dependency is explicitly `HUMAN_FEEDBACK_REQUIRED`; the minimum information is a per-trajectory rating, a pairwise trajectory choice, or an equivalent explicit response. This audit does not design or execute a human study.
"""


def retention_map() -> str:
    return """# Existing Work Retention Map

Nothing is deleted because of the pivot.

| Existing work | Future role | Evidence/claim boundary |
|---|---|---|
| 2-DOF supine rehabilitation model | MAIN_METHOD_MATERIAL | transparent mechanics and trajectory representation; not clinical validation |
| frozen measured asymmetric reference and ROM convention | MAIN_METHOD_MATERIAL | fixed task definition and provenance |
| V3 fixed-ROM branch-aware `beta_flex, beta_extend` | MAIN_METHOD_MATERIAL | low-dimensional coordination family; not yet human-safe |
| physics-informed gray-box identification framework | MAIN_METHOD_MATERIAL_WITH_REVALIDATION | candidate prior/adaptation layer; future evidence must come from executed trials |
| MyoLeg mapping and replay/truth semantics | SUPPORTING_MATERIAL | independent simulator mapping, feasibility and stress-test support |
| MyoLeg V1/V3 full development landscapes | SUPPORTING_MATERIAL | offline development evidence only; held-out remains sealed |
| V2 amplitude/ROM root-cause audit | NEGATIVE_RESULT_FORMULATION_EVIDENCE | explains why V3 fixed-task reformulation was necessary |
| V2/V3 personalization-necessity audits | NEGATIVE_RESULT_FORMULATION_EVIDENCE | synthetic mechanical personalization upper bound was not supported |
| objective-versus-heterogeneity audit | NEGATIVE_RESULT_FORMULATION_EVIDENCE | objective retained ordering; current heterogeneity was limiting |
| amended S1 and structural pilot | NEGATIVE_RESULT_FORMULATION_EVIDENCE | preregistered structural factors remained magnitude-only |
| old same-family five-parameter virtual truth | SHOULD_NOT_DOMINATE_FINAL_PAPER | circularity concern; historical development only |
| repeated synthetic factor/cohort redesign | STOPPED_PATH | must not be used to search for oracle diversity |
| unvalidated PINN/BO claims | SHOULD_NOT_DOMINATE_FINAL_PAPER | methods are future candidates, not completed evidence |
"""


def myoleg_role_payload() -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "KEEP": [
            "offline method development",
            "physics-prior implementation checks",
            "trajectory kinematic/dynamic feasibility support",
            "controlled stress testing and ablation",
            "simulation-based baseline construction with explicit evidence limits",
        ],
        "DOWNGRADE": [
            "patient population truth",
            "final personalization-necessity proof",
            "subject preference ground truth",
            "comfort oracle",
            "human safety or clinical effectiveness evidence",
        ],
        "STOP": [
            "arbitrary factor expansion to induce oracle diversity",
            "Cohort V2 under uncalibrated S1 ranges",
            "larger diagnostic z after a negative preregistered pilot",
            "V4/V5 redesign solely to manufacture personalization",
            "objective-weight search solely to manufacture different oracles",
        ],
        "myoleg_still_useful": True,
        "myoleg_is_future_patient_preference_truth": False,
    }


def final_decision_payload() -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "primary_recommendation": PRIMARY_DECISION,
        "core_question_answer": "NO_CURRENT_JUSTIFICATION_FOR_SYNTHETIC_MYOLEG_HETEROGENEITY_PLUS_TORQUE_OBJECTIVE_AS_PRIMARY_SUBJECT_PREFERENCE_SOURCE",
        "option_a_continue_synthetic_myoleg_personalization": SYNTHETIC_OPTION_DECISION,
        "option_b_universal_mechanical_trajectory_optimization": UNIVERSAL_OPTION_DECISION,
        "option_c_measurement_driven_personalization": "PRIMARY_RECOMMENDATION",
        "mechanical_and_comfort_personalization_separated": True,
        "mechanical_torque_is_comfort_truth": False,
        "human_feedback_required_for_comfort_preference_claim": True,
        "pinn_status": PINN_DECISION,
        "bo_status": BO_DECISION,
        "myoleg_patient_truth_role_stopped": True,
        "myoleg_engineering_support_role_retained": True,
        "cohort_v2_generated": False,
        "held_out_scientific_access_count": 0,
        "next_independent_stage": NEXT_STAGE,
        "next_stage_executed": False,
        "human_ready": False,
        "robot_approved": False,
    }


def report(protocol_sha: str) -> str:
    return f"""# MyoLeg Personalization Formulation Stop-or-Pivot Audit V1

## Primary project decision

`{PRIMARY_DECISION}`

Protocol SHA-256: `{protocol_sha}`

The current evidence does **not** justify continuing to treat MyoLeg-generated parameter heterogeneity plus the frozen mechanical torque objective as the primary source of subject-specific trajectory preference. This does not prove that real patients share one best trajectory. It shows that the current synthetic formulation has not produced the required subject-specific ordering despite independent simulator mapping, fixed-ROM V3 reformulation, objective diagnostics, and preregistered structural-factor testing.

## Route decisions

- Option A, synthetic MyoLeg personalization: `{SYNTHETIC_OPTION_DECISION}` as a primary personalization-truth program.
- Option B, universal mechanical optimization: `{UNIVERSAL_OPTION_DECISION}`.
- Option C, measurement-driven personalization: `PRIMARY_RECOMMENDATION`.

The pivot retains V3, physics models and MyoLeg as priors/engineering tools. The individualized signal must come from subject-specific executed-trial measurements or direct preference feedback—not from repeated synthetic factor redesign.

## Why this decision follows the frozen evidence

- V2 and V3 both failed formal personalization-necessity audits.
- V3 removed task-amplitude confounding but still produced one oracle for 24/24 subjects, zero common/transfer regret, 0.135074% interaction and nearly identical ranking.
- Raw, normalized, time-local, peak and component diagnostics did not reveal a strong ordering signal hidden by RMS; the objective was judged information-retaining and heterogeneity limiting.
- All four amended/preregistered structural factors were integrity-valid but `MAGNITUDE_ONLY`; no fallback was used.
- Population ranges for those S1 factors remain unavailable.

## Mechanical versus comfort personalization

Mechanical-load personalization can target prespecified force, torque or pressure metrics measured during executed trials. Comfort/preference personalization requires direct subject feedback. Torque and pressure may be features or constraints, but neither is comfort truth.

## Future research questions

**Mechanical-only:** Can physics-informed adaptation plus low-budget exploration use subject-specific interaction measurements to select a mechanically preferable V3 coordination path over equal-budget non-personalized and non-adaptive baselines?

**Preference/comfort:** Can direct ratings or pairwise choices, combined with mechanical constraints, support low-budget selection of a subject-reported preferable V3 coordination path?

These are separate studies with different data requirements and claim boundaries.

## PINN and BO

- PINN: `{PINN_DECISION}`. It becomes meaningful only after measured executed-trial residual data exist and simpler models provide a baseline.
- BO: `{BO_DECISION}`. BO selects against a supplied outcome; it cannot invent a personalized objective. Mechanical BO requires measured mechanical outcomes, while preference BO requires direct feedback.

## Existing work retained

The 2-DOF model, measured reference, fixed-ROM V3 family, MyoLeg mapping, truth semantics and negative audits remain useful. V3 and the reference can anchor a future formulation; MyoLeg remains useful for offline feasibility, physics-prior checks and stress testing. Past negative results remain evidence explaining the pivot and are not deleted.

## Explicit stops

- No larger S1 z or outcome-triggered S2/S3.
- No Cohort V2 with uncalibrated synthetic bounds.
- No V4/V5 or objective-weight search solely to induce different oracles.
- No unsupported PINN/BO personalization claim.

## Exact next independent stage

`{NEXT_STAGE}`

That stage should freeze the research question, primary outcome, data role, physics/PINN role, BO role, tactile role, trial budget and validation hierarchy. It was **not** executed here.

## Integrity and status

- Frozen evidence files verified: 13.
- Scientific arrays/replays/new optimizer runs: 0.
- Held-out scientific access: 0.
- New cohort/virtual subjects: 0.
- Objective, normalization, V3 domain, S1 and historical artifacts: unchanged.
- Human ready: no.
- Robot approved: no.
"""


def write_checksums() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(OUTPUT)}" for path in files]
    atomic_text(OUTPUT / "checksums.sha256", "\n".join(lines) + "\n")


def execute() -> None:
    protocol, evidence = verify_frozen_protocol_and_inputs()
    atomic_text(OUTPUT / "CUMULATIVE_EVIDENCE_CHAIN.md", cumulative_evidence_chain(evidence))
    atomic_text(OUTPUT / "SYNTHETIC_PERSONALIZATION_CONTINUATION_ASSESSMENT.md", synthetic_assessment())
    atomic_text(OUTPUT / "UNIVERSAL_OPTIMIZATION_OPTION.md", universal_option())
    atomic_text(OUTPUT / "MEASUREMENT_DRIVEN_PERSONALIZATION_OPTION.md", measurement_option())
    atomic_json(OUTPUT / "MYOLEG_FUTURE_ROLE.json", myoleg_role_payload())
    atomic_text(OUTPUT / "PINN_ROLE_REASSESSMENT.md", pinn_role())
    atomic_text(OUTPUT / "BO_ROLE_REASSESSMENT.md", bo_role())
    atomic_text(OUTPUT / "TACTILE_AND_FEEDBACK_ROLE.md", tactile_role())
    atomic_text(OUTPUT / "EXISTING_WORK_RETENTION_MAP.md", retention_map())
    atomic_json(OUTPUT / "FINAL_PROJECT_DIRECTION_DECISION.json", final_decision_payload())
    atomic_text(OUTPUT / "MYOLEG_PERSONALIZATION_FORMULATION_STOP_OR_PIVOT_REPORT.md", report(FROZEN_PROTOCOL_SHA256))
    atomic_json(
        OUTPUT / "metadata.json",
        {
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "evidence_file_count": len(evidence),
            "primary_decision": PRIMARY_DECISION,
            "synthetic_personalization_option": SYNTHETIC_OPTION_DECISION,
            "universal_optimization_option": UNIVERSAL_OPTION_DECISION,
            "pinn_status": PINN_DECISION,
            "bo_status": BO_DECISION,
            "held_out_scientific_access_count": 0,
            "simulator_replay_count": 0,
            "scientific_array_load_count": 0,
            "new_cohort_count": 0,
            "new_virtual_subject_count": 0,
            "objective_modified": False,
            "normalization_modified": False,
            "v3_candidate_domain_modified": False,
            "s1_modified": False,
            "five_parameter_or_nn_or_pinn_trained": False,
            "bo_run": False,
            "robot_or_hardware": False,
            "human_ready": False,
            "robot_approved": False,
            "next_stage": NEXT_STAGE,
            "next_stage_executed": False,
            "protocol_identity_verified": protocol["protocol_id"] == PROTOCOL_ID,
        },
    )
    write_checksums()
    print(json.dumps(final_decision_payload(), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="freeze protocol and evidence hashes")
    group.add_argument("--execute", action="store_true", help="verify freeze and build decision artifacts")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute()


if __name__ == "__main__":
    main()
