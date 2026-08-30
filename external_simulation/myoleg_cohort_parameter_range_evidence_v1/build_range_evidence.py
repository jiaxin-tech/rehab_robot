"""Build MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_V1.

This is an offline evidence and numerical-integrity stage.  It freezes ranges
for the already-selected six Scheme-A factors, replays only preregistered
endpoints/corners, and emits a future generation protocol.  It deliberately
does not instantiate a cohort, reveal learner outcomes, fit any model, run BO,
or access robot-facing code.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import mujoco
import numpy as np


STAGE_ID = "MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_V1"
OUTCOME = "MYOLEG_COHORT_RANGES_READY_WITH_SYNTHETIC_LIMITATIONS"
SCHEME_ID = "SCHEME_A_MINIMAL_INTERPRETABLE"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"
V2_REFERENCE_ID = "NATIVE_ROM_REFERENCE_CANDIDATE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_cohort_parameter_range_evidence_v1"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_supine_rehab_v1"
    / "myoleg_supine_right_v1.xml"
)
V2_REFERENCE_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)
TRUTH_SEMANTICS_PATH = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_reference_trajectory_replay_v1"
    / "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1.json"
)
FORMAL_REFERENCE_PATH = (
    PROJECT_ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
FORMAL_MANIFEST_PATH = PROJECT_ROOT / "config" / "formal_experiment_manifest.json"
PRIOR_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "external_simulation_audits"
    / "myoleg_virtual_patient_cohort_design_v1"
)
PRIOR_BUILDER_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_virtual_patient_cohort_design_v1"
    / "build_design_audit.py"
)
REPLAY_BUILDER_PATH = (
    PROJECT_ROOT
    / "external_simulation"
    / "myoleg_reference_trajectory_replay_v1"
    / "build_and_replay.py"
)

FROZEN_SHA256 = {
    "base_myoleg_model": "20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d",
    "v2_reference": "208a13cef47ff5407348db27dc0a8570e803f9191fbb29b85fabf3fb71012678",
    "truth_semantics": "750d94b59427cdf25cd026192b889a5ba4345e7cb9bdd674434c4a61771c0adc",
    "formal_reference": "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881",
    "formal_manifest": "f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441",
    "prior_scheme": "a460befdbbfa7dc7f54078673067843467740ed6ecbf7c4cd5cee533e6269bff",
    "prior_taxonomy": "0f8f3e6af995ad973bb1c941e9cc4e2efa96248ee1df85c65f44f38138bab33f",
    "prior_range_evidence": "af250c583d856ba9891fb0449b4a964f9c469a3a0151ed560ce086534fec596c",
    "prior_inventory": "b4eded805c353e65bb38325d64deb85bfbb3eff4c4ff127e9135c5be306ac417",
}

PRIOR_FROZEN_FILES = {
    "prior_scheme": PRIOR_ARTIFACT_DIRECTORY / "PROPOSED_COHORT_SCHEMES.json",
    "prior_taxonomy": PRIOR_ARTIFACT_DIRECTORY / "PARAMETER_TAXONOMY.csv",
    "prior_range_evidence": PRIOR_ARTIFACT_DIRECTORY / "PARAMETER_RANGE_EVIDENCE.json",
    "prior_inventory": PRIOR_ARTIFACT_DIRECTORY / "MYOLEG_PARAMETER_INVENTORY.csv",
}

BIARTICULAR_ACTUATORS = (
    "bflh_r",
    "grac_r",
    "recfem_r",
    "sart_r",
    "semimem_r",
    "semiten_r",
    "tfl_r",
)

# These gates are reused from the frozen V1 design smoke protocol.  The
# force-ratio screen is applied more broadly here as a conservative numerical
# integrity check; it is not a physiological validity threshold.
INTEGRITY_THRESHOLDS = {
    "source_equality_residual_max": 1.0e-3,
    "algebraic_residual_max_nm": 1.0e-8,
    "tracking_q_max_abs_deg": 1.0,
    "peak_force_ratio_vs_nominal_max": 2.0,
    "native_knee_min_deg": 0.0,
    "native_knee_max_deg": 120.0,
}

SOURCE_BIBLIOGRAPHY = {
    "MYOSUITE_DOCS": {
        "title": "Models and Tasks - MyoSuite documentation: myoLeg",
        "year": 2026,
        "url": "https://myosuite.readthedocs.io/en/stable/suite.html",
        "role": "official model provenance; says MyoLeg takes the Rajagopal full-body gait model as close reference",
    },
    "MUJOCO_MUSCLE_36": {
        "title": "MuJoCo 3.6 Modeling - Muscle actuators",
        "year": 2026,
        "url": "https://mujoco.readthedocs.io/en/3.6.0/modeling.html#muscle-actuators",
        "role": "official actuator force, F0, tendon-length and FLV semantics",
    },
    "MUJOCO_XML_36": {
        "title": "MuJoCo 3.6 XML Reference - actuator/muscle",
        "year": 2026,
        "url": "https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#actuator-muscle",
        "role": "official fpmax field definition",
    },
    "RAJAGOPAL2016": {
        "title": "Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait",
        "year": 2016,
        "doi": "10.1109/TBME.2016.2586891",
        "url": "https://pubmed.ncbi.nlm.nih.gov/27392337/",
        "role": "source-model population, architecture and limitations",
    },
    "KUDZIA2022": {
        "title": "Estimating body segment parameters from three-dimensional human body scans",
        "year": 2022,
        "doi": "10.1371/journal.pone.0262296",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262296",
        "role": "adult thigh, shank and foot mass/inertia mean and SD",
    },
    "DURKIN2003": {
        "title": "Analysis of body segment parameter differences between four human populations and the estimation errors of four popular mathematical models",
        "year": 2003,
        "doi": "10.1115/1.1590359",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12968576/",
        "role": "DXA evidence of population and individual BSP variability",
    },
    "SILDER2007": {
        "title": "Identification of passive elastic joint moment-angle relationships in the lower extremity",
        "year": 2007,
        "doi": "10.1016/j.jbiomech.2006.12.017",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2020832/",
        "role": "subject-specific passive hip/knee/ankle moments and biarticular effects",
    },
    "WINTER2010": {
        "title": "The force-length curves of the human rectus femoris and gastrocnemius muscles in vivo",
        "year": 2010,
        "url": "https://pubmed.ncbi.nlm.nih.gov/20147757/",
        "role": "intersubject and intermuscle variability in expressed force-length sections",
    },
    "MAGNUSSON2000": {
        "title": "Passive tensile stress and energy of the human hamstring muscles in vivo",
        "year": 2000,
        "url": "https://pubmed.ncbi.nlm.nih.gov/11085563/",
        "role": "passive hamstring behavior differs between flexible and inflexible groups",
    },
    "UHLRICH2022_CALIBRATION": {
        "title": "Calibration of Passive Muscle Force Curves in OpenSim Musculoskeletal Models",
        "year": 2022,
        "doi": "10.1038/s41598-022-13386-9",
        "url": "https://github.com/stanfordnmbl/PassiveMuscleForceCalibration",
        "role": "official laboratory calibration materials: shifts OpenSim passive-curve strain parameters, not MuJoCo fpmax",
    },
}


# Range values are preregistered constants.  They are written to the proposal
# artifact before any replay is evaluated.  Anthropometric primary half-widths
# are rounded maximum sex-specific one-SD mass CVs from Kudzia et al.; extended
# values are symmetric two-SD stress envelopes.  Passive primary values retain
# the previously frozen +/-5% numerical sensitivity amplitude because no
# defensible population-to-fpmax mapping was found; +/-10% is stress-only.
FACTOR_RANGES: dict[str, dict[str, Any]] = {
    "FEMUR_MASS_INERTIA_SCALE": {
        "factor_type": "ANTHROPOMETRY",
        "targets": ["femur_r"],
        "conservative": [0.88, 1.0, 1.12],
        "extended": [0.76, 1.0, 1.24],
        "evidence_class": "E2",
        "sources": ["KUDZIA2022", "DURKIN2003", "RAJAGOPAL2016"],
        "derivation": "primary +/-12% rounds the larger sex-specific thigh-mass SD/mean (11.7%); extended doubles that fractional SD",
        "mapping_assumptions": [
            "thigh mass maps to femur_r body mass",
            "body inertia receives the same scalar while body_ipos/COM and geometry remain fixed",
            "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
        ],
        "confidence": "MODERATE_FOR_MASS_LOW_FOR_COUPLED_INERTIA",
        "primary_suitability": "PRIMARY_ANTHROPOMETRIC_APPROXIMATION",
    },
    "TIBIA_PATELLA_MASS_INERTIA_SCALE": {
        "factor_type": "ANTHROPOMETRY",
        "targets": ["tibia_r", "patella_r"],
        "conservative": [0.87, 1.0, 1.13],
        "extended": [0.74, 1.0, 1.26],
        "evidence_class": "E2",
        "sources": ["KUDZIA2022", "DURKIN2003", "RAJAGOPAL2016"],
        "derivation": "primary +/-13% rounds the larger sex-specific shank-mass SD/mean (12.8%); extended doubles that fractional SD",
        "mapping_assumptions": [
            "shank mass maps to tibia_r and the mechanically coupled patella_r",
            "patella was not separately estimated in the source study",
            "body inertia receives the same scalar while body_ipos/COM and geometry remain fixed",
            "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
        ],
        "confidence": "MODERATE_FOR_SHANK_MASS_LOW_FOR_PATELLA_AND_COUPLED_INERTIA",
        "primary_suitability": "PRIMARY_ANTHROPOMETRIC_APPROXIMATION",
    },
    "FOOT_COMPLEX_MASS_INERTIA_SCALE": {
        "factor_type": "ANTHROPOMETRY",
        "targets": ["talus_r", "calcn_r", "toes_r"],
        "conservative": [0.82, 1.0, 1.18],
        "extended": [0.64, 1.0, 1.36],
        "evidence_class": "E2",
        "sources": ["KUDZIA2022", "DURKIN2003", "RAJAGOPAL2016"],
        "derivation": "primary +/-18% rounds the reported foot-mass SD/mean (18.2%); extended doubles that fractional SD",
        "mapping_assumptions": [
            "whole-foot mass maps to the talus/calcaneus/toes model complex",
            "small-segment scan estimates have greater measurement variability",
            "body inertia receives the same scalar while body_ipos/COM and geometry remain fixed",
            "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
        ],
        "confidence": "LOW_TO_MODERATE_FOR_MASS_LOW_FOR_COUPLED_INERTIA",
        "primary_suitability": "PRIMARY_ANTHROPOMETRIC_APPROXIMATION",
    },
    "HIP_ONLY_PASSIVE_FP_MAX_SCALE": {
        "factor_type": "PASSIVE_FPMAX",
        "structural_group": "HIP_ONLY",
        "conservative": [0.95, 1.0, 1.05],
        "extended": [0.90, 1.0, 1.10],
        "evidence_class": "E4",
        "sources": ["MUJOCO_MUSCLE_36", "SILDER2007", "UHLRICH2022_CALIBRATION"],
        "derivation": "primary retains the frozen +/-5% sensitivity amplitude; extended is a two-amplitude stress interval",
        "mapping_assumptions": [
            "fpmax is a MuJoCo normalized passive-FLV curve parameter, not measured human passive stiffness",
            "no population-level conversion is available",
        ],
        "confidence": "LOW_SYNTHETIC",
        "primary_suitability": "PRIMARY_SYNTHETIC_HETEROGENEITY_ONLY",
    },
    "KNEE_ONLY_PASSIVE_FP_MAX_SCALE": {
        "factor_type": "PASSIVE_FPMAX",
        "structural_group": "KNEE_ONLY",
        "conservative": [0.95, 1.0, 1.05],
        "extended": [0.90, 1.0, 1.10],
        "evidence_class": "E4",
        "sources": ["MUJOCO_MUSCLE_36", "SILDER2007", "UHLRICH2022_CALIBRATION"],
        "derivation": "primary retains the frozen +/-5% sensitivity amplitude; extended is a two-amplitude stress interval",
        "mapping_assumptions": [
            "fpmax is a MuJoCo normalized passive-FLV curve parameter, not measured human passive stiffness",
            "no population-level conversion is available",
        ],
        "confidence": "LOW_SYNTHETIC",
        "primary_suitability": "PRIMARY_SYNTHETIC_HETEROGENEITY_ONLY",
    },
    "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE": {
        "factor_type": "PASSIVE_FPMAX",
        "structural_group": "HIP_KNEE_BIARTICULAR",
        "conservative": [0.95, 1.0, 1.05],
        "extended": [0.90, 1.0, 1.10],
        "evidence_class": "E4",
        "sources": [
            "MUJOCO_MUSCLE_36",
            "SILDER2007",
            "WINTER2010",
            "MAGNUSSON2000",
            "UHLRICH2022_CALIBRATION",
        ],
        "derivation": "same common +/-5% base interval as other passive groups; no evidence supports a distinct relative fpmax range",
        "mapping_assumptions": [
            "the frozen transmission-derived group is used unchanged",
            "passive biarticular behavior is evidenced, but not a population-to-MuJoCo-fpmax mapping",
        ],
        "confidence": "LOW_SYNTHETIC",
        "primary_suitability": "PRIMARY_SYNTHETIC_HETEROGENEITY_ONLY",
    },
}

SCHEME_A_FACTORS = tuple(FACTOR_RANGES)

INTERACTION_PROFILES = {
    "ALL_CONSERVATIVE_LOW": {factor: spec["conservative"][0] for factor, spec in FACTOR_RANGES.items()},
    "ALL_CONSERVATIVE_HIGH": {factor: spec["conservative"][2] for factor, spec in FACTOR_RANGES.items()},
    # Scheme A contains only mass/inertia and passive factors, so this requested
    # profile is algebraically identical to ALL_CONSERVATIVE_HIGH.  It remains
    # explicitly retained rather than silently dropped.
    "HIGH_MASS_HIGH_PASSIVE": {factor: spec["conservative"][2] for factor, spec in FACTOR_RANGES.items()},
    "LOW_MASS_HIGH_PASSIVE": {
        factor: spec["conservative"][0 if spec["factor_type"] == "ANTHROPOMETRY" else 2]
        for factor, spec in FACTOR_RANGES.items()
    },
    "REPRESENTATIVE_MIXED": {
        "FEMUR_MASS_INERTIA_SCALE": 1.12,
        "TIBIA_PATELLA_MASS_INERTIA_SCALE": 0.87,
        "FOOT_COMPLEX_MASS_INERTIA_SCALE": 1.0,
        "HIP_ONLY_PASSIVE_FP_MAX_SCALE": 1.05,
        "KNEE_ONLY_PASSIVE_FP_MAX_SCALE": 0.95,
        "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE": 1.0,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checksum_manifest(path: Path) -> dict[str, str]:
    checked: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = path.parent / relative.strip()
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"prior artifact checksum failed: {target}")
        checked[relative.strip()] = actual
    return checked


def frozen_input_hashes() -> dict[str, str]:
    paths = {
        "base_myoleg_model": MODEL_PATH,
        "v2_reference": V2_REFERENCE_PATH,
        "truth_semantics": TRUTH_SEMANTICS_PATH,
        "formal_reference": FORMAL_REFERENCE_PATH,
        "formal_manifest": FORMAL_MANIFEST_PATH,
        **PRIOR_FROZEN_FILES,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if any(actual[name] != expected for name, expected in FROZEN_SHA256.items()):
        failures = {
            name: {"expected": expected, "actual": actual[name]}
            for name, expected in FROZEN_SHA256.items()
            if actual[name] != expected
        }
        raise RuntimeError(f"frozen input changed: {failures}")
    manifest = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not (
        manifest["rom_protocol_version"] == "ROM_PROTOCOL_V2"
        and manifest["hip_rom_deg"] == [0.0, 120.0]
        and manifest["knee_rom_deg"] == [5.0, 145.0]
        and manifest["theta_shank_definition"] == "q_hip - q_knee"
        and manifest["active_reference_sha256"] == FROZEN_SHA256["formal_reference"]
    ):
        raise RuntimeError("formal ROM/reference convention changed")
    truth = json.loads(TRUTH_SEMANTICS_PATH.read_text(encoding="utf-8"))
    if truth["semantic_version"] != TRUTH_SEMANTIC_VERSION or truth["truth_field"] != TRUTH_FIELD:
        raise RuntimeError("truth semantics changed")
    return actual


def runtime_environment() -> dict[str, Any]:
    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "myosuite": importlib.metadata.version("myosuite"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    expected = {"python": "3.10.19", "myosuite": "2.12.2", "mujoco": "3.6.0"}
    result["frozen_expected"] = expected
    result["frozen_match"] = all(result[key] == value for key, value in expected.items())
    if not result["frozen_match"]:
        raise RuntimeError("frozen MyoLeg runtime changed")
    return result


def model_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or ""


def proposal_payload(prior_scheme_sha: str) -> dict[str, Any]:
    factors = []
    for factor_id, spec in FACTOR_RANGES.items():
        factors.append({"factor_id": factor_id, "nominal": 1.0, **spec})
    core = {
        "stage_id": STAGE_ID,
        "scheme_id": SCHEME_ID,
        "scheme_source_sha256": prior_scheme_sha,
        "frozen_before_replay": True,
        "evidence_hierarchy": {
            "E1": "DIRECT",
            "E2": "MODEL_DERIVED",
            "E3": "INDIRECT",
            "E4": "STRESS_ONLY_OR_SYNTHETIC_SENSITIVITY",
        },
        "p0_semantics": {
            "condition": "P0_ZERO_MINIMUM_CONTROL_MUSCULOSKELETAL",
            "force_fpmax_indistinguishable": True,
            "f0_factor_present": False,
            "rule": "Do not add F0 as an independent P0 dimension; retain frozen fpmax groups.",
        },
        "factors": factors,
        "interaction_profiles": INTERACTION_PROFILES,
        "integrity_thresholds": INTEGRITY_THRESHOLDS,
        "range_selection_used_mechanical_objective": False,
        "range_selection_used_learner_or_landscape_outcome": False,
    }
    core["proposal_content_sha256"] = canonical_sha256(core)
    return core


def literature_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "parameter_or_factor": "MYOLEG_MODEL_PROVENANCE",
            "source_id": "MYOSUITE_DOCS",
            "source": SOURCE_BIBLIOGRAPHY["MYOSUITE_DOCS"]["title"],
            "year": 2026,
            "population_or_model": "official MyoSuite MyoLeg documentation",
            "sample_size": "NA",
            "reported_quantity": "model relationship",
            "reported_central_tendency": "NA",
            "reported_variability_or_range": "10 joints; 20 DoF; 80 MTUs; Rajagopal is a close reference",
            "evidence_class": "E2",
            "mapping_to_myoleg": "direct official provenance for model family, not proof of one-to-one parameter conversion",
            "limitation": "documentation says close reference rather than exact conversion",
            "used_for_primary_range": "NO",
        },
        {
            "parameter_or_factor": "MYOLEG_MODEL_PROVENANCE",
            "source_id": "RAJAGOPAL2016",
            "source": SOURCE_BIBLIOGRAPHY["RAJAGOPAL2016"]["title"],
            "year": 2016,
            "population_or_model": "OpenSim generic healthy-young gait model",
            "sample_size": "21/22 cadavers plus MRI from 24 young healthy subjects",
            "reported_quantity": "37 DoF; 80 lower-limb Hill-type MTUs; musculotendon architecture",
            "reported_central_tendency": "generic 75 kg, 170 cm male skeleton",
            "reported_variability_or_range": "paper states experimental variability is not captured by the generic model",
            "evidence_class": "E2",
            "mapping_to_myoleg": "source-model context only",
            "limitation": "OpenSim Millard muscle/tendon parameters are not one-to-one with MuJoCo FLV fields",
            "used_for_primary_range": "NO",
        },
        {
            "parameter_or_factor": "FEMUR_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "source": SOURCE_BIBLIOGRAPHY["KUDZIA2022"]["title"],
            "year": 2022,
            "population_or_model": "healthy young adults; 10 male, 11 female; repeated 3D scans",
            "sample_size": 21,
            "reported_quantity": "thigh mass and three-axis inertia",
            "reported_central_tendency": "mass male 10.4 kg; female 12.0 kg",
            "reported_variability_or_range": "mass SD male 0.8 kg (7.7%); female 1.4 kg (11.7%); inertia SD/mean about 10.8-21.3% across sex/axis",
            "evidence_class": "E2",
            "mapping_to_myoleg": "fractional thigh-mass variability -> femur_r mass scale; same inertia scale is approximate",
            "limitation": "small young sample; 3D scan plus density model; no covariance; nominal MyoLeg subject is not the study mean",
            "used_for_primary_range": "YES",
        },
        {
            "parameter_or_factor": "TIBIA_PATELLA_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "source": SOURCE_BIBLIOGRAPHY["KUDZIA2022"]["title"],
            "year": 2022,
            "population_or_model": "healthy young adults; 10 male, 11 female; repeated 3D scans",
            "sample_size": 21,
            "reported_quantity": "shank mass and three-axis inertia",
            "reported_central_tendency": "mass male 3.5 kg; female 3.9 kg",
            "reported_variability_or_range": "mass SD male 0.4 kg (11.4%); female 0.5 kg (12.8%); inertia SD/mean about 17.0-23.7% across sex/axis",
            "evidence_class": "E2",
            "mapping_to_myoleg": "fractional shank-mass variability -> tibia_r plus mechanically coupled patella_r",
            "limitation": "patella not separate; coupled inertia scaling and fixed COM are approximations",
            "used_for_primary_range": "YES",
        },
        {
            "parameter_or_factor": "FOOT_COMPLEX_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "source": SOURCE_BIBLIOGRAPHY["KUDZIA2022"]["title"],
            "year": 2022,
            "population_or_model": "healthy young adults; 10 male, 11 female; repeated 3D scans",
            "sample_size": 21,
            "reported_quantity": "foot mass and three-axis inertia",
            "reported_central_tendency": "mass male/female 1.1 kg",
            "reported_variability_or_range": "mass SD 0.2 kg (18.2%); inertia SD/mean about 24-40%; small segments had >15% repeat-scan CV in some measures",
            "evidence_class": "E2",
            "mapping_to_myoleg": "fractional foot-mass variability -> talus_r/calcn_r/toes_r complex",
            "limitation": "small-segment measurement uncertainty contributes to the observed variability",
            "used_for_primary_range": "YES_WITH_LOW_CONFIDENCE",
        },
        {
            "parameter_or_factor": "ALL_ANTHROPOMETRY_FACTORS",
            "source_id": "DURKIN2003",
            "source": SOURCE_BIBLIOGRAPHY["DURKIN2003"]["title"],
            "year": 2003,
            "population_or_model": "four adult populations measured by DXA",
            "sample_size": "not extracted for numeric range use",
            "reported_quantity": "segment mass, COM and radius of gyration",
            "reported_central_tendency": "NA",
            "reported_variability_or_range": "significant population differences and large within-group individual differences",
            "evidence_class": "E1",
            "mapping_to_myoleg": "supports nonzero subject heterogeneity and correlated anthropometric predictors",
            "limitation": "abstract does not provide factor-specific SD used for bounds",
            "used_for_primary_range": "CONTEXT_ONLY",
        },
        {
            "parameter_or_factor": "ALL_PASSIVE_FP_MAX_FACTORS",
            "source_id": "MUJOCO_MUSCLE_36",
            "source": SOURCE_BIBLIOGRAPHY["MUJOCO_MUSCLE_36"]["title"],
            "year": 2026,
            "population_or_model": "MuJoCo 3.6 actuator model",
            "sample_size": "NA",
            "reported_quantity": "FLV and fpmax semantics",
            "reported_central_tendency": "fpmax is passive normalized force at lmax relative to F0",
            "reported_variability_or_range": "no human population interval",
            "evidence_class": "E2",
            "mapping_to_myoleg": "direct field semantics only",
            "limitation": "abstract actuator-curve parameter; inelastic tendon and inferred L0/LT mapping",
            "used_for_primary_range": "NO",
        },
        {
            "parameter_or_factor": "HIP_ONLY;KNEE_ONLY;HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
            "source_id": "SILDER2007",
            "source": SOURCE_BIBLIOGRAPHY["SILDER2007"]["title"],
            "year": 2007,
            "population_or_model": "20 healthy young adults; 9 male, 11 female",
            "sample_size": 20,
            "reported_quantity": "subject-specific passive hip/knee/ankle moment-angle functions with uni/biarticular terms",
            "reported_central_tendency": "model RMSE 2.5/1.4/0.7 Nm at hip/knee/ankle",
            "reported_variability_or_range": "subject-specific fitted functions; no MuJoCo fpmax scale distribution",
            "evidence_class": "E3",
            "mapping_to_myoleg": "supports passive and biarticular heterogeneity, not a scalar fpmax conversion",
            "limitation": "joint moments include multiple tissues and exponential offsets/gains",
            "used_for_primary_range": "NO_NUMERIC_MAPPING",
        },
        {
            "parameter_or_factor": "HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE",
            "source_id": "WINTER2010",
            "source": SOURCE_BIBLIOGRAPHY["WINTER2010"]["title"],
            "year": 2010,
            "population_or_model": "28 nonspecifically trained human subjects",
            "sample_size": 28,
            "reported_quantity": "expressed force-length sections of rectus femoris and gastrocnemius",
            "reported_central_tendency": "NA",
            "reported_variability_or_range": "intersubject variability differs between muscles",
            "evidence_class": "E3",
            "mapping_to_myoleg": "supports group/muscle differences in force-length behavior",
            "limitation": "not passive fpmax and not a scale interval",
            "used_for_primary_range": "NO",
        },
        {
            "parameter_or_factor": "ALL_PASSIVE_FP_MAX_FACTORS",
            "source_id": "UHLRICH2022_CALIBRATION",
            "source": SOURCE_BIBLIOGRAPHY["UHLRICH2022_CALIBRATION"]["title"],
            "year": 2022,
            "population_or_model": "Rajagopal/OpenSim model calibrated to Silder passive moments",
            "sample_size": "model calibration to 20-subject aggregate data",
            "reported_quantity": "passive curve strainAtZeroForce and strainAtOneNormForce",
            "reported_central_tendency": "NA",
            "reported_variability_or_range": "model-specific tuned curve shifts; no population fpmax distribution",
            "evidence_class": "E2",
            "mapping_to_myoleg": "shows the source-model passive curve needs calibration but changes x-axis curve parameters, not MuJoCo fpmax amplitude",
            "limitation": "cannot be converted to group-specific fpmax scale",
            "used_for_primary_range": "NO_NUMERIC_MAPPING",
        },
        {
            "parameter_or_factor": "ALL_PASSIVE_FP_MAX_FACTORS",
            "source_id": "PRIOR_FROZEN_SMOKE",
            "source": "MYOLEG_VIRTUAL_PATIENT_COHORT_DESIGN_V1 SINGLE_PARAMETER_SENSITIVITY_RESULTS.csv",
            "year": 2026,
            "population_or_model": "frozen native MyoLeg P0 V2 replay",
            "sample_size": "one nominal model; deterministic +/-5% perturbations",
            "reported_quantity": "numerical integrity under grouped fpmax scaling",
            "reported_central_tendency": "nominal scale 1.0",
            "reported_variability_or_range": "0.95-1.05 passed; not a physiological range",
            "evidence_class": "E4",
            "mapping_to_myoleg": "direct numerical precedent for the exact compiled fields and frozen groups",
            "limitation": "single-model sensitivity only",
            "used_for_primary_range": "YES_SYNTHETIC_ONLY",
        },
    ]
    return rows


def anthropometry_rows() -> list[dict[str, Any]]:
    return [
        {
            "factor_id": "FEMUR_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "segment": "thigh",
            "male_mass_mean_kg": 10.4,
            "male_mass_sd_kg": 0.8,
            "female_mass_mean_kg": 12.0,
            "female_mass_sd_kg": 1.4,
            "max_sex_mass_cv_pct": 11.667,
            "reported_inertia_cv_pct_range": "10.8-21.3",
            "conservative_scale": "0.88;1.12",
            "extended_scale": "0.76;1.24",
            "evidence_class": "E2",
            "inertia_mapping_status": "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
            "limitation": "fixed COM/geometry; one scalar cannot reproduce independent mass and inertia variability",
        },
        {
            "factor_id": "TIBIA_PATELLA_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "segment": "shank",
            "male_mass_mean_kg": 3.5,
            "male_mass_sd_kg": 0.4,
            "female_mass_mean_kg": 3.9,
            "female_mass_sd_kg": 0.5,
            "max_sex_mass_cv_pct": 12.821,
            "reported_inertia_cv_pct_range": "17.0-23.7",
            "conservative_scale": "0.87;1.13",
            "extended_scale": "0.74;1.26",
            "evidence_class": "E2",
            "inertia_mapping_status": "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
            "limitation": "patella not measured separately; fixed COM/geometry",
        },
        {
            "factor_id": "FOOT_COMPLEX_MASS_INERTIA_SCALE",
            "source_id": "KUDZIA2022",
            "segment": "foot",
            "male_mass_mean_kg": 1.1,
            "male_mass_sd_kg": 0.2,
            "female_mass_mean_kg": 1.1,
            "female_mass_sd_kg": 0.2,
            "max_sex_mass_cv_pct": 18.182,
            "reported_inertia_cv_pct_range": "24-40",
            "conservative_scale": "0.82;1.18",
            "extended_scale": "0.64;1.36",
            "evidence_class": "E2",
            "inertia_mapping_status": "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
            "limitation": "small-segment scan CV can exceed 15%; talus/calcaneus/toes mapping is aggregated",
        },
    ]


def passive_rows(model: mujoco.MjModel, structural: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for factor_id, spec in FACTOR_RANGES.items():
        if spec["factor_type"] != "PASSIVE_FPMAX":
            continue
        group = spec["structural_group"]
        ids = [
            index
            for index in range(model.nu)
            if model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
            and structural[index]["structural_group"] == group
        ]
        names = [model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in ids]
        values = np.asarray(model.actuator_biasprm[ids, 7], dtype=float)
        rows.append(
            {
                "factor_id": factor_id,
                "frozen_structural_group": group,
                "target_count": len(ids),
                "target_actuators": ";".join(names),
                "native_fpmax_min": float(np.min(values)),
                "native_fpmax_max": float(np.max(values)),
                "conservative_scale": "0.95;1.05",
                "extended_scale": "0.90;1.10",
                "range_evidence_class": "E4",
                "population_mapping_status": "NO_RELIABLE_POPULATION_TO_FPMAX_MAPPING",
                "range_interpretation": "structured musculoskeletal-model heterogeneity; not representative patient variation",
                "group_specific_range_supported": "NO_USE_COMMON_BASE_INTERVAL",
                "sources": ";".join(spec["sources"]),
            }
        )
    return rows


def correlation_rows() -> list[dict[str, Any]]:
    mass = list(SCHEME_A_FACTORS[:3])
    passive = list(SCHEME_A_FACTORS[3:])
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(mass):
        for right in mass[index + 1 :]:
            rows.append(
                {
                    "factor_a": left,
                    "factor_b": right,
                    "scope": "SCHEME_A",
                    "classification": "POSSIBLY_CORRELATED",
                    "evidence": "segment masses share body-size/sex/morphology predictors, but no pairwise covariance was extracted",
                    "source_ids": "DURKIN2003;KUDZIA2022",
                    "sampling_action": "do not claim statistical independence; no covariance matrix frozen",
                }
            )
    for left in mass:
        for right in passive:
            rows.append(
                {
                    "factor_a": left,
                    "factor_b": right,
                    "scope": "SCHEME_A",
                    "classification": "NO_USEFUL_EVIDENCE",
                    "evidence": "no quantitative evidence maps segment scale to MuJoCo group fpmax scale",
                    "source_ids": "RAJAGOPAL2016;MUJOCO_MUSCLE_36",
                    "sampling_action": "marginal space-filling only; independence is a synthetic design limitation",
                }
            )
    for index, left in enumerate(passive):
        for right in passive[index + 1 :]:
            rows.append(
                {
                    "factor_a": left,
                    "factor_b": right,
                    "scope": "SCHEME_A",
                    "classification": "POSSIBLY_CORRELATED",
                    "evidence": "common tissue/system influences are plausible, but groupwise fpmax covariance is unavailable",
                    "source_ids": "SILDER2007;WINTER2010",
                    "sampling_action": "use common marginal interval; do not invent covariance",
                }
            )
    for right in passive:
        rows.append(
            {
                "factor_a": "GLOBAL_RIGHT_TARGET_FP_MAX_SCALE",
                "factor_b": right,
                "scope": "NON_SCHEME_A_STRUCTURAL_EXCLUSION",
                "classification": "KNOWN_CORRELATED",
                "evidence": "the global factor algebraically contains each group-specific scaling direction",
                "source_ids": "PRIOR_FROZEN_SMOKE",
                "sampling_action": "global fpmax factor is excluded from Scheme A",
            }
        )
    rows.append(
        {
            "factor_a": "ALL_MASS_FACTORS",
            "factor_b": "FUTURE_ACTIVE_CONDITION_F0_FORCE_CAPACITY",
            "scope": "NON_SCHEME_A_FUTURE_ACTIVE_CONDITION",
            "classification": "POSSIBLY_CORRELATED",
            "evidence": "Rajagopal reports total lower-limb muscle volume correlated with subject mass and height",
            "source_ids": "RAJAGOPAL2016",
            "sampling_action": "future active-condition design must not assume independence without quantitative review",
        }
    )
    return rows


def apply_factor_scale(
    model: mujoco.MjModel,
    factor_id: str,
    scale: float,
    structural: dict[int, dict[str, Any]],
) -> list[str]:
    spec = FACTOR_RANGES[factor_id]
    targets: list[str] = []
    if spec["factor_type"] == "ANTHROPOMETRY":
        for body in spec["targets"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0:
                raise RuntimeError(f"missing body {body}")
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale
            targets.append(body)
    else:
        group = spec["structural_group"]
        selected = [
            index
            for index in range(model.nu)
            if model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
            and structural[index]["structural_group"] == group
        ]
        if not selected:
            raise RuntimeError(f"empty frozen structural group {group}")
        model.actuator_gainprm[selected, 7] *= scale
        model.actuator_biasprm[selected, 7] *= scale
        targets.extend(model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in selected)
    return targets


def replay_fingerprint(prescribed: dict[str, np.ndarray], controlled: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in (
        prescribed["tau_truth_nm"],
        prescribed["actuator_force_n"],
        prescribed["source_equality_residual"],
        controlled["actual_q_rad"],
        controlled["actual_dq_rad_s"],
        controlled["source_equality_residual"],
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def run_profile(
    profile_id: str,
    scales: dict[str, float],
    structural: dict[int, dict[str, Any]],
    reference: dict[str, Any],
    replay: Any,
) -> dict[str, Any]:
    if set(scales) != set(SCHEME_A_FACTORS):
        raise RuntimeError(f"profile {profile_id} does not cover exactly Scheme A")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    targets: list[str] = []
    for factor_id in SCHEME_A_FACTORS:
        scale = float(scales[factor_id])
        if scale != 1.0:
            targets.extend(apply_factor_scale(model, factor_id, scale, structural))
    prescribed, prescribed_runtime = replay.prescribed_truth(model, reference)
    controlled, controlled_runtime = replay.controlled_replay(model, reference)
    force = np.abs(np.asarray(prescribed["actuator_force_n"], dtype=float))
    peak_flat = int(np.argmax(force))
    peak_sample, peak_actuator = np.unravel_index(peak_flat, force.shape)
    total_force = np.sum(force, axis=1)
    sample_share = np.divide(
        np.max(force, axis=1),
        total_force,
        out=np.zeros_like(total_force),
        where=total_force > 0.0,
    )
    q_error_deg = np.degrees(controlled["actual_q_rad"] - reference["q"])
    knee_deg = np.degrees(controlled["actual_q_rad"][:, 1])
    algebraic = max(
        float(np.max(np.abs(prescribed["inverse_formula_residual_nm"]))),
        float(np.max(np.abs(prescribed["decomposition_residual_nm"]))),
        float(np.max(np.abs(prescribed["muscle_reconstruction_residual_nm"]))),
    )
    arrays: Iterable[np.ndarray] = (
        prescribed["tau_truth_nm"],
        prescribed["actuator_force_n"],
        prescribed["tendon_length_m"],
        controlled["actual_q_rad"],
        controlled["actual_dq_rad_s"],
    )
    return {
        "profile_id": profile_id,
        "factor_scales_json": json.dumps(scales, sort_keys=True, separators=(",", ":")),
        "target_count": len(set(targets)),
        "target_ids": ";".join(sorted(set(targets))),
        "reference_id": V2_REFERENCE_ID,
        "reference_sha256": FROZEN_SHA256["v2_reference"],
        "duration_s": float(reference["time_s"][-1]),
        "sample_count": len(reference["time_s"]),
        "tau_truth_hip_rms_nm": float(np.sqrt(np.mean(prescribed["tau_truth_nm"][:, 0] ** 2))),
        "tau_truth_knee_rms_nm": float(np.sqrt(np.mean(prescribed["tau_truth_nm"][:, 1] ** 2))),
        "tau_truth_hip_peak_abs_nm": float(np.max(np.abs(prescribed["tau_truth_nm"][:, 0]))),
        "tau_truth_knee_peak_abs_nm": float(np.max(np.abs(prescribed["tau_truth_nm"][:, 1]))),
        "source_equality_residual_max": max(
            float(np.max(np.abs(prescribed["source_equality_residual"]))),
            float(np.max(np.abs(controlled["source_equality_residual"]))),
        ),
        "algebraic_residual_max_nm": algebraic,
        "all_state_finite": all(bool(np.isfinite(array).all()) for array in arrays),
        "muscle_state_all_finite": bool(np.isfinite(prescribed["actuator_force_n"]).all()),
        "tendon_state_all_finite": bool(np.isfinite(prescribed["tendon_length_m"]).all()),
        "warning_count": max(
            int(np.max(prescribed["warning_count"])),
            int(np.max(controlled["warning_count"])),
            int(controlled_runtime["warning_count"]),
        ),
        "maximum_actuator_force_abs_n": float(force[peak_sample, peak_actuator]),
        "peak_force_actuator": model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, peak_actuator),
        "peak_force_time_s": float(reference["time_s"][peak_sample]),
        "maximum_single_actuator_force_share": float(np.max(sample_share)),
        "tendon_length_min_m": float(np.min(prescribed["tendon_length_m"])),
        "tendon_length_max_m": float(np.max(prescribed["tendon_length_m"])),
        "prescribed_joint_limit_active_max": int(np.max(prescribed["constraint_joint_limit_active_count"])),
        "prescribed_tendon_limit_active_max": int(np.max(prescribed["constraint_tendon_limit_active_count"])),
        "prescribed_contact_active_max": int(np.max(prescribed["constraint_contact_active_count"])),
        "tracking_q_max_abs_deg": float(np.max(np.abs(q_error_deg))),
        "controlled_knee_min_deg": float(np.min(knee_deg)),
        "controlled_knee_max_deg": float(np.max(knee_deg)),
        "truth_semantics_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "replay_sha256": replay_fingerprint(prescribed, controlled),
        "prescribed_runtime_s": prescribed_runtime["wall_time_s"],
        "controlled_runtime_s": controlled_runtime["wall_time_s"],
        "total_runtime_s": prescribed_runtime["wall_time_s"] + controlled_runtime["wall_time_s"],
    }


def add_gates(row: dict[str, Any], nominal: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["peak_force_ratio_vs_nominal"] = (
        float(row["maximum_actuator_force_abs_n"])
        / float(nominal["maximum_actuator_force_abs_n"])
    )
    result["force_concentration_share_ratio_vs_nominal"] = (
        float(row["maximum_single_actuator_force_share"])
        / max(float(nominal["maximum_single_actuator_force_share"]), 1.0e-12)
    )
    gates = {
        "model_load_and_reference": row["duration_s"] == 24.0 and row["sample_count"] == 401,
        "finite": bool(
            row["all_state_finite"]
            and row["muscle_state_all_finite"]
            and row["tendon_state_all_finite"]
        ),
        "no_solver_warning": int(row["warning_count"]) == 0,
        "equality_integrity": float(row["source_equality_residual_max"])
        <= INTEGRITY_THRESHOLDS["source_equality_residual_max"],
        "truth_algebra": float(row["algebraic_residual_max_nm"])
        <= INTEGRITY_THRESHOLDS["algebraic_residual_max_nm"],
        "tracking": float(row["tracking_q_max_abs_deg"])
        <= INTEGRITY_THRESHOLDS["tracking_q_max_abs_deg"],
        "native_knee_rom": float(row["controlled_knee_min_deg"])
        >= INTEGRITY_THRESHOLDS["native_knee_min_deg"] - 1.0e-10
        and float(row["controlled_knee_max_deg"])
        <= INTEGRITY_THRESHOLDS["native_knee_max_deg"] + 1.0e-10,
        "no_force_explosion": result["peak_force_ratio_vs_nominal"]
        <= INTEGRITY_THRESHOLDS["peak_force_ratio_vs_nominal_max"],
        "no_new_contact_or_limit_mode": (
            int(row["prescribed_joint_limit_active_max"])
            <= int(nominal["prescribed_joint_limit_active_max"])
            and int(row["prescribed_tendon_limit_active_max"])
            <= int(nominal["prescribed_tendon_limit_active_max"])
            and int(row["prescribed_contact_active_max"])
            <= int(nominal["prescribed_contact_active_max"])
        ),
        "truth_identity": row["truth_semantics_version"] == TRUTH_SEMANTIC_VERSION
        and row["truth_field"] == TRUTH_FIELD,
    }
    result["integrity_gate_results_json"] = json.dumps(gates, sort_keys=True, separators=(",", ":"))
    result["all_integrity_gates_pass"] = all(gates.values())
    result["abnormal_force_concentration_observed"] = bool(
        result["peak_force_ratio_vs_nominal"] > INTEGRITY_THRESHOLDS["peak_force_ratio_vs_nominal_max"]
    )
    return result


def write_provenance() -> None:
    text = f"""# MyoLeg model provenance audit

## Provenance chain

MyoSuite's official documentation describes MyoLeg as a 10-joint, 20-DoF,
80-muscle-tendon-unit model that takes the Rajagopal full-body gait model as a
**close reference**.  That wording does not establish a one-to-one converter or
parameter identity.  Rajagopal et al. (2016) built an OpenSim generic healthy
young gait model with lower-limb architecture drawn from cadaver measurements
and MRI muscle-volume data.  The current repository's native supine model is a
derived MyoLeg MJCF that preserves all 80 muscles/tendons and the target knee
equalities while changing root pose, locking non-target coordinates and
disabling world contacts; its frozen SHA is
`{FROZEN_SHA256['base_myoleg_model']}`.

## What can and cannot be called inherited

- Body names, meshes, tendon transmissions, muscle names and numerical MJCF
  fields come from the installed MyoLeg asset and are preserved in the frozen
  derived model unless its manifest explicitly lists a change.
- Rajagopal provides source-model context for geometry, muscle architecture and
  a generic 75 kg / 170 cm male skeleton.  The available official documentation
  does not prove that every MyoLeg body inertia or muscle curve field is an
  unchanged Rajagopal value.
- Therefore this audit calls the parameters `MyoLeg model parameters`, not
  direct measurements from Rajagopal's participants.

## OpenSim-to-MuJoCo muscle semantics

Rajagopal uses Millard-type Hill muscle-tendon units with an explicit source
model parameterization.  MuJoCo instead treats the spatial transmission as a
tendon and the muscle as an abstract force generator, assumes an inelastic
biological tendon for its shortcut mapping, infers `L0` and `LT` from length
ranges, and evaluates:

`FLV = FL(L) * FV(V) * activation + FP(L)`

`actuator_force = -F0 * FLV`

`fpmax` is the normalized passive force at `lmax`, relative to `F0`.  It is an
actuator-curve parameter.  It is **not** a directly measured human passive
stiffness, and the OpenSim passive-curve calibration evidence changes curve
strain/length parameters rather than providing a population distribution for
MuJoCo `fpmax`.

## Consequence for this stage

Anthropometric mass ranges can be anchored to model-derived adult segment
statistics.  The coupled inertia edit remains
`INERTIA_SCALING_IS_MODELING_APPROXIMATION`.  All three `fpmax` intervals remain
synthetic sensitivity ranges, despite literature support that passive and
biarticular mechanics vary between people.
"""
    (ARTIFACT_DIRECTORY / "MYOLEG_MODEL_PROVENANCE.md").write_text(text, encoding="utf-8")


def generation_protocol(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1",
        "status": "FROZEN_PROTOCOL_ONLY_DEFAULT_OFF",
        "execution_authorized": False,
        "cohort_generated": False,
        "readiness_basis": OUTCOME,
        "claim_boundary": "heterogeneous musculoskeletal virtual subjects; not a representative patient cohort",
        "scheme_id": SCHEME_ID,
        "scheme_source_sha256": FROZEN_SHA256["prior_scheme"],
        "factor_order": list(SCHEME_A_FACTORS),
        "ranges": {
            factor: {
                "primary": spec["conservative"],
                "extended_stress_only": spec["extended"],
                "evidence_class": spec["evidence_class"],
                "sources": spec["sources"],
            }
            for factor, spec in FACTOR_RANGES.items()
        },
        "dependency_and_correlation": {
            "quantitative_covariance_frozen": False,
            "reason": "no defensible covariance matrix is available for the exact six MyoLeg scale factors",
            "known_structural_overlap_excluded": "GLOBAL_RIGHT_TARGET_FP_MAX_SCALE",
            "marginal_independence_is_population_claim": False,
            "limitation": "space filling covers a rectangular model-parameter design, not a human joint distribution",
        },
        "sampling": {
            "algorithm": "deterministic centered maximin Latin hypercube",
            "dimension": 6,
            "heterogeneous_subject_count": 32,
            "nominal_control": "existing frozen base model, evaluated separately and not counted among 32 heterogeneous subjects",
            "seed": 20260830,
            "random_engine": "NumPy PCG64",
            "construction": "32 equal strata per factor; midpoint in each stratum; 512 seeded permutation restarts; maximize minimum pairwise Euclidean distance in normalized coordinates; lexicographic tie-break",
            "development_indices_zero_based": [index for index in range(32) if index % 4 != 3],
            "held_out_indices_zero_based": [index for index in range(32) if index % 4 == 3],
            "split_frozen_before_learner_performance_reveal": True,
        },
        "integrity_gates": INTEGRITY_THRESHOLDS,
        "p0_semantics": proposal["p0_semantics"],
        "frozen_inputs": {
            "base_model_sha256": FROZEN_SHA256["base_myoleg_model"],
            "v2_reference_sha256": FROZEN_SHA256["v2_reference"],
            "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
            "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
            "truth_field": TRUTH_FIELD,
            "formal_reference_sha256": FROZEN_SHA256["formal_reference"],
            "formal_manifest_sha256": FROZEN_SHA256["formal_manifest"],
            "proposal_content_sha256": proposal["proposal_content_sha256"],
        },
        "forbidden_in_generation": [
            "change formal/V2 reference",
            "change truth semantics",
            "add independent F0 factor in P0",
            "use held-out learner outcomes to alter subjects/ranges/split",
            "claim clinical or patient-population representativeness",
        ],
    }


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    input_before = frozen_input_hashes()
    prior_checksums_before = verify_checksum_manifest(PRIOR_ARTIFACT_DIRECTORY / "checksums.sha256")
    environment = runtime_environment()
    design = load_module(PRIOR_BUILDER_PATH, "frozen_cohort_design_builder")
    replay = load_module(REPLAY_BUILDER_PATH, "frozen_myoleg_replay_builder")

    scheme_payload = json.loads(PRIOR_FROZEN_FILES["prior_scheme"].read_text(encoding="utf-8"))
    scheme = next(item for item in scheme_payload["schemes"] if item["scheme_id"] == SCHEME_ID)
    if tuple(scheme["subject_level_factors"]) != SCHEME_A_FACTORS:
        raise RuntimeError("frozen Scheme A factor list changed")
    if tuple(scheme["biarticular_underlying_actuators"]) != BIARTICULAR_ACTUATORS:
        raise RuntimeError("frozen biarticular group changed")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    reference = replay.load_reference(V2_REFERENCE_PATH, "MYOLEG_V2_PRIMARY")
    reference_audit = replay.reference_audit(reference, model)
    if reference_audit["duration_s"] != 24.0 or reference_audit["sample_count"] != 401:
        raise RuntimeError("V2 reference identity changed")
    structural = design.structural_muscle_map(model, reference, replay)
    actual_biarticular = tuple(
        model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
        if model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index).endswith("_r")
        and structural[index]["structural_group"] == "HIP_KNEE_BIARTICULAR"
    )
    if actual_biarticular != BIARTICULAR_ACTUATORS:
        raise RuntimeError("compiled biarticular transmission group changed")

    proposal = proposal_payload(FROZEN_SHA256["prior_scheme"])
    # This write occurs before replay evaluation and the proposal contains no
    # replay outcomes, enforcing the direction of inference in the artifact.
    write_json(ARTIFACT_DIRECTORY / "PROPOSED_PARAMETER_RANGES.json", proposal)
    evaluation_manifest = {
        "manifest_id": "MYOLEG_COHORT_RANGE_EVALUATION_MANIFEST_V1",
        "frozen_before_replay": True,
        "proposal_content_sha256": proposal["proposal_content_sha256"],
        "endpoint_levels": [
            "EXTENDED_LOWER",
            "CONSERVATIVE_LOWER",
            "NOMINAL",
            "CONSERVATIVE_UPPER",
            "EXTENDED_UPPER",
        ],
        "interaction_profiles": INTERACTION_PROFILES,
        "reference_sha256": FROZEN_SHA256["v2_reference"],
        "truth_semantics_sha256": FROZEN_SHA256["truth_semantics"],
        "no_range_tuning_after_results": True,
    }
    evaluation_manifest["manifest_content_sha256"] = canonical_sha256(evaluation_manifest)
    write_json(ARTIFACT_DIRECTORY / "RANGE_EVALUATION_MANIFEST.json", evaluation_manifest)

    write_json(ARTIFACT_DIRECTORY / "SOURCE_BIBLIOGRAPHY.json", SOURCE_BIBLIOGRAPHY)
    write_csv(ARTIFACT_DIRECTORY / "PARAMETER_RANGE_LITERATURE_EVIDENCE.csv", literature_rows())
    write_csv(ARTIFACT_DIRECTORY / "ANTHROPOMETRY_RANGE_EVIDENCE.csv", anthropometry_rows())
    write_csv(ARTIFACT_DIRECTORY / "PASSIVE_PROPERTY_RANGE_EVIDENCE.csv", passive_rows(model, structural))
    write_csv(ARTIFACT_DIRECTORY / "PARAMETER_CORRELATION_AUDIT.csv", correlation_rows())
    write_provenance()

    nominal_scales = {factor: 1.0 for factor in SCHEME_A_FACTORS}
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def cached(profile_id: str, scales: dict[str, float]) -> dict[str, Any]:
        key = tuple(float(scales[factor]) for factor in SCHEME_A_FACTORS)
        if key not in cache:
            cache[key] = run_profile(profile_id, scales, structural, reference, replay)
        row = dict(cache[key])
        row["profile_id"] = profile_id
        row["factor_scales_json"] = json.dumps(scales, sort_keys=True, separators=(",", ":"))
        return row

    nominal_raw = cached("NOMINAL", nominal_scales)
    nominal = add_gates(nominal_raw, nominal_raw)
    endpoint_rows: list[dict[str, Any]] = []
    level_spec = (
        ("EXTENDED_LOWER", "EXTENDED_STRESS_ONLY", 0),
        ("CONSERVATIVE_LOWER", "PRIMARY_CONSERVATIVE", 0),
        ("NOMINAL", "NOMINAL", 1),
        ("CONSERVATIVE_UPPER", "PRIMARY_CONSERVATIVE", 2),
        ("EXTENDED_UPPER", "EXTENDED_STRESS_ONLY", 2),
    )
    for factor_id, spec in FACTOR_RANGES.items():
        for endpoint, role, index in level_spec:
            if endpoint.startswith("EXTENDED"):
                scale = float(spec["extended"][index])
            elif endpoint.startswith("CONSERVATIVE"):
                scale = float(spec["conservative"][index])
            else:
                scale = 1.0
            scales = dict(nominal_scales)
            scales[factor_id] = scale
            row = add_gates(cached(f"{factor_id}:{endpoint}", scales), nominal_raw)
            row = {
                "factor_id": factor_id,
                "endpoint": endpoint,
                "range_role": role,
                "scale": scale,
                "evidence_class": spec["evidence_class"],
                **row,
            }
            endpoint_rows.append(row)
    write_csv(ARTIFACT_DIRECTORY / "RANGE_ENDPOINT_REPLAY_RESULTS.csv", endpoint_rows)

    interaction_rows: list[dict[str, Any]] = []
    high_key = tuple(INTERACTION_PROFILES["ALL_CONSERVATIVE_HIGH"][factor] for factor in SCHEME_A_FACTORS)
    for profile_id, scales in INTERACTION_PROFILES.items():
        row = add_gates(cached(profile_id, scales), nominal_raw)
        row["interaction_role"] = "PRIMARY_CONSERVATIVE_CORNER"
        row["duplicate_parameterization_of"] = (
            "ALL_CONSERVATIVE_HIGH"
            if profile_id == "HIGH_MASS_HIGH_PASSIVE"
            and tuple(scales[factor] for factor in SCHEME_A_FACTORS) == high_key
            else ""
        )
        interaction_rows.append(row)
    write_csv(ARTIFACT_DIRECTORY / "RANGE_INTERACTION_SMOKE_RESULTS.csv", interaction_rows)

    if not all(bool(row["all_integrity_gates_pass"]) for row in endpoint_rows + interaction_rows):
        failed = [row["profile_id"] for row in endpoint_rows + interaction_rows if not row["all_integrity_gates_pass"]]
        raise RuntimeError(f"proposed range integrity failed closed: {failed}")

    mean_replay_s = float(np.mean([row["total_runtime_s"] for row in cache.values()]))
    size_rows = []
    for size, split in ((16, "12 development / 4 held-out"), (24, "16 development / 8 held-out"), (32, "24 development / 8 held-out")):
        size_rows.append(
            f"| {size} | {split} | {size / 6.0:.2f} | {size * mean_replay_s:.2f} s |"
        )
    (ARTIFACT_DIRECTORY / "COHORT_SIZE_DESIGN_AUDIT.md").write_text(
        f"""# Cohort size design audit

This is a simulation-design comparison, not a clinical power analysis.  Scheme
A has six factors and no defensible joint human probability distribution.

| heterogeneous subjects | proposed split | points per factor dimension (descriptive only) | measured one-reference-replay lower-bound cost |
|---:|---|---:|---:|
{chr(10).join(size_rows)}

The mean unique-profile P0/V2 integrity replay time on the frozen runtime was
`{mean_replay_s:.6f} s`.  The cost column excludes future learner fitting,
candidate landscapes and repeated method evaluations and is therefore only a
hardware-specific lower bound.

- 16 is too sparse once four are held out.
- 24 preserves eight held-out models but leaves only 16 development profiles.
- 32 permits 24 development and eight held-out profiles and is a natural size
  for deterministic six-dimensional space filling without claiming clinical
  representativeness.

Recommendation: **32 heterogeneous virtual subjects (24 development / 8
held-out), plus the existing nominal base model as a separate control**.
""",
        encoding="utf-8",
    )
    (ARTIFACT_DIRECTORY / "COHORT_SAMPLING_DESIGN_AUDIT.md").write_text(
        """# Cohort sampling design audit

| design | advantage | limitation | decision |
|---|---|---|---|
| deterministic predefined profiles | interpretable | weak six-dimensional coverage | retain for ablations, not primary cohort |
| Latin hypercube | one point per marginal stratum and works at n=32 | does not create a physiological joint distribution | selected with deterministic centered maximin construction |
| Sobol / low discrepancy | strong rectangular-space coverage | awkward exact nominal anchor and split semantics; still assumes a box | acceptable alternative, not selected |
| factorial extremes + nominal | clear corners | extreme-heavy and combinatorial | use only the five preregistered integrity corners |
| hybrid | can mix anchors and coverage | more design degrees of freedom and post-hoc discretion | reject for V1 |

Freeze a centered maximin Latin hypercube: six dimensions, 32 heterogeneous
profiles, NumPy PCG64 seed `20260830`, 512 permutation restarts and a
lexicographic tie-break.  The existing nominal base model is evaluated as a
separate control.  Held-out indices are every fourth generated row starting at
index 3 (`3,7,...,31`) and are frozen before learner performance is revealed.

The design fills the rectangular *model-parameter* ranges.  It does not assert
that factors are statistically independent in humans.  Anthropometric factors
are plausibly correlated and passive-group factors may be correlated, but no
quantitative covariance is defensible for these exact MyoLeg scales.  Therefore
V1 does not invent a covariance matrix and must report the independence-like
marginal design as a synthetic limitation.
""",
        encoding="utf-8",
    )
    protocol = generation_protocol(proposal)
    write_json(
        ARTIFACT_DIRECTORY / "MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_PROTOCOL_V1.json",
        protocol,
    )

    endpoint_primary = [row for row in endpoint_rows if row["range_role"] == "PRIMARY_CONSERVATIVE"]
    max_force_ratio = max(float(row["peak_force_ratio_vs_nominal"]) for row in endpoint_rows + interaction_rows)
    max_share_ratio = max(float(row["force_concentration_share_ratio_vs_nominal"]) for row in endpoint_rows + interaction_rows)
    report = f"""# {STAGE_ID}

## Final outcome

`{OUTCOME}`

Scheme A's structure and every proposed endpoint/corner pass the frozen offline
MyoLeg integrity gates, but all three `fpmax` intervals remain synthetic
model-property ranges.  The cohort may only be described as **heterogeneous
musculoskeletal virtual subjects**, never as a representative patient cohort.

## Q1 - Conservative range for every Scheme A factor

- `FEMUR_MASS_INERTIA_SCALE`: **0.88-1.12** (E2 mass-derived; inertia coupling approximate).
- `TIBIA_PATELLA_MASS_INERTIA_SCALE`: **0.87-1.13** (E2 shank-mass-derived; patella/inertia approximate).
- `FOOT_COMPLEX_MASS_INERTIA_SCALE`: **0.82-1.18** (E2 foot-mass-derived; low confidence for a small segmented body region).
- `HIP_ONLY_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).
- `KNEE_ONLY_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).
- `HIP_KNEE_BIARTICULAR_PASSIVE_FP_MAX_SCALE`: **0.95-1.05** (E4 synthetic sensitivity interval).

Extended stress-only envelopes are 0.76-1.24, 0.74-1.26 and 0.64-1.36 for
the three mass factors, and 0.90-1.10 for every passive group.  These are not
primary population claims.

## Q2 - Evidence interpretation

The fractional mass intervals are model-derived (E2) from reported adult
segment mass mean/SD.  DXA evidence directly supports that segment inertial
properties vary across and within populations but did not supply the numeric
bounds used here.  Proportional inertia scaling, fixed COM and grouped body
mapping are approximations.  The `fpmax` intervals are E4 synthetic ranges,
not direct or model-converted physiological distributions.

## Q3 - Coupled mass/inertia scaling

It is defensible only as a practical fixed-geometry V1 approximation:
`INERTIA_SCALING_IS_MODELING_APPROXIMATION`.  Literature reports substantially
different fractional variability for mass and the three inertia axes; one
scalar cannot represent all of them or replace subject-specific geometry.

## Q4 - Population-level fpmax

No.  MuJoCo `fpmax` is passive normalized force at `lmax` relative to `F0`.
Passive-joint and force-length studies show real heterogeneity, while
Rajagopal/OpenSim calibration changes different curve parameters.  No source
provides a defensible population-to-MuJoCo-`fpmax` conversion.

## Q5 - Different passive-group ranges

No distinct relative ranges are justified.  The frozen structural groups are
retained, including `{', '.join(BIARTICULAR_ACTUATORS)}`, but all three groups
use the common conservative 0.95-1.05 interval.  Structural grouping creates
different torque effects without inventing different marginal widths.

## Q6 - Correlations

Thigh/shank/foot scales are `POSSIBLY_CORRELATED`; passive-group factors are
also `POSSIBLY_CORRELATED`; mass-to-`fpmax` pairs have
`NO_USEFUL_EVIDENCE`.  A global fpmax factor is `KNOWN_CORRELATED` by algebraic
overlap and remains excluded.  No quantitative covariance matrix is frozen.

## Q7 - Numerical integrity

All `{len(endpoint_rows)}` endpoint rows (conservative and extended, including
nominal rows) and all `{len(interaction_rows)}` requested interaction rows pass.
The maximum peak-force ratio versus nominal is `{max_force_ratio:.6f}` and the
maximum concentration-share ratio is `{max_share_ratio:.6f}`.  There are no
solver warnings, nonfinite states, new contact/limit modes, equality failures or
truth-algebra failures.  This proves numerical/model integrity only.

The requested `HIGH_MASS_HIGH_PASSIVE` corner is exactly the same parameter
vector as `ALL_CONSERVATIVE_HIGH` because Scheme A contains only mass and
passive factors.  Both labels are retained and the duplicate is explicit.

## Q8 - Claim boundary

Use **structured musculoskeletal heterogeneity**.  Do not call the complete
cohort physiologically motivated or representative of patients.  Only its
mass marginals have anthropometric motivation, with modeling approximations.

## Q9 - Size and design

Freeze 32 heterogeneous profiles, split 24 development / 8 held-out, with the
nominal frozen model as a separate control.  Use the preregistered deterministic
centered maximin Latin hypercube and seed `20260830`.  This is simulation-space
coverage, not clinical power or a human joint probability model.

## Q10 - Readiness for generation

Yes, but only under the generated default-off protocol and the synthetic claim
boundary.  `MYOLEG_VIRTUAL_PATIENT_COHORT_GENERATION_V1` was **not executed**.

## Frozen boundaries

- primary P0 semantics unchanged; no simultaneous F0/fpmax dimensions
- V2 reference / base model / truth semantics unchanged
- formal reference / ROM protocol / `theta_shank = q_hip - q_knee` unchanged
- no cohort, landscape, five-parameter fit, NN/PINN, BO or robot access
- proposal SHA: `{proposal['proposal_content_sha256']}`
- evaluation-manifest SHA: `{evaluation_manifest['manifest_content_sha256']}`
"""
    (ARTIFACT_DIRECTORY / "MYOLEG_COHORT_PARAMETER_RANGE_EVIDENCE_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    input_after = frozen_input_hashes()
    prior_checksums_after = verify_checksum_manifest(PRIOR_ARTIFACT_DIRECTORY / "checksums.sha256")
    if input_before != input_after or prior_checksums_before != prior_checksums_after:
        raise RuntimeError("a frozen input or prior artifact changed during the stage")

    artifacts_before_metadata = {
        path.name: sha256_file(path)
        for path in sorted(ARTIFACT_DIRECTORY.iterdir())
        if path.is_file() and path.name not in {"metadata.json", "checksums.sha256"}
    }
    metadata = {
        "stage_id": STAGE_ID,
        "outcome": OUTCOME,
        "evidence_level": "OFFLINE_LITERATURE_MODEL_MAPPING_AND_P0_REPLAY_INTEGRITY",
        "scheme_id": SCHEME_ID,
        "scheme_factor_count": len(SCHEME_A_FACTORS),
        "scheme_frozen": True,
        "range_proposal_frozen_before_replay": True,
        "proposal_content_sha256": proposal["proposal_content_sha256"],
        "evaluation_manifest_content_sha256": evaluation_manifest["manifest_content_sha256"],
        "runtime_environment": environment,
        "runtime_s": time.perf_counter() - started,
        "unique_replay_profile_count": len(cache),
        "endpoint_row_count": len(endpoint_rows),
        "interaction_row_count": len(interaction_rows),
        "primary_endpoint_row_count": len(endpoint_primary),
        "all_endpoint_integrity_pass": all(bool(row["all_integrity_gates_pass"]) for row in endpoint_rows),
        "all_interaction_integrity_pass": all(bool(row["all_integrity_gates_pass"]) for row in interaction_rows),
        "fpmax_population_mapping": "NOT_JUSTIFIED",
        "mass_inertia_mapping": "INERTIA_SCALING_IS_MODELING_APPROXIMATION",
        "claim_boundary": "heterogeneous musculoskeletal virtual subjects",
        "recommended_cohort_size": 32,
        "recommended_split": {"development": 24, "held_out": 8},
        "sampling_seed": 20260830,
        "sampling_algorithm": "deterministic centered maximin Latin hypercube",
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "prior_artifact_checksums_before": prior_checksums_before,
        "prior_artifact_checksums_after": prior_checksums_after,
        "builder_script_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "artifact_sha256": artifacts_before_metadata,
        "cohort_generated": False,
        "subject_instances_generated": 0,
        "candidate_landscape_generated": False,
        "five_parameter_fit": False,
        "nn_trained": False,
        "pinn_trained": False,
        "bo_run": False,
        "robot_connected": False,
        "hardware_accessed": False,
        "control_modified": False,
        "safety_modified": False,
        "formal_reference_modified": False,
        "v2_reference_modified": False,
        "rom_protocol_modified": False,
        "truth_semantics_modified": False,
        "next_stage_executed": False,
    }
    write_json(ARTIFACT_DIRECTORY / "metadata.json", metadata)
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(ARTIFACT_DIRECTORY.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (ARTIFACT_DIRECTORY / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage_id": STAGE_ID,
                "outcome": OUTCOME,
                "artifact_count": len(checksum_lines) + 1,
                "endpoint_rows": len(endpoint_rows),
                "interaction_rows": len(interaction_rows),
                "unique_replays": len(cache),
                "all_gates_pass": True,
                "runtime_s": metadata["runtime_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
