"""Deterministic on-demand replay for one frozen MyoLeg-V2 subject/candidate.

This module deliberately regenerates full arrays instead of reading the compact
oracle landscape.  It is the execution-like interface future offline research
must use to reveal a candidate outcome.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
CANDIDATE_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_candidate_domain_design_v1/MYOLEG_V2_CANDIDATE_DOMAIN_V1_MANIFEST.json"
CANDIDATE_BUILDER = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"

FROZEN_COHORT_SHA256 = "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057"
FROZEN_CANDIDATE_SHA256 = "0daebd0dd418f06a92613d116fb484c6627325eb6e395c6c75124ddff0361ae7"
TRUTH_SEMANTIC_VERSION = "MYOLEG_DYNAMICS_TRUTH_SEMANTICS_V1"
TRUTH_FIELD = "TAU_MY0LEG_REQUIRED_DRIVE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(COHORT_MANIFEST) != FROZEN_COHORT_SHA256:
        raise RuntimeError("frozen cohort manifest SHA changed")
    if _sha256(CANDIDATE_MANIFEST) != FROZEN_CANDIDATE_SHA256:
        raise RuntimeError("frozen candidate manifest SHA changed")
    cohort = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if len(cohort["subjects"]) != 32 or len(candidates["ordered_included_candidates"]) != 16675:
        raise RuntimeError("frozen replay identity count changed")
    return cohort, candidates


def replay_subject_candidate(subject_id: str, candidate_id: str) -> dict[str, Any]:
    """Regenerate complete prescribed-replay arrays for one frozen pair.

    The return value contains q/dq/ddq, all generalized-force decomposition
    arrays emitted by the frozen replay, identity metadata, and runtime.  It
    never consults the compact landscape or oracle ordering.
    """

    cohort, candidate_manifest = _frozen_inputs()
    subject = next((row for row in cohort["subjects"] if row["subject_id"] == subject_id), None)
    candidate = next(
        (row for row in candidate_manifest["ordered_included_candidates"] if row["candidate_id"] == candidate_id),
        None,
    )
    if subject is None:
        raise KeyError(f"unknown frozen subject_id: {subject_id}")
    if candidate is None:
        raise KeyError(f"unknown frozen candidate_id: {candidate_id}")

    candidate_builder = _module(CANDIDATE_BUILDER, "_myoleg_v2_candidate_builder_replay_api")
    replay_builder = _module(REPLAY_BUILDER, "_myoleg_truth_replay_api")
    reference = candidate_builder.load_reference_adapter()
    generated = candidate_builder.generate_candidate(reference, *map(float, candidate["alpha"]))
    replay_reference = {
        "time_s": reference["time_s"],
        "q": generated["q"],
        "dq": generated["dq"],
        "ddq": generated["ddq"],
        "phases": reference["phases"],
        "rows": [],
    }
    reconstructed_id, split, model, _ = candidate_builder.model_from_record(subject)
    if reconstructed_id != subject_id:
        raise RuntimeError("subject reconstruction identity mismatch")
    prescribed, runtime = replay_builder.prescribed_truth(model, replay_reference)
    arrays: dict[str, np.ndarray] = {
        "time_s": np.asarray(reference["time_s"]).copy(),
        "q_rad": np.asarray(generated["q"]).copy(),
        "dq_rad_s": np.asarray(generated["dq"]).copy(),
        "ddq_rad_s2": np.asarray(generated["ddq"]).copy(),
    }
    arrays.update({key: np.asarray(value).copy() for key, value in prescribed.items()})
    return {
        "subject_id": subject_id,
        "split": split,
        "candidate_id": candidate_id,
        "proposal_index": int(candidate["proposal_index"]),
        "alpha": [float(value) for value in candidate["alpha"]],
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION,
        "truth_field": TRUTH_FIELD,
        "arrays": arrays,
        "runtime": runtime,
    }

