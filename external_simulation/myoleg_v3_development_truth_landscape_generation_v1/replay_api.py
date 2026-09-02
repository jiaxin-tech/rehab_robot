"""Development-only deterministic on-demand replay for frozen V3 candidates."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization


V3_MANIFEST = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
V3_TABLE = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1/V3_KINEMATIC_CANDIDATE_TABLE.csv"
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
CANDIDATE_BUILDER = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"

FROZEN_V3_MANIFEST_SHA256 = "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745"
FROZEN_V3_TABLE_SHA256 = "376fb3bc036b742714271f42fa457f61657ef31e931faf9f23a748a8985cf774"
FROZEN_COHORT_SHA256 = "31fbdfcf26dad04d13d4fbf62fb69b1ae6a0c14fc3d3acbeb7272dd1cc6a7057"
FROZEN_CANDIDATE_BUILDER_SHA256 = "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e"
FROZEN_REPLAY_BUILDER_SHA256 = "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e"
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


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    actual = {
        V3_MANIFEST: _sha256(V3_MANIFEST), V3_TABLE: _sha256(V3_TABLE),
        COHORT_MANIFEST: _sha256(COHORT_MANIFEST), CANDIDATE_BUILDER: _sha256(CANDIDATE_BUILDER),
        REPLAY_BUILDER: _sha256(REPLAY_BUILDER),
    }
    expected = {
        V3_MANIFEST: FROZEN_V3_MANIFEST_SHA256, V3_TABLE: FROZEN_V3_TABLE_SHA256,
        COHORT_MANIFEST: FROZEN_COHORT_SHA256, CANDIDATE_BUILDER: FROZEN_CANDIDATE_BUILDER_SHA256,
        REPLAY_BUILDER: FROZEN_REPLAY_BUILDER_SHA256,
    }
    if actual != expected:
        raise RuntimeError("frozen V3 replay input SHA changed")
    cohort = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    with V3_TABLE.open(newline="", encoding="utf-8") as stream:
        table = {row["candidate_id"]: row for row in csv.DictReader(stream)}
    if len(table) != 625 or list(table) != manifest["ordered_candidate_ids"]:
        raise RuntimeError("frozen V3 candidate order changed")
    return cohort, manifest, table


def replay_v3_subject_candidate(subject_id: str, candidate_id: str) -> dict[str, Any]:
    """Regenerate full prescribed truth for one development pair.

    Held-out IDs and nominal control are rejected before any simulator module is
    loaded.  This function never reads compact landscape J or oracle data.
    """

    cohort, _, table = _inputs()
    development_ids = set(cohort["development_subject_ids"])
    held_out_ids = set(cohort["held_out_subject_ids"])
    if subject_id in held_out_ids:
        raise PermissionError("held-out scientific replay is sealed")
    if subject_id not in development_ids:
        raise KeyError(f"unknown or non-development subject_id: {subject_id}")
    candidate = table.get(candidate_id)
    if candidate is None:
        raise KeyError(f"unknown frozen V3 candidate_id: {candidate_id}")
    subject = next(row for row in cohort["subjects"] if row["subject_id"] == subject_id)
    candidate_builder = _module(CANDIDATE_BUILDER, "_myoleg_v3_query_candidate_builder")
    replay_builder = _module(REPLAY_BUILDER, "_myoleg_v3_query_replay_builder")
    reference = candidate_builder.load_reference_adapter()
    generated = parameterization.generate_v3_trajectory(
        reference, float(candidate["beta_flex"]), float(candidate["beta_extend"])
    )
    replay_reference = {
        "time_s": reference["time_s"], "q": generated.q, "dq": generated.dq,
        "ddq": generated.ddq, "phases": reference["phases"], "rows": [],
    }
    reconstructed, split, model, _ = candidate_builder.model_from_record(subject)
    if reconstructed != subject_id or split != "DEVELOPMENT":
        raise RuntimeError("development model reconstruction identity mismatch")
    prescribed, runtime = replay_builder.prescribed_truth(model, replay_reference)
    arrays = {
        "time_s": np.asarray(reference["time_s"]).copy(),
        "q_rad": generated.q.copy(), "dq_rad_s": generated.dq.copy(), "ddq_rad_s2": generated.ddq.copy(),
    }
    arrays.update({key: np.asarray(value).copy() for key, value in prescribed.items()})
    return {
        "subject_id": subject_id, "split": split,
        "candidate_id": candidate_id, "candidate_index": int(candidate["candidate_index"]),
        "beta_flex": float(candidate["beta_flex"]), "beta_extend": float(candidate["beta_extend"]),
        "truth_semantic_version": TRUTH_SEMANTIC_VERSION, "truth_field": TRUTH_FIELD,
        "arrays": arrays, "runtime": runtime,
        "compact_landscape_or_oracle_read": False,
    }

