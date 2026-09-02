"""Run the post-manifest, nominal-only V3 simulator-integrity smoke."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1 import parameterization


OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_trajectory_parameterization_design_v1"
MANIFEST_PATH = OUTPUT / "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
SELECTION_PATH = OUTPUT / "V3_NOMINAL_SMOKE_SELECTION.csv"
OUTPUT_PATH = OUTPUT / "V3_NOMINAL_MYOLEG_SMOKE.csv"
CANDIDATE_BUILDER_PATH = ROOT / "external_simulation/myoleg_v2_candidate_domain_design_v1/build_candidate_domain.py"
REPLAY_BUILDER_PATH = ROOT / "external_simulation/myoleg_reference_trajectory_replay_v1/build_and_replay.py"

EXPECTED_MANIFEST_SHA256 = "6fc4259fe8b8c7d34c382f3d7840e97f58441ccb01098b1b3348ceedf8b7a745"
EXPECTED_CANDIDATE_BUILDER_SHA256 = "e8d3741099e8c6ac7f2b63c8b9fbfaf8f72da001c2714bcfff453b6f55ffd92e"
EXPECTED_REPLAY_BUILDER_SHA256 = "d60a9b1651b49307155b8b36bfdd881b595c604288f7c07a3237afe5f5feb32e"

ABS_LIMIT_TORQUE_MAX_NM = 0.005
REL_LIMIT_CONTRIBUTION_MAX = 0.0005
EQUALITY_RESIDUAL_MAX = 0.001
ALGEBRAIC_RESIDUAL_MAX_NM = 1.0e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("nominal smoke already exists; refusing overwrite")
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("frozen V3 candidate manifest changed")
    if sha256_file(CANDIDATE_BUILDER_PATH) != EXPECTED_CANDIDATE_BUILDER_SHA256:
        raise RuntimeError("frozen candidate builder changed")
    if sha256_file(REPLAY_BUILDER_PATH) != EXPECTED_REPLAY_BUILDER_SHA256:
        raise RuntimeError("frozen replay builder changed")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["smoke_completed_at_manifest_freeze"] is not False:
        raise RuntimeError("candidate manifest does not establish pre-smoke freeze")
    selection = read_csv(SELECTION_PATH)
    if len(selection) != 13:
        raise RuntimeError("frozen smoke selection count changed")

    candidate_builder = load_module(CANDIDATE_BUILDER_PATH, "_myoleg_v3_nominal_candidate_builder")
    replay = load_module(REPLAY_BUILDER_PATH, "_myoleg_v3_nominal_replay_builder")
    reference = candidate_builder.load_reference_adapter()
    subject_id, split, model, denominator = candidate_builder.model_from_record(None)
    rows: list[dict[str, Any]] = []
    for selected in selection:
        started = time.perf_counter()
        generated = parameterization.generate_v3_trajectory(
            reference,
            float(selected["beta_flex"]),
            float(selected["beta_extend"]),
        )
        replay_reference = {
            "time_s": reference["time_s"],
            "q": generated.q,
            "dq": generated.dq,
            "ddq": generated.ddq,
            "phases": reference["phases"],
            "rows": [],
        }
        prescribed, runtime = replay.prescribed_truth(model, replay_reference)
        limit = np.abs(prescribed["constraint_joint_limit_internal_nm"][:, 1])
        tau = np.asarray(prescribed["tau_truth_nm"][:, 1])
        absolute = float(np.max(limit))
        relative = float(np.max(limit / np.maximum(np.abs(tau), denominator)))
        equality = float(np.max(np.abs(prescribed["source_equality_residual"])))
        warnings = int(np.max(prescribed["warning_count"]))
        joint_count = int(np.max(prescribed["constraint_joint_limit_active_count"]))
        contact_count = int(np.max(prescribed["constraint_contact_active_count"]))
        tendon_count = int(np.max(prescribed["constraint_tendon_limit_active_count"]))
        algebraic = max(
            float(np.max(np.abs(prescribed[key])))
            for key in (
                "inverse_formula_residual_nm",
                "decomposition_residual_nm",
                "muscle_reconstruction_residual_nm",
            )
        )
        finite = all(
            bool(np.isfinite(prescribed[key]).all())
            for key in (
                "tau_truth_nm",
                "actuator_force_n",
                "tendon_length_m",
                "constraint_internal_nm",
                "inverse_formula_residual_nm",
            )
        )
        passed = bool(
            finite
            and warnings == 0
            and equality <= EQUALITY_RESIDUAL_MAX
            and algebraic <= ALGEBRAIC_RESIDUAL_MAX_NM
            and absolute <= ABS_LIMIT_TORQUE_MAX_NM
            and relative <= REL_LIMIT_CONTRIBUTION_MAX
            and joint_count <= 1
            and contact_count == 0
            and tendon_count == 0
        )
        rows.append({
            "smoke_rank": int(selected["smoke_rank"]),
            "selection_role": selected["selection_role"],
            "candidate_id": selected["candidate_id"],
            "candidate_index": int(selected["candidate_index"]),
            "beta_flex": float(selected["beta_flex"]),
            "beta_extend": float(selected["beta_extend"]),
            "subject_id": subject_id,
            "split": split,
            "model_role": "UNMODIFIED_NOMINAL_MYOLEG_V2",
            "purpose": "SIMULATOR_ARTIFACT_INTEGRITY_ONLY_NOT_SAFETY_OR_RANKING",
            "duration_s": 24.0,
            "sample_count": 401,
            "absolute_joint_limit_knee_contribution_nm": absolute,
            "relative_joint_limit_contribution": relative,
            "joint_limit_active_count": joint_count,
            "contact_active_count": contact_count,
            "tendon_limit_active_count": tendon_count,
            "source_equality_residual_max": equality,
            "algebraic_residual_max_nm": algebraic,
            "solver_warning_count": warnings,
            "all_finite": finite,
            "smoke_integrity_pass": passed,
            "prescribed_replay_wall_time_s": float(runtime["wall_time_s"]),
            "total_candidate_wall_time_s": time.perf_counter() - started,
            "mechanical_objective_computed": False,
            "ranking_computed": False,
        })
    if not all(row["smoke_integrity_pass"] for row in rows):
        failed = [row["candidate_id"] for row in rows if not row["smoke_integrity_pass"]]
        raise RuntimeError(f"nominal smoke integrity failure: {failed}")
    write_csv(OUTPUT_PATH, rows)
    print(json.dumps({
        "candidate_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "smoke_count": len(rows),
        "all_pass": True,
        "total_prescribed_replay_s": sum(float(row["prescribed_replay_wall_time_s"]) for row in rows),
        "objective_or_ranking_computed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
