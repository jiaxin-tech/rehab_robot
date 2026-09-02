"""Generate the preregistered development replay subset in frozen MyoLeg env.

This helper intentionally imports only stdlib and NumPy before invoking the
existing deterministic on-demand replay API.  It rejects held-out identifiers
before any replay call and stores only arrays preregistered by the root-cause
protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "external_simulation_audits/myoleg_v2_personalization_signal_root_cause_audit_v1"
PROTOCOL = OUTPUT / "PERSONALIZATION_SIGNAL_ROOT_CAUSE_PROTOCOL.json"
CACHE = ROOT / "external_simulation/data/myoleg_v2_personalization_signal_root_cause_audit_v1/development_replay_subset.npz"
MANIFEST = OUTPUT / "DEVELOPMENT_REPLAY_CACHE_MANIFEST.json"
REPLAY_API = ROOT / "external_simulation/myoleg_v2_truth_landscape_generation_v1/replay_api.py"
EXPECTED_PROTOCOL_SHA = "2beac2ffb512783bcbe6dfcf60e8d64d9b6be8a5fe2122b8c77da876e6202bbb"
HELD_OUT = {
    "MYOLEG_VP_004", "MYOLEG_VP_008", "MYOLEG_VP_012", "MYOLEG_VP_016",
    "MYOLEG_VP_020", "MYOLEG_VP_024", "MYOLEG_VP_028", "MYOLEG_VP_032",
}
ARRAY_NAMES = (
    "tau_truth_nm", "mass_term_nm", "bias_term_nm", "passive_internal_nm",
    "actuator_internal_nm", "constraint_internal_nm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_replay_api() -> Any:
    spec = importlib.util.spec_from_file_location("_root_cause_replay_cache_api", REPLAY_API)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen replay API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("root-cause protocol SHA changed")
    if CACHE.exists() or MANIFEST.exists():
        raise RuntimeError("replay cache or manifest already exists; refusing overwrite")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    subject_ids = [row["subject_id"] for row in protocol["replay_subset"]["subject_rows"]]
    candidate_rows = protocol["replay_subset"]["candidate_rows"]
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    if len(subject_ids) != 6 or len(candidate_ids) != 20 or set(subject_ids).intersection(HELD_OUT):
        raise RuntimeError("development-only replay plan identity mismatch")
    api = load_replay_api()
    stored = {name: np.zeros((len(subject_ids), len(candidate_ids), 401, 2), dtype=np.float64) for name in ARRAY_NAMES}
    warning_max = np.zeros((len(subject_ids), len(candidate_ids)), dtype=np.int16)
    time_s = None
    started = time.perf_counter()
    for subject_index, subject_id in enumerate(subject_ids):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            if subject_id in HELD_OUT:
                raise PermissionError(f"held-out replay denied before API call: {subject_id}")
            payload = api.replay_subject_candidate(subject_id, candidate_id)
            if payload["split"] != "DEVELOPMENT" or payload["subject_id"] != subject_id:
                raise RuntimeError("replay returned wrong split/identity")
            arrays = payload["arrays"]
            current_time = np.asarray(arrays["time_s"], dtype=float)
            if time_s is None:
                time_s = current_time
            elif not np.array_equal(time_s, current_time):
                raise RuntimeError("replay time grid mismatch")
            for name in ARRAY_NAMES:
                value = np.asarray(arrays[name], dtype=np.float64)
                if value.shape != (401, 2) or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid replay array {name}: {subject_id}/{candidate_id}")
                stored[name][subject_index, candidate_index] = value
            warning_max[subject_index, candidate_index] = int(np.max(arrays["warning_count"]))
            if warning_max[subject_index, candidate_index] != 0:
                raise RuntimeError(f"replay warning: {subject_id}/{candidate_id}")
    if time_s is None:
        raise RuntimeError("empty replay plan")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_name(CACHE.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            subject_ids=np.asarray(subject_ids), candidate_ids=np.asarray(candidate_ids),
            proposal_index=np.asarray([row["proposal_index"] for row in candidate_rows], dtype=np.int32),
            alpha=np.asarray([row["alpha"] for row in candidate_rows], dtype=np.float64),
            time_s=np.asarray(time_s, dtype=np.float64), warning_max=warning_max, **stored,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, CACHE)
    manifest = {
        "cache_id": "MYOLEG_V2_ROOT_CAUSE_DEVELOPMENT_REPLAY_CACHE_V1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "cache_path": str(CACHE.relative_to(ROOT)), "cache_sha256": sha256(CACHE),
        "subject_ids": subject_ids, "candidate_ids": candidate_ids,
        "replay_pair_count": len(subject_ids) * len(candidate_ids), "sample_count_per_pair": 401,
        "stored_array_names": list(ARRAY_NAMES), "warning_count_max": int(np.max(warning_max)),
        "held_out_scientific_truth_access_count": 0, "held_out_replay_api_call_count": 0,
        "runtime_s": time.perf_counter() - started,
        "runtime_environment": {
            "python": platform.python_version(), "python_executable": sys.executable,
            "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"),
            "myosuite": importlib.metadata.version("myosuite"),
        },
        "replay_api_sha256": sha256(REPLAY_API),
    }
    atomic_json(MANIFEST, manifest)
    print(json.dumps({
        "cache_sha256": manifest["cache_sha256"], "replay_pairs": manifest["replay_pair_count"],
        "held_out_scientific_truth_access_count": 0, "runtime_s": manifest["runtime_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
