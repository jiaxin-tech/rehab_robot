"""Validate the frozen V3 development-only replay API without storing arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_simulation.myoleg_v3_development_truth_landscape_generation_v1.replay_api import replay_v3_subject_candidate


OUTPUT = ROOT / "external_simulation_audits/myoleg_v3_development_truth_landscape_generation_v1/V3_REPLAY_API_VALIDATION.json"


def fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("replay API validation already exists")
    first = replay_v3_subject_candidate("MYOLEG_VP_001", "MYOLEG_V3_K0312")
    second = replay_v3_subject_candidate("MYOLEG_VP_001", "MYOLEG_V3_K0312")
    keys_equal = set(first["arrays"]) == set(second["arrays"])
    arrays_equal = keys_equal and all(np.array_equal(first["arrays"][key], second["arrays"][key]) for key in first["arrays"])
    first_sha = fingerprint(first["arrays"])
    second_sha = fingerprint(second["arrays"])
    held_out_blocked = False
    held_out_error = ""
    try:
        replay_v3_subject_candidate("MYOLEG_VP_004", "MYOLEG_V3_K0312")
    except PermissionError as exc:
        held_out_blocked = True
        held_out_error = str(exc)
    payload = {
        "validation_id": "V3_DETERMINISTIC_REPLAY_API_VALIDATION_V1",
        "subject_id": "MYOLEG_VP_001", "candidate_id": "MYOLEG_V3_K0312",
        "truth_semantic_version": first["truth_semantic_version"],
        "truth_field": first["truth_field"],
        "array_key_count": len(first["arrays"]),
        "array_keys_equal": keys_equal, "all_arrays_equal": arrays_equal,
        "first_array_payload_sha256": first_sha, "second_array_payload_sha256": second_sha,
        "held_out_test_subject_id": "MYOLEG_VP_004",
        "held_out_rejected_before_replay": held_out_blocked,
        "held_out_error": held_out_error,
        "compact_landscape_or_oracle_read": False,
        "pass": arrays_equal and first_sha == second_sha and held_out_blocked,
    }
    if not payload["pass"]:
        raise RuntimeError(f"V3 replay API validation failed: {payload}")
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
