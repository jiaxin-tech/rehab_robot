"""Read-only timing reconstruction of recorded attempts; never opens an SDK."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def analyze_attempt(directory: Path) -> dict:
    directory = Path(directory)
    audit_path = directory / "wrench_query_events.jsonl"
    metadata_path = directory / "metadata.json"
    command_path = directory / "trajectory_command.csv"
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    acks = {e["child_event_id"]: e for e in events if e["type"] == "DURABLE_ACK"}
    queries = []
    for event in events:
        if event["type"] not in ("QUERY_SUCCESS", "QUERY_ERROR"):
            continue
        ack = acks.get(event["event_id"])
        sample = (event.get("frame") or {}).get("host_monotonic_time_s")
        queries.append(dict(query_id=event["query_id"], result=event["type"],
            native_duration_ms=(event["completion_ns"] - event["native_start_ns"]) / 1e6,
            intent_to_native_ms=(event["native_start_ns"] - event["intent_ns"]) / 1e6,
            completion_to_durable_ms=(ack["durable_ns"] - event["completion_ns"]) / 1e6 if ack else None,
            sample_age_at_ack_ms=(ack["timestamp_ns"] / 1e9 - sample) * 1e3
                if ack and sample is not None else None))
    with command_path.open(encoding="utf-8-sig", newline="") as stream:
        commands = sum(1 for _ in csv.DictReader(stream))
    outcomes = [e["outcome"] for e in events if e["type"] == "CLEANUP_OUTCOME"]
    contradictory = any(o.get("NORMAL_CLEANUP") is True and o.get("failure_latch") for o in outcomes)
    return dict(directory=str(directory.resolve()), command_rows=commands,
        zero_recorded_trajectory_commands=commands == 0,
        note="Zero recorded commands alone does not prove physical power-off or zero motion.",
        failure_reasons=[e["reason"] for e in events if e["type"] == "FAILURE_LATCH"],
        queries=queries, cleanup_log_contradiction=contradictory,
        release_sha256=metadata.get("trial_release", {}).get("approval_sha256"),
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (audit_path, metadata_path, command_path)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(p.parent for p in args.input_root.glob("*/wrench_query_events.jsonl"))
    if not directories:
        parser.error("no recorded attempts found")
    payload = dict(mode="OFFLINE_LOG_AUDIT_NO_HARDWARE", attempts=[analyze_attempt(p) for p in directories])
    # Exclusive creation prevents overwriting previous evidence or a raw log.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Audited {len(directories)} recorded attempts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
