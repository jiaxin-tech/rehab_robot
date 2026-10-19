import csv
import json

from scripts.audit_empty_load_attempts import analyze_attempt


def test_audit_separates_query_storage_and_sample_age(tmp_path):
    events = [
        dict(type="QUERY_SUCCESS", event_id=5, query_id=1,
             intent_ns=1_000_000_000, native_start_ns=1_110_000_000,
             completion_ns=1_111_000_000, frame=dict(host_monotonic_time_s=1.1105)),
        dict(type="DURABLE_ACK", child_event_id=5, durable_ns=1_165_000_000,
             timestamp_ns=1_180_000_000),
        dict(type="FAILURE_LATCH", reason="wrench_sample_stale_or_clock_invalid"),
        dict(type="CLEANUP_OUTCOME", outcome=dict(NORMAL_CLEANUP=True,
             failure_latch=["wrench_sample_stale_or_clock_invalid"])),
    ]
    (tmp_path / "wrench_query_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with (tmp_path / "trajectory_command.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(["host_time_s", "x"])
    report = analyze_attempt(tmp_path)
    query = report["queries"][0]
    assert query["native_duration_ms"] == 1
    assert query["intent_to_native_ms"] == 110
    assert query["completion_to_durable_ms"] == 54
    assert round(query["sample_age_at_ack_ms"], 6) == 69.5
    assert report["cleanup_log_contradiction"]
    assert report["zero_recorded_trajectory_commands"]
    assert len(report["source_sha256"]) == 3
