"""Prepare versioned 900 s A/B requests without SDK initialization.

The generated request is NOT authorized for hardware. The separate runner
requires completed site records, reviewed budgets and an exact authorization
binding after those records have been frozen.
"""
import argparse
from dataclasses import asdict
import json
import math
import uuid
from pathlib import Path

from collection.wrench_process import WrenchBudgets


def prepare(output, rate_hz, budget_file=None):
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("explicit positive intended wrench rate required")
    budgets = None
    if budget_file:
        budgets = asdict(WrenchBudgets(**json.loads(Path(budget_file).read_text(encoding="utf-8"))))
        if budgets["provenance"] == "NOT_SAFETY_THRESHOLDS":
            raise ValueError("test-only budgets cannot prepare an approved hardware run")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="PREPARED_NOT_AUTHORIZED", live_runner_ready=False,
        schema_version=1, request_id=str(uuid.uuid4()),
        state_poll_hz=250.0, alignment_hz=50.0, supervisor_hz=100.0,
        rt_interval_ms=8, connection=None, expected_identity=None,
        allowed_power_states=None, criteria=None, lifecycle_budgets=None,
        production_entry="collection.real_robot_acquisition.RealRobotAcquisition",
        wrench_provider="collection.wrench_process.WrenchProcessProvider",
        intended_wrench_hz=rate_hz, budgets=budgets, duration_s=900,
        order=["A", "B"], cases={"A":"DISABLED_FOR_STATIONARY_A", "B":"isolated_wrench"},
        raw_streams=["robot_state.csv", "robot_wrench.csv", "aligned_snapshot.csv"],
        sidecars=["wrench_query_events.jsonl", "health_timeline.jsonl", "process_lifecycle.jsonl",
                  "cleanup_outcome.json", "end_state.json", "case_summary.json"],
        missing_evidence=["reviewed stationary authorization", "parent/child identity binding",
                          "reviewed budgets" if budgets is None else "budget/site applicability review",
                          "bounded parent native shutdown and end-state verification",
                          "reviewed recovery after blocked session"],
        robot_connected=False, reference_release="NO_GO", retry=False)
    (out/'stationary_request.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--intended-wrench-hz',required=True,type=float)
    parser.add_argument('--budget-file',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output_dir,args.intended_wrench_hz,args.budget_file),indent=2))


if __name__=='__main__':
    main()
