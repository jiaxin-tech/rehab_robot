"""Prepare and optionally train an offline history; never command hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from measurement_validation.robot_leg_history import prepare_history, read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def replay(manifest, output_dir, *, train=False, timeout_s=120):
    request, audit = prepare_history(manifest)
    if not 0 < timeout_s <= 3600:
        raise ValueError("training timeout must be positive and at most 3600 seconds")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / "training_request.json"
    _write_json(request_path, request)
    _write_json(output / "history_audit.json", audit)
    status = dict(status="HISTORY_PREPARED", data_classification=audit["data_classification"],
                  rounds=len(request["history"]), training_requested=train, motion_approved=False,
                  policy_state_replayed=False, completed_rounds_only=audit["completed_rounds_only"],
                  total_attempt_budget_accounted=audit["total_attempt_budget_accounted"],
                  replay_mode=audit["replay_mode"],
                  training_request_sha256=sha256_file(request_path),
                  baseline_episode_id=audit["baseline_episode_id"])
    _write_json(output / "replay_status.json", status)
    if train:
        try:
            with (output / "training.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round", str(request_path),
                     "--output", str(output / "model_update.json")], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"isolated trainer failed with exit {completed.returncode}; see training.log")
            update_path = output / "model_update.json"
            update = read_json(update_path)
            if update.get("status") not in ("MODEL_UPDATED", "POLICY_REPLAYED"):
                raise RuntimeError("isolated trainer did not return a completed model/policy replay")
            strict = audit["replay_mode"] == "strict_policy"
            update.update(data_classification=audit["data_classification"], offline_history_replay=True,
                          motion_approved=False,
                          replay_scope=audit["budget_scope"], history_manifest_sha256=audit["manifest_sha256"])
            if not strict:
                update.update(action=None, policy_state_replayed=False,
                              completed_rounds_only=True, total_attempt_budget_accounted=False)
            _write_json(update_path, update)
            status.update(status="OFFLINE_POLICY_REPLAYED" if strict else "OFFLINE_MODEL_UPDATED",
                          policy_state_replayed=update["policy_state_replayed"],
                          completed_trials=update.get("completed_trials", len(request["history"])),
                          attempted_trials=update.get("attempted_trials"),
                          model_update_sha256=sha256_file(update_path))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            status.update(status="TRAINING_FAILED", error=f"{type(exc).__name__}: {exc}")
            _write_json(output / "replay_status.json", status)
            raise
        _write_json(output / "replay_status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; existing results are never overwritten")
    parser.add_argument("--train", action="store_true", help="invoke the existing frozen trainer in an isolated subprocess")
    parser.add_argument("--timeout-s", type=float, default=120)
    args = parser.parse_args(argv)
    print(json.dumps(replay(args.manifest, args.output_dir, train=args.train, timeout_s=args.timeout_s), ensure_ascii=False))


if __name__ == "__main__":
    main()
