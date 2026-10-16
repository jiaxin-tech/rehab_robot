"""Bounded native ADS recording. No enable, reset or motion commands."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from collection.twincat_leg import LegCaptureProcess, atomic_json, positive
from measurement_validation.offline_leg import inspect_file
from safety.robot_trial_release import sha256_file


def record(config, directory, duration_s, *, factory=LegCaptureProcess):
    duration_s = positive(duration_s, "duration_s")
    if duration_s > 3600:
        raise ValueError("one segment must not exceed 3600 seconds")
    config, directory = Path(config).resolve(strict=True), Path(directory).resolve()
    worker = factory(config, directory)
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(status="STARTING", motion_commanded=False, duration_requested_s=duration_s,
                 config_path=str(config), config_sha256=sha256_file(config))
    atomic_json(directory / "segment_status.json", state)
    started = False
    try:
        started = True
        worker.start()
        state["status"] = "RECORDING"
        atomic_json(directory / "segment_status.json", state)
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            reason = worker.health_reason()
            if reason:
                raise RuntimeError(reason)
            time.sleep(min(.02, max(0., deadline-time.perf_counter())))
        worker.stop()
        started = False
        quality = inspect_file(directory / "leg.csv")
        atomic_json(directory / "quality.json", quality)
        state.update(status="CAPTURE_COMPLETE", csv_sha256=sha256_file(directory / "leg.csv"),
                     rows=quality["rows"], sample_rate_hz=quality.get("sample_rate_hz"),
                     long_intervals=len(quality["long_intervals"]),
                     note="Capture completion is not calibration or dynamics approval.")
        atomic_json(directory / "segment_status.json", state)
        return state
    except BaseException as exc:
        state.update(status="CAPTURE_FAILED", reason=f"{type(exc).__name__}: {exc}")
        if started:
            try:
                worker.stop()
            except BaseException as cleanup:
                state["cleanup_error"] = str(cleanup)
        atomic_json(directory / "segment_status.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=10.)
    args = parser.parse_args(argv)
    result = record(args.config, args.output_dir, args.duration_s)
    print(f"{result['status']}: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
