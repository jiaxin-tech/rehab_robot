"""Import and quality-check one TwinCAT dummy-leg episode CSV.

This command only reads already recorded files.  It never connects to TwinCAT,
the robot, or any controller, and it refuses to invent a joint angle when no
calibration is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

from measurement_validation.leg_episode import (
    _calibration_from_json,
    quality_report,
    read_leg_episode,
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path, help="dummy-leg episode CSV")
    parser.add_argument("--calibration", type=Path, default=None, help="joint calibration JSON")
    parser.add_argument("--column-map", type=Path, default=None, help="column override JSON")
    parser.add_argument("--minimum-rate-hz", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None, help="write the report to this JSON file")
    args = parser.parse_args(argv)

    calibration = _calibration_from_json(args.calibration) if args.calibration else None
    column_map = (
        json.loads(args.column_map.read_text(encoding="utf-8")) if args.column_map else None
    )
    episode = read_leg_episode(args.episode, calibration=calibration, column_map=column_map)
    report = quality_report(episode, minimum_rate_hz=args.minimum_rate_hz)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
