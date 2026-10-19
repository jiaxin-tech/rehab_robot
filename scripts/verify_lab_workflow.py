"""Run offline synthetic-hardware failure drills and save their evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_loaded_round_lifecycle.py",
    "tests/test_robot_leg_round.py",
    "tests/test_wrench_production_process.py",
    "tests/test_wrench_deadline_regressions.py",
    "tests/test_run_rehab_experiment.py",
    "tests/test_robot_trajectory_executor.py",
)


def verify(output, *, frozen_config=None):
    output = Path(output).resolve()
    environment = dict(os.environ)
    if frozen_config is not None:
        frozen_config = Path(frozen_config).resolve(strict=True)
        environment["REHAB_TEST_ROUND_CONFIG"] = str(frozen_config)
    output.mkdir(parents=True, exist_ok=False)
    # Keep live test files outside the watched source checkout. Retain this
    # unique scratch directory too, including when copying evidence fails.
    scratch = Path(tempfile.mkdtemp(prefix="rehab_offline_")) / "synthetic_runs"
    command = [sys.executable, "-X", "utf8", "-m", "pytest", "-c", "pytest-core.ini",
               *TESTS, "-q", "--basetemp", str(scratch),
               "--junitxml", str(output / "tests.xml")]
    with (output / "verification.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
    copy_error = None
    try:
        shutil.copytree(scratch, output / "synthetic_runs")
    except OSError as exc:
        copy_error = str(exc)
    suites = ET.parse(output / "tests.xml").getroot().findall("testsuite") if (output / "tests.xml").is_file() else []
    evidence = dict(software_verification_only=True, hardware_verified=False,
        synthetic_hardware=True, robot_connection_attempted=False, command=command,
        frozen_config=str(frozen_config) if frozen_config else None,
        scratch_artifacts=str(scratch), artifact_copy_error=copy_error,
        returncode=result.returncode, passed=result.returncode == 0 and copy_error is None,
        counts={key: sum(int(s.get(key, 0)) for s in suites)
                for key in ("tests", "failures", "errors", "skipped")},
        cases=[dict(name=case.get("name"), outcome=("failed" if case.find("failure") is not None
                    or case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed"))
               for suite in suites for case in suite.findall("testcase")],
        limitations=["Virtual trajectory clock does not validate physical RT timing.",
                     "Synthetic CSV files are not experiment or paper results.",
                     "Persistent disk loss may prevent local reports; stderr is a fallback, not durable evidence.",
                     "Software cleanup completion is not physical stop verification."])
    (output / "verification.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
    lines = ["# Offline Lab Workflow Verification", "", "SYNTHETIC HARDWARE ONLY / NOT LAB RESULTS", "",
             f"Passed: {evidence['passed']}; counts: {evidence['counts']}", "",
             "Covers loaded same-session RT preparation, real release/preflight/executor wiring,",
             "capture, actual frozen learner, cleanup, disconnect, query hang, disk failure,",
             "operator cancellation, real worker timeout and collision state rejection.", "",
             "Inspect tests.xml, verification.log and synthetic_runs for per-round raw artifacts.", "",
             *["- " + item for item in evidence["limitations"]]]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (
        "offline_lab_verification_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    parser.add_argument("--frozen-config", type=Path, help="Portable frozen source paths for test fixtures only")
    args = parser.parse_args(argv)
    evidence = verify(args.output, frozen_config=args.frozen_config)
    print(json.dumps(dict(output=str(args.output.resolve()), passed=evidence["passed"],
                          counts=evidence["counts"]), indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
