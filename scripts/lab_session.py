"""Lab workflow launcher. Check is offline; device and motion actions are explicit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("check", "initialize", "diagnostics", "events", "capture", "quality", "calibrate", "observe", "release", "empty", "round", "train")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("required path is missing")
    return (ROOT / value).resolve()


def write(filename, value):
    with Path(filename).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def step(directory, name, module, arguments, *, runner=subprocess.run):
    command = [sys.executable, "-X", "utf8", "-m", module, *map(str, arguments)]
    with (directory / f"{name}.log").open("x", encoding="utf-8") as log:
        result = runner(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return dict(name=name, command=command, returncode=result.returncode,
                log=str(directory / f"{name}.log"))


def check(profile, directory):
    from scripts.run_robot_leg_round import check_config, paths
    from hardware.safety_events import validate_collision_source
    from scripts.acquire_robot_data import load_live_wrench_config
    findings = []
    for name in ("numpy", "pandas", "scipy", "mujoco"):
        if importlib.util.find_spec(name) is None:
            findings.append("Python package missing: " + name)
    if sys.version_info[:2] != (3, 12):
        findings.append("bundled Windows SDK requires CPython 3.12")
    config_path = path(profile["round_config"])
    trial_kind = None
    try:
        findings.extend(check_config(config_path))
        config = paths(config_path)
        trial_kind = config.get("trial_kind")
        if trial_kind not in ("dummy_leg_reference", "dummy_leg_capture_only"):
            findings.append("round requires reviewed dummy_leg_reference or dummy_leg_capture_only, current=" + str(trial_kind))
    except Exception as exc:
        findings.append(f"round inspection: {type(exc).__name__}:{exc}")
    try:
        validate_collision_source("events", profile["diagnostic_event_max_age_s"])
        positive(profile["diagnostic_duration_s"], "diagnostic duration")
        capture = load(path(profile["leg_config"]))
        if not Path(capture["ads_dll_path"]).is_file():
            findings.append("native ADS DLL missing: " + capture["ads_dll_path"])
        wrench = load_live_wrench_config(path(profile["observation_wrench_config"]))
        if any(wrench["connection"][k] != profile[k] for k in ("robot_ip", "local_ip")):
            findings.append("observation and launcher IP mismatch")
    except Exception as exc:
        findings.append(f"capture configuration: {type(exc).__name__}:{exc}")
    try:
        empty_release(profile)
        empty = "CURRENT_HASH_BOUND_RELEASE_PRESENT_NOT_LIVE_VALIDATED"
    except Exception as exc:
        empty = f"BLOCKED: {type(exc).__name__}:{exc}"
    training_blockers = list(findings)
    if trial_kind != "dummy_leg_reference":
        training_blockers.append("training intentionally disabled for current trial_kind=" + str(trial_kind))
    report = dict(offline=True, device_connected=False, motion_approved=False,
        round_trial_kind=trial_kind, round_blockers=findings, training_blockers=training_blockers,
        empty_load=empty, hardware_unverified=[
            "Safety callback initial state and continuity",
            "State/wrench freshness and alignment under production logging",
            "Complete supervised reference execution and cleanup",
            "Independent angle accuracy across intended trajectory range"],
        next_actions=["events", "capture", "observe", "empty (current release required)",
                      "round (reviewed capture-only or training setup required)"])
    write(directory / "blockers.json", report)
    return report


def empty_release(profile):
    from safety.empty_load_commissioning_release import load_and_validate_empty_load_commissioning_release
    empty = profile["empty_load"]
    release = load_and_validate_empty_load_commissioning_release(
        path(empty["release"]), reference_path=path(empty["reference"]),
        safety_config_path=path(empty["safety_config"]), frame_config_path=path(empty["frame_config"]),
        anchor_path=path(empty["anchor"]), wrench_config_path=path(empty["wrench_config"]))
    payload = release.payload
    issued = datetime.fromisoformat(payload["issued_at"].replace("Z", "+00:00"))
    expiry = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
    maximum = positive(profile["empty_load_max_release_lifetime_s"], "release lifetime")
    if not 0 < (expiry-issued).total_seconds() <= maximum:
        raise PermissionError("release lifetime exceeds launcher limit; obtain a genuinely short-lived reviewed release")
    if payload.get("training_allowed") is not False or payload.get("single_cycle") is not True:
        raise PermissionError("empty-load must exclude training and allow only one cycle")
    return release


def execute(action, profile, directory, *, live=False, confirm=input, runner=subprocess.run, session=None,
            duration_s=None, label=None, data=None, manifest=None):
    if action not in ("check", "initialize", "release", "train", "quality", "calibrate") and not live:
        raise PermissionError("device access requires --live; check never connects")
    state = dict(action=action, status="STARTING", steps=[], motion_requested=action in ("empty", "round"))

    def run(name, module, arguments):
        result = step(directory, name, module, arguments, runner=runner)
        state["steps"].append(result)
        if result["returncode"]:
            raise RuntimeError(f"{name} failed ({result['returncode']}); see {result['log']}")

    try:
        if action == "check":
            state["checks"] = check(profile, directory)
            state["status"] = "CHECK_COMPLETE_WITH_BLOCKERS" if state["checks"]["round_blockers"] or state["checks"]["empty_load"].startswith("BLOCKED") else "OFFLINE_CHECK_COMPLETE"
        elif action == "initialize":
            from scripts.run_robot_leg_round import paths
            config_path = path(profile["round_config"])
            config = paths(config_path)
            request = {key: config[key] for key in ("algorithm_root", "algorithm_sha256", "algorithm_dependencies",
                "geometry_m", "candidates", "reference_candidate_id", "setup_id")}
            import hashlib
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            reference = next(p for p in config["candidates"] if p["candidate_id"] == config["reference_candidate_id"])
            request.update(history=[], setup_sha256=digest(config_path), reference_sha256=digest(reference["path"]))
            write(directory / "initialization_request.json", request)
            run("initialize", "scripts.train_robot_leg_round", [directory / "initialization_request.json",
                "--validate-only", "--output", directory / "algorithm_ready.json"])
            if load(directory / "algorithm_ready.json").get("status") != "ALGORITHM_INITIALIZED":
                raise RuntimeError("frozen algorithm initialization did not succeed")
        elif action == "diagnostics":
            for child_action in ("events", "capture", "observe"):
                child_dir = directory / child_action
                child_dir.mkdir(exist_ok=False)
                result = execute(child_action, profile, child_dir, live=True, confirm=confirm, runner=runner)
                state["steps"].append(dict(name=child_action,
                    returncode=0 if result["status"] == "COMPLETE" else 1,
                    log=str(child_dir / "session_report.md")))
            if any(s["returncode"] for s in state["steps"]):
                state.update(status="BLOCKED_OR_FAILED", reason="One or more diagnostic stages failed; independent stages were retained.")
            state["note"] = "Sequential stationary diagnostics, not simultaneous ADS/robot alignment validation. No RT preparation or motion."
        elif action == "events":
            run("events", "scripts.diagnose_collision_events", ["--robot-ip", profile["robot_ip"],
                "--local-ip", profile["local_ip"], "--duration-s", profile["diagnostic_duration_s"],
                "--max-age-s", profile["diagnostic_event_max_age_s"], "--output", directory / "events.json"])
        elif action == "capture":
            from scripts.lab_capture_summary import summarize_capture
            duration = positive(profile["diagnostic_duration_s"] if duration_s is None else duration_s, "capture duration")
            if duration > 3600:
                raise ValueError("one capture must not exceed 3600 seconds")
            if label is not None and (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label)):
                raise ValueError("capture label must contain only ASCII letters, digits, underscores or hyphens")
            run("leg", "scripts.capture_leg_segment", ["--config", path(profile["leg_config"]),
                "--output-dir", directory / "leg", "--duration-s", duration])
            state["capture_label"] = label
            state["quality_summary"] = summarize_capture(directory / "leg")
            state["note"] = "Recording completed; inspect quality_summary and raw quality.json before interpreting measurements. No subsequent motion is approved."
        elif action == "quality":
            if data is None:
                raise ValueError("--data must name a raw CSV or its parent directory")
            run("quality", "measurement_validation.offline_leg", ["quality", path(str(data)),
                "--output-dir", directory / "quality"])
            state["note"] = "Descriptive offline quality report. Host timing gaps are not exact device frame loss; torque changes during pose changes are not stationary drift."
        elif action == "calibrate":
            if data is None or manifest is None:
                raise ValueError("--data and --manifest are required for offline calibration analysis")
            run("calibrate", "measurement_validation.offline_leg", ["calibrate", path(str(manifest)),
                "--data-dir", path(str(data)), "--output", directory / "calibration.json"])
            state["note"] = "Offline fit/validation analysis only; no calibration, safety configuration or movement release was changed."
        elif action == "observe":
            # Unpowered idle observation. Does not prepare RT or reuse the powered commissioning profile.
            from scripts.acquire_robot_data import load_live_wrench_config
            wrench = load_live_wrench_config(path(profile["observation_wrench_config"]))
            if any(wrench["connection"][k] != profile[k] for k in ("robot_ip", "local_ip")):
                raise ValueError("observation and launcher IP mismatch")
            from hardware.safety_events import validate_collision_source
            validate_collision_source("events", profile["diagnostic_event_max_age_s"])
            run("robot", "scripts.acquire_robot_data", ["--ip", profile["robot_ip"], "--local-ip", profile["local_ip"],
                "--episode-dir", directory / "robot", "--duration-s", profile["diagnostic_duration_s"],
                "--mode", "live", "--wrench-config", path(profile["observation_wrench_config"]),
                "--wrench-hz", positive(profile["observation_wrench_hz"], "observation wrench rate"),
                "--collision-source", "events", "--collision-event-max-age-s", profile["diagnostic_event_max_age_s"]])
        elif action == "release":
            empty = profile["empty_load"]
            request_path = path(empty["review_request"])
            request = load(request_path)
            expected = request["required_confirmation"]
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError("review request has no explicit confirmation text")
            print("Review current fixture, anchor, trajectory, payload and files before signing.")
            print(expected)
            if confirm("Enter the exact reviewed confirmation above: ") != expected:
                raise PermissionError("review not confirmed; no release generated")
            args = ["--output", directory / "release.json", "--review-request", request_path,
                    "--confirmation", expected, "--valid-for-minutes", "30"]
            for key, flag in (("reference", "--reference"), ("anchor", "--anchor"),
                ("safety_config", "--safety-config"), ("frame_config", "--frame-config"), ("wrench_config", "--wrench-config")):
                args += [flag, path(empty[key])]
            run("release", "scripts.prepare_empty_load_commissioning_release", args)
            checked_profile = dict(profile, empty_load=dict(empty, release=str(directory / "release.json")))
            approved = empty_release(checked_profile)
            state["release"] = str(directory / "release.json")
            state["release_sha256"] = approved.approval_sha256
            state["expires_at"] = approved.payload["expires_at"]
        elif action == "empty":
            from control.start_anchor import load_start_anchor
            from control.execution_preflight import OPERATOR_CONFIRMATION
            release = empty_release(profile)
            empty = profile["empty_load"]
            shared = ["--reference", path(empty["reference"]), "--safety-config", path(empty["safety_config"]),
                "--frame-config", path(empty["frame_config"]), "--anchor", path(empty["anchor"]),
                "--wrench-config", path(empty["wrench_config"])]
            run("empty_precheck", "scripts.check_empty_load_commissioning_release",
                ["--release", path(empty["release"]), *shared])
            print("ACTUAL MOTION: one fixture-only empty-load cycle; no leg or human; no training.")
            print("Release:", release.approval_sha256, "expires:", release.payload["expires_at"])
            if confirm("Type I CONFIRM EMPTY LOAD ONE CYCLE: ") != "I CONFIRM EMPTY LOAD ONE CYCLE":
                raise PermissionError("operator cancelled; no motion started")
            anchor = load_start_anchor(path(empty["anchor"]))
            run("empty_execute", "scripts.run_rehab_experiment", ["--mode", "execute", "--ip", profile["robot_ip"],
                "--local-ip", profile["local_ip"], "--episode-dir", directory / "robot", "--anchor", path(empty["anchor"]),
                "--anchor-id", anchor.anchor_id, "--frame-config", path(empty["frame_config"]),
                "--safety-config", path(empty["safety_config"]), "--trajectory", "reference_measured_asymmetric_closed_slow",
                "--reference-path", path(empty["reference"]), "--enable-motion", "--prepare-realtime",
                "--collision-source", "query", "--wrench-config", path(empty["wrench_config"]),
                "--wrench-hz", release.payload["wrench_hz"], "--release-approval", path(empty["release"]),
                "--operator-confirmation", OPERATOR_CONFIRMATION])
        elif action == "round":
            from scripts.run_robot_leg_round import check_config, paths, CONFIRMATION
            config_path = path(profile["round_config"])
            blockers = check_config(config_path)
            config = paths(config_path)
            trial_kind = config["trial_kind"]
            if trial_kind not in ("dummy_leg_reference", "dummy_leg_capture_only"):
                blockers.append("round requires a reviewed loaded capture-only or training configuration")
            if blockers:
                state["blockers"] = blockers
                raise PermissionError("loaded round setup blocked; see session_status.json")
            round_dir = directory / "round"
            run("prepare", "scripts.run_robot_leg_round", ["prepare", "--config", config_path, "--session", round_dir])
            print("Review captured anchor and preview:", round_dir / "preview")
            if confirm("Type " + CONFIRMATION + ": ") != CONFIRMATION:
                raise PermissionError("operator cancelled; no motion started")
            run("round", "scripts.run_robot_leg_round", ["run", "--session", round_dir, "--confirmation", CONFIRMATION])
            completed = load(round_dir / "round_status.json")
            expected = ("CAPTURE_ONLY_COMPLETE", "SKIPPED_CAPTURE_ONLY") if trial_kind == "dummy_leg_capture_only" else ("ROUND_COMPLETE", "MODEL_UPDATED")
            if (completed.get("status"), completed.get("training_status")) != expected:
                raise RuntimeError("round did not confirm the expected completion and training status: " + str(expected))
            state.update(round_trial_kind=trial_kind, round_status=completed["status"],
                         training_status=completed["training_status"])
            state["round_report"] = str(round_dir / "report.md")
        elif action == "train":
            if session is None:
                raise ValueError("--session required for retraining a recorded real round")
            original = Path(session).resolve()
            recorded = load(original / "round_status.json")
            if recorded.get("status") != "ROUND_COMPLETE" or recorded.get("training_status") != "MODEL_UPDATED":
                raise PermissionError("offline replay requires an already validated complete training round")
            request = original / "training_request.json"
            run("train", "scripts.train_robot_leg_round", [request, "--output", directory / "model_update.json"])
            state["note"] = "Replay of existing measurements; not a new physical trial. No recommendation is dispatched."
        if state["status"] == "STARTING":
            state["status"] = "COMPLETE"
    except Exception as exc:
        state.update(status="BLOCKED_OR_FAILED", reason=f"{type(exc).__name__}:{exc}")
    except KeyboardInterrupt:
        state.update(status="INTERRUPTED", reason="Operator interrupted. Inspect child cleanup records and physical state.")
        raise
    finally:
        write(directory / "session_status.json", state)
        lines = ["# Lab session", "", "Action: " + action, "Status: " + state["status"], "",
                 state.get("reason", ""), "", "No automatic retry, reset, approval or next-candidate dispatch."]
        lines += [f"- {s['name']}: exit {s['returncode']}; {s['log']}" for s in state["steps"]]
        if state.get("checks"):
            lines += ["", "Empty-load: " + state["checks"]["empty_load"], "", "Loaded round blockers:"]
            lines += ["- " + item for item in state["checks"]["round_blockers"]]
            lines += ["", "Training blockers:"]
            lines += ["- " + item for item in state["checks"]["training_blockers"]]
        if state.get("blockers"):
            lines += ["", "Blockers:"] + ["- " + item for item in state["blockers"]]
        if state.get("round_report"):
            lines += ["", "Round report: " + state["round_report"]]
        if state.get("quality_summary"):
            lines += ["", "Recording completion does not mean dynamics quality passed.",
                      "Quality summary: " + json.dumps(state["quality_summary"], ensure_ascii=False)]
        if state.get("note"):
            lines += ["", state["note"]]
        (directory / "session_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--profile", type=Path, default=ROOT / "config/lab_next_visit.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/next_lab_visit")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--duration-s", type=float, help="capture recording length only; does not change safety timing limits")
    parser.add_argument("--label", help="ASCII capture label recorded in the session metadata")
    parser.add_argument("--data", type=Path, help="raw CSV or directory for offline quality/calibration analysis")
    parser.add_argument("--manifest", type=Path, help="independently measured angle manifest for offline calibration")
    parser.add_argument("--empty-release", type=Path, help="existing separately reviewed short-lived release; never generated by this launcher")
    args = parser.parse_args(argv)
    if args.label is not None and (args.action != "capture" or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label)):
        parser.error("--label requires capture and only ASCII letters, digits, underscores or hyphens")
    if args.duration_s is not None and args.action != "capture":
        parser.error("--duration-s changes only a capture recording length")
    profile = load(args.profile)
    if profile.get("schema_version") != 1:
        raise ValueError("unknown launcher profile schema")
    if args.empty_release:
        profile["empty_load"]["release"] = str(args.empty_release.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = args.action + ("_" + args.label if args.label else "")
    directory = args.output_root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + suffix)
    directory.mkdir(exist_ok=False)
    write(directory / "launcher_profile.json", profile)
    result = execute(args.action, profile, directory, live=args.live, session=args.session,
                     duration_s=args.duration_s, label=args.label, data=args.data, manifest=args.manifest)
    print(json.dumps(dict(directory=str(directory), status=result["status"], reason=result.get("reason")), ensure_ascii=False))
    return 0 if result["status"] in ("COMPLETE", "OFFLINE_CHECK_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
