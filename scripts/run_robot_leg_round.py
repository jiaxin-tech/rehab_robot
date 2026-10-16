"""Prepare, explicitly confirm, and run one real robot/dummy-leg round."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from collection.twincat_leg import LegCaptureProcess, atomic_json, positive, validate_config
from control.execution_preflight import OPERATOR_CONFIRMATION
from control.start_anchor import load_start_anchor
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from measurement_validation.offline_leg import inspect_file
from measurement_validation.robot_leg_round import aligned_measurement, endpoints, validate_analysis
from safety.robot_trial_release import (APPROVAL_CONFIRMATION, EVIDENCE_KINDS, SCOPE,
    build_robot_trial_release_approval, load_and_validate_robot_trial_release, read_json, sha256_file,
    validate_robot_trial_release_payload, write_robot_trial_release)
from scripts.acquire_robot_data import _configured_adapter_factory, load_live_wrench_config
from scripts.capture_start_anchor import run_capture
from scripts.preview_rehab_trajectory import preview_trajectory
from scripts.run_rehab_experiment import run_execute


LOADED_SCOPE = "ROBOT_DUMMY_LEG_REFERENCE_TRIAL"
CONFIRMATION = "I CONFIRM REVIEWED ROBOT LEG ROUND"
ALGORITHM_FILES = ("config.py", "surrogate.py", "policy.py")


def resolve(base, value):
    return (base / value).resolve()


def paths(config_path):
    p = Path(config_path).resolve()
    c = read_json(p)
    base = p.parent
    for key in ("leg_config", "calibration", "analysis", "safety_config", "frame_config", "wrench_config"):
        c[key] = str(resolve(base, c[key]))
    c["algorithm_root"] = str(resolve(base, c["algorithm_root"]))
    for item in c["candidates"]:
        item["path"] = str(resolve(base, item["path"]))
    for record in c["site_evidence"].values():
        record["path"] = str(resolve(base, record["path"]))
    return c


def scope_for(c):
    if c["trial_kind"] not in ("empty_load", "dummy_leg_reference"):
        raise ValueError("unknown trial_kind")
    return SCOPE if c["trial_kind"] == "empty_load" else LOADED_SCOPE


def check_config(config_path):
    problems = []
    try:
        c = paths(config_path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return ["round config: " + str(exc)]
    def check(name, action):
        try:
            action()
        except (ValueError, OSError, KeyError, TypeError, PermissionError) as exc:
            problems.append(name + ": " + str(exc))
    check("scope", lambda: scope_for(c))
    from hardware.safety_events import validate_collision_source
    check("collision events", lambda: validate_collision_source(c.get("collision_source"), c.get("collision_event_max_age_s")))
    def expiry():
        if not isinstance(c.get("expires_at"), str):
            raise ValueError("future timezone-aware release expiry required")
        value = datetime.fromisoformat(c["expires_at"].replace("Z", "+00:00"))
        if value.tzinfo is None or value <= datetime.now(timezone.utc):
            raise ValueError("future timezone-aware release expiry required")
    check("release expiry", expiry)
    if c.get("schema_version") != 1 or c.get("reviewed") is not True:
        problems.append("round setup must be explicitly reviewed")
    for key in ("setup_id", "reviewer", "operator", "robot_ip", "local_ip", "robot_class", "tool_name", "workpiece_name"):
        if not isinstance(c.get(key), str) or not c[key].strip():
            problems.append("missing " + key)
    # Enumerate missing physical facts even when the overall review is pending.
    def missing_facts():
        capture = read_json(c["leg_config"])
        for key, value in capture.items():
            if value is None:
                problems.append("ADS capture missing: " + key)
        for axis in ("axis1", "axis4"):
            for key, value in capture["guards"][axis].items():
                if value is None:
                    problems.append("ADS capture missing: " + axis + "." + key)
        if c["trial_kind"] == "dummy_leg_reference":
            for joint, values in read_json(c["calibration"])["joints"].items():
                if values is None:
                    problems.append("joint calibration missing: " + joint)
                else:
                    for key, value in values.items():
                        if value is None:
                            problems.append("joint calibration missing: " + joint + "." + key)
            for key, value in read_json(c["analysis"]).items():
                if value is None:
                    problems.append("analysis missing: " + key)
    check("missing physical facts", missing_facts)
    for key in ("pre_roll_s", "post_roll_s", "training_timeout_s", "wrench_hz", "max_leg_start_age_s"):
        check(key, lambda key=key: positive(c.get(key), key))
    check("ADS capture mapping", lambda: validate_config(read_json(c["leg_config"])))
    def dll():
        capture = read_json(c["leg_config"])
        if os.name != "nt" or not capture.get("ads_dll_path") or not Path(capture["ads_dll_path"]).is_file():
            raise ValueError("Windows and a matching-bitness real TcAdsDll.dll installation are required")
    check("native ADS runtime", dll)
    check("live wrench", lambda: load_live_wrench_config(c["wrench_config"]))
    def connection():
        wrench = load_live_wrench_config(c["wrench_config"])
        if any(wrench["connection"][key] != c.get(key) for key in ("robot_ip", "local_ip", "robot_class")):
            raise ValueError("round and wrench connection/class must match")
    check("consistent robot connection", connection)
    from safety.experiment_safety import load_experiment_safety_config
    from control.start_anchored_relative_trajectory import load_rehab_frame_config
    check("robot safety", lambda: load_experiment_safety_config(c["safety_config"]).require_execute_allowed())
    def frame():
        if not load_rehab_frame_config(c["frame_config"]).reviewed:
            raise ValueError("frame not reviewed")
    check("rehab frame", frame)
    if c["trial_kind"] == "dummy_leg_reference":
        check("joint angles/timing", lambda: validate_analysis(read_json(c["analysis"]), read_json(c["calibration"])))
        def planned_joint_ranges():
            conversions = validate_analysis(read_json(c["analysis"]), read_json(c["calibration"]))
            reference = pd.read_csv(DEFAULT_REFERENCE_PATH)
            for name, conv in conversions.items():
                q = np.rad2deg(reference[f"q_{name}_rad"].to_numpy(float))
                if not np.isfinite(q).all() or np.any(q < conv.safe_min_deg) or np.any(q > conv.safe_max_deg):
                    raise ValueError(name + " reference exceeds reviewed leg angle range")
        check("planned leg angles", planned_joint_ranges)
    expected_evidence = set(EVIDENCE_KINDS)
    if c["trial_kind"] == "dummy_leg_reference":
        expected_evidence |= {"dummy_leg_installation", "joint_angle_calibration", "cross_device_timing"}
    if set(c["site_evidence"]) != expected_evidence:
        problems.append("site evidence categories differ from required scope: " + ",".join(sorted(expected_evidence)))
    for name, record in c["site_evidence"].items():
        def evidence(record=record):
            if record.get("passed") is not True or sha256_file(record["path"]) != record["sha256"]:
                raise ValueError("missing, unreviewed or changed evidence")
        check("site evidence " + name, evidence)
    def algorithm():
        root = Path(c["algorithm_root"])
        if set(c["algorithm_sha256"]) != set(ALGORITHM_FILES):
            raise ValueError("freeze config.py, surrogate.py and policy.py hashes")
        for name, digest in c["algorithm_sha256"].items():
            if sha256_file(root / name) != digest:
                raise ValueError("frozen algorithm changed: " + name)
        # The public mechanical prior and its dependencies are explicit bound inputs.
        if not c.get("algorithm_dependencies"):
            raise ValueError("public-prior source/XML dependency hashes required")
        for name, digest in c["algorithm_dependencies"].items():
            if sha256_file(name) != digest:
                raise ValueError("algorithm dependency changed: " + name)
    check("algorithm sources", algorithm)
    def catalog():
        ids = [p["candidate_id"] for p in c["candidates"]]
        if len(set(ids)) != len(ids) or c["reference_candidate_id"] not in ids:
            raise ValueError("unique candidate IDs and reference required")
        reference = next(p for p in c["candidates"] if p["candidate_id"] == c["reference_candidate_id"])
        if Path(reference["path"]) != DEFAULT_REFERENCE_PATH.resolve() or reference["sha256"] != sha256_file(DEFAULT_REFERENCE_PATH):
            raise ValueError("first round must use pinned slow reference")
        for point in c["candidates"]:
            if sha256_file(point["path"]) != point["sha256"]:
                raise ValueError("candidate changed")
    check("candidate catalog", catalog)
    return problems


def report(directory, state):
    atomic_json(directory / "round_status.json", state)
    scores = state.get("endpoints")
    lines = ["# 本轮机器人与假腿实验报告", "", f"- 状态：{state['status']}",
        f"- 实验类型：{state.get('trial_kind', '未确定')}", f"- 训练状态：{state.get('training_status', 'NOT_STARTED')}",
        f"- 原因：{state.get('reason') or '无'}", "",
        "训练只使用完整、通过检查的真实采集记录。空载联调不进入训练。", ""]
    if scores:
        lines += [f"E3：{scores['E3']:.6g}；E2：{scores['E2']:.6g}；峰值比：{scores['peak_ratio']:.6g}。", "",
            "本轮为参考初始化，自身归一化得到 1，不代表优化有效。", ""]
    if state.get("training_result"):
        result = state["training_result"]
        lines += ["主方法：残差 GP 均值 Greedy。", "", "下一步决策：" + json.dumps(result["action"], ensure_ascii=False), "",
            "模型更新不等于模型可信，也不构成下一条轨迹的运动许可。", ""]
        lines += ["拟合参数（用于本轮机械模型，不等同于已验证的人体生理参数）：", "",
                  "| 参数 | 数值 |", "| --- | ---: |"]
        for name, value in result["diagnostics"].get("parameters", {}).items():
            lines.append(f"| {name} | {value:.6g} |")
        lines += ["", "拟合诊断：" + json.dumps(result["diagnostics"], ensure_ascii=False), ""]
    if state.get("quality_summary"):
        lines += ["采集质量摘要：" + json.dumps(state["quality_summary"], ensure_ascii=False), "",
            "运动过程中的首尾力矩变化同时包含负载变化，不能直接认定为传感器漂移。", ""]
    if state.get("blockers"):
        lines += ["待补齐：", ""] + ["- " + x for x in state["blockers"]]
    lines += ["", "数据文件：leg/leg.csv、robot/ 内的机器人记录、dispatch.csv。", "",
              "详细来源哈希、对齐方法和拟合诊断保存在本轮 JSON 中。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def prepare(config_path, directory, *, capture_fn=run_capture):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    blockers = check_config(config_path)
    state = dict(status="CHECKING", training_status="NOT_STARTED",
                 config_path=str(Path(config_path).resolve()), config_sha256=sha256_file(config_path))
    if blockers:
        state.update(status="BLOCKED_BEFORE_CONNECTION", blockers=blockers)
        report(directory, state)
        return state
    c = paths(config_path)
    state["trial_kind"] = c["trial_kind"]
    try:
        factory = _configured_adapter_factory(local_ip=c["local_ip"], robot_class=c["robot_class"],
            state_interval_ms=None, collision_source=c["collision_source"],
            collision_event_max_age_s=c["collision_event_max_age_s"])
        capture_fn(robot_ip=c["robot_ip"], output_path=directory / "anchor.json",
            tool_name=c["tool_name"], workpiece_name=c["workpiece_name"], adapter_factory=factory)
        preview_trajectory(anchor=directory / "anchor.json", frame_config=c["frame_config"],
                           reference=DEFAULT_REFERENCE_PATH, output_dir=directory / "preview")
        state.update(status="AWAITING_ONSITE_CONFIRMATION", anchor_sha256=sha256_file(directory / "anchor.json"))
        report(directory, state)
        return state
    except BaseException as exc:
        state.update(status="PREPARATION_FAILED", reason=f"{type(exc).__name__}: {exc}")
        report(directory, state)
        raise


def approve_prepared(c, config_path, directory, state):
    """Called only after the CLI's explicit on-site confirmation."""
    anchor = load_start_anchor(directory / "anchor.json")
    if sha256_file(directory / "anchor.json") != state["anchor_sha256"]:
        raise PermissionError("prepared anchor changed before confirmation")
    anchor = replace(anchor, reviewed=True, notes=anchor.notes + " | explicitly confirmed for this round")
    anchor.save_json(directory / "reviewed_anchor.json")
    trial_setup = None
    if scope_for(c) == LOADED_SCOPE:
        inputs = {name: dict(path=str(Path(path).resolve()), sha256=sha256_file(path)) for name, path in
            dict(leg_acquisition=c["leg_config"], calibration=c["calibration"], analysis=c["analysis"],
                 round_config=str(Path(config_path).resolve())).items()}
        trial_setup = directory / "trial_setup.json"
        atomic_json(trial_setup, dict(scope=LOADED_SCOPE, reviewed=True, setup_id=c["setup_id"], inputs=inputs))
    draft = build_robot_trial_release_approval(reviewer=c["reviewer"], operator=c["operator"],
        expires_at=c["expires_at"], robot_identity=c["robot_identity"], reference_path=DEFAULT_REFERENCE_PATH,
        safety_config_path=c["safety_config"], frame_config_path=c["frame_config"],
        anchor_path=directory / "reviewed_anchor.json", wrench_config_path=c["wrench_config"],
        collision_source=c["collision_source"], collision_event_max_age_s=c["collision_event_max_age_s"],
        wrench_hz=c["wrench_hz"], scope=scope_for(c), trial_setup_path=trial_setup)
    write_robot_trial_release(directory / "release_draft.json", draft)
    site = directory / "site_review.json"
    atomic_json(site, dict(scope=draft["scope"], decision="approved", reviewer=c["reviewer"],
        reviewed_at=datetime.now(timezone.utc).isoformat(), bundle_sha256=draft["bundle_sha256"], records=c["site_evidence"],
        confirmation=CONFIRMATION))
    approved = dict(draft, approved=True, site_review=dict(path=str(site), sha256=sha256_file(site)))
    release_path = directory / "release_approval.json"
    inputs = dict(reference_path=DEFAULT_REFERENCE_PATH, safety_config_path=c["safety_config"],
        frame_config_path=c["frame_config"], anchor_path=directory / "reviewed_anchor.json", wrench_config_path=c["wrench_config"])
    validate_robot_trial_release_payload(approved, artifact_path=release_path, **inputs)
    write_robot_trial_release(release_path, approved)
    return load_and_validate_robot_trial_release(release_path, **inputs)


def run_algorithm_worker(directory, request_name, output_name, timeout, *, validate_only=False):
    command = [sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round",
               str(directory / request_name), "--output", str(directory / output_name)]
    if validate_only:
        command.append("--validate-only")
    with (directory / ("algorithm_initialization.log" if validate_only else "training.log")).open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return read_json(directory / output_name)


def run(directory, *, confirmation, capture_factory=LegCaptureProcess, execute_fn=run_execute):
    if confirmation != CONFIRMATION:
        raise PermissionError("requires exact on-site confirmation: " + CONFIRMATION)
    directory = Path(directory).resolve()
    # An exclusive persistent attempt marker also prevents replay after a crash.
    with (directory / "run_attempt.lock").open("x", encoding="ascii") as lock:
        lock.write(str(os.getpid()))
    return _run_once(directory, capture_factory=capture_factory, execute_fn=execute_fn)


def _run_once(directory, *, capture_factory, execute_fn):
    state = read_json(directory / "round_status.json")
    if state["status"] != "AWAITING_ONSITE_CONFIRMATION":
        raise PermissionError("round is not prepared, or already attempted; motion cannot be replayed")
    config_path = Path(state["config_path"])
    if sha256_file(config_path) != state["config_sha256"]:
        raise PermissionError("round configuration changed after preparation")
    c = paths(config_path)
    blockers = check_config(config_path)
    if blockers:
        state.update(status="BLOCKED_BEFORE_CONNECTION", blockers=blockers)
        report(directory, state)
        return state
    capture = None
    dispatches = []
    state.update(status="STARTING", training_status="NOT_STARTED")
    report(directory, state)
    try:
        release = approve_prepared(c, config_path, directory, state)
        request = None
        if c["trial_kind"] == "dummy_leg_reference":
            state.update(status="INITIALIZING_ALGORITHM")
            report(directory, state)
            request = dict(algorithm_root=c["algorithm_root"], algorithm_sha256=c["algorithm_sha256"],
                algorithm_dependencies=c["algorithm_dependencies"], geometry_m=c["geometry_m"], candidates=c["candidates"],
                setup_id=c["setup_id"], setup_sha256=sha256_file(directory / "trial_setup.json"),
                reference_candidate_id=c["reference_candidate_id"], reference_sha256=sha256_file(DEFAULT_REFERENCE_PATH), history=[])
            atomic_json(directory / "algorithm_initialization_request.json", request)
            ready = run_algorithm_worker(directory, "algorithm_initialization_request.json", "algorithm_ready.json",
                                         c["training_timeout_s"], validate_only=True)
            if ready.get("status") != "ALGORITHM_INITIALIZED":
                raise RuntimeError("algorithm failed to initialize before motion")
            release.require_current(check_files=True)
        capture = capture_factory(c["leg_config"], directory / "leg")
        capture.start()
        calibration = read_json(c["calibration"]) if c["trial_kind"] == "dummy_leg_reference" else None
        analysis = read_json(c["analysis"]) if calibration else None
        conversions = validate_analysis(analysis, calibration) if calibration else None
        reference = pd.read_csv(DEFAULT_REFERENCE_PATH)
        def guard(*, initial=False):
            reason = capture.health_reason()
            if reason:
                return reason
            if conversions:
                snapshot = capture.snapshot()
                age = (time.perf_counter_ns()-snapshot["last_sample_ns"])/1e9
                if age < 0 or age > c["max_leg_start_age_s"]:
                    return "leg_angle_snapshot_stale"
                for name, axis in (("hip", 1), ("knee", 4)):
                    angle = float(conversions[name].to_degrees(snapshot["latest"][f"axis{axis}_position_raw"]))
                    limits = conversions[name]
                    if not limits.safe_min_deg <= angle <= limits.safe_max_deg:
                        return name + "_outside_reviewed_angle_range"
                    if initial and abs(angle-np.rad2deg(reference[f"q_{name}_rad"].iloc[0])) > analysis["max_start_angle_error_deg"]:
                        return name + "_start_angle_mismatch"
            return None
        deadline = time.perf_counter()+c["pre_roll_s"]
        while time.perf_counter() < deadline:
            reason = guard(initial=True)
            if reason:
                raise RuntimeError(reason)
            time.sleep(.01)
        state.update(status="RECORDING_AND_MOVING")
        report(directory, state)
        def dispatched(t, start_ns, end_ns):
            dispatches.append(dict(trajectory_time_s=t, dispatch_start_ns=start_ns, dispatch_end_ns=end_ns))
        result = execute_fn(robot_ip=c["robot_ip"], local_ip=c["local_ip"], episode_dir=directory / "robot",
            anchor_path=directory / "reviewed_anchor.json", requested_anchor_id=load_start_anchor(directory / "reviewed_anchor.json").anchor_id,
            frame_config_path=c["frame_config"], safety_config_path=c["safety_config"],
            trajectory_id="reference_measured_asymmetric_closed_slow", enable_motion=True,
            operator_confirmation=OPERATOR_CONFIRMATION, live_wrench_config=load_live_wrench_config(c["wrench_config"]),
            adapter_factory=_configured_adapter_factory(local_ip=c["local_ip"], robot_class=c["robot_class"],
                state_interval_ms=None, collision_source=c["collision_source"],
                collision_event_max_age_s=c["collision_event_max_age_s"]),
            wrench_hz=c["wrench_hz"], collision_source=c["collision_source"],
            collision_event_max_age_s=c["collision_event_max_age_s"], release_approval_path=directory / "release_approval.json",
            wrench_config_path=c["wrench_config"], execution_guard=lambda: guard(initial=not dispatches), dispatch_observer=dispatched)
        if result.get("completed") is not True or not dispatches:
            raise RuntimeError("robot did not confirm complete reference execution")
        deadline = time.perf_counter()+c["post_roll_s"]
        while time.perf_counter() < deadline:
            reason = guard()
            if reason:
                raise RuntimeError(reason)
            time.sleep(.01)
        capture.stop()
        capture = None
        pd.DataFrame(dispatches).to_csv(directory / "dispatch.csv", index=False)
        quality = inspect_file(directory / "leg/leg.csv")
        atomic_json(directory / "quality.json", quality)
        state["quality_summary"] = dict(long_intervals=len(quality["long_intervals"]),
            duplicate_timestamps=len(quality["receive_time_duplicates_csv_lines"]),
            missing_values_by_column=quality["missing_values_by_column"])
        state["execution_result"] = result
        if c["trial_kind"] == "empty_load":
            state.update(status="EMPTY_LOAD_COMMISSIONING_COMPLETE", training_status="EXCLUDED_EMPTY_LOAD")
        else:
            release.require_current(check_files=True)
            measured, metrics = aligned_measurement(directory / "leg/leg.csv", directory / "dispatch.csv",
                DEFAULT_REFERENCE_PATH, read_json(c["analysis"]), calibration)
            measured.to_csv(directory / "measured.csv", index=False)
            atomic_json(directory / "measured_metrics.json", metrics)
            scores = endpoints(metrics, metrics)
            state.update(status="TRAINING", endpoints=scores)
            report(directory, state)
            request["history"] = [dict(episode_id=directory.name, candidate_id=c["reference_candidate_id"],
                measured_path=str(directory / "measured.csv"), measured_sha256=sha256_file(directory / "measured.csv"), endpoints=scores)]
            atomic_json(directory / "training_request.json", request)
            trained = run_algorithm_worker(directory, "training_request.json", "model_update.json", c["training_timeout_s"])
            state.update(status="ROUND_COMPLETE", training_status="MODEL_UPDATED",
                         training_result=trained)
        report(directory, state)
        return state
    except BaseException as exc:
        state.update(status="ROUND_FAILED", training_status="NOT_UPDATED", reason=f"{type(exc).__name__}: {exc}")
        if dispatches:
            pd.DataFrame(dispatches).to_csv(directory / "dispatch.csv", index=False)
        if capture:
            try:
                capture.stop()
            except BaseException as cleanup:
                state["capture_cleanup_error"] = str(cleanup)
        report(directory, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--output", type=Path)
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--session", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--session", type=Path, required=True)
    execute.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        blockers = check_config(args.config)
        result = dict(offline_config_valid=not blockers, hardware_verified=False, blockers=blockers)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(bool(blockers))
    result = prepare(args.config, args.session) if args.command == "prepare" else run(args.session, confirmation=args.confirmation)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] in ("BLOCKED_BEFORE_CONNECTION", "ROUND_FAILED", "PREPARATION_FAILED"))


if __name__ == "__main__":
    raise SystemExit(main())
