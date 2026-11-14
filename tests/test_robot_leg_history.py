"""Scientific consistency and real frozen-trainer compatibility of offline replay."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from measurement_validation.robot_leg_history import IDENTITY_KEYS, prepare_history, sha256_file
from scripts.replay_robot_leg_history import replay
from experiment_algorithm.binding import current_binding
from measurement_validation.torque_endpoints import RAW_ENDPOINT_METHOD, raw_torque_endpoint_trace, torque_metrics_from_trace

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fixture_manifest(directory, *, real_trainer=False):
    """Synthetic aligned tables only; never evidence from the lab."""
    directory.mkdir(parents=True, exist_ok=True)
    if real_trainer:
        config = current_binding()
        algorithm_root = Path(config["algorithm_root"])
        dependencies = config["algorithm_dependencies"]
        sources = config["algorithm_sha256"]
    else:
        algorithm_root = directory / "algorithm"
        algorithm_root.mkdir()
        for name in ("config.py", "surrogate.py", "policy.py"):
            (algorithm_root / name).write_text("# unit-test-only source\n", encoding="utf-8")
        sources = {name: sha256_file(algorithm_root / name) for name in ("config.py", "surrogate.py", "policy.py")}
        dependencies = {"algorithm/config.py": sources["config.py"]}
    base = pd.read_csv(ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv")
    columns = ["time_s", "cycle_phase", "q_hip_rad", "q_knee_rad", "dq_hip_rad_s", "dq_knee_rad_s", "ddq_hip_rad_s2", "ddq_knee_rad_s2"]
    base = base[columns].copy()
    catalog = []
    for index, candidate_id in enumerate(("reference", "candidate_1", "candidate_2")):
        curve = base.copy()
        # Analytic perturbation for interface coverage, not an approved path.
        time = curve.time_s.to_numpy()
        omega = 2 * np.pi / time[-1]
        curve.q_hip_rad += index * .005 * (1 - np.cos(omega * time))
        curve.dq_hip_rad_s += index * .005 * omega * np.sin(omega * time)
        curve.ddq_hip_rad_s2 += index * .005 * omega**2 * np.cos(omega * time)
        path = directory / f"{candidate_id}.csv"
        curve.to_csv(path, index=False)
        catalog.append(dict(candidate_id=candidate_id, path=path.name, sha256=sha256_file(path),
                            beta_flex=index * .01, beta_extend=0., share_shift=0.))
    template = dict(algorithm_root=str(algorithm_root) if real_trainer else "algorithm",
                    algorithm_sha256=sources, algorithm_dependencies=dependencies,
                    geometry_m=[.42, .30], candidates=catalog,
                    reference_candidate_id="reference", reference_sha256=catalog[0]["sha256"],
                    setup_id="SYNTHETIC_INTERFACE_DEMO", setup_sha256="0" * 64, history=[])
    if real_trainer:
        template.update({key: config[key] for key in ("algorithm_version", "algorithm_source_lock_sha256")})
    template_path = directory / "template.json"
    write_json(template_path, template)
    rounds = []
    for index, item in enumerate(catalog[:2]):
        measured = pd.read_csv(directory / item["path"]).rename(columns={"cycle_phase": "phase"})
        measured["command_time_s"] = measured.time_s
        scale = 1. if index == 0 else .8
        measured["tau_hip_nm"] = scale * (5 + .1 * np.sin(measured.time_s))
        measured["tau_knee_nm"] = scale * (3 + .1 * np.cos(measured.time_s))
        path = directory / f"measured_{index}.csv"
        measured.to_csv(path, index=False)
        rounds.append(dict(episode_id=f"synthetic_{index}", candidate_id=item["candidate_id"],
                           measured_path=path.name, measured_sha256=sha256_file(path),
                           status="VALIDATED_COMPLETE", data_classification="SYNTHETIC_INTERFACE_DEMO",
                           **{key: template[key] for key in IDENTITY_KEYS}))
    manifest = dict(schema_version=1, data_classification="SYNTHETIC_INTERFACE_DEMO",
                    training_template=dict(path=template_path.name, sha256=sha256_file(template_path)), rounds=rounds)
    target = directory / "manifest.json"
    write_json(target, manifest)
    return target, manifest, template


def test_fixed_first_reference_and_resolved_paths(tmp_path, monkeypatch):
    path, _, _ = fixture_manifest(tmp_path / "input")
    monkeypatch.chdir(tmp_path)  # Resolution must not depend on process cwd.
    request, audit = prepare_history(path)
    first, second = request["history"]
    assert first["endpoints"] == dict(E3=1., E2=1., peak_ratio=1.)
    assert second["endpoints"] == pytest.approx(dict(E3=.8, E2=.8, peak_ratio=.8))
    assert audit["baseline_episode_id"] == first["episode_id"]
    assert audit["rounds"][1]["components_nm"] == pytest.approx(np.array(audit["baseline_metrics"]["components_nm"]) * .8)
    assert all(Path(x["measured_path"]).is_absolute() for x in request["history"])
    assert all(Path(x["path"]).is_absolute() for x in request["candidates"])


@pytest.mark.parametrize("kind", ["setup", "geometry", "reference", "duplicate_candidate", "duplicate_episode", "wrong_first", "incomplete_status", "classification", "unknown_candidate", "csv_hash", "template_hash", "algorithm_hash", "catalog_hash"])
def test_identity_completion_and_hash_failures(tmp_path, kind):
    path, manifest, _ = fixture_manifest(tmp_path / "input")
    round2 = manifest["rounds"][1]
    if kind == "setup": round2["setup_sha256"] = "1" * 64
    if kind == "geometry": round2["geometry_m"] = [.41, .3]
    if kind == "reference": round2["reference_sha256"] = "1" * 64
    if kind == "duplicate_candidate": round2["candidate_id"] = "reference"
    if kind == "duplicate_episode": round2["episode_id"] = "synthetic_0"
    if kind == "wrong_first": manifest["rounds"].reverse()
    if kind == "incomplete_status": round2["status"] = "ROUND_FAILED"
    if kind == "classification": round2["data_classification"] = "MEASURED_DATA"
    if kind == "unknown_candidate": round2["candidate_id"] = "unlisted"
    if kind == "csv_hash": round2["measured_sha256"] = "1" * 64
    if kind == "template_hash": manifest["training_template"]["sha256"] = "1" * 64
    if kind == "algorithm_hash": (path.parent / "algorithm/config.py").write_text("changed", encoding="utf-8")
    if kind == "catalog_hash": (path.parent / "candidate_1.csv").write_text("changed", encoding="utf-8")
    write_json(path, manifest)
    with pytest.raises(ValueError):
        prepare_history(path)


@pytest.mark.parametrize("kind", ["missing_state", "missing_torque", "nonfinite", "duplicate_time", "partial", "wrong_phase", "wrong_command_grid", "csv_identity"])
def test_incomplete_or_invalid_measured_tables(tmp_path, kind):
    path, manifest, _ = fixture_manifest(tmp_path / "input")
    record = manifest["rounds"][1]
    csv = path.parent / record["measured_path"]
    frame = pd.read_csv(csv)
    if kind == "missing_state": frame = frame.drop(columns="dq_knee_rad_s")
    if kind == "missing_torque": frame = frame.drop(columns="tau_knee_nm")
    if kind == "nonfinite": frame.loc[4, "tau_hip_nm"] = np.inf
    if kind == "duplicate_time": frame.loc[4, "time_s"] = frame.loc[3, "time_s"]
    if kind == "partial": frame = frame.iloc[:-1]
    if kind == "wrong_phase": frame.loc[4, "phase"] = "extension"
    if kind == "wrong_command_grid": frame.loc[4, "command_time_s"] += .01
    if kind == "csv_identity": frame["candidate_id"] = "reference"
    frame.to_csv(csv, index=False)
    record["measured_sha256"] = sha256_file(csv)
    write_json(path, manifest)
    with pytest.raises(ValueError):
        prepare_history(path)


def test_prepare_only_never_launches_training_or_overwrites(tmp_path, monkeypatch):
    manifest, _, _ = fixture_manifest(tmp_path / "input")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("prepare-only must not train"))
    output = tmp_path / "replay"
    result = replay(manifest, output)
    assert result["status"] == "HISTORY_PREPARED" and result["rounds"] == 2
    assert not (output / "model_update.json").exists()
    with pytest.raises(FileExistsError): replay(manifest, output)


def test_import_does_not_load_hardware_boundaries():
    code = "import sys; import scripts.replay_robot_leg_history; assert not any(n == 'hardware' or n.startswith('hardware.') or 'xCoreSDK' in n for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_rounds_use_existing_isolated_frozen_trainer(tmp_path):
    manifest, _, _ = fixture_manifest(tmp_path / "input", real_trainer=True)
    output = tmp_path / "replay"
    status = replay(manifest, output, train=True)
    update = json.loads((output / "model_update.json").read_text(encoding="utf-8"))
    assert status["status"] == "OFFLINE_MODEL_UPDATED" and status["rounds"] == 2
    assert update["method"] == "FROZEN_RESIDUAL_GP_MEAN_GREEDY"
    assert len(update["history"]) == 2 and len(update["theta"]) == 6
    assert update["history"][1]["endpoints"]["E3"] == pytest.approx(.8)
    assert update["data_classification"] == "SYNTHETIC_INTERFACE_DEMO"
    assert update["offline_history_replay"] and not update["motion_approved"]
    assert not update["policy_state_replayed"]
    assert update["completed_rounds_only"] and not update["total_attempt_budget_accounted"]
    assert update["action"] is None and "diagnostic_action" in update


@pytest.mark.parametrize("change_duration", [False, True])
def test_same_duration_allows_changed_flexion_extension_timing(tmp_path, change_duration):
    path, manifest, template = fixture_manifest(tmp_path / "input")
    candidate = template["candidates"][1]
    curve_path = path.parent / candidate["path"]
    curve = pd.read_csv(curve_path)
    old = curve.time_s.to_numpy()
    turn = old[curve.cycle_phase.eq("flexion")][-1]
    new_turn = turn * 1.05
    curve.time_s = np.where(old <= turn, old * new_turn / turn,
                           new_turn + (old-turn) * (old[-1]-new_turn) / (old[-1]-turn))
    if change_duration:
        curve.time_s *= 1.1
    curve.to_csv(curve_path, index=False)
    candidate["sha256"] = sha256_file(curve_path)
    measured_path = path.parent / manifest["rounds"][1]["measured_path"]
    measured = pd.read_csv(measured_path)
    measured["time_s"] = measured["command_time_s"] = curve.time_s
    measured.to_csv(measured_path, index=False)
    manifest["rounds"][1]["measured_sha256"] = sha256_file(measured_path)
    template_path = path.parent / "template.json"
    write_json(template_path, template)
    manifest["training_template"]["sha256"] = sha256_file(template_path)
    write_json(path, manifest)
    if change_duration:
        with pytest.raises(ValueError, match="duration"):
            prepare_history(path)
    else:
        request, _ = prepare_history(path)
        assert len(request["history"]) == 2


def measured_raw_manifest(directory):
    path, manifest, template = fixture_manifest(directory)
    manifest["data_classification"] = "MEASURED_DATA"
    for i, record in enumerate(manifest["rounds"]):
        record["data_classification"] = "MEASURED_DATA"
        csv_path = directory / record["measured_path"]
        frame = pd.read_csv(csv_path)
        frame["tau_hip_nm"] = frame["tau_knee_nm"] = 1.
        frame.to_csv(csv_path, index=False)
        record["measured_sha256"] = sha256_file(csv_path)
        raw_t = np.linspace(0, 24, 4801 + i * 1200)
        hip = np.ones(len(raw_t))
        if i:
            hip[np.argmin(abs(raw_t - .15))] = 11.
        turn = float(frame.loc[frame.phase.eq("flexion"), "time_s"].iloc[-1])
        trace = raw_torque_endpoint_trace(raw_t, hip, np.ones(len(raw_t)),
            motion_start_s=0., phase_boundary_s=turn, motion_end_s=24.)
        metrics = dict(**torque_metrics_from_trace(trace), endpoint_trace=trace,
            measured_sha256=record["measured_sha256"],
            reference_sha256=template["candidates"][i]["sha256"], source_sha256="1"*64,
            dispatch_sha256="2"*64, analysis_config_sha256="3"*64, calibration_config_sha256="4"*64,
            endpoint_sampling=dict(method=RAW_ENDPOINT_METHOD, endpoint_samples=len(trace["time_s"])))
        metrics_path = directory / f"raw_metrics_{i}.json"
        write_json(metrics_path, metrics)
        record["endpoint_source"] = dict(path=metrics_path.name, sha256=sha256_file(metrics_path))
    write_json(path, manifest)
    return path, manifest


def test_measured_history_uses_high_rate_peak_and_allows_different_sample_counts(tmp_path):
    path, _ = measured_raw_manifest(tmp_path / "input")
    request, audit = prepare_history(path)
    assert request["history"][1]["endpoints"]["peak_ratio"] == 11.
    assert audit["rounds"][1]["peaks_nm"] == [11., 1.]
    assert audit["endpoint_sampling"][0] == RAW_ENDPOINT_METHOD


@pytest.mark.parametrize("kind", ["missing", "hash", "declared_metrics", "candidate_binding", "measured_round_binding", "analysis_definition", "calibration_definition"])
def test_measured_history_rejects_missing_or_tampered_raw_endpoints(tmp_path, kind):
    path, manifest = measured_raw_manifest(tmp_path / "input")
    record = manifest["rounds"][1]
    descriptor = record["endpoint_source"]
    if kind == "missing":
        del record["endpoint_source"]
    elif kind == "hash":
        descriptor["sha256"] = "0"*64
    else:
        target = path.parent / descriptor["path"]
        metrics = json.loads(target.read_text(encoding="utf-8"))
        if kind == "declared_metrics": metrics["peaks_nm"][0] = 1.
        if kind == "candidate_binding": metrics["reference_sha256"] = "0"*64
        if kind == "measured_round_binding": metrics["measured_sha256"] = "0"*64
        if kind == "analysis_definition": metrics["analysis_config_sha256"] = "5"*64
        if kind == "calibration_definition": metrics["calibration_config_sha256"] = "5"*64
        write_json(target, metrics)
        descriptor["sha256"] = sha256_file(target)
    write_json(path, manifest)
    with pytest.raises(ValueError): prepare_history(path)


def test_strict_manifest_retains_failed_attempt_in_budget_ledger(tmp_path):
    path, manifest, _ = fixture_manifest(tmp_path / "input")
    manifest.update(schema_version=2, replay_mode="strict_policy", attempt_ledger_complete=True)
    attempts = manifest.pop("rounds")
    for i, record in enumerate(attempts, 1):
        record.update(attempt_index=i, role="REFERENCE" if i == 1 else "INFORMATION_PROBE", outcome="COMPLETED")
    failed = attempts[1]
    failed.update(outcome="FAILED", failure_reason="incomplete acquisition")
    for key in ("measured_path", "measured_sha256"):
        failed.pop(key)
    manifest["attempts"] = attempts
    write_json(path, manifest)
    request, audit = prepare_history(path)
    assert len(request["attempts"]) == 2 and len(request["history"]) == 1
    assert request["attempts"][1]["outcome"] == "FAILED"
    assert audit["total_attempt_budget_accounted"] and not audit["completed_rounds_only"]


def test_current_algorithm_binding_stays_inside_checkout_and_detects_content_edits(tmp_path, monkeypatch):
    import experiment_algorithm.binding as bindings
    binding = current_binding()
    assert all(Path(path).is_relative_to(ROOT) for path in binding["algorithm_dependencies"])
    lock = json.loads(bindings.LOCK.read_text(encoding="utf-8"))
    lock["source_sha256"] = {"changed.py": "0"*64}
    fake_lock = tmp_path / "lock.json"
    write_json(fake_lock, lock)
    (tmp_path / "changed.py").write_text("actual content", encoding="utf-8")
    monkeypatch.setattr(bindings, "ROOT", tmp_path)
    monkeypatch.setattr(bindings, "LOCK", fake_lock)
    with pytest.raises(ValueError, match="source lock mismatch"):
        bindings.current_binding()


@pytest.mark.parametrize("failed", [False, True])
def test_strict_isolated_trainer_reconstructs_reference_prefix_and_failure_budget(tmp_path, failed):
    path, manifest, _ = fixture_manifest(tmp_path / "input", real_trainer=True)
    attempt = manifest.pop("rounds")[0]
    attempt.update(attempt_index=1, role="REFERENCE", outcome="FAILED" if failed else "COMPLETED")
    if failed:
        attempt.update(failure_reason="synthetic acquisition failure")
        attempt.pop("measured_path")
        attempt.pop("measured_sha256")
    manifest.update(schema_version=2, replay_mode="strict_policy", attempt_ledger_complete=True, attempts=[attempt])
    write_json(path, manifest)
    result = replay(path, tmp_path / "strict", train=True)
    update = json.loads((tmp_path / "strict/model_update.json").read_text(encoding="utf-8"))
    assert result["status"] == "OFFLINE_POLICY_REPLAYED"
    assert update["policy_state_replayed"] and update["total_attempt_budget_accounted"]
    assert update["attempted_trials"] == 1 and update["remaining_trials"] == 7
    assert update["completed_trials"] == int(not failed) and update["failed_trials"] == int(failed)
    assert update["model_fitted"] == (not failed)
    assert update["action"] is not None and update["diagnostic_action"] is None
    if failed:
        assert update["action"]["stop_reason"] == "ATTEMPT_FAILED"


@pytest.mark.parametrize("kind", ["failed_status", "missing_classification", "different_classification", "mismatched_history"])
def test_direct_strict_trainer_rejects_bypassed_manifest_contract(tmp_path, kind):
    path, manifest, _ = fixture_manifest(tmp_path / "input", real_trainer=True)
    record = manifest.pop("rounds")[0]
    record.update(attempt_index=1, role="REFERENCE", outcome="COMPLETED")
    manifest.update(schema_version=2, replay_mode="strict_policy", attempt_ledger_complete=True, attempts=[record])
    write_json(path, manifest)
    request, _ = prepare_history(path)
    if kind == "failed_status": request["attempts"][0]["status"] = "ROUND_FAILED"
    if kind == "missing_classification": request.pop("data_classification")
    if kind == "different_classification": request["attempts"][0]["data_classification"] = "MEASURED_DATA"
    if kind == "mismatched_history": request["history"][0]["episode_id"] = "other_episode"
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round", str(request_path),
                             "--output", str(tmp_path / "update.json")], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode != 0 and not (tmp_path / "update.json").exists()
    assert any(message in result.stderr for message in ("validated complete measurement", "data_classification", "history does not match"))
