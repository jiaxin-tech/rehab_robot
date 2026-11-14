"""Validate adapters to the frozen original algorithm; never connect to hardware."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection.twincat_leg import atomic_json
from config.bundle import configuration
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import load_rehab_frame_config, build_start_anchored_relative_trajectory
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from lower_limb_sim.kinematics import forward_kinematics
from measurement_validation.robot_observation import SOURCE, robot_measurement
from measurement_validation.robot_execution import audit_robot_execution, read_json
from scripts import lab_session as lab
from scripts import original_training_adapter as adapter


@pytest.fixture
def episode(tmp_path):
    c = configuration()
    c["training"]["mode"] = "original_offline"
    c["trajectory"]["mode"] = "original_candidate"
    c["trajectory"]["target_peak_tcp_z_base_m"] = None
    c["empty_load"].update(max_tcp_speed_m_s=.3, max_tcp_acceleration_m_s2=.5)
    c["analysis"].update(sensor_receive_delay_s=0., command_application_delay_s=0.,
        max_gap_s=.03, max_read_duration_s=.01, max_dispatch_time_error_s=.02,
        derivative_rate_hz=100., derivative_window_s=.05,
        max_start_angle_error_deg=2., max_tracking_error_deg=2.)
    config_path = tmp_path / "experiment.json"
    atomic_json(config_path, c)
    anchor = load_start_anchor(Path(__file__).resolve().parents[1] / "results/empty_load_118/anchor.json")
    base = pd.read_csv(DEFAULT_REFERENCE_PATH)
    anchor = replace(anchor, tcp_pose_base=(.5, 0., .5) + anchor.tcp_pose_base[3:],
        tcp_position_base_m=(.5, 0., .5), trajectory_id=str(base.trajectory_id.iloc[0]))
    frame = load_rehab_frame_config(config_path)
    _, _, x0, z0 = forward_kinematics(base.q_hip_rad.iloc[0], base.q_knee_rad.iloc[0], .42, .30)
    # Explicit synthetic calibration only; no claims about laboratory values.
    hip = np.asarray(anchor.tcp_pose_base[:3])-frame.rotation_base_from_rehab @ np.array([x0, 0., z0])
    c["robot_observation"].update(hip_center_in_base_m=hip.tolist(),
        tcp_to_pull_point_in_tcp_m=[0., 0., 0.], wrench_origin_in_tcp_m=[0., 0., 0.],
        L1_m=.42, L2_m=.30, wrench_frame_type="world", rotation_base_from_wrench=np.eye(3).tolist(),
        force_robot_on_leg_sign=1, unloaded_wrench_bias=[0.]*6, compensation_verified=True,
        state_time_offset_s=0., wrench_time_offset_s=0., max_out_of_plane_position_m=.001,
        max_out_of_plane_force_n=.1, max_connection_moment_nm=.01)
    catalog = []
    for index, cid in enumerate(("reference", "candidate")):
        path = tmp_path / (cid + ".csv")
        base.to_csv(path, index=False)
        catalog.append({"candidate_id": cid, "path": str(path), "beta_flex": index*.04,
            "beta_extend": 0., "share_shift": 0.})
    c["training"].update(candidates=catalog, reference_candidate_id="reference")
    atomic_json(config_path, c)

    def create(name="one", *, cid="reference", parent=None, torque_scale=1.):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "robot").mkdir()
        (directory / "leg").mkdir()
        anchor.save_json(directory / "anchor.json")
        atomic_json(directory / "prepared_config.json", c)
        point = next(p for p in catalog if p["candidate_id"] == cid)
        frozen = directory / "original_candidate.csv"
        frozen.write_bytes(Path(point["path"]).read_bytes())
        table, _, meta = build_start_anchored_relative_trajectory(frozen,
            current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=load_rehab_frame_config(config_path))
        meta.update(anchor=anchor.to_dict(), mode="original_candidate", candidate_id=cid,
            reference_path=str(frozen), reference_sha256=adapter.file_hash(frozen))
        atomic_json(directory / "tcp_plan.json", meta)
        safety = dict(c["motion"])
        for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
            safety[key] = min(safety[key], c["empty_load"][key])
        atomic_json(directory / "tcp_safety.json", safety)
        table.to_csv(directory / "tcp_trajectory.csv", index=False)
        dispatch = pd.DataFrame({"trajectory_time_s": table.time_s,
            "dispatch_start_ns": ((1000 + table.time_s)*1e9).round().astype("int64")})
        dispatch["dispatch_end_ns"] = dispatch.dispatch_start_ns + 1000
        dispatch.to_csv(directory / "dispatch.csv", index=False)
        commands = pd.DataFrame({"trajectory_time_s": table.time_s, "command_valid": True})
        for axis in "xyz":
            commands[f"tcp_target_{axis}"] = table[f"tcp_{axis}_base"]
            commands[f"tcp_target_r{axis}"] = table[f"tcp_r{axis}"]
        commands.to_csv(directory / "robot/trajectory_command.csv", index=False)
        elapsed = np.arange(0., table.time_s.iloc[-1]+.0001, .02)
        clock = 1000 + elapsed
        pd.DataFrame({"query_start_s": clock-.001, "query_end_s": clock+.001,
            "valid": True, "frame_type": "world", "fx": 0., "fy": -10.*torque_scale, "fz": 2.*torque_scale,
            "tx": 0., "ty": 0., "tz": 0.}).to_csv(directory / "robot/robot_wrench.csv", index=False)
        robot = pd.DataFrame({"host_time_s": clock, "valid": True})
        for axis in "xyz":
            robot[f"tcp_{axis}"] = np.interp(elapsed, table.time_s, table[f"tcp_{axis}_base"])
            robot[f"tcp_r{axis}"] = table[f"tcp_r{axis}"].iloc[0]
        robot.tcp_z += .0005  # Measured motion differs from the commanded path.
        robot.to_csv(directory / "robot/robot_state.csv", index=False)
        atomic_json(directory / "robot/metadata.json", {"logger": {"status": "completed"},
            "execution_result": {"completed": True, "started_host_time_s": 999.,
                "schedule_origin_host_time_s": 1000.}})
        # Passive response intentionally differs from declared candidate joint angles.
        time_s = np.arange(999.9, 1000. + float(table.time_s.iloc[-1]) + .102, .002)
        ticks = (time_s*1e9).round().astype("int64")
        leg = pd.DataFrame({"host_time_s": time_s, "host_monotonic_ns": ticks,
            "query_start_ns": ticks-100_000, "query_end_ns": ticks})
        for name, axis, offset in (("hip", 1, .2), ("knee", 4, .3)):
            q = offset + .04*np.sin(np.pi*(time_s-1000)/float(table.time_s.iloc[-1]))
            cal = c["calibration"][name]
            leg[f"axis{axis}_position_raw"] = cal["zero_counts"] + np.rad2deg(q)*cal["counts_per_degree"]/cal["flexion_sign"]
            leg[f"axis{axis}_torque_3c1a_nm"] = torque_scale * (2. + np.sin(q))
            leg[f"axis{axis}_statusword_hex"] = "0x0231"
            leg[f"axis{axis}_error_code_hex"] = "0x0000"
            leg[f"axis{axis}_fault"] = False
            leg[f"axis{axis}_operation_enabled"] = False
        leg.to_csv(directory / "leg/leg.csv", index=False)
        binding = adapter.freeze_binding(directory, c, meta)
        assert binding["status"] == "BOUND", binding
        state = {"status": "CAPTURE_COMPLETE", "training_status": "PENDING",
            "trajectory_mode": "original_candidate", "empty_load": False,
            "config_path": str(config_path), "tcp_plan": meta, "original_training_binding": binding,
            "execution": {"completed": True, "same_session_realtime_cleanup_completed": True,
                "commands_dispatched": len(table)}}
        if parent:
            state["parent_session"] = str(parent)
        atomic_json(directory / "round_status.json", state)
        return directory
    create.config = c
    create.config_path = config_path
    create.anchor = anchor
    return create


def test_passive_inputs_run_the_real_original_algorithm(episode):
    from scripts.train_robot_leg_round import train
    source = episode()
    result = adapter.train_original(source, lambda directory, request, timeout: train(request))
    assert result["method"] == "FROZEN_RESIDUAL_GP_MEAN_GREEDY"
    assert len(result["result"]["theta"]) == 6
    assert result["result"]["predictions"]["physics_mean"]
    assert result["result"]["diagnostics"]["parameter_count"] == 6
    assert result["history"][0]["endpoints"] == {"E3": 1., "E2": 1., "peak_ratio": 1.}
    assert "W_mean" not in result
    measured = pd.read_csv(source / "original_measured.csv")
    assert not np.isclose(measured.q_hip_rad.iloc[0], .2)
    assert measured.observation_source.eq(SOURCE).all()
    assert result["dummy_leg_data_used"] is False
    assert all("leg.csv" not in name for name in result["input_sha256"])
    assert "robot measured TCP/wrench" in result["result"]["input_adapter"]
    assert not (source / "passive_training.json").exists()


def stub_worker(directory, request, timeout):
    return {"method": adapter.METHOD, "theta": [1., 10., 10., 1., 1., 1.],
        "observation_source": SOURCE,
        "candidates": ["reference", "candidate"],
        "action": {"candidate_id": "candidate", "role": "SEARCH", "stop_reason": None},
        "predictions": {"mean": [[1., 1., 1.], [.9, 1., 1.]], "std": [[.01]*3, [.02]*3]}}


def test_train_routes_through_original_worker_and_retains_fixed_baseline(episode, monkeypatch):
    first = episode("first")
    seen = []
    def worker(directory, request, timeout):
        seen.append(request)
        return stub_worker(directory, request, timeout)
    monkeypatch.setattr(lab, "run_algorithm_worker", worker)
    assert lab.train_session(first, training_mode="passive")["training_status"] == "COMPLETE"
    second = episode("second", cid="candidate", parent=first, torque_scale=.5)
    lab.train_session(second)
    assert [item["candidate_id"] for item in seen[-1]["history"]] == ["reference", "candidate"]
    assert seen[-1]["history"][-1]["endpoints"]["E3"] == pytest.approx(.5)
    assert seen[-1]["history"][-1]["endpoints"]["E2"] == pytest.approx(.5)


def test_existing_fk_converter_needs_no_external_mapping_table(episode):
    source = episode()
    c = dict(episode.config)
    assert "tcp_bindings" not in c["training"]
    binding = adapter.freeze_binding(source, c, read_json(source / "tcp_plan.json"))
    assert binding["status"] == "BOUND"
    assert binding["source"] == "existing_candidate_fk_converter"


def test_original_algorithm_duplicate_rule_is_preserved(episode, monkeypatch):
    first = episode("first")
    monkeypatch.setattr(lab, "run_algorithm_worker", stub_worker)
    lab.train_session(first)
    repeat = episode("repeat", parent=first)
    with pytest.raises(ValueError, match="DUPLICATE_CANDIDATE_NO_REPEAT_NOISE_MODEL"):
        lab.train_session(repeat)
    assert read_json(repeat / "round_status.json")["training_status"] == "FAILED"


def test_missing_analysis_calibration_is_not_guessed(episode):
    source = episode()
    c = dict(episode.config)
    c["analysis"] = dict(c["analysis"], derivative_rate_hz=None)
    binding = adapter.freeze_binding(source, c, read_json(source / "tcp_plan.json"))
    assert binding["status"] == "UNBOUND"
    assert "derivative_rate_hz" in binding["reason"]


def test_candidate_timeline_is_not_silently_retimed(episode):
    source = episode()
    path = Path(episode.config["training"]["candidates"][0]["path"])
    table = pd.read_csv(path)
    table.time_s *= .5
    table.to_csv(path, index=False)
    binding = adapter.freeze_binding(source, episode.config, read_json(source / "tcp_plan.json"))
    assert binding["status"] == "UNBOUND"
    assert "does not match original catalog" in binding["reason"]


def test_original_recommendation_selects_registered_candidate(episode, monkeypatch):
    source = episode()
    monkeypatch.setattr(lab, "run_algorithm_worker", stub_worker)
    lab.train_session(source)
    assert adapter.selected_tcp_settings(source) == {"candidate_id": "candidate"}
    seen = []
    def prepare(config, destination, **kwargs):
        seen.append(kwargs)
        destination.mkdir()
        atomic_json(destination / "round_status.json", {"status": "PREPARED"})
    monkeypatch.setattr(lab, "prepare", prepare)
    target = source.parent / "next"
    lab.repeat(source, target, use_trained=True)
    assert seen[0]["trajectory_settings"] == adapter.selected_tcp_settings(source)


def test_stale_original_input_prevents_recommendation(episode, monkeypatch):
    source = episode()
    monkeypatch.setattr(lab, "run_algorithm_worker", stub_worker)
    lab.train_session(source)
    path = source / "robot/robot_wrench.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input changed"):
        adapter.selected_tcp_settings(source)


def test_replacement_gp_worker_is_rejected(episode):
    source = episode()
    with pytest.raises(ValueError, match="original six-parameter"):
        adapter.train_original(source, lambda *a: {"method": "ROBOT_WRENCH_GP", "theta": []})


def test_original_stop_is_preserved(episode, monkeypatch):
    source = episode()
    def stopped(*args):
        result = stub_worker(*args)
        result["action"] = {"candidate_id": None, "role": "STOP", "stop_reason": "DOMAIN_EXHAUSTED"}
        return result
    monkeypatch.setattr(lab, "run_algorithm_worker", stopped)
    lab.train_session(source)
    with pytest.raises(ValueError, match="DOMAIN_EXHAUSTED"):
        adapter.selected_tcp_settings(source)


def test_original_prediction_is_copied_not_refitted_by_another_gp(episode, monkeypatch):
    first = episode("first")
    monkeypatch.setattr(lab, "run_algorithm_worker", stub_worker)
    lab.train_session(first)
    second = episode("second", cid="candidate", parent=first)
    binding = read_json(second / "original_training_binding.json")
    prediction = adapter.commit_original_prediction(first, second, binding)
    assert prediction["method"] == adapter.METHOD
    assert prediction["endpoint_family"] == ["E3", "E2", "peak_ratio"]
    assert prediction["mean"] == [.9, 1., 1.]


def test_completion_still_checks_actual_robot_feedback(episode):
    source = episode()
    state = read_json(source / "round_status.json")
    state.update(status="EMPTY_LOAD_COMPLETE", empty_load=True)
    atomic_json(source / "round_status.json", state)
    assert lab.complete(source)["status"] == "COMPLETE"
    path = source / "robot/robot_state.csv"
    robot = pd.read_csv(path)
    robot.tcp_z = .5
    robot.to_csv(path, index=False)
    with pytest.raises(ValueError, match="did not follow"):
        lab.complete(source)


def test_bound_model_never_requires_passive_joint_angle_tracking(episode):
    from measurement_validation.robot_leg_round import aligned_measurement
    source = episode()
    c = episode.config
    path = c["training"]["candidates"][0]["path"]
    with pytest.raises(ValueError, match="start angle inconsistent"):
        aligned_measurement(source / "leg/leg.csv", source / "dispatch.csv", path,
            c["analysis"], {"joints": c["calibration"]})
    measured, _ = aligned_measurement(source / "leg/leg.csv", source / "dispatch.csv", path,
        c["analysis"], {"joints": c["calibration"]}, require_reference_tracking=False,
        schedule_origin_host_time_s=1000.)
    assert measured.q_hip_rad.iloc[0] == pytest.approx(.2, abs=1e-5)


def test_unbound_training_stops_before_capture_or_motion(episode, monkeypatch):
    source = episode()
    binding = {"status": "UNBOUND", "reason": "analysis calibration missing"}
    atomic_json(source / "original_training_binding.json", binding)
    state = read_json(source / "round_status.json")
    state.update(status="PREPARED", original_training_binding=binding)
    atomic_json(source / "round_status.json", state)
    monkeypatch.setattr(lab, "_check_config_for_operation", lambda *a, **k: [])
    with pytest.raises(ValueError, match="offline comparison only"):
        lab.run(source, live=True, capture_factory=lambda *a: pytest.fail("must not capture"),
            execute_fn=lambda **k: pytest.fail("must not move"))
    assert read_json(source / "round_status.json")["status"] == "PREPARED"
    assert not (source / "running.lock").exists()


def test_robot_wrench_drives_original_reconstructed_endpoints(episode):
    source = episode()
    requests = []
    def worker(directory, request, timeout):
        requests.append(request)
        return stub_worker(directory, request, timeout)
    adapter.train_original(source, worker)
    original_metrics = read_json(source / "original_joint_metrics.json")
    path = source / "robot/robot_wrench.csv"
    wrench = pd.read_csv(path)
    wrench.fy *= 2
    wrench.fz *= 2
    wrench.to_csv(path, index=False)
    adapter.train_original(source, worker)
    # First-reference normalization stays one; absolute observed torques change.
    assert requests[0]["observation_source"] == SOURCE
    metrics = read_json(source / "original_joint_metrics.json")
    assert metrics["dummy_leg_data_used"] is False
    np.testing.assert_allclose(metrics["components_nm"], 2*np.array(original_metrics["components_nm"]))


def test_formal_training_does_not_read_or_depend_on_dummy_leg_sensors(episode):
    from scripts.train_robot_leg_round import train
    source = episode()
    first = adapter.train_original(source, lambda d, r, t: train(r))
    measured = pd.read_csv(source / "original_measured.csv")
    # The learner must still work when evaluation-only leg measurements vanish.
    (source / "leg/leg.csv").unlink()
    second = adapter.train_original(source, lambda d, r, t: train(r))
    pd.testing.assert_frame_equal(measured, pd.read_csv(source / "original_measured.csv"))
    np.testing.assert_allclose(first["result"]["theta"], second["result"]["theta"], rtol=0, atol=0)
    assert first["input_sha256"] == second["input_sha256"]


def test_robot_posture_comes_from_measured_tcp_not_nominal_or_leg_angles(episode):
    source = episode()
    execution = audit_robot_execution(source)
    reference = pd.read_csv(episode.config["training"]["candidates"][0]["path"])
    measured, _ = robot_measurement(source, episode.config, reference, execution)
    assert not np.allclose(measured.q_hip_rad, reference.q_hip_rad, atol=1e-5, rtol=0)
    reference.q_hip_rad = -123.  # Nominal angles are not observations.
    reference.q_knee_rad = -456.
    unchanged, _ = robot_measurement(source, episode.config, reference, execution)
    pd.testing.assert_frame_equal(measured, unchanged)


@pytest.mark.parametrize("change,reason", [
    ({"hip_center_in_base_m": None}, "hip_center"),
    ({"compensation_verified": False}, "compensation"),
    ({"force_robot_on_leg_sign": None}, "force_robot_on_leg_sign"),
    ({"state_time_offset_s": None}, "state_time_offset"),
    ({"rotation_base_from_wrench": [[1., 0., 0.]]*3}, "orthonormal"),
    ({"hip_center_in_base_m": [100., 100., 100.]}, "sagittal"),
])
def test_robot_observation_does_not_guess_missing_physics(episode, change, reason):
    import copy
    source = episode()
    c = copy.deepcopy(episode.config)
    c["robot_observation"].update(change)
    reference = pd.read_csv(c["training"]["candidates"][0]["path"])
    with pytest.raises(ValueError, match=reason):
        robot_measurement(source, c, reference, audit_robot_execution(source))


def test_reaction_sign_and_wrench_moment_arm_are_explicit(episode):
    import copy
    from scipy.spatial.transform import Rotation
    source = episode()
    execution = audit_robot_execution(source)
    c = copy.deepcopy(episode.config)
    reference = pd.read_csv(c["training"]["candidates"][0]["path"])
    measured, _ = robot_measurement(source, c, reference, execution)
    offset = np.array([.01, .02, -.01])
    lever = Rotation.from_euler("xyz", execution["anchor_pose"][3:]).apply(offset)
    c["robot_observation"]["tcp_to_pull_point_in_tcp_m"] = offset.tolist()
    c["robot_observation"]["hip_center_in_base_m"] = (np.array(c["robot_observation"]["hip_center_in_base_m"])+lever).tolist()
    wrench_path = source / "robot/robot_wrench.csv"
    wrench = pd.read_csv(wrench_path)
    moment = np.cross(lever, wrench[["fx", "fy", "fz"]].to_numpy())
    wrench[["tx", "ty", "tz"]] = moment
    wrench.to_csv(wrench_path, index=False)
    shifted, _ = robot_measurement(source, c, reference, execution)
    np.testing.assert_allclose(measured[["tau_hip_nm", "tau_knee_nm"]], shifted[["tau_hip_nm", "tau_knee_nm"]], atol=1e-10)
    c["robot_observation"]["force_robot_on_leg_sign"] = -1
    reversed_force, _ = robot_measurement(source, c, reference, execution)
    np.testing.assert_allclose(reversed_force[["tau_hip_nm", "tau_knee_nm"]], -measured[["tau_hip_nm", "tau_knee_nm"]], atol=1e-10)


@pytest.mark.parametrize("column,value,reason", [("tx", 1., "free connection moment"),
    ("fx", 1., "out-of-plane force"), ("frame_type", "tool", "frame differs")])
def test_unsupported_wrench_is_rejected_instead_of_ignored(episode, column, value, reason):
    source = episode()
    execution = audit_robot_execution(source)
    path = source / "robot/robot_wrench.csv"
    wrench = pd.read_csv(path)
    wrench[column] = value
    wrench.to_csv(path, index=False)
    reference = pd.read_csv(episode.config["training"]["candidates"][0]["path"])
    with pytest.raises(ValueError, match=reason):
        robot_measurement(source, episode.config, reference, execution)


def test_historical_dummy_leg_model_cannot_recommend_formal_motion(episode, monkeypatch):
    source = episode()
    monkeypatch.setattr(lab, "run_algorithm_worker", stub_worker)
    lab.train_session(source)
    state = read_json(source / "round_status.json")
    model = read_json(source / "original_training.json")
    model.pop("observation_source")
    state["original_training"] = model
    atomic_json(source / "original_training.json", model)
    atomic_json(source / "round_status.json", state)
    with pytest.raises(ValueError, match="historical dummy-leg-sensor"):
        adapter.selected_tcp_settings(source)


def test_removed_alternative_algorithm_is_not_importable():
    import importlib.util
    assert importlib.util.find_spec("experiment_algorithm.passive_training") is None


def prepare_from_fixture(episode, monkeypatch, *, name, empty_load=False, leg_angle=10.):
    """Use captured fixture data; never open either hardware connection."""
    root = episode.config_path.parent
    monkeypatch.setattr(lab, "_check_config_for_operation", lambda *a, **k: [])
    def capture(**kwargs):
        episode.anchor.save_json(kwargs["output_path"])
        assert kwargs["reference"] == episode.config["training"]["candidates"][0]["path"]
        assert not kwargs.get("independent_tcp")
    def leg_origin(config_path, output_path):
        origin = {"usable_for_motion": True,
            "axes": {joint: {"physical_angle_deg": leg_angle} for joint in ("hip", "knee")}}
        atomic_json(output_path, origin)
        return origin
    destination = root / name
    state = lab.prepare(episode.config_path, destination, empty_load=empty_load,
        capture_fn=capture, leg_origin_capture_fn=leg_origin)
    return destination, state


def test_prepare_preserves_original_candidate_and_ignores_leg_angles(episode, monkeypatch):
    first, state = prepare_from_fixture(episode, monkeypatch, name="prepare_a", leg_angle=10.)
    second, _ = prepare_from_fixture(episode, monkeypatch, name="prepare_b", leg_angle=40.)
    assert state["blockers"] == []
    assert state["original_training_binding"]["source"] == "existing_candidate_fk_converter"
    original = Path(episode.config["training"]["candidates"][0]["path"])
    assert (first / "original_candidate.csv").read_bytes() == original.read_bytes()
    assert (second / "original_candidate.csv").read_bytes() == original.read_bytes()
    a, b = (pd.read_csv(p / "tcp_trajectory.csv") for p in (first, second))
    pd.testing.assert_frame_equal(a, b)
    source = pd.read_csv(original)
    np.testing.assert_array_equal(a.time_s, source.time_s)
    for joint in ("hip", "knee"):
        np.testing.assert_array_equal(a[f"q_{joint}_ref"], source[f"q_{joint}_rad"])
    assert state["tcp_plan"]["joint_profile_transform"]["hip_delta_scale"] == 1.
    assert state["tcp_plan"]["joint_profile_transform"]["knee_phase_shift_s"] == 0.


def test_original_height_and_limits_block_run_without_reshaping(episode, monkeypatch):
    c = episode.config
    c["trajectory"]["target_peak_tcp_z_base_m"] = .9
    c["empty_load"].update(max_tcp_speed_m_s=.03, max_tcp_acceleration_m_s2=.01)
    atomic_json(episode.config_path, c)
    directory, state = prepare_from_fixture(episode, monkeypatch, name="blocked", empty_load=True)
    assert "trajectory_speed_limit_exceeded" in state["blockers"]
    assert "trajectory_acceleration_limit_exceeded" in state["blockers"]
    assert any("differs from requested" in reason for reason in state["blockers"])
    assert state["tcp_plan"]["actual_peak_tcp_z_base_m"] < .9
    table = pd.read_csv(directory / "tcp_trajectory.csv")
    assert table.time_s.iloc[-1] == 24.
    with pytest.raises(ValueError, match="prepared trajectory blocked"):
        lab.run(directory, live=True, empty_load=True,
            capture_factory=lambda *a: pytest.fail("must not capture"),
            execute_fn=lambda **k: pytest.fail("must not move"))
    assert read_json(directory / "round_status.json")["status"] == "PREPARED"


def test_original_execution_uses_frozen_reference_and_no_leg_capture(episode, monkeypatch):
    directory, state = prepare_from_fixture(episode, monkeypatch, name="empty_run", empty_load=True)
    assert state["blockers"] == []
    def execute(**kwargs):
        assert kwargs["reference_path_override"] == directory / "original_candidate.csv"
        assert "tcp_trajectory_settings" not in kwargs
        for t in pd.read_csv(directory / "tcp_trajectory.csv").time_s:
            tick = int(round((1000.+t)*1e9))
            kwargs["dispatch_observer"](t, tick, tick+1000)
        return {"completed": True, "same_session_realtime_cleanup_completed": True}
    result = lab.run(directory, live=True, empty_load=True, execute_fn=execute,
        capture_factory=lambda *a: pytest.fail("must not capture leg during empty run"))
    assert result["status"] == "EMPTY_LOAD_COMPLETE"
    assert len(pd.read_csv(directory / "dispatch.csv")) == 401


@pytest.mark.parametrize("tamper", ["source", "tcp", "text"])
def test_frozen_original_execution_rejects_changes_before_motion(episode, monkeypatch, tamper):
    directory, _ = prepare_from_fixture(episode, monkeypatch, name="tampered", empty_load=True)
    if tamper == "source":
        path = directory / "original_candidate.csv"
        path.write_bytes(path.read_bytes()+b"\n")
    else:
        path = directory / "tcp_trajectory.csv"
        table = pd.read_csv(path, keep_default_na=False)
        table.loc[1, "tcp_z_base" if tamper == "tcp" else "invalid_reason"] = .8 if tamper == "tcp" else "changed"
        table.to_csv(path, index=False)
    with pytest.raises((ValueError, AssertionError)):
        lab.run(directory, live=True, empty_load=True,
            execute_fn=lambda **k: pytest.fail("must not move"))
    assert read_json(directory / "round_status.json")["status"] == "PREPARED"
