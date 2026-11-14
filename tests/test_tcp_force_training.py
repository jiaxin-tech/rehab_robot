"""Robot-only TCP learning, using synthetic recordings and no hardware."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from collection.twincat_leg import atomic_json
from config.bundle import configuration
from control.start_anchor import load_start_anchor
from control.independent_tcp_trajectory import TRAJECTORY_ID
from measurement_validation.robot_execution import read_json
from scripts import lab_session as lab
from scripts import tcp_force_training as force


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    c = configuration()
    c["validation"]["capture_dummy_leg"] = False
    c["force_training"].update(compensation_verified=True, unloaded_wrench_bias=[0.]*6)
    # No patient, dummy-leg or joint-model input is available to this fixture.
    for key in ("robot_observation", "calibration", "leg", "analysis"):
        c.pop(key)
    c["training"].pop("candidates")
    config_path = tmp_path / "config.json"
    atomic_json(config_path, c)
    monkeypatch.setattr(lab, "preview_trajectory", lambda **kwargs: None)
    anchor = load_start_anchor(Path(__file__).resolve().parents[1]/"results/empty_load_118/anchor.json")
    anchor = replace(anchor, trajectory_id=TRAJECTORY_ID,
        tcp_pose_base=(.5, 0., .5)+anchor.tcp_pose_base[3:], tcp_position_base_m=(.5, 0., .5))
    def capture(**kwargs):
        assert kwargs["independent_tcp"] is True
        anchor.save_json(kwargs["output_path"])
    def prepare(name, *, parent=None, selection=None, empty=False):
        directory = tmp_path/name
        state = lab.prepare(config_path, directory, empty_load=empty,
            trajectory_settings=selection, parent_session=parent, repeat_index=1 if parent else None,
            capture_fn=capture, leg_origin_capture_fn=lambda *a: pytest.fail("no leg capture"))
        assert state["blockers"] == []
        return directory
    prepare.config = c
    prepare.config_path = config_path
    prepare.capture = capture
    return prepare


def record(directory, *, scale=1.):
    state = read_json(directory/"round_status.json")
    table = pd.read_csv(directory/"tcp_trajectory.csv")
    origin = time.perf_counter()+1.
    times = table.time_s.to_numpy(float)
    (directory/"robot").mkdir(exist_ok=True)
    dispatch = pd.DataFrame({"trajectory_time_s": times,
        "dispatch_start_ns": ((origin+times)*1e9).round().astype("int64")})
    dispatch["dispatch_end_ns"] = dispatch.dispatch_start_ns+1000
    dispatch.to_csv(directory/"dispatch.csv", index=False)
    commands = pd.DataFrame({"trajectory_time_s": times, "command_valid": True})
    for axis in "xyz":
        commands[f"tcp_target_{axis}"] = table[f"tcp_{axis}_base"]
        commands[f"tcp_target_r{axis}"] = table[f"tcp_r{axis}"]
    commands.to_csv(directory/"robot/trajectory_command.csv", index=False)
    elapsed = np.linspace(0., times[-1], int(round(times[-1]*50))+1)
    clock = origin+elapsed
    pd.DataFrame({"query_start_s": clock-.001, "query_end_s": clock+.001,
        "valid": True, "frame_type": "world", "fx": 0., "fy": 5.*scale, "fz": 10.*scale,
        "tx": 2.*scale, "ty": 0., "tz": 0.}).to_csv(directory/"robot/robot_wrench.csv", index=False)
    robot = pd.DataFrame({"host_time_s": clock, "valid": True})
    for axis in "xyz":
        robot[f"tcp_{axis}"] = np.interp(elapsed, times, table[f"tcp_{axis}_base"])
        robot[f"tcp_r{axis}"] = table[f"tcp_r{axis}"].iloc[0]
    robot.to_csv(directory/"robot/robot_state.csv", index=False)
    execution = {"completed": True, "same_session_realtime_cleanup_completed": True,
        "commands_dispatched": len(times), "schedule_origin_host_time_s": origin,
        "started_host_time_s": origin-.5}
    atomic_json(directory/"robot/metadata.json", {"logger": {"status": "completed"}, "execution_result": execution})
    state.update(status="EMPTY_LOAD_COMPLETE" if state.get("empty_load") else "CAPTURE_COMPLETE", execution=execution)
    atomic_json(directory/"round_status.json", state)
    return execution


def test_robot_only_force_learning_and_next_candidate(experiment):
    first = experiment("first")
    record(first)
    state = lab.train_session(first, training_mode="passive")
    result = state["force_training"]
    assert result["method"] == force.METHOD
    assert result["action"]["candidate_id"] in ("tcp_slower_lift", "tcp_slower_return")
    assert result["six_parameter_identification"] is False
    assert result["dummy_leg_data_used"] is False
    assert "theta" not in result and "E3" not in result
    expected = .5*(np.hypot(5., 10.)/100.+2./200.)
    assert result["history"][0]["values"][0] == pytest.approx(expected)
    assert lab.complete(first)["status"] == "COMPLETE"
    selection = force.selected_candidate(first)
    second = experiment("second", parent=first, selection=selection)
    record(second, scale=.8)
    next_state = lab.train_session(second)
    assert len(next_state["force_training"]["history"]) == 2
    assert next_state["force_training"]["prediction_evaluation"]["status"] == "COMMITTED_BEFORE_MOTION"
    assert not (first/"leg").exists()
    assert not (first/"original_training.json").exists()


def test_unchanged_task_is_required_despite_lower_force(experiment):
    first = experiment("first")
    record(first, scale=.01)
    path = first/"robot/robot_state.csv"
    robot = pd.read_csv(path)
    robot.tcp_z = .5
    robot.to_csv(path, index=False)
    with pytest.raises(ValueError, match="did not follow"):
        lab.train_session(first)
    assert read_json(first/"round_status.json")["training_status"] == "FAILED"


def test_candidate_cannot_shrink_excursion_or_slow_without_limit(experiment):
    c = experiment.config
    _, settings = force.selected_settings(c)
    settings["horizontal_displacement_m"] = 0.
    with pytest.raises(ValueError, match="horizontal task"):
        force.check_task(c, settings, [.5, 0., .5])
    _, settings = force.selected_settings(c)
    settings["lift_duration_s"] = 10000.
    with pytest.raises(ValueError, match="duration"):
        force.check_task(c, settings, [.5, 0., .5])
    _, settings = force.selected_settings(c)
    with pytest.raises(ValueError, match="vertical task"):
        force.check_task(c, settings, [.5, 0., .89])


def test_force_training_refuses_stale_or_tampered_inputs(experiment):
    first = experiment("first")
    record(first)
    lab.train_session(first)
    path = first/"robot/robot_wrench.csv"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="input changed"):
        force.selected_candidate(first)


def test_exact_repeat_is_explicit_evaluation_outside_adaptive_history(experiment):
    first = experiment("first")
    record(first)
    lab.train_session(first)
    target = first.parent/"repeat"
    with pytest.raises(ValueError, match="explicit --evaluation"):
        lab.repeat(first, target, capture_fn=experiment.capture,
            leg_origin_capture_fn=lambda *a: pytest.fail("no leg"))
    state = lab.repeat(first, target, evaluation=True, capture_fn=experiment.capture,
        leg_origin_capture_fn=lambda *a: pytest.fail("no leg"))
    assert state["force_training_binding"]["candidate_id"] == "tcp_reference"
    assert state["trial_purpose"] == "evaluation" and "parent_session" not in state
    record(target, scale=.9)
    result = lab.train_session(target)["force_evaluation"]
    assert result["enters_adaptive_history"] is False and result["action"] is None
    assert not (target / "tcp_force_training.json").exists()


def test_complete_robot_task_requires_no_leg_or_patient_geometry(experiment):
    directory = experiment("empty", empty=True)
    record(directory)
    assert lab.complete(directory)["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="completed with-leg"):
        lab.train_session(directory)


def test_compensation_is_a_robot_requirement_and_never_inferred_from_patient(experiment):
    c = experiment.config
    c["force_training"]["compensation_verified"] = False
    atomic_json(experiment.config_path, c)
    blockers = lab.check_config(experiment.config_path)
    assert any("compensation" in reason for reason in blockers)
    assert not any("hip_center" in reason or "ADS" in reason for reason in blockers)
    assert lab.check_config(experiment.config_path, empty_load=True) == []


def test_robot_only_run_closes_then_trains(experiment):
    directory = experiment("run")
    def execute(**kwargs):
        assert "execution_guard" not in kwargs
        execution = record(directory)
        dispatch = pd.read_csv(directory/"dispatch.csv")
        for row in dispatch.itertuples():
            kwargs["dispatch_observer"](row.trajectory_time_s, row.dispatch_start_ns, row.dispatch_end_ns)
        return execution
    state = lab.run(directory, live=True, execute_fn=execute,
        capture_factory=lambda *a: pytest.fail("no fake-leg dependency"))
    assert state["training_status"] == "COMPLETE"
    assert state["force_training"]["method"] == force.METHOD
    ledger = read_json(directory / "force_trial_ledger.json")
    assert len(ledger["attempts"]) == 1 and ledger["attempts"][0]["status"] == "COMPLETE"
    assert state["force_training"]["attempt_accounting"]["complete"] is True


@pytest.mark.parametrize("invalid", ["flag", "negative_query", "nan"])
def test_invalid_moving_wrench_is_never_treated_as_low_force(experiment, invalid):
    directory = experiment("invalid")
    record(directory)
    path = directory/"robot/robot_wrench.csv"
    wrench = pd.read_csv(path)
    if invalid == "flag":
        wrench.loc[100, "valid"] = False
    elif invalid == "negative_query":
        wrench.loc[100, "query_end_s"] = wrench.loc[100, "query_start_s"]-.001
    else:
        wrench.loc[100, "fz"] = np.nan
    wrench.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid .*wrench"):
        lab.train_session(directory)


def test_robot_only_observation_needs_no_ads_or_leg_fields(experiment):
    target = experiment.config_path.parent/"observe"
    def acquire(**kwargs):
        assert kwargs["observation_guard"] is None
        return {"completed": True, "motion_commanded": False}
    result = lab.observe(experiment.config_path, target, duration_s=.1,
        capture_factory=lambda *a: pytest.fail("must not use ADS"), acquire_fn=acquire)
    assert result["status"] == "OBSERVATION_COMPLETE"
    assert not (target/"leg").exists()


def test_force_total_budget_and_stop_are_preserved(experiment):
    c = experiment.config
    c["force_training"]["total_budget"] = 2
    atomic_json(experiment.config_path, c)
    first = experiment("first")
    record(first)
    lab.train_session(first)
    second = experiment("second", parent=first, selection=force.selected_candidate(first))
    record(second, scale=.9)
    result = lab.train_session(second)["force_training"]
    assert result["action"]["stop_reason"] == "TOTAL_BUDGET_REACHED"
    with pytest.raises(ValueError, match="TOTAL_BUDGET_REACHED"):
        force.selected_candidate(second)


def test_six_parameter_comparison_cannot_start_loaded_motion(tmp_path):
    atomic_json(tmp_path/"round_status.json", {"status": "PREPARED", "trajectory_mode": "original_candidate"})
    with pytest.raises(ValueError, match="offline comparison only"):
        lab.run(tmp_path, live=True, execute_fn=lambda **k: pytest.fail("must not move"))


def fake_run(directory, *, scale=1., fails=False):
    def execute(**kwargs):
        from measurement_validation.force_trial_ledger import NAME, root_for
        ledger = read_json(root_for(directory) / NAME)
        assert ledger["attempts"][-1]["session"] == str(directory)
        assert ledger["attempts"][-1]["status"] == "RUNNING"  # Before any physical dispatch.
        if fails:
            raise RuntimeError("synthetic failed hardware attempt")
        execution = record(directory, scale=scale)
        for row in pd.read_csv(directory / "dispatch.csv").itertuples():
            kwargs["dispatch_observer"](row.trajectory_time_s, row.dispatch_start_ns, row.dispatch_end_ns)
        return execution
    return lab.run(directory, live=True, execute_fn=execute,
                   capture_factory=lambda *a: pytest.fail("no ADS in software test"))


def test_attempt_budget_is_charged_before_run_and_stop_keeps_evaluation_separate(experiment):
    c = experiment.config
    c["force_training"]["total_budget"] = 2
    c["force_training"]["evaluation_budget"] = 2
    atomic_json(experiment.config_path, c)
    first = experiment("first")
    first_state = fake_run(first)
    second = first.parent / "second"
    lab.repeat(first, second, use_trained=True, capture_fn=experiment.capture,
               leg_origin_capture_fn=lambda *a: pytest.fail("no leg"))
    second_state = fake_run(second, scale=.9)
    assert second_state["force_training"]["action"]["stop_reason"] == "TOTAL_BUDGET_REACHED"
    with pytest.raises(ValueError, match="TOTAL_BUDGET_REACHED"):
        lab.repeat(second, first.parent / "blocked", use_trained=True, capture_fn=experiment.capture)
    evaluation = first.parent / "evaluation"
    evaluated = lab.repeat(second, evaluation, evaluation=True, capture_fn=experiment.capture,
                           leg_origin_capture_fn=lambda *a: pytest.fail("no leg"))
    assert evaluated["force_training_binding"]["candidate_id"] == second_state["force_training_binding"]["candidate_id"]
    result = fake_run(evaluation, scale=.85)
    assert "force_training" not in result and result["force_evaluation"]["action"] is None
    assert lab.complete(evaluation)["status"] == "COMPLETE"
    ledger = read_json(first / "force_trial_ledger.json")
    assert [row["purpose"] for row in ledger["attempts"]] == ["adaptive", "adaptive", "evaluation"]
    assert all(row["status"] == "COMPLETE" for row in ledger["attempts"])


def test_failed_run_latches_root_and_prevents_free_sibling_retry(experiment):
    first = experiment("first")
    fake_run(first)
    second = first.parent / "failed"
    lab.repeat(first, second, use_trained=True, capture_fn=experiment.capture,
               leg_origin_capture_fn=lambda *a: pytest.fail("no leg"))
    with pytest.raises(RuntimeError, match="synthetic failed"):
        fake_run(second, fails=True)
    ledger = read_json(first / "force_trial_ledger.json")
    assert len(ledger["attempts"]) == 2 and ledger["attempts"][1]["status"] == "FAILED"
    with pytest.raises(ValueError, match="failed or unfinished"):
        lab.repeat(first, first.parent / "sibling_retry", use_trained=True, capture_fn=experiment.capture)


def test_paper_fixed_period_rejects_conservative_retiming_that_changes_cycle(experiment):
    c = experiment.config
    c["force_training"]["fixed_cycle_duration_s"] = 164.
    _, settings = force.selected_settings(c)
    force.check_task(c, settings, [.5, 0., .5])
    settings["lift_duration_s"] += .1
    with pytest.raises(ValueError, match="fixed comparison cycle"):
        force.check_task(c, settings, [.5, 0., .5])
