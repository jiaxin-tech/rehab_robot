"""Raw measured endpoints retain peaks between the model's command samples."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from measurement_validation.robot_leg_round import aligned_measurement
from measurement_validation.torque_endpoints import (
    endpoints, raw_torque_endpoint_trace, torque_metrics_from_trace,
)


def measurement_fixture(directory, *, outside_peaks=False):
    command = np.arange(7, dtype=float) / 10
    reference = pd.DataFrame({
        "time_s": command,
        "cycle_phase": ["flexion"] * 4 + ["extension"] * 3,
        "q_hip_rad": .5,
        "q_knee_rad": .6,
    })
    reference_path = directory / "reference.csv"
    reference.to_csv(reference_path, index=False)
    sensor = np.arange(-.02, .622, .002)
    origin = 123456789000000
    ticks = origin + np.rint(sensor * 1e9).astype(np.int64)
    hip = np.ones(len(sensor))
    hip[np.argmin(abs(sensor - .15))] = 11.
    if outside_peaks:
        hip[np.argmin(abs(sensor + .01))] = 100.
        hip[np.argmin(abs(sensor - .61))] = 100.
    raw = {
        "sample_index": np.arange(len(sensor)),
        "receive_elapsed_ms": (sensor - sensor[0]) * 1000,
        "host_monotonic_ns": ticks,
        "query_start_ns": ticks - 100000,
        "query_end_ns": ticks,
    }
    for name, axis, angle, torque in (("hip", 1, .5, hip), ("knee", 4, .6, np.full(len(sensor), 2.))):
        raw.update({
            f"axis{axis}_position_raw": np.rint(np.rad2deg(angle) * 1000),
            f"axis{axis}_torque_3c1a_nm": torque,
            f"axis{axis}_statusword_hex": "0x0000",
            f"axis{axis}_error_code_hex": "0x0000",
            f"axis{axis}_fault": False,
            f"axis{axis}_operation_enabled": False,
        })
    csv_path = directory / "leg.csv"
    pd.DataFrame(raw).to_csv(csv_path, index=False)
    dispatch_ticks = origin + np.rint(command * 1e9).astype(np.int64)
    dispatch_path = directory / "dispatch.csv"
    pd.DataFrame({"trajectory_time_s": command,
                  "dispatch_start_ns": dispatch_ticks - 100000,
                  "dispatch_end_ns": dispatch_ticks}).to_csv(dispatch_path, index=False)
    analysis = dict(reviewed=True, alignment_reviewed=True, sensor_receive_delay_s=0.,
                    command_application_delay_s=0., max_gap_s=.01, max_read_duration_s=.01,
                    max_dispatch_time_error_s=.02, derivative_rate_hz=500., derivative_window_s=.02,
                    max_start_angle_error_deg=1., max_tracking_error_deg=2.)
    calibration = dict(reviewed=True, independent_validation_passed=True,
                       joints={name: dict(counts_per_degree=1000., zero_counts=0., flexion_sign=1,
                                          safe_min_deg=-10., safe_max_deg=140.,
                                          torque_sign_to_inverse_dynamics=1)
                               for name in ("hip", "knee")})
    return csv_path, dispatch_path, reference_path, analysis, calibration


def test_narrow_raw_peak_survives_coarse_model_grid(tmp_path):
    args = measurement_fixture(tmp_path)
    measured, metrics = aligned_measurement(*args)
    assert len(measured) == 7
    assert measured.tau_hip_nm.max() == pytest.approx(1.)
    assert metrics["peaks_nm"] == pytest.approx([11., 2.])
    assert metrics["components_nm"][0] > 1.
    assert metrics["endpoint_sampling"]["method"] == "RAW_SENSOR_TIME_WEIGHTED_WITH_INTERPOLATED_BOUNDARIES_V1"
    assert len(metrics["endpoint_trace"]["time_s"]) > len(measured)
    assert metrics["analysis_config_sha256"]
    assert metrics["calibration_config_sha256"]
    assert metrics["raw_torque_preserved"] and not metrics["automatically_tared"]
    rebuilt = torque_metrics_from_trace(metrics["endpoint_trace"])
    assert rebuilt["components_nm"] == pytest.approx(metrics["components_nm"])
    assert rebuilt["peaks_nm"] == pytest.approx(metrics["peaks_nm"])


def test_pre_and_post_roll_peaks_are_excluded(tmp_path):
    args = measurement_fixture(tmp_path, outside_peaks=True)
    _, metrics = aligned_measurement(*args)
    assert metrics["peaks_nm"] == pytest.approx([11., 2.])
    assert min(metrics["endpoint_trace"]["time_s"]) == 0.
    assert max(metrics["endpoint_trace"]["time_s"]) == pytest.approx(.6)


def test_nonuniform_sensor_samples_use_time_weighted_rms():
    time = np.array([0., .01, .45, .5, .72, 1.])
    hip = np.array([1., 2., 3., 4., 2., 1.])
    knee = hip * 2
    trace = raw_torque_endpoint_trace(time, hip, knee,
                                      motion_start_s=0., phase_boundary_s=.5, motion_end_s=1.)
    metrics = torque_metrics_from_trace(trace)
    expected = np.sqrt(np.trapezoid(hip[:4] ** 2, time[:4]) / .5)
    assert metrics["components_nm"][0] == pytest.approx(expected)
    assert metrics["components_nm"][0] != pytest.approx(np.sqrt(np.mean(hip[:4] ** 2)))
    assert metrics["components_nm"][2] == pytest.approx(expected * 2)


def test_exact_phase_boundary_is_interpolated_and_shared():
    time = np.array([-.1, .2, .4, .8, 1.1])
    hip = np.array([1., 2., 5., 3., 2.])
    trace = raw_torque_endpoint_trace(time, hip, hip * 2,
                                      motion_start_s=0., phase_boundary_s=.3, motion_end_s=1.)
    grid = np.asarray(trace["time_s"])
    torque = np.asarray(trace["tau_hip_nm"])
    boundary = trace["phase_boundary_time_s"]
    assert boundary == .3
    assert torque[grid == boundary].item() == pytest.approx(3.5)
    metrics = torque_metrics_from_trace(trace)
    flex, extension = grid <= boundary, grid >= boundary
    assert metrics["components_nm"][0] == pytest.approx(
        np.sqrt(np.trapezoid(torque[flex] ** 2, grid[flex]) / .3))
    assert metrics["components_nm"][1] == pytest.approx(
        np.sqrt(np.trapezoid(torque[extension] ** 2, grid[extension]) / .7))
    assert np.isclose(np.diff(grid).sum(), 1.)


def test_same_raw_reference_keeps_one_normalization_denominator():
    time = np.array([0., .1, .3, .5, .8, 1.])
    reference_trace = raw_torque_endpoint_trace(time, 2 + time, 3 + time,
                                                motion_start_s=0., phase_boundary_s=.5, motion_end_s=1.)
    reference = torque_metrics_from_trace(reference_trace)
    assert endpoints(reference, reference) == dict(E3=1., E2=1., peak_ratio=1.)
    candidate_trace = deepcopy(reference_trace)
    for name in ("tau_hip_nm", "tau_knee_nm"):
        candidate_trace[name] = (np.asarray(candidate_trace[name]) * .8).tolist()
    candidate = torque_metrics_from_trace(candidate_trace)
    assert endpoints(candidate, reference) == pytest.approx(dict(E3=.8, E2=.8, peak_ratio=.8))


@pytest.mark.parametrize("change", ["nonfinite", "wrong_phase", "missing_boundary", "duplicate_time"])
def test_invalid_trace_cannot_supply_measured_endpoints(change):
    trace = raw_torque_endpoint_trace(np.array([0., .25, .5, .75, 1.]), np.ones(5), np.ones(5),
                                      motion_start_s=0., phase_boundary_s=.5, motion_end_s=1.)
    if change == "nonfinite": trace["tau_hip_nm"][1] = float("nan")
    if change == "wrong_phase": trace["phase"][0] = "extension"
    if change == "missing_boundary": trace["phase_boundary_time_s"] = .51
    if change == "duplicate_time": trace["time_s"][1] = trace["time_s"][0]
    with pytest.raises(ValueError):
        torque_metrics_from_trace(trace)
