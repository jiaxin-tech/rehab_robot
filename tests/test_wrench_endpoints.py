import numpy as np
import pytest

from measurement_validation.wrench_endpoints import load_window


def test_raw_short_peak_is_retained_between_model_grid_points():
    clock = np.arange(1001) / 1000
    values = np.ones((len(clock), 2))
    values[501, 0] = 11
    result = load_window(clock, values, np.ones(len(clock), bool), 0, 1, .002)
    assert result["peaks"][0] == 11
    assert result["raw_sample_count"] == 1001
    assert result["rms"][0] == pytest.approx(np.sqrt(1.12))


def test_exact_boundaries_use_full_requested_duration():
    clock = np.array([0., .2, .6, 1.])
    values = np.column_stack([clock, np.ones(4) * 3])
    result = load_window(clock, values, [True] * 4, .1, .9, .5)
    points = np.array([.1, .2, .6, .9])
    expected = np.sqrt(np.trapezoid(points ** 2, points) / .8)
    assert result["rms"] == pytest.approx([expected, 3.])
    assert result["peaks"] == pytest.approx([.9, 3.])
    assert result["duration_s"] == pytest.approx(.8)


def test_host_clock_csv_rounding_is_not_missing_coverage():
    clock = 240000. + np.arange(11) * .02
    end = np.nextafter(clock[-1], np.inf)
    result = load_window(clock, np.ones((11, 2)), [True] * 11, clock[0], end, .02)
    assert result["rms"] == pytest.approx([1., 1.])


@pytest.mark.parametrize("case", ["invalid_bracket", "gap", "outside", "reversed"])
def test_bad_boundary_or_missing_source_is_rejected(case):
    clock = np.array([0., .2, .6, 1.])
    values = np.ones((4, 2))
    valid = np.ones(4, bool)
    begin, end, gap = .1, .9, .5
    if case == "invalid_bracket":
        valid[0] = False
    elif case == "gap":
        gap = .3
    elif case == "outside":
        begin = -.1
    else:
        clock[2] = .1
    with pytest.raises(ValueError):
        load_window(clock, values, valid, begin, end, gap)
