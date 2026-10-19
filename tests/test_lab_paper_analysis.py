import numpy as np
import pytest

from scripts.analyze_lab_paper_progress import (ads_summary, leave_one_angle_out,
    linear_fit, segment, stats)
from tests.test_robot_leg_round import measured_fixture
from safety.robot_trial_release import sha256_file


def test_line_and_angle_group_holdout():
    angles = np.array([0, 15, 30, 45, 60, 45, 30, 15, 0.])
    counts = -100000+2000*angles
    fit = linear_fit(angles, counts)
    assert fit["slope_counts_per_deg"] == pytest.approx(2000)
    assert fit["zero_counts"] == pytest.approx(-100000)
    result = leave_one_angle_out(angles, counts)
    assert result["max_abs_deg"] < 1e-10
    # A corrupted pair must be excluded together from its angle's training fold.
    counts[angles == 30] += 2000
    result = leave_one_angle_out(angles, counts)
    assert np.array(result["errors_deg"])[angles == 30] == pytest.approx([1, 1])


def test_raw_csv_summary_and_checksums(tmp_path):
    csv, _ = measured_fixture(tmp_path)
    summary, dt = ads_summary(csv)
    assert summary["rate_hz"] == pytest.approx(500)
    assert summary["gaps_over_4ms"] == 0
    assert dt.min() == pytest.approx(2)
    spec = dict(file=csv.name, sha256=sha256_file(csv), id="test", angle_deg=30., start_s=0, end_s=1)
    result = segment(tmp_path, spec, "knee")
    assert 490 <= result["rows"] <= 501
    spec["sha256"] = "0"*64
    with pytest.raises(ValueError, match="checksum"):
        segment(tmp_path, spec, "knee")


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_nonfinite_stats_are_not_reported_as_valid(values):
    with pytest.raises(ValueError): stats(values)
