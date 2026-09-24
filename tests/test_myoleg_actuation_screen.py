import json

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.myoleg_benchmark import actuation_screen as screen


def landscape(a, b, feasible=None):
    rows = []
    for i, (subject, values) in enumerate((("A", a), ("B", b))):
        for j, value in enumerate(values):
            rows.append({"subject_id": subject, "candidate_id": f"c{j}", "candidate_index": j,
                         "E3": value, "valid": True,
                         "feasible": True if feasible is None else feasible[i][j]})
    return pd.DataFrame(rows)


def test_shared_assistance_improvement_is_not_personalization():
    _, metrics = screen.summarize(landscape([.8, .9], [.81, .88]))
    assert metrics["oracle_unique_count"] == 1
    assert metrics["common_relative_regret_max"] == 0
    assert not metrics["practical_personalization_signal"]
    assert not metrics["confirmatory_ready"]


def test_distinct_oracles_with_negligible_common_regret_do_not_advance():
    _, metrics = screen.summarize(landscape([.8, .8001], [.8001, .8]))
    assert metrics["oracle_unique_count"] == 2
    assert metrics["exploratory_reference_rule_pass"]
    assert not metrics["practical_personalization_signal"]


def test_all_assisted_choices_worse_than_no_assistance_cannot_advance():
    summary, metrics = screen.summarize(landscape([1.0001, 1.0099], [1.0099, 1.0001]))
    assert metrics["exploratory_reference_rule_pass"]
    assert metrics["common_relative_regret_max"] > .005
    assert (summary.reference_regret < 0).all()
    assert metrics["fallback_common_candidate_id"] == screen.UNASSISTED_REFERENCE
    assert metrics["fallback_common_relative_regret_max"] == 0
    assert not metrics["practical_personalization_signal"]


def test_practical_signal_only_permits_development_and_common_is_shared_feasible():
    _, metrics = screen.summarize(landscape([.6, .9], [.9, .6]))
    assert metrics["practical_personalization_signal"]
    assert metrics["decision"] == "DEVELOPMENT_FOLLOWUP_ELIGIBLE"
    assert not metrics["confirmatory_ready"]
    summary, metrics = screen.summarize(landscape([.5, .8], [.6, .9], [[True, True], [False, True]]))
    assert metrics["common_candidate_id"] == "c1"
    assert summary.common_candidate_id.nunique() == 1


def test_no_feasible_or_shared_candidate_is_explicit():
    summary, metrics = screen.summarize(landscape([.5, .8], [.6, .9], [[True, False], [False, True]]))
    assert metrics["common_candidate_id"] is None
    assert summary.common_regret.isna().all()
    assert not metrics["practical_personalization_signal"]
    summary, metrics = screen.summarize(landscape([.5, .8], [.6, .9], [[False, False], [False, False]]))
    assert (summary.oracle_status == "NO_FEASIBLE_CANDIDATE").all()
    assert metrics["oracle_unique_count"] == 0
    assert not metrics["practical_personalization_signal"]


def test_missing_duplicates_and_sealed_subjects_fail_before_output(tmp_path, monkeypatch):
    rows = landscape([.8, .9], [.81, .88])
    with pytest.raises(ValueError, match="INCOMPLETE"):
        screen.summarize(rows.iloc[:-1])
    with pytest.raises(ValueError, match="DUPLICATED"):
        screen.summarize(pd.concat([rows, rows.iloc[:1]]))
    monkeypatch.setattr(screen, "development_ids", lambda: ("A", "B", "C"))
    for subjects in ([], ["SEALED"], ["A", "A"]):
        with pytest.raises(ValueError, match="DEVELOPMENT_SUBJECTS"):
            screen.run_screen(subjects=subjects, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_pilot_writes_protocol_before_truth_and_uses_unassisted_normalization(tmp_path, monkeypatch):
    output = tmp_path / "pilot"
    monkeypatch.setattr(screen, "development_ids", lambda: ("A", "B", "C"))

    class Native:
        def __init__(self, subject_id, domain, cache_dir):
            assert (output / "protocol.json").is_file()
            assert not (output / "completion.json").exists()
            self.domain = domain
            self.provenance = {"subject_id": subject_id, "test_backend": True}
            self.fresh_simulations = self.disk_hits = 0
            self.max_decomposition_residual_nm = 0.0

        def requested(self, point):
            signs = np.where(self.domain.subject_reference.phases == "flexion", 1., -1.)
            return signs[:, None] * np.array([3., 4.])

    monkeypatch.setattr(screen, "SimulatorBackend", Native)
    screen.run_screen(subjects=["A", "B"], output_dir=output, cache_dir=tmp_path / "cache")
    protocol = json.loads((output / "protocol.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    assert completion["complete"] and completion["responses"] == 54
    assert completion["protocol_sha256"] == screen.file_sha(output / "protocol.json")
    assert protocol["confirmatory_access"] is False
    rows = pd.read_csv(output / "landscape.csv")
    np.testing.assert_allclose(rows.native_E3, 1.0)
    assert (rows.E3 < 1.0).all()  # assisted middle candidate is not normalized to itself
    assert rows.feasible.all()
    with np.load(output / "assistance_waveforms.npz", allow_pickle=False) as artifact:
        assert artifact["assistance_tau_nm"].shape[0] == 27


def test_simulator_failure_never_leaves_a_complete_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "development_ids", lambda: ("A", "B", "C"))

    def fail(*args):
        raise RuntimeError("SIMULATOR_FAILED")

    monkeypatch.setattr(screen, "SimulatorBackend", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="SIMULATOR_FAILED"):
        screen.run_screen(output_dir=output)
    assert json.loads((output / "completion.json").read_text())["complete"] is False
