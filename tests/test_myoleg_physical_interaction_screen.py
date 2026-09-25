"""Offline audit of the physical development screen; no MyoLeg/hardware run."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.mechanical_endpoints import branch_rms_components
from lower_limb_sim.myoleg_benchmark import actuation_screen
from lower_limb_sim.myoleg_benchmark import physical_interaction_screen as screen
from lower_limb_sim.myoleg_benchmark.physical_interaction import (
    PHYSICAL_INTERACTION_FORMULA_VERSION,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FakeNative:
    """Deterministic nonconstant demand, independent of candidate assistance."""

    def __init__(self, subject_id, domain, cache_dir):
        self.subject_id, self.domain = subject_id, domain
        self.provenance = {"subject_id": subject_id, "test_backend": True}
        self.fresh_simulations = self.disk_hits = 0
        self.max_decomposition_residual_nm = 0.0

    def requested(self, point):
        self.fresh_simulations += 1
        signs = np.where(self.domain.subject_reference.phases == "flexion", 1., -1.)
        base = 1.0 + .05 * (ord(self.subject_id) - ord("A"))
        return base * (signs[:, None] * np.array([3., 4.])
                       + .12 * point.trajectory.q + .03 * point.trajectory.dq)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    output = tmp_path_factory.mktemp("physical-screen") / "pilot"
    seen = []

    class ProtocolCheckedNative(FakeNative):
        def __init__(self, subject_id, domain, cache_dir):
            assert (output / "protocol.json").is_file()
            assert not (output / "completion.json").exists()
            protocol = read_json(output / "protocol.json")
            assert len(protocol["profiles"]) == 4
            assert len(protocol["candidates"]) == 27
            self.protocol_sha = screen.file_sha(output / "protocol.json")
            seen.append(subject_id)
            super().__init__(subject_id, domain, cache_dir)

        def requested(self, point):
            # Protocol is frozen before the very first response, and stays so.
            assert self.protocol_sha == screen.file_sha(output / "protocol.json")
            return super().requested(point)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(screen, "development_ids", lambda: ("D", "C", "A", "B"))
        monkeypatch.setattr(screen, "SimulatorBackend", ProtocolCheckedNative)
        screen.run_screen(output_dir=output, cache_dir=output.parent / "cache")
    return output, seen


def test_protocol_precedes_responses_and_declares_crossed_evaluator_scope(pilot):
    output, seen = pilot
    protocol = read_json(output / "protocol.json")
    completion = read_json(output / "completion.json")
    summary = read_json(output / "summary.json")
    assert seen == ["A", "B", "C"]
    assert protocol["native_subject_ids"] == seen
    assert protocol["formula_version"] == PHYSICAL_INTERACTION_FORMULA_VERSION
    assert protocol["confirmatory_access"] is False
    assert protocol["robot_motion"] == "NO-GO"
    assert protocol["v1_expansion"] == "PAUSED"
    assert protocol["learner_access"].startswith("none; evaluator-only")
    assert "crossed" in protocol["statistical_unit"]
    assert "no independent patient inference" in protocol["statistical_unit"]
    assert completion["complete"] and completion["responses"] == 324
    assert summary["native_subject_count"] == 3
    assert summary["profile_count"] == 4
    assert summary["confirmatory_ready"] is False
    assert set(read_json(output / "within_profile_summary.json")) == {
        "ZERO_INTERACTION", "HIP_SPRING", "KNEE_SPRING", "DAMPING",
    }


def test_every_artifact_and_declared_source_is_hashed(pilot):
    output, _ = pilot
    protocol = read_json(output / "protocol.json")
    completion = read_json(output / "completion.json")
    assert completion["protocol_sha256"] == screen.file_sha(output / "protocol.json")
    expected_files = {p.name for p in output.iterdir() if p.is_file()} - {"completion.json"}
    assert set(completion["artifact_sha256"]) == expected_files
    assert len(list(output.glob("traces_*.npz"))) == 12
    for filename, digest in completion["artifact_sha256"].items():
        assert digest == screen.file_sha(output / filename)
    for filename, digest in protocol["code_sha256"].items():
        assert digest == screen.file_sha(screen.ROOT / filename)
    assert protocol["cohort_manifest_sha256"] == screen.file_sha(screen.COHORT_MANIFEST)
    for config in read_json(output / "provenance.json")["configurations"]:
        assert config["trace_sha256"] == screen.file_sha(output / config["trace_file"])


def test_saved_decomposition_rebuilds_metrics_and_same_attachment_reference(pilot):
    output, _ = pilot
    rows = pd.read_csv(output / "landscape.csv")
    protocol = read_json(output / "protocol.json")
    profiles = {item["profile_id"]: item for item in protocol["profiles"]}
    metadata = read_json(output / "provenance.json")["configurations"]
    for config in metadata:
        profile = profiles[config["profile"]["profile_id"]]
        group = rows[rows.subject_id == config["configuration_id"]].set_index("candidate_id")
        with np.load(output / config["trace_file"], allow_pickle=False) as saved:
            q, dq = saved["q_rad"], saved["dq_rad_s"]
            np.testing.assert_array_equal(saved["force_mapping_valid"],
                                          saved["net_force_mapping_valid"] & saved["interaction_force_mapping_valid"])
            for prefix in ("net", "interaction"):
                np.testing.assert_array_equal(saved[f"{prefix}_force_mapping_valid"],
                                              saved[f"{prefix}_force_mapping_invalid_reason"] == "")
            elastic = -np.asarray(profile["stiffness_nm_per_rad"]) * (
                q - np.asarray(profile["reference_q_rad"])) * profile["resistance_scale"]
            viscous = -np.asarray(profile["damping_nm_s_per_rad"]) * dq * profile["resistance_scale"]
            np.testing.assert_allclose(saved["elastic_tau_nm"], elastic)
            np.testing.assert_allclose(saved["viscous_tau_nm"], viscous)
            np.testing.assert_allclose(saved["interaction_tau_nm"], elastic + viscous)
            np.testing.assert_allclose(saved["net_tau_nm"], saved["native_tau_nm"]
                                       - elastic - viscous - saved["assistance_tau_nm"])
            assert np.max(viscous * dq) <= 1e-12
            baseline = saved["native_tau_nm"][0] - saved["interaction_tau_nm"][0]
            np.testing.assert_allclose(saved["unassisted_reference_tau_nm"], baseline)
            baseline_components = np.asarray(branch_rms_components(
                *baseline.T, saved["time_s"][0], saved["phases"]))
            baseline_peaks = np.max(np.abs(baseline), axis=0)
            np.testing.assert_allclose(saved["reference_components_nm"], baseline_components)
            np.testing.assert_allclose(saved["reference_peaks_nm"], baseline_peaks)
            np.testing.assert_allclose(config["unassisted_reference_components_nm"], baseline_components)
            for i, candidate_id in enumerate(saved["candidate_ids"]):
                actual = group.loc[candidate_id]
                tau = saved["net_tau_nm"][i]
                components = np.asarray(branch_rms_components(
                    *tau.T, saved["time_s"][i], saved["phases"]))
                ratios = components / baseline_components
                assert actual.E3 == pytest.approx(np.mean(ratios))
                assert actual.E2 == pytest.approx(np.max(ratios))
                assert actual.peak_ratio == pytest.approx(np.max(np.max(np.abs(tau), axis=0) / baseline_peaks))
                assert bool(actual.feasible) == (actual.E2 <= 1.01 + 1e-12
                                                 and actual.peak_ratio <= 1.10 + 1e-12)
                unassisted = saved["native_tau_nm"][i] - saved["interaction_tau_nm"][i]
                parts = branch_rms_components(*unassisted.T, saved["time_s"][i], saved["phases"])
                assert actual.unassisted_E3 == pytest.approx(np.mean(np.asarray(parts) / baseline_components))
                assert actual.assistance_E3_change == pytest.approx(actual.E3 - actual.unassisted_E3)
            # Duration changes dq once; timing/allocation never change the attachment.
            ids = saved["candidate_ids"]
            for _, duration_rows in group.groupby("duration_scale"):
                indices = [int(np.flatnonzero(ids == value)[0]) for value in duration_rows.index]
                for index in indices[1:]:
                    np.testing.assert_array_equal(saved["interaction_tau_nm"][indices[0]],
                                                  saved["interaction_tau_nm"][index])
            assert group.iloc[0].unassisted_E3 == pytest.approx(1.)
            assert not np.allclose(saved["net_tau_nm"][0], baseline)


def test_zero_attachment_reproduces_previous_actuation_screen(pilot, monkeypatch):
    output, _ = pilot
    monkeypatch.setattr(actuation_screen, "SimulatorBackend", FakeNative)
    old_rows, _, _ = actuation_screen._subject_rows(
        "A", screen.ControlledActuationDomain(), screen.ControlledActuationConfig(), output.parent / "cache")
    old = pd.DataFrame(old_rows).set_index("candidate_id")
    new = pd.read_csv(output / "landscape.csv")
    new = new[new.subject_id == "A::ZERO_INTERACTION"].set_index("candidate_id")
    for column in ("E3", "E2", "peak_ratio", "assistance_E3_change", "assist_peak_l1_nm"):
        np.testing.assert_allclose(new.loc[old.index, column], old[column], atol=1e-14)
    np.testing.assert_array_equal(new.loc[old.index, "feasible"], old.feasible)


@pytest.mark.parametrize("subjects", [[], ["CONFIRMATORY"], ["A", "A"]])
def test_invalid_subject_selection_is_rejected_before_output_or_simulation(tmp_path, monkeypatch, subjects):
    monkeypatch.setattr(screen, "development_ids", lambda: ("A", "B", "C"))

    def forbidden(*args):
        pytest.fail("Invalid selection reached the native backend")

    monkeypatch.setattr(screen, "SimulatorBackend", forbidden)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="ONLY_UNIQUE_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED"):
        screen.run_screen(subjects=subjects, output_dir=output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "development_ids", lambda: ("A",))
    output = tmp_path / "existing"
    output.mkdir()
    # An empty output directory is already protected, before any native access.
    monkeypatch.setattr(screen, "SimulatorBackend", lambda *args: pytest.fail("Existing run was entered"))
    with pytest.raises(FileExistsError):
        screen.run_screen(output_dir=output)
    assert list(output.iterdir()) == []


def test_partial_failure_preserves_protocol_and_marks_run_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "development_ids", lambda: ("A",))

    class PartialFailureNative(FakeNative):
        def requested(self, point):
            if self.fresh_simulations >= 30:
                raise RuntimeError("DELIBERATE_NATIVE_FAILURE")
            return super().requested(point)

    monkeypatch.setattr(screen, "SimulatorBackend", PartialFailureNative)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="DELIBERATE_NATIVE_FAILURE"):
        screen.run_screen(output_dir=output)
    completion = read_json(output / "completion.json")
    assert completion["complete"] is False
    assert completion["protocol_sha256"] == screen.file_sha(output / "protocol.json")
    assert "DELIBERATE_NATIVE_FAILURE" in completion["error"]
    assert (output / "traces_A_ZERO_INTERACTION.npz").is_file()
    assert not (output / "summary.json").exists()
    assert not (output / "REPORT.md").exists()


def test_invalid_force_mapping_blocks_followup_but_keeps_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "development_ids", lambda: ("A",))
    monkeypatch.setattr(screen, "SimulatorBackend", FakeNative)
    original_force = screen._force

    def invalid_force(q, tau):
        mapped = original_force(q, tau)
        return replace(mapped, force_mapping_valid=np.zeros(len(q), dtype=bool),
                       invalid_reason=np.full(len(q), "INJECTED_MAPPING_FAILURE"))

    monkeypatch.setattr(screen, "_force", invalid_force)
    output = tmp_path / "bad_mapping"
    screen.run_screen(output_dir=output)
    assert read_json(output / "completion.json")["complete"] is True
    metrics = read_json(output / "summary.json")
    assert metrics["decision"] == "HOLD_EQUIVALENT_FORCE_MAPPING_INVALID"
    assert metrics["all_force_mappings_valid"] is False
    assert metrics["invalid_force_mapping_trace_count"] == 108
    rows = pd.read_csv(output / "landscape.csv")
    assert not rows.force_mapping_valid.any()
    assert rows.force_mapping_invalid_reasons.str.contains("INJECTED_MAPPING_FAILURE").all()
