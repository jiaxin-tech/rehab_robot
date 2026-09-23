"""Scientific aggregation and winner claims using synthetic result rows only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.myoleg_benchmark import report


def result_row(subject_id, method="MODEL_INFORMED_BO_EI", seed=0, loss=.8, **changes):
    row = dict(
        subject_id=subject_id, family="BETA_TIMING", noise_std=0., seed=seed,
        method=method, budget=4, status="complete", recommendation_id="executed:1",
        true_feasible=True, selection_loss=loss, true_E3=loss, true_E2=.99,
        true_peak_ratio=1., improvement_pct=100 * (1 - loss),
        constraint_violations=0, invalid_trials=0, executed_trials=4, elapsed_s=2.,
    )
    row.update(changes)
    return row


def summaries(rows):
    return report.summarize(report.prepare_results(pd.DataFrame(rows)))


def complete_comparison(*, tied=False):
    protocol = {
        "methods": ["MODEL_INFORMED_BO_EI", "RANDOM"],
        "seeds": {"random": [0, 1, 2]},
        "primary_budget": 4,
        "primary_noise_std": 0.,
    }
    rows = []
    for subject_id in sorted(report.DEVELOPMENT_IDS):
        rows.append(result_row(subject_id, loss=.8))
        for seed in protocol["seeds"]["random"]:
            rows.append(result_row(subject_id, method="RANDOM", seed=seed, loss=.8 if tied else .9))
    return rows, protocol


def test_seeds_are_averaged_within_subject_before_equal_subject_weighting():
    rows = [result_row("MYOLEG_VP_001", loss=.8)]
    rows.extend(result_row("MYOLEG_VP_002", seed=seed, loss=.4) for seed in range(3))
    per_subject, summary = summaries(rows)
    assert per_subject.set_index("subject_id").loc["MYOLEG_VP_002", "seed_count"] == 3
    assert summary.iloc[0].selection_loss == pytest.approx(.6)
    assert summary.iloc[0].selection_loss != pytest.approx(.5)  # Incorrect row-weighted mean.
    assert summary.iloc[0].subject_count == 2
    assert summary.iloc[0].runs == 4
    assert summary.iloc[0].seeds_min == 1
    assert summary.iloc[0].seeds_max == 3


def test_native_model_does_not_enter_development_average_or_subject_count():
    rows = [result_row("MYOLEG_VP_001", loss=.8), result_row("MYOLEG_VP_002", loss=.6)]
    rows.extend(result_row("MYOLEG_NATIVE_P0", seed=seed, loss=.2) for seed in range(5))
    per_subject, summary = summaries(rows)
    development = summary[summary.scope == "DEVELOPMENT"].iloc[0]
    native = summary[summary.scope == "NATIVE"].iloc[0]
    assert development.selection_loss == pytest.approx(.7)
    assert development.subject_count == 2
    assert development.runs == 2
    assert native.selection_loss == pytest.approx(.2)
    assert native.subject_count == 1
    assert native.runs == 5
    assert set(per_subject[per_subject.scope == "DEVELOPMENT"].subject_id) == {
        "MYOLEG_VP_001", "MYOLEG_VP_002",
    }


def test_failed_run_keeps_declared_penalty_in_mean_and_failure_count():
    rows = [
        result_row("MYOLEG_VP_001", loss=.8),
        result_row(
            "MYOLEG_VP_002", loss=1., status="REFERENCE_MEASUREMENT_FAILED",
            recommendation_id=None, true_feasible=False, true_E3=np.nan,
            true_E2=np.nan, true_peak_ratio=np.nan, invalid_trials=1, executed_trials=1,
        ),
    ]
    _, summary = summaries(rows)
    result = summary.iloc[0]
    assert result.selection_loss == pytest.approx(.9)
    assert result.improvement_pct == pytest.approx(10.)
    assert result.failure_rate == pytest.approx(.5)
    assert result.failure_runs == 1
    assert result.true_feasible_rate == pytest.approx(.5)
    # Raw E3 lacks a failed measurement; it must not replace the penalized loss.
    assert result.true_E3 == pytest.approx(.8)
    assert result.runs == 2


@pytest.mark.parametrize("status", ["REFERENCE_MEASUREMENT_FAILED", "complete"])
def test_report_rejects_unearned_improvement_for_failure_or_infeasibility(status):
    row = result_row(
        "MYOLEG_VP_001", loss=.8, status=status, true_feasible=False,
        true_E2=1.05, recommendation_id=None if status != "complete" else "executed:1",
    )
    with pytest.raises(ValueError, match="selection_loss disagrees"):
        report.prepare_results(pd.DataFrame([row]))


@pytest.mark.parametrize("missing", ["subject", "seed", "method"])
def test_missing_planned_coverage_prevents_winner_claim(missing):
    rows, protocol = complete_comparison()
    if missing == "subject":
        rows = [r for r in rows if r["subject_id"] != "MYOLEG_VP_001"]
    elif missing == "seed":
        rows = [r for r in rows if not (
            r["subject_id"] == "MYOLEG_VP_001" and r["method"] == "RANDOM" and r["seed"] == 2
        )]
    else:
        rows = [r for r in rows if r["method"] != "RANDOM"]
    per_subject, summary = summaries(rows)
    conclusion = report._winner_text(per_subject, summary, protocol)
    assert "暂不宣布队列赢家" in conclusion
    assert "逐对区间支持此排序" not in conclusion


def test_complete_exact_tie_does_not_claim_unique_winner():
    rows, protocol = complete_comparison(tied=True)
    per_subject, summary = summaries(rows)
    conclusion = report._winner_text(per_subject, summary, protocol)
    assert "不能宣称唯一赢家" in conclusion
    assert "逐对区间支持此排序" not in conclusion
    pairs = report.pairwise_comparisons(per_subject)
    assert set(pairs.paired_subjects) == {24}
    np.testing.assert_allclose(pairs.mean_a_minus_b, 0., atol=report.NUMERICAL_TOLERANCE)
    assert pairs.ci_includes_zero.all()


def test_complete_consistent_advantage_is_distinguished_from_incomplete_or_tied():
    rows, protocol = complete_comparison()
    per_subject, summary = summaries(rows)
    conclusion = report._winner_text(per_subject, summary, protocol)
    assert "经验排序首位为 MODEL_INFORMED_BO_EI" in conclusion
    assert "逐对区间支持此排序" in conclusion
    assert "不能宣称唯一赢家" not in conclusion
    assert "仍不构成held-out确认" in conclusion


@pytest.mark.parametrize(
    "lo,hi,includes_zero",
    [(-1e-16, -1e-16, True), (1e-16, 1e-16, True),
     (-3e-12, -2e-12, False), (2e-12, 3e-12, False), (np.nan, np.nan, True)],
)
def test_confidence_interval_uses_numerical_not_literal_zero(lo, hi, includes_zero):
    assert report.interval_includes_numerical_zero(lo, hi) is includes_zero


def test_sub_tolerance_difference_is_not_a_unique_advantage():
    rows, protocol = complete_comparison(tied=True)
    for row in rows:
        if row["method"] == "RANDOM":
            row["selection_loss"] = row["true_E3"] = .8 + 2e-13
            row["improvement_pct"] = 100 * (1 - row["selection_loss"])
    per_subject, summary = summaries(rows)
    assert "不能宣称唯一赢家" in report._winner_text(per_subject, summary, protocol)
    assert report.pairwise_comparisons(per_subject).ci_includes_zero.all()


@pytest.fixture
def completed_output(tmp_path, monkeypatch):
    output = tmp_path / "completed"
    output.mkdir()
    protocol = dict(
        subjects=["MYOLEG_VP_001", "MYOLEG_NATIVE_P0"], families=["BETA_TIMING"],
        methods=["MODEL_INFORMED_BO_EI", "RANDOM"], budgets=[1, 4],
        primary_budget=4, primary_noise_std=0., noise_subjects="native",
        noise_levels=[.01], seeds={"random": [0, 1], "paired_noise": [7, 9]},
        source_sha256={"lower_limb_sim\\myoleg_benchmark\\report.py": "f" * 64},
    )
    rows = []
    for subject in protocol["subjects"]:
        for budget in (1, 4):
            rows.append(result_row(subject, budget=budget))
            rows.extend(result_row(subject, method="RANDOM", seed=seed, budget=budget) for seed in (0, 1))
            if subject == "MYOLEG_NATIVE_P0":
                for method in protocol["methods"]:
                    rows.extend(result_row(subject, method=method, seed=seed, noise_std=.01, budget=budget) for seed in (7, 9))
    (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    # Contents need not be loaded by the reporter; only the input identity is recorded.
    (output / "trial_history.csv").write_text("candidate_id,trial\nexecuted:1,1\n", encoding="utf-8")
    completion = dict(
        completed=True, result_rows=len(rows),
        protocol_sha256=hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
    )
    (output / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    monkeypatch.setattr(report, "_plots", lambda *args: None)
    return output, protocol, rows


def test_expected_identities_match_declared_native_only_noise_schedule(completed_output):
    _, protocol, rows = completed_output
    manual_schedule = {tuple(row[key] for key in report.KEYS) for row in rows}
    assert report.expected_result_identities(protocol) == manual_schedule
    assert len(manual_schedule) == 20
    assert not any(subject == "MYOLEG_VP_001" and noise > 0
                   for subject, _, noise, _, _, _ in manual_schedule)


def test_expected_identities_include_development_noise_when_declared(completed_output):
    _, protocol, rows = completed_output
    protocol["noise_subjects"] = "all"
    noisy_native = [row for row in rows if row["subject_id"] == "MYOLEG_NATIVE_P0" and row["noise_std"] > 0]
    rows.extend(dict(row, subject_id="MYOLEG_VP_001") for row in noisy_native)
    expected = {tuple(row[key] for key in report.KEYS) for row in rows}
    assert report.expected_result_identities(protocol) == expected
    assert len(expected) == 28


@pytest.mark.parametrize("completion_state", ["missing", "false", "wrong_protocol"])
def test_report_requires_completed_matching_protocol(completed_output, completion_state):
    output, _, _ = completed_output
    path = output / "completion.json"
    if completion_state == "missing":
        path.unlink()
    else:
        completion = json.loads(path.read_text(encoding="utf-8"))
        if completion_state == "false":
            completion["completed"] = False
        else:
            completion["protocol_sha256"] = "0" * 64
        path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="BENCHMARK_NOT_COMPLETE|COMPLETION_PROTOCOL_SHA_MISMATCH"):
        report.generate_report(output)
    assert not (output / "REPORT.md").exists()
    assert not (output / "analysis_provenance.json").exists()


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_seed_same_count"])
def test_report_requires_exact_result_identities(completed_output, change):
    output, _, rows = completed_output
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(result_row("MYOLEG_VP_001", seed=99))
    else:
        rows[-1]["seed"] = 999
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    with pytest.raises(ValueError, match="RESULT_IDENTITY_COVERAGE_MISMATCH"):
        report.generate_report(output)
    assert not (output / "REPORT.md").exists()


def test_report_rejects_completion_row_count_mismatch(completed_output):
    output, _, _ = completed_output
    path = output / "completion.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    completion["result_rows"] += 1
    path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETION_RESULT_ROW_COUNT_MISMATCH"):
        report.generate_report(output)


def test_report_records_revision_provenance_without_changing_run_inputs(completed_output):
    output, protocol, _ = completed_output
    input_paths = [output / name for name in ("protocol.json", "completion.json", "results.csv", "trial_history.csv")]
    original = {path.name: path.read_bytes() for path in input_paths}
    report.generate_report(output)
    provenance = json.loads((output / "analysis_provenance.json").read_text(encoding="utf-8"))
    assert provenance["report_sha256"] == hashlib.sha256(Path(report.__file__).read_bytes()).hexdigest()
    assert provenance["protocol_report_sha256"] == protocol["source_sha256"]["lower_limb_sim\\myoleg_benchmark\\report.py"]
    assert provenance["report_sha256"] != provenance["protocol_report_sha256"]
    for filename, key in (
        ("protocol.json", "protocol_sha256"), ("completion.json", "completion_sha256"),
        ("results.csv", "results_sha256"), ("trial_history.csv", "trial_history_sha256"),
    ):
        assert provenance[key] == hashlib.sha256(original[filename]).hexdigest()
        assert (output / filename).read_bytes() == original[filename]
    assert provenance["numerical_tolerance"] == 1e-12
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "K=4" in text and "K=8 是次级预算" in text
    assert "词典序来自运行前" in text
    assert "不是实际效果阈值" in text
