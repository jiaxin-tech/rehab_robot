from pathlib import Path
import shutil
import uuid

import pandas as pd

from lower_limb_sim.myoleg_benchmark.candidate_space_diagnostic import (
    DEFAULT_LANDSCAPE,
    _domain_geometry,
    _response_diagnostic,
    run_diagnostic,
)


def test_native_domain_geometry_matches_frozen_v1_candidate_counts():
    beta = _domain_geometry("BETA_TIMING")
    key = _domain_geometry("KEY_POSTURE_TIMING")
    assert beta["candidate_count"] == 324
    assert key["candidate_count"] == 387
    assert beta["parameter_pca"]["rank"] == 3
    assert key["parameter_pca"]["rank"] == 3
    assert beta["trajectory_pca"]["rank_95"] <= 10
    assert key["trajectory_pca"]["rank_95"] <= 10


def test_frozen_landscape_shows_shared_ranking_and_oracle():
    landscape = pd.read_csv(DEFAULT_LANDSCAPE)
    beta, agreement, _ = _response_diagnostic(landscape, "BETA_TIMING")
    key, _, _ = _response_diagnostic(landscape, "KEY_POSTURE_TIMING")
    assert beta["subject_count"] == 24
    assert key["subject_count"] == 24
    assert beta["oracle_unique_count"] == 1
    assert key["oracle_unique_count"] == 1
    assert beta["rank_spearman_median"] > 0.99
    assert key["rank_spearman_median"] > 0.99
    assert beta["cross_subject_to_within_candidate_sd_ratio"] < 0.1
    assert len(agreement) == 24 * 23 // 2


def test_diagnostic_writes_scope_and_summary_outputs():
    output = Path(".cache") / f"candidate-diagnostic-{uuid.uuid4().hex}"
    try:
        payload = run_diagnostic(output_dir=output)
        assert payload["response"]["BETA_TIMING"]["oracle_unique_count"] == 1
        assert (output / "REPORT.md").is_file()
        assert (output / "diagnostic_summary.json").is_file()
        assert (output / "ranking_agreement.csv").is_file()
        assert (output / "oracle_summary.csv").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)
