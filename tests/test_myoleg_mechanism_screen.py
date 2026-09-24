import numpy as np
import pandas as pd

from lower_limb_sim.myoleg_benchmark.mechanism_screen import summarize


def test_mechanism_summary_reports_oracles_and_rank_agreement():
    rows = pd.DataFrame([
        {"subject_id": "A", "candidate_id": "DURATION_COORDINATION_V2:0", "E3": 1.0, "feasible": True},
        {"subject_id": "A", "candidate_id": "DURATION_COORDINATION_V2:1", "E3": .9, "feasible": True},
        {"subject_id": "B", "candidate_id": "DURATION_COORDINATION_V2:0", "E3": 1.0, "feasible": True},
        {"subject_id": "B", "candidate_id": "DURATION_COORDINATION_V2:1", "E3": 1.1, "feasible": True},
    ])
    summary, metrics = summarize(rows)
    assert len(summary) == 2
    assert metrics["oracle_unique_count"] == 2
    assert np.isfinite(metrics["rank_spearman_median"])
