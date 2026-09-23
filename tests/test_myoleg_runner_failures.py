"""Infrastructure failures stay visible without discarding other completed jobs."""
import json
from concurrent.futures import Future

import pytest

from lower_limb_sim.myoleg_benchmark import run as runner


@pytest.mark.parametrize("workers", [1, 2])
def test_runner_records_job_error_and_retains_other_results(tmp_path, monkeypatch, workers):
    output = tmp_path / "run"
    calls = []

    def fake_job(job):
        calls.append(job[2])
        if job[2] == 0:
            raise PermissionError("synthetic cache publication failure")
        return dict(
            results=[dict(subject_id=runner.NOMINAL, family="BETA_TIMING")],
            histories=[dict(trial=1)], diagnostics=[], provenance={},
            domain_size=1, kinematic_rejections=[],
            cache_diagnostics=dict(fresh_simulations=0, disk_hits=1, max_decomposition_residual_nm=0.),
        )

    class ImmediatePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, job):
            future = Future()
            try:
                future.set_result(function(job))
            except Exception as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(runner, "_job", fake_job)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediatePool)
    with pytest.raises(RuntimeError, match="INCOMPLETE_BENCHMARK"):
        runner.main([
            "--subjects", "native", "--families", "BETA_TIMING", "--methods", "REFERENCE",
            "--budgets", "1", "--random-seeds", "1", "--noise-seeds", "1",
            "--noise-levels", ".01", "--workers", str(workers), "--output-dir", str(output),
        ])
    assert calls == [0., .01]
    completion = json.loads((output / "completion.json").read_text())
    assert completion["completed"] is False
    assert completion["failed_jobs"] == 1
    assert completion["result_rows"] == completion["trial_rows"] == 1
    errors = json.loads((output / "job_errors.json").read_text())
    assert errors[0]["error_type"] == "PermissionError"
    assert errors[0]["subject_id"] == runner.NOMINAL
    assert errors[0]["noise_std"] == 0.
    assert "synthetic cache publication failure" in errors[0]["traceback"]
    assert (output / "results.csv").exists()
