"""Causal-prefix and schema checks for the gray-box held-out evaluator."""

import numpy as np

from lower_limb_sim.myoleg_benchmark import experiment
from lower_limb_sim.myoleg_benchmark import graybox_cv
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain


class _Backend:
    def __init__(self):
        self.calls = []

    def requested(self, point):
        self.calls.append(point.candidate_id)
        return np.tile(np.array([10.0, 5.0]), (len(point.time_s), 1))


def test_graybox_fit_reports_declared_boundary(monkeypatch):
    source = native_domain()
    domain = experiment.Domain(source, "BETA_TIMING", grid=(0.0,), shifts=(0.0,))
    point = domain.reference
    tau = np.tile(np.array([10.0, 5.0]), (len(point.time_s), 1))

    class Result:
        estimated_parameters = {
            "mass_scale": 1.6,
            "k_hip_nm_per_rad": 10.0,
            "k_knee_nm_per_rad": 10.0,
            "b_hip_nm_s_per_rad": 1.0,
            "b_knee_nm_s_per_rad": 1.0,
        }
        optimizer_success = True
        cost = 1.0
        valid_training_samples = len(point.time_s) * 2
        optimizer_message = "ok"
        residual_statistics = {"torque_rmse_combined_nm": 2.0}

    monkeypatch.setattr(graybox_cv, "estimate_subject_parameters", lambda *args, **kwargs: Result())
    fit = graybox_cv.fit_graybox([(point, tau)])
    assert fit.success and fit.status == "OK"
    assert fit.boundary_hits == ("mass_scale",)
    assert fit.train_samples == len(point.time_s) * 2


def test_evaluate_prefix_scores_only_candidates_outside_prefix(monkeypatch):
    source = native_domain()
    domain = experiment.Domain(source, "BETA_TIMING", grid=(0.0, 0.03), shifts=(0.0, 0.05))
    backend = _Backend()
    prefix = [domain.reference.candidate_id, next(p.candidate_id for p in domain if p != domain.reference)]
    seen = {}

    def fake_fit(traces, **kwargs):
        seen["train_ids"] = [point.candidate_id for point, _ in traces]
        return graybox_cv.FitResult(
            parameters={name: 1.0 for name in graybox_cv.PARAMETER_NAMES},
            success=True, status="OK", cost=0.0, train_torque_rmse_nm=0.0,
            train_samples=1, boundary_hits=(), optimizer_message="ok",
        )

    monkeypatch.setattr(graybox_cv, "fit_graybox", fake_fit)
    monkeypatch.setattr(
        graybox_cv,
        "predict_joint_torque",
        lambda frame, template, params, length: (
            np.full(len(frame), 10.0), np.full(len(frame), 5.0)
        ),
    )
    predictions, summary = graybox_cv.evaluate_prefix(
        domain, backend, subject_id="synthetic", family="BETA_TIMING",
        method="RANDOM", seed=0, noise_std=0.0,
        prefix_candidate_ids=prefix, budget=2,
    )
    assert seen["train_ids"] == prefix
    assert not set(prefix).intersection(predictions["candidate_id"])
    assert len(predictions) == len(domain) - len(prefix)
    assert len(backend.calls) == len(set(backend.calls)) == len(domain)
    assert summary["prefix_size"] == 2
