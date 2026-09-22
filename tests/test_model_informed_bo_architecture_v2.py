"""Offline integration contracts, not a scientific personalization comparison."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from lower_limb_sim.mechanical_endpoints import (
    E0, E2, MechanicalReferenceContext, BranchRMSReference, branch_rms_components,
    branch_balanced_reference_normalized_rms, full_cycle_dual_joint_rms,
)
from personalization.candidates import V3CandidateDomain
from personalization.identification import TimeSeriesIdentificationPayload
from personalization.observations import EpisodeObservation
from personalization.models.base import Prediction
from personalization.models.physics_graybox import PhysicsSubjectModel, FullDynamicsGrayBoxEndpointAdapter
from personalization.models.time_series_graybox import TimeSeriesFiveParameterGrayBoxAdapter, BranchBalancedE2GrayBoxEndpointAdapter
from personalization.models.residual_gp import PhysicsInformedResidualModel
from personalization.models import time_series_graybox as ts_module
from personalization.selectors.bo import ExpectedImprovementSelector, LowerConfidenceBoundSelector, expected_improvement
from personalization.offline_time_series import ModelConsistentTimeSeriesEnvironment
from personalization.rom_gated_v2.development import determine_synthetic_rom, make_offline_rom_development_cases
from personalization.rom_gated_v2.reference import SubjectSpecificV3CandidateDomain
from personalization.integrated_v2 import OfflineBOConfiguration, run_offline_configuration
from personalization.sequential import run_sequential_personalization

ROOT = Path(__file__).resolve().parents[1]
THETA = dict(mass_scale=1.2, k_hip_nm_per_rad=18., k_knee_nm_per_rad=14.,
             b_hip_nm_s_per_rad=2.2, b_knee_nm_s_per_rad=1.7)


@pytest.fixture(scope="module")
def domain():
    _, profile = determine_synthetic_rom(make_offline_rom_development_cases()[0])
    return SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(profile)


@pytest.fixture(scope="module")
def template():
    return baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS["baseline"])


def environment(domain, template, endpoint="E0"):
    return ModelConsistentTimeSeriesEnvironment(domain, baseline_template=template,
                                               theta=THETA, L1=.42, L2=.30, endpoint=endpoint)


def adapter(domain, template, endpoint="E0"):
    cls = TimeSeriesFiveParameterGrayBoxAdapter if endpoint == "E0" else BranchBalancedE2GrayBoxEndpointAdapter
    return cls(domain, baseline_template=template, L1=.42, L2=.30)


def scalar(candidate, value, index=1, valid=True):
    return EpisodeObservation(f"e{index}", index, candidate.candidate_id,
        *candidate.beta, E0.endpoint_name, value if valid else None, E0.unit,
        0., valid, invalid_reason=None if valid else "injected")


class Posterior:
    def __init__(self, predictions):
        self.predictions = predictions
    def predict(self, candidate):
        return self.predictions[candidate.candidate_id]


def small_posterior():
    domain = V3CandidateDomain.regular_grid([0., .03])
    predictions = {c.candidate_id: Prediction(100., 0., False, {}) for c in domain}
    return domain, predictions


def test_ei_formula_and_limits():
    assert expected_improvement(1., 2., 1.) == pytest.approx(2 / np.sqrt(2*np.pi))
    assert expected_improvement(.5, 0., 1.) == .5
    assert expected_improvement(2., 1e-14, 1.) == 0.
    assert expected_improvement(.5, 0., 1., .1) == pytest.approx(.4)
    assert expected_improvement(.5, .2, 1.) > expected_improvement(.8, .2, 1.)
    assert expected_improvement(1., .8, 1.) > expected_improvement(1., .2, 1.)


def test_ei_measured_incumbent_metadata_and_no_repeats():
    d, p = small_posterior()
    history = [scalar(d.reference, 5.)]
    available = [c for c in d if c != d.reference]
    p[d.reference.candidate_id] = Prediction(-1000., 100., True, {})
    p[available[0].candidate_id] = Prediction(3., 0., True, {})
    p[available[1].candidate_id] = Prediction(-1000., 100., False, {})
    selected = ExpectedImprovementSelector().select_next(history, d, Posterior(p))
    assert selected.candidate == available[0]
    assert selected.acquisition_value == 2.
    assert selected.metadata["incumbent_measured_best"] == 5.
    assert selected.metadata["trial_index"] == 2
    assert selected.metadata["xi"] == 0
    assert selected.metadata["beta"] == list(available[0].beta)


def test_ei_no_valid_incumbent():
    d, p = small_posterior()
    for history in ([], [scalar(d.reference, None, valid=False)]):
        with pytest.raises(RuntimeError, match="EI_REQUIRES_VALID_INCUMBENT"):
            ExpectedImprovementSelector().select_next(history, d, Posterior(p))


@pytest.mark.parametrize("selector", [ExpectedImprovementSelector(), LowerConfidenceBoundSelector(name="test")])
def test_prediction_validity_and_exhaustion(selector):
    d, p = small_posterior()
    history = [scalar(d.reference, 1.)]
    with pytest.raises(RuntimeError, match="NO_VALID_UNEXECUTED_PREDICTION"):
        selector.select_next(history, d, Posterior(p))
    c = next(c for c in d if c != d.reference)
    for prediction in (Prediction(float("nan"), 1., True, {}),
                       Prediction(0., -1., True, {}), Prediction(0., float("inf"), True, {})):
        p[c.candidate_id] = prediction
        with pytest.raises(RuntimeError, match="NO_VALID_UNEXECUTED_PREDICTION"):
            selector.select_next(history, d, Posterior(p))


def test_ei_and_lcb_can_choose_differently():
    d, p = small_posterior()
    a, b = [c for c in d if c != d.reference][:2]
    p[a.candidate_id] = Prediction(-.2, 0., True, {})
    p[b.candidate_id] = Prediction(.65, .6, True, {})
    history = [scalar(d.reference, 0.)]
    assert ExpectedImprovementSelector().select_next(history, d, Posterior(p)).candidate == a
    assert LowerConfidenceBoundSelector(name="LCB").select_next(history, d, Posterior(p)).candidate == b


def test_payload_roundtrip_and_explicit_mapping(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    restored = EpisodeObservation(**obs.as_dict())
    assert restored.identification_payload == obs.identification_payload
    with pytest.raises(ValueError, match="EXPLICIT_PROJECT_FORCE_MAPPING"):
        replace(obs.identification_payload, force_mapping="robot_X_is_task_X")
    with pytest.raises(ValueError, match="TIME_MUST_INCREASE"):
        replace(obs.identification_payload, time_s=[0.] * len(obs.identification_payload.time_s))
    assert isinstance(obs.identification_payload.q, tuple)


def test_known_theta_recovery_and_diagnostics(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    physics = adapter(domain, template)
    physics.fit([obs])
    # Exact model-consistent, full-rank noiseless fixture; numerical optimizer tolerance.
    assert np.allclose(list(physics.theta_hat.values()), list(THETA.values()), atol=2e-4, rtol=2e-4)
    m = physics.metadata()
    assert m["fit_valid_episode_count"] == 1
    assert m["valid_sample_count"] == len(obs.identification_payload.time_s)
    diag = m["identification_diagnostics"]
    assert diag["optimizer_success"] and len(diag["jacobian_singular_values"]) == 5
    assert np.asarray(diag["parameter_covariance"]).shape == (5, 5)
    assert m["parameter_bounds_status"] == "WITHIN_BOUNDS"
    assert m["jacobian_condition_number"] > 1


def test_two_episodes_pointwise_union_and_refinement(domain, template, monkeypatch):
    env = environment(domain, template)
    first = env.evaluate(domain.reference, 1)
    second = env.evaluate(next(c for c in domain if c.beta == (.03, -.03)), 2)
    force = np.asarray(first.identification_payload.planar_force_n).copy()
    force[:, 0] += .02 * np.sin(np.linspace(0, 6, len(force)))
    first = replace(first, identification_payload=replace(first.identification_payload, planar_force_n=force))
    captured = []
    original = ts_module.estimator.estimate_subject_parameters
    def spy(frame, *args, **kwargs):
        captured.append(frame.copy())
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(ts_module.estimator, "estimate_subject_parameters", spy)
    physics = adapter(domain, template)
    physics.fit([first]); before = physics.theta_hat
    physics.fit([first, second]); after = physics.theta_hat
    expected = pd.concat([first.identification_payload.to_frame(), second.identification_payload.to_frame()], ignore_index=True)
    pd.testing.assert_frame_equal(captured[-1], expected)
    assert len(captured[-1]) == 802 and captured[-1].iloc[401].time_s == 0.
    assert not np.allclose(list(before.values()), list(after.values()), atol=1e-8, rtol=0)
    assert physics.metadata()["fit_update_count"] == 2
    assert physics.metadata()["fit_valid_episode_count"] == 2
    assert physics.metadata()["valid_sample_count"] == 802


def test_invalid_missing_payload_and_invalid_samples_excluded(domain, template):
    env = environment(domain, template)
    a = env.evaluate(domain.reference, 1)
    b = env.evaluate(next(c for c in domain if c != domain.reference), 2)
    physics = adapter(domain, template)
    physics.fit([a])
    invalid = replace(b, valid=False, endpoint_value=None, invalid_reason="bad")
    payload_invalid = replace(b, identification_payload=replace(b.identification_payload, valid=False, invalid_reason="bad mapping"))
    for extra in (invalid, payload_invalid, replace(b, identification_payload=None)):
        physics.fit([a, extra])
        assert physics.metadata()["fit_update_count"] == 1
    mask = list(b.identification_payload.sample_valid); mask[0] = False
    b = replace(b, identification_payload=replace(b.identification_payload, sample_valid=mask))
    physics.fit([a, b])
    assert physics.metadata()["valid_sample_count"] == 801


@pytest.mark.parametrize("change", [dict(rom_fingerprint="other"), dict(rom_version=9),
    dict(candidate_id="other"), dict(episode_id="other"), dict(beta_flex=.1), dict(L2=.2)])
def test_payload_context_mismatches_fail(domain, template, change):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    obs = replace(obs, identification_payload=replace(obs.identification_payload, **change))
    with pytest.raises(ValueError, match="MISMATCH"):
        adapter(domain, template).fit([obs])


def test_failed_estimator_never_commits_update(domain, template, monkeypatch):
    env = environment(domain, template)
    a = env.evaluate(domain.reference, 1)
    b = env.evaluate(next(c for c in domain if c != domain.reference), 2)
    physics = adapter(domain, template); physics.fit([a]); before = physics.theta_hat
    failed = replace(physics._estimate, optimizer_success=False, optimizer_message="injected failure")
    monkeypatch.setattr(ts_module.estimator, "estimate_subject_parameters", lambda *a, **kw: failed)
    with pytest.raises(RuntimeError, match="IDENTIFICATION_FAILED"):
        physics.fit([a, b])
    assert physics.theta_hat == before
    assert physics.metadata()["fit_update_count"] == 1
    assert not physics.metadata()["last_attempt_diagnostics"]["optimizer_success"]
    with pytest.raises(RuntimeError, match="SUCCESSFUL_TIMESERIES"):
        physics.predict_value(domain.reference)


def test_shared_e2_branch_math_and_no_cancellation():
    time = np.arange(6.)
    branches = np.array(["flexion"]*3 + ["extension"]*3)
    components = branch_rms_components([2]*3+[.5]*3, [1]*6, time, branches)
    context = MechanicalReferenceContext("leg", 1, "rom", "ref", (1.,))
    ref = BranchRMSReference(context, (1., 1., 1., 1.))
    assert components == (2., .5, 1., 1.)
    assert branch_balanced_reference_normalized_rms(components, ref, context=context) == 2.
    assert branch_balanced_reference_normalized_rms((2., 2., .1, .1), ref, context=context) == 2.
    assert branch_balanced_reference_normalized_rms(ref.components, ref, context=context) == 1.
    assert full_cycle_dual_joint_rms([3]*6, [4]*6, time) == 5.


@pytest.mark.parametrize("denominator", [(0, 1, 1, 1), (float("nan"), 1, 1, 1), (-1,1,1,1)])
def test_e2_invalid_reference(denominator):
    context = MechanicalReferenceContext("leg", 1, "rom", "ref", (1.,))
    with pytest.raises(ValueError, match="REFERENCE_COMPONENT"):
        branch_balanced_reference_normalized_rms((1,1,1,1), BranchRMSReference(context, denominator), context=context)


@pytest.mark.parametrize("change", [dict(rom_profile_id="other_leg"), dict(rom_fingerprint="other_rom"),
                                    dict(reference_version="v2"), dict(model_state=(2.,))])
def test_e2_cross_context_rejected(change):
    context = MechanicalReferenceContext("leg", 1, "rom", "ref", (1.,))
    with pytest.raises(ValueError, match="CONTEXT_MISMATCH"):
        branch_balanced_reference_normalized_rms((1,1,1,1), BranchRMSReference(context, (1,1,1,1)),
                                                context=replace(context, **change))


def test_e2_theta_refit_recomputes_reference_and_residuals(domain, template):
    env = environment(domain, template, "E2")
    a = env.evaluate(domain.reference, 1)
    b = env.evaluate(next(c for c in domain if c.beta == (.03, -.03)), 2)
    force = np.asarray(a.identification_payload.planar_force_n) * 1.01
    a = replace(a, identification_payload=replace(a.identification_payload, planar_force_n=force))
    physics = adapter(domain, template, "E2")
    model = PhysicsInformedResidualModel(PhysicsSubjectModel(physics))
    model.fit([a]); old_ref = physics.predicted_reference()
    model.fit([a, b]); new_ref = physics.predicted_reference()
    assert old_ref.context != new_ref.context
    assert not np.array_equal(old_ref.components, new_ref.components)
    assert physics.metadata()["reference_recomputation_count"] == 2
    assert physics.predict_value(domain.reference) == pytest.approx(1.)
    targets = [o.endpoint_value - physics.predict_value(domain.by_id(o.candidate_id)) for o in (a,b)]
    assert model.residual_gp._y == pytest.approx(targets, abs=1e-12)
    assert model.residual_gp._x.shape == (2,2)


@pytest.mark.parametrize("adapter_endpoint,observation_endpoint", [("E0", "E2"),("E2", "E0")])
def test_e0_e2_mismatch_rejected_before_estimation(domain, template, monkeypatch, adapter_endpoint, observation_endpoint):
    obs = environment(domain, template, observation_endpoint).evaluate(domain.reference, 1)
    def forbidden(*args, **kwargs):
        pytest.fail("must reject endpoint before estimator")
    monkeypatch.setattr(ts_module.estimator, "estimate_subject_parameters", forbidden)
    with pytest.raises(ValueError, match="ENDPOINT_IDENTITY_MISMATCH"):
        PhysicsInformedResidualModel(PhysicsSubjectModel(adapter(domain, template, adapter_endpoint))).fit([obs])


def test_endpoint_unit_mismatch(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    with pytest.raises(ValueError, match="ENDPOINT_IDENTITY_MISMATCH"):
        adapter(domain, template).fit([replace(obs, endpoint_unit="dimensionless")])


@pytest.mark.parametrize("endpoint", ["E0", "E2"])
def test_integrated_three_trial_causal_run(domain, template, endpoint):
    result = run_offline_configuration(environment(domain, template, endpoint), domain,
        configuration=OfflineBOConfiguration(endpoint=endpoint, budget=3), baseline_template=template, L1=.42, L2=.30)
    ledger = result.ledger.entries
    assert len({e.candidate.beta for e in ledger}) == 3
    assert ledger[0].candidate == domain.reference
    for i, entry in enumerate(ledger, 1):
        assert entry.physics_model_state_summary["fit_update_count"] == i
        assert entry.physics_model_state_summary["fit_valid_episode_count"] == i
        assert entry.physics_model_state_summary["valid_sample_count"] == 401*i
        assert entry.residual_model_state_summary["training_count"] == i
        assert entry.observation.endpoint_name == (E0 if endpoint == "E0" else E2).endpoint_name
        if i < 3:
            m = entry.next_selection_metadata
            assert m["acquisition_type"] == "EI" and m["trial_index"] == i+1
            assert m["candidate_id"] == ledger[i].candidate.candidate_id
            assert m["incumbent_measured_best"] == min(e.observation.endpoint_value for e in ledger[:i])
            assert np.isfinite([m["acquisition_value"],m["predictive_mean"],m["predictive_std"]]).all()
    assert result.best_observed_candidate is not None
    assert result.model_recommended_final_candidate is not None
    json.dumps(result.as_dict(), allow_nan=False)


def test_future_data_cannot_change_earlier_theta_or_selection(domain, template):
    class FutureChange(ModelConsistentTimeSeriesEnvironment):
        def evaluate(self, candidate, index):
            obs = super().evaluate(candidate, index)
            if index >= 3:
                obs = replace(obs, endpoint_value=obs.endpoint_value + 10.,
                              identification_payload=replace(obs.identification_payload,
                                  planar_force_n=np.asarray(obs.identification_payload.planar_force_n)*1.01))
            return obs
    normal = environment(domain, template)
    future = FutureChange(domain, baseline_template=template, theta=THETA, L1=.42, L2=.30)
    runs = [run_offline_configuration(e, domain, configuration=OfflineBOConfiguration(endpoint="E0", budget=3),
                                     baseline_template=template, L1=.42, L2=.30) for e in (normal, future)]
    for a,b in zip(runs[0].ledger.entries[:2],runs[1].ledger.entries[:2]):
        assert a.physics_model_state_summary == b.physics_model_state_summary
        assert a.next_selection_metadata == b.next_selection_metadata
    assert runs[0].ledger.executed_candidate_ids == runs[1].ledger.executed_candidate_ids
    assert runs[0].final_model_summary != runs[1].final_model_summary


def test_pure_bo_never_receives_identification_payload(domain, template, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure BO accessed estimator")
    monkeypatch.setattr(ts_module.estimator, "estimate_subject_parameters", forbidden)
    result = run_offline_configuration(environment(domain, template), domain,
        configuration=OfflineBOConfiguration(endpoint="E0", method="PURE_BO_EI", budget=3))
    assert all(e.observation.identification_payload is None for e in result.ledger.entries)
    assert all(not e.physics_model_state_summary for e in result.ledger.entries)
    assert result.final_model_summary["physics_access"] is False


@pytest.mark.parametrize("method,acquisition", [("MODEL_ONLY_GREEDY","MODEL_MEAN"), ("MODEL_INFORMED_BO_LCB","LCB")])
def test_same_timeseries_adapter_for_greedy_and_lcb(domain, template, method, acquisition):
    result = run_offline_configuration(environment(domain, template), domain,
        configuration=OfflineBOConfiguration(endpoint="E0", method=method, budget=2), baseline_template=template, L1=.42, L2=.30)
    first = result.ledger.entries[0]
    assert first.next_selection_metadata["acquisition_type"] == acquisition
    assert first.physics_model_state_summary["identification_mode"] == "TIME_SERIES_TORQUE_RESIDUALS"
    if acquisition == "LCB":
        assert first.next_selection_metadata["kappa"] == 1.5


def test_legacy_adapter_still_scalar(domain):
    legacy = FullDynamicsGrayBoxEndpointAdapter()
    canonical = V3CandidateDomain.from_frozen_artifact().reference
    value = legacy.predict_value(canonical)
    observation = EpisodeObservation("legacy", 1, canonical.candidate_id, *canonical.beta,
        legacy.endpoint_name, value, legacy.endpoint_unit, .05, True)
    legacy.fit([observation])
    assert np.isfinite(legacy.predict_value(canonical))


def test_timeseries_method_cannot_silently_use_scalar_adapter(domain):
    with pytest.raises(ValueError, match="TIMESERIES_ID_ADAPTER_REQUIRED"):
        run_sequential_personalization(None, domain, method="MODEL_INFORMED_BO_EI_TIMESERIES_ID",
            physics_model=PhysicsSubjectModel(FullDynamicsGrayBoxEndpointAdapter()))


def test_scalar_channel_does_not_estimate_theta(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    first, second = adapter(domain, template), adapter(domain, template)
    first.fit([obs])
    second.fit([replace(obs, endpoint_value=obs.endpoint_value + 1000.)])
    assert first.theta_hat == second.theta_hat
    model = PhysicsInformedResidualModel(PhysicsSubjectModel(second))
    model.fit([replace(obs, endpoint_value=obs.endpoint_value + 1000.)])
    assert model.residual_gp._y[0] == pytest.approx(1000., abs=1e-6)


def test_invalid_samples_serialization_keeps_missing_not_zero(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    force = np.asarray(obs.identification_payload.planar_force_n).copy()
    force[0] = np.nan
    modified = replace(obs, identification_payload=replace(obs.identification_payload, planar_force_n=force))
    serialized = json.loads(json.dumps(modified.as_dict(), allow_nan=False))
    assert serialized['identification_payload']['planar_force_n'][0] == [None, None]
    restored = EpisodeObservation(**serialized)
    physics = adapter(domain, template); physics.fit([restored])
    assert physics.metadata()['valid_sample_count'] == 400


def test_no_payload_cannot_fall_back_to_scalar_fitting(domain, template):
    obs = environment(domain, template).evaluate(domain.reference, 1)
    with pytest.raises(RuntimeError, match='TIMESERIES_ID_REQUIRES_VALID_IDENTIFICATION_EPISODE'):
        adapter(domain, template).fit([replace(obs, identification_payload=None)])


def test_greedy_and_residual_physics_share_identical_id_history(domain, template):
    env = environment(domain, template)
    history = [env.evaluate(domain.reference, 1), env.evaluate(next(c for c in domain if c != domain.reference), 2)]
    greedy = adapter(domain, template)
    informed = adapter(domain, template)
    greedy.fit(history)
    PhysicsInformedResidualModel(PhysicsSubjectModel(informed)).fit(history)
    assert greedy.theta_hat == informed.theta_hat
    assert greedy.metadata()['valid_samples_by_episode'] == informed.metadata()['valid_samples_by_episode']


def test_strict_offline_endpoint_contract_for_pure_bo(domain, template):
    with pytest.raises(ValueError, match='ENDPOINT_IDENTITY_MISMATCH'):
        run_offline_configuration(environment(domain, template, 'E0'), domain,
            configuration=OfflineBOConfiguration(endpoint='E2', method='PURE_BO_EI', budget=2))
    with pytest.raises(ValueError, match='EXPLICIT_OFFLINE_ENVIRONMENT_REQUIRED'):
        run_offline_configuration(object(), domain, configuration=OfflineBOConfiguration(endpoint='E0'))


def test_best_of_multiple_valid_measurements_not_last_observation():
    d, p = small_posterior()
    other, candidate = [c for c in d if c != d.reference][:2]
    history = [scalar(d.reference, 1.), scalar(other, 5., index=2)]
    p[candidate.candidate_id] = Prediction(0., 0., True, {})
    result = ExpectedImprovementSelector().select_next(history, d, Posterior(p))
    assert result.metadata['incumbent_measured_best'] == 1.
    assert result.acquisition_value == 1.
