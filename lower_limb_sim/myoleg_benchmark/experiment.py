"""Causal, paired-noise E3 comparisons using only requested trajectory responses.

The learner never receives the simulator, cache, candidate outcomes, or oracle.
Valid but load-infeasible observations remain useful identification/GP data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import time

import numpy as np

from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.e3_candidate_comparison.run import e3, key_posture
from lower_limb_sim.e3_low_budget.run import Adapter, Point, E3
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.mechanical_endpoints import branch_rms_components
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from lower_limb_sim.trajectory_sensitivity.study import trajectory
from personalization.identification import PROJECT_FORCE_MAPPING, TimeSeriesIdentificationPayload
from personalization.models.residual_gp import ResidualGaussianProcess
from personalization.observations import EpisodeObservation
from personalization.selectors.bo import expected_improvement

METHODS = (
    "REFERENCE", "RANDOM", "SPACE_FILLING", "PHYSICS_GREEDY",
    "RESIDUAL_GP_GREEDY", "PURE_BO_EI", "MODEL_INFORMED_BO_EI",
)
FAMILIES = ("BETA_TIMING", "KEY_POSTURE_TIMING")
PHYSICS_METHODS = {"PHYSICS_GREEDY", "RESIDUAL_GP_GREEDY", "MODEL_INFORMED_BO_EI"}


class Domain:
    """Construct candidates from kinematics alone, without reading outcome tables."""

    def __init__(self, source, family, *, grid=None, shifts=None):
        if family not in FAMILIES:
            raise ValueError("UNKNOWN_TRAJECTORY_FAMILY")
        self.profile, self.subject_reference = source.profile, source.subject_reference
        self.family = family
        self.points, self.rejections = [], []
        r = self.subject_reference
        grid = np.round(np.arange(-4, 5) * .03, 8) if grid is None else grid
        shifts = (-.10, -.05, 0., .05, .10) if shifts is None else shifts
        maker = key_posture if family == "KEY_POSTURE_TIMING" else trajectory
        _, reference_kin = maker(r, r.as_v3_mapping(), 0., 0., 0.)
        for a, b, shift in itertools.product(grid, grid, shifts):
            try:
                curve, kin = maker(r, r.as_v3_mapping(), float(a), float(b), float(shift))
                for joint in ("hip", "knee"):
                    for kind, limit in (("speed", 1.5), ("accel", 2.0)):
                        key = f"{joint}_{kind}_peak_deg_s" + ("2" if kind == "accel" else "")
                        if kin[key] > limit * reference_kin[key] + 1e-10:
                            raise ValueError("KINEMATIC_COMPARISON_LIMIT")
                from types import SimpleNamespace
                index = len(self.points)
                self.points.append(Point(
                    f"{family}:{index}", index, float(a), float(b), float(shift),
                    SimpleNamespace(q=curve["q"], dq=curve["dq"], ddq=curve["ddq"]),
                    curve["time_s"],
                ))
            except ValueError as error:
                self.rejections.append(dict(parameters=[float(a), float(b), float(shift)], reason=str(error)))
        self.lookup = {p.candidate_id: p for p in self.points}
        self.reference = next(p for p in self.points if (*p.beta, p.share_shift) == (0., 0., 0.))
        self.features = np.array([p.features for p in self.points])

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def by_id(self, key):
        return self.lookup[key]


def response_metrics(tau, point, phases, reference_components, reference_peaks):
    tau = np.asarray(tau, dtype=float)
    if tau.shape != (len(point.time_s), 2) or not np.isfinite(tau).all():
        raise ValueError("NONFINITE_OR_MALFORMED_TORQUE")
    components = np.asarray(branch_rms_components(*tau.T, point.time_s, phases))
    peaks = np.max(np.abs(tau), axis=0)
    return dict(E3=e3(components, reference_components),
                E2=float(np.max(components / reference_components)),
                peak_ratio=float(np.max(peaks / reference_peaks)))


def is_feasible(metrics, tier):
    return bool(metrics["E2"] <= 1 + tier + 1e-12 and metrics["peak_ratio"] <= 1.1 + 1e-12)


def paired_noise(tau, *, subject_id, family, candidate_id, seed, relative_std, joint_scale):
    """Same noisy trace for the same subject/candidate/seed, independent of method/order."""
    identity = f"{subject_id}|{family}|{candidate_id}|{seed}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
    return np.asarray(tau) + rng.normal(size=np.shape(tau)) * np.asarray(joint_scale) * relative_std


@dataclass(frozen=True)
class Measurement:
    observation: EpisodeObservation
    feasible: bool
    E2: float | None
    peak_ratio: float | None


class Environment:
    """Responds to a requested candidate; carries hidden truth solely in backend."""

    def __init__(self, domain, backend, *, subject_id, noise_std=0., seed=0, tier=.01, physics=False):
        self.domain, self.backend, self.subject_id = domain, backend, subject_id
        self.noise_std, self.seed, self.tier, self.physics = noise_std, seed, tier, physics
        self.calls = []
        self.reference_components = self.reference_peaks = self.joint_scale = None

    def evaluate(self, point, index):
        if not self.calls and point.candidate_id != self.domain.reference.candidate_id:
            raise ValueError("REFERENCE_FIRST")
        if point.candidate_id in self.calls:
            raise ValueError("DUPLICATE_TRIAL")
        self.calls.append(point.candidate_id)
        r, profile = self.domain.subject_reference, self.domain.profile
        episode = f"{self.subject_id}:{self.domain.family}:{self.seed}:{index}"
        try:
            tau = np.asarray(self.backend.requested(point))
            if tau.shape != (len(point.time_s), 2) or not np.isfinite(tau).all():
                raise ValueError("INVALID_SIMULATED_RESPONSE")
            if self.joint_scale is None:
                # Noise magnitude is a simulator condition, never a learner feature.
                self.joint_scale = np.sqrt(np.mean(tau ** 2, axis=0))
            measured = paired_noise(tau, subject_id=self.subject_id, family=self.domain.family,
                                    candidate_id=point.candidate_id, seed=self.seed,
                                    relative_std=self.noise_std, joint_scale=self.joint_scale)
            if self.reference_components is None:
                self.reference_components = np.array(branch_rms_components(*measured.T, point.time_s, r.phases))
                self.reference_peaks = np.max(np.abs(measured), axis=0)
            metrics = response_metrics(measured, point, r.phases, self.reference_components, self.reference_peaks)
            payload = None
            if self.physics:
                curve = point.trajectory
                jacobian_t = leg_jacobian(curve.q[:, 0], curve.q[:, 1], .42, .30).swapaxes(-1, -2)
                force = np.linalg.solve(jacobian_t, measured[..., None])[..., 0]
                payload = TimeSeriesIdentificationPayload(
                    episode_id=episode, candidate_id=point.candidate_id,
                    rom_profile_id=profile.profile_id, rom_version=profile.version,
                    rom_fingerprint=profile.fingerprint, reference_version=r.reference_version,
                    beta_flex=point.beta_flex, beta_extend=point.beta_extend,
                    time_s=point.time_s, q=curve.q, dq=curve.dq, ddq=curve.ddq,
                    planar_force_n=force, sample_valid=np.ones(len(point.time_s), dtype=bool),
                    L1=.42, L2=.30, force_mapping=PROJECT_FORCE_MAPPING,
                    mapping_provenance="Requested MyoLeg prescribed-state torque plus declared synthetic noise; algebraic planar equivalent force, not cuff measurement",
                    classification="MYOLEG_DEVELOPMENT_BENCHMARK_V1",
                )
            # Delta-method approximation for independent torque-sample noise.
            # Reference denominator errors are shared across trials; this scalar
            # approximation does not claim a full correlated observation model.
            uncertainty = float(self.noise_std * np.sqrt(2 / len(point.time_s)))
            obs = EpisodeObservation(episode, index, point.candidate_id, *point.beta,
                                     E3.endpoint_name, metrics["E3"], E3.unit, uncertainty,
                                     True, identification_payload=payload)
            return Measurement(obs, is_feasible(metrics, self.tier), metrics["E2"], metrics["peak_ratio"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            obs = EpisodeObservation(episode, index, point.candidate_id, *point.beta,
                                     E3.endpoint_name, None, E3.unit, None, False,
                                     invalid_reason=f"{type(error).__name__}: {error}")
            return Measurement(obs, False, None, None)


class GaussianProcess3D:
    def __init__(self):
        self.kernel = ResidualGaussianProcess()
        self.x = np.empty((0, 3))
        self.chol = self.alpha = None

    def fit(self, x, y, noise):
        self.x = np.asarray(x, dtype=float).reshape(-1, 3)
        y, noise = np.asarray(y, dtype=float), np.asarray(noise, dtype=float)
        if len(self.x) != len(y) or noise.shape != y.shape or np.any(noise < 0):
            raise ValueError("INVALID_GP_OBSERVATIONS")
        if not all(np.isfinite(a).all() for a in (self.x, y, noise)):
            raise ValueError("NONFINITE_GP_OBSERVATIONS")
        if not len(y):
            self.chol = self.alpha = None
            return
        covariance = self.kernel._kernel(self.x, self.x) + np.diag(np.maximum(noise, 1e-6) ** 2)
        extra = self.kernel.jitter
        for _ in range(6):
            try:
                self.chol = np.linalg.cholesky(covariance + extra * np.eye(len(y)))
                break
            except np.linalg.LinAlgError:
                extra *= 10
        else:
            raise RuntimeError("GP_FACTORIZATION_FAILED")
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, y))

    def predict(self, features):
        features = np.asarray(features)
        if self.alpha is None:
            return np.zeros(len(features)), np.full(len(features), self.kernel.signal_std)
        cross = self.kernel._kernel(self.x, features)
        projected = np.linalg.solve(self.chol, cross)
        return cross.T @ self.alpha, np.sqrt(np.maximum(self.kernel.signal_std ** 2 - np.sum(projected ** 2, axis=0), 0))


class Learner:
    def __init__(self, domain, method, seed):
        self.domain, self.method = domain, method
        self.rng = np.random.default_rng(seed)
        self.gp = GaussianProcess3D()
        self.adapter = (Adapter(domain, baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS["baseline"]), L1=.42, L2=.30)
                        if method in PHYSICS_METHODS else None)

    def select(self, measurements):
        history = [m.observation for m in measurements]
        used = {o.candidate_id for o in history}
        available = [p for p in self.domain if p.candidate_id not in used]
        if not available:
            raise RuntimeError("DOMAIN_EXHAUSTED")
        if self.method == "RANDOM":
            return available[int(self.rng.integers(len(available)))], {}
        if self.method == "SPACE_FILLING":
            used_x = np.array([self.domain.by_id(o.candidate_id).features for o in history])
            distances = np.linalg.norm(np.array([p.features for p in available])[:, None, :] - used_x[None, :, :], axis=2).min(axis=1)
            return available[int(np.argmax(distances))], {}
        valid = [o for o in history if o.valid]
        if not valid:
            raise RuntimeError("NO_VALID_MODEL_OBSERVATIONS")
        if self.adapter is not None:
            self.adapter.fit(history)
            physics = {p.candidate_id: self.adapter.predict_value(p) for p in self.domain}
            baseline = np.array([physics[p.candidate_id] for p in available])
            targets = [o.endpoint_value - physics[o.candidate_id] for o in valid]
        else:
            mean = float(np.mean([o.endpoint_value for o in valid]))
            baseline = np.full(len(available), mean)
            targets = [o.endpoint_value - mean for o in valid]
        if self.method == "PHYSICS_GREEDY":
            scores = baseline
        else:
            self.gp.fit([self.domain.by_id(o.candidate_id).features for o in valid], targets,
                        [o.endpoint_uncertainty or 0. for o in valid])
            residual, std = self.gp.predict([p.features for p in available])
            mean = baseline + residual
            if self.method == "RESIDUAL_GP_GREEDY":
                scores = mean
            else:
                feasible = [m.observation.endpoint_value for m in measurements if m.observation.valid and m.feasible]
                if not feasible:
                    raise RuntimeError("NO_OBSERVED_FEASIBLE_INCUMBENT")
                scores = -np.array([expected_improvement(float(mu), float(sigma), min(feasible)) for mu, sigma in zip(mean, std)])
        selected = available[int(np.argmin(scores))]
        diagnostics = self.adapter.metadata() if self.adapter else {}
        return selected, diagnostics


def recommend(measurements):
    eligible = [m.observation for m in measurements if m.observation.valid and m.feasible]
    return min(eligible, key=lambda o: (o.endpoint_value, o.trial_index)).candidate_id if eligible else None


def run_sequence(domain, backend, *, subject_id, method, budget=8, noise_std=0., seed=0, tier=.01):
    if method not in METHODS or budget < 1 or budget > len(domain):
        raise ValueError("INVALID_METHOD_OR_BUDGET")
    environment = Environment(domain, backend, subject_id=subject_id, noise_std=noise_std,
                              seed=seed, tier=tier, physics=method in PHYSICS_METHODS)
    learner = Learner(domain, method, seed)
    measurements, rows, diagnostics = [], [], []
    point, failure = domain.reference, None
    started = time.perf_counter()
    for k in range(1, (1 if method == "REFERENCE" else budget) + 1):
        measurement = environment.evaluate(point, k)
        measurements.append(measurement)
        o = measurement.observation
        rows.append(dict(trial=k, candidate_id=point.candidate_id, observed_E3=o.endpoint_value,
                         observed_E2=measurement.E2, observed_peak_ratio=measurement.peak_ratio,
                         measurement_valid=o.valid, observed_feasible=measurement.feasible,
                         invalid_reason=o.invalid_reason, recommendation_id=recommend(measurements)))
        if k == 1 and not o.valid:
            failure = "REFERENCE_MEASUREMENT_FAILED"
            break
        if method != "REFERENCE" and k < budget:
            try:
                point, diagnostic = learner.select(measurements)
                diagnostics.append(dict(after_trial=k, **diagnostic))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                failure = f"SELECTION_FAILED: {error}"
                break
    return dict(rows=rows, measurements=measurements, diagnostics=diagnostics,
                failure=failure, elapsed_s=time.perf_counter() - started)
