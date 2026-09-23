"""Evidence-gated, measurement-driven personalization for the MyoLeg benchmark.

This module is deliberately independent from :mod:`experiment`.  It contains a
small, deterministic controller that can be used with a MyoLeg ``Point`` (or a
plain candidate object exposing ``candidate_id`` and ``features``) and an
observation-only evaluator.  The controller never receives a simulator,
replay store, complete outcome table, or an oracle.  It only fits its residual
model from observations returned by the evaluator.

The policy is called SAST-BO (Subject-Adaptive Safe Trust-region Bayesian
Optimization).  EG-CPI-BO is retained as a descriptive alias for the same
evidence-gated constrained acquisition policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from personalization.observations import EpisodeObservation
from personalization.selectors.bo import expected_improvement


ArrayLike = Sequence[float] | np.ndarray


def _features(candidate: Any) -> np.ndarray:
    """Return finite candidate features without inspecting hidden outcomes."""

    value = getattr(candidate, "features", None)
    if value is None:
        value = getattr(candidate, "beta", None)
    if value is None:
        raise TypeError("candidate must expose features or beta")
    out = np.asarray(value, dtype=float).reshape(-1)
    if not len(out) or not np.isfinite(out).all():
        raise ValueError("candidate features must be finite and non-empty")
    return out


def _candidate_id(candidate: Any) -> str:
    value = getattr(candidate, "candidate_id", None)
    if value is None:
        raise TypeError("candidate must expose candidate_id")
    return str(value)


@dataclass(frozen=True)
class CandidateView:
    """Minimal candidate contract useful in unit tests and stress runners.

    ``constraint_predictions`` contains *model predictions*, not hidden truth:
    ``{"E2": (mean, std), "peak_ratio": (mean, std)}``.  It is optional; a
    caller can provide the same predictions through ``constraint_predictor``
    on :class:`SASTBO`.
    """

    candidate_id: str
    features: tuple[float, ...]
    constraint_predictions: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationRecord:
    """Observation-only payload consumed by the controller.

    Values are measured endpoint values.  Constraint values are optional and
    are only used to fit the safety surrogate after an episode has completed.
    """

    candidate_id: str
    features: tuple[float, ...]
    value: float | None
    uncertainty: float = 0.0
    feasible: bool = True
    constraint_values: Mapping[str, float] = field(default_factory=dict)
    trial_index: int = 1
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.trial_index < 1:
            raise ValueError("trial_index must be >= 1")
        x = np.asarray(self.features, dtype=float).reshape(-1)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("observation features must be finite and non-empty")
        if self.valid and (self.value is None or not math.isfinite(float(self.value))):
            raise ValueError("valid observations require a finite value")
        if not self.valid and self.value is not None:
            raise ValueError("invalid observations must have value=None")
        if self.uncertainty < 0 or not math.isfinite(float(self.uncertainty)):
            raise ValueError("uncertainty must be finite and non-negative")
        if any(not math.isfinite(float(value)) for value in self.constraint_values.values()):
            raise ValueError("constraint observations must be finite")
        if any(name in self.constraint_values and float(self.constraint_values[name]) < 0.0 for name in ("E2", "peak_ratio")):
            raise ValueError("constraint ratios must be non-negative")


def observation_from_episode(
    observation: EpisodeObservation,
    candidate: Any,
    *,
    feasible: bool = True,
    constraint_values: Mapping[str, float] | None = None,
) -> ObservationRecord:
    """Convert the repository episode contract to the local controller API.

    ``EpisodeObservation`` intentionally does not contain E2/peak values.  A
    benchmark adapter may pass them via ``constraint_values`` after the
    response has been measured; no evaluator-only data is required.
    """

    if observation.candidate_id != _candidate_id(candidate):
        raise ValueError("OBSERVATION_CANDIDATE_ID_MISMATCH")
    values = {} if constraint_values is None else dict(constraint_values)
    return ObservationRecord(
        candidate_id=observation.candidate_id,
        features=tuple(float(x) for x in _features(candidate)),
        value=None if not observation.valid else float(observation.endpoint_value),
        uncertainty=float(observation.endpoint_uncertainty or 0.0),
        feasible=bool(feasible and observation.valid),
        constraint_values=values,
        trial_index=observation.trial_index,
        valid=bool(observation.valid),
        invalid_reason=observation.invalid_reason,
    )


@dataclass(frozen=True)
class ProbePlan:
    """Frozen first-trial protocol: reference followed by named probes."""

    reference_id: str
    calibration_probe_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = (self.reference_id, *self.calibration_probe_ids)
        if any(not str(item) for item in ids):
            raise ValueError("probe ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("reference and calibration probes must be unique")

    @property
    def frozen_order(self) -> tuple[str, ...]:
        return (self.reference_id, *self.calibration_probe_ids)


@dataclass(frozen=True)
class GateConfig:
    """Pre-registered evidence gate and acquisition settings."""

    interaction_threshold: float = 0.0025
    posterior_threshold: float = 0.95
    min_valid_observations: int = 3
    prior_interaction_std: float = 0.01
    exploration_weight: float = 1.0
    safety_probability: float = 0.95
    e2_limit: float = 1.01
    peak_ratio_limit: float = 1.10
    trust_radius: float = 2.0
    length_scale: float = 0.7
    residual_signal_std: float = 0.6
    constraint_signal_std: float = 0.10
    constraint_noise_floor: float = 0.005
    jitter: float = 1.0e-9

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in self.__dict__.values()):
            raise ValueError("configuration must be finite")
        if self.interaction_threshold < 0 or not 0.5 < self.posterior_threshold < 1:
            raise ValueError("invalid evidence thresholds")
        if self.min_valid_observations < 2:
            raise ValueError("at least two observations are required")
        if self.prior_interaction_std <= 0 or self.length_scale <= 0:
            raise ValueError("prior and length scale must be positive")
        if not 0 < self.safety_probability < 1:
            raise ValueError("safety_probability must lie in (0,1)")
        if min(self.constraint_signal_std, self.constraint_noise_floor, self.residual_signal_std, self.trust_radius, self.jitter) <= 0:
            raise ValueError("model scales and trust radius must be positive")


@dataclass(frozen=True)
class GateResult:
    """Heuristic evidence score; ``posterior_probability`` is a legacy name.

    The score is not a calibrated posterior probability or a test of
    population subject-by-trajectory interaction.  It detects deviation from
    an explicit, frozen common response model on the calibration probes.
    """
    active: bool
    status: str
    posterior_probability: float
    interaction_score: float
    valid_count: int
    reason: str


@dataclass(frozen=True)
class Decision:
    candidate: Any
    mode: str
    reason: str
    acquisition: float | None
    gate: GateResult
    safe_candidate_count: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


class LocalResidualModel:
    """Fixed-kernel local residual model with uncertainty from posterior variance."""

    def __init__(self, *, length_scale: float = 0.7, signal_std: float = 0.6, jitter: float = 1e-9):
        self.length_scale = float(length_scale)
        self.signal_std = float(signal_std)
        self.jitter = float(jitter)
        self.x = np.empty((0, 0), dtype=float)
        self.alpha: np.ndarray | None = None
        self.chol: np.ndarray | None = None

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / self.length_scale
        scaled = math.sqrt(5.0) * distance
        return self.signal_std**2 * (1 + scaled + scaled**2 / 3.0) * np.exp(-scaled)

    def fit(self, features: Sequence[ArrayLike], residuals: Sequence[float], noise: Sequence[float]) -> None:
        y = np.asarray(residuals, dtype=float).reshape(-1)
        self.x = np.asarray(features, dtype=float)
        n = len(y)
        if n == 0:
            self.x = np.empty((0, 0), dtype=float)
            self.alpha = self.chol = None
            return
        if self.x.ndim != 2 or self.x.shape[0] != n or not np.isfinite(self.x).all():
            raise ValueError("invalid residual model features")
        sigma = np.maximum(np.asarray(noise, dtype=float).reshape(-1), 1e-6)
        if sigma.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(sigma).all():
            raise ValueError("invalid residual model observations")
        covariance = self._kernel(self.x, self.x) + np.diag(sigma**2 + self.jitter)
        extra = self.jitter
        for _ in range(8):
            try:
                self.chol = np.linalg.cholesky(covariance + extra * np.eye(n))
                break
            except np.linalg.LinAlgError:
                extra *= 10.0
        else:
            raise RuntimeError("RESIDUAL_MODEL_FACTORIZATION_FAILED")
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, y))

    def predict(self, features: ArrayLike) -> tuple[float, float]:
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if self.chol is None or self.alpha is None or not len(self.x):
            return 0.0, self.signal_std
        cross = self._kernel(self.x, x)[:, 0]
        projected = np.linalg.solve(self.chol, cross)
        mean = float(cross @ self.alpha)
        variance = max(self.signal_std**2 - float(projected @ projected), 0.0)
        return mean, math.sqrt(variance)


ConstraintPredictor = Callable[[Any, Sequence[ObservationRecord]], Mapping[str, tuple[float, float]]]
CommonPredictor = Callable[[Any], float]


class SASTBO:
    """Evidence-gated residual BO controller.

    The first calls are deterministic: reference, then all predeclared
    calibration probes.  Before the gate is active, the controller returns a
    common-policy candidate.  Once active, it uses a local residual GP and
    expected-improvement acquisition over candidates with conservative E2 and
    peak predictions.  No candidate truth is read by this class.
    """

    def __init__(
        self,
        candidates: Iterable[Any],
        *,
        reference_id: str,
        calibration_probe_ids: Sequence[str] = (),
        common_policy_id: str | None = None,
        common_predictor: CommonPredictor | None = None,
        constraint_predictor: ConstraintPredictor | None = None,
        preapproved_candidate_ids: Sequence[str] = (),
        config: GateConfig | None = None,
    ) -> None:
        self.candidates = tuple(candidates)
        self.lookup = {_candidate_id(c): c for c in self.candidates}
        if len(self.lookup) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        self.plan = ProbePlan(str(reference_id), tuple(str(x) for x in calibration_probe_ids))
        missing = [x for x in self.plan.frozen_order if x not in self.lookup]
        if missing:
            raise ValueError(f"probe candidates missing: {missing}")
        self.common_policy_id = common_policy_id or self.plan.reference_id
        if self.common_policy_id not in self.lookup:
            raise ValueError("common_policy_id is not in candidates")
        self.common_predictor = common_predictor or self._default_common_predictor
        self.has_common_model = common_predictor is not None
        self.constraint_predictor = constraint_predictor
        self.config = config or GateConfig()
        self.preapproved_candidate_ids = frozenset(preapproved_candidate_ids)
        allowed_preapproved = set(self.plan.frozen_order) | {self.common_policy_id}
        if not self.preapproved_candidate_ids.issubset(allowed_preapproved):
            raise ValueError("PREAPPROVAL_RESTRICTED_TO_REFERENCE_CALIBRATION_COMMON")
        self.history: list[ObservationRecord] = []
        self.residual_model = LocalResidualModel(
            length_scale=self.config.length_scale,
            signal_std=self.config.residual_signal_std,
            jitter=self.config.jitter,
        )
        # Fit deviations from a conservative offset at the constraint limit.
        # Away from observations the prior reverts to the limit with nonzero
        # uncertainty, hence unknown regions cannot become falsely safe.
        self.constraint_models: dict[str, LocalResidualModel] = {}
        self.constraint_offsets: dict[str, float] = {}
        self.last_gate = GateResult(False, "PERSONALIZATION_INACTIVE", 0.0, 0.0, 0, "NO_OBSERVATIONS")

    def _default_common_predictor(self, candidate: Any) -> float:
        # A caller that has a declared mechanics prior can inject it through
        # ``common_predictor``.  The default deliberately ignores every
        # candidate attribute except kinematic features, so an accidental
        # ``oracle``/``truth``/cache field can never influence selection.
        del candidate
        return 0.0

    def _valid_history(self) -> list[ObservationRecord]:
        return [item for item in self.history if item.valid and item.value is not None]

    def _gate(self) -> GateResult:
        valid = [item for item in self._valid_history() if item.candidate_id in self.plan.frozen_order]
        n = len(valid)
        if not self.has_common_model:
            return GateResult(False, "PERSONALIZATION_INACTIVE", 0.0, 0.0, n, "COMMON_MODEL_REQUIRED")
        if n < self.config.min_valid_observations:
            return GateResult(False, "PERSONALIZATION_INACTIVE", 0.0, 0.0, n, "INSUFFICIENT_CALIBRATION_EVIDENCE")
        try:
            common_values = np.asarray(
                [float(self.common_predictor(self.lookup[item.candidate_id])) for item in valid],
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError):
            return GateResult(False, "PERSONALIZATION_INACTIVE", 0.0, 0.0, n, "COMMON_MODEL_INVALID")
        if common_values.shape != (n,) or not np.isfinite(common_values).all():
            return GateResult(False, "PERSONALIZATION_INACTIVE", 0.0, 0.0, n, "COMMON_MODEL_INVALID")
        residuals = np.asarray([float(item.value) for item in valid]) - common_values
        centered = residuals - np.mean(residuals)
        noise = np.asarray([max(float(item.uncertainty), 1e-6) for item in valid])
        noise_level = float(np.sqrt(np.mean(noise**2) + self.config.prior_interaction_std**2))
        observed_sd = float(np.sqrt(np.mean(centered**2)))
        signal_sd = math.sqrt(max(observed_sd**2 - float(np.mean(noise**2)), 0.0))
        # A preregistered heuristic score, NOT a calibrated posterior/test.
        # Common-model error and unmodelled response scaling may also trigger
        # it; null/positive controls must measure its empirical behavior.
        standard_error = max(noise_level / math.sqrt(n), self.config.prior_interaction_std / math.sqrt(n))
        score = (signal_sd - self.config.interaction_threshold) / standard_error
        probability = _normal_cdf(score)
        active = probability >= self.config.posterior_threshold
        return GateResult(
            active=active,
            status="PERSONALIZATION_ACTIVE" if active else "PERSONALIZATION_INACTIVE",
            posterior_probability=float(probability),
            interaction_score=float(signal_sd),
            valid_count=n,
            reason="INTERACTION_EVIDENCE_SUFFICIENT" if active else "INTERACTION_EVIDENCE_BELOW_GATE",
        )

    def observe(self, observation: ObservationRecord | EpisodeObservation, candidate: Any | None = None, *, feasible: bool = True, constraint_values: Mapping[str, float] | None = None) -> GateResult:
        """Append one evaluator response and update gate/model state."""

        if isinstance(observation, EpisodeObservation):
            if candidate is None:
                candidate = self.lookup.get(observation.candidate_id)
            if candidate is None:
                raise ValueError("candidate is required for EpisodeObservation")
            observation = observation_from_episode(observation, candidate, feasible=feasible, constraint_values=constraint_values)
        if not isinstance(observation, ObservationRecord):
            raise TypeError("observation must be ObservationRecord or EpisodeObservation")
        if observation.candidate_id not in self.lookup:
            raise ValueError("observation candidate is outside this domain")
        expected_features = _features(self.lookup[observation.candidate_id])
        observed_features = np.asarray(observation.features, dtype=float)
        if expected_features.shape != observed_features.shape or not np.allclose(expected_features, observed_features, rtol=0, atol=1e-12):
            raise ValueError("OBSERVATION_FEATURE_MISMATCH")
        if observation.trial_index != len(self.history) + 1:
            raise ValueError("OBSERVATION_TRIAL_INDEX_MISMATCH")
        if not self.history and observation.candidate_id != self.plan.reference_id:
            raise ValueError("REFERENCE_FIRST")
        if observation.candidate_id in {item.candidate_id for item in self.history}:
            raise ValueError("DUPLICATE_OBSERVATION")
        self.history.append(observation)
        self.last_gate = self._gate()
        self._fit_models()
        return self.last_gate

    def _fit_models(self) -> None:
        valid = self._valid_history()
        if not valid:
            self.residual_model.fit([], [], [])
            self.constraint_models = {}
            self.constraint_offsets = {}
            return
        x = [_features(self.lookup[item.candidate_id]) for item in valid]
        noise = [max(float(item.uncertainty), 1e-6) for item in valid]
        if self.has_common_model:
            try:
                residual = [float(item.value) - float(self.common_predictor(self.lookup[item.candidate_id])) for item in valid]
                if np.isfinite(residual).all():
                    self.residual_model.fit(x, residual, noise)
                else:
                    self.residual_model.fit([], [], [])
            except (TypeError, ValueError, OverflowError):
                self.residual_model.fit([], [], [])
        else:
            self.residual_model.fit([], [], [])
        self.constraint_models = {}
        self.constraint_offsets = {}
        for name in ("E2", "peak_ratio"):
            rows = [item for item in valid if name in item.constraint_values]
            if not rows:
                continue
            values = np.asarray([float(item.constraint_values[name]) for item in rows], dtype=float)
            if not np.isfinite(values).all():
                continue
            limit = self.config.e2_limit if name == "E2" else self.config.peak_ratio_limit
            offset = max(float(limit), float(np.mean(values)))
            model = LocalResidualModel(
                length_scale=self.config.length_scale,
                signal_std=self.config.constraint_signal_std,
                jitter=self.config.jitter,
            )
            model.fit(
                [_features(self.lookup[item.candidate_id]) for item in rows],
                values - offset,
                [max(float(item.uncertainty), self.config.constraint_noise_floor) for item in rows],
            )
            self.constraint_models[name] = model
            self.constraint_offsets[name] = offset

    def _constraint_prediction(self, candidate: Any) -> Mapping[str, tuple[float, float]]:
        if self.constraint_predictor is not None:
            value = self.constraint_predictor(candidate, tuple(self.history))
            return {str(k): (float(v[0]), float(v[1])) for k, v in value.items()}
        raw = getattr(candidate, "constraint_predictions", {})
        predictions = {str(k): (float(v[0]), float(v[1])) for k, v in dict(raw).items()}
        for name, model in self.constraint_models.items():
            if name not in predictions:
                residual, std = model.predict(_features(candidate))
                predictions[name] = self.constraint_offsets[name] + residual, std
        return predictions

    def _is_safe(self, candidate: Any) -> bool:
        try:
            prediction = self._constraint_prediction(candidate)
        except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            return False
        required = ("E2", "peak_ratio")
        if not all(name in prediction for name in required):
            return False
        # Bonferroni allocation gives the declared joint target without an
        # assumption that E2 and peak model errors are independent.
        z = NormalDist().inv_cdf(1 - (1 - self.config.safety_probability) / len(required))
        e2, peak = prediction["E2"], prediction["peak_ratio"]
        if any(not math.isfinite(value) for pair in (e2, peak) for value in pair) or e2[0] < 0 or peak[0] < 0 or e2[1] < 0 or peak[1] < 0:
            return False
        return e2[0] + z * e2[1] <= self.config.e2_limit and peak[0] + z * peak[1] <= self.config.peak_ratio_limit

    def _observed_safe(self, observation: ObservationRecord) -> bool:
        values = observation.constraint_values
        return bool(observation.valid and observation.feasible and "E2" in values and "peak_ratio" in values
                    and values["E2"] <= self.config.e2_limit and values["peak_ratio"] <= self.config.peak_ratio_limit)

    def recommend(self) -> str | None:
        """Recommend only an executed, valid, measured-feasible candidate."""
        eligible = [item for item in self.history if self._observed_safe(item)]
        if not eligible:
            return None
        if not self._gate().active:
            safe_ids = {item.candidate_id for item in eligible}
            for candidate_id in (self.common_policy_id, self.plan.reference_id):
                if candidate_id in safe_ids:
                    return candidate_id
            return None
        return min(eligible, key=lambda item: (float(item.value), item.trial_index)).candidate_id

    def _safe_execution_decision(self, candidate: Any, mode: str, reason: str, gate: GateResult, *, calibration: bool = False) -> Decision:
        if _candidate_id(candidate) in self.preapproved_candidate_ids or self._is_safe(candidate):
            return Decision(candidate, mode, reason, None, gate, 1)
        return Decision(None, "SAFETY_LOCK", "REQUESTED_CANDIDATE_NOT_CERTIFIED", None, gate, 0)

    def _within_trust_region(self, candidate: Any) -> bool:
        valid = [item for item in self.history if self._observed_safe(item)]
        if not valid:
            return True
        observed = np.asarray([_features(self.lookup[item.candidate_id]) for item in valid])
        distances = np.linalg.norm(observed - _features(candidate), axis=1)
        return bool(np.min(distances) <= self.config.trust_radius)

    def select_next(self) -> Decision:
        """Select the next candidate and return auditable decision metadata."""

        used = {item.candidate_id for item in self.history}
        if not self.history:
            candidate = self.lookup[self.plan.reference_id]
            return self._safe_execution_decision(candidate, "REFERENCE", "REFERENCE_FIRST", self.last_gate, calibration=True)
        if not self._observed_safe(self.history[-1]):
            return Decision(None, "SAFETY_LOCK", "PREVIOUS_MEASUREMENT_INVALID_OR_UNSAFE", None, self.last_gate, 0)
        for probe_id in self.plan.calibration_probe_ids:
            if probe_id not in used:
                candidate = self.lookup[probe_id]
                return self._safe_execution_decision(candidate, "CALIBRATION", "PREDECLARED_CALIBRATION_PROBE", self.last_gate, calibration=True)
        gate = self._gate()
        self.last_gate = gate
        if not gate.active:
            candidate = self.lookup[self.common_policy_id]
            if _candidate_id(candidate) in used:
                observed = next(item for item in self.history if item.candidate_id == self.common_policy_id)
                if self._observed_safe(observed):
                    return Decision(candidate, "COMMON_POLICY", "PERSONALIZATION_INACTIVE", None, gate, 1)
                return Decision(None, "SAFETY_LOCK", "COMMON_POLICY_OBSERVED_UNSAFE", None, gate, 0)
            return self._safe_execution_decision(candidate, "COMMON_POLICY", "PERSONALIZATION_INACTIVE", gate)

        available = [c for c in self.candidates if _candidate_id(c) not in used and self._within_trust_region(c)]
        safe = [c for c in available if self._is_safe(c)]
        if not safe:
            return Decision(None, "SAFETY_LOCK", "NO_CERTIFIED_SAFE_CANDIDATE", None, gate, 0)
        incumbent = min(float(item.value) for item in self.history if self._observed_safe(item))
        scored: list[tuple[float, Any, float, float]] = []
        for candidate in safe:
            residual_mean, residual_std = self.residual_model.predict(_features(candidate))
            mean = self.common_predictor(candidate) + residual_mean
            acquisition = -expected_improvement(mean, residual_std, incumbent)
            scored.append((acquisition, candidate, residual_mean, residual_std))
        scored.sort(key=lambda item: (item[0], _candidate_id(item[1])))
        acquisition, candidate, residual_mean, residual_std = scored[0]
        return Decision(
            candidate,
            "PERSONALIZED_BO",
            "EVIDENCE_GATE_ACTIVE",
            float(acquisition),
            gate,
            len(safe),
            diagnostics={"residual_mean": residual_mean, "residual_std": residual_std,
                         "acquisition_kind": "EI_WITH_JOINT_UCB_ADMISSIBILITY"},
        )


def run_sequence(controller: SASTBO, evaluator: Callable[[Any, int], ObservationRecord | EpisodeObservation], *, budget: int) -> dict[str, Any]:
    """Run an observation-only sequence through a frozen controller.

    The evaluator receives only the selected candidate and trial index.  It may
    internally use MyoLeg, but its response is the only information returned to
    the controller.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    decisions: list[Decision] = []
    observations: list[ObservationRecord] = []
    for trial in range(1, budget + 1):
        decision = controller.select_next()
        decisions.append(decision)
        candidate = decision.candidate
        if candidate is None:
            return {
                "observations": observations,
                "decisions": decisions,
                "next_decision": decision,
                "failure": None,
                "termination": decision.mode,
                "gate": controller.last_gate,
            }
        if _candidate_id(candidate) in {item.candidate_id for item in observations}:
            # A common fallback is often the reference that was already
            # measured.  It is a recommendation, not a second physical trial;
            # terminate cleanly instead of allowing a duplicate execution.
            return {
                "observations": observations,
                "decisions": decisions,
                "next_decision": decision,
                "failure": None,
                "termination": "RECOMMENDATION_ALREADY_OBSERVED",
                "gate": controller.last_gate,
            }
        response = evaluator(candidate, trial)
        if isinstance(response, EpisodeObservation):
            response = observation_from_episode(response, candidate)
        controller.observe(response)
        observations.append(response)
    final = controller.select_next()
    return {
        "observations": observations,
        "decisions": decisions,
        "next_decision": final,
        "failure": None,
        "termination": None,
        "gate": controller.last_gate,
    }


# Names used in the research plan and paper draft.
EvidenceGate = SASTBO
EGCPIBO = SASTBO


__all__ = [
    "CandidateView",
    "ObservationRecord",
    "observation_from_episode",
    "ProbePlan",
    "GateConfig",
    "GateResult",
    "Decision",
    "LocalResidualModel",
    "SASTBO",
    "EvidenceGate",
    "EGCPIBO",
    "run_sequence",
]
