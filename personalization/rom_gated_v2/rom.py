"""Offline-only Stage 0 contracts for task-specific allowed ROM.

This module contains no robot, control, collection, or hardware imports.  A
``SubjectROMProfile`` describes the range admitted for the current task; it is
not a claim about a person's complete physiological range of motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


OFFLINE_ALGORITHM_DEVELOPMENT_ONLY = "OFFLINE_ALGORITHM_DEVELOPMENT_ONLY"
OFFLINE_ALGORITHM_TEST_ONLY = "OFFLINE_ALGORITHM_TEST_ONLY"
NOT_A_HUMAN_SAFETY_MODEL = "NOT_A_HUMAN_SAFETY_MODEL"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"
CURRENT_TASK_SPECIFIC_ALLOWED_ROM = "CURRENT_TASK_SPECIFIC_ALLOWED_ROM"
FROZEN_SUBJECT_ROM_PROFILE_V1 = "FROZEN_SUBJECT_ROM_PROFILE_V1"
P_LIMIT = "NOT_AVAILABLE"
VALIDATED_FORCE_THRESHOLD = "NOT_AVAILABLE"
SAFETY_MARGIN_NOT_YET_DEFINED = None
ROM_DETERMINATION_BUDGET_POLICY = (
    "ROM_DETERMINATION_BUDGET_SEPARATE_FROM_PERSONALIZATION_ADAPTATION_BUDGET"
)
REAL_ROM_DETERMINATION_DISABLED = "REAL_ROM_DETERMINATION_DISABLED"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _finite_or_none(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite or None")


@dataclass(frozen=True)
class SubjectROMProfile:
    """Versioned task envelope frozen before Stage 1 starts."""

    profile_id: str
    version: int
    hip_min_rad: float
    hip_max_rad: float
    knee_min_rad: float
    knee_max_rad: float
    rom_status: str
    provenance: str
    boundary_evidence: tuple[Mapping[str, Any], ...] = ()
    safety_margin_policy: str | None = SAFETY_MARGIN_NOT_YET_DEFINED
    threshold_policy_ids: tuple[str, ...] = ()
    calibration_episode_ids: tuple[str, ...] = ()
    frozen: bool = False
    coupled_joint_constraint: str | None = None
    configuration_validity_mask: str | None = None
    trajectory_specific_constraint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.version < 1:
            raise ValueError("ROM profile version must be >= 1")
        bounds = (
            self.hip_min_rad,
            self.hip_max_rad,
            self.knee_min_rad,
            self.knee_max_rad,
        )
        if not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("ROM bounds must be finite")
        if self.hip_min_rad >= self.hip_max_rad:
            raise ValueError("hip_min_rad must be less than hip_max_rad")
        if self.knee_min_rad >= self.knee_max_rad:
            raise ValueError("knee_min_rad must be less than knee_max_rad")
        if not self.provenance:
            raise ValueError("ROM provenance is required")
        if self.frozen and self.rom_status != FROZEN_SUBJECT_ROM_PROFILE_V1:
            raise ValueError("a frozen profile must use FROZEN_SUBJECT_ROM_PROFILE_V1")
        if not self.frozen and self.rom_status == FROZEN_SUBJECT_ROM_PROFILE_V1:
            raise ValueError("FROZEN_SUBJECT_ROM_PROFILE_V1 requires frozen=True")
        object.__setattr__(
            self,
            "boundary_evidence",
            tuple(_freeze(item) for item in self.boundary_evidence),
        )
        object.__setattr__(self, "threshold_policy_ids", tuple(self.threshold_policy_ids))
        object.__setattr__(
            self, "calibration_episode_ids", tuple(self.calibration_episode_ids)
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def semantic_scope(self) -> str:
        return CURRENT_TASK_SPECIFIC_ALLOWED_ROM

    def identity_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "hip_min_rad": self.hip_min_rad,
            "hip_max_rad": self.hip_max_rad,
            "knee_min_rad": self.knee_min_rad,
            "knee_max_rad": self.knee_max_rad,
            "rom_status": self.rom_status,
            "provenance": self.provenance,
            "safety_margin_policy": self.safety_margin_policy,
            "threshold_policy_ids": list(self.threshold_policy_ids),
            "calibration_episode_ids": list(self.calibration_episode_ids),
            "frozen": self.frozen,
            "coupled_joint_constraint": self.coupled_joint_constraint,
            "configuration_validity_mask": self.configuration_validity_mask,
            "trajectory_specific_constraint": self.trajectory_specific_constraint,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "semantic_scope": self.semantic_scope,
            "boundary_evidence": [_thaw(item) for item in self.boundary_evidence],
            "metadata": _thaw(self.metadata),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ROMBoundaryObservation:
    """One currently observed Stage 0 configuration and its measurements."""

    observation_id: str
    calibration_episode_id: str
    configuration_id: str
    hip_angle_rad: float | None
    knee_angle_rad: float | None
    force_features: Mapping[str, float | None] = field(default_factory=dict)
    pressure_features: Mapping[str, float | None] = field(default_factory=dict)
    tracking_features: Mapping[str, float | None] = field(default_factory=dict)
    valid: bool = True
    gate_status: str | None = None
    stop_requested: bool = False
    threshold_policy_id: str | None = None
    measurement_quality: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id or not self.calibration_episode_id:
            raise ValueError("observation and calibration episode IDs are required")
        _finite_or_none(self.hip_angle_rad, "hip_angle_rad")
        _finite_or_none(self.knee_angle_rad, "knee_angle_rad")
        if self.valid and (self.hip_angle_rad is None or self.knee_angle_rad is None):
            raise ValueError("valid ROM observations require finite hip and knee angles")
        for name in ("force_features", "pressure_features", "tracking_features"):
            values = getattr(self, name)
            for key, value in values.items():
                _finite_or_none(value, f"{name}.{key}")
            object.__setattr__(self, name, _freeze(values))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "calibration_episode_id": self.calibration_episode_id,
            "configuration_id": self.configuration_id,
            "hip_angle_rad": self.hip_angle_rad,
            "knee_angle_rad": self.knee_angle_rad,
            "force_features": _thaw(self.force_features),
            "pressure_features": _thaw(self.pressure_features),
            "tracking_features": _thaw(self.tracking_features),
            "valid": self.valid,
            "gate_status": self.gate_status,
            "stop_requested": self.stop_requested,
            "threshold_policy_id": self.threshold_policy_id,
            "measurement_quality": self.measurement_quality,
            "metadata": _thaw(self.metadata),
        }


class ROMGateStatus(str, Enum):
    SAFE = "SAFE"
    STOP = "STOP"


@dataclass(frozen=True)
class ROMGateResult:
    status: ROMGateStatus
    reason: str
    threshold_policy_id: str | None
    classification: str

    @property
    def safe(self) -> bool:
        return self.status is ROMGateStatus.SAFE

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "threshold_policy_id": self.threshold_policy_id,
            "classification": self.classification,
        }


class ROMSafetyGate(ABC):
    """Pluggable, observation-only safety gate interface."""

    @abstractmethod
    def evaluate(self, observation: ROMBoundaryObservation) -> ROMGateResult:
        raise NotImplementedError

    @staticmethod
    def fail_closed_precheck(
        observation: ROMBoundaryObservation,
        *,
        classification: str,
    ) -> ROMGateResult | None:
        if observation.stop_requested or observation.gate_status == "EXTERNAL_STOP":
            return ROMGateResult(
                ROMGateStatus.STOP,
                "EXTERNAL_STOP",
                observation.threshold_policy_id,
                classification,
            )
        if observation.gate_status == ROMGateStatus.STOP.value:
            return ROMGateResult(
                ROMGateStatus.STOP,
                "OBSERVATION_REPORTED_STOP",
                observation.threshold_policy_id,
                classification,
            )
        if not observation.valid:
            return ROMGateResult(
                ROMGateStatus.STOP,
                "INVALID_MEASUREMENT_FAIL_CLOSED",
                observation.threshold_policy_id,
                classification,
            )
        return None


class SyntheticThresholdGate(ROMSafetyGate):
    """Explicit development-only scalar gate; not a human safety model."""

    classification = f"{OFFLINE_ALGORITHM_TEST_ONLY}:{NOT_A_HUMAN_SAFETY_MODEL}"

    def __init__(
        self,
        *,
        stop_threshold: float,
        threshold_policy_id: str,
        feature_name: str = "synthetic_boundary_proxy",
    ) -> None:
        if not math.isfinite(stop_threshold):
            raise ValueError("synthetic stop_threshold must be finite")
        if not threshold_policy_id.startswith("SYNTHETIC_"):
            raise ValueError("synthetic threshold policy IDs must start with SYNTHETIC_")
        self.stop_threshold = float(stop_threshold)
        self.threshold_policy_id = threshold_policy_id
        self.feature_name = feature_name

    def evaluate(self, observation: ROMBoundaryObservation) -> ROMGateResult:
        precheck = self.fail_closed_precheck(
            observation, classification=self.classification
        )
        if precheck is not None:
            return precheck
        raw = observation.metadata.get(self.feature_name)
        if raw is None or not math.isfinite(float(raw)):
            return ROMGateResult(
                ROMGateStatus.STOP,
                "SYNTHETIC_FEATURE_MISSING_OR_INVALID_FAIL_CLOSED",
                self.threshold_policy_id,
                self.classification,
            )
        status = (
            ROMGateStatus.STOP
            if float(raw) >= self.stop_threshold
            else ROMGateStatus.SAFE
        )
        return ROMGateResult(
            status,
            "SYNTHETIC_THRESHOLD_REACHED" if status is ROMGateStatus.STOP else "SAFE",
            self.threshold_policy_id,
            self.classification,
        )


class UnavailableValidatedSafetyGate(ROMSafetyGate):
    """Fail-closed placeholder until real threshold policies are validated."""

    def evaluate(self, observation: ROMBoundaryObservation) -> ROMGateResult:
        precheck = self.fail_closed_precheck(
            observation, classification=OFFLINE_ALGORITHM_DEVELOPMENT_ONLY
        )
        if precheck is not None:
            return precheck
        return ROMGateResult(
            ROMGateStatus.STOP,
            "VALIDATED_PRESSURE_FORCE_THRESHOLDS_NOT_AVAILABLE",
            None,
            OFFLINE_ALGORITHM_DEVELOPMENT_ONLY,
        )


class CompositeROMSafetyGate(ROMSafetyGate):
    """Fail-closed composition point for future independently validated gates."""

    def __init__(self, gates: tuple[ROMSafetyGate, ...]) -> None:
        if not gates:
            raise ValueError("at least one ROM safety gate is required")
        self.gates = tuple(gates)

    def evaluate(self, observation: ROMBoundaryObservation) -> ROMGateResult:
        results = tuple(gate.evaluate(observation) for gate in self.gates)
        for result in results:
            if not result.safe:
                return result
        return ROMGateResult(
            ROMGateStatus.SAFE,
            "ALL_CONFIGURED_GATES_SAFE",
            "+".join(
                result.threshold_policy_id
                for result in results
                if result.threshold_policy_id is not None
            )
            or None,
            OFFLINE_ALGORITHM_DEVELOPMENT_ONLY,
        )


@dataclass(frozen=True)
class ROMCalibrationEntry:
    sequence_index: int
    observation: ROMBoundaryObservation
    gate_result: ROMGateResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "observation": self.observation.as_dict(),
            "gate_result": self.gate_result.as_dict(),
        }


class ROMCalibrationLedger:
    """Stage 0 ledger, intentionally separate from the Stage 1 trial ledger."""

    ledger_type = "ROM_CALIBRATION_LEDGER"

    def __init__(self) -> None:
        self._entries: list[ROMCalibrationEntry] = []

    @property
    def entries(self) -> tuple[ROMCalibrationEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: ROMCalibrationEntry) -> None:
        if entry.sequence_index != len(self._entries) + 1:
            raise RuntimeError("ROM calibration indices must be contiguous")
        if entry.observation.observation_id in {
            item.observation.observation_id for item in self._entries
        }:
            raise RuntimeError("duplicate ROM observation ID")
        self._entries.append(entry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ledger_type": self.ledger_type,
            "budget_policy": ROM_DETERMINATION_BUDGET_POLICY,
            "entries": [entry.as_dict() for entry in self._entries],
        }


class ROMDeterminationController:
    """Offline boundary-finding state machine that stops on the first non-SAFE."""

    def __init__(self, gate: ROMSafetyGate) -> None:
        self.gate = gate
        self.ledger = ROMCalibrationLedger()
        self.first_safe_configuration: ROMBoundaryObservation | None = None
        self.last_safe_configuration: ROMBoundaryObservation | None = None
        self.first_stop_configuration: ROMBoundaryObservation | None = None
        self._closed = False
        self._profile: SubjectROMProfile | None = None

    @property
    def can_propose(self) -> bool:
        return not self._closed

    @property
    def profile(self) -> SubjectROMProfile | None:
        return self._profile

    def observe(self, observation: ROMBoundaryObservation) -> ROMGateResult:
        if self._closed:
            raise RuntimeError("ROM determination closed; further expansion is prohibited")
        result = self.gate.evaluate(observation)
        if result.safe:
            if self.last_safe_configuration is not None:
                previous = self.last_safe_configuration
                assert previous.hip_angle_rad is not None
                assert previous.knee_angle_rad is not None
                assert observation.hip_angle_rad is not None
                assert observation.knee_angle_rad is not None
                if (
                    observation.hip_angle_rad < previous.hip_angle_rad
                    or observation.knee_angle_rad < previous.knee_angle_rad
                ):
                    result = ROMGateResult(
                        ROMGateStatus.STOP,
                        "NON_MONOTONIC_EXPANSION_FAIL_CLOSED",
                        result.threshold_policy_id,
                        result.classification,
                    )
        entry = ROMCalibrationEntry(len(self.ledger.entries) + 1, observation, result)
        self.ledger.append(entry)
        if result.safe:
            if self.first_safe_configuration is None:
                self.first_safe_configuration = observation
            self.last_safe_configuration = observation
        else:
            self.first_stop_configuration = observation
            self._closed = True
        return result

    def freeze_profile(
        self,
        *,
        profile_id: str,
        version: int,
        provenance: str,
        safety_margin_policy: str | None = SAFETY_MARGIN_NOT_YET_DEFINED,
        coupled_joint_constraint: str | None = None,
        configuration_validity_mask: str | None = None,
        trajectory_specific_constraint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SubjectROMProfile:
        if self._profile is not None:
            raise RuntimeError("ROM profile is already frozen")
        if self.first_safe_configuration is None or self.last_safe_configuration is None:
            raise RuntimeError("cannot freeze ROM without a defensibly SAFE observation")
        first_safe = self.first_safe_configuration
        last_safe = self.last_safe_configuration
        assert first_safe.hip_angle_rad is not None
        assert first_safe.knee_angle_rad is not None
        assert last_safe.hip_angle_rad is not None
        assert last_safe.knee_angle_rad is not None
        evidence = tuple(
            {
                "observation_id": entry.observation.observation_id,
                "configuration_id": entry.observation.configuration_id,
                "gate_status": entry.gate_result.status.value,
                "gate_reason": entry.gate_result.reason,
            }
            for entry in self.ledger.entries
        )
        threshold_ids = tuple(
            dict.fromkeys(
                entry.gate_result.threshold_policy_id
                for entry in self.ledger.entries
                if entry.gate_result.threshold_policy_id is not None
            )
        )
        calibration_ids = tuple(
            dict.fromkeys(
                entry.observation.calibration_episode_id
                for entry in self.ledger.entries
            )
        )
        self._profile = SubjectROMProfile(
            profile_id=profile_id,
            version=version,
            hip_min_rad=float(first_safe.hip_angle_rad),
            hip_max_rad=float(last_safe.hip_angle_rad),
            knee_min_rad=float(first_safe.knee_angle_rad),
            knee_max_rad=float(last_safe.knee_angle_rad),
            rom_status=FROZEN_SUBJECT_ROM_PROFILE_V1,
            provenance=provenance,
            boundary_evidence=evidence,
            safety_margin_policy=safety_margin_policy,
            threshold_policy_ids=threshold_ids,
            calibration_episode_ids=calibration_ids,
            frozen=True,
            coupled_joint_constraint=coupled_joint_constraint,
            configuration_validity_mask=configuration_validity_mask,
            trajectory_specific_constraint=trajectory_specific_constraint,
            metadata={
                "classification": OFFLINE_ALGORITHM_DEVELOPMENT_ONLY,
                "semantic_scope": CURRENT_TASK_SPECIFIC_ALLOWED_ROM,
                "first_safe_configuration_id": first_safe.configuration_id,
                "last_safe_configuration_id": last_safe.configuration_id,
                "first_stop_configuration_id": (
                    self.first_stop_configuration.configuration_id
                    if self.first_stop_configuration is not None
                    else None
                ),
                "no_boundary_interpolation": True,
                "pressure_limit": P_LIMIT,
                "validated_force_threshold": VALIDATED_FORCE_THRESHOLD,
                **dict(metadata or {}),
            },
        )
        self._closed = True
        return self._profile


class RealROMDeterminationInterface:
    """Fail-closed placeholder with no motion execution dependency."""

    def determine(self, *_: Any, **__: Any) -> SubjectROMProfile:
        raise RuntimeError(f"{REAL_ROM_DETERMINATION_DISABLED}: {NOT_ROBOT_APPROVED}")
