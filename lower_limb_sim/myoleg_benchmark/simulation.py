"""Fresh prescribed-state MyoLeg dynamics with versioned, private response cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from external_simulation.myoleg_reference_trajectory_replay_v1 import build_and_replay as replay
from external_simulation.myoleg_v2_candidate_domain_design_v1.build_candidate_domain import model_fingerprint
from .portable_model import DEFAULT_MODEL_PATH, load_model_with_provenance

ROOT = Path(__file__).resolve().parents[2]
COHORT_MANIFEST = ROOT / "external_simulation_audits/myoleg_virtual_patient_cohort_generation_v1/MYOLEG_VIRTUAL_PATIENT_COHORT_V1_MANIFEST.json"
NOMINAL = "MYOLEG_NATIVE_P0"
# Frozen before cohort execution: tolerances describe numerical portability,
# not a replacement for the historical bitwise fingerprint.
REFERENCE_ATOL = 1e-10
REFERENCE_RTOL = 1e-12
DELTA_BEFORE_RTOL = 1e-13
REFERENCE_COMPONENTS = (
    "tau_truth_nm", "tau_inverse_net_nm", "mass_term_nm", "bias_term_nm",
    "passive_internal_nm", "actuator_internal_nm", "constraint_internal_nm",
    "constraint_equality_internal_nm", "constraint_joint_limit_internal_nm",
    "constraint_tendon_limit_internal_nm", "constraint_friction_internal_nm",
    "constraint_contact_internal_nm", "actuator_force_n", "actuator_length_m",
    "tendon_length_m", "actuator_activation", "muscle_moment_independent_m",
    "muscle_torque_contribution_nm", "key_biarticular_torque_contribution_nm",
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_file_identity(path, expected, *, text_file=False):
    """Accept exact frozen content, or CRLF-only changes for a text source."""
    payload = Path(path).read_bytes()
    raw = hashlib.sha256(payload).hexdigest()
    normalized = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() if text_file else raw
    if raw != expected and normalized != expected:
        raise ValueError(f"FROZEN_SOURCE_CONTENT_MISMATCH: {path}")
    return dict(expected_sha256=expected, raw_sha256=raw, lf_sha256=normalized,
                match="exact" if raw == expected else "newline_equivalent")


def _reference_equivalence(model, record):
    """Check only the subject's public reference, never a candidate landscape.

    Saved source_equality_* and warning_count belong to controlled replay
    because dataset_payload overwrites those keys. Compare the independent
    prescribed-state quantities below and check current warnings separately.
    This environment check never returns frozen observations to the learner.
    """
    path = ROOT / record["reference_replay_truth_path"]
    identity = _frozen_file_identity(path, record["reference_replay_truth_sha256"])
    with np.load(path, allow_pickle=False) as frozen:
        reference = dict(time_s=frozen["time_s"], q=frozen["target_q_rad"],
                         dq=frozen["target_dq_rad_s"], ddq=frozen["target_ddq_rad_s2"],
                         phases=frozen["cycle_phase"])
        actual, _ = replay.prescribed_truth(model, reference)
        errors = {}
        for key in REFERENCE_COMPONENTS:
            observed, expected = actual[key], frozen[key]
            if observed.shape != expected.shape or not np.allclose(
                observed, expected, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL,
            ):
                raise ValueError(f"FROZEN_REFERENCE_PHYSICS_MISMATCH: {key}")
            errors[key] = float(np.max(np.abs(observed - expected)))
        for group in ("equality", "joint_limit", "tendon_limit", "friction", "contact"):
            key = f"constraint_{group}_active_count"
            if not np.array_equal(actual[key], frozen[key]):
                raise ValueError(f"FROZEN_REFERENCE_CONSTRAINT_COUNT_MISMATCH: {group}")
        for key in ("actuator_names", "key_biarticular_names"):
            if not np.array_equal(actual[key], frozen[key]):
                raise ValueError(f"FROZEN_REFERENCE_MUSCLE_IDENTITY_MISMATCH: {key}")
    if np.any(actual["warning_count"]) or np.max(np.abs(actual["decomposition_residual_nm"])) > REFERENCE_ATOL:
        raise ValueError("FROZEN_REFERENCE_DYNAMICS_DIAGNOSTIC_FAILED")
    return dict(reference_path=str(path.relative_to(ROOT)), source_identity=identity,
                sample_count=len(reference["time_s"]), atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL, max_absolute_errors=errors,
                interpretation="public reference numerical equivalence; not bit-identical compiled model")


def _set_delta_field(target, selection, specification, label, differences):
    observed = np.asarray(target[selection])
    expected = np.asarray(specification["before"])
    if observed.shape != expected.shape or not np.allclose(
        observed, expected, rtol=DELTA_BEFORE_RTOL, atol=0.,
    ):
        raise ValueError(f"FROZEN_DELTA_BASE_FIELD_MISMATCH: {label}")
    error = float(np.max(np.abs(observed - expected)))
    if error:
        differences[label] = error
    target[selection] = specification["after"]
    if not np.array_equal(target[selection], np.asarray(specification["after"])):
        raise ValueError(f"FROZEN_DELTA_ASSIGNMENT_MISMATCH: {label}")


def development_ids():
    return tuple(json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))["development_subject_ids"])


def subject_model(subject_id):
    # Reject unknown/sealed IDs before reading their model delta or metadata.
    if subject_id != NOMINAL and subject_id not in development_ids():
        raise ValueError("ONLY_NOMINAL_AND_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED")
    manifest = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    xml_identity = _frozen_file_identity(
        DEFAULT_MODEL_PATH, manifest["base_model_sha256"], text_file=True,
    )
    model, provenance = load_model_with_provenance()
    before_differences = {}
    if subject_id != NOMINAL:
        record = next(r for r in manifest["subjects"] if r["subject_id"] == subject_id)
        if record["split"] != "DEVELOPMENT":
            raise ValueError("HELD_OUT_ACCESS_DISABLED")
        path = ROOT / record["model_delta_path"]
        delta_identity = _frozen_file_identity(path, record["model_delta_sha256"], text_file=True)
        delta = json.loads(path.read_text(encoding="utf-8"))
        if delta["base_model_sha256"] != manifest["base_model_sha256"] or delta["factor_order"] != manifest["factor_order"]:
            raise ValueError("FROZEN_DELTA_IDENTITY_MISMATCH")
        for change in delta["modifications"]:
            fields = change["fields"]
            if change["object_type"] == "body":
                index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, change["object_name"])
                if index < 0:
                    raise ValueError("UNKNOWN_BODY")
                if set(fields) != {"body_mass", "body_inertia"}:
                    raise ValueError("NON_SCHEME_A_DELTA_FIELD")
                for field in ("body_mass", "body_inertia"):
                    _set_delta_field(getattr(model, field), index, fields[field],
                                     f"{change['object_name']}.{field}", before_differences)
            elif change["object_type"] == "actuator":
                index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, change["object_name"])
                if index < 0:
                    raise ValueError("UNKNOWN_ACTUATOR")
                if set(fields) != {"actuator_gainprm_7_fpmax", "actuator_biasprm_7_fpmax"}:
                    raise ValueError("NON_SCHEME_A_DELTA_FIELD")
                for field in ("actuator_gainprm", "actuator_biasprm"):
                    _set_delta_field(getattr(model, field), (index, 7), fields[field + "_7_fpmax"],
                                     f"{change['object_name']}.{field}[7]", before_differences)
            else:
                raise ValueError("NON_SCHEME_A_DELTA")
        if delta["generated_model_fingerprint_sha256"] != record["generated_model_fingerprint_sha256"]:
            raise ValueError("FROZEN_DELTA_FINGERPRINT_IDENTITY_MISMATCH")
        frozen_fingerprint = record["generated_model_fingerprint_sha256"]
        provenance.update(delta_path=str(path.relative_to(ROOT)), delta_sha256=file_sha(path),
                          delta_source_identity=delta_identity)
    else:
        record = manifest["nominal_control"]
        metadata_path = ROOT / record["metadata_path"]
        _frozen_file_identity(metadata_path, record["metadata_sha256"], text_file=True)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frozen_fingerprint = metadata["generated_model_fingerprint_sha256"]
    actual = model_fingerprint(model)
    if actual != frozen_fingerprint:
        provenance["reference_numerical_equivalence"] = _reference_equivalence(model, record)
    provenance.update(model_xml_identity=xml_identity,
                      frozen_compiled_fingerprint=frozen_fingerprint,
                      compiled_fingerprint_match=actual == frozen_fingerprint,
                      compiled_model_validation="bitwise_fingerprint" if actual == frozen_fingerprint else "source_verified_reference_numerical_equivalence",
                      delta_before_max_absolute_differences=before_differences,
                      delta_before_rtol=DELTA_BEFORE_RTOL)
    provenance.update(subject_id=subject_id, cohort_manifest_sha256=file_sha(COHORT_MANIFEST),
                      runtime_compiled_fingerprint=model_fingerprint(model),
                      truth_source_sha256=file_sha(Path(replay.__file__)), mujoco_version=mujoco.__version__,
                      truth_semantics="prescribed-state inverse dynamics; generalized required-drive torque, not measured cuff force or forward tracking")
    return model, provenance


def _publish_cache(temporary, destination):
    """Atomically install a complete trace without replacing another worker's.

    Both paths are on the same filesystem. A duplicate publisher may reuse
    only identical cache content; disk errors remain infrastructure failures.
    """
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                with np.load(temporary, allow_pickle=False) as expected, np.load(
                    destination, allow_pickle=False,
                ) as existing:
                    fields = {"tau_nm", "decomposition_residual_nm", "simulation_identity", "trajectory_sha256"}
                    if set(existing.files) != fields or set(expected.files) != fields:
                        raise ValueError("cache fields differ")
                    if any(not np.array_equal(existing[key], expected[key]) for key in fields):
                        raise ValueError("cache identity or values differ")
            except Exception as error:
                raise OSError(f"CONCURRENT_CACHE_CONTENT_MISMATCH: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


class SimulatorBackend:
    """Cache is hidden from the learner; only requested() reveals one trace."""

    def __init__(self, subject_id, domain, cache_dir):
        self.subject_id, self.domain = subject_id, domain
        self.model, self.provenance = subject_model(subject_id)
        identity = {key: self.provenance[key] for key in (
            "runtime_compiled_fingerprint", "truth_source_sha256", "mujoco_version")}
        # The historical fingerprint omits some dynamical options. Bind all XML
        # bytes and this backend implementation as well, not just its array subset.
        identity.update(model_xml_sha256=file_sha(self.provenance["model_path"]),
                        simulator_source_sha256=file_sha(__file__),
                        delta_sha256=self.provenance.get("delta_sha256"))
        self.identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache_dir = Path(cache_dir) / self.identity_sha / domain.family
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}
        self.fresh_simulations = self.disk_hits = 0
        self.max_decomposition_residual_nm = 0.

    def requested(self, point):
        if point.candidate_id in self.memory:
            return self.memory[point.candidate_id].copy()
        curve = point.trajectory
        digest = hashlib.sha256()
        for value in (point.time_s, curve.q, curve.dq, curve.ddq):
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
        key = digest.hexdigest()
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                tau = cached["tau_nm"].copy()
                residual = float(cached["decomposition_residual_nm"])
                if str(cached["simulation_identity"]) != self.identity_sha or str(cached["trajectory_sha256"]) != key:
                    raise ValueError("SIMULATION_CACHE_IDENTITY_MISMATCH")
            self.disk_hits += 1
        else:
            arrays, _ = replay.prescribed_truth(self.model, dict(
                time_s=point.time_s, q=curve.q, dq=curve.dq, ddq=curve.ddq,
                phases=self.domain.subject_reference.phases))
            tau = arrays["tau_truth_nm"]
            residual = float(np.max(np.abs(arrays["decomposition_residual_nm"])))
            if not np.isfinite(tau).all() or not np.isfinite(residual) or residual > 1e-8 or np.any(arrays["warning_count"]):
                raise ValueError("MYOLEG_DYNAMICS_DIAGNOSTIC_FAILED")
            # Concurrent workers may request the same deterministic trace. Each
            # publishes a complete private temporary file without overwriting a
            # trace that another worker may already be reading on Windows.
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, tau_nm=tau, decomposition_residual_nm=residual,
                                    simulation_identity=self.identity_sha, trajectory_sha256=key)
            _publish_cache(temporary, path)
            self.fresh_simulations += 1
        if tau.shape != (len(point.time_s), 2) or not np.isfinite(tau).all() or not np.isfinite(residual) or residual > 1e-8:
            raise ValueError("INVALID_CACHED_DYNAMICS")
        self.max_decomposition_residual_nm = max(self.max_decomposition_residual_nm, residual)
        self.memory[point.candidate_id] = tau.copy()
        return tau.copy()
