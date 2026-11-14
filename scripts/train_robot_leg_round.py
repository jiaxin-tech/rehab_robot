"""Train the frozen residual-GP mean Greedy in a separate, hardware-free process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiment_algorithm.strict_replay import replay_attempts
from measurement_validation.robot_leg_history import IDENTITY_KEYS, CLASSIFICATIONS, COMPLETE_STATUSES, _numeric, _series, TIME_ATOL, measurement_endpoints
from measurement_validation.torque_endpoints import endpoints


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def train(request, *, validate_only=False):
    # The subprocess keeps the frozen standalone module named 'config' isolated
    # from the robot application's config package.
    root = Path(request["algorithm_root"]).resolve(strict=True)
    local_algorithm = Path(__file__).resolve().parents[1] / "experiment_algorithm/v2"
    version = request.get("algorithm_version", "REHAB_OFFLINE_V2" if root == local_algorithm else "LEGACY_EXTERNAL_SNAPSHOT")
    if version == "REHAB_OFFLINE_V2":
        from experiment_algorithm.binding import current_binding
        expected = current_binding()
        if (root != Path(expected["algorithm_root"]) or
                any(request.get(key) != expected[key] for key in ("algorithm_sha256", "algorithm_dependencies")) or
                request.get("algorithm_source_lock_sha256", expected["algorithm_source_lock_sha256"]) != expected["algorithm_source_lock_sha256"]):
            raise ValueError("request does not match the locked repository algorithm version")
    else:
        sys.path.insert(0, str(root))
    if set(request["algorithm_sha256"]) != {"config.py", "surrogate.py", "policy.py"} or not request["algorithm_dependencies"]:
        raise ValueError("complete frozen algorithm and dependency hashes required")
    for name, expected in request["algorithm_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("frozen algorithm source changed: " + name)
    for name, expected in request["algorithm_dependencies"].items():
        if sha256_file(name) != expected:
            raise ValueError("public prior dependency changed: " + name)
    if version == "REHAB_OFFLINE_V2":
        frozen = importlib.import_module("experiment_algorithm.v2.surrogate")
        policy_module = importlib.import_module("experiment_algorithm.v2.policy")
    else:
        frozen_config = importlib.import_module("config")
        if Path(frozen_config.__file__).resolve() != root / "config.py":
            raise ImportError("legacy training must run in its isolated subprocess")
        frozen = importlib.import_module("surrogate")
        policy_module = importlib.import_module("policy")
    from lower_limb_sim.config import L1, L2
    if request["geometry_m"] != [L1, L2]:
        raise ValueError("reviewed rig geometry must match this public prior; no silent rescaling")
    catalog = request["candidates"]
    points, frames = [], {}
    for item in catalog:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError("candidate trajectory hash changed")
        f = pd.read_csv(path)
        t = f.time_s.to_numpy(float)
        q = f[["q_hip_rad", "q_knee_rad"]].to_numpy(float)
        dq = f[["dq_hip_rad_s", "dq_knee_rad_s"]].to_numpy(float)
        ddq = f[["ddq_hip_rad_s2", "ddq_knee_rad_s2"]].to_numpy(float)
        if not np.isfinite(np.column_stack((t, q, dq, ddq))).all() or np.any(np.diff(t) <= 0):
            raise ValueError("invalid candidate kinematics")
        point = SimpleNamespace(candidate_id=item["candidate_id"], time_s=t,
            trajectory=SimpleNamespace(q=q, dq=dq, ddq=ddq), beta_flex=item["beta_flex"],
            beta_extend=item["beta_extend"], share_shift=item["share_shift"])
        points.append(point)
        frames[point.candidate_id] = f
    if len({p.candidate_id for p in points}) != len(points):
        raise ValueError("duplicate candidate ID")
    ref_id = request["reference_candidate_id"]
    reference = next(p for p in points if p.candidate_id == ref_id)
    phase = frames[ref_id].cycle_phase.to_numpy()
    if request["reference_sha256"] != next(item["sha256"] for item in catalog if item["candidate_id"] == ref_id):
        raise ValueError("reference hash differs from catalog")
    if any(len(p.time_s) != len(reference.time_s) or not np.allclose(p.time_s[[0, -1]], reference.time_s[[0, -1]], rtol=0, atol=TIME_ATOL) or
           not np.array_equal(frames[p.candidate_id].cycle_phase.to_numpy(), phase) for p in points):
        raise ValueError("frozen model needs common sample count and phase labels")
    class Domain:
        def __init__(self):
            self.reference = reference
            self.profile = SimpleNamespace(frozen=True, profile_id=request["setup_id"], version="REAL_LEG_V1",
                                            fingerprint=request["setup_sha256"])
            self.subject_reference = SimpleNamespace(phases=phase, reference_version=request["reference_sha256"])
        def __iter__(self):
            return iter(points)
        def __len__(self):
            return len(points)
        def by_id(self, name):
            return next(p for p in points if p.candidate_id == name)
    context = frozen.ModelContext(Domain())
    class DirectTorqueSurrogate(frozen.Surrogate):
        def _accepted(self, measurements):
            accepted, episodes, candidates = [], set(), set()
            for m in measurements:
                o = m.observation
                if not o.valid:
                    raise ValueError("invalid observation excluded from model")
                if o.episode_id in episodes or o.candidate_id in candidates:
                    raise ValueError("duplicate episode/candidate")
                episodes.add(o.episode_id)
                candidates.add(o.candidate_id)
                f = m.frame
                state = f.loc[:, frozen.STATE_COLUMNS].to_numpy(float)
                tau = f[["tau_hip_nm", "tau_knee_nm"]].to_numpy(float)
                scalar = np.array([o.endpoint_value, m.E2, m.peak_ratio], float)
                if not np.isfinite(state).all() or not np.isfinite(tau).all() or not np.isfinite(scalar).all():
                    raise ValueError("nonfinite direct measurement")
                index = context.index_by_id[o.candidate_id]
                matrix = frozen.mechanical.design_matrix(f, context.template)[0]
                accepted.append((m, index, scalar, f, matrix, tau))
            if not accepted:
                raise ValueError("valid direct joint-torque observation required")
            return accepted
    model = DirectTorqueSurrogate(context)
    if validate_only:
        return dict(status="ALGORITHM_INITIALIZED", method="FROZEN_RESIDUAL_GP_MEAN_GREEDY", algorithm_version=version,
            reference_candidate_id=ref_id, source_hashes_checked=True,
            public_prior=finite_json(frozen.mechanical.public_basis().provenance))
    baseline = None
    source_sampling = None
    def load_measurement(item, index):
        nonlocal baseline, source_sampling
        path = Path(item["measured_path"])
        if sha256_file(path) != item["measured_sha256"]:
            raise ValueError("measured training frame changed")
        f = pd.read_csv(path)
        point = context.domain.by_id(item["candidate_id"])
        time, phases = _series(f, "phase", "measured training frame")
        torque = _numeric(f, ("tau_hip_nm", "tau_knee_nm"), "measured training frame")
        command_time = (_numeric(f, ("command_time_s",), "measured training frame")[:, 0]
                        if "command_time_s" in f else time)
        if len(time) != len(point.time_s) or not np.allclose(command_time, point.time_s, rtol=0, atol=TIME_ATOL) or not np.array_equal(phases, phase):
            raise ValueError("measured training frame has incomplete candidate time/phase coverage")
        if index == 1 and item["candidate_id"] != ref_id:
            raise ValueError("first measurement must be the reference")
        candidate_sha = next(x["sha256"] for x in catalog if x["candidate_id"] == item["candidate_id"])
        metrics, sampling = measurement_endpoints(item, request.get("data_classification"), candidate_sha, f)
        if baseline is None:
            baseline, source_sampling = metrics, sampling
        elif sampling != source_sampling:
            raise ValueError("mixed reference/candidate endpoint sampling conventions")
        scores = endpoints(metrics, baseline)
        if "endpoints" in item and any(not np.isclose(scores[k], item["endpoints"].get(k, np.nan), rtol=1e-12, atol=1e-12) for k in scores):
            raise ValueError("history endpoints differ from the fixed measured reference")
        observation = SimpleNamespace(valid=True, episode_id=item["episode_id"],
            candidate_id=item["candidate_id"], trial_index=index, endpoint_value=scores["E3"],
            endpoint_uncertainty=None, beta_flex=point.beta_flex, beta_extend=point.beta_extend)
        return SimpleNamespace(observation=observation, E2=scores["E2"],
            peak_ratio=scores["peak_ratio"], feasible=(scores["E2"] <= 1.01 and scores["peak_ratio"] <= 1.1), frame=f)
    policy = policy_module.Policy(context, method="GREEDY", surrogate=model)
    mode = request.get("replay_mode", "fit_only")
    if mode == "strict_policy":
        if request.get("data_classification") not in CLASSIFICATIONS:
            raise ValueError("strict_policy requires an explicit data_classification")
        if request.get("attempt_ledger_complete") is not True:
            raise ValueError("strict_policy requires an explicit complete attempt ledger")
        attempts = request.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("strict_policy requires attempts")
        completed = [item for item in attempts if item.get("outcome") == "COMPLETED"]
        declared_history = request.get("history", [])
        if (len(declared_history) != len(completed) or any(
                any(a.get(key) != b.get(key) for key in
                    ("episode_id", "candidate_id", "measured_path", "measured_sha256", "endpoint_source"))
                for a, b in zip(declared_history, completed))):
            raise ValueError("history does not match completed attempts in the ledger")
        for item in attempts:
            if item.get("data_classification") != request["data_classification"]:
                raise ValueError("attempt data_classification mismatch")
            if item.get("outcome") == "COMPLETED" and item.get("status") not in COMPLETE_STATUSES:
                raise ValueError("completed attempt is not a validated complete measurement")
            for key in IDENTITY_KEYS:
                if item.get(key) != request[key]:
                    raise ValueError("attempt " + key + " mismatch")
        result = replay_attempts(policy, attempts, load_measurement)
        measurements = result.pop("measurements")
        action = asdict(result.pop("action"))
        replay_info = dict(policy_state_replayed=True, total_attempt_budget_accounted=True,
                           completed_rounds_only=False, **result)
        diagnostic_action = None
    elif mode == "fit_only":
        measurements = []
        for index, item in enumerate(request["history"], 1):
            measurements.append(load_measurement(item, index))
            model.fit(measurements)
        diagnostic_action = asdict(policy.next_action(measurements))
        action = None
        replay_info = dict(policy_state_replayed=False, total_attempt_budget_accounted=False,
                           completed_rounds_only=True)
    else:
        raise ValueError("replay_mode must be fit_only or strict_policy")
    predictions = model.predict_all() if measurements else None
    return finite_json(dict(status="POLICY_REPLAYED" if mode == "strict_policy" else "MODEL_UPDATED", method="FROZEN_RESIDUAL_GP_MEAN_GREEDY",
        model_fitted=bool(measurements), parameters_source="FITTED_HISTORY" if measurements else "PUBLIC_INITIAL",
        algorithm_version=version, replay_mode=mode, motion_approved=False, **replay_info,
        input_adapter="direct calibrated joint-torque sensor; no fictitious planar force",
        theta=model.theta, diagnostics=model.diagnostics, action=action, diagnostic_action=diagnostic_action,
        candidates=[p.candidate_id for p in points], predictions=predictions,
        algorithm_sha256=request["algorithm_sha256"],
        algorithm_dependencies=request["algorithm_dependencies"], history=request.get("history", []),
        attempts=request.get("attempts"), endpoint_sampling=source_sampling,
        setup_sha256=request["setup_sha256"], public_prior=frozen.mechanical.public_basis().provenance,
        runtime=dict(python=sys.version, packages={name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "mujoco")}),
        note="One reference episode does not establish optimization benefit or model reliability; recommendations are not motion approvals."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = train(request, validate_only=args.validate_only)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
