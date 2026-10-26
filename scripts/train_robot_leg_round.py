"""Train the frozen residual-GP mean Greedy in a separate, hardware-free process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = REPO_ROOT / "experiment_algorithm"
ALGORITHM_FILES = {"config.py", "surrogate.py", "policy.py"}
PACKAGE_DEPENDENCIES = ("__init__.py", "coupled_model.py",
                        "coordinate_consistent_model.py", "coordinates.py")


def repo_path(value):
    """Request-relative paths are always relative to this repository, not cwd."""
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve(strict=True)


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
    from experiment_algorithm import surrogate as frozen
    from experiment_algorithm import policy as policy_module
    catalog = request["candidates"]
    points, frames = [], {}
    for item in catalog:
        path = repo_path(item["path"])
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
    if any(len(p.time_s) != len(reference.time_s) or
           not np.array_equal(frames[p.candidate_id].cycle_phase.to_numpy(), phase) for p in points):
        raise ValueError("frozen model needs common sample count and phase labels")
    class Domain:
        def __init__(self):
            self.reference = reference
            self.profile = SimpleNamespace(frozen=True, profile_id=request.get("setup_id", "dummy_leg"), version="REAL_LEG_V1",
                                            fingerprint="experiment")
            self.subject_reference = SimpleNamespace(phases=phase, reference_version=ref_id)
        def __iter__(self):
            return iter(points)
        def __len__(self):
            return len(points)
        def by_id(self, name):
            return next(p for p in points if p.candidate_id == name)
    basis = frozen.mechanical.public_basis()
    context = frozen.ModelContext(Domain())
    model = frozen.Surrogate(context)
    if validate_only:
        return dict(status="ALGORITHM_INITIALIZED", method="FROZEN_RESIDUAL_GP_MEAN_GREEDY",
            reference_candidate_id=ref_id,
            public_prior=finite_json(frozen.mechanical.public_basis().provenance))
    measurements = []
    for index, item in enumerate(request["history"], 1):
        path = repo_path(item["measured_path"])
        f = pd.read_csv(path)
        scores = item["endpoints"]
        point = context.domain.by_id(item["candidate_id"])
        observation = SimpleNamespace(valid=True, episode_id=item["episode_id"],
            candidate_id=item["candidate_id"], trial_index=index, endpoint_value=scores["E3"],
            endpoint_uncertainty=None, beta_flex=point.beta_flex, beta_extend=point.beta_extend)
        measurements.append(SimpleNamespace(observation=observation, E2=scores["E2"],
            peak_ratio=scores["peak_ratio"], feasible=(scores["E2"] <= 1.01 and scores["peak_ratio"] <= 1.1), frame=f))
        # Append one complete episode at a time, preserving prequential diagnostics.
        model.fit(measurements)
    policy = policy_module.Policy(context, method="GREEDY", surrogate=model)
    action = policy.next_action(measurements)
    predictions = model.predict_all()
    return finite_json(dict(status="MODEL_UPDATED", method="FROZEN_RESIDUAL_GP_MEAN_GREEDY",
        input_adapter="direct calibrated joint-torque sensor; no fictitious planar force",
        theta=model.theta, diagnostics=model.diagnostics, action=asdict(action),
        candidates=[p.candidate_id for p in points], predictions=predictions,
        history=request["history"], public_prior=frozen.mechanical.public_basis().provenance,
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
