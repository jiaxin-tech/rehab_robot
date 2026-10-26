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


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = REPO_ROOT / "experiment_algorithm"
ALGORITHM_FILES = {"config.py", "surrogate.py", "policy.py"}
PACKAGE_DEPENDENCIES = ("__init__.py", "coupled_model.py",
                        "coordinate_consistent_model.py", "coordinates.py")


def repo_path(value):
    """Request-relative paths are always relative to this repository, not cwd."""
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve(strict=True)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_public_prior(provenance, dependency_hashes):
    """Bind the loaded XML and actual asset locations to the verified request."""
    model_path = repo_path(provenance["model_path"])
    asset_root = repo_path(provenance["asset_root"])
    asset_files = provenance["asset_files"]
    if not asset_root.is_dir() or not isinstance(asset_files, list) or not asset_files:
        raise ValueError("public prior requires an asset directory and complete asset provenance")
    used = {model_path}
    for name in asset_files:
        relative = Path(name)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("invalid public prior asset path: " + name)
        path = (asset_root / relative).resolve(strict=True)
        if not path.is_relative_to(asset_root):
            raise ValueError("public prior asset escapes its root: " + name)
        used.add(path)
    for path in used:
        if path not in dependency_hashes:
            raise ValueError("public prior actually used an unhashed dependency: " + str(path))
        if sha256_file(path) != dependency_hashes[path]:
            raise ValueError("public prior dependency changed after loading: " + str(path))
    if provenance["source_sha256"] != dependency_hashes[model_path]:
        raise ValueError("loaded public prior model does not match its verified hash")


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
    root = repo_path(request["algorithm_root"])
    if root != ALGORITHM_ROOT.resolve():
        raise ValueError("algorithm_root must identify this repository's experiment_algorithm package")
    if set(request["algorithm_sha256"]) != ALGORITHM_FILES or not request["algorithm_dependencies"]:
        raise ValueError("complete frozen algorithm and dependency hashes required")
    for name, expected in request["algorithm_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("frozen algorithm source changed: " + name)
    dependency_hashes = {}
    for name, expected in request["algorithm_dependencies"].items():
        path = repo_path(name)
        if sha256_file(path) != expected:
            raise ValueError("public prior dependency changed: " + name)
        dependency_hashes[path] = expected
    if not {root / name for name in PACKAGE_DEPENDENCIES} <= dependency_hashes.keys():
        raise ValueError("complete experiment_algorithm package dependency hashes required")
    # Check hashes before importing the learner or loading its XML.
    modules = {}
    for name in ("", "config", "coupled_model", "coordinates", "coordinate_consistent_model", "surrogate", "policy"):
        module = importlib.import_module("experiment_algorithm" + ("." + name if name else ""))
        expected = root / (name + ".py" if name else "__init__.py")
        if Path(module.__file__).resolve() != expected:
            raise ImportError("algorithm import does not match verified source: " + name)
        modules[name] = module
    frozen = modules["surrogate"]
    policy_module = modules["policy"]
    from lower_limb_sim.config import L1, L2
    if request["geometry_m"] != [L1, L2]:
        raise ValueError("reviewed rig geometry must match this public prior; no silent rescaling")
    catalog = request["candidates"]
    points, frames = [], {}
    for item in catalog:
        path = repo_path(item["path"])
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
    if any(len(p.time_s) != len(reference.time_s) or
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
    basis = frozen.mechanical.public_basis()
    validate_public_prior(basis.provenance, dependency_hashes)
    context = frozen.ModelContext(Domain())
    model = frozen.Surrogate(context)
    if validate_only:
        return dict(status="ALGORITHM_INITIALIZED", method="FROZEN_RESIDUAL_GP_MEAN_GREEDY",
            reference_candidate_id=ref_id, source_hashes_checked=True,
            public_prior=finite_json(frozen.mechanical.public_basis().provenance))
    measurements = []
    for index, item in enumerate(request["history"], 1):
        path = repo_path(item["measured_path"])
        if sha256_file(path) != item["measured_sha256"]:
            raise ValueError("measured training frame changed")
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
        algorithm_sha256=request["algorithm_sha256"],
        algorithm_dependencies=request["algorithm_dependencies"], history=request["history"],
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
