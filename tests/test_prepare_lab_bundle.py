from copy import deepcopy
import hashlib
import json

import pytest

import scripts.prepare_lab_bundle as bundle


def fixture_repo(tmp_path, monkeypatch):
    real_root = bundle.ROOT
    config = deepcopy(bundle.load(real_root / "config/robot_leg_round.json"))
    profile = bundle.load(real_root / "config/lab_next_visit.json")
    root = tmp_path / "rehab_robot"
    (root / "config").mkdir(parents=True)
    frozen = tmp_path / "standalone" / "frozen_algorithm"
    frozen.mkdir(parents=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    config["algorithm_root"] = str(frozen)
    for name in config["algorithm_sha256"]:
        file = frozen / name
        file.write_bytes(("# exact fixture " + name + "\n").encode())
        config["algorithm_sha256"][name] = digest(file)
    external = frozen.parent / "model" / "coupled_model.py"
    external.parent.mkdir()
    external.write_bytes(b"# exact external model\n")
    internal = root / "mechanics.py"
    internal.write_bytes(b"# exact repository dependency\n")
    config["algorithm_dependencies"] = {str(external): digest(external), str(internal): digest(internal)}
    candidate = root / "reference.csv"
    candidate.write_bytes(b"time_s,q_hip_rad\n0,0\n")
    config["candidates"] = [dict(candidate_id="reference", path=str(candidate), sha256=digest(candidate))]
    config["reference_candidate_id"] = "reference"
    (root / "config/robot_leg_round.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "config/lab_next_visit.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(bundle, "ROOT", root)
    return root, frozen, config


def test_portable_draft_copies_exact_bytes_without_reviewing_or_filling_facts(tmp_path, monkeypatch):
    root, frozen, original = fixture_repo(tmp_path, monkeypatch)
    source_bytes = (root / "config/robot_leg_round.json").read_bytes()
    output = bundle.build(root / "results/lab_pack")
    draft = bundle.load(output / "round_training_draft.json")
    assert draft["trial_kind"] == "dummy_leg_reference" and draft["reviewed"] is False
    assert draft["collision_event_max_age_s"] is None
    assert draft["max_leg_start_age_s"] is None
    assert {k: v["passed"] for k, v in draft["site_evidence"].items()} == {k: v["passed"] for k, v in original["site_evidence"].items()}
    for name, digest in draft["algorithm_sha256"].items():
        assert bundle.sha(output / draft["algorithm_root"] / name) == digest
    for name, digest in draft["algorithm_dependencies"].items():
        assert bundle.sha(root / name) == digest
    assert (root / "config/robot_leg_round.json").read_bytes() == source_bytes
    with pytest.raises(FileExistsError):
        bundle.build(output)


def test_changed_frozen_source_blocks_before_creating_bundle(tmp_path, monkeypatch):
    root, frozen, _ = fixture_repo(tmp_path, monkeypatch)
    (frozen / "policy.py").write_bytes(b"changed\n")
    output = root / "results/lab_pack"
    with pytest.raises(ValueError, match="algorithm changed"):
        bundle.build(output)
    assert not output.exists()
