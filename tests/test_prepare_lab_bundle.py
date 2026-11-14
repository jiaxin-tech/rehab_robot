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


def test_existing_absolute_paths_with_rehab_robot_ancestor_are_not_relocated_twice(tmp_path, monkeypatch):
    nested = tmp_path / "rehab_robot" / ".cache" / "fixture"
    nested.mkdir(parents=True)
    root, frozen, _ = fixture_repo(nested, monkeypatch)
    output = bundle.build(root / "results/portable")
    assert output.is_dir()
    assert bundle.sha(output / "frozen/model/coupled_model.py") == bundle.sha(frozen.parent / "model/coupled_model.py")


def test_repository_relative_dependencies_and_config_relative_sources_are_supported(tmp_path, monkeypatch):
    root, frozen, config = fixture_repo(tmp_path, monkeypatch)
    config["algorithm_root"] = str(frozen.relative_to(root.parent)).replace("\\", "/")
    # The algorithm root is relative to config/, so reach the fixture sibling.
    config["algorithm_root"] = "../../" + config["algorithm_root"]
    internal_digest = config["algorithm_dependencies"].pop(str(root / "mechanics.py"))
    config["algorithm_dependencies"]["mechanics.py"] = internal_digest
    config["candidates"][0]["path"] = "../reference.csv"
    (root / "config/robot_leg_round.json").write_text(json.dumps(config), encoding="utf-8")
    output = bundle.build(root / "results/portable_relative")
    assert bundle.load(output / "round_training_draft.json")["algorithm_dependencies"]["mechanics.py"] == internal_digest
