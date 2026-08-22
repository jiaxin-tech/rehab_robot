from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
)
from .large_artifact_reproduction import (
    PROJECT_ROOT,
    artifact_entry,
    inspect_csv,
    load_large_artifact_manifest,
    reproduce_large_artifact,
    sha256_file as artifact_sha256,
    verify_artifact,
)
from .p2_revision_v2_research_prototype import DEFAULT_PROTOTYPE_CONTROLS
from .p2_v2_offline_research_prototype import DEFAULT_CONTROLS as V2_OFFLINE_CONTROLS


PAIR_PLAN_PATH = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "formal_artifacts"
    / "p2_v2_formal_research_protocol_v1"
    / "designated_local_validation_pair_plan.csv"
)
PAIR_PLAN_SHA256 = "ffaf01c65f9097bae35d165c25c2dddf5a617fd97835fd3fa5d50604c4beeb55"
CHECKPOINT_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "formal_artifacts"
    / "p2_checkpoint_and_large_artifact_reproducibility_v1"
)


def test_large_manifest_is_content_addressed_and_machine_independent() -> None:
    manifest = load_large_artifact_manifest()
    assert manifest["schema_version"] == "GENERATED_LARGE_ARTIFACT_MANIFEST_V1"
    assert manifest["normal_regression_requires_large_artifacts"] is False
    assert manifest["formal_artifact_reproduction_requires_large_artifacts"] is True
    assert len(manifest["artifacts"]) == 5
    for entry in manifest["artifacts"]:
        assert not Path(entry["expected_path"]).is_absolute()
        assert len(entry["sha256"]) == 64
        assert entry["required_for_normal_pytest"] is False
        assert entry["required_for_formal_reproduction"] is True


def test_optional_local_cache_is_verified_but_not_required() -> None:
    for entry in load_large_artifact_manifest()["artifacts"]:
        path = PROJECT_ROOT / entry["expected_path"]
        if path.exists():
            verify_artifact(path, entry, verify_sha=True)
        else:
            assert entry["required_for_normal_pytest"] is False


def test_normal_tests_do_not_name_large_csv_paths() -> None:
    normal_tests = sorted((PROJECT_ROOT / "lower_limb_sim").glob("test_*.py"))
    this_test = Path(__file__).resolve()
    filenames = {
        entry["expected_filename"]
        for entry in load_large_artifact_manifest()["artifacts"]
    }
    for test_path in normal_tests:
        if test_path.resolve() == this_test:
            continue
        source = test_path.read_text(encoding="utf-8")
        assert filenames.isdisjoint(
            {filename for filename in filenames if filename in source}
        ), test_path


def test_ignore_rules_are_exact_and_keep_small_scientific_outputs_visible() -> None:
    for entry in load_large_artifact_manifest()["artifacts"]:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", entry["expected_path"]],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert completed.returncode == 0
    small_output = (
        "lower_limb_sim/formal_artifacts/p2_revision_root_cause_audit_v1/"
        "truth_landscape_summary.csv"
    )
    completed = subprocess.run(
        ["git", "check-ignore", "-q", small_output],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert completed.returncode == 1


def test_representative_truth_regeneration_is_deterministic(tmp_path) -> None:
    first = reproduce_large_artifact(
        "truth_landscape_baseline",
        tmp_path / "first",
        representative_subset=True,
    )
    second = reproduce_large_artifact(
        "truth_landscape_baseline",
        tmp_path / "second",
        representative_subset=True,
    )
    assert artifact_sha256(first) == artifact_sha256(second)
    rows, schema = inspect_csv(first)
    assert 100 <= rows < artifact_entry("truth_landscape_baseline")[
        "expected_row_count"
    ]
    assert schema == artifact_entry("truth_landscape_baseline")["expected_schema"]


def test_pair_plan_and_frozen_scientific_identity_are_unchanged() -> None:
    assert hashlib.sha256(PAIR_PLAN_PATH.read_bytes()).hexdigest() == PAIR_PLAN_SHA256
    metadata = json.loads(
        (PAIR_PLAN_PATH.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["designated_local_pair_plan_count"] == 324
    assert metadata["designated_local_pair_plan_sha256"] == PAIR_PLAN_SHA256
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert ACTIVE_REFERENCE_SHA256 == (
        "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    )
    assert sha256_file(ACTIVE_REFERENCE_PATH) == ACTIVE_REFERENCE_SHA256


def test_p2_v2_remains_default_off_and_protected_packages_are_unchanged() -> None:
    DEFAULT_PROTOTYPE_CONTROLS.require_default_off()
    V2_OFFLINE_CONTROLS.require_default_off()
    completed = subprocess.run(
        ["git", "diff", "--", "hardware", "control", "collection", "safety"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""


def test_checkpoint_pathspecs_are_exact_disjoint_and_do_not_stage_large_files() -> None:
    checkpoint_sets = []
    for checkpoint in range(1, 5):
        rows = (
            CHECKPOINT_AUDIT_DIRECTORY / f"checkpoint_{checkpoint}_files.txt"
        ).read_text(encoding="utf-8").splitlines()
        assert rows == sorted(set(rows))
        assert rows
        assert all(not Path(row).is_absolute() and "*" not in row for row in rows)
        assert all((PROJECT_ROOT / row).is_file() for row in rows)
        checkpoint_sets.append(set(rows))
    assert sum(map(len, checkpoint_sets)) == len(set().union(*checkpoint_sets))
    excluded = {
        entry["expected_path"]
        for entry in load_large_artifact_manifest()["artifacts"]
    }
    assert excluded.isdisjoint(set().union(*checkpoint_sets))
    assert "lower_limb_sim/.DS_Store" not in set().union(*checkpoint_sets)


def test_checkpoint_three_is_self_contained_and_precedes_outcomes() -> None:
    checkpoint_three = set(
        (CHECKPOINT_AUDIT_DIRECTORY / "checkpoint_3_files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert {
        "lower_limb_sim/p2_v2_formal_research_protocol.py",
        "lower_limb_sim/run_p2_v2_formal_research_protocol.py",
        "lower_limb_sim/test_p2_v2_formal_research_protocol.py",
        "lower_limb_sim/formal_artifacts/p2_v2_formal_research_protocol_v1/designated_local_validation_pair_plan.csv",
        "lower_limb_sim/formal_artifacts/p2_v2_formal_research_protocol_v1/DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1.json",
        "lower_limb_sim/formal_artifacts/p2_v2_formal_research_protocol_v1/LOCAL_VALIDATION_PROTOCOL_REPORT.md",
        "lower_limb_sim/formal_artifacts/p2_v2_formal_research_protocol_v1/metadata.json",
    }.issubset(checkpoint_three)
    assert not any("p2_v2_offline_research_prototype_v1" in path for path in checkpoint_three)
    metadata = json.loads(
        (PAIR_PLAN_PATH.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["designated_local_outcomes_available"] is False
    protocol_source = (
        PROJECT_ROOT / "lower_limb_sim/p2_v2_formal_research_protocol.py"
    ).read_text(encoding="utf-8")
    assert "p2_v2_offline_research_prototype" not in protocol_source


def test_clean_checkout_prerequisites_are_exactly_in_checkpoint_two() -> None:
    checkpoint_two = set(
        (CHECKPOINT_AUDIT_DIRECTORY / "checkpoint_2_files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    prerequisites = {
        "lower_limb_sim/data/reference_trajectories/processed/reference_full_angles.csv",
        "lower_limb_sim/data/reference_trajectories/processed/detected_cycles.csv",
        "lower_limb_sim/data/reference_trajectories/processed/metadata.json",
        "lower_limb_sim/data/reference_local_active_asymmetric/state_domain_bounds.json",
    }
    assert prerequisites.issubset(checkpoint_two)
    for path in prerequisites:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert completed.returncode == 1


def test_checkpoint_task_created_no_prospective_data_artifact() -> None:
    prohibited_suffixes = (".csv", ".json", ".npy", ".npz")
    prohibited = []
    for path in CHECKPOINT_AUDIT_DIRECTORY.rglob("*"):
        lower = path.name.lower()
        if path.is_file() and "prospective" in lower and lower.endswith(prohibited_suffixes):
            prohibited.append(path)
    assert prohibited == []
