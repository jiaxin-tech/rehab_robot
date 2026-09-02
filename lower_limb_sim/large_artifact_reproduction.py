"""Content-addressed reproduction for generated large offline truth artifacts.

This wrapper delegates scientific calculations to the existing formal runners.
It does not define a new model, objective, candidate policy, or prospective case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import pandas as pd

from .decision_relevant_global_model_reliability import (
    build_trajectory_component_cache,
    geometrically_valid_parameter_lattice,
)
from .p2_revision_root_cause_audit import audit_subject_truth_landscape
from .run_decision_relevant_global_model_reliability import (
    ANALYSIS_CASES,
    DEFAULT_PARAMETER_MAP_PATH,
    _limited_lattice_for_test,
    run_characterization,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_MANIFEST_PATH = (
    MODULE_DIR / "formal_artifacts" / "GENERATED_LARGE_ARTIFACT_MANIFEST.json"
)
ROOT_TRUTH_PREFIX = "truth_landscape_"
MANAGEMENT_POLICY = "REGENERABLE_CONTENT_ADDRESSED_ARTIFACT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_large_artifact_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("management_policy") != MANAGEMENT_POLICY:
        raise ValueError("unexpected large-artifact management policy")
    if payload.get("normal_regression_requires_large_artifacts") is not False:
        raise ValueError("normal regression must not require large artifacts")
    return payload


def artifact_entry(
    logical_artifact_id: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_large_artifact_manifest() if manifest is None else manifest
    matches = [
        item
        for item in registry["artifacts"]
        if item["logical_artifact_id"] == logical_artifact_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate logical artifact id: {logical_artifact_id}")
    return dict(matches[0])


def inspect_csv(path: Path) -> tuple[int, list[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            schema = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        return sum(1 for _ in reader), schema


def verify_artifact(path: Path, entry: dict[str, Any], *, verify_sha: bool) -> None:
    row_count, schema = inspect_csv(path)
    if row_count != int(entry["expected_row_count"]):
        raise RuntimeError(
            f"row-count mismatch for {entry['logical_artifact_id']}: {row_count}"
        )
    if schema != list(entry["expected_schema"]):
        raise RuntimeError(f"schema mismatch for {entry['logical_artifact_id']}")
    if verify_sha and sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {entry['logical_artifact_id']}")


def _representative_lattice(parameter_map_path: Path) -> pd.DataFrame:
    full = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
    # The existing 100-point diagnostic subset is known to retain model-supported
    # points; a 27-point local-only subset can contain none, making the existing
    # support-distance diagnostic undefined.
    subset = _limited_lattice_for_test(full, 100)
    required = full.loc[
        full["hip_delta"].eq(0.0)
        & full["knee_delta"].eq(-5.0)
        & full["phase_delta"].eq(0.0)
    ]
    return (
        pd.concat((subset, required), ignore_index=True)
        .drop_duplicates("trajectory_id")
        .sort_values("trajectory_id", kind="mergesort")
        .reset_index(drop=True)
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def reproduce_large_artifact(
    logical_artifact_id: str,
    output_dir: Path,
    *,
    representative_subset: bool = False,
    verify_manifest: bool = False,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> Path:
    """Reproduce one artifact without overwriting an existing output file."""

    entry = artifact_entry(logical_artifact_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".representative.csv" if representative_subset else ".csv"
    stem = Path(entry["expected_filename"]).stem
    target = output / f"{stem}{suffix}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    if logical_artifact_id == "global_prediction_truth_comparison":
        with tempfile.TemporaryDirectory(prefix="p2_global_truth_") as temporary:
            temporary_output = Path(temporary)
            run_characterization(
                temporary_output,
                parameter_map_path,
                # The existing report layer expects its full nine-case registry;
                # representative mode truncates only the parameter lattice.
                analysis_cases=ANALYSIS_CASES,
                maximum_points=100 if representative_subset else None,
            )
            shutil.copy2(
                temporary_output / entry["expected_filename"],
                target,
            )
    elif logical_artifact_id.startswith(ROOT_TRUTH_PREFIX):
        subject_id = logical_artifact_id.removeprefix(ROOT_TRUTH_PREFIX)
        full = geometrically_valid_parameter_lattice(pd.read_csv(parameter_map_path))
        lattice = _representative_lattice(parameter_map_path) if representative_subset else full
        cache = build_trajectory_component_cache(lattice)
        audit = audit_subject_truth_landscape(subject_id, lattice, cache)
        _write_csv(target, audit.landscape)
    else:
        raise KeyError(f"unsupported artifact: {logical_artifact_id}")

    if representative_subset:
        rows, schema = inspect_csv(target)
        if rows <= 0 or schema != list(entry["expected_schema"]):
            raise RuntimeError("representative regeneration schema check failed")
    else:
        verify_artifact(target, entry, verify_sha=verify_manifest)
    return target


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate one content-addressed large offline truth artifact."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-subset", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    target = reproduce_large_artifact(
        args.artifact,
        args.output_dir,
        representative_subset=args.representative_subset,
        verify_manifest=args.verify_manifest,
        parameter_map_path=args.parameter_map,
    )
    rows, _ = inspect_csv(target)
    print(f"artifact={args.artifact}")
    print(f"output={target}")
    print(f"rows={rows}")
    print(f"sha256={sha256_file(target)}")
    print(f"representative_subset={str(args.representative_subset).lower()}")
    print("prospective_validation_started=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
