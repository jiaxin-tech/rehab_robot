"""Single authority and verified JSON views for existing Path/hash APIs.

Test migration: load_section(name) supplies a dict; section_path(name) supplies
an actual file; export_sections(tmp_path / 'config') creates mutable fixtures.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "config/experiment.json"
CACHE_ROOT = ROOT / ".cache/config-bundle"
EVIDENCE_PREFIXES = ("reference_release/", "lower_limb_sim/data/", ".cache/myosuite-assets/")
SECTIONS = (
    "empty_load_commissioning_safety_20261018", "empty_load_live_observation_20261018",
    "experiment_safety", "formal_experiment_manifest", "lab_next_visit",
    "leg_joint_calibration", "leg_round_analysis", "live_observation_acquisition_20261018",
    "live_observation_acquisition_template", "python_source_scope_manifest",
    "real_identification_config", "reference_commissioning", "rehab_frame_config",
    "robot_leg_round", "static_validation_logging", "twincat_leg_capture",
    "static_validation_record_schema_v1", "site_evidence_records",
)
ROUND_PATHS = ("leg_config", "calibration", "analysis", "safety_config", "frame_config", "wrench_config")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _snapshot():
    raw = BUNDLE_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    if (not isinstance(data, dict) or set(data) != {"schema_version", "sections"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1):
        raise ValueError("experiment bundle schema mismatch")
    sections = data["sections"]
    if not isinstance(sections, dict) or set(sections) != set(SECTIONS):
        raise ValueError("experiment bundle requires exactly the named sections")
    if any(not isinstance(value, dict) for value in sections.values()):
        raise ValueError("configuration sections must be JSON objects")
    for filename, content in sections["site_evidence_records"].items():
        if (not filename or filename in (".", "..") or "/" in filename or chr(92) in filename or ":" in filename
                or PureWindowsPath(filename).drive or not isinstance(content, str)):
            raise ValueError("site evidence must map plain filenames to UTF-8 text")
        content.encode("utf-8")
    json.dumps(data, allow_nan=False)
    digest = hashlib.sha256(raw + Path(__file__).read_bytes()).hexdigest()
    return sections, CACHE_ROOT / digest


def _alias(path):
    if path.parent == ROOT / "config" and path.stem in SECTIONS and path.suffix == ".json":
        return path.stem
    if path == ROOT / "scripts/static_validation_record_schema_v1.json":
        return "static_validation_record_schema_v1"
    return None


def _render(name, sections, directory):
    value = copy.deepcopy(sections[name])

    def path(item):
        if item is None:
            return None
        if not isinstance(item, str) or not item:
            raise ValueError("configuration path must be a nonempty repository-relative string")
        source = Path(item)
        if source.is_absolute() or PureWindowsPath(item).drive or chr(92) in item:
            raise ValueError(f"configuration path must be portable: {item}")
        source = (ROOT / source).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError(f"configuration path escapes repository: {item}")
        section = _alias(source)
        if section:
            return str(directory / f"{section}.json")
        relative = source.relative_to(ROOT).as_posix()
        if relative.startswith("site_evidence/"):
            return str(directory / relative)
        if relative.startswith(EVIDENCE_PREFIXES):
            from utils.evidence import artifact_path
            return str(artifact_path(relative))
        return str(source)

    if name == "robot_leg_round":
        for key in (*ROUND_PATHS, "algorithm_root"):
            value[key] = path(value[key])
        value["algorithm_dependencies"] = {path(k): v for k, v in value["algorithm_dependencies"].items()}
        for record in [*value["candidates"], *value["site_evidence"].values()]:
            record["path"] = path(record["path"])
    elif name == "lab_next_visit":
        for key in ("leg_config", "round_config", "observation_wrench_config"):
            value[key] = path(value[key])
        for key in ("review_request", "release", "reference", "anchor", "safety_config", "frame_config", "wrench_config"):
            value["empty_load"][key] = path(value["empty_load"][key])
    elif name == "reference_commissioning":
        value["trajectory"]["path"] = path(value["trajectory"]["path"])
    elif name == "static_validation_logging":
        value["record_schema"] = path(value["record_schema"])
    elif name == "formal_experiment_manifest":
        value["reference_release_manifest"] = path(value["reference_release_manifest"])
    elif name == "twincat_leg_capture":
        value["ads_dll_path"] = path(value["ads_dll_path"])
    elif "evidence" in value and (name.startswith("live_observation_acquisition") or name == "empty_load_live_observation_20261018"):
        value["evidence"] = {key: path(item) for key, item in value["evidence"].items()}
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_view(destination, expected):
    if destination.exists():
        if destination.read_bytes() != expected:
            raise PermissionError(f"materialized configuration changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".view-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(expected)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _materialize(sections, directory):
    for name in SECTIONS:
        _write_view(directory / f"{name}.json", _render(name, sections, directory))
    for filename, content in sections["site_evidence_records"].items():
        _write_view(directory / "site_evidence" / filename, content.encode("utf-8"))


def section_path(name: str) -> Path:
    """Return a verified real JSON file for a named section."""
    if name not in SECTIONS:
        raise ValueError(f"unknown configuration section: {name}")
    sections, directory = _snapshot()
    _materialize(sections, directory)
    return directory / f"{name}.json"


def resolve_config_path(path: str | Path, *, section: str | None = None) -> Path:
    """Resolve exact aliases; other run/release files pass through.

    experiment.json needs an explicit section. Cached snapshots must match
    current authority. Historical approvals cannot select an old cache.
    """
    source = Path(path).expanduser().resolve()
    name = _alias(source)
    if source == BUNDLE_PATH:
        if section is None:
            raise ValueError("experiment.json requires an explicit configuration section")
        return section_path(section)
    if name:
        return section_path(name)
    if source.is_relative_to(CACHE_ROOT):
        sections, directory = _snapshot()
        is_section = source.parent == directory and source.stem in SECTIONS and source.suffix == ".json"
        is_evidence = source.parent == directory / "site_evidence" and source.name in sections["site_evidence_records"]
        if not (is_section or is_evidence):
            raise PermissionError("stale or unknown materialized configuration; prepare and review a new release")
        _materialize(sections, directory)
    if source.is_relative_to(ROOT):
        relative = source.relative_to(ROOT).as_posix()
        if relative.startswith("site_evidence/"):
            sections, directory = _snapshot()
            if source.parent != ROOT / "site_evidence" or source.name not in sections["site_evidence_records"]:
                raise FileNotFoundError(f"site evidence is not in experiment.json: {relative}")
            _materialize(sections, directory)
            return directory / relative
        if relative.startswith(EVIDENCE_PREFIXES):
            from utils.evidence import artifact_path
            return artifact_path(relative)
    return source


def load_section(name: str, *, resolved: bool = False) -> dict:
    """Independent dict: raw repo paths by default, actual paths optionally."""
    if name not in SECTIONS:
        raise ValueError(f"unknown configuration section: {name}")
    if resolved:
        return json.loads(section_path(name).read_text(encoding="utf-8"))
    sections, _ = _snapshot()
    return copy.deepcopy(sections[name])


def export_sections(directory: str | Path) -> dict[str, Path]:
    """Create mutable legacy-layout fixtures in a NEW test directory.

    files = export_sections(tmp_path / 'config'); mutate files['robot_leg_round'].
    Cross references point into this export. This helper does not approve runs.
    """
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT / "config") or directory.is_relative_to(CACHE_ROOT):
        raise ValueError("fixture export cannot overwrite authority or managed cache")
    sections, _ = _snapshot()
    directory.mkdir(parents=True, exist_ok=False)
    _materialize(sections, directory)
    return {name: directory / f"{name}.json" for name in SECTIONS}
