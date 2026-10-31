"""Read the compact experiment configuration; adapt paths for shared modules."""
from __future__ import annotations
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "config/experiment.json"
CACHE_ROOT = ROOT / ".cache/config"
ALIASES = {"experiment_safety": "motion", "rehab_frame_config": "frame",
           "twincat_leg_capture": "leg", "leg_joint_calibration": "calibration",
           "leg_round_analysis": "analysis", "live_observation_acquisition_20261018": "acquisition"}


def configuration(path=BUNDLE_PATH):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be an object")
    return value


def load_section(name, *, resolved=False, config_path=BUNDLE_PATH):
    c = configuration(config_path)
    key = ALIASES.get(name, name)
    value = copy.deepcopy(c[key])
    if key == "calibration":
        value = {"joints": value}
    if key == "leg":
        value["ads_dll_path"] = str((ROOT / value["ads_dll_path"]).resolve())
    if key == "acquisition":
        r = c["robot"]
        value = dict(connection=dict(robot_ip=r["ip"], local_ip=r["local_ip"], robot_class=r["robot_class"]),
                     expected_identity=value["expected_identity"], budgets=value["budgets"],
                     allowed_power_states=["on", "off"],
                     allowed_operation_states=["idle", "rtcontrolling"])
    return value


def section_path(name, *, config_path=BUNDLE_PATH):
    """A disposable JSON view for older modules accepting a file path."""
    value = load_section(name, config_path=config_path)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / (name + ".json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_config(path, *, section):
    """Read a section directly, also accepting standalone diagnostic configs."""
    source = resolve_config_path(path)
    value = configuration(source)
    key = ALIASES.get(section, section)
    if "robot" in value and key in value:
        return load_section(section, config_path=source)
    return value


def resolve_config_path(path, *, section=None):
    source = Path(path).expanduser().resolve()
    if source.parent == ROOT / "config" and source.stem in ALIASES and not source.exists():
        return BUNDLE_PATH
    if source.is_relative_to(ROOT):
        relative = source.relative_to(ROOT).as_posix()
        if relative.startswith(("reference_release/", "lower_limb_sim/data/", "external_simulation/calibration/")):
            from utils.assets import artifact_path
            return artifact_path(relative)
    return source
