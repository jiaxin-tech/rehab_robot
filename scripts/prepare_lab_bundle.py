"""Copy byte-identical frozen sources and make portable unapproved lab drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from config.bundle import resolve_config_path
import shutil

from scripts.lab_session import ROOT, load, write


def sha(path):
    return hashlib.sha256(resolve_config_path(path).read_bytes()).hexdigest()


def build(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("bundle must be within this checkout for portable repository-relative dependencies")
    config_path = resolve_config_path(ROOT / "config/robot_leg_round.json")
    config = load(config_path)
    copies = []
    dependencies = {}
    algorithm_root = Path(config["algorithm_root"])
    frozen_root = output / "frozen" / algorithm_root.name
    for name, digest in config["algorithm_sha256"].items():
        source = algorithm_root / name
        if sha(source) != digest:
            raise ValueError("frozen algorithm changed: " + str(source))
        copies.append((source, frozen_root / name, digest))
    for name, digest in config["algorithm_dependencies"].items():
        source = Path(name)
        suffix = name.replace("\\", "/").split("/rehab_robot/", 1)
        if source.is_relative_to(ROOT) and not source.is_relative_to(algorithm_root):
            target = source
        elif len(suffix) == 2 and not source.is_file():
            source = ROOT / suffix[1]
            target = source
        else:
            # The standalone model relies on this original sibling layout.
            try:
                relative = source.relative_to(algorithm_root.parent)
            except ValueError as exc:
                raise ValueError("unknown external dependency layout: " + name) from exc
            target = output / "frozen" / relative
            copies.append((source, target, digest))
        if sha(source) != digest:
            raise ValueError("dependency bytes differ: " + str(source))
        dependencies[os.path.relpath(target, ROOT).replace("\\", "/")] = digest
    # Validate all inputs before creating or copying any files.
    candidates = []
    for item in config["candidates"]:
        source = Path(item["path"])
        suffix = item["path"].replace("\\", "/").split("/rehab_robot/", 1)
        if len(suffix) == 2 and not source.is_file():
            source = ROOT / suffix[1]
        if sha(source) != item["sha256"]:
            raise ValueError("candidate source mismatch")
        candidates.append(dict(item, path=os.path.relpath(source, output).replace("\\", "/")))
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source, target, digest in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise ValueError("copy checksum mismatch")
        manifest.append(dict(path=os.path.relpath(target, ROOT).replace("\\", "/"), sha256=digest))
    draft = dict(config, reviewed=False, trial_kind="dummy_leg_reference", realtime_preparation="same_session", algorithm_dependencies=dependencies,
                 algorithm_root=os.path.relpath(frozen_root, output), candidates=candidates)
    for key in ("leg_config", "calibration", "analysis", "safety_config", "frame_config", "wrench_config"):
        draft[key] = os.path.relpath((config_path.parent / config[key]).resolve(), output).replace("\\", "/")
    draft["site_evidence"] = {name: dict(record, path=os.path.relpath(
        (config_path.parent / record["path"]).resolve(), output).replace("\\", "/"))
        for name, record in config["site_evidence"].items()}
    draft["note"] += " | Portable training draft only. No review, evidence, calibration or missing timing value was filled in."
    write(output / "round_training_draft.json", draft)
    profile = load(ROOT / "config/lab_next_visit.json")
    profile["round_config"] = os.path.relpath(output / "round_training_draft.json", ROOT).replace("\\", "/")
    write(output / "launcher.json", profile)
    write(output / "frozen_manifest.json", dict(source_config_sha256=sha(config_path),
        frozen_copies=manifest, dependency_sha256=dependencies,
        approved=False, source_bytes_changed=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
