"""Copy byte-identical frozen sources and make portable unapproved lab drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from scripts.lab_session import ROOT, load, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(name, base):
    """Prefer real existing paths; relocate only missing old-checkout paths.

    A temporary directory can itself contain a parent named rehab_robot. Splitting
    that existing absolute path at its first marker duplicates the checkout.
    """
    source = (base / name).resolve()
    if source.exists():
        return source
    normalized = str(name).replace("\\", "/")
    if "/rehab_robot/" in normalized:
        relocated = (ROOT / normalized.rsplit("/rehab_robot/", 1)[1]).resolve()
        if relocated.exists():
            return relocated
    return source


def build(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("bundle must be within this checkout for portable repository-relative dependencies")
    config_path = ROOT / "config/robot_leg_round.json"
    config = load(config_path)
    copies = []
    dependencies = {}
    algorithm_root = source_path(config["algorithm_root"], config_path.parent)
    frozen_root = output / "frozen" / algorithm_root.name
    for name, digest in config["algorithm_sha256"].items():
        source = algorithm_root / name
        if sha(source) != digest:
            raise ValueError("frozen algorithm changed: " + str(source))
        copies.append((source, frozen_root / name, digest))
    for name, digest in config["algorithm_dependencies"].items():
        source = source_path(name, ROOT)
        if source.is_relative_to(ROOT):
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
        source = source_path(item["path"], config_path.parent)
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
