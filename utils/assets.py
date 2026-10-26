"""Read experiment data and public model assets from a normal ZIP archive."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/experiment_data.zip"
CACHE = ROOT / ".cache/data"
_READY = None


def _relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError("invalid archive path")
    return p


def read_bytes(name):
    _relative(name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        return archive.read(name)


def runtime_root():
    global _READY
    identity = (ARCHIVE.stat().st_mtime_ns, str(CACHE))
    if _READY != identity:
        with zipfile.ZipFile(ARCHIVE) as archive:
            for name in archive.namelist():
                if name.startswith(("repository/reference_release/", "repository/lower_limb_sim/data/",
                                    "repository/.cache/myosuite-assets/", "repository/external_simulation/calibration/")) or name == "repository/bone_return_3_leg.csv":
                    destination = CACHE / _relative(name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
        _READY = identity
    return CACHE / "repository"


def artifact_path(relative):
    return runtime_root() / _relative(relative)


def archive_session(session):
    """Store a closed experiment once; retain local raw streams."""
    session = Path(session).resolve()
    state = json.loads((session / "round_status.json").read_text(encoding="utf-8"))
    if state.get("status") not in ("CAPTURE_COMPLETE", "ROUND_COMPLETE"):
        raise ValueError("only a completed experiment can be archived")
    name = "sessions/" + session.name + "/"
    files = [p for p in session.rglob("*") if p.is_file()]
    with zipfile.ZipFile(ARCHIVE, "a", zipfile.ZIP_DEFLATED) as archive:
        if any(p.startswith(name) for p in archive.namelist()):
            raise FileExistsError("session already archived; choose a new session name")
        for path in files:
            archive.write(path, name + path.relative_to(session).as_posix())
    with (ROOT / "data/experiment_summary.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for key, value in state.get("metrics", {}).items():
            writer.writerow([name, key, json.dumps(value, ensure_ascii=False, allow_nan=False)])
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "extract", "archive-session"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args(argv)
    if args.action == "list":
        with zipfile.ZipFile(ARCHIVE) as archive:
            print("\n".join(n for n in archive.namelist() if n.startswith(args.prefix)))
    elif args.action == "extract":
        if not args.name or not args.output:
            parser.error("extract requires NAME --output PATH")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(read_bytes(args.name))
    else:
        if not args.session:
            parser.error("archive-session requires --session PATH")
        print(archive_session(args.session))


if __name__ == "__main__":
    main()
