"""Read original experiment evidence from one content-addressed, verified archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/evidence.zip"
CACHE = ROOT / ".cache/evidence"
_READY = None


def _relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError("invalid evidence path")
    return p


def _read(archive, index, name):
    item = index["entries"][name]
    value = archive.read("blobs/" + item["sha256"])
    if len(value) != item["size"] or hashlib.sha256(value).hexdigest() != item["sha256"]:
        raise ValueError("evidence SHA-256 mismatch: " + name)
    return value


def read_bytes(name):
    """Read one original by its logical name from index.json."""
    _relative(name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        return _read(archive, json.loads(archive.read("index.json")), name)


def _atomic_write(destination, data):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".evidence-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def runtime_root():
    """Materialize the pinned reference inputs in an ignored, disposable cache."""
    global _READY
    identity = (ARCHIVE.stat().st_mtime_ns, ARCHIVE.stat().st_size)
    if _READY != identity:
        with zipfile.ZipFile(ARCHIVE) as archive:
            index = json.loads(archive.read("index.json"))
            for name in index["runtime"]:
                p = _relative(name)
                if p.parts[0] != "repository":
                    raise ValueError("runtime evidence must originate in the repository")
                target = CACHE.joinpath(*p.parts)
                expected = index["entries"][name]["sha256"]
                if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                    continue
                _atomic_write(target, _read(archive, index, name))
        _READY = identity
    return CACHE / "repository"


def artifact_path(relative):
    return runtime_root().joinpath(*_relative(relative).parts)


def verify():
    """Verify every logical original, including the 25 source-hash-bound CSVs."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        index = json.loads(archive.read("index.json"))
        checked = set()
        for name, item in index["entries"].items():
            _relative(name)
            if item["sha256"] not in checked:
                _read(archive, index, name)
                checked.add(item["sha256"])
        return dict(originals=len(index["entries"]), unique_blobs=len(checked),
                    raw_csvs=sum(n.startswith("raw/") for n in index["entries"]),
                    source_commit=index["source_commit"], limitations=index["limitations"])



def archive_session(directory):
    """Pack a completed round without deleting logs or one-use release ledgers."""
    directory = Path(directory).resolve(strict=True)
    if not directory.is_relative_to(ROOT / "results"):
        raise ValueError("session must be inside this repository's results directory")
    state = json.loads((directory / "round_status.json").read_text(encoding="utf-8"))
    if state.get("status") not in {"ROUND_COMPLETE", "CAPTURE_ONLY_COMPLETE", "EMPTY_LOAD_COMMISSIONING_COMPLETE"}:
        raise ValueError("only closed, completed rounds may be consolidated")
    namespace = "sessions/" + directory.relative_to(ROOT / "results").as_posix()
    inputs = []
    for source in sorted(directory.rglob("*")):
        if source.is_symlink() or not source.resolve().is_relative_to(directory):
            raise ValueError("session contains a link outside its directory")
        if source.is_file():
            before = source.stat()
            value = source.read_bytes()
            after = source.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError("session changed while reading: " + str(source))
            inputs.append((namespace + "/" + source.relative_to(directory).as_posix(), value))
    # This is an offline command. The exclusive lock prevents competing updates.
    lock = ARCHIVE.with_suffix(".lock")
    with lock.open("x"):
        pass
    temporary = None
    try:
        with zipfile.ZipFile(ARCHIVE) as old:
            index = json.loads(old.read("index.json"))
            blobs = {item["sha256"] for item in index["entries"].values()}
            additions = {}
            for name, value in inputs:
                _relative(name)
                sha = hashlib.sha256(value).hexdigest()
                item = dict(sha256=sha, size=len(value), source=name)
                if name in index["entries"] and index["entries"][name] != item:
                    raise ValueError("archived session identity already has different content: " + name)
                index["entries"][name] = item
                if sha not in blobs:
                    additions[sha] = value
            fd, temporary = tempfile.mkstemp(dir=ARCHIVE.parent, suffix=".zip")
            os.close(fd)
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as new:
                for sha in sorted(blobs):
                    value = old.read("blobs/" + sha)
                    if hashlib.sha256(value).hexdigest() != sha:
                        raise ValueError("existing evidence hash mismatch")
                    new.writestr("blobs/" + sha, value)
                for sha, value in additions.items():
                    new.writestr("blobs/" + sha, value)
                new.writestr("index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as checked:
            if checked.testzip():
                raise ValueError("new evidence archive failed CRC verification")
        os.replace(temporary, ARCHIVE)
        summary_path = ROOT / "data/experiment_summary.csv"
        records = []
        if summary_path.exists():
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
        by_key = {(row["source"], row["metric"]): row for row in records}
        def summarize(value, source, sha, prefix=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    summarize(child, source, sha, prefix + "." + key if prefix else key)
            elif not isinstance(value, list):
                by_key[source, prefix] = dict(source=source, metric=prefix,
                    value=json.dumps(value, ensure_ascii=False, allow_nan=False), source_sha256=sha)
        for name, value in inputs:
            if name.endswith("/round_status.json"):
                record = json.loads(value)
                record = {key: record[key] for key in ("status", "training_status", "software_verification_only", "endpoints", "quality_summary") if key in record}
            elif name.endswith("/measured_metrics.json"):
                record = json.loads(value)
            else:
                continue
            summarize(record, name, hashlib.sha256(value).hexdigest())
        table = io.StringIO(newline="")
        writer = csv.DictWriter(table, fieldnames=("source", "metric", "value", "source_sha256"))
        writer.writeheader()
        writer.writerows(by_key.values())
        _atomic_write(summary_path, table.getvalue().encode("utf-8-sig"))
        return dict(session=namespace, files=len(inputs), new_blobs=len(additions), original_logs_retained=True)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    pack = commands.add_parser("archive-session")
    pack.add_argument("--session", type=Path, required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--prefix", default="raw/")
    extract = commands.add_parser("extract")
    extract.add_argument("name")
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    elif args.command == "archive-session":
        print(json.dumps(archive_session(args.session), ensure_ascii=False, indent=2))
    elif args.command == "list":
        with zipfile.ZipFile(ARCHIVE) as archive:
            index = json.loads(archive.read("index.json"))
            for name in sorted(index["entries"]):
                if name.startswith(args.prefix):
                    print(name)
    else:
        data = read_bytes(args.name)
        with args.output.open("xb") as stream:
            stream.write(data)


if __name__ == "__main__":
    main()
