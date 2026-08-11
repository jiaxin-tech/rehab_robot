"""Small, side-effect-free provenance helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def current_git_commit(*, repository: str | Path | None = None) -> str | None:
    """Return the current commit without modifying Git or raising on absence."""

    repository_path = (
        _REPOSITORY_ROOT
        if repository is None
        else Path(repository).expanduser().resolve()
    )
    command = ["git", "-C", str(repository_path), "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


__all__ = ["current_git_commit"]
