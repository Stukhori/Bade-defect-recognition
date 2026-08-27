"""Small, dependency-free helpers shared by the infrastructure modules."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import re
import tempfile


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(UTC)


def format_utc(value: datetime) -> str:
    """Serialize a datetime as an ISO-8601 UTC string ending in ``Z``."""

    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def compact_utc(value: datetime) -> str:
    """Serialize a datetime for use in a readable experiment identifier."""

    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    normalized = value.astimezone(UTC)
    return normalized.strftime("%Y%m%dT%H%M%S") + f"{normalized.microsecond:06d}Z"


def sanitize_path_component(value: str) -> str:
    """Return a conservative, non-empty file-name component."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not sanitized:
        raise ValueError("experiment name has no path-safe characters")
    return sanitized.lower()


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
