"""Validated, atomic persistence for canonical review-form CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

from windblade_review.schema import PassSchema


class ReviewDataError(ValueError):
    """Raised when review data violate the frozen schema or identity."""


@dataclass(frozen=True)
class ReviewSnapshot:
    rows: tuple[dict[str, str], ...]
    completed_cases: int
    answered_required: int
    total_required: int
    first_incomplete_index: int | None

    @property
    def unanswered_required(self) -> int:
        return self.total_required - self.answered_required

    @property
    def complete(self) -> bool:
        return self.first_incomplete_index is None


@dataclass(frozen=True)
class SaveResult:
    changed: bool
    saved_at: str | None
    snapshot: ReviewSnapshot


class ReviewStore:
    def __init__(
        self,
        form_path: str | Path,
        schema: PassSchema,
        expected_ids: tuple[str, ...],
        *,
        replace_file: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
    ) -> None:
        self.path = Path(form_path)
        self.schema = schema
        self.expected_ids = expected_ids
        self.replace_file = replace_file

    def load(self, *, require_complete: bool = False) -> ReviewSnapshot:
        return self._load_path(self.path, require_complete=require_complete)

    def _load_path(self, path: Path, *, require_complete: bool) -> ReviewSnapshot:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != self.schema.headers:
                raise ReviewDataError(f"exact CSV headers changed: {path}")
            rows = [dict(row) for row in reader]
        if len(rows) != len(self.expected_ids):
            raise ReviewDataError(f"review row count changed: {path}")
        observed_ids = tuple(row.get("review_id", "") for row in rows)
        if observed_ids != self.expected_ids or len(set(observed_ids)) != len(observed_ids):
            raise ReviewDataError(f"review IDs or row order changed: {path}")

        answered = 0
        completed = 0
        first_incomplete: int | None = None
        for index, row in enumerate(rows):
            row_complete = True
            for field in self.schema.fields:
                value = row.get(field.name)
                if value is None:
                    raise ReviewDataError(f"missing field {field.name}: {row['review_id']}")
                if field.required:
                    if value:
                        if value not in field.choices:
                            raise ReviewDataError(
                                f"invalid response {field.name}={value!r}: {row['review_id']}"
                            )
                        answered += 1
                    else:
                        row_complete = False
                elif not isinstance(value, str):
                    raise ReviewDataError(f"notes are not text: {row['review_id']}")
            if row_complete:
                completed += 1
            elif first_incomplete is None:
                first_incomplete = index
        if require_complete and first_incomplete is not None:
            raise ReviewDataError(
                f"review is incomplete: {answered}/{len(rows) * len(self.schema.required_fields)} required answers"
            )
        return ReviewSnapshot(
            rows=tuple(rows),
            completed_cases=completed,
            answered_required=answered,
            total_required=len(rows) * len(self.schema.required_fields),
            first_incomplete_index=first_incomplete,
        )

    def save_case(self, review_id: str, values: Mapping[str, str]) -> SaveResult:
        unknown = set(values) - {field.name for field in self.schema.fields}
        if unknown:
            raise ReviewDataError(f"unknown fields cannot be saved: {sorted(unknown)}")
        snapshot = self.load()
        try:
            row_index = self.expected_ids.index(review_id)
        except ValueError as exc:
            raise ReviewDataError(f"unknown review ID: {review_id}") from exc
        rows = [dict(row) for row in snapshot.rows]
        changed = False
        for field_name, value in values.items():
            if not isinstance(value, str):
                raise ReviewDataError(f"review values must be text: {field_name}")
            definition = self.schema.definition(field_name)
            if definition.required and value and value not in definition.choices:
                raise ReviewDataError(f"invalid response {field_name}={value!r}: {review_id}")
            if rows[row_index][field_name] != value:
                rows[row_index][field_name] = value
                changed = True
        if not changed:
            return SaveResult(False, None, snapshot)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.schema.headers, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            self._load_path(temporary, require_complete=False)
            self.replace_file(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        saved_at = datetime.now(timezone.utc).isoformat()
        return SaveResult(True, saved_at, self.load())

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
