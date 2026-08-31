"""In-memory analysis-session records for Application v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from PIL import Image

from windblade_demo.constants import (
    APPLICATION_VERSION,
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_STATE_FINGERPRINT,
    CLASS_LABELS,
    PREPROCESSING_CONTRACT,
)
from windblade_demo.inference import InferenceResult


@dataclass(frozen=True)
class RegionRecord:
    """One user-supplied region and its frozen-classifier output.

    Pixel payloads live only in Streamlit session memory and are deliberately omitted
    from serializable metadata.
    """

    region_id: str
    created_utc: str
    mode: str
    source_name: str
    source_sha256: str
    source_width: int
    source_height: int
    selected_box: tuple[int, int, int, int] | None
    contextual_box: tuple[int, int, int, int] | None
    predicted_class_id: int
    predicted_label: str
    scores: tuple[float, ...]
    logits: tuple[float, ...]
    preprocessing_seconds: float
    inference_seconds: float
    model_input: Image.Image
    thumbnail: Image.Image
    gradcam_status: str = "not_generated"
    gradcam_overlay: Image.Image | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "created_utc": self.created_utc,
            "mode": self.mode,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_dimensions": [self.source_width, self.source_height],
            "selected_box": list(self.selected_box) if self.selected_box else None,
            "contextual_box": list(self.contextual_box) if self.contextual_box else None,
            "predicted_class_id": self.predicted_class_id,
            "predicted_label": self.predicted_label,
            "scores": {
                label: self.scores[index] for index, label in enumerate(CLASS_LABELS)
            },
            "logits": {
                label: self.logits[index] for index, label in enumerate(CLASS_LABELS)
            },
            "preprocessing_seconds": self.preprocessing_seconds,
            "inference_seconds": self.inference_seconds,
            "gradcam_status": self.gradcam_status,
            "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
            "checkpoint_state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
            "preprocessing_contract": PREPROCESSING_CONTRACT,
            "application_version": APPLICATION_VERSION,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_region_id(records: Iterable[RegionRecord]) -> str:
    numbers = []
    for record in records:
        if record.region_id.startswith("R") and record.region_id[1:].isdigit():
            numbers.append(int(record.region_id[1:]))
    return f"R{max(numbers, default=0) + 1}"


def make_region_record(
    *,
    records: Iterable[RegionRecord],
    mode: str,
    source_name: str,
    source_sha256: str,
    source_size: tuple[int, int],
    model_input: Image.Image,
    result: InferenceResult,
    selected_box: tuple[int, int, int, int] | None = None,
    contextual_box: tuple[int, int, int, int] | None = None,
    region_id: str | None = None,
    created_utc: str | None = None,
) -> RegionRecord:
    """Build a record without mutating the caller's session collection."""

    existing = tuple(records)
    chosen_id = region_id or next_region_id(existing)
    if any(item.region_id == chosen_id for item in existing) and region_id is None:
        raise ValueError(f"Region ID {chosen_id} is already present.")
    thumbnail = model_input.copy()
    thumbnail.thumbnail((240, 240), Image.Resampling.BILINEAR)
    return RegionRecord(
        region_id=chosen_id,
        created_utc=created_utc or utc_now(),
        mode=mode,
        source_name=source_name,
        source_sha256=source_sha256,
        source_width=int(source_size[0]),
        source_height=int(source_size[1]),
        selected_box=selected_box,
        contextual_box=contextual_box,
        predicted_class_id=result.predicted_class_id,
        predicted_label=result.predicted_label,
        scores=tuple(result.scores),
        logits=tuple(result.logits),
        preprocessing_seconds=float(result.preprocessing_seconds),
        inference_seconds=float(result.inference_seconds),
        model_input=model_input.copy(),
        thumbnail=thumbnail,
    )


def replace_region(records: Iterable[RegionRecord], replacement: RegionRecord) -> list[RegionRecord]:
    result = list(records)
    for index, current in enumerate(result):
        if current.region_id == replacement.region_id:
            result[index] = replacement
            return result
    raise KeyError(f"Unknown region ID: {replacement.region_id}")


def remove_region(records: Iterable[RegionRecord], region_id: str) -> list[RegionRecord]:
    existing = list(records)
    result = [record for record in existing if record.region_id != region_id]
    if len(result) == len(existing):
        raise KeyError(f"Unknown region ID: {region_id}")
    return result


def with_gradcam(record: RegionRecord, overlay: Image.Image) -> RegionRecord:
    return replace(
        record,
        gradcam_status="generated",
        gradcam_overlay=overlay.copy(),
    )
