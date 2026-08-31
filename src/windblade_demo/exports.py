"""In-memory Application v2 export builders."""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
from typing import Iterable

from PIL import Image

from windblade_demo.constants import (
    APPLICATION_VERSION, CHECKPOINT_FILE_SHA256, CHECKPOINT_STATE_FINGERPRINT,
    CLASS_LABELS, PREPROCESSING_CONTRACT,
)
from windblade_demo.session import RegionRecord, utc_now
from windblade_demo.visualization import annotate_regions


LIMITATION_NOTICE = (
    "User-supplied regions only. Automatic localization is unavailable; model scores are not "
    "calibrated confidence estimates and do not assess structural integrity or operational safety."
)


def session_payload(records: Iterable[RegionRecord], *, exported_utc: str | None = None) -> dict:
    items = list(records)
    return {
        "schema_version": "1.0",
        "application_version": APPLICATION_VERSION,
        "exported_utc": exported_utc or utc_now(),
        "processing": "local in-memory session processing",
        "automatic_localization": "unavailable",
        "limitation_notice": LIMITATION_NOTICE,
        "region_count": len(items),
        "regions": [item.metadata() for item in items],
    }


def json_export(records: Iterable[RegionRecord], *, exported_utc: str | None = None) -> bytes:
    return (json.dumps(session_payload(records, exported_utc=exported_utc), indent=2) + "\n").encode("utf-8")


def csv_export(records: Iterable[RegionRecord]) -> bytes:
    fields = [
        "application_version", "limitation_notice", "checkpoint_file_sha256",
        "checkpoint_state_fingerprint", "preprocessing_contract",
        "region_id", "created_utc", "mode", "source_name", "source_sha256",
        "source_width", "source_height", "selected_box", "contextual_box",
        "predicted_label", *[f"score_{label}" for label in CLASS_LABELS],
        "preprocessing_seconds", "inference_seconds", "gradcam_status",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = {
            "application_version": APPLICATION_VERSION,
            "limitation_notice": LIMITATION_NOTICE,
            "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
            "checkpoint_state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
            "preprocessing_contract": PREPROCESSING_CONTRACT,
            "region_id": record.region_id,
            "created_utc": record.created_utc,
            "mode": record.mode,
            "source_name": record.source_name,
            "source_sha256": record.source_sha256,
            "source_width": record.source_width,
            "source_height": record.source_height,
            "selected_box": record.selected_box,
            "contextual_box": record.contextual_box,
            "predicted_label": record.predicted_label,
            "preprocessing_seconds": record.preprocessing_seconds,
            "inference_seconds": record.inference_seconds,
            "gradcam_status": record.gradcam_status,
        }
        row.update({f"score_{label}": record.scores[i] for i, label in enumerate(CLASS_LABELS)})
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def annotated_image_export(image: Image.Image, records: Iterable[RegionRecord]) -> bytes:
    output = BytesIO()
    annotate_regions(image, records).save(output, format="PNG")
    return output.getvalue()
