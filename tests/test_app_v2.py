from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path

from PIL import Image
import pytest

from windblade_demo.constants import APPLICATION_VERSION, CLASS_LABELS
from windblade_demo.detection_status import DetectorUnavailableError, detect, load_detection_status, load_detector
from windblade_demo.exports import annotated_image_export, csv_export, json_export
from windblade_demo.inference import InferenceResult
from windblade_demo.research import PHASE10_FINGERPRINT, FrozenResearchError, load_phase10
from windblade_demo.session import make_region_record, next_region_id, remove_region, replace_region


ROOT = Path(__file__).resolve().parents[1]


def inference(label_index: int = 1) -> InferenceResult:
    scores = [0.02] * len(CLASS_LABELS)
    scores[label_index] = 0.90
    return InferenceResult(
        predicted_class_id=label_index, predicted_label=CLASS_LABELS[label_index],
        logits=tuple(float(index) for index in range(len(CLASS_LABELS))),
        scores=tuple(scores), preprocessing_seconds=0.01, inference_seconds=0.02,
    )


def record(records=(), *, region_id=None, box=(10, 12, 42, 50)):
    return make_region_record(
        records=records, mode="manual_multi_region", source_name="blade.jpg",
        source_sha256="a" * 64, source_size=(100, 80),
        model_input=Image.new("RGB", (224, 224), "white"), result=inference(),
        selected_box=box, contextual_box=(2, 4, 66, 68), region_id=region_id,
        created_utc="2026-08-31T00:00:00+00:00",
    )


def test_region_ids_are_stable_across_add_replace_remove():
    first = record()
    second = record([first], box=(20, 20, 60, 60))
    assert (first.region_id, second.region_id) == ("R1", "R2")
    replacement = record([first, second], region_id="R1", box=(1, 1, 20, 20))
    updated = replace_region([first, second], replacement)
    assert [item.region_id for item in updated] == ["R1", "R2"]
    assert updated[0].selected_box == (1, 1, 20, 20)
    assert [item.region_id for item in remove_region(updated, "R1")] == ["R2"]
    assert next_region_id([second]) == "R3"


def test_json_and_csv_exports_include_reproducibility_metadata():
    item = record()
    payload = json.loads(json_export([item], exported_utc="2026-08-31T01:00:00+00:00"))
    assert payload["application_version"] == APPLICATION_VERSION
    assert payload["automatic_localization"] == "unavailable"
    assert payload["regions"][0]["selected_box"] == [10, 12, 42, 50]
    assert set(payload["regions"][0]["scores"]) == set(CLASS_LABELS)
    rows = list(csv.DictReader(StringIO(csv_export([item]).decode("utf-8"))))
    assert rows[0]["region_id"] == "R1"
    assert rows[0][f"score_{CLASS_LABELS[1]}"] == "0.9"
    assert rows[0]["application_version"] == APPLICATION_VERSION
    assert len(rows[0]["checkpoint_state_fingerprint"]) == 64
    assert "Automatic localization is unavailable" in rows[0]["limitation_notice"]


def test_annotated_export_draws_user_boxes_without_touching_source():
    source = Image.new("RGB", (100, 80), "white")
    before = source.tobytes()
    exported = annotated_image_export(source, [record()])
    with Image.open(BytesIO(exported)) as opened:
        assert opened.format == "PNG"
        assert opened.size == source.size
        assert opened.convert("RGB").getpixel((10, 12)) != (255, 255, 255)
    assert source.tobytes() == before


def test_phase10_dashboard_sources_are_verified_and_not_recomputed(tmp_path):
    result = load_phase10(ROOT)
    assert result["scientific_output_fingerprint"] == PHASE10_FINGERPRINT
    assert len(result["tables"]["clean_method_comparison"]) == 4
    with pytest.raises(FrozenResearchError, match="unavailable"):
        load_phase10(tmp_path)


def test_phase11a_status_and_future_detector_contract():
    status = load_detection_status(ROOT)
    assert status.available is False
    assert status.integration_decision == "unsupported"
    assert status.phase11b_status == "blocked and unstarted"
    with pytest.raises(DetectorUnavailableError, match="Phase 11B"):
        load_detector()
    with pytest.raises(DetectorUnavailableError, match="Phase 11B"):
        detect(Image.new("RGB", (50, 50)))
