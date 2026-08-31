"""Read-only validation of both local demonstration workflows."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from windblade.deep.checkpoints import state_dict_fingerprint
from windblade_review.schema import load_pass_schema
from windblade_review.store import ReviewDataError, ReviewStore
from windblade_demo.constants import (
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_STATE_FINGERPRINT,
    MODEL_DISPLAY_NAME,
)
from windblade_demo.crops import PixelBox, contextual_crop, prepare_region
from windblade_demo.explain import generate_gradcam
from windblade_demo.inference import infer, load_frozen_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def blank_review_form(path: Path) -> bool:
    rows = read_rows(path)
    human_fields = [
        field
        for field in (rows[0] if rows else {})
        if field not in {"review_id", "anonymous_case_id", "display_order"}
    ]
    return all(not str(row.get(field, "")).strip() for row in rows for field in human_fields)


def validate(root: Path) -> dict[str, Any]:
    manifest = read_rows(root / "data/processed/wtbd_crops_v1/manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    if not train_rows:
        raise RuntimeError("The frozen Phase 3 training rows are unavailable.")
    reference = train_rows[0]
    processed_path = root / "data/processed/wtbd_crops_v1" / reference["output_relative_path"]
    source_path = root / "data/raw/wtbd/WT blade defect dataset/JPEGImages" / reference["source_filename"]
    if not processed_path.is_file() or not source_path.is_file():
        raise RuntimeError("The local frozen image payload required for app validation is unavailable.")

    with Image.open(processed_path) as opened:
        prepared_source = opened.convert("RGB").copy()
    model_load_started = perf_counter()
    loaded = load_frozen_model(root)
    measured_load = perf_counter() - model_load_started
    prepared_input = prepare_region(prepared_source)
    prepared_result = infer(loaded, prepared_input)

    with Image.open(source_path) as opened:
        large_image = opened.convert("RGB").copy()
    selected = PixelBox(
        int(reference["bbox_xmin"]) - 1,
        int(reference["bbox_ymin"]) - 1,
        int(reference["bbox_xmax"]),
        int(reference["bbox_ymax"]),
    )
    manual = contextual_crop(large_image, selected)
    if not np.array_equal(np.asarray(manual.model_input), np.asarray(prepared_source)):
        raise RuntimeError("Manual workflow is not pixel-identical to the frozen Phase 3 reference.")
    manual_result = infer(loaded, manual.model_input)
    if manual_result.logits != prepared_result.logits:
        raise RuntimeError("The two workflows produced different logits for identical pixels.")

    before_state = state_dict_fingerprint(loaded.model.state_dict())
    gradcam_started = perf_counter()
    gradcam = generate_gradcam(loaded, prepared_input, prepared_result.predicted_class_id)
    gradcam_seconds = perf_counter() - gradcam_started
    after_state = state_dict_fingerprint(loaded.model.state_dict())
    after_gradcam = infer(loaded, prepared_input)
    if before_state != after_state or after_gradcam.logits != prepared_result.logits:
        raise RuntimeError("Optional Grad-CAM changed the frozen model or its prediction.")

    checkpoint = loaded.checkpoint_path
    if sha256(checkpoint) != CHECKPOINT_FILE_SHA256:
        raise RuntimeError("Checkpoint SHA-256 changed during validation.")
    packet_root = root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"
    expected_ids = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    review_records = []
    for pass_name, expected_total in (("pass_a", 300), ("pass_b", 240)):
        path = packet_root / pass_name / f"{pass_name}_review_form.csv"
        schema = load_pass_schema(root / "configs/error_analysis.yaml", pass_name)
        try:
            snapshot = ReviewStore(path, schema, expected_ids).load()
        except ReviewDataError as exc:
            raise RuntimeError(f"The {pass_name} review form is invalid: {exc}") from exc
        if snapshot.answered_required not in {0, expected_total}:
            raise RuntimeError(f"The {pass_name} review form is partially completed.")
        review_records.append(
            {
                "pass_name": pass_name,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "blank": snapshot.answered_required == 0,
                "complete": snapshot.complete,
                "answered_required": snapshot.answered_required,
                "total_required": snapshot.total_required,
            }
        )
    if review_records[1]["complete"] and not review_records[0]["complete"]:
        raise RuntimeError("Pass B cannot be complete unless Pass A is complete.")
    phase9b_manifest = root / "experiments/summaries/phase9_error_analysis_v1/phase9b/manifest.json"
    scientific_status = (
        "Phase 9 complete and frozen; Phase 10 not started"
        if phase9b_manifest.is_file()
        and json.loads(phase9b_manifest.read_text(encoding="utf-8")).get("phase9_complete")
        else "Phase 9A complete; Phase 9 incomplete"
    )

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "non-scientific local demonstration validation",
        "scientific_phase_status": scientific_status,
        "dependencies": {
            "streamlit": version("streamlit"),
            "streamlit-cropper": version("streamlit-cropper"),
        },
        "model": {
            "display_name": MODEL_DISPLAY_NAME,
            "checkpoint_file_sha256": sha256(checkpoint),
            "checkpoint_state_fingerprint": before_state,
            "expected_state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
            "device": "cpu",
            "evaluation_mode": not loaded.model.training,
        },
        "workflows": {
            "prepared_crop": {
                "status": "PASS",
                "reference_instance_id": reference["instance_id"],
                "input_shape": [224, 224, 3],
                "score_count": len(prepared_result.scores),
            },
            "manual_region": {
                "status": "PASS",
                "reference_instance_id": reference["instance_id"],
                "phase3_pixel_parity": True,
                "selected_original_box": list(selected.as_tuple()),
                "context_side": manual.geometry.crop_side,
            },
        },
        "optional_gradcam": {
            "status": "PASS",
            "activation_shape": list(gradcam.activation_shape),
            "parameters_unchanged": True,
            "prediction_unchanged": True,
        },
        "descriptive_local_timings_seconds": {
            "model_load_wrapper": measured_load,
            "model_load_recorded": loaded.load_seconds,
            "prepared_preprocessing": prepared_result.preprocessing_seconds,
            "prepared_inference": prepared_result.inference_seconds,
            "manual_preprocessing": manual_result.preprocessing_seconds,
            "manual_inference": manual_result.inference_seconds,
            "optional_gradcam": gradcam_seconds,
        },
        "phase9a_review_forms": review_records,
        "prohibited_actions": {
            "training": 0,
            "fine_tuning": 0,
            "calibration": 0,
            "test_set_evaluation": 0,
            "external_service_calls": 0,
            "permanent_upload_storage": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = validate(root)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
