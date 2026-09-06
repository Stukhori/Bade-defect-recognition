"""Fail-closed experimental region proposals from the frozen Phase 11B detector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from windblade_demo.crops import PixelBox


DETECTOR_PACKAGE_VERSION = "8.3.150"
DETECTOR_CHECKPOINT = Path(
    "experiments/results/phase11b_yolo11n_v1/final/seed_17/epoch82.pt"
)
DETECTOR_CHECKPOINT_BYTES = 16_085_716
DETECTOR_CHECKPOINT_SHA256 = (
    "793547a5ec31954d8e909b2f5c63f378374134353a8d1e0cefdd452a5365eefa"
)
DETECTOR_INTERNAL_CLASSES = {0: "item"}
DETECTOR_SEMANTIC_LABEL = "defect region proposal"
DETECTOR_EXPECTED_SEED = 17
DETECTOR_EXPECTED_EPOCH = 83
ZERO_PROPOSAL_MESSAGE = "No region proposal exceeded the frozen threshold."
PROPOSAL_LIMITATION_NOTICE = (
    "Experimental research feature. The detector dataset contains no healthy/background-only "
    "images, every proposal requires human review, and outputs are not a safety, severity, "
    "progression, remaining-life, or production-readiness assessment."
)
INFERENCE_ARGUMENTS = {
    "device": "cpu",
    "imgsz": 640,
    "conf": 0.39,
    "iou": 0.7,
    "agnostic_nms": True,
    "max_det": 300,
    "save": False,
    "plots": False,
    "verbose": False,
}


class ProposalDetectorError(RuntimeError):
    """Raised when detector identity, metadata, or output violates the frozen contract."""


@dataclass(frozen=True)
class RegionProposal:
    proposal_id: str
    box: PixelBox
    detector_confidence: float

    @property
    def semantic_label(self) -> str:
        return DETECTOR_SEMANTIC_LABEL


@dataclass(frozen=True)
class ProposalDetector:
    model: Any
    checkpoint_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    checkpoint = (root_path / DETECTOR_CHECKPOINT).resolve()
    try:
        checkpoint.relative_to(root_path)
    except ValueError as exc:
        raise ProposalDetectorError("Detector checkpoint path escapes the repository.") from exc
    if not checkpoint.is_file():
        raise ProposalDetectorError("The frozen detector checkpoint is missing.")
    if checkpoint.stat().st_size != DETECTOR_CHECKPOINT_BYTES:
        raise ProposalDetectorError("Detector checkpoint size does not match the frozen contract.")
    if sha256_file(checkpoint) != DETECTOR_CHECKPOINT_SHA256:
        raise ProposalDetectorError("Detector checkpoint SHA-256 does not match the frozen contract.")
    return checkpoint


def _normalise_classes(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ProposalDetectorError("Detector class metadata is missing or malformed.")
    result: dict[int, str] = {}
    for key, label in value.items():
        if isinstance(key, bool) or not isinstance(key, int) or not isinstance(label, str):
            raise ProposalDetectorError("Detector class metadata is missing or malformed.")
        result[key] = label
    return result


def _validate_model_metadata(model: Any) -> None:
    if getattr(model, "task", None) != "detect":
        raise ProposalDetectorError("The frozen checkpoint is not a detection model.")
    if _normalise_classes(getattr(model, "names", None)) != DETECTOR_INTERNAL_CLASSES:
        raise ProposalDetectorError("Detector internal class mapping does not match the frozen contract.")
    checkpoint = getattr(model, "ckpt", None)
    if not isinstance(checkpoint, dict):
        raise ProposalDetectorError("Detector training metadata is unavailable.")
    training = checkpoint.get("train_args")
    if not isinstance(training, dict):
        raise ProposalDetectorError("Detector training metadata is unavailable.")
    if training.get("seed") != DETECTOR_EXPECTED_SEED:
        raise ProposalDetectorError("Detector seed metadata does not match the frozen contract.")
    if training.get("imgsz") != INFERENCE_ARGUMENTS["imgsz"]:
        raise ProposalDetectorError("Detector image-size metadata does not match the frozen contract.")
    if checkpoint.get("epoch") != DETECTOR_EXPECTED_EPOCH - 1:
        raise ProposalDetectorError("Detector epoch metadata does not match the frozen contract.")


def _disable_external_integrations() -> None:
    # Process-local settings are established before importing the detector package.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(
        Path(tempfile.gettempdir()) / "bladescope-ultralytics"
    )
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ["CLEARML_OFFLINE_MODE"] = "1"
    os.environ["MLFLOW_TRACKING_URI"] = ""


def load_proposal_detector(
    root: str | Path,
    *,
    package_version: str | None = None,
    yolo_factory: Callable[..., Any] | None = None,
) -> ProposalDetector:
    """Verify bytes and metadata before returning one CPU-only detection model."""

    checkpoint = verify_checkpoint(root)
    try:
        installed = package_version if package_version is not None else version("ultralytics")
    except PackageNotFoundError as exc:
        raise ProposalDetectorError("The pinned detector runtime is not installed.") from exc
    if installed != DETECTOR_PACKAGE_VERSION:
        raise ProposalDetectorError(
            f"Detector runtime version must be exactly {DETECTOR_PACKAGE_VERSION}; found {installed}."
        )
    _disable_external_integrations()
    if yolo_factory is None:
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        yolo_factory = YOLO
    try:
        model = yolo_factory(str(checkpoint), task="detect")
    except Exception as exc:
        raise ProposalDetectorError("The frozen detector checkpoint could not be loaded.") from exc
    try:
        model.to("cpu")
    except Exception as exc:
        raise ProposalDetectorError("The frozen detector could not be placed on CPU.") from exc
    _validate_model_metadata(model)
    return ProposalDetector(model=model, checkpoint_path=checkpoint)


def _values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ProposalDetectorError("Detector output is malformed.")
    return value


def parse_proposals(result: Any, *, image_size: tuple[int, int]) -> tuple[RegionProposal, ...]:
    """Parse, bound, and deterministically order single-class detector output."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ProposalDetectorError("Source image dimensions are invalid.")
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _values(getattr(boxes, "xyxy", None))
    scores = _values(getattr(boxes, "conf", None))
    classes = _values(getattr(boxes, "cls", None))
    if not (len(xyxy) == len(scores) == len(classes)):
        raise ProposalDetectorError("Detector output fields have inconsistent lengths.")

    parsed: list[tuple[PixelBox, float]] = []
    for coordinates, score_value, class_value in zip(xyxy, scores, classes, strict=True):
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ProposalDetectorError("Detector coordinates are malformed.")
        try:
            coordinate_values = [float(value) for value in coordinates]
            score = float(score_value)
            class_number = float(class_value)
        except (TypeError, ValueError) as exc:
            raise ProposalDetectorError("Detector output contains a non-numeric value.") from exc
        if not all(math.isfinite(value) for value in (*coordinate_values, score, class_number)):
            raise ProposalDetectorError("Detector output contains a non-finite value.")
        if class_number != 0.0 or not class_number.is_integer():
            raise ProposalDetectorError("Detector output contains an unexpected class.")
        if not 0.0 <= score <= 1.0:
            raise ProposalDetectorError("Detector confidence is outside [0, 1].")
        left_f, top_f, right_f, bottom_f = coordinate_values
        if (
            left_f >= right_f or top_f >= bottom_f
            or right_f <= 0 or bottom_f <= 0
            or left_f >= width or top_f >= height
        ):
            raise ProposalDetectorError("Detector output contains a box outside the image area.")
        left = max(0, min(width - 1, math.floor(left_f)))
        top = max(0, min(height - 1, math.floor(top_f)))
        right = max(1, min(width, math.ceil(right_f)))
        bottom = max(1, min(height, math.ceil(bottom_f)))
        if left >= right or top >= bottom:
            raise ProposalDetectorError("Detector output contains an empty bounded box.")
        parsed.append((PixelBox(left, top, right, bottom), score))

    parsed.sort(key=lambda item: (-item[1], item[0].top, item[0].left, item[0].bottom, item[0].right))
    return tuple(
        RegionProposal(f"P{index}", box, score)
        for index, (box, score) in enumerate(parsed, start=1)
    )


def propose_regions(detector: ProposalDetector, image: Image.Image) -> tuple[RegionProposal, ...]:
    """Run one frozen, CPU-only prediction without output files or user controls."""

    source = np.asarray(image.convert("RGB"))
    try:
        results = detector.model.predict(source=source, **INFERENCE_ARGUMENTS)
    except Exception as exc:
        raise ProposalDetectorError("Experimental region proposal inference failed.") from exc
    if not isinstance(results, Iterable):
        raise ProposalDetectorError("Detector result collection is malformed.")
    result_list = list(results)
    if len(result_list) != 1:
        raise ProposalDetectorError("Detector must return exactly one result for one image.")
    return parse_proposals(result_list[0], image_size=image.size)


def reviewed_proposals(
    proposals: Iterable[RegionProposal], selected_ids: Iterable[str], *, reviewed: bool
) -> tuple[RegionProposal, ...]:
    """Fail closed unless a user explicitly reviewed every selected proposal."""

    proposal_items = tuple(proposals)
    selected = tuple(selected_ids)
    if not selected:
        raise ProposalDetectorError("Select at least one proposal before classification.")
    if not reviewed:
        raise ProposalDetectorError("Confirm that you reviewed the selected proposal boxes.")
    by_id = {proposal.proposal_id: proposal for proposal in proposal_items}
    if len(by_id) != len(proposal_items):
        raise ProposalDetectorError("Proposal identifiers are not unique.")
    if len(set(selected)) != len(selected) or any(value not in by_id for value in selected):
        raise ProposalDetectorError("The reviewed proposal selection is invalid.")
    return tuple(by_id[value] for value in selected)
