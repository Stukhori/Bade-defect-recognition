"""Phase 11A readiness reporting and an explicit unavailable-detector contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image


PHASE11A_FINGERPRINT = "3f46cbdc6c7a2e3cf6093ff177dd1948d113fa4c36fa9eb907d7c8621e800461"


class DetectorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int
    score: float
    label: str | None = None


@dataclass(frozen=True)
class DetectionResult:
    boxes: tuple[BoundingBox, ...]
    detector_name: str


@dataclass(frozen=True)
class DetectorStatus:
    available: bool
    phase11a_status: str
    phase11b_status: str
    integration_decision: str
    block_reason: str
    scientific_output_fingerprint: str
    audit: dict[str, Any]
    feasibility: dict[str, Any]
    compute_gate: dict[str, Any]


def load_detection_status(root: str | Path) -> DetectorStatus:
    base = Path(root) / "experiments/summaries/phase11_detection_audit_v1"
    required = {
        name: base / f"{name}.json"
        for name in ("manifest", "reproducibility", "audit_summary", "feasibility", "compute_gate")
    }
    if any(not path.is_file() for path in required.values()):
        raise DetectorUnavailableError("The verified detection-readiness sources could not be loaded.")
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    manifest = values["manifest"]
    repro = values["reproducibility"]
    if (
        manifest.get("scientific_output_fingerprint") != PHASE11A_FINGERPRINT
        or repro.get("scientific_output_fingerprint") != PHASE11A_FINGERPRINT
        or manifest.get("phase11b_training_started") is not False
    ):
        raise DetectorUnavailableError("The detection-readiness source identity failed verification.")
    return DetectorStatus(
        available=False,
        phase11a_status="complete and frozen",
        phase11b_status="blocked and unstarted",
        integration_decision=values["feasibility"]["application_integration"]["decision"],
        block_reason=values["compute_gate"]["block_reason"],
        scientific_output_fingerprint=PHASE11A_FINGERPRINT,
        audit=values["audit_summary"],
        feasibility=values["feasibility"],
        compute_gate=values["compute_gate"],
    )


def load_detector(*_args: Any, **_kwargs: Any) -> None:
    raise DetectorUnavailableError(
        "This application accepts user-selected regions for classification."
    )


def detect(_image: Image.Image, *_args: Any, **_kwargs: Any) -> DetectionResult:
    load_detector()
    raise AssertionError("unreachable")
