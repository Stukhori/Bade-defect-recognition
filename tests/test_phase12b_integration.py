from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from scripts.validate_phase12b import validate
from windblade_demo import detector as detector_module
from windblade_demo.constants import CLASS_LABELS
from windblade_demo.detector import (
    DETECTOR_CHECKPOINT, DETECTOR_CHECKPOINT_BYTES, DETECTOR_CHECKPOINT_SHA256,
    DETECTOR_SEMANTIC_LABEL, INFERENCE_ARGUMENTS, ProposalDetector,
    ProposalDetectorError, ZERO_PROPOSAL_MESSAGE, load_proposal_detector,
    parse_proposals, propose_regions, reviewed_proposals, verify_checkpoint,
)
from windblade_demo.inference import InferenceResult
from windblade_demo.exports import csv_export, json_export
from windblade_demo.session import make_region_record


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeBoxes:
    xyxy: object
    conf: object
    cls: object


@dataclass
class FakeResult:
    boxes: object


class FakeModel:
    task = "detect"
    names = {0: "item"}
    ckpt = {"epoch": 82, "train_args": {"seed": 17, "imgsz": 640}}

    def __init__(self, result: FakeResult | None = None) -> None:
        self.result = result or FakeResult(FakeBoxes([], [], []))
        self.calls: list[dict] = []
        self.devices: list[str] = []

    def to(self, device: str):
        self.devices.append(device)
        return self

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def factory_for(model: FakeModel):
    calls = []

    def factory(path: str, **kwargs):
        calls.append((path, kwargs))
        return model

    return factory, calls


def test_checkpoint_is_exact_and_tracked() -> None:
    checkpoint = verify_checkpoint(ROOT)
    assert checkpoint.stat().st_size == DETECTOR_CHECKPOINT_BYTES
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == DETECTOR_CHECKPOINT_SHA256
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", DETECTOR_CHECKPOINT.as_posix()],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert tracked.returncode == 0


def test_load_is_lazy_fail_closed_and_verifies_exact_metadata() -> None:
    model = FakeModel()
    factory, calls = factory_for(model)
    loaded = load_proposal_detector(ROOT, package_version="8.3.150", yolo_factory=factory)
    assert loaded.model is model
    assert model.devices == ["cpu"]
    assert calls == [(str((ROOT / DETECTOR_CHECKPOINT).resolve()), {"task": "detect"})]


def test_loading_disables_network_and_tracker_integrations(monkeypatch) -> None:
    for key in ("YOLO_OFFLINE", "WANDB_DISABLED", "COMET_MODE", "CLEARML_OFFLINE_MODE"):
        monkeypatch.setenv(key, "unsafe")
    factory, _ = factory_for(FakeModel())
    load_proposal_detector(ROOT, package_version="8.3.150", yolo_factory=factory)
    assert detector_module.os.environ["YOLO_OFFLINE"] == "true"
    assert detector_module.os.environ["WANDB_DISABLED"] == "true"
    assert detector_module.os.environ["COMET_MODE"] == "DISABLED"
    assert detector_module.os.environ["CLEARML_OFFLINE_MODE"] == "1"


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("task", "classify", "detection model"),
        ("names", {0: "defect"}, "class mapping"),
        ("names", {0: "item", 1: "item"}, "class mapping"),
        ("ckpt", {"epoch": 82, "train_args": {"seed": 29, "imgsz": 640}}, "seed metadata"),
        ("ckpt", {"epoch": 82, "train_args": {"seed": 17, "imgsz": 608}}, "image-size metadata"),
        ("ckpt", {"epoch": 81, "train_args": {"seed": 17, "imgsz": 640}}, "epoch metadata"),
    ],
)
def test_checkpoint_metadata_failures_are_rejected(attribute: str, value: object, message: str) -> None:
    model = FakeModel()
    setattr(model, attribute, value)
    factory, _ = factory_for(model)
    with pytest.raises(ProposalDetectorError, match=message):
        load_proposal_detector(ROOT, package_version="8.3.150", yolo_factory=factory)


def test_wrong_runtime_version_fails_before_model_construction() -> None:
    factory, calls = factory_for(FakeModel())
    with pytest.raises(ProposalDetectorError, match="exactly 8.3.150"):
        load_proposal_detector(ROOT, package_version="8.3.151", yolo_factory=factory)
    assert calls == []


def test_missing_and_hash_mismatched_checkpoint_fail_closed(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ProposalDetectorError, match="missing"):
        verify_checkpoint(tmp_path)
    checkpoint = tmp_path / DETECTOR_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-the-checkpoint")
    monkeypatch.setattr(detector_module, "DETECTOR_CHECKPOINT_BYTES", checkpoint.stat().st_size)
    with pytest.raises(ProposalDetectorError, match="SHA-256"):
        verify_checkpoint(tmp_path)


def test_exact_cpu_inference_arguments_and_no_filesystem_output(tmp_path: Path, monkeypatch) -> None:
    model = FakeModel()
    detector = ProposalDetector(model=model, checkpoint_path=ROOT / DETECTOR_CHECKPOINT)
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    assert propose_regions(detector, Image.new("RGB", (40, 30), "navy")) == ()
    assert tuple(tmp_path.iterdir()) == before
    assert len(model.calls) == 1
    call = model.calls[0]
    source = call.pop("source")
    assert source.shape == (30, 40, 3)
    assert call == INFERENCE_ARGUMENTS
    assert call["device"] == "cpu"
    assert call["save"] is False and call["plots"] is False


def test_proposal_parsing_is_deterministic_bounded_and_semantic_only() -> None:
    result = FakeResult(FakeBoxes(
        [[90.2, 50.7, 120.9, 80.3], [-3.2, -2.1, 20.1, 25.8], [5, 4, 12, 11]],
        [0.7, 0.9, 0.7],
        [0.0, 0.0, 0.0],
    ))
    proposals = parse_proposals(result, image_size=(100, 60))
    assert [proposal.proposal_id for proposal in proposals] == ["P1", "P2", "P3"]
    assert [proposal.box.as_tuple() for proposal in proposals] == [
        (0, 0, 21, 26), (5, 4, 12, 11), (90, 50, 100, 60),
    ]
    assert all(proposal.semantic_label == DETECTOR_SEMANTIC_LABEL for proposal in proposals)
    assert all("item" not in proposal.semantic_label for proposal in proposals)


@pytest.mark.parametrize(
    "boxes",
    [
        FakeBoxes([[1, 1, 2, 2]], [0.5], [1.0]),
        FakeBoxes([[1, 1, 2, 2]], [1.1], [0.0]),
        FakeBoxes([[4, 4, 3, 3]], [0.5], [0.0]),
        FakeBoxes([[float("nan"), 1, 2, 2]], [0.5], [0.0]),
        FakeBoxes([[-5, -5, -1, -1]], [0.5], [0.0]),
        FakeBoxes([[21, 21, 25, 25]], [0.5], [0.0]),
    ],
)
def test_malformed_proposals_fail_closed(boxes: FakeBoxes) -> None:
    with pytest.raises(ProposalDetectorError):
        parse_proposals(FakeResult(boxes), image_size=(20, 20))


def test_zero_wording_and_ui_never_expose_internal_label() -> None:
    assert ZERO_PROPOSAL_MESSAGE == "No region proposal exceeded the frozen threshold."
    source = (ROOT / "app/app.py").read_text(encoding="utf-8")
    assert "st.info(ZERO_PROPOSAL_MESSAGE)" in source
    assert '"item"' not in source and "'item'" not in source
    assert "Auto detection" in source
    assert "experimental" not in source.lower()


def test_review_is_required_before_classification() -> None:
    proposal = parse_proposals(
        FakeResult(FakeBoxes([[1, 2, 10, 12]], [0.8], [0.0])), image_size=(20, 20)
    )[0]
    with pytest.raises(ProposalDetectorError, match="Select"):
        reviewed_proposals([proposal], [], reviewed=True)
    with pytest.raises(ProposalDetectorError, match="Confirm"):
        reviewed_proposals([proposal], ["P1"], reviewed=False)
    with pytest.raises(ProposalDetectorError, match="invalid"):
        reviewed_proposals([proposal], ["P2"], reviewed=True)
    assert reviewed_proposals([proposal], ["P1"], reviewed=True) == (proposal,)


def test_detector_confidence_is_distinct_from_classifier_scores_and_exports() -> None:
    scores = (0.1, 0.2, 0.3, 0.1, 0.2, 0.1)
    result = InferenceResult(2, CLASS_LABELS[2], scores, scores, 0.0, 0.0)
    record = make_region_record(
        records=(), mode="auto_detection", source_name="synthetic.png",
        source_sha256="a" * 64, source_size=(30, 20), selected_box=(1, 2, 10, 12),
        contextual_box=(0, 0, 20, 20), model_input=Image.new("RGB", (224, 224)),
        result=result, detector_proposal_id="P1", detector_confidence=0.87,
    )
    metadata = record.metadata()
    assert metadata["detector_confidence"] == 0.87
    assert metadata["detector_proposal_id"] == "P1"
    assert metadata["scores"][CLASS_LABELS[2]] == 0.3
    assert metadata["detector_confidence"] not in metadata["scores"].values()
    exported_json = json.loads(json_export([record]))["regions"][0]
    exported_csv = list(csv.DictReader(StringIO(csv_export([record]).decode("utf-8"))))[0]
    assert exported_json["detector_confidence"] == 0.87
    assert exported_json["scores"][CLASS_LABELS[2]] == 0.3
    assert exported_csv["detector_confidence"] == "0.87"
    assert exported_csv[f"score_{CLASS_LABELS[2]}"] == "0.3"


def test_phase12b_validator_passes_without_running_inference() -> None:
    result = validate(ROOT)
    assert result["status"] == "PASS"
    assert result["phase"] == "12B"
    assert result["external_deployment"] is False
