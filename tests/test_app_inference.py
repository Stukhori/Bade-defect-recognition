from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from windblade.deep.checkpoints import state_dict_fingerprint
from windblade_demo.constants import (
    CHECKPOINT_RELATIVE_PATH,
    CHECKPOINT_STATE_FINGERPRINT,
    CLASS_LABELS,
    MODEL_DISPLAY_NAME,
)
from windblade_demo.explain import generate_gradcam
from windblade_demo.inference import infer, load_frozen_model, model_status, preprocess_model_input


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/processed/wtbd_crops_v1/images/0_0.png"

# Frozen after independent inference from the unchanged Phase 6 seed-17 checkpoint.
REFERENCE_LOGITS = (
    -2.116699457168579,
    8.117175102233887,
    -2.5408267974853516,
    -0.792829692363739,
    -5.986865520477295,
    1.9971497058868408,
)


@pytest.fixture(scope="module")
def loaded():
    if not (ROOT / CHECKPOINT_RELATIVE_PATH).is_file():
        pytest.skip("local frozen Phase 6 checkpoint is not available")
    return load_frozen_model(ROOT)


def reference_image():
    if not REFERENCE.is_file():
        pytest.skip("local frozen Phase 3 crop payload is not available")
    with Image.open(REFERENCE) as opened:
        return opened.convert("RGB").copy()


def test_frozen_checkpoint_identity_and_cpu_eval_mode(loaded):
    assert state_dict_fingerprint(loaded.model.state_dict()) == CHECKPOINT_STATE_FINGERPRINT
    assert not loaded.model.training
    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cpu"}
    status = model_status(loaded)
    assert status["model"] == MODEL_DISPLAY_NAME
    assert status["seed"] == "17"
    assert "Frozen checkpoint verified" in status["status"]


def test_preprocessing_matches_frozen_imagenet_transform(loaded):
    tensor = preprocess_model_input(reference_image())
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    pixel = np.asarray(reference_image(), dtype=np.float32)[0, 0] / 255.0
    expected = (pixel - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(tensor[0, :, 0, 0].numpy(), expected, atol=1e-6)


def test_reference_logits_and_all_scores_are_stable(loaded):
    result = infer(loaded, reference_image())
    assert len(result.logits) == len(result.scores) == len(CLASS_LABELS) == 6
    assert np.allclose(result.logits, REFERENCE_LOGITS, rtol=0.0, atol=1e-6)
    assert sum(result.scores) == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_label == CLASS_LABELS[int(np.argmax(result.scores))]
    assert result.inference_seconds >= 0
    assert result.preprocessing_seconds >= 0


def test_gradcam_does_not_change_parameters_or_predictions(loaded):
    image = reference_image()
    before_state = state_dict_fingerprint(loaded.model.state_dict())
    before = infer(loaded, image)
    visualization = generate_gradcam(loaded, image, before.predicted_class_id)
    after = infer(loaded, image)
    assert visualization.heatmap.size == visualization.overlay.size == (224, 224)
    assert visualization.activation_shape == (1, 576, 7, 7)
    assert state_dict_fingerprint(loaded.model.state_dict()) == before_state
    assert after.logits == pytest.approx(before.logits, abs=0.0, rel=0.0)
    assert after.scores == pytest.approx(before.scores, abs=0.0, rel=0.0)
