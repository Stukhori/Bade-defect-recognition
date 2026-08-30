"""Optional, read-only Grad-CAM wrapper for the existing Phase 9A apparatus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch

from windblade.deep.checkpoints import state_dict_fingerprint
from windblade.error_analysis.gradcam import gradcam_map, render_heatmap, resolve_module
from windblade_demo.inference import LoadedFrozenModel, preprocess_model_input


@dataclass(frozen=True)
class GradCamResult:
    heatmap: Image.Image
    overlay: Image.Image
    activation_shape: tuple[int, ...]


def generate_gradcam(
    loaded: LoadedFrozenModel,
    image: Image.Image,
    target_class: int,
    *,
    alpha: float = 0.45,
) -> GradCamResult:
    """Generate a visualization without changing model parameters or predictions."""

    before = state_dict_fingerprint(loaded.model.state_dict())
    tensor = preprocess_model_input(image)
    target_layer = resolve_module(loaded.model, "features.12")
    try:
        array, shape, _logits = gradcam_map(loaded.model, target_layer, tensor, target_class)
    finally:
        loaded.model.zero_grad(set_to_none=True)
        loaded.model.eval()
    after = state_dict_fingerprint(loaded.model.state_dict())
    if before != after or shape != (1, 576, 7, 7) or not np.isfinite(array).all():
        raise RuntimeError("The optional Grad-CAM invariance check failed.")
    heatmap = render_heatmap(array)
    overlay = Image.blend(image.convert("RGB"), heatmap.convert("RGB"), float(alpha))
    return GradCamResult(heatmap=heatmap, overlay=overlay, activation_shape=shape)
