"""Official torchvision MobileNetV3-Small construction and provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from windblade.deep.checkpoints import state_dict_fingerprint
from windblade.deep.determinism import seed_torch
from windblade.deep.resnet import _jsonable

EXPECTED_MOBILENET_PARAMETERS = 1_524_006
WEIGHT_ENUM = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"


def replace_head(model: nn.Module) -> nn.Module:
    if not isinstance(model.classifier[-1], nn.Linear) or model.classifier[-1].in_features != 1024:
        raise RuntimeError("official MobileNetV3-Small final classifier structure changed")
    model.classifier[-1] = nn.Linear(1024, 6)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != EXPECTED_MOBILENET_PARAMETERS or trainable != total:
        raise RuntimeError(f"unexpected MobileNetV3-Small parameter count: {total}/{trainable}")
    return model


def build_mobilenet(*, pretrained: bool) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    return replace_head(model)


def load_official_model() -> tuple[nn.Module, dict[str, Any]]:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    cache = Path(torch.hub.get_dir()) / "checkpoints" / filename
    was_cached = cache.is_file()
    model = mobilenet_v3_small(weights=weights)
    return model, {
        "weight_enum": WEIGHT_ENUM,
        "url": weights.url,
        "cache_filename": filename,
        "loaded_from_cache": was_cached,
        "downloaded_during_phase6": not was_cached,
        "pretrained_mobilenet_fingerprint": state_dict_fingerprint(model.state_dict()),
        "published_transforms": repr(weights.transforms()),
        "published_metadata": _jsonable(weights.meta),
    }


def model_from_official_state(state: dict[str, torch.Tensor], *, seed: int) -> nn.Module:
    seed_torch(seed)
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(state)
    return replace_head(model)
