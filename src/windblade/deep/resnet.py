"""Frozen standard torchvision ResNet-18 construction and provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from windblade.deep.checkpoints import state_dict_fingerprint

EXPECTED_RESNET18_PARAMETERS = 11_179_590
WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"


def build_resnet18(*, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 6)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != EXPECTED_RESNET18_PARAMETERS or trainable != total:
        raise RuntimeError(f"unexpected ResNet-18 parameter count: {total}/{trainable}")
    return model


def load_official_backbone() -> tuple[nn.Module, dict[str, Any]]:
    """Load only the named official weights and fingerprint the 1000-class model."""

    weights = ResNet18_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    was_cached = cache_path.is_file()
    model = resnet18(weights=weights)
    provenance = {
        "weight_enum": WEIGHT_ENUM,
        "url": weights.url,
        "cache_filename": filename,
        "loaded_from_cache": was_cached,
        "downloaded_during_phase5": not was_cached,
        "pretrained_backbone_fingerprint": state_dict_fingerprint(model.state_dict()),
        "published_transforms": repr(weights.transforms()),
        "published_metadata": _jsonable(weights.meta),
    }
    return model, provenance


def replace_head(backbone: nn.Module) -> nn.Module:
    backbone.fc = nn.Linear(backbone.fc.in_features, 6)
    total = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    if total != EXPECTED_RESNET18_PARAMETERS or trainable != total:
        raise RuntimeError(f"unexpected ResNet-18 parameter count: {total}/{trainable}")
    return backbone


def model_from_official_state(
    official_state: dict[str, torch.Tensor], *, seed: int
) -> nn.Module:
    """Recreate the official backbone and deterministically initialize only its new head."""

    from windblade.deep.determinism import seed_torch

    seed_torch(seed)
    model = resnet18(weights=None)
    model.load_state_dict(official_state)
    return replace_head(model)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
