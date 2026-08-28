from __future__ import annotations

import torch
from torchvision.models import ResNet18_Weights

from windblade.deep.resnet import EXPECTED_RESNET18_PARAMETERS, WEIGHT_ENUM, build_resnet18


def test_official_enum_and_standard_six_class_structure() -> None:
    assert ResNet18_Weights.IMAGENET1K_V1 is not None
    assert WEIGHT_ENUM == "ResNet18_Weights.IMAGENET1K_V1"
    model = build_resnet18(pretrained=False)
    assert model.fc.out_features == 6
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_RESNET18_PARAMETERS
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == EXPECTED_RESNET18_PARAMETERS
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
