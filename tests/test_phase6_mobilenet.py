from __future__ import annotations
from pathlib import Path
import torch
from torchvision.models import MobileNet_V3_Small_Weights
from windblade.config import load_config
from windblade.deep.checkpoints import state_dict_fingerprint
from windblade.deep.mobilenet import EXPECTED_MOBILENET_PARAMETERS, WEIGHT_ENUM, build_mobilenet


def test_mobilenet_structure_and_forward() -> None:
    assert WEIGHT_ENUM == "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    assert MobileNet_V3_Small_Weights.IMAGENET1K_V1 is not None
    model=build_mobilenet(pretrained=False)
    assert model.classifier[-1].in_features == 1024 and model.classifier[-1].out_features == 6
    assert model.classifier[2].p == 0.2
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_MOBILENET_PARAMETERS
    assert all(p.requires_grad for p in model.parameters())
    model.eval()
    with torch.no_grad(): logits=model(torch.zeros(2,3,224,224))
    assert logits.shape == (2,6) and torch.isfinite(logits).all()


def test_state_fingerprint_is_deterministic() -> None:
    model=build_mobilenet(pretrained=False)
    assert state_dict_fingerprint(model.state_dict()) == state_dict_fingerprint(model.state_dict())


def test_phase5_phase6_protocol_parity() -> None:
    root=Path(__file__).resolve().parents[1]
    p5=load_config(root/"configs/resnet18_baseline.yaml").as_dict(); p6=load_config(root/"configs/mobilenet_v3_small_baseline.yaml").as_dict()
    fields=(("dataset","processed_version"),("dataset","processed_fingerprint"),("classes","order"),("input","size"),("input","augmentation"),("input","normalization"),("training","batch_size"),("training","validation_batch_size"),("training","max_epochs"),("training","patience"),("training","min_delta"),("training","optimizer"),("training","betas"),("training","eps"),("training","class_weight"),("search","seed"),("search","learning_rates"),("search","weight_decays"),("final","seeds"),("selection","primary_metric"))
    assert all(p5[s][f] == p6[s][f] for s,f in fields)
    assert p6["search"]["seed"] == 17 and p6["final"]["seeds"] == [17,29,43]
