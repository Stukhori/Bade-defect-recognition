from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from windblade.data.processed import read_csv
from windblade.deep.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WTBDCropDataset,
    balanced_class_weights,
    canonical_transform,
    make_loader,
    split_rows,
)


def test_canonical_transform_is_exact_and_deterministic() -> None:
    image = Image.fromarray(np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3))
    transform = canonical_transform()
    first, second = transform(image), transform(image)
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)
    assert torch.equal(first, second)
    assert first.shape == (3, 224, 224)
    assert torch.isfinite(first).all()
    assert all("Random" not in type(item).__name__ for item in transform.transforms)


def test_manifest_dataset_retains_identity(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (224, 224), (10, 20, 30)).save(path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"instance_id": "7_0", "source_image_id": "7", "class_id": "0", "canonical_label": "craze", "split": "train", "output_relative_path": "image.png", "processed_image_sha256": digest}
    sample = WTBDCropDataset([row], tmp_path, verify_hashes=True)[0]
    assert sample["instance_id"] == "7_0"
    assert sample["source_image_id"] == "7"
    assert sample["processed_image_sha256"] == digest
    assert sample["image"].shape == (3, 224, 224)


def test_real_manifest_split_and_train_only_weights() -> None:
    rows = read_csv("data/processed/wtbd_crops_v1/manifest.csv")
    partitions = split_rows(rows)
    assert {key: len(value) for key, value in partitions.items()} == {"train": 757, "validation": 146, "test": 162}
    sources = {name: {row["source_image_id"] for row in values} for name, values in partitions.items()}
    assert not (sources["train"] & sources["validation"] | sources["train"] & sources["test"] | sources["validation"] & sources["test"])
    weights = balanced_class_weights(partitions["train"])
    counts = torch.tensor([123, 126, 185, 42, 93, 188])
    assert torch.allclose(weights, 757 / (6 * counts))
    assert weights[3] > weights[5]
    try:
        balanced_class_weights(partitions["train"] + partitions["validation"][:1])
    except ValueError as error:
        assert "training rows only" in str(error)
    else:
        raise AssertionError("validation labels affected class weights")


class _Ids(Dataset):
    def __len__(self): return 12
    def __getitem__(self, index): return {"id": index}


def test_training_shuffle_is_seeded() -> None:
    def order(seed: int) -> list[int]:
        return [int(value) for batch in make_loader(_Ids(), batch_size=3, shuffle=True, seed=seed) for value in batch["id"]]
    assert order(17) == order(17)
    assert order(17) != order(29)
