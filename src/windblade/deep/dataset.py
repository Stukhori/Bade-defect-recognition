"""Manifest-driven loading of the immutable Phase 3 crop dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from windblade.data.processed import LABELS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def canonical_transform() -> v2.Compose:
    """Return the identical deterministic transform used for every split."""

    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class WTBDCropDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        processed_root: str | Path,
        *,
        verify_hashes: bool = False,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.root = Path(processed_root)
        self.transform = canonical_transform()
        for row in self.rows:
            class_id = int(row["class_id"])
            if class_id not in range(len(LABELS)) or row["canonical_label"] != LABELS[class_id]:
                raise ValueError(f"invalid frozen label mapping: {row['instance_id']}")
            path = self.root / row["output_relative_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and _sha256(path) != row["processed_image_sha256"]:
                raise ValueError(f"processed image hash mismatch: {row['instance_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.root / row["output_relative_path"]
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                raise ValueError(f"non-canonical Phase 3 crop: {row['instance_id']}")
            tensor = self.transform(image.copy())
        if tuple(tensor.shape) != (3, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid tensor: {row['instance_id']}")
        return {
            "image": tensor,
            "class_id": int(row["class_id"]),
            "instance_id": row["instance_id"],
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "processed_image_sha256": row["processed_image_sha256"],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    result = {"train": [], "validation": [], "test": []}
    source_split: dict[str, str] = {}
    for original in rows:
        row = dict(original)
        split = row["split"]
        if split not in result:
            raise ValueError(f"unexpected split: {split}")
        source_id = row["source_image_id"]
        previous = source_split.setdefault(source_id, split)
        if previous != split:
            raise ValueError(f"source crosses partitions: {source_id}")
        result[split].append(row)
    expected = {"train": 757, "validation": 146, "test": 162}
    if {name: len(items) for name, items in result.items()} != expected:
        raise ValueError("Phase 3 split counts changed")
    return result


def balanced_class_weights(train_rows: Sequence[Mapping[str, str]]) -> torch.Tensor:
    """Calculate N/(K*N_c), exclusively from rows declared as training."""

    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("class weights accept training rows only")
    counts = np.bincount([int(row["class_id"]) for row in train_rows], minlength=len(LABELS))
    if len(train_rows) != 757 or counts.tolist() != [123, 126, 185, 42, 93, 188]:
        raise ValueError("frozen Phase 3 training class counts changed")
    return torch.tensor(len(train_rows) / (len(LABELS) * counts), dtype=torch.float32)


def make_loader(
    dataset: Dataset[dict[str, Any]], *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
