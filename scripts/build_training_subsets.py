"""Build or validate the frozen Phase 3 WTBD training subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data.subsets import build_training_subsets, validate_training_subsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/crop_dataset.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    result = (
        validate_training_subsets(config, root)
        if args.validate_only
        else build_training_subsets(config, root)
    )
    print(
        json.dumps(
            {
                "dataset_version": result["dataset_version"],
                "scientific_seeds": sorted(int(seed) for seed in result["seeds"]),
                "nesting_validated": result["nesting_validated"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
