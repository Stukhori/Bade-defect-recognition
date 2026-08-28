"""Build or validate the deterministic Phase 3 WTBD crop dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data.processed import build_processed_dataset, validate_processed_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/crop_dataset.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    result = (
        validate_processed_dataset(config, root)
        if args.validate_only
        else build_processed_dataset(config, root)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_version": result["dataset_version"],
                "instance_count": result["instance_count"],
                "processed_dataset_fingerprint": result["processed_dataset_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
