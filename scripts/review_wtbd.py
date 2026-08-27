#!/usr/bin/env python3
"""Generate deterministic WTBD identity and duplicate review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data.curation import build_review_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/curation.yaml"))
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="regenerate tabular diagnostics without rendering identity review sheets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    summary = build_review_evidence(load_config(config_path), root, generate_images=not args.no_images)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
