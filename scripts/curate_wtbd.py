#!/usr/bin/env python3
"""Build or validate the versioned WTBD curation manifest and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data.curation import build_curation, validate_existing_curation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/curation.yaml"))
    parser.add_argument("--output-dir", type=Path, help="override the curated metadata output directory")
    parser.add_argument("--validate-only", action="store_true", help="validate existing artifacts without rewriting them")
    parser.add_argument("--strict", action="store_true", help="return exit code 2 while human-review blockers remain")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    if args.validate_only:
        result = validate_existing_curation(config, root)
    else:
        output = None if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else root / args.output_dir)
        result = build_curation(config, root, output_directory=output)
    print(
        json.dumps(
            {
                "status": result.status,
                "manifest": str(result.manifest_path),
                "summary": str(result.summary_path),
                "blockers": list(result.blockers),
            },
            sort_keys=True,
        )
    )
    return 2 if args.strict and result.status != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
