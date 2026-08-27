#!/usr/bin/env python3
"""Run the configuration-driven WTBD forensic dataset audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data.audit import run_wtbd_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_audit.yaml"),
        help="Phase 2 audit YAML configuration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    config = load_config(config_path)
    result = run_wtbd_audit(config, repository_root)
    print(
        json.dumps(
            {
                "status": result.status,
                "summary": str(result.summary_path),
                "documentation": str(result.documentation_path),
                "critical_error_count": len(result.critical_errors),
                "warning_count": len(result.warnings),
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
