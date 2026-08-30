"""Run, apparatus-check, or validate the canonical Phase 9A analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.error_analysis.runner import apparatus_check, run_error_analysis, validate_error_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/error_analysis.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apparatus-check", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.apparatus_check:
        result = apparatus_check(config, root)
    elif args.validate_only:
        result = validate_error_analysis(config, root)
    else:
        result = run_error_analysis(config, root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
