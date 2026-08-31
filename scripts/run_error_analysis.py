"""Run or validate the canonical Phase 9A and Phase 9B analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.error_analysis.phase9b import run_phase9b, validate_phase9b
from windblade.error_analysis.runner import apparatus_check, run_error_analysis, validate_error_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/error_analysis.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apparatus-check", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--phase9b", action="store_true")
    mode.add_argument("--validate-phase9b", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.apparatus_check:
        result = apparatus_check(config, root)
    elif args.validate_only:
        result = validate_error_analysis(config, root)
    elif args.phase9b:
        result = run_phase9b(config, root)
    elif args.validate_phase9b:
        result = validate_phase9b(config, root)
    else:
        result = run_error_analysis(config, root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
