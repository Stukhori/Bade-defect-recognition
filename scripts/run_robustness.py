"""Run, apparatus-check, or validate the canonical Phase 8 robustness experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.robustness.runner import apparatus_check, run_robustness, validate_robustness_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robustness.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apparatus-check", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.apparatus_check:
        result = apparatus_check(config, root)
    elif args.validate_only:
        result = validate_robustness_results(config, root)
    else:
        result = run_robustness(config, root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
