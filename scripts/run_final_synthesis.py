"""Check, run, or validate the canonical Phase 10 final synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.final_synthesis import apparatus_check, run_final_synthesis, validate_final_synthesis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_synthesis.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apparatus-check", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.apparatus_check:
        result = apparatus_check(config, root)
    elif args.validate_only:
        result = validate_final_synthesis(config, root)
    else:
        result = run_final_synthesis(config, root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
