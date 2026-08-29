"""Run or validate the frozen Phase 7 data-efficiency experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.data_efficiency import run_data_efficiency, validate_data_efficiency_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_efficiency.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    result = (
        validate_data_efficiency_results(config, root)
        if args.validate_only
        else run_data_efficiency(config, root)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
