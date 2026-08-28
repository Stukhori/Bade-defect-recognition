"""Run or validate the canonical Phase 5 ResNet-18 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.resnet_experiment import run_resnet18_baseline, validate_resnet18_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/resnet18_baseline.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    result = validate_resnet18_results(config, root) if args.validate_only else run_resnet18_baseline(config, root)
    print(json.dumps(result if args.validate_only else {"status": result["status"], "selected": result["selected"], "aggregate": result["aggregate"]["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
