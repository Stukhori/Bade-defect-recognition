"""Run or validate the canonical Phase 6 MobileNetV3-Small baseline."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from windblade.config import load_config
from windblade.mobilenet_experiment import run_mobilenet, validate_mobilenet_results

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mobilenet_v3_small_baseline.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; config = load_config(root / args.config)
    result = validate_mobilenet_results(config, root) if args.validate_only else run_mobilenet(config, root)
    print(json.dumps(result, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
