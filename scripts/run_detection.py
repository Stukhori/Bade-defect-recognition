"""Run or validate the deterministic Phase 11A full-image detection audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.detection import apparatus_check, run_audit, validate_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detection.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apparatus-check", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.train:
        raise SystemExit(
            "Phase 11B is blocked: CUDA and a separately pinned pre-test detector dependency/weight "
            "record are required. Phase 11A intentionally performs no training."
        )
    if args.apparatus_check:
        result = apparatus_check(config, root)
    elif args.validate_only:
        result = validate_audit(config, root)
    else:
        result = run_audit(config, root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
