#!/usr/bin/env python3
"""Run the Phase 1 synthetic-only infrastructure smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.smoke import run_smoke_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic synthetic Phase 1 smoke experiment."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration path")
    parser.add_argument("--seed", type=int, help="Override the synthetic smoke seed")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the run output root; the override is included in the resolved config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    outcome = run_smoke_experiment(
        config_path=args.config,
        repository_root=repository_root,
        seed_override=args.seed,
        output_root_override=args.output_root,
    )
    print(
        json.dumps(
            {
                "synthetic_only": True,
                "status": "completed",
                "experiment_id": outcome.experiment_id,
                "config_hash": outcome.config_hash,
                "seed": outcome.seed,
                "run_dir": str(outcome.run_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
