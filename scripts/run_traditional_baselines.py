"""Run the canonical Phase 4 HOG/LBP plus RBF-SVM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.config import load_config
from windblade.traditional import run_traditional_baselines, validate_traditional_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/traditional_baselines.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.validate_only:
        result = validate_traditional_results(config, root)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run_traditional_baselines(config, root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_id": result["result_id"],
                "processed_dataset_fingerprint": result["processed_dataset_fingerprint"],
                "validation_grid_fingerprint": result["validation_grid_fingerprint"],
                "selected": {
                    family: {"C": row["C"], "gamma": row["gamma"]}
                    for family, row in result["selected"].items()
                },
                "test_macro_f1": {
                    family: metrics["macro_f1"] for family, metrics in result["metrics"].items()
                },
                "deterministic_repeat_passed": result["deterministic_repeat_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
