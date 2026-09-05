"""Run the pinned Phase 11B Colab apparatus without weakening the test firewall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windblade.detection.phase11b import (
    DriveLayout,
    acquire_official_weight,
    atomic_json,
    environment_preflight,
    generate_training_bundle,
    load_apparatus,
    materialize_dataset,
    run_final_test,
    select_validation_configuration,
    train_seed,
    validate_frozen_inputs,
    verify_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detection_phase11b.yaml")
    parser.add_argument("--drive-root")
    parser.add_argument("--data-root")
    parser.add_argument("--archive")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("apparatus-check")
    commands.add_parser("preflight")
    acquire = commands.add_parser("acquire-weight")
    acquire.add_argument("--apparatus-commit", required=True)
    commands.add_parser("verify-archive")
    commands.add_parser("materialize-trainval")
    train = commands.add_parser("train")
    train.add_argument("--seed", required=True, type=int)
    commands.add_parser("select-validation")
    commands.add_parser("bundle")
    commands.add_parser("final-test")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = (repo / args.config).resolve()
    config = load_apparatus(config_path)
    colab = config["colab"]
    drive_root = Path(args.drive_root or colab["drive_root"])
    data_root = Path(args.data_root or colab["local_data_root"])
    archive = Path(args.archive or (drive_root / colab["archive_relative_path"]))
    weight = drive_root / colab["weight_relative_path"]
    weight_record = drive_root / colab["weight_record_relative_path"]
    receipt = repo / config["firewall"]["committed_selection_receipt"]

    if args.command == "apparatus-check":
        result = validate_frozen_inputs(config, repo)
    elif args.command == "verify-archive":
        result = verify_archive(config, archive)
    elif args.command == "preflight":
        layout = DriveLayout.from_root(drive_root)
        result = environment_preflight(config, repo, archive, drive_root)
        atomic_json(layout.provenance / "environment_preflight.json", result)
    elif args.command == "acquire-weight":
        result = acquire_official_weight(config, repo, weight, args.apparatus_commit, weight_record)
    elif args.command == "materialize-trainval":
        result = materialize_dataset(config, repo, archive, data_root, scope="trainval")
    elif args.command == "train":
        result = train_seed(config, config_path, repo, data_root, drive_root, weight, weight_record, args.seed)
    elif args.command == "select-validation":
        result = select_validation_configuration(config, config_path, repo, data_root, drive_root, receipt)
    elif args.command == "bundle":
        result = generate_training_bundle(config, config_path, repo, drive_root, receipt)
    else:
        result = run_final_test(config, config_path, repo, archive, data_root, drive_root)
    display = dict(result)
    if "artifacts" in display:
        display["artifact_count"] = len(display["artifacts"])
        display.pop("artifacts")
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
