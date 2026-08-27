#!/usr/bin/env python3
"""Acquire only the official Springer Nature Figshare WTBD release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from windblade.data.acquisition import (
    AcquisitionBlockedError,
    DATASET_DOI,
    OFFICIAL_FILENAME,
    OFFICIAL_PAGE_URL,
    acquire_wtbd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to the manually downloaded official Figshare archive",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Network timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        result = acquire_wtbd(
            repository_root=repository_root,
            archive=args.archive,
            timeout_seconds=args.timeout,
        )
    except AcquisitionBlockedError as exc:
        manual_target = repository_root / "data" / "raw" / "wtbd" / OFFICIAL_FILENAME
        print(f"Official automated acquisition is blocked: {exc}", file=sys.stderr)
        print(f"Official dataset DOI: {DATASET_DOI}", file=sys.stderr)
        print(f"Official repository: {OFFICIAL_PAGE_URL}", file=sys.stderr)
        print("Download the official archive in a browser, then run:", file=sys.stderr)
        print(f'  python scripts/acquire_wtbd.py --archive "{manual_target}"', file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "acquired",
                "source": "official Springer Nature Figshare",
                "method": result.acquisition_method,
                "archive": str(result.archive_path),
                "archive_size_bytes": result.size_bytes,
                "archive_sha256": result.sha256,
                "source_record": str(result.source_record_path),
                "extracted_top_level_entries": list(result.extracted_top_level_entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
