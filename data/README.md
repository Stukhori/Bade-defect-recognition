# Data Directory Policy

WTBD version 1 was acquired in Phase 2 from the official Springer Nature Figshare release under CC BY 4.0. Its externally licensed image/XML/archive payload remains excluded from Git.

- `raw/` is reserved for immutable original downloaded source data. Never edit, rename internally, overwrite, or transform raw dataset files in place.
- `metadata/` contains small, versionable provenance, checksums, raw-audit tables, review decisions, the curation manifest, and raw/curated summaries derived without altering raw files.
- `interim/` is reserved for intermediate transformations that can be recreated from raw data and recorded configuration.
- `processed/` contains the versioned `wtbd_crops_v1` metadata and the regenerable, Git-ignored 224 × 224 PNG payload.
- `splits/` contains small, version-controlled instance split and nested source-group subset manifests.

Raw and interim payloads and generated crop pixels are ignored by Git. Phase 3 configs, processed manifests, checksums, fingerprints, summaries, and split definitions are versioned. Regenerate the crop payload with `python scripts/build_wtbd_crops.py --config configs/crop_dataset.yaml`, followed by `python scripts/build_training_subsets.py --config configs/crop_dataset.yaml`.
