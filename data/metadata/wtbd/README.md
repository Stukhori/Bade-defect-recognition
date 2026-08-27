# WTBD Audit Metadata

This tracked directory contains small, deterministic metadata derived from the immutable official WTBD release in `data/raw/wtbd/`. It contains no image pixels and no model-ready crops.

- `source.json` records official Springer Nature Figshare provenance and the archive checksum.
- `audit_resolved_config.yaml` is the exact Phase 2 audit configuration.
- `raw_file_checksums.csv` and `audit_summary.json` define the reproducible raw-dataset fingerprint.
- `images.csv` and `instances.csv` inventory decoded images and primary VOC objects.
- `raw_label_counts.csv` and `class_counts.csv` preserve exact labels and compare canonical counts with published expectations.
- `split_membership.csv` and `split_class_counts.csv` describe the official source-image split.
- `class_cooccurrence.csv`, `bbox_statistics.csv`, `very_small_bbox_counts.csv`, and `overlap_pairs.csv` contain crop-relevant descriptive evidence.
- `duplicate_candidates.csv` contains exact and dHash-based review candidates; dHash matches are not automatically treated as duplicates.
- `upstream_files.csv` inventories supplied code/reference files without executing them.
- `audit_findings.csv` distinguishes critical errors from scientific warnings.

Regenerate these records with:

```bash
uv run python scripts/audit_wtbd.py --config configs/dataset_audit.yaml
```

The official raw release is CC BY 4.0 and remains excluded from Git.
