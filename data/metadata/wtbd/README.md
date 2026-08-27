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
- `curation_manifest.csv` records one versioned interpretation row per logical source-image sample.
- `curation_schema.json` centralizes manifest columns and allowed decision/status enums.
- `identity_diagnostics.csv`, `second_annotator_comparison.csv`, and `review_summary.json` contain identity and annotator evidence.
- `manual_review_decisions.csv` and `near_duplicate_review_decisions.csv` are the only human-decision inputs; pending rows are never silently approved.
- `near_duplicate_review_index.csv` prioritizes exact and non-exact candidates without declaring dHash matches to be leakage.
- `curation_blockers.csv` lists every identity or pair row still requiring review.
- `curation_summary.json` and the `curated_*` CSV files keep provisional curated statistics separate from the raw audit.

Regenerate these records with:

```bash
uv run python scripts/audit_wtbd.py --config configs/dataset_audit.yaml
uv run python scripts/review_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml
```

The official raw release is CC BY 4.0 and remains excluded from Git. Current curation status is `BLOCKED_PENDING_HUMAN_REVIEW`; the provisional curated metadata is not authorized for Phase 3.
