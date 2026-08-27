# Phase 2 — WTBD Curation and Reconciliation

## Status

**BLOCKED_PENDING_HUMAN_REVIEW.** This layer controls the known raw-release ambiguities without editing the official WTBD files. It is reproducible from the immutable raw release, the Phase 2 audit artifacts, `configs/curation.yaml`, and the two review-decision CSV files. Phase 3 has not started.

## Immutable source boundary

The official WTBD archive, extracted JPEGs, primary/secondary XML files, and official split file remain unchanged under `data/raw/wtbd/` and outside Git. The reconciliation commands verify the complete raw fingerprint before and after their work:

`568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb`

All interpretations are stored in versioned metadata. Output-path guards reject attempts to place curation artifacts inside the raw root.

## Identity reconciliation

`scripts/review_wtbd.py` compares every primary XML with its same-ID second-annotator XML and, for all 262 filename mismatches, also compares the image named by the XML file with the image named inside the XML. Diagnostics record filenames, dimensions, classes, coordinates, object counts, same-class box IoUs, exact annotation signatures, thumbnail intensity correlation, and explicit evidence flags.

The four-view sheets under `figures/phase2/identity_review/` show, for each mismatch:

1. XML-named image with primary boxes;
2. embedded-filename image with primary boxes;
3. XML-named image with second-annotator boxes;
4. embedded-filename image with second-annotator boxes.

The automated evidence recommends `xml_name_correct` with high confidence for 119 rows and labels 143 rows `ambiguous`. It also finds that 250 mismatch annotations have an exact primary-annotation signature match to the XML associated with the embedded filename. These are diagnostics, not proof of provenance: **zero automated recommendations are applied as decisions**.

Until reviewed, all 262 identity-mismatch rows have `identity_status=pending_review`, are explicitly excluded by policy, and cannot enter a curated split. Their exact IDs and evidence links are in `curation_blockers.csv`, `identity_diagnostics.csv`, and the identity-review index.

## Second-annotator evidence

The second annotator is never substituted automatically for the primary annotation. Across all 1,065 same-ID XML pairs, the comparison categories are:

| Category | Samples |
|---|---:|
| strong agreement | 922 |
| box disagreement | 15 |
| class disagreement | 87 |
| object-count disagreement | 41 |

The complete per-sample evidence is in `second_annotator_comparison.csv`. A human decision may select primary or secondary annotations only by recording the choice and reviewer identity.

## Human-review workflow

Identity decisions are entered in `data/metadata/wtbd/manual_review_decisions.csv`. Allowed decisions and statuses are generated in `curation_schema.json`. A completed decision requires a reviewer, a compatible resolved filename, a declared annotation source, and an explicit include value. Pending decisions are never interpreted as approval.

Non-exact pair decisions are entered separately in `near_duplicate_review_decisions.csv`. Allowed outcomes are `distinct_capture`, `same_scene`, and `unrelated_false_positive`; pending remains the default. A `same_scene` decision may name a canonical sample. The pipeline excludes every non-canonical member of the resulting group and validates that the group cannot cross curated splits.

Run the workflow with:

```bash
uv run python scripts/review_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml --validate-only
```

Use `--strict` when a nonzero exit is required while review blockers remain. `review_wtbd.py` preserves existing decision files instead of overwriting human work.

## Exact-duplicate policy

Exact groups are formed from both whole-file SHA-256 and decoded-pixel SHA-256. The canonical sample is the lowest natural sample ID and retains its official split; redundant copies are excluded.

| Group | Canonical | Retained split | Excluded redundant sample |
|---|---:|---|---:|
| `exact-001` | 547 | train | 640 (official validation) |
| `exact-002` | 565 | train | 668 (official train) |

The official split file remains untouched. No included exact-duplicate group crosses a curated split.

## Near-duplicate review

The raw audit contains 491 non-exact dHash candidates, including 166 cross-split pairs. `near_duplicate_review_index.csv` prioritizes all 493 candidate rows (the 491 non-exact pairs plus two exact pairs) using exactness, cross-split status, dHash distance, image dimensions, annotation class overlap, object-count agreement, thumbnail pixel differences, and intensity correlation.

No non-exact pair has been automatically declared leakage. All 491 remain pending human review, so Phase 2 remains blocked. The two exact pairs are handled by the exact policy above.

## Raw and provisional curated statistics

The curated statistics are generated from included manifest rows and their primary instances. They are not forced to match the publication.

| Measure | Raw official release | Provisional curated interpretation |
|---|---:|---:|
| images | 1,065 | 801 |
| objects | 1,584 | 1,215 |
| excluded images | 0 | 264 |
| unresolved identity rows included | not applicable | 0 |
| redundant exact copies included | 2 raw copies present | 0 |

The 264 exclusions comprise 262 pending identity rows and two redundant exact copies.

| Class | Raw objects | Provisional curated objects |
|---|---:|---:|
| craze | 257 | 195 |
| corrosion | 257 | 188 |
| surface_injure | 412 | 315 |
| thunderstrike | 92 | 66 |
| crack | 224 | 167 |
| hide_craze | 342 | 284 |

| Split | Official images | Provisional curated images |
|---|---:|---:|
| train | 745 | 564 |
| validation | 159 | 115 |
| test | 161 | 122 |

The authoritative machine-readable versions are `curation_summary.json`, `curated_class_counts.csv`, `curated_split_class_counts.csv`, and `curated_split_membership.csv`.

## Validation and exit gate

The manifest validator enforces centralized enums, complete schema, confirmed annotations for included rows, no included unresolved identity, one resolved image per included sample, and no duplicate group spanning curated splits. Tests also cover raw-path immutability, decision merging, exact canonicalization, reviewed same-scene canonical selection, deterministic statistics, annotator comparison, and fingerprint stability.

The gate does not pass yet:

- all 262 identity rows are safely excluded but await human decisions;
- 491 non-exact near-duplicate pairs await review, including 166 cross-split candidates;
- therefore the provisional 801-image interpretation is not authorized for Phase 3.

`data/metadata/wtbd/curation_blockers.csv` is the exact machine-readable blocker list. No raw file was changed, no publication count was forced, no crop was generated, no model was implemented or trained, and no Phase 3 choice was made.
