# Phase 2 — WTBD Curation and Reconciliation

## Status

**COMPLETE.** The reviewed `wtbd-curation-v1` benchmark passes the Phase 2 exit gate. Phases 0 and 1 remain frozen, and Phase 3 has not started.

This curation is a separate, reproducible interpretation of the official release. It does not modify raw files and does not claim to reconstruct the dataset authors' intended 1,568-instance dataset.

## Immutable source and raw discrepancies

The official archive, JPEGs, XML files, and split file remain unchanged under `data/raw/wtbd/`. The complete fingerprint was verified before import, during regeneration, and after curation:

`568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb`

The raw official release remains forensically discrepant:

- the publication reports 1,568 defect objects, while the primary XML files contain 1,584;
- the observed raw class counts differ from the publication;
- 262 XML files contain stale/mismatched embedded image identities;
- two exact duplicate groups are present;
- the official split contains non-exact same-scene leakage.

Raw audit tables remain separate from curated tables. No raw count was forced to match the publication.

## Human-review import

The reviewed archive `phase2_reviewed_decisions_2026-08-28.zip` had SHA-256:

`587d847afcf014d9276ce78eabce1d79a30349a3c853e453a40d35999b1df1e8`

Its six review-data files and their checksums are preserved under `data/metadata/wtbd/human_review/`. The archive's extra embedded prompt was not treated as project authority and was not imported. `import_manifest.json` records this boundary.

### Identity decisions

All 262 identity-mismatch rows received completed human decisions:

- decision: `mark_annotation_reused`;
- include: `False`;
- resulting identity status: `annotation_reused_from_other_image`;
- resulting reason: `annotation_reused`.

They were reviewed as reused/derived variants of earlier source scenes and excluded to preserve source-scene independence. This is not a claim that their box geometry is invalid. No identity decision remains pending, and no unresolved identity row is included.

The supporting evidence remains in `identity_diagnostics.csv`, `second_annotator_comparison.csv`, and `figures/phase2/identity_review/`. Automated recommendations were never substituted for human decisions.

## Exact duplicates

Exact groups use whole-file and decoded-pixel hashes. The lowest natural sample ID is canonical and retains its official split.

| Group | Canonical retained | Split | Redundant excluded |
|---|---:|---|---:|
| `exact-001` | 547 | train | 640 |
| `exact-002` | 565 | train | 668 |

No exact group is represented in more than one curated split. The official split file was not edited and redundant files were not moved.

## Reviewed non-exact duplicates

The 491 non-exact dHash candidates received:

| Decision | Pairs |
|---|---:|
| `same_scene` | 122 |
| `unrelated_false_positive` | 8 |
| `pending_review` | 361 |

dHash remains a screening heuristic, not proof of duplication. Completed `same_scene` edges form an undirected graph with 50 connected source-scene components and 131 members. For each component, the lowest natural sample ID is the deterministic canonical; 81 non-canonical members are excluded with status `near_duplicate_same_scene` and reason `near_duplicate_same_scene_redundant`. Component IDs `scene-001` through `scene-050` are stored in the manifest and exactly reproduce `reviewed_same_scene_components.csv`.

The 71 reviewed non-exact cross-split pairs that survived identity/exact curation comprise 63 `same_scene` and eight `unrelated_false_positive` decisions. After component deduplication, only the eight reviewed false positives remain represented across splits.

The 361 intentionally pending candidates are non-blocking under the revised evidence-based rule:

| Pending category | Pairs | Blocks Phase 2? |
|---|---:|:---:|
| involves at least one excluded image | 283 | No |
| both retained within one split | 78 | No |
| both retained across different splits | 0 | Yes, if present |

No pending row was silently reclassified.

## Final curated benchmark

| Measure | Raw official release | Curated benchmark |
|---|---:|---:|
| images | 1,065 | 720 |
| objects | 1,584 | 1,065 |
| excluded images | 0 | 345 |

The 345 exclusions consist of 262 reviewed identity variants, two redundant exact copies, and 81 redundant reviewed same-scene component members.

### Image split

| Split | Images | Percentage |
|---|---:|---:|
| train | 510 | 70.83% |
| validation | 101 | 14.03% |
| test | 109 | 15.14% |

### Object counts

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| craze | 169 | 123 | 19 | 27 |
| corrosion | 178 | 126 | 22 | 30 |
| surface_injure | 264 | 185 | 46 | 33 |
| thunderstrike | 60 | 42 | 9 | 9 |
| crack | 131 | 93 | 24 | 14 |
| hide_craze | 263 | 188 | 26 | 49 |
| **Total** | **1,065** | **757** | **146** | **162** |

All six classes remain represented in train, validation, and test.

## Reproduction and validation

```bash
uv run python scripts/review_wtbd.py --config configs/curation.yaml --no-images
uv run python scripts/curate_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml --validate-only --strict
uv run python -m pytest
```

The strict validator independently checks the raw fingerprint, manifest schema, identity decisions, exact groups, recomputed same-scene graph, human component cross-check, cross-split review table, pending-pair partition, split/class statistics, supplied expected-summary assertions, and empty blocker list.

The final machine-readable sources are `curation_manifest.csv`, `curated_instances.csv`, `curated_class_counts.csv`, `curated_split_membership.csv`, `curated_split_class_counts.csv`, `curation_summary.json`, and `curation_blockers.csv`.

## Scientific interpretation

The curated benchmark is intentionally different from the raw release. It is designed to reduce source-scene leakage for this experiment using stale-identity review, exact hashing, human-reviewed perceptual candidates, and connected-component deduplication. The project does not claim that the 720-image/1,065-object result reconstructs the authors' intended dataset or that every remaining within-split visual similarity is an independent physical scene.

No model code or pretrained weights were added, no model was trained, no crop preprocessing was frozen, and Phase 3 was not started.
