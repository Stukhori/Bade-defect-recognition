# Phase 2 — WTBD Dataset Audit

## 1. Source and provenance

- Dataset: WTBD — Wind Turbine Blade Defect dataset.
- Official source: Springer Nature Figshare, DOI `10.6084/m9.figshare.30210175` (version 1).
- Article DOI: `10.1038/s41597-026-06762-x`.
- License: CC BY 4.0.
- Acquisition: automatic official download on 2026-08-27T16:12:59Z.
- Archive: `WT blade defect dataset.zip`, 78958553 bytes.
- Archive SHA-256: `466452f2a0cfc9ef6ba63ea2a3bbc7ea4262057dd07e4fc9e00eedf5bba305b4`.
- Dataset fingerprint: `568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb`.

## 2. Published expectations

The audit tested 1,065 JPEG images, 1,065 primary PASCAL VOC XML files, 1,568 primary objects, six supplied categories, 1024×1024 resolution, and the published per-class counts. It did not treat the separate second-annotator XML directory as additional primary objects.

## 3. Raw file structure

The official archive extracted one top-level directory. Its root entries are:

- `annotation_second_person/`
- `Annotations/`
- `calculate_kappa.py`
- `class_definitions.txt`
- `Fig6_Feature_Visualization.png`
- `generate_split.py`
- `JPEGImages/`
- `preprocessing_demo.py`
- `requirements.txt`
- `train_val_test_split.txt`
- `tsne_analysis.py`

The release includes `Annotations/`, `annotation_second_person/`, `JPEGImages/`, the official split CSV, class definitions, upstream scripts, requirements, and a supplied feature-visualization image. All raw files remain unmodified and Git-ignored. Checksums are in `data/metadata/wtbd/raw_file_checksums.csv`.

## 4. Image integrity

- JPEG files: 1065.
- Successfully decoded: 1065.
- Failed decoding: 0.
- Zero-byte images: 0.
- Resolution counts: `{"1024x1024": 1064, "788x788": 1}`.
- XML/image dimension disagreements: 0.

Every image was decoded with Pillow; dimensions, mode, byte checksum, decoded-pixel checksum, and dHash were recorded rather than trusted from filenames.

## 5. Annotation integrity

- Primary XML files: 1065.
- Second-annotator XML files: 1065.
- Parsed primary objects: 1584.
- Invalid bounding boxes: 0.
- Images without primary XML: 0.
- Primary XML without images: 0.
- XML parse failures: 0.

Geometry is summarized with inclusive PASCAL VOC coordinates because the supplied `calculate_kappa.py` explicitly uses `+1` widths and areas. The supplied t-SNE script instead uses exclusive Python slices; that inconsistency is an upstream reference, not a Phase 3 preprocessing decision.

## 6. Label taxonomy

Exact raw-label counts are recorded in `raw_label_counts.csv`. Canonicalization is limited to unambiguous capitalization variants; raw XML is unchanged.

| Class | Actual | Expected | Difference | Match |
|---|---:|---:|---:|:---:|
| craze | 257 | 259 | -2 | False |
| corrosion | 257 | 254 | 3 | False |
| surface_injure | 412 | 394 | 18 | False |
| thunderstrike | 92 | 92 | 0 | True |
| crack | 224 | 224 | 0 | True |
| hide_craze | 342 | 345 | -3 | False |

## 7. Class distribution

The six classes are imbalanced; `thunderstrike` has the fewest supplied annotations. This is a descriptive dataset property and does not imply physical severity or model difficulty. See `class_counts.csv` and `figures/phase2/class_distribution.png`.

## 8. Source-image/object structure

- Mean objects/image: 1.487324.
- Median: 1.000000; minimum: 1; maximum: 7.
- Exactly 1 object: 696; 2: 268; 3: 75; 4+: 26.
- Images with multiple defect classes: 84.
- Images with repeated instances of the same class: 319.

The source-image co-occurrence matrix is in `class_cooccurrence.csv`. Co-occurrence is not interpreted causally.

## 9. Bounding-box characteristics

For area fraction, min=0.00060844, p05=0.00482306, median=0.05415344, p95=0.28567505, max=0.64680004. For aspect ratio, min=0.042829, median=0.941701, max=22.850000.

Diagnostic counts (not exclusion rules): width <16: 0; height <16: 0; width <32: 24; height <32: 10; area fraction <0.1%: 5; <0.5%: 82; <1%: 220; aspect ratio <0.1: 26; aspect ratio >10: 15; area fraction >50%: 12.

Within-image box pairs: 751; IoU >0: 38; IoU ≥0.25: 0; IoU ≥0.50: 0; IoU ≥0.75: 0. No box was filtered or merged.

## 10. Official split

- Source: `WT blade defect dataset/train_val_test_split.txt`.
- Raw format: `CSV with header ImageID,Subset; raw subsets train,val,test`.
- SHA-256: `87ab0b1a268b1dd915502db1e2bfc424cde5f9243aa57a3855345123315dd571`.

| Split | Source images | Proportion | Instances |
|---|---:|---:|---:|
| train | 745 | 0.6995 | 1100 |
| validation | 159 | 0.1493 | 246 |
| test | 161 | 0.1512 | 238 |

Overlap IDs: 0; duplicate rows: 0; omitted IDs: 0; unknown IDs: 0.

Future crops such as `123_0`, `123_1`, and `123_2` must all inherit image `123`'s official split. Individual crops must never be randomly split.

## 11. Duplicate audit

- Exact file-duplicate groups: 2.
- Exact decoded-pixel groups: 2.
- Cross-split exact duplicate pairs: 1.
- Non-exact dHash candidates at distance ≤4: 491.
- Cross-split non-exact candidates: 166.

Perceptual matches are **candidate near duplicates requiring review**, not established duplicates. Pair metadata are in `duplicate_candidates.csv`; visual sheets prioritize cross-split candidates.

## 12. Visual annotation review

- Per-class annotation contact sheets: `figures/phase2/annotation_examples/`.
- Near-duplicate review sheets: `figures/phase2/near_duplicates/`.
- High-overlap review: `figures/phase2/high_overlap/`.
- Dataset plots: `figures/phase2/`.

Sampling is deterministic with the Phase 2 visualization seed and is for quality control only. No audit crop was added to a model-ready dataset.

## 13. Implications for Crop-Based Classification

WTBD supplies 1584 potential future defect crops across 1065 source images. Crop-relevant concerns for Phase 3 include variable box area/aspect ratio, 220 boxes below 1% of image area, multiple objects/classes in some source images, 38 overlapping pairs, and split-level class imbalance. All crops from one source image must remain grouped. These facts neither validate nor invalidate crop classification and do not determine a context margin.

## Upstream preprocessing reference

The included `preprocessing_demo.py` reads common image extensions with OpenCV, resizes to 1024×1024 with `cv2.INTER_AREA`, and writes new files; it states no normalization. The included t-SNE code extracts annotation ROIs, resizes them to 64×128, and computes HOG/LBP features. None of these choices were executed or adopted for this project. Phase 3 retains authority over crop and preprocessing design.

## 14. Critical errors

- 262 XML filename fields disagree with XML IDs.
- Primary instance count is 1584, expected 1568.
- At least one canonical class count differs from the published expectation.
- 1 exact file/pixel duplicate pairs cross official splits.

## 15. Warnings

- Observed resolution distribution differs from 1065×1024x1024: {'1024x1024': 1064, '788x788': 1}
- Found 2 exact-file and 2 exact-pixel duplicate groups; none were removed.
- Found 491 non-exact dHash candidates at distance <= 4 requiring human review.
- 166 non-exact near-duplicate candidates cross official splits.
- 220 valid boxes occupy less than 1% of the source image.
- Diagnostic elongation flags: 26 boxes have aspect ratio < 0.1 and 15 exceed 10.0.
- 12 boxes exceed the diagnostic area-fraction threshold 0.5.
- 38 within-image annotation pairs overlap (IoU > 0).
- 84 source images contain multiple canonical defect classes.

## 16. Curation/reconciliation layer

The immutable raw audit is followed by the versioned `wtbd-curation-v1` reconciliation layer documented in `docs/phase2_curation.md`. It never edits an image, XML, or official split file.

- Curation status: **BLOCKED_PENDING_HUMAN_REVIEW**.
- Identity mismatches: 262; automatically resolved: 0; manually resolved: 0; policy-excluded while pending: 262.
- Provisional curated images/objects: 801 / 1215.
- Excluded images: 264, including 2 redundant exact copies.
- Exact duplicate groups crossing curated splits: 0.
- Pending non-exact near-duplicate pairs: 491, including 166 cross-split candidates.

Every unresolved row is listed in `data/metadata/wtbd/curation_blockers.csv`. Recommendations are evidence only and are never silently applied as decisions.


## 17. Phase 2 exit-gate status

**INCOMPLETE**. All counts and conclusions in this document were generated from the machine-readable audit, not manually transcribed. No model was trained, no model weights were downloaded, no final classification crops were created, and no Phase 3 preprocessing choice was frozen.
