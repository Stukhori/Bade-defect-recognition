# Phase 2 Human Review — WTBD

Date: 2026-08-28
Reviewer: OpenAI human review layer

## Decision summary

The official WTBD release should **not** be used exactly as released for the planned classification experiment. The review artifacts show substantial source-scene duplication and stale image identities.

### 1. Identity mismatches

All 262 XML identity-mismatch samples are treated as **reused/derived variants of earlier source scenes** and excluded from the core experiment.

Evidence used:

- every mismatched XML points backward to an earlier existing image ID;
- 250/262 primary annotation signatures exactly repeat the declared earlier sample;
- the two annotators strongly agree on the current annotation in most cases;
- supplied human-review sheets show the XML-stem image and declared image are the same underlying blade/turbine scene with photometric, exposure, contrast, grayscale, or closely related preprocessing differences;
- the 12 non-exact-reuse cases were individually inspected in the supplied sheets and show the same pattern.

This is **not** a claim that the current boxes are unusable. The exclusion is a source-independence decision: keeping both a source image and its transformed/processed counterpart would inflate sample count and can create train/validation/test leakage.

Decision written for all 262 rows: `mark_annotation_reused`, `include=False`.

### 2. Exact duplicates

Retain the existing Phase 2 decisions:

- keep 547; exclude 640;
- keep 565; exclude 668.

### 3. Non-exact near duplicates

The supplied near-duplicate sheets were visually reviewed. The high-priority pairs are overwhelmingly photometric/processing variants of the same source photograph rather than independent captures.

For candidates where both images survived provisional identity/exact-duplicate curation, review combined:

- visual evidence from supplied review sheets;
- dHash distance;
- thumbnail intensity correlation;
- annotation class overlap;
- object-count agreement;
- primary bounding-box geometry where available;
- transitive membership in an already-confirmed same-scene component.

Decisions written:

- **122 pair rows:** `same_scene`
- **8 cross-split pair rows:** `unrelated_false_positive`
- remaining unresolved rows are intentionally left `pending_review` because they either involve already-excluded samples or are within a single split and do not create train/test leakage.

Among the 71 non-exact cross-split candidates where **both** samples survived the provisional curation:

- **63 are reviewed as the same source scene**;
- **8 are reviewed as dHash false positives**;
- **0 remain unresolved**.

### 4. Same-scene component policy

Reviewed same-scene edges form **50 connected scene components** among provisional included samples.

Use the lowest natural sample ID as the deterministic canonical representative of each component. Exclude all other members as `near_duplicate_same_scene_redundant`.

This is consistent with the existing exact-duplicate canonical policy and tends to preserve the earlier/original-looking sample rather than later derived variants.

Expected additional exclusions from this rule: **81 images**.

## Expected curated dataset after applying the decisions

Starting provisional curation:

- 801 included images
- 1,215 objects

After reviewed same-scene deduplication:

- **720 included images**
- **1,065 annotated objects**

Expected image split:

| Split | Images | Percent |
|---|---:|---:|
| Train | 510 | 70.83% |
| Validation | 101 | 14.03% |
| Test | 109 | 15.14% |

Expected object counts by class:

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| craze | 169 | 123 | 19 | 27 |
| corrosion | 178 | 126 | 22 | 30 |
| surface_injure | 264 | 185 | 46 | 33 |
| thunderstrike | 60 | 42 | 9 | 9 |
| crack | 131 | 93 | 24 | 14 |
| hide_craze | 263 | 188 | 26 | 49 |
| **Total** | **1,065** | **757** | **146** | **162** |

All six classes remain present in all three partitions.

After applying the reviewed same-scene exclusions, the only remaining cross-split dHash candidates should be the **8 rows explicitly reviewed as false positives**. There should be no unresolved cross-split candidate involving two included images.

## Revised Phase 2 blocking rule

The original curation workflow treated every pending dHash candidate as a blocker. That is too strict because dHash is a screening heuristic, not proof of duplication.

A pending near-duplicate row should block Phase 2 **only when all of the following are true**:

1. both images are still included after identity, exact-duplicate, and completed same-scene curation;
2. the pair crosses curated train/validation/test partitions;
3. no completed human decision classifies the pair as `same_scene`, `distinct_capture`, or `unrelated_false_positive`.

Therefore:

- pending rows involving an already-excluded image are irrelevant to the curated experiment;
- pending pairs fully within one split are warnings, not Phase 2 blockers;
- confirmed same-scene groups must be deduplicated before Phase 3;
- remaining cross-split candidates reviewed as false positives may stay.

## Scientific interpretation

This review changes how WTBD should be described in the research paper.

Do **not** write that the official dataset contains 1,065 independent UAV observations suitable for random use as released. The released files contain processed/reused scene variants and the official split contains same-scene leakage.

For the project, describe the procedure approximately as:

> We performed a source-level forensic audit of WTBD before model training. Images with stale identity metadata, exact duplicates, and high-confidence processed variants of the same source scene were removed using annotation agreement, perceptual similarity, visual review, and connected-component deduplication. The resulting curated benchmark contained 720 source-level images and 1,065 labeled defect instances across the six WTBD classes.

Do not claim the curation reconstructs the authors' intended 1,568-instance dataset. It constructs a defensible, leakage-reduced benchmark for this study.

## Files in this review bundle

- `manual_review_decisions_reviewed.csv`
- `near_duplicate_review_decisions_reviewed.csv`
- `reviewed_same_scene_components.csv`
- `cross_split_review_summary.csv`
- `expected_post_review_summary.json`
- `codex_apply_review_prompt.md`
