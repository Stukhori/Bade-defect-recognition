# Project State

## Current phase

- **Phase:** Phase 2 — WTBD Dataset Acquisition and Forensic Audit
- **Status:** COMPLETE — reviewed curation passes the Phase 2 exit gate
- **Start date:** 2026-08-28
- **Previous phases:** Phase 0 and Phase 1 — complete and frozen
- **Next phase:** Phase 3 — Frozen Split Representation and Crop/Preprocessing Design

Phase 2 is complete. Phase 0 research decisions and Phase 1 infrastructure remain frozen. Phase 3 has not started; beginning it requires a separate explicit authorization.

## Phase 2 — WTBD Dataset Acquisition and Forensic Audit

Phase 2 treats the official WTBD release as immutable source evidence. It may acquire and audit the dataset, but it does not create final classification crops, choose preprocessing or scientific hyperparameters, download model weights, or train models.

### Phase 2 status

**COMPLETE.** The raw forensic audit still records the official-release discrepancies, while the reviewed `wtbd-curation-v1` interpretation passes its separate leakage-control gate. The expanded suite reports **59 passed and 0 failed**. Strict manifest validation reports `PASS` with no blockers.

### Acquisition and immutable provenance

- Official source: Springer Nature Figshare, dataset DOI `10.6084/m9.figshare.30210175`, version 1, CC BY 4.0.
- Method: automatic official download.
- Archive: `WT blade defect dataset.zip`, 78,958,553 bytes.
- Official MD5: `14ad7e2cf7161b9100d1d70fb398b0cf`.
- Archive SHA-256: `466452f2a0cfc9ef6ba63ea2a3bbc7ea4262057dd07e4fc9e00eedf5bba305b4`.
- Raw dataset fingerprint: `568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb` (reproduced before and after review/curation generation).
- Raw contents remain unmodified and excluded from Git.

### Verified release properties

- 1,065/1,065 JPEG images decode successfully; 1,064 are 1024×1024 and `714.jpg` is 788×788, matching its XML dimensions.
- 1,065 primary XML files and 1,065 separate second-annotator XML files are present.
- Image/XML pairing by file ID is complete; no unmatched or duplicate IDs were found.
- All primary XML files parse and all 1,584 parsed bounding boxes pass geometric validation under inclusive VOC coordinates.
- Exact raw labels are the expected six strings, but their observed counts differ from the published counts.
- Official split CSV is disjoint and exhaustive: 745 train, 159 validation, 161 test; zero overlap, omitted, unknown, or duplicate IDs.

### Critical official-release discrepancies

1. **XML identity ambiguity:** 262 primary XML `<filename>` values disagree with their XML/image file IDs. These occur across XML IDs 743–1062 in discontinuous ranges. Automated comparison and four-view sheets now exist, but no recommendation is silently applied as a decision.
2. **Object-count mismatch:** primary XML contains 1,584 objects, not the published 1,568.
3. **Class-count mismatch:** actual counts are craze 257 (expected 259), corrosion 257 (254), surface_injure 412 (394), thunderstrike 92 (92), crack 224 (224), and hide_craze 342 (345).
4. **Cross-split exact duplicate:** `547.jpg` (train) and `640.jpg` (validation) are identical both as files and decoded pixels. A second exact pair, `565.jpg`/`668.jpg`, remains within train.

These findings are preserved as raw evidence. No label, filename field, official split, annotation, or image was altered. The separate curated benchmark is derived only from the versioned manifest and does not claim to reconstruct the authors' intended 1,568-object dataset.

### Raw scientific warnings and review candidates

- 491 non-exact dHash candidates at distance ≤4; 166 cross official splits. dHash remains a screening heuristic rather than automatic proof of duplication.
- 220 boxes occupy less than 1% of source-image area; 5 occupy less than 0.1%.
- Bounding-box area fraction ranges from approximately 0.000608 to 0.646800; aspect ratio ranges from approximately 0.042829 to 22.85.
- Diagnostic flags identify 26 boxes with aspect ratio <0.1, 15 with aspect ratio >10, and 12 occupying more than 50% of their source image; these are not exclusion rules.
- 38 of 751 within-image object pairs overlap at IoU >0; none reach IoU 0.25.
- 84 source images contain multiple canonical classes; 319 contain repeated instances of a class.
- Objects per image: mean 1.487324, median 1, minimum 1, maximum 7; 696 images have one object, 268 have two, 75 have three, and 26 have four or more.

### Curation/reconciliation state

- Curation version: `wtbd-curation-v1`; schema and allowed enums are recorded in `curation_schema.json`.
- Human-review archive SHA-256: `587d847afcf014d9276ce78eabce1d79a30349a3c853e453a40d35999b1df1e8`; imported file hashes and the embedded-instruction exclusion are recorded in `human_review/import_manifest.json`.
- All 262 stale/mismatched identity rows were reviewed as reused/derived source-scene variants and excluded. Zero identity decisions remain pending and no unresolved identity is included.
- Second-annotator comparison: 922 strong agreement, 15 box disagreement, 87 class disagreement, and 41 object-count disagreement cases. Second annotations remain evidence only.
- Exact groups: retain `547` in train and exclude `640`; retain `565` in train and exclude `668`. No included exact group crosses curated splits.
- Human pair decisions: 122 `same_scene`, eight `unrelated_false_positive`, and 361 intentionally pending candidates.
- The 122 same-scene edges form 50 reviewed components with 131 members. The lowest-ID canonical is retained in each component and 81 redundant members are excluded.
- Pending candidates are non-blocking warnings: 283 involve excluded images, 78 retain both images within one split, and zero retain both images across splits.
- Final curated benchmark: 720 images and 1,065 objects; train 510, validation 101, test 109.
- Final class counts: craze 169, corrosion 178, surface_injure 264, thunderstrike 60, crack 131, hide_craze 263. All six classes occur in every split.
- `data/metadata/wtbd/curation_blockers.csv` contains only its header; there are no remaining Phase 2 blockers.

### Phase 2 evidence

- Machine-readable summary: `data/metadata/wtbd/audit_summary.json`.
- Full audit: `docs/phase2_dataset_audit.md`.
- Curation policy, human-review import, final counts, and completed gate: `docs/phase2_curation.md`.
- Curated manifest and summary: `data/metadata/wtbd/curation_manifest.csv` and `curation_summary.json`.
- Empty final blocker table: `data/metadata/wtbd/curation_blockers.csv`.
- Imported review provenance and component/cross-split cross-checks: `data/metadata/wtbd/human_review/`.
- Identity evidence sheets and index: `figures/phase2/identity_review/`.
- Per-class annotation review: `figures/phase2/annotation_examples/`.
- Candidate near-duplicate review: `figures/phase2/near_duplicates/`.
- Split, bbox, co-occurrence, checksum, duplicate, and upstream inventories: `data/metadata/wtbd/`.

### Phase 2 exit-gate record

- [x] Phase 0 and Phase 1 remain frozen and tests pass.
- [x] Official provenance, license, archive checksum, and raw fingerprint are recorded.
- [x] Every image decodes and image/XML file-ID pairing is complete.
- [x] All primary XML parses and all boxes pass geometric validation.
- [x] Official split syntax, coverage, disjointness, and class statistics are audited.
- [x] Exact and perceptual duplicate screening is complete.
- [x] Contact sheets, plots, CSV records, JSON summary, and audit documentation are generated.
- [x] Published/raw count discrepancies are preserved without forcing agreement.
- [x] A versioned manifest, schemas, decision inputs, diagnostics, and raw/curated statistics are generated.
- [x] All 262 identity decisions are imported and their reviewed reused/derived variants are excluded.
- [x] Cross-split exact-copy leakage is removed by a deterministic canonical/exclusion policy.
- [x] Raw fingerprint remains stable after curation generation.
- [x] The reviewed same-scene graph reproduces 50 supplied components and excludes 81 redundant members.
- [x] No confirmed same-scene component has more than one included image.
- [x] No pending cross-split candidate retains both images; the eight retained cross-split candidates are reviewed false positives.
- [x] Pending within-split and excluded-image candidates remain documented as non-blocking warnings.
- [x] All six classes remain represented in train, validation, and test.
- [x] Curated metadata regenerates deterministically and strict validation passes.
- [x] Phase 2 is complete.

No model was implemented or trained, no model weights were downloaded, no final classification crops were generated, and no crop margin or preprocessing choice was frozen.

## Phase 1 — Repository and Reproducibility Infrastructure

Phase 1 establishes engineering infrastructure using synthetic data only. It does not acquire datasets, implement scientific methods, train models, or produce scientific results.

### Frozen Phase 1 decisions

- Python 3.11 reference environment.
- `src/windblade` package layout.
- YAML source configuration and per-run resolved YAML snapshots.
- Deterministic 12-character SHA-256 configuration fingerprints.
- Timestamped, sanitized experiment identifiers.
- Global Python, NumPy, and optional PyTorch seed utility.
- Technical environment and Git provenance in every manifest.
- Deterministic machine-readable JSON result records.
- One isolated run directory containing resolved configuration, manifest, results, and log.
- Standard-library console and file logging.
- Failed-run evidence preservation with non-swallowed exceptions.
- Pytest infrastructure using temporary output directories.
- Immutable `data/raw/` policy.

### Phase 1 status

Complete. Validation used Python 3.11.15 and pytest 9.1.1:

- Full suite: **21 passed, 0 failed**.
- Two independent seed-42 CLI runs produced different timestamped experiment IDs, the identical config hash `cc5ceef2d6b6`, and identical deterministic synthetic payloads.
- A seed-43 CLI run produced config hash `6a8b48d3045f` and a different seeded synthetic payload.
- All three run directories contained exactly `resolved_config.yaml`, `manifest.json`, `results.json`, and `run.log`.
- All three inspected manifests reported `completed`, Python 3.11.15, CPU execution, PyTorch unavailable, Git commit `7ee35b99b58fc22a04c87e789dbe18762a72554c`, and a clean worktree.
- The only files under `data/` are policy documentation and tracked directory placeholders.

### Phase 1 exit-gate record

- [x] Phase 0 research documentation remains unchanged.
- [x] The `windblade` package imports under Python 3.11.
- [x] YAML configuration loading and validation work.
- [x] Configuration hashing is deterministic and order-independent.
- [x] Python and NumPy global seeding work; optional PyTorch handling is non-fatal.
- [x] Environment and Git provenance capture work without optional ML dependencies.
- [x] Timestamped, sanitized experiment IDs are generated.
- [x] Every experiment receives an isolated run directory.
- [x] Resolved configuration snapshots are preserved.
- [x] Manifest JSON, results JSON, and run logs are generated.
- [x] Failed-run status, failure details, configuration, and logs are preserved and tested; exceptions propagate.
- [x] Synthetic smoke execution succeeds on CPU in seconds.
- [x] Same-seed smoke payload determinism is independently validated.
- [x] A different seed changes both the resolved hash and seeded payload.
- [x] Full pytest suite passes: 21 passed, 0 failed.
- [x] Raw-data immutability and version-control policies are documented.
- [x] No WTBD or other real dataset was downloaded.
- [x] No real ML model was trained or implemented.
- [x] No scientific performance number or figure was produced.

## Phase 0 artifacts

- `docs/phase0_research_contract.md` — authoritative research scope, rules, safeguards, and non-claims.
- `docs/source_ledger.md` — supplied literature and dataset ledger only.
- `docs/experiment_matrix.md` — core comparison matrix and frozen-versus-later decision boundary.
- `PROJECT_STATE.md` — phase status and handoff record.

## Frozen decisions

- Working title: **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**.
- Primary question: comparison of handcrafted image-feature methods and transfer-learned CNNs for classifying visible surface-defect crops under limited labeled data and image degradation.
- Task: six-class classification of expert-annotated WTBD defect instances—not arbitrary full-image detection.
- Primary dataset: WTBD, with one crop per PASCAL VOC bounding box.
- Classes: `craze`, `corrosion`, `surface_injure`, `thunderstrike`, `crack`, and `hide_craze`.
- WTBD has no valid healthy/normal class; healthy-versus-damaged claims are prohibited.
- “Earlier detection” is prohibited because WTBD has no temporal onset labels.
- Leakage rule: every crop from the same source image remains in one partition; crops are never independently random-split.
- Split rule: use the supplied standard source-image split when available and valid.
- Core methods: HOG + SVM, LBP + SVM, ImageNet-pretrained ResNet-18, and ImageNet-pretrained MobileNetV3-Small.
- Primary metric: macro F1-score.
- Secondary, class-level, and efficiency measures are those enumerated in the research contract and experiment matrix.
- Training fractions: 25%, 50%, 75%, and 100% of the training partition only; validation and test sets stay fixed.
- Repetition: exactly three reproducible seeds, with individual results, mean, and standard deviation; no best-seed-only reporting.
- Corruptions: Gaussian blur, resolution degradation, brightness reduction, and JPEG compression, each with clean/control, mild, moderate, and severe levels.
- Robustness occurs after model selection on the fixed test set, without retraining or fine-tuning on corrupted test images.
- Test data do not influence training, hyperparameter selection, preprocessing, or model selection.
- Final reported numbers must come from machine-readable outputs.
- Core training excludes generated, synthetic, web-scraped, and secondary-dataset images.
- Unsupported priority, architecture-novelty, state-of-the-art, causal, safety, economic, and deployment claims are prohibited as detailed in the research contract.
- Blade30, DTU Drone Inspection Images, WTBs2025, lightweight YOLO detection, and Grad-CAM are outside the core Phase 0 scope.

## Phase 0 handoff questions and resolution

These were recorded at the end of Phase 0. Phase 1 infrastructure questions are now resolved; scientific and dataset-specific items remain intentionally deferred and must not be chosen by inspecting test or robustness results.

### Resolved in Phase 1

- [x] Supported Python runtime and minimal dependency versions.
- [x] Environment and lockfile strategy.
- [x] Repository layout beyond the Phase 0 documents.
- [x] Configuration schema, smoke command, artifact directory conventions, and machine-readable JSON output.
- [x] Logging, Git/environment provenance, deterministic controls, and validation tests.

### Dataset and preprocessing phase, before model experiments

- Verification procedure for the supplied WTBD source-image split and fallback only if it is unavailable or invalid.
- Exact fixed context-margin definition and rounding behavior.
- Crop boundary, malformed-annotation, and image/annotation integrity handling.
- Exact resize/pad dimensions, interpolation, color conversion, normalization, and method-fair preprocessing rules.
- Exact stratified training-subset construction and handling when perfect stratification is infeasible.
- The three numeric random-seed values; these must be frozen before Phase 3.
- Exact class weighting or balancing protocol, including the explicit choice to use none if applicable.

### Phase 3 preparation, before applicable results are viewed

- HOG and LBP feature-extraction parameters.
- SVM kernel, regularization, multiclass handling, scaling, and validation search protocol.
- CNN optimizer, learning-rate schedule, batch size, epoch limit, early-stopping rule, augmentation, and layer-freezing/fine-tuning protocol.
- Model-selection tie handling and checkpoint-selection details.
- Exact numerical corruption parameters and deterministic implementations for all four severity scales.
- Exact metric-library/version conventions, including undefined per-class metric handling.
- Inference-latency hardware/software conditions, batch size, warm-up, repetition, and summary procedure.
- Model/checkpoint size and trainable-parameter accounting conventions.

## Phase 0 exit-gate record

- [x] Research question is unambiguous and uses “classifying.”
- [x] Task is explicitly six-class defect classification.
- [x] WTBD is the primary dataset.
- [x] Absence of a healthy class is documented.
- [x] “Earlier detection” is explicitly excluded.
- [x] Four core methods are frozen.
- [x] Macro F1-score is the primary metric.
- [x] Four data-efficiency fractions are frozen.
- [x] Four corruption families are frozen.
- [x] Source-image leakage prevention is documented.
- [x] Test-set policy is documented.
- [x] Unsupported novelty claims are prohibited.
- [x] Every supplied source is recorded.
- [x] No model training or dataset download occurred.
- [x] No external literature research occurred.

## Phase boundary

Completed Phase 0 and Phase 1 decisions and documents remain frozen unless the user explicitly requests a documented revision. Phase 2 is complete and its reviewed manifest is the authoritative WTBD benchmark input for the next phase. Phase 3 has not started: do not create final crops, implement or train models, download pretrained weights, or freeze preprocessing without separate explicit authorization.
