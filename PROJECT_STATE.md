# Project State

## Current phase

- **Phase:** Phase 1 — Repository and Reproducibility Infrastructure
- **Status:** COMPLETE and frozen
- **Completion date:** 2026-08-27
- **Previous phase:** Phase 0 — complete and frozen on 2026-08-27
- **Next phase:** Phase 2 — WTBD Dataset Acquisition and Audit

Phase 1 passed its exit gate. Phase 0 research decisions remain frozen, and Phase 2 has not started.

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

Completed Phase 0 and Phase 1 decisions and documents remain frozen unless the user explicitly requests a documented revision. Do not begin Phase 2, download data, preprocess real images, train models, or conduct external research without explicit authorization.
