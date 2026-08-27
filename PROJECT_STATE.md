# Project State

## Current phase

- **Phase:** Phase 1 — Repository and Reproducibility Infrastructure
- **Status:** IN PROGRESS
- **Previous phase:** Phase 0 — complete and frozen on 2026-08-27
- **Next phase:** Phase 2 — WTBD Dataset Acquisition and Audit

Phase 1 is explicitly authorized. Phase 0 research decisions remain frozen, and Phase 2 has not started.

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

Implementation and validation are in progress. Completion requires the full Phase 1 exit gate to pass.

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

## Intentionally unresolved implementation questions

These items are reserved for later phases. They are not omissions from the frozen research design and must not be chosen by inspecting test or robustness results.

### Phase 1 — repository and reproducibility infrastructure

- Supported runtime, package, and dependency versions.
- Environment and lockfile strategy.
- Repository layout beyond the Phase 0 documents.
- Configuration schema, command interface, artifact directory conventions, and machine-readable output format.
- Logging, provenance, hardware metadata, deterministic-execution controls, and validation/CI checks.

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

## Exit-gate record

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

Completed Phase 0 decisions and documents remain frozen unless the user explicitly requests a documented revision. Phase 1 is infrastructure-only. Do not begin Phase 2, download data, preprocess real images, train models, or conduct external research without explicit authorization.
