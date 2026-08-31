# Phase 11A — Full-image detection feasibility and annotation audit

## Status and scientific separation

`PHASE 11A COMPLETE — DETECTION PROTOCOL FROZEN; PHASE 11B COMPUTE-BLOCKED`

Phase 11 is an optional result line separate from the frozen crop-classification study. Phase 11A audits the authoritative full-image annotations and freezes a detection-specific manifest and source-image split. At its freeze point it changed no Streamlit application file, and it changes no Phase 3–10 dataset, split, crop, label, checkpoint, prediction, logit, metric, table, figure, bootstrap index, review judgment, or conclusion. Application v2 was added afterward as validated non-scientific productization; it reads this audit but does not modify it or integrate a detector.

Phase 11B training was not started. The current environment has CPU-only PyTorch and no CUDA device, so the declared three-seed compact-detector plan cannot be completed safely here. No detector dependency was installed, no pretrained weight was downloaded, no checkpoint or prediction was created, and the test set was not evaluated. Phase 12 was not started.

## Authoritative source and Phase 9A transition

The source is WTBD v1 from Springer Nature Figshare (`10.6084/m9.figshare.30210175.v1`), license CC BY 4.0. The locally retained archive is `78,958,553` bytes with SHA-256 `466452f2a0cfc9ef6ba63ea2a3bbc7ea4262057dd07e4fc9e00eedf5bba305b4`. Original images are under `data/raw/wtbd/WT blade defect dataset/JPEGImages`; the authoritative primary annotations are PASCAL VOC XML under `Annotations`.

Coordinates are one-based inclusive integer pixels. Valid bounds are `1 <= xmin <= xmax <= image width` and the corresponding rule for height. YOLO conversion uses continuous edges `[xmin-1, xmax]` and `[ymin-1, ymax]`; normalized width and height therefore retain the inclusive `+1` extent. No box was reconstructed from a screenshot, Phase 9 overlay, or Grad-CAM output.

The Phase 9A fingerprint change from `a5938ec22a0f496c6fd9ce3acd7d999cad56e8c13a85d9be4b4d59860aecd74b` to `14e500fd94fa871bbe1e6bee6494d3158fe003c3799e0ccfbcab8e549adf80fa` is documented and validated as a caption-only correction to 177 Pass B true-class captions across 52 cases. Predictions, logits, checkpoints, selected cases, target indices, true labels, Grad-CAM arrays and images, and quantitative conclusions were unchanged.

## Raw and curated audit

The official release contains 1,065 readable source images and 1,584 valid primary boxes. The frozen `wtbd-curation-v1` layer retains 720 images and 1,065 boxes after the earlier human-reviewed identity, exact-duplicate, and same-scene exclusions. It supplies the detection dataset; Phase 11 makes no new label correction.

| Split | Images | Boxes |
|---|---:|---:|
| Train | 510 | 757 |
| Validation | 101 | 146 |
| Test | 109 | 162 |
| Total | 720 | 1,065 |

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| craze | 123 | 19 | 27 | 169 |
| corrosion | 126 | 22 | 30 | 178 |
| surface_injure | 185 | 46 | 33 | 264 |
| thunderstrike | 42 | 9 | 9 | 60 |
| crack | 93 | 24 | 14 | 131 |
| hide_craze | 188 | 26 | 49 | 263 |

There are 467 single-box images and 253 multi-box images; the maximum is seven boxes, and 64 images contain multiple defect classes. Every retained image has at least one annotation. There are no background-only or healthy images.

All 1,065 retained boxes decode against their source images, have finite coordinates, positive inclusive width/height, declared class IDs, and in-bounds geometry. There are zero invalid boxes, parse failures, duplicate annotations, orphan retained images, orphan retained annotations, or retained exact-image duplicate groups. Twenty-four within-image box pairs overlap at nonzero IoU; none reaches IoU 0.25. These boxes are preserved as supplied.

Diagnostic—not automatically erroneous—geometry counts are:

- 162 boxes occupy less than 1% of their image;
- 9 boxes occupy more than 50%;
- 20 boxes have aspect ratio below 0.10;
- 13 boxes have aspect ratio above 10;
- median box size is 220 × 249 pixels, median area fraction is 0.05298, and median aspect ratio is 0.9335;
- 719 images are 1024 × 1024 and one is 788 × 788.

The raw release contains 262 embedded-filename/XML-stem identity mismatches. The existing human-reviewed curation excludes those reuse/identity failures and other redundant items, for 345 excluded images total. It also removes both exact redundant images and 81 redundant confirmed same-scene variants. Seventy-eight unresolved perceptual-hash candidates remain only within the same curated split; zero unresolved, exact, or confirmed same-scene group crosses train/validation/test.

Phase 9 marked a possible crop/background concern as yes or uncertain for ten unique retained source images represented in its selected review cases. Those sources are included in the deterministic QC packet without reinterpretation.

## Feasibility decisions

| Question | Decision | Limit |
|---|---|---|
| Class-agnostic localization | Supported with explicit limitations | Positive-only, fixed dataset; no healthy controls or operational-domain evidence. |
| Six-class detection | Supported with explicit limitations | All classes occur in every split, but support is imbalanced; thunderstrike has only 60 boxes overall. |
| Healthy-blade false-positive evaluation | Unsupported | No source image has zero annotated defects. |
| Arbitrary full-image inspection | Unsupported | The release does not sample arbitrary healthy/operational scenes. |
| Existing-app integration | Unsupported | No detector result or background false-positive evidence exists. |

The primary future task is class-agnostic localization with `mAP@0.50:0.95` as primary metric. Six-class detection is secondary. This prioritization is fixed before training. It does not imply that a detector is currently available.

## Frozen split and test firewall

The detection split preserves the Phase 2 curated source-image split. All boxes from an image remain together. Exact duplicates and confirmed same-scene redundant variants were removed before the split was adopted; retained unresolved candidates do not cross partitions. Dataset fingerprint: `ad4ab59c3e3c85c6cf0b85b148177bd6b79d24f372f49bdff0043609e6fefc97`. Split fingerprint: `264f8460f203074374c2c098c8fd5d2e55fb7ee1f281a8d505e2dfb0de9a2bc3`.

Future Phase 11B must freeze one compact/nano one-stage YOLO-family variant, its exact package and license, pretrained-weight source and SHA-256, architecture, resolution, optimizer, schedule, batch size, epochs, early stopping, augmentations, NMS, and deterministic settings in a new pre-test apparatus commit. Seeds are fixed at 17, 29, and 43. Checkpoint selection must use validation `mAP@0.50:0.95`; the confidence threshold must be selected on validation by maximum class-agnostic F1 with a lower-threshold tie break. Test evaluation remains forbidden until all three seeds, checkpoints, threshold, and NMS decisions are locked.

Because Phase 11B did not start, detector version, architecture implementation, pretrained-weight source/license/hash, training logs, checkpoints, predictions, threshold value, seed results, AP values, per-class AP, false positives, false negatives, and error decomposition are all `N/A`, not zero.

## Compute gate

- CPU: AMD64 Family 23 Model 24, 8 logical processors.
- RAM: 13.95 GiB.
- Disk at gate: 475.73 GiB total, 344.39 GiB free.
- Python: 3.11.15; PyTorch: 2.13.0+cpu.
- CUDA devices: 0; VRAM: 0 GiB.
- Planned runs: three scientific seeds; estimated local storage allowance: 5 GiB; declared minimum GPU memory: 8 GiB.

Multi-hour CPU training is prohibited. On a suitable CUDA machine, the first action is not test evaluation or training: create and validate a new pre-test apparatus commit that pins the exact official detector package, compact/nano variant, license, official weight, weight SHA-256, and resolved training parameters. Only then may the recorded handoff command be enabled:

```powershell
python scripts/run_detection.py --config configs/detection.yaml --train
```

The current command intentionally refuses with the compute/dependency block.

## QC and machine-readable outputs

The deterministic QC packet contains 26 annotated full-image examples selected by declared rules: per-class coverage, smallest/largest boxes, extreme aspect ratios, multi-box images, overlapping boxes, and Phase 9 crop/background concerns. It draws only the authoritative retained boxes and proposes no edits. Open `experiments/summaries/phase11_detection_audit_v1/qc_packet/index.html`.

Machine-readable artifacts include image, annotation/conversion, split, class-support, geometry, finding, duplicate, and QC-selection tables; class mappings; feasibility and compute decisions; frozen protocol; Phase 10 and application immutability inventories; manifest; reproducibility record; and validation record.

```powershell
python scripts/run_detection.py --config configs/detection.yaml --apparatus-check
python scripts/run_detection.py --config configs/detection.yaml
python scripts/run_detection.py --config configs/detection.yaml --validate-only
python -m pytest tests/test_phase11_detection.py
python -m pytest
```

Two isolated complete Phase 11A generations match exactly for every scientific file, normalized coordinate, manifest table, QC selection, and QC image. Configuration fingerprint: `9f4a20ba4404c9a6072277a504c466a0756143b908e79c7168d2ccf91ff32057`. Scientific-output fingerprint: `3f46cbdc6c7a2e3cf6093ff177dd1948d113fa4c36fa9eb907d7c8621e800461`.

The Phase 3–10 validation chain, Phase 11A validator, review-interface validator, and classifier-app validator all pass. The focused Phase 11A suite passes 21 tests; the complete repository suite passes 256 tests with 11 unchanged scikit-learn `SVC(probability=True)` future warnings and no new warning.

## Limitations and application readiness

The data are defect-positive, mostly 1024 × 1024 images from one dataset release. They cannot measure false positives on healthy blades, represent arbitrary drone flights, establish operational domain coverage, validate rare-class reliability, estimate structural safety, severity, progression, or remaining life, or support fleet-level uncertainty. Geometry diagnostics and the single-reviewer Phase 9 concerns are not objective annotation errors.

Automatic-localization readiness is **not eligible for integration**. Application v2 respects that decision: all three active analysis modes still require a user-supplied crop or rectangle, and the detection interface explicitly reports unavailable. A future checkpoint alone would be insufficient: integration would require reviewed Phase 11B test precision/recall, a validation-selected operating point, background false-positive evidence, license/dependency review, checkpoint availability, and a separate application decision.

Phase 10 remains frozen and unchanged. Phase 12 has not started.

> Phase 11A is complete and the detection protocol is frozen. Phase 11B training is blocked by the documented compute or dependency requirement. Phase 10 remains frozen and Phase 12 has not started.
