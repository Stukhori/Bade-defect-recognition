# Phase 6 — MobileNetV3-Small Transfer-Learning Baseline

## Objective and frozen upstream data

Phase 6 tests whether a lightweight CNN retains the frozen ResNet-18 benchmark performance. It uses only source-isolated `wtbd_crops_v1` (757/146/162 train/validation/test), fingerprint `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.

## Architecture and provenance

Official torchvision `mobilenet_v3_small` uses `MobileNet_V3_Small_Weights.IMAGENET1K_V1`; only its final `1024→1000` Linear layer becomes `1024→6`. All 1,524,006 parameters are trainable. PyTorch/torchvision are `2.13.0+cpu`/`0.28.0`; pretrained fingerprint is `982065f8b6dec87a6c0d70a9a0b132bc976615c1f2d6a4fa2d93bc060caaf1d1`.

## Matched protocol

Phase 5/6 parity is tested: identical pixels, splits, ImageNet normalization, no augmentation, batch sizes 32/64, FP32 full fine-tuning, deterministic algorithms, AdamW, early stopping, grid, and seeds. Train-only class weights are `1.025745273, 1.001322746, 0.681981981, 3.003968239, 1.356630802, 0.671099305`.

## Tuning and frozen winner

| LR | WD | Best epoch | Validation macro-F1 | Balanced accuracy |
|---:|---:|---:|---:|---:|
| 0.0001 | 0 | 17 | 0.873412 | 0.859572 |
| 0.0001 | 0.0001 | 17 | 0.873412 | 0.859572 |
| 0.0003 | 0 | 12 | 0.873147 | 0.855398 |
| 0.0003 | 0.0001 | 12 | 0.872412 | 0.852611 |

The deterministic tie-break froze LR `0.0001`, WD `0` before test. Grid fingerprint: `d61b35f9655a50f266fc52f2ba745c6904cf6e1bd367f0f8bfdcc4f8f77aa5b6`.

## Final results

| Seed | Best epoch | Val F1 | Test macro-F1 | Balanced acc. | Accuracy | Precision | Recall | Train seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 17 | 0.873412 | 0.901740 | 0.908258 | 0.888889 | 0.898527 | 0.908258 | 762.847 |
| 29 | 17 | 0.887441 | 0.894307 | 0.903105 | 0.876543 | 0.888752 | 0.903105 | 765.327 |
| 43 | 27 | 0.905371 | 0.889915 | 0.899931 | 0.882716 | 0.883268 | 0.899931 | 997.557 |
| Mean ± sample SD | — | — | 0.895321 ± 0.005977 | 0.903765 ± 0.004203 | 0.882716 ± 0.006173 | 0.890182 ± 0.007729 | 0.903765 ± 0.004203 | — |

Per-class F1 mean ± SD: craze `0.914645 ± 0.049798`; corrosion `0.791270 ± 0.027800`; surface_injure `0.849840 ± 0.022736`; thunderstrike `0.964912 ± 0.030387`; crack `0.944753 ± 0.035964`; hide_craze `0.906504 ± 0.031471`.

The largest mean confusion is corrosion→hide_craze (0.133), then surface_injure→corrosion (0.071) and craze→corrosion (0.062). Limited thunderstrike support constrains interpretation.

## Efficiency and frozen comparisons

Checkpoint size is 6,219,239 bytes. CPU batch-1 timing is 22.067 ms median, 33.990 ms p95, and 45.32 images/s. Relative to frozen ResNet: 86.37% fewer parameters, 86.12% smaller checkpoint, 79.23% lower recorded median latency, macro-F1 retention 100.0007%, and absolute mean difference +0.0000063. This is descriptive, not statistical equivalence.

| Method | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| HOG + SVM | 0.477988 | 0.457766 | 0.530864 |
| LBP + SVM | 0.592401 | 0.572746 | 0.611111 |
| ResNet-18 | 0.895314 ± 0.014118 | 0.902476 ± 0.017472 | 0.888889 ± 0.021383 |
| MobileNetV3-Small | 0.895321 ± 0.005977 | 0.903765 ± 0.004203 | 0.882716 ± 0.006173 |

## Reproducibility, limitations, and exit gate

The independent seed-17 rerun exactly matched best epoch, scientific history excluding timing, validation/test predictions and metrics, and checkpoint fingerprint. ResNet was not retrained.

Defect crops are already localized; this is not autonomous full-image detection. There is no healthy class or temporal basis for early-detection claims. No augmentation is used; only one curated dataset and validation split are evaluated. Performance and latency apply to recorded hardware. This matched MobileNet/ResNet comparison is not an exhaustive architecture benchmark.

All Phase 6 gates pass. Machine-readable histories, logits, predictions, matrices, fingerprints, summaries, comparisons, and timing are saved. No data-efficiency, robustness, or Phase 7 experiment started.
