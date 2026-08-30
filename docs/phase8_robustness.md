# Phase 8 — Controlled Image-Degradation Robustness

## 1. Objective

Phase 8 measures zero-shot robustness of the four frozen full-data classifiers when the fixed 162-instance WTBD test partition is subjected to predeclared, simulated image-quality degradation. The independent variable is test-image quality. No model training, fine-tuning, hyperparameter selection, validation-based modification, SVM refitting, or scaler refitting occurs.

## 2. Frozen models

- HOG + the clean-trained Phase 4 `StandardScaler` and SVM.
- LBP + the clean-trained Phase 4 `StandardScaler` and SVM.
- ResNet-18 Phase 5 checkpoints for seeds 17, 29, and 43.
- MobileNetV3-Small Phase 6 checkpoints for seeds 17, 29, and 43.

Only the full-data baselines are evaluated. Phase 7 training fractions are not crossed with corruption severity.

## 3. Why no retraining occurs

The question concerns the behavior of already-frozen clean-trained models when image quality worsens. Corruption-aware training would answer a different question and is outside Phase 8. Every inference uses evaluation mode and no gradients; the traditional pipelines call only their frozen feature extraction, `StandardScaler.transform()`, and `SVM.predict()` operations.

## 4. Test-only degradation protocol

Every degraded sample is generated independently from its immutable 224 × 224 RGB Phase 3 PNG. No degradation consumes another degraded output, corruption families are never combined, and clean is stored once as a reference rather than duplicated. The twelve degraded conditions produce exactly 1,944 canonical PNGs (`162 × 12`).

The tracked [corruption manifest](../data/processed/wtbd_robustness_v1/manifest.csv), [checksum manifest](../data/processed/wtbd_robustness_v1/corruption_checksum_manifest.csv), [condition definition](../data/processed/wtbd_robustness_v1/conditions.json), and [dataset summary](../data/processed/wtbd_robustness_v1/summary.json) define `wtbd_robustness_v1`. The PNG payload remains Git-ignored.

## 5. Common-pixel fairness

Each degraded RGB image is generated and hashed once. HOG, LBP, ResNet-18, and MobileNetV3-Small all read that same losslessly stored PNG. HOG/LBP then apply their frozen grayscale conversions; the CNNs apply their frozen tensor conversion and ImageNet normalization. There are no method-specific corruption implementations.

## 6. Gaussian blur definition

Pillow `ImageFilter.GaussianBlur` is applied at radii 0.75 (mild), 1.5 (moderate), and 3.0 (severe). Clean radius 0 references the original image.

## 7. Resolution definition

The clean 224 × 224 image is bilinearly downsampled to 168 × 168, 112 × 112, or 56 × 56 and then bilinearly upsampled to 224 × 224 for mild, moderate, and severe conditions, respectively.

## 8. Brightness definition

RGB values are multiplied by 0.75, 0.50, or 0.25, rounded deterministically with NumPy `rint`, clipped to `[0, 255]`, and cast to uint8. Contrast, gamma, auto-exposure, and histogram normalization are not applied.

## 9. JPEG definition

The clean RGB image undergoes one in-memory Pillow JPEG encode/decode round trip at quality 75, 50, or 25. Encoder options are `subsampling=2`, `optimize=false`, and `progressive=false`. Decoded pixels are stored losslessly as PNG. The recorded environment is Pillow 12.3.0 with JPEG library 8.0.

## 10. Clean reproduction gate

**PASS.** Before any degraded inference, Phase 8 exactly reproduced the frozen HOG and LBP predictions and metrics and all ResNet/MobileNet predictions, logits, and metrics for seeds 17/29/43. The eight machine-readable checks are in [`clean_reproduction/status.json`](../experiments/summaries/phase8_robustness_v1/clean_reproduction/status.json).

## 11. Metrics

Macro-F1 remains primary. Secondary outputs include balanced accuracy, accuracy, macro precision, macro recall, per-class precision/recall/F1/support, and confusion matrices. CNN summaries use the mean and sample standard deviation over seeds 17/29/43 (`ddof=1`). HOG/LBP are single deterministic results and have no standard deviation.

## 12. Absolute loss

For each corrupted condition, `absolute_drop = corrupted macro-F1 − clean macro-F1`. Negative values denote loss. Values are stored in [`robustness_summary.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/robustness_summary.csv).

## 13. Retention

`retention = corrupted macro-F1 / clean macro-F1`; `relative_loss = 1 − retention`. These are descriptive ratios to the frozen clean baseline, not universal robustness scores.

## 14. Prediction-flip analysis

For every degraded inference, Phase 8 records whether the prediction changed relative to the same frozen model/seed on clean pixels. It separately counts harmful flips (clean correct → corrupted incorrect), beneficial flips (clean incorrect → corrupted correct), and all four correct/incorrect transitions. Complete tables are [`prediction_flip_rates.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/prediction_flip_rates.csv), [`error_transitions.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/error_transitions.csv), and [`instance_robustness.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/instance_robustness.csv).

## 15. Full results

Values below are test macro-F1. CNN entries are mean ± sample SD; traditional entries have SD N/A.

| Family | Method | Clean | Mild | Moderate | Severe |
|---|---|---:|---:|---:|---:|
| Blur | HOG + SVM | 0.477988 | 0.454351 | 0.450664 | 0.351435 |
| Blur | LBP + SVM | 0.592401 | 0.555030 | 0.371985 | 0.203079 |
| Blur | ResNet-18 | 0.895314 ± 0.014118 | 0.891084 ± 0.020693 | 0.822025 ± 0.018044 | 0.633788 ± 0.032868 |
| Blur | MobileNetV3-Small | 0.895321 ± 0.005977 | 0.878574 ± 0.022958 | 0.843774 ± 0.014293 | 0.654345 ± 0.041871 |
| Resolution | HOG + SVM | 0.477988 | 0.467346 | 0.465164 | 0.395885 |
| Resolution | LBP + SVM | 0.592401 | 0.530885 | 0.364424 | 0.175100 |
| Resolution | ResNet-18 | 0.895314 ± 0.014118 | 0.876557 ± 0.021513 | 0.853611 ± 0.027808 | 0.723203 ± 0.025957 |
| Resolution | MobileNetV3-Small | 0.895321 ± 0.005977 | 0.879019 ± 0.019438 | 0.861577 ± 0.013832 | 0.733372 ± 0.016645 |
| Brightness | HOG + SVM | 0.477988 | 0.425921 | 0.399174 | 0.318186 |
| Brightness | LBP + SVM | 0.592401 | 0.561298 | 0.444338 | 0.262047 |
| Brightness | ResNet-18 | 0.895314 ± 0.014118 | 0.882760 ± 0.018941 | 0.809782 ± 0.000065 | 0.526303 ± 0.072731 |
| Brightness | MobileNetV3-Small | 0.895321 ± 0.005977 | 0.878433 ± 0.016557 | 0.858144 ± 0.005571 | 0.659088 ± 0.048045 |
| JPEG | HOG + SVM | 0.477988 | 0.439939 | 0.454484 | 0.298325 |
| JPEG | LBP + SVM | 0.592401 | 0.512283 | 0.384648 | 0.218386 |
| JPEG | ResNet-18 | 0.895314 ± 0.014118 | 0.885217 ± 0.013598 | 0.863319 ± 0.003985 | 0.822440 ± 0.020611 |
| JPEG | MobileNetV3-Small | 0.895321 ± 0.005977 | 0.872486 ± 0.006569 | 0.856341 ± 0.011473 | 0.803678 ± 0.030335 |

The complete primary table also includes secondary metrics, absolute drop, retention, relative loss, and prediction flips.

## 16. Per-class results

All six classes, including thunderstrike, are retained in [`per_class_robustness.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/per_class_robustness.csv). Severe-condition CNN mean F1 values are:

| Model | Class | Blur | Resolution | Brightness | JPEG |
|---|---|---:|---:|---:|---:|
| ResNet-18 | craze | 0.593 | 0.643 | 0.527 | 0.802 |
| ResNet-18 | corrosion | 0.408 | 0.587 | 0.554 | 0.812 |
| ResNet-18 | surface_injure | 0.664 | 0.722 | 0.639 | 0.736 |
| ResNet-18 | thunderstrike | 0.478 | 0.762 | 0.376 | 0.886 |
| ResNet-18 | crack | 0.876 | 0.827 | 0.376 | 0.866 |
| ResNet-18 | hide_craze | 0.783 | 0.799 | 0.686 | 0.832 |
| MobileNetV3-Small | craze | 0.478 | 0.684 | 0.604 | 0.848 |
| MobileNetV3-Small | corrosion | 0.609 | 0.642 | 0.680 | 0.722 |
| MobileNetV3-Small | surface_injure | 0.690 | 0.683 | 0.674 | 0.779 |
| MobileNetV3-Small | thunderstrike | 0.571 | 0.662 | 0.493 | 0.791 |
| MobileNetV3-Small | crack | 0.791 | 0.864 | 0.714 | 0.864 |
| MobileNetV3-Small | hide_craze | 0.786 | 0.865 | 0.789 | 0.818 |

The rare thunderstrike class has small support, so class-specific patterns are descriptive and should not be overgeneralized. Detailed qualitative error interpretation is reserved for Phase 9.

## 17. Severe-condition comparison

| Method | Blur retention | Resolution retention | Brightness retention | JPEG retention |
|---|---:|---:|---:|---:|
| HOG + SVM | 73.52% | 82.82% | 66.57% | 62.41% |
| LBP + SVM | 34.28% | 29.56% | 44.23% | 36.86% |
| ResNet-18 | 70.83% | 80.82% | 58.72% | 91.90% |
| MobileNetV3-Small | 73.08% | 81.91% | 73.60% | 89.76% |

Severe prediction-flip rates are HOG 41.36/29.01/29.01/33.33%, LBP 57.41/61.73/51.23/50.62%, ResNet 33.13/24.49/40.12/15.23%, and MobileNet 27.57/22.22/31.48/14.61% for blur/resolution/brightness/JPEG. CNN counts are three-seed means; exact per-seed counts remain machine-readable.

| Method | Severe blur H/B | Severe resolution H/B | Severe brightness H/B | Severe JPEG H/B |
|---|---:|---:|---:|---:|
| HOG + SVM | 24 / 18 | 16 / 16 | 22 / 12 | 28 / 10 |
| LBP + SVM | 58 / 10 | 64 / 13 | 44 / 10 | 48 / 12 |
| ResNet-18 | 42.33 / 8.00 | 31.00 / 5.67 | 53.33 / 7.67 | 17.33 / 4.67 |
| MobileNetV3-Small | 36.00 / 5.00 | 26.67 / 5.67 | 37.67 / 8.67 | 17.00 / 3.67 |

`H/B` denotes harmful/beneficial flip counts. CNN entries are means over the three seeds, not pooled integer counts.

## 18. Family and overall summaries

| Method | Mean degraded-condition macro-F1 | Mean degraded-condition retention |
|---|---:|---:|
| HOG + SVM | 0.410073 | 85.79% |
| LBP + SVM | 0.381959 | 64.48% |
| ResNet-18 | 0.799174 | 89.28% |
| MobileNetV3-Small | 0.814903 | 91.02% |

The twelve-condition mean is specific to these predeclared synthetic transformations. Family-specific means are in [`family_summary.csv`](../experiments/summaries/phase8_robustness_v1/aggregate/family_summary.csv).

Descriptively, both CNNs retained mild degradation well. Severe brightness produced the lowest CNN macro-F1, while severe JPEG preserved the most clean macro-F1 for both CNNs. MobileNet retained more than ResNet under severe blur, resolution loss, and brightness reduction; ResNet retained more under severe JPEG. HOG retained more of its clean score than LBP across these conditions despite LBP's higher clean baseline. These observations do not establish statistical superiority, causal mechanisms, or universal robustness.

## 19. Full reproducibility

**PASS.** The canonical command executed two complete passes. All 1,944 regenerated PNG hashes, the dataset fingerprint, the scientific file set, predictions, CNN logits, metrics, aggregate tables, and flip analyses reproduced exactly. Timing values were excluded. Evidence is in [`reproducibility.json`](../experiments/summaries/phase8_robustness_v1/reproducibility.json).

## 20. Limitations

- Corruptions are controlled synthetic proxies and do not directly map to drone speed, flight altitude, lux, camera settings, or real weather measurements.
- Corruption types are evaluated individually, not jointly.
- Models were trained only on clean images.
- Only the fixed 162-instance WTBD test set is evaluated; there is no external dataset.
- Classification uses already-localized visible surface-defect crops.
- WTBD has no valid healthy class, so healthy-vs-damaged claims are prohibited.
- The study does not support early-detection inference.
- No statistical-superiority, causal, universal-robustness, structural-safety, repair, or deployment claim is made.

## 21. Exit gate

**PASS.** Phases 0–7 remain frozen; the Phase 3 fingerprint is unchanged; all frozen artifacts and clean predictions reproduce; exactly twelve independently generated degraded conditions and 1,944 common-pixel images exist; all metrics, per-class outputs, logits, flips, transitions, matrices, and figures exist; the full two-pass reproduction passes; the independent result validator passes; and the expanded suite reports 149 passed, 0 failed, with 11 unchanged scikit-learn deprecation warnings. Phase 9 has not started.
