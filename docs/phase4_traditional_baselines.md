# Phase 4 — Traditional Computer-Vision Baselines

## 1. Objective

Phase 4 establishes the two frozen non-deep-learning baselines: HOG + RBF-SVM and multi-scale spatial LBP + RBF-SVM. It is the first phase reporting real model-performance results. All feature parameters and the eight-candidate SVM grid were predeclared; validation selected one configuration per family before the test split was materialized for evaluation.

## 2. Dataset

- Processed dataset: `wtbd_crops_v1`.
- Processed fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Samples: 1,065 deterministic 224 × 224 RGB PNG defect crops.
- Instance counts: 757 train, 146 validation, and 162 test.
- Source-image counts: 510 train, 101 validation, and 109 test.
- Frozen class order: craze, corrosion, surface_injure, thunderstrike, crack, hide_craze.

Phase 2 strict validation and the complete Phase 3 gate, including 1,093/1,093 byte-identical regeneration checks, passed immediately before Phase 4. Phase 4 reads only the Phase 3 processed images and manifest. It does not read raw WTBD pixels, change crop geometry, or use partial training fractions.

## 3. HOG representation

Each stored RGB crop is converted to grayscale with Pillow's deterministic luminance conversion. Scikit-image HOG uses 9 orientations, 16 × 16 pixels per cell, 2 × 2 cells per block, `L2-Hys` normalization, `transform_sqrt=true`, and a flattened vector. Fourteen cells per image axis and 13 × 13 block positions produce exactly `13 × 13 × 2 × 2 × 9 = 6,084` finite features.

- Feature configuration hash: `e0723cd80ec462644aec14e3827821d716d04ff375424b5de45ac5ddac4d5cf2`.
- Feature-matrix fingerprint: `a89e1102fd90cf8d2ecc8698b029bd90f05abbdf39cd3861b3ceefaceef3ecbf`.

## 4. LBP representation

The same RGB crop is converted to grayscale. Every 32 × 32 region in a fixed 7 × 7 grid receives two independently normalized uniform-LBP histograms: radius 1 with 8 points and 10 bins, and radius 2 with 16 points and 18 bins. Concatenating all regions produces exactly `49 × (10 + 18) = 1,372` finite features.

- Feature configuration hash: `e952103e7c0664952a0b8c568141bd26d177333c572f7ec134d1270c2592d122`.
- Feature-matrix fingerprint: `2826d502592c6b4066cab2dd64b556a281e79752659401d271986d9990f004e8`.

Feature extraction sees crop pixels only. Labels, bounding boxes, source names, crop scale, occupancy, and split metadata are not extractor inputs. The cache key covers the processed fingerprint, feature configuration, and relevant library versions; cached arrays remain ignored by Git and are validated against instance IDs and fingerprints before reuse.

## 5. SVM

Both families use the same scikit-learn pipeline: `StandardScaler` followed by an RBF `SVC` with `class_weight="balanced"` and probability estimation disabled. Scaling and SVM fitting receive the 757 training instances only. Validation and test data receive transform/predict only. The grid is exactly `C ∈ {0.1, 1, 10, 100}` crossed with `gamma ∈ {scale, auto}`.

## 6. Model selection

The primary selection metric is validation macro-F1. Ties within `1e-12` are resolved by higher balanced accuracy, higher macro recall, lower C, then `scale` before `auto`. The validation search function accepts train and validation inputs only; no test data can be passed to it. Both complete validation grids and all validation predictions were saved before either selected configuration was frozen.

The validation-grid fingerprint is `e897c6b58911baaf27d626d38a598404d538af7bf99f72aa7a1a7a1043ad8bb8`.

## 7. Validation results

| Method | C | gamma | macro-F1 | balanced accuracy | macro precision | macro recall | selected |
|---|---:|---|---:|---:|---:|---:|---|
| HOG | 0.1 | scale | 0.122089 | 0.190821 | 0.122222 | 0.190821 | no |
| HOG | 0.1 | auto | 0.122089 | 0.190821 | 0.122222 | 0.190821 | no |
| HOG | 1 | scale | 0.443025 | 0.459065 | 0.461961 | 0.459065 | no |
| HOG | 1 | auto | 0.443025 | 0.459065 | 0.461961 | 0.459065 | no |
| HOG | 10 | scale | 0.448100 | 0.457539 | 0.460539 | 0.457539 | **yes** |
| HOG | 10 | auto | 0.448100 | 0.457539 | 0.460539 | 0.457539 | no |
| HOG | 100 | scale | 0.448100 | 0.457539 | 0.460539 | 0.457539 | no |
| HOG | 100 | auto | 0.448100 | 0.457539 | 0.460539 | 0.457539 | no |
| LBP | 0.1 | scale | 0.213212 | 0.234624 | 0.228147 | 0.234624 | no |
| LBP | 0.1 | auto | 0.213212 | 0.234624 | 0.228147 | 0.234624 | no |
| LBP | 1 | scale | 0.421536 | 0.429038 | 0.453590 | 0.429038 | no |
| LBP | 1 | auto | 0.421536 | 0.429038 | 0.453590 | 0.429038 | no |
| LBP | 10 | scale | 0.495356 | 0.514518 | 0.499884 | 0.514518 | **yes** |
| LBP | 10 | auto | 0.495356 | 0.514518 | 0.499884 | 0.514518 | no |
| LBP | 100 | scale | 0.495356 | 0.514518 | 0.499884 | 0.514518 | no |
| LBP | 100 | auto | 0.495356 | 0.514518 | 0.499884 | 0.514518 | no |

The tied C=10/C=100 results invoke the predeclared lower-C rule, and the tied gamma values invoke the predeclared `scale` preference. No rejected candidate was evaluated on test.

## 8. Frozen configurations

- HOG + SVM: `C=10`, `gamma=scale`, validation macro-F1 `0.4480996944232238`.
- LBP + SVM: `C=10`, `gamma=scale`, validation macro-F1 `0.495355605764074`.

The immutable YAML records were written at the test gate and reference clean implementation commit `f5ae5ae8dd7b7dd8f315fdbc944138f32cbfb3b8`. Final models were refit from scratch on train only; validation was not merged into training.

## 9. Test results

| Method | accuracy | balanced accuracy | macro precision | macro recall | macro F1 |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.530864 | 0.457766 | 0.593041 | 0.457766 | 0.477988 |
| LBP + SVM | 0.611111 | 0.572746 | 0.670386 | 0.572746 | 0.592401 |

These are the frozen first test evaluations for the selected traditional baselines. They are benchmark results, not claims of universal industrial performance.

## 10. Per-class behavior

| Class | HOG test F1 | LBP test F1 | Test support |
|---|---:|---:|---:|
| craze | 0.596491 | 0.677966 | 27 |
| corrosion | 0.452830 | 0.526316 | 30 |
| surface_injure | 0.500000 | 0.470588 | 33 |
| thunderstrike | 0.363636 | 0.500000 | 9 |
| crack | 0.347826 | 0.666667 | 14 |
| hide_craze | 0.607143 | 0.712871 | 49 |

Descriptively, HOG recalls 2/9 thunderstrike and 4/14 crack samples; its thunderstrike errors most often go to surface injury. LBP recalls 3/9 thunderstrike and 9/14 crack samples. Its larger off-diagonal counts include hide craze predicted for corrosion (6) and surface injury (6), and surface injury predicted for hide craze (8). These observations do not trigger feature or model changes.

## 11. Computational efficiency

Timing uses the same Windows CPU environment (AMD64 Family 23 Model 24, 8 logical CPUs), one warm-up, three repeated test feature passes, and seven repeated prediction passes. Latency values are medians and are descriptive for this computer only.

| Measure | HOG + SVM | LBP + SVM |
|---|---:|---:|
| feature dimensions | 6,084 | 1,372 |
| first full-dataset feature extraction | 17.708 s | 45.016 s |
| test feature latency/image | 10.564 ms | 31.609 ms |
| final train-only SVM fit | 2.924 s | 0.647 s |
| prediction latency/image | 3.549 ms | 0.861 ms |
| combined latency/image | 14.112 ms | 32.470 ms |
| serialized model size | 35,551,712 bytes | 7,687,831 bytes |

HOG model SHA-256 is `30c46fd50b8da5779434dad856f7c0b270ab101a14e27cecc43be4b0d275bb2d`; LBP model SHA-256 is `742aa499caab3376173cd434caaf645a09a9654fd2f0d53a5fe7e005fb8dca1e`. Model binaries and feature caches are regenerable and excluded from Git.

## 12. Reproducibility

- Result ID: `phase4_traditional_v1`.
- Implementation Git commit: `f5ae5ae8dd7b7dd8f315fdbc944138f32cbfb3b8`.
- Python 3.11.15; NumPy 2.4.6; Pillow 12.3.0; scikit-image 0.26.0; scikit-learn 1.9.0; joblib 1.5.3.
- Machine-readable versioned results: `experiments/summaries/phase4_traditional_v1/`.
- Regenerable full results and models: `experiments/results/phase4_traditional_v1/`.
- Feature cache: `experiments/cache/traditional_features/`.

The unchanged canonical rerun reproduced 10/10 scientific prediction, metric, selected-config, and confusion files byte-for-byte. Validation-grid and feature fingerprints, model SHA-256 values, selected hyperparameters, predictions, and metrics were identical. Floating timing measurements and run timestamps are intentionally excluded from byte-level equality.

## 13. Limitations

- These models classify externally supplied defect crops; they are not autonomous full-image detectors.
- Handcrafted feature parameters are fixed rather than exhaustively searched.
- One frozen validation split is used instead of extensive cross-validation to preserve source-image isolation.
- The curated benchmark is modest and class-imbalanced; thunderstrike has only nine test instances.
- Test results apply to the curated WTBD benchmark and do not establish universal wind-farm or industrial performance.
- Scikit-learn 1.9 emits a deprecation notice when `SVC(probability=False)` is specified explicitly; probability estimation was nevertheless disabled, and no probabilities were computed.

## 14. Phase 4 exit status

**COMPLETE.** Both fixed feature matrices validate, exactly 16 validation configurations are recorded, the two winners were frozen before test, exactly those two models were evaluated on test, all required metrics/predictions/confusions/efficiency records exist, and deterministic repetition passes. No CNN, pretrained weight, partial-fraction run, augmentation, or corruption experiment was started.
