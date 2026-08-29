# Phase 7 — Data-Efficiency / Limited-Labeled-Data Experiment

## Research objective and independent variable

Phase 7 asks how the four frozen classification methods behave when 25%, 50%, 75%, or 100% of the labeled training source images are available. The only manipulated variable is labeled training-data availability. Model hyperparameters, image representation, validation data, test data, and evaluation rules remain fixed.

No method was retuned at a reduced fraction. Retuning would confound training-data availability with optimization or representation changes. CNN validation was used only for the already-frozen best-epoch early-stopping rule.

## Frozen nested subsets

The Phase 3 source-image subsets were used without regeneration or replacement. All crops from a source image move together; no source image is partially selected. Within each seed family, 25% ⊂ 50% ⊂ 75% ⊂ 100%. The validation partition remains 146 instances and the test partition remains 162 instances for every run.

| Fraction | Source images | Defect crops | craze | corrosion | surface_injure | thunderstrike | crack | hide_craze |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 128 | 252 | 39 | 40 | 66 | 12 | 28 | 67 |
| 50% | 255 | 440 | 70 | 72 | 111 | 22 | 51 | 114 |
| 75% | 383 | 608 | 98 | 100 | 151 | 32 | 73 | 154 |
| 100% | 510 | 757 | 123 | 126 | 185 | 42 | 93 | 188 |

The three scientific seed families are 17, 29, and 43. For CNNs, each subset seed is paired with the same model seed, so replicate variability combines source-subset composition and neural-network stochasticity. HOG/LBP reduced-data variability reflects subset composition only.

## Frozen method configurations

- HOG + SVM: frozen Phase 4 6,084-dimensional HOG representation, StandardScaler fit on the active training subset, and RBF SVM with `C=10`, `gamma=scale`, and active-subset `class_weight=balanced`.
- LBP + SVM: frozen Phase 4 1,372-dimensional multiscale spatial LBP representation, active-subset StandardScaler, and the same frozen RBF-SVM selection.
- ResNet-18: official `ResNet18_Weights.IMAGENET1K_V1`, fresh six-class head, full fine-tuning, AdamW, learning rate 0.0003, weight decay 0, batch sizes 32/64, maximum 30 epochs, patience 6, `min_delta=0.0001`, no augmentation, ImageNet normalization, FP32, and deterministic execution.
- MobileNetV3-Small: official `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, fresh six-class head, full fine-tuning, AdamW, learning rate 0.0001, and otherwise the same frozen training controls.

CNN loss weights were calculated separately from each active training subset as `N_subset / (6 × N_subset,c)`. Withheld training labels and validation/test labels were excluded. Every fraction began independently from official pretrained weights; no cross-fraction warm start occurred. Within each architecture and seed, the initial six-class head fingerprint matched across 25%, 50%, and 75%.

## Run matrix and full-data anchors

The primary reduced-data matrix contains 36 fits: nine per method, comprising three reduced fractions × three paired seeds. This includes 18 CNN trainings. Each completed run received exactly one primary test evaluation after validation-based best-epoch restoration. No seed was removed.

The 100% endpoints were loaded from the validated Phase 4–6 artifacts and were not retrained. HOG and LBP therefore have one deterministic 100% endpoint with no standard deviation. ResNet and MobileNet use their genuine three-seed full-data results. Detailed inference benchmarking was not repeated because model architecture is unchanged; Phase 5/6 latency records remain the architecture-level references.

## Primary learning curves

Values are test mean ± sample SD across the three predeclared reduced-data replicates. HOG/LBP 100% SD is not applicable.

| Method | 25% macro-F1 | 50% macro-F1 | 75% macro-F1 | 100% macro-F1 |
|---|---:|---:|---:|---:|
| HOG + SVM | 0.334961 ± 0.004791 | 0.410712 ± 0.021288 | 0.450523 ± 0.011041 | 0.477988 (SD N/A) |
| LBP + SVM | 0.396248 ± 0.014417 | 0.522589 ± 0.012073 | 0.548415 ± 0.013633 | 0.592401 (SD N/A) |
| ResNet-18 | 0.741756 ± 0.021156 | 0.855151 ± 0.030380 | 0.890470 ± 0.006729 | 0.895314 ± 0.014118 |
| MobileNetV3-Small | 0.723334 ± 0.017074 | 0.813294 ± 0.017598 | 0.854272 ± 0.004630 | 0.895321 ± 0.005977 |

| Method / fraction | Balanced accuracy mean ± SD | Accuracy mean ± SD |
|---|---:|---:|
| HOG 25 / 50 / 75 / 100% | 0.337096±0.010914 / 0.393439±0.015482 / 0.441520±0.010164 / 0.457766 (N/A) | 0.413580±0.018519 / 0.471193±0.009429 / 0.541152±0.007128 / 0.530864 (N/A) |
| LBP 25 / 50 / 75 / 100% | 0.382308±0.014790 / 0.501940±0.008477 / 0.532781±0.006603 / 0.572746 (N/A) | 0.460905±0.018858 / 0.565844±0.015535 / 0.565844±0.012850 / 0.611111 (N/A) |
| ResNet 25 / 50 / 75 / 100% | 0.741139±0.021851 / 0.849951±0.044085 / 0.897392±0.007886 / 0.902476±0.017472 | 0.755144±0.019843 / 0.843621±0.021678 / 0.878601±0.003564 / 0.888889±0.021383 |
| MobileNet 25 / 50 / 75 / 100% | 0.729651±0.022759 / 0.823601±0.025018 / 0.862522±0.006559 / 0.903765±0.004203 | 0.736626±0.012850 / 0.808642±0.012346 / 0.845679±0.006173 / 0.882716±0.006173 |

The main figure is [`macro_f1_learning_curves.png`](../figures/phase7/macro_f1_learning_curves.png). The full numerical table and all run-level values are in `experiments/summaries/phase7_data_efficiency_v1/aggregate/`.

## Retention, absolute loss, and marginal gains

| Method | Retention at 25 / 50 / 75% | Δ to full at 25 / 50 / 75% | Δ25→50 / Δ50→75 / Δ75→100 |
|---|---|---|---|
| HOG | 70.08% / 85.93% / 94.25% | -0.143027 / -0.067276 / -0.027465 | +0.075750 / +0.039811 / +0.027465 |
| LBP | 66.89% / 88.22% / 92.57% | -0.196153 / -0.069813 / -0.043986 | +0.126341 / +0.025827 / +0.043986 |
| ResNet | 82.85% / 95.51% / 99.46% | -0.153558 / -0.040163 / -0.004844 | +0.113395 / +0.035319 / +0.004844 |
| MobileNet | 80.79% / 90.84% / 95.42% | -0.171987 / -0.082027 / -0.041049 | +0.089960 / +0.040978 / +0.041049 |

The smallest tested fraction reaching at least 95% of each method's full-data macro-F1 is 100% for HOG, 100% for LBP, 50% for ResNet, and 75% for MobileNet. No intermediate budget was interpolated.

Normalized macro-F1 learning-curve AUC is 0.422570 for HOG, 0.521776 for LBP, 0.854719 for ResNet, and 0.825631 for MobileNet. This is trapezoidal area over the four tested label budgets divided by 0.75; it is not ROC-AUC.

## Per-class learning curves

Test per-class F1 mean ± sample SD is shown below. Classical 100% entries are single deterministic values.

| Method / class | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| HOG craze | .598±.056 | .610±.020 | .659±.024 | .596 |
| HOG corrosion | .252±.030 | .356±.039 | .497±.042 | .453 |
| HOG surface_injure | .419±.033 | .434±.029 | .498±.020 | .500 |
| HOG thunderstrike | .200±.000 | .255±.094 | .200±.000 | .364 |
| HOG crack | .084±.073 | .273±.053 | .231±.026 | .348 |
| HOG hide_craze | .456±.026 | .536±.036 | .617±.016 | .607 |
| LBP craze | .527±.029 | .654±.057 | .653±.052 | .678 |
| LBP corrosion | .385±.030 | .495±.040 | .512±.028 | .526 |
| LBP surface_injure | .417±.040 | .491±.021 | .445±.030 | .471 |
| LBP thunderstrike | .200±.000 | .309±.094 | .442±.070 | .500 |
| LBP crack | .290±.042 | .547±.064 | .601±.056 | .667 |
| LBP hide_craze | .560±.048 | .639±.019 | .637±.034 | .713 |
| ResNet craze | .733±.043 | .847±.028 | .876±.017 | .929±.003 |
| ResNet corrosion | .730±.066 | .788±.038 | .795±.010 | .848±.032 |
| ResNet surface_injure | .752±.018 | .823±.069 | .878±.014 | .845±.047 |
| ResNet thunderstrike | .693±.023 | .896±.083 | .967±.058 | .947±.091 |
| ResNet crack | .744±.168 | .914±.022 | .928±.005 | .899±.004 |
| ResNet hide_craze | .799±.043 | .864±.006 | .899±.033 | .904±.041 |
| MobileNet craze | .767±.026 | .822±.048 | .870±.014 | .915±.050 |
| MobileNet corrosion | .614±.082 | .720±.042 | .765±.030 | .791±.028 |
| MobileNet surface_injure | .690±.061 | .735±.050 | .804±.015 | .850±.023 |
| MobileNet thunderstrike | .727±.088 | .912±.031 | .928±.034 | .965±.030 |
| MobileNet crack | .731±.034 | .804±.047 | .873±.025 | .945±.036 |
| MobileNet hide_craze | .811±.036 | .887±.021 | .887±.010 | .907±.031 |

Corrosion and surface-injury performance generally remained below the strongest CNN class results. Some per-class curves are non-monotonic, which is reported descriptively and was not used to alter the experiment.

## Thunderstrike low-data analysis

Thunderstrike has only 12/22/32/42 training instances at 25/50/75/100%. Its F1 means at those budgets are HOG 0.200/0.255/0.200/0.364, LBP 0.200/0.309/0.442/0.500, ResNet 0.693/0.896/0.967/0.947, and MobileNet 0.727/0.912/0.928/0.965. Exact precision, recall, and SD values are recorded in `aggregate/thunderstrike_learning_curve.csv`. Given the small training and test support, individual-budget differences should not be overinterpreted.

## Compute and reproducibility

The nine reduced ResNet runs used 13,896.5 seconds (3.86 hours) of recorded training time. The nine reduced MobileNet runs used 4,181.1 seconds (1.16 hours). Combined primary reduced CNN training time was 18,077.6 seconds (5.02 hours). Training time was descriptive and did not affect selection.

The canonical ResNet seed-17/25% rerun and MobileNet seed-17/25% rerun both passed exact equality for initial-head fingerprint, best epoch, scientific history, validation predictions, test predictions, validation/test metrics, and final checkpoint fingerprint. Timings were excluded from exact equality.

## Descriptive interpretation

Across all tested budgets, the transfer-learned CNN means exceed the handcrafted baselines. ResNet reaches 95% of its own full-data macro-F1 at 50%, whereas MobileNet first reaches that threshold at 75%. ResNet retains 99.46% of its full-data macro-F1 at 75%; its observed 75→100 gain is only 0.004844. LBP exceeds HOG at every tested fraction. These are descriptive observations for this benchmark, not significance, equivalence, causal, or universal data-efficiency claims.

## Limitations

- Only 25%, 50%, 75%, and 100% budgets were tested.
- Only three reduced-data source-subset families were used.
- CNN variability combines subset composition and training randomness; HOG/LBP reduced-data variability reflects subset composition.
- Frozen full-data hyperparameters may not be individually optimal at every low-data fraction. This is intentional so the data-budget comparison changes only one experimental factor.
- The same fixed validation and test partitions are reused across all runs.
- The benchmark is six-class visible defect-crop classification, not full-image detection, structural-safety assessment, or deployment validation.
- WTBD has no valid healthy class and cannot support healthy-versus-damaged claims.
- WTBD provides no temporal onset labels, so no earlier-detection conclusion is supported.
- No external-domain validation was performed.

## Phase 7 gate result

**PASS.** All 12 frozen subset manifests and upstream Phase 3–6 validators pass; the processed dataset fingerprint remains `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`. All 36 reduced fits and 36 primary test evaluations completed, the 100% anchors were reused without training, both canonical CNN reproducibility checks passed, and the required machine-readable predictions, logits, metrics, histories, metadata, aggregates, and 12 figures exist. Phase 8 has not started.
