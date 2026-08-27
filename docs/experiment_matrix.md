# Phase 0 Experiment Matrix

This document freezes the planned experimental comparisons while separating them from numeric implementation details that must be predeclared in later phases. No result is recorded or implied here.

## Status key

- **Frozen now:** part of the Phase 0 research contract and not changeable without an explicit documented revision.
- **Later freeze required:** intentionally unresolved implementation detail that must be placed in version-controlled configuration before the applicable experiment or result inspection.

## Core methods

| Method family | Core method | Representation | Classifier / head | Initialization | Phase 0 status |
|---|---|---|---|---|---|
| Classical | HOG + SVM | Histogram of Oriented Gradients | Support Vector Machine | Not applicable | **Frozen now** |
| Classical | LBP + SVM | Local Binary Pattern texture features | Support Vector Machine | Not applicable | **Frozen now** |
| Deep learning | ResNet-18 pretrained | Learned CNN features | Six-class classification head | ImageNet pretrained weights | **Frozen now** |
| Deep learning | MobileNetV3-Small pretrained | Learned lightweight CNN features | Six-class classification head | ImageNet pretrained weights | **Frozen now** |

No additional core architecture is authorized. In particular, Vision Transformers, EfficientNet, custom CNNs, Faster R-CNN, Mask R-CNN, YOLO, foundation models, and vision-language models are outside the core matrix.

## Data-efficiency matrix

The percentages below apply only to the training partition. The validation and test partitions remain fixed. Each cell is evaluated with exactly three reproducible seeds, uses the same seeded training subset across methods wherever applicable, and reports every seed plus mean and standard deviation.

| Core method | 25% train | 50% train | 75% train | 100% train |
|---|---:|---:|---:|---:|
| HOG + SVM | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| LBP + SVM | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| ResNet-18 pretrained | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| MobileNetV3-Small pretrained | 3 seeds | 3 seeds | 3 seeds | 3 seeds |

This defines 16 model/fraction cells and 48 individual seeded evaluations. Training subsets must be stratified as far as feasible. Selecting or reporting only the best seed is prohibited.

| Data-efficiency detail | Status |
|---|---|
| Fractions: 25%, 50%, 75%, 100% | **Frozen now** |
| Fractions affect training only | **Frozen now** |
| Fixed validation and test sets | **Frozen now** |
| Exactly three reproducible seeds | **Frozen now** |
| Same seeded subsets across methods wherever applicable | **Frozen now** |
| Individual-seed, mean, and standard-deviation reporting | **Frozen now** |
| Exact three numeric seed values | **Later freeze required before Phase 3** |
| Exact stratified-subset construction algorithm and infeasibility handling | **Later freeze required before subset generation** |

## Robustness matrix

Robustness evaluation occurs on the fixed held-out test set after model selection. There is no retraining or fine-tuning on corrupted test images.

| Corruption family | Clean/control | Mild | Moderate | Severe | Phase 0 status |
|---|---:|---:|---:|---:|---|
| Gaussian blur | Evaluate | Evaluate | Evaluate | Evaluate | Family and level labels **frozen now** |
| Resolution degradation | Evaluate | Evaluate | Evaluate | Evaluate | Family and level labels **frozen now** |
| Brightness reduction | Evaluate | Evaluate | Evaluate | Evaluate | Family and level labels **frozen now** |
| JPEG compression | Evaluate | Evaluate | Evaluate | Evaluate | Family and level labels **frozen now** |

For every model and corruption level, report:

- absolute macro-F1;
- macro-F1 drop relative to the clean test condition.

Each family must be reported separately rather than combined into an opaque aggregate “bad image” metric. Transformations must be deterministic for a given configuration and seed.

| Robustness detail | Status |
|---|---|
| Four corruption families | **Frozen now** |
| Clean/control, mild, moderate, severe levels | **Frozen now** |
| Fixed held-out test set, evaluated after model selection | **Frozen now** |
| No retraining or fine-tuning on corrupted test images | **Frozen now** |
| Absolute and clean-relative macro-F1 reporting | **Frozen now** |
| Exact blur kernel/sigma values | **Later freeze required before robustness results are inspected** |
| Exact downsampling scale, resampling, and restoration procedure | **Later freeze required before robustness results are inspected** |
| Exact brightness factors and color-handling procedure | **Later freeze required before robustness results are inspected** |
| Exact JPEG quality values and codec settings | **Later freeze required before robustness results are inspected** |
| Determinism configuration | **Later freeze required before robustness results are inspected** |

## Metrics matrix

| Category | Measure | Role | Phase 0 status |
|---|---|---|---|
| Primary | Macro F1-score | Main model-comparison metric | **Frozen now** |
| Secondary | Accuracy | Overall predictive performance | **Frozen now** |
| Secondary | Balanced accuracy | Class-balanced predictive performance | **Frozen now** |
| Secondary | Macro precision | Class-balanced precision summary | **Frozen now** |
| Secondary | Macro recall | Class-balanced recall summary | **Frozen now** |
| Class-level | Per-class precision | Class-specific analysis | **Frozen now** |
| Class-level | Per-class recall | Class-specific analysis | **Frozen now** |
| Class-level | Per-class F1 | Class-specific analysis | **Frozen now** |
| Error analysis | Confusion matrix | Pairwise class-confusion analysis | **Frozen now** |
| Efficiency | Inference latency per image | Runtime trade-off | **Frozen now** |
| Efficiency | Model/checkpoint size | Storage trade-off | **Frozen now** |
| Efficiency | Trainable parameter count where meaningful | Model-complexity description | **Frozen now** |
| Descriptive | Training time | Training-cost description | **Frozen now** |

Raw accuracy must not be the sole optimization target. Exact metric-library versions, averaging behavior for undefined class values, latency measurement protocol, hardware, warm-up/repetition procedure, and size-accounting convention require later version-controlled specification before measurements are viewed.

## Dataset and partition matrix

| Item | Contract | Phase 0 status |
|---|---|---|
| Primary dataset | WTBD only | **Frozen now** |
| Prediction task | Six-class classification of expert-annotated visible defect crops | **Frozen now** |
| Classes | craze, corrosion, surface_injure, thunderstrike, crack, hide_craze | **Frozen now** |
| Crop unit | One classification sample per annotated bounding box | **Frozen now** |
| Grouping key | Source image ID | **Frozen now** |
| Partition rule | Every crop from one source image remains in one partition | **Frozen now** |
| Preferred split | Supplied standard source-image split when available and valid | **Frozen now** |
| Healthy/normal class | Absent; no healthy-vs-damaged claim | **Frozen now** |
| Context margin | Small and fixed, boundary-clamped, identical across methods | Principle **frozen now**; exact implementation **later freeze required** |
| Resize/pad preprocessing | Consistent across methods under a fair protocol | Principle **frozen now**; exact implementation **later freeze required** |
| Synthetic/web-scraped data | Prohibited from core WTBD experiment | **Frozen now** |

## Model-selection and output rules

| Rule | Phase 0 status |
|---|---|
| Test data are never used for training or hyperparameter selection | **Frozen now** |
| Test results cannot determine preprocessing | **Frozen now** |
| Validation data drive model selection and early stopping | **Frozen now** |
| Deep core methods use ImageNet pretrained weights | **Frozen now** |
| Every run is reproducible from configuration and seed | **Frozen now** |
| Final numbers originate from machine-readable outputs | **Frozen now** |
| Class imbalance is tracked explicitly | **Frozen now** |
| Any balancing/weighting strategy follows a predeclared recorded protocol | Principle **frozen now**; exact strategy **later freeze required** |
| HOG, LBP, SVM, CNN training, validation-selection, and early-stopping hyperparameters | **Later freeze required before applicable experiments** |

## Phase ordering constraint

This matrix is planning documentation only. Phase 0 does not authorize dataset downloading, pipeline implementation, preprocessing, model fitting, hyperparameter selection, corruption-result inspection, or any optional extension. Phase 1 is limited to repository and reproducibility infrastructure as separately authorized.
