# Phase 0 Research Contract

## Project identity

**Working title:** Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation

**Domain:** wind energy; wind-turbine blade inspection; computer vision; machine learning; robustness; data efficiency.

This is an experimental research project. The software developed in later phases will be the experimental apparatus used to answer the research questions; it is not merely an application or demonstration.

## Motivation

The study will provide a controlled comparison of two handcrafted-feature pipelines and two transfer-learned convolutional neural networks for classifying visible wind-turbine blade surface-defect instances. It focuses on three practical experimental constraints: limited labeled training data, degraded image quality, and computational cost. The benchmark will use the same source-image-aware partitions and underlying defect crops for all four methods so that the comparisons remain interpretable.

## Primary research question

**How do traditional handcrafted image-feature methods and transfer-learned convolutional neural networks compare in classifying visible wind-turbine blade surface defects when labeled training data are limited and image quality is degraded?**

The formal task is classification, not full-image detection. The primary experiment operates on expert-annotated defect crops rather than arbitrary full images.

## Research subquestions

- **RQ1 — Clean performance:** How do HOG+SVM, LBP+SVM, ResNet-18, and MobileNetV3-Small compare on clean wind-turbine blade defect images?
- **RQ2 — Data efficiency:** How does the performance of each method change when only 25%, 50%, 75%, or 100% of the available training data are used?
- **RQ3 — Image-quality robustness:** How does the performance of each method change when test images are degraded by blur, reduced resolution, reduced brightness, or JPEG compression?
- **RQ4 — Computational efficiency:** What performance-versus-efficiency trade-offs exist among the methods in terms of model size, parameter count where applicable, and inference latency?
- **RQ5 — Class-specific difficulty:** Which defect categories are consistently easiest or hardest to classify, and how do their error patterns change across method families and image-quality conditions?

## Hypotheses

- **H1:** Transfer-learned CNNs are expected to achieve higher clean-image macro-F1 than handcrafted HOG/LBP + SVM methods.
- **H2:** The performance advantage of deep models may change as labeled training data become scarcer; classical methods may become relatively more competitive at smaller training fractions.
- **H3:** All methods are expected to degrade as image quality worsens, but the amount and pattern of degradation may differ substantially between handcrafted features and learned CNN representations.
- **H4:** MobileNetV3-Small is expected to provide a stronger efficiency/performance trade-off than ResNet-18, even if ResNet-18 achieves equal or higher absolute classification performance.
- **H5:** Small, subtle, or visually similar defect classes are expected to produce more classification confusion than visually distinctive categories.

These are hypotheses, not conclusions. Any or all of them may be rejected by the experimental results.

## Dataset contract

### Primary dataset

The primary dataset is **WTBD — Wind Turbine Blade Defect dataset**:

Lipeng Ji, Junjie Cheng, and Shilong Wu. “Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery.” *Scientific Data*, 2026. DOI: `10.1038/s41597-026-06762-x`. Dataset DOI: `10.6084/m9.figshare.30210175`.

The following supplied facts are frozen:

- 1,065 real UAV-captured blade images.
- 1,568 annotated defect instances.
- Images standardized to 1024 × 1024 JPEG.
- Bounding-box annotations supplied in PASCAL VOC XML format.
- Six categories and instance counts:

  | Category | Instances |
  |---|---:|
  | craze | 259 |
  | corrosion | 254 |
  | surface_injure | 394 |
  | thunderstrike | 92 |
  | crack | 224 |
  | hide_craze | 345 |
  | **Total** | **1,568** |

- Multiple defect instances may occur in one source image.
- Images were captured using a DJI Phantom 4 Pro V2.0 at an approximate UAV-to-blade distance of 6–10 m.
- Data were collected at a commercial coastal wind farm in Shanghai, China, from May through August 2023.
- Capture conditions included sunny, cloudy, and overcast weather.
- The dataset authors excluded images with major motion blur/out-of-focus targets, unusable over/under-exposure, and images containing no visible defects.
- The released dataset therefore has **no valid healthy/normal image class**.
- Two annotators independently checked the annotations; the reported Cohen's kappa is 0.8970.
- The repository provides a standard train/validation/test split index file.

Consequently, the core task is six-class classification among already-visible defect categories. It must not be described as healthy-versus-damaged classification. It must not claim “earlier detection,” because the dataset has no temporal defect-onset labels.

### Instance construction

Each annotated WTBD defect instance will become one classification sample:

1. Parse its PASCAL VOC annotation.
2. Extract the annotated bounding-box crop from the original image.
3. Add a small, fixed context margin around the box.
4. Clamp the crop to the image boundaries.
5. Preserve the ground-truth defect category.
6. Resize or pad consistently in the later preprocessing phase.
7. Record the source image ID for every crop.

The exact context-margin implementation remains intentionally unresolved in Phase 0. It must be frozen before model experiments and applied identically to every method.

### Leakage prevention

All crops derived from one source image must remain in the same train, validation, or test partition. Crops must never be independently random-split. The dataset's supplied standard source-image split will be used when it is available and valid. Every model family must use the same underlying crops and partitions.

### Secondary datasets

Blade30, DTU Drone Inspection Images, and WTBs2025 are not core Phase 0 data and must not be combined with WTBD for core training. They are documented separately in [source_ledger.md](source_ledger.md) as possible later external-validation sources.

WTBs2025 must not be the primary data-efficiency dataset: its ordinary augmentation and GA-DCGAN augmentation provenance could confound training-size comparisons or create leakage unless original-image ancestry is first verified.

No generated, synthetic, or web-scraped image may enter the core WTBD experiment.

## Core methods

The core model set is frozen to four methods:

1. **HOG + SVM:** Histogram of Oriented Gradients features with a Support Vector Machine classifier; an edge/gradient-oriented handcrafted approach.
2. **LBP + SVM:** Local Binary Pattern texture features with a Support Vector Machine classifier; a texture-oriented handcrafted approach.
3. **ResNet-18:** standard torchvision-compatible ResNet-18 with ImageNet pretrained weights, fine-tuned for six-class WTBD defect classification.
4. **MobileNetV3-Small:** standard torchvision-compatible MobileNetV3-Small with ImageNet pretrained weights, fine-tuned for six-class WTBD defect classification; the lightweight/deployment-oriented CNN.

The core model set excludes Vision Transformers, EfficientNet, custom CNN architectures, Faster R-CNN, Mask R-CNN, YOLO, foundation models, and vision-language models. Any such model requires a later, explicitly approved extension.

### Why Vision Transformers are excluded

The purpose is not a broad architecture leaderboard. The supplied methodological context already includes a direct comparison of lightweight CNNs and Vision Transformers in recent wind-turbine-blade literature. This study instead focuses on handcrafted features versus transfer-learned CNNs, data efficiency, degradation robustness, and computational trade-offs. A restricted model set improves interpretability and feasibility.

## Metrics

**Macro F1-score is the primary model-comparison metric.** The six WTBD classes are imbalanced, and ordinary accuracy could hide poor performance on a rare category such as thunderstrike. Models must not be optimized solely for raw accuracy.

Secondary predictive metrics are:

- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix.

Efficiency and descriptive measurements are:

- inference latency per image;
- model or checkpoint size;
- number of trainable parameters where meaningful;
- training time as a descriptive measure.

## Data-efficiency protocol

- Training fractions are frozen at 25%, 50%, 75%, and 100% of the training partition only.
- Validation and test partitions remain fixed for all fractions.
- Training subsets will be stratified as far as feasible.
- The primary repeated experiment uses exactly three reproducible random seeds. Their numeric values must be frozen before Phase 3.
- The same seeded subsets must be used across competing methods wherever applicable.
- Every model/fraction combination will report each individual-seed result, the mean, and the standard deviation. Reporting only the best seed is prohibited.

## Robustness protocol

Robustness evaluation measures already-selected models on worse-quality versions of the fixed held-out test set. Models must not be retrained or fine-tuned on corrupted test images.

Four corruption families are frozen:

1. Gaussian blur;
2. resolution degradation;
3. brightness reduction;
4. JPEG compression.

Each family will have clean/control, mild, moderate, and severe levels. Exact numerical corruption parameters must be placed in a version-controlled configuration before robustness results are inspected. The transformations must be deterministic for a given configuration and seed.

Results will report both absolute macro-F1 at every corruption level and the macro-F1 drop relative to the clean test condition. The families must not be collapsed into one opaque “bad image” metric.

## Training, model-selection, and reporting safeguards

- Test data may never be used for training or hyperparameter selection.
- Test performance may not determine preprocessing choices.
- Validation data are used for model selection and early stopping.
- Core deep models use ImageNet pretrained weights; training from scratch is not a required core experiment.
- Every experiment must be reproducible from a recorded configuration and random seed.
- Every final reported number must originate from machine-readable experiment output, not manual transcription.
- No method may receive a deliberately weak or unfair preprocessing pipeline.
- All model families use the same underlying crops and source-image-aware partitions.
- Class imbalance must be tracked explicitly.
- Any weighting or balancing strategy must be recorded and applied under a predeclared protocol.
- Generated, synthetic, and web-scraped images are prohibited from the core WTBD experiment.

## Non-claims

The project concerns visual classification of already-visible surface-defect instances. The core project does **not** establish:

- whether a blade is structurally safe;
- whether a blade requires repair;
- remaining useful life;
- future failure probability;
- defect growth rate;
- causal effects of weather on blade damage;
- actual financial savings;
- autonomous UAV navigation capability;
- superiority to human inspectors;
- superiority to commercial industrial inspection systems;
- earlier detection of damage;
- hidden or internal defect detection;
- generalization to every wind farm;
- real-time edge deployment unless later measured on actual target hardware.

## Positioning and novelty constraints

The project must not claim that nobody has compared deep learning and traditional computer vision for wind-turbine blade defects. It must not use unsupported phrases such as “the first ever,” “novel state-of-the-art architecture,” or “revolutionary.”

The defensible contribution is a controlled benchmark combining a recent real UAV defect dataset, identical source-image-aware partitions, handcrafted features, standard transfer-learned CNNs, systematically varied labeled-data availability, controlled image degradation, class-level error analysis, and computational-efficiency comparison. Appropriate language includes “This study systematically evaluates…” and “This study investigates…”. The supplied literature establishes context only and does not prove the hypotheses.

## Optional future extensions — outside Phase 0

The following are recorded but not authorized for implementation now:

- **Phase 10:** multi-class/error-analysis extensions if warranted.
- **Phase 11:** full-image object detection/localization using a lightweight YOLO model on the original WTBD bounding-box annotations.
- **External validation:** Blade30 and/or DTU Drone Inspection Images if class mapping and licensing permit scientifically valid evaluation.
- **Possible Grad-CAM analysis:** interpretability/error analysis only after core results are frozen.

## Phase 0 freeze statement

This document freezes the Phase 0 research contract. The research question, six-class task, primary dataset, core methods, primary metric, training fractions, repeated-seed count, corruption families, leakage rules, test-set policy, safeguards, positioning constraints, and non-claims may not be changed in later phases unless an explicit revision is requested and documented. Numeric implementation details explicitly reserved for later phases are not silently implied by this contract.

Phase 0 involved documentation only: no external literature research, dataset download, model implementation, or model training is part of this phase.
