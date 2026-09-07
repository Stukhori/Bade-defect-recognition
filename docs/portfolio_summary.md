# BladeScope Portfolio Summary

## Problem

BladeScope is a reproducible computer-vision research project focused on wind-turbine blade surface defects. The central question is not only whether a model can classify six defect categories, but how its performance changes when training data are limited and image quality is degraded in controlled ways. The work also examines model errors, visual attribution, human interpretation, and class-agnostic defect localization. It does not claim to assess structural safety or replace an operational inspection process.

## Methodology

The project begins with an audit and curation of the Wind Turbine Blade Defect (WTBD) dataset. The retained corpus contains 720 defect-positive full images and 1,065 annotated defect instances across six classes. Source-image-level splitting prevents crops from the same source image from crossing train, validation, and test boundaries. A deterministic crop pipeline produces 224×224 RGB classification inputs while preserving manifests, checksums, configuration snapshots, and dataset fingerprints.

Four classification approaches are compared: HOG + SVM, LBP + SVM, ResNet-18, and MobileNetV3-Small. Hyperparameter and checkpoint decisions use validation data; held-out test evaluation occurs only after selection is frozen. The CNN experiments use seeds 17, 29, and 43. Separate studies measure data efficiency at four declared training-data budgets and robustness under 12 fixed synthetic degradation conditions spanning blur, resolution, brightness, and JPEG compression.

Post-hoc analysis covers confusion patterns, harmful and beneficial prediction flips, cross-model failures, and seed disagreement. Grad-CAM visualizations support a blinded two-pass human-review workflow: judgments about images and annotations were completed before model evidence was revealed. A final statistical synthesis uses paired, true-class-stratified bootstrap resampling while retaining all CNN seeds.

The localization extension uses a compact YOLO11n detector for class-agnostic defect-region proposals. Three training seeds were completed, checkpoint and threshold selection used validation evidence only, and the final held-out result was frozen with no further tuning. BladeScope's Streamlit application combines one exact frozen detector checkpoint with the frozen six-category MobileNetV3-Small crop classifier. Users review detected regions before classification, and prepared-crop and manual-region workflows remain available.

## Key Findings

Frozen clean-test macro-F1 was `0.477988` for HOG + SVM, `0.592401` for LBP + SVM, `0.895314 ± 0.014118` for ResNet-18, and `0.895321 ± 0.005977` for MobileNetV3-Small. Both CNN means were substantially higher than the handcrafted baselines under the declared test conditions. MobileNetV3-Small had the highest observed mean performance and retention across the fixed degradation grid, while ResNet-18 had the highest normalized data-efficiency area. Their paired clean macro-F1 interval does not justify claiming that either model is superior or that they are equivalent.

## Technical Scope and Reproducibility

The implementation uses Python 3.11, PyTorch and torchvision, scikit-learn, NumPy, scikit-image, Pillow, matplotlib, Ultralytics YOLO, Streamlit, pytest, and uv. The repository contains configuration-driven pipelines, source-isolated split manifests, artifact hashes, provenance records, frozen evaluation outputs, focused validators, and a complete automated test suite. Detailed phase documents connect numerical claims to machine-readable tables and scientific-output fingerprints.

## Limitations

WTBD is limited in size and contains no healthy/background-only controls, so detector false-positive behavior on healthy blades is not measured. The synthetic degradations are controlled design points rather than a substitute for field cameras, weather, sites, or drone-flight conditions. Grad-CAM is descriptive rather than causal evidence, and the human review reflects one reviewer. The detector remains an experimental proposal aid. The project does not support safety, severity, progression, remaining-life, or production-readiness claims.

## What Differentiates the Project

BladeScope treats reproducibility and uncertainty as core engineering requirements rather than presentation extras. It connects data curation, multiple model families, constrained-data testing, controlled robustness experiments, interpretability, blinded review, statistical synthesis, localization, and an interactive application without overwriting the boundaries between scientific evidence and demonstration software.
