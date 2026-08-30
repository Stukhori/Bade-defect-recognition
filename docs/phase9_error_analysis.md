# Phase 9A — Quantitative Error Analysis, Grad-CAM, and Blinded Human Review

## Purpose

Phase 9A is a post-hoc, descriptive analysis of the frozen clean and controlled-degradation predictions from Phases 4–8. It quantifies class confusions, prediction transitions, CNN seed stability, cross-method agreement, observed class-difficulty indicators, and associations with Phase 3 crop geometry. It also generates independently normalized Grad-CAM evidence and a two-pass blinded packet for later human review.

Phase 9A does not train, fine-tune, refit, select, or modify a model. It does not change labels, partitions, crops, annotations, corruption definitions, predictions, logits, or checkpoints. Grad-CAM is not treated as proof of causal model reasoning, and no visual explanation is generated automatically.

## Frozen inputs and preflight

- Phase 3 processed-dataset fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Phase 8 corruption-configuration fingerprint: `a6a9e40c9c3de7130892df3cf49698f95718ce9241c8a963849016ce7adc1d57`.
- Phase 8 robustness-dataset fingerprint: `da0eda8956adfef63e001d0d1614279a907c46feffc5e2f180c5acdbe89987db`.
- Frozen methods: HOG+SVM, LBP+SVM, ResNet-18 seeds 17/29/43, and MobileNetV3-Small seeds 17/29/43.
- Frozen conditions: clean plus the twelve Phase 8 family/severity combinations.
- Fixed test partition: 162 instances from 109 source images.

The canonical preflight reruns every Phase 3–8 validator, verifies all 104 prediction sets and CNN logits, verifies crop/source/label identities and geometry, checks all six checkpoint hashes, and audits commit `681ed81`. That commit modified only the Phase 8 report: it added the already-generated sample standard deviations to the per-class presentation table. No prediction, logit, image, label, checkpoint, or aggregate macro-F1 artifact changed.

## Analysis contract

The complete machine-readable contract is [`configs/error_analysis.yaml`](../configs/error_analysis.yaml). Its event definitions, occupancy bins, selection quotas, class order, Grad-CAM layers, normalization/rendering policy, review fields, input fingerprints, and prohibited operations are fixed before scientific generation.

## Event definitions and seed policy

- `stable_correct`: clean correct and degraded correct.
- `harmful_flip`: clean correct and degraded incorrect.
- `beneficial_flip`: clean incorrect and degraded correct.
- `changed_wrong`: clean and degraded incorrect, with the predicted label changed.
- `stable_wrong`: clean and degraded incorrect, with the predicted label unchanged.
- `clean_only`: clean-condition row.

CNN rows remain seed-specific. No primary prediction is formed by averaging softmax values. Strict consensus requires all three seeds; majority agreement requires at least two. Individual seeds remain available beneath all descriptive aggregates. CNN softmax values and SVM scores are not treated as interchangeable calibrated certainty.

## Deterministic exemplar selection

The target packet contains 60 distinct sample-condition cases: 10 clean consensus errors, 20 severe harmful flips, 8 severe stable-wrong cases, 8 severe stable-correct cases, 6 beneficial flips, and 8 CNN seed-disagreement cases. Eligibility uses consensus before majority, candidates are grouped by event, method, family, and true class, instance IDs are sorted immutably, and selection cycles through the frozen class order. A case is never replaced based on visual appearance. Any shortage is reported without weakening the rule.

## Grad-CAM implementation

ResNet-18 uses `layer4.1` with expected activation shape `1×512×7×7`; MobileNetV3-Small uses `features.12` with expected shape `1×576×7×7`. The instantiated torchvision models were inspected to freeze these paths. Every selected CNN case is processed for seeds 17, 29, and 43. Transition cases receive clean and degraded maps; every state receives a true-class map, plus a predicted-class map when incorrect.

Maps use the frozen evaluation transform, ReLU, independent per-map min–max normalization, bilinear resizing to 224×224, the fixed `magma` colormap, and alpha 0.45. The full crop background remains present, and the original annotation is transformed into resized-crop coordinates. Heatmaps are never averaged across seeds. Because maps are normalized separately, their color intensity cannot be compared quantitatively across maps.

## Quantitative results

Pending apparatus commit and canonical two-pass generation.

## Human-review packet

Pending apparatus commit and canonical two-pass generation. Human fields will remain blank until the packet is returned for Phase 9B.

## Limitations

All analyses are post hoc and descriptive. They do not establish statistical significance or causality. Crop geometry may be associated with errors but cannot be claimed to cause them. Class frequency alone cannot be claimed to cause poor performance. Grad-CAM localizes gradient-weighted activation and does not identify a physical defect, prove what caused a prediction, or validate a mechanistic explanation. Qualitative claims are withheld until human review.

## Reproduction commands

```powershell
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --apparatus-check
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --validate-only
uv run python -m pytest --basetemp .phase9_pytest_tmp -p no:cacheprovider
```

## Status

APPARATUS READY — SCIENTIFIC PHASE 9A GENERATION NOT YET RUN
