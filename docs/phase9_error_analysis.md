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

The canonical manifest contains 16,848 rows: 162 test samples × 13 conditions × eight model/seed identities. All 104 prediction sets reproduce the frozen Phase 8 sample IDs, source IDs, labels, predictions, and CNN logits. The tables preserve exact seed-level rows and denominators.

### Clean class confusions and difficulty indicators

- HOG's lowest clean recall was thunderstrike (`0.2222`, support 9); its largest single clean off-diagonal cell was `surface_injure → hide_craze` (10).
- LBP's lowest clean recall was thunderstrike (`0.3333`, support 9); its largest single clean off-diagonal cell was `hide_craze → surface_injure` (8).
- ResNet-18's lowest mean clean recall was corrosion (`0.8000`, support 30). Its lowest mean clean F1 was surface_injure (`0.8454`).
- MobileNetV3-Small's lowest mean clean recall and F1 were corrosion (`0.7556` and `0.7913`, support 30).

These are observed difficulty indicators, not causal explanations. In particular, support alone is not asserted to cause an error pattern.

### Severe prediction transitions

Mean harmful flips per 162-sample severe condition, with sample SD across CNN seeds, were:

| Method | Blur | Resolution | Brightness | JPEG |
|---|---:|---:|---:|---:|
| HOG + SVM | 24 | 16 | 22 | 28 |
| LBP + SVM | 58 | 64 | 44 | 48 |
| ResNet-18 | 42.33 ± 4.04 | 31.00 ± 4.36 | 53.33 ± 9.29 | 17.33 ± 4.73 |
| MobileNetV3-Small | 36.00 ± 4.36 | 26.67 ± 0.58 | 37.67 ± 4.04 | 17.00 ± 2.65 |

Thus, the largest observed severe harmful-flip count occurred under JPEG for HOG, resolution degradation for LBP, and brightness reduction for both CNNs. Beneficial flips and all five degraded-event categories remain in the seed-level transition table; no family is collapsed into an opaque degradation score.

### CNN seed stability

On clean images, ResNet-18 had 7 unanimous failures, 9 majority-only failures, and 26 samples with predicted-label disagreement; MobileNetV3-Small had 13, 4, and 19, respectively. Summed across the four severe conditions (648 sample-condition records per architecture), ResNet-18 had 97 unanimous failures, 78 majority-only failures, and 253 label-disagreement records; MobileNetV3-Small had 100, 67, and 177. These are descriptive counts, not tests of statistical significance.

### Cross-method agreement

Under the strict rule requiring all CNN seeds, clean results included 10 samples misclassified by all four method families, 22 correctly classified by both CNNs but missed by both handcrafted methods, 17 missed by both CNNs, and 38 with exactly one model-family failure. Under the at-least-two-seed majority rule, the corresponding counts were 7, 32, 11, and 44. Across the four severe conditions, 93 sample-condition records had harmful flips shared by at least two model families under strict CNN consensus; the majority-rule count was 140.

### Crop geometry

The complete geometry tables report counts and error/harmful-flip rates for the four fixed occupancy bins, `boundary_shifted`, and `max_side_clipped`, while retaining original contextual crop side continuously. For example, among the substantive clean occupancy bins, the highest observed seed-17/deterministic error rate for each family occurred in `0.10–<0.25` for HOG (36/63), `<0.10` for LBP (11/24), and `0.10–<0.25` for ResNet (8/63) and MobileNet (9/63). The `>=0.50` bin contains only two test samples. These are descriptive associations; crop geometry is not claimed to cause errors.

## Human-review packet

The deterministic packet contains exactly 60 distinct sample-condition cases with no quota shortfall:

- 10 clean consensus errors;
- 20 severe harmful flips;
- 8 severe stable-wrong cases;
- 8 severe stable-correct cases;
- 6 beneficial flips;
- 8 CNN seed-disagreement cases.

The resulting fixed set contains 51 ResNet-18 and 9 MobileNetV3-Small cases and spans all six true classes. This distribution is a consequence of the frozen consensus-first, class-round-robin algorithm and was not altered after viewing results.

The packet is under [`experiments/summaries/phase9_error_analysis_v1/human_review_packet/`](../experiments/summaries/phase9_error_analysis_v1/human_review_packet/). Pass A contains 220 anonymized image panels and reveals only the review ID, clean/degraded crops where applicable, annotation, and dataset true label. It contains no model identity, prediction, correctness, event, Grad-CAM, or seed. Pass B contains 507 linked Grad-CAM evidence records across the three seeds. Both HTML indexes have zero missing image links, both CSV review forms are unfilled, and the review-ID mapping is stored separately.

The human reviewer must complete Pass A fully before opening Pass B, then return both completed CSV forms for Phase 9B. No Codex or vision-model judgment has been substituted for human review.

## Validation and reproducibility

- Apparatus commit: `9758f139d5c35fab3d13d955ce5457fa843b0795`.
- Generated-output commit: `95e1bd7`.
- Analysis-config fingerprint: `9a9c87f132359718636f46d1b9061e8e5539980b6c9b67a3f5336834ce1c9f1e`.
- Frozen-input fingerprint: `dbedb680481e6477b94ff623858e1db4c1981f2b8f5491d0e04a99b5672173b1`.
- Scientific-output fingerprint: `a5938ec22a0f496c6fd9ce3acd7d999cad56e8c13a85d9be4b4d59860aecd74b`.
- Two complete clean-state generations: PASS across 1,973 compared files, including manifests, numerical tables, selected IDs, canonical Grad-CAM hashes, rendered figures, and inventory.
- Independent Phase 9A validator: PASS for 16,848 manifest rows, 104 prediction identities, 96 degraded transition rows, all review assets, blank forms, and unchanged checkpoints/inputs.
- Grad-CAM integrity: 507 finite records; exact in-process map regeneration; all model state fingerprints identical before and after generation. ResNet activations were `1×512×7×7`; MobileNet activations were `1×576×7×7`.
- Expanded test suite: 164 passed, 0 failed, with 11 unchanged scikit-learn deprecation warnings.
- Training, fine-tuning, SVM/scaler refitting, automatic visual interpretation, and Phase 10/app generation: zero.

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

`PHASE 9A COMPLETE — AWAITING HUMAN REVIEW`

Phase 9 is not yet complete or frozen. The next authorized step is Phase 9B after the completed human-review forms are returned. No qualitative image or Grad-CAM interpretation has been made in Phase 9A.
