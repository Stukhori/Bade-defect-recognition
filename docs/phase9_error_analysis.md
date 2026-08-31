# Phase 9 — Quantitative Error Analysis, Grad-CAM, Blinded Review, and Descriptive Synthesis

## Purpose

Phase 9A is a post-hoc, descriptive analysis of the frozen clean and controlled-degradation predictions from Phases 4–8. It quantifies class confusions, prediction transitions, CNN seed stability, cross-method agreement, observed class-difficulty indicators, and associations with Phase 3 crop geometry. It also generates independently normalized Grad-CAM evidence and a two-pass blinded packet. Phase 9B validates the completed human forms, opens the frozen mapping only for incorporation, joins the judgments to the selected-case and seed-level prediction metadata, and produces a deterministic descriptive synthesis.

Neither subphase trains, fine-tunes, refits, selects, or modifies a model. Phase 9 does not change labels, partitions, crops, annotations, corruption definitions, predictions, logits, checkpoints, Grad-CAM arrays, or review-case identities. Grad-CAM is not treated as proof of causal model reasoning, and no visual explanation is generated automatically.

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

The packet is under [`experiments/summaries/phase9_error_analysis_v1/human_review_packet/`](../experiments/summaries/phase9_error_analysis_v1/human_review_packet/). Pass A contains 220 anonymized image panels and reveals only the review ID, clean/degraded crops where applicable, annotation, and dataset true label. It contains no model identity, prediction, correctness, event, Grad-CAM, or seed. Pass B contains 507 linked Grad-CAM evidence records across the three seeds. Both HTML indexes have zero missing image links, and the review-ID mapping is stored separately. Both human forms are complete and frozen. The Pass B judgments were made from the corrected packet documented in [`phase9a_gradcam_reporting_audit.md`](phase9a_gradcam_reporting_audit.md).

The reviewer attested that every Pass A judgment was saved before Pass B was opened, both forms contain the reviewer's own visual judgments, and no review fields were filled automatically. Phase 9B records that exact statement and does not use the separate AI-comparison workbook.

## Phase 9B descriptive human-review synthesis

### Input validation and provenance

- Pass A: 60 ordered review IDs, 300/300 required responses, 30 nonempty notes, SHA-256 `3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e`.
- Pass B: 60 ordered review IDs, 240/240 required responses, 60 nonempty notes, SHA-256 `0f5258e06a4e854d338705bcf1d38ced048f0652a99ccc4639b18c3baae1cd96`.
- All values match the frozen response choices; no value was inferred, repaired, normalized, or rewritten during Phase 9B.
- The mapping remains one-to-one and ordered at SHA-256 `46b19248797997e8aa7236b9c3fbf17f972977f2cdb2c3a958bd12513f25210b`.
- The corrected Pass B ZIP is SHA-256 `fbfe1985e8a1809884c0df57a0e9a8eb265815988ad66583a8d9c5a3e84dcaf5`. Its blank-form and corrected-index hashes match the declared provenance, and all 1,680 non-form files match the repository with zero missing or changed files. The superseded caption-bugged packet was not used.

### Derived tables and figures

The synthesis contains 60 joined review-case rows and 180 joined case-seed prediction rows. It reports all nine required response fields in a 36-row response table, 1,152 case-level cross-tab rows, and 2,080 seed-prediction-level cross-tab rows. Optional notes remain verbatim in the joined case table; they were not converted into themes or treated as systematically coded qualitative data.

Five figures summarize Pass A responses, Pass B responses, activation location by selected event group, degradation-artifact activation by corruption family, and seed-pattern consistency by selected event group. No image was interpreted automatically.

### Descriptive findings

- Defects were judged visible in 50/60 cases and partially visible in 10/60. Corruption obscured diagnostic detail in 25/60 cases: 16 mild and 9 strong.
- Dataset labels were judged visually plausible in 51/60 cases, implausible in 1/60, and uncertain in 8/60. Category ambiguity was marked yes for 12/60, while crop/background concern was marked yes for 14/60.
- Activation was judged primarily inside the annotation for 27/60 cases, partially inside for 24/60, and outside for 9/60. The stable-correct group had 7/8 inside judgments; harmful-flip cases had 11/20 inside and 7/20 partial judgments.
- Activation was never judged concentrated on the degradation artifact: 45/60 were no and 15/60 uncertain. Uncertainty was more frequent within the reviewed Gaussian-blur cases (7/13) and JPEG cases (2/3) than within brightness cases (6/34). These small, selected-group denominators are descriptive only.
- Cross-seed activation patterns were judged consistent in 24/60 cases, partly consistent in 30/60, and inconsistent in 6/60. Within the eight selected seed-disagreement cases, the counts were 3 yes, 3 partly, and 2 no; within the eight stable-correct cases, 7 were yes and 1 partly.
- After reveal, predictions were judged visually understandable in 57/60 cases and uncertain in 3/60; none were marked no. “Understandable” does not mean correct and does not establish that the displayed activation caused the prediction.

These percentages describe one reviewer's post-hoc judgments on a deliberately selected 60-case packet. They are not estimates for the full test set, real turbine inspections, or real-flight deployment.

## Validation and reproducibility

- Apparatus commit: `9758f139d5c35fab3d13d955ce5457fa843b0795`.
- Generated-output commit: `95e1bd7`.
- Analysis-config fingerprint: `9a9c87f132359718636f46d1b9061e8e5539980b6c9b67a3f5336834ce1c9f1e`.
- Frozen-input fingerprint: `dbedb680481e6477b94ff623858e1db4c1981f2b8f5491d0e04a99b5672173b1`.
- Corrected scientific-output fingerprint: `14e500fd94fa871bbe1e6bee6494d3158fe003c3799e0ccfbcab8e549adf80fa` (supersedes `a5938ec22a0f496c6fd9ce3acd7d999cad56e8c13a85d9be4b4d59860aecd74b` after a caption-only Pass B correction).
- The original two complete clean-state generations matched across 1,973 files. The corrected Pass B page rendered identically twice; every other generated hash and the output inventory remained unchanged.
- Independent Phase 9A validator: PASS for 16,848 manifest rows, 104 prediction identities, 96 degraded transition rows, all review assets, all 507 role-specific Grad-CAM targets and captions, both completed forms, and unchanged checkpoints/inputs.
- Grad-CAM integrity: 507 finite records; exact in-process map regeneration; all model state fingerprints identical before and after generation. ResNet activations were `1×512×7×7`; MobileNet activations were `1×576×7×7`.
- Phase 9B apparatus commit: `29ca2fe`; human-review provenance and output commit: `b4e6727`.
- Phase 9B configuration fingerprint: `ac60d00ffb9b0b729c334a9b5b76bf0f3630c09fd7e5b78b7f304f3048263830`.
- Two separate Phase 9B temporary generations matched for all 15 derived files. Derived-output fingerprint: `dc940fe0a802d285b236224b04197f7fb7d6b0ef73062a5bdddbb1238de286d3`.
- Phase 3–8 validators, Phase 9A validator, Phase 9B validator, review-interface validator, and classifier-app validator: PASS.
- Focused Phase 9/review-interface suite: 46 passed. Complete suite: 222 passed, 0 failed, with the same 11 scikit-learn `probability` deprecation warnings.
- Training, fine-tuning, SVM/scaler refitting, automatic visual interpretation, and Phase 10/app generation: zero.
- Caption audit: 330/330 true-class targets and 177/177 predicted-class targets match frozen labels and indices. Independent read-only recomputation reproduced all 507 stored array hashes. The 177 defective true-class captions across 52 cases were corrected without changing arrays or figures.

## Limitations

All analyses are post hoc and descriptive. Phase 9B has one human reviewer, so inter-rater agreement is neither available nor calculated. Response percentages are not hypothesis-test conclusions and do not establish statistical superiority or causality. Optional notes were preserved verbatim but not systematically coded. Crop geometry may be associated with errors but cannot be claimed to cause them. Class frequency alone cannot be claimed to cause poor performance. Grad-CAM localizes gradient-weighted activation and does not identify a physical defect, prove what caused a prediction, or validate a mechanistic explanation; independently normalized colors cannot be compared quantitatively across maps. The controlled synthetic corruptions do not establish robustness under real-flight or universal deployment conditions.

## Reproduction commands

```powershell
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --apparatus-check
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --validate-only
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --phase9b
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --validate-phase9b
uv run python -m pytest --basetemp .phase9_pytest_tmp -p no:cacheprovider
```

## Status

`PHASE 9 COMPLETE — VALIDATED AND FROZEN`

Phase 9 is complete and frozen. At the Phase 9 freeze, Phase 10 had not started; it subsequently completed without modifying any Phase 9 artifact.
