# Phase 10 — Final Statistical Synthesis and Reproducibility Freeze

## Purpose and scope

Phase 10 consolidates the already-frozen clean evaluation, data-efficiency experiment, controlled-degradation robustness experiment, quantitative error analysis, and human-review synthesis. It creates the canonical paper-ready numerical assets and establishes the final scientific freeze for the technical core of the project.

Phase 10 generated no model prediction and performed no training, fine-tuning, hyperparameter or threshold tuning, calibration, ensembling, SVM/scaler refitting, split or label change, preprocessing or corruption change, dataset download, or external research. Phase 11 localization and Phase 12 external validation were not started.

The Streamlit classifier remains a separate non-scientific demonstration. It classifies a prepared visible-defect crop or a manually selected visible region; it does not automatically detect defects, assess structural safety, or establish deployment readiness.

## Frozen starting gate

Phase 10 began from clean, synchronized `main` at Phase 9 freeze commit `89e18d48ef2dd7c2cef35e9b7a4fa6fe72c92ea9`. Before the Phase 10 apparatus was created, all Phase 3–9 validators passed, both completed human-review forms and the reviewer attestation were present, and no Phase 10/11/12 or YOLO path existed.

The complete Phase 10 upstream inventory contains 3,008 files and has fingerprint `20d917d5cbe08f579382575218a62aa8f0162841197a5d5d98d4eec7e4d0f9a2`. It covers the frozen configurations, split and dataset metadata, predictions and logits, aggregate results, six CNN checkpoints, traditional model binaries when present, Phase 3–9 figures, Phase 9 review artifacts, and the corrected Pass B packet. Validation rehashes every entry.

## Statistical-analysis plan

The plan was frozen in [`configs/final_synthesis.yaml`](../configs/final_synthesis.yaml) before Phase 10 result generation. Apparatus commit `0956dd0`; corrected inventory-path commit `9567862`; scientific-results commit `e8135f6`.

- Primary metric: macro-F1.
- Secondary metrics: accuracy, balanced accuracy, macro precision, macro recall, and per-class precision, recall, and F1.
- Clean comparison unit: the same 162 frozen test instances from 109 source images.
- Pairing: every bootstrap resample uses the same instance positions for all methods and all six predeclared pairwise comparisons.
- Bootstrap: 5,000 paired, nonparametric, true-class-stratified resamples; seed `20260831`; equal-tailed 95% percentile intervals.
- CNN handling: each metric is calculated separately for seeds 17, 29, and 43 within every resample and then averaged. Predictions and logits are not ensembled and no seed is selected.
- Deterministic-method handling: HOG and LBP each retain one frozen prediction vector. They receive sample-bootstrap intervals, while seed SD is reported as `N/A`, never zero.
- Missing values: any missing, nonfinite, duplicate, or inconsistently paired scientific row is rejected.
- Pairwise family: all six clean method-family pairs were declared before result generation.
- P-values: omitted. No p-value family or Holm decision procedure is used. The six confidence intervals are pointwise estimation intervals and are not presented as multiplicity-controlled significance tests.
- Inferential scope: clean-test metric intervals and paired clean macro-F1 differences only. Data-efficiency, robustness, error analysis, human review, and cross-phase trade-offs remain descriptive.

The Phase 10 configuration fingerprint is `0691d82dc6addc200806c26a42382db3ebd70cc5ef1cd6065c306a30a7ed2951`.

## Clean test-set synthesis

CNN entries below are means across all three frozen seeds; seed variation is sample SD. Bootstrap intervals describe paired resampling uncertainty over the fixed test sample structure.

| Method | Macro-F1 | Seed SD | 95% bootstrap CI | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.477988 | N/A | [0.380822, 0.566750] | 0.530864 | 0.457766 |
| LBP + SVM | 0.592401 | N/A | [0.496869, 0.676479] | 0.611111 | 0.572746 |
| ResNet-18 | 0.895314 | 0.014118 | [0.858212, 0.929628] | 0.888889 | 0.902476 |
| MobileNetV3-Small | 0.895321 | 0.005977 | [0.854656, 0.933258] | 0.882716 | 0.903765 |

The highest observed mean clean macro-F1 was MobileNetV3-Small's `0.895321`, only `0.0000063` above ResNet-18. Their paired difference interval was `[-0.036869, 0.035522]`; this near-zero point difference does not establish equivalence or superiority. Both CNN means were higher than the two handcrafted point estimates under these frozen conditions. Complete per-class precision, recall, F1, seed SD, bootstrap intervals, and seed-specific confusion matrices remain machine-readable.

### Paired macro-F1 differences

Differences are second method minus first method.

| Pair | Difference | 95% paired bootstrap CI |
|---|---:|---:|
| LBP − HOG | 0.114414 | [0.024580, 0.209007] |
| ResNet − HOG | 0.417327 | [0.329354, 0.515898] |
| MobileNet − HOG | 0.417333 | [0.325969, 0.519849] |
| ResNet − LBP | 0.302913 | [0.216451, 0.400882] |
| MobileNet − LBP | 0.302919 | [0.218487, 0.403168] |
| MobileNet − ResNet | 0.000006 | [-0.036869, 0.035522] |

These intervals are effect estimates in original macro-F1 units, not declarations of statistical or practical superiority. No domain threshold for practical importance was specified.

## Data-efficiency synthesis

Mean macro-F1 at the fixed 25/50/75/100% source-image budgets was:

| Method | 25% | 50% | 75% | 100% | Normalized learning-curve AUC | First grid point at ≥95% full score |
|---|---:|---:|---:|---:|---:|---:|
| HOG + SVM | 0.334961 | 0.410712 | 0.450523 | 0.477988 | 0.422570 | 100% |
| LBP + SVM | 0.396248 | 0.522589 | 0.548415 | 0.592401 | 0.521776 | 100% |
| ResNet-18 | 0.741756 | 0.855151 | 0.890470 | 0.895314 | 0.854719 | 50% |
| MobileNetV3-Small | 0.723334 | 0.813294 | 0.854272 | 0.895321 | 0.825631 | 75% |

The 95% threshold and normalized trapezoidal learning-curve area were already defined in Phase 7. ResNet had the highest observed normalized learning-curve area and reached 95% of its full-data mean at the 50% grid point; MobileNet first did so at 75%. These four budgets do not establish a general sample-complexity law.

## Robustness synthesis

Across the twelve fixed degraded conditions, mean macro-F1 / mean retention was HOG `0.410073 / 85.79%`, LBP `0.381959 / 64.48%`, ResNet `0.799174 / 89.28%`, and MobileNet `0.814903 / 91.02%`.

Severe retention for blur/resolution/brightness/JPEG was:

| Method | Blur | Resolution | Brightness | JPEG |
|---|---:|---:|---:|---:|
| HOG + SVM | 73.52% | 82.82% | 66.57% | 62.41% |
| LBP + SVM | 34.28% | 29.56% | 44.23% | 36.86% |
| ResNet-18 | 70.83% | 80.82% | 58.72% | 91.90% |
| MobileNetV3-Small | 73.08% | 81.91% | 73.60% | 89.76% |

MobileNet had the highest observed mean performance and retention across the declared degraded grid, while ResNet retained more under severe JPEG. HOG retained a larger fraction of its lower clean score than LBP. The twelve transformations are fixed synthetic design points, not random operational environments; these summaries do not establish universal or real-flight robustness.

## Error analysis and human review

The frozen quantitative synthesis retains clean errors, harmful and beneficial flips, changed/stable wrong predictions, stable-correct cases, CNN seed disagreement, cross-method failure categories, class-level patterns, and the complete selected-case human-review distributions.

- Strict clean consensus found 10 test instances misclassified by all four method families, 22 correct for both CNNs but wrong for both handcrafted methods, and 17 missed by both CNNs.
- Clean ResNet seed patterns contained 7 unanimous failures, 9 majority-only failures, and 26 predicted-label disagreements; MobileNet contained 13, 4, and 19.
- The single reviewer judged 51/60 dataset labels visually plausible, activation inside or partly inside the annotation in 51/60 cases, and seed activation patterns yes/partly consistent in 54/60 cases.
- Prediction understandability after reveal was marked yes in 57/60 cases and uncertain in 3/60. Understandability does not mean correctness or establish a causal mechanism.

Human-review results are post-hoc judgments from one reviewer on a deliberately selected packet. They are not objective ground truth, inter-rater reliability is unavailable, and independently normalized Grad-CAM intensity is never compared quantitatively.

## Cross-phase synthesis

The final synthesis deliberately preserves multiple dimensions instead of producing a weighted composite winner.

| Method | Clean macro-F1 | Mean degraded macro-F1 | Mean degraded retention | Mean severe retention | Fraction at ≥95% full | Learning AUC |
|---|---:|---:|---:|---:|---:|---:|
| HOG + SVM | 0.477988 | 0.410073 | 85.79% | 71.33% | 100% | 0.422570 |
| LBP + SVM | 0.592401 | 0.381959 | 64.48% | 36.23% | 100% | 0.521776 |
| ResNet-18 | 0.895314 | 0.799174 | 89.28% | 75.57% | 50% | 0.854719 |
| MobileNetV3-Small | 0.895321 | 0.814903 | 91.02% | 79.59% | 75% | 0.825631 |

ResNet had the highest observed data-efficiency area and earlier 95%-retention grid point. MobileNet had the highest observed clean mean by a negligible margin, smaller seed SD, and the highest mean degraded performance/retention. No single method dominates every declared dimension, and computational size/latency constraints remain relevant.

## Canonical tables

Every table has matching CSV and JSON representations under [`experiments/summaries/phase10_final_synthesis_v1/tables/`](../experiments/summaries/phase10_final_synthesis_v1/tables/):

1. `experimental_data_summary`
2. `clean_method_comparison`
3. `clean_per_class_performance`
4. `data_efficiency_summary`
5. `robustness_retention_summary`
6. `severe_corruption_summary`
7. `paired_macro_f1_differences`
8. `error_human_review_summary`
9. `cross_phase_tradeoffs`
10. `reproducibility_fingerprints`

## Canonical figures

Figures are under [`figures/phase10/`](../figures/phase10/):

1. `clean_macro_f1_bootstrap_ci.png` — 95% sample-bootstrap intervals.
2. `clean_per_class_f1.png` — frozen point estimates / CNN seed means.
3. `data_efficiency_learning_curves.png` — sample SD across the three declared reduced-data replicates; deterministic full-data SD is not drawn.
4. `robustness_curves.png` — four fixed synthetic degradation families.
5. `severe_retention_heatmap.png` — descriptive severe retention ratios.
6. `severe_per_class_f1.png` — severe condition class patterns.
7. `phase9_human_review_summary.png` — selected single-reviewer distributions.

The machine-readable `figure_registry.json` binds every figure to its caption, source table, and source hash.

## Reproducibility package and lineage

The Phase 10 result directory includes:

- `statistical_plan.json` and resolved configuration fingerprint;
- `bootstrap_indices.csv` (SHA-256 `e20254004fe3be1af1c5a45788cedb967021f7dce8e0aff829f9f29f8eb0f2a6`);
- `upstream_inventory.json` and `upstream_validation.json`;
- `clean_confusion_matrices.json`;
- matching CSV/JSON canonical tables;
- `result_registry.json` and `figure_registry.json`;
- deterministic `runtime.json`, `summary.json`, and `manifest.json`;
- `reproducibility.json` and `validation.json`.

The lineage is explicit: frozen dataset and source-isolated splits → frozen preprocessing → frozen predictions/logits → frozen phase metrics → Phase 10 deterministic derivation → canonical table or figure. The 1,944 ignored robustness PNGs are represented by their tracked checksum manifest and dataset fingerprint rather than copied into Git.

Two complete Phase 10 generations into separate clean temporary locations matched exactly for all 37 fingerprinted scientific files, including bootstrap positions, numbers, intervals, figures, captions/metadata, tables, registries, and inventories. The Phase 10 scientific-output fingerprint is `6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0`.

## Validation and commands

```powershell
uv run python scripts/run_final_synthesis.py --config configs/final_synthesis.yaml --apparatus-check
uv run python scripts/run_final_synthesis.py --config configs/final_synthesis.yaml
uv run python scripts/run_final_synthesis.py --config configs/final_synthesis.yaml --validate-only
uv run python -m pytest tests/test_phase10_final_synthesis.py
uv run python -m pytest
```

The independent validator confirms the exact output inventory and fingerprint, 5,000 stored bootstrap rows with class counts `27/30/33/9/14/49`, table/JSON parity, figure/source parity, all 3,008 upstream hashes, all Phase 3–9 validators, and absence of Phase 11/12 work. The focused Phase 10 suite passes 13 tests; the complete repository suite passes 235 tests with 11 unchanged scikit-learn `SVC(probability=True)` future warnings.

## Limitations and future work

- Three CNN seeds are retained but are insufficient for strong population inference about training randomness.
- Bootstrap intervals characterize resampling of the fixed test instance structure, not uncertainty over new datasets, turbine fleets, domains, or deployments.
- Multiple crops from a source image can be correlated; the split is source-isolated, but the bootstrap unit is the classification instance and does not model a source-cluster hierarchy.
- Pointwise intervals are not multiplicity-controlled hypothesis tests, and no practical-effect threshold was declared.
- WTBD has no valid healthy class and no temporal onset labels.
- Crops are already localized visible surface-defect regions. The study does not automatically detect defects or cover hidden/internal damage.
- Synthetic corruptions do not reproduce the distribution of real drone flights, cameras, weather, lighting, or compound degradation.
- Human review has one reviewer and cannot support inter-rater reliability or objective-mechanism claims.
- No external dataset, real-flight validation, structural-safety assessment, or deployment validation was performed.

Optional future work would require a separately authorized Phase 11 localization study and/or Phase 12 external validation. Real-world deployment would additionally require representative operational data, automatic localization, domain validation, monitoring, and an independent safety assessment.

## Final scientific freeze

Result version: `phase10_final_synthesis_v1`. The content freeze is established by scientific-results commit `e8135f6`; the final documentation/push commit is recorded in subsequent Git history. Configuration fingerprint: `0691d82dc6addc200806c26a42382db3ebd70cc5ef1cd6065c306a30a7ed2951`. Scientific-output fingerprint: `6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0`.

Documentation-only corrections may be made only if numbers, statistical definitions, scientific outputs, and conclusions remain unchanged. Any change to splits, labels, preprocessing, predictions, metrics, the statistical plan, or a scientific output requires a new version and a new documented freeze.

`PHASE 10 COMPLETE — VALIDATED AND FROZEN`

The core technical research project is complete. Phases 11 and 12 have not been started.
