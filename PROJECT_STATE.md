# Project State

## Current phase

- **Phase:** Application v2 productization on frozen assets
- **Status:** COMPLETE AND VALIDATED; AUTOMATIC LOCALIZATION UNAVAILABLE
- **Start date:** 2026-08-31
- **Completion date:** 2026-08-31
- **Previous phases:** Phases 0–10 — complete and frozen
- **Next phase:** Phase 11B training on a suitable pinned CUDA environment; Phase 12 not started

Phases 0–10 remain complete and frozen. Phase 11A is complete and frozen. Phase 11B training is compute/dependency-blocked and unstarted. Application v2 productizes only the current frozen classifier and summary assets; it does not alter scientific behavior or integrate a detector. Phase 12 has not started.

## Application v2 — frozen-asset productization

**COMPLETE AND VALIDATED; AUTOMATIC LOCALIZATION UNAVAILABLE.** The initial interface-refresh commit is `1f6a6fe`; Application v2 implementation/validation commits are recorded in subsequent history.

- Gate: Phase 11A application readiness is `unsupported`; Phase 11B has no detector dependency, license record, checkpoint, validation-selected threshold, NMS configuration, predictions, metrics, or reproducible inference. Phase 11C automatic localization was therefore not entered.
- Navigation: Home, Analyze Image, Compare Regions, Research Results, Detection Readiness, and About and Limitations.
- Active modes: prepared crop, manual single region, and manual multi-region. Multi-region work supports stable IDs, overlap, replace, remove, clear, and new-image actions.
- Session product features: thumbnails, score comparison, selected-region Grad-CAM, and in-memory JSON/CSV/annotated-PNG downloads. Session data and uploads are not persisted.
- Evidence dashboards: verified read-only Phase 10 canonical tables and Phase 11A audit/readiness outputs; no metric recomputation or hard-coded substitute values.
- Future detector contract: explicit unavailable exception and typed box/result boundary; no detector dependency, weights, predictions, or fabricated boxes.
- Scientific behavior: Phase 3 pixel parity, Phase 6 checkpoint/state/reference logits, six scores, CPU inference, and Grad-CAM prediction/parameter invariance remain unchanged. Phase 8–11A scientific outputs remain unchanged.
- Frozen classifier checkpoint SHA-256 `9c7a5f18e7d05a320e1296c73bbeb9366636e0e55dc7c6ff2bab6d8808a0e5a5`; state fingerprint `3c17629d1b1748e2f3d9046cb9a3d88c6369786acc1381f105974396c0f46757`. Detector checkpoint, threshold, NMS, detector latency, and detector parity: `N/A`.
- Privacy: local in-memory uploads, telemetry disabled, no persistent upload writes, no external service/API, no tracked upload, and no global caching of user images.
- Dependencies remain Streamlit `1.62.0` and streamlit-cropper `0.3.1`; no detector runtime or detector-license implication was added.
- Validation: focused Application v2 suite **37 passed**; complete repository suite **266 passed** with 11 unchanged scikit-learn future warnings; all read-only Phase 2–11A/app/review validators pass.
- Local loopback health: HTTP 200 / `ok`, then the server was stopped. No external deployment occurred.
- Full record: `docs/app.md`; machine validation: `app/validation/validation.json`.

## Phase 11A — full-image detection feasibility and annotation audit

**COMPLETE AND FROZEN; PHASE 11B BLOCKED.** Audit-apparatus commit `39cf4c7`; initial generated audit commit `956975c`; the completed reporting/final-state commit is recorded in subsequent history.

- Phase 10 immutability: PASS. The 3,008-file Phase 3–9 inventory, every Phase 10 scientific output/figure, and Phase 10 fingerprint `6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0` remain unchanged. The Streamlit application inventory was unchanged at the Phase 11A freeze; later Application v2 productization is separately validated.
- Phase 9A fingerprint transition is explained and validated: `a5938ec...` was superseded by `14e500...` only because 177 incorrect true-class display captions were corrected; arrays, predictions, checkpoints, labels, selected cases, and conclusions did not change.
- Authoritative source: WTBD v1 primary PASCAL VOC XML and full JPEG images; one-based inclusive coordinates. Raw release: 1,065 images/1,584 boxes. Frozen curated detection set: 720 images/1,065 boxes.
- Detection split: 510/101/109 images and 757/146/162 boxes in train/validation/test. All boxes remain with their source image; zero exact, confirmed-scene, or unresolved duplicate group crosses a split.
- Class boxes: craze 169, corrosion 178, surface_injure 264, thunderstrike 60, crack 131, hide_craze 263. Every class occurs in every split.
- Annotation gate: zero invalid boxes, duplicate annotations, orphan retained images/annotations, or retained exact-image groups. Diagnostic counts retained without repair: 162 boxes below 1% area, 9 above 50%, 33 with aspect ratio outside 0.10–10, and 24 nonzero-IoU within-image pairs.
- Background/healthy images: zero. Class-agnostic localization and six-class detection are supported only with explicit limitations; healthy false-positive evaluation, arbitrary inspection claims, and app integration are unsupported.
- Dataset fingerprint `ad4ab59c3e3c85c6cf0b85b148177bd6b79d24f372f49bdff0043609e6fefc97`; split fingerprint `264f8460f203074374c2c098c8fd5d2e55fb7ee1f281a8d505e2dfb0de9a2bc3`; configuration fingerprint `9f4a20ba4404c9a6072277a504c466a0756143b908e79c7168d2ccf91ff32057`; scientific-output fingerprint `3f46cbdc6c7a2e3cf6093ff177dd1948d113fa4c36fa9eb907d7c8621e800461`.
- Exact two-pass audit/conversion/QC generation: PASS. QC packet: 26 deterministically selected, authoritatively boxed full images.
- Validation: Phase 3–10 chain, Phase 11A validator, review-interface validator, and classifier-app validator all PASS. Focused Phase 11A suite: **21 passed, 0 failed**. Complete repository suite: **256 passed, 0 failed**, with 11 unchanged scikit-learn `SVC(probability=True)` future warnings.
- Compute gate: AMD64 Family 23 Model 24, 8 logical CPUs, 13.95 GiB RAM, Python 3.11.15, PyTorch 2.13.0+cpu, CUDA devices 0, VRAM 0. Multi-seed CPU training is prohibited.
- No detector dependency or pretrained weight was acquired; no detector version/checkpoint/prediction/metric exists. All Phase 11B metrics and hashes are `N/A`, not zero. A new pre-test apparatus commit on suitable CUDA hardware must pin the exact compact/nano detector package, license, weight source/hash, and resolved configuration before training.
- Automatic-localization readiness: **not eligible for integration**. Application v2 reads the frozen readiness evidence but every analysis path still requires a user-supplied crop or rectangle. Phase 12 has not started.
- Results: `experiments/summaries/phase11_detection_audit_v1/`; full record: `docs/phase11_detection.md`.

## Phase 10 — Final statistical synthesis and reproducibility freeze

**COMPLETE, VALIDATED, AND FROZEN.** Statistical-plan/apparatus commit `0956dd0`; inventory-path correction commit `9567862`; scientific-results commit `e8135f6`. The final documentation/push commit is recorded in subsequent repository history.

- Frozen starting point: Phase 9 freeze commit `89e18d48ef2dd7c2cef35e9b7a4fa6fe72c92ea9`, clean and synchronized before Phase 10 apparatus creation.
- Statistical plan: macro-F1 primary; accuracy, balanced accuracy, macro precision/recall, and per-class precision/recall/F1 secondary; 162 paired test instances; 5,000 true-class-stratified percentile-bootstrap resamples; seed `20260831`; 95% pointwise intervals; all three CNN seeds averaged at the metric level within every resample; HOG/LBP seed SD `N/A`; six predeclared clean pairs; no p-values or multiplicity-controlled significance claims.
- Configuration fingerprint: `0691d82dc6addc200806c26a42382db3ebd70cc5ef1cd6065c306a30a7ed2951`.
- Complete frozen Phase 3–9 upstream inventory: 3,008 files, fingerprint `20d917d5cbe08f579382575218a62aa8f0162841197a5d5d98d4eec7e4d0f9a2`.
- Clean macro-F1: HOG `0.477988`, LBP `0.592401`, ResNet `0.895314 ± 0.014118`, MobileNet `0.895321 ± 0.005977`. ResNet and MobileNet differ by only `0.0000063`; paired MobileNet-minus-ResNet 95% interval `[-0.036869, 0.035522]`.
- Data efficiency: normalized learning-curve AUC HOG `0.422570`, LBP `0.521776`, ResNet `0.854719`, MobileNet `0.825631`; the fixed grid first reached 95% of full-data performance at 100/100/50/75%, respectively.
- Robustness: mean degraded-condition macro-F1/retention HOG `0.410073/85.79%`, LBP `0.381959/64.48%`, ResNet `0.799174/89.28%`, MobileNet `0.814903/91.02%`. These are descriptive summaries of twelve fixed synthetic design points.
- Phase 9 synthesis remains descriptive: 10 strict-consensus clean failures shared by all four families; single-reviewer label plausibility 51/60; activation inside/partial 51/60; seed pattern yes/partly 54/60.
- Canonical assets: ten tables with exact CSV/JSON counterparts, seven figures with source hashes/captions, seed-specific and mean confusion matrices, stored bootstrap positions, a complete result registry, runtime record, upstream inventory/validation, and reproducibility manifest.
- Exact two-pass reproduction: **PASS** for 37 scientific files. Bootstrap-index SHA-256 `e20254004fe3be1af1c5a45788cedb967021f7dce8e0aff829f9f29f8eb0f2a6`. Phase 10 output fingerprint `6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0`.
- All Phase 3–10 validators pass; table/JSON consistency, figure/source consistency, formula checks, stored class stratification, generated inventory, and upstream immutability pass. Focused Phase 10 tests: **13 passed, 0 failed**. Complete repository suite: **235 passed, 0 failed**, with 11 unchanged scikit-learn `SVC(probability=True)` future warnings.
- Predictions, logits, checkpoints, splits, labels, mappings, review inputs, and prior Phase 3–9 scientific outputs remained unchanged. Training, tuning, refitting, downloading, external research, automatic localization, Phase 11, and Phase 12 work: zero.
- Full record: `docs/phase10_final_synthesis.md`; results: `experiments/summaries/phase10_final_synthesis_v1/`; figures: `figures/phase10/`.

## Phase 9B — Human-review incorporation and descriptive synthesis

**COMPLETE, VALIDATED, AND FROZEN.** Apparatus/correction commit `29ca2fe`; human-review synthesis commit `b4e6727`. The final documentation/freeze commit is recorded in repository history after this state file.

- The reviewer attested exactly: “I completed and saved every Pass A judgment before opening or completing Pass B. Both forms contain my own visual judgments. No review fields were filled automatically.”
- Pass A is complete at `300/300`, with 30 notes and SHA-256 `3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e`. Pass B is complete at `240/240`, with 60 notes and SHA-256 `0f5258e06a4e854d338705bcf1d38ced048f0652a99ccc4639b18c3baae1cd96`. Both source forms were incorporated byte-for-byte without automated entries or scientific-output normalization.
- The anonymous-ID mapping is complete, one-to-one, order-preserving, and bound by SHA-256 `46b19248797997e8aa7236b9c3fbf17f972977f2cdb2c3a958bd12513f25210b`.
- The corrected Pass B packet is the sole packet provenance: ZIP SHA-256 `fbfe1985e8a1809884c0df57a0e9a8eb265815988ad66583a8d9c5a3e84dcaf5`, corrected index SHA-256 `ed8f3ef20449603e8e55647f3d917dbefbf2803d05ed696532859dde92201f29`, and corrected blank-form SHA-256 `7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8`. All 1,680 non-form packet files match the repository. The superseded caption-bugged packet and the AI comparison workbook were not used in the scientific analysis.
- Deterministic outputs contain 60 joined review cases, 180 joined seed-prediction rows, 36 response-summary rows, 1,152 case cross-tab rows, 2,080 prediction cross-tab rows, and five descriptive figures.
- Pass A responses indicate a visible or partially visible defect in all 60 selected cases; 51/60 labels were judged plausible. Pass B responses place activation inside or partially inside the annotated defect in 51/60 cases, with seed consistency marked yes or partly in 54/60 cases. These are single-reviewer, selected-case descriptive counts only; they are not causal, inferential, or population-level claims.
- Phase 9B config fingerprint: `ac60d00ffb9b0b729c334a9b5b76bf0f3630c09fd7e5b78b7f304f3048263830`. Scientific-output fingerprint: `dc940fe0a802d285b236224b04197f7fb7d6b0ef73062a5bdddbb1238de286d3`.
- Two independent temporary generations matched exactly for all 15 scientific derived files. The validator also confirmed 113 frozen upstream input files, 1,971 Phase 9A non-form files, all six checkpoint state identities, 104 prediction sets, and 507 Grad-CAM records with zero mutation.
- Phase 3–8 preflight validators, Phase 9A validation, Phase 9B validation, the review-interface validator, and the classifier-app validator all pass. Focused Phase 9/review tests: **46 passed, 0 failed**. Complete repository suite: **222 passed, 0 failed**, with 11 unchanged scikit-learn `SVC(probability=True)` future warnings.
- No training, fine-tuning, SVM/scaler refitting, checkpoint selection, model interpretation automation, scientific use of AI-authored comparison material, dataset download, or Phase 10 work occurred.

## Separate non-scientific human-review interface

**Status: COMPLETE AND VALIDATED; historical data-entry interface only.** The local Streamlit review interface was the human-data-entry aid for the Phase 9A packet. It is not a scientific experiment and did not itself incorporate the review or start Phase 10.

- Entry point: `app/review_app.py`; operating guide: `docs/human_review_interface.md`; reusable isolated support: `src/windblade_review/`; machine-readable validation: `app/validation/review_interface_validation.json`.
- Launch: `uv run streamlit run app/review_app.py --server.address 127.0.0.1`.
- The public classifier demonstration remains separate at `app/app.py`; the review interface does not import or call its prediction code.
- Pass A is one-case-at-a-time and exposes only the existing anonymous review ID, dataset true label, review images, and annotation visualization. It does not read or expose Pass B, the separate ID mapping, source IDs, predictions, model/seed identity, event information, or Grad-CAM before the gate.
- Pass B requires all 300 valid Pass A responses, explicit attestation, **Validate and lock Pass A**, and a separate deliberate **Begin Pass B** action. The lock makes Pass A read-only in that interface session and displays its completed SHA-256.
- Valid changes autosave to the current pass through a validated, same-directory temporary file and atomic replacement. Exact headers, review IDs, row order, response enums, notes, and the other pass are preserved. Restart resumes at the first incomplete case.
- The interface performs no model or LLM inference, applies no heuristic or automated rule, supplies no judgments or answer suggestions, makes no external service call, and stores only the required CSV responses plus transient in-memory state.
- Validation used temporary copied forms only. Focused review-interface suite: **22 passed, 0 failed**. Complete repository suite: **213 passed, 0 failed**, with 11 unchanged scikit-learn deprecation warnings. Phase 9A and classifier-app validators remain `PASS`; local-only live health returned HTTP 200/`ok` and the server was stopped.
- At the interface-validation commit, both canonical forms were blank. They are now complete and frozen: Pass A SHA-256 `3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e`; Pass B SHA-256 `0f5258e06a4e854d338705bcf1d38ced048f0652a99ccc4639b18c3baae1cd96`.
- The interface changed no mapping entry, packet image/HTML, checkpoint, prediction, quantitative output, or other scientific artifact. Phase 9B subsequently incorporated the returned forms through the separately validated deterministic pipeline. At this Phase 9B completion point, Phase 10 was unstarted; it subsequently completed without changing any interface or Phase 9 artifact.

## Separate non-scientific Application v2

**Status: COMPLETE AND VALIDATED; does not change the scientific phase.** The local Streamlit application is an engineering demonstration of the already frozen classifier, not a scientific experiment or a Phase 10 deliverable.

- Entry point: `app/app.py`; operating record: `docs/app.md`; app-only validation: `app/validation/`.
- Frozen model: Phase 6 full-data MobileNetV3-Small seed 17, used because 17 is the predeclared canonical seed rather than because of a best-result comparison.
- Input modes: prepared crop, larger image with one manually selected region, or larger image with multiple manually selected regions.
- The manual workflow reuses the exact Phase 3 contextual-square geometry and the same RGB/bilinear 224×224 pixel policy; regression tests confirm pixel identity against multiple stored Phase 3 references.
- App output: one of the six frozen WTBD categories plus all six model scores, explicitly labeled as not calibrated confidence estimates.
- Optional Grad-CAM imports the unchanged Phase 9A primitive; model-state and prediction invariance pass.
- App dependencies are pinned separately in `requirements-app.txt`; scientific dependency constraints are unchanged.
- Product features: session-only comparison, stable region IDs, optional Grad-CAM, in-memory JSON/CSV/annotated-PNG exports, frozen Phase 10 research dashboard, and frozen Phase 11A readiness dashboard.
- Validation: all three UI workflows, pure session/export contracts, canonical dashboard sources, classifier invariance, and automatic-localization gate pass. Final suite and live-health results are recorded in the current Application v2 section.
- Training, fine-tuning, calibration, ensembling, test-driven model/seed selection, new test-set evaluation, checkpoint mutation, external service calls, permanent upload storage, and tracked uploaded images: zero.
- Scientific status remains **PHASE 10 COMPLETE — VALIDATED AND FROZEN** with **PHASE 11A COMPLETE AND FROZEN**. Phase 11B is blocked and unstarted; Phase 12 has not started.

The application classifies a manually identified visible surface-defect region. It does not automatically detect or localize defects, assess blade safety or condition, detect hidden/internal damage, estimate severity or remaining life, replace inspection professionals, or establish real-time target-hardware deployment.

## Phase 9A — Quantitative Error Analysis, Grad-CAM, and Blinded Human Review

### Status and frozen identity

**COMPLETE AND FROZEN AS THE PHASE 9A COMPONENT OF PHASE 9.** Apparatus commit `9758f139d5c35fab3d13d955ce5457fa843b0795`; original generated-output commit `95e1bd7`; caption-audit and Phase 9B apparatus commit `29ca2fe`. The required human judgments were subsequently returned, validated, and incorporated in Phase 9B.

- Analysis-config fingerprint: `9a9c87f132359718636f46d1b9061e8e5539980b6c9b67a3f5336834ce1c9f1e`.
- Frozen-input fingerprint: `dbedb680481e6477b94ff623858e1db4c1981f2b8f5491d0e04a99b5672173b1`.
- Scientific-output fingerprint: `14e500fd94fa871bbe1e6bee6494d3158fe003c3799e0ccfbcab8e549adf80fa` after the caption-only Pass B correction; supersedes `a5938ec22a0f496c6fd9ce3acd7d999cad56e8c13a85d9be4b4d59860aecd74b`.
- Upstream fingerprints remain Phase 3 `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`, Phase 8 config `a6a9e40c9c3de7130892df3cf49698f95718ce9241c8a963849016ce7adc1d57`, and Phase 8 robustness data `da0eda8956adfef63e001d0d1614279a907c46feffc5e2f180c5acdbe89987db`.
- Canonical error manifest: 16,848 rows covering every fixed sample and all 104 method/seed-condition prediction sets.
- Review set: 60 unique sample-condition cases with all six event quotas met and no shortfall.
- Grad-CAM: 507 finite evidence records; ResNet `layer4.1` (`1×512×7×7`) and MobileNet `features.12` (`1×576×7×7`); parameters unchanged for all six checkpoint identities.
- Review packet: two linked HTML passes, ten contact sheets, two complete canonical CSV forms, and a separately stored anonymous-ID mapping. The corrected Pass B packet was used for the completed review.
- Training, fine-tuning, SVM/scaler refitting, checkpoint/prediction mutation, automated visual interpretation, and Phase 10/app work: zero.

### Grad-CAM reporting audit before Pass B review

- A Pass B caption defect repeated `predicted_label` on `true_class` captions. The underlying target indices, metadata, arrays, heatmaps, and overlays were correct.
- All 330 true-class target rows match frozen ground truth; all 177 predicted-class target rows match their corresponding frozen predictions. Zero target identity mismatches were found.
- Independent read-only recomputation reproduced all 507 Grad-CAM array hashes, activation shapes, and prediction-logit argmax values; all six checkpoint state fingerprints remained unchanged.
- The 177 affected true-class captions across 52 review cases were corrected. Grad-CAM array/figure fingerprint `84e0dd6c21a6ff0cd92602f83b99e6b0e08c2d175cdc5d10c6c6602abf20c321` is unchanged.
- Completed Pass A is preserved byte-for-byte: `300/300`, SHA-256 `3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e`.
- Four early Pass B answers entered from the defective page were invalidated before the corrected review. The corrected blank form had SHA-256 `7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8`; the later completed corrected form has SHA-256 `0f5258e06a4e854d338705bcf1d38ced048f0652a99ccc4639b18c3baae1cd96`.
- Validator assertions bind both target roles to frozen class IDs/labels and validate every Pass B caption. Phase 9B incorporated the completed corrected review and is now complete.
- Full audit: `docs/phase9a_gradcam_reporting_audit.md`; machine-readable record: `experiments/audits/phase9a_gradcam_reporting_audit.json`.

### Quantitative descriptive highlights

- Lowest clean recall: thunderstrike for HOG (`0.2222`) and LBP (`0.3333`); corrosion for ResNet (`0.8000` mean) and MobileNet (`0.7556` mean).
- Largest mean severe harmful-flip condition: JPEG for HOG (28), resolution for LBP (64), brightness for ResNet (`53.33 ± 9.29`) and MobileNet (`37.67 ± 4.04`).
- Clean unanimous/majority-only/label-disagreement counts: ResNet 7/9/26; MobileNet 13/4/19.
- Four severe conditions combined (648 sample-condition records): ResNet 97 unanimous failures, 78 majority-only failures, and 253 label-disagreement records; MobileNet 100, 67, and 177.
- Strict clean cross-method rule: 10 samples missed by all four families, 22 correct for both CNNs but wrong for both handcrafted methods, 17 missed by both CNNs, and 38 one-family-only failures.

All findings are post-hoc and descriptive. No geometry association is causal, and no visual or Grad-CAM explanation has been asserted.

### Reproducibility and evidence

- Phase 3–8 preflight validators: PASS.
- Commit `681ed81` audit: presentation-only Phase 8 report correction; predictions, CNN logits, corrupted images, labels, checkpoints, and headline macro-F1 values unchanged.
- Two complete Phase 9A generations: exact equality across 1,973 compared files and output fingerprint.
- Independent Phase 9A validator: PASS.
- Expanded suite: **164 passed, 0 failed**, with 11 unchanged scikit-learn deprecation warnings.
- Machine-readable results and packet: `experiments/summaries/phase9_error_analysis_v1/`.
- Grad-CAM and quantitative figures: `figures/phase9/`.
- Full record: `docs/phase9_error_analysis.md`.

### Human-review completion

1. Pass A was completed and saved before Pass B or the separate mapping was opened.
2. Pass A was locked after all 300 fields were valid, and Pass B was then completed from the corrected packet.
3. Both forms contain the reviewer's own visual judgments; no review field was filled automatically.
4. Phase 9B validated and incorporated both immutable forms. Phase 10 was not begun.

## Phase 8 — Controlled Image-Degradation Robustness

### Status and frozen identity

**COMPLETE.** Result `phase8_robustness_v1`; apparatus commit `b222435`; scientific-results commit `e9e7b07`. The Phase 3 fingerprint remains `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.

- Corruption config fingerprint: `a6a9e40c9c3de7130892df3cf49698f95718ce9241c8a963849016ce7adc1d57`.
- Robustness dataset fingerprint: `da0eda8956adfef63e001d0d1614279a907c46feffc5e2f180c5acdbe89987db`.
- Environment: Pillow 12.3.0; JPEG support enabled; JPEG library 8.0.
- Dataset: 162 fixed test references, 12 independently generated degraded conditions, exactly 1,944 224 × 224 RGB PNGs.
- Parameters: blur radii 0.75/1.5/3.0; bilinear resolution 168/112/56 → 224; brightness factors 0.75/0.50/0.25; JPEG quality 75/50/25 with subsampling 2, optimize false, progressive false.
- Clean reproduction: PASS for HOG, LBP, and ResNet/MobileNet seeds 17/29/43; all predictions and metrics exact, and all stored CNN logits exact.
- Training/fine-tuning/refitting: zero.

Macro-F1 by clean/mild/moderate/severe:

| Family | HOG + SVM | LBP + SVM | ResNet-18 mean ± SD | MobileNetV3-Small mean ± SD |
|---|---|---|---|---|
| Blur | 0.477988 / 0.454351 / 0.450664 / 0.351435 | 0.592401 / 0.555030 / 0.371985 / 0.203079 | 0.895314±0.014118 / 0.891084±0.020693 / 0.822025±0.018044 / 0.633788±0.032868 | 0.895321±0.005977 / 0.878574±0.022958 / 0.843774±0.014293 / 0.654345±0.041871 |
| Resolution | 0.477988 / 0.467346 / 0.465164 / 0.395885 | 0.592401 / 0.530885 / 0.364424 / 0.175100 | 0.895314±0.014118 / 0.876557±0.021513 / 0.853611±0.027808 / 0.723203±0.025957 | 0.895321±0.005977 / 0.879019±0.019438 / 0.861577±0.013832 / 0.733372±0.016645 |
| Brightness | 0.477988 / 0.425921 / 0.399174 / 0.318186 | 0.592401 / 0.561298 / 0.444338 / 0.262047 | 0.895314±0.014118 / 0.882760±0.018941 / 0.809782±0.000065 / 0.526303±0.072731 | 0.895321±0.005977 / 0.878433±0.016557 / 0.858144±0.005571 / 0.659088±0.048045 |
| JPEG | 0.477988 / 0.439939 / 0.454484 / 0.298325 | 0.592401 / 0.512283 / 0.384648 / 0.218386 | 0.895314±0.014118 / 0.885217±0.013598 / 0.863319±0.003985 / 0.822440±0.020611 | 0.895321±0.005977 / 0.872486±0.006569 / 0.856341±0.011473 / 0.803678±0.030335 |

Severe retention for blur/resolution/brightness/JPEG is HOG 73.52/82.82/66.57/62.41%, LBP 34.28/29.56/44.23/36.86%, ResNet 70.83/80.82/58.72/91.90%, and MobileNet 73.08/81.91/73.60/89.76%.

Mean degraded-condition macro-F1/retention is HOG 0.410073/85.79%, LBP 0.381959/64.48%, ResNet 0.799174/89.28%, and MobileNet 0.814903/91.02%.

Severe prediction-flip rates for blur/resolution/brightness/JPEG are HOG 41.36/29.01/29.01/33.33%, LBP 57.41/61.73/51.23/50.62%, ResNet 33.13/24.49/40.12/15.23%, and MobileNet 27.57/22.22/31.48/14.61%. Harmful and beneficial counts, including exact CNN per-seed values, are in the Phase 8 aggregate tables.

Class-level descriptive findings: severe HOG/LBP thunderstrike F1 is 0 across all four families; small support constrains interpretation. For ResNet, severe brightness has the lowest thunderstrike/crack means (0.376/0.376), while severe JPEG class means remain 0.736–0.886. For MobileNet, severe blur craze is 0.478 and severe brightness thunderstrike is 0.493; severe JPEG class means remain 0.722–0.864. Detailed qualitative interpretation is reserved for Phase 9.

### Reproducibility and evidence

- Full deterministic regeneration: PASS across two complete runs.
- Regenerated PNG hashes: 1,944/1,944 exact.
- Predictions, CNN logits, metrics, scientific file set, aggregate tables, flip analyses, and both fingerprints: exact.
- Independent Phase 8 result validator: PASS.
- Expanded suite: **149 passed, 0 failed**, with 11 unchanged scikit-learn deprecation warnings.
- Machine-readable results: `experiments/summaries/phase8_robustness_v1/` (483 files including the reproduction record).
- Dataset metadata: `data/processed/wtbd_robustness_v1/`; corrupted PNG payload ignored.
- Figures: `figures/phase8/` (10 main figures, 20 severe/clean mean normalized confusion matrices, and four training-only QC sheets).
- Full record: `docs/phase8_robustness.md`.

### Exit gate

- [x] Phase 7 passed before Phase 8 began; Phases 0–7 remain frozen.
- [x] Phase 3 fingerprint, instance counts, and source-isolated test membership are unchanged.
- [x] All eight frozen model artifacts exist and reproduce clean outputs exactly.
- [x] No model training, fine-tuning, SVM refit, or scaler refit occurred.
- [x] Exactly twelve single-family degraded conditions and 1,944 common-pixel PNGs exist.
- [x] Parameters match the pre-science config; each corruption derives independently from clean.
- [x] Macro-F1, secondary metrics, per-class metrics, drops, retention, flips, transitions, predictions, logits, and matrices exist.
- [x] Required figures and training-only QC sheets exist.
- [x] Full corruption regeneration and evaluation reproduction pass exactly.
- [x] Full pytest and independent result validation pass.
- [x] Phase 9 has not started.

## Phase 7 — Data-Efficiency / Limited-Labeled-Data Experiment

### Status and frozen identity

**COMPLETE.** Result `phase7_data_efficiency_v1`; apparatus commits `cd7f80a` and pre-science gate fix `1ae074f`; result-artifact commit `b67d1f9`. The Phase 3 fingerprint remains `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.

- Primary reduced fits: 36 total — nine each for HOG, LBP, ResNet-18, and MobileNetV3-Small.
- New CNN trainings: 18; primary reduced-data test evaluations: 36.
- Full-data training runs: zero. The Phase 4/5/6 100% anchors were validated and reused.
- Validation/test sizes remained 146/162 instances for every run.
- Subset/model seeds remained paired as 17/29/43; all fractions started independently.
- Same-seed initial CNN head fingerprints match across 25%/50%/75% for each architecture.
- No Phase 7 hyperparameter search, augmentation, inference retiming, or Phase 8 corruption work occurred.

Frozen configuration SHA-256 identities:

- HOG: `2054b1605590aa34a56ab7808b2f95d589d337e4fb651048cac750fa8618de24`.
- LBP: `24ff9776eacd0e26bb61a3bd13f3c840772ee5b46a5f3ac745ac164084ca9cc9`.
- ResNet-18: `6ac9e437f026a5c30433ccc9a670496b75ecab7619b0fb7266ab395c0093c2ad`.
- MobileNetV3-Small: `62d610e150c950a42641f726134ae17227cdac875139828f346c23e248cce87e`.

Subset fingerprints (25/50/75/100%):

- Seed 17: `b231a06c1e90ce97090359e861df054c98e1084c46288571424851b37ab606dd`, `ec9f49a5532a62afac49942d45fe02c214442d944a173a4954152d2bdb9c3a7f`, `bd8d15b8214c52378f83cdf666cd46fc0411272fcc1967654d3c0280a47cb496`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.
- Seed 29: `a2b427a6a682de42ef29280042c0ffbe862f80963b13dbb15aa5bfabf77db10c`, `b6f5222161c60af4373413caf49409b12f0e29720b6dd035ee2e4aad2e620f8c`, `3f2a2f67ca829e008d12e95f3526d8595aff0a61e2ae99a9acb2150a215fc21d`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.
- Seed 43: `3b43aaefb03318b3b5349ffd81626b0f15dd1fc3eb7e3d40523a0e1cf6580f33`, `5a66073adc48f8aa57c7c7711a86676540adfa5a6ee5e59d981d5cc874632a19`, `6ee68e91f499b1e011ebbe17a97cf3f1bc8e314ba28de0331156ad7f17560005`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.

### Primary results

Test macro-F1 mean ± sample SD by labeled-source fraction:

| Method | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| HOG + SVM | 0.334961 ± 0.004791 | 0.410712 ± 0.021288 | 0.450523 ± 0.011041 | 0.477988 (SD N/A) |
| LBP + SVM | 0.396248 ± 0.014417 | 0.522589 ± 0.012073 | 0.548415 ± 0.013633 | 0.592401 (SD N/A) |
| ResNet-18 | 0.741756 ± 0.021156 | 0.855151 ± 0.030380 | 0.890470 ± 0.006729 | 0.895314 ± 0.014118 |
| MobileNetV3-Small | 0.723334 ± 0.017074 | 0.813294 ± 0.017598 | 0.854272 ± 0.004630 | 0.895321 ± 0.005977 |

Retention at 25/50/75% is HOG 70.08/85.93/94.25%, LBP 66.89/88.22/92.57%, ResNet 82.85/95.51/99.46%, and MobileNet 80.79/90.84/95.42%. Marginal macro-F1 gains for 25→50/50→75/75→100 are HOG 0.075750/0.039811/0.027465, LBP 0.126341/0.025827/0.043986, ResNet 0.113395/0.035319/0.004844, and MobileNet 0.089960/0.040978/0.041049.

The smallest tested fraction reaching 95% of full-data macro-F1 is 100% for HOG, 100% for LBP, 50% for ResNet, and 75% for MobileNet. Normalized macro-F1 learning-curve AUC is 0.422570/0.521776/0.854719/0.825631 for HOG/LBP/ResNet/MobileNet.

Thunderstrike has 12/22/32/42 training examples. Its F1 curves are HOG 0.200/0.255/0.200/0.364, LBP 0.200/0.309/0.442/0.500, ResNet 0.693/0.896/0.967/0.947, and MobileNet 0.727/0.912/0.928/0.965. Small support limits interpretation. Corrosion and surface-injury results remain among the more difficult CNN class-level outcomes, and several per-class curves are non-monotonic.

### Compute, reproducibility, and evidence

- Recorded primary reduced CNN training: ResNet 13,896.5 seconds (3.86 h); MobileNet 4,181.1 seconds (1.16 h); total 18,077.6 seconds (5.02 h).
- Canonical ResNet seed-17/25% rerun: PASS for initial head, best epoch, scientific history, validation/test predictions, metrics, and checkpoint fingerprint.
- Canonical MobileNet seed-17/25% rerun: PASS for the same exact comparisons. Timings were excluded.
- Expanded final test suite: **134 passed, 0 failed**, with 11 unchanged scikit-learn deprecation warnings.
- Versioned results: `experiments/summaries/phase7_data_efficiency_v1/`.
- Figures: `figures/phase7/` (12 required figures).
- Full record: `docs/phase7_data_efficiency.md`.

### Exit gate

- [x] Phases 3–6 and all 12 frozen subset manifests validate.
- [x] Exact source, instance, class, validation, and test counts reproduce.
- [x] Frozen HOG/LBP/ResNet/MobileNet configurations were used with no search.
- [x] Reduced CNN weights use active training labels only; every run starts fresh from official pretrained weights.
- [x] Nine reduced fits per method completed; no seed was removed.
- [x] Every primary reduced run received one test evaluation; complete predictions, CNN logits, class metrics, histories, and checkpoint metadata exist.
- [x] Frozen 100% results were reused and detailed inference timing was not repeated.
- [x] Learning curves, retention, absolute loss, marginal gains, tested 95% threshold, normalized learning-curve AUC, per-class analysis, and thunderstrike analysis exist.
- [x] Both canonical CNN reproducibility checks pass.
- [x] Phase 8 has not started.

## Phase 6 — MobileNetV3-Small Transfer-Learning Baseline

**COMPLETE.** Result `phase6_mobilenet_v3_small_v1`; apparatus commit `e5e7be4`. Dataset fingerprint remains `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`. Official pretrained fingerprint: `982065f8b6dec87a6c0d70a9a0b132bc976615c1f2d6a4fa2d93bc060caaf1d1`. PyTorch/torchvision: `2.13.0+cpu`/`0.28.0`; CPU; 1,524,006 trainable parameters.

Frozen LR/WD: `0.0001`/`0`; winning validation macro-F1 `0.873412`. Final best epochs: 17/17/27. Test macro-F1: 0.901740/0.894307/0.889915; aggregate `0.895321 ± 0.005977`. Balanced accuracy: `0.903765 ± 0.004203`; accuracy: `0.882716 ± 0.006173`. Checkpoint size: 6,219,239 bytes; CPU latency: 22.067 ms median and 33.990 ms p95.

Relative to frozen ResNet: 86.37% fewer parameters, 86.12% smaller checkpoint, 79.23% lower median latency, and 100.0007% descriptive macro-F1 retention. Exact seed-17 rerun passed. Three primary test evaluations occurred; no ResNet retraining, data-efficiency, robustness, or Phase 7 work occurred. Evidence: `docs/phase6_mobilenet_v3_small_baseline.md` and `experiments/summaries/phase6_mobilenet_v3_small_v1/`.

## Phase 5 — ResNet-18 Transfer-Learning Baseline

### Status and frozen identity

**COMPLETE.** Phase 4 passed before Phase 5 began. The result validator and exact seed-17 rerun pass. The expanded suite reports **124 passed and 0 failed** with 11 known scikit-learn deprecation warnings.

- Result ID: `phase5_resnet18_v1`.
- Dataset/fingerprint: `wtbd_crops_v1` / `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Counts: 757 train, 146 validation, 162 test; source-image isolation unchanged.
- Runtime: PyTorch `2.13.0+cpu`, torchvision `0.28.0`, device `cpu`; CUDA unavailable.
- Weight enum: `ResNet18_Weights.IMAGENET1K_V1`.
- Pretrained backbone fingerprint: `78d60a5d12431f3233de6606575384b8b65c4c2c8bb4fc1b039001d0e0c1db57`.
- Parameters: 11,179,590 total and trainable.
- Frozen selection: learning rate `0.0003`, weight decay `0`; best tuning validation macro-F1 `0.8973875632`.
- Final seeds/best epochs: 17→7, 29→3, 43→5.
- Checkpoint size: 44,794,379 bytes per seed.
- Seed-17 CPU forward latency: 106.225 ms median and 128.939 ms p95, batch 1, 20 warm-ups and 100 measurements.

### Final results

| Seed | Validation macro-F1 | Test macro-F1 | Balanced accuracy | Accuracy | Macro precision | Macro recall | Training seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.897388 | 0.911068 | 0.919369 | 0.901235 | 0.906522 | 0.919369 | 2962.080 |
| 29 | 0.871300 | 0.883807 | 0.884479 | 0.864198 | 0.892118 | 0.884479 | 1730.863 |
| 43 | 0.855144 | 0.891068 | 0.903580 | 0.901235 | 0.885600 | 0.903580 | 1995.253 |
| Mean ± sample SD | — | 0.895314 ± 0.014118 | 0.902476 ± 0.017472 | 0.888889 ± 0.021383 | 0.894747 ± 0.010706 | 0.902476 ± 0.017472 | — |

Per-class F1 mean ± sample SD in frozen order: craze `0.928511 ± 0.002555`, corrosion `0.847535 ± 0.032036`, surface_injure `0.845446 ± 0.047499`, thunderstrike `0.947368 ± 0.091161`, crack `0.898776 ± 0.003853`, hide_craze `0.904251 ± 0.040824`.

### Evidence and exit gate

- Frozen configuration: `configs/frozen/resnet18.yaml`.
- Versioned outputs: `experiments/summaries/phase5_resnet18_v1/`.
- Regenerable checkpoints/full run: `experiments/results/phase5_resnet18_v1/`.
- Figures: `figures/phase5/`.
- Full record: `docs/phase5_resnet18_baseline.md`.
- [x] Exactly four validation-only candidates used tuning seed 17; tuning test evaluations = 0.
- [x] The winner was frozen before test prediction.
- [x] Fresh final seeds 17/29/43 used full train and validation-only stopping.
- [x] All three checkpoints existed before the test loader was constructed.
- [x] Each final seed received one primary test evaluation; no ensemble or discarded seed.
- [x] Predictions, six logits, histories, metrics, matrices, timing, and fingerprints are machine-readable.
- [x] The independent seed-17 rerun exactly matches best epoch, validation/test predictions, metrics, scientific history, and checkpoint fingerprint.
- [x] No crop/split change, augmentation, MobileNet, data-efficiency, corruption, robustness, or Phase 6 work occurred.

## Phase 4 — Traditional Computer-Vision Baselines

Phase 4 establishes the frozen HOG + RBF-SVM and multi-scale spatial LBP + RBF-SVM baselines using only the full Phase 3 training partition. It is the first phase with real model-performance results.

### Phase 4 status

**COMPLETE.** Phase 3 passed its full gate before feature extraction or training. Exactly eight predeclared SVM candidates per feature family were evaluated using train and validation only. Both winners were frozen before the test gate, and exactly those two selected models were evaluated on test. The expanded suite reports **113 passed and 0 failed**.

### Frozen input and feature identity

- Processed version: `wtbd_crops_v1`.
- Processed fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Instance counts: 757 train, 146 validation, and 162 test; total 1,065.
- HOG dimensions/config hash/feature fingerprint: 6,084 / `e0723cd80ec462644aec14e3827821d716d04ff375424b5de45ac5ddac4d5cf2` / `a89e1102fd90cf8d2ecc8698b029bd90f05abbdf39cd3861b3ceefaceef3ecbf`.
- LBP dimensions/config hash/feature fingerprint: 1,372 / `e952103e7c0664952a0b8c568141bd26d177333c572f7ec134d1270c2592d122` / `2826d502592c6b4066cab2dd64b556a281e79752659401d271986d9990f004e8`.
- Validation-grid fingerprint: `e897c6b58911baaf27d626d38a598404d538af7bf99f72aa7a1a7a1043ad8bb8`.

### Frozen selection and results

| Method | Selected C | gamma | Validation macro-F1 | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| HOG + SVM | 10 | scale | 0.448100 | 0.530864 | 0.457766 | 0.477988 |
| LBP + SVM | 10 | scale | 0.495356 | 0.611111 | 0.572746 | 0.592401 |

Per-class test F1 in frozen class order (craze/corrosion/surface_injure/thunderstrike/crack/hide_craze):

- HOG: 0.596491 / 0.452830 / 0.500000 / 0.363636 / 0.347826 / 0.607143.
- LBP: 0.677966 / 0.526316 / 0.470588 / 0.500000 / 0.666667 / 0.712871.

### Efficiency record

| Measure | HOG + SVM | LBP + SVM |
|---|---:|---:|
| first full feature extraction | 17.708 s | 45.016 s |
| final train-only fit | 2.924 s | 0.647 s |
| feature latency/image | 10.564 ms | 31.609 ms |
| prediction latency/image | 3.549 ms | 0.861 ms |
| combined latency/image | 14.112 ms | 32.470 ms |
| model size | 35,551,712 bytes | 7,687,831 bytes |

Timings are descriptive medians from the recorded Windows CPU environment and do not imply other-hardware performance.

### Phase 4 evidence and exit gate

- Result ID: `phase4_traditional_v1`.
- Clean apparatus commit used by frozen configs: `f5ae5ae8dd7b7dd8f315fdbc944138f32cbfb3b8`.
- Frozen winner configs: `configs/frozen/traditional_hog_svm.yaml` and `traditional_lbp_svm.yaml`.
- Versioned machine-readable outputs: `experiments/summaries/phase4_traditional_v1/`.
- Regenerable models/full run: `experiments/results/phase4_traditional_v1/`.
- Scientific figures: `figures/phase4/`.
- Complete method and result record: `docs/phase4_traditional_baselines.md`.
- [x] Phase 2 strict and Phase 3 complete gates pass before training.
- [x] Processed dataset fingerprint and all 1,065 sample identities remain unchanged.
- [x] HOG is exactly 1,065 × 6,084 and LBP is exactly 1,065 × 1,372; all values are finite.
- [x] Scalers/SVMs fit training instances only; test cannot enter validation selection.
- [x] Exactly 8 HOG and 8 LBP validation candidates are saved.
- [x] Deterministic tie-breaking selected C=10/gamma=scale for both families.
- [x] Winner YAML files were written before test evaluation.
- [x] Only the two selected configurations were evaluated on test.
- [x] Metrics, per-class records, predictions, confusion matrices, timing, model hashes, and sizes are machine-readable.
- [x] Unchanged canonical rerun reproduces scientific files, fingerprints, predictions, metrics, and model hashes.
- [x] Full test suite passes: 113 passed, 0 failed.
- [x] No CNN, pretrained weight, 25/50/75% run, augmentation, corruption, or Phase 5 work was started.

## Phase 3 — Curated Classification Dataset and Preprocessing

Phase 3 transforms every retained Phase 2 annotation into one deterministic classification sample. It freezes the common crop representation and source-grouped data-efficiency manifests without performing feature extraction, model fitting, augmentation, corruption, or performance evaluation.

### Phase 3 status

**COMPLETE.** The clean rebuild generated exactly 1,065 validated 224 × 224 RGB PNG crops from 720 curated source images. All 1,065 PNGs and 28 metadata/QC artifacts reproduced byte-for-byte on a second complete build. The expanded suite reports **93 passed and 0 failed**; Phase 2 strict validation and both Phase 3 validation commands pass.

### Frozen crop and dataset policy

- Processed version: `wtbd_crops_v1`.
- Unit: one classification sample per curated annotated defect instance.
- Stable instance identity: `<source_image_id>_<validated_object_index>`.
- Frozen labels: craze=0, corrosion=1, surface_injure=2, thunderstrike=3, crack=4, hide_craze=5.
- Crop side: `max(64, ceil(1.5 × max(bbox_width, bbox_height)))`, capped at the largest square that fits the decoded source image.
- Geometry: square centered as closely as possible; shift inside image boundaries; no padding or synthetic pixels; complete annotation required.
- Input bounds remain one-based inclusive VOC; recorded crop bounds are zero-based half-open pixel coordinates.
- Neutral output: deterministic bilinear resize to 224 × 224, RGB, lossless PNG; no enhancement, normalization, augmentation, or corruption.
- Common-input rule: every later core method derives from the same Phase 3 image.

### Processed dataset and provenance

- Upstream raw fingerprint: `568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb`.
- Phase 2 curation-manifest SHA-256: `9e5ce3b44457e52f686fb16f62df18a10a576262c9f0f89b96ccdd75d89c0767`.
- Phase 3 resolved-config SHA-256: `e91f2026c3e6ac8dc75adf138014cf07a4e9d8907c638ae21fe52c799460b9b8`.
- Processed dataset fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Class counts: craze 169, corrosion 178, surface_injure 264, thunderstrike 60, crack 131, hide_craze 263.
- Source images by split: train 510, validation 101, test 109.
- Crop instances by split: train 757, validation 146, test 162.
- Every class is present in every split; source-ID split intersections are empty.
- Crop side median: 537 pixels. Defect occupancy min/p05/p25/median/p75/p95/max: 0.028451/0.059817/0.128937/0.239509/0.355987/0.426917/0.646800.
- Four crops use the 64-pixel minimum, 568 are shifted at a boundary, and 167 are clipped to the maximum in-image square. No crop truncates its annotation.

### Frozen data-efficiency subsets

- Scientific subset seeds: 17, 29, and 43.
- Unit: training source image; every selected source contributes all of its defect instances.
- Exact source targets: 128/255/383/510 for 25%/50%/75%/100%.
- Every seed family is strictly nested and every subset contains all six classes.
- Validation and test do not enter the selection objective and remain fixed.
- Partial-fraction instance totals are 252/440/608; the full training set contains 757 instances.
- Seed 17 fingerprints: `b231a06c1e90ce97090359e861df054c98e1084c46288571424851b37ab606dd`, `ec9f49a5532a62afac49942d45fe02c214442d944a173a4954152d2bdb9c3a7f`, `bd8d15b8214c52378f83cdf666cd46fc0411272fcc1967654d3c0280a47cb496`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.
- Seed 29 fingerprints: `a2b427a6a682de42ef29280042c0ffbe862f80963b13dbb15aa5bfabf77db10c`, `b6f5222161c60af4373413caf49409b12f0e29720b6dd035ee2e4aad2e620f8c`, `3f2a2f67ca829e008d12e95f3526d8595aff0a61e2ae99a9acb2150a215fc21d`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.
- Seed 43 fingerprints: `3b43aaefb03318b3b5349ffd81626b0f15dd1fc3eb7e3d40523a0e1cf6580f33`, `5a66073adc48f8aa57c7c7711a86676540adfa5a6ee5e59d981d5cc874632a19`, `6ee68e91f499b1e011ebbe17a97cf3f1bc8e314ba28de0331156ad7f17560005`, `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95`.

### Phase 3 evidence and exit gate

- Frozen config: `configs/crop_dataset.yaml`.
- Processed manifest, checksum manifest, summary, resolved config, and regenerable payload location: `data/processed/wtbd_crops_v1/`.
- Instance split and nested subset manifests: `data/splits/wtbd_crops_v1_split.csv` and `data/splits/wtbd_crops_v1/`.
- Crop statistics, split counts, label map, and subset fingerprints: `data/metadata/wtbd/`.
- Training-only QC sheets: `figures/phase3/crop_qc/`.
- Methodology and complete results: `docs/phase3_crop_preprocessing.md`.
- [x] Phases 0–2 remain frozen and Phase 2 strict validation passes.
- [x] Raw and curated fingerprints/counts match the frozen inputs.
- [x] Exactly 1,065 valid crops were generated with no excluded source.
- [x] Class counts and source/instance split counts reproduce exactly.
- [x] Every crop contains its full annotation and uses no padding.
- [x] Three nested source-group subset families validate at exact target counts.
- [x] Every split and every training subset contains all six classes.
- [x] Processed and subset fingerprints are recorded.
- [x] Clean deterministic regeneration passes for 1,093/1,093 artifacts.
- [x] Full test suite passes: 93 passed, 0 failed.
- [x] No model, pretrained weight, augmentation, corruption, or Phase 4 work was started.

## Phase 2 — WTBD Dataset Acquisition and Forensic Audit

Phase 2 treats the official WTBD release as immutable source evidence. It may acquire and audit the dataset, but it does not create final classification crops, choose preprocessing or scientific hyperparameters, download model weights, or train models.

### Phase 2 status

**COMPLETE.** The raw forensic audit still records the official-release discrepancies, while the reviewed `wtbd-curation-v1` interpretation passes its separate leakage-control gate. The expanded suite reports **59 passed and 0 failed**. Strict manifest validation reports `PASS` with no blockers.

### Acquisition and immutable provenance

- Official source: Springer Nature Figshare, dataset DOI `10.6084/m9.figshare.30210175`, version 1, CC BY 4.0.
- Method: automatic official download.
- Archive: `WT blade defect dataset.zip`, 78,958,553 bytes.
- Official MD5: `14ad7e2cf7161b9100d1d70fb398b0cf`.
- Archive SHA-256: `466452f2a0cfc9ef6ba63ea2a3bbc7ea4262057dd07e4fc9e00eedf5bba305b4`.
- Raw dataset fingerprint: `568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb` (reproduced before and after review/curation generation).
- Raw contents remain unmodified and excluded from Git.

### Verified release properties

- 1,065/1,065 JPEG images decode successfully; 1,064 are 1024×1024 and `714.jpg` is 788×788, matching its XML dimensions.
- 1,065 primary XML files and 1,065 separate second-annotator XML files are present.
- Image/XML pairing by file ID is complete; no unmatched or duplicate IDs were found.
- All primary XML files parse and all 1,584 parsed bounding boxes pass geometric validation under inclusive VOC coordinates.
- Exact raw labels are the expected six strings, but their observed counts differ from the published counts.
- Official split CSV is disjoint and exhaustive: 745 train, 159 validation, 161 test; zero overlap, omitted, unknown, or duplicate IDs.

### Critical official-release discrepancies

1. **XML identity ambiguity:** 262 primary XML `<filename>` values disagree with their XML/image file IDs. These occur across XML IDs 743–1062 in discontinuous ranges. Automated comparison and four-view sheets now exist, but no recommendation is silently applied as a decision.
2. **Object-count mismatch:** primary XML contains 1,584 objects, not the published 1,568.
3. **Class-count mismatch:** actual counts are craze 257 (expected 259), corrosion 257 (254), surface_injure 412 (394), thunderstrike 92 (92), crack 224 (224), and hide_craze 342 (345).
4. **Cross-split exact duplicate:** `547.jpg` (train) and `640.jpg` (validation) are identical both as files and decoded pixels. A second exact pair, `565.jpg`/`668.jpg`, remains within train.

These findings are preserved as raw evidence. No label, filename field, official split, annotation, or image was altered. The separate curated benchmark is derived only from the versioned manifest and does not claim to reconstruct the authors' intended 1,568-object dataset.

### Raw scientific warnings and review candidates

- 491 non-exact dHash candidates at distance ≤4; 166 cross official splits. dHash remains a screening heuristic rather than automatic proof of duplication.
- 220 boxes occupy less than 1% of source-image area; 5 occupy less than 0.1%.
- Bounding-box area fraction ranges from approximately 0.000608 to 0.646800; aspect ratio ranges from approximately 0.042829 to 22.85.
- Diagnostic flags identify 26 boxes with aspect ratio <0.1, 15 with aspect ratio >10, and 12 occupying more than 50% of their source image; these are not exclusion rules.
- 38 of 751 within-image object pairs overlap at IoU >0; none reach IoU 0.25.
- 84 source images contain multiple canonical classes; 319 contain repeated instances of a class.
- Objects per image: mean 1.487324, median 1, minimum 1, maximum 7; 696 images have one object, 268 have two, 75 have three, and 26 have four or more.

### Curation/reconciliation state

- Curation version: `wtbd-curation-v1`; schema and allowed enums are recorded in `curation_schema.json`.
- Human-review archive SHA-256: `587d847afcf014d9276ce78eabce1d79a30349a3c853e453a40d35999b1df1e8`; imported file hashes and the embedded-instruction exclusion are recorded in `human_review/import_manifest.json`.
- All 262 stale/mismatched identity rows were reviewed as reused/derived source-scene variants and excluded. Zero identity decisions remain pending and no unresolved identity is included.
- Second-annotator comparison: 922 strong agreement, 15 box disagreement, 87 class disagreement, and 41 object-count disagreement cases. Second annotations remain evidence only.
- Exact groups: retain `547` in train and exclude `640`; retain `565` in train and exclude `668`. No included exact group crosses curated splits.
- Human pair decisions: 122 `same_scene`, eight `unrelated_false_positive`, and 361 intentionally pending candidates.
- The 122 same-scene edges form 50 reviewed components with 131 members. The lowest-ID canonical is retained in each component and 81 redundant members are excluded.
- Pending candidates are non-blocking warnings: 283 involve excluded images, 78 retain both images within one split, and zero retain both images across splits.
- Final curated benchmark: 720 images and 1,065 objects; train 510, validation 101, test 109.
- Final class counts: craze 169, corrosion 178, surface_injure 264, thunderstrike 60, crack 131, hide_craze 263. All six classes occur in every split.
- `data/metadata/wtbd/curation_blockers.csv` contains only its header; there are no remaining Phase 2 blockers.

### Phase 2 evidence

- Machine-readable summary: `data/metadata/wtbd/audit_summary.json`.
- Full audit: `docs/phase2_dataset_audit.md`.
- Curation policy, human-review import, final counts, and completed gate: `docs/phase2_curation.md`.
- Curated manifest and summary: `data/metadata/wtbd/curation_manifest.csv` and `curation_summary.json`.
- Empty final blocker table: `data/metadata/wtbd/curation_blockers.csv`.
- Imported review provenance and component/cross-split cross-checks: `data/metadata/wtbd/human_review/`.
- Identity evidence sheets and index: `figures/phase2/identity_review/`.
- Per-class annotation review: `figures/phase2/annotation_examples/`.
- Candidate near-duplicate review: `figures/phase2/near_duplicates/`.
- Split, bbox, co-occurrence, checksum, duplicate, and upstream inventories: `data/metadata/wtbd/`.

### Phase 2 exit-gate record

- [x] Phase 0 and Phase 1 remain frozen and tests pass.
- [x] Official provenance, license, archive checksum, and raw fingerprint are recorded.
- [x] Every image decodes and image/XML file-ID pairing is complete.
- [x] All primary XML parses and all boxes pass geometric validation.
- [x] Official split syntax, coverage, disjointness, and class statistics are audited.
- [x] Exact and perceptual duplicate screening is complete.
- [x] Contact sheets, plots, CSV records, JSON summary, and audit documentation are generated.
- [x] Published/raw count discrepancies are preserved without forcing agreement.
- [x] A versioned manifest, schemas, decision inputs, diagnostics, and raw/curated statistics are generated.
- [x] All 262 identity decisions are imported and their reviewed reused/derived variants are excluded.
- [x] Cross-split exact-copy leakage is removed by a deterministic canonical/exclusion policy.
- [x] Raw fingerprint remains stable after curation generation.
- [x] The reviewed same-scene graph reproduces 50 supplied components and excludes 81 redundant members.
- [x] No confirmed same-scene component has more than one included image.
- [x] No pending cross-split candidate retains both images; the eight retained cross-split candidates are reviewed false positives.
- [x] Pending within-split and excluded-image candidates remain documented as non-blocking warnings.
- [x] All six classes remain represented in train, validation, and test.
- [x] Curated metadata regenerates deterministically and strict validation passes.
- [x] Phase 2 is complete.

No model was implemented or trained, no model weights were downloaded, no final classification crops were generated, and no crop margin or preprocessing choice was frozen.

## Phase 1 — Repository and Reproducibility Infrastructure

Phase 1 establishes engineering infrastructure using synthetic data only. It does not acquire datasets, implement scientific methods, train models, or produce scientific results.

### Frozen Phase 1 decisions

- Python 3.11 reference environment.
- `src/windblade` package layout.
- YAML source configuration and per-run resolved YAML snapshots.
- Deterministic 12-character SHA-256 configuration fingerprints.
- Timestamped, sanitized experiment identifiers.
- Global Python, NumPy, and optional PyTorch seed utility.
- Technical environment and Git provenance in every manifest.
- Deterministic machine-readable JSON result records.
- One isolated run directory containing resolved configuration, manifest, results, and log.
- Standard-library console and file logging.
- Failed-run evidence preservation with non-swallowed exceptions.
- Pytest infrastructure using temporary output directories.
- Immutable `data/raw/` policy.

### Phase 1 status

Complete. Validation used Python 3.11.15 and pytest 9.1.1:

- Full suite: **21 passed, 0 failed**.
- Two independent seed-42 CLI runs produced different timestamped experiment IDs, the identical config hash `cc5ceef2d6b6`, and identical deterministic synthetic payloads.
- A seed-43 CLI run produced config hash `6a8b48d3045f` and a different seeded synthetic payload.
- All three run directories contained exactly `resolved_config.yaml`, `manifest.json`, `results.json`, and `run.log`.
- All three inspected manifests reported `completed`, Python 3.11.15, CPU execution, PyTorch unavailable, Git commit `7ee35b99b58fc22a04c87e789dbe18762a72554c`, and a clean worktree.
- The only files under `data/` are policy documentation and tracked directory placeholders.

### Phase 1 exit-gate record

- [x] Phase 0 research documentation remains unchanged.
- [x] The `windblade` package imports under Python 3.11.
- [x] YAML configuration loading and validation work.
- [x] Configuration hashing is deterministic and order-independent.
- [x] Python and NumPy global seeding work; optional PyTorch handling is non-fatal.
- [x] Environment and Git provenance capture work without optional ML dependencies.
- [x] Timestamped, sanitized experiment IDs are generated.
- [x] Every experiment receives an isolated run directory.
- [x] Resolved configuration snapshots are preserved.
- [x] Manifest JSON, results JSON, and run logs are generated.
- [x] Failed-run status, failure details, configuration, and logs are preserved and tested; exceptions propagate.
- [x] Synthetic smoke execution succeeds on CPU in seconds.
- [x] Same-seed smoke payload determinism is independently validated.
- [x] A different seed changes both the resolved hash and seeded payload.
- [x] Full pytest suite passes: 21 passed, 0 failed.
- [x] Raw-data immutability and version-control policies are documented.
- [x] No WTBD or other real dataset was downloaded.
- [x] No real ML model was trained or implemented.
- [x] No scientific performance number or figure was produced.

## Phase 0 artifacts

- `docs/phase0_research_contract.md` — authoritative research scope, rules, safeguards, and non-claims.
- `docs/source_ledger.md` — supplied literature and dataset ledger only.
- `docs/experiment_matrix.md` — core comparison matrix and frozen-versus-later decision boundary.
- `PROJECT_STATE.md` — phase status and handoff record.

## Frozen decisions

- Working title: **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**.
- Primary question: comparison of handcrafted image-feature methods and transfer-learned CNNs for classifying visible surface-defect crops under limited labeled data and image degradation.
- Task: six-class classification of expert-annotated WTBD defect instances—not arbitrary full-image detection.
- Primary dataset: WTBD, with one crop per PASCAL VOC bounding box.
- Classes: `craze`, `corrosion`, `surface_injure`, `thunderstrike`, `crack`, and `hide_craze`.
- WTBD has no valid healthy/normal class; healthy-versus-damaged claims are prohibited.
- “Earlier detection” is prohibited because WTBD has no temporal onset labels.
- Leakage rule: every crop from the same source image remains in one partition; crops are never independently random-split.
- Split rule: use the supplied standard source-image split when available and valid.
- Core methods: HOG + SVM, LBP + SVM, ImageNet-pretrained ResNet-18, and ImageNet-pretrained MobileNetV3-Small.
- Primary metric: macro F1-score.
- Secondary, class-level, and efficiency measures are those enumerated in the research contract and experiment matrix.
- Training fractions: 25%, 50%, 75%, and 100% of the training partition only; validation and test sets stay fixed.
- Repetition: exactly three reproducible seeds, with individual results, mean, and standard deviation; no best-seed-only reporting.
- Corruptions: Gaussian blur, resolution degradation, brightness reduction, and JPEG compression, each with clean/control, mild, moderate, and severe levels.
- Robustness occurs after model selection on the fixed test set, without retraining or fine-tuning on corrupted test images.
- Test data do not influence training, hyperparameter selection, preprocessing, or model selection.
- Final reported numbers must come from machine-readable outputs.
- Core training excludes generated, synthetic, web-scraped, and secondary-dataset images.
- Unsupported priority, architecture-novelty, state-of-the-art, causal, safety, economic, and deployment claims are prohibited as detailed in the research contract.
- Blade30, DTU Drone Inspection Images, WTBs2025, lightweight YOLO detection, and Grad-CAM are outside the core Phase 0 scope.

## Phase 0 handoff questions and resolution

These were recorded at the end of Phase 0. Phase 1 infrastructure questions are now resolved; scientific and dataset-specific items remain intentionally deferred and must not be chosen by inspecting test or robustness results.

### Resolved in Phase 1

- [x] Supported Python runtime and minimal dependency versions.
- [x] Environment and lockfile strategy.
- [x] Repository layout beyond the Phase 0 documents.
- [x] Configuration schema, smoke command, artifact directory conventions, and machine-readable JSON output.
- [x] Logging, Git/environment provenance, deterministic controls, and validation tests.

### Resolved in Phase 3

- [x] The validated supplied source-image split is inherited by every crop.
- [x] The contextual crop multiplier, minimum, coordinate conversion, and rounding behavior are frozen.
- [x] Boundary shifting, annotation containment, and image/annotation integrity handling are frozen and tested.
- [x] The 224 × 224 RGB PNG common input, bilinear interpolation, no-padding policy, and later method-fairness boundary are frozen.
- [x] Deterministic grouped, nested training-subset construction is implemented and validated.
- [x] Scientific data-efficiency seeds are frozen as 17, 29, and 43.

### Resolved in Phases 4–5

- [x] HOG/LBP feature extraction, SVM fitting, and validation selection are frozen by Phase 4.
- [x] ResNet-18 train-only balanced loss, optimizer grid, batch size, epoch limit, early stopping, no-augmentation policy, full fine-tuning, tie handling, checkpointing, and timing are frozen by Phase 5.

### Resolved in Phases 6–7; remaining for Phase 8

- [x] MobileNetV3-Small implementation details and validation-selected optimizer configuration were frozen in Phase 6.
- [x] The four-method limited-labeled-data protocol and results were completed in Phase 7 without per-fraction retuning.
- Exact numerical corruption parameters and deterministic implementations for all four severity scales.
- Any later phase-specific timing additions that do not inherit the frozen Phase 5 procedure.

## Phase 0 exit-gate record

- [x] Research question is unambiguous and uses “classifying.”
- [x] Task is explicitly six-class defect classification.
- [x] WTBD is the primary dataset.
- [x] Absence of a healthy class is documented.
- [x] “Earlier detection” is explicitly excluded.
- [x] Four core methods are frozen.
- [x] Macro F1-score is the primary metric.
- [x] Four data-efficiency fractions are frozen.
- [x] Four corruption families are frozen.
- [x] Source-image leakage prevention is documented.
- [x] Test-set policy is documented.
- [x] Unsupported novelty claims are prohibited.
- [x] Every supplied source is recorded.
- [x] No model training or dataset download occurred.
- [x] No external literature research occurred.

## Phase boundary

Completed Phases 0–10 remain frozen unless the user explicitly requests a documented new version. Phase 11A is complete and frozen; Phase 11B remains blocked and requires a new pre-test apparatus commit on suitable CUDA hardware. Phase 12 external validation has not started and requires separate explicit authorization. Documentation-only corrections may not change frozen numbers, statistical definitions, outputs, or conclusions.
