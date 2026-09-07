# BladeScope

### Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation

BladeScope is a reproducible computer-vision research project studying six-category wind-turbine blade defect recognition under limited training data and controlled image degradation. It combines classical and convolutional classification baselines with data-efficiency and robustness experiments, Grad-CAM and blinded human review, statistical synthesis, experimental defect localization, and a Streamlit demonstration application.

> **License:** This entire repository is licensed under the GNU Affero General
> Public License v3.0 only (`AGPL-3.0-only`). See [LICENSE](LICENSE). Dataset-
> derived assets retain the attribution documented in
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**[Application](#run-the-bladescope-application) · [Portfolio summary](docs/portfolio_summary.md) · [Final report](docs/phase10_final_synthesis.md) · [Results](#key-results) · [Reproducibility](#reproducibility-and-project-record) · [Limitations](#limitations)**

![Frozen clean-test macro-F1 comparison with bootstrap intervals](figures/phase10/clean_macro_f1_bootstrap_ci.png)

*Frozen clean-test comparison. CNN values aggregate the three declared seeds; interval definitions and the deterministic-baseline treatment are documented in the [final synthesis](docs/phase10_final_synthesis.md).*

## Highlights

- **720 retained full images** and **1,065 annotated defect instances/crops** across **six defect classes**.
- Classical HOG/LBP + SVM baselines and CNN baselines using ResNet-18 and MobileNetV3-Small.
- Data-efficiency experiments at four declared training-data budgets and robustness evaluation across 12 controlled synthetic degradation conditions.
- Grad-CAM analysis, a blinded two-pass human-review workflow, and a frozen statistical synthesis with paired stratified bootstrap intervals.
- A three-seed YOLO11n class-agnostic localization study and a human-reviewed automatic-region proposal workflow in the Streamlit app.
- The highest observed clean CNN means were approximately `0.895` macro-F1. Their difference was negligible, but the paired interval does **not** justify a claim of superiority or equivalence.

## Research Question

How reliably can classical and convolutional models recognize six wind-turbine blade defect categories when labeled data are limited? The project also asks how performance changes under controlled image degradation, and how errors, model activations, and localization behavior can be examined without extending the evidence to operational safety or deployment claims.

## Key Results

| Method | Clean macro-F1 | Notes |
|---|---:|---|
| HOG + SVM | `0.477988` | Deterministic handcrafted baseline; seed SD is not applicable. |
| LBP + SVM | `0.592401` | Deterministic handcrafted baseline; seed SD is not applicable. |
| ResNet-18 | `0.895314 ± 0.014118` | Mean ± sample SD across seeds 17, 29, and 43. |
| MobileNetV3-Small | `0.895321 ± 0.005977` | Mean ± sample SD across seeds 17, 29, and 43. |

On the frozen clean test set, both CNN means were substantially higher than the traditional baselines. MobileNetV3-Small had the highest observed mean performance and retention across the declared synthetic degradation grid, while ResNet-18 had the highest normalized data-efficiency area. No composite winner is defined. The paired MobileNet-minus-ResNet macro-F1 interval was `[-0.036869, 0.035522]`; it supports neither a superiority claim nor an equivalence claim.

## Visual Evidence

| Data efficiency | Controlled degradation robustness |
|---|---|
| ![Frozen data-efficiency learning curves](figures/phase10/data_efficiency_learning_curves.png) | ![Frozen robustness curves](figures/phase10/robustness_curves.png) |

The left figure compares four predeclared training-data budgets. The right figure shows the four fixed degradation families at clean, mild, moderate, and severe settings. Both are descriptive summaries of the frozen experiment grid, not claims about arbitrary field conditions.

<!-- Portfolio TODO: add a real Streamlit application screenshot here after one is captured from the deployed or locally validated app; do not use a mock-up. -->

## What I Built

`Dataset audit and curation` → `source-isolated crop generation` → `HOG/LBP baselines` → `ResNet-18 and MobileNetV3-Small baselines` → `data-efficiency study` → `robustness study` → `error analysis and Grad-CAM` → `blinded human review` → `statistical synthesis` → `YOLO11n localization study` → `Streamlit demonstration app`

Each stage has a frozen configuration, machine-readable outputs, validation gates, and explicit provenance. Scientific stages remain separate from the application productization layer.

## Tech Stack

Python 3.11 · PyTorch and torchvision · scikit-learn · NumPy · scikit-image · Pillow · matplotlib · Ultralytics YOLO · Streamlit · pytest · uv

## Limitations

- The dataset is limited in size and contains defect-positive images without healthy/background-only controls; healthy-blade false-positive behavior is not measured.
- Classification covers six dataset categories, and localized crops do not cover hidden or internal damage.
- The 12 synthetic degradation conditions do not fully represent field cameras, drone flights, sites, weather, lighting, or compound degradation.
- Grad-CAM and the single-reviewer judgments are descriptive; they are not proof of causal model reasoning.
- The YOLO11n detector is an experimental, human-reviewed proposal aid, not validated automatic inspection.
- BladeScope does not assess safety, severity, progression, remaining life, or production readiness.
- Detector checkpoint, threshold, and NMS selection were frozen before held-out evaluation; no post-test tuning is permitted.

## Reproducibility and Project Record

The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md). The sections below preserve the detailed installation, acquisition, execution, validation, result-path, and phase-boundary record. Phases 0–10 are complete, validated, and frozen; the Phase 11A audit and Phase 11B detector study are also complete and frozen. Application v3 integrates one exact frozen detector checkpoint as an experimental, human-reviewed region-proposal aid. It is locally validated; no external deployment URL is asserted here.

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

Runtime dependencies include PyYAML, NumPy, Pillow, matplotlib, scikit-image, scikit-learn, joblib, PyTorch, and torchvision; pytest is the development/test dependency.

## Run the BladeScope application

The repository also contains a non-scientific Streamlit application using the frozen Phase 6 MobileNetV3-Small classifier and the frozen Phase 11B proposal detector. Install its separately pinned CPU dependencies and launch it locally:

```bash
uv pip install -r requirements-app.txt
uv run streamlit run app/app.py
```

Application v3 presents **Auto detection** as its primary workflow while retaining its scientific status as an experimental, human-reviewed proposal aid. Proposals are numbered, detector confidence is shown separately, and a user must review and select boxes before the frozen six-category crop classifier runs. The dataset contains no healthy/background-only images, so this does not establish a healthy blade or assess safety, severity, progression, remaining life, or production readiness. Both exact checkpoints are included for clean-clone startup; no training, tuning, calibration, or test-set evaluation occurs. See [`docs/app.md`](docs/app.md) for the exact identities, controls, scope, and validation record.

For Streamlit Community Cloud, deploy branch `main` with entrypoint `app/app.py` and select Python 3.11. The app-local dependency manifest and exact frozen inference checkpoint are tracked, and no secrets are required. See [`docs/deployment.md`](docs/deployment.md) for the deployment checklist and validator.

## Run the local human-review interface

The separate Phase 9A review interface enters human judgments into the existing blinded two-pass packet without loading a model or suggesting answers:

```powershell
uv run streamlit run app/review_app.py --server.address 127.0.0.1
```

The completed Phase 9 forms are now frozen. The interface remains documented for provenance, but it is no longer an active data-entry step. See [`docs/human_review_interface.md`](docs/human_review_interface.md) for the blinding, autosave, recovery, and verification procedure that was used. The tool is scientifically separate from Phase 9B.

## Run the complete test suite

```bash
uv run python -m pytest
```

## Run the synthetic smoke experiment

```bash
uv run python scripts/run_smoke_experiment.py --config configs/smoke.yaml
```

The smoke command uses only a tiny deterministic synthetic array. It does not download data or train a model. Each run writes `resolved_config.yaml`, `manifest.json`, `results.json`, and `run.log` under `experiments/results/<experiment_id>/`.

## Acquire and audit WTBD

Only the official Springer Nature Figshare release is accepted:

```bash
uv run python scripts/acquire_wtbd.py
uv run python scripts/audit_wtbd.py --config configs/dataset_audit.yaml
```

Raw external files remain under `data/raw/wtbd/` and are excluded from Git. Versionable audit evidence is under `data/metadata/wtbd/`, with human-review figures under `figures/phase2/`. See [`docs/phase2_dataset_audit.md`](docs/phase2_dataset_audit.md) before any Phase 3 decision.

## Review and curate WTBD

The curation layer records interpretations without renaming, editing, or moving official files:

```bash
uv run python scripts/review_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml
uv run python scripts/curate_wtbd.py --config configs/curation.yaml --validate-only
```

`--strict` now succeeds only when the reviewed manifest passes every Phase 2 gate. Decisions and imported review provenance are versioned; rerunning review preserves the decision files. See [`docs/phase2_curation.md`](docs/phase2_curation.md).

## Regenerate the Phase 3 classification dataset

With the immutable official WTBD release present under `data/raw/wtbd/`, rebuild the 1,065 lossless crops and then the nested training subsets:

```bash
uv run python scripts/build_wtbd_crops.py --config configs/crop_dataset.yaml
uv run python scripts/build_training_subsets.py --config configs/crop_dataset.yaml
```

The common 224 × 224 RGB PNG payload is under `data/processed/wtbd_crops_v1/images/` and is ignored by Git. Its versioned manifest, checksums, summary, and fingerprint are beside it; split manifests are under `data/splits/wtbd_crops_v1/`, and compact statistics and subset fingerprints are under `data/metadata/wtbd/`. See [`docs/phase3_crop_preprocessing.md`](docs/phase3_crop_preprocessing.md).

## Run the frozen traditional baselines

The canonical Phase 4 command validates Phase 3, extracts or validates cached HOG/LBP features, performs the fixed validation-only SVM grids, freezes both winners, and evaluates only the selected models on test:

```bash
uv run python scripts/run_traditional_baselines.py --config configs/traditional_baselines.yaml
```

Versioned machine-readable results are under `experiments/summaries/phase4_traditional_v1/`; regenerable model binaries and caches remain ignored under `experiments/results/` and `experiments/cache/`. Figures are under `figures/phase4/`. See [`docs/phase4_traditional_baselines.md`](docs/phase4_traditional_baselines.md).

## Run the frozen ResNet-18 baseline

The canonical Phase 5 command validates Phases 3–4, loads the explicit official ImageNet weights, runs the four-candidate validation grid, freezes the selected optimizer configuration, trains seeds 17/29/43, performs gated test evaluation, and verifies a full seed-17 deterministic rerun:

```bash
uv run python scripts/run_resnet18_baseline.py --config configs/resnet18_baseline.yaml
```

The frozen selection is `configs/frozen/resnet18.yaml`. Versioned results are under `experiments/summaries/phase5_resnet18_v1/`; regenerable checkpoints remain ignored under `experiments/results/phase5_resnet18_v1/`. Figures are under `figures/phase5/`. See [`docs/phase5_resnet18_baseline.md`](docs/phase5_resnet18_baseline.md).

## Run the frozen MobileNetV3-Small baseline

```bash
uv run python scripts/run_mobilenet_v3_small_baseline.py --config configs/mobilenet_v3_small_baseline.yaml
```

Frozen config: `configs/frozen/mobilenet_v3_small.yaml`. Results: `experiments/summaries/phase6_mobilenet_v3_small_v1/`. See `docs/phase6_mobilenet_v3_small_baseline.md`.

## Run the frozen data-efficiency experiment

The resumable canonical Phase 7 command validates Phases 3–6, runs only the predeclared 25%/50%/75% reduced-data fits, reuses every frozen 100% endpoint, performs the two canonical reproducibility reruns, and generates the learning-curve aggregates:

```bash
uv run python scripts/run_data_efficiency.py --config configs/data_efficiency.yaml
```

Results are under `experiments/summaries/phase7_data_efficiency_v1/`, figures under `figures/phase7/`, and the complete record is [`docs/phase7_data_efficiency.md`](docs/phase7_data_efficiency.md).

## Run the frozen robustness experiment

The Phase 8 command verifies all upstream gates and frozen artifacts, exactly reproduces clean predictions, generates twelve deterministic test-image degradation conditions, evaluates only the frozen full-data models, and performs a second complete regeneration/evaluation pass:

```bash
uv run python scripts/run_robustness.py --config configs/robustness.yaml
```

## Phase 9: error analysis, blinded review, and descriptive synthesis

Phase 9A performed post-hoc quantitative error analysis on frozen Phase 8 predictions, generated Grad-CAM only from the frozen CNN checkpoints, and prepared the two-pass review packet. Pass A was completed blind to model evidence; Pass B was completed only after the corrected caption packet was verified. Phase 9B validates and joins those human judgments to frozen metadata and generates descriptive summaries only. No training or refitting occurs.

```powershell
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --apparatus-check
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --validate-only
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --phase9b
uv run python scripts/run_error_analysis.py --config configs/error_analysis.yaml --validate-phase9b
```

The default Phase 9A generation command now refuses to run when completed review data exist, preventing accidental replacement of the human inputs. Phase 9B outputs are under `experiments/summaries/phase9_error_analysis_v1/phase9b/`; its five figures are under `figures/phase9/human_review/`.

Across the 60 reviewed cases, the dataset label was judged visually plausible in 51 cases, activation was inside or partially inside the annotation in 51 cases, and seed patterns were fully or partly consistent in 54 cases. These are single-reviewer, post-hoc descriptive observations. They are not hypothesis tests, do not establish causality, and do not make Grad-CAM a proof of model reasoning. See [`docs/phase9_error_analysis.md`](docs/phase9_error_analysis.md) for the full frozen record.

Versioned results are under `experiments/summaries/phase8_robustness_v1/`, tracked dataset metadata under `data/processed/wtbd_robustness_v1/`, figures under `figures/phase8/`, and the complete record is [`docs/phase8_robustness.md`](docs/phase8_robustness.md). The 1,944 corrupted PNG payloads remain Git-ignored.

## Phase 10: final statistical synthesis and reproducibility freeze

Phase 10 reads frozen Phase 3–9 artifacts only. It consolidates the final clean, data-efficiency, robustness, error-analysis, and human-review results; computes 5,000 paired true-class-stratified bootstrap resamples for clean-test uncertainty; and produces ten canonical CSV/JSON tables and seven figures. All three CNN seeds are retained within every resample, deterministic-method seed SD is `N/A`, and no p-values or model-selection decisions are introduced.

```powershell
uv run python scripts/run_final_synthesis.py --config configs/final_synthesis.yaml --apparatus-check
uv run python scripts/run_final_synthesis.py --config configs/final_synthesis.yaml --validate-only
```

Clean macro-F1 was HOG `0.477988`, LBP `0.592401`, ResNet `0.895314 ± 0.014118`, and MobileNet `0.895321 ± 0.005977`. The paired MobileNet-minus-ResNet interval was `[-0.036869, 0.035522]`, so the negligible observed mean difference is not presented as superiority or equivalence. MobileNet had the highest observed mean performance/retention across the declared synthetic degradation grid, while ResNet had the highest normalized data-efficiency area. The synthesis preserves these as separate trade-offs rather than constructing a composite winner.

Results are under `experiments/summaries/phase10_final_synthesis_v1/`, figures under `figures/phase10/`, and the full statistical, reproducibility, limitation, table, and figure record is [`docs/phase10_final_synthesis.md`](docs/phase10_final_synthesis.md).

## Phase 11A: full-image detection feasibility audit

Phase 11A validates the authoritative full images and primary PASCAL VOC boxes, preserves the human-reviewed Phase 2 source/duplicate curation, freezes a source-image-level 510/101/109 detection split, and generates a deterministic 26-image annotation-QC packet. The retained dataset has 720 defect-positive images and 1,065 boxes across six classes, with no invalid boxes or cross-split duplicate leakage—but no healthy/background-only controls.

```powershell
uv run python scripts/run_detection.py --config configs/detection.yaml --apparatus-check
uv run python scripts/run_detection.py --config configs/detection.yaml --validate-only
```

The Phase 11B detector study completed its three frozen YOLO11n runs, validation-only checkpoint and threshold selection, and firewalled held-out evaluation. Its final metrics are committed and no further tuning is permitted. See [`docs/phase11b_detector_training.md`](docs/phase11b_detector_training.md) for the final scientific record and [`docs/phase11b_colab.md`](docs/phase11b_colab.md) for the execution apparatus. The Phase 11A dataset/protocol, compute gate, QC packet, and limitations remain frozen in [`docs/phase11_detection.md`](docs/phase11_detection.md). Application v3 uses one validation-selected frozen checkpoint only as an experimental proposal aid; see [`docs/app.md`](docs/app.md) and [`docs/phase12_integration.md`](docs/phase12_integration.md).

## Phase boundaries

Phases 0–10 remain complete, validated, and frozen. Optional Phase 11A is complete and frozen, and Phase 11B is complete, validated, and frozen with no post-test tuning permitted. Phase 12A historical apparatus evidence remains unchanged. Phase 12B implements and locally validates Application v3 with fixed CPU proposal controls and mandatory human review. No scientific result changed and no external deployment occurred.
