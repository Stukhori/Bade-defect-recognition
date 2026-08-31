# Robust Wind Turbine Blade Defect Recognition

This repository supports the experimental study **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**. Phases 0–10 are complete, validated, and frozen; the core technical research project is complete. Results include the Phase 4 handcrafted baselines, matched Phase 5/6 ResNet-18 and MobileNetV3-Small baselines, the Phase 7 limited-labeled-data comparison, the Phase 8 controlled image-degradation robustness experiment, the Phase 9 post-hoc error analysis and single-reviewer descriptive synthesis, and the Phase 10 final statistical synthesis. The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md).

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

Runtime dependencies include PyYAML, NumPy, Pillow, matplotlib, scikit-image, scikit-learn, joblib, PyTorch, and torchvision; pytest is the development/test dependency.

## Run the separate local demonstration

The repository also contains a non-scientific Streamlit demonstration of the frozen Phase 6 MobileNetV3-Small full-data seed-17 classifier. Install its separately pinned UI dependencies and launch it locally:

```bash
uv pip install -r requirements-app.txt
uv run streamlit run app/app.py
```

It accepts a prepared visible-defect crop or a larger image with a manually selected region. It does not automatically detect defects or assess blade safety. The local frozen checkpoint must already be present; no training, tuning, calibration, or test-set evaluation occurs. See [`docs/app.md`](docs/app.md) for the exact model identity, crop parity, scope, limitations, and validation record.

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

## Phase boundaries

Phases 0–10 are complete, validated, and frozen. Phase 11 localization and Phase 12 external validation are optional and have not been started. The separate Streamlit application remains a non-scientific demonstration: it classifies manually selected visible regions, does not automatically detect defects, and does not assess structural safety.
