# Robust Wind Turbine Blade Defect Recognition

This repository supports the experimental study **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**. Phase 3 has frozen and generated the deterministic classification crops and source-aware data-efficiency subsets from the reviewed Phase 2 WTBD curation. Its exit gate passes; no model results exist. The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md).

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

Runtime dependencies are PyYAML, NumPy, Pillow, matplotlib, scikit-image, scikit-learn, and joblib; pytest is the development/test dependency. PyTorch and torchvision are not required or installed.

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

## Phase boundaries

Phases 0–3 remain frozen, and Phase 4 is complete. HOG + SVM and LBP + SVM selected configurations and first test results are frozen. Phase 5 ResNet-18 work has not started.
