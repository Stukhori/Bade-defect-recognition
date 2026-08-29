# Robust Wind Turbine Blade Defect Recognition

This repository supports the experimental study **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**. Phases 0–6 are complete and frozen. Results include the Phase 4 handcrafted baselines and matched Phase 5/6 ResNet-18 and MobileNetV3-Small baselines. The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md).

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

Runtime dependencies include PyYAML, NumPy, Pillow, matplotlib, scikit-image, scikit-learn, joblib, PyTorch, and torchvision; pytest is the development/test dependency.

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

## Phase boundaries

Phases 0–6 are complete and frozen. Phase 7 — Data-Efficiency Experiment — has not been started.
