# Robust Wind Turbine Blade Defect Recognition

This repository supports the experimental study **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**. Phase 2 acquired and forensically audited the official WTBD release and now provides a non-destructive reconciliation layer. Its exit gate remains `BLOCKED_PENDING_HUMAN_REVIEW`; no model results exist. The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md).

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

Runtime dependencies are PyYAML, NumPy, Pillow, and matplotlib; pytest is the development/test dependency. PyTorch and torchvision are not required or installed.

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

Use `--strict` to return a nonzero status while human-review blockers remain. Decisions belong in the versioned identity and near-duplicate decision CSVs; rerunning review preserves those files. See [`docs/phase2_curation.md`](docs/phase2_curation.md).

## Phase boundaries

Phases 0 and 1 remain frozen. Phase 2 is incomplete pending 262 identity decisions and 491 non-exact near-duplicate decisions. Exact-copy leakage is controlled in the provisional manifest. Phase 3 preprocessing, feature extraction, model fitting, and robustness evaluation have not started.
