# Robust Wind Turbine Blade Defect Recognition

This repository supports the experimental study **Robust Wind Turbine Blade Defect Recognition Under Limited Data and Image Degradation**. The project is currently in Phase 1: repository and reproducibility infrastructure. The scientific contract is frozen in [`docs/phase0_research_contract.md`](docs/phase0_research_contract.md); no scientific results exist yet.

## Reference environment and installation

The reference interpreter is Python 3.11. With [`uv`](https://docs.astral.sh/uv/) already installed, create the environment and install the minimal development dependencies:

```bash
uv sync --extra dev
```

The Phase 1 runtime dependencies are PyYAML and NumPy; pytest is the development/test dependency. PyTorch and torchvision are not required or installed by Phase 1.

## Run the complete test suite

```bash
uv run python -m pytest
```

## Run the synthetic smoke experiment

```bash
uv run python scripts/run_smoke_experiment.py --config configs/smoke.yaml
```

The smoke command uses only a tiny deterministic synthetic array. It does not download data or train a model. Each run writes `resolved_config.yaml`, `manifest.json`, `results.json`, and `run.log` under `experiments/results/<experiment_id>/`.

## Phase boundaries

Phase 0 research decisions remain frozen. Phase 1 builds engineering infrastructure only. WTBD acquisition, dataset auditing, preprocessing, feature extraction, model fitting, and robustness evaluation have not started.
