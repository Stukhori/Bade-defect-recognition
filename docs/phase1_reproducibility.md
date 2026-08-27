# Phase 1 Reproducibility Infrastructure

## Purpose

Infrastructure is established before dataset or model work so later runs can be reconstructed from a code commit, immutable resolved configuration, seed, dataset/split identity, and technical environment. Phase 1 uses synthetic data only and produces no scientific result.

## Reference environment

Python 3.11 is the frozen reference environment. Phase 1 is CPU-compatible and requires no CUDA installation. The package uses a `src/windblade` layout and minimal PyYAML and NumPy runtime dependencies; pytest is a development dependency. PyTorch and torchvision remain optional and are not required by the Phase 1 tests.

The implementation avoids unnecessary operating-system-specific behavior and is intended for Windows, Linux, and macOS where feasible.

## Configuration policy

Human-authored source configurations are YAML files under `configs/`. A source file may use the small `extends` mechanism to inherit a base YAML mapping. Loading resolves inheritance, validates required Phase 1 fields and value constraints, and rejects malformed structures with `ConfigError` messages.

The resolved mapping is deterministically serialized with sorted keys and fingerprinted with the first 12 hexadecimal characters of SHA-256. Key ordering does not affect the hash. Timestamps and other run-time metadata are not injected into the configuration, so transient values do not affect its hash. Intentional command-line overrides, including seed or output root, become part of the resolved configuration and therefore affect the hash.

Every run saves `resolved_config.yaml`. This immutable snapshot records the actual configuration used and prevents later edits to the source YAML from changing the historical meaning of a run.

## Random-seed policy

`set_global_seed(seed)` always seeds Python `random` and NumPy. If PyTorch is importable, it also seeds the CPU RNG, seeds available CUDA RNGs, requests deterministic algorithms with warnings, and disables cuDNN benchmarking while enabling cuDNN deterministic mode.

The applied settings are stored in the manifest. These controls support repeatability in a fixed environment; they do not guarantee bitwise-identical GPU behavior for every operation or across different hardware and library versions.

The smoke payload deliberately depends on the seed. Identical resolved configuration and seed produce the same deterministic payload, while a different seed changes the configuration hash and payload. Experiment IDs differ because they contain a creation timestamp.

## Experiment identity

An experiment ID has three readable components:

```text
<UTC timestamp with microseconds>_<sanitized experiment name>_<12-character config hash>
```

Microseconds make rapid repeated runs independently addressable. Experiment names are restricted to conservative path-safe characters before use.

## Environment provenance

Each manifest captures, when available:

- UTC capture time;
- operating system, release, and platform string;
- Python version and implementation;
- installed project version;
- hostname and standard-library CPU information;
- selected device;
- PyTorch and torchvision versions;
- CUDA availability and device count when PyTorch is installed;
- Git commit hash and dirty-worktree state.

Missing Git, commits, CUDA, PyTorch, or torchvision are represented with `null`, `false`, or an explicit import-status field rather than invented values. No username or unrelated personal information is recorded.

## Artifact layout

Every run is isolated under:

```text
experiments/results/<experiment_id>/
├── resolved_config.yaml
├── manifest.json
├── results.json
└── run.log
```

The manifest references artifact names relative to the run directory. `results.json` is deterministically written and remains the source of machine-readable output; numbers are not manually copied into documentation. Later aggregate CSV support is deferred until it is needed.

## Logging policy

Standard Python logging writes concise INFO messages to the console and `run.log`. Logs identify the experiment, configuration load, seed initialization, environment capture, major synthetic stages, completion or failure, and elapsed time. No third-party tracking platform is used.

## Failure policy

Exceptions inside an established run update `manifest.json` to `failed` when feasible, record the exception type and message, retain existing configuration/log artifacts, and propagate the exception so the command exits non-zero. Failed run directories are not silently deleted.

## Data immutability policy

`data/raw/` is reserved for original downloaded source data and is never modified in place. Interim and processed data must be reproducible derivatives. Small split metadata remains versionable. No dataset is downloaded in Phase 1; exact WTBD storage and acquisition are deferred to Phase 2.

## Version-control policy

Source code, tests, documentation, YAML configurations, small split definitions, and curated small machine-readable summaries are versionable. Virtual environments, caches, dataset payloads, checkpoints, autogenerated run directories, and large logs are ignored. This keeps reproducibility metadata intentional without committing large artifacts by accident.

## Test policy

The complete infrastructure suite is run with:

```bash
uv run python -m pytest
```

Tests use temporary directories and must not write to `experiments/results/`. Configuration validation/hashing, random seeding, JSON preservation, environment and manifest generation, failure evidence, artifact isolation, and smoke determinism must pass before Phase 2 can begin.
