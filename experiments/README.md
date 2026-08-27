# Experiment Artifact Policy

Every execution receives one isolated directory under `results/<experiment_id>/` containing its resolved configuration, manifest, result record, and log. Automatically generated run directories are ignored by Git so routine local validation does not pollute the worktree.

Small, intentionally reviewed aggregate JSON/CSV summaries may later be placed in `summaries/` and committed. Source configurations in `configs/`, split metadata in `data/splits/`, source code, tests, and documentation are also versionable. Large logs, checkpoints, raw images, and derived image caches are not versioned.

Phase 1 output is synthetic infrastructure evidence only and has no scientific interpretation.
