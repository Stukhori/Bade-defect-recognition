# Data Directory Policy

WTBD version 1 was acquired in Phase 2 from the official Springer Nature Figshare release under CC BY 4.0. Its externally licensed image/XML/archive payload remains excluded from Git.

- `raw/` is reserved for immutable original downloaded source data. Never edit, rename internally, overwrite, or transform raw dataset files in place.
- `metadata/` contains small, versionable provenance, checksums, raw-audit tables, review decisions, the curation manifest, and raw/curated summaries derived without altering raw files.
- `interim/` is reserved for intermediate transformations that can be recreated from raw data and recorded configuration.
- `processed/` is reserved for derived, final model-ready data that can be recreated from raw data and recorded configuration.
- `splits/` is reserved for small, version-controlled split metadata. Future split files must preserve source-image grouping and provenance.

Raw, interim, and processed payloads are ignored by Git. Their directory placeholders and this policy are tracked. Metadata and small split definitions remain versionable. The Phase 2 curated split metadata is provisional and review-blocked; it is not a Phase 3 model-ready split. Final classification crops and model-ready preprocessing have not been created.
