# Data Directory Policy

No dataset is present or downloaded in Phase 1.

- `raw/` is reserved for immutable original downloaded source data. Never edit, rename internally, overwrite, or transform raw dataset files in place.
- `interim/` is reserved for intermediate transformations that can be recreated from raw data and recorded configuration.
- `processed/` is reserved for derived, final model-ready data that can be recreated from raw data and recorded configuration.
- `splits/` is reserved for small, version-controlled split metadata. Future split files must preserve source-image grouping and provenance.

Raw, interim, and processed payloads are ignored by Git. Their directory placeholders and this policy are tracked. Small split definitions remain versionable. Dataset acquisition and exact WTBD layout are intentionally deferred to Phase 2.
