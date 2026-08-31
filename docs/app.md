# Application v2 — frozen classifier research workspace

## Status and boundary

Application v2 is a separate, non-scientific local interface around already frozen assets. It adds product workflows and read-only dashboards; it does not alter or extend any scientific experiment. Phase 10 and Phase 11A remain frozen. Phase 11B detector training is blocked and unstarted, so automatic localization remains unavailable. Phase 12 has not started.

Every active analysis path requires a user-supplied crop or rectangle. The software cannot establish that a defect exists, establish that an image is defect-free, assess hidden damage, structural integrity, severity, remaining service life, or operational safety, or replace professional inspection.

## Install and run

Use Python 3.11 and the separately pinned application dependencies:

```powershell
uv sync --extra dev
uv pip install -r requirements-app.txt
uv run streamlit run app/app.py --server.address 127.0.0.1
```

The frozen Phase 6 checkpoint and metadata must already exist at `experiments/results/phase6_mobilenet_v3_small_v1/final/seed_17/`. Once the environment and local artifacts are present, the app needs no network access. Streamlit telemetry is disabled in `.streamlit/config.toml`.

## Navigation and workflows

The sidebar exposes six sections: Home, Analyze Image, Compare Regions, Research Results, Detection Readiness, and About and Limitations.

Analyze Image has three active modes:

1. **Prepared crop** applies EXIF orientation, RGB conversion, Pillow bilinear resize to 224×224, frozen ImageNet normalization, and frozen classifier inference.
2. **Manual single region** maps one display rectangle back to original-image, zero-based half-open coordinates and reuses the exact Phase 3 contextual crop policy.
3. **Manual multi-region** saves and independently classifies any number of user rectangles, including overlaps. Stable IDs (`R1`, `R2`, …) support replace, remove, clear-regions, and new-image actions.

The Phase 3 manual policy uses a square side of `ceil(1.5 × max(box width, box height))`, a 64-pixel minimum where source dimensions permit, boundary shifting without padding, then RGB/bilinear resize to 224×224. Regression tests confirm pixel identity with canonical Phase 3 crops.

## Session comparison and exports

Classification records exist only in the active Streamlit session. Compare Regions shows thumbnails, predicted category, all six scores, source and crop coordinates, preprocessing/inference timing, and Grad-CAM status. Optional Grad-CAM uses the unchanged Phase 9A primitive and remains an activation visualization—not detector evidence or a causal explanation.

The app generates three downloads in memory:

- JSON with application/checkpoint/preprocessing identity, UTC timestamps, source dimensions and hashes, coordinates, logits, all six scores, and limitations;
- CSV with one row per saved region and all six scores;
- PNG with only the user-drawn manual boxes and stable region IDs over the selected source image.

No upload, session record, visualization, or export is written to the server filesystem.

## Frozen research dashboards

Research Results reads and verifies canonical Phase 10 CSV tables for clean method comparison, data efficiency, robustness retention, and error/human-review summaries. It does not recompute metrics. The required Phase 10 scientific-output fingerprint is `6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0`.

Detection Readiness reads the frozen Phase 11A manifest, audit summary, feasibility decisions, compute gate, and reproducibility record. It reports 720 curated full images, 1,065 boxes, zero healthy/background images, the `unsupported` application-integration decision, and the CPU/CUDA block. The Phase 11A scientific-output fingerprint is `3f46cbdc6c7a2e3cf6093ff177dd1948d113fa4c36fa9eb907d7c8621e800461`.

`windblade_demo.detection_status` defines a future-facing box/result interface, but `load_detector()` and `detect()` explicitly raise `DetectorUnavailableError`. It does not import a detector runtime, download weights, use annotations as predictions, or fabricate boxes.

## Frozen classifier identity

- Model: MobileNetV3-Small, Phase 6 full-data seed 17 (the predeclared canonical seed).
- Checkpoint file SHA-256: `9c7a5f18e7d05a320e1296c73bbeb9366636e0e55dc7c6ff2bab6d8808a0e5a5`.
- State fingerprint: `3c17629d1b1748e2f3d9046cb9a3d88c6369786acc1381f105974396c0f46757`.
- Processed dataset fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Class order: `craze`, `corrosion`, `surface_injure`, `thunderstrike`, `crack`, `hide_craze`.
- Runtime: CPU evaluation mode; `torch.inference_mode()` for normal inference.

All six softmax outputs are labeled as model scores, not calibrated confidence estimates. The app verifies the checkpoint file, metadata, dataset identity, architecture, seed, class order, and decoded state before use. It never trains, tunes, calibrates, ensembles, selects, or rewrites a model.

## Input safety and privacy

PNG/JPG/JPEG uploads are limited to 15 MB, 50 megapixels, and 20,000 pixels on either dimension. EXIF orientation is applied; grayscale and alpha inputs are converted to RGB. Empty, corrupt, mislabeled, oversized, invalid-coordinate, and zero-area inputs are rejected. There is no arbitrary path input, analytics, API key, external service, upload persistence, or global cache of user images.

## Validation

```powershell
uv run pytest tests/test_app_inputs.py tests/test_app_crops.py tests/test_app_inference.py tests/test_app_v2.py tests/test_app_smoke.py
uv run python scripts/validate_app.py --output app/validation/validation.json
uv run python scripts/run_detection.py --validate-only
```

The machine-readable application record verifies all three workflows, stable region IDs, session operations, JSON/CSV/PNG exports, Phase 3 pixel parity, checkpoint identity, reference inference, Grad-CAM invariance, Phase 10 source hashes, Phase 11A readiness identity, privacy controls, and scientific invariance. The focused Application v2 suite passes 37 tests; the complete repository suite passes 266 tests with 11 unchanged scikit-learn future warnings. Every read-only Phase 2–11A/app/review validator passes. Live loopback health returned HTTP 200 with body `ok`, then the server was stopped.

No screenshot or uploaded user image is tracked. No detector dependency, detector checkpoint, threshold, NMS setting, prediction, metric, or external deployment exists.
