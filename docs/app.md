# Frozen Classifier Demonstration

## Status and boundary

This is a separate, non-scientific local demonstration of the already frozen classifier. It does not start Phase 9B, complete Phase 9, or start Phase 10–12. Scientific Phases 0–8 remain frozen, and Phase 9A remains complete but awaiting human review.

The application is a non-scientific demonstration of the frozen classifier. It classifies manually identified visible defect regions and does not automatically detect defects or assess blade safety.

It must not be used to claim automatic defect detection or localization, blade condition or safety assessment, hidden/internal damage recognition, severity, remaining useful life, earlier detection, calibrated probabilities, replacement of inspection professionals, deployment performance, or generalization beyond the research evidence.

## Frozen model identity

- Model: MobileNetV3-Small.
- Scientific origin: Phase 6 full-data run, seed 17.
- Selection rationale: seed 17 is the predeclared canonical seed; the app did not choose the seed by test performance.
- Architecture: `torchvision_mobilenet_v3_small` with six outputs.
- Class order: `craze`, `corrosion`, `surface_injure`, `thunderstrike`, `crack`, `hide_craze`.
- Checkpoint: `experiments/results/phase6_mobilenet_v3_small_v1/final/seed_17/best_state_dict.pt`.
- Checkpoint file SHA-256: `9c7a5f18e7d05a320e1296c73bbeb9366636e0e55dc7c6ff2bab6d8808a0e5a5`.
- Checkpoint state fingerprint: `3c17629d1b1748e2f3d9046cb9a3d88c6369786acc1381f105974396c0f46757`.
- Processed-dataset fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.
- Runtime: CPU, evaluation mode, `torch.inference_mode()` for ordinary inference.

The app verifies the checkpoint file hash, metadata, dataset fingerprint, architecture, seed, class order, and decoded state fingerprint before use. It never trains, fine-tunes, calibrates, ensembles, rewrites, or selects a model.

## Install and run

Use Python 3.11. Install the frozen scientific environment first, then the separately pinned app-only dependencies:

```powershell
uv sync --extra dev
uv pip install -r requirements-app.txt
uv run streamlit run app/app.py
```

The local Phase 6 seed-17 checkpoint and its JSON metadata must already exist at the path above. These large scientific result artifacts remain excluded from Git under the repository's established policy. Once the repository, environment, checkpoint, and local image are present, the app needs no network access. Streamlit usage telemetry is disabled in `.streamlit/config.toml`.

## Input workflows

### Prepared visible defect crop

The user supplies one already identified visible defect region. The app applies EXIF orientation, converts to RGB, resizes to 224×224 with Pillow bilinear resampling, applies the frozen ImageNet normalization, and runs the frozen classifier. This workflow does not locate the region.

### Larger image with manual region selection

The user draws a free-aspect rectangle on a display-only image. The pinned cropper component is prevented from applying its own resize. App-owned code maps the display rectangle back to zero-based, half-open original-image pixels using the known display/original scale, clamps it to the image, and rejects non-positive selections.

The mapped box is converted to the Phase 3 one-based inclusive VOC convention and passed to the existing `windblade.data.crops.calculate_square_crop` function. The frozen policy is:

- square side `ceil(1.5 × max(box width, box height))`;
- minimum side 64 pixels where the source dimensions permit;
- shift the square inside the real image while preserving the full selected box;
- no padding;
- RGB and Pillow bilinear resize to 224×224.

The annotated original, original-pixel coordinates, contextual geometry flags, and exact 224×224 model input are shown before the explicit classification button is used.

## Output interpretation

The app shows one predicted category and all six softmax model scores. The interface labels them exactly as **“Model scores — not calibrated confidence estimates.”** Scores are not calibrated probabilities and do not establish safety, severity, or the presence or absence of other defects.

Request inference, preprocessing, one-time model loading, and optional Grad-CAM time are local descriptive engineering measurements only. They are not scientific efficiency results or target-hardware deployment claims.

## Optional Grad-CAM

The optional button calls the unchanged Phase 9A Grad-CAM primitive at MobileNet layer `features.12`. The wrapper checks the expected `1×576×7×7` activation shape, verifies the model state before and after generation, clears gradients, restores evaluation mode, and tests prediction invariance.

The visualization is an activation map for the selected class score. It is not an automatic detector, a defect localization output, a causal explanation, or a safety assessment.

## Input safety and privacy

- Accepted extensions and decoded formats: PNG, JPG, JPEG.
- Maximum encoded upload: 15 MB.
- Maximum decoded size: 50 megapixels and 20,000 pixels on either dimension.
- Empty, corrupt, mislabeled, unsupported, non-finite-coordinate, zero-area, and out-of-range inputs are rejected with user-facing messages.
- Grayscale and RGBA images are converted to RGB; EXIF orientation is applied.
- Upload bytes remain in memory. The app exposes no arbitrary filesystem path, writes no upload, uses no analytics or API key, and calls no external service.
- Replacing an upload clears the stored crop, classification, and optional visualization for that workflow.

## Validation

Run the focused app tests:

```powershell
uv run python -m pytest tests/test_app_inputs.py tests/test_app_crops.py tests/test_app_inference.py tests/test_app_smoke.py
```

Run the read-only two-workflow validation:

```powershell
uv run python scripts/validate_app.py
```

The tests cover upload safety, EXIF/RGB handling, display-to-original coordinate mapping, edge and elongated boxes, byte-identical pixels for multiple frozen Phase 3 references, frozen normalization, checkpoint identity, stored reference logits, all-six-score output, Grad-CAM parameter/prediction invariance, and Streamlit startup. The validator uses one training-partition reference only and performs no test-set evaluation.

The application-specific validation record is stored separately from all scientific summary paths at `app/validation/validation.json`. Its 2026-08-30 run passed both workflows, Phase 3 pixel parity, frozen checkpoint identity, six-score output, and optional Grad-CAM invariance. The focused suite passed 27 tests and the complete repository suite passed 191 tests with the 11 existing scikit-learn deprecation warnings. The frozen Phase 9A validator also passed with unchanged checkpoints, predictions, inputs, and blank review forms. A live local Streamlit process returned HTTP 200 with health body `ok` and was then stopped.
