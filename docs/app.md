# BladeScope Application v3 — automatic proposals and frozen classification

## Status and boundary

BladeScope Application v3 is implemented and locally validated, but it has not been externally deployed. The interface presents **Auto detection** as its primary workflow while the scientific record correctly retains its status as an experimental, human-reviewed region-proposal aid. Integration does not change the detector model, checkpoint, operating threshold, NMS, metrics, or the frozen MobileNet crop classifier.

This is a research feature, not automatic inspection. The detector dataset contains no healthy/background-only images, so a proposal result cannot establish that a blade is healthy or defect-free. Every proposal must be reviewed and explicitly selected by a user before classification. Outputs do not assess safety, severity, progression, remaining life, or production readiness.

## Install and run

Use Python 3.11 and the separately pinned application dependencies:

```powershell
uv sync --extra dev
uv pip install -r requirements-app.txt
uv run streamlit run app/app.py --server.address 127.0.0.1
```

The local app and Streamlit deployment specifications pin Ultralytics `8.3.150`, PyTorch `2.13.0+cpu`, and torchvision `0.28.0+cpu` on supported non-macOS deployments. Both required checkpoints are tracked, so inference requires no runtime model download. Streamlit telemetry is disabled.

## Analysis modes

Analyze Image has four modes, with automatic detection presented first:

1. **Auto detection** lazily loads and caches the exact frozen one-class detector, generates numbered boxes, and displays each detector confidence separately.
2. **Prepared crop** applies the existing RGB/bilinear 224×224 preparation and frozen six-category classifier.
3. **Manual single region** maps one user rectangle to original-image coordinates and applies the frozen contextual-crop policy.
4. **Manual multi-region** preserves stable IDs, overlaps, replace, remove, clear, and new-image actions.

The automatic mode has no threshold, NMS, checkpoint, device, or model controls. A user must select proposals and confirm that the boxes were reviewed. Only accepted boxes enter the same frozen contextual-crop pipeline used by manual regions and then the unchanged six-category MobileNet classifier. Detector confidence remains separate from classifier category scores in the interface, JSON, and CSV.

If no proposal crosses the frozen operating threshold, the interface displays exactly:

> No region proposal exceeded the frozen threshold.

The interface links the user back to the manual-region workflows. The scientific limitation remains documented here: this outcome does not establish a healthy or defect-free blade.

## Detector identity and execution contract

- Model: YOLO11n, frozen detector seed 17, selected epoch 83.
- Checkpoint: `experiments/results/phase11b_yolo11n_v1/final/seed_17/epoch82.pt`.
- Size: 16,085,716 bytes.
- SHA-256: `793547a5ec31954d8e909b2f5c63f378374134353a8d1e0cefdd452a5365eefa`.
- Runtime: Ultralytics `8.3.150`, CPU only.
- Inference: image size 640, confidence threshold `0.39`, NMS IoU `0.7`, class-agnostic NMS, maximum 300 detections, with saving, plots, and verbose output disabled.
- One model only; no ensemble, download fallback, training path, tracker, or filesystem prediction output.

The checkpoint has one internal class. Its internal display metadata is validated but never shown in the UI. Application presentation maps class ID 0 only to **defect region proposal**; this presentation mapping does not alter coordinates, scores, thresholding, NMS, or execution.

## Frozen classifier and crop identity

The existing MobileNetV3-Small seed-17 checkpoint SHA-256 remains `9c7a5f18e7d05a320e1296c73bbeb9366636e0e55dc7c6ff2bab6d8808a0e5a5`, with state fingerprint `3c17629d1b1748e2f3d9046cb9a3d88c6369786acc1381f105974396c0f46757`. The contextual crop remains the frozen 1.5× square, 64-pixel minimum, boundary-shift-without-padding policy followed by RGB/bilinear resize to 224×224. Manual-mode regression tests preserve pixel parity and behavior.

## Session, exports, privacy, and limitations

Uploads, crops, records, proposal results, visualizations, and exports remain in process memory for the active session. JSON and CSV contain optional proposal ID and detector-confidence fields separate from classifier logits and scores. Annotated PNG export draws only saved user-selected or user-accepted regions. No upload or prediction is persisted to the server.

Research Results and Detection Readiness remain read-only views of frozen records. The detector integration does not change scientific results and provides no evidence about arbitrary operational imagery, healthy-blade false-positive behavior, hidden damage, structural integrity, safety, severity, progression, remaining life, or production readiness.

## Validation

```powershell
uv run pytest tests/test_app_inputs.py tests/test_app_crops.py tests/test_app_inference.py tests/test_app_v2.py tests/test_app_smoke.py tests/test_phase12b_integration.py
uv run python scripts/validate_phase12.py
uv run python scripts/validate_phase12b.py
uv run python scripts/validate_deployment.py
```

The prior apparatus remains verifiable as historical evidence. The implementation layer separately validates the tracked checkpoint, code, frozen controls, dependencies, provenance, and upstream hashes. One deterministic in-memory synthetic CPU smoke was performed with the committed checkpoint; no project image, prediction coordinates, or timing were retained. External deployment remains a separate user-controlled action and has not occurred.
