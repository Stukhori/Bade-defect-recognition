# Phase 12 detector-to-application integration record

## Status and boundary

Phase 12A is preserved as the historical **apparatus and runtime-compatibility gate**. At that gate the detector had not been copied or integrated. The frozen apparatus remains
`configs/application_phase12.yaml`, and the identity-only probe record is
`provenance/phase12_runtime_compatibility.json`.

Phase 12B subsequently copied the exact authorized checkpoint byte-for-byte and implemented Application v3 as an experimental, human-reviewed region-proposal feature. Its separate configuration is `configs/application_phase12b.yaml`, its implementation record is `provenance/phase12b_implementation.json`, and its validator is `scripts/validate_phase12b.py`. Phase 12A evidence was not rewritten. A public Streamlit demonstration was deployed later at [https://bladescope.streamlit.app/](https://bladescope.streamlit.app/); deployment did not alter this frozen integration record or its scientific boundaries.

The planned feature name is **Experimental automatic region proposals**. It
must not be described as reliable automatic inspection. Proposed boxes must
remain reviewable, and the existing prepared-crop, manual single-region, and
manual multi-region workflows must remain available.

## Deployment-candidate selection

The single-model candidate is the frozen YOLO11n seed-17 checkpoint selected at
epoch 83:

- original Drive path: `runs/seed_17/weights/epoch82.pt`
- repository path authorized and later used by Phase 12B:
  `experiments/results/phase11b_yolo11n_v1/final/seed_17/epoch82.pt`
- size: 16,085,716 bytes
- SHA-256:
  `793547a5ec31954d8e909b2f5c63f378374134353a8d1e0cefdd452a5365eefa`

This Phase 12 productization choice uses only the frozen validation receipt.
Seed 17 had validation mAP@0.50:0.95 `0.43109`, compared with `0.42675` for
seed 29 and `0.40643` for seed 43. Held-out-test performance was not used for
deployment selection, and Phase 11B is not modified or reinterpreted.

## Internal and application class identities

The unchanged checkpoint is a detection model with exactly one internal class:

```yaml
checkpoint_internal_classes:
  0: item
```

Phase 11B is class-agnostic localization: annotation class ID 0 represents a
defect location. Phase 12 therefore freezes this sole presentation mapping:

```yaml
application_semantic_classes:
  0: defect_region_proposal
```

The mapping is presentation-only. It must not alter box coordinates, detector
scores, thresholding, NMS, or model execution. The internal label `item` must
never be presented to users. Class ID 0 must never be interpreted as healthy,
safe, or defect-free. Any other checkpoint class count, ID, or internal label
must fail closed.

## Runtime compatibility gate

The external checkpoint was tested read-only in an isolated short-path
environment using Python `3.11.15`, Ultralytics `8.3.150`, PyTorch
`2.13.0+cpu`, torchvision `0.28.0+cpu`, and the exact Application v2 CPU
dependency pins. CUDA was not enabled, and telemetry and undeclared trackers
were disabled.

The checkpoint loaded as a one-class detection model. Its training metadata
identified seed 17 and image size 640. One deterministic, in-memory synthetic
RGB image was used for a non-scientific smoke inference. No WTBD or project
image was used, no prediction details or timings were retained, and the
temporary environment was deleted afterward.

Frozen inference controls are:

- device: CPU
- image size: 640
- confidence threshold: `0.39`
- NMS IoU: `0.7`
- class-agnostic NMS: true
- maximum detections: 300
- `save=False`, `plots=False`, and `verbose=False`

No ensembling, checkpoint switching, threshold control, or user-adjustable NMS
is permitted.

## Implemented Application v3 flow

Application v3 draws numbered detector boxes and lists detector confidence separately. It requires the user to select proposals and confirm review before any proposed box is classified. Accepted boxes pass through the existing frozen contextual-crop preparation and then the frozen six-class MobileNet classifier. The three earlier prepared/manual workflows remain available. Detector confidence and classifier category scores remain separate in the UI and exports.

If no proposal crosses the frozen threshold, the exact message is:

> No region proposal exceeded the frozen threshold.

This outcome must never be interpreted as evidence of a healthy blade.

## Limitations and deployment boundary

The training and evaluation data contain no healthy/background-only images,
so healthy-blade false-positive behavior is unknown. The system does not
assess safety, severity, progression, or remaining life and is not established
as production-ready.

Phase 12B authorized only the local implementation and validation recorded here. It does not establish false-positive behavior on healthy blades, external-domain performance, production readiness, or safety fitness. The later public Streamlit demonstration does not expand that evidence or validation scope.

The implementation uses Ultralytics `8.3.150`, PyTorch `2.13.0+cpu`, and torchvision `0.28.0+cpu`. It verifies the checkpoint task, exact one-class metadata, seed, image-size metadata, byte size, and SHA-256 before prediction. The runtime is CPU-only, lazily cached, and configured without downloads, telemetry, external trackers, filesystem prediction output, ensembling, or user-adjustable detector controls. One deterministic synthetic in-memory smoke test used no project image and retained no coordinates or timings.
