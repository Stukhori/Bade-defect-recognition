# Phase 12A detector-to-application integration apparatus

## Status and boundary

Phase 12A is an **apparatus and runtime-compatibility gate only**. The gate
passed, but the detector has not been copied into the repository or integrated
into Application v2. No deployment occurred. The frozen apparatus is
`configs/application_phase12.yaml`, and the identity-only probe record is
`provenance/phase12_runtime_compatibility.json`.

The planned feature name is **Experimental automatic region proposals**. It
must not be described as reliable automatic inspection. Proposed boxes must
remain reviewable, and the existing prepared-crop, manual single-region, and
manual multi-region workflows must remain available.

## Deployment-candidate selection

The single-model candidate is the frozen YOLO11n seed-17 checkpoint selected at
epoch 83:

- original Drive path: `runs/seed_17/weights/epoch82.pt`
- proposed future tracked path:
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

## Proposed application flow

A future, separately authorized implementation may pass reviewable detector
boxes into the existing frozen contextual-crop preparation and then the frozen
six-class MobileNet classifier. Detector confidence and classifier category
scores must remain separate.

If no proposal crosses the frozen threshold, the exact message is:

> No region proposal exceeded the frozen threshold.

This outcome must never be interpreted as evidence of a healthy blade.

## Limitations and next gate

The training and evaluation data contain no healthy/background-only images,
so healthy-blade false-positive behavior is unknown. The system does not
assess safety, severity, progression, or remaining life and is not established
as production-ready.

Phase 12A does not authorize checkpoint copying, dependency changes in the
application, detector invocation from the UI, application implementation, or
external deployment. Those actions require a separate validated integration
step.
