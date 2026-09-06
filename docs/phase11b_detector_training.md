# Phase 11B detector training and held-out-test freeze

## Final scientific status

Phase 11B is **complete, validated, and frozen**. The scientific task was
class-agnostic localization with YOLO11n. The final status is
`FINAL_TEST_COMPLETE_NO_FURTHER_TUNING`.

The only authoritative source for held-out-test numbers is the committed
machine-readable result:
`provenance/phase11b_final_test_metrics.json`. Its SHA-256 is
`e6bac76aa5a2e3d68ed7b93f5be180228addba1f1f90ddafab398d198496bcbb`.
The values below reproduce that frozen record without additional computation,
reinterpretation, or rounding.

## Frozen detector and execution environment

- Detector: YOLO11n
- Ultralytics: `8.3.150`
- PyTorch: `2.6.0+cu124`
- torchvision: `0.21.0+cu124`
- Python: `3.11.13`
- GPU: Tesla T4

The frozen training controls were image size 640, batch size 16, AdamW,
initial learning rate `0.001`, and a cosine learning-rate schedule. Seeds 17,
29, and 43 each completed 100 epochs. No automatic batch sizing or automatic
optimizer selection was used.

## Validation-only selection

Checkpoint and operating-threshold selection used validation evidence only.
The selected epochs were:

| Seed | Selected epoch |
| ---: | -------------: |
| 17 | 83 |
| 29 | 86 |
| 43 | 81 |

The pooled validation-selected operating threshold is `0.39`, with validation
F1 `0.6567967698519516`. The threshold, checkpoints, and locked NMS policy were
frozen before the held-out test. Test results must not be used for further
tuning or checkpoint selection.

## Frozen held-out-test result

| Seed | Test mAP@0.50:0.95 |
| ---: | ------------------: |
| 17 | 0.3243185659488687 |
| 29 | 0.3338538682270522 |
| 43 | 0.3198505104055161 |

Aggregate test mAP@0.50:0.95 is
`0.32600764819381234 ± 0.007152849550503072` sample SD across the three
declared seeds.

## Frozen identities

- Configuration SHA-256:
  `fc0ab33a25bafb5b92da88f67343bca9bbcecb6c715d867b06f4ac74f90cff1b`
- Validation-selection receipt SHA-256:
  `6c236e9d7220b443f17a628a3d8f621afc56be3777949eeca47f462879e46509`
- Final held-out-test metrics SHA-256:
  `e6bac76aa5a2e3d68ed7b93f5be180228addba1f1f90ddafab398d198496bcbb`

These identities bind the frozen configuration, validation-selected
checkpoints and threshold, locked NMS settings, and final metrics. The final
status prohibits post-test tuning.

## Limitations and phase boundary

The detection dataset contains no healthy or background-only images.
Consequently, these results do not measure false-positive behavior on healthy
blades. They do not support claims about arbitrary operational inspection,
safety, defect severity or progression, remaining life, or production
readiness.

Application v2 remains unchanged and does not automatically localize defects.
Any detector integration or external validation belongs to Phase 12, which
must be separately specified, authorized, and validated. Phase 12 remains
unstarted.
