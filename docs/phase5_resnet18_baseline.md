# Phase 5 — ResNet-18 Transfer-Learning Baseline

## 1. Objective

Phase 5 establishes the standard transfer-learned CNN baseline for the curated WTBD crop benchmark. Phase 4 passed before this phase began. This is a controlled six-class defect-crop classification experiment, not full-image detection.

## 2. Dataset

The only input is `wtbd_crops_v1`, fingerprint `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`. The fixed source-isolated partitions contain 757 train, 146 validation, and 162 test crops. Source images never cross partitions. Exact Phase 3 224 × 224 RGB PNG pixels are loaded through the manifest; raw WTBD images are not read or recropped.

## 3. Architecture and pretrained provenance

The model is the unmodified torchvision ResNet-18 except that its 1,000-class head is replaced by one `512 → 6` linear layer. All layers are fine-tuned. Total and trainable parameter counts are both 11,179,590.

- PyTorch: `2.13.0+cpu`; torchvision: `0.28.0`; actual device: CPU.
- Explicit weight enum: `ResNet18_Weights.IMAGENET1K_V1`.
- Official cache filename: `resnet18-f37072fd.pth`; downloaded from the torchvision URL during Phase 5.
- Pretrained 1,000-class state fingerprint before head replacement: `78d60a5d12431f3233de6606575384b8b65c4c2c8bb4fc1b039001d0e0c1db57`.
- Apparatus commit recorded by the freeze: `0f08692b8bb3cd695242ff3edee2588b4e62c628`.

The fingerprint hashes sorted tensor names, dtypes, shapes, and contiguous CPU bytes. Published torchvision metadata and preprocessing metadata are preserved in `pretrained_backbone.json`.

## 4. Input preprocessing

Every split uses the same deterministic operation: convert RGB pixels to float32 in `[0,1]`, then normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. No resize, crop, flip, rotation, color change, blur, JPEG operation, or other augmentation is applied. This deliberate policy keeps the Phase 3 pixels common to handcrafted and learned methods and avoids conflating the baseline with later robustness work.

## 5. Class imbalance

`CrossEntropyLoss` weights are calculated from train only as `757 / (6 × N_c)`:

| Class | Train count | Float32 weight |
|---|---:|---:|
| craze | 123 | 1.025745273 |
| corrosion | 126 | 1.001322746 |
| surface_injure | 185 | 0.681981981 |
| thunderstrike | 42 | 3.003968239 |
| crack | 93 | 1.356630802 |
| hide_craze | 188 | 0.671099305 |

No weighted sampler, over/undersampling, or second imbalance correction is used.

## 6. Training

Training uses full FP32 AdamW for all parameters, batch size 32, validation batch size 64, `betas=(0.9,0.999)`, `eps=1e-8`, no scheduler, and no mixed precision. Maximum length is 30 epochs. Validation macro-F1 controls early stopping with patience 6 and minimum improvement 0.0001; balanced accuracy, macro recall, and earlier epoch resolve checkpoint ties. Python, NumPy, and PyTorch RNGs are seeded; deterministic algorithms are required; cuDNN benchmark is disabled; and `num_workers=0`.

## 7. Hyperparameter selection

Only seed 17, train, and validation entered the four-candidate search. No tuning candidate was evaluated on test.

| Candidate | Learning rate | Weight decay | Best epoch | Validation macro-F1 | Balanced accuracy | Macro recall |
|---|---:|---:|---:|---:|---:|---:|
| config_01 | 0.0001 | 0 | 4 | 0.857433 | 0.844692 | 0.844692 |
| config_02 | 0.0001 | 0.0001 | 4 | 0.857433 | 0.844692 | 0.844692 |
| config_03 | 0.0003 | 0 | 7 | 0.897388 | 0.883746 | 0.883746 |
| config_04 | 0.0003 | 0.0001 | 8 | 0.873299 | 0.861209 | 0.861209 |

Grid fingerprint: `162b6a86dea395bb59c9e3e72b297512f4ae69e41363edfe30241ab4c5d8a202`.

## 8. Frozen configuration

The deterministic selector chose learning rate `0.0003` and weight decay `0`. `configs/frozen/resnet18.yaml` was written before final test prediction and cannot be revised in response to test results.

## 9. Three-seed final experiment

Each seed starts again from the same official ImageNet state and trains on the same 757 crops. Seed 17 is a fresh final replicate, not a reused tuning checkpoint.

| Seed | Epochs | Best epoch | Validation macro-F1 | Training seconds | Checkpoint fingerprint |
|---:|---:|---:|---:|---:|---|
| 17 | 13 | 7 | 0.897388 | 2962.080 | `e87ca88dad134833a54171674332ea76920f957e7d7e1df10b9ec308ebeff8d8` |
| 29 | 9 | 3 | 0.871300 | 1730.863 | `53c4dd81213f271aa6b0e00c71dff7bad172bae3f5a560ba112bd6dc761b2848` |
| 43 | 11 | 5 | 0.855144 | 1995.253 | `320ce93b293e6b810c1c6e1c4e845a68443fcfaae24ca7875a0648c352c3e23f` |

## 10. Test performance

Each best checkpoint was evaluated once on test only after all three checkpoint files existed. Seeds estimate training variance; predictions are not ensembled. Standard deviations are sample SD (`ddof=1`).

| Seed | Macro-F1 | Balanced accuracy | Accuracy | Macro precision | Macro recall |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.911068 | 0.919369 | 0.901235 | 0.906522 | 0.919369 |
| 29 | 0.883807 | 0.884479 | 0.864198 | 0.892118 | 0.884479 |
| 43 | 0.891068 | 0.903580 | 0.901235 | 0.885600 | 0.903580 |
| Mean ± SD | 0.895314 ± 0.014118 | 0.902476 ± 0.017472 | 0.888889 ± 0.021383 | 0.894747 ± 0.010706 | 0.902476 ± 0.017472 |

## 11. Class-specific performance

| Class | Precision mean ± SD | Recall mean ± SD | F1 mean ± SD |
|---|---:|---:|---:|
| craze | 0.897815 ± 0.027501 | 0.962963 ± 0.037037 | 0.928511 ± 0.002555 |
| corrosion | 0.903948 ± 0.068088 | 0.800000 ± 0.033333 | 0.847535 ± 0.032036 |
| surface_injure | 0.887731 ± 0.061756 | 0.818182 ± 0.121212 | 0.845446 ± 0.047499 |
| thunderstrike | 0.933333 ± 0.115470 | 0.962963 ± 0.064150 | 0.947368 ± 0.091161 |
| crack | 0.852288 ± 0.024905 | 0.952381 ± 0.041239 | 0.898776 ± 0.003853 |
| hide_craze | 0.893364 ± 0.079878 | 0.918367 ± 0.020408 | 0.904251 ± 0.040824 |

The mean normalized confusion matrix is descriptive across seeds, not an ensemble. The largest mean off-diagonal patterns are corrosion → surface_injure (0.100), corrosion → hide_craze (0.078), and surface_injure → hide_craze (0.071). Thunderstrike has high mean F1 but also the largest F1 SD, consistent with its small support; no universal difficulty claim follows.

## 12. Computational efficiency

The canonical environment was Windows CPU on an AMD64 processor with eight logical CPUs; CUDA and cuDNN were unavailable. Every checkpoint is 44,794,379 bytes. Seed-17 batch-1 forward timing used 20 warm-ups and 100 measurements: median 0.106225 seconds, p95 0.128939 seconds, and 9.41395 images/s derived from the median. These values are descriptive for the recorded machine.

## 13. Comparison with traditional baselines

On the curated WTBD crop benchmark:

| Method | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| HOG + SVM | 0.477988 | 0.457766 | 0.530864 |
| LBP + SVM | 0.592401 | 0.572746 | 0.611111 |
| ResNet-18 three-seed mean | 0.895314 | 0.902476 | 0.888889 |

This controlled result does not establish universal superiority, human-inspector superiority, or industrial deployment performance.

## 14. Reproducibility

The full seed-17 final training was independently rerun. Best epoch, validation predictions, test predictions, test metrics, scientific history excluding time, and checkpoint fingerprint all matched exactly. Machine-readable configurations, histories, raw logits, predictions, metrics, matrices, fingerprints, environment, and equality checks are under `experiments/summaries/phase5_resnet18_v1/`. Regenerable `.pt` files remain ignored under `experiments/results/phase5_resnet18_v1/`.

## 15. Limitations

- This is visible defect-crop classification, not autonomous full-image detection.
- WTBD has no valid healthy class and supports no healthy-versus-damaged claim.
- WTBD has no temporal onset labels, so no early-detection claim is supported.
- The dataset remains relatively small and substantially imbalanced.
- Crops from one source photograph can be correlated, although source-level split isolation prevents cross-partition leakage.
- No augmentation was used in this core CNN baseline.
- Selection uses one validation split rather than extensive cross-validation.
- Conclusions are specific to the curated WTBD benchmark.
- No structural safety, repair, industrial deployment, or human-versus-AI superiority claim is supported.

## 16. Phase 5 exit gate

Phase 4 passed before Phase 5 began. The processed fingerprint and 757/146/162 split remain unchanged. Official weights, standard architecture, exact parameter count, deterministic preprocessing, train-only balanced loss, four-candidate validation selection, pre-test freeze, three final seeds, single primary test evaluations, per-seed/aggregate/class-level results, logits, histories, checkpoint fingerprints, efficiency measures, and exact seed-17 rerun all pass. The full suite reports 124 passed and 0 failed. No MobileNet, data-efficiency, augmentation, corruption, robustness, or Phase 6 experiment was started.
