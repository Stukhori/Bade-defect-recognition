# Phase 3 — Curated Classification Dataset and Crop Preprocessing

## 1. Objective

Phase 3 defines the neutral common visual input for the four frozen classification methods. It transforms each retained Phase 2 annotation into one deterministic contextual crop and freezes leakage-safe instance splits and nested training-data subsets. It does not extract scientific HOG/LBP features, train any model, load pretrained weights, implement augmentation, or run robustness experiments.

## 2. Upstream provenance

- Raw WTBD fingerprint: `568c00e99f5ca8d205c5b48b3c058ca8f3b93d2e4de9986ec7d01af75b33babb`.
- Reviewed curation: `wtbd-curation-v1`.
- Phase 2 curation-manifest SHA-256: `9e5ce3b44457e52f686fb16f62df18a10a576262c9f0f89b96ccdd75d89c0767`.
- Curated input: 720 source images and 1,065 annotated objects.
- Curated source splits: 510 train, 101 validation, and 109 test images.
- Strict Phase 2 validation remained `PASS` before and after Phase 3 generation.

The raw-release audit remains historically incomplete because the official release contains documented inconsistencies. Phase 3 consumes the separate reviewed Phase 2 curation and does not rewrite that history or claim to reconstruct the published object totals.

## 3. Classification sample definition

One classification sample is one curated annotated defect instance. Stable IDs use `<source_image_id>_<object_index>`, where the object index is inherited from the validated annotation ordering. All crops from one source photograph inherit that photograph's curated split. Crops are never independently split.

The explicit label order in `classification_label_map.json` is:

0. `craze`
1. `corrosion`
2. `surface_injure`
3. `thunderstrike`
4. `crack`
5. `hide_craze`

Runtime alphabetical ordering is not used.

## 4. Crop geometry

For inclusive VOC box width `w` and height `h`:

```text
base_side    = max(w, h)
context_side = ceil(1.5 * base_side)
crop_side    = max(64, context_side)
crop_side    = min(crop_side, image_width, image_height)
```

The square is centered on the annotation as closely as possible. If the centered window crosses a source boundary, it is shifted inside the decoded image. It is never padded and the complete annotation must remain inside it. The generator preserves Phase 2's one-based inclusive VOC values in the manifest and explicitly records crop coordinates as zero-based half-open pixel bounds for Pillow. This conversion preserves the complete annotated pixel extent.

The 1.5 multiplier supplies moderate blade context while keeping the defect prominent. Square crops avoid distortion of elongated boxes; the 64-pixel minimum supplies real-image context for very small annotations. Shifting instead of padding guarantees that every crop pixel comes from the WTBD photograph. These choices are predeclared and may not be tuned from validation or test performance.

## 5. Resize and format

Each contextual square is converted to RGB and resized to 224 × 224 with Pillow's modern bilinear resampling API. It is saved as an 8-bit-per-channel lossless PNG with deterministic encoder settings. No enhancement, equalization, sharpening, denoising, grayscale conversion, normalization, augmentation, blur, or compression corruption is applied.

PNG prevents an additional uncontrolled lossy encoding stage before the later JPEG-robustness experiment. All four core methods must derive their input from these exact images; only representation-intrinsic later transformations, such as grayscale conversion for HOG/LBP or ImageNet normalization for CNNs, may differ.

## 6. Split inheritance

The instance counts produced by the frozen source-image split are:

| Split | Source images | Instances | craze | corrosion | surface_injure | thunderstrike | crack | hide_craze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 510 | 757 | 123 | 126 | 185 | 42 | 93 | 188 |
| validation | 101 | 146 | 19 | 22 | 46 | 9 | 24 | 26 |
| test | 109 | 162 | 27 | 30 | 33 | 9 | 14 | 49 |

Source-ID intersections across the three splits are empty. Every class occurs in every split, no excluded Phase 2 source occurs in the processed manifest, and no reviewed same-scene component crosses retained splits.

## 7. Training fractions

Data-efficiency fractions count curated **training source images**, not individual instances:

- 25%: 128 source images.
- 50%: 255 source images.
- 75%: 383 source images.
- 100%: all 510 training source images.

For every seed, inclusion is strictly nested: 25% ⊂ 50% ⊂ 75% ⊂ 100%. Selecting one source selects all of its defect instances. Validation and test membership never enters the selection objective and remains unchanged for every fraction.

## 8. Scientific seeds

The scientific subset seeds are `17`, `29`, and `43`. They govern seeded tie ordering in training-source subset construction. They are not yet model-initialization, visualization, or corruption seeds.

## 9. Grouped stratification

The deterministic greedy procedure first computes the full training set's six-class instance-count vector. At each target it extends the already selected source set, scoring every remaining source by the normalized squared deviation between the candidate cumulative class vector and the ideal vector for that source-image target. A large deterministic priority is assigned to filling any still-absent class. The scientific seed supplies tie ordering only. Selected sources are never removed as the target grows.

The summary below reports class counts in frozen label order (`craze/corrosion/surface_injure/thunderstrike/crack/hide_craze`). Distribution deviation is the L1 distance between subset and full-training class shares.

| Seed | Fraction | Sources | Instances | Per-class instances | L1 deviation | Manifest SHA-256 |
|---:|---:|---:|---:|---|---:|---|
| 17 | 0.25 | 128 | 252 | 39/40/66/12/28/67 | 0.070087 | `b231a06c1e90ce97090359e861df054c98e1084c46288571424851b37ab606dd` |
| 17 | 0.50 | 255 | 440 | 70/72/111/22/51/114 | 0.037258 | `ec9f49a5532a62afac49942d45fe02c214442d944a173a4954152d2bdb9c3a7f` |
| 17 | 0.75 | 383 | 608 | 98/100/151/32/73/154 | 0.017821 | `bd8d15b8214c52378f83cdf666cd46fc0411272fcc1967654d3c0280a47cb496` |
| 17 | 1.00 | 510 | 757 | 123/126/185/42/93/188 | 0 | `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95` |
| 29 | 0.25 | 128 | 252 | 39/40/66/12/28/67 | 0.070087 | `a2b427a6a682de42ef29280042c0ffbe862f80963b13dbb15aa5bfabf77db10c` |
| 29 | 0.50 | 255 | 440 | 70/72/111/22/51/114 | 0.037258 | `b6f5222161c60af4373413caf49409b12f0e29720b6dd035ee2e4aad2e620f8c` |
| 29 | 0.75 | 383 | 608 | 98/100/151/32/73/154 | 0.017821 | `3f2a2f67ca829e008d12e95f3526d8595aff0a61e2ae99a9acb2150a215fc21d` |
| 29 | 1.00 | 510 | 757 | 123/126/185/42/93/188 | 0 | `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95` |
| 43 | 0.25 | 128 | 252 | 39/40/66/12/28/67 | 0.070087 | `3b43aaefb03318b3b5349ffd81626b0f15dd1fc3eb7e3d40523a0e1cf6580f33` |
| 43 | 0.50 | 255 | 440 | 70/72/111/22/51/114 | 0.037258 | `5a66073adc48f8aa57c7c7711a86676540adfa5a6ee5e59d981d5cc874632a19` |
| 43 | 0.75 | 383 | 608 | 98/100/151/32/73/154 | 0.017821 | `6ee68e91f499b1e011ebbe17a97cf3f1bc8e314ba28de0331156ad7f17560005` |
| 43 | 1.00 | 510 | 757 | 123/126/185/42/93/188 | 0 | `32819d7250690290f9f7ea19325b053affbbf30bcaca21a3ea0fe5c4f2584b95` |

Different seeds produce different valid memberships at partial fractions even though this procedure achieves the same aggregate class-count vectors here. The 25% source subsets contain 252 instances (33.2893% of full training instances), the 50% subsets contain 440 (58.1242%), and the 75% subsets contain 608 (80.3170%).

## 10. Processed dataset statistics

The output contains exactly 1,065 crops with frozen class counts: craze 169, corrosion 178, surface injury 264, thunderstrike 60, crack 131, and hide craze 263.

| Metric | min | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bbox width | 20 | 43.4 | 117 | 220 | 376 | 754 | 1013 |
| bbox height | 23 | 50 | 126 | 249 | 431 | 878.8 | 1023 |
| bbox area | 800 | 4890.6 | 20400 | 55550 | 121660 | 320978 | 678219 |
| crop side | 64 | 138 | 330 | 537 | 801 | 1024 | 1024 |
| bbox/source area | 0.000763 | 0.004664 | 0.019455 | 0.052977 | 0.116024 | 0.306108 | 0.646800 |
| defect occupancy | 0.028451 | 0.059817 | 0.128937 | 0.239509 | 0.355987 | 0.426917 | 0.646800 |
| resize scale | 0.21875 | 0.21875 | 0.279650 | 0.417132 | 0.678788 | 1.623188 | 3.5 |

Four crops use the 64-pixel minimum, 568 require boundary shifting, and 167 have their requested contextual side clipped to the largest square that fits the real image. Clipping never truncates an annotation. Class-specific versions of every distribution are in `phase3_crop_statistics.csv`; no statistic is used to filter a sample.

## 11. QC findings

Five deterministic QC sheets were generated from training data only: six random examples per class, smallest boxes, largest boxes, most elongated boxes, and boxes closest to image boundaries. Visual review confirmed that the red annotation extents remain within the blue crop windows and that the derived panels contain real source pixels without visible padding. These sheets are technical preprocessing checks, not scientific results. No ordinary labeled validation/test contact sheet was generated.

## 12. Fingerprints and reproducibility

- Phase 3 resolved-config SHA-256: `e91f2026c3e6ac8dc75adf138014cf07a4e9d8907c638ae21fe52c799460b9b8`.
- Processed dataset fingerprint: `4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991`.

The fingerprint covers the Phase 3 config hash, Phase 2 curation-manifest hash, processed manifest, and sorted crop checksum manifest. Each manifest row records official source identity and checksum, object index, raw/canonical labels, class ID, original bounds, crop geometry, split, output path, and PNG checksum. Subset manifests and their hashes complete the lineage from official image to later fraction membership.

Regenerate with:

```bash
uv run python scripts/build_wtbd_crops.py --config configs/crop_dataset.yaml
uv run python scripts/build_training_subsets.py --config configs/crop_dataset.yaml
```

The PNG payload is intentionally ignored by Git; configs, manifests, checksums, summaries, split definitions, and QC evidence are versioned. A clean second build reproduced all 1,065 PNGs and 28 other Phase 3 artifacts byte-for-byte: 1,093/1,093 hashes matched.

## 13. Limitations

- Classification assumes an externally supplied region of interest and is not autonomous full-image defect localization.
- Multiple defect crops from one source image remain correlated; this motivates source-image-aware splitting and subsampling.
- Seventy-eight unresolved perceptual-similarity candidates remain within individual retained splits from Phase 2, while no retained pending candidate crosses train, validation, or test.
- Crop preprocessing is predeclared rather than optimized against test results.
- QC confirms technical crop construction only and does not establish classifiability or model performance.

## 14. Phase 3 exit status

**COMPLETE.** Phase 2 strict validation passes, all 1,065 classification samples and frozen split/subset artifacts validate, deterministic clean regeneration passes, and the full test suite passes. No model, feature experiment, pretrained weight, augmentation, corruption, or Phase 4 work was started.
