# Phase 9A Grad-CAM reporting audit

## Finding

The reported defect was confirmed as a caption-only Pass B rendering error. The generator used `predicted_label` for every evidence caption even when `target_role` was `true_class`. The stored Grad-CAM target indices, target labels, arrays, heatmaps, overlays, and frozen prediction metadata were correct.

For P9A-001, every ResNet seed used true-class index `2` (`surface_injure`) and predicted-class index `0` (`craze`). The corrected page now labels the corresponding evidence `true_class: surface_injure` and `predicted_class: craze` for seeds 17, 29, and 43.

## Complete audit

- Grad-CAM manifest rows: 507.
- True-class target rows: 330; target-index or target-label mismatches against frozen ground truth: 0.
- Predicted-class target rows: 177; target-index or target-label mismatches against the corresponding frozen prediction: 0.
- Incorrect-prediction display identities: 177 across 52 review cases.
- Defective captions before repair: 177, all on `true_class` rows.
- Defective captions after repair: 0.
- Target-index/label counts across all rows: class 0/craze 93, class 1/corrosion 127, class 2/surface_injure 118, class 3/thunderstrike 30, class 4/crack 42, and class 5/hide_craze 97.
- Grad-CAM heatmap/overlay fingerprint before and after repair: `84e0dd6c21a6ff0cd92602f83b99e6b0e08c2d175cdc5d10c6c6602abf20c321`.
- Independent read-only recomputation: all 507 arrays were recalculated from the frozen checkpoints, stored inputs, and recorded target indices; 507/507 array hashes, activation shapes, and prediction-logit argmax values matched, and all six model-state fingerprints were unchanged.

Because all target identities were correct, no Grad-CAM array, heatmap, overlay, checkpoint, prediction, or quantitative result was regenerated or changed. Only the Pass B HTML captions and their recorded hashes required correction.

## Review-form handling

The completed Pass A form was preserved byte-for-byte at SHA-256 `3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e` with all 300 required responses intact.

The four Pass B responses entered from the defective page were invalidated. Pass B was regenerated as the exact blank 60-row form with `0/240` required responses and SHA-256 `7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8`.

## Corrected artifacts and safeguards

- Corrected Pass B page SHA-256: `ed8f3ef20449603e8e55647f3d917dbefbf2803d05ed696532859dde92201f29`.
- Corrected Phase 9A output fingerprint: `14e500fd94fa871bbe1e6bee6494d3158fe003c3799e0ccfbcab8e549adf80fa`.
- Machine-readable audit: `experiments/audits/phase9a_gradcam_reporting_audit.json`.
- Independent recomputation record: `experiments/audits/phase9a_gradcam_array_verification.json`.
- Validator assertions now require `target_role=true_class` to match frozen ground truth and `target_role=predicted_class` to match the corresponding frozen prediction, including both class index and label.
- The validator also verifies target-role coverage, target-bearing filenames, and every Pass B caption/asset association.

No image was interpreted, no human judgment was supplied, and Phase 9B was not started. Human review may resume with the corrected, blank Pass B form.
