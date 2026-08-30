# App validation record

Validation completed locally on 2026-08-30. This directory is intentionally separate from the scientific result and summary paths.

- Machine-readable two-workflow record: `validation.json` — `PASS`.
- Prepared-region UI flow: upload, exact model-input preview, explicit classification, six scores, zero Streamlit errors — `PASS`.
- Manual-region UI flow: larger-image upload, rectangle component, display/original coordinate handling, contextual preview, explicit classification, six scores, zero Streamlit errors — `PASS`.
- Frozen training-reference parity: manual contextual crop pixels exactly equal the stored Phase 3 crop — `PASS`.
- Optional Grad-CAM: expected activation shape, unchanged model state, unchanged prediction — `PASS`.
- Focused app suite: 27 passed, 0 failed.
- Complete repository suite: 191 passed, 0 failed; 11 unchanged scikit-learn deprecation warnings.
- Phase 9A validator: `PASS`; checkpoints, predictions, and input fingerprints unchanged; both review forms blank.
- Live server: `http://127.0.0.1:8501/_stcore/health` returned HTTP 200 and body `ok`; server stopped afterward.
- External service calls, training, fine-tuning, calibration, test-set evaluation, permanent upload storage, and tracked uploaded files: zero.

No screenshots were generated or committed; the UI checks use Streamlit's supported local test driver plus a live server health check.
