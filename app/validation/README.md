# Application validation record

`validation.json` is the machine-readable Application v2 validation record, kept outside scientific result paths.

- Prepared-crop, manual single-region, and manual multi-region workflows: `PASS`.
- Stable region IDs and replace/remove/clear/new-image contracts: `PASS`.
- Session-only comparison plus in-memory JSON, CSV, and annotated-PNG export: `PASS`.
- Frozen Phase 3 contextual-crop pixel parity: `PASS`.
- Phase 6 checkpoint/state identity, six-score inference, and optional Grad-CAM parameter/prediction invariance: `PASS`.
- Frozen Phase 10 canonical-table identity and no-recomputation dashboard: `PASS`.
- Frozen Phase 11A readiness identity and `unsupported` automatic-integration gate: `PASS`.
- Automatic detector inference, training, tuning, calibration, test-set evaluation, external service calls, upload persistence, and tracked uploads: zero.
- Phase 10 and Phase 11A scientific-output fingerprints: unchanged.
- Focused Application v2 suite: 37 passed; complete repository suite: 266 passed with 11 unchanged scikit-learn future warnings.
- Live loopback health: HTTP 200 / `ok`; server stopped afterward.

The app uses Streamlit's local test driver and a loopback-only live health check. No UI screenshot is committed.

`deployment.json` separately validates the Streamlit Community Cloud contract: repository/branch/entrypoint, Python 3.11 selection, app-local pinned requirements, root configuration, tracked checkpoint and metadata, exact checkpoint SHA/state identity, CPU evaluation-mode loading, zero runtime artifact downloads, and no required secrets.
