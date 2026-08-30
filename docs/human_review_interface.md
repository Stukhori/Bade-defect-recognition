# Local Phase 9A human-review interface

## Scope and launch

This separate Streamlit interface is a local, non-scientific data-entry aid for the existing Phase 9A packet. It does not run a classifier, LLM, heuristic, or inference function; suggest answers; expose the anonymous-ID mapping; or change any scientific result. Install the already pinned UI requirements, then launch from the repository root:

```powershell
uv pip install -r requirements-app.txt
uv run streamlit run app/review_app.py --server.address 127.0.0.1
```

The explicit loopback address keeps the service on the local computer. Streamlit telemetry is disabled by the repository configuration. The canonical review forms are edited in place; do not run multiple review-interface processes against the same forms.

## Pass A workflow

Pass A shows one anonymous case at a time with only the information already allowed by the frozen Pass A HTML: review ID, dataset true label, review image, and existing annotation visualization. It does not load or show model identity, seed, prediction, correctness, error event, Grad-CAM, Pass B, source sample ID, or the separate mapping.

For each case, select every required response and optionally add notes. Use `uncertain` rather than guessing. Dataset-label plausibility is not certainty, and an image can be visually ambiguous even when its label appears plausible. Use Previous, Save & Next, or Jump to first incomplete case. A case is complete only when all five required responses are valid.

When all 300 required answers are present, check this attestation exactly:

> I completed Pass A without opening Pass B or the separate ID mapping. I understand that locking Pass A prevents later changes after model evidence is revealed.

Then press **Validate and lock Pass A**. The interface validates all 60 rows, exact IDs, headers, order, and enum values; displays the completed Pass A SHA-256; and makes Pass A read-only for that interface session. It does not open Pass B automatically. Press **Begin Pass B** separately only when ready for model evidence. The scientific attestation must still be repeated when Phase 9B begins; this in-session lock is a workflow safeguard, not proof of reviewer behavior.

## Pass B workflow

Only after the completed, attested Pass A lock and the separate Begin Pass B action does the interface load the existing Pass B page and images. It then shows the already generated model/seed/prediction/event information and Grad-CAM evidence for one anonymous case at a time.

Grad-CAM maps are independently normalized visualizations. Their colors cannot be compared quantitatively across maps, and they do not prove what caused a prediction. Apparent annotation overlap does not establish causal reasoning. Complete all four required fields per case, or 240 required Pass B answers total, and use `uncertain` instead of guessing.

## Frozen response schemas

Pass A uses these exact configured values:

| CSV field | Allowed values |
|---|---|
| `defect_visible` | `yes`, `partially`, `no`, `uncertain` |
| `corruption_obscures_diagnostic_detail` | `none`, `mild`, `strong`, `uncertain` |
| `dataset_label_visually_plausible` | `yes`, `no`, `uncertain` |
| `visually_ambiguous_between_categories` | `yes`, `no`, `uncertain` |
| `possible_crop_or_background_problem` | `yes`, `no`, `uncertain` |
| `reviewer_notes` | optional free text |

Pass B uses these exact configured values:

| CSV field | Allowed values |
|---|---|
| `activation_primarily_inside_annotation` | `inside`, `partial`, `outside`, `diffuse`, `uncertain` |
| `activation_concentrated_on_degradation_artifact` | `yes`, `no`, `uncertain` |
| `pattern_consistent_across_cnn_seeds` | `yes`, `partly`, `no`, `uncertain` |
| `prediction_visually_understandable_after_reveal` | `yes`, `no`, `uncertain` |
| `reviewer_notes` | optional free text |

## Saving, recovery, and final verification

Every valid changed response is autosaved as UTF-8 CSV through a validated same-directory temporary file and atomic replacement. Notes safely preserve commas, quotes, and Unicode. The interface shows required-answer counts and the last successful save time. Closing the browser does not discard saved answers: reopening resumes at the first incomplete case. Because the lock is intentionally transient, a restarted session with complete Pass A requires the attestation and lock action again before Pass B can be opened; a partially completed Pass B then resumes at its first incomplete case.

To inspect the current forms without changing them:

```powershell
uv run python scripts/validate_review_interface.py --check-forms-only
```

Final completion requires Pass A `300/300` and Pass B `240/240`, with `complete: true` for both. Before returning the forms, close the review interface, rerun that command, and retain the reported hashes. Return the two completed canonical CSVs for Phase 9B. Do not manually commit completed human-review forms; Phase 9B must validate and incorporate them under its explicit protocol.

The exact Phase 9B attestation sentence is:

> I completed Pass A without opening Pass B or the separate ID mapping. I understand that locking Pass A prevents later changes after model evidence is revealed.
