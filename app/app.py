"""Application v2: local analysis workspace around frozen scientific assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_cropper import st_cropper

from windblade_demo.constants import (
    APPLICATION_VERSION, CHECKPOINT_STATE_FINGERPRINT, CLASS_DESCRIPTIONS,
    CLASS_LABELS, HUMAN_LABELS, MODEL_DISPLAY_NAME, PREPROCESSING_CONTRACT,
)
from windblade_demo.crops import (
    SelectionValidationError, annotated_selection, contextual_crop, display_image,
    map_display_box, prepare_region,
)
from windblade_demo.detection_status import DetectorUnavailableError, load_detection_status
from windblade_demo.explain import generate_gradcam
from windblade_demo.exports import annotated_image_export, csv_export, json_export
from windblade_demo.inference import FrozenModelError, infer, load_frozen_model
from windblade_demo.inputs import UploadValidationError, decode_upload
from windblade_demo.research import FrozenResearchError, load_phase10
from windblade_demo.session import (
    RegionRecord, make_region_record, remove_region, replace_region, with_gradcam,
)
from windblade_demo.visualization import annotate_regions


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = (
    "Home", "Analyze Image", "Compare Regions", "Research Results",
    "Detection Readiness", "About and Limitations",
)
ANALYSIS_MODES = ("Prepared crop", "Manual single region", "Manual multi-region")

st.set_page_config(
    page_title="Wind Turbine Blade Defect Recognition", page_icon="🔎", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp {
        color: #102a43;
        background:
            radial-gradient(circle at 88% 4%, rgba(112, 190, 238, .20), transparent 24rem),
            linear-gradient(180deg, #edf6fd 0%, #f8fbfe 38%, #f2f7fb 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e5f1fb 0%, #f4f8fc 68%, #eaf3fb 100%);
        border-right: 1px solid #c8ddec;
    }
    .block-container { max-width: 1280px; padding-top: 1.2rem; padding-bottom: 3rem; }
    .hero { position: relative; overflow: hidden; padding: 1.65rem 1.8rem;
            border-radius: 22px; color: white;
            background: linear-gradient(125deg, #0b3158 0%, #0d5f9f 58%, #49a4dc 100%);
            box-shadow: 0 16px 38px rgba(18, 76, 122, .22); margin-bottom: 1rem; }
    .hero::before { content: ""; position: absolute; width: 19rem; height: 19rem;
                    right: -5rem; top: -11rem; border-radius: 50%;
                    border: 2px solid rgba(255,255,255,.20);
                    box-shadow: 0 0 0 2.7rem rgba(255,255,255,.055),
                                0 0 0 5.6rem rgba(255,255,255,.035); }
    .hero::after { content: ""; position: absolute; width: 22rem; height: 4rem;
                   right: 2rem; bottom: -2.1rem; border-radius: 50%;
                   border-top: 2px solid rgba(255,255,255,.24);
                   transform: rotate(-7deg); }
    .hero > * { position: relative; z-index: 1; }
    .hero h1 { margin: .2rem 0 0; font-size: clamp(1.85rem, 4vw, 2.6rem); line-height: 1.1; }
    .hero p { margin: .65rem 0 0; max-width: 840px; opacity: .94; font-size: 1.02rem; }
    .badge { display: inline-block; padding: .28rem .58rem; margin: 0 .32rem .3rem 0;
             border-radius: 999px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
             font-size: .72rem; font-weight: 750; letter-spacing: .055em; }
    .notice { border-left: 5px solid #2b8fd2; background: #eaf5fd; padding: .9rem 1rem;
              border-radius: 10px; margin: .75rem 0 1.2rem; color: #153d5c; }
    .card { background: rgba(255,255,255,.90); border: 1px solid #c9deed; border-radius: 15px;
            padding: 1rem 1.15rem; box-shadow: 0 6px 20px rgba(23,75,116,.09); }
    .eyebrow { color: #116eaf; font-size: .76rem; font-weight: 800; letter-spacing: .08em; }
    .status-ok { border-left: 4px solid #2589c9; background: #e8f4fc; padding: .8rem 1rem; border-radius: 10px; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.90); border:1px solid #c9deed; padding:.7rem; border-radius:12px; }
    div[data-testid="stFileUploader"] { background:rgba(255,255,255,.90); border:1px solid #c9deed; border-radius:14px; padding:.55rem .8rem; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; }
    @media (max-width: 700px) { .hero { padding:1.15rem; border-radius:14px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Verifying and loading the frozen checkpoint…")
def cached_model():
    return load_frozen_model(ROOT)


@st.cache_data(show_spinner=False)
def cached_research():
    return load_phase10(ROOT)


@st.cache_data(show_spinner=False)
def cached_detection_status():
    return load_detection_status(ROOT)


def initialize_session() -> None:
    st.session_state.setdefault("analysis_records", [])
    st.session_state.setdefault("source_images", {})
    st.session_state.setdefault("multi_uploader_nonce", 0)
    st.session_state.setdefault("next_region_number", 1)


def records() -> list[RegionRecord]:
    return st.session_state["analysis_records"]


def save_source(decoded: Any) -> None:
    st.session_state["source_images"][decoded.byte_sha256] = decoded.image.copy()


def add_record(record: RegionRecord) -> None:
    st.session_state["analysis_records"] = [*records(), record]


def score_rows(record: RegionRecord) -> list[dict[str, Any]]:
    return [
        {"Category": HUMAN_LABELS[label], "Model score": record.scores[index]}
        for index, label in enumerate(CLASS_LABELS)
    ]


def render_scores(record: RegionRecord, *, key: str) -> None:
    rows = score_rows(record)
    st.subheader(HUMAN_LABELS[record.predicted_label])
    st.caption(f"{record.region_id} · model scores, not calibrated confidence estimates")
    st.vega_lite_chart(
        data=rows,
        spec={"mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#08766f"},
         "encoding": {
             "x": {"field": "Model score", "type": "quantitative", "scale": {"domain": [0, 1]}},
             "y": {"field": "Category", "type": "nominal", "sort": "-x"},
             "tooltip": ["Category", {"field": "Model score", "format": ".6f"}]},
         "height": 210},
        width="stretch", key=f"scores_{key}",
    )


def classify_record(
    *, mode: str, decoded: Any, model_input,
    selected_box: tuple[int, int, int, int] | None = None,
    contextual_box: tuple[int, int, int, int] | None = None,
    replacement_id: str | None = None,
) -> RegionRecord:
    result = infer(cached_model(), model_input)
    chosen_id = replacement_id
    if chosen_id is None:
        chosen_id = f"R{st.session_state['next_region_number']}"
    record = make_region_record(
        records=records(), mode=mode, source_name=decoded.filename,
        source_sha256=decoded.byte_sha256, source_size=decoded.image.size,
        model_input=model_input, result=result, selected_box=selected_box,
        contextual_box=contextual_box, region_id=chosen_id,
    )
    if replacement_id is None:
        st.session_state["next_region_number"] += 1
    return record


def render_hero(title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div><span class="badge">APPLICATION v{APPLICATION_VERSION}</span>'
        '<span class="badge">LOCAL PROCESSING</span><span class="badge">FROZEN CLASSIFIER</span></div>'
        f'<h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True,
    )


def render_scope_notice() -> None:
    st.markdown(
        '<div class="notice"><strong>Region-based analysis.</strong> '
        'Choose a prepared crop or draw one or more rectangles on an image. The classifier '
        'evaluates each region you supply.</div>',
        unsafe_allow_html=True,
    )


def go_to(page: str) -> None:
    st.session_state["navigation"] = page


def go_to_analysis(mode: str) -> None:
    st.session_state["analysis_mode"] = mode
    st.session_state["navigation"] = "Analyze Image"


def render_home() -> None:
    render_hero(
        "Blade image research workspace",
        "Explore a frozen six-category crop classifier, compare your own regions, export this browser session, and inspect the project's frozen research evidence.",
    )
    render_scope_notice()
    st.markdown("### Start an analysis workflow")
    first, second, third = st.columns(3)
    with first:
        st.markdown('<div class="card"><div class="eyebrow">PREPARED</div><h3>Classify a crop</h3><p>Use an image already centered on one visible region.</p></div>', unsafe_allow_html=True)
        st.button("Analyze prepared crop", type="primary", width="stretch", on_click=go_to_analysis, args=("Prepared crop",))
    with second:
        st.markdown('<div class="card"><div class="eyebrow">SINGLE</div><h3>Draw one region</h3><p>Select and classify one rectangle on a larger image.</p></div>', unsafe_allow_html=True)
        st.button("Analyze one manual region", width="stretch", on_click=go_to_analysis, args=("Manual single region",))
    with third:
        st.markdown('<div class="card"><div class="eyebrow">MULTI</div><h3>Build a region set</h3><p>Add, replace, compare, and export multiple manual regions.</p></div>', unsafe_allow_html=True)
        st.button("Analyze multiple regions", width="stretch", on_click=go_to_analysis, args=("Manual multi-region",))
    action_a, action_b, action_c = st.columns(3)
    action_a.button("Compare saved regions", width="stretch", on_click=go_to, args=("Compare Regions",))
    action_b.button("Open research results", width="stretch", on_click=go_to, args=("Research Results",))
    action_c.button("Check detection readiness", width="stretch", on_click=go_to, args=("Detection Readiness",))
    st.markdown("### Current apparatus")
    a, b, c = st.columns(3)
    a.metric("Classifier", "Verified")
    b.metric("Active input modes", "3")
    c.metric("Saved regions", str(len(records())))


def render_prepared() -> None:
    st.subheader("Prepared crop classification")
    st.caption("Upload a crop already centered on one visible region. The app does not verify that a defect is present.")
    uploaded = st.file_uploader("Choose a prepared PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="prepared_v2")
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    model_input = prepare_region(decoded.image)
    left, right = st.columns(2)
    left.image(decoded.image, caption="Uploaded crop", width="stretch")
    right.image(model_input, caption="Exact RGB 224×224 model input", width="stretch")
    if st.button("Classify and add prepared crop", type="primary", key="prepared_classify"):
        with st.spinner("Running the frozen model on CPU…"):
            record = classify_record(mode="prepared_crop", decoded=decoded, model_input=model_input)
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
    latest = next((item for item in reversed(records()) if item.region_id == st.session_state.get("latest_region_id")), None)
    if latest and latest.source_sha256 == decoded.byte_sha256:
        render_scores(latest, key=f"prepared_{latest.region_id}")


def manual_selection(decoded: Any, *, key: str):
    displayed = display_image(decoded.image)
    rectangle = st_cropper(
        displayed, realtime_update=True, box_color="#F59E0B", aspect_ratio=None,
        return_type="box", should_resize_image=False, stroke_width=3, key=key,
    )
    selected = map_display_box(rectangle, display_size=displayed.size, original_size=decoded.image.size)
    crop = contextual_crop(decoded.image, selected)
    st.caption(
        f"Display {displayed.width}×{displayed.height} → original {decoded.image.width}×{decoded.image.height}; "
        f"selected original box {selected.as_tuple()}."
    )
    return selected, crop


def render_manual_single() -> None:
    st.subheader("Manual single-region classification")
    st.caption("Upload a larger image and draw one rectangle around a visible region.")
    uploaded = st.file_uploader("Choose a larger PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="single_v2")
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    selected, crop = manual_selection(decoded, key=f"single_cropper_{decoded.byte_sha256[:12]}")
    left, right = st.columns([1.35, 1])
    left.image(annotated_selection(decoded.image, crop), caption="Orange: your rectangle · Teal: contextual crop", width="stretch")
    right.image(crop.model_input, caption="Exact contextual RGB 224×224 model input", width="stretch")
    if st.button("Classify and add selected region", type="primary", key="single_classify"):
        with st.spinner("Running the frozen model on CPU…"):
            geometry = crop.geometry
            record = classify_record(
                mode="manual_single_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(),
                contextual_box=(geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax),
            )
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
    latest = next((item for item in reversed(records()) if item.region_id == st.session_state.get("latest_region_id")), None)
    if latest and latest.source_sha256 == decoded.byte_sha256 and latest.selected_box == selected.as_tuple():
        render_scores(latest, key=f"single_{latest.region_id}")


def render_manual_multi() -> None:
    st.subheader("Manual multi-region analysis")
    st.caption("Draw one rectangle at a time. Saved regions receive stable IDs (R1, R2, …), may overlap, and are classified independently.")
    top_left, top_right = st.columns([3, 1])
    with top_right:
        if st.button("New image", width="stretch", key="multi_new_image"):
            st.session_state["multi_uploader_nonce"] += 1
            st.session_state.pop("multi_active_hash", None)
            st.rerun()
    with top_left:
        uploaded = st.file_uploader(
            "Choose one source image", type=["png", "jpg", "jpeg"],
            key=f"multi_v2_{st.session_state['multi_uploader_nonce']}",
        )
    if uploaded is None:
        return
    decoded = decode_upload(uploaded.getvalue(), uploaded.name)
    save_source(decoded)
    st.session_state["multi_active_hash"] = decoded.byte_sha256
    source_records = [item for item in records() if item.source_sha256 == decoded.byte_sha256 and item.mode == "manual_multi_region"]
    if source_records:
        st.image(annotate_regions(decoded.image, source_records), caption=f"Saved manual regions: {len(source_records)}", width="stretch")
    selected, crop = manual_selection(decoded, key=f"multi_cropper_{decoded.byte_sha256[:12]}")
    preview_left, preview_right = st.columns([1.4, 1])
    preview_left.image(annotated_selection(decoded.image, crop), caption="Current unsaved rectangle and contextual crop", width="stretch")
    preview_right.image(crop.model_input, caption="Current exact model input", width="stretch")
    geometry = crop.geometry
    contextual_box = (geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax)
    action_left, action_mid, action_right = st.columns(3)
    if action_left.button("Add and classify region", type="primary", width="stretch", key="multi_add"):
        with st.spinner("Classifying and saving this region…"):
            record = classify_record(
                mode="manual_multi_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(), contextual_box=contextual_box,
            )
            add_record(record)
            st.session_state["latest_region_id"] = record.region_id
            st.rerun()
    selected_id = action_mid.selectbox(
        "Saved region", [item.region_id for item in source_records], disabled=not source_records,
        key=f"multi_selected_{decoded.byte_sha256[:8]}",
    ) if source_records else None
    if action_right.button("Replace with current rectangle", width="stretch", disabled=not selected_id, key="multi_replace"):
        with st.spinner("Reclassifying the replacement region…"):
            replacement = classify_record(
                mode="manual_multi_region", decoded=decoded, model_input=crop.model_input,
                selected_box=selected.as_tuple(), contextual_box=contextual_box, replacement_id=selected_id,
            )
            st.session_state["analysis_records"] = replace_region(records(), replacement)
            st.session_state["latest_region_id"] = replacement.region_id
            st.rerun()
    remove_left, clear_right = st.columns(2)
    if remove_left.button("Remove selected region", disabled=not selected_id, width="stretch", key="multi_remove"):
        st.session_state["analysis_records"] = remove_region(records(), selected_id)
        st.rerun()
    if clear_right.button("Clear regions for this image", disabled=not source_records, width="stretch", key="multi_clear"):
        st.session_state["analysis_records"] = [
            item for item in records()
            if not (item.source_sha256 == decoded.byte_sha256 and item.mode == "manual_multi_region")
        ]
        st.rerun()


def render_analyze() -> None:
    render_hero("Analyze image", "Choose exactly how you will supply each visible region to the frozen crop classifier.")
    render_scope_notice()
    mode = st.radio("Analysis mode", ANALYSIS_MODES, horizontal=True, key="analysis_mode")
    try:
        if mode == "Prepared crop":
            render_prepared()
        elif mode == "Manual single region":
            render_manual_single()
        else:
            render_manual_multi()
    except (UploadValidationError, SelectionValidationError, FrozenModelError, RuntimeError) as exc:
        st.error(str(exc))


def render_compare() -> None:
    render_hero("Compare regions", "Inspect every region saved in this browser session and export reproducible metadata without server-side persistence.")
    items = records()
    if not items:
        st.info("No regions are saved yet. Open Analyze Image and classify a user-supplied region first.")
        st.button("Go to Analyze Image", type="primary", on_click=go_to, args=("Analyze Image",))
        return
    sort_by = st.selectbox("Sort comparison", ("Region ID", "Top score", "Prediction"), key="compare_sort")
    if sort_by == "Top score":
        items = sorted(items, key=lambda item: max(item.scores), reverse=True)
    elif sort_by == "Prediction":
        items = sorted(items, key=lambda item: (HUMAN_LABELS[item.predicted_label], int(item.region_id[1:])))
    else:
        items = sorted(items, key=lambda item: int(item.region_id[1:]))
    rows = [
        {"Region": item.region_id, "Mode": item.mode.replace("_", " "), "Source": item.source_name,
         "Prediction": HUMAN_LABELS[item.predicted_label], "Top score": max(item.scores),
         "Grad-CAM": item.gradcam_status}
        for item in items
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    chart_rows = [
        {"Region": item.region_id, "Category": HUMAN_LABELS[label], "Model score": item.scores[index]}
        for item in items for index, label in enumerate(CLASS_LABELS)
    ]
    st.vega_lite_chart(
        data=chart_rows,
        spec={"mark": "bar", "encoding": {
            "x": {"field": "Region", "type": "nominal"},
            "y": {"field": "Model score", "type": "quantitative"},
            "color": {"field": "Category", "type": "nominal"},
            "xOffset": {"field": "Category"},
            "tooltip": ["Region", "Category", {"field": "Model score", "format": ".6f"}]},
         "height": 300}, width="stretch",
    )
    selected_id = st.selectbox("Inspect region", [item.region_id for item in items], key="compare_selected")
    selected = next(item for item in items if item.region_id == selected_id)
    image_col, scores_col = st.columns([1, 1.7])
    image_col.image(selected.thumbnail, caption=f"{selected.region_id} exact model input", width="stretch")
    with scores_col:
        render_scores(selected, key=f"compare_{selected.region_id}")
    st.caption(f"Selected box: {selected.selected_box or 'prepared crop'} · contextual box: {selected.contextual_box or 'not applicable'}")
    if st.button("Generate Grad-CAM for selected region", key="compare_gradcam"):
        with st.spinner("Generating a read-only activation visualization…"):
            visual = generate_gradcam(cached_model(), selected.model_input, selected.predicted_class_id)
            st.session_state["analysis_records"] = replace_region(items, with_gradcam(selected, visual.overlay))
            st.rerun()
    if selected.gradcam_overlay is not None:
        st.warning("Grad-CAM is a crop-classifier activation visualization—not detector evidence, a causal explanation, or a safety assessment.")
        st.image(selected.gradcam_overlay, caption=f"{selected.region_id} Grad-CAM overlay", width=420)
    st.markdown("### Session exports")
    st.caption("Exports are generated in memory when requested. Uploaded images and analysis records are not written to the server.")
    export_a, export_b, export_c = st.columns(3)
    export_a.download_button("Download JSON", json_export(items), "blade-session.json", "application/json", width="stretch")
    export_b.download_button("Download CSV", csv_export(items), "blade-session.csv", "text/csv", width="stretch")
    manual_sources = sorted({item.source_sha256 for item in items if item.selected_box is not None})
    if manual_sources:
        source_hash = export_c.selectbox(
            "Annotated source", manual_sources,
            format_func=lambda value: next(item.source_name for item in items if item.source_sha256 == value),
            key="export_source",
        )
        matching = [item for item in items if item.source_sha256 == source_hash and item.selected_box is not None]
        image = st.session_state["source_images"].get(source_hash)
        if image is not None:
            export_c.download_button(
                "Download annotated PNG", annotated_image_export(image, matching),
                "blade-manual-regions.png", "image/png", width="stretch",
            )
    controls_a, controls_b = st.columns(2)
    if controls_a.button("Remove inspected region", width="stretch"):
        st.session_state["analysis_records"] = remove_region(items, selected_id)
        st.rerun()
    if controls_b.button("Clear session", width="stretch"):
        st.session_state["analysis_records"] = []
        st.session_state["source_images"] = {}
        st.session_state["next_region_number"] = 1
        st.rerun()


def render_research() -> None:
    render_hero("Research results", "A read-only dashboard over verified benchmark tables. Values are loaded from checked artifacts and are never recomputed in the app.")
    try:
        research = cached_research()
    except FrozenResearchError as exc:
        st.error(str(exc))
        return
    st.markdown('<div class="status-ok"><strong>Research analysis complete and locked.</strong> Canonical source fingerprint verified.</div>', unsafe_allow_html=True)
    clean = research["tables"]["clean_method_comparison"]
    summary = research["summary"]
    a, b, c, d = st.columns(4)
    a.metric("Test instances", clean[0]["test_instances"])
    b.metric("Test source images", clean[0]["test_sources"])
    c.metric("Methods compared", str(len(clean)))
    d.metric("Bootstrap resamples", str(summary["bootstrap_resamples"]))
    st.markdown("### Clean held-out performance")
    clean_view = [
        {"Method": row["method_name"], "Macro F1": float(row["macro_f1"]),
         "95% bootstrap CI low": float(row["macro_f1_bootstrap_ci_low"]),
         "95% bootstrap CI high": float(row["macro_f1_bootstrap_ci_high"]),
         "Accuracy": float(row["accuracy"]), "Balanced accuracy": float(row["balanced_accuracy"])}
        for row in clean
    ]
    st.dataframe(clean_view, hide_index=True, width="stretch")
    st.bar_chart(clean_view, x="Method", y="Macro F1", horizontal=True)
    st.markdown("### Data efficiency")
    efficiency_view = [
        {"Method": row["method_name"], "Training fraction": float(row["training_fraction"]),
         "Macro F1": float(row["macro_f1_mean"])}
        for row in research["tables"]["data_efficiency_summary"]
    ]
    st.line_chart(efficiency_view, x="Training fraction", y="Macro F1", color="Method")
    st.markdown("### Robustness retention")
    robust_view = [
        {"Method": row["method_name"], "Condition": row["condition_id"],
         "Retention (%)": float(row["retention_percent"])}
        for row in research["tables"]["robustness_retention_summary"] if row["condition_id"] != "clean"
    ]
    st.dataframe(robust_view, hide_index=True, width="stretch")
    st.markdown("### Error and human-review summary")
    st.dataframe(research["tables"]["error_human_review_summary"], hide_index=True, width="stretch")
    st.info("These are descriptive research summaries, not live estimates for the uploaded image. Intervals and seed variability retain the documented evaluation definitions.")
    st.caption(f"Verified research-source fingerprint: {research['scientific_output_fingerprint']}")


def render_detection() -> None:
    render_hero("Detection readiness", "Explore the curated full-image annotation audit and the evidence supporting future detector development.")
    try:
        status = cached_detection_status()
    except DetectorUnavailableError as exc:
        st.error(str(exc))
        return
    render_scope_notice()
    st.markdown("### Frozen dataset audit")
    audit = status.audit
    first, second, third, fourth = st.columns(4)
    first.metric("Curated images", audit["curated_image_count"])
    second.metric("Curated boxes", audit["curated_box_count"])
    third.metric("Multi-box images", audit["images_with_multiple_boxes"])
    fourth.metric("Healthy/background images", audit["background_or_healthy_images"])
    split_rows = [
        {"Split": split.title(), "Images": audit["split_image_counts"][split], "Boxes": audit["split_box_counts"][split]}
        for split in ("train", "validation", "test")
    ]
    st.dataframe(split_rows, hide_index=True, width="stretch")
    class_rows = [{"Class": HUMAN_LABELS.get(label, label), "Boxes": count} for label, count in audit["classes"].items()]
    st.bar_chart(class_rows, x="Class", y="Boxes", horizontal=True)
    duplicate_a, duplicate_b, duplicate_c = st.columns(3)
    duplicate_a.metric("Cross-split duplicate/related pairs", audit["cross_split_duplicate_or_related_pair_count"])
    duplicate_b.metric("Retained exact-duplicate groups", audit["retained_exact_duplicate_groups"])
    duplicate_c.metric("Invalid boxes", audit["suspicious_geometry"]["invalid_boxes"])
    provenance = audit["annotation_provenance"]
    st.caption(
        f"Annotation format: {audit['annotation_format']} · source: {provenance['dataset_name']} v{provenance['dataset_version']} · "
        f"license {provenance['license']} · dataset DOI {provenance['versioned_dataset_doi']}"
    )
    st.markdown("### Development considerations")
    st.markdown(
        "- The curated boxes support experiments in defect localization and six-category detection.\n"
        "- Healthy and background-only blade images would strengthen false-positive evaluation.\n"
        "- External turbine imagery would strengthen evidence across cameras, sites, and inspection conditions."
    )
    st.info("Future detector development can use the locked training and evaluation plan, add background evidence, select an operating point with validation data, and then assess application integration.")
    st.caption(f"Verified annotation-audit fingerprint: {status.scientific_output_fingerprint}")


def render_about() -> None:
    render_hero("About and limitations", "What this local research application does and how to interpret its user-selected-region workflow.")
    st.markdown("### Frozen apparatus")
    st.write(MODEL_DISPLAY_NAME)
    st.code(f"Checkpoint state fingerprint: {CHECKPOINT_STATE_FINGERPRINT}\nPreprocessing: {PREPROCESSING_CONTRACT}")
    st.markdown("### Six output categories")
    st.dataframe(
        [{"Category": HUMAN_LABELS[label], "Brief dataset-label guide": CLASS_DESCRIPTIONS[label]} for label in CLASS_LABELS],
        hide_index=True, width="stretch",
    )
    st.caption("These descriptions are plain-language guides to the supplied dataset labels, not new diagnoses or a physical severity taxonomy.")
    st.markdown("### Required interpretation limits")
    st.markdown(
        "- Each result describes a crop or rectangle selected by the user; selection is separate from classification.\n"
        "- Model scores are **not calibrated confidence estimates**.\n"
        "- The crop classifier was not externally validated for arbitrary drone imagery or healthy-blade screening.\n"
        "- Grad-CAM describes crop-classifier activations; it is not detector evidence or a causal explanation.\n"
        "- Outputs do not assess structural integrity, defect severity, remaining service life, or operational safety."
    )
    st.markdown("### Privacy and persistence")
    st.info("Uploads, crops, session history, visualizations, and exports remain in process memory for the active session. The app makes no external API calls and does not persist uploads or analysis history.")
    st.markdown("### Scientific state")
    st.write("The crop-classification research and full-image annotation audit are complete and locked. Future research can extend this work with trained full-image localization and external validation.")


initialize_session()
with st.sidebar:
    st.markdown("## Blade research app")
    page = st.radio("Navigation", NAVIGATION, key="navigation")
    st.caption(f"Application v{APPLICATION_VERSION}")
    st.divider()
    st.metric("Session regions", len(records()))
    st.success("Frozen classifier · CPU · local processing")
    if st.button("Clear all session data", width="stretch", disabled=not records()):
        st.session_state["analysis_records"] = []
        st.session_state["source_images"] = {}
        st.session_state["next_region_number"] = 1
        st.rerun()
    st.caption("PNG/JPG/JPEG · max 15 MB · no upload persistence or telemetry")

if page == "Home":
    render_home()
elif page == "Analyze Image":
    render_analyze()
elif page == "Compare Regions":
    render_compare()
elif page == "Research Results":
    render_research()
elif page == "Detection Readiness":
    render_detection()
else:
    render_about()
