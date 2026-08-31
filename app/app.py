"""Streamlit entry point for the non-scientific frozen-classifier demo."""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_cropper import st_cropper

from windblade_demo.constants import CLASS_LABELS, HUMAN_LABELS, MAX_UPLOAD_BYTES, MODEL_DISPLAY_NAME
from windblade_demo.crops import (
    ContextualCrop,
    SelectionValidationError,
    annotated_selection,
    contextual_crop,
    display_image,
    map_display_box,
    prepare_region,
)
from windblade_demo.explain import generate_gradcam
from windblade_demo.inference import FrozenModelError, infer, load_frozen_model, model_status
from windblade_demo.inputs import UploadValidationError, decode_upload


st.set_page_config(
    page_title="Blade Image Research Demo",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f8fa; }
    .block-container { max-width: 1220px; padding-top: 1.4rem; padding-bottom: 3rem; }
    .hero { padding: 1.55rem 1.7rem; border-radius: 20px; color: white;
            background: linear-gradient(125deg, #12344d 0%, #0b7a75 72%, #15958d 100%);
            box-shadow: 0 14px 34px rgba(18,52,77,.18); margin-bottom: 1rem; }
    .hero h1 { margin: .2rem 0 0; font-size: clamp(1.8rem, 4vw, 2.55rem); line-height: 1.1; }
    .hero p { margin: .65rem 0 0; max-width: 780px; opacity: .94; font-size: 1.02rem; }
    .badge { display: inline-block; padding: .28rem .58rem; margin: 0 .32rem .3rem 0;
             border-radius: 999px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
             font-size: .72rem; font-weight: 750; letter-spacing: .055em; }
    .scope { border-left: 5px solid #e69f00; background: #fff8e6; padding: .9rem 1rem;
             border-radius: 10px; margin: .75rem 0 1.2rem; color: #332700; }
    .result-card { background: white; border: 1px solid #d9e6e8; border-radius: 14px;
                   padding: 1rem 1.2rem; box-shadow: 0 5px 18px rgba(18,52,77,.08); }
    .eyebrow { color: #0b7a75; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    .workflow-head { display:flex; gap:.65rem; align-items:center; margin:.25rem 0 .75rem; }
    .step-number { display:inline-grid; place-items:center; width:1.75rem; height:1.75rem;
                   border-radius:50%; background:#0b7a75; color:white; font-weight:800; }
    .muted-panel { background:#eaf2f4; border:1px solid #d0e0e4; border-radius:12px;
                   padding:.85rem 1rem; color:#294656; }
    div[data-testid="stMetric"] { background:white; border:1px solid #d9e6e8; padding:.7rem;
                                  border-radius:12px; }
    div[data-testid="stFileUploader"] { background:white; border:1px solid #d9e6e8;
                                         border-radius:14px; padding:.6rem .85rem; }
    @media (max-width: 700px) { .hero { padding:1.15rem; border-radius:14px; } }
    </style>
    <div class="hero">
      <div><span class="badge">LOCAL PROCESSING</span><span class="badge">FROZEN MODEL</span><span class="badge">MANUAL REGION INPUT</span></div>
      <h1>Blade image research demo</h1>
      <p>Classify one visible defect region with the frozen six-category MobileNet model—either from a prepared crop or a region you select.</p>
    </div>
    <div class="scope"><strong>Experimental-use notice:</strong> Automatic localization is unavailable because Phase 11B detector training and evaluation are incomplete. This demo classifies only a region supplied or selected by you.</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Verifying and loading the frozen checkpoint…")
def cached_model():
    return load_frozen_model()


def reset_for_upload(namespace: str, fingerprint: str) -> None:
    identity_key = f"{namespace}_upload_identity"
    if st.session_state.get(identity_key) != fingerprint:
        for suffix in ("result", "model_input", "gradcam", "selection"):
            st.session_state.pop(f"{namespace}_{suffix}", None)
        st.session_state[identity_key] = fingerprint


def clear_workflow(namespace: str) -> None:
    for key in list(st.session_state):
        if key.startswith(f"{namespace}_"):
            del st.session_state[key]


def step_heading(number: int, title: str) -> None:
    st.markdown(
        f'<div class="workflow-head"><span class="step-number">{number}</span><strong>{title}</strong></div>',
        unsafe_allow_html=True,
    )


def render_result(namespace: str, model_input, result: Any, show_explanation: bool) -> None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">CLASSIFICATION RESULT</div>', unsafe_allow_html=True)
    st.subheader(HUMAN_LABELS[result.predicted_label])
    st.caption("Model scores — not calibrated confidence estimates")
    rows = [
        {"Category": HUMAN_LABELS[label], "Model score": result.scores[index]}
        for index, label in enumerate(CLASS_LABELS)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.vega_lite_chart(
        {"values": rows},
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#0B7A75"},
            "encoding": {
                "x": {"field": "Model score", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "y": {"field": "Category", "type": "nominal", "sort": "-x"},
                "tooltip": ["Category", {"field": "Model score", "format": ".6f"}],
            },
            "height": 220,
        },
        width="stretch",
    )
    loaded = cached_model()
    status = model_status(loaded)
    left, middle, right = st.columns(3)
    left.metric("Request inference", f"{result.inference_seconds * 1000:.1f} ms")
    middle.metric("Preprocessing", f"{result.preprocessing_seconds * 1000:.1f} ms")
    right.metric("One-time model load", f"{loaded.load_seconds:.2f} s")
    st.caption(f"{status['model']} · {status['status']}")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_explanation and st.button("Generate optional Grad-CAM visualization", key=f"{namespace}_gradcam_button"):
        with st.spinner("Generating the optional activation visualization…"):
            st.session_state[f"{namespace}_gradcam"] = generate_gradcam(
                loaded, model_input, result.predicted_class_id
            )
    gradcam = st.session_state.get(f"{namespace}_gradcam") if show_explanation else None
    if gradcam is not None:
        st.warning(
            "Grad-CAM is an activation visualization for the selected class score. "
            "It is not automatic defect localization, a causal explanation, or a safety assessment."
        )
        first, second = st.columns(2)
        first.image(gradcam.heatmap, caption="Grad-CAM heatmap", width="stretch")
        second.image(gradcam.overlay, caption="Overlay on the exact model input", width="stretch")


with st.sidebar:
    st.markdown("## Workflow")
    mode = st.radio(
        "Choose how to supply a visible region",
        ("Prepared visible defect crop", "Larger image — manually select a region"),
        captions=("Upload a crop already centered on a defect.", "Draw one rectangle on a larger image."),
    )
    show_explanation = st.toggle("Offer optional Grad-CAM", value=True)
    namespace = "prepared" if mode == "Prepared visible defect crop" else "manual"
    if st.button("Reset current workflow", width="stretch"):
        clear_workflow(namespace)
        st.rerun()
    st.divider()
    st.markdown("### Frozen apparatus")
    st.caption(MODEL_DISPLAY_NAME)
    st.success("Checkpoint verified at load. No training, tuning, calibration, or model selection.")
    st.markdown("### Upload privacy")
    st.caption("PNG/JPG/JPEG · maximum 15 MB · decoded and processed in memory")
    st.caption("No upload persistence, external API, image transmission, or telemetry.")

st.markdown(
    '<div class="muted-panel"><strong>Two manual workflows are available.</strong> '
    'A frozen automatic detector does not exist, so the application does not offer automatic localization.</div>',
    unsafe_allow_html=True,
)

if mode == "Prepared visible defect crop":
    st.header("Classify a prepared defect crop")
    st.caption("Use this when one visible defect region has already been identified and cropped.")
    step_heading(1, "Upload one prepared region")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Choose a PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="prepared_uploader"
        )
    if uploaded is not None:
        try:
            decoded = decode_upload(uploaded.getvalue(), uploaded.name)
            reset_for_upload("prepared", decoded.byte_sha256)
            step_heading(2, "Review the exact classifier input")
            original_col, input_col = st.columns(2)
            original_col.image(decoded.image, caption="Uploaded region", width="stretch")
            model_input = prepare_region(decoded.image)
            input_col.image(
                model_input,
                caption="Exact model input: RGB, bilinear resize to 224×224",
                width="stretch",
            )
            st.info("The supplied region will be classified into the six frozen WTBD categories. The app does not verify that a defect is present.")
            step_heading(3, "Run the frozen classifier")
            if st.button("Classify prepared region", type="primary"):
                with st.spinner("Running the frozen model on CPU…"):
                    loaded = cached_model()
                    st.session_state["prepared_result"] = infer(loaded, model_input)
                    st.session_state["prepared_model_input"] = model_input
                    st.session_state.pop("prepared_gradcam", None)
            if st.session_state.get("prepared_result") is not None:
                render_result(
                    "prepared",
                    st.session_state["prepared_model_input"],
                    st.session_state["prepared_result"],
                    show_explanation,
                )
        except (UploadValidationError, FrozenModelError, RuntimeError) as exc:
            st.error(str(exc))
else:
    st.header("Manually select a defect region")
    st.caption("Upload a larger image, then draw one rectangle around a visible region you want classified.")
    step_heading(1, "Upload one larger image")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Choose a PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="manual_uploader"
        )
    if uploaded is not None:
        try:
            decoded = decode_upload(uploaded.getvalue(), uploaded.name)
            reset_for_upload("manual", decoded.byte_sha256)
            step_heading(2, "Draw and verify one region")
            displayed = display_image(decoded.image)
            rectangle = st_cropper(
                displayed,
                realtime_update=True,
                box_color="#F59E0B",
                aspect_ratio=None,
                return_type="box",
                should_resize_image=False,
                stroke_width=3,
                key=f"manual_cropper_{decoded.byte_sha256[:12]}",
            )
            selected = map_display_box(
                rectangle,
                display_size=displayed.size,
                original_size=decoded.image.size,
            )
            crop = contextual_crop(decoded.image, selected)
            st.caption(
                f"Display {displayed.width}×{displayed.height} → original {decoded.image.width}×{decoded.image.height}; "
                f"selected original box (left, top, right, bottom) = {selected.as_tuple()}."
            )
            overview_col, crop_col = st.columns([1.4, 1])
            overview_col.image(
                annotated_selection(decoded.image, crop),
                caption="Orange: your region · Teal: frozen contextual square",
                width="stretch",
            )
            crop_col.image(
                crop.model_input,
                caption="Exact contextual crop sent to the model: RGB 224×224",
                width="stretch",
            )
            st.caption(
                f"Context square: {crop.geometry.crop_side} original pixels; "
                f"minimum-side applied={crop.geometry.minimum_side_applied}; "
                f"boundary-shifted={crop.geometry.boundary_shifted}; no padding."
            )
            selection_identity = (decoded.byte_sha256, selected.as_tuple())
            if st.session_state.get("manual_selection") != selection_identity:
                st.session_state["manual_selection"] = selection_identity
                st.session_state.pop("manual_result", None)
                st.session_state.pop("manual_model_input", None)
                st.session_state.pop("manual_gradcam", None)
            step_heading(3, "Run the frozen classifier")
            if st.button("Classify selected region", type="primary"):
                with st.spinner("Running the frozen model on CPU…"):
                    loaded = cached_model()
                    st.session_state["manual_result"] = infer(loaded, crop.model_input)
                    st.session_state["manual_model_input"] = crop.model_input
                    st.session_state.pop("manual_gradcam", None)
            if st.session_state.get("manual_result") is not None:
                render_result(
                    "manual",
                    st.session_state["manual_model_input"],
                    st.session_state["manual_result"],
                    show_explanation,
                )
        except (UploadValidationError, SelectionValidationError, FrozenModelError, RuntimeError) as exc:
            st.error(str(exc))

st.divider()
with st.expander("Limits of this research demonstration", expanded=False):
    st.markdown(
        "- It does **not automatically locate defects** or establish that an image is defect-free.\n"
        "- Model scores are **not calibrated confidence estimates**.\n"
        "- It was not externally validated for arbitrary drone imagery or healthy-blade screening.\n"
        "- Grad-CAM describes crop-classifier activations; it is not detector evidence or a causal explanation."
    )
    st.warning(
        "This research demonstration identifies and classifies visible image patterns. "
        "It does not assess structural integrity, defect severity, remaining service life, or operational safety."
    )
st.caption(
    "UI-only refresh · frozen Phase 6 crop classifier · Phase 11B detector unavailable · "
    "Phase 10 and Phase 11A scientific results unchanged · Phase 12 not started."
)
