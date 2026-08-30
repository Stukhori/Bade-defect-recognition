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
    page_title="Wind Turbine Blade Defect Region Classifier",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(160deg, #f7fafc 0%, #edf5f5 100%); }
    .hero { padding: 1.3rem 1.5rem; border-radius: 18px; color: white;
            background: linear-gradient(120deg, #12344d 0%, #0b7a75 100%);
            box-shadow: 0 10px 30px rgba(18,52,77,.16); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .45rem 0 0; opacity: .92; }
    .scope { border-left: 5px solid #f59e0b; background: #fff8e8; padding: .9rem 1rem;
             border-radius: 8px; margin: .75rem 0 1.2rem; }
    .result-card { background: white; border: 1px solid #d9e6e8; border-radius: 14px;
                   padding: 1rem 1.2rem; box-shadow: 0 5px 18px rgba(18,52,77,.08); }
    .eyebrow { color: #0b7a75; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    </style>
    <div class="hero">
      <div class="eyebrow" style="color:#b9f3ed">LOCAL • OFFLINE • CPU</div>
      <h1>Visible defect region classifier</h1>
      <p>A careful demonstration of one frozen six-category WTBD classifier.</p>
    </div>
    <div class="scope"><strong>Scope:</strong> This demonstration classifies a manually identified visible defect region into one of six WTBD categories. It does not automatically detect defects, assess blade safety or condition, detect hidden/internal damage, estimate severity or remaining life, or replace inspection professionals.</div>
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


def render_result(namespace: str, model_input, result: Any) -> None:
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

    if st.button("Generate optional Grad-CAM visualization", key=f"{namespace}_gradcam_button"):
        with st.spinner("Generating the optional activation visualization…"):
            st.session_state[f"{namespace}_gradcam"] = generate_gradcam(
                loaded, model_input, result.predicted_class_id
            )
    gradcam = st.session_state.get(f"{namespace}_gradcam")
    if gradcam is not None:
        st.warning(
            "Grad-CAM is an activation visualization for the selected class score. "
            "It is not automatic defect localization, a causal explanation, or a safety assessment."
        )
        first, second = st.columns(2)
        first.image(gradcam.heatmap, caption="Grad-CAM heatmap", width="stretch")
        second.image(gradcam.overlay, caption="Overlay on the exact model input", width="stretch")


with st.sidebar:
    st.header("Frozen apparatus")
    st.write(MODEL_DISPLAY_NAME)
    st.success("No training, tuning, calibration, or model selection occurs in this app.")
    st.caption("Accepted: PNG, JPG, JPEG · maximum 15 MB · processed in memory")
    st.caption("Uploaded images are not permanently stored or sent to an external service.")

mode = st.radio(
    "Choose an input workflow",
    ("Prepared visible defect crop", "Larger image — manually select a region"),
    horizontal=True,
)

if mode == "Prepared visible defect crop":
    st.subheader("Prepared region")
    st.write(
        "Use this when a visible defect region has already been identified and cropped. "
        "The app classifies the supplied region; it does not locate a defect."
    )
    uploaded = st.file_uploader(
        "Upload one prepared crop", type=["png", "jpg", "jpeg"], key="prepared_uploader"
    )
    if uploaded is not None:
        try:
            decoded = decode_upload(uploaded.getvalue(), uploaded.name)
            reset_for_upload("prepared", decoded.byte_sha256)
            original_col, input_col = st.columns(2)
            original_col.image(decoded.image, caption="Uploaded region", width="stretch")
            model_input = prepare_region(decoded.image)
            input_col.image(
                model_input,
                caption="Exact model input: RGB, bilinear resize to 224×224",
                width="stretch",
            )
            st.info("One manually identified visible region will be classified into the six frozen WTBD categories.")
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
                )
        except (UploadValidationError, FrozenModelError, RuntimeError) as exc:
            st.error(str(exc))
else:
    st.subheader("Manual region selection")
    st.write(
        "Draw a rectangle around one visible defect region. The app maps the display rectangle "
        "back to original pixels and applies the frozen Phase 3 contextual crop."
    )
    uploaded = st.file_uploader(
        "Upload one larger image", type=["png", "jpg", "jpeg"], key="manual_uploader"
    )
    if uploaded is not None:
        try:
            decoded = decode_upload(uploaded.getvalue(), uploaded.name)
            reset_for_upload("manual", decoded.byte_sha256)
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
                )
        except (UploadValidationError, SelectionValidationError, FrozenModelError, RuntimeError) as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "Non-scientific local demonstration. Scientific Phases 0–8 remain frozen; "
    "Phase 9A remains awaiting human review."
)
