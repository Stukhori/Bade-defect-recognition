"""Local two-pass data-entry interface for the frozen Phase 9A review packet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from windblade_review.packet import ReviewCase, ReviewPacketError, load_review_packet
from windblade_review.schema import PassSchema, ReviewSchemaError, load_pass_schema
from windblade_review.store import ReviewDataError, ReviewSnapshot, ReviewStore
from windblade_review.workflow import (
    PASS_A_ATTESTATION,
    GateError,
    pass_b_access_allowed,
    validate_pass_a_lock,
)


st.set_page_config(page_title="Phase 9A Human Review", page_icon="📝", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5f7f8; }
    .review-hero { background: linear-gradient(120deg,#213547,#3b6670); color:white;
      border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
    .review-hero h1 { margin:0; font-size:1.8rem; }
    .review-hero p { margin:.4rem 0 0; opacity:.92; }
    .blind { border-left:5px solid #d49416; background:#fff8e5; padding:.8rem 1rem;
      border-radius:8px; margin:.7rem 0 1rem; }
    </style>
    <div class="review-hero"><h1>Phase 9A human review</h1>
    <p>Local, judgment-neutral entry for the existing blinded two-pass packet.</p></div>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/error_analysis.yaml"
PACKET_ROOT = ROOT / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"
DEFAULT_FORM_ROOT = PACKET_ROOT
FORM_ROOT = Path(os.environ.get("WINDBLADE_REVIEW_FORM_ROOT", str(DEFAULT_FORM_ROOT))).resolve()

FIELD_LABELS = {
    "defect_visible": "Defect visibility",
    "corruption_obscures_diagnostic_detail": "Corruption obscures diagnostic detail",
    "dataset_label_visually_plausible": "Dataset label visually plausible",
    "visually_ambiguous_between_categories": "Visually ambiguous between categories",
    "possible_crop_or_background_problem": "Possible crop or background problem",
    "activation_primarily_inside_annotation": "Activation location relative to annotation",
    "activation_concentrated_on_degradation_artifact": "Activation concentrated on degradation artifact",
    "pattern_consistent_across_cnn_seeds": "Pattern consistent across CNN seeds",
    "prediction_visually_understandable_after_reveal": "Prediction visually understandable after reveal",
}

FIELD_HELP = {
    "defect_visible": "Rate only whether the annotated visible defect can be seen. Use uncertain rather than guessing.",
    "corruption_obscures_diagnostic_detail": "Rate the degree to which displayed degradation hides potentially diagnostic visual detail.",
    "dataset_label_visually_plausible": "Visual plausibility is not certainty and does not relabel the frozen sample.",
    "visually_ambiguous_between_categories": "Ambiguity may be marked even when the dataset label appears plausible.",
    "possible_crop_or_background_problem": "Rate whether crop framing or background may complicate visual review.",
    "activation_primarily_inside_annotation": "Describe the displayed spatial pattern only; overlap does not establish causal reasoning.",
    "activation_concentrated_on_degradation_artifact": "Rate only the apparent displayed concentration; use uncertain rather than guessing.",
    "pattern_consistent_across_cnn_seeds": "Compare the displayed seed-specific maps without selecting a preferred seed.",
    "prediction_visually_understandable_after_reveal": "Rate visual understandability after reveal, not correctness or causal reasoning.",
}


def form_path(pass_name: str) -> Path:
    filename = "pass_a_review_form.csv" if pass_name == "pass_a" else "pass_b_review_form.csv"
    return FORM_ROOT / pass_name / filename


def status_panel(snapshot: ReviewSnapshot, total_cases: int) -> None:
    progress = snapshot.completed_cases / total_cases
    st.progress(progress, text=f"{snapshot.completed_cases}/{total_cases} cases complete")
    first, second, third = st.columns(3)
    first.metric("Complete cases", f"{snapshot.completed_cases}/{total_cases}")
    second.metric("Required answers", f"{snapshot.answered_required}/{snapshot.total_required}")
    third.metric("Unanswered fields", snapshot.unanswered_required)
    if st.session_state.get("last_review_save"):
        st.caption(f"Last successful atomic save: {st.session_state['last_review_save']}")


def display_case(case: ReviewCase, pass_name: str) -> None:
    st.subheader(case.review_id)
    if pass_name == "pass_a":
        st.write(f"Dataset true label: **{case.true_label}**")
    else:
        st.write(case.metadata)
    columns = st.columns(min(3, len(case.assets)))
    for index, asset in enumerate(case.assets):
        columns[index % len(columns)].image(str(asset.path), caption=asset.caption, width="stretch")


def widget_values(schema: PassSchema, row: dict[str, str], review_id: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in schema.fields:
        key = f"{schema.pass_name}_{review_id}_{field.name}"
        if field.required:
            selected = row[field.name]
            index = field.choices.index(selected) if selected else None
            answer = st.radio(
                FIELD_LABELS[field.name],
                field.choices,
                index=index,
                key=key,
                help=FIELD_HELP[field.name],
                horizontal=True,
            )
            values[field.name] = "" if answer is None else str(answer)
        else:
            values[field.name] = st.text_area(
                "Optional reviewer notes",
                value=row[field.name],
                key=key,
                help="Optional free text. Notes are preserved exactly, including punctuation and Unicode.",
                height=100,
            )
    return values


def current_index(pass_name: str, snapshot: ReviewSnapshot, total: int) -> int:
    key = f"{pass_name}_index"
    if key not in st.session_state:
        st.session_state[key] = snapshot.first_incomplete_index or 0
    st.session_state[key] = max(0, min(int(st.session_state[key]), total - 1))
    return int(st.session_state[key])


def render_review_pass(pass_name: str, schema: PassSchema, packet: Any, store: ReviewStore) -> ReviewSnapshot:
    snapshot = store.load()
    status_panel(snapshot, len(packet.cases))
    index = current_index(pass_name, snapshot, len(packet.cases))
    case = packet.cases[index]
    st.caption(f"Case {index + 1} of {len(packet.cases)}")
    display_case(case, pass_name)
    row = dict(snapshot.rows[index])
    values = widget_values(schema, row, case.review_id)
    if any(values[field.name] != row[field.name] for field in schema.fields):
        result = store.save_case(case.review_id, values)
        if result.changed:
            st.session_state["last_review_save"] = result.saved_at
            snapshot = result.snapshot

    previous, save_next, first_incomplete = st.columns(3)
    if previous.button("Previous", disabled=index == 0, key=f"{pass_name}_previous"):
        st.session_state[f"{pass_name}_index"] = index - 1
        st.rerun()
    if save_next.button("Save & Next", type="primary", key=f"{pass_name}_next"):
        missing = [name for name in schema.required_fields if not values[name]]
        if missing:
            st.error("Complete every required field before moving forward: " + ", ".join(missing))
        elif index < len(packet.cases) - 1:
            st.session_state[f"{pass_name}_index"] = index + 1
            st.rerun()
        else:
            st.success("This is the final case. Use completion validation below.")
    if first_incomplete.button("Jump to first incomplete case", key=f"{pass_name}_incomplete"):
        refreshed = store.load()
        if refreshed.first_incomplete_index is None:
            st.info("Every required field in this pass is complete.")
        else:
            st.session_state[f"{pass_name}_index"] = refreshed.first_incomplete_index
            st.rerun()
    return store.load()


try:
    # Initial execution loads only Pass A schema, HTML, assets for the current
    # case, and form. Pass B and the separate ID mapping are not accessed.
    pass_a_schema = load_pass_schema(CONFIG_PATH, "pass_a")
    pass_a_packet = load_review_packet(ROOT, PACKET_ROOT, "pass_a")
    pass_a_store = ReviewStore(form_path("pass_a"), pass_a_schema, pass_a_packet.review_ids)
    pass_a_snapshot = pass_a_store.load()

    if st.session_state.get("pass_a_locked") and not pass_a_snapshot.complete:
        st.session_state["pass_a_locked"] = False
        st.session_state["pass_b_started"] = False
        st.warning("Pass A changed after locking and is incomplete; Pass B access was closed.")

    if not st.session_state.get("pass_a_locked", False):
        st.markdown(
            '<div class="blind"><strong>Pass A blinding:</strong> model identity, predictions, correctness, event category, Grad-CAM evidence, source IDs, Pass B, and the separate ID mapping are not loaded or shown.</div>',
            unsafe_allow_html=True,
        )
        pass_a_snapshot = render_review_pass("pass_a", pass_a_schema, pass_a_packet, pass_a_store)
        if pass_a_snapshot.complete:
            st.divider()
            st.subheader("Pass A completion gate")
            attested = st.checkbox(PASS_A_ATTESTATION, key="pass_a_attestation")
            if st.button("Validate and lock Pass A", type="primary", key="lock_pass_a"):
                try:
                    completed_hash = validate_pass_a_lock(pass_a_store, attested)
                except GateError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["pass_a_locked"] = True
                    st.session_state["pass_a_locked_hash"] = completed_hash
                    st.session_state["pass_b_started"] = False
                    st.rerun()
        else:
            st.warning(
                f"Pass A is incomplete: {pass_a_snapshot.unanswered_required} required fields remain. "
                "Pass B is inaccessible."
            )
    elif not st.session_state.get("pass_b_started", False):
        st.success("Pass A is validated and locked read-only for this interface session.")
        st.code(st.session_state.get("pass_a_locked_hash", pass_a_store.sha256()), language=None)
        st.warning("Pass B reveals model identities, predictions, event information, and Grad-CAM evidence.")
        if st.button("Begin Pass B", type="primary", key="begin_pass_b"):
            st.session_state["pass_b_started"] = True
            st.rerun()

    if pass_b_access_allowed(
        pass_a_snapshot,
        pass_a_locked=bool(st.session_state.get("pass_a_locked")),
        pass_b_started=bool(st.session_state.get("pass_b_started")),
    ):
        # This is the only runtime branch that loads any Pass B material.
        pass_b_schema = load_pass_schema(CONFIG_PATH, "pass_b")
        pass_b_packet = load_review_packet(ROOT, PACKET_ROOT, "pass_b")
        pass_b_store = ReviewStore(form_path("pass_b"), pass_b_schema, pass_b_packet.review_ids)
        st.markdown(
            '<div class="blind"><strong>Grad-CAM warning:</strong> Grad-CAM maps are independently normalized visualizations. Their colors cannot be compared quantitatively across maps, and they do not prove what caused a prediction.</div>',
            unsafe_allow_html=True,
        )
        pass_b_snapshot = render_review_pass("pass_b", pass_b_schema, pass_b_packet, pass_b_store)
        if pass_b_snapshot.complete:
            st.divider()
            if st.button("Validate completed Pass B", type="primary", key="validate_pass_b"):
                pass_b_store.load(require_complete=True)
                st.success("Pass B is complete and schema-valid.")
                st.code(pass_b_store.sha256(), language=None)
        else:
            st.warning(f"Pass B is incomplete: {pass_b_snapshot.unanswered_required} required fields remain.")
except (ReviewDataError, ReviewPacketError, ReviewSchemaError, OSError) as exc:
    st.error(f"Review interface stopped without repairing data: {exc}")

st.divider()
st.caption(
    "Local human-data-entry aid only. No model, LLM, heuristic, analytics, external service, or answer suggestion is used."
)
