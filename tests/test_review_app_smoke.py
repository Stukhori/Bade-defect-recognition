from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from windblade_review.packet import load_review_packet
from windblade_review.schema import load_pass_schema
from windblade_review.store import ReviewStore


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/review_app.py"
CONFIG = ROOT / "configs/error_analysis.yaml"
PACKET = ROOT / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"


def copied_forms(tmp_path: Path) -> Path:
    forms = tmp_path / "forms"
    (forms / "pass_a").mkdir(parents=True)
    (forms / "pass_b").mkdir(parents=True)
    shutil.copy2(PACKET / "pass_a/pass_a_review_form.csv", forms / "pass_a/pass_a_review_form.csv")
    shutil.copy2(PACKET / "pass_b/pass_b_review_form.csv", forms / "pass_b/pass_b_review_form.csv")
    return forms


def complete_a(forms: Path) -> None:
    schema = load_pass_schema(CONFIG, "pass_a")
    packet = load_review_packet(ROOT, PACKET, "pass_a")
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", schema, packet.review_ids)
    values = {field.name: (field.choices[0] if field.required else "") for field in schema.fields}
    for review_id in packet.review_ids:
        store.save_case(review_id, values)


def test_initial_and_query_parameter_runs_remain_pass_a_only(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    forms = copied_forms(tmp_path)
    monkeypatch.setenv("WINDBLADE_REVIEW_FORM_ROOT", str(forms))
    app = testing.AppTest.from_file(str(APP))
    app.query_params["pass"] = "pass_b"
    app.run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[0].value == "P9A-001"
    labels = {radio.label for radio in app.radio}
    assert "Defect visibility" in labels
    assert "Activation location relative to annotation" not in labels
    assert not any(button.label == "Begin Pass B" for button in app.button)


def test_completed_copy_requires_attested_lock_then_deliberate_b_start(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    forms = copied_forms(tmp_path)
    complete_a(forms)
    monkeypatch.setenv("WINDBLADE_REVIEW_FORM_ROOT", str(forms))
    app = testing.AppTest.from_file(str(APP)).run(timeout=30)
    lock = next(button for button in app.button if button.label == "Validate and lock Pass A")
    lock.click().run(timeout=30)
    assert any("attestation" in error.value.lower() for error in app.error)
    app.checkbox[0].set_value(True).run(timeout=30)
    next(button for button in app.button if button.label == "Validate and lock Pass A").click().run(timeout=30)
    assert not app.error
    assert any(button.label == "Begin Pass B" for button in app.button)
    assert "Activation location relative to annotation" not in {radio.label for radio in app.radio}
    next(button for button in app.button if button.label == "Begin Pass B").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    labels = {radio.label for radio in app.radio}
    assert "Activation location relative to annotation" in labels
    assert "Defect visibility" not in labels
