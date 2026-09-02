from __future__ import annotations

from pathlib import Path

import pytest

from windblade_demo.constants import HUMAN_LABELS


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app/app.py"
NAVIGATION = [
    "Home", "Analyze Image", "Compare Regions", "Research Results",
    "Detection Readiness", "About and Limitations",
]


def button(app, label):
    return next(item for item in app.button if item.label == label)


def open_analysis(app):
    next(item for item in app.radio if item.label == "Navigation").set_value("Analyze Image").run(timeout=30)
    mode = next(item for item in app.radio if item.label == "Analysis mode")
    assert mode.options == ["Prepared crop", "Manual single region", "Manual multi-region"]
    return app


def test_application_v2_starts_on_home_without_an_upload():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert app.radio[0].label == "Navigation"
    assert app.radio[0].options == NAVIGATION
    assert len(app.file_uploader) == 0


def test_application_v2_withholds_automatic_localization_and_shows_safety_scope():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    rendered = "\n".join(element.value for element in app.markdown)
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Automatic localization is unavailable" in rendered
    assert "operational safety" in source
    assert "Detector training and evaluation have not been completed" in source
    assert "ultralytics" not in source.lower()


def test_prepared_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/processed/wtbd_crops_v1/images/1_0.png"
    if not image.is_file():
        pytest.skip("local frozen Phase 3 crop payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/png").run(timeout=30)
    button(app, "Classify and add prepared crop").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())
    assert app.session_state["analysis_records"][0].region_id == "R1"
    next(item for item in app.radio if item.label == "Navigation").set_value("Compare Regions").run(timeout=30)
    assert {item.label for item in app.get("download_button")} >= {"Download JSON", "Download CSV"}
    assert not app.exception


def test_manual_single_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    if not image.is_file():
        pytest.skip("local frozen WTBD source image payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    next(item for item in app.radio if item.label == "Analysis mode").set_value("Manual single region").run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/jpeg").run(timeout=30)
    button(app, "Classify and add selected region").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())
    assert app.session_state["analysis_records"][0].selected_box is not None


def test_manual_multi_region_adds_stable_session_record():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    if not image.is_file():
        pytest.skip("local frozen WTBD source image payload is not available")
    app = open_analysis(testing.AppTest.from_file(str(APP_PATH)).run(timeout=30))
    next(item for item in app.radio if item.label == "Analysis mode").set_value("Manual multi-region").run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/jpeg").run(timeout=30)
    button(app, "Add and classify region").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    saved = app.session_state["analysis_records"]
    assert len(saved) == 1
    assert saved[0].region_id == "R1"
    assert saved[0].mode == "manual_multi_region"


@pytest.mark.parametrize("page", ["Research Results", "Detection Readiness", "About and Limitations"])
def test_read_only_pages_render_without_errors(page):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.radio[0].set_value(page).run(timeout=30)
    assert not app.exception
    assert not app.error
