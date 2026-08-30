from __future__ import annotations

from pathlib import Path

import pytest

from windblade_demo.constants import HUMAN_LABELS


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app/app.py"


def test_streamlit_app_starts_without_an_upload():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert len(app.radio) == 1


def test_prepared_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/processed/wtbd_crops_v1/images/1_0.png"
    if not image.is_file():
        pytest.skip("local frozen Phase 3 crop payload is not available")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/png").run(timeout=30)
    assert app.button[0].label == "Classify prepared region"
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())


def test_manual_upload_and_classify_ui_workflow():
    testing = pytest.importorskip("streamlit.testing.v1")
    image = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    if not image.is_file():
        pytest.skip("local frozen WTBD source image payload is not available")
    app = testing.AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.radio[0].set_value(app.radio[0].options[1]).run(timeout=30)
    app.file_uploader[0].upload(image.name, image.read_bytes(), "image/jpeg").run(timeout=30)
    assert app.button[0].label == "Classify selected region"
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.subheader[-1].value in set(HUMAN_LABELS.values())
