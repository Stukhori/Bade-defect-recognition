from __future__ import annotations

from pathlib import Path

import pytest


def test_streamlit_app_starts_without_an_upload():
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app/app.py"
    app = testing.AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert len(app.radio) == 1
