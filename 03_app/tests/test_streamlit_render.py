"""
Streamlit AppTest coverage: execute the real app script for every page and
assert it renders without raising.

This is stronger than a unit test of the helpers because it runs the actual
Streamlit script, so any import error, bad attribute, or widget misuse surfaces.

Run:
    .venv/Scripts/python.exe -m pytest 03_app/tests/test_streamlit_render.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
APP_FILE = APP_DIR / "app.py"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Overview",
    "Single Text Analysis",
    "Batch CSV Analysis",
    "BOC Case Study",
    "About & Limitations",
]

TIMEOUT = 120


def _fresh_app() -> AppTest:
    at = AppTest.from_file(str(APP_FILE), default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_default_page_renders():
    at = _fresh_app()
    assert not at.exception, f"app raised on first render: {at.exception}"
    assert at.title, "expected a page title"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    """Switch to each module and confirm no exception is raised."""
    at = _fresh_app()
    # the navigation radio is the first radio in the sidebar
    assert at.sidebar.radio, "navigation radio missing"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"page {page!r} raised: {at.exception}"


def _all_text(at) -> str:
    """Collect rendered text across markdown, headers, subheaders and captions."""
    parts = []
    for attr in ("markdown", "header", "subheader", "caption", "title"):
        for el in getattr(at, attr, []) or []:
            v = getattr(el, "value", None)
            if v:
                parts.append(str(v))
    return " ".join(parts)


def test_overview_mentions_both_models():
    at = _fresh_app()
    text = _all_text(at)
    assert "DistilBERT" in text
    assert "FinBERT" in text


def test_case_study_loads_real_metrics():
    """The case study must read the experiment files, not invent numbers."""
    at = _fresh_app()
    at.sidebar.radio[0].set_value("BOC Case Study").run()
    assert not at.exception, f"case study raised: {at.exception}"

    metric_pairs = {str(m.label): str(m.value) for m in at.metric}
    # dataset facts (unique labels)
    assert metric_pairs.get("Records") == "20"
    assert metric_pairs.get("Independent articles") == "20"
    assert metric_pairs.get("Negative / Neutral / Positive") == "6 / 7 / 7"

    # Both models expose "Accuracy"/"Macro F1"/"Negative recall" with the SAME
    # label, so collect the values in render order rather than by label.
    values = [str(m.value) for m in at.metric]
    joined = " | ".join(values)
    assert "70.0%" in values, f"distilbert accuracy missing: {joined}"
    assert "65.0%" in values, f"finbert accuracy missing: {joined}"
    assert "0.8333" in values, f"distilbert negative recall missing: {joined}"
    assert "0.6667" in values, f"finbert negative recall missing: {joined}"

    # tables (comparison, classification report, confusion matrices)
    assert len(at.dataframe) >= 3, "expected comparison/report/confusion tables"


def test_about_page_lists_limitations():
    at = _fresh_app()
    at.sidebar.radio[0].set_value("About & Limitations").run()
    assert not at.exception
    body = _all_text(at)
    low = body.lower()
    assert "limitation" in low
    # the substantive limitation content must be present
    assert "direction-blindness" in low or "direction" in low
    assert "not calibrated" in low or "calibrated" in low


def test_single_text_empty_input_shows_error():
    """Submitting blank text must surface a friendly error, not crash."""
    at = _fresh_app()
    at.sidebar.radio[0].set_value("Single Text Analysis").run()
    # clear the text area then press Analyse
    at.text_area[0].set_value("").run()
    at.button[0].click().run()
    assert not at.exception, f"blank input raised: {at.exception}"
    assert any("enter some text" in e.value.lower() for e in at.error), \
        "expected a friendly empty-input error"


def test_single_text_analysis_runs_end_to_end():
    """Full inference through the UI on the default example sentence."""
    at = _fresh_app()
    at.sidebar.radio[0].set_value("Single Text Analysis").run()
    at.text_area[0].set_value(
        "Bank of China's net interest margin declined in the first half of 2025."
    ).run()
    at.button[0].click().run()
    assert not at.exception, f"single-text inference raised: {at.exception}"
    body = _all_text(at)
    assert ("negative" in body.lower()) or ("neutral" in body.lower()) \
        or ("positive" in body.lower()), "no sentiment label rendered"
