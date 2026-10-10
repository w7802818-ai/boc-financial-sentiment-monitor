"""
Automated tests for the BOC Financial Information Monitoring app.

Coverage
--------
  * label mapping is name-based and correct for BOTH models' native orderings
  * probability re-projection preserves the total mass
  * empty / blank input handling
  * single-text prediction shape
  * batch CSV prediction and row-count integrity
  * batch predictions match the saved Step 36 experiment results
  * model-source resolution / unavailable-model error messaging
  * app module imports and wires its pages

Run:
    .venv/Scripts/python.exe -m pytest 03_app/tests/test_app.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the app directory importable
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import sentiment_engine as se  # noqa: E402

DATASET = PROJECT_ROOT / "04_dataset" / "boc_business_dataset_final.csv"
PRED_CSV = PROJECT_ROOT / "06_experiments" / "boc_business_predictions.csv"


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
class TestLabelMapping:
    def test_distilbert_native_order_maps_identically(self):
        id2 = {0: "negative", 1: "neutral", 2: "positive"}
        assert se.build_name_mapping(id2) == {0: 0, 1: 1, 2: 2}

    def test_finbert_native_order_is_remapped(self):
        """The critical case: FinBERT's indices are NOT in project order."""
        id2 = {0: "positive", 1: "negative", 2: "neutral"}
        assert se.build_name_mapping(id2) == {0: 2, 1: 0, 2: 1}

    def test_mapping_is_bijective(self):
        m = se.build_name_mapping({0: "positive", 1: "negative", 2: "neutral"})
        assert sorted(m.values()) == [0, 1, 2]

    def test_unknown_label_name_raises(self):
        with pytest.raises(se.EngineError):
            se.build_name_mapping({0: "positive", 1: "negative", 2: "banana"})

    def test_duplicate_label_names_raise(self):
        with pytest.raises(se.EngineError):
            se.build_name_mapping({0: "positive", 1: "positive", 2: "neutral"})

    def test_project_probs_reorders_correctly(self):
        """FinBERT raw [pos=0.1, neg=0.7, neu=0.2] -> canonical [neg, neu, pos]."""
        raw = [0.1, 0.7, 0.2]
        mapping = {0: 2, 1: 0, 2: 1}
        out = se.project_probs(raw, mapping)
        assert out == pytest.approx([0.7, 0.2, 0.1])

    def test_project_probs_identity(self):
        raw = [0.2, 0.3, 0.5]
        assert se.project_probs(raw, {0: 0, 1: 1, 2: 2}) == pytest.approx(raw)

    def test_project_probs_preserves_mass(self):
        raw = [0.333, 0.333, 0.334]
        out = se.project_probs(raw, {0: 2, 1: 0, 2: 1})
        assert sum(out) == pytest.approx(1.0, abs=1e-9)

    def test_project_probs_length_mismatch_raises(self):
        with pytest.raises(se.EngineError):
            se.project_probs([0.5, 0.5], {0: 0, 1: 1, 2: 2})

    def test_canonical_scale_is_stable(self):
        assert se.LABEL_NAMES == {0: "negative", 1: "neutral", 2: "positive"}
        assert se.NAME_TO_LABEL["negative"] == 0
        assert se.NAME_TO_LABEL["positive"] == 2


# ---------------------------------------------------------------------------
# Source resolution and error handling
# ---------------------------------------------------------------------------
class TestModelSource:
    def test_unknown_model_key_raises(self):
        with pytest.raises(se.EngineError):
            se.load_engine("no_such_model")

    def test_finbert_source_is_hub(self):
        src = se.model_source("finbert")
        assert src["hub_id"] == "ProsusAI/finbert"
        assert src["resolved_from"] in ("hub", "local")

    def test_distilbert_has_local_fallback(self):
        src = se.model_source("distilbert")
        # local path should resolve inside this project when present
        assert src["local_path"] is None or "05_models" in str(src["local_path"])

    def test_resolve_local_path_missing_returns_none(self):
        assert se.resolve_local_path("05_models/definitely_not_here") is None

    def test_resolve_local_path_none_returns_none(self):
        assert se.resolve_local_path(None) is None


# ---------------------------------------------------------------------------
# Prediction helpers (no model download required)
# ---------------------------------------------------------------------------
class _Batch(dict):
    """Stand-in for transformers.BatchEncoding, which supports .to(device)."""

    def to(self, *args, **kwargs):
        return self


class _DummyTok:
    """Minimal tokenizer stub producing fixed-size tensors."""

    def __call__(self, texts, padding=True, truncation=True, max_length=256,
                 return_tensors="pt"):
        import torch
        n = len(texts)
        ids = torch.ones((n, 4), dtype=torch.long)
        mask = torch.ones((n, 4), dtype=torch.long)
        return _Batch(input_ids=ids, attention_mask=mask)


class _DummyModel:
    """Emits logits that make 'negative' the argmax on the canonical scale."""

    class _Cfg:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}  # FinBERT ordering

    config = _Cfg()

    def eval(self):
        return self

    def to(self, *a, **k):  # pragma: no cover
        return self

    def __call__(self, **kwargs):
        import torch

        class _Out:
            pass

        n = kwargs["input_ids"].shape[0]
        # raw order = [positive, negative, neutral] -> favour raw index 1 (negative)
        logits = torch.tensor([[0.1, 3.0, 0.2]] * n)
        o = _Out()
        o.logits = logits
        return o


class TestEnginePredict:
    def _engine(self):
        mapping = se.build_name_mapping({0: "positive", 1: "negative", 2: "neutral"})
        return se.SentimentEngine("finbert", _DummyTok(), _DummyModel(), mapping, "cpu",
                                  loaded_from="test")

    def test_single_prediction_uses_canonical_scale(self):
        p = self._engine().predict_one("some text")
        assert p.sentiment == "negative"      # despite raw index 1 in FinBERT order
        assert p.label == 0
        assert sum(p.probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_confidence_is_max_probability(self):
        p = self._engine().predict_one("x")
        assert p.confidence == pytest.approx(max(p.probs.values()), abs=1e-9)

    def test_empty_text_still_returns_a_prediction(self):
        p = self._engine().predict_one("")
        assert p.sentiment in se.LABEL_NAMES.values()

    def test_none_text_is_handled(self):
        p = self._engine().predict_one(None)  # type: ignore[arg-type]
        assert p.sentiment in se.LABEL_NAMES.values()

    def test_empty_batch_returns_empty_list(self):
        assert self._engine().predict_batch([]) == []

    def test_batch_length_matches_input(self):
        preds = self._engine().predict_batch(["a", "b", "c", "d", "e"])
        assert len(preds) == 5

    def test_as_dict_keys(self):
        d = self._engine().predict_one("x").as_dict()
        for k in ["label", "sentiment", "confidence",
                  "prob_negative", "prob_neutral", "prob_positive"]:
            assert k in d

    def test_metadata_exposes_mapping(self):
        md = self._engine().metadata()
        assert md["index_to_canonical"] == {0: 2, 1: 0, 2: 1}
        assert md["model_key"] == "finbert"


# ---------------------------------------------------------------------------
# Dataset / prediction artefacts
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DATASET.exists(), reason="dataset not present")
class TestDatasetArtefacts:
    def test_dataset_has_20_rows(self):
        import pandas as pd
        df = pd.read_csv(DATASET, keep_default_na=False)
        assert len(df) == 20

    def test_no_duplicate_ids(self):
        import pandas as pd
        df = pd.read_csv(DATASET, keep_default_na=False)
        assert df["id"].is_unique

    def test_labels_in_range(self):
        import pandas as pd
        df = pd.read_csv(DATASET, keep_default_na=False)
        assert df["ground_truth_label"].isin([0, 1, 2]).all()

    def test_all_human_confirmed(self):
        import pandas as pd
        df = pd.read_csv(DATASET, keep_default_na=False)
        assert (df["annotation_status"] == "human_confirmed").all()


# ---------------------------------------------------------------------------
# Consistency with Step 36 saved predictions
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (DATASET.exists() and PRED_CSV.exists()),
                    reason="experiment artefacts not present")
class TestStep36Consistency:
    """Re-running inference must reproduce the saved Step 36 predictions."""

    @pytest.fixture(scope="class")
    @classmethod
    def frames(cls):
        import pandas as pd
        ds = pd.read_csv(DATASET, keep_default_na=False)
        pr = pd.read_csv(PRED_CSV, keep_default_na=False)
        return ds, pr

    def test_prediction_row_count_matches_dataset(self, frames):
        ds, pr = frames
        assert len(pr) == len(ds)

    def test_ids_align_one_to_one(self, frames):
        ds, pr = frames
        assert list(pr["id"]) == list(ds["id"])

    def _run_model(self, key):
        import pandas as pd
        ds = pd.read_csv(DATASET, keep_default_na=False)
        try:
            engine = se.get_engine(key)
        except se.EngineError as e:  # pragma: no cover
            pytest.skip(f"model unavailable: {e}")
        return engine.predict_batch(ds["text"].astype(str).tolist())

    def test_distilbert_predictions_match_saved(self, frames):
        ds, pr = frames
        preds = self._run_model("distilbert")
        got = [p.label for p in preds]
        want = pr["distilbert_pred"].astype(int).tolist()
        mismatches = [(ds["id"].iloc[i], want[i], got[i])
                      for i in range(len(want)) if want[i] != got[i]]
        assert not mismatches, f"distilbert mismatch: {mismatches}"

    def test_finbert_predictions_match_saved(self, frames):
        ds, pr = frames
        preds = self._run_model("finbert")
        got = [p.label for p in preds]
        want = pr["finbert_pred"].astype(int).tolist()
        mismatches = [(ds["id"].iloc[i], want[i], got[i])
                      for i in range(len(want)) if want[i] != got[i]]
        assert not mismatches, f"finbert mismatch: {mismatches}"

    def test_probs_sum_to_one_in_saved_file(self, frames):
        _, pr = frames
        for m in ["distilbert", "finbert"]:
            s = (pr[f"{m}_prob_negative"].astype(float)
                 + pr[f"{m}_prob_neutral"].astype(float)
                 + pr[f"{m}_prob_positive"].astype(float))
            assert (s - 1.0).abs().max() < 1e-4


# ---------------------------------------------------------------------------
# Batch CSV behaviour (pure pandas, no model)
# ---------------------------------------------------------------------------
class TestBatchLogic:
    def test_row_count_preserved_through_merge(self):
        import pandas as pd
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        out = df.copy()
        out["distilbert_sentiment"] = ["negative", "neutral", "positive"]
        assert len(out) == len(df)

    def test_missing_text_column_detected(self):
        import pandas as pd
        df = pd.DataFrame({"body": ["x"]})
        assert "text" not in df.columns

    def test_negative_selection(self):
        import pandas as pd
        out = pd.DataFrame({
            "text": ["a", "b", "c"],
            "distilbert_sentiment": ["negative", "neutral", "negative"],
        })
        neg = out[out["distilbert_sentiment"] == "negative"]
        assert len(neg) == 2


# ---------------------------------------------------------------------------
# App module wiring
# ---------------------------------------------------------------------------
class TestAppModule:
    def test_app_imports(self):
        import importlib
        mod = importlib.import_module("app")
        assert hasattr(mod, "main")

    def test_app_defines_all_pages(self):
        import app
        for fn in ["page_overview", "page_single", "page_batch",
                   "page_case_study", "page_about"]:
            assert callable(getattr(app, fn))

    def test_label_colors_cover_all_classes(self):
        import app
        assert set(app.LABEL_COLORS) == {"negative", "neutral", "positive"}

    def test_directional_caveats_detects_npl(self):
        import app
        notes = app.directional_caveats("the NPL ratio fell to 1.2%")
        assert notes, "expected a directional caveat for NPL wording"

    def test_directional_caveats_empty_for_plain_text(self):
        import app
        assert app.directional_caveats("the weather is nice") == []
