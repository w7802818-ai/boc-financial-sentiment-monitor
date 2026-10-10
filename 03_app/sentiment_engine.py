"""
Sentiment inference engine for the BOC Financial Information Monitoring app.

Design contract
---------------
* One canonical label scale is used everywhere:
      0 = negative, 1 = neutral, 2 = positive
* Model outputs are translated onto that scale **by label NAME**, never by raw
  index. This matters because the two supported models disagree on ordering:

      boc_distilbert_sentiment : {0: negative, 1: neutral, 2: positive}
      ProsusAI/finbert         : {0: positive, 1: negative, 2: neutral}

  Trusting the index would silently invert FinBERT's negative and positive
  predictions.
* Inference runs on CPU by default so the app works on Streamlit Cloud.
* Model loading is cached with st.cache_resource (when Streamlit is present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Canonical label scale
# ---------------------------------------------------------------------------
LABEL_NAMES: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
NAME_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}

LABEL_COLORS: Dict[str, str] = {
    "negative": "#C62828",
    "neutral": "#616161",
    "positive": "#2E7D32",
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# `hub_id` is the canonical public identifier. `local_path` is an optional
# development fallback used when the model has not been published to the Hub.
MODELS: Dict[str, Dict[str, object]] = {
    "distilbert": {
        "display": "DistilBERT (fine-tuned on Financial PhraseBank)",
        "hub_id": os.environ.get("HF_MODEL_ID", "").strip() or None,
        "local_path": "05_models/boc_distilbert_sentiment",
        "max_len": 256,
    },
    "finbert": {
        "display": "FinBERT (ProsusAI/finbert, pretrained)",
        "hub_id": "ProsusAI/finbert",
        "local_path": None,
        "max_len": 256,
    },
}

DEFAULT_BATCH_SIZE = 8


class EngineError(RuntimeError):
    """Raised when a model cannot be loaded or a request is invalid."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    label: int
    sentiment: str
    confidence: float
    probs: Dict[str, float] = field(default_factory=dict)
    model_key: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "sentiment": self.sentiment,
            "confidence": round(self.confidence, 6),
            "prob_negative": round(self.probs.get("negative", 0.0), 6),
            "prob_neutral": round(self.probs.get("neutral", 0.0), 6),
            "prob_positive": round(self.probs.get("positive", 0.0), 6),
        }


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
def build_name_mapping(id2label: Dict) -> Dict[int, int]:
    """Translate a model's `id2label` into {raw_index: canonical_label}.

    Mapping is driven purely by the label NAME, so an unusual ordering such as
    FinBERT's is handled correctly.
    """
    mapping: Dict[int, int] = {}
    for raw_idx, name in id2label.items():
        key = str(name).strip().lower()
        if key not in NAME_TO_LABEL:
            raise EngineError(
                f"Model declares an unknown label name {name!r}; "
                f"expected one of {sorted(NAME_TO_LABEL)}."
            )
        mapping[int(raw_idx)] = NAME_TO_LABEL[key]

    if sorted(mapping.values()) != [0, 1, 2]:
        raise EngineError(
            f"Label mapping is not a bijection onto {{0,1,2}}: {mapping}"
        )
    return mapping


def project_probs(raw_probs: Sequence[float], mapping: Dict[int, int]) -> List[float]:
    """Reorder a model's raw probability vector onto the canonical scale."""
    if len(raw_probs) != len(mapping):
        raise EngineError(
            f"Expected {len(mapping)} probabilities, received {len(raw_probs)}."
        )
    out = [0.0, 0.0, 0.0]
    for raw_idx, canon_idx in mapping.items():
        out[canon_idx] = float(raw_probs[raw_idx])
    return out


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """Locate the project root from this file's position (03_app/sentiment_engine.py)."""
    return Path(__file__).resolve().parent.parent


def resolve_local_path(local_path: Optional[str]) -> Optional[Path]:
    if not local_path:
        return None
    p = Path(local_path)
    if not p.is_absolute():
        p = _project_root() / p
    return p if p.exists() else None


def model_source(key: str) -> Dict[str, object]:
    """Describe where a model would be loaded from, for display purposes."""
    cfg = MODELS[key]
    local = resolve_local_path(cfg.get("local_path"))  # type: ignore[arg-type]
    return {
        "key": key,
        "display": cfg["display"],
        "hub_id": cfg.get("hub_id"),
        "local_path": str(local) if local else None,
        "resolved_from": "local" if local else ("hub" if cfg.get("hub_id") else "unavailable"),
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class SentimentEngine:
    """Wraps a single tokenizer/model pair with canonical-scale predictions."""

    def __init__(self, key: str, tokenizer, model, mapping: Dict[int, int],
                 device: str, loaded_from: str, max_len: int = 256):
        self.key = key
        self.tokenizer = tokenizer
        self.model = model
        self.mapping = mapping
        self.device = device
        self.loaded_from = loaded_from
        self.max_len = max_len
        self.id2label = dict(model.config.id2label)

    # -- introspection -----------------------------------------------------
    @property
    def display_name(self) -> str:
        return MODELS[self.key]["display"]  # type: ignore[return-value]

    # -- prediction --------------------------------------------------------
    def predict_one(self, text: str) -> Prediction:
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: Sequence[str],
                      batch_size: int = DEFAULT_BATCH_SIZE) -> List[Prediction]:
        # local import keeps module importable without torch for pure-logic tests
        import numpy as np
        import torch

        if not texts:
            return []
        cleaned = ["" if t is None else str(t) for t in texts]

        out: List[Prediction] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(cleaned), batch_size):
                chunk = cleaned[start:start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    canon = project_probs(row, self.mapping)
                    arr = np.asarray(canon, dtype=float)
                    label = int(arr.argmax())
                    out.append(
                        Prediction(
                            label=label,
                            sentiment=LABEL_NAMES[label],
                            confidence=float(arr[label]),
                            probs={LABEL_NAMES[i]: float(arr[i]) for i in range(3)},
                            model_key=self.key,
                        )
                    )
        return out

    def metadata(self) -> Dict[str, object]:
        return {
            "model_key": self.key,
            "display_name": self.display_name,
            "loaded_from": self.loaded_from,
            "device": self.device,
            "id2label": self.id2label,
            "index_to_canonical": self.mapping,
            "max_len": self.max_len,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_transformers(source: str, local_files_only: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=local_files_only
    )
    return tok, mdl


def load_engine(key: str) -> SentimentEngine:
    """Load a model by key, preferring the Hub and falling back to local files.

    Raises EngineError with an actionable message when neither source works.
    """
    if key not in MODELS:
        raise EngineError(f"Unknown model key {key!r}. Choose from {sorted(MODELS)}.")

    cfg = MODELS[key]
    hub_id = cfg.get("hub_id")
    local = resolve_local_path(cfg.get("local_path"))  # type: ignore[arg-type]
    device = _pick_device()
    errors: List[str] = []

    # 1) Hub first (Streamlit Cloud has no access to local C:\ paths)
    if hub_id:
        try:
            tok, mdl = _load_transformers(str(hub_id), local_files_only=False)
            mdl.to(device).eval()
            mapping = build_name_mapping(mdl.config.id2label)
            return SentimentEngine(key, tok, mdl, mapping, device,
                                   loaded_from=f"hub:{hub_id}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hub load of {hub_id!r} failed: {type(exc).__name__}: {exc}")
            # retry from the local cache in offline mode
            try:
                tok, mdl = _load_transformers(str(hub_id), local_files_only=True)
                mdl.to(device).eval()
                mapping = build_name_mapping(mdl.config.id2label)
                return SentimentEngine(key, tok, mdl, mapping, device,
                                       loaded_from=f"hub-cache:{hub_id}")
            except Exception as exc2:  # noqa: BLE001
                errors.append(f"cached load of {hub_id!r} failed: {type(exc2).__name__}: {exc2}")

    # 2) Local project files (development convenience)
    if local:
        try:
            tok, mdl = _load_transformers(str(local), local_files_only=True)
            mdl.to(device).eval()
            mapping = build_name_mapping(mdl.config.id2label)
            return SentimentEngine(key, tok, mdl, mapping, device,
                                   loaded_from=f"local:{local}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local load from {local} failed: {type(exc).__name__}: {exc}")

    detail = " | ".join(errors) if errors else "no model identifier configured"
    raise EngineError(
        f"Could not load model {key!r}. {detail}\n"
        "If deploying to Streamlit Cloud, publish the model to the Hugging Face Hub "
        "and set the HF_MODEL_ID environment variable (see README)."
    )


# ---------------------------------------------------------------------------
# Streamlit caching
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only inside Streamlit
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def get_engine(key: str) -> SentimentEngine:
        """Cached model loader. Falls back to plain caching outside Streamlit."""
        return load_engine(key)

    CACHE_BACKEND = "st.cache_resource"
except Exception:  # pragma: no cover
    _PLAIN_CACHE: Dict[str, SentimentEngine] = {}

    def get_engine(key: str) -> SentimentEngine:
        if key not in _PLAIN_CACHE:
            _PLAIN_CACHE[key] = load_engine(key)
        return _PLAIN_CACHE[key]

    CACHE_BACKEND = "module-level dict"


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def predict_texts(texts: Sequence[str], model_keys: Sequence[str]) -> Dict[str, List[Prediction]]:
    """Run one or more models over the same texts. Missing keys are omitted."""
    out: Dict[str, List[Prediction]] = {}
    for key in model_keys:
        out[key] = get_engine(key).predict_batch(texts)
    return out


def disagreement(a: Prediction, b: Prediction) -> bool:
    return a.label != b.label
