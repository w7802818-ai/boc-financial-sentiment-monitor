# 03_app — Streamlit application

This directory contains the Streamlit application for the HKUST ISOM5240 group
project *"AI-Powered Financial Information Monitoring for Bank of China"*.

See the repository-level [`README.md`](../README.md) for the project overview and
[`DEPLOYMENT.md`](../DEPLOYMENT.md) for the Streamlit Community Cloud procedure.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit entrypoint — defines the 5 pages and the UI |
| `sentiment_engine.py` | Model registry, loading, and label normalisation |
| `requirements.txt` | Deployment dependencies (see version notes inside) |
| `tests/` | pytest suite (51 tests) |

---

## The label-mapping rule

Both models are normalised onto one canonical scale:

```
0 = negative     1 = neutral     2 = positive
```

The two models order their native output classes **differently**:

| Model | Native `id2label` |
|---|---|
| `boc_distilbert_sentiment` (fine-tuned) | `{0: negative, 1: neutral, 2: positive}` |
| `ProsusAI/finbert` (pretrained) | `{0: positive, 1: negative, 2: neutral}` |

`sentiment_engine.build_name_mapping()` therefore maps raw indices to the
canonical scale **by label name**. Using raw indices would silently swap
FinBERT's negative and positive predictions. This is covered by unit tests.

---

## Paths — no absolute local paths

`app.py` resolves everything relative to its own location:

```python
APP_DIR       = Path(__file__).resolve().parent          # .../03_app
PROJECT_ROOT  = APP_DIR.parent                            # repository root
DATASET_FINAL = PROJECT_ROOT / "04_dataset" / "boc_business_dataset_final.csv"
EXPERIMENTS   = PROJECT_ROOT / "06_experiments"
```

There are **no** `C:\` paths, no user names, and no machine-specific references.
The app works from any checkout directory, including Streamlit Cloud.

Expected data layout relative to the repository root:

```
04_dataset/boc_business_dataset_final.csv
04_dataset/boc_external_human_confirmed.csv
06_experiments/boc_business_predictions.csv
06_experiments/boc_business_model_comparison.csv
06_experiments/boc_business_classification_report.csv
06_experiments/boc_business_confusion_matrices.csv
06_experiments/boc_business_error_analysis.csv
06_experiments/boc_model_label_map_audit.csv
```

If a file is absent, the corresponding page degrades gracefully with an
informational message rather than crashing.

---

## Model loading

Loading order implemented in `sentiment_engine.load_engine()`:

1. **Hugging Face Hub** (`hub_id`) — the deployment path.
2. **Local HF cache**, offline — used when the Hub request fails.
3. **Local project directory** (`05_models/boc_distilbert_sentiment`) — development only; never present on Streamlit Cloud.

| Model | `hub_id` | Local fallback |
|---|---|---|
| `distilbert` | `os.environ["HF_MODEL_ID"]` | `05_models/boc_distilbert_sentiment` |
| `finbert` | `ProsusAI/finbert` (public) | none |

> ⚠️ **Streamlit Cloud cannot read `C:\Users\...`.** Deploying the fine-tuned
> DistilBERT requires it to be published to the Hugging Face Hub and
> `HF_MODEL_ID` to be set. There is **no** public repository for it yet and none
> is fabricated here.

### Configuring the fine-tuned model

```toml
# .streamlit/secrets.toml  (or Streamlit Cloud → App settings → Secrets)
HF_MODEL_ID = "your-username/boc-distilbert-sentiment"
```

Or, equivalently, as an environment variable. FinBERT needs no configuration.

If `HF_MODEL_ID` is unset, the DistilBERT module reports a clear load error
rather than substituting another model.

---

## Running locally

```bash
# from the repository root
pip install -r 03_app/requirements.txt
streamlit run 03_app/app.py
```

Then open the printed local URL (default <http://localhost:8501>).

### Offline / restricted networks

If the Hugging Face Hub is unreachable, the engine falls back to the local model
cache automatically:

```bash
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

---

## Inference

* Runs on **CPU** by default and uses CUDA only if available (Streamlit Cloud is CPU-only).
* Models are cached with `st.cache_resource`.
* Sequence length 256, batch size 8.
* Predictions are deterministic; no sampling is used.

---

## Tests

```bash
python -m pytest 03_app/tests/ -v
```

51 tests covering label mapping for both native orderings, probability
projection, empty/blank input, single and batch prediction shape, batch row-count
integrity, model-source resolution, CSV validation, app wiring, and
reproduction of the saved Step 36 predictions.

---

## Repository hygiene

Do **not** commit model weights, datasets beyond the small CSV evaluation set, or
any credentials. Streamlit Cloud reads models from the Hub and data from the
repository paths listed above.
