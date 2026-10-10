# AI-Powered Financial Information Monitoring for Bank of China

A Streamlit application that applies **two transformer sentiment models** to Bank
of China (BOC) disclosures and related English financial news.

Built for the HKUST ISOM5240 group project.

> **Research prototype — non-commercial academic use only.** It is a screening
> aid for human analysts — **not** investment advice, and **not** evidence of any
> regulatory violation or realised financial event.

> ⚠️ **Licensing.** The fine-tuned DistilBERT model is derived from
> Financial PhraseBank v1.0, which is licensed **CC BY-NC-SA 3.0
> (NonCommercial)**. The model and this demo are therefore for
> **non-commercial use only**. See [`NOTICE`](NOTICE) for details and the
> attribution obligations.

---

## Live demo

> Deployment not yet performed. Once the app is live on Streamlit Community
> Cloud, the public URL goes here. **No URL is invented in advance.**

---

## The two models

Both models are normalised onto a single canonical scale:

```
0 = negative     1 = neutral     2 = positive
```

| Model | Role | Source |
|---|---|---|
| **Fine-tuned DistilBERT** | Primary model | `distilbert-base-uncased` fine-tuned on Financial PhraseBank |
| **ProsusAI/finbert** | Comparison baseline | Public pretrained checkpoint |

### ⚠️ The label-mapping trap

The two models order their native outputs **differently**:

| Model | Native `id2label` |
|---|---|
| Fine-tuned DistilBERT | `{0: negative, 1: neutral, 2: positive}` |
| `ProsusAI/finbert` | `{0: positive, 1: negative, 2: neutral}` |

All predictions are therefore remapped **by label name**, never by raw index.
Mapping by index silently swaps FinBERT's positive and negative output.
`sentiment_engine.build_name_mapping()` implements this, and it is covered by
unit tests.

---

## Dataset

The BOC case study uses a small curated evaluation set:

| Item | Value |
|---|---|
| Records | 20 |
| Composition | 6 negative / 7 neutral / 7 positive |
| Sources | BOC official disclosures + external financial news |
| Labelling | Human-confirmed, with an agent-suggested audit trail |
| Independent articles | 20 (one record per article) |

**Scale is deliberately small and this is a known limitation.** Per-class metrics
are indicative only.

---

## Application modules

| Module | Purpose |
|---|---|
| **Overview** | Project goal, the two models, business framing |
| **Single Text Analysis** | One text item → label, confidence, class probabilities, optional model comparison |
| **Batch CSV Analysis** | Upload a CSV with a `text` column → batch predictions, distribution, adverse-item listing, CSV download |
| **BOC Case Study** | Real held-out metrics for the 20-record dataset, confusion matrices, typical errors |
| **About & Limitations** | Training provenance, dataset scale, responsible-use notes |

---

## Held-out results (20-record BOC set)

| Model | Accuracy | Macro F1 | Negative recall |
|---|---|---|---|
| Fine-tuned DistilBERT | 0.70 | 0.7009 | 0.8333 |
| ProsusAI/finbert | 0.65 | 0.6444 | 0.6667 |

On the larger Financial PhraseBank test split (n=339), FinBERT scores higher
(accuracy 0.9823 vs 0.9587). The two results disagree because the in-domain BOC
set is very small — this is documented, not hidden.

---

## Repository layout

```
.
├── 03_app/
│   ├── app.py                 # Streamlit entrypoint (5 pages)
│   ├── sentiment_engine.py    # model loading + label normalisation
│   ├── requirements.txt       # deployment dependencies
│   ├── README.md              # app-level notes
│   └── tests/                 # 51 pytest tests
├── 04_dataset/                # BOC evaluation set (CSV only)
├── 06_experiments/            # result CSVs the Case Study page reads
├── .streamlit/config.toml     # runtime + theme configuration
├── .env.example               # template for HF_MODEL_ID
├── .gitignore
└── DEPLOYMENT.md              # Streamlit Cloud deployment guide
```

Model weights are **not** in this repository — they are served from the Hugging
Face Hub.

---

## Running locally

```bash
# 1. install (Python 3.12 recommended)
pip install -r 03_app/requirements.txt

# 2. configure the fine-tuned model
#    copy .env.example -> .env and set HF_MODEL_ID=<user>/<repo>

# 3. run
streamlit run 03_app/app.py
```

Then open the printed local URL (default <http://localhost:8501>).

### Offline mode

If the Hugging Face Hub is unreachable, models fall back to the local cache:

```bash
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

---

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `HF_MODEL_ID` | **Yes**, for DistilBERT | Hugging Face repo id of the fine-tuned model |
| `HF_TOKEN` | No | Only if the model repo is private |

On Streamlit Cloud, set these under **App settings → Secrets**:

```toml
HF_MODEL_ID = "your-username/boc-distilbert-sentiment"
```

FinBERT needs no configuration (`ProsusAI/finbert` is public).

---

## Tests

```bash
python -m pytest 03_app/tests/ -v
```

51 tests covering label mapping for both native orderings, probability
projection, blank/empty input handling, single and batch prediction shape, batch
row-count integrity, model-source resolution, CSV validation, app wiring, and
reproduction of the saved Step 36 predictions.

---

## Limitations

1. **Small evaluation set** — 20 human-labelled records; per-class figures are indicative only.
2. **Possible event overlap** — some records describe the same news event on the same date.
3. **Possible FinBERT training-data overlap** — pretraining corpora may contain similar phrasing.
4. **Direction-blindness** — models can follow sentiment vocabulary rather than metric direction. *"The NPL ratio decreased"* is favourable but has been predicted negative. The app shows a caveat; it does **not** override the model.
5. **English only.**
6. **Confidence is not calibrated** — it is the softmax score, not P(correct).
7. **Not a risk judgement.**

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full Streamlit Community Cloud
procedure, resource expectations, and troubleshooting.

---

## Citation

```bibtex
@misc{boc-financial-monitoring-2026,
  title  = {AI-Powered Financial Information Monitoring for Bank of China},
  author = {HKUST ISOM5240 Group Project},
  year   = {2026}
}
```

## Acknowledgements

- Base model: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- Dataset: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
- Comparison model: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
