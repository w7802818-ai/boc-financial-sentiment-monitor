"""
AI-Powered Financial Information Monitoring for Bank of China
=============================================================

A Streamlit application that applies two sentiment models to Bank of China
financial information:

    * DistilBERT fine-tuned on a financial sentiment corpus
    * ProsusAI/finbert (pretrained)

Modules
-------
  A. Overview               project goal, model summary, business framing
  B. Single Text Analysis   one free-text item, optional side-by-side comparison
  C. Batch CSV Analysis     upload a CSV with a `text` column, export predictions
  D. BOC Case Study         real held-out results for the 20-record BOC dataset
  E. About & Limitations    training provenance, dataset scale, responsible use

Run locally
-----------
    streamlit run 03_app/app.py

Deployment notes and known blockers are documented in 03_app/README.md.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATASET_FINAL = PROJECT_ROOT / "04_dataset" / "boc_business_dataset_final.csv"
DATASET_CONFIRMED = PROJECT_ROOT / "04_dataset" / "boc_external_human_confirmed.csv"
EXPERIMENTS = PROJECT_ROOT / "06_experiments"
PREDICTIONS_CSV = EXPERIMENTS / "boc_business_predictions.csv"
COMPARISON_CSV = EXPERIMENTS / "boc_business_model_comparison.csv"
CLASS_REPORT_CSV = EXPERIMENTS / "boc_business_classification_report.csv"
CONFUSION_CSV = EXPERIMENTS / "boc_business_confusion_matrices.csv"
ERROR_CSV = EXPERIMENTS / "boc_business_error_analysis.csv"
LABEL_MAP_CSV = EXPERIMENTS / "boc_model_label_map_audit.csv"

# ---------------------------------------------------------------------------
# Canonical labels
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_COLORS = {
    "negative": "#C62828",
    "neutral": "#616161",
    "positive": "#2E7D32",
}
LABEL_EMOJI = {"negative": "▼", "neutral": "■", "positive": "▲"}

MAX_CSV_BYTES = 2 * 1024 * 1024  # 2 MB upload guard
MAX_CSV_ROWS = 500

# Direction-sensitive financial vocabulary. Used ONLY to raise an explanatory
# caveat next to a prediction - it never overrides the model output.
DIRECTIONAL_TERMS = {
    "non-performing": "a non-performing loan ratio moving DOWN is generally favourable, "
                       "while moving UP is adverse; sentiment words alone can mislead.",
    "npl": "the direction of the NPL ratio (up vs down) determines whether the news is "
           "favourable; do not read the label as a verdict on asset quality.",
    "impairment": "rising impairment losses are typically adverse even when headline "
                  "profit grows.",
    "net interest margin": "a narrowing net interest margin pressures profitability; a "
                           "widening one is favourable.",
    "margin": "margin direction matters: compression is adverse, expansion is favourable.",
    "provision": "higher provisions usually signal deteriorating asset quality.",
    "capital adequacy": "a rising capital adequacy ratio is generally favourable for "
                        "loss-absorbing capacity.",
    "penalty": "a regulatory penalty is an adverse compliance event, but a single fine "
               "does not establish misconduct beyond what the source states.",
    "fine": "a regulatory fine is an adverse compliance event.",
}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if DATASET_FINAL.exists():
        return pd.read_csv(DATASET_FINAL, keep_default_na=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV, keep_default_na=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_experiment_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    return pd.read_csv(p, keep_default_na=False) if p.exists() else pd.DataFrame()


def load_label_map() -> pd.DataFrame:
    if LABEL_MAP_CSV.exists():
        return pd.read_csv(LABEL_MAP_CSV, keep_default_na=False)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def label_badge(sentiment: str) -> str:
    color = LABEL_COLORS.get(sentiment, "#616161")
    return (f"<span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-size:0.85rem;font-weight:600;'>"
            f"{LABEL_EMOJI.get(sentiment, '')} {sentiment}</span>")


def show_negative_notice(sentiment: str) -> None:
    """A restrained prompt for adverse-leaning output. Never asserts wrongdoing."""
    if sentiment != "negative":
        return
    st.markdown(
        "<div style='border-left:5px solid #C62828;background:#FDECEA;"
        "padding:10px 14px;border-radius:4px;margin:8px 0;'>"
        "<b>⚠️ Potential adverse financial information</b><br>"
        "<span style='font-size:0.9rem;'>The model classified this item as "
        "<b>negative</b>. This is a text-classification output only — it does "
        "<b>not</b> confirm any regulatory violation, misconduct, or realised "
        "financial risk event. Always verify against the original source.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def show_disagreement_notice(a_sent: str, b_sent: str, a_name: str, b_name: str) -> None:
    if a_sent == b_sent:
        return
    st.markdown(
        f"<div style='border-left:5px solid #EF6C00;background:#FFF3E0;"
        f"padding:10px 14px;border-radius:4px;margin:8px 0;'>"
        f"<b>⚖️ Model disagreement</b><br>"
        f"<span style='font-size:0.9rem;'>{a_name} predicts <b>{a_sent}</b> while "
        f"{b_name} predicts <b>{b_sent}</b>. Conflicting labels indicate the text "
        f"is ambiguous under the current models; treat the result as inconclusive "
        f"and review the source.</span></div>",
        unsafe_allow_html=True,
    )


def directional_caveats(text: str) -> list[str]:
    low = (text or "").lower()
    return [msg for term, msg in DIRECTIONAL_TERMS.items() if term in low]


def show_directional_caveats(text: str) -> None:
    notes = directional_caveats(text)
    if not notes:
        return
    with st.expander("Direction-sensitive wording detected — read with care", expanded=False):
        st.caption(
            "These notes explain why a sentiment label can diverge from the "
            "financial meaning of the sentence. They do not change the model output."
        )
        for n in notes:
            st.markdown(f"- {n}")


def prob_bar_chart(probs: dict, title: str = "Class probabilities"):
    """Horizontal probability bars with a fixed canonical colour scheme."""
    order = ["negative", "neutral", "positive"]
    chart_df = pd.DataFrame({
        "class": order,
        "probability": [float(probs.get(k, 0.0)) for k in order],
    })
    st.markdown(f"**{title}**")
    for k in order:
        p = float(probs.get(k, 0.0))
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
            f"<span style='width:80px;font-size:0.85rem;'>{k}</span>"
            f"<span style='flex:1;background:#EEE;border-radius:6px;height:14px;display:block;'>"
            f"<span style='display:block;width:{p*100:.1f}%;height:14px;border-radius:6px;"
            f"background:{LABEL_COLORS[k]};'></span></span>"
            f"<span style='width:56px;text-align:right;font-size:0.85rem;'>{p:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_prediction(pred, model_label: str, key_suffix: str = "") -> None:
    st.markdown(f"**{model_label}**")
    st.markdown(label_badge(pred.sentiment), unsafe_allow_html=True)
    st.caption(
        f"confidence = {pred.confidence:.1%} — this is the model's softmax score for the "
        f"chosen class, **not** a calibrated probability that the prediction is correct."
    )
    prob_bar_chart(pred.probs, "Class probabilities")
    show_negative_notice(pred.sentiment)


# ---------------------------------------------------------------------------
# Page A. Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.header("Overview")
    st.markdown(
        """
### Project objective

This application monitors financial information about **Bank of China Limited**
(3988.HK / 601988.SH) by applying transformer-based sentiment classification to
its disclosures and related financial news. It is the applied layer of an
ISOM5240 group project that compares two modelling pipelines on identical
business text.

### The two models

| Pipeline | Model | Training | Label order (native) |
|---|---|---|---|
| 1 | **DistilBERT** (`boc_distilbert_sentiment`) | Fine-tuned on a financial sentiment corpus | `0 negative, 1 neutral, 2 positive` |
| 2 | **FinBERT** (`ProsusAI/finbert`) | Pretrained, no fine-tuning | `0 positive, 1 negative, 2 neutral` |

> **Why label mapping matters.** The two models order their output classes
> differently. This app translates both onto one canonical scale
> (`0 negative / 1 neutral / 2 positive`) **by label name**, so a "negative"
> from DistilBERT and a "negative" from FinBERT mean the same thing. Comparing
> raw class indices would silently invert FinBERT's results.

### How the app is used

1. **Single Text Analysis** — paste one piece of financial text and inspect the
   predicted sentiment, class probabilities and any model disagreement.
2. **Batch CSV Analysis** — upload many items at once and export the labelled
   results for downstream review.
3. **BOC Case Study** — review the measured performance of both models on a
   human-labelled Bank of China dataset.

### Business framing

The tool is designed to **triage** large volumes of financial text so that
analysts can focus attention on the items most likely to carry adverse
information. It is a screening aid, not a decision system.
"""
    )

    st.divider()
    st.subheader("What this tool is not")
    st.markdown(
        """
- It does **not** detect fraud, regulatory violations, or confirmed risk events.
- It does **not** provide investment advice or a risk rating.
- Its confidence scores are **not** calibrated probabilities of correctness.
"""
    )


# ---------------------------------------------------------------------------
# Page B. Single Text Analysis
# ---------------------------------------------------------------------------
def page_single(min_df_bytes: int = 0) -> None:
    st.header("Single Text Analysis")
    st.caption(
        "Paste one English financial news sentence or paragraph. The models were "
        "trained on English financial text; other languages are not supported."
    )

    default_text = (
        "Bank of China's net interest margin declined in the first half of 2025, "
        "reflecting pressure on lending profitability amid lower interest rates."
    )
    text = st.text_area("Financial text", value=default_text, height=150,
                        key="single_text_input")
    choice = st.radio(
        "Model",
        ["DistilBERT", "FinBERT", "Both (side-by-side)"],
        horizontal=True,
        key="single_model_choice",
    )

    run = st.button("Analyse", type="primary", key="single_run")

    if not run:
        return

    if not text or not text.strip():
        st.error("Please enter some text before analysing.")
        return

    from sentiment_engine import EngineError, get_engine

    keys = {"DistilBERT": ["distilbert"],
            "FinBERT": ["finbert"],
            "Both (side-by-side)": ["distilbert", "finbert"]}[choice]

    results = {}
    for k in keys:
        try:
            with st.spinner(f"Running {k}…"):
                results[k] = get_engine(k).predict_one(text)
        except EngineError as e:
            st.error(f"**{k} could not be loaded.**\n\n{e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"**{k} failed during inference:** {type(e).__name__}: {e}")

    if not results:
        return

    show_directional_caveats(text)
    st.divider()

    if len(results) == 1:
        k, pred = next(iter(results.items()))
        engine = get_engine(k)
        render_prediction(pred, engine.display_name)
        st.caption(f"model source: `{engine.loaded_from}` · device: `{engine.device}`")
    else:
        c1, c2 = st.columns(2)
        with c1:
            render_prediction(results["distilbert"], "DistilBERT (fine-tuned)")
        with c2:
            render_prediction(results["finbert"], "FinBERT (pretrained)")
        show_disagreement_notice(
            results["distilbert"].sentiment,
            results["finbert"].sentiment,
            "DistilBERT", "FinBERT",
        )
        agree = results["distilbert"].sentiment == results["finbert"].sentiment
        st.info(
            f"Models **agree** on this item." if agree
            else "Models **disagree** — the item is ambiguous under the current models."
        )

    st.divider()
    with st.expander("Traceability"):
        st.json({
            "input_text": text,
            "models_run": keys,
            "predictions": {k: v.as_dict() for k, v in results.items()},
        })


# ---------------------------------------------------------------------------
# Page C. Batch CSV Analysis
# ---------------------------------------------------------------------------
def page_batch() -> None:
    st.header("Batch CSV Analysis")
    st.caption(
        f"Upload a CSV containing a `text` column (max {MAX_CSV_ROWS} rows, "
        f"max {MAX_CSV_BYTES // (1024*1024)} MB). Each row is classified independently."
    )

    choice = st.radio(
        "Model",
        ["DistilBERT", "FinBERT", "Both"],
        horizontal=True,
        key="batch_model_choice",
    )

    up = st.file_uploader("CSV file", type=["csv"], key="batch_upload")

    use_sample = st.checkbox("Use the built-in 20-record BOC sample instead",
                             key="batch_use_sample")

    df_in: pd.DataFrame | None = None

    if use_sample:
        df_in = load_dataset().copy()
        if df_in.empty:
            st.error("Built-in BOC dataset not found in 04_dataset/.")
            return
        st.success(f"Loaded built-in sample: {len(df_in)} records.")
    elif up is not None:
        raw = up.getvalue()
        if len(raw) > MAX_CSV_BYTES:
            st.error(
                f"File is too large ({len(raw) / (1024*1024):.1f} MB). "
                f"Maximum allowed is {MAX_CSV_BYTES // (1024*1024)} MB."
            )
            return
        try:
            df_in = pd.read_csv(io.BytesIO(raw))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not parse the CSV file: {type(e).__name__}: {e}")
            return
        if "text" not in df_in.columns:
            st.error(
                "The CSV must contain a column named `text`. "
                f"Found columns: {list(df_in.columns)}"
            )
            return
        if len(df_in) > MAX_CSV_ROWS:
            st.error(f"CSV has {len(df_in)} rows; the limit is {MAX_CSV_ROWS}.")
            return

    if df_in is None:
        st.info("Upload a CSV or tick the sample box to continue.")
        return

    st.write(f"**Input preview** — {len(df_in)} rows")
    st.dataframe(df_in.head(10), width='stretch')

    if st.button("Run batch prediction", type="primary", key="batch_run"):
        from sentiment_engine import EngineError, get_engine

        keys = {"DistilBERT": ["distilbert"], "FinBERT": ["finbert"],
                "Both": ["distilbert", "finbert"]}[choice]

        texts = df_in["text"].fillna("").astype(str).tolist()
        out = df_in.copy()
        loaded, failed = [], []

        for k in keys:
            try:
                engine = get_engine(k)
                preds = engine.predict_batch(texts)
                out[f"{k}_sentiment"] = [p.sentiment for p in preds]
                out[f"{k}_label"] = [p.label for p in preds]
                out[f"{k}_confidence"] = [round(p.confidence, 4) for p in preds]
                out[f"{k}_prob_negative"] = [round(p.probs["negative"], 4) for p in preds]
                out[f"{k}_prob_neutral"] = [round(p.probs["neutral"], 4) for p in preds]
                out[f"{k}_prob_positive"] = [round(p.probs["positive"], 4) for p in preds]
                loaded.append(k)
            except EngineError as e:
                failed.append((k, str(e)))
                st.error(f"**{k} could not be loaded.**\n\n{e}")
            except Exception as e:  # noqa: BLE001
                failed.append((k, f"{type(e).__name__}: {e}"))
                st.error(f"**{k} failed during batch inference:** {type(e).__name__}: {e}")

        if not loaded:
            return
        if failed:
            st.warning("Some models failed; results below cover only the loaded models: "
                       + ", ".join(loaded))

        # disagreement column when both models ran
        if len(loaded) == 2:
            out["models_agree"] = (
                out[f"{loaded[0]}_sentiment"] == out[f"{loaded[1]}_sentiment"]
            ).astype(int)

        # row-count integrity
        if len(out) != len(df_in):
            st.error(f"Internal error: output rows ({len(out)}) != input rows ({len(df_in)}).")
            return
        st.success(f"Scored {len(out)} rows with {', '.join(loaded)}.")

        st.divider()
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows scored", len(out))
        neg_total, neu_total, pos_total = 0, 0, 0
        for k in loaded:
            vc = out[f"{k}_sentiment"].value_counts()
            neg_total += int(vc.get("negative", 0))
            neu_total += int(vc.get("neutral", 0))
            pos_total += int(vc.get("positive", 0))
        m2.metric("Negative outputs", neg_total)
        m3.metric("Neutral outputs", neu_total)
        m4.metric("Positive outputs", pos_total)

        for k in loaded:
            st.markdown(f"**{k} — sentiment distribution**")
            dist = out[f"{k}_sentiment"].value_counts().reindex(
                ["negative", "neutral", "positive"]).fillna(0).astype(int)
            st.bar_chart(dist)

        if len(loaded) == 2:
            disagree_n = int((out["models_agree"] == 0).sum())
            if disagree_n:
                st.warning(
                    f"⚖️ The two models disagree on **{disagree_n}** of {len(out)} rows "
                    f"({disagree_n/len(out):.1%}). Treat those rows as inconclusive."
                )

        # negative listing
        neg_rows = pd.DataFrame()
        for k in loaded:
            sub = out[out[f"{k}_sentiment"] == "negative"]
            if not sub.empty:
                sub = sub.assign(_flagged_by=k)
                neg_rows = pd.concat([neg_rows, sub], ignore_index=True)
        st.divider()
        st.subheader("Potentially adverse records")
        st.caption(
            "Items the model classified as negative. This is a screening signal, not "
            "evidence of a regulatory or financial event."
        )
        if neg_rows.empty:
            st.info("No records were classified as negative.")
        else:
            keep = ["_flagged_by"] + [c for c in ["id", "source_name", "title", "text"]
                                      if c in neg_rows.columns]
            keep += [c for c in neg_rows.columns if c.endswith("_sentiment")
                     or c.endswith("_confidence")]
            st.dataframe(neg_rows[keep].drop_duplicates(), width='stretch')

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download results CSV",
            data=csv_bytes,
            file_name="boc_batch_sentiment_results.csv",
            mime="text/csv",
            key="batch_download",
        )
        st.session_state["batch_results"] = out


# ---------------------------------------------------------------------------
# Page D. BOC Case Study
# ---------------------------------------------------------------------------
def page_case_study() -> None:
    st.header("BOC Case Study")
    st.caption(
        "Measured results on a human-labelled Bank of China dataset. Every number "
        "below is read from the saved experiment files — none are hardcoded."
    )

    preds = load_predictions()
    comparison = load_experiment_csv(str(COMPARISON_CSV))
    class_report = load_experiment_csv(str(CLASS_REPORT_CSV))
    confusion = load_experiment_csv(str(CONFUSION_CSV))
    errors = load_experiment_csv(str(ERROR_CSV))
    dataset = load_dataset()

    if preds.empty or comparison.empty:
        st.error(
            "Experiment result files were not found in 06_experiments/. "
            "Run the Step 36 evaluation scripts first."
        )
        return

    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(dataset) if not dataset.empty else len(preds))
    c2.metric("Independent articles", preds["article_id"].nunique()
              if "article_id" in preds else "—")
    if not dataset.empty:
        vc = dataset["ground_truth_label"].astype(int).value_counts()
        c3.metric("Negative / Neutral / Positive",
                  f"{int(vc.get(0,0))} / {int(vc.get(1,0))} / {int(vc.get(2,0))}")
    c4.metric("Source types", dataset["source_type"].nunique()
              if not dataset.empty else "—")

    st.divider()
    st.subheader("Model comparison (same 20 samples)")

    disp = comparison.copy()
    if "model" in disp.columns:
        disp["model"] = disp["model"].map(
            {"distilbert": "DistilBERT (fine-tuned)", "finbert": "FinBERT (pretrained)"}
        ).fillna(disp["model"])

    metric_cols = [c for c in ["model", "n_samples", "accuracy", "macro_f1",
                               "weighted_f1", "negative_recall",
                               "negative_false_positives",
                               "negative_false_negatives"] if c in disp.columns]
    st.dataframe(disp[metric_cols], width='stretch', hide_index=True)

    if "negative_recall" in comparison.columns:
        vals = comparison["negative_recall"].astype(str).tolist()
        if all(v.strip().upper() == "N/A" for v in vals):
            st.info("Negative recall is **N/A** — the ground truth contains no negative samples.")

    # highlight headline numbers
    for _, row in comparison.iterrows():
        name = {"distilbert": "DistilBERT (fine-tuned)",
                "finbert": "FinBERT (pretrained)"}.get(str(row.get("model")), row.get("model"))
        with st.expander(f"{name} — detail"):
            cc = st.columns(3)
            cc[0].metric("Accuracy", f"{float(row['accuracy']):.1%}")
            cc[1].metric("Macro F1", f"{float(row['macro_f1']):.3f}")
            cc[2].metric("Negative recall", str(row.get("negative_recall", "N/A")))

    st.divider()
    st.subheader("Per-class metrics")
    if not class_report.empty:
        pivot = class_report.copy()
        if "model" in pivot.columns:
            pivot["model"] = pivot["model"].map(
                {"distilbert": "DistilBERT", "finbert": "FinBERT"}).fillna(pivot["model"])
        st.dataframe(pivot, width='stretch', hide_index=True)
    else:
        st.info("Classification report not available.")

    st.divider()
    st.subheader("Confusion matrices")
    if not confusion.empty:
        for m in sorted(confusion["model"].unique()):
            st.markdown(f"**{m}** — rows = true, columns = predicted")
            sub = confusion[confusion["model"] == m]
            mat = sub.pivot(index="true_class", columns="pred_class", values="count")
            mat = mat.reindex(index=["negative", "neutral", "positive"],
                              columns=["negative", "neutral", "positive"]).fillna(0).astype(int)
            st.dataframe(mat, width='stretch')
    else:
        st.info("Confusion matrices not available.")

    st.divider()
    st.subheader("Typical model errors")
    st.caption(
        "The recurring failure mode is direction-blindness: the models react to "
        "sentiment vocabulary rather than the direction of the financial metric."
    )
    if errors.empty:
        st.info("No misclassifications recorded.")
    else:
        for _, e in errors.head(12).iterrows():
            st.markdown(
                f"**{e.get('id','')}** · true `{e.get('true_sentiment','')}` → "
                f"predicted `{e.get('pred_sentiment','')}` "
                f"({e.get('model','')}, conf {float(e.get('confidence',0)):.1%})"
            )
            if e.get("title"):
                st.caption(f"{e['title']}")

    st.divider()
    with st.expander("Label-mapping audit (why index order matters)"):
        lm = load_label_map()
        if lm.empty:
            st.info("Label map audit file not found.")
        else:
            st.dataframe(lm, width='stretch', hide_index=True)
            st.caption(
                "FinBERT natively orders classes as 0=positive, 1=negative, 2=neutral. "
                "All app predictions are remapped by label name onto "
                "0=negative, 1=neutral, 2=positive."
            )

    with st.expander("Traceability — source files"):
        st.json({
            "dataset": str(DATASET_FINAL.relative_to(PROJECT_ROOT)),
            "predictions": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
            "comparison": str(COMPARISON_CSV.relative_to(PROJECT_ROOT)),
            "rows_dataset": int(len(dataset)),
            "rows_predictions": int(len(preds)),
        })


# ---------------------------------------------------------------------------
# Page E. About & Limitations
# ---------------------------------------------------------------------------
def page_about() -> None:
    st.header("About & Limitations")

    st.subheader("Training data provenance")
    st.markdown(
        """
- **Financial PhraseBank** is the source corpus used to fine-tune the
  DistilBERT pipeline. Sentences come from English financial news and were
  annotated for sentiment. It is a general financial corpus, **not** a
  Bank of China corpus.
- **FinBERT (`ProsusAI/finbert`)** is a pretrained financial-language model used
  as-is, without fine-tuning on this project's data.
- Neither model was trained on the Bank of China evaluation records.
"""
    )

    st.subheader("Evaluation dataset")
    ds = load_dataset()
    if not ds.empty:
        vc = ds["ground_truth_label"].astype(int).value_counts()
        st.markdown(
            f"""
- Size: **{len(ds)} records** (10 official disclosures + 10 external news items).
- Labels: human-assigned — **{int(vc.get(0,0))} negative / {int(vc.get(1,0))} neutral / {int(vc.get(2,0))} positive**.
- Source types: `official_disclosure` (filings, quarterly and annual reports) and
  `external_news` (Reuters, Chinese business media).
- Annotation status: every record is `human_confirmed`; agent suggestions were
  never promoted automatically.
"""
        )
    else:
        st.markdown("- Dataset file not available in this environment.")

    st.subheader("Known limitations")
    st.markdown(
        """
1. **Small evaluation set.** 20 records is far too few for a stable estimate of
   real-world performance. Per-class and per-source figures are indicative only.
2. **Possible event overlap.** Some records describe the same news event on the
   same date. Where two records cover one event they are not fully independent
   observations, which can flatter or penalise the measured accuracy.
3. **Potential training-data overlap for FinBERT.** FinBERT was pretrained on
   large-scale financial text. Some benchmark phrasing may be similar to its
   pretraining data, so its advantage or disadvantage here should be read with
   caution.
4. **Direction-blindness.** Both models have been observed to follow sentiment
   words rather than the direction of the financial metric — for example reading
   "the NPL ratio **decreased**" (favourable) as negative. The app surfaces a
   caveat for such wording but does not override the model.
5. **English only.** The models operate on English financial text.
6. **Confidence is not calibrated.** The displayed confidence is the softmax
   score for the chosen class. It is **not** a calibrated probability that the
   prediction is correct.
7. **Not a risk judgement.** Outputs are text-classification signals. They must
   not be used as evidence of regulatory violations, misconduct, or realised
   financial events, and they are not investment advice.
"""
    )

    st.subheader("Responsible-use statement")
    st.markdown(
        "All predictions should be verified against the original source document "
        "before any operational use. The application is a triage aid for human "
        "analysts, and the human reviewer remains responsible for every decision."
    )

    st.divider()
    with st.expander("Environment and reproducibility"):
        import sentiment_engine as se
        st.json({
            "cache_backend": se.CACHE_BACKEND,
            "models": {k: se.model_source(k) for k in se.MODELS},
            "canonical_scale": se.LABEL_NAMES,
            "hf_model_id_env": os.environ.get("HF_MODEL_ID", "(not set)") or "(not set)",
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="AI-Powered Financial Information Monitoring for Bank of China",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 AI-Powered Financial Information Monitoring for Bank of China")
    st.caption(
        "Sentiment screening of Bank of China disclosures and financial news using "
        "two transformer models. Research prototype — not investment or compliance advice."
    )

    # make 03_app importable when Streamlit runs this file directly
    import sys
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    page = st.sidebar.radio(
        "Module",
        ["Overview", "Single Text Analysis", "Batch CSV Analysis",
         "BOC Case Study", "About & Limitations"],
        key="nav",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
**Canonical labels**
<span style='color:{LABEL_COLORS['negative']}'>■ negative</span> ·
<span style='color:{LABEL_COLORS['neutral']}'>■ neutral</span> ·
<span style='color:{LABEL_COLORS['positive']}'>■ positive</span>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Models run on CPU. First run downloads model weights.")

    if page == "Overview":
        page_overview()
    elif page == "Single Text Analysis":
        page_single()
    elif page == "Batch CSV Analysis":
        page_batch()
    elif page == "BOC Case Study":
        page_case_study()
    else:
        page_about()


if __name__ == "__main__":
    main()
