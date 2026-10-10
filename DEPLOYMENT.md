# Deployment Guide — Streamlit Community Cloud

Step-by-step procedure for deploying the BOC Financial Information Monitoring app
to Streamlit Community Cloud, plus resource expectations and troubleshooting.

> **Status: not yet deployed.** This document describes the procedure. No public
> URL exists yet and none is fabricated here.

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| GitHub account | Hosts the public deployment repository |
| Hugging Face account | Hosts the fine-tuned DistilBERT model (public repo) |
| Streamlit Community Cloud | Free tier, sign in with GitHub |
| Python (target) | 3.12 (recommended; 3.11–3.12 are supported by all pinned deps) |

---

## 2. Step 1 — Publish the model to the Hugging Face Hub

The fine-tuned DistilBERT exists **only on local disk** and Streamlit Cloud
cannot read local Windows paths. It must be published first.

```bash
# from the project root, with your HF token available
set HF_TOKEN=hf_xxxxxxxx
python 06_experiments/publish_hf_model.py \
    --repo-id <your-username>/boc-distilbert-sentiment --execute
```

Notes:

- The script **refuses to run without `HF_TOKEN`** and never hard-codes a token.
- Without `--execute` it is a safe dry-run that uploads nothing.
- It publishes only allow-listed artefacts and aborts on any secret-scan hit.
- The model card (`README.md` inside the model directory) is published with it.

Resulting URL: `https://huggingface.co/<your-username>/boc-distilbert-sentiment`

**Upload size: ~256 MB** (7 files, dominated by `model.safetensors` ≈ 255 MB).

---

## 3. Step 2 — Publish the app repository to GitHub

The prepared deployment directory `deploy_repo/` contains only the files the app
needs (≈ 128 KB — no model weights).

```bash
cd deploy_repo
git init
git add .
git commit -m "Initial deployment commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

> Do **not** run this until you have explicitly authorised creating the public
> repository. See `06_experiments/deployment_readiness_report.md` for the
> authorisation checklist.

### What is in the repository

```
03_app/          app.py, sentiment_engine.py, requirements.txt, README.md, tests/
04_dataset/      2 CSVs (BOC evaluation set)
06_experiments/  6 result CSVs read by the Case Study page
.streamlit/      config.toml
README.md, DEPLOYMENT.md, .gitignore, .gitattributes, .env.example
```

---

## 4. Step 3 — Create the Streamlit Cloud app

1. Sign in at <https://share.streamlit.io> with GitHub.
2. **New app** → *Deploy a public app from GitHub*.
3. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `<your-username>/<repo-name>` |
   | Branch | `main` |
   | **Main file path** | `03_app/app.py` |
   | App URL | your preferred subdomain |

   The main file path **must** be `03_app/app.py`. Streamlit Cloud also needs
   `03_app/requirements.txt`, which it detects automatically from the entrypoint's
   directory.

4. Click **Deploy**.

---

## 5. Step 4 — Configure secrets

The fine-tuned model is loaded via `HF_MODEL_ID`.

**App settings → Secrets**, paste:

```toml
HF_MODEL_ID = "your-username/boc-distilbert-sentiment"
```

Then **Save** — the app reboots automatically.

- `HF_TOKEN` is **not** required if the model repository is public.
- If the model repo is private, add `HF_TOKEN = "hf_..."` to the same secrets block.
- FinBERT needs no configuration (`ProsusAI/finbert` is public).

> Streamlit Cloud exposes secrets as environment variables, so
> `os.environ.get("HF_MODEL_ID")` in `sentiment_engine.py` reads them directly —
> no code change is needed.

---

## 6. Model loading on the Cloud

| Model | Cloud behaviour |
|---|---|
| DistilBERT | Loads from `HF_MODEL_ID` on the Hub. If unset → clear error; only FinBERT remains usable. |
| FinBERT | Loads from `ProsusAI/finbert`. Always available. |
| Local fallback | `05_models/` is **not** in the repo and **never** used on the Cloud. |

First load downloads ~256 MB (DistilBERT) + ~440 MB (FinBERT) into the container
cache. Subsequent requests are served from `st.cache_resource`.

---

## 7. Dependency installation

`03_app/requirements.txt` pins compatible ranges. Key notes:

| Package | Range | Why |
|---|---|---|
| `streamlit` | `>=1.49,<2.0` | **Required ≥1.49** — the app uses `st.dataframe(width="stretch")`, which superseded `use_container_width` in 1.49 |
| `torch` | `>=2.1,<3.0` | CPU wheels only on the Cloud; no CUDA |
| `transformers` | `>=4.40,<6.0` | Validated on 5.19.x |
| `pandas` | `>=2.0,<3.1` | Validated on 3.0.6 |
| `plotly` | `>=5.20,<8.0` | Validated on 7.1.0 |

Validated combination: `python 3.12.10`, `streamlit 1.65.0`,
`torch 2.14.1+cpu`, `transformers 5.19.0`, `pandas 3.0.6`, `numpy 2.5.3`,
`plotly 7.1.0`.

---

## 8. Resource expectations

| Resource | Expectation |
|---|---|
| RAM | ~1.5–2 GB resident with both models loaded (DistilBERT ~66M + FinBERT ~110M params, float32) |
| Disk | ~700 MB for downloaded model weights in the container cache |
| CPU | Inference-only; a single short text is sub-second on the Cloud's shared CPU |
| Cold start | 1–3 minutes on first boot (dependency install + model download) |
| Concurrency | Free tier is shared; batch CSV is capped at 2 MB / 500 rows |

The app's built-in upload guards (`MAX_CSV_BYTES = 2 MB`, `MAX_CSV_ROWS = 500`)
keep the free tier within memory limits.

---

## 9. Smoke test — run after deployment

Perform these checks against the **live public URL**. Record the actual results;
do not assume success.

| # | Check | Expected |
|---|---|---|
| 1 | Public URL loads | App renders, no exception traceback |
| 2 | **Page 1 — Overview** | Project goal, both models, business framing visible |
| 3 | **Page 2 — Single Text Analysis** | Enter *"Bank of China reported a 5% increase in net profit."* → positive label + confidence + 3 class probabilities |
| 4 | **Dual-model comparison** | Enable comparison on the same text → both models' outputs shown side by side |
| 5 | **Page 3 — Batch CSV Analysis** | Load the built-in 20-record dataset → 20 rows scored; distribution chart renders |
| 6 | **Page 4 — BOC Case Study** | Accuracy / macro F1 tables and confusion matrices load for both models |
| 7 | **Page 5 — About & Limitations** | Limitations list present |
| 8 | **Download CSV** | Batch page's download button produces a valid CSV with predictions |
| 9 | Negative-item listing | Negative records listed with the directional caveat shown |
| 10 | Unknown-model guard | If `HF_MODEL_ID` is unset, a clear error appears (not a crash) |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not load model 'distilbert'. no model identifier configured` | `HF_MODEL_ID` not set | Add it under App settings → Secrets |
| `ProxyError` / `502` on model download | Hub unreachable, or private repo without token | Add `HF_TOKEN` to secrets, or set `HF_HUB_OFFLINE=1` if weights are pre-cached |
| `ModuleNotFoundError` | Missing dependency | Confirm `03_app/requirements.txt` is present at the entrypoint's level and rebuild |
| App renders but a page shows "file not found" | A data CSV missing from the repo | Re-check the `04_dataset/` and `06_experiments/` file list in `README.md` |
| `TypeError` on `st.dataframe(width=...)` | Streamlit resolved below 1.49 | Tighten the pin to `streamlit>=1.49` |
| Out-of-memory on upload | CSV too large | The 2 MB / 500-row guards should trigger; reduce the file |
| Local path error (`C:\...`) | Local fallback used on the Cloud | Should be impossible — `05_models/` is gitignored; verify it was not committed |

---

## 11. Rollback / teardown

- **Reboot** the app: App settings → *Reboot app*.
- **Change secrets**: App settings → Secrets → Save (auto-reboot).
- **Take offline**: App settings → *Delete app* (the GitHub repo remains).
- **Unpublish the model**: delete the repository on the Hugging Face Hub.

---

## 12. Authorisation checklist

These external actions require the account owner's explicit approval and are
**not** performed automatically:

- [ ] `HF_TOKEN` provided for Hugging Face publication (Step 1)
- [ ] Authorisation to create the **public** GitHub repository (Step 2)
- [ ] Authorisation to publish the app on Streamlit Community Cloud (Step 3)
- [ ] `HF_MODEL_ID` secret value confirmed (Step 4)
