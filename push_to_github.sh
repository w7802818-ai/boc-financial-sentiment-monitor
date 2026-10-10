#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# push_to_github.sh — prepare and push the BOC deployment repository.
#
# SAFETY: this script does NOT run automatically and does NOT create a remote
# repository. It only ever pushes to a remote URL that YOU pass in. Run it
# manually, after you have authorised publishing.
#
# Usage:
#   bash push_to_github.sh <git-remote-url> ["commit message"]
#
# Example:
#   bash push_to_github.sh https://github.com/<user>/boc-financial-monitoring.git
#
# Before running:
#   1. Create the (public) repository on GitHub yourself.
#   2. Make sure no secrets are present (the script scans first).
# ---------------------------------------------------------------------------
set -euo pipefail

REMOTE_URL="${1:-}"
COMMIT_MSG="${2:-Initial deployment commit: BOC financial information monitoring app}"

if [ -z "$REMOTE_URL" ]; then
  echo "ERROR: no remote URL provided."
  echo "Usage: bash push_to_github.sh <git-remote-url> [\"commit message\"]"
  exit 1
fi

# Move to the directory containing this script (the deployment repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================="
echo "BOC deployment -> GitHub"
echo "  repo dir   : $SCRIPT_DIR"
echo "  remote url : $REMOTE_URL"
echo "=============================================================="
echo

# --- Guard 1: refuse if a secrets file is present --------------------------
echo "[1/6] Checking for secrets..."
BLOCKED=0
for f in .env .streamlit/secrets.toml secrets.toml hf_token.txt; do
  if [ -f "$f" ]; then
    echo "  !! found $f - remove it before pushing"
    BLOCKED=1
  fi
done

# --- Guard 2: scan tracked text files for tokens / local paths -------------
# Delegated to the Python scanner, which handles path escaping reliably on all
# platforms. It ignores generic documentation placeholders such as C:\Users\...
if [ -f scan_for_secrets.py ]; then
  SCAN_PY="python"
  command -v python >/dev/null 2>&1 || SCAN_PY="python3"
  if ! $SCAN_PY scan_for_secrets.py . ; then
    BLOCKED=1
  fi
else
  echo "  !! scan_for_secrets.py not found - cannot verify safety."
  BLOCKED=1
fi

if [ "$BLOCKED" -eq 1 ]; then
  echo
  echo "ABORTED: resolve the issues above, then re-run."
  exit 1
fi
echo "  OK - no secrets or local paths detected."
echo

# --- Guard 3: refuse to commit model weights --------------------------------
echo "[2/6] Checking for model weights..."
if find . -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" | grep -q .; then
  echo "  !! model weight files present - they should live on the Hugging Face Hub"
  exit 1
fi
echo "  OK - none present."
echo

# --- Init / commit ---------------------------------------------------------
echo "[3/6] Initialising git (if needed)..."
if [ ! -d .git ]; then
  git init
  git branch -M main
else
  echo "  existing git repository found."
fi

echo "[4/6] Staging and committing..."
git add .
if git diff --cached --quiet; then
  echo "  nothing to commit."
else
  git commit -m "$COMMIT_MSG"
fi
echo

# --- Remote + push ----------------------------------------------------------
echo "[5/6] Configuring remote..."
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi
echo "  origin -> $REMOTE_URL"
echo

echo "[6/6] Pushing to main..."
git push -u origin main

echo
echo "=============================================================="
echo "PUSHED."
echo
echo "Next: create the Streamlit Cloud app with main file path 03_app/app.py"
echo "and set the HF_MODEL_ID secret. See DEPLOYMENT.md."
echo "=============================================================="
