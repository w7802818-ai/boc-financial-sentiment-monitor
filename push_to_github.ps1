# ---------------------------------------------------------------------------
# push_to_github.ps1 — prepare and push the BOC deployment repository (Windows).
#
# SAFETY: this script does NOT run automatically and does NOT create a remote
# repository. It only ever pushes to a remote URL that YOU pass in. Run it
# manually, after you have authorised publishing.
#
# Usage:
#   .\push_to_github.ps1 -RemoteUrl "https://github.com/<user>/<repo>.git"
#
# Before running:
#   1. Create the (public) repository on GitHub yourself.
#   2. Make sure no secrets are present (the script scans first).
# ---------------------------------------------------------------------------
param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,

    [string]$CommitMessage = "Initial deployment commit: BOC financial information monitoring app"
)

$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot
Set-Location $RepoDir

Write-Output "=============================================================="
Write-Output "BOC deployment -> GitHub"
Write-Output "  repo dir   : $RepoDir"
Write-Output "  remote url : $RemoteUrl"
Write-Output "=============================================================="
Write-Output ""

# --- Guard 1: refuse if a secrets file is present --------------------------
Write-Output "[1/6] Checking for secrets..."
$blocked = $false
foreach ($f in @(".env", ".streamlit\secrets.toml", "secrets.toml", "hf_token.txt")) {
    if (Test-Path $f) {
        Write-Output "  !! found $f - remove it before pushing"
        $blocked = $true
    }
}

# --- Guard 2: scan text files for tokens / local absolute paths -------------
# Delegated to the Python scanner, which handles path escaping reliably on all
# platforms. It ignores generic documentation placeholders such as C:\Users\...
if (Test-Path "scan_for_secrets.py") {
    & python "scan_for_secrets.py" "."
    if ($LASTEXITCODE -ne 0) { $blocked = $true }
} else {
    Write-Output "  !! scan_for_secrets.py not found - cannot verify safety."
    $blocked = $true
}

if ($blocked) {
    Write-Output ""
    Write-Output "ABORTED: resolve the issues above, then re-run."
    exit 1
}
Write-Output "  OK - no secrets or local paths detected."
Write-Output ""

# --- Guard 3: refuse to commit model weights --------------------------------
Write-Output "[2/6] Checking for model weights..."
$weights = Get-ChildItem -Recurse -Include "*.safetensors", "*.bin", "*.pt" -File -ErrorAction SilentlyContinue
if ($weights) {
    Write-Output "  !! model weight files present - they should live on the Hugging Face Hub"
    exit 1
}
Write-Output "  OK - none present."
Write-Output ""

# --- Init / commit ---------------------------------------------------------
Write-Output "[3/6] Initialising git (if needed)..."
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
} else {
    Write-Output "  existing git repository found."
}

Write-Output "[4/6] Staging and committing..."
git add .
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Output "  nothing to commit."
} else {
    git commit -m $CommitMessage
}
Write-Output ""

# --- Remote + push ----------------------------------------------------------
Write-Output "[5/6] Configuring remote..."
git remote get-url origin *> $null
if ($LASTEXITCODE -eq 0) {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}
Write-Output "  origin -> $RemoteUrl"
Write-Output ""

Write-Output "[6/6] Pushing to main..."
git push -u origin main

Write-Output ""
Write-Output "=============================================================="
Write-Output "PUSHED."
Write-Output ""
Write-Output "Next: create the Streamlit Cloud app with main file path 03_app/app.py"
Write-Output "and set the HF_MODEL_ID secret. See DEPLOYMENT.md."
Write-Output "=============================================================="
