# Create train-only venv (no bittensor). Prefer Python 3.12.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

Write-Host "==> Creating directories"
New-Item -ItemType Directory -Force -Path runs, checkpoints, data, tokenizer | Out-Null

$env:Path = "C:\Users\1\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing uv"
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "C:\Users\1\.local\bin;$env:Path"
}

# Prefer venv on C: — D: often hits Windows file-lock issues during pip install
$venvRoot = "C:\Users\1\.venvs\sn38-train"
Write-Host "==> Creating $venvRoot with Python 3.12"
New-Item -ItemType Directory -Force -Path "C:\Users\1\.venvs" | Out-Null
$env:UV_LINK_MODE = "copy"
uv venv $venvRoot --python 3.12
Write-Host "==> Installing train requirements"
uv pip install --python "$venvRoot\Scripts\python.exe" -r scripts\train\requirements.txt
if (-not (Test-Path .venv-train)) {
    cmd /c mklink /J .venv-train $venvRoot
}

Write-Host ""
Write-Host "==> Done. Activate and smoke-test:"
Write-Host "    .\.venv-train\Scripts\Activate.ps1"
Write-Host "    python -m scripts.train.smoke_cpu"
Write-Host ""
Write-Host "==> On GPU machine, real train:"
Write-Host "    python -m scripts.train.train --config scripts/train/config_2018.yaml"
