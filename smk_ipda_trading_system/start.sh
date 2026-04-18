#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# QUIMERIA / SMK — startup script
# Run from the project root (quimeria/)
# ─────────────────────────────────────────────────────────────────
set -e

BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"
SMK_DIR="${SMK_DIR:-./smk_ipda_trading_system}"   # set this to your actual SMK path

echo ""
echo "  ▓▓▓  QUIMERIA / SOVEREIGN MARKET KERNEL  ▓▓▓"
echo ""

# 1. Install deps
echo "[1/3] Installing Python deps..."
pip install -q fastapi uvicorn httpx numpy pandas scipy statsmodels scikit-learn pytz python-multipart

# 2. Symlink or copy SMK modules into backend so imports resolve
echo "[2/3] Linking SMK modules..."
if [ -d "$SMK_DIR" ]; then
  for pkg in core lambda_sensors causality liquidity risk market detectors; do
    SRC="$SMK_DIR/$pkg"
    DST="$BACKEND_DIR/$pkg"
    if [ -d "$SRC" ] && [ ! -e "$DST" ]; then
      ln -sf "$(realpath "$SRC")" "$DST"
      echo "  linked: $pkg"
    fi
  done
  # Top-level files
  for f in lambda_fusion_engine.py ipda_utils.py; do
    SRC="$SMK_DIR/$f"
    DST="$BACKEND_DIR/$f"
    if [ -f "$SRC" ] && [ ! -e "$DST" ]; then
      ln -sf "$(realpath "$SRC")" "$DST"
      echo "  linked: $f"
    fi
  done
else
  echo "  ⚠  SMK_DIR not found at '$SMK_DIR' — set SMK_DIR env var"
  echo "     Pipeline will run in fallback mode (numpy-only detectors)"
fi

# 3. Launch
echo "[3/3] Starting server on http://0.0.0.0:8000 ..."
echo ""
cd "$BACKEND_DIR"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
