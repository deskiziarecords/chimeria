#!/usr/bin/env bash
# ============================================================
# QUIMERIA / SOVEREIGN MARKET KERNEL — Ubuntu Server Launcher
# Run from project root: ./start.sh
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo " ░▒▓  QUIMERIA / SOVEREIGN MARKET KERNEL  ▓▒░"
echo ""

# Detect project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
SMK_ROOT="$PROJECT_ROOT"

echo -e "${GREEN}[INFO]${NC} Project root : $PROJECT_ROOT"
echo -e "${GREEN}[INFO]${NC} Backend dir  : $BACKEND_DIR"
echo -e "${GREEN}[INFO]${NC} SMK root     : $SMK_ROOT"
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 not found. Install with: sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}[INFO]${NC} Python version: $PYTHON_VERSION"

# Check if running as root (discourage)
if [ "$EUID" -eq 0 ]; then 
    echo -e "${YELLOW}[WARN]${NC} Running as root is not recommended for the backend service"
fi

# Install/update dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip3 install -q --upgrade pip
pip3 install -q fastapi uvicorn httpx numpy pandas scipy statsmodels scikit-learn pytz python-multipart

# Verify critical imports work
echo "[2/4] Verifying module imports..."
cd "$BACKEND_DIR"
python3 -c "
import sys
sys.path.insert(0, '$SMK_ROOT')
try:
    from smk_pipeline import SMKPipeline
    print('✓ SMKPipeline imports successfully')
except Exception as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" || {
    echo -e "${YELLOW}[WARN]${NC} SMK modules not fully available - will run in fallback mode"
}

# Export environment variables
export SMK_DIR="$SMK_ROOT"
export PYTHONPATH="$SMK_ROOT:$PYTHONPATH"
echo "[3/4] Environment configured:"
echo "      SMK_DIR=$SMK_DIR"
echo "      PYTHONPATH includes SMK root"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

echo ""
echo "[4/4] Starting FastAPI server..."
echo -e "${GREEN}       API will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}       Apache frontend should proxy /api to this${NC}"
echo ""
echo "       Logs: tail -f $PROJECT_ROOT/logs/backend.log"
echo "       Stop: Ctrl+C or pkill -f 'uvicorn main:app'"
echo ""

# Run with logging
cd "$BACKEND_DIR"
exec python3 -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    2>&1 | tee -a "$PROJECT_ROOT/logs/backend.log"
