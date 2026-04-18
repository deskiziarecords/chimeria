@echo off
REM ============================================================
REM  QUIMERIA / SOVEREIGN MARKET KERNEL  —  Windows Launcher
REM  Double-click this file OR run from CMD in the project root
REM ============================================================

echo.
echo  ░▒▓  QUIMERIA / SOVEREIGN MARKET KERNEL  ▓▒░
echo.

REM Detect where this batch file lives — that is the project root
set "PROJECT_ROOT=%~dp0"
REM Remove trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "SMK_ROOT=%PROJECT_ROOT%"

echo [INFO] Project root : %PROJECT_ROOT%
echo [INFO] Backend dir  : %BACKEND_DIR%
echo [INFO] SMK root     : %SMK_ROOT%
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from python.org and add to PATH.
    pause
    exit /b 1
)

REM Install deps
echo [1/3] Installing Python deps...
pip install fastapi uvicorn httpx numpy pandas scipy scikit-learn pytz python-multipart --quiet

REM Export SMK_DIR so smk_pipeline.py can find the modules
set "SMK_DIR=%SMK_ROOT%"
echo [2/3] SMK_DIR set to: %SMK_DIR%
echo.

REM Start server
echo [3/3] Starting server on http://0.0.0.0:8000
echo       Open browser to: http://localhost:8000
echo.
echo       Press Ctrl+C to stop
echo.

cd /d "%BACKEND_DIR%"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
