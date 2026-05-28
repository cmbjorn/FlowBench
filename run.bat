@echo off
setlocal

:: Change to the folder containing this script
cd /d "%~dp0"

:: First-time setup: create venv and install dependencies if not present
if not exist ".venv\Scripts\activate.bat" (
    echo No virtual environment found. Running first-time setup...
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        echo Make sure Python 3.10-3.12 is installed and on your PATH.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo Installing dependencies ^(this takes a minute the first time^)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed. See messages above.
        pause
        exit /b 1
    )
    echo Setup complete.
    echo.
) else (
    call .venv\Scripts\activate.bat
)

:: Kill any previous instance on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" 2^>nul') do taskkill /PID %%a /F >nul 2>&1

:: Open browser (it will retry until the server is ready)
start "" "http://127.0.0.1:8501"

echo.
echo Starting FlowBench at http://127.0.0.1:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py
pause
