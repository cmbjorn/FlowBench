@echo off
setlocal

:: Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found - using system Python.
    echo If you see import errors, create a venv first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
)

echo Installing / updating dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. See messages above.
    pause
    exit /b 1
)
echo Dependencies OK.

echo Stopping any running Streamlit instances...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" 2^>nul') do taskkill /PID %%a /F >nul 2>&1

echo.
echo Starting FlowBench...
echo If the browser does not open automatically, navigate to:
echo     http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
streamlit run app.py
pause
