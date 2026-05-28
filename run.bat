@echo off
setlocal

:: Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found. Run setup once first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Stopping any existing FlowBench on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" 2^>nul') do taskkill /PID %%a /F >nul 2>&1

echo.
echo Starting FlowBench...
echo If the browser does not open, navigate to:  http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py
pause
