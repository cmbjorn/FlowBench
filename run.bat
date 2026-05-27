@echo off
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

echo Starting FlowBench...
streamlit run app.py
