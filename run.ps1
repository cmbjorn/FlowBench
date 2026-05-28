# Change to the folder containing this script
Set-Location $PSScriptRoot

# First-time setup: create venv and install dependencies if not present
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "No virtual environment found. Running first-time setup..." -ForegroundColor Yellow
    Write-Host ""
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Could not create virtual environment." -ForegroundColor Red
        Write-Host "Make sure Python 3.10-3.12 is installed and on your PATH." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    & ".venv\Scripts\Activate.ps1"
    Write-Host "Installing dependencies (this takes a minute the first time)..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed. See messages above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host ""
} else {
    & ".venv\Scripts\Activate.ps1"
}

# Kill any previous instance on port 8501
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

# Open browser (it will retry until the server is ready)
Start-Process "http://127.0.0.1:8501"

Write-Host ""
Write-Host "Starting FlowBench at http://127.0.0.1:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop."
Write-Host ""
streamlit run app.py
Read-Host "Press Enter to exit"
