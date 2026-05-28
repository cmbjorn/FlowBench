# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "No .venv found - using system Python." -ForegroundColor Yellow
    Write-Host "If you see import errors, create a venv first:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv" -ForegroundColor Yellow
    Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "    pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Installing / updating dependencies..."
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed. See messages above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Dependencies OK."

Write-Host "Stopping any running Streamlit instances..."
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "Starting FlowBench..."
Write-Host "If the browser does not open automatically, navigate to: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""
streamlit run app.py
Read-Host "Press Enter to exit"
