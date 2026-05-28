# Change to the folder containing this script
Set-Location $PSScriptRoot

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "No .venv found in $PWD" -ForegroundColor Yellow
    Write-Host "Run setup once first:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv" -ForegroundColor Yellow
    Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "    pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Stopping any existing FlowBench on port 8501..."
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "Starting FlowBench..." -ForegroundColor Green
Write-Host "If the browser does not open, navigate to: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop."
Write-Host ""
streamlit run app.py
Read-Host "Press Enter to exit"
