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

Write-Host "Starting FlowBench..."
streamlit run app.py
