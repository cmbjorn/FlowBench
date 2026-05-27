#!/usr/bin/env bash
# Kill any process holding port 8501, any matching "streamlit run", and any
# orphaned kaleido subprocesses left over from a previous crash, then start fresh.

set -e

echo "Installing / updating dependencies..."
pip install -q -r requirements.txt
echo "Dependencies OK."

echo "Stopping any running Streamlit instances..."
lsof -ti :8501 | xargs kill -9 2>/dev/null || true
pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -9 -f kaleido 2>/dev/null || true
sleep 1

echo "Starting FlowBench..."
streamlit run app.py
