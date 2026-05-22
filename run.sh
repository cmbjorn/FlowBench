#!/usr/bin/env bash
# Kill any process holding port 8501, any matching "streamlit run", and any
# orphaned kaleido subprocesses left over from a previous crash, then start fresh.
echo "Stopping any running Streamlit instances..."
lsof -ti :8501 | xargs kill -9 2>/dev/null
pkill -f "streamlit run app.py" 2>/dev/null
pkill -9 -f kaleido 2>/dev/null
sleep 1
echo "Starting Multiphase Pressure Drop Calculator..."
streamlit run app.py
