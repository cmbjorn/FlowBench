#!/usr/bin/env bash
# Kill any process holding port 8501 or matching "streamlit run", then start fresh.
echo "Stopping any running Streamlit instances..."
lsof -ti :8501 | xargs kill -9 2>/dev/null
pkill -f "streamlit run app.py" 2>/dev/null
sleep 1
echo "Starting Multiphase Pressure Drop Calculator..."
streamlit run app.py
