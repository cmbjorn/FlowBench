"""
FlowBench launcher — double-click or run: python launch.py

Works on Windows, macOS, and Linux without any shell script quirks.
Creates a virtual environment and installs dependencies on first run.
"""
import os
import sys
import subprocess
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

VENV = os.path.join(ROOT, ".venv")
if sys.platform == "win32":
    PYTHON = os.path.join(VENV, "Scripts", "python.exe")
    PIP    = os.path.join(VENV, "Scripts", "pip.exe")
else:
    PYTHON = os.path.join(VENV, "bin", "python")
    PIP    = os.path.join(VENV, "bin", "pip")

URL = "http://127.0.0.1:8501"

# ── First-time setup ─────────────────────────────────────────────────────────
if not os.path.exists(PYTHON):
    print("No virtual environment found — running first-time setup...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV])
    print("Installing dependencies (this takes a minute the first time)...")
    subprocess.check_call([PIP, "install", "-r", "requirements.txt"])
    print("Setup complete.\n")

# ── Launch Streamlit ─────────────────────────────────────────────────────────
print(f"Starting FlowBench at {URL}")
print("Close this window or press Ctrl+C to stop.\n")

proc = subprocess.Popen([
    PYTHON, "-m", "streamlit", "run", "app.py",
    "--server.address=127.0.0.1",
    "--server.port=8501",
    "--server.headless=true",
])

# Give the server a moment to start, then open the browser
time.sleep(5)
webbrowser.open(URL)

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
