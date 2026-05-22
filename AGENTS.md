# AGENTS.md — Guidance for AI coding agents

Purpose
-------
This file tells AI coding agents what matters most to be immediately productive in this repository.

Quick commands
--------------
- Setup (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

- Run the Streamlit app:

```bash
streamlit run app.py
```

- Run the automated test suite (smoke + validation):

```bash
python test_suite.py
```

Key files
---------
- `app.py` — Streamlit UI and user input orchestration.
- `multiphase_engine.py` — Core physics engine (two-phase properties, ΔP calculations).
- `validation_cases.py` — 7 regression reference cases — do not modify values without author approval.
- `test_suite.py` — Runs validation and unit checks; use this as the canonical test harness.
- `report_generator.py` — Builds Word (.docx) reports for results.
- `requirements.txt` — Python dependencies; install into a venv.

What agents should know
----------------------
- Python 3.9+ target. Use a virtual environment.
- The app is a Streamlit app; launching via `streamlit run app.py` opens the UI.
- The `test_suite.py` script is the primary automated checks runner — run it after changes to `multiphase_engine.py` or `validation_cases.py`.
- `validation_cases.py` contains the regression anchors. Treat changes there as high-impact and request human review.
- Prefer linking to existing docs (README.md) rather than copying large blocks of text.

Conventions & pitfalls
----------------------
- Pressure units: many helpers use `bara` ↔ `Pa` conversions; follow existing conversion patterns in `test_suite.py` and `app.py`.
- Streamlit-specific code lives in `app.py` — keep computational logic in `multiphase_engine.py` so it remains testable without Streamlit.
- Report generation produces binary `.docx` output; tests assert file-like `BytesIO` results — avoid relying on filesystem paths in tests.

Suggested next customizations
----------------------------
- Create a small skill to run `python test_suite.py` and summarize results (pass/fail counts).
- Add a CI job that runs tests and `pip install -r requirements.txt` on PRs.

Links
-----
- Project README: see [README.md](README.md)

Notes
-----
Keep this file minimal — link to `README.md` and the test harness for details.
