# AGENTS.md — Guidance for AI coding agents

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

## Key files

| File | Role |
|---|---|
| `app.py` | Streamlit UI — inputs, session state, visualisations, export |
| `multiphase_engine.py` | Physics engine — fluid properties, ΔP, flow regime, VLE flash |
| `report_generator.py` | Word (.docx) and comparison report builder |
| `validation_cases.py` | Regression reference cases — do not modify values without author approval |
| `test_suite.py` | Automated test runner — run after changes to the engine |

## Architecture notes

- `app.py` calls into `multiphase_engine` and `report_generator`. Keep computational logic in the engine so it remains testable without Streamlit.
- Four flow modes: `"liquid_only"`, `"gas_only"`, `"gas_liquid"`, `"vle"`. All code paths branch on this key.
- Pressure units: use `bara` in UI, `Pa` internally. Follow existing conversion patterns.
- Session state keys are namespaced per case with `k("key_name")` helpers inside `run_case()`.
- `validation_cases.py` contains regression anchors — treat changes there as high-impact.

## Testing

```bash
python test_suite.py
```

Runs validation and unit checks. Check pass/fail counts before and after engine changes.
