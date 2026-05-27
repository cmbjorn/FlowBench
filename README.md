# FlowBench

Streamlit app for steady-state pipe hydraulics — pressure drop, flow regime, and velocity analysis for single-phase and two-phase flows.

---

## What it does

FlowBench calculates segment-by-segment pressure drop along a pipeline, marching from inlet to outlet and re-evaluating fluid properties at each segment. It covers four flow modes:

| Mode | Model |
|---|---|
| **Single-phase liquid** | Darcy-Weisbach, Churchill friction factor. CoolProp physical properties. KOH solution available as a built-in empirical model (20–40 wt%, 10–90 °C). |
| **Single-phase gas** | Isothermal compressible Darcy-Weisbach. Ideal-gas density re-evaluated at each segment from the marched pressure. No Fanno-flow or choking check. |
| **Gas + liquid (two-phase)** | Six ΔP correlations (Beggs-Brill, Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, Kim-Mudawar) × two void-fraction models (Homogeneous, Rouhani-1). ΔP decomposed into frictional, gravitational, and accelerational components. |
| **Saturated / VLE** | Single-component pure fluid on its saturation curve. Quality x evolves isenthalpically as pressure falls. CoolProp provides all phase properties at each segment. Phase distribution reported at each stream boundary. |

---

## What it is not

- Not a process simulator or flow network solver — one flow path at a time.
- No transient effects, slugging frequency, or liquid inventory.
- No Fanno-flow choking, no relief-valve sizing.
- Two-phase correlations carry ±20–30 % uncertainty; they were developed for oil/gas and may not transfer well to all services. Always validate against commissioning data.
- Single-phase Darcy-Weisbach accuracy is limited by roughness assumptions (±5–15 % typical).

---

## Workflow

**Tabs A / B** — individual branch lines. Set flow mode, fluid, pipe geometry (segments, fittings, liner), then calculate. Export Word (.docx) or Excel (.xlsx) from each tab.

**Header A / B** — collecting manifolds with multiple taps. Computes the governing-arm pressure drop and goal-seeks to a target separator pressure.

**Compare** — side-by-side overlay of two cases. Runs all 12 correlation × void-fraction combinations for both cases and shows the ΔP uncertainty band.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
streamlit run app.py
```

Requires **Python 3.10 or 3.11** (recommended). Opens at `http://localhost:8501`.

### Windows notes

CoolProp is a C++ extension. Pre-built wheels exist for Python 3.10 and 3.11 on 64-bit Windows — use one of those Python versions and `pip install` will work without a compiler.

If you see `ImportError: CoolProp` or the app hangs on localhost without showing anything:

```bash
pip install "CoolProp>=6.4.0"
```

If pip tries to compile from source (no wheel for your Python version), install
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the "Desktop development with C++" workload first, then retry.

If port 8501 is blocked by Windows Firewall, either allow it or run on a different port:

```bash
streamlit run app.py --server.port 8502
```

---

## Pipe library

DN20–DN250, PN20/25/40. Five materials (SS316L, Duplex SS 2205, Carbon Steel, Hastelloy C-276, Titanium Gr. 2). Optional fluoropolymer liner (PTFE, FEP, PFA, PVDF). 17 fitting types via equivalent-length method (Crane TP-410). Inline components: valves (Kv or ΔP mode), heat exchangers.

---

## File structure

```
FlowBench/
├── app.py                  Streamlit UI — inputs, visualisations, export
├── multiphase_engine.py    Physics engine — properties, ΔP, regime, VLE
├── report_generator.py     Word (.docx) report builder
├── validation_cases.py     Reference cases for regression checks
├── test_suite.py           Automated test runner
├── requirements.txt        Python dependencies
└── run.sh                  Helper: kill port 8501 and relaunch
```

---

## References

- Beggs, H.D. and Brill, J.P. (1973). SPE-4007-PA.
- Friedel, L. (1979). Improved friction pressure drop correlations for horizontal and vertical two-phase pipe flow. *European Two-Phase Flow Group Meeting*, Ispra.
- Lockhart, R.W. and Martinelli, R.C. (1949). *Chem. Eng. Prog.*, 45(1), 39–48.
- Müller-Steinhagen, H. and Heck, K. (1986). *Chem. Eng. Process.*, 20(6), 297–308.
- Chisholm, D. (1973). *Int. J. Heat Mass Transfer*, 16(2), 347–358.
- Kim, S.M. and Mudawar, I. (2012). *Int. J. Heat Mass Transfer*, 55(13–14), 3246–3261.
- Bell, I.H. et al. (2014). CoolProp. *Ind. Eng. Chem. Res.*, 53(6), 2498–2508.
- API RP 14E (2007). *Recommended Practice for Design and Installation of Offshore Production Platform Piping Systems*.
- Crane Co. (2013). *Flow of Fluids Through Valves, Fittings, and Pipe* — TP-410.

---

## License

MIT — see [LICENSE](LICENSE).
