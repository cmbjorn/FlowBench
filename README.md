# FlowBench

A Streamlit-based engineering workbench for steady-state flow calculations — pipe hydraulics, equipment sizing, and fluid thermodynamics in a single browser-based tool.

---

## What it does

FlowBench is organised into two areas: a **Pipeline Hydraulics** module for multi-segment pressure-drop analysis, and a suite of **Engineering Calculators** for common equipment sizing tasks.

### Pipeline Hydraulics (Tabs A / B and Header A / B)

Segment-by-segment pressure marching along a pipeline — inlet fluid properties are re-evaluated at each segment from the current pressure.

| Flow mode | Model |
|---|---|
| **Single-phase liquid** | Darcy-Weisbach with Churchill (1977) friction factor. Fluid properties (ρ, μ, σ) from CoolProp. Built-in KOH empirical model for 20–40 wt%, 10–90 °C. |
| **Single-phase gas** | Isothermal compressible Darcy-Weisbach. Ideal-gas density re-evaluated at each segment from the marched pressure. |
| **Gas + liquid (two-phase)** | Six ΔP correlations × two void-fraction models (12 combinations). ΔP decomposed into frictional, gravitational, and accelerational components. |
| **Saturated / VLE** | Single-component pure fluid on its saturation curve. Quality evolves isenthalpically as pressure falls. CoolProp saturation tables at each segment. |

**Two-phase correlations:** Beggs-Brill, Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, Kim-Mudawar.  
**Void-fraction models:** Homogeneous, Rouhani-1 drift-flux.

**Per-case outputs:** segment ΔP table, pipeline schematic (coloured by flow regime), pressure profile, horizontal and vertical flow regime maps with operating points, phase distribution (VLE), method sensitivity sweep, API RP 14E erosion check, valve Kv sizing. Export to Word (.docx) or Excel (.xlsx).

**Header cases** extend each branch line to a collecting manifold with multiple tap connections. The tool computes the governing-arm pressure drop and goal-seeks to a target separator pressure.

**Compare tab** overlays Cases A and B across all 12 correlation × void-fraction combinations and displays the resulting ΔP uncertainty band.

**Goal Seek tab** back-calculates the required line inlet pressure from a fixed separator pressure, or forward-calculates separator pressure from a fixed source — for both the A and B systems simultaneously.

---

### Engineering Calculators

| Tool | Description |
|---|---|
| **Fanno Flow** | Adiabatic compressible duct flow (Fanno line). Inlet Mach number from flow conditions, pipe friction → exit Mach, static and stagnation pressure/temperature, choking limit and margin. |
| **RO (Restriction Orifice)** | ISO 5167 orifice sizing for gas and liquid service. Single-stage and multistage arrays. Discharge coefficient from the Reader-Harris/Gallagher correlation. |
| **PSV (Pressure Safety Valve)** | API 520 / API 526 orifice area sizing for gas, steam, and liquid relief. Back-pressure correction factors for conventional, balanced-bellows, and pilot-operated valves. Standard orifice letter selection. |
| **Control Valve** | IEC 60534 Kv/Cv sizing for liquid (with cavitation and flashing checks), gas, and steam service. Suggested valve body size at a target opening. |
| **Dissolved Gas Flash** | Henry's law gas dissolution in water and KOH solutions (H₂, O₂, CO₂, N₂, CH₄). Dissolved concentration vs pressure and temperature; flash calculation for dissolved gas released on depressurisation. |
| **Pump** | Centrifugal pump H-Q curve fitting (3-point or tabular), system curve, operating point by intersection, speed-scaling (affinity laws), NPSH available/required check, shaft and motor power. PD pump design pressure. |
| **Line Size** | Quick DN selection — finds the minimum pipe diameter meeting velocity and ΔP/100 m criteria for a given fluid and flow rate. Service presets for process liquid, pump suction/discharge, gas, steam, and slurry. |

---

## Limitations

- One flow path at a time — not a network solver.
- Steady-state only; no transients, slugging frequency, or liquid inventory.
- Two-phase correlations: ±20–30 % typical uncertainty. Developed primarily for oil/gas service; validate against commissioning data for other fluids.
- Single-phase Darcy-Weisbach: ±5–15 % depending on roughness assumptions.
- Fanno flow: no real-gas corrections; ideal gas only.
- VLE mode: single-component pure fluids only (no mixtures).

---

## Disclaimer

FlowBench is provided for general engineering reference only. No warranty is given for accuracy, completeness, or fitness for any particular purpose. The authors accept no liability for errors, omissions, bugs, or misuse of this tool. Validate all results independently before use in any design, procurement, or safety-critical application.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
streamlit run app.py
```

Requires **Python 3.10–3.12**. Opens at `http://localhost:8501`.

### Windows notes

**Starting the app** — double-click `run.bat` (Command Prompt) or right-click `run.ps1` → *Run with PowerShell*. Both scripts automatically activate the `.venv` virtual environment if it exists. If the browser does not open on its own, navigate manually to `http://localhost:8501`.

**First-time setup** — create the virtual environment once before running the scripts:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

After that, `run.bat` / `run.ps1` handle activation automatically on every subsequent launch.

**PowerShell execution policy** — if Windows blocks `run.ps1`, run this once in PowerShell (admin not required):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**CoolProp** — pre-built wheels exist for Python 3.10–3.12 on 64-bit Windows; `pip install` works without a compiler. If pip tries to compile from source (wrong Python version or 32-bit), install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload first.

**Windows Firewall** — the app binds to `localhost` (127.0.0.1) so Windows Firewall should not block it. If it does prompt, allow access. Alternatively run on a different port:

```bat
streamlit run app.py --server.port 8502
```

---

## Pipe library

DN20–DN250, PN20/25/40. Five materials: SS316L, Duplex SS 2205, Carbon Steel, Hastelloy C-276, Titanium Gr. 2. Optional fluoropolymer liner: PTFE, FEP, PFA, PVDF. Fittings via equivalent-length method (Crane TP-410) — 17 fitting types. Inline components: control valves (Kv or ΔP mode), heat exchangers.

---

## File structure

```
FlowBench/
├── app.py                    Streamlit UI — all tabs, visualisations, export
├── multiphase_engine.py      Pipeline physics — fluid properties, ΔP, VLE
├── pump_engine.py            Pump hydraulics — H-Q curves, NPSH, power
├── fanno_engine.py           Fanno-flow (adiabatic compressible duct)
├── ro_engine.py              Restriction orifice sizing
├── psv_engine.py             Pressure safety valve sizing
├── cv_engine.py              Control valve sizing
├── dissolution_engine.py     Gas dissolution and Henry's law flash
├── report_generator.py       Word (.docx) and Excel report builders
├── validation_cases.py       Reference cases for regression checks
├── test_suite.py             Automated test runner
├── requirements.txt          Python dependencies
├── models/
│   └── pipe.py               SegmentRow dataclass
├── physics/
│   └── friction.py           Churchill friction factor
├── standards/
│   ├── piping.py             Pipe database, roughness, fitting Le/D, flange classes
│   ├── pressure_relief.py    PSV orifice letters and flange sizes
│   └── electrical.py        Standard motor frame sizes
└── workflows/
    ├── pipeline_case.py      Segment-loop calculation (extracted from app.py)
    └── pump_case.py          Pump operating-point calculation
```

---

## References

**Pipeline hydraulics**
- Beggs, H.D. and Brill, J.P. (1973). SPE-4007-PA.
- Friedel, L. (1979). European Two-Phase Flow Group Meeting, Ispra.
- Lockhart, R.W. and Martinelli, R.C. (1949). *Chem. Eng. Prog.*, 45(1), 39–48.
- Müller-Steinhagen, H. and Heck, K. (1986). *Chem. Eng. Process.*, 20(6), 297–308.
- Chisholm, D. (1973). *Int. J. Heat Mass Transfer*, 16(2), 347–358.
- Kim, S.M. and Mudawar, I. (2012). *Int. J. Heat Mass Transfer*, 55(13–14), 3246–3261.
- API RP 14E (2007). Offshore Production Platform Piping Systems.
- Crane Co. (2013). *Flow of Fluids Through Valves, Fittings, and Pipe* — TP-410.

**Thermophysical properties**
- Bell, I.H. et al. (2014). CoolProp. *Ind. Eng. Chem. Res.*, 53(6), 2498–2508.

**Equipment sizing**
- API Std 520 (2020). Sizing, Selection, and Installation of Pressure-Relieving Devices.
- API Std 526 (2017). Flanged Steel Pressure-Relief Valves.
- IEC 60534-2-1 (2011). Industrial-Process Control Valves — Flow Capacity.
- ISO 5167-2 (2022). Measurement of Fluid Flow — Orifice Plates.

---

## License

MIT — see [LICENSE](LICENSE).
