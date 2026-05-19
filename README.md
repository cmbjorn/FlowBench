# Multiphase Pipe Hydraulic Calculator

**Industrial piping system pressure drop calculator for multiphase flows (gas + liquid) with dynamic water vapor saturation.**

---

## Overview

This tool calculates pressure drops across complex, multi-directional piping networks carrying multiphase mixtures. Designed for hydrogen electrolyzer applications where gas streams (H₂/O₂ with water vapor) flow alongside dense liquid phases (30 wt% aqueous KOH).

### Key Capabilities

- **Beggs & Brill (1973)** two-phase pressure drop correlation — horizontal and vertical segments
- **Pressure marching** — gas density and void fraction re-evaluated at each segment inlet for compressibility
- **Temperature-dependent KOH properties** — density, viscosity, surface tension (0–100°C)
- **Per-segment pipe specification** — DN class, PN pressure rating, pipe material (SS316L, Duplex SS, Carbon Steel, Hastelloy C-276, Titanium Gr.2)
- **Fluoropolymer liner option** — PTFE, FEP, PFA, PVDF with liner thickness reducing effective bore
- **Minor losses** — equivalent length method (Crane TP-410), 17 fitting types
- **Erosion check** — API RP 14E, V_e = 122/√ρ_mix (m/s), C=100 continuous service
- **Flow regime detection** — stratified, slug, annular, bubble/slug, churn/annular
- **Interactive pipeline schematic** and pressure profile visualisation
- **Word document report export** (.docx) with charts embedded
- **Validation framework** — 7 reference cases with regression checks

---

## Installation

### Prerequisites

- Python **3.9 or later** — download from https://www.python.org/downloads/
  - **Windows:** tick *"Add Python to PATH"* during installation

### Setup

```bash
# 1. Open a terminal / Command Prompt in the project folder
cd path/to/multiphase_hydraulics

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

---

## How to Use

### 1. Set Process Boundaries
- Inlet pressure (1–100 bara) and temperature (5–95°C)
- Or load one of the built-in presets (H₂ side / O₂ side electrolyzer)

### 2. Define Flow Rates
- H₂ mass flow (kg/h), O₂ mass flow (kg/h), KOH lye volume flow (m³/h)

### 3. Build Pipe Geometry
- Add segments with **+ Add Segment** / **- Remove Last**
- Each segment: orientation (horizontal / vertical up / down), DN, PN, material, length, fittings
- Optional fluoropolymer liner — tick **Lined**, select material and thickness (mm)

### 4. Review Results
- **Phase Thermodynamics** — gas density, KOH properties, vapor saturation (inlet conditions)
- **Segment Analysis** — flow regime, superficial velocities, erosion ratio, ΔP per segment
- **Erosion check** — green / amber / red banner based on API RP 14E V_m/V_e ratio
- **Pipeline Schematic** — visual layout coloured by flow regime
- **Pressure Profile** — pressure vs. pipeline distance
- **Export Report** — generates a Word document with tables and embedded charts

### 5. Validate
- Run any of the 7 built-in reference cases and compare calculated vs. expected ΔP

---

## Technical Details

### Model Assumptions

1. KOH concentration constant at 30 wt% (no evaporation effect on concentration)
2. Ideal gas behaviour for H₂, O₂, H₂O vapour mixture
3. Continuous liquid phase; no flooding or flow inversion
4. Bore = f(DN, PN) only — ANSI B36.10/19 schedule, material-independent for metallic pipe
5. Lined segments: effective ID = metal bore − 2 × liner thickness; liner roughness overrides metal
6. Roughness by material — Crane TP-410 (metallic) / manufacturer data (liners)
7. Pressure marching: gas density and void fraction re-evaluated at each segment inlet
8. Erosion check: API RP 14E, V_e = 122/√ρ_mix (m/s), C=100 continuous service
9. Steady-state only — no transient effects
10. Void fraction: homogeneous model α = (x/ρg) / (x/ρg + (1−x)/ρl)

### KOH Property Correlations (30 wt%)

| Property | Correlation |
|---|---|
| Density (kg/m³) | ρ(T) = 1295 − 0.3375·(T − 20) |
| Viscosity (Pa·s) | μ(T) = μ_ref · exp(1200·(1/T − 1/T_ref)) |
| Surface tension (N/m) | σ(T) = 0.074 − 0.001125·(T − 20) |

### Pipe Materials and Roughness (Crane TP-410)

| Material | ε (m) |
|---|---|
| SS316L | 1.5 × 10⁻⁵ |
| Duplex SS 2205 | 1.5 × 10⁻⁵ |
| Carbon Steel | 4.6 × 10⁻⁵ |
| Hastelloy C-276 | 1.5 × 10⁻⁵ |
| Titanium Gr. 2 | 1.5 × 10⁻⁵ |
| PTFE / FEP / PFA liner | 5.0 × 10⁻⁸ |
| PVDF liner | 1.5 × 10⁻⁷ |

---

## File Structure

```
multiphase_hydraulics/
├── app.py                  Streamlit UI — inputs, outputs, visualisations, report export
├── multiphase_engine.py    Physics engine — KOH properties, Beggs & Brill, erosion check
├── report_generator.py     Word document (.docx) report generator
├── validation_cases.py     7 reference cases for regression testing
├── requirements.txt        Python dependencies
├── .streamlit/
│   └── config.toml         UI theme
├── LICENSE                 MIT
└── README.md               This file
```

---

## Limitations

- Beggs & Brill was developed for oil/gas systems — uncertainty on H₂/KOH is ±20–30%
- KOH correlations assume 30 wt% concentration (not adjustable)
- Single flow path only — no parallel branches or network calculations
- Steady-state; no transient, slug frequency, or liquid inventory calculations
- Temperature assumed constant along pipeline (no heat loss model)

---

## References

- Beggs, H.D. and Brill, J.P. (1973). "A Study of Two-Phase Flow in Inclined Pipes." *Journal of Petroleum Technology*, 25(5), 607–617.
- API RP 14E (2007). *Recommended Practice for Design and Installation of Offshore Production Platform Piping Systems*.
- Crane Co. (2013). *Flow of Fluids Through Valves, Fittings, and Pipe* — Technical Paper 410.
- CoolProp — open-source thermodynamic properties library
- fluids — Python fluid dynamics library

---

## License

MIT — see [LICENSE](LICENSE).
