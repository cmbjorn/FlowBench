# validation_cases.py
"""
Reference validation cases for multiphase flow pressure drop calculations.

Expected values are computed from the corrected engine (Beggs & Brill with
angles in degrees, correct lye mass-flow). They serve as regression anchors —
if you change the engine, run these to detect unintended shifts.

Note: Beggs & Brill was developed for oil/gas systems. These H2/KOH cases
exercise the correlation outside its original validation domain; results should
be treated as engineering estimates, not certified reference values.

--- Library canonical reference (independent check) ---
The fluids library ships its own docstring example for Beggs_Brill:
  m=0.6 kg/s, x=0.1, rhol=915, rhog=2.67, mul=180e-6, mug=14e-6,
  sigma=0.0487, P=1e7 Pa, D=0.05 m, angle=0°, L=1 m
  → 686.97 Pa  (verified: fluids 1.0.x)
This uses oil/gas properties so it cannot be expressed as H2/KOH engine
inputs; it is verified separately in the test suite.
"""

VALIDATION_CASES = {
    "reference_horizontal_stratified": {
        "name": "Horizontal Stratified Flow — Low Velocity",
        "description": (
            "Horizontal DN100 pipe, 50 m, H2/O2/KOH mixture at low superficial "
            "velocities (Vsg ≈ 0.17 m/s, Vsl ≈ 0.10 m/s). "
            "Friction-dominated; gravitational component is zero. "
            "Expected ΔP ≈ 0.28 kPa from Beggs & Brill friction correlation."
        ),
        "inputs": {
            "P_bara":    5.0,
            "T_C":      25.0,
            "m_H2_kgh":  2.0,
            "m_O2_kgh":  0.5,
            "q_lye_m3h": 3.0,
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 50.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 0.280,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    "reference_vertical_upflow": {
        "name": "Vertical Upflow — Gravity-Dominated",
        "description": (
            "Vertical DN80 riser, 30 m upflow, H2/O2/KOH at 8 bara, 40 °C. "
            "Gravity dominates (hydrostatic head ~88–379 kPa depending on holdup). "
            "Beggs & Brill holdup model gives ΔP ≈ 217 kPa. "
            "Bubble/Slug regime at Vsg ≈ 0.48 m/s."
        ),
        "inputs": {
            "P_bara":    8.0,
            "T_C":      40.0,
            "m_H2_kgh":  5.0,
            "m_O2_kgh":  1.2,
            "q_lye_m3h": 2.5,
            "pipe_dn":  "DN80",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Upflow", "length": 30.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 217.317,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    "reference_mixed_horizontal_vertical": {
        "name": "Mixed Geometry — Horizontal Run + Vertical Riser",
        "description": (
            "DN100 system: 25 m horizontal with one 90° elbow, then 15 m vertical riser. "
            "Horizontal friction negligible vs riser gravity head. "
            "Total ΔP ≈ 162 kPa, dominated by 15 m upflow section."
        ),
        "inputs": {
            "P_bara":    6.0,
            "T_C":      50.0,
            "m_H2_kgh":  3.5,
            "m_O2_kgh":  0.8,
            "q_lye_m3h": 2.0,
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal",     "length": 25.0,
                 "fittings": "90° Standard Elbow", "fitting_count": 1},
                {"type": "Vertical Upflow","length": 15.0,
                 "fittings": "None",               "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 162.467,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    # ── New cases added to extend coverage ────────────────────────────────────

    "downflow_gravity_recovery": {
        "name": "Vertical Downflow — Gravity-Assisted (Pressure Gain)",
        "description": (
            "DN80 vertical downflow, 20 m. Gravity drives the liquid column downward, "
            "recovering more pressure head than friction consumes, so the net ΔP is "
            "negative (outlet pressure > inlet pressure). "
            "Annular regime at Vsg ≈ 0.83 m/s, Vsl ≈ 0.23 m/s. "
            "Tests that the engine correctly handles pressure-gain segments."
        ),
        "inputs": {
            "P_bara":    10.0,
            "T_C":      60.0,
            "m_H2_kgh": 10.0,
            "m_O2_kgh":  3.0,
            "q_lye_m3h": 4.0,
            "pipe_dn":  "DN80",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Downflow", "length": 20.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": -49.104,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    "high_pressure_horizontal": {
        "name": "High-Pressure Horizontal — Elevated Gas Density",
        "description": (
            "DN150 horizontal pipe, 50 m, at 30 bara and 70 °C. "
            "Higher pressure triples the gas density (ρ_g ≈ 3.25 kg/m³ vs ~1 kg/m³ at 10 bara), "
            "increasing the homogeneous mixture density and altering the holdup. "
            "Stratified regime at low superficial velocities. "
            "Tests pressure-dependent gas density behaviour."
        ),
        "inputs": {
            "P_bara":    30.0,
            "T_C":      70.0,
            "m_H2_kgh": 50.0,
            "m_O2_kgh": 25.0,
            "q_lye_m3h": 20.0,
            "pipe_dn":  "DN150",
            "pipe_pn":  "PN40",
            "segments": [
                {"type": "Horizontal", "length": 50.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 1.292,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    "large_riser_high_flow": {
        "name": "Large-Diameter Riser — High Flow, Bubble/Slug",
        "description": (
            "DN200 vertical riser, 15 m, at 16 bara and 80 °C with high gas and lye flows. "
            "Tests the engine on the largest standard pipe size with combined H2 and O2 gas. "
            "Bubble/Slug regime at Vsg ≈ 0.83 m/s, Vsl ≈ 0.43 m/s. "
            "Gravity-dominated: ΔP ≈ 94 kPa over 15 m."
        ),
        "inputs": {
            "P_bara":    16.0,
            "T_C":      80.0,
            "m_H2_kgh": 100.0,
            "m_O2_kgh":  50.0,
            "q_lye_m3h": 50.0,
            "pipe_dn":  "DN200",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Upflow", "length": 15.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 93.758,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference"
    },

    "near_single_phase_liquid": {
        "name": "Near-Single-Phase Liquid — Trace Gas Fraction",
        "description": (
            "DN100 horizontal pipe, 50 m, with trace gas (H2 = O2 = 0.1 kg/h) and "
            "8 m³/h lye recirculation. Mass quality x ≈ 0.002%, void fraction α ≈ 1.8%. "
            "Result should approach single-phase Darcy-Weisbach for KOH lye. "
            "D-W predicts ≈ 507 Pa; Beggs-Brill gives ≈ 674 Pa (~33% higher, "
            "acceptable since BB is a multiphase correlation not tuned for single-phase). "
            "Useful as a lower-bound sanity check on friction losses."
        ),
        "inputs": {
            "P_bara":    10.0,
            "T_C":      60.0,
            "m_H2_kgh":  0.1,
            "m_O2_kgh":  0.1,
            "q_lye_m3h": 8.0,
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 50.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": 0.674,
        "tolerance_pct": 5.0,
        "source": (
            "Beggs & Brill (1973) correlation — H2/KOH inputs, regression reference. "
            "Independent check: Darcy-Weisbach for KOH lye at same conditions gives "
            "≈ 0.507 kPa (BB is ~33% higher, consistent with multiphase-correlation bias)."
        )
    },
}


def get_validation_case(case_name):
    return VALIDATION_CASES.get(case_name, None)


def list_validation_cases():
    return [
        (key, case["name"], case["description"])
        for key, case in VALIDATION_CASES.items()
    ]


def get_case_info(case_name):
    case = VALIDATION_CASES.get(case_name)
    if not case:
        return "Validation case not found."

    segs = case["inputs"]["segments"]
    seg_lines = "\n".join(
        f"  - {s['type']}, {s['length']} m"
        + (f", {s['fitting_count']}× {s['fittings']}" if s["fittings"] != "None" else "")
        for s in segs
    )

    return f"""
**{case['name']}**

{case['description']}

**Expected ΔP:** {case['expected_total_dp_kpa']:.3f} kPa  (tolerance ±{case['tolerance_pct']:.0f}%)

**Inputs:**
- Pressure: {case['inputs']['P_bara']} bara · Temperature: {case['inputs']['T_C']} °C
- H₂: {case['inputs']['m_H2_kgh']} kg/h · O₂: {case['inputs']['m_O2_kgh']} kg/h · Lye: {case['inputs']['q_lye_m3h']} m³/h
- Pipe: {case['inputs']['pipe_dn']} / {case['inputs']['pipe_pn']}
- Segments:
{seg_lines}

**Source:** {case['source']}
"""
