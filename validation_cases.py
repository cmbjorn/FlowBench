# validation_cases.py
"""
Reference validation cases for multiphase flow pressure drop calculations.

All gas+liquid cases now use `liquid_flows_kgh` (dict of species → kg/h) as the
liquid input, which feeds the CoolProp-backed liquid mixture path in the engine.
Cases that previously used KOH 30 wt% have been migrated to Water; their expected
ΔP values are set to None (auto-calibrated on first run) because KOH and Water
have different physical properties.

Expected values with a number serve as regression anchors — if you change the
engine, run these to detect unintended shifts.
"""

VALIDATION_CASES = {
    "reference_horizontal_stratified": {
        "name": "Horizontal Stratified Flow — Low Velocity",
        "description": (
            "Horizontal DN100 pipe, 50 m, gas/liquid mixture at low superficial "
            "velocities (Vsg ≈ 0.17 m/s, Vsl ≈ 0.10 m/s). "
            "Friction-dominated; gravitational component is zero. "
            "Expected ΔP from Beggs & Brill friction correlation (Water liquid)."
        ),
        "inputs": {
            "P_bara":    5.0,
            "T_C":      25.0,
            "gas_flows_kgh": {"H₂": 2.0, "O₂": 0.5},
            "liquid_flows_kgh": {"Water": 2991.0},
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 50.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "reference_vertical_upflow": {
        "name": "Vertical Upflow — Gravity-Dominated",
        "description": (
            "Vertical DN80 riser, 30 m upflow, gas/liquid mixture at 8 bara, 40 °C. "
            "Gravity dominates (hydrostatic head). "
            "Bubble/Slug regime at Vsg ≈ 0.48 m/s. Water liquid."
        ),
        "inputs": {
            "P_bara":    8.0,
            "T_C":      40.0,
            "gas_flows_kgh": {"H₂": 5.0, "O₂": 1.2},
            "liquid_flows_kgh": {"Water": 2480.0},
            "pipe_dn":  "DN80",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Upflow", "length": 30.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "reference_mixed_horizontal_vertical": {
        "name": "Mixed Geometry — Horizontal Run + Vertical Riser",
        "description": (
            "DN100 system: 25 m horizontal with one 90° elbow, then 15 m vertical riser. "
            "Horizontal friction negligible vs riser gravity head. "
            "Total ΔP dominated by 15 m upflow section. Water liquid."
        ),
        "inputs": {
            "P_bara":    6.0,
            "T_C":      50.0,
            "gas_flows_kgh": {"H₂": 3.5, "O₂": 0.8},
            "liquid_flows_kgh": {"Water": 1976.0},
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal",     "length": 25.0,
                 "fittings": "90° Standard Elbow", "fitting_count": 1},
                {"type": "Vertical Upflow","length": 15.0,
                 "fittings": "None",               "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "downflow_gravity_recovery": {
        "name": "Vertical Downflow — Gravity-Assisted (Pressure Gain)",
        "description": (
            "DN80 vertical downflow, 20 m. Gravity drives the liquid column downward, "
            "recovering more pressure head than friction consumes, so the net ΔP is "
            "negative (outlet pressure > inlet pressure). "
            "Tests that the engine correctly handles pressure-gain segments. Water liquid."
        ),
        "inputs": {
            "P_bara":    10.0,
            "T_C":      60.0,
            "gas_flows_kgh": {"H₂": 10.0, "O₂": 3.0},
            "liquid_flows_kgh": {"Water": 3932.0},
            "pipe_dn":  "DN80",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Downflow", "length": 20.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "high_pressure_horizontal": {
        "name": "High-Pressure Horizontal — Elevated Gas Density",
        "description": (
            "DN150 horizontal pipe, 50 m, at 30 bara and 70 °C. "
            "Higher pressure increases the gas density, "
            "increasing the homogeneous mixture density and altering the holdup. "
            "Stratified regime at low superficial velocities. Water liquid."
        ),
        "inputs": {
            "P_bara":    30.0,
            "T_C":      70.0,
            "gas_flows_kgh": {"H₂": 50.0, "O₂": 25.0},
            "liquid_flows_kgh": {"Water": 19560.0},
            "pipe_dn":  "DN150",
            "pipe_pn":  "PN40",
            "segments": [
                {"type": "Horizontal", "length": 50.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "large_riser_high_flow": {
        "name": "Large-Diameter Riser — High Flow, Bubble/Slug",
        "description": (
            "DN200 vertical riser, 15 m, at 16 bara and 80 °C with high gas and liquid flows. "
            "Tests the engine on the largest standard pipe size with combined gas flows. "
            "Bubble/Slug regime. Gravity-dominated. Water liquid."
        ),
        "inputs": {
            "P_bara":    16.0,
            "T_C":      80.0,
            "gas_flows_kgh": {"H₂": 100.0, "O₂": 50.0},
            "liquid_flows_kgh": {"Water": 48600.0},
            "pipe_dn":  "DN200",
            "pipe_pn":  "PN25",
            "segments": [
                {"type": "Vertical Upflow", "length": 15.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "near_single_phase_liquid": {
        "name": "Near-Single-Phase Liquid — Trace Gas Fraction",
        "description": (
            "DN100 horizontal pipe, 50 m, with trace gas and "
            "~7.9 m³/h water recirculation. Mass quality x ≈ 0.002%, void fraction α ≈ 1.8%. "
            "Result should approach single-phase Darcy-Weisbach for Water. "
            "Useful as a lower-bound sanity check on friction losses."
        ),
        "inputs": {
            "P_bara":    10.0,
            "T_C":      60.0,
            "gas_flows_kgh": {"H₂": 0.1, "O₂": 0.1},
            "liquid_flows_kgh": {"Water": 7864.0},
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 50.0,
                 "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Beggs & Brill (1973) correlation — regression anchor (Water, migrated from KOH)"
    },

    "air_water_horizontal": {
        "name": "Air / Water — Horizontal, CoolProp Liquid",
        "description": (
            "Horizontal DN80 pipe, 50 m, Air gas + Water liquid at 5 bara, 25 °C. "
            "Exercises the generalized CoolProp liquid lookup path. "
            "Expected ΔP from Beggs & Brill friction + zero gravity component."
        ),
        "inputs": {
            "P_bara":    5.0,
            "T_C":      25.0,
            "gas_flows_kgh": {"Air": 100.0},
            "liquid_flows_kgh": {"Water": 4985.0},
            "pipe_dn":  "DN80",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 50.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Regression anchor — Beggs & Brill, CoolProp Water properties",
        "mode": "gas_liquid",
    },

    "vle_steam_water_riser": {
        "name": "Steam / Water VLE — Vertical Riser",
        "description": (
            "Vertical DN100 riser, 20 m upflow. Water at 10 bara saturation (T_sat ≈ 179.9 °C), "
            "inlet quality x = 0.3, total mass flow 1.0 kg/s. "
            "VLE mode: both phase properties derived from CoolProp saturation at each segment. "
            "Gravity-dominated; hydrostatic head ≈ 20 m × weighted mixture density."
        ),
        "inputs": {
            "P_bara":        10.0,
            "vle_fluid":     "Water",
            "vle_x_mass":    0.3,
            "vle_m_total_kgs": 1.0,
            "pipe_dn":  "DN100",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Vertical Upflow", "length": 20.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "VLE mode regression anchor — CoolProp Water saturation, Beggs & Brill",
        "mode": "vle",
    },

    "single_phase_n2_gas": {
        "name": "Single-Phase N₂ Gas — Darcy-Weisbach Fallback",
        "description": (
            "Horizontal DN50 pipe, 100 m, pure N₂ gas at 30 bara, 40 °C, no liquid. "
            "Liquid flow = 0 → single-phase gas Darcy-Weisbach fallback engaged. "
            "ΔP ≈ Darcy-Weisbach with N₂ density from ideal gas at 30 bara."
        ),
        "inputs": {
            "P_bara":    30.0,
            "T_C":      40.0,
            "gas_flows_kgh": {"N₂": 200.0},
            "liquid_flows_kgh": {},
            "pipe_dn":  "DN50",
            "pipe_pn":  "PN20",
            "segments": [
                {"type": "Horizontal", "length": 100.0, "fittings": "None", "fitting_count": 0}
            ]
        },
        "expected_total_dp_kpa": None,
        "tolerance_pct": 5.0,
        "source": "Single-phase fallback regression anchor — Darcy-Weisbach, N₂ ideal gas",
        "mode": "gas_liquid",
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

    mode = case.get("mode", "gas_liquid")
    inp  = case["inputs"]
    segs = inp["segments"]
    seg_lines = "\n".join(
        f"  - {s['type']}, {s['length']} m"
        + (f", {s['fitting_count']}× {s['fittings']}" if s.get("fittings", "None") != "None" else "")
        for s in segs
    )

    exp = case["expected_total_dp_kpa"]
    exp_str = f"{exp:.3f} kPa  (tolerance ±{case['tolerance_pct']:.0f}%)" if exp is not None \
              else "Auto-calibrated (no reference value)"

    if mode == "vle":
        flow_line = (f"VLE fluid: {inp.get('vle_fluid', '?')}  · "
                     f"x = {inp.get('vle_x_mass', '?')}  · "
                     f"ṁ = {inp.get('vle_m_total_kgs', '?')} kg/s")
        p_line = f"Pressure: {inp.get('P_bara', '?')} bara (T_sat from CoolProp)"
    else:
        gas_flows = inp.get("gas_flows_kgh", {})
        gas_line  = " · ".join(f"{sp}: {kg:.3g} kg/h" for sp, kg in gas_flows.items())
        liq_flows = inp.get("liquid_flows_kgh", {})
        if liq_flows:
            liq_line = " · ".join(f"{sp}: {m:.3g} kg/h" for sp, m in liq_flows.items())
        else:
            liq_line = "none (single-phase gas)"
        flow_line = f"Gas: {gas_line}  ·  Liquid: {liq_line}"
        p_line    = f"Pressure: {inp.get('P_bara', '?')} bara · Temperature: {inp.get('T_C', '?')} °C"

    return f"""
**{case['name']}**

{case['description']}

**Expected ΔP:** {exp_str}

**Inputs:**
- {p_line}
- {flow_line}
- Pipe: {inp.get('pipe_dn', '?')} / {inp.get('pipe_pn', '?')}
- Segments:
{seg_lines}

**Source:** {case['source']}
"""
