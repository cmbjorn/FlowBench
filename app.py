# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import multiphase_engine as engine
import validation_cases as val_cases
import report_generator
import hashlib
import json

st.set_page_config(
    page_title="Multiphase Hydraulic Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}
hr { margin: 0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Multiphase Pipe Hydraulic Engine")
st.caption(
    "Two-phase pressure drop · H₂ / O₂ / 30 wt% KOH · "
    "Beggs & Brill (1973) · SS316 · Steady-state"
)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Documentation")

    with st.expander("About", expanded=False):
        st.markdown("""
        Calculate pressure drops in piping systems carrying multiphase flows
        (gas + liquid) with dynamic water vapor saturation and temperature effects.

        **Correlation:** Beggs & Brill (1973) — horizontal and inclined pipes
        **Liquid:** 30 wt% aqueous KOH, temperature-dependent ρ, μ, σ
        **Gas:** H₂ + O₂ + H₂O vapor, ideal gas
        **Minor losses:** Equivalent length method (Le/D)
        """)

    with st.expander("Model Assumptions", expanded=False):
        st.markdown("""
        1. KOH concentration constant (no evaporation effect on concentration)
        2. Ideal gas behavior for H₂, O₂, H₂O vapor mixture
        3. Continuous liquid phase; no flooding or flow inversion
        4. Bore = f(DN, PN) only — ANSI B36.10/19 schedule, material-independent for metallic pipe
        5. Lined segments: effective ID = metal bore − 2 × liner thickness; liner roughness overrides metal
        6. Roughness set per segment by material (Crane TP-410) or liner (manufacturer data)
        7. Pressure marching: gas density and void fraction re-evaluated at each segment inlet pressure
        8. Erosion check: API RP 14E, V_e = 122/√ρ_mix (m/s), C=100 continuous service
        8. Validated temperature range: 5–95 °C
        9. Validated pressure range: 1–100 bara
        10. Steady-state only — no transient effects
        11. Void fraction displayed is homogeneous model α = (x/ρg) / (x/ρg + (1−x)/ρl)
        """)

    with st.expander("Flow Regimes", expanded=False):
        st.markdown("""
        **Horizontal**
        - Stratified — Vsg < 1 m/s
        - Intermittent (Slug) — 1 ≤ Vsg < 5 m/s
        - Annular/Dispersed — Vsg ≥ 5 m/s

        **Vertical Upflow**
        - Bubble/Slug — Vsg < 3 m/s
        - Churn/Annular — Vsg ≥ 3 m/s

        **Vertical Downflow**
        - Falling Film — Vsl > 2 m/s
        - Annular — Vsl ≤ 2 m/s
        """)

    with st.expander("Property Correlations", expanded=False):
        st.markdown("""
        **KOH Density**  ρ(T) = 1295 − 0.3375·(T − 20) kg/m³
        **KOH Viscosity**  μ(T) = μ_ref · exp(1200·(1/T − 1/T_ref)) Pa·s
        **Surface Tension**  σ(T) = 0.074 − 0.001125·(T − 20) N/m
        Valid range: 0–100 °C
        """)

    with st.expander("References", expanded=False):
        st.markdown("""
        - Beggs & Brill (1973) — SPE-4007-PA
        - CoolProp — open-source thermodynamic library
        - fluids — Python fluid dynamics library
        - Yaws' Chemical Properties Handbook
        """)

# ============================================================================
# SESSION STATE
# ============================================================================
if "segments" not in st.session_state:
    st.session_state.segments = [
        {"type": "Horizontal",      "dn": "DN50", "pn": "PN40", "material": "SS316L",
         "length": 12.0, "fittings": "None", "fitting_count": 0,
         "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
        {"type": "Vertical Upflow", "dn": "DN50", "pn": "PN40", "material": "SS316L",
         "length":  4.5, "fittings": "None", "fitting_count": 0,
         "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
    ]
# Migrate segments from earlier sessions
_VALID_MATS   = set(engine.MATERIAL_ROUGHNESS.keys())
_VALID_LINERS = set(engine.LINER_ROUGHNESS.keys())
for _seg in st.session_state.segments:
    _seg.setdefault("dn", "DN50")
    _seg.setdefault("pn", "PN40")
    _seg.setdefault("material", "SS316L")
    _seg.setdefault("lined", False)
    _seg.setdefault("liner_material", "FEP")
    _seg.setdefault("liner_thickness_mm", 1.0)
    if _seg["material"] not in _VALID_MATS:
        _seg["material"] = "SS316L"
    if _seg["liner_material"] not in _VALID_LINERS:
        _seg["liner_material"] = "FEP"

# ============================================================================
# MAIN COLUMNS
# ============================================================================
col_in, col_out = st.columns([1, 1.2])

# ── LEFT: INPUTS ─────────────────────────────────────────────────────────────
with col_in:
    st.header("Inputs")

    PRESETS = {
        "Custom":                              {"P": 10.0, "T": 50.0, "H2":   8.0, "O2":  2.0, "lye":  1.5},
        "H₂ side — 80 °C, 16 bara, 30% KOH": {"P": 16.0, "T": 80.0, "H2": 100.0, "O2":  0.1, "lye": 50.0},
        "O₂ side — 80 °C, 16 bara, 30% KOH": {"P": 16.0, "T": 80.0, "H2":   0.1, "O2": 50.0, "lye": 50.0},
    }

    with st.container(border=True):
        st.subheader("Quick Presets")
        selected_preset = st.selectbox("Load preset conditions", list(PRESETS.keys()))
        preset_vals = PRESETS[selected_preset]

    with st.container(border=True):
        st.subheader("Process Boundaries")
        p1, p2 = st.columns(2)
        P_bara = p1.number_input("Inlet Pressure (bara)", min_value=1.0, max_value=100.0,
                                  value=preset_vals["P"], step=1.0)
        T_C    = p2.number_input("Temperature (°C)",      min_value=5.0,  max_value=95.0,
                                  value=preset_vals["T"], step=5.0)

    with st.container(border=True):
        st.subheader("Mass & Volume Flows")
        f1, f2, f3 = st.columns(3)
        m_H2  = f1.number_input("H₂ (kg/h)",   min_value=0.1, value=preset_vals["H2"],  step=1.0)
        m_O2  = f2.number_input("O₂ (kg/h)",   min_value=0.0, value=preset_vals["O2"],  step=0.1)
        q_lye = f3.number_input("Lye (m³/h)",  min_value=0.1, value=preset_vals["lye"], step=1.0)

    with st.container(border=True):
        st.subheader("Pipe Geometry")
        current_specs = []

        DN_OPTIONS     = list(engine.PIPE_DATABASE.keys())
        PN_OPTIONS     = ["PN20", "PN25", "PN40"]
        MAT_OPTIONS    = list(engine.MATERIAL_ROUGHNESS.keys())
        LINER_OPTIONS  = list(engine.LINER_ROUGHNESS.keys())
        fitting_options = ["None"] + list(engine.FITTING_Le_over_D.keys())

        for i, seg in enumerate(st.session_state.segments):
            st.markdown(f"**Segment #{i+1}**")

            # Row 1: orientation, DN, PN, length
            g1, g2, g3, g4 = st.columns([1.3, 0.8, 0.7, 0.7])
            t = g1.selectbox(
                "Orientation",
                ["Horizontal", "Vertical Upflow", "Vertical Downflow"],
                key=f"t_{i}",
                index=["Horizontal", "Vertical Upflow", "Vertical Downflow"].index(seg["type"])
            )
            dn = g2.selectbox("DN", DN_OPTIONS, key=f"dn_{i}",
                              index=DN_OPTIONS.index(seg.get("dn", "DN50")))
            pn = g3.selectbox("PN", PN_OPTIONS, key=f"pn_{i}",
                              index=PN_OPTIONS.index(seg.get("pn", "PN40")))
            l  = g4.number_input("Length (m)", min_value=0.1, value=float(seg["length"]),
                                 step=1.0, key=f"l_{i}")

            # Row 2: material, minor loss, qty, lined checkbox
            g5, g6, g7, g8 = st.columns([1.1, 2.0, 0.6, 0.65])
            _mat_default = seg.get("material", "SS316L")
            mat = g5.selectbox("Material", MAT_OPTIONS, key=f"m_{i}",
                               index=MAT_OPTIONS.index(_mat_default) if _mat_default in MAT_OPTIONS else 0)
            fitting_idx = 0
            if seg["fittings"] in engine.FITTING_Le_over_D:
                fitting_idx = list(engine.FITTING_Le_over_D.keys()).index(seg["fittings"]) + 1
            f = g6.selectbox("Minor Loss", fitting_options, key=f"f_{i}", index=fitting_idx)
            c = g7.number_input("Qty", min_value=0, value=int(seg["fitting_count"]), key=f"c_{i}")
            g8.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            lined = g8.checkbox("Lined", value=bool(seg.get("lined", False)), key=f"lined_{i}")

            # Row 3: liner options — only shown when lined is checked
            _lmat    = seg.get("liner_material", "FEP")
            _lthk_mm = float(seg.get("liner_thickness_mm", 1.0))
            if lined:
                g9, g10 = st.columns([1.6, 1.0])
                _lmat = g9.selectbox(
                    "Liner Material", LINER_OPTIONS, key=f"lmat_{i}",
                    index=LINER_OPTIONS.index(_lmat) if _lmat in LINER_OPTIONS else 0,
                )
                _lthk_mm = g10.number_input(
                    "Liner Thickness (mm)", min_value=0.1, max_value=20.0,
                    value=_lthk_mm, step=0.5, key=f"lthk_{i}",
                )

            D_seg  = engine.PIPE_DATABASE[dn][pn]
            D_eff  = D_seg - 2 * (_lthk_mm / 1000.0) if lined else D_seg
            rough  = engine.LINER_ROUGHNESS[_lmat] if lined else engine.MATERIAL_ROUGHNESS[mat]
            if lined:
                st.caption(
                    f"Bore {D_seg*1000:.1f} mm → **ID {D_eff*1000:.1f} mm**"
                    f"  ·  ε {rough*1e6:.3g} µm  ·  {mat} + {_lmat} {_lthk_mm:.1f} mm liner"
                )
            else:
                st.caption(f"ID {D_seg*1000:.1f} mm  ·  ε {rough*1e6:.2g} µm  ·  {mat}")

            current_specs.append({
                "type": t, "dn": dn, "pn": pn, "material": mat, "length": l,
                "fittings": f if f != "None" else "None",
                "fitting_count": c,
                "lined": lined,
                "liner_material": _lmat,
                "liner_thickness_mm": _lthk_mm,
            })

        st.session_state.segments = current_specs

        b1, b2 = st.columns(2)
        if b1.button("+ Add Segment"):
            _last = st.session_state.segments[-1]
            st.session_state.segments.append({
                "type": "Horizontal",
                "dn": _last.get("dn", "DN50"),
                "pn": _last.get("pn", "PN40"),
                "material": _last.get("material", "SS316L"),
                "length": 2.0, "fittings": "None", "fitting_count": 0,
                "lined": _last.get("lined", False),
                "liner_material": _last.get("liner_material", "FEP"),
                "liner_thickness_mm": _last.get("liner_thickness_mm", 1.0),
            })
            st.rerun()
        if b2.button("- Remove Last") and len(st.session_state.segments) > 1:
            st.session_state.segments.pop()
            st.rerun()

# ── RIGHT: OUTPUTS ───────────────────────────────────────────────────────────
with col_out:
    st.header("Output")

    is_valid, warn_list = engine.validate_input_bounds(P_bara, T_C, m_H2, m_O2, q_lye)
    for w in warn_list:
        st.warning(w)

    props = engine.calculate_two_phase_properties(P_bara, T_C, m_H2, m_O2, q_lye)

    # ── Phase Thermodynamics ──────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Phase Thermodynamics — Inlet")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ρ_gas",     f"{props['rho_g']:.3f} kg/m³")
        c2.metric("ρ_liquid",  f"{props['rho_l']:.1f} kg/m³")
        c3.metric("μ_liquid",  f"{props['mu_l']*1e3:.3f} mPa·s")
        c4.metric("σ_surface", f"{props['sigma']*1e3:.2f} mN/m")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Mass Quality x", f"{props['x_gas']*100:.3f} %")
        c6.metric("Void Fraction α", f"{props['alpha']*100:.1f} %")
        c7.metric("P_sat (H₂O)",    f"{props['P_sat_H2O_pa']/1e5:.3f} bara")
        c8.metric("H₂O Vapor",      f"{props['m_vapor_h2o_kgh']:.3f} kg/h")

    # ── Inlet Flow Conditions (based on Segment 1 effective bore) ───────────
    _seg1    = st.session_state.segments[0]
    _D_bore1 = engine.PIPE_DATABASE[_seg1["dn"]][_seg1["pn"]]
    _lthk1   = _seg1.get("liner_thickness_mm", 1.0) / 1000.0
    _D_in    = _D_bore1 - 2 * _lthk1 if _seg1.get("lined", False) else _D_bore1
    _A_in    = 0.25 * np.pi * _D_in ** 2
    Vsg_in = (props['x_gas'] * props['m_total_kgs'] / props['rho_g']) / _A_in \
             if props['rho_g'] > 0 else 0.0
    Vsl_in = ((1 - props['x_gas']) * props['m_total_kgs'] / props['rho_l']) / _A_in \
             if props['rho_l'] > 0 else 0.0
    Re_sl  = props['rho_l'] * Vsl_in * _D_in / props['mu_l'] \
             if props['mu_l'] > 0 else 0.0

    with st.container(border=True):
        _liner_note1 = (f" + {_seg1['liner_material']} {_seg1['liner_thickness_mm']:.1f}mm liner"
                        if _seg1.get("lined") else "")
        st.subheader(f"Inlet Flow Conditions  ({_seg1['dn']} / {_seg1['pn']}{_liner_note1})")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("V_sg (gas)",    f"{Vsg_in:.3f} m/s")
        v2.metric("V_sl (liquid)", f"{Vsl_in:.3f} m/s")
        v3.metric("V_m (mixture)", f"{Vsg_in + Vsl_in:.3f} m/s")
        v4.metric("Re_liquid",     f"{Re_sl:,.0f}")

    # ── Segment Calculations ──────────────────────────────────────────────────
    current_P          = P_bara * 1e5
    grid_records       = []
    cumulative_positions = []
    cumulative_distance  = 0.0
    pressure_profile_x   = [0.0]
    pressure_profile_y   = [P_bara]
    regime_bands         = []

    for i, seg in enumerate(st.session_state.segments):
        D_seg     = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        _lined    = seg.get("lined", False)
        _lthk_m   = seg.get("liner_thickness_mm", 1.0) / 1000.0
        _lmat     = seg.get("liner_material", "FEP")
        D_eff     = D_seg - 2 * _lthk_m if _lined else D_seg
        rough_seg = (engine.LINER_ROUGHNESS[_lmat] if _lined
                     else engine.MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])

        # Re-evaluate thermodynamic properties at the local segment inlet pressure
        # so that gas density, void fraction and superficial velocities reflect
        # actual local conditions rather than fixed inlet values.
        props_seg = engine.calculate_two_phase_properties(
            current_P / 1e5, T_C, m_H2, m_O2, q_lye
        )

        angle = {"Horizontal": 0.0,
                 "Vertical Upflow": np.pi / 2.0,
                 "Vertical Downflow": -np.pi / 2.0}[seg["type"]]

        le_fit = 0.0
        if seg["fittings"] in engine.FITTING_Le_over_D:
            le_fit = engine.FITTING_Le_over_D[seg["fittings"]] * D_eff * seg["fitting_count"]
        L_eff = seg["length"] + le_fit

        dP_Pa, regime, dP_per_dz, Vsg, Vsl = engine.calculate_segment_pressure_drop(
            props_seg, D_eff, rough_seg, L_eff, angle
        )

        V_m = Vsg + Vsl
        V_e, rho_mix = engine.calculate_erosion_velocity(
            props_seg["rho_g"], props_seg["rho_l"], props_seg["x_gas"]
        )
        erosion_ratio = V_m / V_e if V_e > 0 else 0.0

        end_P = current_P - dP_Pa
        _mat_str = seg.get("material", "SS316L")
        if _lined:
            _mat_str += f" / {_lmat} {seg.get('liner_thickness_mm', 1.0):.1f}mm"
        grid_records.append({
            "Seg":            f"#{i+1}",
            "Pipe":           f"{seg['dn']}/{seg['pn']}",
            "ID (mm)":        round(D_eff * 1000, 1),
            "Material":       _mat_str,
            "Type":           seg["type"],
            "P_in (bara)":    round(current_P / 1e5, 4),
            "ρ_g (kg/m³)":   round(props_seg["rho_g"], 4),
            "L (m)":          seg["length"],
            "L_eff (m)":      round(L_eff, 2),
            "Regime":         regime,
            "V_sg (m/s)":     round(Vsg, 3),
            "V_sl (m/s)":     round(Vsl, 3),
            "V_m (m/s)":      round(V_m, 3),
            "V_e (m/s)":      round(V_e, 2),
            "V_m/V_e":        round(erosion_ratio, 3),
            "dP/dz (Pa/m)":   round(dP_per_dz, 2),
            "ΔP (kPa)":       round(dP_Pa / 1000, 3),
            "P_out (bara)":   round(end_P / 1e5, 4),
        })

        current_P = end_P
        cumulative_distance += L_eff
        cumulative_positions.append(cumulative_distance)
        pressure_profile_x.append(cumulative_distance)
        pressure_profile_y.append(max(0.1, current_P / 1e5))
        regime_bands.append(regime)

    # ── Erosion velocity check (API RP 14E, C=100 continuous service) ─────────
    _max_ratio  = max((r["V_m/V_e"] for r in grid_records), default=0.0)
    _worst_seg  = next((r["Seg"] for r in grid_records if r["V_m/V_e"] == _max_ratio), "")
    if _max_ratio >= 1.0:
        st.error(
            f"**Erosion limit exceeded** — Segment {_worst_seg}: "
            f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, C=100).  "
            f"Reduce velocity or increase pipe diameter."
        )
    elif _max_ratio >= 0.8:
        st.warning(
            f"**Approaching erosion limit** — Segment {_worst_seg}: "
            f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, C=100, limit = 1.0)."
        )
    else:
        st.success(
            f"Erosion check OK — worst segment {_worst_seg}: "
            f"V_m/V_e = {_max_ratio:.2f}  (API RP 14E, C=100, limit = 1.0)."
        )

    st.subheader("Segment Analysis")
    st.dataframe(
        pd.DataFrame(grid_records),
        column_config={
            "ID (mm)":        st.column_config.NumberColumn(format="%.1f"),
            "P_in (bara)":    st.column_config.NumberColumn(format="%.4f"),
            "ρ_g (kg/m³)":   st.column_config.NumberColumn(format="%.4f"),
            "L (m)":          st.column_config.NumberColumn(format="%.1f"),
            "L_eff (m)":      st.column_config.NumberColumn(format="%.1f"),
            "V_sg (m/s)":     st.column_config.NumberColumn(format="%.3f"),
            "V_sl (m/s)":     st.column_config.NumberColumn(format="%.3f"),
            "V_m (m/s)":      st.column_config.NumberColumn(format="%.3f"),
            "V_e (m/s)":      st.column_config.NumberColumn(format="%.2f"),
            "V_m/V_e":        st.column_config.NumberColumn(format="%.3f"),
            "dP/dz (Pa/m)":   st.column_config.NumberColumn(format="%.2f"),
            "ΔP (kPa)":       st.column_config.NumberColumn(format="%.3f"),
            "P_out (bara)":   st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True
    )

    # ── System Totals ─────────────────────────────────────────────────────────
    total_dp_kpa         = ((P_bara * 1e5) - current_P) / 1000.0
    outlet_pressure_bara = max(0.1, current_P / 1e5)
    pipe_length_m        = sum(s["length"] for s in st.session_state.segments)

    with st.container(border=True):
        st.subheader("System Totals")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total ΔP",        f"{total_dp_kpa:.3f} kPa",
                  delta=f"−{total_dp_kpa/100:.4f} bar", delta_color="inverse")
        s2.metric("Outlet Pressure",  f"{outlet_pressure_bara:.4f} bara")
        s3.metric("Pipe Length",      f"{pipe_length_m:.1f} m")
        s4.metric("Effective Length", f"{cumulative_distance:.1f} m")

# ============================================================================
# VISUALISATIONS — build figures first so they are accessible to the report
# ============================================================================

# ── Build pipeline schematic figure ─────────────────────────────────────────
_nodes = [(0.0, 0.0)]
for _seg in st.session_state.segments:
    _xl, _yl = _nodes[-1]
    if _seg["type"] == "Horizontal":
        _nodes.append((_xl + _seg["length"], _yl))
    elif _seg["type"] == "Vertical Upflow":
        _nodes.append((_xl, _yl + _seg["length"]))
    else:
        _nodes.append((_xl, _yl - _seg["length"]))

_RHEX = {
    "Stratified":          "#3B82F6",
    "Intermittent (Slug)": "#D97706",
    "Annular/Dispersed":   "#059669",
    "Bubble/Slug":         "#7C3AED",
    "Churn/Annular":       "#DC2626",
    "Falling Film":        "#0891B2",
    "Annular":             "#0891B2",
}
_DN_LW = {
    "DN40": 5, "DN50": 7, "DN80": 9, "DN100": 11,
    "DN150": 15, "DN200": 19, "DN250": 23,
}

fig_sch = go.Figure()
_seen_reg = set()

for _i, (_seg, _rec) in enumerate(zip(st.session_state.segments, grid_records)):
    _x0, _y0 = _nodes[_i]
    _x1, _y1 = _nodes[_i + 1]
    _regime  = _rec["Regime"]
    _col     = next((v for k, v in _RHEX.items() if k in _regime), "#64748B")
    _lw      = _DN_LW.get(_seg["dn"], 10)
    _show    = _regime not in _seen_reg
    _seen_reg.add(_regime)

    _liner_hover = (
        f"Liner: {_seg['liner_material']} {_seg['liner_thickness_mm']:.1f} mm"
        f"  →  ID {_rec['ID (mm)']:.1f} mm<br>"
        if _seg.get("lined") else ""
    )
    fig_sch.add_trace(go.Scatter(
        x=[_x0, _x1], y=[_y0, _y1],
        mode="lines",
        line=dict(color=_col, width=_lw),
        name=_regime, legendgroup=_regime,
        showlegend=_show,
        hovertemplate=(
            f"<b>Segment #{_i+1}  {_seg['dn']}/{_seg['pn']}</b><br>"
            + _liner_hover
            + f"{_seg['type']},  L = {_seg['length']:.1f} m<br>"
            f"Regime: {_regime}<br>"
            f"ΔP: {_rec['ΔP (kPa)']:.3f} kPa  ·  "
            f"V_sg: {_rec['V_sg (m/s)']:.3f} m/s<extra></extra>"
        ),
    ))
    fig_sch.add_annotation(
        x=_x0 + (_x1 - _x0) * 0.65, y=_y0 + (_y1 - _y0) * 0.65,
        ax=_x0 + (_x1 - _x0) * 0.50, ay=_y0 + (_y1 - _y0) * 0.50,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.8,
        arrowwidth=2.5, arrowcolor=_col,
    )
    fig_sch.add_annotation(
        x=(_x0 + _x1) / 2, y=(_y0 + _y1) / 2,
        text=f"<b>#{_i+1}</b> {_seg['dn']}",
        showarrow=False,
        font=dict(size=10, color="#1E293B"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=_col, borderwidth=1.5, borderpad=3,
        xanchor="center", yanchor="middle",
    )

fig_sch.add_trace(go.Scatter(
    x=[_nodes[0][0]], y=[_nodes[0][1]],
    mode="markers+text",
    marker=dict(size=14, color="#059669", symbol="circle",
                line=dict(color="white", width=2)),
    text=["IN"], textposition="bottom center",
    textfont=dict(size=11, color="#059669"),
    showlegend=False,
    hovertemplate=f"Inlet  ·  {P_bara:.2f} bara<extra></extra>",
))
_out_tpos = "top center" if _nodes[-1][1] >= _nodes[0][1] else "bottom center"
fig_sch.add_trace(go.Scatter(
    x=[_nodes[-1][0]], y=[_nodes[-1][1]],
    mode="markers+text",
    marker=dict(size=14, color="#DC2626", symbol="circle",
                line=dict(color="white", width=2)),
    text=["OUT"], textposition=_out_tpos,
    textfont=dict(size=11, color="#DC2626"),
    showlegend=False,
    hovertemplate=f"Outlet  ·  {outlet_pressure_bara:.4f} bara<extra></extra>",
))

_all_x = [n[0] for n in _nodes]
_all_y = [n[1] for n in _nodes]
_xspan = max(_all_x) - min(_all_x)
_yspan = max(_all_y) - min(_all_y)
_xpad  = max(_xspan * 0.12, 2.0)
_ypad  = max(_yspan * 0.12, max(_xspan * 0.10, 2.0))

fig_sch.update_layout(
    template="plotly_white",
    height=480,
    margin=dict(l=60, r=20, t=30, b=50),
    hovermode="closest",
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(
        title="Horizontal Distance (m)",
        gridcolor="#F1F5F9", zeroline=True,
        zerolinecolor="#CBD5E1", linecolor="#E2E8F0",
        range=[min(_all_x) - _xpad, max(_all_x) + _xpad],
    ),
    yaxis=dict(
        title="Elevation (m)",
        gridcolor="#F1F5F9", zeroline=True,
        zerolinecolor="#CBD5E1", linecolor="#E2E8F0",
        range=[min(_all_y) - _ypad, max(_all_y) + _ypad],
    ),
    legend=dict(
        title="Flow Regime",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E2E8F0", borderwidth=1,
        font=dict(size=11),
    ),
    font=dict(size=12, color="#374151"),
)

# ── Build pressure profile figure ────────────────────────────────────────────
REGIME_COLORS = {
    "Stratified":          "rgba(219, 234, 254, 0.70)",
    "Intermittent (Slug)": "rgba(254, 243, 199, 0.70)",
    "Annular/Dispersed":   "rgba(209, 250, 229, 0.70)",
    "Bubble/Slug":         "rgba(237, 233, 254, 0.70)",
    "Churn/Annular":       "rgba(254, 226, 226, 0.70)",
    "Falling Film":        "rgba(207, 250, 254, 0.70)",
    "Annular":             "rgba(207, 250, 254, 0.70)",
}
REGIME_BORDER = {
    "Stratified":          "#93C5FD",
    "Intermittent (Slug)": "#FCD34D",
    "Annular/Dispersed":   "#6EE7B7",
    "Bubble/Slug":         "#C4B5FD",
    "Churn/Annular":       "#FCA5A5",
    "Falling Film":        "#67E8F9",
    "Annular":             "#67E8F9",
}

fig_prof = go.Figure()
_x0_prof = 0.0
for _x1_prof, _regime_prof in zip(cumulative_positions, regime_bands):
    _fill   = next((v for k, v in REGIME_COLORS.items() if k in _regime_prof), "rgba(241,245,249,0.60)")
    _border = next((v for k, v in REGIME_BORDER.items() if k in _regime_prof), "#CBD5E1")
    fig_prof.add_shape(
        type="rect", x0=_x0_prof, x1=_x1_prof, y0=0, y1=1, yref="paper",
        fillcolor=_fill,
        line=dict(color=_border, width=1),
        layer="below"
    )
    fig_prof.add_annotation(
        x=(_x0_prof + _x1_prof) / 2, y=0.96, yref="paper",
        text=_regime_prof[:18], showarrow=False,
        font=dict(size=9, color="#475569"),
    )
    _x0_prof = _x1_prof

fig_prof.add_trace(go.Scatter(
    x=pressure_profile_x,
    y=pressure_profile_y,
    mode="lines+markers",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=7, color="#2563EB"),
    hovertemplate="Distance: %{x:.2f} m<br>Pressure: %{y:.4f} bara<extra></extra>",
))
fig_prof.update_layout(
    xaxis_title="Pipeline Distance (m)",
    yaxis_title="Pressure (bara)",
    template="plotly_white",
    height=370,
    margin=dict(l=60, r=20, t=30, b=50),
    hovermode="x unified",
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0"),
    font=dict(size=12, color="#374151"),
)

# ── Display in tabs ───────────────────────────────────────────────────────────
st.divider()
tab_sch, tab_prof_tab = st.tabs(["Pipeline Schematic", "Pressure Profile"])

with tab_sch:
    st.plotly_chart(fig_sch, use_container_width=True)
    st.caption(
        "Line width ∝ pipe diameter (DN)  ·  "
        "Colour = flow regime  ·  "
        "Arrow = flow direction  ·  "
        "Axes: projected horizontal distance and elevation in metres"
    )

with tab_prof_tab:
    st.plotly_chart(fig_prof, use_container_width=True)

# ============================================================================
# REPORT EXPORT
# ============================================================================
st.divider()
st.subheader("Export Report")

# Detect input changes so a stale report is not silently reused
_rpt_hash = hashlib.md5(
    json.dumps({
        "P": P_bara, "T": T_C, "H2": m_H2, "O2": m_O2, "lye": q_lye,
        "segs": [
            (s["type"], s["dn"], s["pn"], s["length"], s["fittings"], s["fitting_count"],
             s.get("lined", False), s.get("liner_material", "FEP"), s.get("liner_thickness_mm", 1.0))
            for s in st.session_state.segments
        ],
    }, sort_keys=True).encode()
).hexdigest()

if st.session_state.get("_rpt_hash") != _rpt_hash:
    st.session_state["_rpt_hash"]  = None
    st.session_state["_rpt_bytes"] = None

_rc1, _rc2 = st.columns([1, 2])

with _rc1:
    if st.button("Generate Report", type="primary", use_container_width=True):
        with st.spinner("Rendering charts and building document…"):
            _buf = report_generator.generate_report(
                P_bara=P_bara, T_C=T_C, m_H2=m_H2, m_O2=m_O2, q_lye=q_lye,
                props=props,
                grid_records=grid_records,
                segments=st.session_state.segments,
                total_dp_kpa=total_dp_kpa,
                outlet_pressure_bara=outlet_pressure_bara,
                pipe_length_m=pipe_length_m,
                cumulative_distance=cumulative_distance,
                fig_sch=fig_sch,
                fig_prof=fig_prof,
            )
            st.session_state["_rpt_bytes"] = _buf.getvalue()
            st.session_state["_rpt_hash"]  = _rpt_hash

with _rc2:
    if st.session_state.get("_rpt_bytes"):
        if st.session_state.get("_rpt_hash") == _rpt_hash:
            st.download_button(
                label="Download  (.docx)",
                data=st.session_state["_rpt_bytes"],
                file_name="multiphase_hydraulic_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.info("Inputs changed — click Generate Report to refresh.")

# ============================================================================
# VALIDATION
# ============================================================================
st.divider()
st.subheader("Validation & Benchmarking")

tab_val, tab_calc = st.tabs(["Reference Cases", "Current Calculation"])

with tab_val:
    st.markdown("### Compare Against Reference Cases")

    _cases = val_cases.list_validation_cases()
    case_options = [item[0] for item in _cases]

    selected_case = st.selectbox(
        "Select Reference Case",
        options=case_options,
        format_func=lambda x: val_cases.get_validation_case(x)["name"]
    )

    if selected_case:
        case = val_cases.get_validation_case(selected_case)
        st.info(val_cases.get_case_info(selected_case))

        if st.button("Run Validation Case"):
            vi = case["inputs"]
            v_props = engine.calculate_two_phase_properties(
                vi["P_bara"], vi["T_C"], vi["m_H2_kgh"], vi["m_O2_kgh"], vi["q_lye_m3h"]
            )
            v_D = engine.PIPE_DATABASE[vi["pipe_dn"]][vi["pipe_pn"]]
            v_r = engine.MATERIAL_ROUGHNESS["SS316L"]  # reference cases are all SS316L
            v_total_dp = 0.0

            for seg in vi["segments"]:
                angle = {"Horizontal": 0.0,
                         "Vertical Upflow": np.pi / 2.0,
                         "Vertical Downflow": -np.pi / 2.0}[seg["type"]]
                le_fit = 0.0
                if seg["fittings"] in engine.FITTING_Le_over_D:
                    le_fit = engine.FITTING_Le_over_D[seg["fittings"]] * v_D * seg["fitting_count"]
                dP_Pa, *_ = engine.calculate_segment_pressure_drop(
                    v_props, v_D, v_r, seg["length"] + le_fit, angle
                )
                v_total_dp += dP_Pa

            calc_kpa     = v_total_dp / 1000.0
            expected_kpa = case["expected_total_dp_kpa"]
            tol_pct      = case.get("tolerance_pct", 5.0)
            err_pct      = abs(calc_kpa - expected_kpa) / expected_kpa * 100.0
            passed       = err_pct <= tol_pct

            r1, r2, r3 = st.columns(3)
            r1.metric("Calculated ΔP", f"{calc_kpa:.3f} kPa")
            r2.metric("Expected ΔP",   f"{expected_kpa:.3f} kPa")
            r3.metric("Deviation",     f"{err_pct:.2f} %",
                      delta=f"Pass (≤{tol_pct:.0f}%)" if passed else f"Fail (>{tol_pct:.0f}%)",
                      delta_color="normal" if passed else "inverse")

            if passed:
                st.success(f"Regression passed — deviation {err_pct:.2f}% within ±{tol_pct:.0f}%")
            else:
                st.warning(f"Regression failed — deviation {err_pct:.2f}% exceeds ±{tol_pct:.0f}%")

with tab_calc:
    st.markdown("### Current Calculation Summary")

    d1, d2, d3 = st.columns(3)
    d1.metric("System Pressure", f"{P_bara:.2f} bara")
    d2.metric("Temperature",     f"{T_C:.1f} °C")
    d3.metric("Total ΔP",        f"{total_dp_kpa:.3f} kPa")

    d4, d5, d6 = st.columns(3)
    d4.metric("Mass Quality x",   f"{props['x_gas']*100:.3f} %")
    d5.metric("Void Fraction α",  f"{props['alpha']*100:.1f} %")
    d6.metric("Outlet Pressure",  f"{outlet_pressure_bara:.4f} bara")

    st.markdown("**Segment detail:**")
    st.dataframe(
        pd.DataFrame(grid_records),
        column_config={
            "ID (mm)":        st.column_config.NumberColumn(format="%.1f"),
            "P_in (bara)":    st.column_config.NumberColumn(format="%.4f"),
            "ρ_g (kg/m³)":   st.column_config.NumberColumn(format="%.4f"),
            "L (m)":          st.column_config.NumberColumn(format="%.1f"),
            "L_eff (m)":      st.column_config.NumberColumn(format="%.1f"),
            "V_sg (m/s)":     st.column_config.NumberColumn(format="%.3f"),
            "V_sl (m/s)":     st.column_config.NumberColumn(format="%.3f"),
            "V_m (m/s)":      st.column_config.NumberColumn(format="%.3f"),
            "V_e (m/s)":      st.column_config.NumberColumn(format="%.2f"),
            "V_m/V_e":        st.column_config.NumberColumn(format="%.3f"),
            "dP/dz (Pa/m)":   st.column_config.NumberColumn(format="%.2f"),
            "ΔP (kPa)":       st.column_config.NumberColumn(format="%.3f"),
            "P_out (bara)":   st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True
    )
