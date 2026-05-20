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
    "Two-phase pressure drop · Beggs & Brill (1973) · "
    "Generic gas + liquid · Steady-state"
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
        **Gas species:** H₂, O₂, N₂, Air, or Custom
        **Liquid:** KOH 30 wt%, KOH 15 wt%, Water, or Custom
        **Minor losses:** Equivalent length method (Le/D, Crane TP-410)
        """)

    with st.expander("Model Assumptions", expanded=False):
        st.markdown("""
        1. Ideal gas behaviour for gas mixture
        2. Continuous liquid phase; no flooding or flow inversion
        3. Bore = f(DN, PN) only — ANSI B36.10/19 schedule, material-independent
        4. Lined segments: effective ID = metal bore − 2 × liner thickness
        5. Roughness set per segment by material or liner
        6. Pressure marching: gas density and void fraction re-evaluated at each segment inlet
        7. Erosion check: API RP 14E, V_e = 122/√ρ_mix (m/s), C=100 continuous service
        8. Validated temperature range: 5–95 °C
        9. Validated pressure range: 1–100 bara
        10. Steady-state only — no transient effects
        11. Void fraction: homogeneous model α = (x/ρg) / (x/ρg + (1−x)/ρl)
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

    with st.expander("Liquid Property Correlations", expanded=False):
        st.markdown("""
        **KOH 30 wt%**
        ρ(T) = 1295 − 0.3375·(T − 20) kg/m³
        μ(T) = μ_ref · exp(1200·(1/T − 1/T_ref)) Pa·s
        σ(T) = 0.074 − 0.001125·(T − 20) N/m

        **KOH 15 wt%**
        ρ(T) = 1139 − 0.50·(T − 20) kg/m³
        μ(T) = μ_ref · exp(1540·(1/T − 1/T_ref)) Pa·s

        **Water** — CoolProp (IAPWS-IF97)

        Valid range: 0–100 °C
        """)

    with st.expander("References", expanded=False):
        st.markdown("""
        - Beggs & Brill (1973) — SPE-4007-PA
        - CoolProp — open-source thermodynamic library
        - fluids — Python fluid dynamics library
        - Crane TP-410 (2013)
        - API RP 14E (2007)
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

# Initialise gas/liquid selection state
if "gas_species_widget" not in st.session_state:
    st.session_state["gas_species_widget"] = ["H₂", "O₂"]
if "liquid_type_widget" not in st.session_state:
    st.session_state["liquid_type_widget"] = "KOH 30 wt%"

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
        "Custom": {
            "P": 10.0, "T": 50.0,
            "gas_flows": {"H₂": 8.0, "O₂": 2.0},
            "liquid_type": "KOH 30 wt%", "lye": 1.5,
        },
        "H₂ side — 80 °C, 16 bara, KOH 30%": {
            "P": 16.0, "T": 80.0,
            "gas_flows": {"H₂": 100.0, "O₂": 0.1},
            "liquid_type": "KOH 30 wt%", "lye": 50.0,
        },
        "O₂ side — 80 °C, 16 bara, KOH 30%": {
            "P": 16.0, "T": 80.0,
            "gas_flows": {"H₂": 0.1, "O₂": 50.0},
            "liquid_type": "KOH 30 wt%", "lye": 50.0,
        },
        "N₂ purge — 25 °C, 5 bara, Water": {
            "P": 5.0, "T": 25.0,
            "gas_flows": {"N₂": 20.0},
            "liquid_type": "Water", "lye": 2.0,
        },
    }

    with st.container(border=True):
        st.subheader("Quick Presets")
        selected_preset = st.selectbox("Load preset conditions", list(PRESETS.keys()),
                                       key="preset_sel")
        preset_vals = PRESETS[selected_preset]

        # When preset changes, sync session state for species / liquid / flows
        if st.session_state.get("_last_preset") != selected_preset:
            st.session_state["_last_preset"] = selected_preset
            st.session_state["gas_species_widget"] = list(preset_vals["gas_flows"].keys())
            st.session_state["liquid_type_widget"]  = preset_vals["liquid_type"]
            for _sp, _fl in preset_vals["gas_flows"].items():
                st.session_state[f"gflow_{_sp}"] = float(_fl)
            st.session_state["q_lye_widget"] = float(preset_vals["lye"])

    # ── Process Boundaries ────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Process Boundaries")
        p1, p2 = st.columns(2)
        P_bara = p1.number_input("Inlet Pressure (bara)", min_value=1.0, max_value=100.0,
                                  value=float(preset_vals["P"]), step=1.0)
        T_C    = p2.number_input("Temperature (°C)",      min_value=5.0,  max_value=95.0,
                                  value=float(preset_vals["T"]), step=5.0)

    # ── Gas Phase ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Gas Phase")
        _all_species = list(engine.GAS_SPECIES.keys())

        selected_species = st.multiselect(
            "Gas species (select one or more)",
            _all_species,
            key="gas_species_widget",
        )
        if not selected_species:
            st.warning("Select at least one gas species.")
            selected_species = ["H₂"]

        # Per-species mass flow inputs
        gas_flows_kgh = {}
        _sp_count = len(selected_species)
        _ncols = min(_sp_count, 3)
        _flow_cols = st.columns(_ncols)
        for _ci, _sp in enumerate(selected_species):
            _default_flow = float(preset_vals["gas_flows"].get(_sp, 1.0))
            _col = _flow_cols[_ci % _ncols]
            gas_flows_kgh[_sp] = _col.number_input(
                f"{_sp}  (kg/h)",
                min_value=0.0,
                value=_default_flow,
                step=1.0,
                key=f"gflow_{_sp}",
            )

        # Custom gas properties shown only when "Custom" is selected
        custom_gas = None
        if "Custom" in selected_species:
            st.markdown("*Custom gas — physical properties*")
            _cg1, _cg2 = st.columns(2)
            _cg_mw  = _cg1.number_input("Molecular Weight (g/mol)",
                                         min_value=1.0, value=28.0, step=1.0)
            _cg_mu  = _cg2.number_input("Viscosity (µPa·s)",
                                         min_value=1.0, value=18.5, step=0.5)
            custom_gas = {"MW_gmol": _cg_mw, "mu_upas": _cg_mu}

    # ── Liquid Phase ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Liquid Phase")
        _liq_idx = engine.LIQUID_PHASES.index(st.session_state["liquid_type_widget"]) \
                   if st.session_state["liquid_type_widget"] in engine.LIQUID_PHASES else 0
        liquid_type = st.selectbox(
            "Liquid type",
            engine.LIQUID_PHASES,
            index=_liq_idx,
            key="liquid_type_widget",
        )
        q_lye = st.number_input(
            "Volume flow (m³/h)",
            min_value=0.0,
            value=float(preset_vals["lye"]),
            step=1.0,
            key="q_lye_widget",
        )

        # Custom liquid properties shown only for "Custom" liquid type
        custom_liquid = None
        if liquid_type == "Custom":
            st.markdown("*Custom liquid — physical properties*")
            _cl1, _cl2, _cl3 = st.columns(3)
            _cl_rho   = _cl1.number_input("Density (kg/m³)",    min_value=100.0, value=1000.0, step=10.0)
            _cl_mu    = _cl2.number_input("Viscosity (mPa·s)",  min_value=0.01,  value=1.0,    step=0.1)
            _cl_sigma = _cl3.number_input("Surface tension (mN/m)", min_value=1.0, value=72.0, step=1.0)
            custom_liquid = {"rho_kgm3": _cl_rho, "mu_mpas": _cl_mu, "sigma_mnm": _cl_sigma}

    # ── Pipe Geometry ─────────────────────────────────────────────────────────
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

    is_valid, warn_list = engine.validate_input_bounds(
        P_bara, T_C, gas_flows_kgh, liquid_type, q_lye
    )
    for w in warn_list:
        st.warning(w)

    props = engine.calculate_two_phase_properties(
        P_bara, T_C, gas_flows_kgh, liquid_type, q_lye,
        custom_gas=custom_gas, custom_liquid=custom_liquid,
    )

    # ── Inlet Physical Properties ─────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Inlet Physical Properties")

        col_gas, col_liq = st.columns(2)

        # Gas phase — composition table + key properties
        with col_gas:
            st.markdown("**Gas phase**")
            _comp     = props["composition"]
            _n_total  = sum(v["mol_h"] for v in _comp.values())

            _comp_rows = []
            for _sp, _data in _comp.items():
                _comp_rows.append({
                    "Component": _sp,
                    "kg/h":  round(_data["kg_h"],    3),
                    "mol/h": round(_data["mol_h"],    1),
                    "mol %": f"{_data['mol_frac']*100:.1f}",
                })
            _comp_rows.append({
                "Component": "Total",
                "kg/h":  round(props["m_gas_total_kgh"], 3),
                "mol/h": round(_n_total, 1),
                "mol %": "100.0",
            })
            st.dataframe(
                pd.DataFrame(_comp_rows),
                column_config={
                    "kg/h":  st.column_config.NumberColumn(format="%.3f"),
                    "mol/h": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True, use_container_width=True,
            )
            g1, g2, g3 = st.columns(3)
            g1.metric("ρ_gas",  f"{props['rho_g']:.3f} kg/m³")
            g2.metric("MW_mix", f"{props['MW_mix_gmol']:.2f} g/mol")
            if props.get("P_sat_H2O_pa", 0) > 0:
                g3.metric("P_sat H₂O", f"{props['P_sat_H2O_pa']/1e5:.3f} bara")

        # Liquid phase + two-phase summary
        with col_liq:
            st.markdown(f"**Liquid phase ({liquid_type})**")
            l1, l2 = st.columns(2)
            l1.metric("ṁ_liquid",  f"{props['m_lye_kgh']:.1f} kg/h")
            l2.metric("ρ_liquid",  f"{props['rho_l']:.1f} kg/m³")
            l3, l4 = st.columns(2)
            l3.metric("μ_liquid",  f"{props['mu_l']*1e3:.3f} mPa·s")
            l4.metric("σ",         f"{props['sigma']*1e3:.2f} mN/m")

            st.markdown("**Two-phase mixture**")
            _m_total_kgh = props["m_gas_total_kgh"] + props["m_liquid_total_kgh"]
            m1, m2, m3 = st.columns(3)
            m1.metric("ṁ_total",        f"{_m_total_kgh:.2f} kg/h")
            m2.metric("Mass quality x",  f"{props['x_gas']*100:.3f} %")
            m3.metric("Void fraction α", f"{props['alpha']*100:.2f} %")

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

        # Re-evaluate at local segment inlet pressure (pressure marching)
        props_seg = engine.calculate_two_phase_properties(
            current_P / 1e5, T_C, gas_flows_kgh, liquid_type, q_lye,
            custom_gas=custom_gas, custom_liquid=custom_liquid,
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
        "P": P_bara, "T": T_C,
        "gas_flows": {k: float(v) for k, v in gas_flows_kgh.items()},
        "liquid_type": liquid_type,
        "lye": q_lye,
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
                P_bara=P_bara, T_C=T_C,
                gas_flows_kgh=gas_flows_kgh, liquid_type=liquid_type, q_lye=q_lye,
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
# EXCEL EXPORT
# ============================================================================
st.divider()
st.subheader("Export Data  (.xlsx)")

def _build_xlsx():
    from io import BytesIO
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    BLUE   = "2563EB"
    LGRAY  = "F1F5F9"
    STRIPE = "F8FAFC"

    _hdr_font  = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
    _hdr_fill  = PatternFill("solid", fgColor=BLUE)
    _sec_font  = Font(bold=True, size=9, name="Calibri")
    _sec_fill  = PatternFill("solid", fgColor=LGRAY)
    _dat_font  = Font(size=9, name="Calibri")
    _alt_fill  = PatternFill("solid", fgColor=STRIPE)
    _thin      = Side(style="thin", color="CBD5E1")
    _border    = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)

    def _hdr(ws, row, text, ncols):
        c = ws.cell(row=row, column=1, value=text)
        c.font = _hdr_font; c.fill = _hdr_fill
        c.alignment = Alignment(vertical="center")
        c.border = _border
        for col in range(2, ncols + 2):
            cc = ws.cell(row=row, column=col)
            cc.fill = _hdr_fill; cc.border = _border
        return row + 1

    def _sec(ws, row, text, ncols):
        c = ws.cell(row=row, column=1, value=text)
        c.font = _sec_font; c.fill = _sec_fill; c.border = _border
        for col in range(2, ncols + 2):
            cc = ws.cell(row=row, column=col)
            cc.fill = _sec_fill; cc.border = _border
        return row + 1

    def _dat(ws, row, label, value, alt=False):
        fill = _alt_fill if alt else None
        lc = ws.cell(row=row, column=1, value=label)
        lc.font = _dat_font; lc.border = _border
        if fill: lc.fill = fill
        vc = ws.cell(row=row, column=2, value=value)
        vc.font = _dat_font; vc.border = _border
        if fill: vc.fill = fill
        return row + 1

    wb = openpyxl.Workbook()

    # ── Sheet 1: System ───────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "System"
    ws.freeze_panes = "A2"

    r = _hdr(ws, 1, "System Summary", 1)

    # Boundary conditions
    alt = False
    r = _sec(ws, r, "Boundary Conditions", 1)
    for lbl, val in [
        ("Inlet Pressure (bara)",             round(P_bara, 4)),
        ("Temperature (°C)",                  round(T_C, 2)),
        ("Outlet Pressure (bara)",            round(outlet_pressure_bara, 4)),
        ("Total ΔP (kPa)",                    round(total_dp_kpa, 4)),
        ("Total ΔP (bar)",                    round(total_dp_kpa / 100, 6)),
        ("Pipe Length (m)",                   round(pipe_length_m, 3)),
        ("Effective Length incl. fittings (m)", round(cumulative_distance, 3)),
    ]:
        r = _dat(ws, r, lbl, val, alt); alt = not alt

    # Gas phase
    r += 1; alt = False
    r = _sec(ws, r, "Gas Phase  —  inlet conditions", 1)
    for sp, flow in gas_flows_kgh.items():
        r = _dat(ws, r, f"{sp} mass flow (kg/h)", round(float(flow), 4), alt); alt = not alt
    for lbl, val in [
        ("Total gas mass flow (kg/h)",        round(props["m_gas_total_kgh"], 4)),
        ("Gas density ρ_g (kg/m³)",           round(props["rho_g"], 4)),
        ("Gas mixture MW (g/mol)",            round(props["MW_mix_gmol"], 3)),
        ("Gas viscosity μ_g (µPa·s)",         round(props["mu_g"] * 1e6, 3)),
        ("Mass quality x (%)",                round(props["x_gas"] * 100, 5)),
        ("Void fraction α (%)",               round(props["alpha"] * 100, 4)),
    ]:
        r = _dat(ws, r, lbl, val, alt); alt = not alt
    if props.get("P_sat_H2O_pa", 0) > 0:
        for lbl, val in [
            ("H₂O vapour flow (kg/h)",           round(props["m_vapor_h2o_kgh"], 5)),
            ("H₂O saturation pressure (bara)",   round(props["P_sat_H2O_pa"] / 1e5, 5)),
        ]:
            r = _dat(ws, r, lbl, val, alt); alt = not alt

    # Liquid phase
    r += 1; alt = False
    r = _sec(ws, r, f"Liquid Phase  —  {liquid_type}", 1)
    for lbl, val in [
        ("Volume flow (m³/h)",                round(q_lye, 4)),
        ("Mass flow (kg/h)",                  round(props["m_lye_kgh"], 3)),
        ("Density ρ_l (kg/m³)",               round(props["rho_l"], 3)),
        ("Viscosity μ_l (mPa·s)",             round(props["mu_l"] * 1e3, 4)),
        ("Surface tension σ (mN/m)",          round(props["sigma"] * 1e3, 4)),
    ]:
        r = _dat(ws, r, lbl, val, alt); alt = not alt

    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 20

    # ── Sheet 2: Segments ─────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Segments")
    ws2.freeze_panes = "B2"

    n = len(grid_records)

    # Header row
    c = ws2.cell(row=1, column=1, value="Parameter")
    c.font = _hdr_font; c.fill = _hdr_fill; c.border = _border
    for j, rec in enumerate(grid_records):
        seg = st.session_state.segments[j]
        label = f"Seg {rec['Seg']}  {rec['Pipe']}"
        if seg.get("lined"):
            label += f" +{seg['liner_material']}"
        cc = ws2.cell(row=1, column=j + 2, value=label)
        cc.font = _hdr_font; cc.fill = _hdr_fill; cc.border = _border

    # Parameter rows
    _ROWS = [
        ("Orientation",                        "Type"),
        ("Pipe class",                         "Pipe"),
        ("Inner diameter (mm)",                "ID (mm)"),
        ("Material",                           "Material"),
        ("Physical length (m)",                "L (m)"),
        ("Effective length incl. fittings (m)","L_eff (m)"),
        ("P_in  Inlet pressure (bara)",        "P_in (bara)"),
        ("P_out  Outlet pressure (bara)",      "P_out (bara)"),
        ("ΔP  Pressure drop (kPa)",            "ΔP (kPa)"),
        ("dP/dz  Pressure gradient (Pa/m)",    "dP/dz (Pa/m)"),
        ("Gas density ρ_g (kg/m³)",            "ρ_g (kg/m³)"),
        ("Flow regime",                        "Regime"),
        ("V_sg  Superficial gas velocity (m/s)",     "V_sg (m/s)"),
        ("V_sl  Superficial liquid velocity (m/s)",  "V_sl (m/s)"),
        ("V_m  Mixture velocity (m/s)",              "V_m (m/s)"),
        ("V_e  Erosion limit API RP 14E (m/s)",      "V_e (m/s)"),
        ("V_m/V_e  Erosion ratio (–)",               "V_m/V_e"),
    ]

    for i, (label, key) in enumerate(_ROWS):
        row = i + 2
        alt = (i % 2 == 1)
        lc = ws2.cell(row=row, column=1, value=label)
        lc.font = _dat_font; lc.border = _border
        if alt: lc.fill = _alt_fill
        for j, rec in enumerate(grid_records):
            vc = ws2.cell(row=row, column=j + 2, value=rec.get(key, ""))
            vc.font = _dat_font; vc.border = _border
            if alt: vc.fill = _alt_fill

    ws2.column_dimensions["A"].width = 46
    for j in range(n):
        ws2.column_dimensions[get_column_letter(j + 2)].width = 18

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

_xl1, _xl2 = st.columns([1, 2])
with _xl1:
    if st.button("Generate Excel", use_container_width=True):
        try:
            _xl_bytes = _build_xlsx().getvalue()
            st.session_state["_xl_bytes"] = _xl_bytes
        except Exception as _xe:
            st.error(f"Excel export failed: {_xe}")

with _xl2:
    if st.session_state.get("_xl_bytes"):
        st.download_button(
            label="Download  (.xlsx)",
            data=st.session_state["_xl_bytes"],
            file_name="multiphase_hydraulic_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

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
            # Build generic gas_flows_kgh dict from legacy case keys
            _v_gas = {}
            if vi.get("m_H2_kgh", 0) > 0:
                _v_gas["H₂"] = vi["m_H2_kgh"]
            if vi.get("m_O2_kgh", 0) > 0:
                _v_gas["O₂"] = vi["m_O2_kgh"]
            _v_liquid = vi.get("liquid_type", "KOH 30 wt%")

            v_props = engine.calculate_two_phase_properties(
                vi["P_bara"], vi["T_C"], _v_gas, _v_liquid, vi["q_lye_m3h"]
            )
            v_D = engine.PIPE_DATABASE[vi["pipe_dn"]][vi["pipe_pn"]]
            v_r = engine.MATERIAL_ROUGHNESS["SS316L"]
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
