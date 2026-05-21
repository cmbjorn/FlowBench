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
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
hr { margin: 0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Multiphase Pipe Hydraulic Engine")
st.caption("Two-phase pressure drop · Six correlations · Generic gas + liquid · Steady-state")

with st.expander("About this calculator", expanded=False):
    st.markdown("""
    Calculates steady-state pressure drops for piping networks carrying a **two-phase mixture of gas and liquid** — a gas (H₂, O₂, N₂, CO₂, CH₄, Ar, He, Air, or custom) entrained in a liquid (KOH 30 wt%, KOH 15 wt%, Water, Methanol, Ethanol, or custom).

    **Method** — Beggs & Brill (1973) two-phase correlation as the default, with Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, and Kim-Mudawar available for cross-checking. Gas density is updated at every segment inlet (pressure marching). Each segment's pressure drop is decomposed into frictional, gravitational, and accelerational components. Flow regime is classified automatically — Taitel-Dukler + Mandhane for horizontal segments, Wallis/Taitel (1980) for vertical. Erosion velocity checked against API RP 14E, C = 100.

    **Workflow** — Set inlet pressure, temperature, and flow rates → build the pipe geometry segment by segment (DN, PN, material, length, fittings, optional fluoropolymer liner) → read the segment analysis table and pressure profile → compare two alternative configurations side by side (Case A vs B) → export a Word or Excel report.

    **Accuracy note** — Beggs & Brill was developed for oil/gas systems. Uncertainty for H₂/KOH is ±20–30 %. Use results for design guidance and relative comparison between alternatives, not as a substitute for detailed simulation or regulatory compliance.
    """)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Documentation")
    with st.expander("Capabilities", expanded=False):
        st.markdown("""
        **Cases** — Run Case A and Case B independently, then compare side by side.
        Useful for: alternative pipe routings, diameter studies, full-flow vs. turndown.

        **Gas species** — H₂, O₂, N₂, CO₂, CH₄, Ar, He, Air, or Custom (user-defined MW and viscosity).
        Water vapour added automatically for aqueous liquids via Dalton's Law.

        **Liquid types** — KOH 30 wt%, KOH 15 wt%, Water, Methanol, Ethanol, or Custom.

        **ΔP correlations** — Beggs-Brill (default), Friedel, Lockhart-Martinelli,
        Müller-Steinhagen & Heck, Chisholm, Kim-Mudawar.

        **Minor losses** — Equivalent length (Le/D, Crane TP-410). 17 fitting types.

        **Pipe geometry** — DN40–DN250, PN20/PN25/PN40, 5 metallic materials,
        optional fluoropolymer liner (PTFE, FEP, PFA, PVDF).

        **Erosion check** — API RP 14E, V_e = C/√ρ_mix, C = 100 continuous service.

        **Exports** — Word report (.docx) with embedded charts; Excel (.xlsx).
        """)
    with st.expander("Model Assumptions", expanded=False):
        st.markdown("""
        1. Ideal gas behaviour for gas mixture
        2. Continuous liquid phase; no flooding or flow inversion
        3. Bore = f(DN, PN) only — ANSI B36.10/19, material-independent
        4. Lined segments: effective ID = metal bore − 2 × liner thickness
        5. Pressure marching: gas density re-evaluated at each segment inlet
        6. Erosion check: API RP 14E, V_e = 122/√ρ_mix (m/s), C=100
        7. Steady-state only — no transient effects
        8. Void fraction: homogeneous α = (x/ρg)/(x/ρg+(1−x)/ρl), or Rouhani-1 slip model
        9. Flow regime: Taitel-Dukler + Mandhane (horizontal), Wallis/Taitel (vertical)
        """)
    with st.expander("Flow Regimes", expanded=False):
        st.markdown("""
        **Horizontal** — Stratified / Intermittent (Slug) / Annular-Dispersed
        **Vertical Up** — Bubble/Slug / Churn/Annular
        **Vertical Down** — Falling Film / Annular
        """)
    with st.expander("Liquid Correlations", expanded=False):
        st.markdown("""
        **KOH 30 wt%** ρ = 1295−0.3375(T−20) · μ Arrhenius E/R=1200 K
        **KOH 15 wt%** ρ = 1139−0.50(T−20) · μ Arrhenius E/R=1540 K
        **Water · Methanol · Ethanol** — CoolProp (IAPWS-IF97 / DIPPR)
        """)
    with st.expander("References", expanded=False):
        st.markdown("""
        - Beggs & Brill (1973) — SPE-4007-PA
        - CoolProp — open-source thermodynamic library
        - fluids — Python fluid dynamics library
        - Crane TP-410 (2013) · API RP 14E (2007)
        """)
    with st.expander("Validation", expanded=False):
        st.markdown("Run a built-in reference case and compare the calculated ΔP against the expected value.")
        _sb_cases     = val_cases.list_validation_cases()
        _sb_case_opts = [item[0] for item in _sb_cases]
        _sb_sel       = st.selectbox("Reference case", options=_sb_case_opts,
                                     format_func=lambda x: val_cases.get_validation_case(x)["name"],
                                     key="sb_val_sel")
        if _sb_sel:
            _sb_case = val_cases.get_validation_case(_sb_sel)
            st.info(val_cases.get_case_info(_sb_sel))
            if st.button("Run", key="sb_run_val", use_container_width=True):
                _sb_vi = _sb_case["inputs"]
                _sb_vg = {}
                if _sb_vi.get("m_H2_kgh", 0) > 0: _sb_vg["H₂"] = _sb_vi["m_H2_kgh"]
                if _sb_vi.get("m_O2_kgh", 0) > 0: _sb_vg["O₂"] = _sb_vi["m_O2_kgh"]
                _sb_vl = _sb_vi.get("liquid_type", "KOH 30 wt%")
                _sb_vp = engine.calculate_two_phase_properties(
                    _sb_vi["P_bara"], _sb_vi["T_C"], _sb_vg, _sb_vl, _sb_vi["q_lye_m3h"])
                _sb_D  = engine.PIPE_DATABASE[_sb_vi["pipe_dn"]][_sb_vi["pipe_pn"]]
                _sb_r  = engine.MATERIAL_ROUGHNESS["SS316L"]
                _sb_dp = 0.0
                for _sb_seg in _sb_vi["segments"]:
                    _sb_ang = {"Horizontal": 0.0, "Vertical Upflow": np.pi/2,
                               "Vertical Downflow": -np.pi/2}[_sb_seg["type"]]
                    _sb_le  = 0.0
                    if _sb_seg["fittings"] in engine.FITTING_Le_over_D:
                        _sb_le = engine.FITTING_Le_over_D[_sb_seg["fittings"]]*_sb_D*_sb_seg["fitting_count"]
                    _sb_res = engine.calculate_segment_pressure_drop(
                        _sb_vp, _sb_D, _sb_r, _sb_seg["length"]+_sb_le, _sb_ang)
                    _sb_dp += _sb_res["dP_Pa"]
                _sb_calc = _sb_dp / 1000.0
                _sb_exp  = _sb_case["expected_total_dp_kpa"]
                _sb_tol  = _sb_case.get("tolerance_pct", 5.0)
                _sb_err  = abs(_sb_calc - _sb_exp) / _sb_exp * 100.0
                _sb_pass = _sb_err <= _sb_tol
                r1, r2, r3 = st.columns(3)
                r1.metric("Calculated", f"{_sb_calc:.3f} kPa")
                r2.metric("Expected",   f"{_sb_exp:.3f} kPa")
                r3.metric("Deviation",  f"{_sb_err:.2f} %",
                          delta=f"Pass (≤{_sb_tol:.0f}%)" if _sb_pass else f"Fail (>{_sb_tol:.0f}%)",
                          delta_color="normal" if _sb_pass else "inverse")
                if _sb_pass:
                    st.success(f"Passed — {_sb_err:.2f}% within ±{_sb_tol:.0f}%")
                else:
                    st.warning(f"Failed — {_sb_err:.2f}% exceeds ±{_sb_tol:.0f}%")

# ============================================================================
# PRESETS  (shared across both cases)
# ============================================================================
PRESETS = {
    "Custom": {
        "P": 2.0, "T": 25.0,
        "gas_flows": {"Air": 10.0},
        "liquid_type": "Water", "lye": 1.0,
    },
    "Air + Water — 25 °C, 2 bara": {
        "P": 2.0, "T": 25.0,
        "gas_flows": {"Air": 10.0},
        "liquid_type": "Water", "lye": 1.0,
    },
    "N₂ + Water — 60 °C, 8 bara": {
        "P": 8.0, "T": 60.0,
        "gas_flows": {"N₂": 20.0},
        "liquid_type": "Water", "lye": 2.0,
    },
    "CO₂ + Water — 25 °C, 50 bara": {
        "P": 50.0, "T": 25.0,
        "gas_flows": {"CO₂": 50.0},
        "liquid_type": "Water", "lye": 5.0,
    },
    "CH₄ + Water — 40 °C, 30 bara": {
        "P": 30.0, "T": 40.0,
        "gas_flows": {"CH₄": 30.0},
        "liquid_type": "Water", "lye": 3.0,
    },
    "H₂ side — 80 °C, 16 bara, KOH 30%": {
        "P": 16.0, "T": 80.0,
        "gas_flows": {"H₂": 26.0},
        "liquid_type": "KOH 30 wt%", "lye": 13.75,
    },
    "O₂ side — 80 °C, 16 bara, KOH 30%": {
        "P": 16.0, "T": 80.0,
        "gas_flows": {"O₂": 206.0},
        "liquid_type": "KOH 30 wt%", "lye": 13.75,
    },
    "N₂ purge — 25 °C, 5 bara, Water": {
        "P": 5.0, "T": 25.0,
        "gas_flows": {"N₂": 20.0},
        "liquid_type": "Water", "lye": 2.0,
    },
    "H₂ + H₂O (sat.) — 80 °C, 16 bara": {
        "P": 16.0, "T": 80.0,
        "gas_flows": {"H₂": 26.0},
        "liquid_type": "Water", "lye": 0.001,
    },
    "O₂ + H₂O (sat.) — 80 °C, 16 bara": {
        "P": 16.0, "T": 80.0,
        "gas_flows": {"O₂": 206.0},
        "liquid_type": "Water", "lye": 0.001,
    },
}

_DEFAULT_SEGMENTS = [
    {"type": "Horizontal",      "dn": "DN50", "pn": "PN40", "material": "SS316L",
     "length": 12.0, "fittings": "None", "fitting_count": 0,
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
    {"type": "Vertical Upflow", "dn": "DN50", "pn": "PN40", "material": "SS316L",
     "length":  4.5, "fittings": "None", "fitting_count": 0,
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
]

# Default geometry for Case C (header / manifold — larger-bore pipes)
_DEFAULT_HDR_SEGMENTS = [
    {"type": "Horizontal",        "dn": "DN100", "pn": "PN40", "material": "SS316L",
     "length": 10.0, "fittings": "None", "fitting_count": 0,
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
    {"type": "Horizontal",        "dn": "DN100", "pn": "PN40", "material": "SS316L",
     "length":  5.0, "fittings": "None", "fitting_count": 0,
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
    {"type": "Vertical Downflow", "dn": "DN150", "pn": "PN40", "material": "SS316L",
     "length":  4.0, "fittings": "None", "fitting_count": 0,
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
]

_VALID_MATS   = set(engine.MATERIAL_ROUGHNESS.keys())
_VALID_LINERS = set(engine.LINER_ROUGHNESS.keys())

# ============================================================================
# HEADER (CASE C) — default geometries
# ============================================================================
def _hdr_seg(**kw):
    return {
        "type":              kw.get("type", "Horizontal"),
        "dn":                kw.get("dn", "DN100"),
        "pn":                kw.get("pn", "PN40"),
        "material":          kw.get("material", "SS316L"),
        "length":            kw.get("length", 2.5),
        "fittings":          kw.get("fittings", "None"),
        "fitting_count":     kw.get("fitting_count", 0),
        "lined":             kw.get("lined", False),
        "liner_material":    kw.get("liner_material", "FEP"),
        "liner_thickness_mm":kw.get("liner_thickness_mm", 1.0),
        "has_tee":           kw.get("has_tee", True),
        "branch_gas_kgh":    kw.get("branch_gas_kgh", {}),
        "branch_q_lye":      kw.get("branch_q_lye", 0.0),
    }

_DEFAULT_HDR_LINEAR = [_hdr_seg() for _ in range(4)]
_DEFAULT_HDR_LEFT   = [_hdr_seg() for _ in range(3)]
_DEFAULT_HDR_RIGHT  = [_hdr_seg() for _ in range(3)]
_DEFAULT_HDR_EXIT   = [_hdr_seg(type="Vertical Downflow", dn="DN150", has_tee=False)]

# ============================================================================
# REGIME COLOUR HELPERS
# Keyword-based matching so any regime string from the engine is coloured
# correctly, including compound strings like "intermittent / slug".
# Order matters: more specific keywords are checked first.
# ============================================================================
_REGIME_LINE_KW = [          # (keyword, hex colour)  — schematic line colour
    ("churn",        "#DC2626"),   # red
    ("falling",      "#0891B2"),   # cyan  (before "annular" to catch falling film/annular)
    ("bubb",         "#7C3AED"),   # purple (catches both "bubble" and "bubbly")
    ("mist",         "#059669"),   # green
    ("annular",      "#059669"),   # green
    ("slug",         "#D97706"),   # amber
    ("intermittent", "#D97706"),   # amber
    ("elongated",    "#D97706"),   # amber
    ("wave",         "#3B82F6"),   # blue
    ("stratified",   "#3B82F6"),   # blue
    ("dispersed",    "#059669"),   # green
]
_REGIME_FILL_KW = [          # (keyword, rgba fill)   — pressure profile band
    ("churn",        "rgba(254,226,226,0.70)"),
    ("falling",      "rgba(207,250,254,0.70)"),
    ("bubb",         "rgba(237,233,254,0.70)"),
    ("mist",         "rgba(209,250,229,0.70)"),
    ("annular",      "rgba(209,250,229,0.70)"),
    ("slug",         "rgba(254,243,199,0.70)"),
    ("intermittent", "rgba(254,243,199,0.70)"),
    ("elongated",    "rgba(254,243,199,0.70)"),
    ("wave",         "rgba(219,234,254,0.70)"),
    ("stratified",   "rgba(219,234,254,0.70)"),
    ("dispersed",    "rgba(209,250,229,0.70)"),
]
_REGIME_BORDER_KW = [        # (keyword, border colour) — pressure profile band
    ("churn",        "#FCA5A5"),
    ("falling",      "#67E8F9"),
    ("bubb",         "#C4B5FD"),
    ("mist",         "#6EE7B7"),
    ("annular",      "#6EE7B7"),
    ("slug",         "#FCD34D"),
    ("intermittent", "#FCD34D"),
    ("elongated",    "#FCD34D"),
    ("wave",         "#93C5FD"),
    ("stratified",   "#93C5FD"),
    ("dispersed",    "#6EE7B7"),
]

def _regime_color(regime_str, kw_list, default):
    r = regime_str.lower()
    return next((col for kw, col in kw_list if kw in r), default)

# ============================================================================
# CASE RUNNER  — renders one full case and returns results for Compare tab
# ============================================================================
def run_case(cid: str, accent: str, default_segments=None) -> dict:
    """
    Render inputs + outputs for one case.

    cid              : "a", "b", or "c"
    accent           : hex colour used for the pressure profile trace
    default_segments : segment list used for first-run initialisation
    Returns dict of results consumed by the Compare tab and report generator.
    """
    k = lambda name: f"{cid}_{name}"   # namespaced session-state / widget key

    # ── Session state init ────────────────────────────────────────────────────
    if k("segments") not in st.session_state:
        import copy
        _init = default_segments if default_segments is not None else _DEFAULT_SEGMENTS
        st.session_state[k("segments")] = copy.deepcopy(_init)
    if k("gas_species_widget") not in st.session_state:
        st.session_state[k("gas_species_widget")] = ["H₂"]
    if k("liquid_type_widget") not in st.session_state:
        st.session_state[k("liquid_type_widget")] = "Water"

    # Migrate segments
    for _seg in st.session_state[k("segments")]:
        _seg.setdefault("dn", "DN50"); _seg.setdefault("pn", "PN40")
        _seg.setdefault("material", "SS316L"); _seg.setdefault("lined", False)
        _seg.setdefault("liner_material", "FEP"); _seg.setdefault("liner_thickness_mm", 1.0)
        if _seg["material"] not in _VALID_MATS:    _seg["material"] = "SS316L"
        if _seg["liner_material"] not in _VALID_LINERS: _seg["liner_material"] = "FEP"

    col_in, col_out = st.columns([1, 1.2])

    # ── INPUTS ────────────────────────────────────────────────────────────────
    with col_in:
        st.subheader("Inputs")

        # Presets
        with st.container(border=True):
            st.markdown("**Quick Presets**")
            selected_preset = st.selectbox("Load preset conditions",
                                           list(PRESETS.keys()), key=k("preset_sel"))
            preset_vals = PRESETS[selected_preset]

            if st.session_state.get(k("last_preset")) != selected_preset:
                st.session_state[k("last_preset")] = selected_preset
                st.session_state[k("P_bara")] = float(preset_vals["P"])
                st.session_state[k("T_C")]    = float(preset_vals["T"])
                st.session_state[k("gas_species_widget")] = list(preset_vals["gas_flows"].keys())
                st.session_state[k("liquid_type_widget")] = preset_vals["liquid_type"]
                for _sp, _fl in preset_vals["gas_flows"].items():
                    st.session_state[k(f"gflow_{_sp}")] = float(_fl)
                st.session_state[k("q_lye_widget")] = float(preset_vals["lye"])

        # Process Boundaries
        with st.container(border=True):
            st.markdown("**Process Boundaries**")
            p1, p2 = st.columns(2)
            P_bara = p1.number_input("Inlet Pressure (bara)", min_value=1.0, max_value=100.0,
                                     value=float(preset_vals["P"]), step=1.0, key=k("P_bara"))
            T_C    = p2.number_input("Temperature (°C)", min_value=5.0, max_value=95.0,
                                     value=float(preset_vals["T"]), step=5.0, key=k("T_C"))

        # Gas Phase
        with st.container(border=True):
            st.markdown("**Gas Phase**")
            _all_species = list(engine.GAS_SPECIES.keys())
            selected_species = st.multiselect(
                "Gas species", _all_species, key=k("gas_species_widget"))
            if not selected_species:
                st.warning("Select at least one gas species.")
                selected_species = ["H₂"]

            gas_flows_kgh = {}
            _ncols = min(len(selected_species), 3)
            _fcols = st.columns(_ncols)
            for _ci, _sp in enumerate(selected_species):
                _default = float(preset_vals["gas_flows"].get(_sp, 1.0))
                gas_flows_kgh[_sp] = _fcols[_ci % _ncols].number_input(
                    f"{_sp}  (kg/h)", min_value=0.0, value=_default, step=0.1,
                    key=k(f"gflow_{_sp}"))

            custom_gas = None
            if "Custom" in selected_species:
                st.markdown("*Custom gas properties*")
                _cg1, _cg2 = st.columns(2)
                _cg_mw = _cg1.number_input("MW (g/mol)", min_value=1.0, value=28.0,
                                            step=1.0, key=k("cg_mw"))
                _cg_mu = _cg2.number_input("μ (µPa·s)", min_value=1.0, value=18.5,
                                            step=0.5, key=k("cg_mu"))
                custom_gas = {"MW_gmol": _cg_mw, "mu_upas": _cg_mu}

        # Liquid Phase
        with st.container(border=True):
            st.markdown("**Liquid Phase**")
            _liq_idx = (engine.LIQUID_PHASES.index(st.session_state[k("liquid_type_widget")])
                        if st.session_state[k("liquid_type_widget")] in engine.LIQUID_PHASES else 0)
            liquid_type = st.selectbox("Liquid type", engine.LIQUID_PHASES,
                                       index=_liq_idx, key=k("liquid_type_widget"))
            q_lye = st.number_input("Volume flow (m³/h)", min_value=0.0,
                                    value=float(preset_vals["lye"]), step=0.25,
                                    key=k("q_lye_widget"))
            custom_liquid = None
            if liquid_type == "Custom":
                st.markdown("*Custom liquid properties*")
                _cl1, _cl2, _cl3 = st.columns(3)
                _cl_rho   = _cl1.number_input("ρ (kg/m³)", min_value=100.0, value=1000.0,
                                               step=10.0, key=k("cl_rho"))
                _cl_mu    = _cl2.number_input("μ (mPa·s)", min_value=0.01, value=1.0,
                                               step=0.1, key=k("cl_mu"))
                _cl_sigma = _cl3.number_input("σ (mN/m)", min_value=1.0, value=72.0,
                                               step=1.0, key=k("cl_sigma"))
                custom_liquid = {"rho_kgm3": _cl_rho, "mu_mpas": _cl_mu, "sigma_mnm": _cl_sigma}

        # Calculation Settings
        with st.container(border=True):
            st.markdown("**Calculation Settings**")
            _cs1, _cs2 = st.columns(2)
            correlation = _cs1.selectbox(
                "ΔP correlation", engine.TWO_PHASE_CORRELATIONS,
                key=k("correlation"),
                help="Two-phase frictional pressure drop correlation. "
                     "Beggs-Brill (default) also models inclined flow; "
                     "others use gravity added separately.")
            voidage_method = _cs2.selectbox(
                "Void fraction model", engine.VOIDAGE_METHODS,
                key=k("voidage_method"),
                help="Homogeneous: α from density ratio (fast, conservative). "
                     "Rouhani-1: slip-flow model (more accurate for stratified/annular).")

        # Pipe Geometry
        with st.container(border=True):
            st.markdown("**Pipe Geometry**")
            current_specs = []
            DN_OPTIONS    = list(engine.PIPE_DATABASE.keys())
            PN_OPTIONS    = ["PN20", "PN25", "PN40"]
            MAT_OPTIONS   = list(engine.MATERIAL_ROUGHNESS.keys())
            LINER_OPTIONS = list(engine.LINER_ROUGHNESS.keys())
            fit_options   = ["None"] + list(engine.FITTING_Le_over_D.keys())

            for i, seg in enumerate(st.session_state[k("segments")]):
                st.markdown(f"**Segment #{i+1}**")
                g1, g2, g3, g4 = st.columns([1.3, 0.8, 0.7, 0.7])
                t = g1.selectbox("Orientation",
                    ["Horizontal", "Vertical Upflow", "Vertical Downflow"],
                    key=k(f"t_{i}"),
                    index=["Horizontal","Vertical Upflow","Vertical Downflow"].index(seg["type"]))
                dn = g2.selectbox("DN", DN_OPTIONS, key=k(f"dn_{i}"),
                                  index=DN_OPTIONS.index(seg.get("dn","DN50")),
                                  help="Nominal pipe diameter (ANSI B36.10/19). "
                                       "Internal bore depends on DN + PN.")
                pn = g3.selectbox("PN", PN_OPTIONS, key=k(f"pn_{i}"),
                                  index=PN_OPTIONS.index(seg.get("pn","PN40")),
                                  help="Pressure rating class. Determines pipe schedule "
                                       "and wall thickness, setting the internal bore.")
                l  = g4.number_input("Length (m)", min_value=0.1,
                                     value=float(seg["length"]), step=1.0, key=k(f"l_{i}"))

                g5, g6, g7, g8 = st.columns([1.1, 2.0, 0.6, 0.65])
                _mat_def = seg.get("material","SS316L")
                mat = g5.selectbox("Material", MAT_OPTIONS, key=k(f"m_{i}"),
                                   index=MAT_OPTIONS.index(_mat_def) if _mat_def in MAT_OPTIONS else 0)
                _fit_idx = 0
                if seg["fittings"] in engine.FITTING_Le_over_D:
                    _fit_idx = list(engine.FITTING_Le_over_D.keys()).index(seg["fittings"]) + 1
                f = g6.selectbox("Minor Loss", fit_options, key=k(f"f_{i}"), index=_fit_idx,
                                 help="Fitting type for equivalent-length minor loss "
                                      "(Crane TP-410 Le/D method). Select None if not applicable.")
                c = g7.number_input("Qty", min_value=0, value=int(seg["fitting_count"]),
                                    key=k(f"c_{i}"),
                                    help="Number of this fitting type in the segment.")
                g8.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
                lined = g8.checkbox("Lined", value=bool(seg.get("lined",False)), key=k(f"lined_{i}"),
                                    help="Fluoropolymer liner (PTFE/FEP/PFA/PVDF) reduces effective "
                                         "bore and overrides wall roughness.")

                _lmat    = seg.get("liner_material","FEP")
                _lthk_mm = float(seg.get("liner_thickness_mm",1.0))
                if lined:
                    g9, g10 = st.columns([1.6,1.0])
                    _lmat = g9.selectbox("Liner Material", LINER_OPTIONS, key=k(f"lmat_{i}"),
                        index=LINER_OPTIONS.index(_lmat) if _lmat in LINER_OPTIONS else 0)
                    _lthk_mm = g10.number_input("Liner Thickness (mm)", min_value=0.1,
                        max_value=20.0, value=_lthk_mm, step=0.5, key=k(f"lthk_{i}"))

                D_seg = engine.PIPE_DATABASE[dn][pn]
                D_eff = D_seg - 2*(_lthk_mm/1000.0) if lined else D_seg
                rough = engine.LINER_ROUGHNESS[_lmat] if lined else engine.MATERIAL_ROUGHNESS[mat]
                if lined:
                    st.caption(f"Bore {D_seg*1000:.1f} mm → **ID {D_eff*1000:.1f} mm**"
                               f"  ·  ε {rough*1e6:.3g} µm  ·  {mat} + {_lmat} {_lthk_mm:.1f} mm")
                else:
                    st.caption(f"ID {D_seg*1000:.1f} mm  ·  ε {rough*1e6:.2g} µm  ·  {mat}")

                current_specs.append({
                    "type": t, "dn": dn, "pn": pn, "material": mat, "length": l,
                    "fittings": f if f != "None" else "None", "fitting_count": c,
                    "lined": lined, "liner_material": _lmat, "liner_thickness_mm": _lthk_mm,
                })

            st.session_state[k("segments")] = current_specs
            b1, b2 = st.columns(2)
            if b1.button("+ Add Segment", key=k("add_seg")):
                import copy
                _last = st.session_state[k("segments")][-1]
                st.session_state[k("segments")].append({
                    "type": "Horizontal", "dn": _last.get("dn","DN50"),
                    "pn": _last.get("pn","PN40"), "material": _last.get("material","SS316L"),
                    "length": 2.0, "fittings": "None", "fitting_count": 0,
                    "lined": _last.get("lined",False),
                    "liner_material": _last.get("liner_material","FEP"),
                    "liner_thickness_mm": _last.get("liner_thickness_mm",1.0),
                })
                st.rerun()
            if b2.button("- Remove Last", key=k("rem_seg")) and \
               len(st.session_state[k("segments")]) > 1:
                st.session_state[k("segments")].pop()
                st.rerun()

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    with col_out:
        st.subheader("Results")

        is_valid, warn_list = engine.validate_input_bounds(
            P_bara, T_C, gas_flows_kgh, liquid_type, q_lye)
        for w in warn_list:
            st.warning(w)

        props = engine.calculate_two_phase_properties(
            P_bara, T_C, gas_flows_kgh, liquid_type, q_lye,
            custom_gas=custom_gas, custom_liquid=custom_liquid)

        # Phase Properties
        with st.container(border=True):
            st.subheader("Phase Properties")
            col_gas, col_liq = st.columns(2)
            with col_gas:
                st.markdown("**Gas phase**")
                _comp     = props["composition"]
                _n_total  = sum(v["mol_h"] for v in _comp.values())
                _comp_rows = []
                for _sp, _data in _comp.items():
                    _comp_rows.append({"Component": _sp,
                                       "kg/h":  round(_data["kg_h"], 3),
                                       "mol/h": round(_data["mol_h"], 1),
                                       "mol %": f"{_data['mol_frac']*100:.1f}"})
                _comp_rows.append({"Component": "Total",
                                   "kg/h":  round(props["m_gas_total_kgh"], 3),
                                   "mol/h": round(_n_total, 1), "mol %": "100.0"})
                st.dataframe(pd.DataFrame(_comp_rows),
                             column_config={"kg/h": st.column_config.NumberColumn(format="%.3f"),
                                            "mol/h": st.column_config.NumberColumn(format="%.1f")},
                             hide_index=True, use_container_width=True)
                g1, g2, g3 = st.columns(3)
                g1.metric("ρ_gas",  f"{props['rho_g']:.3f} kg/m³")
                g2.metric("MW_mix", f"{props['MW_mix_gmol']:.2f} g/mol")
                if props.get("P_sat_H2O_pa", 0) > 0:
                    g3.metric("P_sat H₂O", f"{props['P_sat_H2O_pa']/1e5:.3f} bara")
            with col_liq:
                st.markdown(f"**Liquid phase ({liquid_type})**")
                l1, l2 = st.columns(2)
                l1.metric("ṁ_liquid", f"{props['m_lye_kgh']:.1f} kg/h")
                l2.metric("ρ_liquid", f"{props['rho_l']:.1f} kg/m³")
                l3, l4 = st.columns(2)
                l3.metric("μ_liquid", f"{props['mu_l']*1e3:.3f} mPa·s")
                l4.metric("σ",        f"{props['sigma']*1e3:.2f} mN/m")
                st.markdown("**Two-phase mixture**")
                _m_total_kgh = props["m_gas_total_kgh"] + props["m_liquid_total_kgh"]
                m1, m2, m3 = st.columns(3)
                m1.metric("ṁ_total",        f"{_m_total_kgh:.2f} kg/h")
                m2.metric("Mass quality x",  f"{props['x_gas']*100:.3f} %")
                m3.metric("Void fraction α", f"{props['alpha']*100:.2f} %")

        # Inlet Flow Conditions
        _seg1    = st.session_state[k("segments")][0]
        _D_bore1 = engine.PIPE_DATABASE[_seg1["dn"]][_seg1["pn"]]
        _lthk1   = _seg1.get("liner_thickness_mm",1.0) / 1000.0
        _D_in    = _D_bore1 - 2*_lthk1 if _seg1.get("lined",False) else _D_bore1
        _A_in    = 0.25 * np.pi * _D_in**2
        Vsg_in = (props["x_gas"]*props["m_total_kgs"]/props["rho_g"])/_A_in \
                 if props["rho_g"] > 0 else 0.0
        Vsl_in = ((1-props["x_gas"])*props["m_total_kgs"]/props["rho_l"])/_A_in \
                 if props["rho_l"] > 0 else 0.0
        Re_sl  = props["rho_l"]*Vsl_in*_D_in/props["mu_l"] if props["mu_l"] > 0 else 0.0

        with st.container(border=True):
            _ln1 = (f" + {_seg1['liner_material']} {_seg1['liner_thickness_mm']:.1f}mm"
                    if _seg1.get("lined") else "")
            st.subheader(f"Inlet Flow Conditions  ({_seg1['dn']}/{_seg1['pn']}{_ln1})")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("V_sg (gas)",    f"{Vsg_in:.3f} m/s")
            v2.metric("V_sl (liquid)", f"{Vsl_in:.3f} m/s")
            v3.metric("V_m (mixture)", f"{Vsg_in+Vsl_in:.3f} m/s")
            v4.metric("Re_liquid",     f"{Re_sl:,.0f}")

        # Segment loop
        current_P          = P_bara * 1e5
        grid_records       = []
        cumulative_positions = []
        cumulative_distance  = 0.0
        pressure_profile_x   = [0.0]
        pressure_profile_y   = [P_bara]
        regime_bands         = []
        total_dp_fric_kpa    = 0.0
        total_dp_grav_kpa    = 0.0
        total_dp_accel_kpa   = 0.0

        for i, seg in enumerate(st.session_state[k("segments")]):
            D_seg   = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
            _lined  = seg.get("lined",False)
            _lthk_m = seg.get("liner_thickness_mm",1.0) / 1000.0
            _lmat   = seg.get("liner_material","FEP")
            D_eff   = D_seg - 2*_lthk_m if _lined else D_seg
            rough_seg = (engine.LINER_ROUGHNESS[_lmat] if _lined
                         else engine.MATERIAL_ROUGHNESS[seg.get("material","SS316L")])

            props_seg = engine.calculate_two_phase_properties(
                current_P/1e5, T_C, gas_flows_kgh, liquid_type, q_lye,
                custom_gas=custom_gas, custom_liquid=custom_liquid)

            angle = {"Horizontal": 0.0,
                     "Vertical Upflow": np.pi/2.0,
                     "Vertical Downflow": -np.pi/2.0}[seg["type"]]
            le_fit = 0.0
            if seg["fittings"] in engine.FITTING_Le_over_D:
                le_fit = engine.FITTING_Le_over_D[seg["fittings"]]*D_eff*seg["fitting_count"]
            L_eff = seg["length"] + le_fit

            seg_result = engine.calculate_segment_pressure_drop(
                props_seg, D_eff, rough_seg, L_eff, angle,
                correlation=correlation, voidage_method=voidage_method)

            dP_Pa      = seg_result["dP_Pa"]
            regime     = seg_result["regime"]
            dP_per_dz  = seg_result["dP_per_dz"]
            Vsg        = seg_result["Vsg"]
            Vsl        = seg_result["Vsl"]
            alpha_seg  = seg_result["alpha"]
            dP_fric_Pa = seg_result["dP_fric_Pa"]
            dP_grav_Pa = seg_result["dP_grav_Pa"]
            dP_accel_Pa= seg_result["dP_accel_Pa"]

            V_m = Vsg + Vsl
            V_e, _ = engine.calculate_erosion_velocity(
                props_seg["rho_g"], props_seg["rho_l"], props_seg["x_gas"])
            erosion_ratio = V_m / V_e if V_e > 0 else 0.0

            end_P    = current_P - dP_Pa
            _mat_str = seg.get("material","SS316L")
            if _lined:
                _mat_str += f" / {_lmat} {seg.get('liner_thickness_mm',1.0):.1f}mm"
            grid_records.append({
                # --- actionable columns first ---
                "Seg":             f"#{i+1}",
                "Type":            seg["type"],
                "Pipe":            f"{seg['dn']}/{seg['pn']}",
                "ID (mm)":         round(D_eff*1000, 1),
                "L (m)":           seg["length"],
                "Regime":          regime,
                "ΔP (kPa)":        round(dP_Pa/1000, 3),
                "P_in (bara)":     round(current_P/1e5, 4),
                "P_out (bara)":    round(end_P/1e5, 4),
                "V_m (m/s)":       round(V_m, 3),
                "V_m/V_e":         round(erosion_ratio, 3),
                "V_sg (m/s)":      round(Vsg, 3),
                "V_sl (m/s)":      round(Vsl, 3),
                "V_e (m/s)":       round(V_e, 2),
                "ΔP_fric (kPa)":   round(dP_fric_Pa/1000, 3),
                "ΔP_grav (kPa)":   round(dP_grav_Pa/1000, 3),
                "ΔP_accel (kPa)":  round(dP_accel_Pa/1000, 3),
                # --- internal / secondary columns ---
                "Material":        _mat_str,
                "ρ_g (kg/m³)":    round(props_seg["rho_g"], 4),
                "L_eff (m)":       round(L_eff, 2),
                "α (void)":        round(alpha_seg, 4),
                "dP/dz (Pa/m)":    round(dP_per_dz, 2),
            })
            total_dp_fric_kpa  += dP_fric_Pa / 1000.0
            total_dp_grav_kpa  += dP_grav_Pa / 1000.0
            total_dp_accel_kpa += dP_accel_Pa / 1000.0
            current_P = end_P
            cumulative_distance += L_eff
            cumulative_positions.append(cumulative_distance)
            pressure_profile_x.append(cumulative_distance)
            pressure_profile_y.append(max(0.1, current_P/1e5))
            regime_bands.append(regime)

        # Erosion banner
        _max_ratio = max((r["V_m/V_e"] for r in grid_records), default=0.0)
        _worst_seg = next((r["Seg"] for r in grid_records if r["V_m/V_e"]==_max_ratio), "")
        if _max_ratio >= 1.0:
            st.error(f"**Erosion limit exceeded** — Segment {_worst_seg}: "
                     f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, C=100). "
                     f"Reduce velocity or increase pipe diameter.")
        elif _max_ratio >= 0.8:
            st.warning(f"**Approaching erosion limit** — Segment {_worst_seg}: "
                       f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, limit = 1.0).")
        else:
            st.success(f"Erosion check OK — worst {_worst_seg}: "
                       f"V_m/V_e = {_max_ratio:.2f}  (API RP 14E, C=100, limit = 1.0).")

        # System Totals  (shown before the detail table)
        total_dp_kpa         = ((P_bara*1e5) - current_P) / 1000.0
        outlet_pressure_bara = max(0.1, current_P/1e5)
        pipe_length_m        = sum(s["length"] for s in st.session_state[k("segments")])

        with st.container(border=True):
            st.subheader("System Totals")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total ΔP",       f"{total_dp_kpa:.3f} kPa",
                      delta=f"−{total_dp_kpa/100:.4f} bar", delta_color="inverse")
            s2.metric("Outlet Pressure", f"{outlet_pressure_bara:.4f} bara")
            s3.metric("Pipe Length",     f"{pipe_length_m:.1f} m")
            s4.metric("Eff. Length",     f"{cumulative_distance:.1f} m")
            st.divider()
            st.markdown("**ΔP decomposition**")
            d1, d2, d3 = st.columns(3)
            d1.metric("Σ Frictional",    f"{total_dp_fric_kpa:.3f} kPa",
                      help="Sum of frictional component across all segments")
            d2.metric("Σ Gravitational", f"{total_dp_grav_kpa:.3f} kPa",
                      help="Sum of gravitational component (negative = pressure recovery from downflow)")
            d3.metric("Σ Accelerational",f"{total_dp_accel_kpa:.3f} kPa",
                      help="Residual (B&B inclination correction). Zero for non-B&B correlations.")

        # Segment table
        st.subheader("Segment Analysis")
        st.dataframe(pd.DataFrame(grid_records),
            column_config={
                "ID (mm)":         st.column_config.NumberColumn(format="%.1f"),
                "L (m)":           st.column_config.NumberColumn(format="%.1f"),
                "ΔP (kPa)":        st.column_config.NumberColumn(format="%.3f"),
                "P_in (bara)":     st.column_config.NumberColumn(format="%.4f"),
                "P_out (bara)":    st.column_config.NumberColumn(format="%.4f"),
                "V_m (m/s)":       st.column_config.NumberColumn(format="%.3f"),
                "V_m/V_e":         st.column_config.NumberColumn(format="%.3f"),
                "V_sg (m/s)":      st.column_config.NumberColumn(format="%.3f"),
                "V_sl (m/s)":      st.column_config.NumberColumn(format="%.3f"),
                "V_e (m/s)":       st.column_config.NumberColumn(format="%.2f"),
                "ΔP_fric (kPa)":   st.column_config.NumberColumn(format="%.3f"),
                "ΔP_grav (kPa)":   st.column_config.NumberColumn(format="%.3f"),
                "ΔP_accel (kPa)":  st.column_config.NumberColumn(format="%.3f"),
                "ρ_g (kg/m³)":    st.column_config.NumberColumn(format="%.4f"),
                "L_eff (m)":       st.column_config.NumberColumn(format="%.2f"),
                "α (void)":        st.column_config.NumberColumn(format="%.4f"),
                "dP/dz (Pa/m)":    st.column_config.NumberColumn(format="%.2f"),
            }, hide_index=True, use_container_width=True)

        # ΔP decomposition stacked bar chart
        _seg_labels = [r["Seg"] + " " + r["Pipe"] for r in grid_records]
        _dp_fric    = [r["ΔP_fric (kPa)"] for r in grid_records]
        _dp_grav    = [r["ΔP_grav (kPa)"] for r in grid_records]
        _dp_accel   = [r["ΔP_accel (kPa)"] for r in grid_records]
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(name="Frictional",    x=_seg_labels, y=_dp_fric,
                                    marker_color="#2563EB", opacity=0.90))
        fig_decomp.add_trace(go.Bar(name="Gravitational", x=_seg_labels, y=_dp_grav,
                                    marker_color="#D97706", opacity=0.90))
        if any(abs(v) > 1e-4 for v in _dp_accel):
            fig_decomp.add_trace(go.Bar(name="Accelerational", x=_seg_labels, y=_dp_accel,
                                        marker_color="#059669", opacity=0.90))
        fig_decomp.update_layout(
            barmode="relative", yaxis_title="ΔP (kPa)", xaxis_title="Segment",
            title=dict(text=f"ΔP Decomposition — {correlation}  ·  void: {voidage_method}",
                       font=dict(size=13), x=0),
            template="plotly_white", height=300,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", zeroline=True,
                       zerolinecolor="#94A3B8", zerolinewidth=1),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                        borderwidth=1, orientation="h", y=1.08),
            font=dict(size=12, color="#374151"))
        st.plotly_chart(fig_decomp, use_container_width=True, key=k("fig_decomp"))

    # ── VISUALISATIONS ────────────────────────────────────────────────────────
    _nodes = [(0.0, 0.0)]
    for _seg in st.session_state[k("segments")]:
        _xl, _yl = _nodes[-1]
        if _seg["type"] == "Horizontal":
            _nodes.append((_xl+_seg["length"], _yl))
        elif _seg["type"] == "Vertical Upflow":
            _nodes.append((_xl, _yl+_seg["length"]))
        else:
            _nodes.append((_xl, _yl-_seg["length"]))

    _DN_LW = {"DN40":5,"DN50":7,"DN80":9,"DN100":11,"DN150":15,"DN200":19,"DN250":23}

    fig_sch = go.Figure()
    _seen_reg = set()
    for _i, (_seg, _rec) in enumerate(zip(st.session_state[k("segments")], grid_records)):
        _x0,_y0 = _nodes[_i]; _x1,_y1 = _nodes[_i+1]
        _reg = _rec["Regime"]
        _col = _regime_color(_reg, _REGIME_LINE_KW, "#64748B")
        _lw  = _DN_LW.get(_seg["dn"], 10)
        _show = _reg not in _seen_reg; _seen_reg.add(_reg)
        _lh = (f"Liner: {_seg['liner_material']} {_seg['liner_thickness_mm']:.1f} mm"
               f"  →  ID {_rec['ID (mm)']:.1f} mm<br>" if _seg.get("lined") else "")
        fig_sch.add_trace(go.Scatter(
            x=[_x0,_x1], y=[_y0,_y1], mode="lines",
            line=dict(color=_col, width=_lw), name=_reg, legendgroup=_reg,
            showlegend=_show,
            hovertemplate=(f"<b>Seg #{_i+1}  {_seg['dn']}/{_seg['pn']}</b><br>"
                           +_lh+f"{_seg['type']},  L={_seg['length']:.1f} m<br>"
                           f"Regime: {_reg}<br>ΔP: {_rec['ΔP (kPa)']:.3f} kPa  ·  "
                           f"V_sg: {_rec['V_sg (m/s)']:.3f} m/s<extra></extra>")))
        fig_sch.add_annotation(
            x=_x0+(_x1-_x0)*0.65, y=_y0+(_y1-_y0)*0.65,
            ax=_x0+(_x1-_x0)*0.5, ay=_y0+(_y1-_y0)*0.5,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.8, arrowwidth=2.5, arrowcolor=_col)
        fig_sch.add_annotation(
            x=(_x0+_x1)/2, y=(_y0+_y1)/2, text=f"<b>#{_i+1}</b> {_seg['dn']}",
            showarrow=False, font=dict(size=10,color="#1E293B"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=_col, borderwidth=1.5,
            borderpad=3, xanchor="center", yanchor="middle")

    _all_x=[n[0] for n in _nodes]; _all_y=[n[1] for n in _nodes]
    _xspan=max(_all_x)-min(_all_x); _yspan=max(_all_y)-min(_all_y)
    _xpad=max(_xspan*0.12,2.0); _ypad=max(_yspan*0.12,max(_xspan*0.10,2.0))
    fig_sch.add_trace(go.Scatter(x=[_nodes[0][0]],y=[_nodes[0][1]],
        mode="markers+text", marker=dict(size=14,color="#059669",symbol="circle",
        line=dict(color="white",width=2)), text=["IN"], textposition="bottom center",
        textfont=dict(size=11,color="#059669"), showlegend=False,
        hovertemplate=f"Inlet  ·  {P_bara:.2f} bara<extra></extra>"))
    _otp = "top center" if _nodes[-1][1] >= _nodes[0][1] else "bottom center"
    fig_sch.add_trace(go.Scatter(x=[_nodes[-1][0]],y=[_nodes[-1][1]],
        mode="markers+text", marker=dict(size=14,color="#DC2626",symbol="circle",
        line=dict(color="white",width=2)), text=["OUT"], textposition=_otp,
        textfont=dict(size=11,color="#DC2626"), showlegend=False,
        hovertemplate=f"Outlet  ·  {outlet_pressure_bara:.4f} bara<extra></extra>"))
    fig_sch.update_layout(template="plotly_white", height=440,
        margin=dict(l=60,r=20,t=30,b=50), hovermode="closest",
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Horizontal Distance (m)", gridcolor="#F1F5F9",
                   zeroline=True, zerolinecolor="#CBD5E1", linecolor="#E2E8F0",
                   range=[min(_all_x)-_xpad, max(_all_x)+_xpad]),
        yaxis=dict(title="Elevation (m)", gridcolor="#F1F5F9",
                   zeroline=True, zerolinecolor="#CBD5E1", linecolor="#E2E8F0",
                   range=[min(_all_y)-_ypad, max(_all_y)+_ypad]),
        legend=dict(title="Flow Regime", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E2E8F0", borderwidth=1, font=dict(size=11)),
        font=dict(size=12,color="#374151"))

    fig_prof = go.Figure()
    _x0p = 0.0
    for _x1p, _rp in zip(cumulative_positions, regime_bands):
        _fill  = _regime_color(_rp, _REGIME_FILL_KW,   "rgba(241,245,249,0.60)")
        _bord  = _regime_color(_rp, _REGIME_BORDER_KW, "#CBD5E1")
        fig_prof.add_shape(type="rect", x0=_x0p, x1=_x1p, y0=0, y1=1, yref="paper",
                           fillcolor=_fill, line=dict(color=_bord,width=1), layer="below")
        fig_prof.add_annotation(x=(_x0p+_x1p)/2, y=0.96, yref="paper",
                                text=_rp[:18], showarrow=False,
                                font=dict(size=9,color="#475569"))
        _x0p = _x1p
    fig_prof.add_trace(go.Scatter(x=pressure_profile_x, y=pressure_profile_y,
        mode="lines+markers", line=dict(color=accent, width=2.5),
        marker=dict(size=7, color=accent),
        hovertemplate="Distance: %{x:.2f} m<br>Pressure: %{y:.4f} bara<extra></extra>"))
    fig_prof.update_layout(xaxis_title="Pipeline Distance (m)", yaxis_title="Pressure (bara)",
        template="plotly_white", height=340, margin=dict(l=60,r=20,t=30,b=50),
        hovermode="x unified", showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9",zeroline=False,linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9",zeroline=False,linecolor="#E2E8F0"),
        font=dict(size=12,color="#374151"))

    st.divider()
    tab_sch, tab_prof_tab = st.tabs(["Pipeline Schematic", "Pressure Profile"])
    with tab_sch:
        st.plotly_chart(fig_sch, use_container_width=True, key=k("fig_sch"))
        st.caption("Line width ∝ DN  ·  Colour = flow regime  ·  Arrow = flow direction")
    with tab_prof_tab:
        st.plotly_chart(fig_prof, use_container_width=True, key=k("fig_prof"))

    # ── EXPORTS ───────────────────────────────────────────────────────────────
    st.divider()
    ex_tab_w, ex_tab_x = st.tabs(["Export Word (.docx)", "Export Excel (.xlsx)"])
    with ex_tab_w:
        _rpt_hash = hashlib.md5(json.dumps({
            "P": P_bara, "T": T_C,
            "gas_flows": {_kk: float(_vv) for _kk,_vv in gas_flows_kgh.items()},
            "liquid_type": liquid_type, "lye": q_lye,
            "segs": [(s["type"],s["dn"],s["pn"],s["length"],s["fittings"],
                      s["fitting_count"],s.get("lined",False),
                      s.get("liner_material","FEP"),s.get("liner_thickness_mm",1.0))
                     for s in st.session_state[k("segments")]],
        }, sort_keys=True).encode()).hexdigest()
        if st.session_state.get(k("rpt_hash")) != _rpt_hash:
            st.session_state[k("rpt_hash")]  = None
            st.session_state[k("rpt_bytes")] = None
        _wc1, _wc2 = st.columns([1,2])
        with _wc1:
            if st.button("Generate Report", type="primary",
                         use_container_width=True, key=k("gen_rpt")):
                with st.spinner("Building document…"):
                    _buf = report_generator.generate_report(
                        P_bara=P_bara, T_C=T_C,
                        gas_flows_kgh=gas_flows_kgh, liquid_type=liquid_type, q_lye=q_lye,
                        props=props, grid_records=grid_records,
                        segments=st.session_state[k("segments")],
                        total_dp_kpa=total_dp_kpa,
                        outlet_pressure_bara=outlet_pressure_bara,
                        pipe_length_m=pipe_length_m,
                        cumulative_distance=cumulative_distance,
                        fig_sch=fig_sch, fig_prof=fig_prof)
                    st.session_state[k("rpt_bytes")] = _buf.getvalue()
                    st.session_state[k("rpt_hash")]  = _rpt_hash
        with _wc2:
            if st.session_state.get(k("rpt_bytes")) and \
               st.session_state.get(k("rpt_hash")) == _rpt_hash:
                st.download_button("Download  (.docx)",
                    data=st.session_state[k("rpt_bytes")],
                    file_name=f"hydraulic_report_case_{cid}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key=k("dl_rpt"))
            elif st.session_state.get(k("rpt_bytes")):
                st.info("Inputs changed — regenerate report.")

    with ex_tab_x:
        def _build_xlsx_case():
            from io import BytesIO
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter
            BLUE="2563EB"; LGRAY="F1F5F9"; STRIPE="F8FAFC"
            _hf=Font(bold=True,color="FFFFFF",size=10,name="Calibri")
            _hfill=PatternFill("solid",fgColor=BLUE)
            _sf=Font(bold=True,size=9,name="Calibri")
            _sfill=PatternFill("solid",fgColor=LGRAY)
            _df=Font(size=9,name="Calibri")
            _af=PatternFill("solid",fgColor=STRIPE)
            _th=Side(style="thin",color="CBD5E1")
            _bdr=Border(left=_th,right=_th,top=_th,bottom=_th)
            def _hdr(ws,row,text,nc):
                c=ws.cell(row=row,column=1,value=text); c.font=_hf; c.fill=_hfill; c.border=_bdr
                for col in range(2,nc+2):
                    cc=ws.cell(row=row,column=col); cc.fill=_hfill; cc.border=_bdr
                return row+1
            def _sec(ws,row,text,nc):
                c=ws.cell(row=row,column=1,value=text); c.font=_sf; c.fill=_sfill; c.border=_bdr
                for col in range(2,nc+2):
                    cc=ws.cell(row=row,column=col); cc.fill=_sfill; cc.border=_bdr
                return row+1
            def _dat(ws,row,label,value,alt=False):
                fill=_af if alt else None
                lc=ws.cell(row=row,column=1,value=label); lc.font=_df; lc.border=_bdr
                if fill: lc.fill=fill
                vc=ws.cell(row=row,column=2,value=value); vc.font=_df; vc.border=_bdr
                if fill: vc.fill=fill
                return row+1
            wb=openpyxl.Workbook()
            ws=wb.active; ws.title="System"; ws.freeze_panes="A2"
            r=_hdr(ws,1,"System Summary",1)
            alt=False
            r=_sec(ws,r,"Boundary Conditions",1)
            for lbl,val in [("Inlet Pressure (bara)",round(P_bara,4)),
                             ("Temperature (°C)",round(T_C,2)),
                             ("Outlet Pressure (bara)",round(outlet_pressure_bara,4)),
                             ("Total ΔP (kPa)",round(total_dp_kpa,4)),
                             ("Total ΔP (bar)",round(total_dp_kpa/100,6)),
                             ("Pipe Length (m)",round(pipe_length_m,3)),
                             ("Effective Length (m)",round(cumulative_distance,3))]:
                r=_dat(ws,r,lbl,val,alt); alt=not alt
            r+=1; alt=False
            r=_sec(ws,r,"Gas Phase  —  inlet conditions",1)
            for _sp,_fl in gas_flows_kgh.items():
                r=_dat(ws,r,f"{_sp} mass flow (kg/h)",round(float(_fl),4),alt); alt=not alt
            for lbl,val in [("Total gas (kg/h)",round(props["m_gas_total_kgh"],4)),
                             ("ρ_g (kg/m³)",round(props["rho_g"],4)),
                             ("MW_mix (g/mol)",round(props["MW_mix_gmol"],3)),
                             ("x_gas (%)",round(props["x_gas"]*100,5)),
                             ("α void (%)",round(props["alpha"]*100,4))]:
                r=_dat(ws,r,lbl,val,alt); alt=not alt
            if props.get("P_sat_H2O_pa",0)>0:
                for lbl,val in [("H₂O vapour (kg/h)",round(props["m_vapor_h2o_kgh"],5)),
                                 ("P_sat H₂O (bara)",round(props["P_sat_H2O_pa"]/1e5,5))]:
                    r=_dat(ws,r,lbl,val,alt); alt=not alt
            r+=1; alt=False
            r=_sec(ws,r,f"Liquid Phase  —  {liquid_type}",1)
            for lbl,val in [("Volume flow (m³/h)",round(q_lye,4)),
                             ("Mass flow (kg/h)",round(props["m_lye_kgh"],3)),
                             ("ρ_l (kg/m³)",round(props["rho_l"],3)),
                             ("μ_l (mPa·s)",round(props["mu_l"]*1e3,4)),
                             ("σ (mN/m)",round(props["sigma"]*1e3,4))]:
                r=_dat(ws,r,lbl,val,alt); alt=not alt
            ws.column_dimensions["A"].width=42; ws.column_dimensions["B"].width=20
            ws2=wb.create_sheet("Segments"); ws2.freeze_panes="B2"
            n=len(grid_records)
            c=ws2.cell(row=1,column=1,value="Parameter"); c.font=_hf; c.fill=_hfill; c.border=_bdr
            for j,rec in enumerate(grid_records):
                lbl=f"Seg {rec['Seg']}  {rec['Pipe']}"
                cc=ws2.cell(row=1,column=j+2,value=lbl); cc.font=_hf; cc.fill=_hfill; cc.border=_bdr
            _ROWS=[("Orientation","Type"),("Pipe class","Pipe"),("Inner diameter (mm)","ID (mm)"),
                   ("Material","Material"),("Physical length (m)","L (m)"),
                   ("Effective length (m)","L_eff (m)"),
                   ("P_in  Inlet pressure (bara)","P_in (bara)"),
                   ("P_out  Outlet pressure (bara)","P_out (bara)"),
                   ("ΔP  Pressure drop (kPa)","ΔP (kPa)"),
                   ("dP/dz  Pressure gradient (Pa/m)","dP/dz (Pa/m)"),
                   ("Gas density ρ_g (kg/m³)","ρ_g (kg/m³)"),("Flow regime","Regime"),
                   ("V_sg  Superficial gas (m/s)","V_sg (m/s)"),
                   ("V_sl  Superficial liquid (m/s)","V_sl (m/s)"),
                   ("V_m  Mixture velocity (m/s)","V_m (m/s)"),
                   ("V_e  Erosion limit API RP 14E (m/s)","V_e (m/s)"),
                   ("V_m/V_e  Erosion ratio (–)","V_m/V_e")]
            for i,(label,key) in enumerate(_ROWS):
                row=i+2; alt=(i%2==1)
                lc=ws2.cell(row=row,column=1,value=label); lc.font=_df; lc.border=_bdr
                if alt: lc.fill=_af
                for j,rec in enumerate(grid_records):
                    vc=ws2.cell(row=row,column=j+2,value=rec.get(key,"")); vc.font=_df; vc.border=_bdr
                    if alt: vc.fill=_af
            ws2.column_dimensions["A"].width=46
            for j in range(n):
                ws2.column_dimensions[get_column_letter(j+2)].width=18
            buf=BytesIO(); wb.save(buf); buf.seek(0); return buf

        _xc1, _xc2 = st.columns([1, 2])
        with _xc1:
            if st.button("Generate Excel", use_container_width=True, key=k("gen_xl")):
                try:
                    st.session_state[k("xl_bytes")] = _build_xlsx_case().getvalue()
                except Exception as _xe:
                    st.error(f"Excel export failed: {_xe}")
        with _xc2:
            if st.session_state.get(k("xl_bytes")):
                st.download_button("Download  (.xlsx)",
                    data=st.session_state[k("xl_bytes")],
                    file_name=f"hydraulic_data_case_{cid}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, key=k("dl_xl"))

    return {
        "P_bara":               P_bara,
        "T_C":                  T_C,
        "gas_flows_kgh":        gas_flows_kgh,
        "liquid_type":          liquid_type,
        "q_lye":                q_lye,
        "props":                props,
        "grid_records":         grid_records,
        "total_dp_kpa":         total_dp_kpa,
        "total_dp_fric_kpa":    total_dp_fric_kpa,
        "total_dp_grav_kpa":    total_dp_grav_kpa,
        "total_dp_accel_kpa":   total_dp_accel_kpa,
        "outlet_pressure_bara": outlet_pressure_bara,
        "pipe_length_m":        pipe_length_m,
        "cumulative_distance":  cumulative_distance,
        "pressure_profile_x":   pressure_profile_x,
        "pressure_profile_y":   pressure_profile_y,
        "segments":             st.session_state[k("segments")],
        "correlation":          correlation,
        "voidage_method":       voidage_method,
        "custom_gas":           custom_gas,
        "custom_liquid":        custom_liquid,
        "fig_sch":              fig_sch,
        "fig_prof":             fig_prof,
    }


# ============================================================================
# TOP-LEVEL TABS
# ============================================================================
# ============================================================================
# GOAL-SEEK HELPERS  — used by the Compare tab
# ============================================================================

def _calc_dp_at_p(res, P_bara_override):
    """
    Re-run the pipeline in res at a different inlet pressure.
    Returns (total_dp_kpa, outlet_bara).  Uses the same correlation / voidage
    / segments / flow rates as the original case.
    """
    current_P = P_bara_override * 1e5
    total_dp  = 0.0
    corr  = res.get("correlation",     engine.TWO_PHASE_CORRELATIONS[0])
    void  = res.get("voidage_method",  engine.VOIDAGE_METHODS[0])
    cgas  = res.get("custom_gas")
    cliq  = res.get("custom_liquid")
    for seg in res["segments"]:
        D_seg  = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        lined  = seg.get("lined", False)
        lthk_m = seg.get("liner_thickness_mm", 1.0) / 1000.0
        lmat   = seg.get("liner_material", "FEP")
        D_eff  = D_seg - 2 * lthk_m if lined else D_seg
        rough  = (engine.LINER_ROUGHNESS[lmat] if lined
                  else engine.MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
        props  = engine.calculate_two_phase_properties(
            current_P / 1e5, res["T_C"],
            res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
            custom_gas=cgas, custom_liquid=cliq)
        angle  = {"Horizontal": 0.0, "Vertical Upflow": np.pi / 2.0,
                  "Vertical Downflow": -np.pi / 2.0}[seg["type"]]
        le_fit = 0.0
        if seg["fittings"] in engine.FITTING_Le_over_D:
            le_fit = engine.FITTING_Le_over_D[seg["fittings"]] * D_eff * seg["fitting_count"]
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)
        total_dp  += seg_res["dP_Pa"]
        current_P -= seg_res["dP_Pa"]
        current_P  = max(1e4, current_P)   # guard: never pass sub-vacuum to thermodynamics
    dp_kpa      = total_dp / 1000.0
    outlet_bara = current_P / 1e5
    return dp_kpa, outlet_bara


def _goal_seek_inlet(res_c, res_a, res_b, P_target_out, tol=0.0005, max_iter=25):
    """
    Find the Case C inlet pressure such that the worst-case branch outlet
    (min of A_out, B_out) equals P_target_out.

    Strategy: monotonic relationship ⟹ simple successive-substitution.
    Returns a result dict.
    """
    # Seed with first-order estimate from current ΔP values
    P_c_in = (P_target_out
              + max(res_a["total_dp_kpa"], res_b["total_dp_kpa"]) / 100.0
              + res_c["total_dp_kpa"] / 100.0)

    dp_c = dp_a = dp_b = 0.0
    P_c_out = P_a_out = P_b_out = 0.0

    for i in range(max_iter):
        dp_c, P_c_out = _calc_dp_at_p(res_c, P_c_in)
        dp_a, P_a_out = _calc_dp_at_p(res_a, P_c_out)
        dp_b, P_b_out = _calc_dp_at_p(res_b, P_c_out)

        worst_out = min(P_a_out, P_b_out)
        error     = worst_out - P_target_out

        if abs(error) < tol:
            return {
                "P_c_in": P_c_in, "P_c_out": P_c_out,
                "dp_c": dp_c, "dp_a": dp_a, "dp_b": dp_b,
                "P_a_out": P_a_out, "P_b_out": P_b_out,
                "worst_out": worst_out, "iterations": i + 1, "converged": True,
            }
        # Direct correction — valid because ΔP barely changes with P
        P_c_in -= error   # if error > 0 (too high) reduce; if < 0 (too low) increase

    return {
        "P_c_in": P_c_in, "P_c_out": P_c_out,
        "dp_c": dp_c, "dp_a": dp_a, "dp_b": dp_b,
        "P_a_out": P_a_out, "P_b_out": P_b_out,
        "worst_out": worst_out, "iterations": max_iter,
        "converged": abs(min(P_a_out, P_b_out) - P_target_out) < tol * 20,
    }


tab_a, tab_b, tab_c, tab_cmp = st.tabs(["Case A", "Case B", "Case C — Header", "Compare A vs B"])

with tab_a:
    results_a = run_case("a", accent="#2563EB")

with tab_b:
    results_b = run_case("b", accent="#D97706")

with tab_c:
    st.info(
        "**Case C — Header / Manifold**  "
        "Pre-loaded with larger-bore defaults (DN100/DN150). "
        "Adjust geometry, flow rates, and pipe specs as needed.",
        icon="ℹ️",
    )
    results_c = run_case("c", accent="#059669", default_segments=_DEFAULT_HDR_SEGMENTS)

# ============================================================================
# COMPARE TAB
# ============================================================================

def _sens_hash(ra, rb):
    """Stable hash of all inputs that determine sensitivity results."""
    _data = {
        "a": {
            "P": ra["P_bara"], "T": ra["T_C"],
            "gas": {_k: float(_v) for _k, _v in ra["gas_flows_kgh"].items()},
            "liq": ra["liquid_type"], "lye": ra["q_lye"],
            "segs": [(s["type"], s["dn"], s["pn"], float(s["length"]),
                      s["fittings"], int(s["fitting_count"]),
                      bool(s.get("lined", False)), s.get("liner_material", "FEP"),
                      float(s.get("liner_thickness_mm", 1.0)))
                     for s in ra["segments"]],
        },
        "b": {
            "P": rb["P_bara"], "T": rb["T_C"],
            "gas": {_k: float(_v) for _k, _v in rb["gas_flows_kgh"].items()},
            "liq": rb["liquid_type"], "lye": rb["q_lye"],
            "segs": [(s["type"], s["dn"], s["pn"], float(s["length"]),
                      s["fittings"], int(s["fitting_count"]),
                      bool(s.get("lined", False)), s.get("liner_material", "FEP"),
                      float(s.get("liner_thickness_mm", 1.0)))
                     for s in rb["segments"]],
        },
    }
    return hashlib.md5(json.dumps(_data, sort_keys=True).encode()).hexdigest()


_CORR_SHORT = {
    "Beggs-Brill": "BB", "Friedel": "Friedel",
    "Lockhart_Martinelli": "L-M", "Muller_Steinhagen_Heck": "MSH",
    "Chisholm": "Chisholm", "Kim_Mudawar": "Kim-M",
}
_VOID_SHORT = {
    "Homogeneous": "Homo",
    "Rouhani-1 (slip)": "Rouhani-1",
}

with tab_cmp:
    st.subheader("Case A  vs.  Case B")

    ra, rb = results_a, results_b

    # ── Side-by-side headline metrics ─────────────────────────────────────────
    with st.container(border=True):
        _lc, _mc, _rc = st.columns([2, 2, 2])

        _lc.markdown("**Metric**")
        _mc.markdown(f"**Case A**")
        _rc.markdown(f"**Case B**")
        st.divider()

        def _cmp_row(label, va, vb, fmt="{}", better="lower", unit=""):
            """Render one comparison row with delta badge."""
            _lc, _mc, _rc = st.columns([2, 2, 2])
            try:
                _delta = vb - va
                _pct   = (_delta / abs(va) * 100) if abs(va) > 1e-9 else 0.0
                _sign  = "+" if _delta > 0 else ""
                _delta_str = f"{_sign}{fmt.format(_delta)} {unit}  ({_sign}{_pct:.1f}%)"
                if better == "lower":
                    _color = "normal" if _delta > 1e-9 else ("inverse" if _delta < -1e-9 else "off")
                elif better == "higher":
                    _color = "inverse" if _delta > 1e-9 else ("normal" if _delta < -1e-9 else "off")
                else:  # neutral — show delta magnitude without colour judgement
                    _color = "off"
            except Exception:
                _delta_str = "—"
                _color = "off"
            _lc.markdown(label)
            _mc.metric("Case A", f"{fmt.format(va)} {unit}", label_visibility="collapsed")
            _rc.metric("Case B", f"{fmt.format(vb)} {unit}", delta=_delta_str,
                       delta_color=_color, label_visibility="collapsed")

        _cmp_row("Inlet pressure",        ra["P_bara"],               rb["P_bara"],               fmt="{:.2f}", unit="bara", better="neutral")
        _cmp_row("Outlet pressure",       ra["outlet_pressure_bara"], rb["outlet_pressure_bara"], fmt="{:.4f}", unit="bara", better="higher")
        _cmp_row("Total ΔP",              ra["total_dp_kpa"],         rb["total_dp_kpa"],         fmt="{:.3f}", unit="kPa",  better="lower")
        _cmp_row("  ↳ Frictional",        ra["total_dp_fric_kpa"],    rb["total_dp_fric_kpa"],    fmt="{:.3f}", unit="kPa",  better="lower")
        _cmp_row("  ↳ Gravitational",     ra["total_dp_grav_kpa"],    rb["total_dp_grav_kpa"],    fmt="{:.3f}", unit="kPa",  better="neutral")
        _cmp_row("Pipe length",           ra["pipe_length_m"],        rb["pipe_length_m"],        fmt="{:.1f}", unit="m",    better="lower")
        _cmp_row("Effective length",      ra["cumulative_distance"],  rb["cumulative_distance"],  fmt="{:.1f}", unit="m",    better="lower")
        _max_a = max((r["V_m/V_e"] for r in ra["grid_records"]), default=0.0)
        _max_b = max((r["V_m/V_e"] for r in rb["grid_records"]), default=0.0)
        _cmp_row("Worst V_m/V_e",         _max_a,                     _max_b,                     fmt="{:.3f}", unit="–",    better="lower")

    # ── System Total ΔP  (Header C + Branch A or B) ──────────────────────────
    st.markdown("#### System Total ΔP — Header (C) + Branch")
    with st.container(border=True):
        _dp_c    = results_c["total_dp_kpa"]
        _dp_a    = ra["total_dp_kpa"]
        _dp_b    = rb["total_dp_kpa"]
        _total_a = _dp_c + _dp_a
        _total_b = _dp_c + _dp_b
        _p_in_c  = results_c["P_bara"]          # system inlet = Case C inlet
        _p_out_a = _p_in_c - _total_a / 100.0
        _p_out_b = _p_in_c - _total_b / 100.0
        _worst_total  = max(_total_a, _total_b)
        _worst_label  = "C + A" if _total_a >= _total_b else "C + B"
        _p_out_worst  = _p_in_c - _worst_total / 100.0

        st.caption(
            f"System inlet = Case C inlet pressure  ·  "
            f"Case C header ΔP: **{_dp_c:.3f} kPa**  ·  "
            f"Branch A ΔP: **{_dp_a:.3f} kPa**  ·  "
            f"Branch B ΔP: **{_dp_b:.3f} kPa**"
        )
        _s1, _s2, _s3 = st.columns(3)
        _s1.metric(
            "Path A  (C + A)",
            f"{_total_a:.3f} kPa",
            delta=f"{_p_in_c:.2f} → {_p_out_a:.4f} bara",
            delta_color="off",
            help="Total ΔP from header inlet through branch A to branch outlet",
        )
        _s2.metric(
            "Path B  (C + B)",
            f"{_total_b:.3f} kPa",
            delta=f"{_p_in_c:.2f} → {_p_out_b:.4f} bara",
            delta_color="off",
            help="Total ΔP from header inlet through branch B to branch outlet",
        )
        _s3.metric(
            f"Worst case  ({_worst_label})",
            f"{_worst_total:.3f} kPa",
            delta=f"{_p_in_c:.2f} → {_p_out_worst:.4f} bara",
            delta_color="off",
            help="Higher of the two total path ΔPs — governs required inlet pressure",
        )

    # ── Goal Seek — Required Inlet Pressure ──────────────────────────────────
    st.markdown("#### Goal Seek — Required C Inlet Pressure")
    with st.container(border=True):
        st.caption(
            "Set the **target outlet pressure** at the end of branches A and B. "
            "The solver iterates Case C → A and C → B with pressure marching until "
            "the worst-case branch outlet matches the target."
        )
        _gs_col1, _gs_col2 = st.columns([1, 1])
        with _gs_col1:
            _p_target = st.number_input(
                "Target outlet pressure (bara)",
                min_value=0.1,
                max_value=float(results_c["P_bara"]) * 2.0,
                value=float(round(min(ra["outlet_pressure_bara"],
                                      rb["outlet_pressure_bara"]), 2)),
                step=0.05,
                format="%.3f",
                key="gs_p_target",
                help="Desired minimum outlet pressure at the end of the worst-case branch (A or B).",
            )
            _gs_run = st.button(
                "Calculate Required Inlet",
                type="primary",
                use_container_width=True,
                key="gs_run",
            )

        if _gs_run:
            with st.spinner("Iterating…"):
                _gsr = _goal_seek_inlet(results_c, ra, rb, _p_target)
            st.session_state["gs_result"] = _gsr
            st.session_state["gs_target"] = _p_target

        _gsr = st.session_state.get("gs_result")
        if _gsr:
            _gs_target_shown = st.session_state.get("gs_target", _p_target)
            with _gs_col2:
                if _gsr["converged"]:
                    st.success(
                        f"Converged in {_gsr['iterations']} iteration(s).  "
                        f"Residual: {abs(_gsr['worst_out'] - _gs_target_shown)*1000:.2f} mbar"
                    )
                else:
                    st.warning(
                        f"Did not fully converge after {_gsr['iterations']} iterations — "
                        f"result is approximate (residual "
                        f"{abs(_gsr['worst_out'] - _gs_target_shown)*1000:.1f} mbar)"
                    )

            st.divider()
            _r1c1, _r1c2, _r1c3 = st.columns(3)
            _r1c1.metric(
                "Required C inlet pressure",
                f"{_gsr['P_c_in']:.4f} bara",
                help="Set this as Case C's inlet pressure to achieve the target.",
            )
            _r1c2.metric(
                "C outlet / branch inlet",
                f"{_gsr['P_c_out']:.4f} bara",
                delta=f"Header ΔP = {_gsr['dp_c']:.3f} kPa",
                delta_color="off",
            )
            _r1c3.metric(
                "Target outlet",
                f"{_gs_target_shown:.4f} bara",
            )
            _r2c1, _r2c2, _r2c3 = st.columns(3)
            _worse_branch = "A" if _gsr["P_a_out"] <= _gsr["P_b_out"] else "B"
            _r2c1.metric(
                "Branch A outlet",
                f"{_gsr['P_a_out']:.4f} bara",
                delta=f"ΔP = {_gsr['dp_a']:.3f} kPa",
                delta_color="off",
            )
            _r2c2.metric(
                "Branch B outlet",
                f"{_gsr['P_b_out']:.4f} bara",
                delta=f"ΔP = {_gsr['dp_b']:.3f} kPa",
                delta_color="off",
            )
            _r2c3.metric(
                f"Governing branch (worst = {_worse_branch})",
                f"{_gsr['worst_out']:.4f} bara",
                delta=f"vs target: {(_gsr['worst_out'] - _gs_target_shown)*1000:+.1f} mbar",
                delta_color="off",
            )
            st.info(
                f"**To achieve {_gs_target_shown:.3f} bara at the worst-case branch outlet**, "
                f"set **Case C inlet pressure to {_gsr['P_c_in']:.4f} bara**.  "
                f"Branch {'A' if _gsr['P_a_out'] > _gsr['P_b_out'] else 'B'} has "
                f"{abs(_gsr['P_a_out'] - _gsr['P_b_out'])*1000:.1f} mbar of pressure margin."
            )

    # ── Overlaid pressure profiles ─────────────────────────────────────────────
    st.markdown("#### Pressure Profiles")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=ra["pressure_profile_x"], y=ra["pressure_profile_y"],
        mode="lines+markers", name="Case A",
        line=dict(color="#2563EB", width=2.5), marker=dict(size=7, color="#2563EB"),
        hovertemplate="Case A  |  Distance: %{x:.2f} m<br>Pressure: %{y:.4f} bara<extra></extra>"))
    fig_cmp.add_trace(go.Scatter(
        x=rb["pressure_profile_x"], y=rb["pressure_profile_y"],
        mode="lines+markers", name="Case B",
        line=dict(color="#D97706", width=2.5, dash="dash"), marker=dict(size=7, color="#D97706"),
        hovertemplate="Case B  |  Distance: %{x:.2f} m<br>Pressure: %{y:.4f} bara<extra></extra>"))
    fig_cmp.update_layout(
        xaxis_title="Pipeline Distance (m)", yaxis_title="Pressure (bara)",
        template="plotly_white", height=360, margin=dict(l=60,r=20,t=30,b=50),
        hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        font=dict(size=12, color="#374151"))
    st.plotly_chart(fig_cmp, use_container_width=True, key="fig_cmp")

    # ── Per-segment ΔP comparison ──────────────────────────────────────────────
    st.markdown("#### Pressure Drop by Segment")
    _segs_a = [f"A-{r['Seg']} {r['Pipe']}" for r in ra["grid_records"]]
    _segs_b = [f"B-{r['Seg']} {r['Pipe']}" for r in rb["grid_records"]]
    _dp_a   = [r["ΔP (kPa)"] for r in ra["grid_records"]]
    _dp_b   = [r["ΔP (kPa)"] for r in rb["grid_records"]]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Case A", x=_segs_a, y=_dp_a,
                             marker_color="#2563EB", opacity=0.85))
    fig_bar.add_trace(go.Bar(name="Case B", x=_segs_b, y=_dp_b,
                             marker_color="#D97706", opacity=0.85))
    fig_bar.update_layout(
        barmode="group", yaxis_title="ΔP (kPa)", xaxis_title="Segment",
        template="plotly_white", height=320, margin=dict(l=60,r=20,t=20,b=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        font=dict(size=12, color="#374151"))
    st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")

    # ── Method Sensitivity Analysis ────────────────────────────────────────────
    st.markdown("#### Method Sensitivity Analysis")

    _sh = _sens_hash(ra, rb)
    if st.session_state.get("sens_hash") != _sh:
        st.session_state.pop("sens_a", None)
        st.session_state.pop("sens_b", None)
        st.session_state.pop("sens_hash", None)

    _sc1, _sc2 = st.columns([1, 3])
    with _sc1:
        _run_sens = st.button("Run Sensitivity Analysis", type="primary",
                              use_container_width=True, key="run_sens")
    with _sc2:
        if "sens_a" in st.session_state and "sens_b" in st.session_state:
            st.success("Results shown below — click again to refresh after input changes.")
        else:
            st.caption(
                "Runs all 12 combinations (6 correlations × 2 void-fraction models). "
                "Quantifies the full ΔP range for Case A and B due to method uncertainty.")

    if _run_sens:
        with st.spinner("Running 24 calculations (12 per case)…"):
            st.session_state["sens_a"] = engine.run_sensitivity(
                ra["P_bara"], ra["T_C"], ra["gas_flows_kgh"], ra["liquid_type"], ra["q_lye"],
                ra["segments"],
                custom_gas=ra.get("custom_gas"), custom_liquid=ra.get("custom_liquid"))
            st.session_state["sens_b"] = engine.run_sensitivity(
                rb["P_bara"], rb["T_C"], rb["gas_flows_kgh"], rb["liquid_type"], rb["q_lye"],
                rb["segments"],
                custom_gas=rb.get("custom_gas"), custom_liquid=rb.get("custom_liquid"))
            st.session_state["sens_hash"] = _sh
        st.rerun()

    fig_sens = None
    if "sens_a" in st.session_state and "sens_b" in st.session_state:
        _sa = st.session_state["sens_a"]
        _sb = st.session_state["sens_b"]

        _ylabels, _dp_a_vals, _dp_b_vals, _ok_a, _ok_b = [], [], [], [], []
        for _r_a, _r_b in zip(_sa, _sb):
            _c = _CORR_SHORT.get(_r_a["correlation"], _r_a["correlation"])
            _v = _VOID_SHORT.get(_r_a["voidage"], _r_a["voidage"])
            _ylabels.append(f"{_c} / {_v}")
            _dp_a_vals.append(_r_a["total_dp_kpa"] if _r_a["ok"] else None)
            _dp_b_vals.append(_r_b["total_dp_kpa"] if _r_b["ok"] else None)
            _ok_a.append(_r_a["ok"])
            _ok_b.append(_r_b["ok"])

        _va = [v for v in _dp_a_vals if v is not None]
        _vb = [v for v in _dp_b_vals if v is not None]

        fig_sens = go.Figure()

        # Shaded range bands
        if len(_va) >= 2:
            fig_sens.add_vrect(x0=min(_va), x1=max(_va),
                               fillcolor="rgba(37,99,235,0.07)", line_width=0,
                               annotation_text="A range", annotation_position="top left",
                               annotation_font=dict(size=9, color="#2563EB"))
        if len(_vb) >= 2:
            fig_sens.add_vrect(x0=min(_vb), x1=max(_vb),
                               fillcolor="rgba(217,119,6,0.07)", line_width=0,
                               annotation_text="B range", annotation_position="bottom right",
                               annotation_font=dict(size=9, color="#D97706"))

        # Connecting lines — one trace with None separators
        _conn_x, _conn_y = [], []
        for _ci in range(len(_ylabels)):
            if _ok_a[_ci] and _ok_b[_ci]:
                _conn_x.extend([_dp_a_vals[_ci], _dp_b_vals[_ci], None])
                _conn_y.extend([_ylabels[_ci],   _ylabels[_ci],   None])
        if _conn_x:
            fig_sens.add_trace(go.Scatter(
                x=_conn_x, y=_conn_y, mode="lines",
                line=dict(color="#CBD5E1", width=1.2),
                showlegend=False, hoverinfo="skip"))

        # Case B dots (amber diamonds, drawn first so A circles sit on top)
        _xb_p = [_dp_b_vals[_ci] for _ci in range(len(_ylabels)) if _ok_b[_ci]]
        _yb_p = [_ylabels[_ci]   for _ci in range(len(_ylabels)) if _ok_b[_ci]]
        fig_sens.add_trace(go.Scatter(
            x=_xb_p, y=_yb_p, mode="markers", name="Case B",
            marker=dict(color="#D97706", size=11, symbol="diamond",
                        line=dict(color="#92400E", width=1.5)),
            hovertemplate="Case B  |  %{y}<br>Total ΔP: %{x:.3f} kPa<extra></extra>"))

        # Case A dots (blue circles)
        _xa_p = [_dp_a_vals[_ci] for _ci in range(len(_ylabels)) if _ok_a[_ci]]
        _ya_p = [_ylabels[_ci]   for _ci in range(len(_ylabels)) if _ok_a[_ci]]
        fig_sens.add_trace(go.Scatter(
            x=_xa_p, y=_ya_p, mode="markers", name="Case A",
            marker=dict(color="#2563EB", size=11, symbol="circle",
                        line=dict(color="#1E40AF", width=1.5)),
            hovertemplate="Case A  |  %{y}<br>Total ΔP: %{x:.3f} kPa<extra></extra>"))

        # Dashed reference lines for the currently-selected method in each case
        _sel_lbl_a = (f"{_CORR_SHORT.get(ra['correlation'], ra['correlation'])} / "
                      f"{_VOID_SHORT.get(ra['voidage_method'], ra['voidage_method'])}")
        _sel_lbl_b = (f"{_CORR_SHORT.get(rb['correlation'], rb['correlation'])} / "
                      f"{_VOID_SHORT.get(rb['voidage_method'], rb['voidage_method'])}")
        _sel_a_dp = next(
            (r["total_dp_kpa"] for r in _sa if r["ok"]
             and f"{_CORR_SHORT.get(r['correlation'], r['correlation'])} / "
                 f"{_VOID_SHORT.get(r['voidage'], r['voidage'])}" == _sel_lbl_a), None)
        _sel_b_dp = next(
            (r["total_dp_kpa"] for r in _sb if r["ok"]
             and f"{_CORR_SHORT.get(r['correlation'], r['correlation'])} / "
                 f"{_VOID_SHORT.get(r['voidage'], r['voidage'])}" == _sel_lbl_b), None)
        if _sel_a_dp is not None:
            fig_sens.add_vline(x=_sel_a_dp,
                               line=dict(color="#2563EB", width=1.5, dash="dash"),
                               annotation_text=f"A selected: {_sel_a_dp:.2f} kPa",
                               annotation_position="top right",
                               annotation_font=dict(size=9, color="#2563EB"))
        if _sel_b_dp is not None:
            fig_sens.add_vline(x=_sel_b_dp,
                               line=dict(color="#D97706", width=1.5, dash="dot"),
                               annotation_text=f"B selected: {_sel_b_dp:.2f} kPa",
                               annotation_position="bottom right",
                               annotation_font=dict(size=9, color="#D97706"))

        fig_sens.update_layout(
            title=dict(text="Total ΔP — all 12 method combinations  (● Case A  ◆ Case B)",
                       font=dict(size=12), x=0),
            xaxis_title="Total ΔP (kPa)", yaxis_title=None,
            template="plotly_white", height=460,
            margin=dict(l=155, r=40, t=55, b=50),
            hovermode="closest", paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                       categoryorder="array",
                       categoryarray=list(reversed(_ylabels))),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                        borderwidth=1, orientation="h", y=1.10, x=0),
            font=dict(size=11, color="#374151"))
        st.plotly_chart(fig_sens, use_container_width=True, key="fig_sens")

        _n_failed = sum(1 for ok in _ok_a + _ok_b if not ok)
        if _n_failed > 0:
            st.caption(f"{_n_failed} combination(s) failed to converge and are excluded from the chart.")

        # Summary table
        if _va and _vb:
            _a_min, _a_max = min(_va), max(_va)
            _b_min, _b_max = min(_vb), max(_vb)
            _overlap = _a_min <= _b_max and _b_min <= _a_max
            st.dataframe(pd.DataFrame([
                {"": "Case A",
                 "Min (kPa)":      f"{_a_min:.3f}",
                 "Selected (kPa)": f"{ra['total_dp_kpa']:.3f}",
                 "Max (kPa)":      f"{_a_max:.3f}",
                 "Spread (kPa)":   f"{_a_max - _a_min:.3f}"},
                {"": "Case B",
                 "Min (kPa)":      f"{_b_min:.3f}",
                 "Selected (kPa)": f"{rb['total_dp_kpa']:.3f}",
                 "Max (kPa)":      f"{_b_max:.3f}",
                 "Spread (kPa)":   f"{_b_max - _b_min:.3f}"},
            ]), hide_index=True, use_container_width=True)
            if _overlap:
                st.warning(
                    "Ranges **overlap** — the relative ordering of Case A vs B depends on "
                    "which correlation is chosen.")
            else:
                st.success(
                    "Ranges **do not overlap** — one case is unambiguously lower-ΔP "
                    "across all methods.")

        # ── Flow Regime Consistency ────────────────────────────────────────────
        st.markdown("**Flow Regime Consistency across Methods**")
        st.caption(
            "Regime is independent of ΔP correlation — Vsg, Vsl, and angle are fixed. "
            "Only the void fraction model (α) can shift vertical-segment thresholds "
            "(bubble/slug/churn). Each column header shows the first correlation that "
            "produced that regime; all other correlations with the same void model agree.")

        def _regime_table(sens_results, segments_list, label):
            """Build regime DataFrame for one case across all converged combinations."""
            ok_results = [r for r in sens_results if r["ok"] and r["segment_regimes"]]
            if not ok_results or not segments_list:
                return None

            n_segs = len(segments_list)
            # For each segment, collect the set of unique regimes across all combinations
            rows = []
            for i, seg in enumerate(segments_list):
                seg_id   = f"#{i+1}"
                orient   = seg["type"].replace("Vertical Upflow", "V Up").replace(
                               "Vertical Downflow", "V Down").replace("Horizontal", "Horiz")
                dn_str   = f"{seg['dn']}/{seg['pn']}"
                # Group regimes by void fraction model (correlation doesn't change regime)
                by_void = {}
                for r in ok_results:
                    v = _VOID_SHORT.get(r["voidage"], r["voidage"])
                    regime_str = r["segment_regimes"][i] if i < len(r["segment_regimes"]) else "—"
                    by_void.setdefault(v, set()).add(regime_str)
                # Build row: one column per void model, showing unique regimes found
                row = {"Seg": seg_id, "Pipe": dn_str, "Orient": orient}
                for v_short, regime_set in by_void.items():
                    row[v_short] = " / ".join(sorted(regime_set)) if regime_set else "—"
                # Consensus across ALL combinations
                all_regimes = set()
                for r in ok_results:
                    if i < len(r["segment_regimes"]):
                        all_regimes.add(r["segment_regimes"][i])
                row["Unanimous"] = "✓" if len(all_regimes) == 1 else f"✗  ({len(all_regimes)} distinct)"
                rows.append(row)
            return pd.DataFrame(rows)

        _rt_a = _regime_table(_sa, ra["segments"], "Case A")
        _rt_b = _regime_table(_sb, rb["segments"], "Case B")

        _rca, _rcb = st.columns(2)
        with _rca:
            st.markdown("**Case A**")
            if _rt_a is not None:
                st.dataframe(_rt_a, hide_index=True, use_container_width=True)
            else:
                st.caption("No regime data available.")
        with _rcb:
            st.markdown("**Case B**")
            if _rt_b is not None:
                st.dataframe(_rt_b, hide_index=True, use_container_width=True)
            else:
                st.caption("No regime data available.")

    # ── Comparison report export ──────────────────────────────────────────────
    st.divider()
    _cr1, _cr2 = st.columns([1, 2])
    with _cr1:
        if st.button("Generate Comparison Report", type="primary",
                     use_container_width=True, key="gen_cmp_rpt"):
            with st.spinner("Building comparison document…"):
                try:
                    _sens_report_data = None
                    if "sens_a" in st.session_state and "sens_b" in st.session_state and fig_sens is not None:
                        _sens_report_data = {
                            "sa":  st.session_state["sens_a"],
                            "sb":  st.session_state["sens_b"],
                            "fig": fig_sens,
                        }
                    _cbuf = report_generator.generate_comparison_report(
                        results_a=ra, results_b=rb,
                        fig_cmp=fig_cmp, fig_bar=fig_bar,
                        sensitivity_data=_sens_report_data)
                    st.session_state["cmp_rpt_bytes"] = _cbuf.getvalue()
                except Exception as _ce:
                    st.error(f"Report failed: {_ce}")
    with _cr2:
        if st.session_state.get("cmp_rpt_bytes"):
            st.download_button(
                "Download Comparison Report  (.docx)",
                data=st.session_state["cmp_rpt_bytes"],
                file_name="hydraulic_comparison_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_cmp_rpt")

    # ── Combined report (A + B + C + sensitivity in one document) ─────────────
    st.markdown("#### Combined Report — A + B + C")
    st.caption(
        "Single document: one Method section, side-by-side conditions/totals for all three cases, "
        "individual segment analyses, comparison charts, and sensitivity (if run)."
    )
    _combined_c1, _combined_c2 = st.columns([1, 2])
    with _combined_c1:
        if st.button("Generate Combined Report", type="primary",
                     use_container_width=True, key="gen_combined_rpt"):
            with st.spinner("Building combined report (A + B + C)…"):
                try:
                    _sens_data = None
                    if "sens_a" in st.session_state and "sens_b" in st.session_state and fig_sens is not None:
                        _sens_data = {
                            "sa":  st.session_state["sens_a"],
                            "sb":  st.session_state["sens_b"],
                            "fig": fig_sens,
                        }
                    _combined_buf = report_generator.generate_combined_report(
                        cases=[ra, rb, results_c],
                        case_labels=["Case A", "Case B", "Case C"],
                        fig_cmp=fig_cmp,
                        fig_bar=fig_bar,
                        sensitivity_data=_sens_data,
                    )
                    st.session_state["combined_rpt_bytes"] = _combined_buf.getvalue()
                except Exception as _combined_err:
                    st.error(f"Combined report failed: {_combined_err}")
    with _combined_c2:
        if st.session_state.get("combined_rpt_bytes"):
            st.download_button(
                "Download Combined Report  (.docx)",
                data=st.session_state["combined_rpt_bytes"],
                file_name="hydraulic_combined_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_combined_rpt")

    st.divider()

    # ── Full segment tables side by side ──────────────────────────────────────
    st.markdown("#### Segment Detail")
    _col_cfg = {
        "ID (mm)":          st.column_config.NumberColumn(format="%.1f"),
        "P_in (bara)":      st.column_config.NumberColumn(format="%.4f"),
        "P_out (bara)":     st.column_config.NumberColumn(format="%.4f"),
        "α (void)":         st.column_config.NumberColumn(format="%.4f"),
        "ΔP_fric (kPa)":    st.column_config.NumberColumn(format="%.3f"),
        "ΔP_grav (kPa)":    st.column_config.NumberColumn(format="%.3f"),
        "ΔP (kPa)":         st.column_config.NumberColumn(format="%.3f"),
        "V_m (m/s)":        st.column_config.NumberColumn(format="%.3f"),
        "V_m/V_e":          st.column_config.NumberColumn(format="%.3f"),
    }
    _cmp_cols = ["Seg","Pipe","ID (mm)","Type","L (m)","Regime",
                 "α (void)","V_m (m/s)","V_m/V_e",
                 "ΔP_fric (kPa)","ΔP_grav (kPa)","ΔP (kPa)",
                 "P_in (bara)","P_out (bara)"]
    _ta, _tb = st.columns(2)
    with _ta:
        st.markdown("**Case A**")
        st.dataframe(pd.DataFrame(ra["grid_records"])[_cmp_cols],
                     column_config=_col_cfg, hide_index=True, use_container_width=True)
    with _tb:
        st.markdown("**Case B**")
        st.dataframe(pd.DataFrame(rb["grid_records"])[_cmp_cols],
                     column_config=_col_cfg, hide_index=True, use_container_width=True)
