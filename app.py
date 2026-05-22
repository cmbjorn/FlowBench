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
    page_title="Multiphase Pressure Drop Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
[data-testid="stMetricDelta"] { font-size: 0.80rem !important; }

/* ── Section labels inside containers (**Bold** pattern) ─── */
[data-testid="stMarkdownContainer"] p strong {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}

/* ── Body text and captions ───────────────────────────────── */
[data-testid="stMarkdownContainer"] p {
    font-size: 0.92rem;
    line-height: 1.55;
    color: #1E293B;
}
[data-testid="stCaptionContainer"] p {
    font-size: 0.82rem !important;
    color: #64748B !important;
    line-height: 1.5 !important;
}

/* ── Subheaders ───────────────────────────────────────────── */
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    letter-spacing: -0.01em !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr { margin: 0.75rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Multiphase Pressure Drop Calculator")
st.caption("Two-phase pressure drop · H₂ · O₂ · KOH · Six correlations · Steady-state")

with st.expander("About this calculator", expanded=False):
    st.markdown("""
    Sizes the **hydrogen and oxygen gas pipelines** from an alkaline or PEM electrolyzer to the gas–liquid separator. Each pipeline carries a two-phase mixture of gas (H₂ or O₂) entrained in electrolyte (KOH 30 wt%, KOH 15 wt%, Water, or custom) at elevated pressure.

    **Typical workflow**
    1. **Hydrogen pipe** tab — set inlet pressure, temperature, H₂ mass flow, and electrolyte flow; build the pipe geometry segment by segment (DN, PN, material, length, fittings, optional FEP/PTFE/PFA liner).
    2. **Oxygen pipe** tab — same for the O₂ side (gas is heavier, ΔP differs).
    3. **H₂ Header / O₂ Header** tabs — configure the collecting manifold (tap positions, header pipe, T-segment) for each gas.
    4. **Generator ΔP** tab — goal-seek both systems simultaneously; the difference in branch inlet pressures is the **differential pressure across the electrolyzer stack**.
    5. **Compare** tab — overlay pressure profiles and export a combined Word or Excel report.

    **Method** — Beggs & Brill (1973) default, with Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, and Kim-Mudawar for cross-checking. Pressure marching: gas density updated at every segment inlet. ΔP split into frictional, gravitational, and accelerational components. Flow regime classified automatically (Taitel-Dukler + Mandhane for horizontal; Wallis/Taitel for vertical). Erosion check: API RP 14E, C = 100.

    **Accuracy note** — Correlations were developed for oil/gas systems. Uncertainty for H₂/O₂ over KOH is ±20–30 %. Use for design guidance and relative comparison; validate against commissioning data.
    """)

# ============================================================================
# SAVE / LOAD  (defined here so sidebar can call them on first render)
# ============================================================================
def _collect_save_state() -> dict:
    s = st.session_state
    data: dict = {
        "version": 1,
        "label_a": s.get("label_a", "Case A"),
        "label_b": s.get("label_b", "Case B"),
    }
    for cid in ("a", "b"):
        gas_flows = {
            k[len(f"{cid}_gflow_"):]: v
            for k, v in s.items()
            if k.startswith(f"{cid}_gflow_")
        }
        data[cid] = {
            "segments":           s.get(f"{cid}_segments", []),
            "P_bara":             s.get(f"{cid}_P_bara", 20.0),
            "T_C":                s.get(f"{cid}_T_C", 60.0),
            "gas_species_widget": s.get(f"{cid}_gas_species_widget", ["H₂"]),
            "gas_flows":          gas_flows,
            "liquid_type_widget": s.get(f"{cid}_liquid_type_widget", "Water"),
            "q_lye_widget":       s.get(f"{cid}_q_lye_widget", 1.0),
            "correlation":        s.get(f"{cid}_correlation", "Beggs-Brill"),
            "voidage_method":     s.get(f"{cid}_voidage_method", "Homogeneous"),
            "cg_mw":              s.get(f"{cid}_cg_mw"),
            "cg_mu":              s.get(f"{cid}_cg_mu"),
            "cl_rho":             s.get(f"{cid}_cl_rho"),
            "cl_mu":              s.get(f"{cid}_cl_mu"),
            "cl_sigma":           s.get(f"{cid}_cl_sigma"),
        }
    for cid in ("c", "d"):
        n_left  = int(s.get(f"{cid}_n_left",  3))
        n_right = int(s.get(f"{cid}_n_right", 3))
        data[cid] = {
            "P_target_sep":    s.get(f"{cid}_P_target_sep", 16.5),
            "T_C":             s.get(f"{cid}_T_C", 60.0),
            "hdr_dn":          s.get(f"{cid}_hdr_dn", "DN100"),
            "hdr_pn":          s.get(f"{cid}_hdr_pn", "PN40"),
            "hdr_mat":         s.get(f"{cid}_hdr_mat", "SS316L"),
            "hdr_lined":       s.get(f"{cid}_hdr_lined", False),
            "hdr_lmat":        s.get(f"{cid}_hdr_lmat", "FEP"),
            "hdr_lthk":        s.get(f"{cid}_hdr_lthk", 1.0),
            "hdr_fits":        s.get(f"{cid}_hdr_fits", []),
            "n_left":          n_left,
            "n_right":         n_right,
            "left_positions":  [s.get(f"{cid}_pos_left_{i}",  float((i+1)*2.5)) for i in range(n_left)],
            "right_positions": [s.get(f"{cid}_pos_right_{i}", float((i+1)*2.5)) for i in range(n_right)],
            "t_dn":            s.get(f"{cid}_t_dn", "DN150"),
            "t_pn":            s.get(f"{cid}_t_pn", "PN40"),
            "t_mat":           s.get(f"{cid}_t_mat", "SS316L"),
            "t_len":           s.get(f"{cid}_t_len", 1.0),
            "correlation":     s.get(f"{cid}_correlation", "Beggs-Brill"),
            "voidage_method":  s.get(f"{cid}_voidage_method", "Homogeneous"),
        }
    return data


def _apply_save_state(data: dict) -> None:
    import copy
    s = st.session_state
    s["label_a"] = data.get("label_a", "Case A")
    s["label_b"] = data.get("label_b", "Case B")

    for cid in ("a", "b"):
        cd = data.get(cid, {})
        if not cd:
            continue
        if "segments" in cd:
            s[f"{cid}_segments"] = copy.deepcopy(cd["segments"])
        for key in ("P_bara", "T_C", "gas_species_widget", "liquid_type_widget",
                    "q_lye_widget", "correlation", "voidage_method",
                    "cg_mw", "cg_mu", "cl_rho", "cl_mu", "cl_sigma"):
            if cd.get(key) is not None:
                s[f"{cid}_{key}"] = cd[key]
        for sp, flow in cd.get("gas_flows", {}).items():
            s[f"{cid}_gflow_{sp}"] = flow

    for cid in ("c", "d"):
        cd = data.get(cid, {})
        if not cd:
            continue
        for key in ("P_target_sep", "T_C", "hdr_dn", "hdr_pn", "hdr_mat",
                    "hdr_lined", "hdr_lmat", "hdr_lthk", "hdr_fits",
                    "n_left", "n_right", "t_dn", "t_pn", "t_mat", "t_len",
                    "correlation", "voidage_method"):
            if cd.get(key) is not None:
                s[f"{cid}_{key}"] = cd[key]
        for i, pos in enumerate(cd.get("left_positions", [])):
            s[f"{cid}_pos_left_{i}"] = float(pos)
        for i, pos in enumerate(cd.get("right_positions", [])):
            s[f"{cid}_pos_right_{i}"] = float(pos)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Documentation")
    with st.expander("Capabilities", expanded=False):
        st.markdown("""
        **Cases** — Run the Hydrogen pipe and Oxygen pipe independently, then compare side by side.
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
                    _sb_le  = engine._seg_le_fit(_sb_seg, _sb_D)
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

    st.divider()
    st.header("Session")

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_json = json.dumps(_collect_save_state(), indent=2, ensure_ascii=False)
    st.download_button(
        "Save session (.json)",
        data=_save_json,
        file_name="hydraulic_session.json",
        mime="application/json",
        use_container_width=True,
        help="Download all current inputs as a JSON file you can reload later.",
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    _uploaded = st.file_uploader(
        "Load session (.json)",
        type="json",
        label_visibility="collapsed",
        help="Upload a previously saved session JSON to restore all inputs.",
        key="session_uploader",
    )
    if _uploaded is not None:
        try:
            _loaded = json.loads(_uploaded.read())
            if _loaded.get("version") != 1:
                st.warning("Unrecognised file format — not loaded.")
            else:
                _apply_save_state(_loaded)
                st.success("Session loaded.")
                st.rerun()
        except Exception as _e:
            st.error(f"Could not load session: {_e}")

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
    {"type": "Horizontal",      "dn": "DN50", "pn": "PN20", "material": "SS316L",
     "length": 3.0,
     "fittings_list": [{"type": "90° Standard Elbow", "qty": 2}],
     "lined": True, "liner_material": "FEP", "liner_thickness_mm": 1.5},
    {"type": "Vertical Upflow", "dn": "DN50", "pn": "PN20", "material": "SS316L",
     "length": 5.0,
     "fittings_list": [],
     "lined": True, "liner_material": "FEP", "liner_thickness_mm": 1.5},
    {"type": "Horizontal",      "dn": "DN50", "pn": "PN20", "material": "SS316L",
     "length": 2.0,
     "fittings_list": [{"type": "90° Standard Elbow", "qty": 2},
                       {"type": "Expansion — Sudden",  "qty": 1}],
     "lined": True, "liner_material": "FEP", "liner_thickness_mm": 1.5},
]

_VALID_MATS   = set(engine.MATERIAL_ROUGHNESS.keys())
_VALID_LINERS = set(engine.LINER_ROUGHNESS.keys())


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
def _sum_le_fit(seg, D_eff):
    """Sum equivalent pipe length from all fittings. Handles old and new segment format."""
    fl = seg.get("fittings_list")
    if fl is not None:
        total = 0.0
        for fit in fl:
            t = fit.get("type", "")
            q = fit.get("qty", 0)
            if t in engine.FITTING_Le_over_D and q > 0:
                total += engine.FITTING_Le_over_D[t] * D_eff * q
        return total
    f = seg.get("fittings", "None")
    c = seg.get("fitting_count", 0)
    if f in engine.FITTING_Le_over_D and c > 0:
        return engine.FITTING_Le_over_D[f] * D_eff * c
    return 0.0


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

                g5, g8 = st.columns([3.5, 0.65])
                _mat_def = seg.get("material","SS316L")
                mat = g5.selectbox("Material", MAT_OPTIONS, key=k(f"m_{i}"),
                                   index=MAT_OPTIONS.index(_mat_def) if _mat_def in MAT_OPTIONS else 0)
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

                # ── Multi-row fittings ───────────────────────────────────────────
                _fk = k(f"fits_{i}")
                if _fk not in st.session_state:
                    _init_fl = seg.get("fittings_list")
                    if _init_fl is None:
                        _old_f = seg.get("fittings", "None")
                        _old_c = seg.get("fitting_count", 0)
                        _init_fl = ([{"type": _old_f, "qty": int(_old_c)}]
                                    if _old_f in engine.FITTING_Le_over_D and _old_c > 0
                                    else [])
                    st.session_state[_fk] = list(_init_fl)

                _fit_rows = st.session_state[_fk]
                _n_fits   = len(_fit_rows)
                _all_fit_types = list(engine.FITTING_Le_over_D.keys())

                _fl_col, _fa_col = st.columns([4, 1])
                _fl_col.caption("**Minor losses**" if _n_fits else "Minor losses — none")
                if _fa_col.button("+ Add", key=k(f"fadd_{i}"), help="Add a fitting type"):
                    for jj in range(_n_fits):
                        _ft = st.session_state.get(k(f"ftype_{i}_{jj}"))
                        _fq = st.session_state.get(k(f"fqty_{i}_{jj}"), 1)
                        if _ft:
                            st.session_state[_fk][jj]["type"] = _ft
                            st.session_state[_fk][jj]["qty"]  = int(_fq)
                    st.session_state[_fk].append({"type": _all_fit_types[0], "qty": 1})
                    st.rerun()

                for j, _fr in enumerate(list(_fit_rows)):
                    _fc1, _fc2, _fc3 = st.columns([3.5, 0.7, 0.3])
                    _ftype_key = k(f"ftype_{i}_{j}")
                    _fqty_key  = k(f"fqty_{i}_{j}")
                    _cur_type  = _fit_rows[j]["type"]
                    _cur_qty   = _fit_rows[j]["qty"]
                    _fc1.selectbox("Fitting", _all_fit_types,
                        index=_all_fit_types.index(_cur_type) if _cur_type in _all_fit_types else 0,
                        key=_ftype_key, label_visibility="collapsed")
                    _fc2.number_input("Qty", min_value=1, value=_cur_qty,
                        key=_fqty_key, label_visibility="collapsed")
                    if _fc3.button("×", key=k(f"frem_{i}_{j}"), help="Remove this fitting"):
                        for jj in range(_n_fits):
                            _ft = st.session_state.get(k(f"ftype_{i}_{jj}"))
                            _fq = st.session_state.get(k(f"fqty_{i}_{jj}"), 1)
                            if _ft:
                                st.session_state[_fk][jj]["type"] = _ft
                                st.session_state[_fk][jj]["qty"]  = int(_fq)
                        st.session_state[_fk].pop(j)
                        for jj in range(j, _n_fits - 1):
                            st.session_state[k(f"ftype_{i}_{jj}")] = st.session_state[_fk][jj]["type"]
                            st.session_state[k(f"fqty_{i}_{jj}")] = st.session_state[_fk][jj]["qty"]
                        st.session_state.pop(k(f"ftype_{i}_{_n_fits-1}"), None)
                        st.session_state.pop(k(f"fqty_{i}_{_n_fits-1}"), None)
                        st.rerun()

                _fittings_list = []
                for jj in range(_n_fits):
                    _ft = st.session_state.get(k(f"ftype_{i}_{jj}"))
                    _fq = st.session_state.get(k(f"fqty_{i}_{jj}"), 1)
                    if _ft and _ft in engine.FITTING_Le_over_D and int(_fq) > 0:
                        _fittings_list.append({"type": _ft, "qty": int(_fq)})

                current_specs.append({
                    "type": t, "dn": dn, "pn": pn, "material": mat, "length": l,
                    "fittings_list": _fittings_list,
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
                    "length": 2.0, "fittings_list": [],
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
            le_fit = _sum_le_fit(seg, D_eff)
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
                "L_eq (m)":        round(le_fit, 3),
                "Fittings":        (", ".join(f"{f['type']} ×{f['qty']}"
                                              for f in seg.get("fittings_list", [])
                                              if f.get("qty", 0) > 0)
                                    or "—"),
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

    _DN_LW = {"DN20":2,"DN25":3,"DN40":5,"DN50":7,"DN80":9,"DN100":11,"DN150":15,"DN200":19,"DN250":23}

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
            "segs": [(s["type"],s["dn"],s["pn"],s["length"],
                      tuple(sorted((f["type"],f["qty"]) for f in s.get("fittings_list",[]))),
                      s.get("lined",False),
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
                        fig_sch=fig_sch, fig_prof=fig_prof,
                        case_label=st.session_state.get(
                            f"label_{cid}", f"Case {cid.upper()}"))
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
        le_fit = _sum_le_fit(seg, D_eff)
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)
        total_dp  += seg_res["dP_Pa"]
        current_P -= seg_res["dP_Pa"]
        current_P  = max(1e4, current_P)   # guard: never pass sub-vacuum to thermodynamics
    dp_kpa      = total_dp / 1000.0
    outlet_bara = current_P / 1e5
    return dp_kpa, outlet_bara


def _march_header_simple(tap_dists, gas_per_tap, liq_per_tap,
                          hdr, P_start_Pa, T_C, liquid_type, corr, void):
    """Pressure-march one header arm from farthest tap to T-junction.

    tap_dists   : list of distances from T (m) for each tap — sorted internally.
    gas_per_tap : {species: kg/h} for ONE A-line.
    liq_per_tap : m³/h liquid for ONE A-line.
    hdr         : dict with dn, pn, material, lined, liner_material,
                  liner_thickness_mm, fittings_list.
    Returns (total_dp_Pa, P_T_Pa, dp_fric_Pa, dp_grav_Pa, records).
    """
    if not tap_dists or not gas_per_tap or liq_per_tap <= 0:
        return 0.0, P_start_Pa, 0.0, 0.0, []

    dists = sorted(tap_dists, reverse=True)   # farthest first
    n     = len(dists)
    boundaries = dists + [0.0]                # segment endpoints, T at 0

    D_nom  = engine.PIPE_DATABASE[hdr["dn"]][hdr["pn"]]
    lined  = hdr.get("lined", False)
    lthk_m = hdr.get("liner_thickness_mm", 1.0) / 1000.0
    lmat   = hdr.get("liner_material", "FEP")
    D_eff  = D_nom - 2 * lthk_m if lined else D_nom
    rough  = (engine.LINER_ROUGHNESS[lmat] if lined
              else engine.MATERIAL_ROUGHNESS[hdr.get("material", "SS316L")])
    le_fit = _sum_le_fit(hdr, D_eff)

    current_P   = P_start_Pa
    total_dp    = dp_fric = dp_grav = 0.0
    running_gas = {}
    running_liq = 0.0
    records     = []

    for i in range(n):
        for sp, kgh in gas_per_tap.items():
            running_gas[sp] = running_gas.get(sp, 0.0) + kgh
        running_liq += liq_per_tap

        seg_len = boundaries[i] - boundaries[i + 1]
        if seg_len <= 1e-6:
            continue

        eff_len = seg_len + le_fit / n    # distribute fittings equally
        props_seg = engine.calculate_two_phase_properties(
            current_P / 1e5, T_C, running_gas, liquid_type, running_liq)
        seg_res = engine.calculate_segment_pressure_drop(
            props_seg, D_eff, rough, eff_len, 0.0,
            correlation=corr, voidage_method=void)

        dP_Pa = seg_res["dP_Pa"]
        end_P = current_P - dP_Pa
        V_m   = seg_res["Vsg"] + seg_res["Vsl"]
        V_e, _ = engine.calculate_erosion_velocity(
            props_seg["rho_g"], props_seg["rho_l"], props_seg["x_gas"])

        records.append({
            "Seg":          f"#{i+1}",
            "Taps in seg":  i + 1,
            "From T (m)":   round(boundaries[i], 2),
            "To T (m)":     round(boundaries[i + 1], 2),
            "L (m)":        round(seg_len, 3),
            "Pipe":         f"{hdr['dn']}/{hdr['pn']}",
            "ID (mm)":      round(D_eff * 1000, 1),
            "Regime":       seg_res["regime"],
            "ΔP (kPa)":     round(dP_Pa / 1000, 3),
            "P_in (bara)":  round(current_P / 1e5, 4),
            "P_out (bara)": round(end_P / 1e5, 4),
            "V_m (m/s)":    round(V_m, 3),
            "V_m/V_e":      round(V_m / V_e if V_e > 0 else 0.0, 3),
            "Q_gas_kgh":    round(sum(running_gas.values()), 3),
            "Q_liq_m3h":    round(running_liq, 3),
            "ΔP_fric (kPa)": round(seg_res["dP_fric_Pa"] / 1000, 3),
            "ΔP_grav (kPa)": round(seg_res["dP_grav_Pa"] / 1000, 3),
        })

        total_dp  += dP_Pa
        dp_fric   += seg_res["dP_fric_Pa"]
        dp_grav   += seg_res["dP_grav_Pa"]
        current_P  = max(1e4, end_P)

    return total_dp, current_P, dp_fric, dp_grav, records


def _march_single_seg(seg, P_in_Pa, T_C, gas_flows, liquid_type, q_lye, corr, void):
    """ΔP for one horizontal pipe segment carrying combined flow.
    Returns (dp_Pa, P_out_Pa, dp_fric_Pa, dp_grav_Pa, record_dict).
    """
    D_nom = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
    mat   = seg.get("material", "SS316L")
    rough = engine.MATERIAL_ROUGHNESS.get(mat, engine.MATERIAL_ROUGHNESS["SS316L"])
    le_fit = _sum_le_fit(seg, D_nom)

    props   = engine.calculate_two_phase_properties(
        P_in_Pa / 1e5, T_C, gas_flows, liquid_type, q_lye)
    seg_res = engine.calculate_segment_pressure_drop(
        props, D_nom, rough, seg["length"] + le_fit, 0.0,
        correlation=corr, voidage_method=void)

    dp_Pa = seg_res["dP_Pa"]
    P_out = max(1e4, P_in_Pa - dp_Pa)
    V_m   = seg_res["Vsg"] + seg_res["Vsl"]
    V_e, _ = engine.calculate_erosion_velocity(
        props["rho_g"], props["rho_l"], props["x_gas"])

    record = {
        "Seg": "T-seg",
        "Pipe": f"{seg['dn']}/{seg['pn']}",
        "ID (mm)": round(D_nom * 1000, 1),
        "L (m)": seg["length"],
        "Regime": seg_res["regime"],
        "ΔP (kPa)": round(dp_Pa / 1000, 3),
        "P_in (bara)": round(P_in_Pa / 1e5, 4),
        "P_out (bara)": round(P_out / 1e5, 4),
        "V_m (m/s)": round(V_m, 3),
        "V_m/V_e": round(V_m / V_e if V_e > 0 else 0.0, 3),
        "Q_gas_kgh": round(sum(gas_flows.values()), 3),
        "Q_liq_m3h": round(q_lye, 3),
        "ΔP_fric (kPa)": round(seg_res["dP_fric_Pa"] / 1000, 3),
        "ΔP_grav (kPa)": round(seg_res["dP_grav_Pa"] / 1000, 3),
    }
    return dp_Pa, P_out, seg_res["dP_fric_Pa"], seg_res["dP_grav_Pa"], record


def _calc_header_dp_at_p(res_hdr, P_in_bara):
    """Re-run header (both arms) + T-segment at an overridden tap inlet pressure.
    Returns (total_dp_kpa, P_separator_bara).
    """
    P_start = P_in_bara * 1e5
    hdr  = res_hdr["header_pipe"]
    gpt  = res_hdr["gas_per_tap"]
    lpt  = res_hdr["liq_per_tap"]
    T_C  = res_hdr["T_C"]
    liq  = res_hdr["liquid_type"]
    corr = res_hdr.get("correlation",    engine.TWO_PHASE_CORRELATIONS[0])
    void = res_hdr.get("voidage_method", engine.VOIDAGE_METHODS[0])

    dp_l, P_T_l, *_ = _march_header_simple(
        res_hdr["left_taps"],  gpt, lpt, hdr, P_start, T_C, liq, corr, void)
    dp_r, P_T_r, *_ = _march_header_simple(
        res_hdr["right_taps"], gpt, lpt, hdr, P_start, T_C, liq, corr, void)

    dp_worst = max(dp_l, dp_r)
    P_T      = min(P_T_l, P_T_r)      # worst-arm pressure arriving at T

    t_seg = res_hdr.get("t_seg")
    if t_seg and t_seg.get("length", 0) > 0:
        dp_t, P_sep, *_ = _march_single_seg(
            t_seg, P_T, T_C, res_hdr["gas_flows_kgh"], liq,
            res_hdr["q_lye"], corr, void)
        return (dp_worst + dp_t) / 1000.0, P_sep / 1e5

    return dp_worst / 1000.0, P_T / 1e5


def _apply_dn_override(res, dn_alt):
    """Return a deep copy of a branch result dict with dn replaced in all segments.
    Header pipe and T-segment are not modified."""
    import copy
    r = copy.deepcopy(res)
    for seg in r.get("segments", []):
        seg["dn"] = dn_alt
    return r


def _goal_seek_header(res_hdr, P_target_sep, tol=0.0005, max_iter=25):
    """Find the header inlet pressure (= line outlet) to achieve P_target_sep
    at the separator (end of T-segment).  Returns a result dict.
    """
    P_in = P_target_sep + res_hdr["total_dp_kpa"] / 100.0
    dp = P_sep = 0.0
    for i in range(max_iter):
        dp, P_sep = _calc_header_dp_at_p(res_hdr, P_in)
        error = P_sep - P_target_sep
        if abs(error) < tol:
            return {"P_hdr_in": P_in, "P_sep": P_sep, "dp_hdr": dp,
                    "iterations": i + 1, "converged": True}
        P_in -= error
    return {"P_hdr_in": P_in, "P_sep": P_sep, "dp_hdr": dp,
            "iterations": max_iter, "converged": abs(P_sep - P_target_sep) < tol * 20}


def _goal_seek_stack(res_line, res_hdr, P_target_sep, tol=0.0005, max_iter=30):
    """Find the line inlet pressure (= stack outlet) to achieve P_target_sep
    at the separator.  Flow: line → header arms → T-segment → separator.
    Returns a result dict.
    """
    P_line_in = (P_target_sep
                 + res_line["total_dp_kpa"] / 100.0
                 + res_hdr["total_dp_kpa"]  / 100.0)

    dp_line = dp_hdr = 0.0
    P_line_out = P_sep = 0.0

    for i in range(max_iter):
        dp_line, P_line_out = _calc_dp_at_p(res_line, P_line_in)
        dp_hdr,  P_sep      = _calc_header_dp_at_p(res_hdr, P_line_out)
        error = P_sep - P_target_sep
        if abs(error) < tol:
            return {
                "P_line_in":  P_line_in,  "P_line_out": P_line_out,
                "P_sep":      P_sep,
                "dp_line":    dp_line,    "dp_hdr":     dp_hdr,
                "iterations": i + 1,      "converged":  True,
            }
        P_line_in -= error

    return {
        "P_line_in":  P_line_in,  "P_line_out": P_line_out,
        "P_sep":      P_sep,
        "dp_line":    dp_line,    "dp_hdr":     dp_hdr,
        "iterations": max_iter,
        "converged":  abs(P_sep - P_target_sep) < tol * 20,
    }


# ============================================================================
# HEADER PIPING SCHEMATIC
# ============================================================================
def _make_header_schematic(
        left_positions, right_positions, t_seg_spec,
        P_inlet_bara, worst_arm,
        dp_l_kpa, dp_r_kpa,
        P_T_l_bara, P_T_r_bara, P_sep_bara):
    """Return a Plotly figure showing the physical header piping layout."""

    PIPE_Y  = 1.8   # header pipe y-level
    TAP_TOP = 3.1   # top of tap risers (flow enters from branch above)
    TSEG_Y  = 0.5   # T-segment / separator level

    max_l  = max(left_positions,  default=0.5)
    max_r  = max(right_positions, default=0.5)
    t_len  = float(t_seg_spec.get("length", 1.0))
    pad    = max((max_l + max_r) * 0.06, 0.5)

    LEFT_C  = "#2563EB"
    RIGHT_C = "#D97706"
    TSEG_C  = "#475569"
    JOINT_C = "#1E293B"

    shapes = []

    # ── Header pipe: left arm ─────────────────────────────────────────────────
    x_left_end  = -max_l if left_positions  else -0.7
    x_right_end =  max_r if right_positions else  0.7
    shapes.append(dict(type="line",
        x0=x_left_end, y0=PIPE_Y, x1=0, y1=PIPE_Y,
        line=dict(color=LEFT_C if left_positions else "#CBD5E1", width=9)))

    # ── Header pipe: right arm ────────────────────────────────────────────────
    shapes.append(dict(type="line",
        x0=0, y0=PIPE_Y, x1=x_right_end, y1=PIPE_Y,
        line=dict(color=RIGHT_C if right_positions else "#CBD5E1", width=9)))

    # ── T-segment: vertical drop then horizontal to separator ─────────────────
    shapes.append(dict(type="line",
        x0=0, y0=PIPE_Y, x1=0, y1=TSEG_Y,
        line=dict(color=TSEG_C, width=14)))
    if t_len > 0:
        shapes.append(dict(type="line",
            x0=0, y0=TSEG_Y, x1=t_len, y1=TSEG_Y,
            line=dict(color=TSEG_C, width=14)))

    # ── Separator box ─────────────────────────────────────────────────────────
    sep_x = t_len if t_len > 0 else 0
    sep_w, sep_h = 1.3, 0.75
    shapes.append(dict(type="rect",
        x0=sep_x, y0=TSEG_Y - sep_h / 2, x1=sep_x + sep_w, y1=TSEG_Y + sep_h / 2,
        line=dict(color=JOINT_C, width=2), fillcolor="#DBEAFE"))

    # ── Tap risers ────────────────────────────────────────────────────────────
    left_sorted  = sorted(left_positions,  reverse=True)   # farthest first
    right_sorted = sorted(right_positions)                  # nearest first
    for pos in left_sorted:
        shapes.append(dict(type="line",
            x0=-pos, y0=PIPE_Y, x1=-pos, y1=TAP_TOP,
            line=dict(color=LEFT_C, width=3)))
    for pos in right_sorted:
        shapes.append(dict(type="line",
            x0=pos, y0=PIPE_Y, x1=pos, y1=TAP_TOP,
            line=dict(color=RIGHT_C, width=3)))

    fig = go.Figure()

    # ── Tap inlet markers (triangles pointing down = flow direction) ──────────
    tap_xs   = [-p for p in left_sorted] + list(right_sorted)
    tap_cs   = [LEFT_C] * len(left_sorted) + [RIGHT_C] * len(right_sorted)
    tap_lbls = (
        [f"L{len(left_sorted) - i}<br>{p:.1f} m" for i, p in enumerate(left_sorted)] +
        [f"R{i + 1}<br>{p:.1f} m"                for i, p in enumerate(right_sorted)]
    )
    if tap_xs:
        fig.add_trace(go.Scatter(
            x=tap_xs, y=[TAP_TOP] * len(tap_xs),
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color=tap_cs,
                        line=dict(color="white", width=1)),
            text=tap_lbls,
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="skip", showlegend=False,
        ))

    # ── T-junction marker ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0], y=[PIPE_Y],
        mode="markers+text",
        marker=dict(symbol="circle", size=20, color=JOINT_C,
                    line=dict(color="white", width=2)),
        text=["<b>T</b>"],
        textposition="middle center",
        textfont=dict(size=11, color="white"),
        hoverinfo="skip", showlegend=False,
    ))

    # ── Annotations ───────────────────────────────────────────────────────────
    anns = []

    # Tap inlet pressure banner (top-left)
    anns.append(dict(
        x=x_left_end, y=TAP_TOP + 0.55,
        text=f"<b>Tap inlet pressure: {P_inlet_bara:.3f} bara</b>",
        showarrow=False, xanchor="left",
        font=dict(size=10, color="#1E293B"),
        xref="x", yref="y",
    ))

    # Left arm: ΔP label + flow arrow
    if left_positions:
        w_mark = "  ⚠ governing" if worst_arm == "Left" else ""
        anns.append(dict(
            x=-max_l * 0.5, y=PIPE_Y - 0.35,
            text=f"ΔP = {dp_l_kpa:.2f} kPa{w_mark}",
            showarrow=False, xanchor="center",
            font=dict(size=9, color=LEFT_C),
            xref="x", yref="y",
        ))
        anns.append(dict(
            x=-max_l * 0.18, y=PIPE_Y + 0.28,
            ax=-max_l * 0.72, ay=PIPE_Y + 0.28,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3,
            arrowwidth=2, arrowcolor=LEFT_C,
        ))
        anns.append(dict(
            x=x_left_end + 0.1, y=PIPE_Y + 0.28,
            text="<i>← Left arm</i>",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=LEFT_C),
            xref="x", yref="y",
        ))

    # Right arm: ΔP label + flow arrow
    if right_positions:
        w_mark = "  ⚠ governing" if worst_arm == "Right" else ""
        anns.append(dict(
            x=max_r * 0.5, y=PIPE_Y - 0.35,
            text=f"ΔP = {dp_r_kpa:.2f} kPa{w_mark}",
            showarrow=False, xanchor="center",
            font=dict(size=9, color=RIGHT_C),
            xref="x", yref="y",
        ))
        anns.append(dict(
            x=max_r * 0.18, y=PIPE_Y + 0.28,
            ax=max_r * 0.72, ay=PIPE_Y + 0.28,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3,
            arrowwidth=2, arrowcolor=RIGHT_C,
        ))
        anns.append(dict(
            x=x_right_end - 0.1, y=PIPE_Y + 0.28,
            text="<i>Right arm →</i>",
            showarrow=False, xanchor="right",
            font=dict(size=9, color=RIGHT_C),
            xref="x", yref="y",
        ))

    # T-segment: pressure at T and flow arrow down
    anns.append(dict(
        x=0.25, y=(PIPE_Y + TSEG_Y) / 2,
        text=f"P_T ≈ {min(P_T_l_bara, P_T_r_bara):.3f} bara",
        showarrow=False, xanchor="left",
        font=dict(size=9, color=TSEG_C),
        xref="x", yref="y",
    ))
    anns.append(dict(
        x=0, y=TSEG_Y + 0.18,
        ax=0, ay=PIPE_Y - 0.18,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2,
        arrowwidth=2, arrowcolor=TSEG_C,
    ))
    if t_len > 0:
        anns.append(dict(
            x=t_len * 0.5, y=TSEG_Y + 0.22,
            text=f"T-seg  {t_len:.1f} m",
            showarrow=False, xanchor="center",
            font=dict(size=9, color=TSEG_C),
            xref="x", yref="y",
        ))

    # Separator label
    anns.append(dict(
        x=sep_x + sep_w / 2, y=TSEG_Y,
        text=f"<b>SEP</b><br>{P_sep_bara:.3f} bara",
        showarrow=False, xanchor="center",
        font=dict(size=9, color="#1E40AF"),
        xref="x", yref="y",
    ))

    x_lo = x_left_end  - pad - 4.0   # extra room for inlet pressure label
    x_hi = sep_x + sep_w + pad + 0.5

    fig.update_layout(
        shapes=shapes,
        annotations=anns,
        showlegend=False,
        height=340,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(visible=False, range=[x_lo, x_hi]),
        yaxis=dict(visible=False, range=[-0.3, TAP_TOP + 1.4]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(text="Header piping layout", font=dict(size=13, color="#1E293B"), x=0.5),
    )
    return fig


# ============================================================================
# HEADER CASE RUNNER  (Case C — uniform header with n A-line taps on each side)
# ============================================================================
def run_header_case(cid: str = "c", accent: str = "#059669",
                    results_a: dict = None) -> dict:
    """
    Uniform header with n A-line taps on each side of the T-junction.
    Flow enters from n_left + n_right copies of Case A and exits at T.
    Tap positions (distance from T) are individually configurable.
    """
    k = lambda name: f"{cid}_{name}"

    # ── Flow source: read from results_a, fall back to session state defaults
    _ra = results_a or {}
    gas_per_tap = dict(_ra.get("gas_flows_kgh") or {"H₂": 5.0})
    liq_per_tap = float(_ra.get("q_lye") or 0.5)
    liquid_type = str(_ra.get("liquid_type") or "KOH 30 wt%")
    T_C_a       = float(_ra.get("T_C") or 60.0)

    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("Inputs")

        with st.container(border=True):
            st.markdown("**Fluid — from Case A (read-only)**")
            _gas_str = "  ·  ".join(f"{sp}: {v:.2f} kg/h" for sp, v in gas_per_tap.items())
            st.caption(f"Gas per tap:  {_gas_str}")
            st.caption(f"Liquid per tap:  {liq_per_tap:.3f} m³/h  ·  {liquid_type}")

        with st.container(border=True):
            st.markdown("**Process Conditions**")
            p1, p2 = st.columns(2)
            P_target_sep = p1.number_input(
                "Target separator pressure (bara)",
                min_value=1.0, max_value=200.0,
                value=float(st.session_state.get(k("P_target_sep"), 16.5)),
                step=0.5, format="%.2f", key=k("P_target_sep"),
                help="Pressure required at the T-junction / separator connection. "
                     "The required tap inlet pressure is found automatically.")
            T_C = p2.number_input("Temperature (°C)",
                                  min_value=5.0, max_value=95.0,
                                  value=float(st.session_state.get(k("T_C"), T_C_a)),
                                  step=5.0, key=k("T_C"))

        with st.container(border=True):
            st.markdown("**Header Pipe** — uniform along full length")
            DN_OPT  = list(engine.PIPE_DATABASE.keys())
            PN_OPT  = ["PN20", "PN25", "PN40"]
            MAT_OPT = list(engine.MATERIAL_ROUGHNESS.keys())
            h1, h2, h3 = st.columns(3)
            hdr_dn  = h1.selectbox("DN",       DN_OPT,  key=k("hdr_dn"),
                                   index=DN_OPT.index(st.session_state.get(k("hdr_dn"), "DN100"))
                                         if st.session_state.get(k("hdr_dn"), "DN100") in DN_OPT else 0)
            hdr_pn  = h2.selectbox("PN",       PN_OPT,  key=k("hdr_pn"),
                                   index=PN_OPT.index(st.session_state.get(k("hdr_pn"), "PN40"))
                                         if st.session_state.get(k("hdr_pn"), "PN40") in PN_OPT else 2)
            hdr_mat = h3.selectbox("Material", MAT_OPT, key=k("hdr_mat"),
                                   index=MAT_OPT.index(st.session_state.get(k("hdr_mat"), "SS316L"))
                                         if st.session_state.get(k("hdr_mat"), "SS316L") in MAT_OPT else 0)
            h4, h5, h6 = st.columns(3)
            hdr_lined = h4.checkbox("Lined", value=bool(st.session_state.get(k("hdr_lined"), False)),
                                    key=k("hdr_lined"))
            LINER_OPT = list(engine.LINER_ROUGHNESS.keys())
            if hdr_lined:
                hdr_lmat  = h5.selectbox("Liner", LINER_OPT, key=k("hdr_lmat"))
                hdr_lthk  = h6.number_input("Thickness (mm)", min_value=0.1, max_value=20.0,
                                             value=float(st.session_state.get(k("hdr_lthk"), 1.0)),
                                             step=0.5, key=k("hdr_lthk"))
            else:
                hdr_lmat = "FEP";  hdr_lthk = 1.0
            _D_nom = engine.PIPE_DATABASE[hdr_dn][hdr_pn]
            _D_eff = _D_nom - 2 * hdr_lthk / 1000.0 if hdr_lined else _D_nom
            st.caption(f"ID {_D_eff*1000:.1f} mm")
            st.caption("**Header fittings**" if st.session_state.get(k("hdr_fits"), []) else "Header fittings — none")
            _hdr_fk = k("hdr_fits")
            if _hdr_fk not in st.session_state:
                st.session_state[_hdr_fk] = []
            _hdr_fit_rows  = st.session_state[_hdr_fk]
            _hdr_n_fits    = len(_hdr_fit_rows)
            _all_fit_types = list(engine.FITTING_Le_over_D.keys())
            if st.button("+ Add fitting", key=k("hdr_fadd"), help="Add a fitting to the header"):
                for jj in range(_hdr_n_fits):
                    _ft = st.session_state.get(k(f"hdr_ftype_{jj}"))
                    _fq = st.session_state.get(k(f"hdr_fqty_{jj}"), 1)
                    if _ft:
                        st.session_state[_hdr_fk][jj]["type"] = _ft
                        st.session_state[_hdr_fk][jj]["qty"]  = int(_fq)
                st.session_state[_hdr_fk].append({"type": _all_fit_types[0], "qty": 1})
                st.rerun()
            for j, _hfr in enumerate(list(_hdr_fit_rows)):
                _hfc1, _hfc2, _hfc3 = st.columns([3.5, 0.7, 0.3])
                _hfc1.selectbox("Fitting", _all_fit_types,
                    index=_all_fit_types.index(_hfr["type"]) if _hfr["type"] in _all_fit_types else 0,
                    key=k(f"hdr_ftype_{j}"), label_visibility="collapsed")
                _hfc2.number_input("Qty", min_value=1, value=_hfr["qty"],
                    key=k(f"hdr_fqty_{j}"), label_visibility="collapsed")
                if _hfc3.button("×", key=k(f"hdr_frem_{j}"), help="Remove"):
                    for jj in range(_hdr_n_fits):
                        _ft = st.session_state.get(k(f"hdr_ftype_{jj}"))
                        _fq = st.session_state.get(k(f"hdr_fqty_{jj}"), 1)
                        if _ft:
                            st.session_state[_hdr_fk][jj]["type"] = _ft
                            st.session_state[_hdr_fk][jj]["qty"]  = int(_fq)
                    st.session_state[_hdr_fk].pop(j)
                    for jj in range(j, _hdr_n_fits - 1):
                        st.session_state[k(f"hdr_ftype_{jj}")] = st.session_state[_hdr_fk][jj]["type"]
                        st.session_state[k(f"hdr_fqty_{jj}")] = st.session_state[_hdr_fk][jj]["qty"]
                    st.session_state.pop(k(f"hdr_ftype_{_hdr_n_fits-1}"), None)
                    st.session_state.pop(k(f"hdr_fqty_{_hdr_n_fits-1}"), None)
                    st.rerun()

        _hdr_fits_list = []
        _hdr_fk = k("hdr_fits")
        for jj in range(len(st.session_state.get(_hdr_fk, []))):
            _ft = st.session_state.get(k(f"hdr_ftype_{jj}"))
            _fq = st.session_state.get(k(f"hdr_fqty_{jj}"), 1)
            if _ft and _ft in engine.FITTING_Le_over_D and int(_fq) > 0:
                _hdr_fits_list.append({"type": _ft, "qty": int(_fq)})
        hdr_spec = {
            "dn": hdr_dn, "pn": hdr_pn, "material": hdr_mat,
            "lined": hdr_lined, "liner_material": hdr_lmat, "liner_thickness_mm": hdr_lthk,
            "fittings_list": _hdr_fits_list,
        }

        with st.container(border=True):
            st.markdown("**Calculation Settings**")
            _cs1, _cs2 = st.columns(2)
            correlation    = _cs1.selectbox("ΔP correlation",
                                            engine.TWO_PHASE_CORRELATIONS, key=k("correlation"))
            voidage_method = _cs2.selectbox("Void fraction",
                                            engine.VOIDAGE_METHODS, key=k("voidage_method"))

        # ── Tap position editor ────────────────────────────────────────────────
        def _tap_editor(side, default_n=3, default_spacing=2.5):
            """Render n + position inputs for one arm. Returns list of distances (m)."""
            n = st.number_input(f"Taps — {side} arm", min_value=0, max_value=8,
                                value=int(st.session_state.get(k(f"n_{side}"), default_n)),
                                step=1, key=k(f"n_{side}"))
            if n == 0:
                return []
            # Show positions in rows of 4
            positions = []
            cols_per_row = 4
            _rows = [range(i, min(i + cols_per_row, n))
                     for i in range(0, n, cols_per_row)]
            for _row_idxs in _rows:
                _cols = st.columns(len(_row_idxs))
                for _ci, _ti in enumerate(_row_idxs):
                    _pk = k(f"pos_{side}_{_ti}")
                    if _pk not in st.session_state:
                        st.session_state[_pk] = float((_ti + 1) * default_spacing)
                    positions.append(_cols[_ci].number_input(
                        f"{side[0].upper()}{_ti+1} (m)", min_value=0.1, step=0.5,
                        key=_pk))
            return positions

        st.markdown("---")
        st.markdown("#### Tap Positions  *(distance from T-junction)*")
        _diag_left  = st.session_state.get(k("n_left"), 3)
        _diag_right = st.session_state.get(k("n_right"), 3)
        st.caption(
            "  ".join([f"←─ L{i}" for i in range(int(_diag_left), 0, -1)])
            + "  ──T──  "
            + "  ".join([f"R{i} ─→" for i in range(1, int(_diag_right) + 1)])
        )
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**← Left arm**")
            left_positions  = _tap_editor("left")
        with rc:
            st.markdown("**Right arm →**")
            right_positions = _tap_editor("right")

        with st.container(border=True):
            st.markdown("**T-Segment** *(at junction — may be larger than header)*")
            st.caption("Carries combined flow from all taps. Horizontal. End = separator connection.")
            ts1, ts2, ts3, ts4 = st.columns(4)
            t_dn  = ts1.selectbox("DN", DN_OPT, key=k("t_dn"),
                                   index=DN_OPT.index(st.session_state.get(k("t_dn"), "DN150"))
                                         if st.session_state.get(k("t_dn"), "DN150") in DN_OPT else 0)
            t_pn  = ts2.selectbox("PN", PN_OPT, key=k("t_pn"),
                                   index=PN_OPT.index(st.session_state.get(k("t_pn"), "PN40"))
                                         if st.session_state.get(k("t_pn"), "PN40") in PN_OPT else 2)
            t_mat = ts3.selectbox("Material", MAT_OPT, key=k("t_mat"),
                                   index=MAT_OPT.index(st.session_state.get(k("t_mat"), "SS316L"))
                                         if st.session_state.get(k("t_mat"), "SS316L") in MAT_OPT else 0)
            t_len = ts4.number_input("Length (m)", min_value=0.0,
                                      value=float(st.session_state.get(k("t_len"), 1.0)),
                                      step=0.5, key=k("t_len"))
            _t_D = engine.PIPE_DATABASE[t_dn][t_pn]
            st.caption(f"ID {_t_D*1000:.1f} mm")

        t_seg_spec = {"dn": t_dn, "pn": t_pn, "material": t_mat,
                      "length": t_len, "fittings_list": []}

    # ── CALCULATION ───────────────────────────────────────────────────────────
    with col_out:
        st.subheader("Results")

        n_left  = len(left_positions)
        n_right = len(right_positions)
        n_total = n_left + n_right
        total_gas = {sp: kgh * n_total for sp, kgh in gas_per_tap.items()}
        total_liq = liq_per_tap * n_total

        # ── Goal-seek: find tap inlet pressure for target separator pressure ──
        _hdr_for_gs = {
            "header_pipe":    hdr_spec,
            "gas_per_tap":    gas_per_tap,
            "liq_per_tap":    liq_per_tap,
            "T_C":            T_C,
            "liquid_type":    liquid_type,
            "left_taps":      left_positions,
            "right_taps":     right_positions,
            "t_seg":          t_seg_spec,
            "correlation":    correlation,
            "voidage_method": voidage_method,
            "gas_flows_kgh":  total_gas,
            "q_lye":          total_liq,
            "total_dp_kpa":   0.5,   # coarse starting offset; converges regardless
        }
        if left_positions or right_positions:
            _gs = _goal_seek_header(_hdr_for_gs, P_target_sep)
            P_inlet_bara = _gs["P_hdr_in"]
            _gs_resid    = abs(_gs["P_sep"] - P_target_sep) * 1000
            if _gs["converged"]:
                st.success(
                    f"Goal-seek converged in {_gs['iterations']} iter  ·  "
                    f"residual {_gs_resid:.2f} mbar"
                )
            else:
                st.warning(
                    f"Goal-seek did not fully converge ({_gs['iterations']} iter, "
                    f"residual {_gs_resid:.1f} mbar)"
                )
        else:
            P_inlet_bara = P_target_sep
            st.info("Add taps to compute the required inlet pressure.")

        # ── Forward march at the found inlet pressure ─────────────────────────
        P_start = P_inlet_bara * 1e5
        dp_l_Pa, P_T_l, fric_l, grav_l, rec_l = _march_header_simple(
            left_positions,  gas_per_tap, liq_per_tap,
            hdr_spec, P_start, T_C, liquid_type, correlation, voidage_method)
        dp_r_Pa, P_T_r, fric_r, grav_r, rec_r = _march_header_simple(
            right_positions, gas_per_tap, liq_per_tap,
            hdr_spec, P_start, T_C, liquid_type, correlation, voidage_method)

        dp_l_kpa   = dp_l_Pa / 1000.0
        dp_r_kpa   = dp_r_Pa / 1000.0
        P_T_l_bara = P_T_l / 1e5
        P_T_r_bara = P_T_r / 1e5
        worst_arm  = "Left" if dp_l_kpa >= dp_r_kpa else "Right"
        dp_worst   = max(dp_l_kpa, dp_r_kpa)
        P_T_worst  = min(P_T_l_bara, P_T_r_bara)

        # T-segment calculation
        dp_t_kpa   = 0.0
        P_sep_bara = P_T_worst
        rec_t      = None
        if t_seg_spec["length"] > 0 and total_gas and total_liq > 0:
            try:
                dp_t_Pa, P_sep, fric_t, grav_t, rec_t = _march_single_seg(
                    t_seg_spec, P_T_worst * 1e5, T_C, total_gas, liquid_type,
                    total_liq, correlation, voidage_method)
                dp_t_kpa   = dp_t_Pa / 1000.0
                P_sep_bara = P_sep / 1e5
            except Exception as _e:
                st.warning(f"T-segment calculation failed: {_e}")

        with st.container(border=True):
            st.subheader("T-Junction")
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Required tap inlet pressure",
                       f"{P_inlet_bara:.4f} bara",
                       delta=f"Target sep: {P_target_sep:.2f} bara",
                       delta_color="off",
                       help="Pressure required at each header tap = branch line outlet pressure")
            _c2.metric("Left arm ΔP",
                       f"{dp_l_kpa:.3f} kPa",
                       delta=f"{P_inlet_bara:.4f} → {P_T_l_bara:.4f} bara",
                       delta_color="off")
            _c3.metric("Right arm ΔP",
                       f"{dp_r_kpa:.3f} kPa",
                       delta=f"{P_inlet_bara:.4f} → {P_T_r_bara:.4f} bara",
                       delta_color="off")
            _c4.metric("Separator connection pressure",
                       f"{P_sep_bara:.4f} bara",
                       delta=f"T-seg ΔP: {dp_t_kpa:.3f} kPa",
                       delta_color="off")
            st.caption(
                f"Total flow at T:  "
                + "  ·  ".join(f"{sp} {v:.2f} kg/h" for sp, v in total_gas.items())
                + f"  ·  Liquid {total_liq:.3f} m³/h"
                + f"  ·  ({n_left}L + {n_right}R = {n_total} taps)"
            )

        # ── Piping schematic ──────────────────────────────────────────────────
        fig_sch_hdr = _make_header_schematic(
            left_positions, right_positions, t_seg_spec,
            P_inlet_bara, worst_arm,
            dp_l_kpa, dp_r_kpa,
            P_T_l_bara, P_T_r_bara, P_sep_bara)
        st.plotly_chart(fig_sch_hdr, use_container_width=True,
                        key=f"{cid}_hdr_schematic")

        _col_hdr = ["Seg", "Taps in seg", "From T (m)", "To T (m)", "L (m)",
                    "Pipe", "ID (mm)", "Regime",
                    "Q_gas_kgh", "Q_liq_m3h",
                    "ΔP_fric (kPa)", "ΔP_grav (kPa)", "ΔP (kPa)",
                    "P_in (bara)", "P_out (bara)", "V_m (m/s)", "V_m/V_e"]
        _col_cfg_hdr = {
            "ID (mm)":       st.column_config.NumberColumn(format="%.1f"),
            "From T (m)":    st.column_config.NumberColumn(format="%.2f"),
            "To T (m)":      st.column_config.NumberColumn(format="%.2f"),
            "Q_gas_kgh":     st.column_config.NumberColumn(label="Q gas (kg/h)", format="%.3f"),
            "Q_liq_m3h":     st.column_config.NumberColumn(label="Q liq (m³/h)", format="%.3f"),
            "P_in (bara)":   st.column_config.NumberColumn(format="%.4f"),
            "P_out (bara)":  st.column_config.NumberColumn(format="%.4f"),
            "ΔP (kPa)":      st.column_config.NumberColumn(format="%.3f"),
            "ΔP_fric (kPa)": st.column_config.NumberColumn(format="%.3f"),
            "ΔP_grav (kPa)": st.column_config.NumberColumn(format="%.3f"),
            "V_m (m/s)":     st.column_config.NumberColumn(format="%.3f"),
            "V_m/V_e":       st.column_config.NumberColumn(format="%.3f"),
        }
        tl, tr = st.columns(2)
        with tl:
            st.markdown(f"**Left arm**  {'⚠️ governing' if worst_arm == 'Left' else ''}")
            if rec_l:
                _df_l = pd.DataFrame(rec_l)
                st.dataframe(_df_l[[c for c in _col_hdr if c in _df_l.columns]],
                             column_config=_col_cfg_hdr,
                             hide_index=True, use_container_width=True)
        with tr:
            st.markdown(f"**Right arm**  {'⚠️ governing' if worst_arm == 'Right' else ''}")
            if rec_r:
                _df_r = pd.DataFrame(rec_r)
                st.dataframe(_df_r[[c for c in _col_hdr if c in _df_r.columns]],
                             column_config=_col_cfg_hdr,
                             hide_index=True, use_container_width=True)

        if rec_t:
            st.markdown("**T-segment** (junction → separator)")
            _df_t = pd.DataFrame([rec_t])
            _t_cols = [c for c in _col_hdr if c in _df_t.columns]
            st.dataframe(_df_t[_t_cols], column_config=_col_cfg_hdr,
                         hide_index=True, use_container_width=True)

        fig_hdr = None
        if rec_l or rec_r:
            fig_hdr = go.Figure()
            def _arm_trace(records, label, color, positions):
                if not records:
                    return
                dists_sorted = sorted(positions, reverse=True)
                xs = [dists_sorted[0]] if dists_sorted else [0.0]
                ys = [P_inlet_bara]
                for r in records:
                    xs.append(r["To T (m)"])
                    ys.append(r["P_out (bara)"])
                fig_hdr.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=label,
                    line=dict(color=color, width=2), marker=dict(size=6)))
            _arm_trace(rec_l, "Left arm",  "#2563EB", left_positions)
            _arm_trace(rec_r, "Right arm", "#D97706", right_positions)
            fig_hdr.update_layout(
                xaxis_title="Distance from T-junction (m)",
                xaxis=dict(autorange="reversed"),
                yaxis_title="Pressure (bara)",
                height=300, margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", y=1.1),
                template="plotly_white",
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_gridcolor="#F1F5F9", yaxis_gridcolor="#F1F5F9",
            )
            st.plotly_chart(fig_hdr, use_container_width=True,
                            key=f"{cid}_hdr_pressure_profile")

    # ── RETURN DICT ────────────────────────────────────────────────────────────
    try:
        _props_out = engine.calculate_two_phase_properties(
            P_T_worst, T_C, total_gas, liquid_type, total_liq) if total_liq > 0 and total_gas else {}
    except Exception:
        _props_out = {}

    _all_recs  = ([dict(r, Seg=f"L{r['Seg']}") for r in rec_l] +
                  [dict(r, Seg=f"R{r['Seg']}") for r in rec_r])
    _pipe_len  = (max(left_positions,  default=[0.0]) +
                  max(right_positions, default=[0.0]))
    _total_dp_fric = (fric_l + fric_r) / 1000.0
    _total_dp_grav = (grav_l + grav_r) / 1000.0

    return {
        "P_bara":               P_inlet_bara,
        "P_target_sep":         P_target_sep,
        "T_C":                  T_C,
        "total_dp_kpa":         dp_worst + dp_t_kpa,
        "outlet_pressure_bara": P_sep_bara,
        "outlet_pressure_mbar": P_sep_bara * 1000.0,
        "P_T_bara":             P_T_worst,
        "P_separator_bara":     P_sep_bara,
        "dp_header_kpa":        dp_worst,
        "dp_t_seg_kpa":         dp_t_kpa,
        "total_dp_fric_kpa":    _total_dp_fric,
        "total_dp_grav_kpa":    _total_dp_grav,
        "pipe_length_m":        _pipe_len,
        "cumulative_distance":  _pipe_len,
        "liquid_type":          liquid_type,
        "gas_flows_kgh":        total_gas,
        "q_lye":                total_liq,
        "props":                _props_out,
        "segments":             [],
        "grid_records":         _all_recs,
        "correlation":          correlation,
        "voidage_method":       voidage_method,
        "fig_sch":              fig_sch_hdr,
        "fig_prof":             fig_hdr,
        # Fields for goal-seek re-run
        "left_taps":            left_positions,
        "right_taps":           right_positions,
        "gas_per_tap":          gas_per_tap,
        "liq_per_tap":          liq_per_tap,
        "header_pipe":          hdr_spec,
        "t_seg":                t_seg_spec,
        # Per-arm detail
        "dp_left_kpa":          dp_l_kpa,
        "dp_right_kpa":         dp_r_kpa,
        "P_T_left_bara":        P_T_l_bara,
        "P_T_right_bara":       P_T_r_bara,
        "worst_arm":            worst_arm,
        "n_left":               n_left,
        "n_right":              n_right,
    }


if "label_a" not in st.session_state:
    st.session_state["label_a"] = "Case A"
if "label_b" not in st.session_state:
    st.session_state["label_b"] = "Case B"

# Forward any pending goal-seek apply values BEFORE widgets render
if "stack_apply_a_pending" in st.session_state:
    st.session_state["a_P_bara"] = st.session_state.pop("stack_apply_a_pending")
if "stack_apply_b_pending" in st.session_state:
    st.session_state["b_P_bara"] = st.session_state.pop("stack_apply_b_pending")

with st.container(border=True):
    _lab_col1, _lab_col2 = st.columns(2)
    _lab_col1.text_input("Case A label", key="label_a", max_chars=40,
                         help="Name used in all tabs, headers, charts, and reports.")
    _lab_col2.text_input("Case B label", key="label_b", max_chars=40,
                         help="Name used in all tabs, headers, charts, and reports.")

_la = st.session_state["label_a"]
_lb = st.session_state["label_b"]
_lc = f"{_la} Header"
_ld = f"{_lb} Header"

tab_a, tab_b, tab_c, tab_d, tab_cmp, tab_stack, tab_dn = st.tabs(
    [_la, _lb, _lc, _ld,
     f"Compare {_la} vs {_lb}", "Generator ΔP", "DN Study"])

with tab_a:
    results_a = run_case("a", accent="#2563EB")

with tab_b:
    results_b = run_case("b", accent="#D97706")

with tab_c:
    st.info(
        f"**{_lc}**  "
        f"A uniform pipe with n {_la} taps on each side of a central T-junction. "
        f"Each tap feeds one copy of {_la}'s branch flow. "
        "Worst-arm ΔP = farthest tap → T → separator.",
        icon="ℹ️",
    )
    results_c = run_header_case("c", accent="#059669", results_a=results_a)

with tab_d:
    st.info(
        f"**{_ld}**  "
        f"A uniform pipe with n {_lb} taps on each side of a central T-junction. "
        f"Each tap feeds one copy of {_lb}'s branch flow. "
        "Worst-arm ΔP = farthest tap → T → separator.",
        icon="ℹ️",
    )
    results_d = run_header_case("d", accent="#7C3AED", results_a=results_b)

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
                      tuple(sorted((f["type"],f["qty"]) for f in s.get("fittings_list",[]))),
                      bool(s.get("lined", False)), s.get("liner_material", "FEP"),
                      float(s.get("liner_thickness_mm", 1.0)))
                     for s in ra["segments"]],
        },
        "b": {
            "P": rb["P_bara"], "T": rb["T_C"],
            "gas": {_k: float(_v) for _k, _v in rb["gas_flows_kgh"].items()},
            "liq": rb["liquid_type"], "lye": rb["q_lye"],
            "segs": [(s["type"], s["dn"], s["pn"], float(s["length"]),
                      tuple(sorted((f["type"],f["qty"]) for f in s.get("fittings_list",[]))),
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
    _la = st.session_state.get("label_a", "Case A")
    _lb = st.session_state.get("label_b", "Case B")
    st.subheader(f"{_la}  vs.  {_lb}")

    ra, rb = results_a, results_b

    # ── Side-by-side headline metrics ─────────────────────────────────────────
    with st.container(border=True):
        _cl, _cm, _cr = st.columns([2, 2, 2])

        _cl.markdown("**Metric**")
        _cm.markdown(f"**{_la}**")
        _cr.markdown(f"**{_lb}**")
        st.divider()

        def _cmp_row(label, va, vb, fmt="{}", better="lower", unit=""):
            """Render one comparison row with delta badge."""
            _cl, _cm, _cr = st.columns([2, 2, 2])
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
            _cl.markdown(label)
            _cm.metric(_la, f"{fmt.format(va)} {unit}", label_visibility="collapsed")
            _cr.metric(_lb, f"{fmt.format(vb)} {unit}", delta=_delta_str,
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

    # ── Overlaid pressure profiles ─────────────────────────────────────────────
    st.markdown("#### Pressure Profiles")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=ra["pressure_profile_x"], y=ra["pressure_profile_y"],
        mode="lines+markers", name=_la,
        line=dict(color="#2563EB", width=2.5), marker=dict(size=7, color="#2563EB"),
        hovertemplate=f"{_la}  |  Distance: %{{x:.2f}} m<br>Pressure: %{{y:.4f}} bara<extra></extra>"))
    fig_cmp.add_trace(go.Scatter(
        x=rb["pressure_profile_x"], y=rb["pressure_profile_y"],
        mode="lines+markers", name=_lb,
        line=dict(color="#D97706", width=2.5, dash="dash"), marker=dict(size=7, color="#D97706"),
        hovertemplate=f"{_lb}  |  Distance: %{{x:.2f}} m<br>Pressure: %{{y:.4f}} bara<extra></extra>"))
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
    fig_bar.add_trace(go.Bar(name=_la, x=_segs_a, y=_dp_a,
                             marker_color="#2563EB", opacity=0.85))
    fig_bar.add_trace(go.Bar(name=_lb, x=_segs_b, y=_dp_b,
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
                f"Quantifies the full ΔP range for {_la} and {_lb} due to method uncertainty.")

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
                               annotation_text=f"{_la} range", annotation_position="top left",
                               annotation_font=dict(size=9, color="#2563EB"))
        if len(_vb) >= 2:
            fig_sens.add_vrect(x0=min(_vb), x1=max(_vb),
                               fillcolor="rgba(217,119,6,0.07)", line_width=0,
                               annotation_text=f"{_lb} range", annotation_position="bottom right",
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
            x=_xb_p, y=_yb_p, mode="markers", name=_lb,
            marker=dict(color="#D97706", size=11, symbol="diamond",
                        line=dict(color="#92400E", width=1.5)),
            hovertemplate=f"{_lb}  |  %{{y}}<br>Total ΔP: %{{x:.3f}} kPa<extra></extra>"))

        # dots (blue circles)
        _xa_p = [_dp_a_vals[_ci] for _ci in range(len(_ylabels)) if _ok_a[_ci]]
        _ya_p = [_ylabels[_ci]   for _ci in range(len(_ylabels)) if _ok_a[_ci]]
        fig_sens.add_trace(go.Scatter(
            x=_xa_p, y=_ya_p, mode="markers", name=_la,
            marker=dict(color="#2563EB", size=11, symbol="circle",
                        line=dict(color="#1E40AF", width=1.5)),
            hovertemplate=f"{_la}  |  %{{y}}<br>Total ΔP: %{{x:.3f}} kPa<extra></extra>"))

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
                               annotation_text=f"{_la} selected: {_sel_a_dp:.2f} kPa",
                               annotation_position="top right",
                               annotation_font=dict(size=9, color="#2563EB"))
        if _sel_b_dp is not None:
            fig_sens.add_vline(x=_sel_b_dp,
                               line=dict(color="#D97706", width=1.5, dash="dot"),
                               annotation_text=f"{_lb} selected: {_sel_b_dp:.2f} kPa",
                               annotation_position="bottom right",
                               annotation_font=dict(size=9, color="#D97706"))

        fig_sens.update_layout(
            title=dict(text=f"Total ΔP — all 12 method combinations  (● {_la}  ◆ {_lb})",
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
                {"": _la,
                 "Min (kPa)":      f"{_a_min:.3f}",
                 "Selected (kPa)": f"{ra['total_dp_kpa']:.3f}",
                 "Max (kPa)":      f"{_a_max:.3f}",
                 "Spread (kPa)":   f"{_a_max - _a_min:.3f}"},
                {"": _lb,
                 "Min (kPa)":      f"{_b_min:.3f}",
                 "Selected (kPa)": f"{rb['total_dp_kpa']:.3f}",
                 "Max (kPa)":      f"{_b_max:.3f}",
                 "Spread (kPa)":   f"{_b_max - _b_min:.3f}"},
            ]), hide_index=True, use_container_width=True)
            if _overlap:
                st.warning(
                    f"Ranges **overlap** — the relative ordering of {_la} vs {_lb} depends on "
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

        _rt_a = _regime_table(_sa, ra["segments"], _la)
        _rt_b = _regime_table(_sb, rb["segments"], _lb)

        _rca, _rcb = st.columns(2)
        with _rca:
            st.markdown(f"**{_la}**")
            if _rt_a is not None:
                st.dataframe(_rt_a, hide_index=True, use_container_width=True)
            else:
                st.caption("No regime data available.")
        with _rcb:
            st.markdown(f"**{_lb}**")
            if _rt_b is not None:
                st.dataframe(_rt_b, hide_index=True, use_container_width=True)
            else:
                st.caption("No regime data available.")

    # ── Generate All Reports ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Reports")
    _sens_avail = ("sens_a" in st.session_state and "sens_b" in st.session_state
                   and fig_sens is not None)

    def _build_sens_data():
        if _sens_avail:
            return {"sa": st.session_state["sens_a"],
                    "sb": st.session_state["sens_b"],
                    "fig": fig_sens}
        return None

    def _build_stack_dp_data():
        _sh = st.session_state.get("stack_gsr_h2")
        _so = st.session_state.get("stack_gsr_o2")
        if _sh and _so:
            return {
                "label_a": _la,
                "label_b": _lb,
                "gsr_h2":  _sh,
                "gsr_o2":  _so,
                "P_sep_h2": st.session_state.get("stack_sep_h2"),
                "P_sep_o2": st.session_state.get("stack_sep_o2"),
            }
        return None

    _all_col1, _all_col2 = st.columns([1, 2])
    with _all_col1:
        if st.button("Generate All Reports", type="primary",
                     use_container_width=True, key="gen_all_rpts"):
            _errors = []
            with st.spinner("Building all reports…"):
                try:
                    _buf_a_rpt = report_generator.generate_report(
                        P_bara=ra["P_bara"], T_C=ra["T_C"],
                        gas_flows_kgh=ra["gas_flows_kgh"],
                        liquid_type=ra["liquid_type"], q_lye=ra["q_lye"],
                        props=ra["props"], grid_records=ra["grid_records"],
                        segments=ra["segments"],
                        total_dp_kpa=ra["total_dp_kpa"],
                        outlet_pressure_bara=ra["outlet_pressure_bara"],
                        pipe_length_m=ra["pipe_length_m"],
                        cumulative_distance=ra["cumulative_distance"],
                        fig_sch=ra.get("fig_sch"), fig_prof=ra.get("fig_prof"),
                        case_label=_la)
                    st.session_state["rpt_a_bytes"] = _buf_a_rpt.getvalue()
                except Exception as _e:
                    _errors.append(f"{_la}: {_e}")

                try:
                    _buf_b_rpt = report_generator.generate_report(
                        P_bara=rb["P_bara"], T_C=rb["T_C"],
                        gas_flows_kgh=rb["gas_flows_kgh"],
                        liquid_type=rb["liquid_type"], q_lye=rb["q_lye"],
                        props=rb["props"], grid_records=rb["grid_records"],
                        segments=rb["segments"],
                        total_dp_kpa=rb["total_dp_kpa"],
                        outlet_pressure_bara=rb["outlet_pressure_bara"],
                        pipe_length_m=rb["pipe_length_m"],
                        cumulative_distance=rb["cumulative_distance"],
                        fig_sch=rb.get("fig_sch"), fig_prof=rb.get("fig_prof"),
                        case_label=_lb)
                    st.session_state["rpt_b_bytes"] = _buf_b_rpt.getvalue()
                except Exception as _e:
                    _errors.append(f"{_lb}: {_e}")

                try:
                    _cbuf = report_generator.generate_comparison_report(
                        results_a=ra, results_b=rb,
                        label_a=_la, label_b=_lb,
                        fig_cmp=fig_cmp, fig_bar=fig_bar,
                        sensitivity_data=_build_sens_data(),
                        stack_dp=_build_stack_dp_data())
                    st.session_state["cmp_rpt_bytes"] = _cbuf.getvalue()
                except Exception as _e:
                    _errors.append(f"Comparison: {_e}")

                try:
                    def _build_dn_study_data():
                        _dn_p = st.session_state.get("dn_study_dn_primary")
                        _dn_a = st.session_state.get("dn_study_dn_alt")
                        _gh_a = st.session_state.get("dn_study_gsr_h2_alt")
                        _go_a = st.session_state.get("dn_study_gsr_o2_alt")
                        _gh_p = st.session_state.get("stack_gsr_h2")
                        _go_p = st.session_state.get("stack_gsr_o2")
                        if not all([_dn_p, _dn_a, _gh_a, _go_a, _gh_p, _go_p]):
                            return None
                        _dp_p_mb = (_gh_p["P_line_in"] - _go_p["P_line_in"]) * 1000.0
                        _dp_a_mb = (_gh_a["P_line_in"] - _go_a["P_line_in"]) * 1000.0
                        _seg0    = ra["segments"][0] if ra.get("segments") else {}
                        _pn0     = _seg0.get("pn", "PN20")
                        _lined0  = _seg0.get("lined", False)
                        _lthk0_m = _seg0.get("liner_thickness_mm", 1.0) / 1000.0
                        _D_p_b   = engine.PIPE_DATABASE.get(_dn_p, {}).get(
                                        _pn0, list(engine.PIPE_DATABASE.get(_dn_p, {None: 0}).values())[0])
                        _D_a_b   = engine.PIPE_DATABASE.get(_dn_a, {}).get(
                                        _pn0, list(engine.PIPE_DATABASE.get(_dn_a, {None: 0}).values())[0])
                        _D_p_e   = _D_p_b - 2*_lthk0_m if _lined0 else _D_p_b
                        _D_a_e   = _D_a_b - 2*_lthk0_m if _lined0 else _D_a_b
                        _scale   = (_D_p_e / _D_a_e)**2 if _D_a_e > 0 else 1.0
                        _rec_a0  = ra["grid_records"][0]  if ra.get("grid_records")  else {}
                        _rec_b0  = rb["grid_records"][0]  if rb.get("grid_records")  else {}
                        return {
                            "dn_primary": _dn_p, "dn_alt": _dn_a,
                            "label_a": _la, "label_b": _lb,
                            "gsr_h2_primary": _gh_p, "gsr_o2_primary": _go_p,
                            "gsr_h2_alt":     _gh_a, "gsr_o2_alt":     _go_a,
                            "dp_gen_primary_mbar": _dp_p_mb,
                            "dp_gen_alt_mbar":     _dp_a_mb,
                            "vel_data": {
                                "vm_a_primary": float(_rec_a0.get("V_m (m/s)", 0)),
                                "vm_b_primary": float(_rec_b0.get("V_m (m/s)", 0)),
                                "vm_a_alt":     float(_rec_a0.get("V_m (m/s)", 0)) * _scale,
                                "vm_b_alt":     float(_rec_b0.get("V_m (m/s)", 0)) * _scale,
                                "ve_a":         float(_rec_a0.get("V_e (m/s)", 0)),
                                "ve_b":         float(_rec_b0.get("V_e (m/s)", 0)),
                                "D_p_mm":       _D_p_e * 1000,
                                "D_a_mm":       _D_a_e * 1000,
                                "vel_scale":    _scale,
                            },
                            "p_sep_h2": st.session_state.get("stack_sep_h2", 0),
                            "p_sep_o2": st.session_state.get("stack_sep_o2", 0),
                        }
                    _combined_buf = report_generator.generate_combined_report(
                        cases=[ra, rb, results_c, results_d],
                        case_labels=[_la, _lb, _lc, _ld],
                        fig_cmp=fig_cmp, fig_bar=fig_bar,
                        sensitivity_data=_build_sens_data(),
                        stack_dp=_build_stack_dp_data(),
                        dn_study_data=_build_dn_study_data())
                    st.session_state["combined_rpt_bytes"] = _combined_buf.getvalue()
                except Exception as _e:
                    _errors.append(f"Combined: {_e}")

            if _errors:
                for _err in _errors:
                    st.error(f"Report failed: {_err}")
            else:
                st.success("All reports ready — download below.")

    with _all_col2:
        _dl_cols = st.columns(2)
        if st.session_state.get("rpt_a_bytes"):
            _dl_cols[0].download_button(
                f"Download {_la}  (.docx)",
                data=st.session_state["rpt_a_bytes"],
                file_name=f"report_{_la.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_rpt_a_all")
        if st.session_state.get("rpt_b_bytes"):
            _dl_cols[1].download_button(
                f"Download {_lb}  (.docx)",
                data=st.session_state["rpt_b_bytes"],
                file_name=f"report_{_lb.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_rpt_b_all")
        _dl_cols2 = st.columns(2)
        if st.session_state.get("cmp_rpt_bytes"):
            _dl_cols2[0].download_button(
                f"Download Comparison  (.docx)",
                data=st.session_state["cmp_rpt_bytes"],
                file_name="report_comparison.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_cmp_rpt")
        if st.session_state.get("combined_rpt_bytes"):
            _dl_cols2[1].download_button(
                "Download Combined  (.docx)",
                data=st.session_state["combined_rpt_bytes"],
                file_name="report_combined.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_combined_rpt")

    if not _sens_avail:
        st.caption("ℹ Run the Sensitivity Analysis above first to include it in the reports.")
    if not st.session_state.get("stack_gsr_h2"):
        st.caption("ℹ Run the Generator ΔP calculation (Generator ΔP tab) first to include it in the reports.")

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
        st.markdown(f"**{_la}**")
        st.dataframe(pd.DataFrame(ra["grid_records"])[_cmp_cols],
                     column_config=_col_cfg, hide_index=True, use_container_width=True)
    with _tb:
        st.markdown(f"**{_lb}**")
        st.dataframe(pd.DataFrame(rb["grid_records"])[_cmp_cols],
                     column_config=_col_cfg, hide_index=True, use_container_width=True)


# ============================================================================
# STACK ΔP TAB
# ============================================================================
with tab_stack:
    _la = st.session_state.get("label_a", "Case A")
    _lb = st.session_state.get("label_b", "Case B")
    _lc = f"{_la} Header"
    _ld = f"{_lb} Header"

    st.subheader("Generator Differential Pressure")
    st.caption(
        f"Goal-seek both the {_la} system ({_la} → {_lc}) and {_lb} system ({_lb} → {_ld}) "
        "to find the required line inlet pressures for given separator pressures. "
        f"The **Generator ΔP** = P_inlet_{_la} − P_inlet_{_lb} is the differential pressure "
        "across the process unit."
    )

    with st.container(border=True):
        st.markdown("##### Separator Target Pressures")
        _sk_col1, _sk_col2 = st.columns(2)
        with _sk_col1:
            _p_sep_h2 = st.number_input(
                f"{_la} separator pressure (bara)",
                min_value=0.1, max_value=200.0,
                value=float(round(results_c.get("P_separator_bara",
                                                results_a["outlet_pressure_bara"]), 3)),
                step=0.1, format="%.3f",
                key="stack_p_sep_h2",
                help=f"Target pressure at the {_la} gas-liquid separator."
            )
        with _sk_col2:
            _p_sep_o2 = st.number_input(
                f"{_lb} separator pressure (bara)",
                min_value=0.1, max_value=200.0,
                value=float(round(results_d.get("P_separator_bara",
                                                results_b["outlet_pressure_bara"]), 3)),
                step=0.1, format="%.3f",
                key="stack_p_sep_o2",
                help=f"Target pressure at the {_lb} gas-liquid separator."
            )

        _sk_run = st.button("Calculate Generator ΔP", type="primary",
                            use_container_width=True, key="stack_run")

    if _sk_run:
        with st.spinner(f"Solving {_la} and {_lb} systems…"):
            _gsr_h2 = _goal_seek_stack(results_a, results_c, _p_sep_h2)
            _gsr_o2 = _goal_seek_stack(results_b, results_d, _p_sep_o2)
        st.session_state["stack_gsr_h2"]  = _gsr_h2
        st.session_state["stack_gsr_o2"]  = _gsr_o2
        st.session_state["stack_sep_h2"]  = _p_sep_h2
        st.session_state["stack_sep_o2"]  = _p_sep_o2

    _gsr_h2 = st.session_state.get("stack_gsr_h2")
    _gsr_o2 = st.session_state.get("stack_gsr_o2")

    if _gsr_h2 and _gsr_o2:
        _shown_sep_h2 = st.session_state.get("stack_sep_h2", _p_sep_h2)
        _shown_sep_o2 = st.session_state.get("stack_sep_o2", _p_sep_o2)

        _conv_h2 = _gsr_h2["converged"]
        _conv_o2 = _gsr_o2["converged"]
        if _conv_h2 and _conv_o2:
            st.success(
                f"{_la} converged in {_gsr_h2['iterations']} iter.  "
                f"{_lb} converged in {_gsr_o2['iterations']} iter."
            )
        else:
            if not _conv_h2:
                st.warning(f"{_la} did not fully converge ({_gsr_h2['iterations']} iter, "
                           f"residual {abs(_gsr_h2['P_sep'] - _shown_sep_h2)*1000:.1f} mbar)")
            if not _conv_o2:
                st.warning(f"{_lb} did not fully converge ({_gsr_o2['iterations']} iter, "
                           f"residual {abs(_gsr_o2['P_sep'] - _shown_sep_o2)*1000:.1f} mbar)")

        st.divider()

        _p_in_a = _gsr_h2["P_line_in"]
        _p_in_b = _gsr_o2["P_line_in"]
        _dp_stack = _p_in_a - _p_in_b

        with st.container(border=True):
            st.markdown(f"##### Generator Differential Pressure  (P_inlet_{_la} − P_inlet_{_lb})")
            _sk1, _sk2, _sk3 = st.columns(3)
            _sk1.metric(
                f"{_la} inlet pressure",
                f"{_p_in_a:.4f} bara",
                delta=f"{_la} ΔP = {_gsr_h2['dp_line']:.3f} kPa",
                delta_color="off",
            )
            _sk2.metric(
                f"{_lb} inlet pressure",
                f"{_p_in_b:.4f} bara",
                delta=f"{_lb} ΔP = {_gsr_o2['dp_line']:.3f} kPa",
                delta_color="off",
            )
            _dp_stack_kpa  = _dp_stack * 100.0
            _dp_stack_mbar = _dp_stack_kpa * 10.0
            _sk3.metric(
                f"Generator ΔP  ({_la} − {_lb})",
                f"{_dp_stack:.4f} bara",
                delta=f"{_dp_stack_kpa:.2f} kPa  ·  {_dp_stack_mbar:.1f} mbar",
                delta_color="off",
            )

        st.markdown("##### System Pressure Breakdown")
        with st.container(border=True):
            _bk1, _bk2 = st.columns(2)
            with _bk1:
                st.markdown(f"**{_la} system  ({_la} → {_lc} → Separator)**")
                st.metric(f"{_la} inlet",
                          f"{_gsr_h2['P_line_in']:.4f} bara")
                st.metric(f"{_la} outlet / {_lc} inlet",
                          f"{_gsr_h2['P_line_out']:.4f} bara",
                          delta=f"{_la} ΔP = {_gsr_h2['dp_line']:.3f} kPa",
                          delta_color="off")
                st.metric(f"{_la} Separator",
                          f"{_gsr_h2['P_sep']:.4f} bara",
                          delta=f"{_lc}+T ΔP = {_gsr_h2['dp_hdr']:.3f} kPa",
                          delta_color="off")
            with _bk2:
                st.markdown(f"**{_lb} system  ({_lb} → {_ld} → Separator)**")
                st.metric(f"{_lb} inlet",
                          f"{_gsr_o2['P_line_in']:.4f} bara")
                st.metric(f"{_lb} outlet / {_ld} inlet",
                          f"{_gsr_o2['P_line_out']:.4f} bara",
                          delta=f"{_lb} ΔP = {_gsr_o2['dp_line']:.3f} kPa",
                          delta_color="off")
                st.metric(f"{_lb} Separator",
                          f"{_gsr_o2['P_sep']:.4f} bara",
                          delta=f"{_ld}+T ΔP = {_gsr_o2['dp_hdr']:.3f} kPa",
                          delta_color="off")

        st.divider()
        st.markdown("##### Apply Results to Cases")
        st.caption(
            "Write the required inlet pressures back into Case A and Case B "
            "so the branch calculations reflect the goal-seek solution."
        )
        _ap1, _ap2 = st.columns(2)
        if _ap1.button(
            f"Apply {_p_in_a:.4f} bara → {_la}",
            use_container_width=True, key="stack_apply_a",
        ):
            st.session_state["stack_apply_a_pending"] = float(_p_in_a)
            st.rerun()
        if _ap2.button(
            f"Apply {_p_in_b:.4f} bara → {_lb}",
            use_container_width=True, key="stack_apply_b",
        ):
            st.session_state["stack_apply_b_pending"] = float(_p_in_b)
            st.rerun()

# ============================================================================
# DN STUDY TAB
# ============================================================================
with tab_dn:
    st.subheader("DN Study — Branch Line Size Comparison")
    st.caption(
        "Re-runs the full system (branches A and B + goal-seek) with a different branch DN. "
        "Header sizes are unchanged. All other inputs (flows, pressure, correlation) are identical."
    )

    _prereq_ok = (
        results_a is not None and results_b is not None
        and results_c is not None and results_d is not None
    )
    _gsr_ok = (
        st.session_state.get("stack_gsr_h2") is not None
        and st.session_state.get("stack_gsr_o2") is not None
    )

    if not _prereq_ok:
        st.warning("Run all four cases (A, B, C, D) before using DN Study.")
    elif not _gsr_ok:
        st.warning("Run the Generator ΔP calculation before using DN Study.")
    else:
        _dn_primary = results_a["segments"][0]["dn"] if results_a.get("segments") else "DN50"
        _dn_all_opts = list(engine.PIPE_DATABASE.keys())
        _dn_alt_opts = [dn for dn in _dn_all_opts if dn != _dn_primary]
        _dn_default_idx = _dn_alt_opts.index("DN40") if "DN40" in _dn_alt_opts else 0

        _inp_col, _res_col = st.columns([1, 2])

        with _inp_col:
            with st.container(border=True):
                st.markdown("**Study Settings**")
                dn_alt = st.selectbox(
                    "Alternative branch DN", _dn_alt_opts,
                    index=_dn_default_idx, key="dn_study_alt_sel"
                )
                _p_sep_h2_dn = st.session_state.get("stack_sep_h2", 16.5)
                _p_sep_o2_dn = st.session_state.get("stack_sep_o2", 16.5)
                st.caption(
                    f"Separator targets (from Generator ΔP tab):  "
                    f"H₂ {_p_sep_h2_dn:.2f} bara  ·  O₂ {_p_sep_o2_dn:.2f} bara"
                )
                _run_dn_study = st.button(
                    "Run DN Study", type="primary",
                    use_container_width=True, key="dn_study_run"
                )

        if _run_dn_study:
            with st.spinner(f"Computing {_dn_primary} vs {dn_alt}…"):
                _ra_alt = _apply_dn_override(results_a, dn_alt)
                _rb_alt = _apply_dn_override(results_b, dn_alt)
                _gsr_h2_alt = _goal_seek_stack(_ra_alt, results_c, _p_sep_h2_dn)
                _gsr_o2_alt = _goal_seek_stack(_rb_alt, results_d, _p_sep_o2_dn)
            st.session_state["dn_study_dn_primary"] = _dn_primary
            st.session_state["dn_study_dn_alt"]     = dn_alt
            st.session_state["dn_study_gsr_h2_alt"] = _gsr_h2_alt
            st.session_state["dn_study_gsr_o2_alt"] = _gsr_o2_alt
            st.rerun()

        _gsr_h2_p = st.session_state.get("stack_gsr_h2")
        _gsr_o2_p = st.session_state.get("stack_gsr_o2")
        _gsr_h2_a = st.session_state.get("dn_study_gsr_h2_alt")
        _gsr_o2_a = st.session_state.get("dn_study_gsr_o2_alt")
        _dn_p_lbl = st.session_state.get("dn_study_dn_primary", _dn_primary)
        _dn_a_lbl = st.session_state.get("dn_study_dn_alt", "")

        if _gsr_h2_a and _gsr_o2_a and _dn_a_lbl:
            with _res_col:
                # ── Generator ΔP ─────────────────────────────────────────────
                _dp_gen_p_mbar = (_gsr_h2_p["P_line_in"] - _gsr_o2_p["P_line_in"]) * 1000.0
                _dp_gen_a_mbar = (_gsr_h2_a["P_line_in"] - _gsr_o2_a["P_line_in"]) * 1000.0
                _dp_gen_delta  = _dp_gen_a_mbar - _dp_gen_p_mbar

                with st.container(border=True):
                    st.markdown("**Generator ΔP**")
                    _gc1, _gc2, _gc3 = st.columns(3)
                    _gc1.metric(f"{_dn_p_lbl} (primary)", f"{_dp_gen_p_mbar:.1f} mbar")
                    _gc2.metric(
                        f"{_dn_a_lbl} (alternative)",
                        f"{_dp_gen_a_mbar:.1f} mbar",
                        delta=f"{_dp_gen_delta:+.1f} mbar",
                        delta_color="off",
                    )
                    _winner = _dn_p_lbl if abs(_dp_gen_p_mbar) <= abs(_dp_gen_a_mbar) else _dn_a_lbl
                    _gc3.metric("Lower |ΔP|", _winner)

                # ── ΔP comparison table ───────────────────────────────────────
                _dp_rows = []
                for _case_lbl, _dp_p_kpa, _dp_a_kpa in [
                    (f"{_la} branch", _gsr_h2_p["dp_line"], _gsr_h2_a["dp_line"]),
                    (f"{_lb} branch", _gsr_o2_p["dp_line"], _gsr_o2_a["dp_line"]),
                    (f"{_la} header", _gsr_h2_p["dp_hdr"],  _gsr_h2_a["dp_hdr"]),
                    (f"{_lb} header", _gsr_o2_p["dp_hdr"],  _gsr_o2_a["dp_hdr"]),
                ]:
                    _pct = ((_dp_a_kpa - _dp_p_kpa) / _dp_p_kpa * 100
                            if _dp_p_kpa and abs(_dp_p_kpa) > 1e-9 else 0.0)
                    _dp_rows.append({
                        "Case":                      _case_lbl,
                        f"{_dn_p_lbl} ΔP (kPa)":    round(_dp_p_kpa, 3),
                        f"{_dn_a_lbl} ΔP (kPa)":    round(_dp_a_kpa, 3),
                        "Change (%)":                f"{_pct:+.1f}",
                    })
                st.dataframe(
                    pd.DataFrame(_dp_rows),
                    hide_index=True, use_container_width=True,
                    column_config={
                        f"{_dn_p_lbl} ΔP (kPa)": st.column_config.NumberColumn(format="%.3f"),
                        f"{_dn_a_lbl} ΔP (kPa)": st.column_config.NumberColumn(format="%.3f"),
                    }
                )

                # ── Velocity estimate (first segment, ID-ratio scaling) ───────
                _seg0      = results_a["segments"][0]
                _pn0       = _seg0["pn"]
                _lined0    = _seg0.get("lined", False)
                _lthk0_m   = _seg0.get("liner_thickness_mm", 1.0) / 1000.0
                _D_p_bore  = engine.PIPE_DATABASE[_dn_p_lbl].get(_pn0, list(engine.PIPE_DATABASE[_dn_p_lbl].values())[0])
                _D_a_bore  = engine.PIPE_DATABASE[_dn_a_lbl].get(_pn0, list(engine.PIPE_DATABASE[_dn_a_lbl].values())[0])
                _D_p_eff   = _D_p_bore - 2 * _lthk0_m if _lined0 else _D_p_bore
                _D_a_eff   = _D_a_bore - 2 * _lthk0_m if _lined0 else _D_a_bore
                _vel_scale = (_D_p_eff / _D_a_eff) ** 2 if _D_a_eff > 0 else 1.0

                _rec_a0 = results_a["grid_records"][0] if results_a.get("grid_records") else {}
                _rec_b0 = results_b["grid_records"][0] if results_b.get("grid_records") else {}
                _vm_a_p = float(_rec_a0.get("V_m (m/s)", 0))
                _vm_b_p = float(_rec_b0.get("V_m (m/s)", 0))
                _ve_a   = float(_rec_a0.get("V_e (m/s)", 0))
                _ve_b   = float(_rec_b0.get("V_e (m/s)", 0))
                _vm_a_a = _vm_a_p * _vel_scale
                _vm_b_a = _vm_b_p * _vel_scale
                _ratio_a = _vm_a_a / _ve_a if _ve_a > 0 else 0.0
                _ratio_b = _vm_b_a / _ve_b if _ve_b > 0 else 0.0

                with st.container(border=True):
                    st.markdown("**Inlet velocity — first segment (estimated)**")
                    _vc1, _vc2 = st.columns(2)
                    with _vc1:
                        st.markdown(f"*{_la} branch*")
                        st.metric(f"{_dn_p_lbl}", f"{_vm_a_p:.2f} m/s",
                                  help=f"V_m/V_e = {_vm_a_p/_ve_a:.2f}" if _ve_a > 0 else None)
                        _dc_a = "inverse" if _ratio_a > 1.0 else ("normal" if _ratio_a > 0.8 else "off")
                        st.metric(f"{_dn_a_lbl}", f"{_vm_a_a:.2f} m/s",
                                  delta=f"V_m/V_e = {_ratio_a:.2f}", delta_color=_dc_a)
                    with _vc2:
                        st.markdown(f"*{_lb} branch*")
                        st.metric(f"{_dn_p_lbl}", f"{_vm_b_p:.2f} m/s",
                                  help=f"V_m/V_e = {_vm_b_p/_ve_b:.2f}" if _ve_b > 0 else None)
                        _dc_b = "inverse" if _ratio_b > 1.0 else ("normal" if _ratio_b > 0.8 else "off")
                        st.metric(f"{_dn_a_lbl}", f"{_vm_b_a:.2f} m/s",
                                  delta=f"V_m/V_e = {_ratio_b:.2f}", delta_color=_dc_b)
                    st.caption(
                        f"Velocity scales as (ID ratio)²: "
                        f"{_dn_p_lbl} {_D_p_eff*1000:.1f} mm → {_dn_a_lbl} {_D_a_eff*1000:.1f} mm  "
                        f"·  factor {_vel_scale:.2f}×.  "
                        f"Erosion limit V_e from primary case (API RP 14E, C = 100)."
                    )

                # ── Recommendation ────────────────────────────────────────────
                _vel_ok  = _ratio_a <= 1.0 and _ratio_b <= 1.0
                _dp_alt_better = abs(_dp_gen_a_mbar) < abs(_dp_gen_p_mbar)
                if _dp_alt_better and _vel_ok:
                    st.success(
                        f"**{_dn_a_lbl}** gives lower Generator |ΔP| "
                        f"({_dp_gen_a_mbar:.1f} vs {_dp_gen_p_mbar:.1f} mbar) "
                        f"with acceptable velocities."
                    )
                elif _dp_alt_better and not _vel_ok:
                    st.warning(
                        f"**{_dn_a_lbl}** gives lower Generator |ΔP| but estimated "
                        f"V_m/V_e exceeds 1.0 — verify erosion before selecting."
                    )
                else:
                    st.info(
                        f"**{_dn_p_lbl}** (primary) gives lower Generator |ΔP|. "
                        f"{_dn_a_lbl} appears oversized for this duty."
                    )

                # ── Report ────────────────────────────────────────────────────
                st.divider()
                _dn_rpt_col1, _dn_rpt_col2 = st.columns(2)
                with _dn_rpt_col1:
                    if st.button("Generate DN Study Report", use_container_width=True,
                                 key="dn_study_gen_rpt"):
                        try:
                            _dn_buf = report_generator.generate_dn_study_report(
                                dn_primary=_dn_p_lbl,
                                dn_alt=_dn_a_lbl,
                                label_a=_la, label_b=_lb,
                                gsr_h2_primary=_gsr_h2_p,
                                gsr_o2_primary=_gsr_o2_p,
                                gsr_h2_alt=_gsr_h2_a,
                                gsr_o2_alt=_gsr_o2_a,
                                dp_gen_primary_mbar=_dp_gen_p_mbar,
                                dp_gen_alt_mbar=_dp_gen_a_mbar,
                                vel_data={
                                    "vm_a_primary": _vm_a_p, "vm_b_primary": _vm_b_p,
                                    "vm_a_alt":     _vm_a_a, "vm_b_alt":     _vm_b_a,
                                    "ve_a":         _ve_a,   "ve_b":         _ve_b,
                                    "D_p_mm":       _D_p_eff * 1000,
                                    "D_a_mm":       _D_a_eff * 1000,
                                    "vel_scale":    _vel_scale,
                                },
                                p_sep_h2=_p_sep_h2_dn,
                                p_sep_o2=_p_sep_o2_dn,
                            )
                            st.session_state["dn_study_rpt_bytes"] = _dn_buf.getvalue()
                        except Exception as _re:
                            st.error(f"Report failed: {_re}")
                with _dn_rpt_col2:
                    if st.session_state.get("dn_study_rpt_bytes"):
                        st.download_button(
                            f"Download  {_dn_p_lbl}_vs_{_dn_a_lbl}.docx",
                            data=st.session_state["dn_study_rpt_bytes"],
                            file_name=f"dn_study_{_dn_p_lbl}_vs_{_dn_a_lbl}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True, key="dn_study_dl_rpt",
                        )
