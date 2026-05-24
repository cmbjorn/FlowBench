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
from fluids.two_phase import (Taitel_Dukler_regime as _TD_regime,
                               Mandhane_Gregory_Aziz_regime as _MGA_regime)

st.set_page_config(
    page_title="FlowBench — General Flow Workbench",
    page_icon="⚙️",
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

st.title("FlowBench")

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
        for key in ("P_bara", "T_C", "gas_species_widget", "correlation", "voidage_method",
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
    st.header("FlowBench")
    with st.expander("About & Capabilities", expanded=False):
        st.markdown("""
        Two-phase steady-state hydraulic workbench — pressure drop for any
        gas–liquid combination that CoolProp can handle.

        **Flow modes**
        - **Gas + Liquid** — multi-component gas and liquid, each as individual
          species (kg/h). Equilibrium flash (Peng-Robinson EOS) at inlet.
        - **Saturated / VLE** — pure fluid at saturation; CoolProp derives phase
          properties from the saturation curve at each segment.

        **Workflow:** tabs **A / B** for individual lines, **Header A / B** for
        collecting manifolds, **Compare** for overlay and uncertainty sweep (12
        correlation × void-fraction combinations). Export Word or Excel from any tab.

        **Correlations** — Beggs-Brill (default), Friedel, Lockhart-Martinelli,
        Müller-Steinhagen & Heck, Chisholm, Kim-Mudawar.

        **Pipe library** — DN40–DN250, PN20/25/40, 5 materials, optional liner
        (PTFE, FEP, PFA, PVDF). 17 fitting types (Crane TP-410).

        **Accuracy** — ±20–30 % for non-hydrocarbon fluids. Validate against
        commissioning data before design decisions.
        """)
    with st.expander("Model details", expanded=False):
        st.markdown("""
        **Assumptions**
        1. Ideal gas behaviour for gas mixture
        2. Continuous liquid phase; no flooding or flow inversion
        3. Bore = f(DN, PN) only — ANSI B36.10/19, material-independent
        4. Lined segments: effective ID = metal bore − 2 × liner thickness
        5. Pressure marching: gas density re-evaluated at each segment inlet
        6. Steady-state only — no transient effects
        7. Void fraction: homogeneous α = (x/ρg)/(x/ρg+(1−x)/ρl), or Rouhani-1

        **Flow regimes**
        Horizontal: Stratified / Intermittent / Annular-Dispersed (Taitel-Dukler, Mandhane)
        Vertical up: Bubble-Slug / Churn-Annular (Wallis/Taitel)
        Vertical down: Falling Film / Annular

        **Liquid species** — CoolProp backed (IAPWS-IF97 for Water; DIPPR / REFPROP for
        organics and refrigerants). Mixture: ρ and σ mass-weighted, μ log-mean.

        **References** — Beggs & Brill (1973) SPE-4007-PA · CoolProp · fluids library ·
        Crane TP-410 (2013) · API RP 14E (2007)
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
                _sb_vi   = _sb_case["inputs"]
                _sb_mode = _sb_case.get("mode", "gas_liquid")
                _sb_D    = engine.PIPE_DATABASE[_sb_vi["pipe_dn"]][_sb_vi["pipe_pn"]]
                _sb_r    = engine.MATERIAL_ROUGHNESS["SS316L"]
                _sb_dp   = 0.0
                _sb_P_Pa = _sb_vi["P_bara"] * 1e5
                try:
                    for _sb_seg in _sb_vi["segments"]:
                        _sb_ang = {"Horizontal": 0.0, "Vertical Upflow": np.pi/2,
                                   "Vertical Downflow": -np.pi/2}[_sb_seg["type"]]
                        _sb_le  = 0.0
                        if _sb_seg.get("fittings", "None") in engine.FITTING_Le_over_D:
                            _sb_le = (engine.FITTING_Le_over_D[_sb_seg["fittings"]]
                                      * _sb_D * _sb_seg.get("fitting_count", 0))
                        if _sb_mode == "vle":
                            _sb_vp = engine.calculate_vle_properties(
                                _sb_vi["vle_fluid"], _sb_P_Pa / 1e5,
                                _sb_vi["vle_x_mass"], _sb_vi["vle_m_total_kgs"])
                        else:
                            _sb_vp = engine.calculate_two_phase_properties(
                                _sb_P_Pa / 1e5, _sb_vi["T_C"],
                                _sb_vi.get("gas_flows_kgh", {}),
                                "Custom", 0.0,
                                liquid_flows_kgh=_sb_vi.get("liquid_flows_kgh"))
                        _sb_res = engine.calculate_segment_pressure_drop(
                            _sb_vp, _sb_D, _sb_r, _sb_seg["length"] + _sb_le, _sb_ang)
                        _sb_dp  += _sb_res["dP_Pa"]
                        _sb_P_Pa = max(1e4, _sb_P_Pa - _sb_res["dP_Pa"])
                    _sb_calc = _sb_dp / 1000.0
                    _sb_exp  = _sb_case["expected_total_dp_kpa"]
                    _sb_tol  = _sb_case.get("tolerance_pct", 5.0)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Calculated", f"{_sb_calc:.3f} kPa")
                    if _sb_exp is None:
                        r2.metric("Expected", "—  (auto-calib.)")
                        r3.metric("Deviation", "—")
                        st.info(f"Auto-calibrated case: {_sb_calc:.3f} kPa (no reference to compare).")
                    else:
                        _sb_err  = abs(_sb_calc - _sb_exp) / abs(_sb_exp) * 100.0
                        _sb_pass = _sb_err <= _sb_tol
                        r2.metric("Expected",  f"{_sb_exp:.3f} kPa")
                        r3.metric("Deviation", f"{_sb_err:.2f} %",
                                  delta=f"Pass (≤{_sb_tol:.0f}%)" if _sb_pass else f"Fail (>{_sb_tol:.0f}%)",
                                  delta_color="normal" if _sb_pass else "inverse")
                        if _sb_pass:
                            st.success(f"Passed — {_sb_err:.2f}% within ±{_sb_tol:.0f}%")
                        else:
                            st.warning(f"Failed — {_sb_err:.2f}% exceeds ±{_sb_tol:.0f}%")
                except Exception as _sb_err_ex:
                    st.error(f"Validation run failed: {_sb_err_ex}")

    st.divider()
    st.header("Session")

    # ── Case labels ───────────────────────────────────────────────────────────
    _slcol1, _slcol2 = st.columns(2)
    _slcol1.text_input("Tab A label", key="label_a", max_chars=20,
                       help="Short name used in tabs, charts, and reports.")
    _slcol2.text_input("Tab B label", key="label_b", max_chars=20,
                       help="Short name used in tabs, charts, and reports.")

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

_DEFAULT_SEGMENTS = [
    {"type": "Horizontal",      "dn": "DN50", "pn": "PN40", "material": "SS316L",
     "length": 20.0,
     "fittings_list": [{"type": "90° Standard Elbow", "qty": 2}],
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
    {"type": "Vertical Upflow", "dn": "DN50", "pn": "PN40", "material": "SS316L",
     "length": 10.0,
     "fittings_list": [],
     "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0},
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


@st.cache_data(show_spinner="Computing regime map…")
def _compute_regime_grid(
    rhol: float, rhog: float, mul: float, mug: float,
    sigma: float, D: float, roughness: float, use_horiz: bool,
) -> tuple:
    """
    Sweep a 50×50 log-log V_sl × V_sg grid and classify each cell using
    exactly the same correlations as the engine (Taitel-Dukler 1976 +
    Mandhane-Gregory-Aziz 1974 for horizontal; Wallis + void-fraction
    thresholds for vertical upflow).

    Returns (td_grid, full_grid, vsl_list, vsg_list) — all plain Python
    lists so Streamlit can cache and serialise them cleanly.
    """
    N = 50
    _g = 9.80665
    vsl_arr = np.logspace(-3, 1, N)   # V_sl: 0.001 → 10 m/s
    vsg_arr = np.logspace(-3, 2, N)   # V_sg: 0.001 → 100 m/s
    A = np.pi / 4 * D ** 2

    td_grid   = [[""] * N for _ in range(N)]   # coloring key (TD regime or vert label)
    full_grid = [[""] * N for _ in range(N)]   # hover label

    if use_horiz:
        for i, vsg in enumerate(vsg_arr):
            for j, vsl in enumerate(vsl_arr):
                ml = vsl * rhol * A
                mg = vsg * rhog * A
                m  = ml + mg
                x  = mg / m if m > 0 else 0.0
                try:
                    td  = _TD_regime(m=m, x=x, rhol=rhol, rhog=rhog,
                                     mul=mul, mug=mug, D=D,
                                     angle=0.0, roughness=roughness)[0]
                    mga = _MGA_regime(m=m, x=x, rhol=rhol, rhog=rhog,
                                      mul=mul, mug=mug, sigma=sigma, D=D)[0]
                    td_grid[i][j]   = td
                    full_grid[i][j] = f"{td} / {mga}"
                except Exception:
                    td_grid[i][j]   = "intermittent"
                    full_grid[i][j] = "intermittent / slug"
    else:
        # Vertical upflow: Wallis annular onset + homogeneous void fraction
        # (mirrors the engine's _classify_regime logic for |θ|≥75°)
        try:
            V_ann = 3.1 * (_g * sigma * (rhol - rhog) / rhog ** 2) ** 0.25
        except Exception:
            V_ann = 1e9
        for i, vsg in enumerate(vsg_arr):
            for j, vsl in enumerate(vsl_arr):
                mg = vsg * rhog * A
                ml = vsl * rhol * A
                m  = ml + mg
                x  = mg / m if m > 0 else 0.0
                alpha_h = vsg / (vsg + vsl)   # homogeneous void fraction
                if x > 0.90:
                    reg = "mist / annular"
                elif vsg >= V_ann:
                    reg = "annular"
                elif alpha_h >= 0.52:
                    reg = "churn"
                elif alpha_h >= 0.25:
                    reg = "slug"
                else:
                    reg = "bubble"
                td_grid[i][j]   = reg
                full_grid[i][j] = reg

    return td_grid, full_grid, vsl_arr.tolist(), vsg_arr.tolist()


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
        st.session_state[k("gas_species_widget")] = ["Air"]
    if k("liquid_species_widget") not in st.session_state:
        st.session_state[k("liquid_species_widget")] = ["Water"]
    if k("flow_mode") not in st.session_state:
        st.session_state[k("flow_mode")] = "gas_liquid"
    if k("vle_fluid_widget") not in st.session_state:
        st.session_state[k("vle_fluid_widget")] = "Water"
    if k("P_bara") not in st.session_state:
        st.session_state[k("P_bara")] = 10.0
    if k("T_C") not in st.session_state:
        st.session_state[k("T_C")] = 25.0
    if k("vle_m_kgs_widget") not in st.session_state:
        st.session_state[k("vle_m_kgs_widget")] = 1.0
    if k("vle_x_widget") not in st.session_state:
        st.session_state[k("vle_x_widget")] = 0.3

    # Migrate segments
    for _seg in st.session_state[k("segments")]:
        _seg.setdefault("kind", "pipe")
        _seg.setdefault("dn", "DN50"); _seg.setdefault("pn", "PN40")
        _seg.setdefault("material", "SS316L"); _seg.setdefault("lined", False)
        _seg.setdefault("liner_material", "FEP"); _seg.setdefault("liner_thickness_mm", 1.0)
        if _seg.get("kind", "pipe") == "pipe":
            if _seg["material"] not in _VALID_MATS:    _seg["material"] = "SS316L"
            if _seg["liner_material"] not in _VALID_LINERS: _seg["liner_material"] = "FEP"

    col_in, col_out = st.columns([1, 1.2])

    # ── INPUTS ────────────────────────────────────────────────────────────────
    with col_in:
        st.subheader("Inputs")

        # Flow mode selector
        flow_mode = st.radio(
            "Flow mode",
            ["Gas + Liquid", "Two-phase Saturated (VLE)"],
            horizontal=True,
            key=k("flow_mode_radio"),
            index=0 if st.session_state.get(k("flow_mode"), "gas_liquid") == "gas_liquid" else 1,
            help="**Gas + Liquid**: separate gas mixture and liquid stream.  "
                 "**Two-phase Saturated (VLE)**: single pure fluid at saturation — "
                 "CoolProp derives both phase properties from the saturation curve at each segment.",
        )
        st.session_state[k("flow_mode")] = "gas_liquid" if flow_mode == "Gas + Liquid" else "vle"
        _is_vle = (st.session_state[k("flow_mode")] == "vle")

        # Inlet Conditions
        with st.container(border=True):
            st.markdown("**Inlet Conditions**")
            if _is_vle:
                P_bara = st.number_input("Inlet Pressure (bara)", min_value=0.1, max_value=300.0,
                                         step=1.0, key=k("P_bara"))
                T_C = None  # derived from saturation in VLE mode
            else:
                p1, p2 = st.columns(2)
                P_bara = p1.number_input("Inlet Pressure (bara)", min_value=1.0, max_value=300.0,
                                         step=1.0, key=k("P_bara"))
                T_C    = p2.number_input("Temperature (°C)", min_value=-60.0, max_value=400.0,
                                         step=5.0, key=k("T_C"))

        # ── VLE input mode ────────────────────────────────────────────────────
        if _is_vle:
            with st.container(border=True):
                st.markdown("**Saturated Fluid (VLE)**")
                _vle_display_list = list(engine.VLE_FLUID_DISPLAY.keys())
                _vle_names_to_id  = engine.VLE_FLUID_DISPLAY
                _vle_saved_fluid  = st.session_state.get(k("vle_fluid_widget"), "Water")
                # Find display name that maps to the saved fluid ID
                _vle_display_saved = next(
                    (dn for dn, fid in _vle_names_to_id.items() if fid == _vle_saved_fluid),
                    _vle_display_list[0])
                _vle_sel_idx = (_vle_display_list.index(_vle_display_saved)
                                if _vle_display_saved in _vle_display_list else 0)
                vle_display_name = st.selectbox(
                    "Fluid", _vle_display_list, index=_vle_sel_idx, key=k("vle_fluid_sel"),
                    help="CoolProp saturation data is used. T_sat is derived from inlet pressure.")
                vle_fluid_id = _vle_names_to_id[vle_display_name]
                st.session_state[k("vle_fluid_widget")] = vle_fluid_id

                _vv1, _vv2 = st.columns(2)
                vle_m_kgs = _vv1.number_input(
                    "Total mass flow (kg/s)", min_value=0.001, max_value=500.0,
                    step=0.1, format="%.3f", key=k("vle_m_kgs_widget"))
                vle_x = _vv2.slider(
                    "Inlet quality x  (0 = all liquid, 1 = all vapour)",
                    min_value=0.0, max_value=1.0,
                    step=0.01, key=k("vle_x_widget"))

                # Show derived saturation properties
                try:
                    _vle_props_preview = engine.calculate_vle_properties(
                        vle_fluid_id, P_bara, vle_x, vle_m_kgs)
                    _vc1, _vc2, _vc3, _vc4 = st.columns(4)
                    _vc1.metric("T_sat", f"{_vle_props_preview['T_sat_C']:.1f} °C")
                    _vc2.metric("ρ_liq", f"{_vle_props_preview['rho_l']:.1f} kg/m³")
                    _vc3.metric("ρ_vap", f"{_vle_props_preview['rho_g']:.3f} kg/m³")
                    _vc4.metric("σ", f"{_vle_props_preview['sigma']*1e3:.2f} mN/m")
                except Exception as _vle_err:
                    st.error(f"CoolProp VLE error: {_vle_err}")
                    vle_fluid_id = "Water"
                    vle_m_kgs   = 1.0
                    vle_x       = 0.5

            # Dummy values for non-VLE variables (keep references valid)
            gas_flows_kgh    = {}
            liquid_type      = f"{vle_fluid_id} (VLE)"
            q_lye            = 0.0
            liquid_flows_kgh = None
            custom_gas       = None
            custom_liquid    = None
            use_coolprop     = False
            T_C              = None  # will be set from VLE props

        else:
            # ── Gas + Liquid mode ─────────────────────────────────────────────
            vle_fluid_id = None
            vle_m_kgs    = None
            vle_x        = None

            # Gas Phase
            with st.container(border=True):
                st.markdown("**Gas Phase**")
                # Grouped species selector
                _all_species = list(engine.GAS_SPECIES.keys())
                selected_species = st.multiselect(
                    "Gas species  (select one or more)",
                    _all_species, key=k("gas_species_widget"),
                    help="All species are CoolProp-backed. "
                         "Categories: Common Process, Hydrocarbons, Refrigerants.")
                if not selected_species:
                    st.info("Select at least one gas species, or switch to Saturated / VLE mode.")
                    selected_species = []

                gas_flows_kgh = {}
                if selected_species:
                    _ncols = min(len(selected_species), 3)
                    _fcols = st.columns(_ncols)
                    for _ci, _sp in enumerate(selected_species):
                        _fkey = k(f"gflow_{_sp}")
                        if _fkey not in st.session_state:
                            st.session_state[_fkey] = 100.0
                        gas_flows_kgh[_sp] = _fcols[_ci % _ncols].number_input(
                            f"{_sp}  (kg/h)", min_value=0.0, step=0.1, key=_fkey)

                custom_gas = None
                if "Custom" in selected_species:
                    st.markdown("*Custom gas properties*")
                    _cg1, _cg2 = st.columns(2)
                    _cg_mw = _cg1.number_input("MW (g/mol)", min_value=1.0, value=28.0,
                                                step=1.0, key=k("cg_mw"))
                    _cg_mu = _cg2.number_input("μ (µPa·s)", min_value=1.0, value=18.5,
                                                step=0.5, key=k("cg_mu"))
                    custom_gas = {"MW_gmol": _cg_mw, "mu_upas": _cg_mu}

                # Use CoolProp for gas mixture when at least one species has a CoolProp ID
                use_coolprop = any(engine.GAS_SPECIES.get(sp, {}).get("coolprop_id")
                                   for sp in selected_species if sp != "Custom")

            # Liquid Phase
            with st.container(border=True):
                st.markdown("**Liquid Phase**")
                _liq_options = list(engine.LIQUID_COOLPROP_ID.keys())
                liquid_species = st.multiselect(
                    "Liquid species  (select one or more)",
                    _liq_options,
                    key=k("liquid_species_widget"),
                    help="All species use CoolProp. Mixture properties are mass-weighted. "
                         "Leave empty for single-phase gas (Darcy-Weisbach fallback).",
                )

                if not liquid_species:
                    st.caption("No liquid selected — calculating as **single-phase gas** (Darcy-Weisbach).")

                liquid_flows_kgh = {}
                if liquid_species:
                    _lncols = min(len(liquid_species), 3)
                    _lfcols = st.columns(_lncols)
                    for _li, _ls in enumerate(liquid_species):
                        _lfk = k(f"lflow_{_ls}")
                        if _lfk not in st.session_state:
                            st.session_state[_lfk] = 1000.0
                        _lf = _lfcols[_li % _lncols].number_input(
                            f"{_ls}  (kg/h)", min_value=0.0, step=100.0, key=_lfk,
                        )
                        if _lf > 0:
                            liquid_flows_kgh[_ls] = _lf

                if liquid_flows_kgh:
                    try:
                        _T_K  = (T_C + 273.15) if T_C is not None else 298.15
                        _P_pa = P_bara * 1e5
                        _rl, _mul, _sigl = engine.liquid_mixture_props(
                            liquid_flows_kgh, _T_K, _P_pa)
                        _lm1, _lm2, _lm3 = st.columns(3)
                        _lm1.metric("ρ_mix", f"{_rl:.1f} kg/m³")
                        _lm2.metric("μ_mix", f"{_mul*1e3:.3f} mPa·s")
                        _lm3.metric("σ_mix", f"{_sigl*1e3:.2f} mN/m")
                        # Compute q_lye for header/helper compat
                        q_lye = sum(liquid_flows_kgh.values()) / _rl
                        if len(liquid_flows_kgh) == 1:
                            liquid_type   = next(iter(liquid_flows_kgh))
                            custom_liquid = None
                        else:
                            liquid_type   = "Custom"
                            custom_liquid = {"rho_kgm3": _rl,
                                             "mu_mpas":  _mul * 1e3,
                                             "sigma_mnm": _sigl * 1e3}
                    except Exception:
                        q_lye         = 0.0
                        liquid_type   = "Custom"
                        custom_liquid = None
                else:
                    q_lye         = 0.0
                    liquid_type   = "Custom"
                    custom_liquid = None

        # Pipe Geometry
        with st.container(border=True):
            st.markdown("**Pipe Geometry**")
            current_specs = []
            DN_OPTIONS    = list(engine.PIPE_DATABASE.keys())
            PN_OPTIONS    = ["PN20", "PN25", "PN40"]
            MAT_OPTIONS   = list(engine.MATERIAL_ROUGHNESS.keys())
            LINER_OPTIONS = list(engine.LINER_ROUGHNESS.keys())
            for i, seg in enumerate(st.session_state[k("segments")]):
                _seg_kind = seg.get("kind", "pipe")
                _kind_labels = {"pipe": "Pipe segment", "valve": "Valve", "hx": "Heat exchanger"}
                _seg_hdr_col, _seg_del_col = st.columns([9, 1])
                _seg_hdr_col.markdown(f"**#{i+1} — {_kind_labels.get(_seg_kind, 'Pipe segment')}**")
                if len(st.session_state[k("segments")]) > 1:
                    if _seg_del_col.button("×", key=k(f"del_seg_{i}"), help="Remove this segment"):
                        st.session_state[k("segments")].pop(i)
                        st.rerun()

                # ── VALVE ──────────────────────────────────────────────────────
                if _seg_kind == "valve":
                    _vm1, _vm2, _vm3, _vm4 = st.columns([1.2, 1.0, 1.0, 1.0])
                    _v_mode = _vm1.radio(
                        "Mode", ["Specify ΔP → Kv", "Specify Kv → ΔP"],
                        key=k(f"v_mode_{i}"),
                        index=0 if seg.get("valve_mode", "dp") == "dp" else 1,
                        horizontal=False,
                        help="ΔP→Kv: budget a pressure drop, get required Kv (line sizing).\n"
                             "Kv→ΔP: enter a known Kv, get the pressure drop.")
                    _v_dn = _vm2.selectbox("Nominal DN", DN_OPTIONS, key=k(f"v_dn_{i}"),
                                           index=DN_OPTIONS.index(seg.get("dn", "DN50")))
                    _v_pn = _vm3.selectbox("PN", PN_OPTIONS, key=k(f"v_pn_{i}"),
                                           index=PN_OPTIONS.index(seg.get("pn", "PN40")))
                    _v_char = _vm4.selectbox(
                        "Characteristic", ["equal-percentage", "linear"],
                        key=k(f"v_char_{i}"),
                        index=["equal-percentage", "linear"].index(
                            seg.get("characteristic", "equal-percentage")),
                        help="Equal-percentage (R=50): Kv_eff = Kv_rated × 50^(f−1).  "
                             "Linear: Kv_eff = Kv_rated × f.")

                    _v_mode_key = "dp" if _v_mode.startswith("Specify ΔP") else "kv"

                    if _v_mode_key == "dp":
                        _vc1, _vc2 = st.columns([1.2, 1.0])
                        _v_dp_kpa = _vc1.number_input(
                            "Target ΔP  (kPa)", min_value=0.1, max_value=10000.0,
                            value=float(seg.get("dp_kpa", 50.0)), step=5.0, format="%.1f",
                            key=k(f"v_dp_{i}"),
                            help="Pressure drop budget allocated to this valve.")
                        _v_open = _vc2.slider(
                            "Opening (%)", min_value=1, max_value=100,
                            value=int(seg.get("opening_pct", 100)),
                            key=k(f"v_open_{i}"),
                            help="Valve opening at operating point.")
                        _v_kv = seg.get("Kv_m3h", 10.0)  # kept for storage, computed at runtime
                        st.caption(f"Required Kv calculated from ΔP = {_v_dp_kpa:.1f} kPa "
                                   f"at {_v_open} % opening — shown in results.")
                    else:
                        _vc1, _vc2 = st.columns([1.2, 1.0])
                        _v_kv = _vc1.number_input(
                            "Kv rated  (m³/h·bar^0.5)", min_value=0.001, max_value=50000.0,
                            value=float(seg.get("Kv_m3h", 10.0)), step=1.0, format="%.3f",
                            key=k(f"v_kv_{i}"),
                            help="Valve flow coefficient at full-open. "
                                 "Kv = Cv / 1.156")
                        _v_open = _vc2.slider(
                            "Opening (%)", min_value=1, max_value=100,
                            value=int(seg.get("opening_pct", 100)),
                            key=k(f"v_open_{i}"),
                            help="Valve opening position.")
                        _f_open = max(0.001, _v_open / 100.0)
                        _kv_eff_p = (_v_kv * (50.0 ** (_f_open - 1.0))
                                     if _v_char == "equal-percentage"
                                     else _v_kv * _f_open)
                        _v_dp_kpa = seg.get("dp_kpa", 50.0)  # kept for storage
                        st.caption(
                            f"Kv_eff = **{_kv_eff_p:.3f}**  ·  "
                            f"Cv_rated ≈ {_v_kv * 1.156:.2f}  ·  "
                            f"Cv_eff ≈ {_kv_eff_p * 1.156:.2f}")

                    current_specs.append({
                        "kind": "valve", "dn": _v_dn, "pn": _v_pn,
                        "valve_mode": _v_mode_key,
                        "Kv_m3h": float(_v_kv), "dp_kpa": float(_v_dp_kpa),
                        "opening_pct": float(_v_open),
                        "characteristic": _v_char,
                    })
                    continue

                # ── HEAT EXCHANGER ─────────────────────────────────────────────
                elif _seg_kind == "hx":
                    _hg1, _hg2, _hg3 = st.columns([1.0, 1.0, 1.0])
                    _h_dn = _hg1.selectbox("Nominal DN", DN_OPTIONS, key=k(f"hx_dn_{i}"),
                                           index=DN_OPTIONS.index(seg.get("dn", "DN50")))
                    _h_pn = _hg2.selectbox("PN", PN_OPTIONS, key=k(f"hx_pn_{i}"),
                                           index=PN_OPTIONS.index(seg.get("pn", "PN40")))
                    _h_dir = _hg3.selectbox(
                        "Flow direction", ["Horizontal", "Vertical Upflow", "Vertical Downflow"],
                        key=k(f"hx_dir_{i}"),
                        index=["Horizontal", "Vertical Upflow", "Vertical Downflow"].index(
                            seg.get("type", "Horizontal")))
                    _hg4, _hg5 = st.columns(2)
                    _h_duty = _hg4.number_input(
                        "Heat duty  (kW)", min_value=-100000.0, max_value=100000.0,
                        value=float(seg.get("duty_kw", 0.0)), step=10.0, format="%.1f",
                        key=k(f"hx_duty_{i}"),
                        help="Heat added to the process stream. "
                             "Positive = heating (temperature rises). "
                             "Negative = cooling (temperature falls). "
                             "Set to 0 if you only want the pressure drop.")
                    _h_dp = _hg5.number_input(
                        "Pressure drop  (kPa)", min_value=0.0, max_value=10000.0,
                        value=float(seg.get("dp_kpa", 20.0)), step=1.0, format="%.1f",
                        key=k(f"hx_dp_{i}"),
                        help="Shell- or tube-side ΔP from the equipment datasheet. "
                             "Enter the ΔP for the stream flowing through this pipeline.")
                    if _h_duty > 0:
                        st.caption(f"Heating duty: +{_h_duty:.1f} kW  ·  ΔP = {_h_dp:.1f} kPa")
                    elif _h_duty < 0:
                        st.caption(f"Cooling duty: {_h_duty:.1f} kW  ·  ΔP = {_h_dp:.1f} kPa")
                    else:
                        st.caption(f"No heat exchange  ·  ΔP = {_h_dp:.1f} kPa")
                    current_specs.append({
                        "kind": "hx", "dn": _h_dn, "pn": _h_pn,
                        "type": _h_dir,
                        "duty_kw": _h_duty, "dp_kpa": _h_dp,
                    })
                    continue

                # ── PIPE SEGMENT ───────────────────────────────────────────────
                g1, g2, g3, g4 = st.columns([1.3, 0.8, 0.7, 0.7])
                t = g1.selectbox("Flow direction",
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
            _ab1, _ab2, _ab3, _ab4 = st.columns(4)
            if _ab1.button("+ Pipe segment", key=k("add_seg"), use_container_width=True):
                _last = st.session_state[k("segments")][-1]
                _last_pipe = next(
                    (s for s in reversed(st.session_state[k("segments")]) if s.get("kind","pipe") == "pipe"),
                    {"dn": "DN50", "pn": "PN40", "material": "SS316L", "lined": False,
                     "liner_material": "FEP", "liner_thickness_mm": 1.0})
                st.session_state[k("segments")].append({
                    "kind": "pipe", "type": "Horizontal",
                    "dn": _last_pipe.get("dn", "DN50"), "pn": _last_pipe.get("pn", "PN40"),
                    "material": _last_pipe.get("material", "SS316L"),
                    "length": 5.0, "fittings_list": [],
                    "lined": _last_pipe.get("lined", False),
                    "liner_material": _last_pipe.get("liner_material", "FEP"),
                    "liner_thickness_mm": _last_pipe.get("liner_thickness_mm", 1.0),
                })
                st.rerun()
            if _ab2.button("+ Valve", key=k("add_valve"), use_container_width=True):
                _last_dn = next(
                    (s.get("dn","DN50") for s in reversed(st.session_state[k("segments")])), "DN50")
                _last_pn = next(
                    (s.get("pn","PN40") for s in reversed(st.session_state[k("segments")])), "PN40")
                st.session_state[k("segments")].append({
                    "kind": "valve", "dn": _last_dn, "pn": _last_pn,
                    "Kv_m3h": 10.0, "opening_pct": 100.0,
                    "characteristic": "equal-percentage",
                })
                st.rerun()
            if _ab3.button("+ Heat exchanger", key=k("add_hx"), use_container_width=True):
                _last_dn = next(
                    (s.get("dn","DN50") for s in reversed(st.session_state[k("segments")])), "DN50")
                _last_pn = next(
                    (s.get("pn","PN40") for s in reversed(st.session_state[k("segments")])), "PN40")
                st.session_state[k("segments")].append({
                    "kind": "hx", "dn": _last_dn, "pn": _last_pn,
                    "type": "Horizontal", "duty_kw": 0.0, "dp_kpa": 20.0,
                })
                st.rerun()
            if _ab4.button("− Remove last", key=k("rem_seg"), use_container_width=True) and \
               len(st.session_state[k("segments")]) > 1:
                st.session_state[k("segments")].pop()
                st.rerun()

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

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    with col_out:
        _key_results_ph = st.empty()   # filled after segment loop — ΔP summary

        # Initialise effective liquid state (overwritten in gas+liquid path below)
        _eff_liquid_flows = liquid_flows_kgh
        _eff_gas_flows    = gas_flows_kgh
        _eff_liq_type     = liquid_type
        _eff_q_lye        = q_lye
        _eff_custom_liq   = custom_liquid

        if _is_vle:
            try:
                props = engine.calculate_vle_properties(
                    vle_fluid_id, P_bara, vle_x, vle_m_kgs)
                T_C = props["T_sat_C"]
            except Exception as _vle_calc_err:
                st.error(f"VLE calculation failed: {_vle_calc_err}")
                st.stop()
        else:
            is_valid, warn_list = engine.validate_input_bounds(
                P_bara, T_C, gas_flows_kgh, liquid_type, q_lye,
                liquid_flows_kgh=liquid_flows_kgh)
            for w in warn_list:
                st.warning(w)

            # ── Equilibrium Flash (Peng-Robinson EOS) ────────────────────────
            _flash = engine.flash_pt(gas_flows_kgh, liquid_type, q_lye, T_C, P_bara,
                                     liquid_flows_kgh=liquid_flows_kgh)

            if _flash["feasible"]:
                _vf_pct = _flash["VF_mass"] * 100
                if _vf_pct >= 1.0:
                    st.info(
                        f"Equilibrium flash at {P_bara:.1f} bara / {T_C:.0f} °C — "
                        f"vapour fraction {_vf_pct:.1f} wt%. "
                        "Adjusted compositions used for pressure-drop calculation.",
                        icon="ℹ️",
                    )
                else:
                    st.caption(
                        f"Flash: VF = {_vf_pct:.2f} wt% — effectively single-phase liquid "
                        f"at {P_bara:.1f} bara / {T_C:.0f} °C."
                    )
                with st.expander(
                    f"Feed / phase-split detail  (VF = {_flash['VF_mass']*100:.1f} wt%)",
                    expanded=False):
                    _feed   = _flash["feed_kgh"]
                    _g_ph   = _flash["gas_phase_kgh"]
                    _l_ph   = _flash["liquid_phase_kgh"]
                    _frows  = []
                    for _sp in _flash["species"]:
                        _frows.append({
                            "Species":           _sp,
                            "Feed (kg/h)":       round(_feed.get(_sp, 0), 4),
                            "Gas phase (kg/h)":  round(_g_ph.get(_sp, 0), 4),
                            "Liquid phase (kg/h)": round(_l_ph.get(_sp, 0), 4),
                        })
                    st.dataframe(
                        pd.DataFrame(_frows),
                        column_config={c: st.column_config.NumberColumn(format="%.4f")
                                       for c in ["Feed (kg/h)", "Gas phase (kg/h)",
                                                 "Liquid phase (kg/h)"]},
                        hide_index=True, use_container_width=True)
                    _fv1, _fv2 = st.columns(2)
                    _fv1.metric("Vapour fraction (mass)", f"{_flash['VF_mass']*100:.2f} %")
                    _fv2.metric("Vapour fraction (mol)",  f"{_flash['VF_mol']*100:.2f} %")
                    st.caption("Peng-Robinson EOS · binary interaction parameters = 0 "
                               "· composition assumed constant along pipe")

                # Derive effective inputs for pressure-drop engine
                _fl_gas = _flash["gas_phase_kgh"]
                _fl_liq = _flash["liquid_phase_kgh"]
                if _fl_gas and _fl_liq:
                    _fl_rho, _fl_mu, _fl_sig = engine.liquid_mixture_props(
                        _fl_liq, T_C + 273.15, P_bara * 1e5)
                    _eff_gas_flows    = _fl_gas
                    _eff_liq_type     = "Custom"
                    _eff_q_lye        = sum(_fl_liq.values()) / _fl_rho
                    _eff_custom_liq   = {"rho_kgm3": _fl_rho,
                                         "mu_mpas":  _fl_mu * 1e3,
                                         "sigma_mnm": _fl_sig * 1e3}
                    _eff_liquid_flows = None  # encoded in Custom above
                elif not _fl_liq:
                    _eff_gas_flows    = _fl_gas or gas_flows_kgh
                    _eff_liq_type     = liquid_type
                    _eff_q_lye        = 0.0
                    _eff_custom_liq   = custom_liquid
                    _eff_liquid_flows = liquid_flows_kgh
                else:
                    _eff_gas_flows    = gas_flows_kgh
                    _eff_liq_type     = liquid_type
                    _eff_q_lye        = q_lye
                    _eff_custom_liq   = custom_liquid
                    _eff_liquid_flows = liquid_flows_kgh
            else:
                if _flash["warnings"]:
                    st.caption(f"Flash equilibrium: {_flash['warnings'][0]}")
                _eff_gas_flows    = gas_flows_kgh
                _eff_liq_type     = liquid_type
                _eff_q_lye        = q_lye
                _eff_custom_liq   = custom_liquid
                _eff_liquid_flows = liquid_flows_kgh

            try:
                props = engine.calculate_two_phase_properties(
                    P_bara, T_C, _eff_gas_flows, _eff_liq_type, _eff_q_lye,
                    custom_gas=custom_gas, custom_liquid=_eff_custom_liq,
                    use_coolprop=use_coolprop,
                    liquid_flows_kgh=_eff_liquid_flows)
            except Exception as _calc_err:
                st.error(f"Property calculation failed: {_calc_err}")
                st.stop()

        # Segment loop
        current_P          = P_bara * 1e5
        current_T_C        = T_C if T_C is not None else 25.0  # updated at HX segments
        grid_records       = []
        stream_records     = []
        valve_sizing       = []   # collects ΔP-mode valve results for display
        cumulative_positions = []
        cumulative_distance  = 0.0
        pressure_profile_x   = [0.0]
        pressure_profile_y   = [P_bara]
        regime_bands         = []
        total_dp_fric_kpa    = 0.0
        total_dp_grav_kpa    = 0.0
        total_dp_accel_kpa   = 0.0

        def _props_at_current():
            """Evaluate fluid properties at the current P and T (used for stream snaps and valve/HX)."""
            if _is_vle:
                return engine.calculate_vle_properties(
                    vle_fluid_id, max(1000.0, current_P)/1e5, vle_x, vle_m_kgs)
            return engine.calculate_two_phase_properties(
                max(1000.0, current_P)/1e5, current_T_C,
                _eff_gas_flows, _eff_liq_type, _eff_q_lye,
                custom_gas=custom_gas, custom_liquid=_eff_custom_liq,
                use_coolprop=use_coolprop,
                liquid_flows_kgh=_eff_liquid_flows)

        def _snap_stream(label, sp):
            """Append one stream-balance row using pre-computed props dict sp."""
            _x  = sp.get("x_gas", 0.0)
            _rg = sp.get("rho_g", 0.0)
            _rl = sp.get("rho_l", 1000.0)
            if 0.0 < _x < 1.0:
                _rh = 1.0 / (_x / max(_rg, 1e-9) + (1.0 - _x) / max(_rl, 1e-9))
            else:
                _rh = _rg if _x >= 1.0 else _rl
            _rec = {
                "Stream":        label,
                "P (bara)":      round(max(1e-4, current_P / 1e5), 4),
                "T (°C)":        round(current_T_C, 1),
            }
            if _is_vle:
                _rec[f"{vle_fluid_id} vapour  kg/h"] = round(sp.get("m_gas_total_kgh", 0.0), 2)
                _rec[f"{vle_fluid_id} liquid  kg/h"] = round(sp.get("m_liquid_total_kgh", 0.0), 2)
            else:
                for _gsp, _gflow in (_eff_gas_flows or {}).items():
                    _rec[f"gas:{_gsp}"] = round(_gflow, 3)
                _rec["Ṁ_gas (kg/h)"] = round(sp.get("m_gas_total_kgh", 0.0), 2)
                for _lsp, _lflow in (_eff_liquid_flows or {}).items():
                    _rec[f"liq:{_lsp}"] = round(_lflow, 3)
                _rec["Ṁ_liq (kg/h)"] = round(sp.get("m_liquid_total_kgh", 0.0), 2)
            _rec.update({
                "x (−)":         round(_x, 5),
                "α (−)":         round(sp.get("alpha", 0.0), 4),
                "ρ_g (kg/m³)":  round(_rg, 3),
                "ρ_l (kg/m³)":  round(_rl, 3),
                "ρ_hom (kg/m³)":round(_rh, 2),
            })
            stream_records.append(_rec)

        _snap_stream("S0 — Inlet", props)

        for i, seg in enumerate(st.session_state[k("segments")]):
            _seg_kind = seg.get("kind", "pipe")

            # ── VALVE ──────────────────────────────────────────────────────────
            if _seg_kind == "valve":
                _vp       = _props_at_current()
                _v_mode   = seg.get("valve_mode", "kv")
                if _v_mode == "dp":
                    _vres = engine.calculate_valve_kv(
                        _vp, seg.get("dp_kpa", 50.0) * 1000.0,
                        seg.get("opening_pct", 100.0),
                        seg.get("characteristic", "equal-percentage"))
                    _v_label = (f"ΔP={seg.get('dp_kpa',50):.1f} kPa → "
                                f"Kv_req={_vres['Kv_rated']:.2f}  "
                                f"Kv_eff={_vres['Kv_eff']:.2f}  "
                                f"{seg.get('opening_pct',100):.0f}% open")
                    valve_sizing.append({
                        "seg":       i + 1,
                        "dn":        seg.get("dn", ""),
                        "pn":        seg.get("pn", ""),
                        "dp_kpa":    seg.get("dp_kpa", 50.0),
                        "opening":   seg.get("opening_pct", 100.0),
                        "char":      seg.get("characteristic", "equal-percentage"),
                        "Q_m3h":     _vres["Q_m3h"],
                        "rho_hom":   _vres["rho_hom"],
                        "Kv_eff":    _vres["Kv_eff"],
                        "Kv_rated":  _vres["Kv_rated"],
                        "Cv_rated":  _vres["Kv_rated"] * 1.156,
                        "P_in":      current_P / 1e5,
                    })
                else:
                    _vres = engine.calculate_valve_dp(
                        _vp, seg.get("Kv_m3h", 1.0),
                        seg.get("opening_pct", 100.0),
                        seg.get("characteristic", "equal-percentage"))
                    _v_label = (f"Kv={seg.get('Kv_m3h',1.0):.2g}  "
                                f"Kv_eff={_vres['Kv_eff']:.3f}  "
                                f"{seg.get('opening_pct',100):.0f}% open")
                _v_dP_Pa  = _vres["dP_Pa"]
                _v_end_P  = current_P - _v_dP_Pa
                _v_D      = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
                grid_records.append({
                    "Seg":           f"#{i+1}",
                    "Type":          "Valve",
                    "Pipe":          f"{seg['dn']}/{seg['pn']}",
                    "ID (mm)":       round(_v_D*1000, 1),
                    "L (m)":         0.0,
                    "L_eq (m)":      0.0,
                    "Fittings":      _v_label,
                    "Regime":        f"Q={_vres['Q_m3h']:.3f} m³/h  ρ_hom={_vres['rho_hom']:.1f} kg/m³",
                    "ΔP (kPa)":      round(_v_dP_Pa/1000, 3),
                    "P_in (bara)":   round(current_P/1e5, 4),
                    "P_out (bara)":  round(_v_end_P/1e5, 4),
                    "V_m (m/s)":     0.0,   "V_m/V_e":       0.0,
                    "V_sg (m/s)":    0.0,   "V_sl (m/s)":    0.0,
                    "V_e (m/s)":     0.0,
                    "ΔP_fric (kPa)": round(_v_dP_Pa/1000, 3),
                    "ΔP_grav (kPa)": 0.0,  "ΔP_accel (kPa)": 0.0,
                    "Material":      "—",
                    "ρ_g (kg/m³)":  round(_vp["rho_g"], 4),
                    "L_eff (m)":     0.0,
                    "α (void)":      round(_vp["alpha"], 4),
                    "dP/dz (Pa/m)":  0.0,
                })
                total_dp_fric_kpa += _v_dP_Pa / 1000.0
                current_P = max(1000.0, _v_end_P)
                pressure_profile_x.append(cumulative_distance)
                pressure_profile_y.append(max(0.1, current_P/1e5))
                regime_bands.append("Valve")
                _snap_stream(f"S{i+1} — #{i+1} Valve", _props_at_current())
                continue

            # ── HEAT EXCHANGER ─────────────────────────────────────────────────
            if _seg_kind == "hx":
                _hx_duty = float(seg.get("duty_kw", 0.0))
                _hx_dP_Pa = float(seg.get("dp_kpa", 0.0)) * 1000.0
                _hx_D     = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
                _hx_end_P = current_P - _hx_dP_Pa
                _delta_T  = 0.0
                if _hx_duty != 0.0:
                    _hp = _props_at_current()
                    _Cp = engine.estimate_mixture_cp(
                        _hp, current_T_C + 273.15, max(1000.0, current_P))
                    if _hp["m_total_kgs"] > 0 and _Cp > 0:
                        _delta_T = (_hx_duty * 1000.0) / (_hp["m_total_kgs"] * _Cp)
                current_T_C += _delta_T
                _hx_sign = f"+{_hx_duty:.1f}" if _hx_duty >= 0 else f"{_hx_duty:.1f}"
                grid_records.append({
                    "Seg":           f"#{i+1}",
                    "Type":          "Heat Exchanger",
                    "Pipe":          f"{seg['dn']}/{seg['pn']}",
                    "ID (mm)":       round(_hx_D*1000, 1),
                    "L (m)":         0.0,
                    "L_eq (m)":      0.0,
                    "Fittings":      f"Q={_hx_sign} kW",
                    "Regime":        f"ΔT={_delta_T:+.2f}°C  →  T_out={current_T_C:.1f}°C",
                    "ΔP (kPa)":      round(_hx_dP_Pa/1000, 3),
                    "P_in (bara)":   round(current_P/1e5, 4),
                    "P_out (bara)":  round(_hx_end_P/1e5, 4),
                    "V_m (m/s)":     0.0,   "V_m/V_e":       0.0,
                    "V_sg (m/s)":    0.0,   "V_sl (m/s)":    0.0,
                    "V_e (m/s)":     0.0,
                    "ΔP_fric (kPa)": round(_hx_dP_Pa/1000, 3),
                    "ΔP_grav (kPa)": 0.0,  "ΔP_accel (kPa)": 0.0,
                    "Material":      "—",
                    "ρ_g (kg/m³)":  0.0,
                    "L_eff (m)":     0.0,
                    "α (void)":      0.0,
                    "dP/dz (Pa/m)":  0.0,
                })
                total_dp_fric_kpa += _hx_dP_Pa / 1000.0
                current_P = max(1000.0, _hx_end_P)
                pressure_profile_x.append(cumulative_distance)
                pressure_profile_y.append(max(0.1, current_P/1e5))
                regime_bands.append("Heat Exchanger")
                _snap_stream(f"S{i+1} — #{i+1} Heat Exch.", _props_at_current())
                continue

            # ── PIPE SEGMENT ───────────────────────────────────────────────────
            D_seg   = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
            _lined  = seg.get("lined",False)
            _lthk_m = seg.get("liner_thickness_mm",1.0) / 1000.0
            _lmat   = seg.get("liner_material","FEP")
            D_eff   = D_seg - 2*_lthk_m if _lined else D_seg
            rough_seg = (engine.LINER_ROUGHNESS[_lmat] if _lined
                         else engine.MATERIAL_ROUGHNESS[seg.get("material","SS316L")])

            if _is_vle:
                props_seg = engine.calculate_vle_properties(
                    vle_fluid_id, current_P/1e5, vle_x, vle_m_kgs)
            else:
                props_seg = engine.calculate_two_phase_properties(
                    current_P/1e5, current_T_C, _eff_gas_flows, _eff_liq_type, _eff_q_lye,
                    custom_gas=custom_gas, custom_liquid=_eff_custom_liq,
                    use_coolprop=use_coolprop,
                    liquid_flows_kgh=_eff_liquid_flows)

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
            current_P = max(1000.0, end_P)  # floor at ~0.01 bara; prevents CoolProp crash on next seg
            _snap_stream(f"S{i+1} — #{i+1} {seg['type']}", _props_at_current())
            cumulative_distance += L_eff
            cumulative_positions.append(cumulative_distance)
            pressure_profile_x.append(cumulative_distance)
            pressure_profile_y.append(max(0.1, current_P/1e5))
            regime_bands.append(regime)

        # Erosion — only banner on actionable conditions; OK is shown inline in table
        _max_ratio = max((r["V_m/V_e"] for r in grid_records), default=0.0)
        _worst_seg = next((r["Seg"] for r in grid_records if r["V_m/V_e"]==_max_ratio), "")
        if _max_ratio >= 1.0:
            st.error(f"**Erosion limit exceeded** — Segment {_worst_seg}: "
                     f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, C=100). "
                     f"Reduce velocity or increase pipe diameter.")
        elif _max_ratio >= 0.8:
            st.warning(f"**Approaching erosion limit** — Segment {_worst_seg}: "
                       f"V_m/V_e = **{_max_ratio:.2f}** (API RP 14E, limit = 1.0).")

        # ── Valve Sizing Results ──────────────────────────────────────────────
        if valve_sizing:
            with st.container(border=True):
                st.markdown("**Valve Sizing**")
                for _vs in valve_sizing:
                    st.markdown(f"Segment #{_vs['seg']} — {_vs['dn']}/{_vs['pn']}  "
                                f"·  ΔP = {_vs['dp_kpa']:.1f} kPa  "
                                f"·  {_vs['opening']:.0f} % open  "
                                f"·  {_vs['char']}")
                    _vc1, _vc2, _vc3, _vc4, _vc5 = st.columns(5)
                    _vc1.metric("Q",       f"{_vs['Q_m3h']:.3f} m³/h")
                    _vc2.metric("ρ_hom",   f"{_vs['rho_hom']:.1f} kg/m³")
                    _vc3.metric("Kv_eff",  f"{_vs['Kv_eff']:.2f} m³/h·bar^0.5")
                    _vc4.metric("Kv_rated (req.)", f"{_vs['Kv_rated']:.2f} m³/h·bar^0.5")
                    _vc5.metric("Cv_rated (req.)", f"{_vs['Cv_rated']:.2f} US gal/min·psi^0.5")
                    st.caption(f"P_in = {_vs['P_in']:.3f} bara  ·  "
                               f"Kv_rated is the minimum full-open Kv needed to pass "
                               f"{_vs['Q_m3h']:.3f} m³/h at {_vs['dp_kpa']:.1f} kPa "
                               f"with valve at {_vs['opening']:.0f} % opening.")
                    if len(valve_sizing) > 1:
                        st.divider()

        # ── Segment Analysis ──────────────────────────────────────────────────
        _primary_cols = ["Seg", "Type", "Pipe", "ID (mm)", "L (m)", "Fittings",
                         "Regime", "ΔP (kPa)", "P_in (bara)", "P_out (bara)",
                         "V_m (m/s)", "V_m/V_e"]
        _detail_cols  = ["V_sg (m/s)", "V_sl (m/s)", "V_e (m/s)",
                         "ΔP_fric (kPa)", "ΔP_grav (kPa)", "ΔP_accel (kPa)",
                         "ρ_g (kg/m³)", "L_eff (m)", "α (void)", "dP/dz (Pa/m)", "Material"]
        _grid_df = pd.DataFrame(grid_records)
        _seg_col_cfg = {
            "ID (mm)":         st.column_config.NumberColumn(format="%.1f"),
            "L (m)":           st.column_config.NumberColumn(format="%.1f"),
            "ΔP (kPa)":        st.column_config.NumberColumn(format="%.3f"),
            "P_in (bara)":     st.column_config.NumberColumn(format="%.4f"),
            "P_out (bara)":    st.column_config.NumberColumn(format="%.4f"),
            "V_m (m/s)":       st.column_config.NumberColumn(format="%.3f"),
            "V_m/V_e":         st.column_config.ProgressColumn(
                                   format="%.3f", min_value=0.0, max_value=1.0),
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
        }
        _existing_primary = [c for c in _primary_cols if c in _grid_df.columns]
        st.subheader("Segment Analysis")
        st.dataframe(_grid_df[_existing_primary],
                     column_config=_seg_col_cfg,
                     hide_index=True, use_container_width=True)
        _existing_detail = [c for c in _detail_cols if c in _grid_df.columns]
        if _existing_detail:
            with st.expander("Detail columns"):
                st.dataframe(_grid_df[["Seg"] + _existing_detail],
                             column_config=_seg_col_cfg,
                             hide_index=True, use_container_width=True)

        # ── Key Results — fill the top-of-column placeholder ─────────────────
        total_dp_kpa         = ((P_bara*1e5) - current_P) / 1000.0
        outlet_pressure_bara = max(0.1, current_P/1e5)
        pipe_length_m        = sum(s.get("length", 0.0) for s in st.session_state[k("segments")])
        _T_out  = current_T_C
        _T_in   = T_C if T_C is not None else _T_out
        _dT     = _T_out - _T_in
        _has_hx = any(s.get("kind") == "hx" for s in st.session_state[k("segments")])

        with _key_results_ph.container():
            with st.container(border=True):
                # Accent banner: ΔP at a glance
                st.markdown(
                    f'<div style="background:{accent};border-radius:6px;'
                    f'padding:0.45rem 1rem 0.3rem;margin-bottom:0.6rem">'
                    f'<span style="color:white;font-size:1.55rem;font-weight:700">'
                    f'ΔP = {total_dp_kpa:.2f} kPa</span>'
                    f'<span style="color:rgba(255,255,255,0.75);font-size:0.95rem;'
                    f'margin-left:1rem">({total_dp_kpa/100:.4f} bar'
                    f' &nbsp;·&nbsp; {total_dp_kpa*1000:.0f} Pa)</span>'
                    f'</div>',
                    unsafe_allow_html=True)

                # Pressure / temperature metrics
                _kr1, _kr2, _kr3, _kr4 = st.columns(4)
                _kr1.metric("P_in",  f"{P_bara:.3f} bara")
                _kr2.metric("P_out", f"{outlet_pressure_bara:.4f} bara",
                            delta=-(total_dp_kpa / 100), delta_color="inverse")
                _kr3.metric("T_in",  f"{_T_in:.1f} °C")
                _kr4.metric("T_out", f"{_T_out:.1f} °C",
                            delta=(_dT if _has_hx else None))

                # Outlet composition (compact)
                _out_gas_flows = _eff_gas_flows or {}
                _out_liq_kgh   = props.get("m_lye_kgh", 0.0)
                _gas_parts = [f"**{sp}** {m:.3g} kg/h"
                              for sp, m in _out_gas_flows.items() if m > 0]
                _gas_total = sum(_out_gas_flows.values())
                if _gas_parts:
                    st.markdown("**Outlet composition** &nbsp; *(constant along pipe)*",
                                unsafe_allow_html=True)
                    _oc1, _oc2 = st.columns([3, 1])
                    _oc1.caption("Gas phase:  " + "  ·  ".join(_gas_parts)
                                 + f"  ·  Total {_gas_total:.3g} kg/h")
                    _oc2.caption(f"Liquid:  {_out_liq_kgh:.3g} kg/h")

                # ΔP decomposition (collapsed by default)
                with st.expander("ΔP decomposition"):
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Σ Frictional",     f"{total_dp_fric_kpa:.3f} kPa",
                              help="Sum of frictional component across all segments")
                    d2.metric("Σ Gravitational",  f"{total_dp_grav_kpa:.3f} kPa",
                              help="Gravitational head loss (negative = recovery on downflow)")
                    d3.metric("Σ Accelerational", f"{total_dp_accel_kpa:.3f} kPa",
                              help="Residual B&B inclination correction; zero for other correlations.")
                    st.caption(f"Pipe length {pipe_length_m:.1f} m  ·  "
                               f"Eff. length (+ fittings) {cumulative_distance:.1f} m")

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
        _seg_type = _seg.get("type", "Horizontal")
        _seg_len  = _seg.get("length", 0.0)
        if _seg_type == "Horizontal":
            _nodes.append((_xl+_seg_len, _yl))
        elif _seg_type == "Vertical Upflow":
            _nodes.append((_xl, _yl+_seg_len))
        else:
            _nodes.append((_xl, _yl-_seg_len))

    _DN_LW = {"DN20":2,"DN25":3,"DN40":5,"DN50":7,"DN80":9,"DN100":11,"DN150":15,"DN200":19,"DN250":23}

    fig_sch = go.Figure()
    _seen_reg = set()
    for _i, (_seg, _rec) in enumerate(zip(st.session_state[k("segments")], grid_records)):
        _x0,_y0 = _nodes[_i]; _x1,_y1 = _nodes[_i+1]
        _reg = _rec["Regime"]
        _col = _regime_color(_reg, _REGIME_LINE_KW, "#64748B")
        _lw  = _DN_LW.get(_seg.get("dn","DN50"), 10)
        _show = _reg not in _seen_reg; _seen_reg.add(_reg)
        _seg_kind = _seg.get("kind", "pipe")
        _lh = (f"Liner: {_seg['liner_material']} {_seg['liner_thickness_mm']:.1f} mm"
               f"  →  ID {_rec['ID (mm)']:.1f} mm<br>" if _seg.get("lined") else "")
        _hover = (f"<b>Seg #{_i+1}  {_seg.get('dn','')}/{_seg.get('pn','')}</b><br>"
                  +_lh+f"{_seg.get('type', _seg_kind)},  L={_seg.get('length',0.0):.1f} m<br>"
                  f"Regime: {_reg}<br>ΔP: {_rec['ΔP (kPa)']:.3f} kPa  ·  "
                  f"V_sg: {_rec['V_sg (m/s)']:.3f} m/s<extra></extra>")
        if _seg_kind in ("valve", "hx"):
            # Zero-length inline component — render as a diamond marker
            _sym = "diamond" if _seg_kind == "valve" else "square"
            fig_sch.add_trace(go.Scatter(
                x=[_x0], y=[_y0], mode="markers",
                marker=dict(symbol=_sym, size=14, color=_col,
                            line=dict(color="white", width=2)),
                name=_reg, legendgroup=_reg, showlegend=_show,
                hovertemplate=_hover))
            fig_sch.add_annotation(
                x=_x0, y=_y0,
                text=f"<b>#{_i+1}</b>",
                showarrow=False, font=dict(size=9, color="#1E293B"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor=_col, borderwidth=1.5,
                borderpad=2, xanchor="left", yanchor="bottom", xshift=10)
        else:
            fig_sch.add_trace(go.Scatter(
                x=[_x0,_x1], y=[_y0,_y1], mode="lines",
                line=dict(color=_col, width=_lw), name=_reg, legendgroup=_reg,
                showlegend=_show,
                hovertemplate=_hover))
            fig_sch.add_annotation(
                x=_x0+(_x1-_x0)*0.65, y=_y0+(_y1-_y0)*0.65,
                ax=_x0+(_x1-_x0)*0.5, ay=_y0+(_y1-_y0)*0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.8, arrowwidth=2.5, arrowcolor=_col)
            _reg_short = _reg.split("/")[0].strip()[:18]
            fig_sch.add_annotation(
                x=(_x0+_x1)/2, y=(_y0+_y1)/2,
                text=f"<b>#{_i+1}</b> {_seg.get('dn','')}<br>"
                     f"<span style='font-size:9px;color:{_col}'>{_reg_short}</span>",
                showarrow=False, font=dict(size=10, color="#1E293B"),
                bgcolor="rgba(255,255,255,0.88)", bordercolor=_col, borderwidth=1.5,
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

    # ── Flow Regime Map (V_sg vs V_sl, log-log) ─────────────────────────────
    _pipe_recs = [r for r in grid_records if r.get("V_sg (m/s)", 0) > 0 or r.get("V_sl (m/s)", 0) > 0]

    st.divider()
    tab_sch, tab_prof_tab, tab_regime_map = st.tabs(
        ["Pipeline Schematic", "Pressure Profile", "Flow Regime Map"])
    with tab_sch:
        st.plotly_chart(fig_sch, use_container_width=True, key=k("fig_sch"))
        st.caption("Line width ∝ DN  ·  Colour = flow regime  ·  Regime name on each segment")
    with tab_prof_tab:
        st.plotly_chart(fig_prof, use_container_width=True, key=k("fig_prof"))
    with tab_regime_map:
        if _pipe_recs:
            _n_vert = sum(1 for r in _pipe_recs if r.get("Type", "Horizontal") != "Horizontal")
            _map_choice = st.radio(
                "Reference map:",
                ["Horizontal – Taitel-Dukler + Mandhane-Gregory-Aziz",
                 "Vertical upflow – Wallis / void-fraction"],
                index=1 if _n_vert > len(_pipe_recs) / 2 else 0,
                horizontal=True, key=k("rmap_orient"),
            )
            _use_horiz = "Horizontal" in _map_choice

            # ── Compute regime grid using engine correlations ─────────────────
            _p      = props          # inlet flash result, in scope from run_case
            _D_repr = (_pipe_recs[0]["ID (mm)"] / 1000.0) if _pipe_recs else 0.05
            _td_grid, _full_grid, _vsl_arr, _vsg_arr = _compute_regime_grid(
                rhol=float(_p["rho_l"]),
                rhog=float(_p["rho_g"]),
                mul=float(_p["mu_l"]),
                mug=float(_p["mu_g"]),
                sigma=float(_p.get("sigma") or 0.072),
                D=float(_D_repr),
                roughness=4.6e-5,
                use_horiz=_use_horiz,
            )

            # ── Map regime strings → integer z-values & colours ───────────────
            _all_regs = sorted(set(r for row in _td_grid for r in row if r))
            _reg_to_idx = {r: i for i, r in enumerate(_all_regs)}
            _idx_to_col = [_regime_color(r, _REGIME_LINE_KW, "#94A3B8") for r in _all_regs]
            _n_reg = len(_all_regs)

            _z = [[_reg_to_idx.get(_td_grid[i][j], 0) for j in range(len(_vsl_arr))]
                  for i in range(len(_vsg_arr))]

            # Discrete colorscale: each regime gets a flat band
            _cs = []
            for _ci, _cc in enumerate(_idx_to_col):
                _cs.extend([[_ci / _n_reg, _cc], [(_ci + 1) / _n_reg, _cc]])

            # ── Build figure ──────────────────────────────────────────────────
            fig_regime = go.Figure()

            # Background heatmap (computed regime zones)
            fig_regime.add_trace(go.Heatmap(
                x=_vsl_arr, y=_vsg_arr, z=_z,
                text=_full_grid,
                colorscale=_cs,
                zmin=0, zmax=_n_reg,
                showscale=False,
                opacity=0.30,
                hovertemplate=(
                    "V_sl = %{x:.4f} m/s<br>"
                    "V_sg = %{y:.4f} m/s<br>"
                    "Regime: %{text}<extra></extra>"
                ),
            ))

            # Zone labels at each regime's log-space centroid
            _zone_acc = {}   # regime → [Σlog_vsl, Σlog_vsg, count]
            _log_vsl = np.log10(_vsl_arr)
            _log_vsg = np.log10(_vsg_arr)
            for _gi, _row in enumerate(_td_grid):
                for _gj, _reg in enumerate(_row):
                    if not _reg:
                        continue
                    if _reg not in _zone_acc:
                        _zone_acc[_reg] = [0.0, 0.0, 0]
                    _zone_acc[_reg][0] += _log_vsl[_gj]
                    _zone_acc[_reg][1] += _log_vsg[_gi]
                    _zone_acc[_reg][2] += 1

            for _zreg, (_svsl, _svsg, _cnt) in _zone_acc.items():
                if _cnt == 0:
                    continue
                fig_regime.add_annotation(
                    x=10 ** (_svsl / _cnt), y=10 ** (_svsg / _cnt),
                    xref="x", yref="y",
                    text=f"<b>{_zreg}</b>",
                    showarrow=False,
                    font=dict(size=9, color="#1E293B"),
                    bgcolor="rgba(255,255,255,0.55)",
                    borderpad=2,
                )

            # Operating point markers
            _seen_reg_map = set()
            for _r in _pipe_recs:
                _vsg = max(_r["V_sg (m/s)"], 1e-4)
                _vsl = max(_r["V_sl (m/s)"], 1e-4)
                _reg = _r["Regime"]
                _col = _regime_color(_reg, _REGIME_LINE_KW, "#64748B")
                _sym = "circle" if _r.get("Type", "Horizontal") == "Horizontal" else "diamond"
                _show_leg = _reg not in _seen_reg_map; _seen_reg_map.add(_reg)
                fig_regime.add_trace(go.Scatter(
                    x=[_vsl], y=[_vsg], mode="markers+text",
                    marker=dict(size=16, color=_col, symbol=_sym,
                                line=dict(color="white", width=1.5)),
                    text=[_r["Seg"]], textposition="middle center",
                    textfont=dict(size=9, color="white"),
                    name=_reg, legendgroup=_reg, showlegend=_show_leg,
                    hovertemplate=(
                        f"<b>{_r['Seg']}  {_r['Pipe']}</b><br>"
                        f"Orientation: {_r.get('Type', '—')}<br>"
                        f"Regime: {_reg}<br>"
                        f"V_sl = {_vsl:.4f} m/s<br>"
                        f"V_sg = {_vsg:.4f} m/s<br>"
                        f"ΔP = {_r['ΔP (kPa)']:.3f} kPa"
                        "<extra></extra>"
                    ),
                ))

            fig_regime.update_layout(
                template="plotly_white", height=470,
                margin=dict(l=70, r=20, t=40, b=70),
                xaxis=dict(title="V_sl  superficial liquid velocity (m/s)", type="log",
                           range=[-3, 1], gridcolor="#F1F5F9", linecolor="#E2E8F0"),
                yaxis=dict(title="V_sg  superficial gas velocity (m/s)", type="log",
                           range=[-3, 2], gridcolor="#F1F5F9", linecolor="#E2E8F0"),
                legend=dict(title="Computed regime", bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#E2E8F0", borderwidth=1, font=dict(size=11)),
                font=dict(size=12, color="#374151"),
                title=dict(text="Flow Regime Map — operating points per segment",
                           font=dict(size=13), x=0),
            )
            st.plotly_chart(fig_regime, use_container_width=True, key=k("fig_regime"))
            _corr_note = ("Taitel-Dukler (1976) + Mandhane-Gregory-Aziz (1974)"
                          if _use_horiz else "Wallis annular criterion + void-fraction thresholds")
            st.caption(
                f"Background zones computed for inlet conditions: "
                f"ρ_l = {_p['rho_l']:.1f} kg/m³, ρ_g = {_p['rho_g']:.4f} kg/m³, "
                f"D = {_D_repr*1000:.1f} mm  —  {_corr_note}.  "
                "● horizontal  ◆ vertical segment"
            )
        else:
            st.info("No pipe segments with velocity data to plot.")

    # ── HEAT & MASS BALANCE (full width, below charts) ───────────────────────
    if stream_records:
        _hmb_rows_def = [("P  (bara)", "P (bara)"), ("T  (°C)", "T (°C)")]
        if _is_vle:
            _hmb_rows_def += [
                (f"  {vle_fluid_id} vapour  kg/h", f"{vle_fluid_id} vapour  kg/h"),
                (f"  {vle_fluid_id} liquid  kg/h", f"{vle_fluid_id} liquid  kg/h"),
            ]
        else:
            for _gsp in (_eff_gas_flows or {}):
                _hmb_rows_def.append((f"  {_gsp}  kg/h", f"gas:{_gsp}"))
            if _eff_gas_flows:
                _hmb_rows_def.append(("  Σ gas  kg/h", "Ṁ_gas (kg/h)"))
            for _lsp in (_eff_liquid_flows or {}):
                _hmb_rows_def.append((f"  {_lsp}  kg/h", f"liq:{_lsp}"))
            if _eff_liquid_flows:
                _hmb_rows_def.append(("  Σ liquid  kg/h", "Ṁ_liq (kg/h)"))
        _hmb_rows_def += [
            ("x  (−)",       "x (−)"),
            ("α  (−)",       "α (−)"),
            ("ρ_hom  kg/m³", "ρ_hom (kg/m³)"),
        ]
        _hmb_table = []
        for _prop_label, _key in _hmb_rows_def:
            _row = {"Property": _prop_label}
            for _sr in stream_records:
                _row[_sr["Stream"]] = _sr.get(_key, "—")
            _hmb_table.append(_row)
        _hmb_df      = pd.DataFrame(_hmb_table)
        _stream_cols = [sr["Stream"] for sr in stream_records]
        _num_col_cfg = {col: st.column_config.NumberColumn(format="%.4g")
                        for col in _stream_cols}
        st.subheader("Heat & Mass Balance")
        st.dataframe(_hmb_df, column_config=_num_col_cfg,
                     hide_index=True, use_container_width=True)


    # ── EXPORTS ───────────────────────────────────────────────────────────────
    st.divider()
    ex_tab_w, ex_tab_x = st.tabs(["Export Word (.docx)", "Export Excel (.xlsx)"])
    with ex_tab_w:
        _rpt_hash = hashlib.md5(json.dumps({
            "P": P_bara, "T": T_C,
            "gas_flows": {_kk: float(_vv) for _kk,_vv in gas_flows_kgh.items()},
            "liq_flows": {_kk: float(_vv) for _kk,_vv in liquid_flows_kgh.items()},
            "segs": [(s.get("type", s.get("kind","")),s.get("dn",""),s.get("pn",""),s.get("length",0.0),
                      tuple(sorted((f["type"],f["qty"]) for f in s.get("fittings_list",[]))),
                      s.get("lined",False),
                      s.get("liner_material","FEP"),s.get("liner_thickness_mm",1.0))
                     for s in st.session_state[k("segments")]],
        }, sort_keys=True).encode()).hexdigest()
        if st.session_state.get(k("rpt_hash")) != _rpt_hash:
            st.session_state[k("rpt_hash")]    = None
            st.session_state[k("rpt_bytes")]   = None
            st.session_state[k("rpt_hash_ch")] = None
            st.session_state[k("rpt_bytes_ch")]= None
        def _rpt_kwargs():
            return dict(
                P_bara=P_bara, T_C=T_C,
                gas_flows_kgh=gas_flows_kgh,
                liquid_type=liquid_type, q_lye=q_lye,
                props=props, grid_records=grid_records,
                segments=st.session_state[k("segments")],
                total_dp_kpa=total_dp_kpa,
                outlet_pressure_bara=outlet_pressure_bara,
                pipe_length_m=pipe_length_m,
                cumulative_distance=cumulative_distance,
                case_label=st.session_state.get(f"label_{cid}", f"Case {cid.upper()}"))
        _wg1, _wg2 = st.columns(2)
        with _wg1:
            if st.button("Generate (tables only)", type="primary",
                         use_container_width=True, key=k("gen_rpt")):
                with st.spinner("Building document…"):
                    _buf = report_generator.generate_report(
                        **_rpt_kwargs(), fig_sch=None, fig_prof=None)
                    st.session_state[k("rpt_bytes")] = _buf.getvalue()
                    st.session_state[k("rpt_hash")]  = _rpt_hash
        with _wg2:
            if st.button("Generate with charts", use_container_width=True,
                         key=k("gen_rpt_ch")):
                with st.spinner("Building document with charts…"):
                    _buf = report_generator.generate_report(
                        **_rpt_kwargs(), fig_sch=fig_sch, fig_prof=fig_prof)
                    st.session_state[k("rpt_bytes_ch")] = _buf.getvalue()
                    st.session_state[k("rpt_hash_ch")]  = _rpt_hash
        _wd1, _wd2 = st.columns(2)
        with _wd1:
            if st.session_state.get(k("rpt_bytes")) and \
               st.session_state.get(k("rpt_hash")) == _rpt_hash:
                st.download_button("Download tables only  (.docx)",
                    data=st.session_state[k("rpt_bytes")],
                    file_name=f"hydraulic_report_case_{cid}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key=k("dl_rpt"))
            elif st.session_state.get(k("rpt_bytes")):
                st.info("Inputs changed — regenerate.")
        with _wd2:
            if st.session_state.get(k("rpt_bytes_ch")) and \
               st.session_state.get(k("rpt_hash_ch")) == _rpt_hash:
                st.download_button("Download with charts  (.docx)",
                    data=st.session_state[k("rpt_bytes_ch")],
                    file_name=f"hydraulic_report_case_{cid}_charts.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key=k("dl_rpt_ch"))
            elif st.session_state.get(k("rpt_bytes_ch")):
                st.info("Inputs changed — regenerate.")

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
        "liquid_flows_kgh":     liquid_flows_kgh,
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
        # VLE mode fields (None when not in VLE mode)
        "flow_mode":            "vle" if _is_vle else "gas_liquid",
        "vle_fluid":            vle_fluid_id,
        "vle_x_mass":           vle_x,
        "vle_m_total_kgs":      vle_m_kgs,
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
    Returns (total_dp_kpa, outlet_bara).  Handles pipe, valve, and HX segments.
    """
    current_P   = P_bara_override * 1e5
    current_T_C = res.get("T_C", 25.0)
    total_dp    = 0.0
    corr  = res.get("correlation",     engine.TWO_PHASE_CORRELATIONS[0])
    void  = res.get("voidage_method",  engine.VOIDAGE_METHODS[0])
    cgas  = res.get("custom_gas")
    cliq  = res.get("custom_liquid")
    _is_vle_res = (res.get("flow_mode") == "vle" and res.get("vle_fluid"))

    def _get_props():
        if _is_vle_res:
            return engine.calculate_vle_properties(
                res["vle_fluid"], max(1e4, current_P) / 1e5,
                res["vle_x_mass"], res["vle_m_total_kgs"])
        return engine.calculate_two_phase_properties(
            max(1e4, current_P) / 1e5, current_T_C,
            res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
            custom_gas=cgas, custom_liquid=cliq,
            liquid_flows_kgh=res.get("liquid_flows_kgh"))

    for seg in res["segments"]:
        _kind = seg.get("kind", "pipe")

        if _kind == "valve":
            props    = _get_props()
            v_res    = engine.calculate_valve_dp(
                props, seg.get("Kv_m3h", 1.0),
                seg.get("opening_pct", 100.0),
                seg.get("characteristic", "equal-percentage"))
            total_dp  += v_res["dP_Pa"]
            current_P -= v_res["dP_Pa"]
            current_P  = max(1e4, current_P)
            continue

        if _kind == "hx":
            dp_pa      = seg.get("dp_kpa", 0.0) * 1000.0
            duty_kw    = seg.get("duty_kw", 0.0)
            if duty_kw != 0.0:
                props = _get_props()
                Cp = engine.estimate_mixture_cp(
                    props, current_T_C + 273.15, max(1e4, current_P))
                if props["m_total_kgs"] > 0 and Cp > 0:
                    current_T_C += (duty_kw * 1000.0) / (props["m_total_kgs"] * Cp)
            total_dp  += dp_pa
            current_P -= dp_pa
            current_P  = max(1e4, current_P)
            continue

        # Pipe segment
        D_seg  = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        lined  = seg.get("lined", False)
        lthk_m = seg.get("liner_thickness_mm", 1.0) / 1000.0
        lmat   = seg.get("liner_material", "FEP")
        D_eff  = D_seg - 2 * lthk_m if lined else D_seg
        rough  = (engine.LINER_ROUGHNESS[lmat] if lined
                  else engine.MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
        props  = _get_props()
        angle  = {"Horizontal": 0.0, "Vertical Upflow": np.pi / 2.0,
                  "Vertical Downflow": -np.pi / 2.0}[seg["type"]]
        le_fit = _sum_le_fit(seg, D_eff)
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)
        total_dp  += seg_res["dP_Pa"]
        current_P -= seg_res["dP_Pa"]
        current_P  = max(1e4, current_P)

    return total_dp / 1000.0, current_P / 1e5


def _calc_regimes_at_p(res, P_bara_override):
    """Run segments at the given inlet pressure; return list of dicts with
    seg label, pipe label, and regime string — one entry per segment."""
    current_P = P_bara_override * 1e5
    corr = res.get("correlation",    engine.TWO_PHASE_CORRELATIONS[0])
    void = res.get("voidage_method", engine.VOIDAGE_METHODS[0])
    cgas = res.get("custom_gas")
    cliq = res.get("custom_liquid")
    _is_vle_res = (res.get("flow_mode") == "vle" and res.get("vle_fluid"))
    current_T_C = res.get("T_C", 25.0)
    out = []
    for i, seg in enumerate(res["segments"]):
        _kind = seg.get("kind", "pipe")
        label = f"#{i + 1}"
        pipe_label = f"{seg.get('dn', '—')}/{seg.get('pn', '—')}"

        if _kind == "valve":
            if _is_vle_res:
                props = engine.calculate_vle_properties(
                    res["vle_fluid"], current_P / 1e5,
                    res["vle_x_mass"], res["vle_m_total_kgs"])
            else:
                props = engine.calculate_two_phase_properties(
                    current_P / 1e5, current_T_C,
                    res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
                    custom_gas=cgas, custom_liquid=cliq,
                    liquid_flows_kgh=res.get("liquid_flows_kgh"))
            vres = engine.calculate_valve_dp(
                props, seg.get("Kv_m3h", 1.0), seg.get("opening_pct", 100.0),
                seg.get("characteristic", "equal-percentage"))
            out.append({"seg": label, "pipe": pipe_label, "regime": "Valve"})
            current_P -= vres["dP_Pa"]
            current_P  = max(1e4, current_P)
            continue

        if _kind == "hx":
            duty_kw = seg.get("duty_kw", 0.0)
            if duty_kw != 0.0:
                if _is_vle_res:
                    props = engine.calculate_vle_properties(
                        res["vle_fluid"], current_P / 1e5,
                        res["vle_x_mass"], res["vle_m_total_kgs"])
                else:
                    props = engine.calculate_two_phase_properties(
                        current_P / 1e5, current_T_C,
                        res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
                        custom_gas=cgas, custom_liquid=cliq,
                        liquid_flows_kgh=res.get("liquid_flows_kgh"))
                Cp = engine.estimate_mixture_cp(props, current_T_C + 273.15, max(1e4, current_P))
                if props["m_total_kgs"] > 0 and Cp > 0:
                    current_T_C += (duty_kw * 1000.0) / (props["m_total_kgs"] * Cp)
            out.append({"seg": label, "pipe": pipe_label, "regime": "Heat Exchanger"})
            current_P -= seg.get("dp_kpa", 0.0) * 1000.0
            current_P  = max(1e4, current_P)
            continue

        D_seg  = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        lined  = seg.get("lined", False)
        lthk_m = seg.get("liner_thickness_mm", 1.0) / 1000.0
        lmat   = seg.get("liner_material", "FEP")
        D_eff  = D_seg - 2 * lthk_m if lined else D_seg
        rough  = (engine.LINER_ROUGHNESS[lmat] if lined
                  else engine.MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
        if _is_vle_res:
            props = engine.calculate_vle_properties(
                res["vle_fluid"], current_P / 1e5,
                res["vle_x_mass"], res["vle_m_total_kgs"])
        else:
            props  = engine.calculate_two_phase_properties(
                current_P / 1e5, current_T_C,
                res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
                custom_gas=cgas, custom_liquid=cliq,
                liquid_flows_kgh=res.get("liquid_flows_kgh"))
        angle  = {"Horizontal": 0.0, "Vertical Upflow": np.pi / 2.0,
                  "Vertical Downflow": -np.pi / 2.0}[seg["type"]]
        le_fit = _sum_le_fit(seg, D_eff)
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)
        out.append({
            "seg":    label,
            "pipe":   pipe_label,
            "regime": seg_res["regime"],
        })
        current_P -= seg_res["dP_Pa"]
        current_P  = max(1e4, current_P)
    return out


def _march_header_simple(tap_dists, gas_per_tap, liq_per_tap,
                          hdr, P_start_Pa, T_C, liquid_type, corr, void,
                          custom_liquid=None):
    """Pressure-march one header arm from farthest tap to T-junction.

    tap_dists    : list of distances from T (m) for each tap — sorted internally.
    gas_per_tap  : {species: kg/h} for ONE A-line.
    liq_per_tap  : m³/h liquid for ONE A-line.
    hdr          : dict with dn, pn, material, lined, liner_material,
                   liner_thickness_mm, fittings_list.
    custom_liquid: optional {rho_kgm3, mu_mpas, sigma_mnm} for Custom liquid type.
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
            current_P / 1e5, T_C, running_gas, liquid_type, running_liq,
            custom_liquid=custom_liquid)
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


def _march_single_seg(seg, P_in_Pa, T_C, gas_flows, liquid_type, q_lye, corr, void,
                       custom_liquid=None):
    """ΔP for one horizontal pipe segment carrying combined flow.
    Returns (dp_Pa, P_out_Pa, dp_fric_Pa, dp_grav_Pa, record_dict).
    """
    D_nom = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
    mat   = seg.get("material", "SS316L")
    rough = engine.MATERIAL_ROUGHNESS.get(mat, engine.MATERIAL_ROUGHNESS["SS316L"])
    le_fit = _sum_le_fit(seg, D_nom)

    props   = engine.calculate_two_phase_properties(
        P_in_Pa / 1e5, T_C, gas_flows, liquid_type, q_lye,
        custom_liquid=custom_liquid)
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
    hdr   = res_hdr["header_pipe"]
    gpt   = res_hdr["gas_per_tap"]
    lpt   = res_hdr["liq_per_tap"]
    T_C   = res_hdr["T_C"]
    liq   = res_hdr["liquid_type"]
    cliq  = res_hdr.get("custom_liquid")
    corr  = res_hdr.get("correlation",    engine.TWO_PHASE_CORRELATIONS[0])
    void  = res_hdr.get("voidage_method", engine.VOIDAGE_METHODS[0])

    dp_l, P_T_l, *_ = _march_header_simple(
        res_hdr["left_taps"],  gpt, lpt, hdr, P_start, T_C, liq, corr, void,
        custom_liquid=cliq)
    dp_r, P_T_r, *_ = _march_header_simple(
        res_hdr["right_taps"], gpt, lpt, hdr, P_start, T_C, liq, corr, void,
        custom_liquid=cliq)

    dp_worst = max(dp_l, dp_r)
    P_T      = min(P_T_l, P_T_r)      # worst-arm pressure arriving at T

    t_seg = res_hdr.get("t_seg")
    if t_seg and t_seg.get("length", 0) > 0:
        dp_t, P_sep, *_ = _march_single_seg(
            t_seg, P_T, T_C, res_hdr["gas_flows_kgh"], liq,
            res_hdr["q_lye"], corr, void, custom_liquid=cliq)
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


def _forward_stack(res_line, res_hdr, P_source_bara):
    """Forward march from a fixed inlet pressure to the separator.
    No iteration — deterministic in one pass.
    Returns a result dict compatible with _goal_seek_stack output.
    """
    dp_line, P_line_out = _calc_dp_at_p(res_line, P_source_bara)
    dp_hdr,  P_sep      = _calc_header_dp_at_p(res_hdr, P_line_out)
    return {
        "P_line_in":  P_source_bara,
        "P_line_out": P_line_out,
        "P_sep":      P_sep,
        "dp_line":    dp_line,
        "dp_hdr":     dp_hdr,
        "converged":  True,
        "iterations": 1,
    }


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
        P_line_in = max(0.1, P_line_in)  # prevent runaway below hard vacuum

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
    _ra          = results_a or {}
    gas_per_tap  = dict(_ra.get("gas_flows_kgh") or {"H₂": 5.0})
    liq_per_tap  = float(_ra.get("q_lye") or 0.5)
    liquid_type  = str(_ra.get("liquid_type") or "Custom")
    custom_liquid_hdr = _ra.get("custom_liquid")   # None for single-species CoolProp
    T_C_a        = float(_ra.get("T_C") or 60.0)

    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("Inputs")

        with st.container(border=True):
            st.markdown("**Fluid — from Case A (read-only)**")
            _gas_str = "  ·  ".join(f"{sp}: {v:.2f} kg/h" for sp, v in gas_per_tap.items())
            st.caption(f"Gas per tap:  {_gas_str}")
            st.caption(f"Liquid per tap:  {liq_per_tap:.3f} m³/h  ·  {liquid_type or 'none'}")

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
            "custom_liquid":  custom_liquid_hdr,
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
            hdr_spec, P_start, T_C, liquid_type, correlation, voidage_method,
            custom_liquid=custom_liquid_hdr)
        dp_r_Pa, P_T_r, fric_r, grav_r, rec_r = _march_header_simple(
            right_positions, gas_per_tap, liq_per_tap,
            hdr_spec, P_start, T_C, liquid_type, correlation, voidage_method,
            custom_liquid=custom_liquid_hdr)

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
                    total_liq, correlation, voidage_method,
                    custom_liquid=custom_liquid_hdr)
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
    st.session_state["label_a"] = "A"
if "label_b" not in st.session_state:
    st.session_state["label_b"] = "B"

# Forward any pending goal-seek apply values BEFORE widgets render
if "stack_apply_a_pending" in st.session_state:
    st.session_state["a_P_bara"] = max(0.1, st.session_state.pop("stack_apply_a_pending"))
if "stack_apply_b_pending" in st.session_state:
    st.session_state["b_P_bara"] = max(0.1, st.session_state.pop("stack_apply_b_pending"))

_la = st.session_state["label_a"]
_lb = st.session_state["label_b"]
_lc = f"Header {_la}"
_ld = f"Header {_lb}"

tab_a, tab_b, tab_c, tab_d, tab_cmp, tab_gs = st.tabs(
    [_la, _lb, _lc, _ld, "Compare", "Goal Seek"])

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
            "segs": [(s.get("type", s.get("kind","")), s.get("dn",""), s.get("pn",""), float(s.get("length",0.0)),
                      tuple(sorted((f["type"],f["qty"]) for f in s.get("fittings_list",[]))),
                      bool(s.get("lined", False)), s.get("liner_material", "FEP"),
                      float(s.get("liner_thickness_mm", 1.0)))
                     for s in ra["segments"]],
        },
        "b": {
            "P": rb["P_bara"], "T": rb["T_C"],
            "gas": {_k: float(_v) for _k, _v in rb["gas_flows_kgh"].items()},
            "liq": rb["liquid_type"], "lye": rb["q_lye"],
            "segs": [(s.get("type", s.get("kind","")), s.get("dn",""), s.get("pn",""), float(s.get("length",0.0)),
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
        _cmp_row("Worst erosion ratio V_m/V_e",  _max_a, _max_b,
                 fmt="{:.3f}", unit="— (limit 1.0)", better="lower")

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
                ra["P_bara"], ra.get("T_C"), ra["gas_flows_kgh"], ra["liquid_type"], ra["q_lye"],
                ra["segments"],
                custom_gas=ra.get("custom_gas"), custom_liquid=ra.get("custom_liquid"),
                vle_fluid=ra.get("vle_fluid"), vle_x_mass=ra.get("vle_x_mass"),
                vle_m_total_kgs=ra.get("vle_m_total_kgs"))
            st.session_state["sens_b"] = engine.run_sensitivity(
                rb["P_bara"], rb.get("T_C"), rb["gas_flows_kgh"], rb["liquid_type"], rb["q_lye"],
                rb["segments"],
                custom_gas=rb.get("custom_gas"), custom_liquid=rb.get("custom_liquid"),
                vle_fluid=rb.get("vle_fluid"), vle_x_mass=rb.get("vle_x_mass"),
                vle_m_total_kgs=rb.get("vle_m_total_kgs"))
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
                orient   = seg.get("type", seg.get("kind", "—")).replace(
                               "Vertical Upflow", "V Up").replace(
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

    # ── Reports ───────────────────────────────────────────────────────────────
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

    # ── Comparison report ─────────────────────────────────────────────────────
    _cmp_btn_col, _cmp_dl_col = st.columns(2)
    with _cmp_btn_col:
        if st.button("Generate Comparison (tables only)", type="primary",
                     use_container_width=True, key="gen_cmp_rpt"):
            with st.spinner("Building comparison report…"):
                try:
                    _sd = _build_sens_data()
                    _cbuf = report_generator.generate_comparison_report(
                        results_a=ra, results_b=rb,
                        label_a=_la, label_b=_lb,
                        fig_cmp=None, fig_bar=None,
                        sensitivity_data={**_sd, "fig": None} if _sd else None,
                        stack_dp=_build_stack_dp_data())
                    st.session_state["cmp_rpt_bytes"] = _cbuf.getvalue()
                    st.success("Ready.")
                except Exception as _e:
                    st.error(f"Failed: {_e}")
        if st.button("Generate Comparison with charts", use_container_width=True,
                     key="gen_cmp_rpt_ch"):
            with st.spinner("Building comparison report with charts…"):
                try:
                    _cbuf = report_generator.generate_comparison_report(
                        results_a=ra, results_b=rb,
                        label_a=_la, label_b=_lb,
                        fig_cmp=fig_cmp, fig_bar=fig_bar,
                        sensitivity_data=_build_sens_data(),
                        stack_dp=_build_stack_dp_data())
                    st.session_state["cmp_rpt_bytes_ch"] = _cbuf.getvalue()
                    st.success("Ready.")
                except Exception as _e:
                    st.error(f"Failed: {_e}")
    with _cmp_dl_col:
        if st.session_state.get("cmp_rpt_bytes"):
            st.download_button(
                "Download Comparison tables  (.docx)",
                data=st.session_state["cmp_rpt_bytes"],
                file_name="report_comparison.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_cmp_rpt")
        if st.session_state.get("cmp_rpt_bytes_ch"):
            st.download_button(
                "Download Comparison with charts  (.docx)",
                data=st.session_state["cmp_rpt_bytes_ch"],
                file_name="report_comparison_charts.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_cmp_rpt_ch")

    # ── Combined report ───────────────────────────────────────────────────────
    _cmb_btn_col, _cmb_dl_col = st.columns(2)
    with _cmb_btn_col:
        if st.button("Generate Combined (tables only)", type="primary",
                     use_container_width=True, key="gen_combined_rpt"):
            with st.spinner("Building combined report…"):
                try:
                    _sd = _build_sens_data()
                    _combined_buf = report_generator.generate_combined_report(
                        cases=[ra, rb, results_c, results_d],
                        case_labels=[_la, _lb, _lc, _ld],
                        fig_cmp=None, fig_bar=None,
                        sensitivity_data={**_sd, "fig": None} if _sd else None,
                        stack_dp=_build_stack_dp_data(),
                        dn_study_data=_build_dn_study_data())
                    st.session_state["combined_rpt_bytes"] = _combined_buf.getvalue()
                    st.success("Ready.")
                except Exception as _e:
                    st.error(f"Failed: {_e}")
        if st.button("Generate Combined with charts", use_container_width=True,
                     key="gen_combined_rpt_ch"):
            with st.spinner("Building combined report with charts…"):
                try:
                    _combined_buf = report_generator.generate_combined_report(
                        cases=[ra, rb, results_c, results_d],
                        case_labels=[_la, _lb, _lc, _ld],
                        fig_cmp=fig_cmp, fig_bar=fig_bar,
                        sensitivity_data=_build_sens_data(),
                        stack_dp=_build_stack_dp_data(),
                        dn_study_data=_build_dn_study_data())
                    st.session_state["combined_rpt_bytes_ch"] = _combined_buf.getvalue()
                    st.success("Ready.")
                except Exception as _e:
                    st.error(f"Failed: {_e}")
    with _cmb_dl_col:
        if st.session_state.get("combined_rpt_bytes"):
            st.download_button(
                "Download Combined tables  (.docx)",
                data=st.session_state["combined_rpt_bytes"],
                file_name="report_combined.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_combined_rpt")
        if st.session_state.get("combined_rpt_bytes_ch"):
            st.download_button(
                "Download Combined with charts  (.docx)",
                data=st.session_state["combined_rpt_bytes_ch"],
                file_name="report_combined_charts.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True, key="dl_combined_rpt_ch")

    st.caption("ℹ Individual case reports (A, B) are available in their respective tabs.")
    if not _sens_avail:
        st.caption("ℹ Run the Sensitivity Analysis above first to include it in the reports.")
    if not st.session_state.get("stack_gsr_h2"):
        st.caption("ℹ Run the Goal Seek calculation first to include it in the reports.")
    if not st.session_state.get("dn_study_dn_primary"):
        st.caption("ℹ Run the DN Study first to include it in the Combined report.")

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
# GOAL SEEK TAB
# ============================================================================
with tab_gs:
    _la = st.session_state.get("label_a", "Case A")
    _lb = st.session_state.get("label_b", "Case B")
    _lc = f"{_la} Header"
    _ld = f"{_lb} Header"

    st.subheader("Goal Seek")

    _gs_direction = st.radio(
        "Mode",
        [
            "Fix separator pressure — back-calculate inlet",
            "Fix source pressure — forward-calculate separator",
        ],
        horizontal=True,
        key="gs_direction",
        label_visibility="collapsed",
    )
    _sink_mode = _gs_direction.startswith("Fix separator")

    # ── Sink-fixed mode (back-calculation) ───────────────────────────────────
    if _sink_mode:
        st.caption(
            f"Goal-seek both the {_la} system ({_la} → {_lc}) and {_lb} system ({_lb} → {_ld}) "
            "to find the required line inlet pressures for given separator pressures. "
            f"**Generator ΔP** = P_inlet_{_la} − P_inlet_{_lb}."
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
                    help=f"Target pressure at the {_la} gas-liquid separator.",
                )
            with _sk_col2:
                _p_sep_o2 = st.number_input(
                    f"{_lb} separator pressure (bara)",
                    min_value=0.1, max_value=200.0,
                    value=float(round(results_d.get("P_separator_bara",
                                                    results_b["outlet_pressure_bara"]), 3)),
                    step=0.1, format="%.3f",
                    key="stack_p_sep_o2",
                    help=f"Target pressure at the {_lb} gas-liquid separator.",
                )
            _sk_run = st.button("Calculate", type="primary",
                                use_container_width=True, key="stack_run")

        if _sk_run:
            with st.spinner(f"Solving {_la} and {_lb} systems…"):
                _gsr_h2 = _goal_seek_stack(results_a, results_c, _p_sep_h2)
                _gsr_o2 = _goal_seek_stack(results_b, results_d, _p_sep_o2)
            st.session_state["stack_gsr_h2"] = _gsr_h2
            st.session_state["stack_gsr_o2"] = _gsr_o2
            st.session_state["stack_sep_h2"] = _p_sep_h2
            st.session_state["stack_sep_o2"] = _p_sep_o2
            st.session_state["gs_mode_used"] = "sink"

    # ── Source-fixed mode (forward march) ────────────────────────────────────
    else:
        st.caption(
            f"Fix the {_la} and {_lb} source (inlet) pressures and march forward through "
            f"the branch pipeline and header to find the resulting separator pressures. "
            f"**Generator ΔP** = P_source_{_la} − P_source_{_lb} (set by you)."
        )

        with st.container(border=True):
            st.markdown("##### Source (Inlet) Pressures")
            _fwd_col1, _fwd_col2 = st.columns(2)
            with _fwd_col1:
                _p_src_h2 = st.number_input(
                    f"{_la} source pressure (bara)",
                    min_value=0.1, max_value=200.0,
                    value=float(round(results_a.get("inlet_pressure_bara",
                                                    results_a["outlet_pressure_bara"]), 3)),
                    step=0.1, format="%.3f",
                    key="gs_p_src_h2",
                    help=f"Fixed inlet pressure for the {_la} branch.",
                )
            with _fwd_col2:
                _p_src_o2 = st.number_input(
                    f"{_lb} source pressure (bara)",
                    min_value=0.1, max_value=200.0,
                    value=float(round(results_b.get("inlet_pressure_bara",
                                                    results_b["outlet_pressure_bara"]), 3)),
                    step=0.1, format="%.3f",
                    key="gs_p_src_o2",
                    help=f"Fixed inlet pressure for the {_lb} branch.",
                )
            _fwd_run = st.button("Calculate", type="primary",
                                 use_container_width=True, key="gs_fwd_run")

        if _fwd_run:
            with st.spinner(f"Marching {_la} and {_lb} systems forward…"):
                _gsr_h2 = _forward_stack(results_a, results_c, _p_src_h2)
                _gsr_o2 = _forward_stack(results_b, results_d, _p_src_o2)
            st.session_state["stack_gsr_h2"] = _gsr_h2
            st.session_state["stack_gsr_o2"] = _gsr_o2
            st.session_state["stack_sep_h2"] = _gsr_h2["P_sep"]
            st.session_state["stack_sep_o2"] = _gsr_o2["P_sep"]
            st.session_state["gs_mode_used"] = "source"

    # ── Shared results display ────────────────────────────────────────────────
    _gsr_h2 = st.session_state.get("stack_gsr_h2")
    _gsr_o2 = st.session_state.get("stack_gsr_o2")

    if _gsr_h2 and _gsr_o2:
        _shown_sep_h2 = st.session_state.get("stack_sep_h2", 0.0)
        _shown_sep_o2 = st.session_state.get("stack_sep_o2", 0.0)

        _conv_h2 = _gsr_h2["converged"]
        _conv_o2 = _gsr_o2["converged"]
        if _conv_h2 and _conv_o2:
            _iter_note = (
                f"{_la} converged in {_gsr_h2['iterations']} iter.  "
                f"{_lb} converged in {_gsr_o2['iterations']} iter."
                if _sink_mode else
                f"Forward march complete — {_la} and {_lb} solved in one pass."
            )
            st.success(_iter_note)
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
            st.markdown(f"##### Generator ΔP  (P_inlet_{_la} − P_inlet_{_lb})")
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
    # DN STUDY (merged into Goal Seek tab)
    # ============================================================================
    st.divider()
    st.subheader("DN Study — Branch Line Size Comparison")
    st.caption(
        "Re-runs the full system (branches A and B + goal-seek) with a different branch DN. "
        "Header sizes are unchanged. All other inputs (flows, pressure, correlation) are identical."
    )

    _prereq_ok = (
        results_a is not None and results_b is not None
        and results_c is not None and results_d is not None
    )
    _gsr_ok = st.session_state.get("gs_mode_used") is not None

    if not _prereq_ok:
        st.warning("Run all four cases (A, B, C, D) before using DN Study.")
    elif not _gsr_ok:
        st.warning("Run the Goal Seek calculation above before using DN Study.")
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
                    f"Separator targets (from Goal Seek above):  "
                    f"{_la} {_p_sep_h2_dn:.2f} bara  ·  {_lb} {_p_sep_o2_dn:.2f} bara"
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
                _reg_a_alt  = _calc_regimes_at_p(_ra_alt, _gsr_h2_alt["P_line_in"])
                _reg_b_alt  = _calc_regimes_at_p(_rb_alt, _gsr_o2_alt["P_line_in"])
            st.session_state["dn_study_dn_primary"] = _dn_primary
            st.session_state["dn_study_dn_alt"]     = dn_alt
            st.session_state["dn_study_gsr_h2_alt"] = _gsr_h2_alt
            st.session_state["dn_study_gsr_o2_alt"] = _gsr_o2_alt
            st.session_state["dn_study_regimes_a"]  = _reg_a_alt
            st.session_state["dn_study_regimes_b"]  = _reg_b_alt
            st.rerun()

        _gsr_h2_p = st.session_state.get("stack_gsr_h2")
        _gsr_o2_p = st.session_state.get("stack_gsr_o2")
        _gsr_h2_a = st.session_state.get("dn_study_gsr_h2_alt")
        _gsr_o2_a = st.session_state.get("dn_study_gsr_o2_alt")
        _dn_p_lbl = st.session_state.get("dn_study_dn_primary", _dn_primary)
        _dn_a_lbl = st.session_state.get("dn_study_dn_alt", "")

        if _gsr_h2_a and _gsr_o2_a and _dn_a_lbl:
            with _res_col:
                # ── Generator ΔP ──────────────────────────────────────────────
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

                # ── Velocity estimate (first segment, ID-ratio scaling) ────────
                _seg0     = results_a["segments"][0]
                _pn0      = _seg0["pn"]
                _lined0   = _seg0.get("lined", False)
                _lthk0_m  = _seg0.get("liner_thickness_mm", 1.0) / 1000.0
                _D_p_bore = engine.PIPE_DATABASE[_dn_p_lbl].get(
                    _pn0, list(engine.PIPE_DATABASE[_dn_p_lbl].values())[0])
                _D_a_bore = engine.PIPE_DATABASE[_dn_a_lbl].get(
                    _pn0, list(engine.PIPE_DATABASE[_dn_a_lbl].values())[0])
                _D_p_eff  = _D_p_bore - 2 * _lthk0_m if _lined0 else _D_p_bore
                _D_a_eff  = _D_a_bore - 2 * _lthk0_m if _lined0 else _D_a_bore
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

                # ── Flow Regime ───────────────────────────────────────────────
                _reg_a_stored = st.session_state.get("dn_study_regimes_a", [])
                _reg_b_stored = st.session_state.get("dn_study_regimes_b", [])
                if _reg_a_stored or _reg_b_stored:
                    with st.container(border=True):
                        st.markdown("**Flow Regime by Segment**")
                        _rr1, _rr2 = st.columns(2)
                        with _rr1:
                            st.caption(f"*{_la} branch*")
                            _prim_a_recs = results_a.get("grid_records", [])
                            _rrA = []
                            for _ri, _ar in enumerate(_reg_a_stored):
                                _pr = _prim_a_recs[_ri]["Regime"] if _ri < len(_prim_a_recs) else "—"
                                _rrA.append({
                                    "Seg":     _ar["seg"],
                                    _dn_p_lbl: _pr,
                                    _dn_a_lbl: _ar["regime"],
                                    "Changed": "⚠" if _pr != _ar["regime"] else "✓",
                                })
                            st.dataframe(pd.DataFrame(_rrA), hide_index=True,
                                         use_container_width=True)
                        with _rr2:
                            st.caption(f"*{_lb} branch*")
                            _prim_b_recs = results_b.get("grid_records", [])
                            _rrB = []
                            for _ri, _br in enumerate(_reg_b_stored):
                                _pr = _prim_b_recs[_ri]["Regime"] if _ri < len(_prim_b_recs) else "—"
                                _rrB.append({
                                    "Seg":     _br["seg"],
                                    _dn_p_lbl: _pr,
                                    _dn_a_lbl: _br["regime"],
                                    "Changed": "⚠" if _pr != _br["regime"] else "✓",
                                })
                            st.dataframe(pd.DataFrame(_rrB), hide_index=True,
                                         use_container_width=True)

                # ── Recommendation ────────────────────────────────────────────
                _vel_ok        = _ratio_a <= 1.0 and _ratio_b <= 1.0
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
                        f"{_dn_a_lbl} appears undersized for this duty."
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
