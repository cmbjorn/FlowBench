"""
Pump hydraulics engine for FlowBench.

Covers:
- Fluid properties (KOH solution + CoolProp liquids) including vapour pressure
- H-Q curve fitting (3-point parametric or tabular, quadratic polynomial)
- Efficiency curve (parabolic)
- System curve (static head + friction)
- Operating point (bisection solver)
- NPSH available
- Shaft and motor power
- Pump design pressure (six methods per the standards table)
- ANSI B16.5 pressure class lookup (Groups 1.1 and 2.3)

No Streamlit imports — pure calculation only.
"""

import math

# ── Constants ─────────────────────────────────────────────────────────────────
g = 9.80665          # m/s²
R_u = 8.31446        # J/mol·K


# ============================================================================
# 1. FLUID PROPERTIES
# ============================================================================

def koh_properties(T_C: float, conc_wt_pct: float):
    """
    ρ (kg/m³), μ (Pa·s), Pv (bara) for aqueous KOH at T_C, conc_wt_pct wt%.
    Pv uses CoolProp water saturation pressure × activity coefficient (Raoult
    + KOH ionic dissociation correction).
    """
    import multiphase_engine as eng
    rho, mu, _ = eng.koh_properties(T_C, conc_wt_pct)

    # Vapour pressure: CoolProp water Psat × water activity in KOH
    try:
        import CoolProp.CoolProp as CP
        Pv_water_Pa = CP.PropsSI('P', 'T', T_C + 273.15, 'Q', 0, 'Water')
    except Exception:
        # Fallback: Antoine equation for water (Pa)
        T_K = T_C + 273.15
        Pv_water_Pa = math.exp(23.1964 - 3816.44 / (T_K - 46.13))

    # KOH activity correction — Raoult's law with ionic dissociation (K⁺ + OH⁻)
    # x_ions = 2 × n_KOH / (2 × n_KOH + n_H2O)
    w = max(0.0, min(40.0, conc_wt_pct)) / 100.0
    if w > 0:
        n_koh  = (w / 0.0561)          # mol KOH per kg solution
        n_h2o  = ((1 - w) / 0.018015)  # mol H2O per kg solution
        x_ions = 2 * n_koh / (2 * n_koh + n_h2o)
        a_water = max(0.3, 1.0 - x_ions)
    else:
        a_water = 1.0

    Pv_bara = Pv_water_Pa * a_water / 1e5
    return rho, mu, Pv_bara


def coolprop_liquid_properties(fluid_name: str, T_C: float, P_bara: float):
    """
    ρ (kg/m³), μ (Pa·s), Pv (bara) for a CoolProp liquid.
    fluid_name must be a key in multiphase_engine.LIQUID_COOLPROP_ID.
    """
    import multiphase_engine as eng
    import CoolProp.CoolProp as CP

    cp_id = eng.LIQUID_COOLPROP_ID.get(fluid_name, fluid_name)
    T_K   = T_C + 273.15
    P_pa  = P_bara * 1e5

    try:
        rho = CP.PropsSI('D', 'T', T_K, 'P', P_pa, cp_id)
        mu  = CP.PropsSI('V', 'T', T_K, 'P', P_pa, cp_id)
    except Exception:
        rho, mu = 1000.0, 1e-3

    try:
        Pv_pa = CP.PropsSI('P', 'T', T_K, 'Q', 0, cp_id)
        Pv_bara = Pv_pa / 1e5
    except Exception:
        Pv_bara = 0.023   # conservative fallback (~water at 20 °C)

    return rho, mu, Pv_bara


# ============================================================================
# 2. H-Q CURVE  (centrifugal pump, quadratic: H = a + b·Q + c·Q²)
# ============================================================================

def fit_hq_3point(H_shutoff: float, H_bep: float, Q_bep: float,
                  H_runout: float = None, Q_runout: float = None):
    """
    Fit quadratic H-Q polynomial from shut-off, BEP, and optional runout.
    Returns (a, b, c) for  H = a + b·Q + c·Q²   (Q in m³/h, H in m).

    Without runout: forces dH/dQ = 0 at Q = 0 (flat top), so b = 0.
      a = H_shutoff,  c = (H_bep - H_shutoff) / Q_bep²

    With runout: full 3-point solve.
    """
    if Q_bep <= 0:
        raise ValueError("Q_bep must be > 0")
    if H_shutoff <= H_bep:
        raise ValueError("Shut-off head must exceed BEP head")

    if H_runout is None or Q_runout is None:
        a = H_shutoff
        b = 0.0
        c = (H_bep - H_shutoff) / (Q_bep ** 2)
    else:
        # Solve:  [1  0    0  ][a]   [H_shutoff]
        #         [1  Q_b  Q_b²][b] = [H_bep    ]
        #         [1  Q_r  Q_r²][c]   [H_runout ]
        Q_b, Q_r = Q_bep, Q_runout
        det = Q_b * Q_r ** 2 - Q_r * Q_b ** 2
        if abs(det) < 1e-12:
            raise ValueError("Q_bep and Q_runout too close — cannot fit curve")
        a = H_shutoff
        b = ((H_bep - a) * Q_r ** 2 - (H_runout - a) * Q_b ** 2) / det
        c = ((H_runout - a) * Q_b  -  (H_bep - a) * Q_r) / det

    return (a, b, c)


def fit_hq_tabular(Q_points, H_points):
    """
    Least-squares quadratic fit to tabular Q/H data.
    Returns (a, b, c).
    """
    import numpy as np
    Q = np.array(Q_points, dtype=float)
    H = np.array(H_points, dtype=float)
    coeffs = np.polyfit(Q, H, 2)   # [c, b, a]  (numpy highest-power first)
    return (float(coeffs[2]), float(coeffs[1]), float(coeffs[0]))


def eval_hq(coeffs, Q_m3h: float) -> float:
    """H (m) at flow Q (m³/h). Returns max(0, H)."""
    a, b, c = coeffs
    return max(0.0, a + b * Q_m3h + c * Q_m3h ** 2)


def scale_hq_to_speed(coeffs, n_ratio: float):
    """
    Apply affinity laws to scale H-Q coefficients to a different speed.
    n_ratio = n_new / n_rated.
    Q scales as n, H as n² → coefficients transform as:
      a_new = a · n²,  b_new = b · n,  c_new = c   (unchanged)
    """
    a, b, c = coeffs
    return (a * n_ratio ** 2, b * n_ratio, c)


def hq_shutoff(coeffs) -> float:
    """Head at Q = 0 (shut-off head in metres)."""
    return coeffs[0]


def hq_max_flow(coeffs, H_min: float = 0.0, Q_max_search: float = 5000.0) -> float:
    """Flow at which H = H_min (runout / zero-head point), via bisection."""
    if eval_hq(coeffs, 0) <= H_min:
        return 0.0
    lo, hi = 0.0, Q_max_search
    for _ in range(60):
        mid = (lo + hi) / 2
        if eval_hq(coeffs, mid) > H_min:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================================
# 3. EFFICIENCY CURVE  (parabolic, peak at Q_bep)
# ============================================================================

def fit_eta_parabolic(eta_bep_pct: float, Q_bep: float, Q_shutoff: float = 0.0,
                      Q_runout: float = None):
    """
    Returns (eta_bep, Q_bep, Q_half_width) for eval_eta().
    Q_half_width is chosen so η → 0 at shut-off and runout.
    """
    if Q_runout is None:
        Q_runout = 2.0 * Q_bep
    half = max(Q_bep - Q_shutoff, Q_runout - Q_bep)
    return (eta_bep_pct, Q_bep, half)


def eval_eta(eta_params, Q_m3h: float) -> float:
    """η (%) at Q. Parabolic, bounded 0–100%."""
    eta_bep, Q_bep, half = eta_params
    return max(0.0, min(100.0, eta_bep * (1.0 - ((Q_m3h - Q_bep) / half) ** 2)))


# ============================================================================
# 4. SYSTEM CURVE
# ============================================================================

def system_head(Q_m3h: float, H_static_m: float, k_friction: float) -> float:
    """
    H_sys (m) = H_static + k_friction × Q².
    H_static can be negative (pumping downhill / suction above discharge).
    k_friction ≥ 0 (m / (m³/h)²).
    """
    return H_static_m + k_friction * Q_m3h ** 2


def k_from_reference(Q_ref_m3h: float, H_friction_ref_m: float) -> float:
    """
    Derive k_friction (m/(m³/h)²) from one (Q, H_friction) reference point.
    k = H_friction / Q².
    """
    if Q_ref_m3h <= 0:
        return 0.0
    return H_friction_ref_m / (Q_ref_m3h ** 2)


def head_to_bar(H_m: float, rho_kgm3: float) -> float:
    """Convert metres of head to bar differential pressure."""
    return H_m * rho_kgm3 * g / 1e5


# ============================================================================
# 5. OPERATING POINT
# ============================================================================

def find_operating_point(coeffs_hq, H_static_m: float, k_friction: float,
                          Q_max_m3h: float = 2000.0):
    """
    Bisection solver for intersection of H_pump(Q) and H_sys(Q).
    Returns (Q_op, H_op) or raises ValueError if no crossing found.
    """
    def f(Q):
        return eval_hq(coeffs_hq, Q) - system_head(Q, H_static_m, k_friction)

    # Check boundary conditions
    f0 = f(0.0)
    fmax = f(Q_max_m3h)

    if f0 <= 0:
        # Pump head at Q=0 is already below system curve — no intersection
        return (0.0, eval_hq(coeffs_hq, 0.0))

    if fmax >= 0:
        # Pump curve always above system curve in range — return max flow
        return (Q_max_m3h, eval_hq(coeffs_hq, Q_max_m3h))

    lo, hi = 0.0, Q_max_m3h
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid

    Q_op = (lo + hi) / 2
    H_op = eval_hq(coeffs_hq, Q_op)
    return (Q_op, H_op)


# ============================================================================
# 6. NPSH AVAILABLE
# ============================================================================

def npsh_available(P_suction_bara: float, Pv_bara: float, rho_kgm3: float,
                   z_suction_m: float, h_loss_suction_m: float) -> float:
    """
    NPSH_A (m) = (P_suction − Pv) × 1e5 / (ρ × g) + z_suction − h_loss_suction

    P_suction_bara : absolute pressure at the suction vessel / reference point
    Pv_bara        : vapour pressure of liquid at pump inlet temperature
    z_suction_m    : liquid level above pump centreline (+ve if flooded, -ve if suction lift)
    h_loss_suction_m: friction + fitting losses in suction line (always positive, reduces NPSH_A)
    """
    pressure_head = (P_suction_bara - Pv_bara) * 1e5 / (rho_kgm3 * g)
    return pressure_head + z_suction_m - h_loss_suction_m


def npsh_margin_status(npsh_a: float, npsh_r: float):
    """Returns (margin_m, status_str, color) where color is 'green'/'amber'/'red'."""
    margin = npsh_a - npsh_r
    if margin < 0:
        return margin, "NPSH deficit — cavitation certain", "red"
    elif margin < 0.5:
        return margin, "Margin < 0.5 m — cavitation risk", "red"
    elif margin < 1.5:
        return margin, "Margin < 1.5 m — monitor closely", "orange"
    else:
        return margin, "Adequate margin", "green"


# ============================================================================
# 7. POWER
# ============================================================================

def shaft_power_kw(rho_kgm3: float, Q_m3h: float, H_m: float,
                   eta_pump_pct: float) -> float:
    """Hydraulic shaft power (kW). P = ρgQH / η."""
    if eta_pump_pct <= 0 or Q_m3h <= 0:
        return 0.0
    Q_m3s = Q_m3h / 3600.0
    return rho_kgm3 * g * Q_m3s * H_m / (eta_pump_pct / 100.0) / 1000.0


def motor_power_kw(P_shaft_kw: float, eta_motor_pct: float) -> float:
    """Electrical input power to motor (kW)."""
    if eta_motor_pct <= 0:
        return 0.0
    return P_shaft_kw / (eta_motor_pct / 100.0)


from standards.electrical import next_motor_frame_kw


# ============================================================================
# 8. PUMP DESIGN PRESSURE
# ============================================================================

# Method identifiers matching the standards table
DESIGN_PRESSURE_METHODS = {
    1: "Full shut-in, max VSD speed  (NORSOK P-001 / API 610)",
    2: "Full shut-in, rated speed  (API 610 + B31.3 strict)",
    3: "Full shut-in, upstream PSV accumulation as suction basis  (B31.3)",
    4: "PSV on pump discharge — standard process 10 %  (B31.3 + PSV)",
    5: "PSV on pump discharge — fire case 21 %  (B31.3 / ASME Sec VIII)",
    6: "PSV-governed + SIL-rated SIS credit  (EN 13480 / PED)",
}


def centrifugal_design_pressure(
    method: int,
    # Inputs for shut-in methods (1, 2, 3)
    P_suction_bara: float   = 1.0,    # suction reference pressure (bara)
    H_shutoff_rated_m: float = 50.0,  # shut-off head at rated speed (m)
    rho_kgm3: float          = 1000.0,
    n_max_ratio: float       = 1.0,   # n_max / n_rated (VSD max speed ratio)
    # Inputs for PSV methods (4, 5, 6)
    PSV_set_barg: float      = None,
    accumulation_pct: float  = 10.0,  # overridden to 21 % for fire, kept for method 6
    Patm_bara: float         = 1.01325,
) -> dict:
    """
    Calculate pump discharge design pressure for centrifugal pump.

    Returns dict with keys:
        P_design_bara, P_design_barg, method, method_label,
        governing_scenario, notes
    """
    label = DESIGN_PRESSURE_METHODS.get(method, "Unknown method")

    if method == 1:
        H_shutoff_max = H_shutoff_rated_m * (n_max_ratio ** 2)
        dP_shutin_bar = H_shutoff_max * rho_kgm3 * g / 1e5
        P_design_bara = P_suction_bara + dP_shutin_bar
        notes = (f"H₀(max speed) = {H_shutoff_max:.1f} m  "
                 f"(rated {H_shutoff_rated_m:.1f} m × {n_max_ratio:.3f}²).  "
                 f"No credit for PSV or protective systems.")

    elif method == 2:
        dP_shutin_bar = H_shutoff_rated_m * rho_kgm3 * g / 1e5
        P_design_bara = P_suction_bara + dP_shutin_bar
        notes = (f"H₀ = {H_shutoff_rated_m:.1f} m at rated speed.  "
                 f"No credit for PSV or protective systems.")

    elif method == 3:
        # Suction basis is the upstream PSV accumulation pressure (already passed as P_suction_bara)
        dP_shutin_bar = H_shutoff_rated_m * rho_kgm3 * g / 1e5
        P_design_bara = P_suction_bara + dP_shutin_bar
        notes = (f"Suction basis = upstream PSV accumulation pressure "
                 f"({P_suction_bara:.2f} bara).  "
                 f"H₀ = {H_shutoff_rated_m:.1f} m at rated speed.")

    elif method in (4, 5, 6):
        if PSV_set_barg is None:
            raise ValueError("PSV_set_barg required for methods 4, 5, 6")
        if method == 5:
            accumulation_pct = 21.0
        acc = accumulation_pct / 100.0
        P_design_bara = (PSV_set_barg + Patm_bara) * (1.0 + acc)
        notes = (f"PSV set = {PSV_set_barg:.2f} barg.  "
                 f"Accumulation = {accumulation_pct:.0f} %.  "
                 + ("SIL-rated SIS must be documented per IEC 61511." if method == 6 else
                    "PSV must be installed on pump discharge side of first isolation valve."))
    else:
        raise ValueError(f"Unknown design pressure method: {method}")

    P_design_barg = P_design_bara - Patm_bara

    return {
        "P_design_bara":       round(P_design_bara, 3),
        "P_design_barg":       round(P_design_barg, 3),
        "method":              method,
        "method_label":        label,
        "notes":               notes,
    }


def pd_pump_design_pressure(PSV_set_barg: float, accumulation_pct: float = 10.0,
                             Patm_bara: float = 1.01325) -> dict:
    """Design pressure for positive-displacement pump (PSV mandatory)."""
    acc = accumulation_pct / 100.0
    P_design_bara = (PSV_set_barg + Patm_bara) * (1.0 + acc)
    P_design_barg = P_design_bara - Patm_bara
    return {
        "P_design_bara": round(P_design_bara, 3),
        "P_design_barg": round(P_design_barg, 3),
        "notes": (f"PD pump — PSV mandatory. Set = {PSV_set_barg:.2f} barg, "
                  f"accumulation = {accumulation_pct:.0f} %."),
    }


# ============================================================================
# 9. ANSI B16.5 PRESSURE CLASS LOOKUP
# ============================================================================
from standards.piping import MATERIAL_GROUPS, ansi_class_lookup
