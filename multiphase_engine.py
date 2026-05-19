# multiphase_engine.py
import numpy as np
import CoolProp.CoolProp as CP
from fluids.two_phase import Beggs_Brill

# ============================================================================
# 1. INDUSTRIAL STANDARDS DATABASE
# ============================================================================
# Inner diameters (m) based on ANSI B36.10 / B36.19 schedule pipe:
#   PN20 / PN25  ≈  Schedule 40   (same bore for all metallic alloys at a given schedule)
#   PN40         ≈  Schedule 80
# Material does NOT affect bore at the same schedule — wall thickness is defined
# by schedule number, not alloy.  All five metallic materials below use this table.
PIPE_DATABASE = {
    "DN40":  {"PN20": 0.0409, "PN25": 0.0409, "PN40": 0.0381},
    "DN50":  {"PN20": 0.0525, "PN25": 0.0525, "PN40": 0.0493},
    "DN80":  {"PN20": 0.0779, "PN25": 0.0779, "PN40": 0.0737},
    "DN100": {"PN20": 0.1023, "PN25": 0.1023, "PN40": 0.0972},
    "DN150": {"PN20": 0.1541, "PN25": 0.1541, "PN40": 0.1463},
    "DN200": {"PN20": 0.2027, "PN25": 0.2027, "PN40": 0.1937},
    "DN250": {"PN20": 0.2545, "PN25": 0.2545, "PN40": 0.2429},
}

# Absolute roughness (m) by pipe material — Crane TP-410 / ASHRAE
# All entries are metallic schedule pipe; polymer materials (HDPE, GRP) are
# excluded because they use SDR-based sizing incompatible with the PN lookup above.
MATERIAL_ROUGHNESS = {
    "SS316L":            1.5e-5,   # electropolished stainless steel
    "Duplex SS 2205":    1.5e-5,   # drawn schedule pipe, similar to SS316L
    "Carbon Steel":      4.6e-5,   # commercial steel (Crane TP-410)
    "Hastelloy C-276":   1.5e-5,   # nickel alloy, drawn tube
    "Titanium Gr. 2":    1.5e-5,   # drawn tube, similar surface finish
}

# Absolute roughness (m) for fluoropolymer pipe liners.
# Lined bore = metal bore − 2 × liner thickness; liner dominates roughness.
# Values from manufacturer data for chemically bonded / compression-fitted liners.
LINER_ROUGHNESS = {
    "PTFE":  5.0e-8,   # polytetrafluoroethylene — ultra-smooth, ~0.05 µm
    "FEP":   5.0e-8,   # fluorinated ethylene propylene — comparable to PTFE
    "PFA":   5.0e-8,   # perfluoroalkoxy — similar to FEP/PTFE
    "PVDF":  1.5e-7,   # polyvinylidene fluoride — slightly rougher, ~0.15 µm
}

# Equivalent-length (Le/D) factors for minor losses — Crane TP-410
# For reducers/expansions: Le/D quoted at the smaller-bore pipe; β ≈ 0.75 assumed.
FITTING_Le_over_D = {
    # ── Elbows ──────────────────────────────────────────────────────────────
    "90° Standard Elbow":              30,
    "90° Long Radius Elbow (1.5D)":    16,
    "45° Elbow":                       16,
    "180° Return Bend":                50,
    # ── Tees ────────────────────────────────────────────────────────────────
    "Tee — Branch Flow":               60,
    "Tee — Run Through":               20,
    # ── Isolating Valves ────────────────────────────────────────────────────
    "Gate Valve — Fully Open":          8,
    "Globe Valve — Fully Open":       340,
    "Ball Valve — Fully Open":          3,
    "Butterfly Valve":                 45,
    # ── Check Valves ────────────────────────────────────────────────────────
    "Swing Check Valve":              100,
    "Lift Check Valve":               600,
    # ── Reducers / Expansions (Crane TP-410, β ≈ 0.75) ─────────────────────
    "Concentric Reducer — Gradual (15°)":  5,
    "Concentric Reducer — Sudden":        26,
    "Eccentric Reducer — Gradual (15°)":   5,
    "Expansion — Gradual (15°)":          10,
    "Expansion — Sudden":                 30,
}

# ============================================================================
# 2. KOH LIQUID PROPERTIES (30 wt% Aqueous Solution)
# Temperature-dependent correlations for density and viscosity
# ============================================================================

def koh_density_kgm3(T_C):
    """
    Density of 30 wt% aqueous KOH solution (kg/m³) as function of temperature (°C).
    Linear approximation based on standard thermodynamic data.
    Valid range: 0–100°C
    Reference: Yaws' Chemical Properties Handbook
    """
    # Reference: at 20°C, ρ ≈ 1.295 kg/m³; at 80°C, ρ ≈ 1.268 kg/m³
    rho_ref_20C = 1295.0  # kg/m³
    slope = -0.3375  # kg/m³ per °C (negative slope as T increases)
    rho_l = rho_ref_20C + slope * (T_C - 20.0)
    return max(1100.0, rho_l)  # Clamp to physical lower bound


def koh_viscosity_pas(T_C):
    """
    Dynamic viscosity of 30 wt% aqueous KOH solution (Pa·s) as function of temperature (°C).
    Arrhenius-type correlation: μ(T) = μ_ref * exp(E_a / R * (1/T - 1/T_ref))
    Valid range: 0–100°C
    Reference: NIST Database for aqueous KOH
    """
    # Reference: at 20°C, μ ≈ 1.4e-3 Pa·s; at 80°C, μ ≈ 5.0e-4 Pa·s
    mu_ref_20C = 1.4e-3  # Pa·s at 20°C
    T_ref_K = 293.15  # 20°C in Kelvin
    T_K = T_C + 273.15
    
    # Effective activation energy parameter (fitted to KOH data)
    E_a_over_R = 1200.0  # K
    
    mu_l = mu_ref_20C * np.exp(E_a_over_R * (1.0 / T_K - 1.0 / T_ref_K))
    return max(2e-4, mu_l)  # Clamp to avoid unphysical values


def koh_surface_tension_nm(T_C):
    """
    Surface tension of 30 wt% aqueous KOH solution (N/m) as function of temperature (°C).
    Simplified linear approximation.
    Valid range: 0–100°C
    """
    # Reference: at 20°C, σ ≈ 0.074 N/m; at 80°C, σ ≈ 0.065 N/m
    sigma_ref_20C = 0.074  # N/m at 20°C
    slope = -0.001125  # N/m per °C
    sigma = sigma_ref_20C + slope * (T_C - 20.0)
    return max(0.040, sigma)  # Clamp to physical minimum


# ============================================================================
# 3. THERMODYNAMIC CORE SOLVER
# ============================================================================

def calculate_two_phase_properties(P_bara, T_C, m_H2_kgh, m_O2_kgh, q_lye_m3h):
    """
    Solves the localized gas mixture densities, H2O vapor pressure limits,
    and temperature-dependent KOH liquid property adjustments.
    
    Returns:
        dict: Thermodynamic and hydrodynamic properties for flow calculations
    """
    P_pa = P_bara * 100000.0
    T_K = T_C + 273.15
    
    # A. Liquid Properties (30 wt% Aqueous KOH, Temperature-Dependent)
    rho_l = koh_density_kgm3(T_C)
    mu_l = koh_viscosity_pas(T_C)
    sigma = koh_surface_tension_nm(T_C)
    
    m_lye_kgh = q_lye_m3h * rho_l
    
    # B. Vapor Phase Saturation Scaling via Dalton's Law
    try:
        P_sat_H2O = CP.PropsSI('P', 'T', T_K, 'Q', 1, 'Water')
    except Exception:
        P_sat_H2O = 0.1 * P_pa  # Fallback if CoolProp fails
    
    # Safety clamp: saturation pressure cannot exceed system pressure
    if P_sat_H2O >= P_pa:
        P_sat_H2O = P_pa * 0.95
    
    # Molar balance for gas mixture
    n_H2 = (m_H2_kgh * 1000.0) / 2.016
    n_O2 = (m_O2_kgh * 1000.0) / 31.998
    n_dry = n_H2 + n_O2
    
    # Partial pressure of water vapor (Dalton's Law)
    y_H2O = P_sat_H2O / P_pa
    n_H2O = n_dry * y_H2O / (1.0 - y_H2O) if y_H2O < 1.0 else 0.0
    
    m_H2O_vapor_kgh = (n_H2O * 18.015) / 1000.0
    
    # Total phase masses
    m_gas_total_kgh = m_H2_kgh + m_O2_kgh + m_H2O_vapor_kgh
    m_liquid_total_kgh = max(0.1, m_lye_kgh - m_H2O_vapor_kgh)
    
    m_total_kgs = (m_gas_total_kgh + m_liquid_total_kgh) / 3600.0
    x_gas = m_gas_total_kgh / (m_gas_total_kgh + m_liquid_total_kgh)
    
    # Gas Mixture Density (Ideal Gas Law)
    n_total = n_dry + n_H2O
    if n_total > 0:
        MW_mix = ((n_H2 * 2.016) + (n_O2 * 31.998) + (n_H2O * 18.015)) / n_total / 1000.0
    else:
        MW_mix = 0.002  # Fallback
    
    R_universal = 8.314
    rho_g = (P_pa * MW_mix) / (R_universal * T_K)
    mu_g = 1.2e-5  # Pa·s (weakly temperature-dependent, constant approximation)

    # Homogeneous void fraction α = (x/ρg) / (x/ρg + (1-x)/ρl)
    alpha = 0.0
    if x_gas > 0 and rho_g > 0 and rho_l > 0:
        alpha = (x_gas / rho_g) / (x_gas / rho_g + (1.0 - x_gas) / rho_l)

    return {
        "m_total_kgs": m_total_kgs,
        "x_gas": x_gas,
        "alpha": alpha,
        "rho_l": rho_l,
        "rho_g": rho_g,
        "mu_l": mu_l,
        "mu_g": mu_g,
        "sigma": sigma,
        "m_vapor_h2o_kgh": m_H2O_vapor_kgh,
        "P_sat_H2O_pa": P_sat_H2O,
        "T_C": T_C,
        "P_pa": P_pa
    }


def calculate_erosion_velocity(rho_g, rho_l, x_gas, C=100):
    """
    API RP 14E erosion velocity for two-phase flow.

    V_e = C_SI / sqrt(rho_mix)   [m/s]

    C = 100 → continuous service (conservative, recommended default)
    C = 125 → intermittent service

    SI conversion of the original US-customary formula:
        C_SI = C * 0.3048 * sqrt(16.018)  ≈  C * 1.2197
    so C=100 → C_SI ≈ 122 m/s·(kg/m³)^0.5

    rho_mix is the no-slip (homogeneous) mixture density:
        rho_mix = 1 / (x/rho_g + (1-x)/rho_l)

    Returns:
        tuple: (V_erosion m/s, rho_mix kg/m³)
    """
    if x_gas <= 0.0:
        rho_mix = rho_l
    elif x_gas >= 1.0:
        rho_mix = rho_g
    else:
        rho_mix = 1.0 / (x_gas / rho_g + (1.0 - x_gas) / rho_l)

    C_SI = C * 0.3048 * (16.018 ** 0.5)   # ≈ 121.97 for C=100
    V_erosion = C_SI / (rho_mix ** 0.5)
    return V_erosion, rho_mix


def validate_input_bounds(P_bara, T_C, m_H2_kgh, m_O2_kgh, q_lye_m3h):
    """
    Performs sanity checks on input parameters.
    
    Returns:
        tuple: (is_valid: bool, warnings: list of str)
    """
    warnings = []
    
    if P_bara < 1.0 or P_bara > 100.0:
        warnings.append(f"⚠️ System pressure {P_bara:.1f} bara outside typical range [1–100 bara]")
    
    if T_C < 5.0 or T_C > 95.0:
        warnings.append(f"⚠️ Temperature {T_C:.1f}°C outside validated range [5–95°C]")
    
    if m_H2_kgh < 0.1:
        warnings.append(f"⚠️ Hydrogen flow {m_H2_kgh:.2f} kg/h is very low; consider minimum 0.5 kg/h")
    
    if 0 < m_O2_kgh < 0.1:
        warnings.append(f"⚠️ Oxygen flow {m_O2_kgh:.2f} kg/h is very low; consider minimum 0.5 kg/h")
    
    if q_lye_m3h < 0.1:
        warnings.append(f"⚠️ Lye volume {q_lye_m3h:.2f} m³/h is very low; consider minimum 0.2 m³/h")
    
    # Check if vapor saturation is feasible
    T_K = T_C + 273.15
    try:
        P_sat_H2O = CP.PropsSI('P', 'T', T_K, 'Q', 1, 'Water')
        if P_sat_H2O > P_bara * 100000.0:
            warnings.append(f"⚠️ Water saturation pressure exceeds system pressure; significant flashing will occur")
    except Exception:
        pass
    
    return len(warnings) == 0, warnings


def calculate_segment_pressure_drop(props, D_inner, roughness, L_eff, angle_rad):
    """
    Calculates pressure drop across a single pipe segment using Beggs & Brill correlation.
    
    Args:
        props: Dictionary from calculate_two_phase_properties()
        D_inner: Inner pipe diameter (m)
        roughness: Absolute roughness (m)
        L_eff: Effective length including fittings (m)
        angle_rad: Pipe inclination angle in radians (0=horizontal, π/2=vertical up)
    
    Returns:
        tuple: (dP_Pa: total pressure drop, flow_regime: str, dP_per_dz: Pa/m, Vsg: m/s, Vsl: m/s)
    """
    try:
        dP_Pa = Beggs_Brill(
            m=props["m_total_kgs"],
            x=props["x_gas"],
            rhol=props["rho_l"],
            rhog=props["rho_g"],
            mul=props["mu_l"],
            mug=props["mu_g"],
            sigma=props["sigma"],
            P=props["P_pa"],
            D=D_inner,
            angle=np.degrees(angle_rad),  # fluids expects degrees, not radians
            roughness=roughness,
            L=L_eff
        )

        dP_per_dz = dP_Pa / L_eff if L_eff > 0 else 0.0

        A_cs = 0.25 * np.pi * D_inner ** 2
        Vsg = ((props["x_gas"] * props["m_total_kgs"]) / props["rho_g"]) / A_cs if props["rho_g"] > 0 else 0.0
        Vsl = (((1.0 - props["x_gas"]) * props["m_total_kgs"]) / props["rho_l"]) / A_cs if props["rho_l"] > 0 else 0.0

        if abs(angle_rad) < 1e-9:  # Horizontal
            if Vsg < 1.0:
                regime = "Stratified"
            elif Vsg < 5.0:
                regime = "Intermittent (Slug)"
            else:
                regime = "Annular/Dispersed"
        else:  # Vertical
            if angle_rad > 0:  # Upflow
                regime = "Bubble/Slug" if Vsg < 3.0 else "Churn/Annular"
            else:  # Downflow
                regime = "Falling Film" if Vsl > 2.0 else "Annular"

        return dP_Pa, regime, dP_per_dz, Vsg, Vsl

    except Exception as e:
        error_msg = f"Calculation Error: {str(e)[:50]}"
        import warnings as _w
        _w.warn(error_msg)
        return 0.0, error_msg, 0.0, 0.0, 0.0
