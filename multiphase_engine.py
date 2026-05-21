# multiphase_engine.py
import numpy as np
import CoolProp.CoolProp as CP
from fluids.two_phase import (
    Beggs_Brill, two_phase_dP, two_phase_dP_dz_gravitational,
    Taitel_Dukler_regime, Mandhane_Gregory_Aziz_regime,
)
from fluids.two_phase_voidage import liquid_gas_voidage

_g = 9.80665

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
MATERIAL_ROUGHNESS = {
    "SS316L":            1.5e-5,
    "Duplex SS 2205":    1.5e-5,
    "Carbon Steel":      4.6e-5,
    "Hastelloy C-276":   1.5e-5,
    "Titanium Gr. 2":    1.5e-5,
}

# Absolute roughness (m) for fluoropolymer pipe liners.
LINER_ROUGHNESS = {
    "PTFE":  5.0e-8,
    "FEP":   5.0e-8,
    "PFA":   5.0e-8,
    "PVDF":  1.5e-7,
}

# Equivalent-length (Le/D) factors for minor losses — Crane TP-410
FITTING_Le_over_D = {
    "90° Standard Elbow":              30,
    "90° Long Radius Elbow (1.5D)":    16,
    "45° Elbow":                       16,
    "180° Return Bend":                50,
    "Tee — Branch Flow":               60,
    "Tee — Run Through":               20,
    "Gate Valve — Fully Open":          8,
    "Globe Valve — Fully Open":       340,
    "Ball Valve — Fully Open":          3,
    "Butterfly Valve":                 45,
    "Swing Check Valve":              100,
    "Lift Check Valve":               600,
    "Concentric Reducer — Gradual (15°)":  5,
    "Concentric Reducer — Sudden":        26,
    "Eccentric Reducer — Gradual (15°)":   5,
    "Expansion — Gradual (15°)":          10,
    "Expansion — Sudden":                 30,
}


def _seg_le_fit(seg, D_eff):
    """Sum equivalent pipe length from all fittings. Handles old and new segment format."""
    fl = seg.get("fittings_list")
    if fl is not None:
        total = 0.0
        for fit in fl:
            t = fit.get("type", "")
            q = fit.get("qty", 0)
            if t in FITTING_Le_over_D and q > 0:
                total += FITTING_Le_over_D[t] * D_eff * q
        return total
    f = seg.get("fittings", "None")
    c = seg.get("fitting_count", 0)
    if f in FITTING_Le_over_D and c > 0:
        return FITTING_Le_over_D[f] * D_eff * c
    return 0.0


# ============================================================================
# 2. CALCULATION METHOD REGISTRIES
# ============================================================================
# Subset of fluids.two_phase correlations that work with our available inputs
# (rhog, mul, mug, sigma all present).  Method strings must match the keys in
# fluids.two_phase.two_phase_correlations exactly.
TWO_PHASE_CORRELATIONS = [
    "Beggs-Brill",
    "Friedel",
    "Lockhart_Martinelli",
    "Muller_Steinhagen_Heck",
    "Chisholm",
    "Kim_Mudawar",
]

# Void fraction models offered in the UI.
VOIDAGE_METHODS = [
    "Homogeneous",
    "Rouhani-1 (slip)",
]

# ============================================================================
# 3. GAS SPECIES DATABASE
# ============================================================================
# MW in kg/mol; coolprop_id used for CoolProp viscosity lookup; mu_ref is
# a temperature-independent fallback (Pa·s).
GAS_SPECIES = {
    "H₂":     {"MW": 2.016e-3,   "coolprop_id": "Hydrogen",       "mu_ref": 8.9e-6},
    "O₂":     {"MW": 31.998e-3,  "coolprop_id": "Oxygen",         "mu_ref": 20.4e-6},
    "N₂":     {"MW": 28.014e-3,  "coolprop_id": "Nitrogen",       "mu_ref": 17.8e-6},
    "CO₂":    {"MW": 44.010e-3,  "coolprop_id": "CarbonDioxide",  "mu_ref": 15.0e-6},
    "CH₄":    {"MW": 16.043e-3,  "coolprop_id": "Methane",        "mu_ref": 11.1e-6},
    "Ar":     {"MW": 39.948e-3,  "coolprop_id": "Argon",          "mu_ref": 22.6e-6},
    "He":     {"MW": 4.003e-3,   "coolprop_id": "Helium",         "mu_ref": 19.7e-6},
    "Air":    {"MW": 28.97e-3,   "coolprop_id": None,             "mu_ref": 18.5e-6},
    "Custom": {"MW": None,        "coolprop_id": None,             "mu_ref": None},
}

# ============================================================================
# 3. LIQUID PHASE DATABASE
# ============================================================================

LIQUID_PHASES = ["KOH 30 wt%", "KOH 15 wt%", "Water", "Methanol", "Ethanol", "Custom"]

# Aqueous liquids add H₂O vapour to the gas phase via Dalton's Law.
LIQUID_AQUEOUS = {
    "KOH 30 wt%": True,
    "KOH 15 wt%": True,
    "Water":       True,
    "Methanol":    False,
    "Ethanol":     False,
    "Custom":      False,
}

# ============================================================================
# 3A. KOH 30 wt% LIQUID PROPERTIES
# Temperature-dependent correlations. Valid range: 0–100 °C.
# Reference: Yaws' Chemical Properties Handbook; NIST aqueous KOH data.
# ============================================================================

def koh_density_kgm3(T_C):
    rho_ref_20C = 1295.0   # kg/m³ at 20 °C
    slope = -0.3375        # kg/m³ per °C
    return max(1100.0, rho_ref_20C + slope * (T_C - 20.0))


def koh_viscosity_pas(T_C):
    mu_ref_20C = 1.4e-3    # Pa·s at 20 °C
    T_ref_K    = 293.15
    E_a_over_R = 1200.0    # K
    mu = mu_ref_20C * np.exp(E_a_over_R * (1.0 / (T_C + 273.15) - 1.0 / T_ref_K))
    return max(2e-4, mu)


def koh_surface_tension_nm(T_C):
    sigma_ref_20C = 0.074  # N/m at 20 °C
    slope = -0.001125      # N/m per °C
    return max(0.040, sigma_ref_20C + slope * (T_C - 20.0))


# ============================================================================
# 3B. KOH 15 wt% LIQUID PROPERTIES
# Valid range: 0–100 °C.
# References: CRC Handbook of Chemistry and Physics; fitted to tabulated data.
# ============================================================================

def koh15_density_kgm3(T_C):
    rho_ref_20C = 1139.0   # kg/m³ at 20 °C (CRC Handbook)
    slope = -0.50          # kg/m³ per °C
    return max(1050.0, rho_ref_20C + slope * (T_C - 20.0))


def koh15_viscosity_pas(T_C):
    mu_ref_20C = 1.1e-3    # Pa·s at 20 °C
    T_ref_K    = 293.15
    E_a_over_R = 1540.0    # K (fitted to μ(80°C) ≈ 0.45 mPa·s)
    mu = mu_ref_20C * np.exp(E_a_over_R * (1.0 / (T_C + 273.15) - 1.0 / T_ref_K))
    return max(1.5e-4, mu)


def koh15_surface_tension_nm(T_C):
    sigma_ref_20C = 0.073  # N/m at 20 °C
    slope = -0.00014       # N/m per °C
    return max(0.040, sigma_ref_20C + slope * (T_C - 20.0))


# ============================================================================
# 3C. LIQUID PROPERTIES VIA COOLPROP AND GAS VISCOSITY HELPER
# ============================================================================

# CoolProp fluid ID and surface-tension fallback (N/m) for each supported liquid.
_COOLPROP_LIQUID = {
    "Water":    ("Water",    0.072),
    "Methanol": ("Methanol", 0.022),
    "Ethanol":  ("Ethanol",  0.022),
}


def _coolprop_liquid_properties(liquid_type, T_C, P_bara):
    """Density (kg/m³), dynamic viscosity (Pa·s), surface tension (N/m) via CoolProp."""
    fluid, sigma_fallback = _COOLPROP_LIQUID[liquid_type]
    T_K  = T_C + 273.15
    P_pa = P_bara * 1e5
    try:
        rho = CP.PropsSI('D', 'T', T_K, 'P', P_pa, fluid)
        mu  = CP.PropsSI('V', 'T', T_K, 'P', P_pa, fluid)
    except Exception:
        rho = 800.0
        mu  = 1e-3
    try:
        sigma = CP.PropsSI('surface_tension', 'T', T_K, 'Q', 0, fluid)
    except Exception:
        sigma = sigma_fallback
    return rho, mu, sigma


def _water_properties(T_C, P_bara):
    """Kept for back-compatibility; delegates to the generic helper."""
    return _coolprop_liquid_properties("Water", T_C, P_bara)


def _get_species_viscosity(species, T_K, P_pa):
    """Dynamic viscosity (Pa·s) for a pure gas species via CoolProp, fallback to mu_ref."""
    cid = GAS_SPECIES[species].get("coolprop_id")
    if cid:
        try:
            return CP.PropsSI('V', 'T', T_K, 'P', P_pa, cid)
        except Exception:
            pass
    return GAS_SPECIES[species]["mu_ref"]


# ============================================================================
# 4. THERMODYNAMIC CORE SOLVER
# ============================================================================

def calculate_two_phase_properties(
    P_bara, T_C,
    gas_flows_kgh,      # dict: {species_name: kg/h}  e.g. {"H₂": 8.0, "O₂": 2.0}
    liquid_type,        # str: one of LIQUID_PHASES
    q_lye_m3h,
    custom_gas=None,    # {"MW_gmol": float, "mu_upas": float}  — only for "Custom" species
    custom_liquid=None, # {"rho_kgm3": float, "mu_mpas": float, "sigma_mnm": float}
):
    """
    Generic two-phase thermodynamic solver.

    Returns a dict of physical properties ready for Beggs & Brill calculations.
    All existing dict keys are preserved for backward compatibility.
    """
    P_pa = P_bara * 1e5
    T_K  = T_C + 273.15

    # ── Liquid properties ────────────────────────────────────────────────────
    if liquid_type == "KOH 30 wt%":
        rho_l = koh_density_kgm3(T_C)
        mu_l  = koh_viscosity_pas(T_C)
        sigma = koh_surface_tension_nm(T_C)
    elif liquid_type == "KOH 15 wt%":
        rho_l = koh15_density_kgm3(T_C)
        mu_l  = koh15_viscosity_pas(T_C)
        sigma = koh15_surface_tension_nm(T_C)
    elif liquid_type in _COOLPROP_LIQUID:   # Water, Methanol, Ethanol
        rho_l, mu_l, sigma = _coolprop_liquid_properties(liquid_type, T_C, P_bara)
    else:  # Custom
        cl    = custom_liquid or {}
        rho_l = cl.get("rho_kgm3", 1000.0)
        mu_l  = cl.get("mu_mpas",  1.0) * 1e-3
        sigma = cl.get("sigma_mnm", 72.0) * 1e-3

    m_lye_kgh = q_lye_m3h * rho_l
    aqueous   = LIQUID_AQUEOUS.get(liquid_type, False)

    # ── Dry gas moles (mol/h = kg/h ÷ MW [kg/mol]) ──────────────────────────
    dry_moles = {}
    for sp, m_kgh in gas_flows_kgh.items():
        if sp == "Custom" and custom_gas:
            MW = custom_gas["MW_gmol"] * 1e-3
        else:
            MW = GAS_SPECIES[sp]["MW"]
        dry_moles[sp] = m_kgh / MW
    n_dry = sum(dry_moles.values())

    # ── Water vapour (Dalton's Law, aqueous liquids only) ────────────────────
    P_sat_H2O        = 0.0
    m_H2O_vapor_kgh  = 0.0
    n_H2O            = 0.0
    if aqueous:
        try:
            P_sat_H2O = CP.PropsSI('P', 'T', T_K, 'Q', 1, 'Water')
        except Exception:
            P_sat_H2O = 0.1 * P_pa
        if P_sat_H2O >= P_pa:
            P_sat_H2O = P_pa * 0.95
        y_H2O = P_sat_H2O / P_pa
        n_H2O = n_dry * y_H2O / (1.0 - y_H2O) if y_H2O < 1.0 else 0.0
        m_H2O_vapor_kgh = n_H2O * 18.015e-3  # mol/h × kg/mol = kg/h

    # ── Gas mixture composition ───────────────────────────────────────────────
    all_moles = dict(dry_moles)
    if n_H2O > 0:
        all_moles["H₂O (vapour)"] = n_H2O
    n_total = sum(all_moles.values())

    composition = {}
    for sp, n_mol_h in all_moles.items():
        if sp == "H₂O (vapour)":
            MW = 18.015e-3
        elif sp == "Custom" and custom_gas:
            MW = custom_gas["MW_gmol"] * 1e-3
        else:
            MW = GAS_SPECIES[sp]["MW"]
        composition[sp] = {
            "mol_h":    n_mol_h,
            "kg_h":     n_mol_h * MW,
            "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
        }

    m_gas_total_kgh = sum(v["kg_h"] for v in composition.values())
    MW_mix_kgmol    = m_gas_total_kgh / n_total if n_total > 0 else 2.016e-3

    # ── Gas density (ideal gas) ───────────────────────────────────────────────
    rho_g = (P_pa * MW_mix_kgmol) / (8.314 * T_K)

    # ── Gas mixture viscosity (mole-fraction weighted) ────────────────────────
    mu_g = 0.0
    _custom_mu = (custom_gas["mu_upas"] * 1e-6) if custom_gas else 1.2e-5
    for sp, data in composition.items():
        y = data["mol_frac"]
        if sp == "H₂O (vapour)":
            mu_i = 1.2e-5
        elif sp == "Custom":
            mu_i = _custom_mu
        else:
            mu_i = _get_species_viscosity(sp, T_K, P_pa)
        mu_g += y * mu_i
    mu_g = max(5e-6, mu_g)

    # ── Phase mass balance ────────────────────────────────────────────────────
    m_liquid_total_kgh = max(0.1, m_lye_kgh - m_H2O_vapor_kgh)
    m_total_kgs        = (m_gas_total_kgh + m_liquid_total_kgh) / 3600.0
    x_gas              = m_gas_total_kgh / (m_gas_total_kgh + m_liquid_total_kgh)

    # ── Void fraction (homogeneous model) ─────────────────────────────────────
    alpha = 0.0
    if x_gas > 0 and rho_g > 0 and rho_l > 0:
        alpha = (x_gas / rho_g) / (x_gas / rho_g + (1.0 - x_gas) / rho_l)

    return {
        # ── Core properties (consumed by Beggs & Brill and erosion check) ──
        "m_total_kgs":        m_total_kgs,
        "x_gas":              x_gas,
        "alpha":              alpha,
        "rho_l":              rho_l,
        "rho_g":              rho_g,
        "mu_l":               mu_l,
        "mu_g":               mu_g,
        "sigma":              sigma,
        # ── Water vapour bookkeeping ────────────────────────────────────────
        "m_vapor_h2o_kgh":    m_H2O_vapor_kgh,
        "P_sat_H2O_pa":       P_sat_H2O,
        # ── State ──────────────────────────────────────────────────────────
        "T_C":                T_C,
        "P_pa":               P_pa,
        # ── Composition & display quantities (new) ──────────────────────────
        "composition":        composition,        # {species: {mol_h, kg_h, mol_frac}}
        "MW_mix_gmol":        MW_mix_kgmol * 1000.0,
        "liquid_type":        liquid_type,
        "m_gas_total_kgh":    m_gas_total_kgh,
        "m_lye_kgh":          m_lye_kgh,
        "m_liquid_total_kgh": m_liquid_total_kgh,
    }


def validate_input_bounds(P_bara, T_C, gas_flows_kgh, liquid_type, q_lye_m3h):
    """Sanity checks on inputs. Returns (is_valid: bool, warnings: list[str])."""
    warnings = []
    total_gas = sum(gas_flows_kgh.values())

    if P_bara < 1.0 or P_bara > 100.0:
        warnings.append(f"⚠️ System pressure {P_bara:.1f} bara outside typical range [1–100 bara]")
    if T_C < 5.0 or T_C > 95.0:
        warnings.append(f"⚠️ Temperature {T_C:.1f}°C outside validated range [5–95°C]")
    if total_gas < 0.05:
        warnings.append(f"⚠️ Total gas flow {total_gas:.3f} kg/h is very low")
    if 0 < q_lye_m3h < 1e-4:
        warnings.append(f"⚠️ Liquid volume flow {q_lye_m3h:.4f} m³/h is extremely low — verify intent")
    if LIQUID_AQUEOUS.get(liquid_type, False):
        try:
            P_sat = CP.PropsSI('P', 'T', T_C + 273.15, 'Q', 1, 'Water')
            if P_sat > P_bara * 1e5:
                warnings.append("⚠️ Water saturation pressure exceeds system pressure; flashing likely")
        except Exception:
            pass
    return len(warnings) == 0, warnings


# ============================================================================
# 5. PRESSURE DROP SOLVER
# ============================================================================

def _classify_regime(m, x, rhol, rhog, mul, mug, sigma, alpha, D, roughness, angle_deg):
    """
    Automatic flow regime classification — method selected by pipe orientation.

    |θ| ≤ 15°  Horizontal / near-horizontal
        Taitel & Dukler (1976) primary + Mandhane, Gregory & Aziz (1974) secondary.
        Returns "<T-D regime> / <MGA regime>".

    |θ| ≥ 75°  Vertical
        Upflow   — Wallis/Taitel (1980) annular-onset criterion plus void-fraction
                   thresholds for bubble / slug / churn transitions.
        Downflow — Wallis annular criterion; otherwise falling film / slug.
        Gas-dominated (x > 0.90) → mist / annular.

    15° < |θ| < 75°  Inclined
        Taitel-Dukler called at θ = 0 (X-parameter is angle-independent);
        result labelled "(inclined)".  No validated library method exists here.
    """
    A   = 0.25 * np.pi * D ** 2
    Vsg = (x * m / rhog) / A         if rhog > 0 else 0.0
    abs_ang = abs(angle_deg)

    # ── Horizontal / near-horizontal ─────────────────────────────────────────
    if abs_ang <= 15.0:
        try:
            td  = Taitel_Dukler_regime(
                m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                D=D, angle=angle_deg, roughness=roughness)[0]
            mga = Mandhane_Gregory_Aziz_regime(
                m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                sigma=sigma, D=D)[0]
            return f"{td} / {mga}"
        except Exception:
            return "intermittent / slug"          # safe fallback

    # ── Vertical ─────────────────────────────────────────────────────────────
    elif abs_ang >= 75.0:
        try:
            V_ann = 3.1 * (_g * sigma * (rhol - rhog) / rhog ** 2) ** 0.25
        except Exception:
            V_ann = 1e9
        if angle_deg > 0:                         # upflow
            if x > 0.90:
                return "mist / annular"
            if Vsg >= V_ann:
                return "annular"
            if alpha >= 0.52:
                return "churn"
            if alpha >= 0.25:
                return "slug"
            return "bubble"
        else:                                      # downflow
            if x > 0.90:
                return "falling film"
            if Vsg >= V_ann:
                return "falling film / annular"
            return "falling film / slug"

    # ── Inclined ─────────────────────────────────────────────────────────────
    else:
        try:
            td = Taitel_Dukler_regime(
                m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                D=D, angle=0.0, roughness=roughness)[0]
            return f"{td} (inclined)"
        except Exception:
            return "intermittent (inclined)"

def calculate_segment_pressure_drop(
    props, D_inner, roughness, L_eff, angle_rad,
    correlation="Beggs-Brill",
    voidage_method="Homogeneous",
):
    """
    Pressure drop across one pipe segment with ΔP decomposition.

    Args:
        props:          dict from calculate_two_phase_properties()
        D_inner:        effective inner diameter (m), after liner if applicable
        roughness:      absolute roughness (m)
        L_eff:          effective length including minor losses (m)
        angle_rad:      inclination (0 = horizontal, +π/2 = vertical up)
        correlation:    one of TWO_PHASE_CORRELATIONS
        voidage_method: one of VOIDAGE_METHODS

    Returns:
        dict with keys: dP_Pa, dP_fric_Pa, dP_grav_Pa, dP_accel_Pa,
                        regime, dP_per_dz, Vsg, Vsl, alpha
    """
    _err_result = lambda msg: {
        "dP_Pa": 0.0, "dP_fric_Pa": 0.0, "dP_grav_Pa": 0.0, "dP_accel_Pa": 0.0,
        "regime": msg, "dP_per_dz": 0.0, "Vsg": 0.0, "Vsl": 0.0,
        "alpha": props.get("alpha", 0.0),
    }
    try:
        m     = props["m_total_kgs"]
        x     = props["x_gas"]
        rhol  = props["rho_l"]
        rhog  = props["rho_g"]
        mul   = props["mu_l"]
        mug   = props["mu_g"]
        sigma = props["sigma"]
        P_pa  = props["P_pa"]

        angle_deg = np.degrees(angle_rad)
        A_cs = 0.25 * np.pi * D_inner ** 2
        Vsg  = (x * m / rhog) / A_cs       if rhog > 0 else 0.0
        Vsl  = ((1.0 - x) * m / rhol) / A_cs if rhol > 0 else 0.0

        # ── Void fraction ─────────────────────────────────────────────────────
        if voidage_method == "Rouhani-1 (slip)" and 0 < x < 1:
            try:
                alpha = float(liquid_gas_voidage(
                    x=x, rhol=rhol, rhog=rhog,
                    D=D_inner, m=m, sigma=sigma,
                    Method="Rouhani 1",
                ))
                alpha = max(0.0, min(1.0, alpha))
            except Exception:
                alpha = props["alpha"]
        else:
            alpha = props["alpha"]

        # ── Gravitational component (can be negative for downflow) ────────────
        dP_grav_Pa = two_phase_dP_dz_gravitational(
            angle=angle_deg, alpha=alpha, rhol=rhol, rhog=rhog,
        ) * L_eff

        # ── Frictional component + total ─────────────────────────────────────
        if correlation == "Beggs-Brill":
            # Use direct Beggs_Brill for total (gravity already included).
            dP_total = Beggs_Brill(
                m=m, x=x, rhol=rhol, rhog=rhog,
                mul=mul, mug=mug, sigma=sigma, P=P_pa,
                D=D_inner, angle=angle_deg, roughness=roughness, L=L_eff,
            )
            # Friction-only: same correlation at horizontal (angle=0).
            dP_fric_Pa = Beggs_Brill(
                m=m, x=x, rhol=rhol, rhog=rhog,
                mul=mul, mug=mug, sigma=sigma, P=P_pa,
                D=D_inner, angle=0.0, roughness=roughness, L=L_eff,
            )
            # Acceleration is the residual (includes B&B inclination correction).
            dP_accel_Pa = dP_total - dP_fric_Pa - dP_grav_Pa
        else:
            # Other correlations: two_phase_dP is friction-only by design.
            dP_fric_Pa = two_phase_dP(
                m=m, x=x, rhol=rhol, rhog=rhog,
                mul=mul, mug=mug, sigma=sigma,
                D=D_inner, L=L_eff, roughness=roughness,
                Method=correlation,
            )
            dP_total    = dP_fric_Pa + dP_grav_Pa
            dP_accel_Pa = 0.0  # negligible for subsonic adiabatic flow

        dP_per_dz = dP_total / L_eff if L_eff > 0 else 0.0

        # ── Flow regime — automatic method selection by orientation ──────────
        regime = _classify_regime(
            m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
            sigma=sigma, alpha=alpha, D=D_inner, roughness=roughness,
            angle_deg=angle_deg,
        )

        return {
            "dP_Pa":       dP_total,
            "dP_fric_Pa":  dP_fric_Pa,
            "dP_grav_Pa":  dP_grav_Pa,
            "dP_accel_Pa": dP_accel_Pa,
            "regime":      regime,
            "dP_per_dz":   dP_per_dz,
            "Vsg":         Vsg,
            "Vsl":         Vsl,
            "alpha":       alpha,
        }

    except Exception as e:
        import warnings as _w
        _w.warn(f"Pressure drop error ({correlation}): {str(e)[:80]}")
        return _err_result(f"Error ({correlation[:12]}): {str(e)[:30]}")


# ============================================================================
# 5A. SENSITIVITY ANALYSIS — all correlation × void-fraction combinations
# ============================================================================

def run_sensitivity(
    P_bara, T_C, gas_flows_kgh, liquid_type, q_lye_m3h, segments,
    custom_gas=None, custom_liquid=None,
):
    """
    Run all 12 combinations (6 correlations × 2 void-fraction models) and
    return the total ΔP for each using pressure marching.

    Returns list of dicts — one per combination, ordered as in TWO_PHASE_CORRELATIONS
    (outer) × VOIDAGE_METHODS (inner):
        label        str   e.g. "Beggs-Brill / Homogeneous"
        correlation  str   key in TWO_PHASE_CORRELATIONS
        voidage      str   key in VOIDAGE_METHODS
        total_dp_kpa float   None if convergence failed
        ok           bool
        error        str | None
    """
    results = []
    for corr in TWO_PHASE_CORRELATIONS:
        for void in VOIDAGE_METHODS:
            try:
                current_P   = P_bara * 1e5
                total_dp    = 0.0
                seg_regimes = []
                for seg in segments:
                    D_seg     = PIPE_DATABASE[seg["dn"]][seg["pn"]]
                    lined     = seg.get("lined", False)
                    lthk_m    = seg.get("liner_thickness_mm", 1.0) / 1000.0
                    lmat      = seg.get("liner_material", "FEP")
                    D_eff     = D_seg - 2 * lthk_m if lined else D_seg
                    roughness = (LINER_ROUGHNESS[lmat] if lined
                                 else MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
                    props_seg = calculate_two_phase_properties(
                        current_P / 1e5, T_C, gas_flows_kgh, liquid_type, q_lye_m3h,
                        custom_gas=custom_gas, custom_liquid=custom_liquid)
                    angle  = {"Horizontal":        0.0,
                              "Vertical Upflow":   np.pi / 2.0,
                              "Vertical Downflow": -np.pi / 2.0}[seg["type"]]
                    le_fit = _seg_le_fit(seg, D_eff)
                    L_eff  = seg["length"] + le_fit
                    res    = calculate_segment_pressure_drop(
                        props_seg, D_eff, roughness, L_eff, angle,
                        correlation=corr, voidage_method=void)
                    total_dp  += res["dP_Pa"]
                    current_P -= res["dP_Pa"]
                    seg_regimes.append(res["regime"])
                results.append({
                    "label":           f"{corr} / {void}",
                    "correlation":     corr,
                    "voidage":         void,
                    "total_dp_kpa":    total_dp / 1000.0,
                    "segment_regimes": seg_regimes,
                    "ok":              True,
                    "error":           None,
                })
            except Exception as exc:
                results.append({
                    "label":           f"{corr} / {void}",
                    "correlation":     corr,
                    "voidage":         void,
                    "total_dp_kpa":    None,
                    "segment_regimes": [],
                    "ok":              False,
                    "error":           str(exc)[:80],
                })
    return results


# ============================================================================
# 6. EROSION VELOCITY CHECK  (API RP 14E)
# ============================================================================

def calculate_erosion_velocity(rho_g, rho_l, x_gas, C=100):
    """
    API RP 14E erosion velocity for two-phase flow.

    V_e = C_SI / sqrt(rho_mix)   [m/s]

    C = 100 → continuous service (conservative, recommended default)
    C = 125 → intermittent service

    SI conversion: C_SI = C × 0.3048 × sqrt(16.018)  ≈  C × 1.2197
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

    C_SI = C * 0.3048 * (16.018 ** 0.5)
    V_erosion = C_SI / (rho_mix ** 0.5)
    return V_erosion, rho_mix
