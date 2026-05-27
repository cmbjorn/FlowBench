"""
Control valve sizing — IEC 60534-2-1 (equivalent to ISA 75.01.01).

Supported services:
  Liquid   : choked-flow / cavitation corrected Kv
  Gas/Vapour: compressible flow with expansion factor Y, choke detection
  Steam    : gas path with CoolProp-derived γ and density

Reuses fanno_engine for gas species data and the universal gas constant.
"""

import math
import fanno_engine as _fan

R_u   = _fan.R_u
GASES = _fan.GASES

# IEC 60534-2-1 numerical constant for gas mass flow equation
# W [kg/h] = N6 × FP × Kv × Y × √(x × P1_bar × ρ1_kgm3)
_N6 = 31.6

# Kv ↔ Cv conversion (Kv: m³/h at 1 bar ΔP; Cv: US gal/min at 1 psi ΔP)
KV_TO_CV = 1.0 / 0.865
CV_TO_KV = 0.865

# Standard conditions for Nm³ (volumetric gas flow)
_T_STD_K  = 273.15    # 0 °C
_P_STD_Pa = 101_325.0 # 1 atm

# ── Valve types ─────────────────────────────────────────────────────────────
# FL  : liquid pressure recovery factor      (higher = lower recovery = less prone to choke)
# xT  : pressure drop ratio at choked flow  (higher = higher pressure at choke)
# Fd  : valve style modifier (for noise)
# Source: IEC 60534-2-1:2011 Annex A (indicative — use vendor data when available)
VALVE_TYPES: dict[str, dict] = {
    "Globe (full trim)":      {"FL": 0.90, "xT": 0.72, "Fd": 1.00},
    "Globe (reduced trim)":   {"FL": 0.85, "xT": 0.65, "Fd": 0.85},
    "Butterfly (60° open)":   {"FL": 0.68, "xT": 0.40, "Fd": 0.70},
    "Butterfly (90° open)":   {"FL": 0.55, "xT": 0.30, "Fd": 0.42},
    "Ball (standard trim)":   {"FL": 0.75, "xT": 0.55, "Fd": 0.98},
    "Rotary plug":            {"FL": 0.85, "xT": 0.60, "Fd": 0.50},
    "Custom":                 {"FL": None, "xT": None, "Fd": None},
}

# Indicative Kv_100 (max Kv at fully open) by valve body size.
# These are representative of typical globe-valve trims; ball / butterfly
# may be substantially higher. Use for body-size guidance only.
INDICATIVE_KV: dict[str, float] = {
    'DN 15  (NPS 1/2")':  1.2,
    'DN 20  (NPS 3/4")':  3.0,
    'DN 25  (NPS 1")':    5.0,
    'DN 32  (NPS 1 1/4")': 8.0,
    'DN 40  (NPS 1 1/2")': 16.0,
    'DN 50  (NPS 2")':    25.0,
    'DN 65  (NPS 2 1/2")': 40.0,
    'DN 80  (NPS 3")':    63.0,
    'DN 100 (NPS 4")':   100.0,
    'DN 150 (NPS 6")':   250.0,
    'DN 200 (NPS 8")':   400.0,
    'DN 250 (NPS 10")':  630.0,
    'DN 300 (NPS 12")': 1000.0,
}

# Thermodynamic critical pressure fallbacks (Pa) for CoolProp fluids
_PC_FALLBACK: dict[str, float] = {
    "Water":          220.6e5,
    "Methanol":        81.0e5,
    "Ethanol":         63.8e5,
    "Propane":         42.5e5,
    "n-Butane":        38.0e5,
    "Ammonia":        113.5e5,
    "R134a":           40.6e5,
    "Benzene":         48.9e5,
    "Toluene":         41.1e5,
    "n-Pentane":       33.7e5,
    "n-Hexane":        30.3e5,
    "n-Heptane":       27.4e5,
    "CarbonDioxide":   73.8e5,
    "Acetone":         47.1e5,
    "CycloHexane":     40.7e5,
}


# ── Property helpers ─────────────────────────────────────────────────────────

def critical_pressure(coolprop_id: str) -> float:
    """Thermodynamic critical pressure (Pa). Tries CoolProp, falls back to table."""
    try:
        import CoolProp.CoolProp as CP
        return float(CP.PropsSI("Pcrit", coolprop_id))
    except Exception:
        return _PC_FALLBACK.get(coolprop_id, 0.0)


def nm3h_to_kgh(Q_nm3h: float, MW_kgmol: float, Z_std: float = 1.0) -> float:
    """Convert normal volumetric flow (Nm³/h at 0 °C, 1 atm) to mass flow (kg/h)."""
    rho_std = _P_STD_Pa * MW_kgmol / (R_u * _T_STD_K * 1000.0 * Z_std)
    return Q_nm3h * rho_std


# ── Liquid sizing ────────────────────────────────────────────────────────────

def _ff_liquid(Pv_Pa: float, Pc_Pa: float) -> float:
    """Liquid critical pressure ratio factor Ff = 0.96 − 0.28 √(Pv/Pc)."""
    if Pc_Pa <= 0:
        return 0.96
    return 0.96 - 0.28 * math.sqrt(max(0.0, Pv_Pa / Pc_Pa))


def cv_liquid_size(
    Q_m3h: float,
    P1_Pa: float,
    P2_Pa: float,
    rho_kgm3: float,
    Pv_Pa: float,
    Pc_Pa: float,
    FL: float,
    FP: float = 1.0,
) -> dict:
    """
    IEC 60534-2-1 liquid control valve sizing.

    Parameters
    ----------
    Q_m3h    : volumetric flow, m³/h
    P1_Pa    : upstream pressure, Pa
    P2_Pa    : downstream pressure, Pa
    rho_kgm3 : liquid density, kg/m³
    Pv_Pa    : vapour pressure at T1, Pa  (0 if unknown)
    Pc_Pa    : critical pressure, Pa       (0 if unknown)
    FL       : liquid pressure recovery factor (valve-specific)
    FP       : piping geometry factor (1.0 for no reducers)

    Returns
    -------
    dict — Kv_req, Cv_req, choked, cavitating, Ff, FL, sigma,
            dP_bar, dP_choked_bar, dP_eff_bar
    """
    dP_Pa = max(P1_Pa - P2_Pa, 0.0)
    Ff     = _ff_liquid(Pv_Pa, Pc_Pa)
    FL_eff = FL * FP
    dP_chok = FL_eff**2 * (P1_Pa - Ff * Pv_Pa) if Pv_Pa > 0 else math.inf
    choked   = dP_Pa >= dP_chok
    dP_eff   = min(dP_Pa, dP_chok)

    rho_rel = rho_kgm3 / 1000.0
    Kv = Q_m3h / (FP * math.sqrt(max(dP_eff, 1.0) / 1e5 / rho_rel))
    Cv = Kv * KV_TO_CV

    sigma = (P1_Pa - Pv_Pa) / dP_Pa if (dP_Pa > 0 and Pv_Pa > 0) else math.inf
    cavitating = sigma < 1.0 / FL**2 if FL > 0 else False

    return {
        "Kv_req":        Kv,
        "Cv_req":        Cv,
        "choked":        choked,
        "cavitating":    cavitating,
        "Ff":            Ff,
        "FL":            FL,
        "FP":            FP,
        "sigma":         sigma,
        "dP_bar":        dP_Pa / 1e5,
        "dP_choked_bar": dP_chok / 1e5 if dP_chok < math.inf else math.inf,
        "dP_eff_bar":    dP_eff / 1e5,
    }


# ── Gas / vapour sizing ───────────────────────────────────────────────────────

def cv_gas_size(
    W_kgh: float,
    P1_Pa: float,
    P2_Pa: float,
    T1_K: float,
    MW_kgmol: float,
    gamma: float,
    Z: float,
    xT: float,
    FP: float = 1.0,
) -> dict:
    """
    IEC 60534-2-1 gas / vapour control valve sizing.

    Parameters
    ----------
    W_kgh    : mass flow, kg/h
    P1_Pa    : upstream pressure, Pa
    P2_Pa    : downstream pressure, Pa
    T1_K     : upstream temperature, K
    MW_kgmol : molecular weight, kg/kmol
    gamma    : ratio of specific heats Cp/Cv
    Z        : compressibility factor (1.0 ideal)
    xT       : pressure drop ratio at choked flow (valve-specific)
    FP       : piping geometry factor (1.0 for no reducers)

    Returns
    -------
    dict — Kv_req, Cv_req, choked, x, x_choked, Y, Fgamma, rho1_kgm3,
            dP_bar, P1_bar, Q_nm3h
    """
    P1_bar = P1_Pa / 1e5
    dP_bar = (P1_Pa - P2_Pa) / 1e5
    x      = min(dP_bar / P1_bar, 1.0) if P1_bar > 0 else 0.0

    Fgamma  = gamma / 1.4
    x_choked = Fgamma * xT
    choked   = x >= x_choked
    x_eff    = min(x, x_choked)

    # Y = 1 − x/(3·Fγ·xT), clamped to 0.667 at choke
    Y = max(0.667, 1.0 - x_eff / (3.0 * Fgamma * xT)) if xT > 0 else 0.667

    R_spec = R_u / MW_kgmol * 1000.0        # J/(kg·K)
    rho1   = P1_Pa / (R_spec * T1_K * Z)    # kg/m³ at inlet

    denom = _N6 * FP * Y * math.sqrt(max(x_eff * P1_bar * rho1, 1e-12))
    Kv    = W_kgh / denom if denom > 0 else math.inf
    Cv    = Kv * KV_TO_CV

    # Normal volumetric flow (Nm³/h at 0 °C, 1 atm)
    rho_std = _P_STD_Pa * MW_kgmol / (R_u * _T_STD_K * 1000.0)
    Q_nm3h  = W_kgh / rho_std

    return {
        "Kv_req":    Kv,
        "Cv_req":    Cv,
        "choked":    choked,
        "x":         x,
        "x_choked":  x_choked,
        "x_eff":     x_eff,
        "Y":         Y,
        "Fgamma":    Fgamma,
        "rho1_kgm3": rho1,
        "dP_bar":    dP_bar,
        "P1_bar":    P1_bar,
        "Q_nm3h":    Q_nm3h,
        "FP":        FP,
    }


# ── Steam sizing ─────────────────────────────────────────────────────────────

def cv_steam_size(
    W_kgh: float,
    P1_Pa: float,
    P2_Pa: float,
    T1_K: float,
    xT: float,
    FP: float = 1.0,
) -> dict:
    """
    Control valve sizing for steam — uses CoolProp for γ and ρ₁.
    Falls back to γ = 1.33, ideal-gas density if CoolProp unavailable or
    fluid is subcooled (warns via 'notes' key).
    """
    gamma = 1.33
    Z     = 1.0
    MW    = 18.015  # kg/kmol
    notes = []

    try:
        import CoolProp.CoolProp as CP
        T_sat = CP.PropsSI("T", "P", P1_Pa, "Q", 1.0, "Water")
        if T1_K < T_sat - 0.5:
            notes.append(
                f"UNPHYSICAL: T₁ = {T1_K-273.15:.1f} °C is below saturation "
                f"({T_sat-273.15:.1f} °C) at this pressure. Use Liquid service instead."
            )
        else:
            Cp  = CP.PropsSI("C", "T", T1_K, "P", P1_Pa, "Water")
            Cv_ = CP.PropsSI("O", "T", T1_K, "P", P1_Pa, "Water")
            if Cv_ > 0:
                gamma = Cp / Cv_
            rho1 = CP.PropsSI("D", "T", T1_K, "P", P1_Pa, "Water")
            Z = P1_Pa / (rho1 * R_u / MW * 1000.0 * T1_K)
            notes.append(f"CoolProp: γ = {gamma:.4f}, ρ₁ = {rho1:.3f} kg/m³")
    except Exception:
        notes.append("CoolProp unavailable — using γ = 1.33, ideal-gas density.")

    result = cv_gas_size(W_kgh, P1_Pa, P2_Pa, T1_K, MW, gamma, Z, xT, FP)
    result["gamma"]  = gamma
    result["notes"]  = notes
    return result


# ── Valve body size recommendation ───────────────────────────────────────────

def suggest_valve_size(Kv_req: float, target_opening: float = 0.80) -> dict:
    """
    Suggest the smallest standard valve body where Kv_100 × target_opening ≥ Kv_req.

    Parameters
    ----------
    Kv_req        : required Kv at design flow
    target_opening: fraction of Kv_100 at design flow (default 0.80 = 80 %)

    Returns
    -------
    dict — size_label, Kv_100, Kv_min (Kv_req/target_opening), opening_pct
    """
    Kv_min = Kv_req / target_opening
    for label, Kv_100 in INDICATIVE_KV.items():
        if Kv_100 >= Kv_min:
            return {
                "size_label":   label,
                "Kv_100":       Kv_100,
                "Kv_min":       Kv_min,
                "opening_pct":  Kv_req / Kv_100 * 100.0,
                "oversized":    False,
            }
    # Larger than table
    last_label = list(INDICATIVE_KV.keys())[-1]
    last_kv    = list(INDICATIVE_KV.values())[-1]
    return {
        "size_label":  last_label + " (too small)",
        "Kv_100":      last_kv,
        "Kv_min":      Kv_min,
        "opening_pct": Kv_req / last_kv * 100.0,
        "oversized":   True,
    }
