"""
Dissolved gas flash calculator.

Physics:
  - Henry's Law:  C [mol/L] = K_H(T) [mol/L/bar] × P_gas [bar]
  - Sechenov:     K_H_soln = K_H_water × 10^(−K_s × c_salt)
  - Flash:        ΔC = C₁ − C₂  (positive → gas released)

Supported gases:  H₂, O₂
Supported solvents: Water, KOH (any wt%)

References:
  Battino et al. (1984) JPCRD vol 13 — H₂ and O₂ in water (tabular K_H)
  Tremosa et al. (2019) — Sechenov K_s for H₂ in KOH ≈ 0.069 L/mol
  Sander (2015) Atm. Chem. Phys. — O₂ K_s in KOH ≈ 0.132 L/mol
"""

import math
from typing import Literal

import multiphase_engine as _eng

# Universal gas constant [J/(mol·K)]
R_u = 8.314462

# KOH molar mass [g/mol]
MW_KOH = 56.105

# Standard conditions: 0 °C, 1 atm
_T_STD_K  = 273.15
_P_STD_Pa = 101_325.0

# ── Tabular K_H for H₂ in water ────────────────────────────────────────────
# K_H in mol/L/atm from Battino et al. (1984), Table 2.
# (i.e. C [mol/L] = K_H × P_H2 [atm])
# T range: 0–95 °C; converted to bar below.
# The minimum near 45–55 °C (non-monotonic) is captured.
_H2_BATTINO: list[tuple[float, float]] = [
    # (T_C, K_H [mol/L/atm])
    ( 0.0,  9.63e-4),
    ( 5.0,  8.84e-4),
    (10.0,  8.14e-4),
    (15.0,  7.63e-4),
    (20.0,  7.22e-4),
    (25.0,  6.92e-4),
    (30.0,  6.74e-4),
    (35.0,  6.62e-4),
    (40.0,  6.58e-4),
    (45.0,  6.58e-4),   # near-minimum plateau
    (50.0,  6.60e-4),
    (55.0,  6.66e-4),
    (60.0,  6.76e-4),
    (65.0,  6.89e-4),
    (70.0,  7.05e-4),
    (75.0,  7.23e-4),
    (80.0,  7.43e-4),
    (85.0,  7.65e-4),
    (90.0,  7.89e-4),
    (95.0,  8.15e-4),
]

# ── Tabular K_H for O₂ in water ────────────────────────────────────────────
# K_H in mol/L/atm from Battino et al. (1983) / Tromans (1998).
# O₂ is monotonically decreasing (higher T → less soluble up to ~100 °C).
_O2_BATTINO: list[tuple[float, float]] = [
    # (T_C, K_H [mol/L/atm])
    ( 0.0,  2.18e-3),
    ( 5.0,  1.91e-3),
    (10.0,  1.70e-3),
    (15.0,  1.52e-3),
    (20.0,  1.38e-3),
    (25.0,  1.26e-3),
    (30.0,  1.16e-3),
    (35.0,  1.09e-3),
    (40.0,  1.03e-3),
    (45.0,  9.80e-4),
    (50.0,  9.43e-4),
    (55.0,  9.15e-4),
    (60.0,  8.97e-4),
    (65.0,  8.87e-4),
    (70.0,  8.84e-4),
    (75.0,  8.88e-4),
    (80.0,  8.98e-4),
    (85.0,  9.14e-4),
    (90.0,  9.35e-4),
    (95.0,  9.62e-4),
]

# Sechenov constants K_s [L/mol] at 25 °C (log₁₀ form).
# Tremosa 2019 for H₂/KOH; Sander 2015 for O₂/NaOH (used as proxy for KOH).
_KS_25: dict[str, float] = {
    "H2": 0.069,
    "O2": 0.132,
}

# Tabular data registry
_TABLES: dict[str, list[tuple[float, float]]] = {
    "H2": _H2_BATTINO,
    "O2": _O2_BATTINO,
}

GAS_LABELS: dict[str, str] = {
    "H2": "Hydrogen (H₂)",
    "O2": "Oxygen (O₂)",
}

# Conversion: 1 atm = 1.01325 bar
_ATM_TO_BAR = 1.01325


# ── Interpolation ────────────────────────────────────────────────────────────

def _interp_kh_water(T_C: float, gas: str) -> float:
    """
    K_H in water [mol/L/bar] at T_C by linear interpolation in Battino tables.
    Extrapolates linearly outside the table range.
    """
    table = _TABLES[gas]
    if T_C <= table[0][0]:
        slope = (table[1][1] - table[0][1]) / (table[1][0] - table[0][0])
        kh_atm = table[0][1] + slope * (T_C - table[0][0])
    elif T_C >= table[-1][0]:
        slope = (table[-1][1] - table[-2][1]) / (table[-1][0] - table[-2][0])
        kh_atm = table[-1][1] + slope * (T_C - table[-1][0])
    else:
        for i in range(len(table) - 1):
            t0, k0 = table[i]
            t1, k1 = table[i + 1]
            if t0 <= T_C <= t1:
                f = (T_C - t0) / (t1 - t0)
                kh_atm = k0 + f * (k1 - k0)
                break
        else:
            kh_atm = table[-1][1]
    return max(kh_atm / _ATM_TO_BAR, 0.0)   # convert to mol/L/bar


# ── Sechenov correction ───────────────────────────────────────────────────────

def _ks_at_T(gas: str, T_C: float) -> float:
    """Sechenov K_s [L/mol] at temperature T_C using power-law T-dependence."""
    ks_25 = _KS_25[gas]
    T_K = T_C + 273.15
    # Mild temperature dependence from literature: K_s ~ (298.15/T)^0.3
    return ks_25 * (298.15 / T_K) ** 0.3


def koh_molarity(T_C: float, wt_pct: float) -> float:
    """KOH molarity [mol/L] at temperature T_C and concentration wt_pct."""
    rho_kgm3, _mu, _sig = _eng.koh_properties(T_C, wt_pct)
    rho_gL = rho_kgm3 * 1.0   # g/L (density numerically same as kg/m³ in g/L)
    mass_KOH_per_L = rho_gL * (wt_pct / 100.0)   # g/L
    return mass_KOH_per_L / MW_KOH   # mol/L


def kh_solution(T_C: float, gas: str, wt_pct_koh: float = 0.0) -> float:
    """
    Effective K_H [mol/L/bar] in KOH solution at T_C.
    wt_pct_koh=0 → pure water.
    """
    kh_water = _interp_kh_water(T_C, gas)
    if wt_pct_koh <= 0.0:
        return kh_water
    c_koh = koh_molarity(T_C, wt_pct_koh)
    ks = _ks_at_T(gas, T_C)
    # Sechenov: K_H_soln = K_H_water × 10^(−K_s × c_koh)
    return kh_water * 10.0 ** (-ks * c_koh)


# ── Henry concentration ───────────────────────────────────────────────────────

def dissolved_conc(T_C: float, P_total_bar: float, gas: str,
                   wt_pct_koh: float = 0.0,
                   y_gas: float = 1.0) -> dict:
    """
    Equilibrium dissolved concentration of gas at given T and P.

    Parameters
    ----------
    T_C          : temperature, °C
    P_total_bar  : total pressure, bar(a)
    gas          : "H2" or "O2"
    wt_pct_koh   : KOH concentration, wt%  (0 = pure water)
    y_gas        : mole fraction of gas in vapour phase (default 1.0 = pure gas)

    Returns
    -------
    dict: K_H_water, K_H_soln, c_koh_mol_L, K_s, P_gas_bar, C_mol_L
    """
    kh_w = _interp_kh_water(T_C, gas)
    kh_s = kh_solution(T_C, gas, wt_pct_koh)
    c_koh = koh_molarity(T_C, wt_pct_koh) if wt_pct_koh > 0 else 0.0
    ks    = _ks_at_T(gas, T_C)
    P_gas = P_total_bar * y_gas
    C     = kh_s * P_gas

    return {
        "K_H_water":    kh_w,
        "K_H_soln":     kh_s,
        "c_koh_mol_L":  c_koh,
        "K_s":          ks,
        "P_gas_bar":    P_gas,
        "C_mol_L":      C,
    }


# ── Flash calculation ─────────────────────────────────────────────────────────

def _std_vol_per_L(delta_C_mol_L: float, MW_g_mol: float) -> dict:
    """Convert Δc [mol/L_liquid] to volume and mass metrics."""
    # Nm³ gas per m³ liquid  (at 0 °C, 1 atm)
    mol_m3 = max(delta_C_mol_L, 0.0) * 1000.0   # mol/m³_liquid
    V_Nm3_m3 = mol_m3 * R_u * _T_STD_K / _P_STD_Pa   # Nm³/m³_liquid
    V_mL_L   = V_Nm3_m3 * 1000.0 * 1000.0 / 1000.0   # mL/L  (= V_Nm3_m3 × 1000 mL/Nm³ × 1/1000 m³/L)
    # Simpler: 1 mol of ideal gas at STP = 22.414 L = 22414 mL
    V_mL_L   = max(delta_C_mol_L, 0.0) * 22_414.0     # mL/L
    mass_gL  = max(delta_C_mol_L, 0.0) * MW_g_mol
    return {
        "V_mL_per_L":   V_mL_L,
        "V_Nm3_per_m3": max(delta_C_mol_L, 0.0) * 22.414,   # Nm³/m³
        "mass_g_per_L": mass_gL,
    }


_MW_GAS = {"H2": 2.016, "O2": 31.999}


def flash_dissolved_gas(
    gas: str,
    T1_C: float,
    P1_bar: float,
    T2_C: float,
    P2_bar: float,
    wt_pct_koh: float = 30.0,
    y_gas: float = 1.0,
) -> dict:
    """
    Calculate gas released when saturated liquid flashes from (T1, P1) to (T2, P2).

    All concentrations in mol/L liquid.  Released volume in mL/L and Nm³/m³.

    Parameters
    ----------
    gas         : "H2" or "O2"
    T1_C, P1_bar: inlet conditions (fully saturated at these conditions)
    T2_C, P2_bar: outlet conditions after HX / valve
    wt_pct_koh  : KOH concentration, wt%  (0 = pure water)
    y_gas       : mole fraction of gas in vapour space (1.0 = pure gas atmosphere)

    Returns
    -------
    dict with full breakdown of each effect and combined flash result.
    """
    if gas not in _TABLES:
        raise ValueError(f"Unsupported gas '{gas}'. Choose from {list(_TABLES)}")

    mw = _MW_GAS[gas]

    # ── State 1: inlet (saturated) ──────────────────────────────────────────
    s1 = dissolved_conc(T1_C, P1_bar, gas, wt_pct_koh, y_gas)

    # ── State 2: outlet ─────────────────────────────────────────────────────
    s2 = dissolved_conc(T2_C, P2_bar, gas, wt_pct_koh, y_gas)

    # ── Intermediate states for decomposition ───────────────────────────────
    # Pressure effect only: T1 → T1, P1 → P2
    s_P = dissolved_conc(T1_C, P2_bar, gas, wt_pct_koh, y_gas)
    # Temperature effect only: T1 → T2, P1 → P1
    s_T = dissolved_conc(T2_C, P1_bar, gas, wt_pct_koh, y_gas)

    dC_combined = s1["C_mol_L"] - s2["C_mol_L"]
    dC_pressure = s1["C_mol_L"] - s_P["C_mol_L"]
    dC_temp     = s1["C_mol_L"] - s_T["C_mol_L"]

    combined = _std_vol_per_L(dC_combined, mw)
    pressure_only = _std_vol_per_L(dC_pressure, mw)
    temp_only     = _std_vol_per_L(dC_temp, mw)

    # Actual gas volume fraction at outlet conditions (what the downstream pump sees)
    T2_K   = T2_C + 273.15
    P2_Pa  = P2_bar * 1e5
    n_released_mol_per_m3 = max(dC_combined, 0.0) * 1000.0      # mol/m³_liquid
    V_gas_actual_m3_per_m3 = n_released_mol_per_m3 * R_u * T2_K / P2_Pa
    vol_pct_outlet = V_gas_actual_m3_per_m3 / (1.0 + V_gas_actual_m3_per_m3) * 100.0

    return {
        "gas":          gas,
        "T1_C":         T1_C,
        "P1_bar":       P1_bar,
        "T2_C":         T2_C,
        "P2_bar":       P2_bar,
        "wt_pct_koh":   wt_pct_koh,
        "y_gas":        y_gas,

        # State 1 (inlet, saturated)
        "C1_mol_L":     s1["C_mol_L"],
        "K_H1_water":   s1["K_H_water"],
        "K_H1_soln":    s1["K_H_soln"],
        "K_s1":         s1["K_s"],
        "c_koh1_mol_L": s1["c_koh_mol_L"],

        # State 2 (outlet)
        "C2_mol_L":     s2["C_mol_L"],
        "K_H2_water":   s2["K_H_water"],
        "K_H2_soln":    s2["K_H_soln"],
        "K_s2":         s2["K_s"],

        # Intermediate
        "C_P_mol_L":    s_P["C_mol_L"],   # after P drop, same T
        "C_T_mol_L":    s_T["C_mol_L"],   # after cooling, same P

        # Net changes (positive → gas released from liquid; negative → stays in solution)
        "dC_combined_mol_L":  dC_combined,
        "dC_pressure_mol_L":  dC_pressure,
        "dC_temp_mol_L":      dC_temp,

        # Volumetric / mass output — combined flash
        "released_mL_per_L":   combined["V_mL_per_L"],
        "released_Nm3_per_m3": combined["V_Nm3_per_m3"],
        "released_g_per_L":    combined["mass_g_per_L"],

        # Decomposed effects
        "pressure_effect":  pressure_only,
        "temp_effect":      temp_only,
        "combined_effect":  combined,

        # Actual gas volume fraction at outlet conditions (m³_gas / m³_total mixture)
        "V_gas_actual_m3_per_m3": V_gas_actual_m3_per_m3,
        "vol_pct_outlet":         vol_pct_outlet,
    }
