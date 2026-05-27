"""Fanno flow physics — adiabatic, constant-area, frictional compressible duct."""

import math
import numpy as np

R_u = 8.31446  # J/(mol·K)

# Gas species: (MW kg/mol, gamma, CoolProp name or None)
GASES = {
    "Air":         (0.028965, 1.400, "Air"),
    "Hydrogen":    (0.002016, 1.405, "Hydrogen"),
    "Oxygen":      (0.031999, 1.395, "Oxygen"),
    "Nitrogen":    (0.028014, 1.400, "Nitrogen"),
    "Methane":     (0.016043, 1.304, "Methane"),
    "CO₂":         (0.044010, 1.289, "CarbonDioxide"),
    "Helium":      (0.004003, 1.667, "Helium"),
    "Argon":       (0.039948, 1.667, "Argon"),
    "Steam (H₂O)": (0.018015, 1.330, "Water"),
    "Ammonia":     (0.017031, 1.310, "Ammonia"),
    "Custom":      (None,     None,  None),
}

# Reference dynamic viscosity (Pa·s) at ~25 °C — fallback when CoolProp unavailable
MU_REF = {
    "Air":         1.85e-5,
    "Hydrogen":    8.90e-6,
    "Oxygen":      2.04e-5,
    "Nitrogen":    1.78e-5,
    "Methane":     1.11e-5,
    "CO₂":         1.49e-5,
    "Helium":      1.96e-5,
    "Argon":       2.27e-5,
    "Steam (H₂O)": 9.80e-6,
    "Ammonia":     1.01e-5,
    "Custom":      2.00e-5,
}


# ---------------------------------------------------------------------------
# Fanno-flow relations (all referenced to the sonic state Ma = 1 = * state)
# ---------------------------------------------------------------------------

def fanno_param(Ma: float, gamma: float) -> float:
    """4fL*/D — normalised friction length from Ma to the sonic limit."""
    if Ma <= 0:
        return math.inf
    g1 = gamma + 1.0
    g2 = gamma - 1.0
    denom = 2.0 + g2 * Ma ** 2
    return (1.0 - Ma ** 2) / (gamma * Ma ** 2) + g1 / (2.0 * gamma) * math.log(g1 * Ma ** 2 / denom)


def T_ratio(Ma: float, gamma: float) -> float:
    """T/T* — static temperature ratio to the critical state."""
    return (gamma + 1.0) / (2.0 + (gamma - 1.0) * Ma ** 2)


def P_ratio(Ma: float, gamma: float) -> float:
    """P/P* — static pressure ratio to the critical state."""
    return (1.0 / Ma) * math.sqrt(T_ratio(Ma, gamma))


def rho_ratio(Ma: float, gamma: float) -> float:
    """ρ/ρ* — density ratio to the critical state."""
    return P_ratio(Ma, gamma) / T_ratio(Ma, gamma)


def P0_ratio(Ma: float, gamma: float) -> float:
    """P₀/P₀* — stagnation-pressure ratio to the critical state."""
    g1 = gamma + 1.0
    g2 = gamma - 1.0
    base = (2.0 / g1) * (1.0 + g2 / 2.0 * Ma ** 2)
    exp = g1 / (2.0 * g2)
    return (1.0 / Ma) * base ** exp


def solve_Ma(target_4fLD: float, gamma: float, supersonic: bool = False) -> float:
    """Invert fanno_param via bisection. Subsonic: Ma ∈ (0,1); supersonic: Ma ∈ (1,50)."""
    if target_4fLD <= 0.0:
        return 1.0
    lo, hi = (1e-6, 1.0 - 1e-9) if not supersonic else (1.0 + 1e-9, 50.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = fanno_param(mid, gamma)
        if not supersonic:
            if val > target_4fLD:
                lo = mid
            else:
                hi = mid
        else:
            if val < target_4fLD:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


def churchill_f(Re: float, D: float, roughness: float) -> float:
    """Churchill (1977) explicit friction factor — valid laminar and turbulent."""
    if Re < 1.0:
        return 64.0
    eps_D = roughness / D
    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * eps_D)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)


def gas_viscosity(species: str, T_K: float, P_Pa: float) -> float:
    """Dynamic viscosity (Pa·s). Tries CoolProp; falls back to reference table."""
    cp_name = GASES.get(species, (None, None, None))[2]
    if cp_name:
        try:
            import CoolProp.CoolProp as CP
            return CP.PropsSI("V", "T", T_K, "P", P_Pa, cp_name)
        except Exception:
            pass
    return MU_REF.get(species, 2e-5)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def fanno_solve(
    P1_Pa: float,
    T1_K: float,
    mdot_kgs: float,
    D_m: float,
    L_m: float,
    roughness_m: float,
    MW_kgmol: float,
    gamma: float,
    species: str,
    N_points: int = 200,
) -> dict:
    """
    Solve Fanno flow from inlet conditions.

    Returns a dict with:
      Ma1, Ma2, P1_bara, P2_bara, T1_K, T2_K, P01_bara, P02_bara,
      V1_ms, V2_ms, a1_ms, a2_ms, Re1, f1,
      L_star_m, choked, margin_pct, dP_static_kPa, dP_stag_kPa,
      x_arr, Ma_arr, P_arr_bara, T_arr_K, P0_arr_bara, V_arr_ms
    """
    R_spec = R_u / MW_kgmol
    A = math.pi / 4.0 * D_m ** 2

    rho1 = P1_Pa / (R_spec * T1_K)
    a1 = math.sqrt(gamma * R_spec * T1_K)
    V1 = mdot_kgs / (rho1 * A)
    Ma1 = V1 / a1

    P01 = P1_Pa * (1.0 + (gamma - 1.0) / 2.0 * Ma1 ** 2) ** (gamma / (gamma - 1.0))

    mu1 = gas_viscosity(species, T1_K, P1_Pa)
    Re1 = rho1 * V1 * D_m / mu1
    f1 = churchill_f(Re1, D_m, roughness_m)

    available_4fLD = 4.0 * f1 * L_m / D_m
    fp1 = fanno_param(Ma1, gamma)
    L_star_m = fp1 * D_m / (4.0 * f1)

    fp2_target = fp1 - available_4fLD
    choked = fp2_target < 0.0
    supersonic = Ma1 > 1.0

    if choked:
        Ma2 = 1.0
        fp2 = 0.0
    else:
        fp2 = fp2_target
        Ma2 = solve_Ma(fp2, gamma, supersonic=supersonic)

    Tr1  = T_ratio(Ma1, gamma)
    Pr1  = P_ratio(Ma1, gamma)
    P0r1 = P0_ratio(Ma1, gamma)

    T_star  = T1_K / Tr1
    P_star  = P1_Pa / Pr1
    P0_star = P01 / P0r1

    T2  = T_star  * T_ratio(Ma2, gamma)
    P2  = P_star  * P_ratio(Ma2, gamma)
    P02 = P0_star * P0_ratio(Ma2, gamma)
    a2  = math.sqrt(gamma * R_spec * T2)
    V2  = Ma2 * a2

    margin_pct = (L_star_m - L_m) / L_star_m * 100.0 if L_star_m > 0 else math.inf

    x_arr  = np.linspace(0.0, L_m, N_points)
    Ma_arr = np.empty(N_points)
    P_arr  = np.empty(N_points)
    T_arr  = np.empty(N_points)
    P0_arr = np.empty(N_points)
    V_arr  = np.empty(N_points)

    for i, x in enumerate(x_arr):
        remaining = max(fp1 - 4.0 * f1 * x / D_m, 0.0)
        Ma_i  = solve_Ma(remaining, gamma, supersonic=supersonic)
        T_i   = T_star  * T_ratio(Ma_i, gamma)
        P_i   = P_star  * P_ratio(Ma_i, gamma)
        P0_i  = P0_star * P0_ratio(Ma_i, gamma)
        a_i   = math.sqrt(gamma * R_spec * T_i)
        Ma_arr[i] = Ma_i
        P_arr[i]  = P_i  / 1e5
        T_arr[i]  = T_i
        P0_arr[i] = P0_i / 1e5
        V_arr[i]  = Ma_i * a_i

    return {
        "Ma1": Ma1, "Ma2": Ma2,
        "P1_bara": P1_Pa / 1e5, "P2_bara": P2 / 1e5,
        "T1_K": T1_K, "T2_K": T2,
        "P01_bara": P01 / 1e5, "P02_bara": P02 / 1e5,
        "V1_ms": V1, "V2_ms": V2,
        "a1_ms": a1, "a2_ms": a2,
        "Re1": Re1, "f1": f1,
        "L_star_m": L_star_m,
        "choked": choked,
        "margin_pct": margin_pct,
        "dP_static_kPa": (P1_Pa - P2) / 1e3,
        "dP_stag_kPa":   (P01 - P02) / 1e3,
        "x_arr": x_arr,
        "Ma_arr": Ma_arr,
        "P_arr_bara": P_arr,
        "T_arr_K":    T_arr,
        "P0_arr_bara": P0_arr,
        "V_arr_ms":   V_arr,
    }
