"""
Restriction Orifice (RO) sizing and rating.

Correlations:
  ISO 5167-2:2022  — discharge coefficient C (Reader-Harris/Gallagher, corner taps)
                     expansion factor ε for compressible flow
  API RP 520 Pt I  — critical (choked) mass flow for gas
  IEC 60534-2      — cavitation index Kc for liquid service

Reuses fanno_engine for gas species data and viscosity.
"""

import math
import fanno_engine as _fan
import multiphase_engine as _eng

# Re-export constants used by the UI
R_u   = _fan.R_u
GASES = _fan.GASES

# ---------------------------------------------------------------------------
# Cavitation thresholds (Kc = ΔP / (P₁ − Pᵥ))
# ---------------------------------------------------------------------------
KC_INCIPIENT = 0.50   # onset: noise and erosion begin
KC_SEVERE    = 0.70   # heavy erosion / structural risk
KC_CHOKED    = 1.00   # liquid choke (P₂ approaches Pᵥ)

# ISO 5167-2 validity window for RHG equation (corner taps)
_BETA_MIN = 0.10
_BETA_MAX = 0.75
_RE_MIN   = 5_000


# ---------------------------------------------------------------------------
# Core orifice relations
# ---------------------------------------------------------------------------

def critical_pressure_ratio(gamma: float) -> float:
    """r_c = (2/(γ+1))^(γ/(γ−1)) — choke condition at orifice throat."""
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def expansion_factor(beta: float, P1_Pa: float, P2_Pa: float, gamma: float) -> float:
    """
    ISO 5167-2 expansion factor ε (corner taps).
    Accounts for gas compressibility downstream of the orifice.
    Returns 1.0 for liquid (or when P2 ≥ P1).
    """
    if P2_Pa >= P1_Pa or gamma <= 1.0:
        return 1.0
    r = max(P2_Pa / P1_Pa, 1e-6)
    return 1.0 - (0.351 + 0.256 * beta**4 + 0.93 * beta**8) * (1.0 - r ** (1.0 / gamma))


def cd_iso5167(beta: float, Re_D: float) -> float:
    """
    ISO 5167-2 discharge coefficient C (Reader-Harris/Gallagher, corner taps).
    Falls back to C = 0.60 outside the RHG validity window.

    Note: ISO uses ṁ = C·ε·(π/4·d²)·√(2·ΔP·ρ₁) — C already includes the
    velocity-of-approach factor √(1−β⁴) used in some other formulations.
    """
    if beta < _BETA_MIN or beta > _BETA_MAX or Re_D < _RE_MIN:
        return 0.60
    A = (19_000.0 * beta / Re_D) ** 0.8
    return (
        0.5961
        + 0.0261 * beta**2
        - 0.216  * beta**8
        + 0.000521 * (1e6 * beta / Re_D) ** 0.7
        + (0.0188 + 0.0063 * A) * beta**3.5 * (1e6 / Re_D) ** 0.3
    )


def ma_throat(P1_Pa: float, P2_Pa: float, gamma: float) -> float:
    """
    Mach number at orifice throat assuming isentropic expansion.
    Returns 1.0 (sonic) when P₂/P₁ ≤ critical pressure ratio.
    """
    r_c = critical_pressure_ratio(gamma)
    r   = P2_Pa / P1_Pa if P1_Pa > 0 else 0.0
    if r <= r_c:
        return 1.0
    return math.sqrt(2.0 / (gamma - 1.0) * ((1.0 / r) ** ((gamma - 1.0) / gamma) - 1.0))


def _kc(P1_Pa: float, P2_Pa: float, Pv_Pa: float) -> float:
    """Cavitation index Kc = ΔP / (P₁ − Pᵥ)."""
    ref = P1_Pa - Pv_Pa
    return (P1_Pa - P2_Pa) / ref if ref > 0 else math.inf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _re_pipe(mdot_kgs: float, D_m: float, mu_pas: float) -> float:
    return 4.0 * mdot_kgs / (math.pi * D_m * mu_pas)


def _choked_flow_coeff(gamma: float) -> float:
    """Dimensionless choked-flow coefficient Φ so that ṁ = Cd·A₀·P₁·Φ/√(R_spec·T₁)."""
    return math.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


# ---------------------------------------------------------------------------
# Gas RO: rate (bore → flow)
# ---------------------------------------------------------------------------

def ro_gas_rate(
    P1_Pa: float, P2_Pa: float, T1_K: float,
    MW_kgmol: float, gamma: float, species: str,
    d_m: float, D_m: float,
) -> dict:
    """Rate a gas RO: given orifice bore d, return mass flow and diagnostics."""
    R_spec  = R_u / MW_kgmol
    rho1    = P1_Pa / (R_spec * T1_K)
    mu      = _fan.gas_viscosity(species, T1_K, P1_Pa)
    beta    = d_m / D_m
    A0      = math.pi / 4.0 * d_m**2
    r_c     = critical_pressure_ratio(gamma)
    choked  = (P2_Pa / P1_Pa) <= r_c
    Phi     = _choked_flow_coeff(gamma)

    C   = 0.60
    eps = 1.0 if choked else expansion_factor(beta, P1_Pa, P2_Pa, gamma)

    for _ in range(40):
        if choked:
            mdot = C * A0 * P1_Pa * Phi / math.sqrt(R_spec * T1_K)
        else:
            mdot = C * eps * A0 * math.sqrt(2.0 * (P1_Pa - P2_Pa) * rho1)

        Re_D  = _re_pipe(mdot, D_m, mu)
        C_new = cd_iso5167(beta, Re_D)
        eps_new = 1.0 if choked else expansion_factor(beta, P1_Pa, P2_Pa, gamma)
        if abs(C_new - C) < 1e-7 and abs(eps_new - eps) < 1e-7:
            C, eps = C_new, eps_new
            break
        C, eps = C_new, eps_new

    Re_D = _re_pipe(mdot, D_m, mu)
    return {
        "mdot_kgs": mdot,
        "C": C, "eps": eps, "beta": beta,
        "Re_D": Re_D, "rho1_kgm3": rho1,
        "choked": choked, "r_c": r_c,
        "Ma_throat": ma_throat(P1_Pa, P2_Pa, gamma),
    }


# ---------------------------------------------------------------------------
# Gas RO: size (flow → bore)
# ---------------------------------------------------------------------------

def ro_gas_size(
    P1_Pa: float, P2_Pa: float, T1_K: float,
    MW_kgmol: float, gamma: float, species: str,
    mdot_kgs: float, D_m: float,
) -> dict:
    """Size a gas RO: given required mass flow, return orifice bore."""
    R_spec = R_u / MW_kgmol
    rho1   = P1_Pa / (R_spec * T1_K)
    mu     = _fan.gas_viscosity(species, T1_K, P1_Pa)
    Re_D   = _re_pipe(mdot_kgs, D_m, mu)   # independent of d
    r_c    = critical_pressure_ratio(gamma)
    choked = (P2_Pa / P1_Pa) <= r_c
    Phi    = _choked_flow_coeff(gamma)
    dP     = P1_Pa - P2_Pa

    C    = 0.60
    beta = 0.40
    d_m  = D_m * beta

    for _ in range(60):
        eps = 1.0 if choked else expansion_factor(beta, P1_Pa, P2_Pa, gamma)
        if choked:
            A0  = mdot_kgs / (C * P1_Pa * Phi / math.sqrt(R_spec * T1_K))
        else:
            A0  = mdot_kgs / (C * eps * math.sqrt(2.0 * dP * rho1))
        d_new  = math.sqrt(4.0 * A0 / math.pi)
        beta   = d_new / D_m
        C_new  = cd_iso5167(beta, Re_D)
        if abs(d_new - d_m) / max(d_m, 1e-9) < 1e-7 and abs(C_new - C) < 1e-7:
            d_m, C = d_new, C_new
            break
        d_m, C = d_new, C_new

    eps = 1.0 if choked else expansion_factor(beta, P1_Pa, P2_Pa, gamma)
    return {
        "d_m": d_m,
        "C": C, "eps": eps, "beta": beta,
        "Re_D": Re_D, "rho1_kgm3": rho1,
        "choked": choked, "r_c": r_c,
        "Ma_throat": ma_throat(P1_Pa, P2_Pa, gamma),
    }


# ---------------------------------------------------------------------------
# Liquid RO: rate (bore → flow)
# ---------------------------------------------------------------------------

def ro_liquid_rate(
    P1_Pa: float, P2_Pa: float,
    rho_kgm3: float, mu_pas: float, Pv_Pa: float,
    d_m: float, D_m: float,
) -> dict:
    """Rate a liquid RO: given orifice bore d, return mass flow and cavitation status."""
    dP   = P1_Pa - P2_Pa
    beta = d_m / D_m
    A0   = math.pi / 4.0 * d_m**2
    C    = 0.60

    for _ in range(40):
        mdot  = C * A0 * math.sqrt(2.0 * dP * rho_kgm3)
        Re_D  = _re_pipe(mdot, D_m, mu_pas)
        C_new = cd_iso5167(beta, Re_D)
        if abs(C_new - C) < 1e-7:
            C = C_new
            break
        C = C_new

    Re_D = _re_pipe(mdot, D_m, mu_pas)
    Kc   = _kc(P1_Pa, P2_Pa, Pv_Pa)
    return {
        "mdot_kgs": mdot,
        "C": C, "beta": beta, "Re_D": Re_D,
        "Kc": Kc,
        "cavitating": Kc >= KC_INCIPIENT,
        "cavitation_severe": Kc >= KC_SEVERE,
        "choked": P2_Pa <= Pv_Pa,
    }


# ---------------------------------------------------------------------------
# Liquid RO: size (flow → bore)
# ---------------------------------------------------------------------------

def ro_liquid_size(
    P1_Pa: float, P2_Pa: float,
    rho_kgm3: float, mu_pas: float, Pv_Pa: float,
    mdot_kgs: float, D_m: float,
) -> dict:
    """Size a liquid RO: given required mass flow, return orifice bore."""
    dP   = P1_Pa - P2_Pa
    Re_D = _re_pipe(mdot_kgs, D_m, mu_pas)
    C    = 0.60
    beta = 0.40
    d_m  = D_m * beta

    for _ in range(60):
        A0    = mdot_kgs / (C * math.sqrt(2.0 * dP * rho_kgm3))
        d_new = math.sqrt(4.0 * A0 / math.pi)
        beta  = d_new / D_m
        C_new = cd_iso5167(beta, Re_D)
        if abs(d_new - d_m) / max(d_m, 1e-9) < 1e-7 and abs(C_new - C) < 1e-7:
            d_m, C = d_new, C_new
            break
        d_m, C = d_new, C_new

    Kc = _kc(P1_Pa, P2_Pa, Pv_Pa)
    return {
        "d_m": d_m,
        "C": C, "beta": beta, "Re_D": Re_D,
        "Kc": Kc,
        "cavitating": Kc >= KC_INCIPIENT,
        "cavitation_severe": Kc >= KC_SEVERE,
        "choked": P2_Pa <= Pv_Pa,
    }


# ---------------------------------------------------------------------------
# Multi-stage gas: equal pressure-ratio distribution
# ---------------------------------------------------------------------------

def multistage_gas(
    P1_Pa: float, P2_Pa: float, T1_K: float,
    MW_kgmol: float, gamma: float, species: str,
    mdot_kgs: float, D_m: float,
    margin: float = 0.90,
) -> dict:
    """
    Recommend N gas RO stages so no stage chokes (P_out/P_in > r_c × margin).
    Returns stage count, pressure schedule, and sized bore per stage.
    """
    r_c        = critical_pressure_ratio(gamma)
    r_c_design = r_c * margin
    r_overall  = P2_Pa / P1_Pa

    if r_overall >= r_c_design:
        N = 1
    else:
        N = math.ceil(math.log(r_overall) / math.log(r_c_design))
        N = max(N, 2)

    r_stage    = r_overall ** (1.0 / N)
    P_schedule = [P1_Pa * r_overall ** (k / N) for k in range(N + 1)]

    stages = []
    for k in range(N):
        P_in  = P_schedule[k]
        P_out = P_schedule[k + 1]
        res   = ro_gas_size(P_in, P_out, T1_K, MW_kgmol, gamma, species, mdot_kgs, D_m)
        stages.append({
            "stage": k + 1,
            "P_in_bara":  P_in  / 1e5,
            "P_out_bara": P_out / 1e5,
            "dP_bar":    (P_in - P_out) / 1e5,
            "d_mm":       res["d_m"] * 1000,
            "beta":       res["beta"],
            "C":          res["C"],
            "eps":        res["eps"],
            "Ma_throat":  res["Ma_throat"],
            "Re_D":       res["Re_D"],
        })

    return {
        "N": N,
        "r_c": r_c, "r_c_design": r_c_design,
        "r_stage": r_stage,
        "P_schedule_bara": [p / 1e5 for p in P_schedule],
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Multi-stage liquid: equal ΔP distribution
# ---------------------------------------------------------------------------

def multistage_liquid(
    P1_Pa: float, P2_Pa: float,
    rho_kgm3: float, mu_pas: float, Pv_Pa: float,
    mdot_kgs: float, D_m: float,
    kc_limit: float = KC_INCIPIENT,
) -> dict:
    """
    Recommend N liquid RO stages so worst-stage Kc stays below kc_limit.
    Equal ΔP per stage; worst Kc is at the last stage (lowest upstream P).
    """
    dP_total  = P1_Pa - P2_Pa
    Kc_single = _kc(P1_Pa, P2_Pa, Pv_Pa)

    if Kc_single < kc_limit:
        N = 1
    else:
        denom = kc_limit * (P2_Pa - Pv_Pa)
        if denom <= 0:
            N = max(10, math.ceil(Kc_single / kc_limit))
        else:
            N = math.ceil(dP_total * (1.0 - kc_limit) / denom)
            N = max(N, 2)

    dP_stage   = dP_total / N
    P_schedule = [P1_Pa - k * dP_stage for k in range(N + 1)]

    stages = []
    for k in range(N):
        P_in  = P_schedule[k]
        P_out = P_schedule[k + 1]
        res   = ro_liquid_size(P_in, P_out, rho_kgm3, mu_pas, Pv_Pa, mdot_kgs, D_m)
        stages.append({
            "stage":      k + 1,
            "P_in_bara":  P_in  / 1e5,
            "P_out_bara": P_out / 1e5,
            "dP_bar":     dP_stage / 1e5,
            "d_mm":       res["d_m"] * 1000,
            "beta":       res["beta"],
            "C":          res["C"],
            "Kc":         res["Kc"],
            "Re_D":       res["Re_D"],
        })

    return {
        "N": N,
        "Kc_single": Kc_single,
        "kc_limit":  kc_limit,
        "P_schedule_bara": [p / 1e5 for p in P_schedule],
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Liquid properties helpers
# ---------------------------------------------------------------------------

# Reference viscosity at 20 °C (Pa·s) for fluids where CoolProp has no transport model
_MU_LIQ_REF = {
    "Acetone": 3.06e-4,
}


def liquid_properties(coolprop_id: str, T_C: float, P_Pa: float) -> dict:
    """
    ρ, μ, Pᵥ for any CoolProp liquid at temperature T_C (°C) and pressure P_Pa.
    Pᵥ is the saturation pressure at T (returns 0 if above the critical temperature).
    For fluids without a CoolProp transport model, μ falls back to a reference table.
    """
    import CoolProp.CoolProp as CP
    T_K = T_C + 273.15
    rho = CP.PropsSI("D", "T", T_K, "P", P_Pa, coolprop_id)
    try:
        mu = CP.PropsSI("V", "T", T_K, "P", P_Pa, coolprop_id)
    except Exception:
        mu = _MU_LIQ_REF.get(coolprop_id, 1e-3)   # 1 cP generic fallback
    try:
        Pv = CP.PropsSI("P", "T", T_K, "Q", 0, coolprop_id)
    except Exception:
        Pv = 0.0   # above critical temperature — no bubble point
    return {"rho_kgm3": rho, "mu_pas": mu, "Pv_Pa": Pv}


def water_properties(T_C: float, P_Pa: float = 1e5) -> dict:
    """Backward-compatible wrapper — use liquid_properties('Water', ...) directly."""
    try:
        return liquid_properties("Water", T_C, P_Pa)
    except Exception:
        T   = T_C
        rho = 999.84 - 0.0643 * T - 0.00363 * T**2
        mu  = 1e-3 * math.exp(-3.5656 + 507.4 / (T + 149.4))
        Pv  = 611.21 * math.exp((18.678 - T / 234.5) * T / (257.14 + T))
        return {"rho_kgm3": rho, "mu_pas": mu, "Pv_Pa": Pv}


# Water-activity depression of KOH solutions (Raoult-based empirical fit, CRC Handbook)
# a_w ≈ 1 − 0.00705·w  (w in wt%, valid 0–40 wt%, accuracy ±3 %)
_KOH_ACTIVITY_SLOPE = 0.00705

KOH_CONC_DEFAULT = 30.0   # wt% — primary design point per user requirement


def koh_liquid_properties(T_C: float, conc_wt_pct: float = KOH_CONC_DEFAULT) -> dict:
    """
    ρ, μ, Pᵥ for aqueous KOH solution.

    Delegates ρ and μ to multiphase_engine.koh_properties (Gilliam et al. 2007).
    Pᵥ = activity × Pᵥ_water, where water activity is estimated from a linear
    Raoult fit calibrated to CRC Handbook vapour-pressure data (0–40 wt%).

    Parameters
    ----------
    T_C         : temperature, °C (model valid 10–90 °C)
    conc_wt_pct : KOH concentration, wt% (0–40 wt%)

    Returns
    -------
    dict with keys rho_kgm3, mu_pas, Pv_Pa  — same schema as liquid_properties()
    """
    rho, mu, _sigma = _eng.koh_properties(T_C, conc_wt_pct)
    a_w = max(0.0, 1.0 - _KOH_ACTIVITY_SLOPE * conc_wt_pct)
    try:
        Pv_water = liquid_properties("Water", T_C, 1e5)["Pv_Pa"]
    except Exception:
        T = T_C
        Pv_water = 611.21 * math.exp((18.678 - T / 234.5) * T / (257.14 + T))
    return {"rho_kgm3": rho, "mu_pas": mu, "Pv_Pa": a_w * Pv_water}
