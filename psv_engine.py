"""
Pressure Safety Valve (PSV) sizing — API 520 Part I, SI edition.
Standard orifice selection — API 526.

Supported services: Gas / Vapour, Steam (via CoolProp), Liquid.

Reuses fanno_engine for gas species data and the universal gas constant.
"""

import math
import fanno_engine as _fan

R_u  = _fan.R_u
GASES = _fan.GASES

from standards.pressure_relief import (
    API526_ORIFICES, API526_FLANGE_NPS,
    KD_GAS, KD_LIQUID, KC_DISC, KC_NONE,
    flange_nps,
)
from standards.piping import nps_to_dn, min_flange_class


# ── Gas coefficient ─────────────────────────────────────────────────────────

def c_gas(gamma: float) -> float:
    """API 520 Part I gas coefficient: C = 0.03948·√(γ·(2/(γ+1))^((γ+1)/(γ−1)))."""
    return 0.03948 * math.sqrt(gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))


# ── Backpressure corrections ────────────────────────────────────────────────

def kb_conventional(P_back_Pa: float, P1_Pa: float) -> float:
    """Kb for conventional spring-loaded PSV.
    Kb = 1.0 when flow is choked (back pressure < critical ratio).
    Subcritical regime is unusual; warn rather than correct here.
    """
    return 1.0


def kb_balanced_bellows(P_back_Pa: float, P1_Pa: float, gamma: float) -> float:
    """
    Kb for balanced-bellows PSV — API 520 Figure 31 approximation.
    Below the critical pressure ratio the bellows valve remains choked (Kb = 1.0).
    Above it, Kb decreases; a conservative curve-fit is used.
    """
    r = P_back_Pa / P1_Pa if P1_Pa > 0 else 0.0
    r_c = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if r <= r_c:
        return 1.0
    # API 520 Fig 31 conservative linear drop from r_c to 0.9 (≈ complete loss at 90% BP)
    return max(0.0, 1.0 - (r - r_c) / (0.9 - r_c))


def kb_pilot(P_back_Pa: float, P1_Pa: float) -> float:
    """Kb for pilot-operated PSV — typically 1.0 up to P_back/P₁ ≈ 0.85."""
    r = P_back_Pa / P1_Pa if P1_Pa > 0 else 0.0
    return 1.0 if r < 0.85 else 0.0


# ── Orifice selection ───────────────────────────────────────────────────────

def select_orifice(A_req_mm2: float) -> tuple[str, float]:
    """Smallest API 526 orifice whose effective area ≥ A_req_mm2."""
    for letter, area in API526_ORIFICES.items():
        if area >= A_req_mm2:
            return letter, area
    return "T+", math.inf   # larger than the largest standard size


# ── Gas / vapour sizing (API 520 Part I, §3.3) ─────────────────────────────

def psv_gas_size(
    W_kgh: float,
    P1_kPa: float,
    T1_K: float,
    MW_kgmol: float,
    gamma: float,
    Kb: float = 1.0,
    Kc: float = 1.0,
    Z: float = 1.0,
) -> dict:
    """
    API 520 Part I gas / vapour PSV area (SI).

    Parameters
    ----------
    W_kgh    : required relieving mass flow, kg/h
    P1_kPa   : upstream relieving pressure, kPa absolute
                (set pressure + allowable overpressure + atmospheric)
    T1_K     : relieving temperature, K
    MW_kgmol : molecular weight, kg/kmol (= g/mol)
    gamma    : ratio of specific heats Cp/Cv
    Kb       : back-pressure correction factor (1.0 for conventional / pilot)
    Kc       : combination factor (KC_NONE or KC_DISC)
    Z        : gas compressibility factor (1.0 ideal)

    Returns
    -------
    dict — A_req_mm2, orifice_letter, orifice_area_mm2, C_coeff, Kd, Kb, Kc,
            critical_pressure_ratio, capacity_selected_kgh
    """
    C = c_gas(gamma)
    # API 520 SI equation:  A[mm²] = W / (C · Kd · P₁[kPa] · Kb · Kc) · √(T·Z/M)
    A_req = W_kgh / (C * KD_GAS * P1_kPa * Kb * Kc) * math.sqrt(T1_K * Z / MW_kgmol)
    letter, area = select_orifice(A_req)

    # Rated capacity of the selected orifice
    cap = C * KD_GAS * P1_kPa * Kb * Kc * area * math.sqrt(MW_kgmol / (T1_K * Z))

    return {
        "A_req_mm2":             A_req,
        "orifice_letter":        letter,
        "orifice_area_mm2":      area,
        "C_coeff":               C,
        "Kd":                    KD_GAS,
        "Kb":                    Kb,
        "Kc":                    Kc,
        "Z":                     Z,
        "critical_pressure_ratio": (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0)),
        "capacity_selected_kgh": cap,
    }


# ── Steam sizing (CoolProp for γ; same formula as gas) ─────────────────────

def psv_steam_size(
    W_kgh: float,
    P1_kPa: float,
    T1_K: float,
    Kb: float = 1.0,
    Kc: float = 1.0,
) -> dict:
    """
    PSV sizing for steam using CoolProp for γ and MW.
    Falls back to γ = 1.33 (saturated steam), MW = 18.015 if CoolProp unavailable.
    """
    gamma = 1.33
    MW    = 18.015
    Z     = 1.0
    notes = []

    try:
        import CoolProp.CoolProp as CP
        P1_Pa = P1_kPa * 1000.0
        # Guard: check saturation temperature — refuse to compute liquid properties as steam
        T_sat = CP.PropsSI("T", "P", P1_Pa, "Q", 1.0, "Water")
        if T1_K < T_sat - 0.5:
            notes.append(
                f"UNPHYSICAL: T₁ = {T1_K:.1f} K ({T1_K-273.15:.1f} °C) is below the "
                f"saturation temperature {T_sat:.1f} K ({T_sat-273.15:.1f} °C) at this pressure. "
                "Increase the relieving temperature or switch to Liquid service."
            )
        else:
            Cp = CP.PropsSI("C", "T", T1_K, "P", P1_Pa, "Water")
            Cv = CP.PropsSI("O", "T", T1_K, "P", P1_Pa, "Water")
            if Cv > 0:
                gamma = Cp / Cv
            rho = CP.PropsSI("D", "T", T1_K, "P", P1_Pa, "Water")
            Z = P1_Pa / (rho * (R_u / MW * 1000.0) * T1_K)
            notes.append(
                f"CoolProp: γ = {gamma:.4f}, Z = {Z:.4f} "
                f"({'superheated' if T1_K > T_sat + 0.5 else 'saturated'} steam)"
            )
    except Exception:
        notes.append("CoolProp unavailable — using γ = 1.33 (saturated steam), Z = 1.0")

    result = psv_gas_size(W_kgh, P1_kPa, T1_K, MW, gamma, Kb, Kc, Z)
    result["gamma"]    = gamma
    result["MW"]       = MW
    result["notes"]    = notes
    return result


# ── Liquid sizing (API 520 Part I, §3.5) ───────────────────────────────────

def _kv_liquid(W_kgh: float, A_mm2: float, mu_cP: float) -> float:
    """
    Viscosity correction factor Kv — API 520 Figure 13.
    Reynolds-number parameter: R = 17 900 · W / (μ · √A)
      W  kg/h, μ cP, A mm².
    Curve-fit: Kv = 1/(0.9935 + 2.878/R^0.5 + 342.75/R^1.5) per API 520.
    """
    if A_mm2 <= 0 or mu_cP <= 0:
        return 1.0
    R = 17_900.0 * W_kgh / (mu_cP * math.sqrt(A_mm2))
    if R >= 10_000.0:
        return 1.0
    return 1.0 / (0.9935 + 2.878 / math.sqrt(R) + 342.75 / R**1.5)


def psv_liquid_size(
    Q_m3h: float,
    P1_kPa: float,
    P2_kPa: float,
    rho_kgm3: float,
    mu_cP: float = 1.0,
    Kw: float = 1.0,
    Kc: float = 1.0,
) -> dict:
    """
    API 520 Part I liquid PSV area (SI).

    ṁ = Kd · Kw · Kc · Kv · A · √(2·ρ·ΔP)   [SI: kg/s, m², kg/m³, Pa]

    Parameters
    ----------
    Q_m3h   : required relieving volumetric flow, m³/h
    P1_kPa  : upstream relieving pressure, kPa absolute
    P2_kPa  : back pressure, kPa absolute
    rho_kgm3: liquid density, kg/m³
    mu_cP   : dynamic viscosity, cP (= mPa·s)
    Kw      : back-pressure correction for balanced-bellows (1.0 conventional)
    Kc      : combination factor (1.0 no disc, 0.9 with disc)

    Returns
    -------
    dict — A_req_mm2, orifice_letter, orifice_area_mm2, Kd, Kw, Kv, Kc, Re_v, dP_kPa,
            capacity_selected_m3h
    """
    dP_Pa  = max((P1_kPa - P2_kPa) * 1000.0, 100.0)
    W_kgh  = Q_m3h * rho_kgm3              # kg/h
    mdot   = Q_m3h * rho_kgm3 / 3600.0    # kg/s

    Kv = 1.0
    A_mm2 = 0.0
    for _ in range(12):
        A_m2  = mdot / (KD_LIQUID * Kw * Kc * Kv * math.sqrt(2.0 * rho_kgm3 * dP_Pa))
        A_mm2 = A_m2 * 1e6
        Kv_new = _kv_liquid(W_kgh, A_mm2, mu_cP)
        if abs(Kv_new - Kv) < 1e-6:
            break
        Kv = Kv_new

    letter, area = select_orifice(A_mm2)

    Re_v = 17_900.0 * W_kgh / (mu_cP * math.sqrt(area)) if area < math.inf else 0.0

    # Rated volumetric capacity of the selected orifice
    cap_m3h = (KD_LIQUID * Kw * Kc * Kv * (area * 1e-6) *
               math.sqrt(2.0 * rho_kgm3 * dP_Pa) / rho_kgm3 * 3600.0)

    return {
        "A_req_mm2":            A_mm2,
        "orifice_letter":       letter,
        "orifice_area_mm2":     area,
        "Kd":                   KD_LIQUID,
        "Kw":                   Kw,
        "Kv":                   Kv,
        "Kc":                   Kc,
        "Re_v":                 Re_v,
        "dP_kPa":               dP_Pa / 1000.0,
        "capacity_selected_m3h": cap_m3h,
    }


# ── Convenience: back-pressure check ───────────────────────────────────────

def backpressure_pct(P_back_kPa: float, P_set_kPa: float) -> float:
    """Built-up back pressure as % of set pressure — API 520 limit checks."""
    return (P_back_kPa / P_set_kPa) * 100.0 if P_set_kPa > 0 else 0.0
