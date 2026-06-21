"""
Pressure Safety Valve (PSV) sizing — API 520 Part I, SI edition.
Standard orifice selection — API 526.

Supported services: Gas / Vapour, Steam (via CoolProp), Liquid, Two-phase / Flashing.

API 521 relief-load scenario calculators:
  fire_heat_input, wetted_surface_area, scenario_relief_load.

API 520 Part II: inlet ΔP check helpers live in the app layer (they need the
pipe database and friction engine already imported there).

Reuses fanno_engine for gas species data and the universal gas constant.
"""

import math
import fanno_engine as _fan

R_u   = _fan.R_u
GASES = _fan.GASES

from standards.pressure_relief import (
    API526_ORIFICES, API526_FLANGE_NPS,
    KD_GAS, KD_LIQUID, KD_TWOPHASE, KC_DISC, KC_NONE,
    flange_nps,
)
from standards.piping import nps_to_dn, min_flange_class


# ── Gas coefficient ─────────────────────────────────────────────────────────

def c_gas(gamma: float) -> float:
    """API 520 Part I gas coefficient: C = 0.03948·√(γ·(2/(γ+1))^((γ+1)/(γ−1)))."""
    return 0.03948 * math.sqrt(gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))


# ── Backpressure corrections — gas/steam ────────────────────────────────────

def kb_conventional(P_back_Pa: float, P1_Pa: float) -> float:
    """Kb = 1.0 for conventional spring-loaded PSV (choked flow assumed)."""
    return 1.0


def kb_balanced_bellows(P_back_Pa: float, P1_Pa: float, gamma: float) -> float:
    """Kb for balanced-bellows PSV — API 520 Figure 31 conservative curve-fit."""
    r = P_back_Pa / P1_Pa if P1_Pa > 0 else 0.0
    r_c = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if r <= r_c:
        return 1.0
    return max(0.0, 1.0 - (r - r_c) / (0.9 - r_c))


def kb_pilot(P_back_Pa: float, P1_Pa: float) -> float:
    """Kb for pilot-operated PSV — 1.0 up to P_back/P₁ ≈ 0.85."""
    r = P_back_Pa / P1_Pa if P1_Pa > 0 else 0.0
    return 1.0 if r < 0.85 else 0.0


# ── Backpressure correction — liquid balanced bellows (API 520 Fig. 14) ─────

def kw_balanced_bellows_liquid(P_back_Pa: float, P1_Pa: float) -> float:
    """
    Kw for liquid balanced-bellows PSV — API 520 Part I Figure 14 curve-fit.

    Back-pressure ratio r = P_back / P₁ (both absolute):
      r ≤ 0.15          → Kw = 1.0
      0.15 < r ≤ 0.50   → linear decay to Kw ≈ 0.60
      0.50 < r ≤ 0.75   → steeper decay to Kw = 0.0 at r = 0.75
      r > 0.75           → Kw = 0.0
    """
    r = P_back_Pa / P1_Pa if P1_Pa > 0 else 0.0
    if r <= 0.15:
        return 1.0
    if r <= 0.50:
        return 1.0 - 1.143 * (r - 0.15)
    return max(0.0, 0.60 - 2.4 * (r - 0.50))


# ── Subcritical gas flow factor (API 520 Part I, §3.4) ──────────────────────

def f2_subcritical(k: float, r: float) -> float:
    """
    Isentropic subcritical nozzle flow factor F₂ — API 520 Part I §3.4.

    F₂ = √(k/(k-1) · (r^(2/k) - r^((k+1)/k)))   where r = P₂/P₁ (absolute).

    Valid when r > critical pressure ratio (subcritical regime).
    Returns 0 for r ≤ 0 or r ≥ 1.
    """
    if r <= 0.0 or r >= 1.0 or k <= 1.0:
        return 0.0
    term = r ** (2.0 / k) - r ** ((k + 1.0) / k)
    return math.sqrt(max(0.0, k / (k - 1.0) * term))


# ── Orifice selection ───────────────────────────────────────────────────────

def select_orifice(A_req_mm2: float) -> tuple[str, float]:
    """Smallest API 526 orifice whose effective area ≥ A_req_mm2."""
    for letter, area in API526_ORIFICES.items():
        if area >= A_req_mm2:
            return letter, area
    return "T+", math.inf


# ── Gas / vapour sizing (API 520 Part I §3.3 + §3.4) ───────────────────────

def psv_gas_size(
    W_kgh: float,
    P1_kPa: float,
    T1_K: float,
    MW_kgmol: float,
    gamma: float,
    Kb: float = 1.0,
    Kc: float = 1.0,
    Z: float = 1.0,
    P_back_Pa: float = 0.0,
) -> dict:
    """
    API 520 Part I gas / vapour PSV area (SI).

    When P_back_Pa / (P1_kPa·1000) > critical pressure ratio, the subcritical
    isentropic nozzle formula (API 520 §3.4) is used automatically.  The
    caller-supplied Kb is retained for critical-flow paths but ignored in the
    subcritical path (F₂ captures the full backpressure effect there).

    Parameters
    ----------
    W_kgh     : required relieving mass flow, kg/h
    P1_kPa    : upstream relieving pressure, kPa absolute
    T1_K      : relieving temperature, K
    MW_kgmol  : molecular weight, kg/kmol
    gamma     : ratio of specific heats Cp/Cv
    Kb        : back-pressure correction (1.0 conventional/pilot; from kb_* functions)
    Kc        : combination factor (KC_NONE or KC_DISC)
    Z         : compressibility factor
    P_back_Pa : absolute back pressure, Pa (0 → assume critical flow)
    """
    P1_Pa = P1_kPa * 1000.0
    r_c   = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    r     = (P_back_Pa / P1_Pa) if (P_back_Pa > 0 and P1_Pa > 0) else 0.0
    subcritical = r_c < r < 1.0

    C = c_gas(gamma)

    if subcritical:
        # API 520 §3.4 — isentropic subcritical nozzle
        # G = P₁ · √(2γ·M / ((γ-1)·Z·Ru·T) · (r^(2/γ) - r^((γ+1)/γ)))  [kg/(m²·s)]
        # R_u from fanno_engine is J/(mol·K); convert MW from kg/kmol to kg/mol
        term = r ** (2.0 / gamma) - r ** ((gamma + 1.0) / gamma)
        G    = P1_Pa * math.sqrt(
            2.0 * gamma * (MW_kgmol / 1000.0) / ((gamma - 1.0) * Z * R_u * T1_K) * term
        )
        A_req  = (W_kgh / 3600.0) / (KD_GAS * Kc * G) * 1e6
        letter, area = select_orifice(A_req)
        cap    = KD_GAS * Kc * G * (area * 1e-6) * 3600.0
        F2_val = f2_subcritical(gamma, r)
    else:
        # API 520 §3.3 — critical (choked) flow
        A_req  = W_kgh / (C * KD_GAS * P1_kPa * Kb * Kc) * math.sqrt(T1_K * Z / MW_kgmol)
        letter, area = select_orifice(A_req)
        cap    = C * KD_GAS * P1_kPa * Kb * Kc * area * math.sqrt(MW_kgmol / (T1_K * Z))
        F2_val = None

    return {
        "A_req_mm2":             A_req,
        "orifice_letter":        letter,
        "orifice_area_mm2":      area,
        "C_coeff":               C,
        "Kd":                    KD_GAS,
        "Kb":                    Kb,
        "Kc":                    Kc,
        "Z":                     Z,
        "critical_pressure_ratio": r_c,
        "subcritical":           subcritical,
        "F2":                    F2_val,
        "capacity_selected_kgh": cap,
    }


# ── Steam sizing (CoolProp for γ; same formula as gas) ─────────────────────

def psv_steam_size(
    W_kgh: float,
    P1_kPa: float,
    T1_K: float,
    Kb: float = 1.0,
    Kc: float = 1.0,
    P_back_Pa: float = 0.0,
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

    result = psv_gas_size(W_kgh, P1_kPa, T1_K, MW, gamma, Kb, Kc, Z, P_back_Pa)
    result["gamma"] = gamma
    result["MW"]    = MW
    result["notes"] = notes
    return result


# ── Liquid sizing (API 520 Part I, §3.5) ───────────────────────────────────

def _kv_liquid(W_kgh: float, A_mm2: float, mu_cP: float) -> float:
    """
    Viscosity correction factor Kv — API 520 Figure 13.
    R = 17 900 · W / (μ · √A)  [W kg/h, μ cP, A mm²]
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
    ṁ = Kd · Kw · Kc · Kv · A · √(2·ρ·ΔP)
    """
    dP_Pa = max((P1_kPa - P2_kPa) * 1000.0, 100.0)
    W_kgh = Q_m3h * rho_kgm3
    mdot  = Q_m3h * rho_kgm3 / 3600.0

    Kv    = 1.0
    A_mm2 = 0.0
    for _ in range(12):
        A_m2   = mdot / (KD_LIQUID * Kw * Kc * Kv * math.sqrt(2.0 * rho_kgm3 * dP_Pa))
        A_mm2  = A_m2 * 1e6
        Kv_new = _kv_liquid(W_kgh, A_mm2, mu_cP)
        if abs(Kv_new - Kv) < 1e-6:
            break
        Kv = Kv_new

    letter, area = select_orifice(A_mm2)
    Re_v  = 17_900.0 * W_kgh / (mu_cP * math.sqrt(area)) if area < math.inf else 0.0
    cap_m3h = (KD_LIQUID * Kw * Kc * Kv * (area * 1e-6) *
               math.sqrt(2.0 * rho_kgm3 * dP_Pa) / rho_kgm3 * 3600.0)

    return {
        "A_req_mm2":             A_mm2,
        "orifice_letter":        letter,
        "orifice_area_mm2":      area,
        "Kd":                    KD_LIQUID,
        "Kw":                    Kw,
        "Kv":                    Kv,
        "Kc":                    Kc,
        "Re_v":                  Re_v,
        "dP_kPa":                dP_Pa / 1000.0,
        "capacity_selected_m3h": cap_m3h,
    }


# ── Two-phase / flashing (API 520 Part I Appendix D, Omega method) ──────────

def _f_omega(eta: float, omega: float) -> float:
    """
    Dimensionless mass-flux-squared term from the Omega-model energy balance.

    G(η)² = 2·P₀/v₀ · f_omega(η, ω)

    For ω = 1 (L'Hôpital limit): f = (1 - η²) / 2
    For ω ≠ 1: f = (1-η)/(1-ω) + ω·ln((1-ω)η + ω) / (1-ω)²
    """
    if abs(omega - 1.0) < 1e-9:
        return (1.0 - eta ** 2) / 2.0
    t = (1.0 - omega) * eta + omega
    if t <= 0.0:
        return 0.0
    return (1.0 - eta) / (1.0 - omega) + omega * math.log(t) / (1.0 - omega) ** 2


def _critical_eta_omega(omega: float) -> float:
    """
    Critical pressure ratio η_c for the Omega method (Leung 1986, API 520 App. D).

    Solves the sonic condition G_energy(η_c) = G_sonic(η_c):
        η_c² / (2·ω) = f_omega(η_c, ω)

    Bisection on [1e-9, 1 - 1e-9]: F(η) = η²/(2ω) - f_omega(η, ω).
    F(0) < 0 and F(1) > 0 for all physically meaningful ω > 0.
    """
    def F(eta):
        return eta ** 2 / (2.0 * omega) - _f_omega(eta, omega)

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if F(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def omega_parameter(
    fluid: str,
    P0_Pa: float,
    T0_K: float,
    x0: float = 0.0,
) -> tuple[float, float, list[str]]:
    """
    Leung's Omega parameter for two-phase nozzle flow — API 520 Appendix D.

    For subcooled / bubble-point inlet (x0 = 0):
        ω_s = C_pf · T₀ · P₀ · v_f0 / h_fg0²

    For two-phase inlet (x0 > 0): blended ω using void fraction α₀.

    Returns (omega, v0_m3kg, notes).
      v0_m3kg is the inlet specific volume (used by psv_twophase_size).
    """
    notes: list[str] = []
    try:
        import CoolProp.CoolProp as CP

        # Saturation properties at P₀
        T_sat = CP.PropsSI("T", "P", P0_Pa, "Q", 0.0, fluid)
        rho_f = CP.PropsSI("D", "P", P0_Pa, "Q", 0.0, fluid)
        rho_g = CP.PropsSI("D", "P", P0_Pa, "Q", 1.0, fluid)
        h_f   = CP.PropsSI("H", "P", P0_Pa, "Q", 0.0, fluid)
        h_g   = CP.PropsSI("H", "P", P0_Pa, "Q", 1.0, fluid)
        h_fg  = h_g - h_f                          # J/kg
        v_f   = 1.0 / rho_f                        # m³/kg

        # Liquid Cp at inlet
        T_eval = min(T0_K, T_sat - 0.1)            # stay liquid-side for subcooled
        Cp_f   = CP.PropsSI("C", "T", T_eval, "P", P0_Pa, fluid)

        omega_s = Cp_f * T_sat * P0_Pa * v_f / h_fg ** 2

        if x0 <= 0.0:
            # Subcooled or bubble-point inlet
            v0 = 1.0 / CP.PropsSI("D", "T", T_eval, "P", P0_Pa, fluid)
            omega = omega_s
            notes.append(
                f"Subcooled/bubble-point inlet: ω_s = {omega:.5f} "
                f"(Cp_f={Cp_f:.1f} J/(kg·K), h_fg={h_fg/1e3:.1f} kJ/kg, "
                f"T_sat={T_sat-273.15:.1f} °C at P₀)"
            )
        else:
            # Two-phase inlet: blend ω using void fraction α₀
            v_g   = 1.0 / rho_g
            v0    = x0 * v_g + (1.0 - x0) * v_f
            alpha = x0 * v_g / v0                  # void fraction
            omega = alpha * (v_g / v_f) + omega_s * (1.0 - alpha * (v_g / v_f))
            notes.append(
                f"Two-phase inlet x₀={x0:.3f}: ω = {omega:.5f} "
                f"(α₀={alpha:.3f}, ω_s={omega_s:.5f})"
            )

        return omega, v0, notes

    except Exception as exc:
        notes.append(f"CoolProp error computing Omega: {exc}")
        return 1.0, 0.001, notes


def psv_twophase_size(
    W_kgh: float,
    P0_kPa: float,
    P_back_kPa: float,
    omega: float,
    v0_m3kg: float,
    Kc: float = 1.0,
) -> dict:
    """
    Two-phase / flashing PSV area — API 520 Appendix D, Omega method (Leung 1986).

    Parameters
    ----------
    W_kgh      : required relieving mass flow, kg/h
    P0_kPa     : relieving pressure (P₁), kPa absolute
    P_back_kPa : back pressure, kPa absolute
    omega      : Leung Omega parameter (from omega_parameter())
    v0_m3kg    : specific volume of the mixture at inlet, m³/kg
    Kc         : combination factor (1.0 or 0.9 with rupture disc)

    Returns
    -------
    dict — A_req_mm2, orifice_letter, orifice_area_mm2, omega, eta_c, G_c_kgm2s,
            Kd, Kc, subcritical, capacity_selected_kgh
    """
    P0_Pa   = P0_kPa * 1000.0
    eta_c   = _critical_eta_omega(omega)
    eta_bp  = P_back_kPa / P0_kPa if P0_kPa > 0 else 0.0

    subcritical = eta_bp > eta_c

    if subcritical:
        # Flow is subcritical — throat pressure equals back pressure
        f_val = _f_omega(eta_bp, omega)
        G     = math.sqrt(max(0.0, 2.0 * P0_Pa / v0_m3kg * f_val))
    else:
        # Critical flow — G_c = η_c · √(P₀/v₀) / √ω
        G = eta_c * math.sqrt(P0_Pa / v0_m3kg) / math.sqrt(omega)

    if G <= 0.0:
        A_req  = math.inf
        letter = "T+"
        area   = math.inf
        cap    = 0.0
    else:
        A_req          = (W_kgh / 3600.0) / (KD_TWOPHASE * Kc * G) * 1e6
        letter, area   = select_orifice(A_req)
        cap            = KD_TWOPHASE * Kc * G * (area * 1e-6) * 3600.0

    return {
        "A_req_mm2":             A_req,
        "orifice_letter":        letter,
        "orifice_area_mm2":      area,
        "omega":                 omega,
        "eta_c":                 eta_c,
        "G_c_kgm2s":             G,
        "Kd":                    KD_TWOPHASE,
        "Kc":                    Kc,
        "subcritical":           subcritical,
        "capacity_selected_kgh": cap,
    }


# ── API 521 — Fire case (§5.15) ─────────────────────────────────────────────

def fire_heat_input(A_wetted_m2: float, F_env: float = 1.0) -> float:
    """
    Total heat absorption from an open pool fire — API 521 §5.15.1.1 (SI).

    Q [W] = 43 200 · F · A_wetted^0.82

    Environmental factors F (API 521 Table 5):
      1.0 — no drainage / firefighting provisions
      0.5 — adequate drainage + remote-operated water spray
      0.3 — drainage + on-site firefighting organisation
    """
    return 43_200.0 * F_env * (A_wetted_m2 ** 0.82)


def wetted_surface_area(
    vessel_type: str,
    D_m: float,
    H_or_L_m: float,
) -> float:
    """
    Wetted surface area for fire-case sizing — API 521 §5.15.3 (SI).

    Parameters
    ----------
    vessel_type : "vertical", "horizontal", or "sphere"
    D_m         : vessel shell outer diameter, m
    H_or_L_m    : for vertical — liquid level height (m, max 7.6 m per API 521);
                  for horizontal — shell length (m);
                  for sphere    — not used (pass 0).

    Returns wetted surface area in m².
    """
    vt = vessel_type.lower()
    if vt == "vertical":
        H_eff = min(H_or_L_m, 7.6)
        return math.pi * D_m * H_eff + 0.5 * math.pi * D_m ** 2   # cylinder + bottom head
    if vt == "horizontal":
        L = H_or_L_m
        return math.pi * D_m * L + 0.5 * math.pi * D_m ** 2       # cylinder + two hemi-heads
    # sphere (equator and below)
    return math.pi * D_m ** 2


def scenario_relief_load(
    scenario: str,
    *,
    fluid: str = "Water",
    P1_kPa: float,
    T_K: float,
    # fire case
    vessel_type: str = "vertical",
    D_m: float = 1.0,
    H_or_L_m: float = 3.0,
    F_env: float = 1.0,
    # loss of cooling
    Q_duty_kW: float = 0.0,
    # thermal expansion
    V_vessel_m3: float = 1.0,
    Q_heat_kW: float = 0.0,
    # blocked discharge is a pass-through (user specifies W / Q directly)
    W_direct_kgh: float = 0.0,
    Q_direct_m3h: float = 0.0,
) -> dict:
    """
    API 521 relief-load calculators — return required relief rate for PSV sizing.

    Scenarios
    ---------
    "fire"              : pool fire heat input → vapor relief rate
    "loss_of_cooling"   : condenser/cooler duty lost → vapor generation rate
    "blocked_discharge" : pass-through (use normal process flow)
    "thermal_expansion" : blocked liquid + external heat → liquid volumetric relief

    Returns dict:
      mode        : "gas" or "liquid" (determines which sizing function to call)
      W_kgh       : required mass flow (gas/steam service)
      Q_m3h       : required volumetric flow (liquid service)
      notes       : list of calculation summary strings
    """
    notes: list[str] = []
    mode  = "gas"
    W_kgh = 0.0
    Q_m3h = 0.0

    try:
        import CoolProp.CoolProp as CP
        P_Pa = P1_kPa * 1000.0

        if scenario == "fire":
            A_w    = wetted_surface_area(vessel_type, D_m, H_or_L_m)
            Q_fire = fire_heat_input(A_w, F_env)          # W
            # Latent heat at relieving (set) pressure
            h_fg   = (CP.PropsSI("H", "P", P_Pa, "Q", 1.0, fluid) -
                      CP.PropsSI("H", "P", P_Pa, "Q", 0.0, fluid))   # J/kg
            W_kgh  = Q_fire / h_fg * 3600.0
            notes.append(
                f"A_wetted = {A_w:.2f} m²  |  Q_fire = {Q_fire/1e3:.1f} kW  "
                f"|  h_fg = {h_fg/1e3:.1f} kJ/kg  |  W_vapor = {W_kgh:.1f} kg/h"
            )
            mode = "gas"

        elif scenario == "loss_of_cooling":
            h_fg  = (CP.PropsSI("H", "P", P_Pa, "Q", 1.0, fluid) -
                     CP.PropsSI("H", "P", P_Pa, "Q", 0.0, fluid))
            W_kgh = Q_duty_kW * 1000.0 / h_fg * 3600.0
            notes.append(
                f"Q_duty = {Q_duty_kW:.1f} kW  |  h_fg = {h_fg/1e3:.1f} kJ/kg  "
                f"|  W_vapor = {W_kgh:.1f} kg/h"
            )
            mode = "gas"

        elif scenario == "blocked_discharge":
            W_kgh = W_direct_kgh
            Q_m3h = Q_direct_m3h
            mode  = "liquid" if W_direct_kgh == 0.0 and Q_direct_m3h > 0 else "gas"
            notes.append("Blocked discharge: relief rate = normal process flow (user-specified).")

        elif scenario == "thermal_expansion":
            # Blocked liquid with external heat: volumetric expansion relief
            beta  = CP.PropsSI("isobaric_expansion_coefficient", "T", T_K, "P", P_Pa, fluid)
            Cp    = CP.PropsSI("C",                              "T", T_K, "P", P_Pa, fluid)
            rho   = CP.PropsSI("D",                             "T", T_K, "P", P_Pa, fluid)
            Q_m3h = beta * Q_heat_kW * 1000.0 / (Cp * rho) * 3600.0
            mode  = "liquid"
            notes.append(
                f"β = {beta:.5f} 1/K  |  Cp = {Cp:.1f} J/(kg·K)  |  "
                f"ρ = {rho:.1f} kg/m³  |  Q_relief = {Q_m3h:.4f} m³/h"
            )

    except Exception as exc:
        notes.append(f"CoolProp error in scenario calculation: {exc}")

    return {
        "mode":   mode,
        "W_kgh":  W_kgh,
        "Q_m3h":  Q_m3h,
        "notes":  notes,
    }


# ── Convenience: back-pressure check ───────────────────────────────────────

def backpressure_pct(P_back_kPa: float, P_set_kPa: float) -> float:
    """Built-up back pressure as % of set pressure — API 520 limit checks."""
    return (P_back_kPa / P_set_kPa) * 100.0 if P_set_kPa > 0 else 0.0
