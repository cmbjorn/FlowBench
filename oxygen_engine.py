"""Gaseous-oxygen velocity limits and safety screening per EIGA Doc 13/20
(*Oxygen Pipeline and Piping Systems*).

Pure Python, no Streamlit. The heart is the EIGA pressure–velocity model:

* **Exemption (§4.3, App. B)** — below its exemption pressure (and at/above the
  minimum thickness) a metal is burn-resistant and has *no* velocity limit.
* **Velocity curves (§4.4.2 impingement / §4.4.3 non-impingement)** — apply when
  a metal is used above its exemption pressure (or is non-exempt, e.g. carbon
  steel). The curves address the *particle-impact* ignition mechanism only;
  other mechanisms (App. D) and noise / vibration / ΔP must be checked
  separately (§4.4.2).
* **Low-pressure / low-purity escape hatches** — P < 0.21 MPa abs (§4.4.2) and
  O₂ < 35 vol% (§4.3.2.2).

All pressures here are **absolute**. The standard's general convention is gauge
(§3.2.15) but Figures 2/3 are explicitly in MPa abs, so we work in abs and make
the basis explicit to the user. Exemption pressures (App. B) are applied as
absolute too — the conservative reading for claiming exemption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log10, pi

_R = 8.314          # J/mol/K
_GAMMA = 1.4        # diatomic (O₂/N₂) — speed-of-sound / Mach screening
_LOW_P_MPA = 0.21   # §4.4.2 low-pressure exemption threshold (MPa abs)
_SCOPE_MAX_MPA = 21.0  # Doc 13 scope limit (§2)
_LB_PER_KG = 2.2046226  # kg → lb (Carucci–Mueller uses lb/s, °R)
_ATM_BAR = 1.01325

# ── EIGA velocity curves (P in MPa abs → V in m/s) ────────────────────────────
def impingement_velocity_limit(P_mpa_abs: float) -> float:
    """Figure 2 impingement-site curve (§4.4.2)."""
    if P_mpa_abs <= 1.5:
        return 30.0                      # 0.3–1.5 MPa plateau (100 ft/s)
    if P_mpa_abs <= 10.0:
        return 45.0 / P_mpa_abs          # P·V = 45 MPa·m/s
    return 4.5                           # 10–20 MPa (15 ft/s)


def nonimpingement_velocity_limit(P_mpa_abs: float) -> float:
    """Figure 3 non-impingement-site curve (§4.4.3)."""
    if P_mpa_abs <= 1.5:
        return 60.0                      # 0.3–1.5 MPa plateau (200 ft/s)
    if P_mpa_abs <= 10.0:
        return 80.0 / P_mpa_abs          # P·V = 80 MPa·m/s
    return 8.0                           # 10–20 MPa (26.6 ft/s)


# ── Exemption pressures (Appendix B), keyed by alloy family ───────────────────
# Each family lists (min_thickness_mm, exemption_pressure_MPa_abs) tiers; the
# highest-pressure tier whose thickness requirement is met applies.
EIGA_EXEMPTION: dict[str, list[tuple[float, float]]] = {
    "stainless_300": [(3.18, 1.38), (6.35, 2.58)],  # 304/304L, 316/316L, 321, 347
    "carbon_steel":  [],                            # not an exempt family
    "copper":        [(0.0, 20.68)],
    "copper_nickel": [(0.0, 20.68)],
    "brass":         [(0.0, 20.68)],
    "tin_bronze":    [(0.0, 20.68)],
    "nickel_200":    [(0.0, 20.68)],
    "monel_400":     [(0.762, 20.68)],
    "monel_k500":    [(0.762, 20.68)],
    "inconel_600":   [(3.18, 8.61)],
    "inconel_625":   [(3.18, 6.90)],
    "hastelloy_c276": [(3.18, 8.61)],
    "hastelloy_c22": [(3.18, 9.70)],
}

# Material label → EIGA alloy family (friendly labels for the UI dropdown plus
# the QuickPipe short codes for backward compatibility).
_MATERIAL_FAMILY: dict[str, str] = {
    "Carbon steel": "carbon_steel",
    "Stainless 316/304": "stainless_300",
    "Monel 400": "monel_400",
    "Monel K-500": "monel_k500",
    "Copper": "copper",
    "Copper-nickel": "copper_nickel",
    "Brass": "brass",
    "Tin bronze": "tin_bronze",
    "Nickel 200": "nickel_200",
    "Inconel 600": "inconel_600",
    "Inconel 625": "inconel_625",
    "Hastelloy C-276": "hastelloy_c276",
    "Hastelloy C-22": "hastelloy_c22",
    "SS316L": "stainless_300",   # QuickPipe code
    "CS": "carbon_steel",        # QuickPipe code
}

# Ordered choices for the UI material picker (exempt alloys make "use exempt
# material" an actionable selection, not just advice).
MATERIAL_CHOICES: list[str] = [
    "Carbon steel", "Stainless 316/304", "Monel 400", "Copper", "Copper-nickel",
    "Nickel 200", "Inconel 625", "Inconel 600", "Brass", "Tin bronze",
    "Hastelloy C-276",
]

# Families exempt for oxygen-impact at any pressure within Doc 13 scope.
_EFFECTIVELY_UNLIMITED = 20.68  # MPa — Cu/Ni/Monel etc. (≥ scope max)


def _exemption(material: str, wall_mm: float) -> tuple[float | None, str]:
    """Return (exemption_pressure_MPa_abs | None, status).

    status ∈ {"ok", "thin", "none"}:
      * "ok"   — exempt family, thickness met → exemption pressure returned
      * "thin" — exempt family but wall below the minimum thickness → no exemption
      * "none" — not an exempt family (e.g. carbon steel) → no exemption
    """
    fam = _MATERIAL_FAMILY.get(material, material)
    tiers = EIGA_EXEMPTION.get(fam)
    if not tiers:
        return (None, "none")
    applicable = [(t, p) for (t, p) in tiers if wall_mm >= t]
    if not applicable:
        return (None, "thin")
    return (max(applicable, key=lambda tp: tp[1])[1], "ok")


@dataclass
class EigaVelocity:
    v_actual: float                  # m/s (superficial, min cross-section)
    v_limit: float | None            # m/s, None when not velocity-limited
    site: str                        # "impingement" | "non-impingement"
    limited: bool                    # a particle-impact velocity limit applies
    exempt: bool                     # material/conditions exempt the line
    ok: bool | None                  # v_actual ≤ v_limit (None if not limited)
    margin: float | None             # v_limit / v_actual
    basis: str                       # one-line explanation of the governing rule
    notes: list[str] = field(default_factory=list)


def eiga_velocity_assessment(P_bar_abs: float, v_actual: float, material: str,
                             wall_mm: float, *, o2_vol_frac: float = 1.0,
                             impingement: bool = True, relief: bool = False,
                             choked: bool = False) -> EigaVelocity:
    """Assess a gaseous-oxygen line against the EIGA Doc 13 velocity model.

    ``relief`` flags a relief transient (PSV lifting): per §4.4.1 the curve is
    *not* a velocity to reduce — it selects the tail-pipe material at the rated
    relief flow. ``choked`` notes that the velocity is the sonic discharge value.
    """
    P_mpa = P_bar_abs / 10.0          # 1 bar = 0.1 MPa
    site = "impingement" if impingement else "non-impingement"
    notes: list[str] = []

    # Low-purity exemption (§4.3.2.2): < 35 vol% O₂, ferrous/non-ferrous exempt.
    if o2_vol_frac <= 0.35:
        return EigaVelocity(
            v_actual, None, site, False, True, None, None,
            f"O₂ {o2_vol_frac*100:.0f} vol% ≤ 35 % → exempt from velocity limits (§4.3.2.2).",
            notes)

    exP, status = _exemption(material, wall_mm)
    if exP is not None and P_mpa <= exP:
        cap = "no upper limit within Doc 13 scope" if exP >= _EFFECTIVELY_UNLIMITED \
            else f"≤ exemption {exP*10:.1f} bara"
        return EigaVelocity(
            v_actual, None, site, False, True, None, None,
            f"{material} at {P_bar_abs:.1f} bara {cap} (wall {wall_mm:.2f} mm) → "
            f"burn-resistant, no EIGA velocity limit (§4.3.1.2). Other ignition "
            f"mechanisms (App. D) still apply.",
            notes)

    # Velocity-limited: explain why the exemption did not hold.
    if status == "thin":
        notes.append(f"Wall {wall_mm:.2f} mm below the {_min_thickness(material):.2f} mm "
                     f"minimum → exemption lost; treated as flammable (§4.3.1.2).")
        fam = _MATERIAL_FAMILY.get(material, material)
        tiers = EIGA_EXEMPTION.get(fam) or []
        if tiers:
            t0, p0 = min(tiers, key=lambda tp: tp[0])      # thinnest exemption tier
            if P_mpa <= p0:
                notes.append(f"Thicken to ≥{t0:.2f} mm to regain the exemption "
                             f"({material} burn-resistant ≤{p0*10:.1f} bara) — then no "
                             f"velocity limit applies.")
    elif status == "none":
        notes.append(f"{material} is not an exempt alloy → always velocity-limited.")
    elif exP is not None and P_mpa > exP:
        notes.append(f"Above {material} exemption pressure ({exP*10:.1f} bara) → "
                     f"velocity curve applies (§4.3.1.2).")

    v_limit = (impingement_velocity_limit if impingement
               else nonimpingement_velocity_limit)(P_mpa)
    ok = v_actual <= v_limit

    if choked:
        notes.append("Flow is choked — the discharge reaches local sonic velocity "
                     "(Mach 1); the value shown is the governing exit velocity "
                     "(§3.2.20, minimum cross-section).")
    if not ok and relief:
        notes.append("§4.4.1: relief flow is a transient, outside the design-velocity "
                     "basis — at the rated relief flow the curve selects the tail-pipe "
                     "MATERIAL (exempt: Monel / Cu / Ni for ≥8 D downstream of the "
                     "valve, §5.2.3.5), it is not a velocity to reduce.")
    if P_mpa < _LOW_P_MPA:
        notes.append("P < 0.21 MPa abs: §4.4.2 permits CS / thin-wall SS without a "
                     "velocity limit on a case-by-case basis with risk assessment.")
    if P_mpa > _SCOPE_MAX_MPA:
        notes.append("P > 21 MPa abs is outside Doc 13 scope (§2).")

    margin = (v_limit / v_actual) if v_actual > 0 else None
    fig = "2" if impingement else "3"
    basis = (f"{site.capitalize()} curve at {P_bar_abs:.1f} bara → "
             f"v_limit {v_limit:.1f} m/s (Fig {fig}).")
    return EigaVelocity(v_actual, v_limit, site, True, False, ok, margin, basis, notes)


def _min_thickness(material: str) -> float:
    """Smallest tabulated minimum thickness for the material's family (mm)."""
    fam = _MATERIAL_FAMILY.get(material, material)
    tiers = EIGA_EXEMPTION.get(fam) or [(0.0, 0.0)]
    return min(t for t, _ in tiers)


# ── Supporting checks for near-atmospheric vent lines (§4.4.2 design factors) ──
@dataclass
class Check:
    name: str
    value: str          # formatted value for display
    severity: str       # "ok" | "warn" | "alarm" | "info"
    note: str


def speed_of_sound(MW_mix_gmol: float, T_C: float, gamma: float = _GAMMA) -> float:
    """Ideal-gas speed of sound (m/s)."""
    M = MW_mix_gmol / 1000.0
    return (gamma * _R * (T_C + 273.15) / M) ** 0.5


def critical_pressure_ratio(gamma: float = _GAMMA) -> float:
    """Choked-flow critical pressure ratio P*/P0 (0.528 for γ=1.4)."""
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def vent_exit_velocity(W_kgs: float, P_line_bar_abs: float, T_C: float, MW_gmol: float,
                       id_m: float, discharge_bar_abs: float = _ATM_BAR,
                       gamma: float = _GAMMA) -> tuple[float, float, bool, float]:
    """Governing (discharge-end) velocity for a gas venting to atmosphere, with
    choked-flow recognition. Returns (v_exit_ms, mach_exit, choked, T_exit_C).

    For a compressible vent the velocity peaks at the low-pressure exit and is
    bounded by the local sonic velocity — the EIGA velocity (§3.2.20, actual
    volumetric flow / minimum cross-section) is evaluated there, not at the
    higher-pressure inlet. Returns sonic (Mach 1) when the line→discharge
    pressure ratio chokes, or when a constant-area subsonic solution would
    exceed sonic (the pipe cannot pass the flow subsonically)."""
    M = MW_gmol / 1000.0
    T0 = T_C + 273.15
    area = pi * id_m * id_m / 4.0
    if area <= 0 or M <= 0 or W_kgs <= 0 or P_line_bar_abs <= 0:
        return 0.0, 0.0, False, T_C

    def _sonic():
        T_e = T0 * 2.0 / (gamma + 1.0)               # sonic static temperature
        return (gamma * _R * T_e / M) ** 0.5, 1.0, True, T_e - 273.15

    if discharge_bar_abs / P_line_bar_abs <= critical_pressure_ratio(gamma):
        return _sonic()
    # Subsonic: isentropic expansion to the discharge pressure, ideal-gas density.
    T_e = T0 * (discharge_bar_abs / P_line_bar_abs) ** ((gamma - 1.0) / gamma)
    rho_e = discharge_bar_abs * 1e5 * M / (_R * T_e)
    v = W_kgs / (rho_e * area)
    a_e = (gamma * _R * T_e / M) ** 0.5
    if v >= a_e:                                      # area too small → chokes
        return _sonic()
    return v, v / a_e, False, T_e - 273.15


def carucci_mueller_pwl(W_kgs: float, P1_bar_abs: float, P2_bar_abs: float,
                        T_C: float, MW_gmol: float) -> float:
    """Sound power level (dB re 1 pW) generated by a gas pressure let-down, per
    Carucci & Mueller (1982) — the basis of the EI AVIFG / NORSOK L-002
    acoustic-induced-vibration screen.

    Evaluated in the original units (mass flow lb/s, temperature °R) with the
    published constant 126.1 so the result matches the cited source. ``P1`` is
    the upstream (set / source) pressure and ``P2`` the discharge pressure across
    the noise source (the valve/PRV), both absolute.
    """
    dP = P1_bar_abs - P2_bar_abs
    if W_kgs <= 0 or P1_bar_abs <= 0 or dP <= 0 or MW_gmol <= 0:
        return 0.0
    W_lbs = W_kgs * _LB_PER_KG
    T_R = (T_C + 273.15) * 1.8
    ratio = min(dP / P1_bar_abs, 1.0)          # ΔP/P1, capped at the sonic let-down
    return 10.0 * log10((ratio ** 3.6) * (W_lbs ** 2 * T_R / MW_gmol) ** 1.2) + 126.1


def noise_check(pwl: float, Di_mm: float, wall_mm: float, *, mach: float | None = None,
                warn: float = 155.0, alarm: float = 161.0) -> Check:
    """Acoustic-fatigue (AIV) screen from the sound power level + downstream D/t.

    Bands follow Carucci & Mueller / EI AVIFG practice: < 155 dB low likelihood,
    155–161 dB review/mitigate, ≥ 161 dB high. Large downstream D/t raises
    susceptibility — surfaced in the note for the full EI likelihood-of-failure.
    """
    if pwl <= 0:
        return Check("Noise / AIV", "n/a", "info",
                     "No pressure let-down specified — set the source/set pressure "
                     "to estimate the sound power level.")
    sev = "alarm" if pwl >= alarm else "warn" if pwl >= warn else "ok"
    dt = (Di_mm / wall_mm) if (Di_mm > 0 and wall_mm > 0) else 0.0
    bits = [f"Carucci & Mueller PWL. AIV screen (EI AVIFG / NORSOK L-002): "
            f"<{warn:g} low · {warn:g}–{alarm:g} review · ≥{alarm:g} high."]
    if dt:
        bits.append(f"Downstream D/t = {dt:.0f} (higher → more susceptible; "
                    f"run the full EI LOF for borderline results).")
    if mach is not None:
        bits.append(f"Mach {mach:.2f}.")
    bits.append("Radiated / occupational noise → API 521 + vent silencer (EIGA §5.4.8.1).")
    return Check("Noise / AIV", f"PWL {pwl:.0f} dB", sev, " ".join(bits))


def rho_v2_check(rho_mix: float, v_actual: float,
                 *, warn: float = 5000.0, alarm: float = 20000.0) -> Check:
    """Momentum flux ρv² (Pa) → flow-induced-vibration screen (EI AVIFG).

    Indicative bands only; a full likelihood-of-failure assessment per the
    Energy Institute guidelines should follow for borderline/high results.
    """
    val = rho_mix * v_actual ** 2
    sev = "alarm" if val >= alarm else "warn" if val >= warn else "ok"
    note = ("Indicative FIV screen (EI AVIFG). High ρv² → check small-bore "
            "connections, supports and welds; do the full LOF assessment.")
    return Check("Vibration (ρv²)", f"{val:,.0f} Pa", sev, note)


def backpressure_check(dp_total_bar: float, ref_bar: float, *,
                       limit_frac: float = 0.10, service: str = "") -> Check:
    """Built-up back-pressure as a fraction of a reference (set/source) pressure.

    For a PSV tail-pipe the reference is the relief set pressure; the default 10 %
    limit suits a conventional spring PSV (raise to ~0.30–0.50 for balanced-bellows
    or pilot-operated). For a continuous vent the reference is the source pressure.
    """
    frac = (dp_total_bar / ref_bar) if ref_bar > 0 else 0.0
    sev = "alarm" if frac > limit_frac else "warn" if frac > 0.8 * limit_frac else "ok"
    note = (f"Built-up back-pressure {frac*100:.1f} % of {ref_bar:.2f} bar "
            f"(limit {limit_frac*100:.0f} %). {service}".strip())
    return Check("Back-pressure", f"{dp_total_bar:.3f} bar ({frac*100:.1f} %)", sev, note)


def particle_check(x_gas: float, T_C: float) -> Check:
    """Two-phase / particle presence — particle impact is the key impingement
    ignition mechanism (§4.2.2, §5.2.1). Cold vents add ice/condensate particles."""
    if x_gas < 1.0:
        return Check("Two-phase / particles", f"x_gas = {x_gas:.3f}", "alarm",
                     "Liquid/condensate present → particle-impact ignition risk at "
                     "bends; halve allowable velocity at impingement sites, add "
                     "filtration (≤150 µm, §4.4.2) and avoid dead-ends (§5.2.3.8).")
    cold = T_C < 0.0
    return Check("Two-phase / particles", "dry gas" + (" (cold)" if cold else ""),
                 "warn" if cold else "ok",
                 ("Sub-zero vent: watch ice/CO₂ particles from moisture (§4.4.2)."
                  if cold else "Dry, single-phase; keep system oxygen-clean (§6)."))


# ── Ignition-mechanism checklist (Appendix D), condition-driven ───────────────
def ignition_mechanisms(P_bar_abs: float, velocity_ok: bool | None,
                        impingement: bool) -> list[Check]:
    """Advisory App. D checklist flagged by the conditions present in this line."""
    out: list[Check] = []
    pi_active = impingement and (velocity_ok is False)
    out.append(Check(
        "Particle impact", "active" if pi_active else "screened",
        "alarm" if pi_active else "info",
        "Conditions: particulates + high velocity + impingement site (App. D). "
        "Controlled by the velocity curve + cleanliness (§6)."))
    adiabatic = P_bar_abs > 10.0
    out.append(Check(
        "Adiabatic compression", "review" if adiabatic else "low",
        "warn" if adiabatic else "info",
        "Rapid pressurisation of dead-ends / fast-opening valves; low-AIT "
        "non-metals can ignite above ~10 bar (App. D). Slow-open valves."))
    out.append(Check(
        "Resonance (dead-end tees)", "review", "info",
        "Short dead-end tees can resonate under high cross-flow (App. D); "
        "connect branches on top of the main line (§5.2.3.8)."))
    return out


# ── Oxygen service definitions (the dropdown cases) ───────────────────────────
@dataclass
class OxygenService:
    label: str
    impingement: bool            # which EIGA curve governs
    bp_limit_frac: float         # default back-pressure limit vs reference
    mach_warn: float
    mach_alarm: float
    note: str
    relief: bool = False         # relief transient (§4.4.1) vs normal venting


OXYGEN_SERVICES: dict[str, OxygenService] = {
    "O₂ — PSV tail-pipe": OxygenService(
        "O₂ — PSV tail-pipe", impingement=True, bp_limit_frac=0.10,
        mach_warn=0.5, mach_alarm=0.7, relief=True,
        note="Relief discharge downstream of a PRV. §4.4.1: relief velocity is a "
             "transient (not the design-velocity basis) — at the rated relief flow "
             "the impingement curve (§5.2.3.4) selects the tail-pipe MATERIAL, not a "
             "velocity to reduce. Model the line at the tail-pipe pressure; enter the "
             "relief SET pressure above for noise / back-pressure."),
    "O₂ — continuous vent": OxygenService(
        "O₂ — continuous vent", impingement=True, bp_limit_frac=0.10,
        mach_warn=0.3, mach_alarm=0.6, relief=False,
        note="Steady (normal) vent/bleed discharge → design-velocity impingement "
             "curve applies (§4.4.2/§5.2.3.4). Noise/AIV, vibration and particle "
             "impingement at bends govern; SS common for condensation resistance."),
}


def mixture_density(props: dict) -> float:
    """No-slip (homogeneous) density (kg/m³) from a props dict."""
    x = props.get("x_gas", 1.0)
    rg, rl = props.get("rho_g", 0.0), props.get("rho_l", 0.0)
    if 0.0 < x < 1.0 and rg > 0 and rl > 0:
        return 1.0 / (x / rg + (1.0 - x) / rl)
    return rg if x >= 1.0 else rl


def insitu_velocity(props: dict, id_m: float) -> float:
    """Superficial velocity (m/s) from a props dict and pipe bore — lets the UI
    keep velocity, density and noise on one consistent (fresh) basis."""
    rho = mixture_density(props)
    area = pi * id_m * id_m / 4.0
    if rho <= 0 or area <= 0:
        return 0.0
    return props.get("m_total_kgs", 0.0) / (rho * area)


def o2_vol_fraction(props: dict) -> float:
    """O₂ mole (≈ volume) fraction from a props dict (0–1)."""
    comp = props.get("composition") or {}
    for key in ("O₂", "O2", "Oxygen"):
        if key in comp:
            return float(comp[key].get("mol_frac") or 0.0)
    return 0.0


@dataclass
class OxygenAssessment:
    velocity: EigaVelocity
    checks: list[Check]
    service: OxygenService


def assess_oxygen_service(service_key: str, props: dict, v_actual: float, *,
                          P_bar_abs: float, T_C: float, material: str,
                          wall_mm: float, dp_total_bar: float = 0.0,
                          ref_pressure_bar: float = 0.0, id_mm: float = 0.0,
                          discharge_bar_abs: float = _ATM_BAR,
                          impingement: bool | None = None) -> OxygenAssessment:
    """Bundle the EIGA velocity verdict with the six near-atmospheric checks.

    ``props`` is a FlowBench props dict (see fluids.props_at); ``v_actual`` the
    local superficial velocity; ``ref_pressure_bar`` the PSV set / vent source
    pressure (upstream of the noise source and back-pressure reference; falls
    back to the inlet abs pressure); ``discharge_bar_abs`` the pressure the line
    discharges to (atmospheric for a vent); ``id_mm`` the downstream pipe bore
    for the AIV D/t screen. ``impingement`` overrides the service default.
    """
    svc = OXYGEN_SERVICES.get(service_key) or next(iter(OXYGEN_SERVICES.values()))
    imp = svc.impingement if impingement is None else impingement
    o2 = o2_vol_fraction(props)

    x_gas = props.get("x_gas", 1.0)
    rho_mix = mixture_density(props)
    MW = props.get("MW_mix_gmol", 28.97)
    W = props.get("m_total_kgs", 0.0)

    # Governing velocity: a vent accelerates to the low-pressure exit and may
    # choke (sonic) — evaluate there, not at the higher-pressure inlet (§3.2.20).
    # The naive m/(ρA) ``v_actual`` can read supersonic (it ignores choking); the
    # choke-aware exit value is the physical bound, so prefer it when available.
    v_exit, mach_exit, choked, T_exit = vent_exit_velocity(
        W, P_bar_abs, T_C, MW, id_mm / 1000.0, discharge_bar_abs)
    v_gov = v_exit if v_exit > 0 else v_actual

    vel = eiga_velocity_assessment(P_bar_abs, v_gov, material, wall_mm,
                                   o2_vol_frac=o2, impingement=imp,
                                   relief=svc.relief, choked=choked)
    if id_mm > 0 and v_exit > 0:
        vel.notes.insert(0, (
            f"Governing velocity {v_exit:.0f} m/s at the discharge"
            + (f" (choked / sonic, exit ≈ {T_exit:.0f} °C)" if choked
               else f" (Mach {mach_exit:.2f})") + "."))

    # Noise: sound power from the pressure let-down across the source (set/source
    # pressure → discharge), driving the AIV screen in the downstream pipe.
    P1 = ref_pressure_bar or P_bar_abs
    pwl = carucci_mueller_pwl(W, P1, discharge_bar_abs, T_C, MW)

    checks = [
        particle_check(x_gas, T_C),
        rho_v2_check(rho_mix, v_gov),
        noise_check(pwl, id_mm, wall_mm, mach=mach_exit),
        backpressure_check(dp_total_bar, P1, limit_frac=svc.bp_limit_frac,
                           service=svc.label),
    ]
    checks += ignition_mechanisms(P_bar_abs, vel.ok, imp)
    return OxygenAssessment(vel, checks, svc)
