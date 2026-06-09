# multiphase_engine.py
import math
import numpy as np
try:
    import CoolProp.CoolProp as CP
    _CP_AVAILABLE = True
except ImportError as _cp_err:
    raise ImportError(
        "\n\nCoolProp is required but could not be imported.\n"
        "Install a pre-built wheel with:\n"
        "    pip install 'CoolProp>=6.4.0'\n"
        "If pip tries to compile from source and fails, install Visual Studio "
        "Build Tools (Windows) or the Xcode CLI (macOS) first.\n"
        f"Original error: {_cp_err}"
    ) from _cp_err
from fluids.two_phase import (
    Beggs_Brill, two_phase_dP, two_phase_dP_dz_gravitational,
    Taitel_Dukler_regime, Mandhane_Gregory_Aziz_regime,
)
from fluids.two_phase_voidage import liquid_gas_voidage
from fluids.friction import friction_factor as _darcy_friction_factor

try:
    from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, FlashVL
    from thermo.eos_mix import PRMIX
    _THERMO_AVAILABLE = True
except ImportError:
    _THERMO_AVAILABLE = False

_g = 9.80665

# ── CoolProp AbstractState cache ────────────────────────────────────────────
# Reusing AbstractState objects is ~75× faster than repeated PropsSI() calls
# because PropsSI rebuilds the EOS state from scratch every time.
# Keyed by fluid_id string; one object per pure-fluid identifier.
_AS_CACHE: dict[str, "CP.AbstractState"] = {}


def _get_as(fluid_id: str) -> "CP.AbstractState":
    """Return (or lazily create) a cached AbstractState for *fluid_id*."""
    if fluid_id not in _AS_CACHE:
        _AS_CACHE[fluid_id] = CP.AbstractState("HEOS", fluid_id)
    return _AS_CACHE[fluid_id]


# ── FlashVL construction cache ───────────────────────────────────────────────
# ChemicalConstantsPackage + FlashVL construction costs ~6 ms per call.
# Cache by sorted tuple of thermo IDs — the flasher only depends on species,
# not on T or P.
_FLASH_CACHE: dict[tuple, object] = {}

# ============================================================================
# 1. INDUSTRIAL STANDARDS DATABASE
# ============================================================================
from standards.piping import (
    PIPE_DATABASE, MATERIAL_ROUGHNESS, LINER_ROUGHNESS, FITTING_Le_over_D
)


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
# a temperature-independent fallback (Pa·s at ~25 °C).
GAS_SPECIES = {
    # ── Common process gases ─────────────────────────────────────────────────
    "H₂":          {"MW": 2.016e-3,   "coolprop_id": "Hydrogen",         "mu_ref": 8.9e-6},
    "O₂":          {"MW": 31.998e-3,  "coolprop_id": "Oxygen",           "mu_ref": 20.4e-6},
    "N₂":          {"MW": 28.014e-3,  "coolprop_id": "Nitrogen",         "mu_ref": 17.8e-6},
    "CO₂":         {"MW": 44.010e-3,  "coolprop_id": "CarbonDioxide",    "mu_ref": 15.0e-6},
    "CO":          {"MW": 28.010e-3,  "coolprop_id": "CarbonMonoxide",   "mu_ref": 17.6e-6},
    "Air":         {"MW": 28.97e-3,   "coolprop_id": None,               "mu_ref": 18.5e-6},
    "Ar":          {"MW": 39.948e-3,  "coolprop_id": "Argon",            "mu_ref": 22.6e-6},
    "He":          {"MW": 4.003e-3,   "coolprop_id": "Helium",           "mu_ref": 19.7e-6},
    "NH₃":         {"MW": 17.031e-3,  "coolprop_id": "Ammonia",          "mu_ref": 10.0e-6},
    "H₂S":         {"MW": 34.081e-3,  "coolprop_id": "HydrogenSulfide",  "mu_ref": 12.5e-6},
    "SO₂":         {"MW": 64.066e-3,  "coolprop_id": "SulfurDioxide",    "mu_ref": 12.5e-6},
    "Cl₂":         {"MW": 70.906e-3,  "coolprop_id": "Chlorine",         "mu_ref": 13.5e-6},
    "N₂O":         {"MW": 44.013e-3,  "coolprop_id": "NitrousOxide",     "mu_ref": 14.5e-6},
    "H₂O (steam)": {"MW": 18.015e-3,  "coolprop_id": "Water",            "mu_ref": 9.6e-6},
    # ── Hydrocarbons ─────────────────────────────────────────────────────────
    "CH₄":         {"MW": 16.043e-3,  "coolprop_id": "Methane",          "mu_ref": 11.1e-6},
    "C₂H₆":        {"MW": 30.069e-3,  "coolprop_id": "Ethane",           "mu_ref": 9.0e-6},
    "C₃H₈":        {"MW": 44.096e-3,  "coolprop_id": "Propane",          "mu_ref": 8.0e-6},
    "n-C₄H₁₀":     {"MW": 58.122e-3,  "coolprop_id": "n-Butane",         "mu_ref": 7.4e-6},
    "i-C₄H₁₀":     {"MW": 58.122e-3,  "coolprop_id": "IsoButane",        "mu_ref": 7.6e-6},
    "C₂H₄":        {"MW": 28.054e-3,  "coolprop_id": "Ethylene",         "mu_ref": 10.0e-6},
    "C₃H₆":        {"MW": 42.080e-3,  "coolprop_id": "Propylene",        "mu_ref": 8.2e-6},
    "n-C₅H₁₂":     {"MW": 72.148e-3,  "coolprop_id": "n-Pentane",        "mu_ref": 6.8e-6},
    # ── Refrigerants (vapour phase) ──────────────────────────────────────────
    "R-134a":      {"MW": 102.032e-3, "coolprop_id": "R134a",            "mu_ref": 11.2e-6},
    "R-22":        {"MW": 86.468e-3,  "coolprop_id": "R22",              "mu_ref": 12.0e-6},
    "R-32":        {"MW": 52.023e-3,  "coolprop_id": "R32",              "mu_ref": 12.9e-6},
    "R-125":       {"MW": 120.022e-3, "coolprop_id": "R125",             "mu_ref": 14.0e-6},
    # ── Custom ───────────────────────────────────────────────────────────────
    "Custom":      {"MW": None,        "coolprop_id": None,               "mu_ref": None},
}

# UI groupings for the gas species selector.
GAS_CATEGORIES = {
    "Common Process": [
        "H₂", "O₂", "N₂", "CO₂", "CO", "Air", "Ar", "He",
        "NH₃", "H₂S", "SO₂", "Cl₂", "N₂O", "H₂O (steam)",
    ],
    "Hydrocarbons": [
        "CH₄", "C₂H₆", "C₃H₈", "n-C₄H₁₀", "i-C₄H₁₀",
        "C₂H₄", "C₃H₆", "n-C₅H₁₂",
    ],
    "Refrigerants": ["R-134a", "R-22", "R-32", "R-125"],
    "Custom": ["Custom"],
}

# ============================================================================
# 3. LIQUID PHASE DATABASE  (CoolProp-backed species only)
#    Special fluids (KOH 30/15 wt%, Custom) are archived in _specials.py.
# ============================================================================

# CoolProp fluid IDs for liquid-phase species.
LIQUID_COOLPROP_ID = {
    "Water":           "Water",
    "Methanol":        "Methanol",
    "Ethanol":         "Ethanol",
    "Acetone":         "Acetone",
    "Benzene":         "Benzene",
    "Toluene":         "Toluene",
    "n-Pentane":       "n-Pentane",
    "n-Hexane":        "n-Hexane",
    "n-Heptane":       "n-Heptane",
    "Cyclohexane":     "CycloHexane",
    "Propane (liq.)":  "Propane",
    "n-Butane (liq.)": "n-Butane",
    "Ammonia (liq.)":  "Ammonia",
    "R-134a (liq.)":   "R134a",
    "CO₂ (liq.)":      "CarbonDioxide",
}

# Surface tension fallbacks (N/m) used when CoolProp cannot provide it.
LIQUID_SIGMA_FALLBACK = {
    "Water":           0.072,
    "Methanol":        0.022,
    "Ethanol":         0.022,
    "Acetone":         0.023,
    "Benzene":         0.029,
    "Toluene":         0.028,
    "n-Pentane":       0.016,
    "n-Hexane":        0.018,
    "n-Heptane":       0.020,
    "Cyclohexane":     0.025,
    "Propane (liq.)":  0.007,
    "n-Butane (liq.)": 0.012,
    "Ammonia (liq.)":  0.021,
    "R-134a (liq.)":   0.008,
    "CO₂ (liq.)":      0.003,
}

# Aqueous species trigger H₂O vapour addition to the gas phase (Dalton's Law).
LIQUID_AQUEOUS = {
    "Water":            True,
    "Methanol":         False,
    "Ethanol":          False,
    "Acetone":          False,
    "Benzene":          False,
    "Toluene":          False,
    "n-Pentane":        False,
    "n-Hexane":         False,
    "n-Heptane":        False,
    "Cyclohexane":      False,
    "Propane (liq.)":   False,
    "n-Butane (liq.)":  False,
    "Ammonia (liq.)":   False,
    "R-134a (liq.)":    False,
    "CO₂ (liq.)":       False,
}

# Liquids handled by built-in correlations (not CoolProp).
# The UI pre-computes properties and passes them as custom_liquid;
# the engine uses this set to bypass liquid_mixture_props for these species.
LIQUID_BUILTIN = frozenset({"KOH solution"})


def koh_properties(T_C: float, conc_wt_pct: float) -> tuple:
    """
    Physical properties of aqueous KOH solution.

    Args:
        T_C:           Temperature (°C), valid 10–90 °C.
        conc_wt_pct:   KOH concentration (wt%), valid 0–40 wt%.

    Returns:
        (rho_kg_m3, mu_Pa_s, sigma_N_m)

    Correlations:
        Density   — fit to ICT / Perry's tabulated data at 25 °C + linear T correction.
                    Accuracy ≈ ±1 %, 0–40 wt%, 10–90 °C.
        Viscosity — Vogel equation for water × exponential concentration factor
                    calibrated to alkaline-electrolyzer reference values
                    (Gilliam et al. 2007, Int. J. Hydrogen Energy 32:359).
                    Accuracy ≈ ±15 % — adequate for hydraulic sizing.
        Surface tension — water σ(T) + small KOH correction (~1 % per 10 wt%).
    """
    import math
    w  = max(0.0, min(40.0, conc_wt_pct))
    T  = max(10.0, min(90.0, T_C))
    TK = T + 273.15

    # Density (kg/m³)
    rho_25 = 997.0 + 7.725 * w + 0.0638 * w ** 2
    rho    = max(900.0, rho_25 - 0.45 * (T - 25.0))

    # Dynamic viscosity (Pa·s) — Vogel water × KOH concentration factor
    mu_w   = math.exp(-3.7188 + 578.919 / (TK - 137.546)) * 1e-3
    a_conc = 0.0388 - 7.45e-5 * (T - 25.0)
    mu     = max(1e-4, mu_w * math.exp(a_conc * w))

    # Surface tension (N/m)
    sigma_w = max(0.040, 0.0728 - 1.66e-4 * (T - 20.0))
    sigma   = sigma_w * (1.0 + 0.001 * w)

    return rho, mu, sigma


# ============================================================================
# 3A. EQUILIBRIUM FLASH (Peng-Robinson EOS via thermo library)
# ============================================================================

# Maps every display-name species to a thermo Chemical identifier.
# None entries are handled specially (Air → N₂/O₂/Ar split).
# Custom is absent — flash requires EOS data that custom inputs don't provide.
SPECIES_THERMO_ID = {
    # Gas species
    "H₂":           "hydrogen",
    "O₂":           "oxygen",
    "N₂":           "nitrogen",
    "CO₂":          "carbon dioxide",
    "CO":           "carbon monoxide",
    "Air":          None,           # expanded to N₂/O₂/Ar below
    "Ar":           "argon",
    "He":           "helium",
    "NH₃":          "ammonia",
    "H₂S":          "hydrogen sulfide",
    "SO₂":          "sulfur dioxide",
    "N₂O":          "nitrous oxide",
    "Cl₂":          "chlorine",
    "H₂O (steam)":  "water",
    "CH₄":          "methane",
    "C₂H₆":         "ethane",
    "C₃H₈":         "propane",
    "n-C₄H₁₀":      "n-butane",
    "i-C₄H₁₀":      "isobutane",
    "C₂H₄":         "ethylene",
    "C₃H₆":         "propylene",
    "n-C₅H₁₂":      "n-pentane",
    "R-134a":        "R134a",
    "R-22":          "chlorodifluoromethane",
    "R-32":          "difluoromethane",
    "R-125":         "pentafluoroethane",
    # Liquid species
    "Water":         "water",
    "Methanol":      "methanol",
    "Ethanol":       "ethanol",
    "Acetone":       "acetone",
    "Benzene":       "benzene",
    "Toluene":       "toluene",
    "n-Pentane":     "n-pentane",
    "n-Hexane":      "n-hexane",
    "n-Heptane":     "n-heptane",
    "Cyclohexane":   "cyclohexane",
    "Propane (liq.)":   "propane",
    "n-Butane (liq.)":  "n-butane",
    "Ammonia (liq.)":   "ammonia",
    "R-134a (liq.)":    "R134a",
    "CO₂ (liq.)":       "carbon dioxide",
}

# Air dry-air molar composition (mol/mol) — ICAO standard atmosphere
_AIR_MOL_FRACS = {"N₂": 0.7809, "O₂": 0.2095, "Ar": 0.0093}
_AIR_MW_KG_MOL = 28.966e-3  # kg/mol


def _expand_air(feed_kgh: dict) -> dict:
    """Replace 'Air' in feed_kgh with equivalent N₂/O₂/Ar flows."""
    if "Air" not in feed_kgh:
        return dict(feed_kgh)
    expanded = {k: v for k, v in feed_kgh.items() if k != "Air"}
    air_mol_h = feed_kgh["Air"] / _AIR_MW_KG_MOL
    for sp, frac in _AIR_MOL_FRACS.items():
        MW = GAS_SPECIES[sp]["MW"]
        expanded[sp] = expanded.get(sp, 0.0) + air_mol_h * frac * MW
    return expanded


def _merge_water_species(feed_kgh: dict) -> dict:
    """Merge 'H₂O (steam)' and 'Water' into a single 'Water' entry."""
    merged = dict(feed_kgh)
    if "H₂O (steam)" in merged:
        merged["Water"] = merged.get("Water", 0.0) + merged.pop("H₂O (steam)")
    return merged


def flash_pt(gas_flows_kgh: dict, liquid_type: str, q_lye_m3h: float,
             T_C: float, P_bara: float,
             liquid_flows_kgh: dict | None = None) -> dict:
    """
    Isothermal two-phase PT flash of the combined gas+liquid feed.

    Uses Peng-Robinson EOS (thermo library). Returns a dict with:
        feasible  : bool
        VF_mol    : molar vapour fraction
        VF_mass   : mass vapour fraction
        gas_phase_kgh    : {species: kg/h} after equilibrium
        liquid_phase_kgh : {species: kg/h} after equilibrium
        feed_kgh         : unified feed composition
        warnings  : list[str]

    Returns feasible=False for Custom liquid or if thermo is unavailable.
    """
    if not _THERMO_AVAILABLE:
        return {"feasible": False,
                "warnings": ["thermo library not installed — flash unavailable."]}

    if liquid_flows_kgh is None and liquid_type == "Custom":
        return {"feasible": False,
                "warnings": ["Custom liquid — no equation-of-state data. "
                             "Using specified phase split."]}
    if "Custom" in gas_flows_kgh:
        return {"feasible": False,
                "warnings": ["Custom gas species — no equation-of-state data. "
                             "Using specified phase split."]}

    # Build unified feed: expand Air, merge water variants, add liquid
    feed = _expand_air(gas_flows_kgh)
    feed = _merge_water_species(feed)

    # Add liquid contribution
    if liquid_flows_kgh:
        for sp, m_kgh in liquid_flows_kgh.items():
            if m_kgh > 0:
                feed[sp] = feed.get(sp, 0.0) + m_kgh
    elif q_lye_m3h > 0:
        try:
            cp_id = LIQUID_COOLPROP_ID.get(liquid_type)
            liq_rho = CP.PropsSI("D", "T", T_C + 273.15, "P", P_bara * 1e5,
                                  cp_id) if cp_id else 1000.0
        except Exception:
            liq_rho = 1000.0
        liq_kgh = q_lye_m3h * liq_rho
        liq_key = liquid_type
        feed[liq_key] = feed.get(liq_key, 0.0) + liq_kgh

    # Check every species has a thermo ID
    missing = [sp for sp in feed if sp not in SPECIES_THERMO_ID]
    if missing:
        return {"feasible": False,
                "warnings": [f"Species not in flash database: {missing}. "
                             "Using specified phase split."]}

    species = [sp for sp in feed if feed[sp] > 0]
    if not species:
        return {"feasible": False, "warnings": ["Zero total feed."]}

    thermo_ids = [SPECIES_THERMO_ID[sp] for sp in species]

    # MW for each species — prefer GAS_SPECIES table, fallback to thermo Chemical
    MWs = {}
    for sp in species:
        if sp in GAS_SPECIES and GAS_SPECIES[sp]["MW"]:
            MWs[sp] = GAS_SPECIES[sp]["MW"]
        elif sp in LIQUID_COOLPROP_ID:
            try:
                MWs[sp] = CP.PropsSI("M", "T", T_C + 273.15, "P", P_bara * 1e5,
                                      LIQUID_COOLPROP_ID[sp])
            except Exception:
                MWs[sp] = 18e-3
        else:
            MWs[sp] = 18e-3  # fallback (water)

    mol_flows = {sp: feed[sp] / MWs[sp] for sp in species}  # mol/h
    n_total = sum(mol_flows.values())
    if n_total <= 0:
        return {"feasible": False, "warnings": ["Zero total molar feed."]}
    zs = [mol_flows[sp] / n_total for sp in species]

    try:
        _flash_key = tuple(sorted(zip(thermo_ids, species)))
        if _flash_key not in _FLASH_CACHE:
            constants, _fprops = ChemicalConstantsPackage.from_IDs(thermo_ids)
            kijs = [[0.0] * len(species) for _ in species]
            eos_kw = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                          omegas=constants.omegas, kijs=kijs)
            _FLASH_CACHE[_flash_key] = FlashVL(
                constants, _fprops,
                liquid=CEOSLiquid(PRMIX, eos_kwargs=eos_kw,
                                  HeatCapacityGases=_fprops.HeatCapacityGases),
                gas=CEOSGas(PRMIX, eos_kwargs=eos_kw,
                            HeatCapacityGases=_fprops.HeatCapacityGases),
            )
        flasher = _FLASH_CACHE[_flash_key]
        res = flasher.flash(T=T_C + 273.15, P=P_bara * 1e5, zs=zs)
    except Exception as exc:
        return {"feasible": False,
                "warnings": [f"Flash solver failed: {str(exc)[:120]}"]}

    VF_mol = float(res.VF) if res.VF is not None else 0.0
    VF_mol = max(0.0, min(1.0, VF_mol))

    gas_phase_kgh = {}
    liquid_phase_kgh = {}

    if VF_mol >= 1.0 - 1e-10:
        gas_phase_kgh = {sp: feed[sp] for sp in species}
    elif VF_mol <= 1e-10:
        liquid_phase_kgh = {sp: feed[sp] for sp in species}
    else:
        for i, sp in enumerate(species):
            mols_gas = n_total * VF_mol * res.gas.zs[i]
            mols_liq = n_total * (1.0 - VF_mol) * res.liquid0.zs[i]
            if mols_gas * MWs[sp] > 1e-6:
                gas_phase_kgh[sp] = mols_gas * MWs[sp]
            if mols_liq * MWs[sp] > 1e-6:
                liquid_phase_kgh[sp] = mols_liq * MWs[sp]

    m_gas  = sum(gas_phase_kgh.values()) if gas_phase_kgh else 0.0
    m_liq  = sum(liquid_phase_kgh.values()) if liquid_phase_kgh else 0.0
    VF_mass = m_gas / (m_gas + m_liq) if (m_gas + m_liq) > 0 else 0.0

    return {
        "feasible":          True,
        "VF_mol":            VF_mol,
        "VF_mass":           VF_mass,
        "gas_phase_kgh":     gas_phase_kgh,
        "liquid_phase_kgh":  liquid_phase_kgh,
        "feed_kgh":          {sp: feed[sp] for sp in species},
        "species":           species,
        "warnings":          [],
    }


def liquid_mixture_props(liquid_phase_kgh: dict, T_K: float, P_pa: float):
    """
    Compute rho (kg/m³), mu (Pa·s), sigma (N/m) for a mixed liquid phase.

    Uses mass-weighted CoolProp properties for each recognisable species.
    Returns (rho, mu, sigma) — falls back to water defaults for unknowns.
    """
    m_total = sum(liquid_phase_kgh.values())
    if m_total <= 0:
        return 1000.0, 1e-3, 0.072

    rho_sum = mu_log_sum = sigma_sum = 0.0
    for sp, m_kgh in liquid_phase_kgh.items():
        w = m_kgh / m_total
        # Determine CoolProp ID: prefer LIQUID_COOLPROP_ID, fallback SPECIES_THERMO_ID
        cp_id = LIQUID_COOLPROP_ID.get(sp) or SPECIES_THERMO_ID.get(sp)
        if cp_id is None:
            rho_sum += w * 1000.0
            mu_log_sum += w * math.log(1e-3)
            sigma_sum += w * 0.072
            continue
        try:
            _as = _get_as(cp_id)
            _as.update(CP.PT_INPUTS, P_pa, T_K)
            rho_i = _as.rhomass()
            mu_i  = _as.viscosity()
            sig_fb = LIQUID_SIGMA_FALLBACK.get(sp, 0.020)
            try:
                _as.update(CP.QT_INPUTS, 0.0, T_K)
                sigma_i = _as.surface_tension()
            except Exception:
                sigma_i = sig_fb
            rho_sum     += w * rho_i
            mu_log_sum  += w * math.log(max(mu_i, 1e-9))
            sigma_sum   += w * sigma_i
        except Exception:
            rho_sum     += w * 1000.0
            mu_log_sum  += w * math.log(1e-3)
            sigma_sum   += w * 0.072

    rho   = max(rho_sum,  100.0)
    mu    = math.exp(mu_log_sum) if mu_log_sum != 0.0 else 1e-3
    sigma = max(sigma_sum, 1e-4)
    return rho, mu, sigma


# ============================================================================
# 3B. LIQUID PROPERTIES VIA COOLPROP AND GAS VISCOSITY HELPER
#     KOH polynomial models archived in _specials.py
# ============================================================================

def _coolprop_liquid_by_id(fluid_id, T_K, P_pa, sigma_fallback=0.020):
    """
    Density (kg/m³), dynamic viscosity (Pa·s), surface tension (N/m) for any
    CoolProp-backed liquid.  Surface tension uses the saturation curve at T
    (acceptable engineering approximation for subcooled liquids).
    """
    try:
        _as = _get_as(fluid_id)
        _as.update(CP.PT_INPUTS, P_pa, T_K)
        rho = _as.rhomass()
        mu  = _as.viscosity()
    except Exception:
        rho = 800.0
        mu  = 1e-3
    try:
        _as = _get_as(fluid_id)
        _as.update(CP.QT_INPUTS, 0.0, T_K)
        sigma = _as.surface_tension()
    except Exception:
        sigma = sigma_fallback
    return rho, mu, sigma



def _species_gas_props(sp, T_K, P_pa, custom_gas=None):
    """Return (MW_kg_mol, coolprop_id) for *sp* in the gas phase.

    Checks GAS_SPECIES first, then LIQUID_COOLPROP_ID — the latter covers
    species that an equilibrium flash has placed in the vapour phase
    (e.g. Benzene, Toluene, n-Hexane).  Returns (None, None) if unknown.
    """
    if sp == "Custom" and custom_gas:
        return custom_gas["MW_gmol"] * 1e-3, None
    info = GAS_SPECIES.get(sp)
    if info:
        return info["MW"], info.get("coolprop_id")
    cp_id = LIQUID_COOLPROP_ID.get(sp)
    if cp_id:
        try:
            MW = CP.PropsSI("M", "T", T_K, "P", P_pa, cp_id)
            return MW, cp_id
        except Exception:
            return None, cp_id
    return None, None


def _get_species_viscosity(species, T_K, P_pa):
    """Dynamic viscosity (Pa·s) for a gas-phase species via CoolProp, with fallbacks.

    Handles species from GAS_SPECIES and also liquid-display-name species that
    may appear in the gas phase after an equilibrium flash (e.g., 'Water').
    """
    info = GAS_SPECIES.get(species)
    if info:
        cid = info.get("coolprop_id")
        if cid:
            try:
                _as = _get_as(cid)
                _as.update(CP.PT_INPUTS, P_pa, T_K)
                return _as.viscosity()
            except Exception:
                pass
        return info.get("mu_ref") or 1e-5
    # Species not in GAS_SPECIES — may be a liquid-named species in the gas phase
    cp_id = LIQUID_COOLPROP_ID.get(species)
    if cp_id:
        try:
            _as = _get_as(cp_id)
            _as.update(CP.PT_INPUTS, P_pa, T_K)
            return _as.viscosity()
        except Exception:
            pass
    return 1e-5  # last-resort default


def _coolprop_mixture_properties(gas_flows_kgh, T_C, P_bara, custom_gas=None):
    """Attempt to compute mixture density (kg/m3), viscosity (Pa·s),
    and MW_mix (kg/mol) using CoolProp for the provided gas flows.
    Falls back to mole-weighted / ideal-gas approximations on failure.
    Returns (rho_g, mu_g, MW_mix_kgmol, composition_dict)
    """
    T_K = T_C + 273.15
    P_pa = P_bara * 1e5

    # Build mole flows and mapping to CoolProp ids
    mol_flows = {}
    cp_parts = []
    for sp, m_kgh in gas_flows_kgh.items():
        MW, cid = _species_gas_props(sp, T_K, P_pa, custom_gas)
        if MW is None or MW <= 0:
            continue
        mol_flows[sp] = m_kgh / MW
        if cid:
            cp_parts.append((cid, mol_flows[sp]))

    n_total = sum(mol_flows.values())
    if n_total <= 0:
        raise ValueError("No valid gas species with positive flow provided")

    # Try CoolProp mixture evaluation if we have at least one coolprop id
    try:
        if cp_parts:
            # Build mixture string like 'Methane[0.5]&Ethane[0.5]'
            mix_str = "&".join(f"{cid}[{frac}]" for cid, frac in (
                (cid, mol / n_total) for cid, mol in cp_parts
            ))
            # Try density (Dmass) and dynamic viscosity (V) via PropsSI
            rho = CP.PropsSI('Dmass', 'T', T_K, 'P', P_pa, mix_str)
            mu = CP.PropsSI('V', 'T', T_K, 'P', P_pa, mix_str)
            # Estimate MW_mix from mass / mol: compute mass flow and mol flow
            m_gas_total_kgh = sum(gas_flows_kgh.get(sp, 0.0) for sp in mol_flows.keys())
            MW_mix_kgmol = m_gas_total_kgh / n_total if n_total > 0 else None
            # Build composition dict
            composition = {}
            for sp, n_mol_h in mol_flows.items():
                MW, cid = _species_gas_props(sp, T_K, P_pa, custom_gas)
                composition[sp] = {
                    "mol_h": n_mol_h,
                    "kg_h": n_mol_h * (MW or 0.0),
                    "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
                    "coolprop_id": cid,
                }
            return rho, mu, MW_mix_kgmol, composition
    except Exception:
        # Fall through to fallback calculations
        pass

    # Fallback: ideal gas density and mole-fraction weighted viscosity
    m_gas_total_kgh = sum(gas_flows_kgh.get(sp, 0.0) for sp in mol_flows.keys())
    MW_mix_kgmol = m_gas_total_kgh / n_total if n_total > 0 else list(GAS_SPECIES.values())[0]["MW"]
    rho_ideal = (P_pa * MW_mix_kgmol) / (8.314 * T_K)
    mu_mix = 0.0
    for sp, n_mol_h in mol_flows.items():
        y = n_mol_h / n_total if n_total > 0 else 0.0
        if sp == "Custom" and custom_gas:
            mu_i = custom_gas.get("mu_upas", 1.2) * 1e-6
        else:
            mu_i = _get_species_viscosity(sp, T_K, P_pa)
        mu_mix += y * mu_i
    mu_mix = max(5e-6, mu_mix)
    composition = {}
    for sp, n_mol_h in mol_flows.items():
        MW, cid = _species_gas_props(sp, T_K, P_pa, custom_gas)
        composition[sp] = {
            "mol_h": n_mol_h,
            "kg_h": n_mol_h * (MW or 0.0),
            "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
            "coolprop_id": cid,
        }
    return rho_ideal, mu_mix, MW_mix_kgmol, composition


# ============================================================================
# 4. THERMODYNAMIC CORE SOLVER
# ============================================================================

def calculate_two_phase_properties(
    P_bara, T_C,
    gas_flows_kgh,           # dict: {species_name: kg/h}  e.g. {"H₂": 8.0, "O₂": 2.0}
    liquid_type,             # str: CoolProp species name or "Custom" (flash override)
    q_lye_m3h,               # volumetric flow [m³/h] — used when liquid_flows_kgh is None
    custom_gas=None,         # {"MW_gmol": float, "mu_upas": float}
    custom_liquid=None,      # {"rho_kgm3": float, "mu_mpas": float, "sigma_mnm": float}
    use_coolprop=False,
    liquid_flows_kgh=None,   # dict: {species: kg/h} — CoolProp multi-species mode
):
    """
    Generic two-phase thermodynamic solver.

    Returns a dict of physical properties ready for Beggs & Brill calculations.
    All existing dict keys are preserved for backward compatibility.
    """
    P_pa = P_bara * 1e5
    T_K  = T_C + 273.15

    # ── Liquid properties ────────────────────────────────────────────────────
    if liquid_flows_kgh:
        _has_builtin = any(sp in LIQUID_BUILTIN for sp in liquid_flows_kgh)
        if _has_builtin and custom_liquid:
            # Built-in liquid (e.g. KOH): UI pre-computes properties and passes
            # them as custom_liquid; mass flow still comes from liquid_flows_kgh.
            cl        = custom_liquid
            rho_l     = cl.get("rho_kgm3", 1000.0)
            mu_l      = cl.get("mu_mpas",  1.0) * 1e-3
            sigma     = cl.get("sigma_mnm", 72.0) * 1e-3
            m_lye_kgh = sum(liquid_flows_kgh.values())
            aqueous   = True  # KOH is aqueous — include water-vapour correction
        else:
            # Primary path: multi-species CoolProp mixture
            rho_l, mu_l, sigma = liquid_mixture_props(liquid_flows_kgh, T_K, P_pa)
            m_lye_kgh = sum(liquid_flows_kgh.values())
            aqueous   = any(LIQUID_AQUEOUS.get(sp, False) for sp in liquid_flows_kgh)
    elif liquid_type in LIQUID_COOLPROP_ID:
        # Backward-compat: single species by name
        fluid_id  = LIQUID_COOLPROP_ID[liquid_type]
        sigma_fb  = LIQUID_SIGMA_FALLBACK.get(liquid_type, 0.020)
        rho_l, mu_l, sigma = _coolprop_liquid_by_id(fluid_id, T_K, P_pa, sigma_fb)
        m_lye_kgh = q_lye_m3h * rho_l
        aqueous   = LIQUID_AQUEOUS.get(liquid_type, False)
    else:
        # Custom: used by the PR-EOS flash-result override path
        cl        = custom_liquid or {}
        rho_l     = cl.get("rho_kgm3", 1000.0)
        mu_l      = cl.get("mu_mpas",  1.0) * 1e-3
        sigma     = cl.get("sigma_mnm", 72.0) * 1e-3
        m_lye_kgh = q_lye_m3h * rho_l
        aqueous   = False

    # ── Dry gas moles (mol/h = kg/h ÷ MW [kg/mol]) ──────────────────────────
    dry_moles = {}
    for sp, m_kgh in gas_flows_kgh.items():
        if sp == "Custom" and custom_gas:
            MW = custom_gas["MW_gmol"] * 1e-3
        else:
            MW = GAS_SPECIES.get(sp, {}).get("MW")
            if MW is None:
                # Flash may put liquid-named species in the gas phase (e.g., "Water")
                cp_id = LIQUID_COOLPROP_ID.get(sp)
                if cp_id:
                    try:
                        MW = CP.PropsSI("M", "T", T_K, "P", P_pa, cp_id)
                    except Exception:
                        pass
        if MW is None or MW <= 0:
            # skip unknown species here; they may be handled by CoolProp
            continue
        dry_moles[sp] = m_kgh / MW
    n_dry = sum(dry_moles.values())

    # ── Water vapour (Dalton's Law, aqueous liquids only) ────────────────────
    P_sat_H2O        = 0.0
    m_H2O_vapor_kgh  = 0.0
    n_H2O            = 0.0
    if aqueous:
        try:
            _as_w = _get_as('Water')
            _as_w.update(CP.QT_INPUTS, 1.0, T_K)
            P_sat_H2O = _as_w.p()
        except Exception:
            P_sat_H2O = 0.1 * P_pa
        # KOH activity correction: water vapour pressure is suppressed by ionic
        # dissociation (K⁺ + OH⁻).  Use same Raoult/dissociation formula as pump_engine.
        _koh_wt = (custom_liquid or {}).get("koh_conc_wt")
        if _koh_wt and _koh_wt > 0:
            _w   = min(40.0, float(_koh_wt)) / 100.0
            _nk  = _w / 0.0561           # mol KOH per kg solution
            _nh  = (1.0 - _w) / 0.018015 # mol H₂O per kg solution
            _xi  = 2.0 * _nk / (2.0 * _nk + _nh)
            P_sat_H2O *= max(0.3, 1.0 - _xi)
        if P_sat_H2O >= P_pa:
            P_sat_H2O = P_pa * 0.95
        y_H2O = P_sat_H2O / P_pa
        n_H2O = n_dry * y_H2O / (1.0 - y_H2O) if y_H2O < 1.0 else 0.0
        m_H2O_vapor_kgh = n_H2O * 18.015e-3  # mol/h × kg/mol = kg/h

    # ── Gas mixture composition and properties (optionally CoolProp) ────────
    # If requested, attempt CoolProp-based mixture eval; otherwise fall back
    # to existing ideal-gas + mole-fraction viscosity approach.
    composition = None
    rho_g = None
    mu_g = None
    MW_mix_kgmol = None

    if use_coolprop:
        try:
            rho_g, mu_g, MW_mix_kgmol, composition = _coolprop_mixture_properties(
                gas_flows_kgh, T_C, P_bara, custom_gas=custom_gas)
        except Exception:
            # Fallback to legacy approach below
            rho_g = None
            mu_g = None
            MW_mix_kgmol = None
            composition = None

    if composition is None:
        # ── Gas mixture composition ─────────────────────────────────────────
        all_moles = dict(dry_moles)
        if n_H2O > 0:
            all_moles["H₂O (vapour)"] = n_H2O
        n_total = sum(all_moles.values())

        composition = {}
        for sp, n_mol_h in all_moles.items():
            if sp == "H₂O (vapour)":
                MW, cid = 18.015e-3, "Water"
            elif sp == "Custom" and custom_gas:
                MW, cid = custom_gas["MW_gmol"] * 1e-3, None
            else:
                MW, cid = _species_gas_props(sp, T_K, P_pa, custom_gas)
                MW = MW or 2.016e-3
            composition[sp] = {
                "mol_h":    n_mol_h,
                "kg_h":     n_mol_h * MW,
                "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
                "coolprop_id": cid,
            }

        m_gas_total_kgh = sum(v["kg_h"] for v in composition.values())
        MW_mix_kgmol    = m_gas_total_kgh / n_total if n_total > 0 else 2.016e-3

        # ── Gas density (ideal gas) ────────────────────────────────────────
        rho_g = (P_pa * MW_mix_kgmol) / (8.314 * T_K)

        # ── Gas mixture viscosity (mole-fraction weighted) ────────────────
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
    m_gas_total_kgh    = sum(v["kg_h"] for v in composition.values())
    m_liq_raw          = m_lye_kgh - m_H2O_vapor_kgh
    # No phantom liquid: single-phase cases get x=1 or x=0 exactly.
    # Only apply a small guard (0.01) when both phases genuinely present to
    # avoid x=NaN when liquid flow is set to an effectively zero value.
    if m_gas_total_kgh > 0 and m_liq_raw <= 0:
        m_liquid_total_kgh = 0.0          # gas-only
    elif m_gas_total_kgh <= 0 and m_liq_raw > 0:
        m_liquid_total_kgh = m_liq_raw    # liquid-only
    else:
        m_liquid_total_kgh = max(0.01, m_liq_raw)  # two-phase; tiny guard
    m_total_kgs = (m_gas_total_kgh + m_liquid_total_kgh) / 3600.0
    _m_sum      = m_gas_total_kgh + m_liquid_total_kgh
    x_gas       = m_gas_total_kgh / _m_sum if _m_sum > 0 else 0.0

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
        "liquid_type":        (" + ".join(liquid_flows_kgh.keys()) if liquid_flows_kgh else liquid_type),
        "m_gas_total_kgh":    m_gas_total_kgh,
        "m_lye_kgh":          m_lye_kgh,
        "m_liquid_total_kgh": m_liquid_total_kgh,
    }


def validate_input_bounds(P_bara, T_C, gas_flows_kgh, liquid_type, q_lye_m3h,
                          liquid_flows_kgh=None):
    """Sanity checks on inputs. Returns (is_valid: bool, warnings: list[str])."""
    warnings = []
    total_gas = sum(gas_flows_kgh.values())

    if P_bara < 1.0 or P_bara > 100.0:
        warnings.append(f"⚠️ System pressure {P_bara:.1f} bara outside typical range [1–100 bara]")
    if T_C < 5.0 or T_C > 95.0:
        warnings.append(f"⚠️ Temperature {T_C:.1f}°C outside validated range [5–95°C]")
    if 0 < total_gas < 0.05:
        warnings.append(f"⚠️ Total gas flow {total_gas:.3f} kg/h is very low")

    if liquid_flows_kgh:
        total_liq_kgh = sum(liquid_flows_kgh.values())
        if 0 < total_liq_kgh < 0.1:
            warnings.append(f"⚠️ Total liquid flow {total_liq_kgh:.3f} kg/h is very low — verify intent")
        is_aqueous = any(LIQUID_AQUEOUS.get(sp, False) for sp in liquid_flows_kgh)
    else:
        if 0 < q_lye_m3h < 1e-4:
            warnings.append(f"⚠️ Liquid volume flow {q_lye_m3h:.4f} m³/h is extremely low — verify intent")
        is_aqueous = LIQUID_AQUEOUS.get(liquid_type, False)

    if is_aqueous:
        try:
            _as_w = _get_as('Water')
            _as_w.update(CP.QT_INPUTS, 1.0, T_C + 273.15)
            P_sat = _as_w.p()
            if P_sat > P_bara * 1e5:
                warnings.append("⚠️ Water saturation pressure exceeds system pressure; flashing likely")
        except Exception:
            pass
    return len(warnings) == 0, warnings


# ============================================================================
# 4A. VLE (SINGLE-COMPONENT SATURATED TWO-PHASE) MODE
# ============================================================================

# Fluids with reliable CoolProp saturation data available for VLE mode.
VLE_FLUIDS = [
    "Water", "Ammonia", "Propane", "n-Butane", "n-Pentane", "n-Hexane",
    "n-Heptane", "Ethanol", "Methanol", "Benzene", "Toluene",
    "CycloHexane", "R134a", "R22", "R32", "R125", "CarbonDioxide",
    "Ethane", "Ethylene", "Acetone",
]

# Human-readable display names → CoolProp IDs for VLE selector.
VLE_FLUID_DISPLAY = {
    "Water (steam/water)":     "Water",
    "Ammonia (NH₃)":           "Ammonia",
    "Propane":                 "Propane",
    "n-Butane":                "n-Butane",
    "n-Pentane":               "n-Pentane",
    "n-Hexane":                "n-Hexane",
    "n-Heptane":               "n-Heptane",
    "Ethanol":                 "Ethanol",
    "Methanol":                "Methanol",
    "Benzene":                 "Benzene",
    "Toluene":                 "Toluene",
    "Cyclohexane":             "CycloHexane",
    "R-134a":                  "R134a",
    "R-22":                    "R22",
    "R-32":                    "R32",
    "R-125":                   "R125",
    "CO₂ (supercritical)":     "CarbonDioxide",
    "Ethane":                  "Ethane",
    "Ethylene":                "Ethylene",
    "Acetone":                 "Acetone",
}


def vle_inlet_enthalpy(fluid_id: str, P_bara: float, x_mass: float) -> float:
    """Specific enthalpy (J/kg) of a pure saturated mixture at inlet conditions."""
    P_pa = P_bara * 1e5
    _as = _get_as(fluid_id)
    _as.update(CP.PQ_INPUTS, P_pa, 0.0)
    h_l = _as.hmass()
    _as.update(CP.PQ_INPUTS, P_pa, 1.0)
    h_v = _as.hmass()
    return h_l + max(0.0, min(1.0, x_mass)) * (h_v - h_l)


def _vle_quality_from_enthalpy(fluid_id: str, P_pa: float, h_spec: float) -> float:
    """Mass quality at pressure P_pa given constant specific enthalpy h_spec (isenthalpic flash)."""
    _as = _get_as(fluid_id)
    _as.update(CP.PQ_INPUTS, P_pa, 0.0)
    h_l = _as.hmass()
    _as.update(CP.PQ_INPUTS, P_pa, 1.0)
    h_v = _as.hmass()
    if h_spec <= h_l:
        return 0.0   # fully condensed — subcooled liquid
    if h_spec >= h_v:
        return 1.0   # fully vaporised — superheated vapour
    return (h_spec - h_l) / (h_v - h_l)


def calculate_vle_properties(fluid_id, P_bara, x_mass, m_total_kgs, h_spec=None):
    """
    Single-component saturated two-phase properties via CoolProp.

    Args:
        fluid_id:      CoolProp fluid name (e.g. "Water", "Propane", "R134a")
        P_bara:        Inlet pressure (bara)
        x_mass:        Mass quality at inlet (0 = liquid, 1 = vapour). Used only
                       when h_spec is None (first call / preview).
        m_total_kgs:   Total mass flow (kg/s)
        h_spec:        Specific enthalpy (J/kg) carried from the inlet. When
                       provided, x_mass is derived via isenthalpic flash so that
                       vapour fraction evolves correctly as pressure drops along
                       the pipe.

    Returns the same dict shape as calculate_two_phase_properties() so the
    pressure-drop solver and pressure-marching loop are unchanged.
    """
    P_pa = P_bara * 1e5
    if h_spec is not None:
        x_mass = _vle_quality_from_enthalpy(fluid_id, P_pa, h_spec)
    else:
        x_mass = max(0.0, min(1.0, x_mass))

    try:
        _as = _get_as(fluid_id)
        _as.update(CP.PQ_INPUTS, P_pa, 0.0)    # saturated liquid
        T_sat = _as.T()
        rho_l = _as.rhomass()
        mu_l  = _as.viscosity()
        try:
            sigma = _as.surface_tension()
        except Exception:
            sigma = 0.020
        _as.update(CP.PQ_INPUTS, P_pa, 1.0)    # saturated vapour
        rho_g = _as.rhomass()
        mu_g  = _as.viscosity()
        MW    = _as.molar_mass()
    except Exception as exc:
        raise ValueError(
            f"CoolProp VLE lookup failed for '{fluid_id}' at {P_bara:.2f} bara: {exc}"
        )

    T_C       = T_sat - 273.15
    m_gas_kgh = x_mass * m_total_kgs * 3600.0
    m_liq_kgh = (1.0 - x_mass) * m_total_kgs * 3600.0

    alpha = 0.0
    if x_mass > 0 and rho_g > 0 and rho_l > 0:
        alpha = (x_mass / rho_g) / (x_mass / rho_g + (1.0 - x_mass) / rho_l)

    composition = {
        fluid_id: {
            "mol_h":       m_gas_kgh / MW if MW > 0 else 0.0,
            "kg_h":        m_gas_kgh,
            "mol_frac":    1.0,
            "coolprop_id": fluid_id,
        }
    }

    return {
        "m_total_kgs":        m_total_kgs,
        "x_gas":              x_mass,
        "alpha":              alpha,
        "rho_l":              rho_l,
        "rho_g":              rho_g,
        "mu_l":               mu_l,
        "mu_g":               mu_g,
        "sigma":              sigma,
        "m_vapor_h2o_kgh":    0.0,
        "P_sat_H2O_pa":       0.0,
        "T_C":                T_C,
        "P_pa":               P_pa,
        "composition":        composition,
        "MW_mix_gmol":        MW * 1000.0,
        "liquid_type":        f"{fluid_id} (VLE)",
        "m_gas_total_kgh":    m_gas_kgh,
        "m_lye_kgh":          m_liq_kgh,
        "m_liquid_total_kgh": m_liq_kgh,
        # VLE-specific bookkeeping
        "vle_fluid":          fluid_id,
        "T_sat_C":            T_C,
    }


# ============================================================================
# 5. PRESSURE DROP SOLVER
# ============================================================================

_SINGLE_PHASE_V_THRESHOLD = 1e-4  # m/s — below this superficial velocity, treat phase as absent


def _single_phase_dp(props, D, roughness, L_eff, angle_rad, phase):
    """
    Darcy-Weisbach pressure drop for single-phase gas or liquid flow.
    Used when the other phase is negligible (superficial velocity < threshold).
    """
    if phase == 'gas':
        rho = props["rho_g"]
        mu  = props["mu_g"]
    else:
        rho = props["rho_l"]
        mu  = props["mu_l"]

    m   = props["m_total_kgs"]
    A   = 0.25 * np.pi * D ** 2
    V   = m / (rho * A) if rho > 0 and A > 0 else 0.0
    Re  = max(1.0, rho * V * D / mu) if mu > 0 else 1e6
    eD  = roughness / D if D > 0 else 0.0

    f        = _darcy_friction_factor(Re=Re, eD=eD)
    dP_fric  = f * (L_eff / D) * 0.5 * rho * V ** 2
    dP_grav  = rho * _g * L_eff * np.sin(angle_rad)
    dP_total = dP_fric + dP_grav

    regime   = "Single-phase gas" if phase == 'gas' else "Single-phase liquid"
    Vsg = V if phase == 'gas'   else 0.0
    Vsl = V if phase == 'liquid' else 0.0

    return {
        "dP_Pa":       dP_total,
        "dP_fric_Pa":  dP_fric,
        "dP_grav_Pa":  dP_grav,
        "dP_accel_Pa": 0.0,
        "regime":      regime,
        "dP_per_dz":   dP_total / L_eff if L_eff > 0 else 0.0,
        "Vsg":         Vsg,
        "Vsl":         Vsl,
        "alpha":       1.0 if phase == 'gas' else 0.0,
    }


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
        "alpha": props.get("alpha", 0.0), "slug_info": None,
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
        Vsg  = (x * m / rhog) / A_cs         if rhog > 0 else 0.0
        Vsl  = ((1.0 - x) * m / rhol) / A_cs if rhol > 0 else 0.0

        # ── Single-phase fallback (one phase absent) ──────────────────────────
        if Vsg < _SINGLE_PHASE_V_THRESHOLD:
            return _single_phase_dp(props, D_inner, roughness, L_eff, angle_rad, 'liquid')
        if Vsl < _SINGLE_PHASE_V_THRESHOLD:
            return _single_phase_dp(props, D_inner, roughness, L_eff, angle_rad, 'gas')

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
        _bb_clamped = False
        if correlation == "Beggs-Brill":
            # Beggs-Brill returns total dP (friction + gravity internally).
            dP_total = Beggs_Brill(
                m=m, x=x, rhol=rhol, rhog=rhog,
                mul=mul, mug=mug, sigma=sigma, P=P_pa,
                D=D_inner, angle=angle_deg, roughness=roughness, L=L_eff,
            )
            # Friction residual: total minus the external gravity term. B&B
            # computes gravity with its own holdup, so this residual absorbs any
            # holdup-model difference — an accepted decomposition approximation.
            _bb_resid   = dP_total - dP_grav_Pa
            _bb_clamped = _bb_resid < -1e-6
            if _bb_clamped:
                # Outside its validity window (high gas velocity in a small bore)
                # B&B can return a total below the gravity head, i.e. a negative
                # friction — which previously got clamped to zero and reported a
                # misleading ΔP ≈ 0. Instead fall back to a friction-only
                # correlation (Friedel) so the result is a real, large, positive
                # number. The decomposition still reconciles (total = fric + grav).
                try:
                    dP_fric_Pa = two_phase_dP(
                        m=m, x=x, rhol=rhol, rhog=rhog,
                        mul=mul, mug=mug, sigma=sigma,
                        D=D_inner, L=L_eff, roughness=roughness, Method="Friedel")
                except Exception:
                    dP_fric_Pa = 0.0
            else:
                dP_fric_Pa = max(0.0, _bb_resid)
            dP_total    = dP_fric_Pa + dP_grav_Pa
            dP_accel_Pa = 0.0
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

        _slug_info = None
        if "slug" in regime or "intermittent" in regime:
            _slug_info = slug_dynamics(Vsl, Vsg, D_inner, rhol, angle_rad, P_pa=props.get("P_pa"))

        # Validity guard: the incompressible two-phase correlations stay
        # meaningful below ~Mach 0.3 in the gas phase (segment-by-segment
        # marching updates density between segments, so moderate compressibility
        # is already captured). Past ~Mach 0.3 a single segment's ΔP is unreliable
        # and the flow is heading toward choking — flag it so callers can warn.
        # The B&B negative-friction clamp is handled by the Friedel fallback
        # above, so it no longer forces an out-of-range flag on its own.
        try:
            _a_gas  = (1.3 * P_pa / rhog) ** 0.5 if (rhog > 0 and P_pa > 0) else float("inf")
            _mach_g = Vsg / _a_gas if _a_gas > 0 else 0.0
        except Exception:
            _mach_g = 0.0
        _out_of_range = bool(_mach_g > 0.3)

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
            "slug_info":   _slug_info,
            "out_of_range": _out_of_range,
            "mach_gas":    _mach_g,
        }

    except Exception as e:
        import warnings as _w
        _w.warn(f"Pressure drop error ({correlation}): {str(e)[:80]}")
        _er = _err_result(f"Error ({correlation[:12]}): {str(e)[:30]}")
        _er["out_of_range"] = True
        return _er


# ============================================================================
# 5A. SLUG FLOW DYNAMICS
# ============================================================================

def slug_dynamics(Vsl: float, Vsg: float, D: float, rho_l: float, theta_rad: float,
                  P_pa: float = None):
    """
    Slug flow characterisation for one pipe segment.

    Correlations:
      Frequency  : Gregory-Scott (1969) — horizontal empirical (validated D≈25–50mm, Vsl≤1 m/s)
      Velocity   : Bendiksen (1984) — generalised for inclination
      Holdup     : Gregory et al. (1978) — slug body liquid holdup
      Length     : Brill & Mukherjee 30D rule-of-thumb
      Pulse/force: Momentum balance at 90° elbow, DLF=2 per ASME B31.3
      Severity   : NORSOK P-001 momentum flux; ASME B31.3 ΔP%; structural resonance frequency

    Returns dict of slug properties, or None if inputs are non-physical.
    """
    import math
    g  = 9.81
    Vm = Vsl + Vsg
    if Vm <= 0.0 or D <= 0.0 or rho_l <= 0.0 or Vsl <= 0.0:
        return None

    # Slug frequency — Gregory-Scott (1969)
    slug_freq_hz = 0.0226 * (Vm / D) * (Vsl / math.sqrt(g * D)) ** 1.2

    # Frequency validity flag (outside correlation's validated range)
    freq_extrapolated = D < 0.025 or Vsl > 1.0

    # Slug translational velocity — Bendiksen (1984)
    Vd     = (0.54 * math.cos(theta_rad) + 0.35 * math.sin(theta_rad)) * math.sqrt(g * D)
    V_slug = 1.2 * Vm + Vd

    # Liquid holdup in slug body — Gregory et al. (1978)
    H_Ls = 1.0 / (1.0 + (Vm / 8.66) ** 1.39)

    # Slug length — Brill & Mukherjee 30D
    L_slug_m = 30.0 * D

    # Pressure pulse at 90° elbow — momentum balance (Pa)
    dP_pulse_Pa  = math.sqrt(2.0) * rho_l * H_Ls * V_slug ** 2
    dP_design_Pa = 2.0 * dP_pulse_Pa          # DLF = 2.0

    # Force on 90° elbow (N)
    A_pipe     = math.pi / 4.0 * D ** 2
    F_elbow_N  = dP_pulse_Pa  * A_pipe
    F_design_N = dP_design_Pa * A_pipe

    # Dynamic pressure reference (lower bound)
    q_Pa = 0.5 * rho_l * V_slug ** 2

    # ── Severity classification ───────────────────────────────────────────────
    # A. Momentum flux — NORSOK P-001 (ρV² design limit 200,000 kg/m/s²)
    momentum_flux = rho_l * V_slug ** 2
    if momentum_flux < 50_000:
        sev_momentum = "Low"
    elif momentum_flux < 150_000:
        sev_momentum = "Moderate"
    else:
        sev_momentum = "Severe"

    # B. ΔP pulse as % of operating pressure — ASME B31.3 occasional-load framework
    if P_pa and P_pa > 0:
        dp_pct = dP_pulse_Pa / P_pa * 100.0
        if dp_pct < 5.0:
            sev_dp = "Low"
        elif dp_pct < 15.0:
            sev_dp = "Moderate"
        else:
            sev_dp = "Severe"
    else:
        dp_pct = None
        sev_dp = "—"

    # C. Frequency vs resonance risk (typical pipe natural frequencies 2–10 Hz)
    if slug_freq_hz < 0.5:
        sev_freq = "Low"
    elif slug_freq_hz < 2.0:
        sev_freq = "Moderate"
    else:
        sev_freq = "High"

    # Overall: worst of A, B, C  (High ranks same as Severe)
    _rank   = {"Low": 0, "Moderate": 1, "Severe": 2, "High": 2, "—": 0}
    severity = max([sev_momentum, sev_dp, sev_freq], key=lambda s: _rank.get(s, 0))

    return {
        "slug_freq_hz":      slug_freq_hz,
        "slug_freq_per_min": slug_freq_hz * 60.0,
        "freq_extrapolated": freq_extrapolated,
        "V_slug_ms":         V_slug,
        "H_Ls":              H_Ls,
        "L_slug_m":          L_slug_m,
        "dP_pulse_kPa":      dP_pulse_Pa  / 1000.0,
        "dP_design_kPa":     dP_design_Pa / 1000.0,
        "F_elbow_N":         F_elbow_N,
        "F_design_N":        F_design_N,
        "q_dyn_kPa":         q_Pa / 1000.0,
        "momentum_flux":     momentum_flux,
        "dp_pct_P":          dp_pct,
        "sev_momentum":      sev_momentum,
        "sev_dp":            sev_dp,
        "sev_freq":          sev_freq,
        "severity":          severity,
    }


# ============================================================================
# 5B. SENSITIVITY ANALYSIS — all correlation × void-fraction combinations
# ============================================================================

def run_sensitivity(
    P_bara, T_C, gas_flows_kgh, liquid_type, q_lye_m3h, segments,
    custom_gas=None, custom_liquid=None,
    liquid_flows_kgh=None,
    # VLE mode — when set, gas_flows_kgh / liquid_type / q_lye_m3h are ignored
    vle_fluid=None, vle_x_mass=None, vle_m_total_kgs=None,
):
    """
    Run all 12 combinations (6 correlations × 2 void-fraction models) and
    return the total ΔP for each using pressure marching.

    Supports both Gas+Liquid and VLE modes.  Pass vle_fluid / vle_x_mass /
    vle_m_total_kgs to activate VLE mode.

    Returns list of dicts — one per combination, ordered as in TWO_PHASE_CORRELATIONS
    (outer) × VOIDAGE_METHODS (inner):
        label        str   e.g. "Beggs-Brill / Homogeneous"
        correlation  str   key in TWO_PHASE_CORRELATIONS
        voidage      str   key in VOIDAGE_METHODS
        total_dp_kpa float   None if convergence failed
        ok           bool
        error        str | None
    """
    _angle_map = {
        "Horizontal":        0.0,
        "Vertical Upflow":   np.pi / 2.0,
        "Vertical Downflow": -np.pi / 2.0,
    }
    # Compute inlet enthalpy once for VLE isenthalpic flash
    _vle_h_spec = None
    if vle_fluid is not None and vle_x_mass is not None:
        try:
            _vle_h_spec = vle_inlet_enthalpy(vle_fluid, P_bara, vle_x_mass)
        except Exception:
            _vle_h_spec = None

    results = []
    for corr in TWO_PHASE_CORRELATIONS:
        for void in VOIDAGE_METHODS:
            try:
                current_P   = P_bara * 1e5
                _h          = _vle_h_spec   # per-run enthalpy, advanced by HX segments
                total_dp    = 0.0
                seg_regimes = []
                for seg in segments:
                    # Heat-exchanger segments: advance enthalpy (VLE) and apply
                    # their fixed pressure drop, then skip the pipe-flow calculation.
                    if seg.get("type") == "heat_exchanger":
                        duty_kw = float(seg.get("duty_kw", 0.0))
                        if vle_fluid is not None and vle_m_total_kgs and duty_kw:
                            _h = (_h or 0.0) + (duty_kw * 1000.0) / vle_m_total_kgs
                        hx_dp = float(seg.get("dp_kpa", 0.0)) * 1000.0
                        total_dp  += hx_dp
                        current_P -= hx_dp
                        current_P  = max(1e4, current_P)
                        continue

                    D_seg     = PIPE_DATABASE[seg["dn"]][seg["pn"]]
                    lined     = seg.get("lined", False)
                    lthk_m    = seg.get("liner_thickness_mm", 1.0) / 1000.0
                    lmat      = seg.get("liner_material", "FEP")
                    D_eff     = D_seg - 2 * lthk_m if lined else D_seg
                    roughness = (LINER_ROUGHNESS[lmat] if lined
                                 else MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
                    if vle_fluid is not None:
                        props_seg = calculate_vle_properties(
                            vle_fluid, current_P / 1e5, vle_x_mass, vle_m_total_kgs,
                            h_spec=_h)
                    else:
                        props_seg = calculate_two_phase_properties(
                            current_P / 1e5, T_C, gas_flows_kgh, liquid_type, q_lye_m3h,
                            custom_gas=custom_gas, custom_liquid=custom_liquid,
                            liquid_flows_kgh=liquid_flows_kgh)
                    angle  = _angle_map[seg["type"]]
                    le_fit = _seg_le_fit(seg, D_eff)
                    L_eff  = seg["length"] + le_fit
                    res    = calculate_segment_pressure_drop(
                        props_seg, D_eff, roughness, L_eff, angle,
                        correlation=corr, voidage_method=void)
                    total_dp  += res["dP_Pa"]
                    current_P -= res["dP_Pa"]
                    current_P  = max(1e4, current_P)
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


# ============================================================================
# VALVE PRESSURE DROP  (IEC 60534 / ISA-75)
# ============================================================================

def calculate_valve_dp(props, Kv_m3h_rated, opening_pct=100.0,
                       characteristic="equal-percentage", rangeability=50.0):
    """
    Two-phase valve pressure drop using the IEC 60534 liquid sizing equation
    with homogeneous mixture density.

    ΔP = (Q / Kv_eff)² · SG_hom    [bar]

    where:
      Q       = total volumetric flow  [m³/h]
      Kv_eff  = effective Kv at the given opening
      SG_hom  = ρ_hom / 999  (relative to water at 15°C)

    Inherent characteristics:
      equal-percentage : Kv_eff = Kv_rated · R^(f−1)  (R = rangeability, default 50)
      linear           : Kv_eff = Kv_rated · f
    where f = opening_pct / 100.

    Valid for non-choked, incompressible-equivalent flow.  For highly
    compressible (gas-dominated) streams this is approximate; use as a
    first estimate and check against manufacturer's Fp correction.

    Args:
        props          : properties dict from calculate_two_phase_properties or calculate_vle_properties
        Kv_m3h_rated   : rated Kv at full-open [m³/h per bar^0.5]
        opening_pct    : valve opening [%], 0–100
        characteristic : "equal-percentage" or "linear"
        rangeability   : R for equal-percentage curve (default 50)

    Returns dict with dP_Pa, Kv_eff, Q_m3h, rho_hom, and ΔP components.
    """
    f = max(0.001, min(1.0, opening_pct / 100.0))
    if characteristic == "equal-percentage":
        Kv_eff = Kv_m3h_rated * (rangeability ** (f - 1.0))
    else:
        Kv_eff = Kv_m3h_rated * f
    Kv_eff = max(Kv_eff, 1e-9)

    x     = props["x_gas"]
    rho_g = props["rho_g"]
    rho_l = props["rho_l"]

    if x <= 0.0:
        rho_hom = rho_l
    elif x >= 1.0:
        rho_hom = rho_g
    else:
        rho_hom = 1.0 / (x / rho_g + (1.0 - x) / rho_l)
    rho_hom = max(rho_hom, 0.01)

    Q_m3h  = props["m_total_kgs"] * 3600.0 / rho_hom
    SG     = rho_hom / 999.0
    dP_bar = (Q_m3h / Kv_eff) ** 2 * SG
    dP_Pa  = dP_bar * 1e5

    return {
        "dP_Pa":       dP_Pa,
        "Kv_eff":      Kv_eff,
        "Q_m3h":       Q_m3h,
        "rho_hom":     rho_hom,
        "dP_fric_Pa":  dP_Pa,
        "dP_grav_Pa":  0.0,
        "dP_accel_Pa": 0.0,
        "regime":      "Valve",
    }


def calculate_valve_kv(props, dp_Pa, opening_pct=100.0,
                       characteristic="equal-percentage", rangeability=50.0):
    """
    Back-calculate required Kv from a target pressure drop (IEC 60534).

    Kv_eff  = Q / sqrt(ΔP_bar / SG_hom)
    Kv_rated = Kv_eff / f_char(opening)

    Returns dict with Kv_eff, Kv_rated, Q_m3h, rho_hom, dP_Pa.
    """
    x     = props["x_gas"]
    rho_g = props["rho_g"]
    rho_l = props["rho_l"]
    if x <= 0.0:
        rho_hom = rho_l
    elif x >= 1.0:
        rho_hom = rho_g
    else:
        rho_hom = 1.0 / (x / rho_g + (1.0 - x) / rho_l)
    rho_hom = max(rho_hom, 0.01)

    Q_m3h  = props["m_total_kgs"] * 3600.0 / rho_hom
    SG     = rho_hom / 999.0
    dp_bar = max(dp_Pa, 1.0) / 1e5
    Kv_eff = Q_m3h / (dp_bar / SG) ** 0.5

    f = max(0.001, min(1.0, opening_pct / 100.0))
    if characteristic == "equal-percentage":
        f_char = rangeability ** (f - 1.0)
    else:
        f_char = f
    Kv_rated = Kv_eff / max(f_char, 1e-9)

    return {
        "dP_Pa":    dp_Pa,
        "Kv_eff":   Kv_eff,
        "Kv_rated": Kv_rated,
        "Q_m3h":    Q_m3h,
        "rho_hom":  rho_hom,
    }


# ============================================================================
# MIXTURE HEAT CAPACITY  (for HX temperature change)
# ============================================================================

def estimate_mixture_cp(props, T_K, P_pa):
    """
    Mass-weighted mixture heat capacity [J/kg/K].

    Liquid Cp: CoolProp lookup via LIQUID_COOLPROP_ID; falls back to 4 200 J/kg/K.
    Gas   Cp:  CoolProp per species in composition dict; falls back to 1 040 J/kg/K (air).

    Returns Cp_mix [J/kg/K].
    """
    x = props.get("x_gas", 0.0)

    # ── Liquid Cp ─────────────────────────────────────────────────────────────
    liq_name = props.get("liquid_type", "")
    # liquid_type may be "Water + Methanol" for mixtures — try first token
    liq_token = liq_name.split(" + ")[0].strip()
    liq_id = LIQUID_COOLPROP_ID.get(liq_token) or LIQUID_COOLPROP_ID.get(liq_name)
    Cp_l = 4200.0
    if liq_id:
        try:
            Cp_l = float(CP.PropsSI("C", "T", T_K, "P", P_pa, liq_id))
        except Exception:
            pass

    # ── Gas Cp ────────────────────────────────────────────────────────────────
    comp = props.get("composition", {})
    m_gas_total = sum(d.get("kg_h", 0.0) for d in comp.values())
    Cp_g = 1040.0
    if m_gas_total > 0.0 and comp:
        Cp_g_weighted = 0.0
        for sp, d in comp.items():
            cp_id = d.get("coolprop_id")
            w_frac = d.get("kg_h", 0.0) / m_gas_total
            if cp_id:
                try:
                    Cp_g_weighted += float(CP.PropsSI("C", "T", T_K, "P", P_pa, cp_id)) * w_frac
                except Exception:
                    Cp_g_weighted += 1040.0 * w_frac
            else:
                Cp_g_weighted += 1040.0 * w_frac
        if Cp_g_weighted > 0.0:
            Cp_g = Cp_g_weighted

    return (1.0 - x) * Cp_l + x * Cp_g
