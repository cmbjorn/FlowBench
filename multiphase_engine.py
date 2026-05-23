# multiphase_engine.py
import math
import numpy as np
import CoolProp.CoolProp as CP
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

# ============================================================================
# 1. INDUSTRIAL STANDARDS DATABASE
# ============================================================================
# Inner diameters (m) based on ANSI B36.10 / B36.19 schedule pipe:
#   PN20 / PN25  ≈  Schedule 40   (same bore for all metallic alloys at a given schedule)
#   PN40         ≈  Schedule 80
# Material does NOT affect bore at the same schedule — wall thickness is defined
# by schedule number, not alloy.  All five metallic materials below use this table.
PIPE_DATABASE = {
    "DN20":  {"PN20": 0.0209, "PN25": 0.0209, "PN40": 0.0189},
    "DN25":  {"PN20": 0.0266, "PN25": 0.0266, "PN40": 0.0243},
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
# 3. LIQUID PHASE DATABASE
# ============================================================================

LIQUID_PHASES = [
    # Water-based
    "KOH 30 wt%", "KOH 15 wt%", "Water",
    # Organic solvents
    "Methanol", "Ethanol", "Acetone", "Benzene", "Toluene",
    # Hydrocarbons (liquid)
    "n-Pentane", "n-Hexane", "n-Heptane", "Cyclohexane",
    # LPG / cryogenic
    "Propane (liq.)", "n-Butane (liq.)", "Ammonia (liq.)",
    # Refrigerants (liquid)
    "R-134a (liq.)", "CO₂ (liq.)",
    # Custom
    "Custom",
]

# UI groupings for the liquid selector.
LIQUID_CATEGORIES = {
    "Water-based":          ["KOH 30 wt%", "KOH 15 wt%", "Water"],
    "Organic solvents":     ["Methanol", "Ethanol", "Acetone", "Benzene", "Toluene"],
    "Hydrocarbons (liq.)":  ["n-Pentane", "n-Hexane", "n-Heptane", "Cyclohexane"],
    "LPG / cryogenic":      ["Propane (liq.)", "n-Butane (liq.)", "Ammonia (liq.)"],
    "Refrigerants (liq.)":  ["R-134a (liq.)", "CO₂ (liq.)"],
    "Custom":               ["Custom"],
}

# CoolProp fluid IDs for CoolProp-backed liquids.
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
    "KOH 30 wt%":      0.072,
    "KOH 15 wt%":      0.072,
    "Custom":          0.020,
}

# Aqueous liquids add H₂O vapour to the gas phase via Dalton's Law.
LIQUID_AQUEOUS = {
    "KOH 30 wt%":      True,
    "KOH 15 wt%":      True,
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
    "Custom":           False,
}

# ============================================================================
# 3A. EQUILIBRIUM FLASH (Peng-Robinson EOS via thermo library)
# ============================================================================

# Maps every display-name species to a thermo Chemical identifier.
# None entries are handled specially (Air → N₂/O₂/Ar split).
# KOH and Custom are absent — flash is not applicable for electrolytes/unknowns.
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
             T_C: float, P_bara: float) -> dict:
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

    Returns feasible=False for KOH, Custom, or if thermo is unavailable.
    """
    if not _THERMO_AVAILABLE:
        return {"feasible": False,
                "warnings": ["thermo library not installed — flash unavailable."]}

    if "KOH" in liquid_type:
        return {"feasible": False,
                "warnings": [f"{liquid_type} is an electrolyte — equilibrium flash "
                             "is not applicable. Using specified phase split."]}
    if liquid_type == "Custom":
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
    if q_lye_m3h > 0:
        try:
            cp_id = LIQUID_COOLPROP_ID.get(liquid_type)
            liq_rho = CP.PropsSI("D", "T", T_C + 273.15, "P", P_bara * 1e5,
                                  cp_id) if cp_id else 1000.0
        except Exception:
            liq_rho = 1000.0
        liq_kgh = q_lye_m3h * liq_rho
        # Map liquid display name to feed key (normalise to SPECIES_THERMO_ID keys)
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
        constants, props = ChemicalConstantsPackage.from_IDs(thermo_ids)
        kijs = [[0.0] * len(species) for _ in species]
        eos_kw = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=kijs)
        flasher = FlashVL(
            constants, props,
            liquid=CEOSLiquid(PRMIX, eos_kwargs=eos_kw,
                              HeatCapacityGases=props.HeatCapacityGases),
            gas=CEOSGas(PRMIX, eos_kwargs=eos_kw,
                        HeatCapacityGases=props.HeatCapacityGases),
        )
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
            rho_i   = CP.PropsSI("D", "T", T_K, "P", P_pa, cp_id)
            mu_i    = CP.PropsSI("V", "T", T_K, "P", P_pa, cp_id)
            sig_fb  = LIQUID_SIGMA_FALLBACK.get(sp, 0.020)
            try:
                sigma_i = CP.PropsSI("I", "T", T_K, "Q", 0, cp_id)
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
# 3B. KOH 30 wt% LIQUID PROPERTIES
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

def _coolprop_liquid_by_id(fluid_id, T_K, P_pa, sigma_fallback=0.020):
    """
    Density (kg/m³), dynamic viscosity (Pa·s), surface tension (N/m) for any
    CoolProp-backed liquid.  Surface tension uses the saturation curve at T
    (acceptable engineering approximation for subcooled liquids).
    """
    try:
        rho = CP.PropsSI('D', 'T', T_K, 'P', P_pa, fluid_id)
        mu  = CP.PropsSI('V', 'T', T_K, 'P', P_pa, fluid_id)
    except Exception:
        rho = 800.0
        mu  = 1e-3
    try:
        sigma = CP.PropsSI('surface_tension', 'T', T_K, 'Q', 0, fluid_id)
    except Exception:
        sigma = sigma_fallback
    return rho, mu, sigma


def _coolprop_liquid_properties(liquid_type, T_C, P_bara):
    """Density (kg/m³), dynamic viscosity (Pa·s), surface tension (N/m) via CoolProp.
    Kept for back-compat; now delegates to _coolprop_liquid_by_id."""
    fluid_id = LIQUID_COOLPROP_ID.get(liquid_type, liquid_type)
    sigma_fb = LIQUID_SIGMA_FALLBACK.get(liquid_type, 0.020)
    return _coolprop_liquid_by_id(fluid_id, T_C + 273.15, P_bara * 1e5, sigma_fb)


def _water_properties(T_C, P_bara):
    """Kept for back-compatibility; delegates to the generic helper."""
    return _coolprop_liquid_properties("Water", T_C, P_bara)


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
                return CP.PropsSI('V', 'T', T_K, 'P', P_pa, cid)
            except Exception:
                pass
        return info.get("mu_ref") or 1e-5
    # Species not in GAS_SPECIES — may be a liquid-named species in the gas phase
    cp_id = LIQUID_COOLPROP_ID.get(species)
    if cp_id:
        try:
            return CP.PropsSI('V', 'T', T_K, 'P', P_pa, cp_id)
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
        if sp == "Custom" and custom_gas:
            MW = custom_gas["MW_gmol"] * 1e-3
        else:
            MW = GAS_SPECIES.get(sp, {}).get("MW")
        if MW is None or MW <= 0:
            continue
        mol_flows[sp] = m_kgh / MW
        cid = GAS_SPECIES.get(sp, {}).get("coolprop_id")
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
                if sp == "Custom" and custom_gas:
                    MW = custom_gas["MW_gmol"] * 1e-3
                else:
                    MW = GAS_SPECIES.get(sp, {}).get("MW")
                composition[sp] = {
                    "mol_h": n_mol_h,
                    "kg_h": n_mol_h * MW,
                    "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
                    "coolprop_id": GAS_SPECIES.get(sp, {}).get("coolprop_id"),
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
        MW = custom_gas["MW_gmol"] * 1e-3 if sp == "Custom" and custom_gas else GAS_SPECIES.get(sp, {}).get("MW")
        composition[sp] = {
            "mol_h": n_mol_h,
            "kg_h": n_mol_h * MW,
            "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
            "coolprop_id": GAS_SPECIES.get(sp, {}).get("coolprop_id"),
        }
    return rho_ideal, mu_mix, MW_mix_kgmol, composition


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
    use_coolprop=False, # opt-in: try CoolProp mixture calculations for gas phase
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
    elif liquid_type in LIQUID_COOLPROP_ID:
        fluid_id  = LIQUID_COOLPROP_ID[liquid_type]
        sigma_fb  = LIQUID_SIGMA_FALLBACK.get(liquid_type, 0.020)
        rho_l, mu_l, sigma = _coolprop_liquid_by_id(fluid_id, T_K, P_pa, sigma_fb)
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
            P_sat_H2O = CP.PropsSI('P', 'T', T_K, 'Q', 1, 'Water')
        except Exception:
            P_sat_H2O = 0.1 * P_pa
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
                MW = 18.015e-3
            elif sp == "Custom" and custom_gas:
                MW = custom_gas["MW_gmol"] * 1e-3
            else:
                MW = GAS_SPECIES.get(sp, {}).get("MW", 2.016e-3)
            composition[sp] = {
                "mol_h":    n_mol_h,
                "kg_h":     n_mol_h * MW,
                "mol_frac": n_mol_h / n_total if n_total > 0 else 0.0,
                "coolprop_id": GAS_SPECIES.get(sp, {}).get("coolprop_id"),
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


def calculate_vle_properties(fluid_id, P_bara, x_mass, m_total_kgs):
    """
    Single-component saturated two-phase properties via CoolProp.

    Args:
        fluid_id:      CoolProp fluid name (e.g. "Water", "Propane", "R134a")
        P_bara:        Inlet pressure (bara)
        x_mass:        Mass quality (0 = all liquid, 1 = all vapour)
        m_total_kgs:   Total mass flow (kg/s)

    Returns the same dict shape as calculate_two_phase_properties() so the
    pressure-drop solver and pressure-marching loop are unchanged.
    At each segment the caller re-invokes this function at the updated inlet
    pressure, so T_sat and all phase properties track the saturation curve.
    """
    P_pa   = P_bara * 1e5
    x_mass = max(0.0, min(1.0, x_mass))

    try:
        T_sat = CP.PropsSI('T',  'P', P_pa, 'Q', 0.5, fluid_id)
        rho_l = CP.PropsSI('D',  'P', P_pa, 'Q', 0,   fluid_id)
        rho_g = CP.PropsSI('D',  'P', P_pa, 'Q', 1,   fluid_id)
        mu_l  = CP.PropsSI('V',  'P', P_pa, 'Q', 0,   fluid_id)
        mu_g  = CP.PropsSI('V',  'P', P_pa, 'Q', 1,   fluid_id)
        MW    = CP.PropsSI('M',  'P', P_pa, 'Q', 1,   fluid_id)   # kg/mol
        try:
            sigma = CP.PropsSI('surface_tension', 'T', T_sat, 'Q', 0, fluid_id)
        except Exception:
            sigma = 0.020  # generic fallback
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
                    if vle_fluid is not None:
                        props_seg = calculate_vle_properties(
                            vle_fluid, current_P / 1e5, vle_x_mass, vle_m_total_kgs)
                    else:
                        props_seg = calculate_two_phase_properties(
                            current_P / 1e5, T_C, gas_flows_kgh, liquid_type, q_lye_m3h,
                            custom_gas=custom_gas, custom_liquid=custom_liquid)
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
