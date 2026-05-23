"""
_specials.py
============
Archived special-fluid models.

NOT imported by the main engine or any UI module.
These are stored here for future reinstatement as an optional add-on.

To re-enable:
  - Import the functions you need into multiphase_engine.py
  - Add the liquid types back to LIQUID_PHASES / LIQUID_CATEGORIES / LIQUID_AQUEOUS
  - Add KOH branches back to calculate_two_phase_properties and flash_pt

Archived models
---------------
  KOH 30 wt% — polynomial correlations, valid 0–100 °C
    ρ = 1295 − 0.3375(T − 20)  [kg/m³]
    μ = Arrhenius, E/R = 1200 K, μ₂₀ = 1.4 mPa·s
    σ = 74 − 1.125(T − 20) mN/m
    References: Yaws' Chemical Properties Handbook; NIST aqueous KOH data

  KOH 15 wt% — polynomial correlations, valid 0–100 °C
    ρ = 1139 − 0.50(T − 20)  [kg/m³]
    μ = Arrhenius, E/R = 1540 K, μ₂₀ = 1.1 mPa·s
    σ ≈ 73 mN/m (nearly flat)
    References: CRC Handbook; fitted to tabulated data

Archived lists
--------------
  LIQUID_PHASES      — full list including KOH entries
  LIQUID_CATEGORIES  — UI grouping dict including KOH
"""

import numpy as np


# ── KOH 30 wt% ────────────────────────────────────────────────────────────────

def koh_density_kgm3(T_C):
    rho_ref_20C = 1295.0
    slope = -0.3375
    return max(1100.0, rho_ref_20C + slope * (T_C - 20.0))


def koh_viscosity_pas(T_C):
    mu_ref_20C = 1.4e-3
    T_ref_K    = 293.15
    E_a_over_R = 1200.0
    mu = mu_ref_20C * np.exp(E_a_over_R * (1.0 / (T_C + 273.15) - 1.0 / T_ref_K))
    return max(2e-4, mu)


def koh_surface_tension_nm(T_C):
    sigma_ref_20C = 0.074
    slope = -0.001125
    return max(0.040, sigma_ref_20C + slope * (T_C - 20.0))


# ── KOH 15 wt% ────────────────────────────────────────────────────────────────

def koh15_density_kgm3(T_C):
    rho_ref_20C = 1139.0
    slope = -0.50
    return max(1050.0, rho_ref_20C + slope * (T_C - 20.0))


def koh15_viscosity_pas(T_C):
    mu_ref_20C = 1.1e-3
    T_ref_K    = 293.15
    E_a_over_R = 1540.0
    mu = mu_ref_20C * np.exp(E_a_over_R * (1.0 / (T_C + 273.15) - 1.0 / T_ref_K))
    return max(1.5e-4, mu)


def koh15_surface_tension_nm(T_C):
    sigma_ref_20C = 0.073
    slope = -0.00014
    return max(0.040, sigma_ref_20C + slope * (T_C - 20.0))


# ── Archived liquid lists ──────────────────────────────────────────────────────

LIQUID_PHASES = [
    "KOH 30 wt%", "KOH 15 wt%", "Water",
    "Methanol", "Ethanol", "Acetone", "Benzene", "Toluene",
    "n-Pentane", "n-Hexane", "n-Heptane", "Cyclohexane",
    "Propane (liq.)", "n-Butane (liq.)", "Ammonia (liq.)",
    "R-134a (liq.)", "CO₂ (liq.)",
    "Custom",
]

LIQUID_CATEGORIES = {
    "Water-based":         ["KOH 30 wt%", "KOH 15 wt%", "Water"],
    "Organic solvents":    ["Methanol", "Ethanol", "Acetone", "Benzene", "Toluene"],
    "Hydrocarbons (liq.)": ["n-Pentane", "n-Hexane", "n-Heptane", "Cyclohexane"],
    "LPG / cryogenic":     ["Propane (liq.)", "n-Butane (liq.)", "Ammonia (liq.)"],
    "Refrigerants (liq.)": ["R-134a (liq.)", "CO₂ (liq.)"],
    "Special":             ["KOH 30 wt%", "KOH 15 wt%", "Custom"],
}
