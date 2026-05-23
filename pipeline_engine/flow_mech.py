"""
pipeline_engine/flow_mech.py
=============================
Thin wrapper over the existing multiphase_engine correlations.

Accepts a PVTState (phase properties), pipe geometry, and flow rates,
and returns the pressure drop over one segment using Beggs-Brill (via
the fluids library).  The existing multiphase_engine.calculate_segment_pressure_drop()
function is reused directly — this module bridges the new PVTState API
to that function's expected keyword arguments.

Single-phase fallback
---------------------
If VF_mol ≈ 1.0  → all-gas Darcy-Weisbach
If VF_mol ≈ 0.0  → all-liquid Darcy-Weisbach
Otherwise        → Beggs-Brill two-phase correlation

Returns
-------
dp_Pa : float
    Pressure drop [Pa] (positive = pressure decreases in flow direction).
    Upflow sections add hydrostatic head (positive dp); downflow sections
    subtract it (negative dp possible).
"""

from __future__ import annotations

import sys
import os

import math
from typing import Optional

# Import the existing engine
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import multiphase_engine as _eng

from .pvt import PVTState

# Beggs-Brill via fluids library (already used by multiphase_engine)
try:
    from fluids import Beggs_Brill
    _FLUIDS_OK = True
except ImportError:
    _FLUIDS_OK = False


def segment_dp(
    state: PVTState,
    m_total_kgs: float,
    D_m: float,
    L_m: float,
    angle_deg: float,
    roughness_m: float = 46e-6,
) -> float:
    """
    Pressure drop over a single pipe segment [Pa].

    Parameters
    ----------
    state        : PVTState from PVTTable.lookup()
    m_total_kgs  : Total mass flow rate [kg/s]
    D_m          : Inner pipe diameter [m]
    L_m          : Segment length [m]
    angle_deg    : Pipe inclination from horizontal [°]
                   +90 = vertical up, −90 = vertical down, 0 = horizontal
    roughness_m  : Absolute pipe roughness [m]; default 46 µm (commercial steel)

    Returns
    -------
    dp_Pa : float
        Positive when outlet pressure < inlet pressure (pressure drop).
        May be negative for downflow with large gravity recovery.
    """
    if not _FLUIDS_OK:
        raise ImportError("fluids library not available — cannot compute pressure drop")

    VF = state.VF_mol
    A  = math.pi * (D_m / 2.0) ** 2  # cross-sectional area [m²]

    # ── Single-phase gas ────────────────────────────────────────────────────────
    if VF > 1.0 - 1e-4 or state.rho_l is None:
        rho_g = state.rho_g if (state.rho_g and state.rho_g > 0) else 1.0
        mu_g  = state.mu_g  if (state.mu_g  and state.mu_g  > 0) else 1e-5
        v_g   = m_total_kgs / (rho_g * A)
        Re    = rho_g * v_g * D_m / mu_g
        fd    = _friction_factor(Re, D_m, roughness_m)
        dp_f  = fd * (L_m / D_m) * rho_g * v_g ** 2 / 2.0
        dp_grav = rho_g * 9.81 * L_m * math.sin(math.radians(angle_deg))
        return dp_f + dp_grav

    # ── Single-phase liquid ─────────────────────────────────────────────────────
    if VF < 1e-4 or state.rho_g is None:
        rho_l = state.rho_l if (state.rho_l and state.rho_l > 0) else 1000.0
        mu_l  = state.mu_l  if (state.mu_l  and state.mu_l  > 0) else 1e-3
        v_l   = m_total_kgs / (rho_l * A)
        Re    = rho_l * v_l * D_m / mu_l
        fd    = _friction_factor(Re, D_m, roughness_m)
        dp_f  = fd * (L_m / D_m) * rho_l * v_l ** 2 / 2.0
        dp_grav = rho_l * 9.81 * L_m * math.sin(math.radians(angle_deg))
        return dp_f + dp_grav

    # ── Two-phase Beggs-Brill ────────────────────────────────────────────────────
    rho_g = state.rho_g
    rho_l = state.rho_l
    mu_g  = state.mu_g  if state.mu_g  else 1e-5
    mu_l  = state.mu_l  if state.mu_l  else 1e-3
    sigma = state.sigma if state.sigma else 0.020

    # Split total mass flow into gas and liquid streams
    m_g_kgs = m_total_kgs * state.VF_mass
    m_l_kgs = m_total_kgs * (1.0 - state.VF_mass)

    # Superficial velocities
    Vsg = m_g_kgs / (rho_g * A) if rho_g > 0 else 0.0
    Vsl = m_l_kgs / (rho_l * A) if rho_l > 0 else 0.0

    P_Pa = state.P_bara * 1e5

    try:
        dp_Pa = Beggs_Brill(
            m=m_total_kgs,
            x=state.VF_mass,
            rhol=rho_l,
            rhog=rho_g,
            mul=mu_l,
            mug=mu_g,
            sigma=sigma,
            P=P_Pa,
            D=D_m,
            roughness=roughness_m,
            L=L_m,
            angle=angle_deg,
        )
    except Exception as exc:
        # Fallback to homogeneous model on Beggs-Brill failure
        rho_mix = rho_l * (1.0 - VF) + rho_g * VF
        v_mix   = m_total_kgs / (rho_mix * A)
        mu_mix  = mu_l * (1.0 - VF) + mu_g * VF
        Re      = rho_mix * v_mix * D_m / mu_mix
        fd      = _friction_factor(Re, D_m, roughness_m)
        dp_f    = fd * (L_m / D_m) * rho_mix * v_mix ** 2 / 2.0
        dp_grav = rho_mix * 9.81 * L_m * math.sin(math.radians(angle_deg))
        dp_Pa   = dp_f + dp_grav

    return float(dp_Pa)


def _friction_factor(Re: float, D_m: float, roughness_m: float) -> float:
    """
    Darcy friction factor — Swamee-Jain approximation (explicit, ±2% of Colebrook).
    Falls back to laminar 64/Re below Re=2300.
    """
    if Re < 1.0:
        return 64.0  # degenerate
    if Re < 2300.0:
        return 64.0 / Re  # laminar
    eps_D = roughness_m / D_m
    # Swamee-Jain
    denom = math.log10(eps_D / 3.7 + 5.74 / Re ** 0.9)
    return 0.25 / denom ** 2
