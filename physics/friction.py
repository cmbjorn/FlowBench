"""
Darcy friction factor correlations.

All functions take dimensionless inputs and return the dimensionless
Darcy-Weisbach friction factor f, where ΔP_fric = f · (L/D) · ½ρV².
"""

import math


def churchill_f(Re: float, eps_D: float) -> float:
    """
    Darcy friction factor — Churchill (1977), valid for all Re and roughness.

    Args:
        Re:    Reynolds number (dimensionless)
        eps_D: relative roughness = absolute_roughness / inner_diameter

    Returns:
        Darcy friction factor f (dimensionless)

    Covers laminar (f = 64/Re), transition, and fully turbulent regimes
    with a single continuous formula. No regime switching required.
    """
    if Re < 1.0:
        return 64.0
    if Re < 2300.0:
        return 64.0 / Re
    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * eps_D)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)
