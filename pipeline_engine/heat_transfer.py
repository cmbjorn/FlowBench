"""
pipeline_engine/heat_transfer.py
=================================
Thermal models for pipeline segments.

Three modes
-----------
isothermal   : no heat exchange — H_out = H_in, T_out = T_in
u_value      : overall heat-transfer coefficient U [W/m²/K]
               Q_loss = U · π · D · ΔL · (T_fluid − T_amb)
surface      : external convection + radiation + solar + rain
               Churchill-Bernstein correlation for forced convection,
               Stefan-Boltzmann radiation, optional solar irradiance,
               optional rain evaporative cooling

All modes return Q_loss_W (positive = heat leaving the fluid).
The caller converts Q to enthalpy: H_out = H_in − Q_loss / ṁ_total_kgs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

# Stefan-Boltzmann constant [W/m²/K⁴]
_SIGMA_SB = 5.670374419e-8
# Air kinematic viscosity at ~20 °C [m²/s] (used as fallback when not supplied)
_AIR_NU_DEFAULT = 15.11e-6
# Air thermal conductivity at ~20 °C [W/m/K]
_AIR_K_DEFAULT = 0.02551
# Air Prandtl number at ~20 °C
_AIR_PR_DEFAULT = 0.7296


# ── Surface environment ────────────────────────────────────────────────────────

@dataclass
class SurfaceConditions:
    """
    Ambient conditions for the external pipe surface.

    Parameters
    ----------
    T_amb_C         : Ambient air temperature [°C]
    wind_ms         : Wind speed [m/s]; 0 → natural convection (not modelled — treated as calm)
    solar_W_m2      : Solar irradiance on pipe projected area [W/m²]; 0 → no solar
    pipe_emissivity : Pipe outer surface emissivity [-]; typical steel ≈ 0.8
    rain             : True → add evaporative cooling (simplified latent heat model)
    air_nu          : Air kinematic viscosity [m²/s] (default 15.11e-6 at 20 °C)
    air_k           : Air thermal conductivity [W/m/K] (default 0.0255)
    air_Pr          : Air Prandtl number (default 0.730)
    """
    T_amb_C:         float = 20.0
    wind_ms:         float = 5.0
    solar_W_m2:      float = 0.0
    pipe_emissivity: float = 0.8
    rain:            bool  = False
    air_nu:          float = _AIR_NU_DEFAULT
    air_k:           float = _AIR_K_DEFAULT
    air_Pr:          float = _AIR_PR_DEFAULT


# ── Isothermal ─────────────────────────────────────────────────────────────────

def q_loss_isothermal() -> float:
    """Returns 0 — no heat exchange."""
    return 0.0


# ── U-value (overall HTC) ─────────────────────────────────────────────────────

def q_loss_u_value(
    U_W_m2K: float,
    D_m: float,
    L_m: float,
    T_fluid_C: float,
    T_amb_C: float,
) -> float:
    """
    Q_loss = U · (π · D · L) · (T_fluid − T_amb)  [W]

    Parameters
    ----------
    U_W_m2K    : Overall heat-transfer coefficient [W/m²/K]
    D_m        : Pipe outer diameter [m]
    L_m        : Segment length [m]
    T_fluid_C  : Fluid temperature at segment inlet [°C]
    T_amb_C    : Ambient temperature [°C]

    Returns positive Q when fluid is hotter than ambient.
    """
    A_ext = math.pi * D_m * L_m  # outer surface area [m²]
    return U_W_m2K * A_ext * (T_fluid_C - T_amb_C)


# ── Churchill-Bernstein external forced convection ────────────────────────────

def _h_ext_forced(
    D_m: float,
    wind_ms: float,
    T_fluid_C: float,
    cond: SurfaceConditions,
) -> float:
    """
    External heat-transfer coefficient via Churchill-Bernstein (1977) [W/m²/K].

    Covers Re·Pr^(1/3) from 0 to 10^7 (cylinder in cross-flow).
    If wind_ms ≈ 0 returns a minimum natural-convection estimate (h=5 W/m²K).
    """
    if wind_ms < 0.1:
        return 5.0  # natural convection minimum estimate

    Re = wind_ms * D_m / cond.air_nu
    Pr = cond.air_Pr

    # Churchill-Bernstein (1977) Nusselt correlation
    Re_sqrt = Re ** 0.5
    Re_5_8  = Re ** (5.0 / 8.0)
    Pr_1_3  = Pr ** (1.0 / 3.0)

    Nu = (0.3
          + (0.62 * Re_sqrt * Pr_1_3)
            / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
          * (1.0 + (Re_5_8 / 282000.0) ** (8.0 / 5.0)) ** 0.625
          )

    return Nu * cond.air_k / D_m


def q_loss_surface(
    D_m: float,
    L_m: float,
    T_fluid_C: float,
    cond: SurfaceConditions,
) -> float:
    """
    Total heat loss from pipe outer surface to environment [W].

    Components:
    1. Forced convection (Churchill-Bernstein)
    2. Radiation (Stefan-Boltzmann, linearised around T_amb)
    3. Solar gain (negative Q — adds heat to fluid)
    4. Rain evaporative cooling (simple latent-heat model)

    Parameters
    ----------
    D_m        : Pipe outer diameter [m]
    L_m        : Segment length [m]
    T_fluid_C  : Fluid temperature [°C] (used as outer wall temperature — thin-wall approx)
    cond       : SurfaceConditions dataclass

    Returns positive Q when fluid is hotter than ambient (net heat loss).
    """
    T_s  = T_fluid_C + 273.15        # K (surface = fluid for thin-wall)
    T_a  = cond.T_amb_C + 273.15     # K (ambient)
    A    = math.pi * D_m * L_m       # outer surface area [m²]

    # 1. Convection
    h_conv = _h_ext_forced(D_m, cond.wind_ms, T_fluid_C, cond)
    Q_conv = h_conv * A * (T_fluid_C - cond.T_amb_C)

    # 2. Radiation (net emission)
    Q_rad = cond.pipe_emissivity * _SIGMA_SB * A * (T_s ** 4 - T_a ** 4)

    # 3. Solar gain (projected area = D · L, not full π·D·L)
    A_proj = D_m * L_m  # projected area perpendicular to sun
    Q_solar = -cond.solar_W_m2 * A_proj  # negative → adds heat (reduces net loss)

    # 4. Rain evaporative cooling (simplified: 40 W/m² per rain event)
    # Based on typical rain-on-pipe evaporation at moderate wind speed
    Q_rain = 40.0 * A if cond.rain else 0.0

    return Q_conv + Q_rad + Q_solar + Q_rain


# ── Unified interface ─────────────────────────────────────────────────────────

ThermalMode = Literal["isothermal", "u_value", "surface"]


@dataclass
class ThermalConfig:
    """
    Thermal configuration for a pipeline segment.

    mode       : "isothermal" | "u_value" | "surface"
    U_W_m2K    : Used when mode="u_value"
    surface    : Used when mode="surface"
    T_amb_C    : Ambient temperature [°C] used for both u_value and surface modes
    """
    mode:    ThermalMode       = "isothermal"
    U_W_m2K: float             = 0.0
    surface: Optional[SurfaceConditions] = field(default=None)
    T_amb_C: float             = 20.0


def segment_heat_loss(
    config: ThermalConfig,
    D_m: float,
    L_m: float,
    T_fluid_C: float,
) -> float:
    """
    Compute Q_loss [W] for a single pipe segment.

    Parameters
    ----------
    config     : ThermalConfig
    D_m        : Pipe outer diameter [m]
    L_m        : Segment length [m]
    T_fluid_C  : Fluid bulk temperature [°C]

    Returns
    -------
    Q_loss_W   : Heat leaving the fluid [W]; negative → fluid gains heat
    """
    if config.mode == "isothermal":
        return q_loss_isothermal()

    elif config.mode == "u_value":
        return q_loss_u_value(
            U_W_m2K=config.U_W_m2K,
            D_m=D_m,
            L_m=L_m,
            T_fluid_C=T_fluid_C,
            T_amb_C=config.T_amb_C,
        )

    elif config.mode == "surface":
        sc = config.surface or SurfaceConditions(T_amb_C=config.T_amb_C)
        return q_loss_surface(
            D_m=D_m,
            L_m=L_m,
            T_fluid_C=T_fluid_C,
            cond=sc,
        )

    else:
        raise ValueError(f"Unknown thermal mode: {config.mode!r}")
