"""
Pressure relief standards lookup tables.

Sources:
    API 520 Part I — sizing and selection of pressure-relieving devices
    API 526        — flanged steel pressure-relief valves
"""

# ── API 526 effective orifice areas (mm²) ───────────────────────────────────
API526_ORIFICES: dict[str, float] = {
    "D":   71.0,
    "E":  126.5,
    "F":  198.1,
    "G":  324.5,
    "H":  506.4,
    "J":  830.3,
    "K":  1_186.0,
    "L":  1_841.0,
    "M":  2_322.0,
    "N":  2_800.0,
    "P":  4_116.0,
    "Q":  7_129.0,
    "R": 10_323.0,
    "T": 16_774.0,
}

# ── API 520 Table 5 discharge coefficients ───────────────────────────────────
KD_GAS    = 0.975   # gas / vapour / steam
KD_LIQUID = 0.65    # liquid

# ── API 520 §4.7 combination factor (rupture disc upstream) ─────────────────
KC_DISC = 0.9
KC_NONE = 1.0

KD_TWOPHASE = 0.85  # API 520 Table 5, footnote b — two-phase homogeneous nozzle

# ── API 526 standard inlet × outlet flange sizes (NPS, inches) ──────────────
# Source: API 526, 7th Ed., Table 1.
API526_FLANGE_NPS: dict[str, tuple[float, float]] = {
    "D": (1.0,  2.0),
    "E": (1.0,  2.0),
    "F": (1.5,  2.5),
    "G": (1.5,  3.0),
    "H": (2.0,  3.0),
    "J": (3.0,  4.0),
    "K": (3.0,  4.0),
    "L": (4.0,  6.0),
    "M": (4.0,  6.0),
    "N": (4.0,  6.0),
    "P": (6.0,  8.0),
    "Q": (6.0, 10.0),
    "R": (6.0, 10.0),
    "T": (8.0, 10.0),
}  # (inlet NPS in, outlet NPS in)


def flange_nps(orifice_letter: str) -> tuple[float, float] | None:
    """Return (inlet_NPS_in, outlet_NPS_in) for a given API 526 orifice letter, or None."""
    return API526_FLANGE_NPS.get(orifice_letter)
