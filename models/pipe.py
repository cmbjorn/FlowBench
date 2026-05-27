from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentRow:
    """Typed result row for one pipeline segment (pipe, valve, or heat exchanger).

    Construct with named fields; call .to_dict() to get the display-key dict
    expected by the segment DataFrame and downstream consumers.
    """
    seg: str
    type: str
    pipe: str
    id_mm: float
    l_m: float
    l_eq_m: float
    fittings: str
    regime: str
    dp_kPa: float
    p_in_bara: float
    p_out_bara: float
    v_m_ms: float
    v_m_ve: float
    v_sg_ms: float
    v_sl_ms: float
    v_e_ms: float
    dp_fric_kPa: float
    dp_grav_kPa: float
    dp_accel_kPa: float
    material: str
    rho_g: float
    l_eff_m: float
    alpha_void: float
    dp_dz: float
    dp_fric_100m_kPa: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "Seg":             self.seg,
            "Type":            self.type,
            "Pipe":            self.pipe,
            "ID (mm)":         self.id_mm,
            "L (m)":           self.l_m,
            "L_eq (m)":        self.l_eq_m,
            "Fittings":        self.fittings,
            "Regime":          self.regime,
            "ΔP (kPa)":        self.dp_kPa,
            "P_in (bara)":     self.p_in_bara,
            "P_out (bara)":    self.p_out_bara,
            "V_m (m/s)":       self.v_m_ms,
            "V_m/V_e":         self.v_m_ve,
            "V_sg (m/s)":      self.v_sg_ms,
            "V_sl (m/s)":      self.v_sl_ms,
            "V_e (m/s)":       self.v_e_ms,
            "ΔP_fric (kPa)":   self.dp_fric_kPa,
            "ΔP_grav (kPa)":   self.dp_grav_kPa,
            "ΔP_accel (kPa)":  self.dp_accel_kPa,
            "Material":        self.material,
            "ρ_g (kg/m³)":    self.rho_g,
            "L_eff (m)":       self.l_eff_m,
            "α (void)":        self.alpha_void,
            "dP/dz (Pa/m)":    self.dp_dz,
        }
        if self.dp_fric_100m_kPa is not None:
            d["ΔP_fric/100m (kPa)"] = self.dp_fric_100m_kPa
        return d
