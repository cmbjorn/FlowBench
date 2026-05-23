"""
pipeline_engine/solver.py
==========================
Non-isothermal two-phase pipeline segment solver.

Algorithm (marching)
--------------------
For each pipe segment i (from inlet to outlet):
  1. Look up phase properties at (P_i, T_i) via the PVT source.
  2. Compute pressure drop ΔP_i  (Beggs-Brill or single-phase D-W).
  3. Compute heat loss Q_i [W]   (isothermal / U-value / surface model).
  4. Update enthalpy: H_{i+1} = H_i − Q_i / ṁ
  5. Update pressure: P_{i+1} = P_i − ΔP_i / 1e5   (Pa → bara)
  6. Recover T_{i+1} = PVT.T_from_PH(P_{i+1}, H_{i+1})
     → JT cooling + heat loss both captured in the single enthalpy step.

PVT mode toggle
---------------
use_flash=True  (default)
    PVTFlash: full PR-EOS flash at every segment lookup and T_from_PH call.
    Exact condensation, correct latent heat, ~0.2–1 ms per segment.

use_flash=False
    PVTTable: 20×20 (P,T) grid built once at the start, then O(1) bilinear
    interpolation.  ~100–450 ms upfront, then <0.001 ms per lookup.
    Better for sensitivity sweeps; may miss sharp phase boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union

import math

from .pvt import PVTTable, PVTFlash, PVTState
from .heat_transfer import ThermalConfig, segment_heat_loss
from .flow_mech import segment_dp

_PVTSource = Union[PVTFlash, PVTTable]


# ── Pipe geometry ──────────────────────────────────────────────────────────────

@dataclass
class PipeSegment:
    """
    One straight pipe run (or fitting group treated as equivalent length).

    Parameters
    ----------
    length_m    : Physical pipe length [m]
    angle_deg   : Inclination from horizontal [°]; +90=up, -90=down, 0=horizontal
    D_inner_m   : Inner pipe diameter [m]
    D_outer_m   : Outer pipe diameter [m] (for heat transfer area)
    roughness_m : Absolute roughness [m]; default 46 µm (commercial steel)
    label       : Optional descriptive label
    """
    length_m:    float
    angle_deg:   float = 0.0
    D_inner_m:   float = 0.1022   # DN100 default
    D_outer_m:   float = 0.114    # DN100 default
    roughness_m: float = 46e-6
    label:       str   = ""


# ── Segment result ─────────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """Computed state at the OUTLET of one pipe segment."""
    label:     str
    P_in_bara: float
    T_in_C:    float
    P_out_bara:float
    T_out_C:   float
    dp_Pa:     float          # pressure drop [Pa] (positive = loss)
    Q_loss_W:  float          # heat leaving fluid [W]
    dH_Jkg:    float          # specific enthalpy change [J/kg]
    VF_mol_in: float          # vapour fraction at inlet
    VF_mol_out:float          # vapour fraction at outlet
    rho_g_in:  Optional[float]
    rho_l_in:  Optional[float]


# ── Solver configuration ───────────────────────────────────────────────────────

@dataclass
class SolverConfig:
    """
    Inlet conditions, fluid composition, and solver settings.

    Parameters
    ----------
    composition_kgh : {display_name: kg/h}  fluid composition (ratios matter)
    m_total_kgs     : Total mass flow rate [kg/s]
    P_in_bara       : Inlet pressure [bara]
    T_in_C          : Inlet temperature [°C]
    thermal         : ThermalConfig  (isothermal / u_value / surface)
    use_flash       : True  → PVTFlash (exact, ~0.2–1 ms/segment)
                      False → PVTTable (interpolated, fast after build)
    P_table_min     : PVT table lower P bound [bara]  (use_flash=False only)
    P_table_max     : PVT table upper P bound [bara]
    T_table_min     : PVT table lower T bound [°C]
    T_table_max     : PVT table upper T bound [°C]
    n_P, n_T        : PVT table grid size (default 20×20)
    """
    composition_kgh: Dict[str, float]
    m_total_kgs:     float
    P_in_bara:       float
    T_in_C:          float
    thermal:         ThermalConfig = field(default_factory=ThermalConfig)
    use_flash:       bool  = True
    segment_length_m: Optional[float] = None  # auto-subdivide; None = no subdivision
    # Table bounds — only used when use_flash=False
    P_table_min:     Optional[float] = None   # default: P_in * 0.2
    P_table_max:     Optional[float] = None   # default: P_in * 1.1
    T_table_min:     Optional[float] = None   # default: T_in − 80
    T_table_max:     Optional[float] = None   # default: T_in + 50
    n_P:             int = 20
    n_T:             int = 20

    def __post_init__(self) -> None:
        """Build the PVT source once so repeated solve_pipeline calls are cheap."""
        if self.use_flash:
            self._pvt: _PVTSource = PVTFlash(self.composition_kgh)
        else:
            P_lo = self.P_table_min if self.P_table_min is not None else max(0.5, self.P_in_bara * 0.2)
            P_hi = self.P_table_max if self.P_table_max is not None else self.P_in_bara * 1.1
            T_lo = self.T_table_min if self.T_table_min is not None else self.T_in_C - 80.0
            T_hi = self.T_table_max if self.T_table_max is not None else self.T_in_C + 50.0
            self._pvt = PVTTable(
                self.composition_kgh,
                P_min_bara=P_lo, P_max_bara=P_hi,
                T_min_C=T_lo,    T_max_C=T_hi,
                n_P=self.n_P,    n_T=self.n_T,
            )


# ── Solver result ──────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    segments:    List[SegmentResult]
    P_out_bara:  float
    T_out_C:     float
    dp_total_Pa: float
    Q_total_W:   float
    pvt_mode:    str   = "flash"   # "flash" or "table"
    converged:   bool  = True

    @property
    def dp_total_kPa(self) -> float:
        return self.dp_total_Pa / 1000.0

    @property
    def dp_total_bara(self) -> float:
        return self.dp_total_Pa / 1e5


# ── Main solver ────────────────────────────────────────────────────────────────

def solve_pipeline(
    segments: List[PipeSegment],
    config: SolverConfig,
) -> SolverResult:
    """
    March through all pipe segments from inlet to outlet.

    The PVT source (PVTFlash or PVTTable) is built automatically from
    config.composition_kgh and config.use_flash.

    Returns
    -------
    SolverResult
    """
    pvt      = config._pvt
    pvt_mode = "flash" if config.use_flash else "table"
    m        = config.m_total_kgs

    P_cur = config.P_in_bara
    T_cur = config.T_in_C
    H_cur = pvt.lookup(P_cur, T_cur).H_Jkg

    results:  List[SegmentResult] = []
    dp_total = 0.0
    Q_total  = 0.0

    for seg_idx, seg in enumerate(segments):
        # ── Subdivide this segment if requested ───────────────────────────────
        n_steps  = _n_steps(seg.length_m, config.segment_length_m)
        step_len = seg.length_m / n_steps

        seg_P_in    = P_cur
        seg_T_in    = T_cur
        seg_VF_in   = pvt.lookup(P_cur, T_cur).VF_mol
        seg_rho_g   = pvt.lookup(P_cur, T_cur).rho_g
        seg_rho_l   = pvt.lookup(P_cur, T_cur).rho_l
        seg_dp      = 0.0
        seg_Q       = 0.0
        seg_dH      = 0.0

        for _ in range(n_steps):
            state = pvt.lookup(P_cur, T_cur)

            dp_Pa = segment_dp(
                state=state,
                m_total_kgs=m,
                D_m=seg.D_inner_m,
                L_m=step_len,
                angle_deg=seg.angle_deg,
                roughness_m=seg.roughness_m,
            )
            Q_W = segment_heat_loss(
                config=config.thermal,
                D_m=seg.D_outer_m,
                L_m=step_len,
                T_fluid_C=T_cur,
            )

            dH     = -Q_W / m if m > 0 else 0.0
            H_cur  = H_cur + dH
            P_cur  = P_cur - dp_Pa / 1e5
            T_cur  = pvt.T_from_PH(P_cur, H_cur)

            seg_dp += dp_Pa
            seg_Q  += Q_W
            seg_dH += dH

        state_out = pvt.lookup(P_cur, T_cur)
        label     = seg.label or f"Seg {seg_idx + 1}"

        results.append(SegmentResult(
            label=label,
            P_in_bara=seg_P_in,   T_in_C=seg_T_in,
            P_out_bara=P_cur,     T_out_C=T_cur,
            dp_Pa=seg_dp,
            Q_loss_W=seg_Q,
            dH_Jkg=seg_dH,
            VF_mol_in=seg_VF_in,
            VF_mol_out=state_out.VF_mol,
            rho_g_in=seg_rho_g,
            rho_l_in=seg_rho_l,
        ))

        dp_total += seg_dp
        Q_total  += seg_Q

    return SolverResult(
        segments=results,
        P_out_bara=P_cur,
        T_out_C=T_cur,
        dp_total_Pa=dp_total,
        Q_total_W=Q_total,
        pvt_mode=pvt_mode,
    )


def _n_steps(length_m: float, max_step: Optional[float]) -> int:
    """Number of sub-steps for a segment of length_m given max_step size."""
    if max_step is None or max_step <= 0:
        return 1
    return max(1, math.ceil(length_m / max_step))
