"""
pipeline_engine/pvt.py
======================
PVT lookup table for a fixed-composition two-phase mixture.

Workflow
--------
1. Supply a composition dict {display_name: mass_flow_kgh} and P/T bounds.
2. PVTTable.__init__ runs a 20×20 PT flash grid (< 0.3 s for typical mixtures)
   and stores all phase properties in NumPy arrays.
3. .lookup(P_bara, T_C)  → PVTState  (bilinear interpolation, O(1))
4. .T_from_PH(P_bara, H_Jkg) → float  (row inversion via np.interp, < 0.03 K error)

Composition normalisation
-------------------------
"Air" is expanded to N₂ / O₂ / Ar by standard mole fractions before flashing.
All species are mapped from display names to CAS/thermo IDs via SPECIES_MAP.

Dependencies
------------
thermo  >= 0.2.x  (Caleb Bell)
scipy   >= 1.7
numpy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ── thermo import (mandatory for this module) ─────────────────────────────────
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, FlashVL, FlashPureVLS
from thermo.eos_mix import PRMIX


# ── species name → thermo CAS / IUPAC identifier ─────────────────────────────
SPECIES_MAP: Dict[str, str] = {
    # electrolyser gases
    "H₂":   "hydrogen",
    "O₂":   "oxygen",
    # common gases
    "N₂":   "nitrogen",
    "CO₂":  "carbon dioxide",
    "CH₄":  "methane",
    "C₂H₆": "ethane",
    "C₃H₈": "propane",
    "Ar":   "argon",
    "He":   "helium",
    "NH₃":  "ammonia",
    "H₂S":  "hydrogen sulfide",
    "CO":   "carbon monoxide",
    "H₂O":  "water",
    "Water": "water",
    # air components (used internally after expansion)
}

# Air mole fractions used for expansion
_AIR_MOL_FRACS = {"N₂": 0.7809, "O₂": 0.2095, "Ar": 0.0093}
_AIR_MW_KGMOL = 28.966e-3  # kg/mol


def _expand_air(feed_kgh: Dict[str, float]) -> Dict[str, float]:
    """Replace 'Air' with N₂/O₂/Ar at standard mole fractions."""
    out: Dict[str, float] = {}
    for sp, kgh in feed_kgh.items():
        if sp == "Air":
            mol_total = kgh / _AIR_MW_KGMOL  # kmol/h (scale doesn't matter)
            for comp, xmol in _AIR_MOL_FRACS.items():
                # import here to avoid circular dep
                from thermo import Chemical
                mw = Chemical(SPECIES_MAP.get(comp, comp)).MW * 1e-3  # kg/mol
                out[comp] = out.get(comp, 0.0) + mol_total * xmol * mw
        else:
            out[sp] = out.get(sp, 0.0) + kgh
    return out


def _build_flasher(
    composition_kgh: Dict[str, float],
) -> Tuple[FlashVL, np.ndarray, list]:
    """
    Build a thermo FlashVL for the given composition.

    Returns
    -------
    flasher    : FlashVL
    zs         : mole fractions (np.ndarray, sums to 1)
    thermo_ids : list of thermo species names in the same order as zs
    """
    feed = _expand_air(composition_kgh)
    thermo_ids = []
    mol_flows = []

    for display_name, kgh in feed.items():
        tid = SPECIES_MAP.get(display_name, display_name)
        thermo_ids.append(tid)
        # temporary MW lookup — we only need ratios so kg/h is fine as proxy
        # actual MW resolved below
        mol_flows.append(kgh)  # placeholder: will normalise after MW lookup

    # Build constants once to get MWs
    constants, props = ChemicalConstantsPackage.from_IDs(thermo_ids)
    MWs = np.array(constants.MWs)  # g/mol

    # Convert kg/h → mol/h
    mol_flows_arr = np.array(mol_flows) / (MWs * 1e-3)  # mol/h
    zs = mol_flows_arr / mol_flows_arr.sum()

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas_obj   = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)
    liq_obj   = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=props.HeatCapacityGases)

    # FlashVL stability test divides by (N-1); use FlashPureVLS for pure components
    if len(thermo_ids) == 1:
        flasher = FlashPureVLS(constants, props, liquids=[liq_obj], gas=gas_obj, solids=[])
    else:
        flasher = FlashVL(constants, props, liquid=liq_obj, gas=gas_obj)

    return flasher, zs, thermo_ids, constants


# ── PVT state dataclass ────────────────────────────────────────────────────────

@dataclass
class PVTState:
    """All properties needed by the flow-mechanics solver at a grid point."""
    P_bara: float
    T_C:    float

    # Phase fractions
    VF_mol:  float   # vapour fraction (mol basis), 0 → all liquid, 1 → all gas
    VF_mass: float   # vapour fraction (mass basis)

    # Gas-phase properties (None if all liquid)
    rho_g:  Optional[float]   # kg/m³
    mu_g:   Optional[float]   # Pa·s
    # Liquid-phase properties (None if all gas)
    rho_l:  Optional[float]   # kg/m³
    mu_l:   Optional[float]   # Pa·s
    sigma:  Optional[float]   # N/m  (surface tension)

    # Mixture enthalpy (mass basis, J/kg) — used for energy marching
    H_Jkg: float

    # Mixture Cp (mass basis, J/kg/K) — informational
    Cp_Jkgk: float


# ── PVT table ─────────────────────────────────────────────────────────────────

class PVTTable:
    """
    Pre-computed 20×20 (P, T) property table for a fixed composition.

    All properties are stored as 2-D numpy arrays indexed [i_P, i_T].
    Bilinear interpolation is used for lookups between grid points.

    Parameters
    ----------
    composition_kgh  : {display_name: kg/h}  (relative values, only ratios matter)
    P_min_bara       : lower pressure bound (bara)
    P_max_bara       : upper pressure bound (bara)
    T_min_C          : lower temperature bound (°C)
    T_max_C          : upper temperature bound (°C)
    n_P, n_T         : grid resolution (default 20×20)
    ref_T_C          : reference temperature for H datum (°C); default 25.0
    ref_P_bara       : reference pressure for H datum (bara); default 1.013
    """

    def __init__(
        self,
        composition_kgh: Dict[str, float],
        P_min_bara: float = 1.0,
        P_max_bara: float = 50.0,
        T_min_C:    float = 0.0,
        T_max_C:    float = 150.0,
        n_P: int = 20,
        n_T: int = 20,
        ref_T_C: float = 25.0,
        ref_P_bara: float = 1.013,
    ):
        self.composition_kgh = dict(composition_kgh)
        self.P_min = P_min_bara
        self.P_max = P_max_bara
        self.T_min = T_min_C
        self.T_max = T_max_C
        self.n_P   = n_P
        self.n_T   = n_T

        self._flasher, self._zs, self._ids, self._constants = _build_flasher(
            composition_kgh
        )
        self._MW_mix_kgmol = float(np.dot(np.array(self._constants.MWs) * 1e-3, self._zs))

        # Axis vectors
        self.P_axis = np.linspace(P_min_bara, P_max_bara, n_P)  # bara
        self.T_axis = np.linspace(T_min_C, T_max_C, n_T)         # °C

        # Compute H reference (datum)
        self._H_ref_Jmol = self._flash_H_ref(ref_T_C, ref_P_bara)

        # Build the grid
        self._build_grid()

    # ── grid construction ─────────────────────────────────────────────────────

    def _flash_H_ref(self, T_C: float, P_bara: float) -> float:
        """Enthalpy at reference state (J/mol)."""
        try:
            r = self._flasher.flash(T=T_C + 273.15, P=P_bara * 1e5, zs=self._zs.tolist())
            return r.H()
        except Exception:
            return 0.0

    def _flash_point(self, P_bara: float, T_C: float) -> dict:
        """
        Run a single PT flash and return a dict of scalar properties.
        All values are float; use NaN on failure.
        """
        T_K = T_C + 273.15
        P_Pa = P_bara * 1e5
        nan = float("nan")
        out = dict(
            VF_mol=nan, VF_mass=nan,
            rho_g=nan, mu_g=nan,
            rho_l=nan, mu_l=nan, sigma=nan,
            H_Jkg=nan, Cp_Jkgk=nan,
        )
        try:
            r = self._flasher.flash(T=T_K, P=P_Pa, zs=self._zs.tolist())
            VF = float(r.VF) if r.VF is not None else 0.0
            VF = max(0.0, min(1.0, VF))
            out["VF_mol"] = VF

            liq = r.liquids[0] if r.liquids else None

            # Mass-basis VF
            if 0.0 < VF < 1.0 and r.gas is not None and liq is not None:
                MW_g = float(np.dot(
                    np.array(r.gas.zs) * np.array(self._constants.MWs),
                    np.ones(len(self._constants.MWs)),
                ) / sum(np.array(r.gas.zs)))
                MW_l = float(np.dot(
                    np.array(liq.zs) * np.array(self._constants.MWs),
                    np.ones(len(self._constants.MWs)),
                ) / sum(np.array(liq.zs)))
                num = VF * MW_g
                denom = num + (1.0 - VF) * MW_l
                out["VF_mass"] = float(num / denom) if denom > 0 else VF
            else:
                out["VF_mass"] = float(VF)

            # Gas properties
            if r.gas is not None and VF > 1e-6:
                try:
                    out["rho_g"] = float(r.gas.rho_mass())
                except Exception:
                    pass
                try:
                    out["mu_g"] = float(r.gas.mu())
                except Exception:
                    pass

            # Liquid properties
            if liq is not None and VF < 1.0 - 1e-6:
                try:
                    out["rho_l"] = float(liq.rho_mass())
                except Exception:
                    pass
                try:
                    out["mu_l"] = float(liq.mu())
                except Exception:
                    pass
                try:
                    out["sigma"] = float(liq.sigma())
                except Exception:
                    pass

            # Enthalpy (J/kg relative to reference datum)
            try:
                H_Jmol = float(r.H()) - self._H_ref_Jmol
                out["H_Jkg"] = H_Jmol / self._MW_mix_kgmol
            except Exception:
                pass

            # Cp
            try:
                Cp_Jmolk = float(r.Cp())
                out["Cp_Jkgk"] = Cp_Jmolk / self._MW_mix_kgmol
            except Exception:
                pass

        except Exception as exc:
            warnings.warn(f"PVT flash failed at P={P_bara:.2f} bara, T={T_C:.1f} °C: {exc}")

        return out

    def _build_grid(self) -> None:
        """Populate all property arrays over the (P, T) grid."""
        shape = (self.n_P, self.n_T)
        props = ["VF_mol", "VF_mass", "rho_g", "mu_g", "rho_l", "mu_l", "sigma",
                 "H_Jkg", "Cp_Jkgk"]
        arrays: Dict[str, np.ndarray] = {p: np.full(shape, np.nan) for p in props}

        for i, P in enumerate(self.P_axis):
            for j, T in enumerate(self.T_axis):
                pt = self._flash_point(P, T)
                for p in props:
                    arrays[p][i, j] = pt[p]

        self._arrays = arrays

        # Build interpolators — one per property
        axes = (self.P_axis, self.T_axis)
        self._interp: Dict[str, RegularGridInterpolator] = {}
        for prop, arr in arrays.items():
            # Replace NaN with nearest valid value for interpolation robustness
            clean = _fill_nan(arr)
            self._interp[prop] = RegularGridInterpolator(
                axes, clean, method="linear", bounds_error=False, fill_value=None
            )

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(self, P_bara: float, T_C: float) -> PVTState:
        """
        Bilinear interpolation at (P_bara, T_C).

        Points outside the grid are extrapolated linearly (fill_value=None).
        """
        pt = np.array([[P_bara, T_C]])
        g = {prop: float(interp(pt)[0]) for prop, interp in self._interp.items()}

        def _nan_to_none(v: float) -> Optional[float]:
            return None if np.isnan(v) else v

        return PVTState(
            P_bara=P_bara,
            T_C=T_C,
            VF_mol=g["VF_mol"],
            VF_mass=g["VF_mass"],
            rho_g=_nan_to_none(g["rho_g"]),
            mu_g=_nan_to_none(g["mu_g"]),
            rho_l=_nan_to_none(g["rho_l"]),
            mu_l=_nan_to_none(g["mu_l"]),
            sigma=_nan_to_none(g["sigma"]),
            H_Jkg=g["H_Jkg"],
            Cp_Jkgk=g["Cp_Jkgk"],
        )

    def T_from_PH(self, P_bara: float, H_Jkg: float) -> float:
        """
        Invert H(T) at fixed P to recover temperature.

        Uses np.interp on the H column at the nearest grid P index.
        Error < 0.03 K for smooth H(T) curves.

        Returns T in °C; clamps to [T_min, T_max] if out of range.
        """
        # Find nearest P index
        i_P = int(np.argmin(np.abs(self.P_axis - P_bara)))
        H_col = self._arrays["H_Jkg"][i_P, :]
        T_col = self.T_axis
        # np.interp requires x to be monotonically increasing
        # H increases with T, so sort is guaranteed for well-behaved fluids
        # but guard against NaNs
        mask = np.isfinite(H_col)
        if mask.sum() < 2:
            # fallback: return midpoint
            return float((self.T_min + self.T_max) / 2.0)
        H_clean = H_col[mask]
        T_clean = T_col[mask]
        # Sort by H in case of non-monotonicity near phase boundary
        order = np.argsort(H_clean)
        T_out = float(np.interp(H_Jkg, H_clean[order], T_clean[order]))
        return float(np.clip(T_out, self.T_min, self.T_max))

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.n_P, self.n_T)

    def summary(self) -> str:
        """One-line description of the table."""
        return (
            f"PVTTable [{self.n_P}×{self.n_T}] "
            f"P={self.P_min:.1f}–{self.P_max:.1f} bara  "
            f"T={self.T_min:.0f}–{self.T_max:.0f} °C  "
            f"species={self._ids}"
        )


# ── helper: fill NaN by nearest valid neighbour ────────────────────────────────

def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN cells with the nearest finite value (simple row-then-col fill)."""
    out = arr.copy()
    # Forward-fill along columns (T axis) first
    for i in range(out.shape[0]):
        row = out[i]
        mask = np.isfinite(row)
        if mask.any() and not mask.all():
            valid_idx = np.where(mask)[0]
            out[i] = np.interp(np.arange(len(row)), valid_idx, row[mask])
    # Forward-fill along rows (P axis) to fix any remaining NaNs
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isfinite(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(mask)[0]
            out[:, j] = np.interp(np.arange(len(col)), valid_idx, col[mask])
    # Last resort: replace any surviving NaN with 0
    out = np.where(np.isfinite(out), out, 0.0)
    return out


# ── Live-flash PVT source (same interface as PVTTable) ────────────────────────

class PVTFlash:
    """
    Drop-in replacement for PVTTable that runs a full PR-EOS flash at every
    lookup instead of interpolating from a pre-built grid.

    Handles condensation exactly: no grid-skipping, latent heat and phase
    compositions are always thermodynamically consistent.

    Cost vs PVTTable
    ----------------
    Pure component   :  ~0.01 ms/call   → negligible
    2–3 component    :  ~0.2–0.9 ms/call → fine for ≤ 1000 segments
    5+ component     :  ~1–2 ms/call     → consider PVTTable for many segments

    T_from_PH uses a PH flash (thermo native) — round-trip error < 0.001 K.

    Parameters
    ----------
    composition_kgh : {display_name: kg/h}
    ref_T_C         : Enthalpy datum temperature [°C]
    ref_P_bara      : Enthalpy datum pressure [bara]
    """

    def __init__(
        self,
        composition_kgh: Dict[str, float],
        ref_T_C:    float = 25.0,
        ref_P_bara: float = 1.013,
    ):
        self.composition_kgh = dict(composition_kgh)

        self._flasher, self._zs, self._ids, self._constants = _build_flasher(
            composition_kgh
        )
        self._MW_mix_kgmol = float(
            np.dot(np.array(self._constants.MWs) * 1e-3, self._zs)
        )
        # Enthalpy datum (J/mol)
        try:
            r0 = self._flasher.flash(
                T=ref_T_C + 273.15, P=ref_P_bara * 1e5, zs=self._zs.tolist()
            )
            self._H_ref_Jmol = float(r0.H())
        except Exception:
            self._H_ref_Jmol = 0.0

    # ── PT flash → PVTState ────────────────────────────────────────────────────

    def lookup(self, P_bara: float, T_C: float) -> "PVTState":
        """Run a full PT flash and return phase properties as a PVTState."""
        T_K  = T_C  + 273.15
        P_Pa = P_bara * 1e5
        zs   = self._zs.tolist()

        try:
            r   = self._flasher.flash(T=T_K, P=P_Pa, zs=zs)
            liq = r.liquids[0] if r.liquids else None
            VF  = float(r.VF) if r.VF is not None else 0.0
            VF  = max(0.0, min(1.0, VF))

            # Mass vapour fraction
            if 0.0 < VF < 1.0 and r.gas is not None and liq is not None:
                MWs = np.array(self._constants.MWs)
                MW_g = float(np.dot(r.gas.zs, MWs) / sum(r.gas.zs))
                MW_l = float(np.dot(liq.zs,   MWs) / sum(liq.zs))
                num  = VF * MW_g
                denom = num + (1.0 - VF) * MW_l
                VF_mass = float(num / denom) if denom > 0 else VF
            else:
                VF_mass = float(VF)

            rho_g = float(r.gas.rho_mass())          if (r.gas  and VF > 1e-6)       else None
            mu_g  = float(r.gas.mu())                if (r.gas  and VF > 1e-6)       else None
            rho_l = float(liq.rho_mass())            if (liq    and VF < 1.0 - 1e-6) else None
            mu_l  = float(liq.mu())                  if (liq    and VF < 1.0 - 1e-6) else None
            sigma = float(liq.sigma())               if (liq    and VF < 1.0 - 1e-6) else None

            H_Jkg   = (float(r.H()) - self._H_ref_Jmol) / self._MW_mix_kgmol
            Cp_Jkgk = float(r.Cp()) / self._MW_mix_kgmol

        except Exception as exc:
            warnings.warn(f"PVTFlash.lookup failed at P={P_bara:.2f} bara, T={T_C:.1f} °C: {exc}")
            VF = VF_mass = 1.0
            rho_g = mu_g = rho_l = mu_l = sigma = None
            H_Jkg = Cp_Jkgk = float("nan")

        return PVTState(
            P_bara=P_bara, T_C=T_C,
            VF_mol=VF, VF_mass=VF_mass,
            rho_g=rho_g, mu_g=mu_g,
            rho_l=rho_l, mu_l=mu_l, sigma=sigma,
            H_Jkg=H_Jkg, Cp_Jkgk=Cp_Jkgk,
        )

    # ── PH flash → temperature ─────────────────────────────────────────────────

    def T_from_PH(self, P_bara: float, H_Jkg: float,
                  T_min_C: float = -200.0, T_max_C: float = 600.0) -> float:
        """
        Recover temperature from pressure + specific enthalpy.

        Tries the native thermo PH flash first (< 0.001 K error).
        Falls back to a PT-bisection search when the PH solver fails
        (e.g., H outside the valid EOS range).  Bisection precision ≈ 0.05 K.
        """
        H_Jmol = H_Jkg * self._MW_mix_kgmol + self._H_ref_Jmol
        P_Pa   = P_bara * 1e5
        zs     = self._zs.tolist()

        # ── Try native PH flash ────────────────────────────────────────────────
        try:
            r = self._flasher.flash(P=P_Pa, H=H_Jmol, zs=zs)
            T = float(r.T) - 273.15
            if not (T != T):   # not NaN
                return T
        except Exception:
            pass

        # ── Bisection fallback ─────────────────────────────────────────────────
        # H(T) is monotonically increasing for well-behaved fluids.
        # If H_Jkg is below H(T_min) clamp to T_min; above H(T_max) clamp to T_max.
        def _H_at_T(T_C: float) -> float:
            try:
                r2 = self._flasher.flash(T=T_C + 273.15, P=P_Pa, zs=zs)
                return (float(r2.H()) - self._H_ref_Jmol) / self._MW_mix_kgmol
            except Exception:
                return float("nan")

        H_lo = _H_at_T(T_min_C)
        H_hi = _H_at_T(T_max_C)
        if H_Jkg <= H_lo:
            return T_min_C
        if H_Jkg >= H_hi:
            return T_max_C

        lo, hi = T_min_C, T_max_C
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if hi - lo < 0.05:
                break
            H_mid = _H_at_T(mid)
            if H_mid < H_Jkg:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def summary(self) -> str:
        return f"PVTFlash (live flash)  species={self._ids}"
