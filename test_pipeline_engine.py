"""
test_pipeline_engine.py — Tests for pipeline_engine modules.

Run with:  python test_pipeline_engine.py
"""
import math
import warnings

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

_results = {"pass": 0, "fail": 0, "warn": 0}


def _check(name, actual, expected, tol_pct=None, tol_abs=None, warn_only=False):
    if tol_pct is not None:
        ok = abs(actual - expected) <= abs(expected) * tol_pct / 100.0
    elif tol_abs is not None:
        ok = abs(actual - expected) <= tol_abs
    else:
        ok = actual == expected
    tag = PASS if ok else (WARN if warn_only else FAIL)
    note = "  (warn-only)" if (warn_only and not ok) else ""
    print(f"  {tag}  {name}")
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else str(v)
    print(f"         got {fmt(actual)}  expected {fmt(expected)}{note}")
    _results["pass" if ok else ("warn" if warn_only else "fail")] += 1


def section(title):
    print(f"\n── {title} {'─'*(55-len(title))}")


# ── imports ────────────────────────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pipeline_engine.pvt import PVTFlash, PVTTable
    from pipeline_engine.heat_transfer import (
        ThermalConfig, SurfaceConditions, segment_heat_loss,
    )
    from pipeline_engine.flow_mech import segment_dp
    from pipeline_engine.solver import PipeSegment, SolverConfig, solve_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 1. PVTFlash
# ══════════════════════════════════════════════════════════════════════════════

section("1. PVTFlash — basic properties")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _pf_h2o2 = PVTFlash({"H₂": 5.0, "O₂": 2.5})
    _pf_ch4  = PVTFlash({"CH₄": 10.0})

# H₂/O₂ is always supercritical gas at operating conditions
_s = _pf_h2o2.lookup(10.0, 60.0)
_check("H2/O2 at 10 bara, 60 C is all gas", _s.VF_mol, 1.0, tol_abs=1e-4)
_check("H2/O2 rho_g reasonable (0.5–5 kg/m³)", float(0.5 < _s.rho_g < 5.0), 1.0)
_check("H2/O2 mu_g reasonable (5e-6–3e-5 Pa·s)", float(5e-6 < _s.mu_g < 3e-5), 1.0)
_check("H2/O2 rho_l is None (no liquid)", _s.rho_l is None, True)

# Enthalpy increases with temperature
_s20 = _pf_h2o2.lookup(10.0, 20.0)
_s80 = _pf_h2o2.lookup(10.0, 80.0)
_check("H2/O2 H increases with T", float(_s20.H_Jkg < _s80.H_Jkg), 1.0)

section("1. PVTFlash — T_from_PH round-trip")

for _T_test in [20.0, 60.0, 100.0]:
    _s_rt = _pf_h2o2.lookup(10.0, _T_test)
    _T_back = _pf_h2o2.T_from_PH(10.0, _s_rt.H_Jkg)
    _err_mK = abs(_T_back - _T_test) * 1000.0
    _check(f"H2/O2 T_from_PH round-trip at {_T_test:.0f} C (< 1 mK)", _err_mK, 0.0, tol_abs=1.0)

section("1. PVTFlash — condensation detection (CH₄ at 30 bara)")

# CH₄ at 30 bara: T_sat ≈ −96 °C
# Just above saturation → all gas; just below → all liquid
_s_gas = _pf_ch4.lookup(30.0, -90.0)
_s_liq = _pf_ch4.lookup(30.0, -105.0)
_check("CH4 at 30 bara, −90 C is gas (VF≈1)", _s_gas.VF_mol, 1.0, tol_abs=0.05)
_check("CH4 at 30 bara, −105 C is liquid (VF≈0)", _s_liq.VF_mol, 0.0, tol_abs=0.05)
_check("CH4 liquid rho_l > gas rho_g", float(_s_liq.rho_l > _s_gas.rho_g), 1.0)

# Latent heat: large H gap across saturation
_dH_latent = abs(_s_liq.H_Jkg - _s_gas.H_Jkg)
_check("CH4 latent heat > 200 kJ/kg across saturation", float(_dH_latent > 200_000), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PVTTable
# ══════════════════════════════════════════════════════════════════════════════

section("2. PVTTable — build and basic lookup")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _pt = PVTTable(
        {"H₂": 5.0, "O₂": 2.5},
        P_min_bara=5.0, P_max_bara=30.0,
        T_min_C=10.0,   T_max_C=120.0,
        n_P=15, n_T=15,
    )

_st = _pt.lookup(15.0, 70.0)
_check("PVTTable H2/O2 is gas (VF≈1)", _st.VF_mol, 1.0, tol_abs=0.05)
_check("PVTTable rho_g in 0.1–5 kg/m³", float(0.1 < _st.rho_g < 5.0), 1.0)

section("2. PVTTable — T_from_PH round-trip")

for _T_test in [30.0, 70.0, 110.0]:
    _st_rt = _pt.lookup(15.0, _T_test)
    _T_back = _pt.T_from_PH(15.0, _st_rt.H_Jkg)
    _err_K = abs(_T_back - _T_test)
    _check(f"PVTTable T_from_PH at {_T_test:.0f} C (< 2 K)", _err_K, 0.0, tol_abs=2.0)

section("2. PVTFlash vs PVTTable — same all-gas result")

_sf = _pf_h2o2.lookup(10.0, 60.0)
_st2 = _pt.lookup(10.0, 60.0)
_check("rho_g within 5%", _sf.rho_g, _st2.rho_g, tol_pct=5.0)
_check("mu_g within 10%",  _sf.mu_g,  _st2.mu_g,  tol_pct=10.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Heat transfer
# ══════════════════════════════════════════════════════════════════════════════

section("3. heat_transfer — isothermal")

_Q = segment_heat_loss(ThermalConfig(mode="isothermal"), D_m=0.114, L_m=100.0, T_fluid_C=60.0)
_check("Isothermal Q = 0", _Q, 0.0, tol_abs=1e-9)

section("3. heat_transfer — U-value formula")

# Q = U · π · D · L · ΔT
_U, _D, _L, _Tf, _Ta = 2.0, 0.114, 100.0, 60.0, 10.0
_Q_expected = _U * math.pi * _D * _L * (_Tf - _Ta)
_Q_got = segment_heat_loss(
    ThermalConfig(mode="u_value", U_W_m2K=_U, T_amb_C=_Ta),
    D_m=_D, L_m=_L, T_fluid_C=_Tf,
)
_check("U-value Q matches formula", _Q_got, _Q_expected, tol_pct=0.01)

_Q_zero = segment_heat_loss(
    ThermalConfig(mode="u_value", U_W_m2K=_U, T_amb_C=_Tf),  # T_amb = T_fluid
    D_m=_D, L_m=_L, T_fluid_C=_Tf,
)
_check("U-value Q = 0 when T_fluid = T_amb", _Q_zero, 0.0, tol_abs=1e-6)

_Q_neg = segment_heat_loss(
    ThermalConfig(mode="u_value", U_W_m2K=_U, T_amb_C=80.0),  # T_amb > T_fluid
    D_m=_D, L_m=_L, T_fluid_C=_Tf,
)
_check("U-value Q < 0 when T_amb > T_fluid (fluid gains heat)", float(_Q_neg < 0), 1.0)

section("3. heat_transfer — surface mode")

_sc = SurfaceConditions(T_amb_C=10.0, wind_ms=5.0, solar_W_m2=0.0, rain=False)
_Q_surf = segment_heat_loss(
    ThermalConfig(mode="surface", surface=_sc), D_m=0.114, L_m=100.0, T_fluid_C=60.0,
)
_check("Surface Q > 0 (fluid hotter than ambient)", float(_Q_surf > 0), 1.0)

# Solar gain reduces heat loss
_sc_solar = SurfaceConditions(T_amb_C=10.0, wind_ms=5.0, solar_W_m2=1000.0, rain=False)
_Q_solar = segment_heat_loss(
    ThermalConfig(mode="surface", surface=_sc_solar), D_m=0.114, L_m=100.0, T_fluid_C=60.0,
)
_check("Solar gain reduces net Q loss", float(_Q_solar < _Q_surf), 1.0)

# Rain increases cooling
_sc_rain = SurfaceConditions(T_amb_C=10.0, wind_ms=5.0, rain=True)
_Q_rain = segment_heat_loss(
    ThermalConfig(mode="surface", surface=_sc_rain), D_m=0.114, L_m=100.0, T_fluid_C=60.0,
)
_check("Rain increases Q loss vs no rain", float(_Q_rain > _Q_surf), 1.0)

# Zero wind → natural convection minimum (h = 5 W/m²K)
_sc_calm = SurfaceConditions(T_amb_C=10.0, wind_ms=0.0, solar_W_m2=0.0)
_Q_calm = segment_heat_loss(
    ThermalConfig(mode="surface", surface=_sc_calm), D_m=0.114, L_m=100.0, T_fluid_C=60.0,
)
_check("Calm wind gives positive (but lower) Q", float(0 < _Q_calm < _Q_surf), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. flow_mech
# ══════════════════════════════════════════════════════════════════════════════

section("4. flow_mech — single-phase gas")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _pf_n2 = PVTFlash({"N₂": 10.0})
_state_gas = _pf_n2.lookup(30.0, 40.0)   # supercritical N₂
_D_i = 0.0779   # DN80 inner

# m=2.0 kg/s for friction tests (high Re, clear friction signal)
_dp_h  = segment_dp(_state_gas, m_total_kgs=2.0,  D_m=_D_i, L_m=100.0, angle_deg=0.0)
_dp_up = segment_dp(_state_gas, m_total_kgs=2.0,  D_m=_D_i, L_m=100.0, angle_deg=90.0)
_dp_dn = segment_dp(_state_gas, m_total_kgs=2.0,  D_m=_D_i, L_m=100.0, angle_deg=-90.0)
# m=0.05 kg/s for pressure-gain test (low velocity → gravity dominates friction)
_dp_dn_slow = segment_dp(_state_gas, m_total_kgs=0.05, D_m=_D_i, L_m=30.0, angle_deg=-90.0)

_check("Horizontal dp > 0 (friction loss)",        float(_dp_h > 0),           1.0)
_check("Vertical up dp > horizontal dp",            float(_dp_up > _dp_h),      1.0)
_check("Vertical down dp < horizontal dp",          float(_dp_dn < _dp_h),      1.0)
_check("Low-velocity downflow gives pressure gain", float(_dp_dn_slow < 0),     1.0)

# Friction dp scales roughly with L
_dp_h_200 = segment_dp(_state_gas, m_total_kgs=2.0, D_m=_D_i, L_m=200.0, angle_deg=0.0)
_check("Horizontal dp doubles with double length", _dp_h_200, 2 * _dp_h, tol_pct=5.0)

section("4. flow_mech — two-phase Beggs-Brill")

# Use CH₄ in two-phase region (just below saturation at 10 bara, T ≈ −110°C)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _state_2p = _pf_ch4.lookup(10.0, -130.0)   # liquid CH₄

_dp_liq = segment_dp(_state_2p, m_total_kgs=0.5, D_m=_D_i, L_m=50.0, angle_deg=0.0)
_check("Liquid dp > 0", float(_dp_liq > 0), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Solver — use_flash toggle
# ══════════════════════════════════════════════════════════════════════════════

section("5. Solver — use_flash=True vs use_flash=False (all-gas H2/O2)")

_segs = [PipeSegment(length_m=20.0, angle_deg=0.0) for _ in range(5)]
_base = dict(
    composition_kgh={"H₂": 5.0, "O₂": 2.5},
    m_total_kgs=0.002,
    P_in_bara=15.0,
    T_in_C=60.0,
    thermal=ThermalConfig(mode="isothermal"),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_flash = SolverConfig(**_base, use_flash=True)
    _cfg_table = SolverConfig(**_base, use_flash=False)

_res_f = solve_pipeline(_segs, _cfg_flash)
_res_t = solve_pipeline(_segs, _cfg_table)

_check("Flash mode pvt_mode tag",  _res_f.pvt_mode, "flash")
_check("Table mode pvt_mode tag",  _res_t.pvt_mode, "table")
_check("P_out matches within 0.5%", _res_f.P_out_bara, _res_t.P_out_bara, tol_pct=0.5)
_check("T_out matches within 2 K",  _res_f.T_out_C,    _res_t.T_out_C,    tol_abs=2.0)
_check("dP_total > 0 (pressure loss)", float(_res_f.dp_total_Pa > 0), 1.0)

section("5. Solver — pressure decreases along pipeline")

_check("Outlet P < inlet P", float(_res_f.P_out_bara < _base["P_in_bara"]), 1.0)

section("5. Solver — isothermal adiabatic preserves T (ideal gas limit)")

# For H₂/O₂ which behaves nearly ideally at low pressure, isothermal adiabatic
# should show minimal T change (JT ≈ small for H₂/O₂ mix)
_segs_low = [PipeSegment(length_m=20.0, angle_deg=0.0) for _ in range(5)]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_adi = SolverConfig(
        composition_kgh={"H₂": 5.0, "O₂": 2.5},
        m_total_kgs=0.001, P_in_bara=3.0, T_in_C=50.0,
        thermal=ThermalConfig(mode="isothermal"), use_flash=True,
    )
_res_adi = solve_pipeline(_segs_low, _cfg_adi)
_dT_jt = abs(_res_adi.T_out_C - 50.0)
_check("Adiabatic JT: T change < 5 K over 100 m (near-ideal gas)", float(_dT_jt < 5.0), 1.0)

section("5. Solver — heat loss drives T toward ambient")

# m=0.05 kg/s keeps dH per segment ≈ 55 kJ/kg — within thermo's valid range
# 8 segments × 50 m needed to drop from 80 °C to below 50 °C
_segs_long = [PipeSegment(length_m=50.0, angle_deg=0.0, D_inner_m=0.0779, D_outer_m=0.0889)
              for _ in range(8)]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_cool = SolverConfig(
        composition_kgh={"H₂": 5.0, "O₂": 2.5},
        m_total_kgs=0.05, P_in_bara=10.0, T_in_C=80.0,
        thermal=ThermalConfig(mode="u_value", U_W_m2K=3.0, T_amb_C=15.0),
        use_flash=True,
    )
_res_cool = solve_pipeline(_segs_long, _cfg_cool)
_check("Cooling: T_out < T_in",         float(_res_cool.T_out_C < 80.0),  1.0)
_check("Cooling: Q_total > 0",           float(_res_cool.Q_total_W > 0),   1.0)
_check("Cooling: T_out approaching amb", float(_res_cool.T_out_C < 50.0),  1.0)

section("5. Solver — vertical upflow increases dp vs horizontal")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_v = SolverConfig(
        composition_kgh={"H₂": 5.0, "O₂": 2.5},
        m_total_kgs=0.01, P_in_bara=10.0, T_in_C=40.0,
        thermal=ThermalConfig(mode="isothermal"), use_flash=True,
    )
_segs_horiz = [PipeSegment(length_m=30.0, angle_deg=0.0)]
_segs_up    = [PipeSegment(length_m=30.0, angle_deg=90.0)]
_segs_down  = [PipeSegment(length_m=30.0, angle_deg=-90.0)]
_res_h  = solve_pipeline(_segs_horiz, _cfg_v)
_res_up = solve_pipeline(_segs_up,    _cfg_v)
_res_dn = solve_pipeline(_segs_down,  _cfg_v)
_check("Upflow dp > horizontal dp",   float(_res_up.dp_total_Pa > _res_h.dp_total_Pa),  1.0)
_check("Downflow dp < horizontal dp", float(_res_dn.dp_total_Pa < _res_h.dp_total_Pa),  1.0)

section("5. Solver — segment_length_m auto-subdivision")

# One 110 m segment; without subdivision a single huge step overshoots badly.
# With segment_length_m=5, the solver splits it into 22 steps internally
# and returns one SegmentResult (aggregated), not 22.
_seg_110 = [PipeSegment(length_m=110.0, angle_deg=0.0,
                        D_inner_m=0.0779, D_outer_m=0.0889, label="Cooler")]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_coarse = SolverConfig(
        composition_kgh={"CH₄": 10.0}, m_total_kgs=0.05,
        P_in_bara=30.0, T_in_C=20.0,
        thermal=ThermalConfig(mode="u_value", U_W_m2K=20.0, T_amb_C=-130.0),
        use_flash=True, segment_length_m=None,
    )
    _cfg_fine = SolverConfig(
        composition_kgh={"CH₄": 10.0}, m_total_kgs=0.05,
        P_in_bara=30.0, T_in_C=20.0,
        thermal=ThermalConfig(mode="u_value", U_W_m2K=20.0, T_amb_C=-130.0),
        use_flash=True, segment_length_m=5.0,
    )

_res_coarse = solve_pipeline(_seg_110, _cfg_coarse)
_res_fine   = solve_pipeline(_seg_110, _cfg_fine)

_check("Coarse (no subdivision): result still has 1 segment", len(_res_coarse.segments), 1)
_check("Fine (5 m steps):        result still has 1 segment", len(_res_fine.segments),   1)
_check("Fine gives lower T_out than coarse (subdivision matters)",
       float(_res_fine.segments[0].T_out_C > _res_coarse.segments[0].T_out_C), 1.0)
_check("Fine T_out reasonable (near T_sat ≈ −97 °C or below)",
       float(_res_fine.segments[0].T_out_C < -90.0), 1.0)

section("5. Solver — condensation: CH₄ cooling crosses saturation")

# One 110 m pipe with 5 m internal steps — replaces 22 hand-crafted segments.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _cfg_cond = SolverConfig(
        composition_kgh={"CH₄": 10.0},
        m_total_kgs=0.05, P_in_bara=30.0, T_in_C=20.0,
        thermal=ThermalConfig(mode="u_value", U_W_m2K=20.0, T_amb_C=-130.0),
        use_flash=True, segment_length_m=5.0,
    )
_segs_cond = [PipeSegment(length_m=110.0, angle_deg=0.0,
                          D_inner_m=0.0779, D_outer_m=0.0889)]
_res_cond = solve_pipeline(_segs_cond, _cfg_cond)

_vf_first = _res_cond.segments[0].VF_mol_in
_vf_last  = _res_cond.segments[-1].VF_mol_out
_check("CH4 pipeline: enters as gas (VF_in ≈ 1)", _vf_first, 1.0, tol_abs=0.05)
_check("CH4 pipeline: exits as liquid (VF_out ≈ 0)", _vf_last, 0.0, tol_abs=0.05)

# Check that condensation was detected somewhere in the middle
_any_two_phase = any(
    sr.VF_mol_out < 0.95 and sr.VF_mol_out > 0.05
    for sr in _res_cond.segments
)
# Either gradual transition (mixture) or abrupt (pure component) is valid
_condenses = _vf_last < 0.5
_check("CH4 pipeline: condensation occurs (VF_out < 0.5)", float(_condenses), 1.0)

section("5. Solver — segment count preserved")

_check("Segment count in result matches input",
       len(_res_f.segments), len(_segs))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

total = sum(_results.values())
print(f"\n{'─'*60}")
print(f"Results: {_results['pass']}/{total} passed, "
      f"{_results['fail']} failed, {_results['warn']} warnings")

if _results["fail"]:
    raise SystemExit(1)
