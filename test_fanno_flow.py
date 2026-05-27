"""Tests for fanno_engine.py — Fanno flow physics and solver."""

import math
import sys
import unittest

import fanno_engine as fanno

_fanno_param = fanno.fanno_param
_T_ratio     = fanno.T_ratio
_P_ratio     = fanno.P_ratio
_rho_ratio   = fanno.rho_ratio
_P0_ratio    = fanno.P0_ratio
_solve_Ma    = fanno.solve_Ma
_churchill_f = fanno.churchill_f
fanno_solve  = fanno.fanno_solve
_PIPE_DB     = None   # pipe DB lives in multiphase_engine; tested separately below
_GASES       = fanno.GASES
R_u          = fanno.R_u

GAMMA_AIR = 1.4
MW_AIR    = 0.028965  # kg/mol


# ===========================================================================
# 1. Fanno relations against textbook tables (γ = 1.4)
# ===========================================================================

class TestFannoParam(unittest.TestCase):
    """4fL*/D values from NACA / Anderson appendix tables."""

    def _check(self, Ma, expected, tol=0.001):
        got = _fanno_param(Ma, GAMMA_AIR)
        self.assertAlmostEqual(got, expected, delta=tol,
                               msg=f"_fanno_param({Ma}) = {got:.6f}, expected {expected}")

    def test_sonic_is_zero(self):
        self.assertAlmostEqual(_fanno_param(1.0, GAMMA_AIR), 0.0, delta=1e-10)

    def test_Ma_03(self):     self._check(0.3, 5.2993)
    def test_Ma_05(self):     self._check(0.5, 1.0691)
    def test_Ma_07(self):     self._check(0.7, 0.2081)
    def test_Ma_20(self):     self._check(2.0, 0.3050)
    def test_Ma_30(self):     self._check(3.0, 0.5222)

    def test_zero_returns_inf(self):
        self.assertEqual(_fanno_param(0.0, GAMMA_AIR), math.inf)

    def test_negative_returns_inf(self):
        self.assertEqual(_fanno_param(-0.5, GAMMA_AIR), math.inf)

    def test_subsonic_decreases_toward_sonic(self):
        vals = [_fanno_param(Ma, GAMMA_AIR) for Ma in [0.1, 0.3, 0.5, 0.7, 0.9, 0.999]]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(a, b)

    def test_supersonic_increases_from_sonic(self):
        vals = [_fanno_param(Ma, GAMMA_AIR) for Ma in [1.001, 1.5, 2.0, 3.0, 4.0]]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b)


class TestTemperatureRatio(unittest.TestCase):
    def test_sonic(self):
        self.assertAlmostEqual(_T_ratio(1.0, GAMMA_AIR), 1.0, delta=1e-12)

    def test_Ma_05(self):
        self.assertAlmostEqual(_T_ratio(0.5, GAMMA_AIR), 2.4 / 2.1, delta=1e-8)

    def test_Ma_20(self):
        self.assertAlmostEqual(_T_ratio(2.0, GAMMA_AIR), 2.4 / 3.6, delta=1e-8)

    def test_subsonic_above_one(self):
        for Ma in [0.1, 0.3, 0.5, 0.8]:
            self.assertGreater(_T_ratio(Ma, GAMMA_AIR), 1.0)

    def test_supersonic_below_one(self):
        for Ma in [1.1, 2.0, 3.0]:
            self.assertLess(_T_ratio(Ma, GAMMA_AIR), 1.0)


class TestPressureRatio(unittest.TestCase):
    def test_sonic(self):
        self.assertAlmostEqual(_P_ratio(1.0, GAMMA_AIR), 1.0, delta=1e-12)

    def test_Ma_05(self):
        self.assertAlmostEqual(_P_ratio(0.5, GAMMA_AIR), (1/0.5)*math.sqrt(2.4/2.1), delta=1e-6)

    def test_Ma_20(self):
        self.assertAlmostEqual(_P_ratio(2.0, GAMMA_AIR), (1/2.0)*math.sqrt(2.4/3.6), delta=1e-6)

    def test_subsonic_above_one(self):
        for Ma in [0.1, 0.3, 0.5, 0.8]:
            self.assertGreater(_P_ratio(Ma, GAMMA_AIR), 1.0)

    def test_supersonic_below_one(self):
        for Ma in [1.1, 2.0, 3.0]:
            self.assertLess(_P_ratio(Ma, GAMMA_AIR), 1.0)


class TestP0Ratio(unittest.TestCase):
    def test_sonic(self):
        self.assertAlmostEqual(_P0_ratio(1.0, GAMMA_AIR), 1.0, delta=1e-10)

    def test_always_ge_one(self):
        for Ma in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 3.0]:
            self.assertGreaterEqual(_P0_ratio(Ma, GAMMA_AIR), 1.0 - 1e-10)

    def test_Ma_05(self):
        self.assertAlmostEqual(_P0_ratio(0.5, GAMMA_AIR), 1.3398, delta=0.001)

    def test_Ma_20(self):
        self.assertAlmostEqual(_P0_ratio(2.0, GAMMA_AIR), 1.6875, delta=0.001)


class TestRhoRatio(unittest.TestCase):
    def test_sonic(self):
        self.assertAlmostEqual(_rho_ratio(1.0, GAMMA_AIR), 1.0, delta=1e-12)

    def test_consistency_with_P_and_T(self):
        for Ma in [0.3, 0.5, 0.7, 1.5, 2.0]:
            expected = _P_ratio(Ma, GAMMA_AIR) / _T_ratio(Ma, GAMMA_AIR)
            self.assertAlmostEqual(_rho_ratio(Ma, GAMMA_AIR), expected, delta=1e-10)


# ===========================================================================
# 2. Mach number inversion
# ===========================================================================

class TestSolveMa(unittest.TestCase):
    def _round_trip(self, Ma, supersonic=False, tol=1e-5):
        fp = _fanno_param(Ma, GAMMA_AIR)
        Ma_back = _solve_Ma(fp, GAMMA_AIR, supersonic=supersonic)
        self.assertAlmostEqual(Ma_back, Ma, delta=tol,
                               msg=f"round-trip failed for Ma={Ma}: got {Ma_back}")

    def test_subsonic_05(self):    self._round_trip(0.5)
    def test_subsonic_03(self):    self._round_trip(0.3)
    def test_subsonic_07(self):    self._round_trip(0.7)
    def test_subsonic_09(self):    self._round_trip(0.9)
    def test_supersonic_20(self):  self._round_trip(2.0, supersonic=True)
    def test_supersonic_30(self):  self._round_trip(3.0, supersonic=True)

    def test_target_zero_returns_sonic(self):
        self.assertAlmostEqual(_solve_Ma(0.0, GAMMA_AIR), 1.0, delta=1e-9)

    def test_negative_target_returns_sonic(self):
        self.assertAlmostEqual(_solve_Ma(-1.0, GAMMA_AIR), 1.0, delta=1e-9)


# ===========================================================================
# 3. Churchill friction factor
# ===========================================================================

class TestChurchillF(unittest.TestCase):
    def test_laminar(self):
        self.assertAlmostEqual(_churchill_f(500.0, 0.05, 1.5e-5), 64.0/500, delta=0.0005)

    def test_laminar_boundary(self):
        self.assertAlmostEqual(_churchill_f(2000.0, 0.05, 1.5e-5), 64.0/2000, delta=0.003)

    def test_turbulent_vs_colebrook(self):
        Re = 1e5; D = 0.1; eps = 1.5e-5
        f_ch = _churchill_f(Re, D, eps)
        eps_D = eps / D
        f_cb = 0.02
        for _ in range(50):
            f_cb = (1 / (-2*math.log10(eps_D/3.7 + 2.51/(Re*math.sqrt(f_cb))))) ** 2
        self.assertAlmostEqual(f_ch, f_cb, delta=0.0002)

    def test_very_low_re_guard(self):
        self.assertEqual(_churchill_f(0.5, 0.05, 1.5e-5), 64.0)

    def test_rougher_gives_higher_f(self):
        self.assertGreater(_churchill_f(1e6, 0.05, 4.6e-5), _churchill_f(1e6, 0.05, 1.5e-5))

    def test_always_positive(self):
        for Re in [10, 100, 1000, 1e4, 1e5, 1e6]:
            self.assertGreater(_churchill_f(Re, 0.05, 1.5e-5), 0.0)


# ===========================================================================
# 4. fanno_solve integration
# ===========================================================================

_BASE = dict(
    P1_Pa=5.0e5, T1_K=293.15, mdot_kgs=500.0/3600.0,
    D_m=0.0779, L_m=100.0, roughness_m=1.5e-5,
    MW_kgmol=MW_AIR, gamma=GAMMA_AIR, species="Air", N_points=100,
)


class TestFannoSolveNormal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = fanno_solve(**_BASE)

    def test_not_choked(self):
        self.assertFalse(self.res["choked"])

    def test_Ma1_subsonic(self):
        self.assertGreater(self.res["Ma1"], 0.0)
        self.assertLess(self.res["Ma1"], 1.0)

    def test_Ma2_subsonic(self):
        self.assertGreater(self.res["Ma2"], 0.0)
        self.assertLess(self.res["Ma2"], 1.0)

    def test_friction_accelerates_subsonic(self):
        self.assertGreater(self.res["Ma2"], self.res["Ma1"])

    def test_static_pressure_drops(self):
        self.assertGreater(self.res["P1_bara"], self.res["P2_bara"])

    def test_static_temperature_drops(self):
        self.assertGreater(self.res["T1_K"], self.res["T2_K"])

    def test_stagnation_pressure_drops(self):
        self.assertGreater(self.res["P01_bara"], self.res["P02_bara"])

    def test_velocity_increases(self):
        self.assertGreater(self.res["V2_ms"], self.res["V1_ms"])

    def test_L_star_exceeds_pipe_length(self):
        self.assertGreater(self.res["L_star_m"], _BASE["L_m"])

    def test_margin_positive(self):
        self.assertGreater(self.res["margin_pct"], 0.0)

    def test_dP_static_positive(self):
        self.assertGreater(self.res["dP_static_kPa"], 0.0)

    def test_dP_stag_positive(self):
        self.assertGreater(self.res["dP_stag_kPa"], 0.0)

    def test_profile_arrays_correct_length(self):
        for key in ("x_arr", "Ma_arr", "P_arr_bara", "T_arr_K", "P0_arr_bara", "V_arr_ms"):
            self.assertEqual(len(self.res[key]), _BASE["N_points"], msg=key)

    def test_profile_Ma_monotone_increasing(self):
        Ma = self.res["Ma_arr"]
        for i in range(len(Ma) - 1):
            self.assertGreaterEqual(Ma[i+1], Ma[i] - 1e-10)

    def test_profile_P_monotone_decreasing(self):
        P = self.res["P_arr_bara"]
        for i in range(len(P) - 1):
            self.assertGreaterEqual(P[i], P[i+1] - 1e-10)

    def test_profile_T_monotone_decreasing(self):
        T = self.res["T_arr_K"]
        for i in range(len(T) - 1):
            self.assertGreaterEqual(T[i], T[i+1] - 1e-10)

    def test_profile_start_matches_inlet(self):
        self.assertAlmostEqual(self.res["Ma_arr"][0],     self.res["Ma1"],     delta=1e-6)
        self.assertAlmostEqual(self.res["P_arr_bara"][0], self.res["P1_bara"], delta=1e-6)
        self.assertAlmostEqual(self.res["T_arr_K"][0],    self.res["T1_K"],    delta=1e-4)

    def test_profile_end_matches_exit(self):
        self.assertAlmostEqual(self.res["Ma_arr"][-1],     self.res["Ma2"],     delta=1e-4)
        self.assertAlmostEqual(self.res["P_arr_bara"][-1], self.res["P2_bara"], delta=1e-4)
        self.assertAlmostEqual(self.res["T_arr_K"][-1],    self.res["T2_K"],    delta=0.01)

    def test_stagnation_temperature_conserved(self):
        g = GAMMA_AIR
        T0_in = self.res["T1_K"] * (1 + (g-1)/2 * self.res["Ma1"]**2)
        for Ma_i, T_i in zip(self.res["Ma_arr"], self.res["T_arr_K"]):
            T0_i = T_i * (1 + (g-1)/2 * Ma_i**2)
            self.assertAlmostEqual(T0_i, T0_in, delta=0.02)

    def test_mass_conservation_along_profile(self):
        R_spec = R_u / MW_AIR
        A = math.pi / 4 * _BASE["D_m"]**2
        mdot = _BASE["mdot_kgs"]
        for P_bara, T_K, V_ms in zip(self.res["P_arr_bara"], self.res["T_arr_K"], self.res["V_arr_ms"]):
            rho = P_bara * 1e5 / (R_spec * T_K)
            self.assertAlmostEqual(rho * V_ms * A, mdot, delta=mdot * 0.001)


class TestFannoSolveChoked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = fanno_solve(**dict(_BASE, L_m=1e6))

    def test_choked_flag(self):
        self.assertTrue(self.res["choked"])

    def test_exit_Ma_is_sonic(self):
        self.assertAlmostEqual(self.res["Ma2"], 1.0, delta=1e-9)

    def test_exit_pressure_equals_critical(self):
        P_star = self.res["P1_bara"] * 1e5 / _P_ratio(self.res["Ma1"], GAMMA_AIR)
        self.assertAlmostEqual(self.res["P2_bara"], P_star / 1e5, delta=1e-3)


class TestFannoSolveZeroLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = fanno_solve(**dict(_BASE, L_m=0.001))

    def test_not_choked(self):
        self.assertFalse(self.res["choked"])

    def test_negligible_pressure_drop(self):
        self.assertAlmostEqual(self.res["dP_static_kPa"], 0.0, delta=0.1)

    def test_Ma_nearly_unchanged(self):
        self.assertAlmostEqual(self.res["Ma1"], self.res["Ma2"], delta=1e-4)


class TestFannoSolveKnownMach(unittest.TestCase):
    def test_known_Ma1_and_fanno_param(self):
        g = GAMMA_AIR; T1 = 300.0; P1 = 2.0e5; D = 0.05
        R_spec = R_u / MW_AIR
        a1 = math.sqrt(g * R_spec * T1)
        rho1 = P1 / (R_spec * T1)
        A = math.pi / 4 * D**2
        mdot = rho1 * 0.5 * a1 * A  # design for Ma1 = 0.5
        res = fanno_solve(P1_Pa=P1, T1_K=T1, mdot_kgs=mdot,
                          D_m=D, L_m=0.001, roughness_m=1.5e-5,
                          MW_kgmol=MW_AIR, gamma=g, species="Air")
        self.assertAlmostEqual(res["Ma1"], 0.5, delta=0.001)
        self.assertAlmostEqual(_fanno_param(res["Ma1"], g), 1.0691, delta=0.002)


class TestFannoSolveHigherMa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # DN20 (D=0.0209 m) at 500 kg/h gives Ma ≈ 0.20
        cls.res = fanno_solve(**dict(_BASE, D_m=0.0209, L_m=20.0))

    def test_Ma1_above_01(self):
        self.assertGreater(self.res["Ma1"], 0.1)

    def test_L_star_finite_and_positive(self):
        self.assertGreater(self.res["L_star_m"], 0.0)
        self.assertTrue(math.isfinite(self.res["L_star_m"]))


# ===========================================================================
# 5. Gas species table
# ===========================================================================

class TestGasSpecies(unittest.TestCase):
    def test_standard_gases_have_positive_mw_and_gamma(self):
        for name, (mw, g, _) in _GASES.items():
            if name == "Custom":
                continue
            self.assertGreater(mw, 0.0, msg=f"{name} MW")
            self.assertGreater(g,  1.0, msg=f"{name} gamma")

    def test_monatomic_gamma_near_5_3(self):
        for name in ("Helium", "Argon"):
            _, g, _ = _GASES[name]
            self.assertAlmostEqual(g, 5/3, delta=0.01, msg=name)

    def test_diatomic_gamma_near_1_4(self):
        for name in ("Air", "Nitrogen", "Hydrogen", "Oxygen"):
            _, g, _ = _GASES[name]
            self.assertAlmostEqual(g, 1.4, delta=0.01, msg=name)

    def test_custom_all_none(self):
        mw, g, cp = _GASES["Custom"]
        self.assertIsNone(mw)
        self.assertIsNone(g)
        self.assertIsNone(cp)


# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"PASSED {passed}/{result.testsRun}")
    if result.failures or result.errors:
        sys.exit(1)
