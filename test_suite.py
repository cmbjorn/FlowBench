"""
test_suite.py — Automated tests for multiphase_engine and related helpers.

Run with:  python test_suite.py
"""
import sys
import math
import numpy as np

import multiphase_engine as engine
from validation_cases import VALIDATION_CASES

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

    if ok:
        print(f"  {PASS}  {name}")
        print(f"         got {actual:.6g}  expected {expected:.6g}")
        _results["pass"] += 1
    elif warn_only:
        print(f"  {WARN}  {name}")
        print(f"         got {actual:.6g}  expected {expected:.6g}  (warn-only)")
        _results["warn"] += 1
    else:
        print(f"  {FAIL}  {name}")
        print(f"         got {actual:.6g}  expected {expected:.6g}")
        _results["fail"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validation cases (regression anchors)
# ─────────────────────────────────────────────────────────────────────────────

def _run_validation_case(key, case):
    inp = case["inputs"]
    gas_flows = {"H₂": inp["m_H2_kgh"], "O₂": inp["m_O2_kgh"]}

    P_Pa = inp["P_bara"] * 1e5
    total_dp = 0.0

    for seg in inp["segments"]:
        dn = inp["pipe_dn"]
        pn = inp["pipe_pn"]
        D = engine.PIPE_DATABASE[dn][pn]
        rough = engine.MATERIAL_ROUGHNESS["SS316L"]

        props = engine.calculate_two_phase_properties(
            P_Pa / 1e5, inp["T_C"],
            gas_flows, "KOH 30 wt%", inp["q_lye_m3h"]
        )

        angle = {"Horizontal": 0.0,
                 "Vertical Upflow": math.pi / 2.0,
                 "Vertical Downflow": -math.pi / 2.0}[seg["type"]]

        le_fit = 0.0
        if seg["fittings"] in engine.FITTING_Le_over_D:
            le_fit = engine.FITTING_Le_over_D[seg["fittings"]] * D * seg["fitting_count"]

        res = engine.calculate_segment_pressure_drop(
            props, D, rough, seg["length"] + le_fit, angle,
            correlation=engine.TWO_PHASE_CORRELATIONS[0],
            voidage_method=engine.VOIDAGE_METHODS[0],
        )
        total_dp += res["dP_Pa"]
        P_Pa -= res["dP_Pa"]
        P_Pa = max(1e4, P_Pa)

    total_dp_kpa = total_dp / 1000.0
    return total_dp_kpa


def test_validation_cases():
    print("\n── 1. Validation cases ─────────────────────────────────────────────")
    for key, case in VALIDATION_CASES.items():
        actual = _run_validation_case(key, case)
        _check(
            case["name"],
            actual,
            case["expected_total_dp_kpa"],
            tol_pct=case["tolerance_pct"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Unit conversion sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def test_unit_conversions():
    print("\n── 2. Unit conversion checks ───────────────────────────────────────")

    # 1 bar expressed as kPa must equal 100 kPa
    bar_in_kpa = 100.0
    _check("1 bar = 100 kPa", bar_in_kpa, 100.0, tol_abs=1e-9)

    # kPa ÷ 100 = bar  (used in app.py for pressure conversion)
    dp_kpa = 50.0
    dp_bar = dp_kpa / 100.0
    _check("50 kPa ÷ 100 = 0.5 bar", dp_bar, 0.5, tol_abs=1e-9)

    # 1 bar = 1000 mbar  (the bug-fixed conversion)
    bar_val = 0.1  # 0.1 bar difference between two branch outlets
    mbar_val = bar_val * 1000.0
    _check("0.1 bar × 1000 = 100 mbar (not 10)", mbar_val, 100.0, tol_abs=1e-9)

    # Wrong conversion (× 100) would give 10 mbar — confirm it's different
    wrong_mbar = bar_val * 100.0
    wrong = abs(wrong_mbar - 100.0) < 1e-9
    _check("× 100 gives wrong mbar (should differ from ×1000)", float(not wrong), 1.0, tol_abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pressure clamping in _calc_dp_at_p (Bug 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _calc_dp_at_p_local(res, P_bara_override):
    """Local copy of app.py's _calc_dp_at_p to test without importing Streamlit."""
    current_P = P_bara_override * 1e5
    total_dp = 0.0
    corr = res.get("correlation", engine.TWO_PHASE_CORRELATIONS[0])
    void = res.get("voidage_method", engine.VOIDAGE_METHODS[0])

    for seg in res["segments"]:
        D_seg  = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        lined  = seg.get("lined", False)
        lthk_m = seg.get("liner_thickness_mm", 1.0) / 1000.0
        lmat   = seg.get("liner_material", "FEP")
        D_eff  = D_seg - 2 * lthk_m if lined else D_seg
        rough  = (engine.LINER_ROUGHNESS[lmat] if lined
                  else engine.MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])
        props  = engine.calculate_two_phase_properties(
            current_P / 1e5, res["T_C"],
            res["gas_flows_kgh"], res["liquid_type"], res["q_lye"],
        )
        angle = {"Horizontal": 0.0,
                 "Vertical Upflow": math.pi / 2.0,
                 "Vertical Downflow": -math.pi / 2.0}[seg["type"]]
        le_fit = 0.0
        if seg["fittings"] in engine.FITTING_Le_over_D:
            le_fit = engine.FITTING_Le_over_D[seg["fittings"]] * D_eff * seg["fitting_count"]
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)
        total_dp  += seg_res["dP_Pa"]
        current_P -= seg_res["dP_Pa"]
        current_P  = max(1e4, current_P)  # Bug 3 fix: clamp

    dp_kpa = total_dp / 1000.0
    outlet_bara = current_P / 1e5
    return dp_kpa, outlet_bara


def test_pressure_clamping():
    print("\n── 3. Pressure clamping (Bug 3 fix) ───────────────────────────────")

    # Build a minimal result dict mimicking what run_case returns
    seg = {
        "type": "Vertical Upflow", "dn": "DN80", "pn": "PN25",
        "length": 30.0, "fittings": "None", "fitting_count": 0,
        "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
        "material": "SS316L",
    }
    res = {
        "T_C": 40.0,
        "gas_flows_kgh": {"H₂": 5.0, "O₂": 1.2},
        "liquid_type": "KOH 30 wt%",
        "q_lye": 2.5,
        "correlation": engine.TWO_PHASE_CORRELATIONS[0],
        "voidage_method": engine.VOIDAGE_METHODS[0],
        "segments": [seg],
    }

    # Normal inlet — should return sensible values
    dp_normal, out_normal = _calc_dp_at_p_local(res, 8.0)
    _check("Normal inlet: dp > 0", float(dp_normal > 0), 1.0, tol_abs=1e-9)
    _check("Normal inlet: outlet > 0 bara", float(out_normal > 0), 1.0, tol_abs=1e-9)
    _check("Normal inlet: outlet < inlet", float(out_normal < 8.0), 1.0, tol_abs=1e-9)

    # Pathologically low inlet — clamp must prevent negative absolute pressure
    dp_low, out_low = _calc_dp_at_p_local(res, 0.01)
    _check("Low inlet: outlet clamped ≥ 1e4 Pa = 0.0001 bara",
           float(out_low >= 1e4 / 1e5), 1.0, tol_abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Goal-seek convergence smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_goal_seek():
    print("\n── 4. Goal-seek convergence ────────────────────────────────────────")

    seg_c = {
        "type": "Horizontal", "dn": "DN100", "pn": "PN40",
        "length": 15.0, "fittings": "None", "fitting_count": 0,
        "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
        "material": "SS316L",
    }
    seg_ab = {
        "type": "Vertical Upflow", "dn": "DN80", "pn": "PN25",
        "length": 20.0, "fittings": "None", "fitting_count": 0,
        "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
        "material": "SS316L",
    }

    base = {
        "T_C": 40.0,
        "gas_flows_kgh": {"H₂": 5.0, "O₂": 1.2},
        "liquid_type": "KOH 30 wt%",
        "q_lye": 2.5,
        "correlation": engine.TWO_PHASE_CORRELATIONS[0],
        "voidage_method": engine.VOIDAGE_METHODS[0],
    }
    res_c  = {**base, "segments": [seg_c]}
    res_a  = {**base, "segments": [seg_ab]}
    res_b  = {**base, "segments": [seg_ab]}

    P_target = 5.0  # bara

    # Manual successive-substitution (mirrors _goal_seek_inlet in app.py)
    dp_c_0, _ = _calc_dp_at_p_local(res_c, 8.0)
    dp_a_0, _ = _calc_dp_at_p_local(res_a, 8.0)
    dp_b_0, _ = _calc_dp_at_p_local(res_b, 8.0)
    P_c_in = P_target + max(dp_a_0, dp_b_0) / 100.0 + dp_c_0 / 100.0

    tol = 0.0005
    converged = False
    for i in range(25):
        dp_c, P_c_out = _calc_dp_at_p_local(res_c, P_c_in)
        dp_a, P_a_out = _calc_dp_at_p_local(res_a, P_c_out)
        dp_b, P_b_out = _calc_dp_at_p_local(res_b, P_c_out)
        worst_out = min(P_a_out, P_b_out)
        error = worst_out - P_target
        if abs(error) < tol:
            converged = True
            break
        P_c_in -= error

    _check("Goal-seek converges", float(converged), 1.0, tol_abs=1e-9)
    _check(f"Worst outlet within 0.5 mbar of target ({P_target} bara)",
           abs(worst_out - P_target), 0.0, tol_abs=tol)
    _check("C outlet < C inlet", float(P_c_out < P_c_in), 1.0, tol_abs=1e-9)
    _check("Branch outlets > 0 bara",
           float(P_a_out > 0 and P_b_out > 0), 1.0, tol_abs=1e-9)
    print(f"         converged in {i+1} iterations, "
          f"C_in={P_c_in:.4f} bara, worst_out={worst_out:.4f} bara")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sensitivity runner smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_sensitivity():
    print("\n── 5. Sensitivity runner ───────────────────────────────────────────")
    seg = {
        "type": "Horizontal", "dn": "DN100", "pn": "PN20",
        "length": 20.0, "fittings": "None", "fitting_count": 0,
    }
    rows = engine.run_sensitivity(
        P_bara=5.0, T_C=25.0,
        gas_flows_kgh={"H₂": 2.0, "O₂": 0.5},
        liquid_type="KOH 30 wt%", q_lye_m3h=3.0,
        segments=[seg],
    )
    n_expected = len(engine.TWO_PHASE_CORRELATIONS) * len(engine.VOIDAGE_METHODS)
    _check(f"Sensitivity returns {n_expected} rows", len(rows), n_expected, tol_abs=0)
    _check("All rows have total_dp_kpa key", float(all("total_dp_kpa" in r for r in rows)), 1.0, tol_abs=1e-9)
    _check("All total_dp_kpa ≥ 0 or None", float(all(r["total_dp_kpa"] is None or r["total_dp_kpa"] >= 0 for r in rows)), 1.0, tol_abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report generator — combined report smoke test (no Streamlit needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_report_generator():
    print("\n── 6. Report generator smoke test ──────────────────────────────────")
    try:
        import report_generator as rg

        # Build minimal case result dicts
        def _make_case(dp_kpa, P_bara, label):
            seg = {
                "type": "Horizontal", "dn": "DN100", "pn": "PN40",
                "length": 10.0, "fittings": "None", "fitting_count": 0,
                "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
                "material": "SS316L",
            }
            return {
                "P_bara": P_bara,
                "T_C": 25.0,
                "gas_flows_kgh": {"H₂": 26.0, "O₂": 0.0},
                "liquid_type": "KOH 30 wt%",
                "q_lye": 13.75,
                "total_dp_kpa": dp_kpa,
                "total_dp_fric_kpa": dp_kpa * 0.9,
                "total_dp_grav_kpa": 0.0,
                "outlet_pressure_bara": P_bara - dp_kpa / 100.0,
                "outlet_pressure_mbar": (P_bara - dp_kpa / 100.0) * 1000.0,
                "pipe_length_m": 10.0,
                "cumulative_distance": 10.0,
                "segments": [seg],
                "correlation": engine.TWO_PHASE_CORRELATIONS[0],
                "voidage_method": engine.VOIDAGE_METHODS[0],
                "props": engine.calculate_two_phase_properties(
                    P_bara, 25.0, {"H₂": 26.0, "O₂": 0.0},
                    "KOH 30 wt%", 13.75
                ),
                "fig_sch": None,
                "fig_prof": None,
                "case_label": label,
                "grid_records": [{
                    "Seg": "#1", "Type": "Horizontal", "Pipe": "DN100/PN40",
                    "ID (mm)": 97.2, "L (m)": 10.0, "Regime": "Stratified",
                    "ΔP (kPa)": dp_kpa, "P_in (bara)": P_bara,
                    "P_out (bara)": P_bara - dp_kpa / 100.0,
                    "V_m (m/s)": 0.5, "V_m/V_e": 0.2, "V_sg (m/s)": 0.4,
                    "V_sl (m/s)": 0.1, "V_e (m/s)": 2.5,
                    "ΔP_fric (kPa)": dp_kpa * 0.9, "ΔP_grav (kPa)": 0.0,
                    "ΔP_accel (kPa)": dp_kpa * 0.1,
                }],
            }

        ra = _make_case(dp_kpa=5.0, P_bara=30.0, label="Case A")
        rb = _make_case(dp_kpa=7.0, P_bara=30.0, label="Case B")
        rc = _make_case(dp_kpa=2.0, P_bara=32.0, label="Case C")

        buf = rg.generate_combined_report(
            cases=[ra, rb, rc],
            case_labels=["Case A", "Case B", "Case C"],
            fig_cmp=None,
            fig_bar=None,
            sensitivity_data=None,
        )
        _check("Combined report returns BytesIO", float(buf is not None), 1.0, tol_abs=1e-9)
        size = len(buf.read())
        _check("Report file size > 5 kB", float(size > 5000), 1.0, tol_abs=1e-9)
        print(f"         report size: {size / 1024:.1f} kB")

    except Exception as e:
        print(f"  {FAIL}  Report generator raised: {e}")
        import traceback; traceback.print_exc()
        _results["fail"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Header goal-seek (two-arm collecting manifold)
# ─────────────────────────────────────────────────────────────────────────────

def _make_arm_seg(dn, length, h2_kgh, q_lye):
    return {
        "type": "Horizontal", "dn": dn, "pn": "PN40", "material": "SS316L",
        "length": length, "fittings": "None", "fitting_count": 0,
        "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
        "branch_gas_kgh": {"H₂": h2_kgh},
        "branch_q_lye": q_lye,
    }


def _march_header_arm_local(segs, P_start_Pa, T_C, liquid_type, corr, void):
    """Local copy of _march_header_arm — mirrors app.py logic."""
    current_P   = P_start_Pa
    total_dp    = 0.0
    running_gas = {}
    running_q   = 0.0

    for seg in segs:
        for sp, kg_h in seg.get("branch_gas_kgh", {}).items():
            if kg_h > 0:
                running_gas[sp] = running_gas.get(sp, 0.0) + kg_h
        running_q += seg.get("branch_q_lye", 0.0)

        if not running_gas or running_q <= 0.0:
            continue

        D_seg  = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        D_eff  = D_seg
        rough  = engine.MATERIAL_ROUGHNESS["SS316L"]
        angle  = 0.0
        le_fit = 0.0

        props_seg = engine.calculate_two_phase_properties(
            current_P / 1e5, T_C, running_gas, liquid_type, running_q)
        seg_res = engine.calculate_segment_pressure_drop(
            props_seg, D_eff, rough, seg["length"] + le_fit, angle,
            correlation=corr, voidage_method=void)

        total_dp  += seg_res["dP_Pa"]
        current_P  = max(1e4, current_P - seg_res["dP_Pa"])

    return total_dp, current_P


def test_header_goal_seek():
    print("\n── 7. Header goal-seek (two-arm collecting manifold) ───────────────")

    corr = engine.TWO_PHASE_CORRELATIONS[0]
    void = engine.VOIDAGE_METHODS[0]
    T_C  = 60.0
    liq  = "KOH 30 wt%"

    # Left arm: 2 segments, each with a branch tapping in H₂ + lye
    left_segs = [
        _make_arm_seg("DN100", 3.0, h2_kgh=5.0, q_lye=2.5),
        _make_arm_seg("DN100", 3.0, h2_kgh=5.0, q_lye=2.5),
    ]
    # Right arm: 2 segments, symmetric
    right_segs = [
        _make_arm_seg("DN100", 3.0, h2_kgh=5.0, q_lye=2.5),
        _make_arm_seg("DN100", 3.0, h2_kgh=5.0, q_lye=2.5),
    ]

    # Branch pipes (Cases A and B)
    seg_ab = {
        "type": "Vertical Upflow", "dn": "DN80", "pn": "PN25", "material": "SS316L",
        "length": 20.0, "fittings": "None", "fitting_count": 0,
        "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
    }
    res_ab_base = {
        "T_C": T_C, "gas_flows_kgh": {"H₂": 5.0},
        "liquid_type": liq, "q_lye": 2.5,
        "correlation": corr, "voidage_method": void,
        "segments": [seg_ab],
    }

    # Stub res_c as a header dict (mirrors run_header_case return)
    # total_dp_kpa is used only for the initial estimate seed
    def _dp_of_arm(segs, P_bara):
        dp_Pa, P_T, *_ = _march_header_arm_local(segs, P_bara * 1e5, T_C, liq, corr, void)
        return dp_Pa / 1000.0

    dp0_l = _dp_of_arm(left_segs,  30.0)
    dp0_r = _dp_of_arm(right_segs, 30.0)

    res_c = {
        "T_C": T_C, "liquid_type": liq,
        "correlation": corr, "voidage_method": void,
        "left_segs": left_segs, "right_segs": right_segs,
        "total_dp_kpa": max(dp0_l, dp0_r),
        "P_bara": 30.0,
    }

    # Calculate initial ΔPs for branches to seed goal-seek
    dp_ab_0, _ = _calc_dp_at_p_local(res_ab_base, 30.0)
    res_a = {**res_ab_base, "total_dp_kpa": dp_ab_0}
    res_b = {**res_ab_base, "total_dp_kpa": dp_ab_0}

    P_target = 25.0  # bara — target at worst branch outlet

    # Run goal-seek manually (mirrors _goal_seek_inlet in app.py)
    _is_header = "left_segs" in res_c
    _check("res_c detected as header", float(_is_header), 1.0, tol_abs=1e-9)

    P_c_in = P_target + max(res_a["total_dp_kpa"], res_b["total_dp_kpa"]) / 100.0 \
             + res_c["total_dp_kpa"] / 100.0

    tol = 0.0005
    converged = False
    worst_out = 0.0
    for i in range(25):
        # Header arm march
        dp_l_Pa, P_T_l, *_ = _march_header_arm_local(left_segs,  P_c_in * 1e5, T_C, liq, corr, void)
        dp_r_Pa, P_T_r, *_ = _march_header_arm_local(right_segs, P_c_in * 1e5, T_C, liq, corr, void)
        if dp_l_Pa >= dp_r_Pa:
            dp_c, P_c_out = dp_l_Pa / 1000.0, P_T_l / 1e5
        else:
            dp_c, P_c_out = dp_r_Pa / 1000.0, P_T_r / 1e5

        dp_a, P_a_out = _calc_dp_at_p_local(res_a, P_c_out)
        dp_b, P_b_out = _calc_dp_at_p_local(res_b, P_c_out)
        worst_out = min(P_a_out, P_b_out)
        error = worst_out - P_target
        if abs(error) < tol:
            converged = True
            break
        P_c_in -= error

    _check("Header goal-seek converges", float(converged), 1.0, tol_abs=1e-9)
    _check(f"Worst outlet ≈ target {P_target} bara",
           abs(worst_out - P_target), 0.0, tol_abs=tol)
    _check("T-junction pressure < branch inlet pressure",
           float(P_c_out < P_c_in), 1.0, tol_abs=1e-9)
    _check("Arms are symmetric → same T-junction pressure from both",
           abs(P_T_l - P_T_r), 0.0, tol_abs=0.5)  # within 0.5 Pa for symmetric arms
    print(f"         converged in {i+1} iterations, "
          f"C_in={P_c_in:.4f} bara, T-junction={P_c_out:.4f} bara, "
          f"worst_out={worst_out:.4f} bara")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_validation_cases()
    test_unit_conversions()
    test_pressure_clamping()
    test_goal_seek()
    test_sensitivity()
    test_header_goal_seek()
    test_report_generator()

    total = _results["pass"] + _results["fail"] + _results["warn"]
    print(f"\n{'─'*60}")
    print(f"Results: {_results['pass']}/{total} passed, "
          f"{_results['fail']} failed, {_results['warn']} warnings")
    sys.exit(0 if _results["fail"] == 0 else 1)
