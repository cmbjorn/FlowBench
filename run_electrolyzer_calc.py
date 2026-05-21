#!/usr/bin/env python3
"""
Electrolyzer system hydraulic study
H2 side: Case A (H2 branch) + Case C (H2 header)
O2 side: Case B (O2 branch) + Case D (O2 header)
Goal-seek: P_T-junction = 16.5 bara
"""
import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import multiphase_engine as engine
import report_generator
from io import BytesIO

# ─── Process parameters ───────────────────────────────────────────────────────
T_C          = 80.0
LIQUID_TYPE  = "KOH 30 wt%"
P_TARGET     = 16.5       # bara at T-junction (T-segment length = 0)

GAS_A = {"H₂":  26.0}    # kg/h per stack
GAS_B = {"O₂": 206.0}    # kg/h per stack
Q_LYE = 15.0              # m³/h KOH per stack

LABEL_A = "H₂ Branch (A)"
LABEL_B = "O₂ Branch (B)"

# ─── Pipe specs ──────────────────────────────────────────────────────────────
DN_BR   = "DN50";   PN_BR  = "PN20"
DN_HDR  = "DN250";  PN_HDR = "PN20"
LINER   = "FEP";    LINER_THK_MM = 1.5
MAT     = "SS316L"

# ─── Branch line segments (identical geometry for A and B) ───────────────────
SEGS = [
    {"type": "Horizontal",      "dn": DN_BR, "pn": PN_BR, "material": MAT,
     "length": 2.5, "fittings": "None", "fitting_count": 0,
     "lined": True, "liner_material": LINER, "liner_thickness_mm": LINER_THK_MM},
    {"type": "Vertical Upflow", "dn": DN_BR, "pn": PN_BR, "material": MAT,
     "length": 4.5, "fittings": "None", "fitting_count": 0,
     "lined": True, "liner_material": LINER, "liner_thickness_mm": LINER_THK_MM},
    {"type": "Horizontal",      "dn": DN_BR, "pn": PN_BR, "material": MAT,
     "length": 2.0, "fittings": "None", "fitting_count": 0,
     "lined": True, "liner_material": LINER, "liner_thickness_mm": LINER_THK_MM},
]

# ─── Header spec (no liner) ──────────────────────────────────────────────────
HDR_SPEC = {"dn": DN_HDR, "pn": PN_HDR, "material": MAT,
            "lined": False, "liner_material": "FEP", "liner_thickness_mm": 1.0,
            "fittings": "None", "fitting_count": 0}

# 8 taps: 4 per side at 1, 2, 3, 4 m from T
LEFT_TAPS  = [1.0, 2.0, 3.0, 4.0]
RIGHT_TAPS = [1.0, 2.0, 3.0, 4.0]
N_TOTAL    = len(LEFT_TAPS) + len(RIGHT_TAPS)   # 8


# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH LINE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════
def calc_branch(gas_flows, P_in_bara, corr="Beggs-Brill", void="Homogeneous"):
    """March through 3 branch segments. Returns a result dict."""
    P = P_in_bara * 1e5
    total_dp = dp_fric = dp_grav = 0.0
    records = []
    px = [0.0]; py = [P_in_bara]
    cum_len = 0.0

    angle_map = {"Horizontal": 0.0, "Vertical Upflow": np.pi/2.0,
                 "Vertical Downflow": -np.pi/2.0}

    for i, seg in enumerate(SEGS):
        D_nom = engine.PIPE_DATABASE[seg["dn"]][seg["pn"]]
        lthk  = seg["liner_thickness_mm"] / 1000.0
        D_eff = D_nom - 2 * lthk if seg["lined"] else D_nom
        rough = (engine.LINER_ROUGHNESS[seg["liner_material"]] if seg["lined"]
                 else engine.MATERIAL_ROUGHNESS[seg["material"]])
        angle = angle_map[seg["type"]]

        props   = engine.calculate_two_phase_properties(
            P / 1e5, T_C, gas_flows, LIQUID_TYPE, Q_LYE)
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_eff, rough, seg["length"], angle,
            correlation=corr, voidage_method=void)

        dP    = seg_res["dP_Pa"]
        P_out = max(1e4, P - dP)
        V_m   = seg_res["Vsg"] + seg_res["Vsl"]
        V_e, _ = engine.calculate_erosion_velocity(
            props["rho_g"], props["rho_l"], props["x_gas"])

        cum_len += seg["length"]
        records.append({
            "Seg":           f"#{i+1}",
            "Pipe":          f"{seg['dn']}/{seg['pn']}",
            "ID (mm)":       round(D_eff * 1000, 1),
            "Type":          seg["type"],
            "L (m)":         seg["length"],
            "Regime":        seg_res["regime"],
            "ΔP (kPa)":      round(dP / 1000, 3),
            "P_in (bara)":   round(P / 1e5, 4),
            "P_out (bara)":  round(P_out / 1e5, 4),
            "V_m (m/s)":     round(V_m, 3),
            "V_e (m/s)":     round(V_e, 3),
            "V_m/V_e":       round(V_m / V_e if V_e > 0 else 0.0, 3),
            "α (void)":      round(props["alpha"], 4),
            "ΔP_fric (kPa)": round(seg_res["dP_fric_Pa"] / 1000, 3),
            "ΔP_grav (kPa)": round(seg_res["dP_grav_Pa"] / 1000, 3),
        })
        total_dp += dP
        dp_fric  += seg_res["dP_fric_Pa"]
        dp_grav  += seg_res["dP_grav_Pa"]
        px.append(cum_len); py.append(P_out / 1e5)
        P = P_out

    try:
        props_out = engine.calculate_two_phase_properties(
            P / 1e5, T_C, gas_flows, LIQUID_TYPE, Q_LYE)
    except Exception:
        props_out = {}

    return {
        "P_bara":               P_in_bara,
        "T_C":                  T_C,
        "total_dp_kpa":         total_dp / 1000.0,
        "outlet_pressure_bara": P / 1e5,
        "outlet_pressure_mbar": P / 100.0,
        "total_dp_fric_kpa":    dp_fric / 1000.0,
        "total_dp_grav_kpa":    dp_grav / 1000.0,
        "pipe_length_m":        cum_len,
        "cumulative_distance":  cum_len,
        "liquid_type":          LIQUID_TYPE,
        "gas_flows_kgh":        gas_flows,
        "q_lye":                Q_LYE,
        "segments":             SEGS,
        "grid_records":         records,
        "correlation":          corr,
        "voidage_method":       void,
        "props":                props_out,
        "pressure_profile_x":   px,
        "pressure_profile_y":   py,
        "segment_regimes":      [r["Regime"] for r in records],
        "fig_sch":              None,
        "fig_prof":             None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER ARM MARCH
# ═══════════════════════════════════════════════════════════════════════════════
def _march_arm(tap_dists, gas_per_tap, liq_per_tap, P_start_Pa,
               corr="Beggs-Brill", void="Homogeneous"):
    if not tap_dists or not gas_per_tap or liq_per_tap <= 0:
        return 0.0, P_start_Pa, 0.0, 0.0, []

    dists = sorted(tap_dists, reverse=True)
    n = len(dists)
    boundaries = dists + [0.0]

    D_nom = engine.PIPE_DATABASE[HDR_SPEC["dn"]][HDR_SPEC["pn"]]
    rough = engine.MATERIAL_ROUGHNESS[HDR_SPEC["material"]]

    P = P_start_Pa
    total_dp = dp_fric = dp_grav = 0.0
    rgas = {}; rliq = 0.0; records = []

    for i in range(n):
        for sp, kgh in gas_per_tap.items():
            rgas[sp] = rgas.get(sp, 0.0) + kgh
        rliq += liq_per_tap

        seg_len = boundaries[i] - boundaries[i + 1]
        if seg_len <= 1e-6:
            continue

        props   = engine.calculate_two_phase_properties(
            P / 1e5, T_C, rgas, LIQUID_TYPE, rliq)
        seg_res = engine.calculate_segment_pressure_drop(
            props, D_nom, rough, seg_len, 0.0,
            correlation=corr, voidage_method=void)

        dP    = seg_res["dP_Pa"]
        P_out = max(1e4, P - dP)
        V_m   = seg_res["Vsg"] + seg_res["Vsl"]
        V_e, _ = engine.calculate_erosion_velocity(
            props["rho_g"], props["rho_l"], props["x_gas"])

        records.append({
            "Seg":           f"#{i+1}",
            "Taps in seg":   i + 1,
            "From T (m)":    round(boundaries[i], 2),
            "To T (m)":      round(boundaries[i + 1], 2),
            "L (m)":         round(seg_len, 3),
            "Pipe":          f"{HDR_SPEC['dn']}/{HDR_SPEC['pn']}",
            "ID (mm)":       round(D_nom * 1000, 1),
            "Regime":        seg_res["regime"],
            "ΔP (kPa)":      round(dP / 1000, 3),
            "P_in (bara)":   round(P / 1e5, 4),
            "P_out (bara)":  round(P_out / 1e5, 4),
            "V_m (m/s)":     round(V_m, 3),
            "V_m/V_e":       round(V_m / V_e if V_e > 0 else 0, 3),
            "Q_gas_kgh":     round(sum(rgas.values()), 3),
            "Q_liq_m3h":     round(rliq, 3),
            "ΔP_fric (kPa)": round(seg_res["dP_fric_Pa"] / 1000, 3),
            "ΔP_grav (kPa)": round(seg_res["dP_grav_Pa"] / 1000, 3),
        })

        total_dp += dP
        dp_fric  += seg_res["dP_fric_Pa"]
        dp_grav  += seg_res["dP_grav_Pa"]
        P = P_out

    return total_dp, P, dp_fric, dp_grav, records


def calc_header_dp_at_p(gas_per_tap, liq_per_tap, P_in_bara,
                         corr="Beggs-Brill", void="Homogeneous"):
    """Both arms + T-segment (length 0). Returns (dp_worst_kpa, P_T_bara)."""
    P_start = P_in_bara * 1e5
    dp_l, P_T_l, fl, gl, rl = _march_arm(LEFT_TAPS,  gas_per_tap, liq_per_tap, P_start, corr, void)
    dp_r, P_T_r, fr, gr, rr = _march_arm(RIGHT_TAPS, gas_per_tap, liq_per_tap, P_start, corr, void)
    dp_worst = max(dp_l, dp_r)
    P_T      = min(P_T_l, P_T_r)
    return dp_worst / 1000.0, P_T / 1e5, (fl + fr) / 1000.0, (gl + gr) / 1000.0, rl, rr


def build_header_result(gas_per_tap, liq_per_tap, P_in_bara,
                         corr="Beggs-Brill", void="Homogeneous"):
    """Full header result dict (mirrors run_header_case return)."""
    dp_kpa, P_T, dfric, dgrav, rec_l, rec_r = calc_header_dp_at_p(
        gas_per_tap, liq_per_tap, P_in_bara, corr, void)
    dp_l, P_T_l, *_ = _march_arm(LEFT_TAPS,  gas_per_tap, liq_per_tap, P_in_bara * 1e5, corr, void)
    dp_r, P_T_r, *_ = _march_arm(RIGHT_TAPS, gas_per_tap, liq_per_tap, P_in_bara * 1e5, corr, void)
    worst = "Left" if dp_l >= dp_r else "Right"
    n_total = len(LEFT_TAPS) + len(RIGHT_TAPS)
    total_gas = {sp: kgh * n_total for sp, kgh in gas_per_tap.items()}
    total_liq = liq_per_tap * n_total

    try:
        props_out = engine.calculate_two_phase_properties(
            P_T, T_C, total_gas, LIQUID_TYPE, total_liq)
    except Exception:
        props_out = {}

    all_recs = ([dict(r, Seg=f"L{r['Seg']}") for r in rec_l] +
                [dict(r, Seg=f"R{r['Seg']}") for r in rec_r])

    return {
        "P_bara":               P_in_bara,
        "T_C":                  T_C,
        "total_dp_kpa":         dp_kpa,
        "outlet_pressure_bara": P_T,
        "outlet_pressure_mbar": P_T * 1000.0,
        "P_T_bara":             P_T,
        "P_separator_bara":     P_T,
        "dp_header_kpa":        dp_kpa,
        "dp_t_seg_kpa":         0.0,
        "total_dp_fric_kpa":    dfric,
        "total_dp_grav_kpa":    dgrav,
        "pipe_length_m":        max(LEFT_TAPS + RIGHT_TAPS, default=0.0),
        "cumulative_distance":  max(LEFT_TAPS + RIGHT_TAPS, default=0.0),
        "liquid_type":          LIQUID_TYPE,
        "gas_flows_kgh":        total_gas,
        "q_lye":                total_liq,
        "props":                props_out,
        "segments":             [],
        "grid_records":         all_recs,
        "correlation":          corr,
        "voidage_method":       void,
        "fig_sch":              None,
        "fig_prof":             None,
        "left_taps":            LEFT_TAPS,
        "right_taps":           RIGHT_TAPS,
        "gas_per_tap":          gas_per_tap,
        "liq_per_tap":          liq_per_tap,
        "header_pipe":          HDR_SPEC,
        "t_seg":                {"dn": DN_HDR, "pn": PN_HDR, "material": MAT,
                                 "length": 0.0, "fittings": "None", "fitting_count": 0},
        "worst_arm":            worst,
        "n_left":               len(LEFT_TAPS),
        "n_right":              len(RIGHT_TAPS),
        "dp_left_kpa":          dp_l / 1000.0,
        "dp_right_kpa":         dp_r / 1000.0,
        "P_T_left_bara":        P_T_l / 1e5,
        "P_T_right_bara":       P_T_r / 1e5,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL SEEK
# ═══════════════════════════════════════════════════════════════════════════════
def goal_seek_stack(gas_flows, gas_per_tap, liq_per_tap, P_target,
                    tol=0.0005, max_iter=30,
                    corr="Beggs-Brill", void="Homogeneous"):
    r0 = calc_branch(gas_flows, P_target + 0.5, corr, void)
    dp_h, _ = calc_header_dp_at_p(gas_per_tap, liq_per_tap,
                                   r0["outlet_pressure_bara"], corr, void)[:2]
    P_line_in = P_target + r0["total_dp_kpa"] / 100.0 + dp_h

    dp_line = dp_hdr = 0.0
    P_line_out = P_sep = 0.0
    for i in range(max_iter):
        r = calc_branch(gas_flows, P_line_in, corr, void)
        dp_hdr, P_sep = calc_header_dp_at_p(
            gas_per_tap, liq_per_tap, r["outlet_pressure_bara"], corr, void)[:2]
        dp_line    = r["total_dp_kpa"]
        P_line_out = r["outlet_pressure_bara"]
        error = P_sep - P_target
        if abs(error) < tol:
            return {"P_line_in": P_line_in, "P_line_out": P_line_out,
                    "P_sep": P_sep, "dp_line": dp_line, "dp_hdr": dp_hdr,
                    "iterations": i + 1, "converged": True, "results_line": r}
        P_line_in -= error

    r = calc_branch(gas_flows, P_line_in, corr, void)
    return {"P_line_in": P_line_in, "P_line_out": r["outlet_pressure_bara"],
            "P_sep": P_sep, "dp_line": dp_line, "dp_hdr": dp_hdr,
            "iterations": max_iter,
            "converged": abs(P_sep - P_target) < tol * 20,
            "results_line": r}


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ELECTROLYZER PIPING HYDRAULIC STUDY")
print(f"Target P at T-junction: {P_TARGET} bara")
print(f"T = {T_C}°C  |  Liquid: {LIQUID_TYPE}")
print(f"H₂/stack: {GAS_A['H₂']} kg/h  |  O₂/stack: {GAS_B['O₂']} kg/h")
print(f"KOH/stack: {Q_LYE} m³/h  |  Stacks per header: {N_TOTAL}")
print("=" * 60)

# ─── H2 side ──────────────────────────────────────────────────────────────────
print("\n── H₂ side (A + C) ──────────────────────────────────────")
gsr_h2 = goal_seek_stack(GAS_A, GAS_A, Q_LYE, P_TARGET)
P_in_A = gsr_h2["P_line_in"]
print(f"  Converged: {gsr_h2['converged']}  ({gsr_h2['iterations']} iter)")
print(f"  P_inlet_A  = {P_in_A:.4f} bara")
print(f"  P_outlet_A = {gsr_h2['P_line_out']:.4f} bara  (ΔP_A = {gsr_h2['dp_line']:.3f} kPa)")
print(f"  Header C ΔP = {gsr_h2['dp_hdr']:.3f} kPa")
print(f"  P_T (H₂)   = {gsr_h2['P_sep']:.4f} bara")

res_a = calc_branch(GAS_A, P_in_A)
res_c = build_header_result(GAS_A, Q_LYE, gsr_h2["P_line_out"])

# ─── O2 side ──────────────────────────────────────────────────────────────────
print("\n── O₂ side (B + D) ──────────────────────────────────────")
gsr_o2 = goal_seek_stack(GAS_B, GAS_B, Q_LYE, P_TARGET)
P_in_B = gsr_o2["P_line_in"]
print(f"  Converged: {gsr_o2['converged']}  ({gsr_o2['iterations']} iter)")
print(f"  P_inlet_B  = {P_in_B:.4f} bara")
print(f"  P_outlet_B = {gsr_o2['P_line_out']:.4f} bara  (ΔP_B = {gsr_o2['dp_line']:.3f} kPa)")
print(f"  Header D ΔP = {gsr_o2['dp_hdr']:.3f} kPa")
print(f"  P_T (O₂)   = {gsr_o2['P_sep']:.4f} bara")

res_b = calc_branch(GAS_B, P_in_B)
res_d = build_header_result(GAS_B, Q_LYE, gsr_o2["P_line_out"])

# ─── Stack ΔP ─────────────────────────────────────────────────────────────────
dp_stack_bara = P_in_A - P_in_B
print("\n── STACK DIFFERENTIAL PRESSURE ──────────────────────────")
print(f"  P_inlet_A (H₂) = {P_in_A:.4f} bara")
print(f"  P_inlet_B (O₂) = {P_in_B:.4f} bara")
print(f"  ΔP_stack       = {dp_stack_bara:.4f} bara")
print(f"                 = {dp_stack_bara*100:.3f} kPa")
print(f"                 = {dp_stack_bara*1000:.1f} mbar")

# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Sensitivity analysis (12 combinations each) ─────────")
sens_a = engine.run_sensitivity(P_in_A, T_C, GAS_A, LIQUID_TYPE, Q_LYE, SEGS)
sens_b = engine.run_sensitivity(P_in_B, T_C, GAS_B, LIQUID_TYPE, Q_LYE, SEGS)

_CS = {"Beggs-Brill": "BB", "Friedel": "Friedel", "Lockhart_Martinelli": "L-M",
       "Muller_Steinhagen_Heck": "MSH", "Chisholm": "Chisholm", "Kim_Mudawar": "Kim-M"}
_VS = {"Homogeneous": "Homo", "Rouhani-1 (slip)": "Rouhani-1"}

print(f"  {'Method':<22}  {'A (H₂) kPa':>12}  {'B (O₂) kPa':>12}")
print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
for sa, sb in zip(sens_a, sens_b):
    c = _CS.get(sa["correlation"], sa["correlation"])
    v = _VS.get(sa["voidage"], sa["voidage"])
    lbl = f"{c} / {v}"
    va = f"{sa['total_dp_kpa']:.3f}" if sa["ok"] else "FAIL"
    vb = f"{sb['total_dp_kpa']:.3f}" if sb["ok"] else "FAIL"
    print(f"  {lbl:<22}  {va:>12}  {vb:>12}")

va_ok = [r["total_dp_kpa"] for r in sens_a if r["ok"]]
vb_ok = [r["total_dp_kpa"] for r in sens_b if r["ok"]]
if va_ok:
    print(f"\n  H₂ ΔP range: {min(va_ok):.3f} – {max(va_ok):.3f} kPa  "
          f"(selected {res_a['total_dp_kpa']:.3f} kPa)")
if vb_ok:
    print(f"  O₂ ΔP range: {min(vb_ok):.3f} – {max(vb_ok):.3f} kPa  "
          f"(selected {res_b['total_dp_kpa']:.3f} kPa)")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Generating Word reports ──────────────────────────────")
out_dir = os.path.dirname(__file__)

# Report A (H2 branch)
buf_a = report_generator.generate_report(
    P_bara=P_in_A, T_C=T_C,
    gas_flows_kgh=GAS_A, liquid_type=LIQUID_TYPE, q_lye=Q_LYE,
    props=res_a["props"], grid_records=res_a["grid_records"],
    segments=SEGS, total_dp_kpa=res_a["total_dp_kpa"],
    outlet_pressure_bara=res_a["outlet_pressure_bara"],
    pipe_length_m=res_a["pipe_length_m"],
    cumulative_distance=res_a["cumulative_distance"],
    case_label=LABEL_A)
path_a = os.path.join(out_dir, "report_case_A_H2.docx")
with open(path_a, "wb") as f:
    f.write(buf_a.getvalue())
print(f"  Saved: {path_a}")

# Report B (O2 branch)
buf_b = report_generator.generate_report(
    P_bara=P_in_B, T_C=T_C,
    gas_flows_kgh=GAS_B, liquid_type=LIQUID_TYPE, q_lye=Q_LYE,
    props=res_b["props"], grid_records=res_b["grid_records"],
    segments=SEGS, total_dp_kpa=res_b["total_dp_kpa"],
    outlet_pressure_bara=res_b["outlet_pressure_bara"],
    pipe_length_m=res_b["pipe_length_m"],
    cumulative_distance=res_b["cumulative_distance"],
    case_label=LABEL_B)
path_b = os.path.join(out_dir, "report_case_B_O2.docx")
with open(path_b, "wb") as f:
    f.write(buf_b.getvalue())
print(f"  Saved: {path_b}")

# Comparison report (A vs B)
buf_cmp = report_generator.generate_comparison_report(
    results_a=res_a, results_b=res_b,
    label_a=LABEL_A, label_b=LABEL_B,
    sensitivity_data={"sa": sens_a, "sb": sens_b, "fig": None})
path_cmp = os.path.join(out_dir, "report_comparison_H2_vs_O2.docx")
with open(path_cmp, "wb") as f:
    f.write(buf_cmp.getvalue())
print(f"  Saved: {path_cmp}")

# Combined report (A + B + C)
buf_combined = report_generator.generate_combined_report(
    cases=[res_a, res_b, res_c],
    case_labels=[LABEL_A, LABEL_B, "C — H₂ Header"],
    sensitivity_data={"sa": sens_a, "sb": sens_b, "fig": None})
path_combined = os.path.join(out_dir, "report_combined_electrolyzer.docx")
with open(path_combined, "wb") as f:
    f.write(buf_combined.getvalue())
print(f"  Saved: {path_combined}")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {LABEL_A}:")
print(f"    Inlet: {P_in_A:.4f} bara  →  Outlet: {res_a['outlet_pressure_bara']:.4f} bara")
print(f"    ΔP:    {res_a['total_dp_kpa']:.3f} kPa  "
      f"(fric {res_a['total_dp_fric_kpa']:.3f}  +  grav {res_a['total_dp_grav_kpa']:.3f})")
print(f"  {LABEL_B}:")
print(f"    Inlet: {P_in_B:.4f} bara  →  Outlet: {res_b['outlet_pressure_bara']:.4f} bara")
print(f"    ΔP:    {res_b['total_dp_kpa']:.3f} kPa  "
      f"(fric {res_b['total_dp_fric_kpa']:.3f}  +  grav {res_b['total_dp_grav_kpa']:.3f})")
print(f"  H₂ header ΔP:  {gsr_h2['dp_hdr']:.3f} kPa")
print(f"  O₂ header ΔP:  {gsr_o2['dp_hdr']:.3f} kPa")
print(f"  P_separator (H₂): {gsr_h2['P_sep']:.4f} bara")
print(f"  P_separator (O₂): {gsr_o2['P_sep']:.4f} bara")
print(f"  ─────────────────────────────────────")
print(f"  STACK ΔP = {dp_stack_bara:.4f} bara  =  {dp_stack_bara*100:.3f} kPa  "
      f"=  {dp_stack_bara*1000:.1f} mbar")
print("=" * 60)
