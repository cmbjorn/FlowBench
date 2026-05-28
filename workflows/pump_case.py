"""
Pump sizing workflow — centrifugal and positive-displacement.

compute_pump_case() orchestrates the full calculation chain (operating
point, power, NPSH, design pressure, ANSI class, method comparison) with
no Streamlit dependency.  Raises ValueError on invalid inputs so the
caller can decide how to surface the error.
"""

import numpy as np
import pump_engine as pe

_N_PLOT = 200


def compute_pump_case(
    *,
    is_pd: bool,
    rho: float,
    Pv: float,
    P_bara: float,
    hq_coeffs,
    eta_params,
    n_ratio: float,
    Qbep_used: float,
    H_static: float,
    k_fric: float,
    eta_motor: float,
    z_suc: float,
    h_suc_loss: float,
    NPSH_R: float,
    dp_method: int | None,
    P_suc_max: float,
    PSV_set: float | None,
    mat_group: str,
    pd_acc_pct: float,
) -> dict:
    """Return all pump results as a plain dict (no Streamlit calls).

    For PD pumps only ``PSV_set``, ``pd_acc_pct``, ``mat_group``, ``rho``
    are used; the centrifugal parameters are ignored.

    Raises:
        ValueError: if ``hq_coeffs`` is None for a centrifugal pump.
    """
    if is_pd:
        dp_res = pe.pd_pump_design_pressure(PSV_set, pd_acc_pct)
        ansi   = pe.ansi_class_lookup(dp_res["P_design_barg"], mat_group)
        return {
            "op_ok":          False,
            "dp_res":         dp_res,
            "P_design_barg":  dp_res["P_design_barg"],
            "P_design_bara":  dp_res["P_design_bara"],
            "ansi":           ansi,
            "method_comparison": [],
        }

    # ── Centrifugal ──────────────────────────────────────────────────────────
    if hq_coeffs is None:
        raise ValueError("Fix H-Q curve inputs to see results.")

    H0_rated = pe.hq_shutoff(hq_coeffs)
    H0_max   = pe.hq_shutoff(pe.scale_hq_to_speed(hq_coeffs, n_ratio))
    H0_bar   = pe.head_to_bar(H0_max, rho)

    dp_res = pe.centrifugal_design_pressure(
        method            = dp_method,
        P_suction_bara    = P_suc_max if dp_method <= 3 else P_bara,
        H_shutoff_rated_m = H0_rated,
        rho_kgm3          = rho,
        n_max_ratio       = n_ratio,
        PSV_set_barg      = PSV_set,
    )
    ansi = pe.ansi_class_lookup(dp_res["P_design_barg"], mat_group)

    Q_max    = pe.hq_max_flow(hq_coeffs)
    Qop, Hop = pe.find_operating_point(hq_coeffs, H_static, k_fric, Q_max * 1.05)
    eta_op   = pe.eval_eta(eta_params, Qop)
    P_shaft  = pe.shaft_power_kw(rho, Qop, Hop, max(eta_op, 1.0))
    P_motor  = pe.motor_power_kw(P_shaft, eta_motor)
    P_frame  = pe.next_motor_frame_kw(P_motor)
    NPSH_A   = pe.npsh_available(P_bara, Pv, rho, z_suc, h_suc_loss)
    npsh_margin, npsh_status, npsh_color = pe.npsh_margin_status(NPSH_A, NPSH_R)
    Hop_bar  = pe.head_to_bar(Hop, rho)

    pset = PSV_set if PSV_set is not None else 0.0
    method_comparison = []
    for m in range(1, 7):
        try:
            mr = pe.centrifugal_design_pressure(
                method            = m,
                P_suction_bara    = P_suc_max if m <= 3 else P_bara,
                H_shutoff_rated_m = H0_rated,
                rho_kgm3          = rho,
                n_max_ratio       = n_ratio,
                PSV_set_barg      = pset if m >= 4 else None,
            )
            ansi_m = pe.ansi_class_lookup(mr["P_design_barg"], mat_group)
            method_comparison.append({
                "#":             m,
                "method_label":  pe.DESIGN_PRESSURE_METHODS[m].split("(")[0].strip(),
                "P_design_barg": mr["P_design_barg"],
                "ansi_label":    ansi_m["class_label"],
            })
        except Exception:
            method_comparison.append({
                "#":             m,
                "method_label":  pe.DESIGN_PRESSURE_METHODS[m].split("(")[0].strip(),
                "P_design_barg": None,
                "ansi_label":    "—",
            })

    Q_plot   = list(np.linspace(0.0, Q_max * 1.05, _N_PLOT))
    H_plot   = [pe.eval_hq(hq_coeffs, q)       for q in Q_plot]
    Hs_plot  = [pe.system_head(q, H_static, k_fric) for q in Q_plot]
    eta_plot = [pe.eval_eta(eta_params, q)      for q in Q_plot]

    return {
        "op_ok":             True,
        "Q_op":              Qop,
        "H_op":              Hop,
        "eta_op":            eta_op,
        "P_shaft":           P_shaft,
        "P_motor":           P_motor,
        "P_frame":           P_frame,
        "NPSH_A":            NPSH_A,
        "npsh_margin":       npsh_margin,
        "npsh_status":       npsh_status,
        "npsh_color":        npsh_color,
        "Hop_bar":           Hop_bar,
        "P_design_barg":     dp_res["P_design_barg"],
        "P_design_bara":     dp_res["P_design_bara"],
        "dp_res":            dp_res,
        "ansi":              ansi,
        "H0_max":            H0_max,
        "H0_bar":            H0_bar,
        "Q_max":             Q_max,
        "method_comparison": method_comparison,
        "Q_plot":            Q_plot,
        "H_plot":            H_plot,
        "Hs_plot":           Hs_plot,
        "eta_plot":          eta_plot,
    }
