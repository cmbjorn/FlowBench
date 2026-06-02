"""
Pipeline case workflow — pure calculation, no Streamlit imports.

Entry point: compute_pipeline_case()

Runs the full segment loop (pipe, valve, heat exchanger segments) given
pre-computed inlet fluid properties and a list of segment definitions.
Returns a plain dict that the Streamlit layer can cache in session state
and render without re-running the calculation.
"""

from __future__ import annotations

import numpy as np

import multiphase_engine as engine
from models.pipe import SegmentRow
from standards.piping import (
    PIPE_DATABASE,
    MATERIAL_ROUGHNESS,
    LINER_ROUGHNESS,
    FITTING_Le_over_D,
    sum_le_fit,
)

# Pipe inclination angles (radians) keyed by segment type label.
_ANGLES: dict[str, float] = {
    "Horizontal":        0.0,
    "Vertical Upflow":   np.pi / 2.0,
    "Vertical Downflow": -np.pi / 2.0,
}




def compute_pipeline_case(
    *,
    P_bara: float,
    T_C: float,
    eff_gas_flows: dict,
    eff_liq_type: str,
    eff_q_lye: float,
    eff_custom_liq: dict | None,
    eff_liquid_flows: dict | None,
    is_vle: bool,
    vle_fluid_id: str | None,
    vle_x: float | None,
    vle_m_kgs: float | None,
    vle_h_inlet: float | None,
    segments: list[dict],
    correlation: str,
    voidage_method: str,
    custom_gas: dict | None,
    use_coolprop: bool,
    props: dict,
) -> dict:
    """
    Run all pipeline segments and return a results dict.

    Args:
        P_bara         : inlet pressure (bara)
        T_C            : inlet temperature (°C) — ignored in VLE mode
        eff_gas_flows  : {species: kg/h} after flash adjustment
        eff_liq_type   : liquid type string
        eff_q_lye      : liquid volume flow (m³/h)
        eff_custom_liq : custom liquid property dict or None
        eff_liquid_flows: {species: kg/h} or None
        is_vle         : True when in saturated two-phase (VLE) mode
        vle_fluid_id   : CoolProp fluid identifier (VLE mode only)
        vle_x          : inlet mass quality (VLE mode only)
        vle_m_kgs      : total mass flow (kg/s, VLE mode only)
        vle_h_inlet    : inlet specific enthalpy (J/kg, VLE mode only)
        segments       : list of segment definition dicts
        correlation    : pressure-drop correlation name
        voidage_method : void fraction method name
        custom_gas     : custom gas property dict or None
        use_coolprop   : whether to use CoolProp for gas properties
        props          : inlet fluid properties dict (pre-computed)

    Returns dict with keys:
        current_P, current_T_C, vle_x,
        grid_records, stream_records, valve_sizing,
        cumulative_distance, cumulative_positions,
        pressure_profile_x, pressure_profile_y,
        regime_bands,
        total_dp_fric_kpa, total_dp_grav_kpa, total_dp_accel_kpa
    """
    current_P   = P_bara * 1e5
    current_T_C = T_C if T_C is not None else 25.0
    # Track current VLE specific enthalpy so heat exchangers properly advance
    # the quality (isenthalpic flash at each downstream pressure).
    _vle_h = vle_h_inlet   # updated when an HX segment adds/removes heat

    grid_records         = []
    stream_records       = []
    valve_sizing         = []
    slug_records         = []
    cumulative_positions = []
    cumulative_distance  = 0.0
    pressure_profile_x   = [0.0]
    pressure_profile_y   = [P_bara]
    regime_bands         = []
    total_dp_fric_kpa    = 0.0
    total_dp_grav_kpa    = 0.0
    total_dp_accel_kpa   = 0.0

    # ── Fluid properties at current loop state ───────────────────────────────
    def _props_at_current() -> dict:
        if is_vle:
            return engine.calculate_vle_properties(
                vle_fluid_id, max(1000.0, current_P) / 1e5,
                vle_x, vle_m_kgs, h_spec=_vle_h)
        return engine.calculate_two_phase_properties(
            max(1000.0, current_P) / 1e5, current_T_C,
            eff_gas_flows, eff_liq_type, eff_q_lye,
            custom_gas=custom_gas, custom_liquid=eff_custom_liq,
            use_coolprop=use_coolprop,
            liquid_flows_kgh=eff_liquid_flows)

    # ── Append one row to the stream balance table ───────────────────────────
    def _snap_stream(label: str, sp: dict) -> None:
        _x  = sp.get("x_gas", 0.0)
        _rg = sp.get("rho_g", 0.0)
        _rl = sp.get("rho_l", 1000.0)
        if 0.0 < _x < 1.0:
            _rh = 1.0 / (_x / max(_rg, 1e-9) + (1.0 - _x) / max(_rl, 1e-9))
        else:
            _rh = _rg if _x >= 1.0 else _rl
        _rec: dict = {
            "Stream":   label,
            "P (bara)": round(max(1e-4, current_P / 1e5), 4),
            "T (°C)":   round(current_T_C, 1),
        }
        if is_vle:
            _rec[f"{vle_fluid_id} vapour  kg/h"] = round(sp.get("m_gas_total_kgh", 0.0), 2)
            _rec[f"{vle_fluid_id} liquid  kg/h"] = round(sp.get("m_liquid_total_kgh", 0.0), 2)
        else:
            for _gsp, _gflow in (eff_gas_flows or {}).items():
                _rec[f"gas:{_gsp}"] = round(_gflow, 3)
            _rec["Ṁ_gas (kg/h)"] = round(sp.get("m_gas_total_kgh", 0.0), 2)
            for _lsp, _lflow in (eff_liquid_flows or {}).items():
                _rec[f"liq:{_lsp}"] = round(_lflow, 3)
            _rec["Ṁ_liq (kg/h)"] = round(sp.get("m_liquid_total_kgh", 0.0), 2)
        _rec.update({
            "x (−)":          round(_x, 5),
            "α (−)":          round(sp.get("alpha", 0.0), 4),
            "ρ_g (kg/m³)":   round(_rg, 3),
            "ρ_l (kg/m³)":   round(_rl, 3),
            "ρ_hom (kg/m³)": round(_rh, 2),
        })
        stream_records.append(_rec)

    # ── Inlet stream snapshot ────────────────────────────────────────────────
    _snap_stream("S0 — Inlet", props)

    # ── Segment loop ─────────────────────────────────────────────────────────
    for i, seg in enumerate(segments):
        _seg_kind = seg.get("kind", "pipe")

        # ── VALVE ────────────────────────────────────────────────────────────
        if _seg_kind == "valve":
            _vp     = _props_at_current()
            _v_mode = seg.get("valve_mode", "kv")
            if _v_mode == "dp":
                _vres = engine.calculate_valve_kv(
                    _vp, seg.get("dp_kpa", 50.0) * 1000.0,
                    seg.get("opening_pct", 100.0),
                    seg.get("characteristic", "equal-percentage"))
                _v_label = (f"ΔP={seg.get('dp_kpa', 50):.1f} kPa → "
                            f"Kv_req={_vres['Kv_rated']:.2f}  "
                            f"Kv_eff={_vres['Kv_eff']:.2f}  "
                            f"{seg.get('opening_pct', 100):.0f}% open")
                valve_sizing.append({
                    "seg":      i + 1,
                    "dn":       seg.get("dn", ""),
                    "pn":       seg.get("pn", ""),
                    "dp_kpa":   seg.get("dp_kpa", 50.0),
                    "opening":  seg.get("opening_pct", 100.0),
                    "char":     seg.get("characteristic", "equal-percentage"),
                    "Q_m3h":    _vres["Q_m3h"],
                    "rho_hom":  _vres["rho_hom"],
                    "Kv_eff":   _vres["Kv_eff"],
                    "Kv_rated": _vres["Kv_rated"],
                    "Cv_rated": _vres["Kv_rated"] * 1.156,
                    "P_in":     current_P / 1e5,
                })
            else:
                _vres = engine.calculate_valve_dp(
                    _vp, seg.get("Kv_m3h", 1.0),
                    seg.get("opening_pct", 100.0),
                    seg.get("characteristic", "equal-percentage"))
                _v_label = (f"Kv={seg.get('Kv_m3h', 1.0):.2g}  "
                            f"Kv_eff={_vres['Kv_eff']:.3f}  "
                            f"{seg.get('opening_pct', 100):.0f}% open")
            _v_dP_Pa = _vres["dP_Pa"]
            _v_end_P = current_P - _v_dP_Pa
            _v_D     = PIPE_DATABASE[seg["dn"]][seg["pn"]]
            grid_records.append(SegmentRow(
                seg=f"#{i+1}", type="Valve",
                pipe=f"{seg['dn']}/{seg['pn']}",
                id_mm=round(_v_D * 1000, 1),
                l_m=0.0, l_eq_m=0.0,
                fittings=_v_label,
                regime=f"Q={_vres['Q_m3h']:.3f} m³/h  ρ_hom={_vres['rho_hom']:.1f} kg/m³",
                dp_kPa=round(_v_dP_Pa / 1000, 3),
                p_in_bara=round(current_P / 1e5, 4),
                p_out_bara=round(_v_end_P / 1e5, 4),
                v_m_ms=0.0, v_m_ve=0.0,
                v_sg_ms=0.0, v_sl_ms=0.0, v_e_ms=0.0,
                dp_fric_kPa=round(_v_dP_Pa / 1000, 3),
                dp_grav_kPa=0.0, dp_accel_kPa=0.0,
                material="—",
                rho_g=round(_vp["rho_g"], 4),
                l_eff_m=0.0,
                alpha_void=round(_vp["alpha"], 4),
                dp_dz=0.0,
            ).to_dict())
            total_dp_fric_kpa += _v_dP_Pa / 1000.0
            current_P = max(1000.0, _v_end_P)
            pressure_profile_x.append(cumulative_distance)
            pressure_profile_y.append(max(0.1, current_P / 1e5))
            regime_bands.append("Valve")
            _snap_stream(f"S{i+1} — #{i+1} Valve", _props_at_current())
            continue

        # ── HEAT EXCHANGER ───────────────────────────────────────────────────
        if _seg_kind == "hx":
            _hx_duty  = float(seg.get("duty_kw", 0.0))
            _hx_dP_Pa = float(seg.get("dp_kpa", 0.0)) * 1000.0
            _hx_D     = PIPE_DATABASE[seg["dn"]][seg["pn"]]
            _hx_end_P = current_P - _hx_dP_Pa
            _delta_T  = 0.0
            if _hx_duty != 0.0:
                if is_vle:
                    # VLE mode: advance enthalpy directly so downstream flash
                    # gives the correct quality (no Cp estimation needed).
                    if vle_m_kgs and vle_m_kgs > 0:
                        _vle_h = (_vle_h or 0.0) + (_hx_duty * 1000.0) / vle_m_kgs
                else:
                    _hp = _props_at_current()
                    _Cp = engine.estimate_mixture_cp(
                        _hp, current_T_C + 273.15, max(1000.0, current_P))
                    if _hp["m_total_kgs"] > 0 and _Cp > 0:
                        _delta_T = (_hx_duty * 1000.0) / (_hp["m_total_kgs"] * _Cp)
            current_T_C += _delta_T
            _hx_sign = f"+{_hx_duty:.1f}" if _hx_duty >= 0 else f"{_hx_duty:.1f}"
            grid_records.append(SegmentRow(
                seg=f"#{i+1}", type="Heat Exchanger",
                pipe=f"{seg['dn']}/{seg['pn']}",
                id_mm=round(_hx_D * 1000, 1),
                l_m=0.0, l_eq_m=0.0,
                fittings=f"Q={_hx_sign} kW",
                regime=f"ΔT={_delta_T:+.2f}°C  →  T_out={current_T_C:.1f}°C",
                dp_kPa=round(_hx_dP_Pa / 1000, 3),
                p_in_bara=round(current_P / 1e5, 4),
                p_out_bara=round(_hx_end_P / 1e5, 4),
                v_m_ms=0.0, v_m_ve=0.0,
                v_sg_ms=0.0, v_sl_ms=0.0, v_e_ms=0.0,
                dp_fric_kPa=round(_hx_dP_Pa / 1000, 3),
                dp_grav_kPa=0.0, dp_accel_kPa=0.0,
                material="—",
                rho_g=0.0, l_eff_m=0.0, alpha_void=0.0, dp_dz=0.0,
            ).to_dict())
            total_dp_fric_kpa += _hx_dP_Pa / 1000.0
            current_P = max(1000.0, _hx_end_P)
            pressure_profile_x.append(cumulative_distance)
            pressure_profile_y.append(max(0.1, current_P / 1e5))
            regime_bands.append("Heat Exchanger")
            _snap_stream(f"S{i+1} — #{i+1} Heat Exch.", _props_at_current())
            continue

        # ── PIPE SEGMENT ─────────────────────────────────────────────────────
        D_seg     = PIPE_DATABASE[seg["dn"]][seg["pn"]]
        _lined    = seg.get("lined", False)
        _lthk_m   = seg.get("liner_thickness_mm", 1.0) / 1000.0
        _lmat     = seg.get("liner_material", "FEP")
        D_eff     = D_seg - 2 * _lthk_m if _lined else D_seg
        rough_seg = (LINER_ROUGHNESS[_lmat] if _lined
                     else MATERIAL_ROUGHNESS[seg.get("material", "SS316L")])

        if is_vle:
            props_seg = engine.calculate_vle_properties(
                vle_fluid_id, current_P / 1e5, vle_x, vle_m_kgs,
                h_spec=_vle_h)
        else:
            props_seg = engine.calculate_two_phase_properties(
                current_P / 1e5, current_T_C,
                eff_gas_flows, eff_liq_type, eff_q_lye,
                custom_gas=custom_gas, custom_liquid=eff_custom_liq,
                use_coolprop=use_coolprop,
                liquid_flows_kgh=eff_liquid_flows)

        angle  = _ANGLES[seg["type"]]
        le_fit = sum_le_fit(seg, D_eff)
        L_eff  = seg["length"] + le_fit

        seg_result = engine.calculate_segment_pressure_drop(
            props_seg, D_eff, rough_seg, L_eff, angle,
            correlation=correlation, voidage_method=voidage_method)

        dP_Pa       = seg_result["dP_Pa"]
        regime      = seg_result["regime"]
        dP_per_dz   = seg_result["dP_per_dz"]
        Vsg         = seg_result["Vsg"]
        Vsl         = seg_result["Vsl"]
        alpha_seg   = seg_result["alpha"]
        dP_fric_Pa  = seg_result["dP_fric_Pa"]
        dP_grav_Pa  = seg_result["dP_grav_Pa"]
        dP_accel_Pa = seg_result["dP_accel_Pa"]

        V_m = Vsg + Vsl
        V_e, _ = engine.calculate_erosion_velocity(
            props_seg["rho_g"], props_seg["rho_l"], props_seg["x_gas"])
        erosion_ratio = V_m / V_e if V_e > 0 else 0.0

        end_P    = current_P - dP_Pa
        _mat_str = seg.get("material", "SS316L")
        if _lined:
            _mat_str += f" / {_lmat} {seg.get('liner_thickness_mm', 1.0):.1f}mm"

        grid_records.append(SegmentRow(
            seg=f"#{i+1}",
            type=seg["type"],
            pipe=f"{seg['dn']}/{seg['pn']}",
            id_mm=round(D_eff * 1000, 1),
            l_m=seg["length"],
            l_eq_m=round(le_fit, 3),
            fittings=(", ".join(f"{f['type']} ×{f['qty']}"
                                for f in seg.get("fittings_list", [])
                                if f.get("qty", 0) > 0)
                      or "—"),
            regime=regime,
            dp_kPa=round(dP_Pa / 1000, 3),
            p_in_bara=round(current_P / 1e5, 4),
            p_out_bara=round(end_P / 1e5, 4),
            v_m_ms=round(V_m, 3),
            v_m_ve=round(erosion_ratio, 3),
            v_sg_ms=round(Vsg, 3),
            v_sl_ms=round(Vsl, 3),
            v_e_ms=round(V_e, 2),
            dp_fric_kPa=round(dP_fric_Pa / 1000, 3),
            dp_fric_100m_kPa=round(dP_fric_Pa / 1000 / max(L_eff, 0.001) * 100, 2),
            dp_grav_kPa=round(dP_grav_Pa / 1000, 3),
            dp_accel_kPa=round(dP_accel_Pa / 1000, 3),
            material=_mat_str,
            rho_g=round(props_seg["rho_g"], 4),
            l_eff_m=round(L_eff, 2),
            alpha_void=round(alpha_seg, 4),
            dp_dz=round(dP_per_dz, 2),
        ).to_dict())

        _slug = seg_result.get("slug_info")
        if _slug is not None:
            slug_records.append({
                "Seg":                f"#{i+1}",
                "DN":                 seg["dn"],
                "Regime":             regime,
                "Severity":           _slug["severity"],
                "Sev (momentum)":     _slug["sev_momentum"],
                "Sev (ΔP%)":          _slug["sev_dp"],
                "Sev (freq)":         _slug["sev_freq"],
                "f_slug (Hz)":        round(_slug["slug_freq_hz"],      3),
                "f_slug (slugs/min)": round(_slug["slug_freq_per_min"], 1),
                "Freq valid?":        "⚠ extrapolated" if _slug["freq_extrapolated"] else "✓",
                "V_slug (m/s)":       round(_slug["V_slug_ms"],         2),
                "H_Ls":               round(_slug["H_Ls"],              3),
                "L_slug (m)":         round(_slug["L_slug_m"],          2),
                "ρV² (kg/m/s²)":      round(_slug["momentum_flux"],     0),
                "ΔP/P (%)":           f"{_slug['dp_pct_P']:.1f}" if _slug["dp_pct_P"] is not None else "—",
                "ΔP_pulse (kPa)":     round(_slug["dP_pulse_kPa"],      2),
                "ΔP_design (kPa)":    round(_slug["dP_design_kPa"],     2),
                "F_elbow (N)":        round(_slug["F_elbow_N"],         1),
                "F_design (N)":       round(_slug["F_design_N"],        1),
                "q_dyn (kPa)":        round(_slug["q_dyn_kPa"],         2),
            })

        total_dp_fric_kpa  += dP_fric_Pa / 1000.0
        total_dp_grav_kpa  += dP_grav_Pa / 1000.0
        total_dp_accel_kpa += dP_accel_Pa / 1000.0
        current_P = max(1000.0, end_P)
        _snap_stream(f"S{i+1} — #{i+1} {seg['type']}", _props_at_current())
        cumulative_distance += L_eff
        cumulative_positions.append(cumulative_distance)
        pressure_profile_x.append(cumulative_distance)
        pressure_profile_y.append(max(0.1, current_P / 1e5))
        regime_bands.append(regime)

    return {
        "current_P":            current_P,
        "current_T_C":          current_T_C,
        "vle_x":                vle_x,
        "slug_records":         slug_records,
        "grid_records":         grid_records,
        "stream_records":       stream_records,
        "valve_sizing":         valve_sizing,
        "cumulative_distance":  cumulative_distance,
        "cumulative_positions": cumulative_positions,
        "pressure_profile_x":   pressure_profile_x,
        "pressure_profile_y":   pressure_profile_y,
        "regime_bands":         regime_bands,
        "total_dp_fric_kpa":    total_dp_fric_kpa,
        "total_dp_grav_kpa":    total_dp_grav_kpa,
        "total_dp_accel_kpa":   total_dp_accel_kpa,
    }
