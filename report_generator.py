# report_generator.py
"""
Generates a Word (.docx) calculation report from multiphase hydraulics engine results.
"""
from io import BytesIO
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


_HDR_BG = "2563EB"
_HDR_FG = RGBColor(0xFF, 0xFF, 0xFF)
_ALT_BG = "F1F5F9"


def _shd(cell, fill_hex):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def _style_header(row, font_size=9):
    for cell in row.cells:
        _shd(cell, _HDR_BG)
        for para in cell.paragraphs:
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after  = Pt(2)
            for run in para.runs:
                run.font.bold       = True
                run.font.color.rgb  = _HDR_FG
                run.font.size       = Pt(font_size)


def _set_col_widths(table, widths_inches):
    for row in table.rows:
        for cell, w in zip(row.cells, widths_inches):
            cell.width = Inches(w)


def _cell_font(cell, size_pt=9):
    for para in cell.paragraphs:
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(1)
        for run in para.runs:
            run.font.size = Pt(size_pt)


def _kv_table(doc, rows_data, col_widths=(2.5, 3.5)):
    """Two-column key-value table with blue header."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=2)
    tbl.style = "Table Grid"
    _style_header(tbl.rows[0])
    tbl.rows[0].cells[0].text = "Parameter"
    tbl.rows[0].cells[1].text = "Value"
    for i, (label, value) in enumerate(rows_data, start=1):
        row = tbl.rows[i]
        row.cells[0].text = label
        row.cells[1].text = value
        if i % 2 == 0:
            _shd(row.cells[0], _ALT_BG)
            _shd(row.cells[1], _ALT_BG)
        _cell_font(row.cells[0])
        _cell_font(row.cells[1])
    _set_col_widths(tbl, col_widths)
    return tbl


def generate_report(
    P_bara, T_C,
    gas_flows_kgh,        # dict {species: kg/h}
    liquid_type,          # str
    q_lye,
    props,
    grid_records,
    segments,
    total_dp_kpa,
    outlet_pressure_bara,
    pipe_length_m,
    cumulative_distance,
    fig_sch=None,
    fig_prof=None,
):
    doc = Document()

    # A4 portrait, 20 mm margins
    sec = doc.sections[0]
    sec.page_width    = Inches(8.27)
    sec.page_height   = Inches(11.69)
    sec.left_margin   = Inches(0.9)
    sec.right_margin  = Inches(0.9)
    sec.top_margin    = Inches(0.9)
    sec.bottom_margin = Inches(0.9)

    # ── Title block ──────────────────────────────────────────────────────────
    h = doc.add_heading("Multiphase Pipe Hydraulic Engine", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        f"Calculation Report  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)

    _gas_label = " + ".join(gas_flows_kgh.keys())
    desc = doc.add_paragraph(
        f"Beggs & Brill (1973) correlation  ·  "
        f"{_gas_label} / {liquid_type}  ·  Steady-state"
    )
    desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if desc.runs:
        desc.runs[0].font.size = Pt(9)
        desc.runs[0].font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    doc.add_paragraph()

    # ── 1. Method ────────────────────────────────────────────────────────────
    doc.add_heading("1. Method", level=1)

    _method_paras = [
        (
            "Pressure drop is calculated using the Beggs & Brill (1973) empirical "
            "correlation for two-phase gas–liquid flow in horizontal and inclined pipes. "
            "The pipeline is divided into user-defined segments; gas density and void "
            "fraction are re-evaluated at each segment inlet pressure (pressure marching) "
            "to capture the effect of gas compressibility along the route."
        ),
        (
            "Gas mixture density is computed from the ideal-gas law with a "
            "mole-fraction-weighted molecular weight. Dynamic viscosity of each pure gas "
            "species is obtained from CoolProp (REFPROP-quality equations of state); "
            "mixture viscosity uses linear mole-fraction weighting. Where the liquid phase "
            "is aqueous, water-vapour partial pressure is added to the gas via Dalton's Law "
            "using CoolProp saturation data."
        ),
        (
            "Friction factors and minor-loss equivalent lengths follow Crane TP-410. "
            "Erosion velocity is checked per API RP 14E (C = 100, continuous service). "
            "The homogeneous void-fraction model is used: α = (x/ρ_g) / (x/ρ_g + (1−x)/ρ_l)."
        ),
        (
            "Key open-source packages: fluids (two-phase correlations, friction factors) · "
            "CoolProp (thermodynamic and transport properties) · "
            "python-docx (this report)."
        ),
    ]
    for _txt in _method_paras:
        _p = doc.add_paragraph(_txt)
        _p.paragraph_format.space_after = Pt(4)
        if _p.runs:
            _p.runs[0].font.size = Pt(9)

    doc.add_paragraph()

    # ── 2. Process Conditions ────────────────────────────────────────────────
    doc.add_heading("2. Process Conditions", level=1)
    _cond_rows = [
        ("Inlet Pressure",     f"{P_bara:.2f} bara"),
        ("Temperature",        f"{T_C:.1f} °C"),
    ]
    for _sp, _flow in gas_flows_kgh.items():
        _cond_rows.append((f"{_sp} Mass Flow", f"{_flow:.3f} kg/h"))
    _cond_rows += [
        ("Liquid Type",        liquid_type),
        ("Liquid Volume Flow", f"{q_lye:.3f} m³/h"),
        ("Number of Segments", str(len(segments))),
    ]
    _kv_table(doc, _cond_rows)
    doc.add_paragraph()

    # ── 3. Phase Thermodynamics ──────────────────────────────────────────────
    doc.add_heading("3. Phase Thermodynamics", level=1)
    _thermo_rows = [
        ("Gas Density ρ_g",               f"{props['rho_g']:.4f} kg/m³"),
        ("Gas Mixture MW",                f"{props['MW_mix_gmol']:.3f} g/mol"),
        ("Liquid Type",                   props.get("liquid_type", liquid_type)),
        ("Liquid Density ρ_l",            f"{props['rho_l']:.2f} kg/m³"),
        ("Liquid Dynamic Viscosity μ_l",  f"{props['mu_l']*1e3:.4f} mPa·s"),
        ("Gas Dynamic Viscosity μ_g",     f"{props['mu_g']*1e6:.2f} µPa·s"),
        ("Surface Tension σ",             f"{props['sigma']*1e3:.3f} mN/m"),
        ("Mass Quality x",                f"{props['x_gas']*100:.4f} %"),
        ("Void Fraction α (homogeneous)", f"{props['alpha']*100:.2f} %"),
    ]
    if props.get("P_sat_H2O_pa", 0) > 0:
        _thermo_rows += [
            ("H₂O Saturation Pressure", f"{props['P_sat_H2O_pa']/1e5:.4f} bara"),
            ("H₂O Vapour Flow",         f"{props['m_vapor_h2o_kgh']:.4f} kg/h"),
        ]
    _kv_table(doc, _thermo_rows)
    doc.add_paragraph()

    # ── 4. Segment Analysis ──────────────────────────────────────────────────
    doc.add_heading("4. Segment Analysis", level=1)

    # Subset of columns that fits A4 portrait (6.47" available between margins)
    _COLS   = ["Seg", "Pipe", "ID (mm)", "Type", "L (m)", "Regime",
               "V_m (m/s)", "V_e (m/s)", "V_m/V_e", "ΔP (kPa)", "P_out (bara)"]
    _WIDTHS = [0.28, 0.55,   0.45,     0.88,   0.42,   0.90,
               0.55,         0.55,        0.48,      0.55,      0.56]

    if grid_records:
        tbl3 = doc.add_table(rows=len(grid_records) + 1, cols=len(_COLS))
        tbl3.style = "Table Grid"
        _style_header(tbl3.rows[0], font_size=8)
        for j, col in enumerate(_COLS):
            tbl3.rows[0].cells[j].text = col

        for i, rec in enumerate(grid_records, start=1):
            row = tbl3.rows[i]
            if i % 2 == 0:
                for cell in row.cells:
                    _shd(cell, _ALT_BG)
            for j, col in enumerate(_COLS):
                # Map report column name back to grid_records key
                key = col.replace("ΔP", "ΔP")  # passthrough unicode
                cell = row.cells[j]
                cell.text = str(rec.get(col, rec.get(key, "")))
                _cell_font(cell, size_pt=8)

        _set_col_widths(tbl3, _WIDTHS)

    doc.add_paragraph()

    # ── 5. System Totals ─────────────────────────────────────────────────────
    doc.add_heading("5. System Totals", level=1)
    _kv_table(doc, [
        ("Total Pressure Drop ΔP",                f"{total_dp_kpa:.4f} kPa"),
        ("Total Pressure Drop ΔP",                f"{total_dp_kpa / 100:.6f} bar"),
        ("Outlet Pressure",                            f"{outlet_pressure_bara:.4f} bara"),
        ("Pipe Length (physical segments)",            f"{pipe_length_m:.2f} m"),
        ("Effective Length (incl. fittings)",          f"{cumulative_distance:.2f} m"),
    ])

    # ── 6. Visualisations ────────────────────────────────────────────────────
    if fig_sch is not None or fig_prof is not None:
        try:
            import plotly.io as pio

            doc.add_page_break()
            doc.add_heading("6. Visualisations", level=1)

            if fig_sch is not None:
                doc.add_heading("Pipeline Schematic", level=2)
                img = pio.to_image(fig_sch, format="png", width=900, height=520, scale=2)
                doc.add_picture(BytesIO(img), width=Inches(6.2))
                doc.add_paragraph()

            if fig_prof is not None:
                doc.add_heading("Pressure Profile", level=2)
                img = pio.to_image(fig_prof, format="png", width=900, height=400, scale=2)
                doc.add_picture(BytesIO(img), width=Inches(6.2))

        except Exception:
            p = doc.add_paragraph(
                "Chart images could not be embedded (kaleido renderer error)."
            )
            if p.runs:
                p.runs[0].font.size = Pt(9)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    _gas_str = " / ".join(gas_flows_kgh.keys())
    note = doc.add_paragraph(
        f"Disclaimer: Results are based on the Beggs & Brill (1973) correlation, originally "
        f"developed for oil/gas systems. Application to {_gas_str} / {liquid_type} duty "
        f"carries an estimated uncertainty of ±20–30 %. Treat as an engineering estimate "
        f"and validate against plant data before use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# ============================================================================
# COMPARISON REPORT  (Case A vs Case B)
# ============================================================================

def _kv3_table(doc, rows_data, col_widths=(2.2, 2.1, 2.1)):
    """Three-column comparison table: Parameter | Case A | Case B."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=3)
    tbl.style = "Table Grid"
    _style_header(tbl.rows[0])
    for j, label in enumerate(["Parameter", "Case A", "Case B"]):
        tbl.rows[0].cells[j].text = label
    for i, (label, va, vb) in enumerate(rows_data, start=1):
        row = tbl.rows[i]
        row.cells[0].text = label
        row.cells[1].text = str(va)
        row.cells[2].text = str(vb)
        if i % 2 == 0:
            for cell in row.cells:
                _shd(cell, _ALT_BG)
        for cell in row.cells:
            _cell_font(cell)
    _set_col_widths(tbl, col_widths)
    return tbl


def generate_comparison_report(results_a, results_b, fig_cmp=None, fig_bar=None):
    """
    Generate a Word report comparing Case A and Case B side by side.

    results_a / results_b: dicts returned by run_case() in app.py containing
    P_bara, T_C, gas_flows_kgh, liquid_type, q_lye, props, grid_records,
    total_dp_kpa, outlet_pressure_bara, pipe_length_m, cumulative_distance.
    """
    doc = Document()
    sec = doc.sections[0]
    sec.page_width    = Inches(8.27);  sec.page_height   = Inches(11.69)
    sec.left_margin   = Inches(0.9);   sec.right_margin  = Inches(0.9)
    sec.top_margin    = Inches(0.9);   sec.bottom_margin = Inches(0.9)

    # ── Title ────────────────────────────────────────────────────────────────
    h = doc.add_heading("Multiphase Pipe Hydraulic Engine — Comparison Report", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        f"Case A  vs.  Case B  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)

    doc.add_paragraph()

    # ── 1. Method ────────────────────────────────────────────────────────────
    doc.add_heading("1. Method", level=1)
    for _txt in [
        ("Pressure drop is calculated using the Beggs & Brill (1973) empirical correlation "
         "for two-phase gas–liquid flow in horizontal and inclined pipes. Gas density and "
         "void fraction are re-evaluated at each segment inlet pressure (pressure marching)."),
        ("Gas viscosity uses CoolProp pure-component values with mole-fraction weighting. "
         "Where the liquid is aqueous, water-vapour partial pressure follows Dalton's Law "
         "(CoolProp saturation data). Minor losses follow Crane TP-410; erosion velocity "
         "per API RP 14E (C = 100). Void fraction: homogeneous model."),
        ("Key packages: fluids (two-phase correlations) · CoolProp (thermodynamic "
         "properties) · python-docx (this report)."),
    ]:
        _p = doc.add_paragraph(_txt)
        _p.paragraph_format.space_after = Pt(4)
        if _p.runs:
            _p.runs[0].font.size = Pt(9)
    doc.add_paragraph()

    # ── 2. Process Conditions ────────────────────────────────────────────────
    doc.add_heading("2. Process Conditions", level=1)

    # Collect all unique species across both cases
    _all_species = list(dict.fromkeys(
        list(results_a["gas_flows_kgh"].keys()) +
        list(results_b["gas_flows_kgh"].keys())
    ))
    _cond_rows = [
        ("Inlet Pressure (bara)",
         f"{results_a['P_bara']:.2f}",
         f"{results_b['P_bara']:.2f}"),
        ("Temperature (°C)",
         f"{results_a['T_C']:.1f}",
         f"{results_b['T_C']:.1f}"),
    ]
    for _sp in _all_species:
        _cond_rows.append((
            f"{_sp} mass flow (kg/h)",
            f"{results_a['gas_flows_kgh'].get(_sp, 0.0):.3f}",
            f"{results_b['gas_flows_kgh'].get(_sp, 0.0):.3f}",
        ))
    _cond_rows += [
        ("Liquid type",
         results_a["liquid_type"],
         results_b["liquid_type"]),
        ("Liquid volume flow (m³/h)",
         f"{results_a['q_lye']:.3f}",
         f"{results_b['q_lye']:.3f}"),
    ]
    _kv3_table(doc, _cond_rows)
    doc.add_paragraph()

    # ── 3. Phase Thermodynamics ──────────────────────────────────────────────
    doc.add_heading("3. Phase Thermodynamics  (inlet conditions)", level=1)
    pa, pb = results_a["props"], results_b["props"]
    _thermo_rows = [
        ("Gas density ρ_g (kg/m³)",          f"{pa['rho_g']:.4f}",             f"{pb['rho_g']:.4f}"),
        ("Gas mixture MW (g/mol)",            f"{pa['MW_mix_gmol']:.3f}",       f"{pb['MW_mix_gmol']:.3f}"),
        ("Liquid density ρ_l (kg/m³)",        f"{pa['rho_l']:.2f}",             f"{pb['rho_l']:.2f}"),
        ("Liquid viscosity μ_l (mPa·s)",      f"{pa['mu_l']*1e3:.4f}",          f"{pb['mu_l']*1e3:.4f}"),
        ("Gas viscosity μ_g (µPa·s)",         f"{pa['mu_g']*1e6:.2f}",          f"{pb['mu_g']*1e6:.2f}"),
        ("Surface tension σ (mN/m)",          f"{pa['sigma']*1e3:.3f}",         f"{pb['sigma']*1e3:.3f}"),
        ("Mass quality x (%)",                f"{pa['x_gas']*100:.4f}",         f"{pb['x_gas']*100:.4f}"),
        ("Void fraction α (%)",               f"{pa['alpha']*100:.2f}",         f"{pb['alpha']*100:.2f}"),
    ]
    if pa.get("P_sat_H2O_pa", 0) > 0 or pb.get("P_sat_H2O_pa", 0) > 0:
        _thermo_rows += [
            ("H₂O saturation pressure (bara)",
             f"{pa.get('P_sat_H2O_pa',0)/1e5:.4f}" if pa.get('P_sat_H2O_pa',0) > 0 else "—",
             f"{pb.get('P_sat_H2O_pa',0)/1e5:.4f}" if pb.get('P_sat_H2O_pa',0) > 0 else "—"),
            ("H₂O vapour flow (kg/h)",
             f"{pa.get('m_vapor_h2o_kgh',0):.4f}" if pa.get('P_sat_H2O_pa',0) > 0 else "—",
             f"{pb.get('m_vapor_h2o_kgh',0):.4f}" if pb.get('P_sat_H2O_pa',0) > 0 else "—"),
        ]
    _kv3_table(doc, _thermo_rows)
    doc.add_paragraph()

    # ── 4. System Totals ─────────────────────────────────────────────────────
    doc.add_heading("4. System Totals", level=1)
    _dp_delta = results_b["total_dp_kpa"] - results_a["total_dp_kpa"]
    _max_ve_a = max((r["V_m/V_e"] for r in results_a["grid_records"]), default=0.0)
    _max_ve_b = max((r["V_m/V_e"] for r in results_b["grid_records"]), default=0.0)
    _tot_rows = [
        ("Outlet pressure (bara)",
         f"{results_a['outlet_pressure_bara']:.4f}",
         f"{results_b['outlet_pressure_bara']:.4f}"),
        ("Total ΔP (kPa)",
         f"{results_a['total_dp_kpa']:.4f}",
         f"{results_b['total_dp_kpa']:.4f}"),
        ("Total ΔP (bar)",
         f"{results_a['total_dp_kpa']/100:.6f}",
         f"{results_b['total_dp_kpa']/100:.6f}"),
        ("ΔP difference B − A (kPa)",
         "—",
         f"{_dp_delta:+.4f}"),
        ("Pipe length (m)",
         f"{results_a['pipe_length_m']:.2f}",
         f"{results_b['pipe_length_m']:.2f}"),
        ("Effective length incl. fittings (m)",
         f"{results_a['cumulative_distance']:.2f}",
         f"{results_b['cumulative_distance']:.2f}"),
        ("Worst V_m/V_e (–)",
         f"{_max_ve_a:.3f}",
         f"{_max_ve_b:.3f}"),
    ]
    _kv3_table(doc, _tot_rows)
    doc.add_paragraph()

    # ── 5. Segment Analysis ──────────────────────────────────────────────────
    _COLS   = ["Seg","Pipe","ID (mm)","Type","L (m)","Regime",
               "V_m (m/s)","V_e (m/s)","V_m/V_e","ΔP (kPa)","P_out (bara)"]
    _WIDTHS = [0.28, 0.55, 0.45, 0.88, 0.42, 0.90, 0.55, 0.55, 0.48, 0.55, 0.56]

    def _seg_table(doc, grid_records):
        if not grid_records:
            return
        tbl = doc.add_table(rows=len(grid_records)+1, cols=len(_COLS))
        tbl.style = "Table Grid"
        _style_header(tbl.rows[0], font_size=8)
        for j, col in enumerate(_COLS):
            tbl.rows[0].cells[j].text = col
        for i, rec in enumerate(grid_records, start=1):
            row = tbl.rows[i]
            if i % 2 == 0:
                for cell in row.cells:
                    _shd(cell, _ALT_BG)
            for j, col in enumerate(_COLS):
                row.cells[j].text = str(rec.get(col, ""))
                _cell_font(row.cells[j], size_pt=8)
        _set_col_widths(tbl, _WIDTHS)

    doc.add_heading("5. Segment Analysis — Case A", level=1)
    _seg_table(doc, results_a["grid_records"])
    doc.add_paragraph()
    doc.add_heading("6. Segment Analysis — Case B", level=1)
    _seg_table(doc, results_b["grid_records"])
    doc.add_paragraph()

    # ── 7. Visualisations ────────────────────────────────────────────────────
    if fig_cmp is not None or fig_bar is not None:
        try:
            import plotly.io as pio
            doc.add_page_break()
            doc.add_heading("7. Visualisations", level=1)
            if fig_cmp is not None:
                doc.add_heading("Pressure Profiles — Case A vs B", level=2)
                img = pio.to_image(fig_cmp, format="png", width=900, height=400, scale=2)
                doc.add_picture(BytesIO(img), width=Inches(6.2))
                doc.add_paragraph()
            if fig_bar is not None:
                doc.add_heading("ΔP by Segment — Case A vs B", level=2)
                img = pio.to_image(fig_bar, format="png", width=900, height=340, scale=2)
                doc.add_picture(BytesIO(img), width=Inches(6.2))
        except Exception:
            p = doc.add_paragraph("Chart images could not be embedded (kaleido renderer error).")
            if p.runs:
                p.runs[0].font.size = Pt(9)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    _sp_a = " / ".join(results_a["gas_flows_kgh"].keys())
    _sp_b = " / ".join(results_b["gas_flows_kgh"].keys())
    _sp_str = _sp_a if _sp_a == _sp_b else f"{_sp_a}  |  {_sp_b}"
    note = doc.add_paragraph(
        f"Disclaimer: Results are based on the Beggs & Brill (1973) correlation, originally "
        f"developed for oil/gas systems. Application to {_sp_str} duty carries an estimated "
        f"uncertainty of ±20–30 %. Treat as an engineering estimate and validate against "
        f"plant data before use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
