# report_generator.py
"""
Generates a Word (.docx) calculation report from multiphase hydraulics engine results.
"""
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


_HDR_BG = "2563EB"
_HDR_FG = RGBColor(0xFF, 0xFF, 0xFF)
_ALT_BG = "F1F5F9"

_IMG_TIMEOUT = 20  # seconds before giving up on kaleido


def _fig_to_png(fig, width=900, height=400, scale=2):
    """Render a Plotly figure to PNG bytes with a hard timeout.
    Returns bytes on success, None if kaleido hangs or fails.
    Kaleido can deadlock when called from a Streamlit run thread;
    running it in a separate thread lets us abort cleanly.
    """
    import plotly.io as pio
    try:
        with ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(pio.to_image, fig,
                              format="png", width=width, height=height, scale=scale)
            return _fut.result(timeout=_IMG_TIMEOUT)
    except (_FuturesTimeout, Exception):
        return None


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
        f"Two-phase pressure drop  ·  "
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
            "Six two-phase ΔP correlations are available: Beggs & Brill (1973, default), "
            "Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, and "
            "Kim-Mudawar. The selected correlation is shown in the Segment Analysis table. "
            "The pipeline is divided into user-defined segments with pressure marching — "
            "gas density is re-evaluated at each segment inlet to capture compressibility. "
            "Each segment's ΔP is decomposed into frictional, gravitational, and "
            "accelerational components."
        ),
        (
            "Void fraction uses either the homogeneous model "
            "(α = (x/ρ_g) / (x/ρ_g + (1−x)/ρ_l)) or the Rouhani-1 slip-flow model. "
            "Flow regime is classified automatically: Taitel-Dukler + Mandhane-Gregory-Aziz "
            "for horizontal segments (|θ| ≤ 15°); Wallis/Taitel (1980) annular-onset "
            "criterion with void-fraction thresholds for vertical segments (|θ| ≥ 75°). "
            "Gas properties use the ideal-gas law and CoolProp viscosities; water vapour "
            "is included via Dalton's Law for aqueous liquids."
        ),
        (
            "Minor losses: equivalent-length method, Crane TP-410. "
            "Erosion: API RP 14E, C = 100 continuous service. "
            "Packages: fluids · CoolProp · python-docx."
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
        doc.add_page_break()
        doc.add_heading("6. Visualisations", level=1)

        if fig_sch is not None:
            doc.add_heading("Pipeline Schematic", level=2)
            img = _fig_to_png(fig_sch, width=900, height=520, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out — export without kaleido)")
            doc.add_paragraph()

        if fig_prof is not None:
            doc.add_heading("Pressure Profile", level=2)
            img = _fig_to_png(fig_prof, width=900, height=400, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out — export without kaleido)")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    _gas_str = " / ".join(gas_flows_kgh.keys())
    note = doc.add_paragraph(
        f"Disclaimer: The correlations implemented were developed primarily for oil/gas "
        f"systems. Application to {_gas_str} / {liquid_type} duty carries an estimated "
        f"uncertainty of ±20–30 %. Use the sensitivity analysis (Compare tab) to bracket "
        f"the range across all available methods. Treat as an engineering estimate and "
        f"validate against plant data before use in safety-critical design."
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

def _kv_n_table(doc, headers, rows_data, col_widths=None):
    """Generic n-column table. headers[0] = 'Parameter', rest = case labels."""
    n = len(headers)
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=n)
    tbl.style = "Table Grid"
    _style_header(tbl.rows[0])
    for j, h in enumerate(headers):
        tbl.rows[0].cells[j].text = h
    for i, row_vals in enumerate(rows_data, start=1):
        row = tbl.rows[i]
        for j, v in enumerate(row_vals):
            row.cells[j].text = str(v)
        if i % 2 == 0:
            for cell in row.cells:
                _shd(cell, _ALT_BG)
        for cell in row.cells:
            _cell_font(cell)
    if col_widths:
        _set_col_widths(tbl, col_widths)
    return tbl


def _kv3_table(doc, rows_data, col_widths=(2.2, 2.1, 2.1),
               label_a="Case A", label_b="Case B"):
    """Three-column comparison table: Parameter | <label_a> | <label_b>."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=3)
    tbl.style = "Table Grid"
    _style_header(tbl.rows[0])
    for j, label in enumerate(["Parameter", label_a, label_b]):
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


def generate_comparison_report(
    results_a, results_b,
    label_a="Case A", label_b="Case B",
    fig_cmp=None, fig_bar=None,
    sensitivity_data=None,
):
    """
    Generate a Word report comparing two cases side by side.

    results_a / results_b: dicts returned by run_case() in app.py.
    label_a / label_b: custom case names used throughout the report.
    sensitivity_data: optional dict {"sa": [...], "sb": [...], "fig": Figure}
                      from run_sensitivity(); adds a Method Sensitivity section.
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
        f"{label_a}  vs.  {label_b}  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)

    doc.add_paragraph()

    # ── 1. Method ────────────────────────────────────────────────────────────
    doc.add_heading("1. Method", level=1)
    for _txt in [
        ("Six two-phase ΔP correlations are available: Beggs & Brill (1973, default), "
         "Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, and Kim-Mudawar. "
         f"{label_a} and {label_b} may use different correlations and void-fraction models. "
         "Gas density is re-evaluated at each segment inlet (pressure marching); each "
         "segment's ΔP is split into frictional, gravitational, and accelerational components."),
        ("Void fraction: homogeneous model or Rouhani-1 slip-flow model. "
         "Flow regime classified automatically — Taitel-Dukler + Mandhane-Gregory-Aziz "
         "for horizontal, Wallis/Taitel (1980) for vertical segments. "
         "Gas properties: ideal-gas law, CoolProp viscosities, Dalton's Law for water vapour. "
         "Minor losses: Crane TP-410. Erosion: API RP 14E, C = 100."),
        ("Where a sensitivity analysis was run, Section 8 shows total ΔP and flow regimes "
         "across all 12 method combinations (6 correlations × 2 void-fraction models). "
         "Packages: fluids · CoolProp · python-docx."),
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
    _kv3_table(doc, _cond_rows, label_a=label_a, label_b=label_b)
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
    _kv3_table(doc, _thermo_rows, label_a=label_a, label_b=label_b)
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
        (f"ΔP difference {label_b} − {label_a} (kPa)",
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
    _kv3_table(doc, _tot_rows, label_a=label_a, label_b=label_b)
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

    doc.add_heading(f"5. Segment Analysis — {label_a}", level=1)
    _seg_table(doc, results_a["grid_records"])
    doc.add_paragraph()
    doc.add_heading(f"6. Segment Analysis — {label_b}", level=1)
    _seg_table(doc, results_b["grid_records"])
    doc.add_paragraph()

    # ── 7. Visualisations ────────────────────────────────────────────────────
    if fig_cmp is not None or fig_bar is not None:
        doc.add_page_break()
        doc.add_heading("7. Visualisations", level=1)
        if fig_cmp is not None:
            doc.add_heading(f"Pressure Profiles — {label_a} vs {label_b}", level=2)
            img = _fig_to_png(fig_cmp, width=900, height=400, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")
            doc.add_paragraph()
        if fig_bar is not None:
            doc.add_heading(f"ΔP by Segment — {label_a} vs {label_b}", level=2)
            img = _fig_to_png(fig_bar, width=900, height=340, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")

    # ── 8. Method Sensitivity Analysis ───────────────────────────────────────
    if sensitivity_data is not None:
        _sa = sensitivity_data.get("sa", [])
        _sb = sensitivity_data.get("sb", [])
        _fig_s = sensitivity_data.get("fig")

        doc.add_page_break()
        doc.add_heading("8. Method Sensitivity Analysis", level=1)

        _intro = doc.add_paragraph(
            "All 12 method combinations (6 ΔP correlations × 2 void-fraction models) "
            "were evaluated to quantify uncertainty in the total pressure drop due to "
            "method selection.  Each combination runs the full pipeline with pressure "
            "marching.  Combinations that failed to converge are excluded."
        )
        if _intro.runs:
            _intro.runs[0].font.size = Pt(9)
        doc.add_paragraph()

        # Build summary table
        _CORR_SHORT = {
            "Beggs-Brill": "BB", "Friedel": "Friedel",
            "Lockhart_Martinelli": "L-M", "Muller_Steinhagen_Heck": "MSH",
            "Chisholm": "Chisholm", "Kim_Mudawar": "Kim-M",
        }
        _VOID_SHORT = {
            "Homogeneous": "Homo",
            "Rouhani-1 (slip)": "Rouhani-1",
        }
        _va = [r["total_dp_kpa"] for r in _sa if r["ok"]]
        _vb = [r["total_dp_kpa"] for r in _sb if r["ok"]]
        if _va and _vb:
            _a_min, _a_max = min(_va), max(_va)
            _b_min, _b_max = min(_vb), max(_vb)
            _a_sel = results_a["total_dp_kpa"]
            _b_sel = results_b["total_dp_kpa"]
            _overlap = _a_min <= _b_max and _b_min <= _a_max
            _sum_rows = [
                (f"{label_a} — minimum ΔP (kPa)",        f"{_a_min:.3f}",   "—"),
                (f"{label_a} — selected method (kPa)",   f"{_a_sel:.3f}",   "—"),
                (f"{label_a} — maximum ΔP (kPa)",        f"{_a_max:.3f}",   "—"),
                (f"{label_a} — spread (kPa)",             f"{_a_max-_a_min:.3f}", "—"),
                (f"{label_b} — minimum ΔP (kPa)",        "—",               f"{_b_min:.3f}"),
                (f"{label_b} — selected method (kPa)",   "—",               f"{_b_sel:.3f}"),
                (f"{label_b} — maximum ΔP (kPa)",        "—",               f"{_b_max:.3f}"),
                (f"{label_b} — spread (kPa)",             "—",               f"{_b_max-_b_min:.3f}"),
                ("Ranges overlap?",
                 "Yes — ordering depends on method" if _overlap else "No — unambiguous",
                 ""),
            ]
            _kv3_table(doc, _sum_rows, label_a=label_a, label_b=label_b)
            doc.add_paragraph()

        # Per-method ΔP detail table
        _detail_rows = []
        for _r_a, _r_b in zip(_sa, _sb):
            _c = _CORR_SHORT.get(_r_a["correlation"], _r_a["correlation"])
            _v = _VOID_SHORT.get(_r_a["voidage"], _r_a["voidage"])
            _label = f"{_c} / {_v}"
            _va_str = f"{_r_a['total_dp_kpa']:.3f}" if _r_a["ok"] else f"FAIL: {_r_a['error']}"
            _vb_str = f"{_r_b['total_dp_kpa']:.3f}" if _r_b["ok"] else f"FAIL: {_r_b['error']}"
            _detail_rows.append((_label, _va_str, _vb_str))
        _kv3_table(doc, _detail_rows, label_a=label_a, label_b=label_b)
        doc.add_paragraph()

        # Flow regime consistency tables
        doc.add_heading("Flow Regime Consistency", level=2)
        _reg_note = doc.add_paragraph(
            "Regime classification uses fixed Vsg, Vsl, and pipe angle — "
            "it is independent of ΔP correlation. Only the void fraction model (α) "
            "can shift vertical-segment thresholds (bubble/slug/churn). "
            "✓ = all 12 combinations predict the same regime for that segment."
        )
        if _reg_note.runs:
            _reg_note.runs[0].font.size = Pt(8)
            _reg_note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        doc.add_paragraph()

        def _regime_report_table(doc, sens_results, segments_list, case_label):
            ok_results = [r for r in sens_results if r["ok"] and r.get("segment_regimes")]
            if not ok_results or not segments_list:
                return
            void_keys = list(dict.fromkeys(
                _VOID_SHORT.get(r["voidage"], r["voidage"]) for r in ok_results))
            cols = ["Seg", "Pipe", "Orientation"] + void_keys + ["Unanimous"]
            tbl = doc.add_table(rows=len(segments_list) + 1, cols=len(cols))
            tbl.style = "Table Grid"
            _style_header(tbl.rows[0], font_size=8)
            for j, col in enumerate(cols):
                tbl.rows[0].cells[j].text = col
            for i, seg in enumerate(segments_list):
                row = tbl.rows[i + 1]
                if i % 2 == 0:
                    for cell in row.cells:
                        _shd(cell, _ALT_BG)
                by_void = {}
                all_regimes = set()
                for r in ok_results:
                    v = _VOID_SHORT.get(r["voidage"], r["voidage"])
                    rg = r["segment_regimes"][i] if i < len(r["segment_regimes"]) else "—"
                    by_void.setdefault(v, set()).add(rg)
                    all_regimes.add(rg)
                row.cells[0].text = f"#{i+1}"
                dn_str = f"{seg.get('dn','?')}/{seg.get('pn','?')}"
                lined_str = (f" + {seg.get('liner_material','?')} {seg.get('liner_thickness_mm',1.0):.1f}mm"
                             if seg.get("lined") else "")
                row.cells[1].text = dn_str + lined_str
                row.cells[2].text = seg["type"]
                for j, v in enumerate(void_keys):
                    row.cells[3 + j].text = " | ".join(sorted(by_void.get(v, {"—"})))
                row.cells[3 + len(void_keys)].text = (
                    "✓" if len(all_regimes) == 1 else f"✗ ({len(all_regimes)})")
                for cell in row.cells:
                    _cell_font(cell, size_pt=8)
            _set_col_widths(tbl, [0.30, 0.70, 0.90] + [1.60] * len(void_keys) + [0.50])

        doc.add_paragraph(f"{label_a} — Flow Regime by Method")
        _regime_report_table(doc, _sa, results_a.get("segments", []), label_a)
        doc.add_paragraph()
        doc.add_paragraph(f"{label_b} — Flow Regime by Method")
        _regime_report_table(doc, _sb, results_b.get("segments", []), label_b)
        doc.add_paragraph()

        # Chart
        if _fig_s is not None:
            _img_s = _fig_to_png(_fig_s, width=900, height=480, scale=2)
            if _img_s:
                doc.add_picture(BytesIO(_img_s), width=Inches(6.2))
            else:
                doc.add_paragraph("(sensitivity chart rendering timed out)")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    _sp_a = " / ".join(results_a["gas_flows_kgh"].keys())
    _sp_b = " / ".join(results_b["gas_flows_kgh"].keys())
    _sp_str = _sp_a if _sp_a == _sp_b else f"{_sp_a}  |  {_sp_b}"
    note = doc.add_paragraph(
        f"Disclaimer: The correlations implemented were developed primarily for oil/gas "
        f"systems. Application to {_sp_str} duty carries an estimated uncertainty of "
        f"±20–30 %. Use the sensitivity analysis (Section 8, if present) to bracket the "
        f"range across all available methods. Treat as an engineering estimate and validate "
        f"against plant data before use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# ============================================================================
# COMBINED REPORT  (Case A + B + optional C, comparison, sensitivity)
# ============================================================================

def generate_combined_report(
    cases,
    case_labels=None,
    fig_cmp=None,
    fig_bar=None,
    sensitivity_data=None,
):
    """
    Generate a single Word report combining all cases, comparison, and sensitivity.

    cases       : list of result dicts from run_case() – 2 or 3 items.
    case_labels : e.g. ["Case A", "Case B", "Case C"].
    fig_cmp     : overlaid pressure-profile figure (A vs B).
    fig_bar     : per-segment ΔP bar chart (A vs B).
    sensitivity_data : dict {"sa": [...], "sb": [...], "fig": Figure} or None.
    """
    n = len(cases)
    if case_labels is None:
        case_labels = [f"Case {chr(65 + i)}" for i in range(n)]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width    = Inches(8.27);  sec.page_height   = Inches(11.69)
    sec.left_margin   = Inches(0.9);   sec.right_margin  = Inches(0.9)
    sec.top_margin    = Inches(0.9);   sec.bottom_margin = Inches(0.9)

    # ── Title ────────────────────────────────────────────────────────────────
    h = doc.add_heading("Multiphase Pipe Hydraulic Engine — Combined Study Report", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "  ·  ".join(case_labels)
        + f"  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)
    doc.add_paragraph()

    # Section counter
    _sec = [0]
    def _h1(title):
        _sec[0] += 1
        doc.add_heading(f"{_sec[0]}. {title}", level=1)

    # Column widths: 4-col for 3 cases, 3-col for 2 cases (fit within 6.47" text area)
    _col_w = (1.9, 1.5, 1.5, 1.5) if n >= 3 else (2.2, 2.1, 2.1)
    _headers = ["Parameter"] + case_labels[:n]

    def _nt(rows):
        return _kv_n_table(doc, _headers, rows, col_widths=_col_w)

    # ── 1. Method ────────────────────────────────────────────────────────────
    _h1("Method")
    for _txt in [
        ("Six two-phase ΔP correlations are available: Beggs & Brill (1973, default), "
         "Friedel, Lockhart-Martinelli, Müller-Steinhagen & Heck, Chisholm, and "
         "Kim-Mudawar. Cases may use different correlations and void-fraction models. "
         "The pipeline is divided into user-defined segments with pressure marching — "
         "gas density is re-evaluated at each segment inlet. Each segment's ΔP is "
         "decomposed into frictional, gravitational, and accelerational components."),
        ("Void fraction: homogeneous model or Rouhani-1 slip-flow model. "
         "Flow regime classified automatically — Taitel-Dukler + Mandhane-Gregory-Aziz "
         "for horizontal, Wallis/Taitel (1980) for vertical segments. "
         "Gas properties: ideal-gas law, CoolProp viscosities, Dalton's Law for water "
         "vapour. Minor losses: Crane TP-410. Erosion: API RP 14E, C = 100. "
         "Packages: fluids · CoolProp · python-docx."),
    ]:
        _p = doc.add_paragraph(_txt)
        _p.paragraph_format.space_after = Pt(4)
        if _p.runs:
            _p.runs[0].font.size = Pt(9)
    doc.add_paragraph()

    # ── 2. Process Conditions ────────────────────────────────────────────────
    _h1("Process Conditions")
    _all_sp = list(dict.fromkeys(sp for c in cases for sp in c["gas_flows_kgh"]))
    _cond = [
        ("Inlet pressure (bara)",)     + tuple(f"{c['P_bara']:.2f}"   for c in cases),
        ("Temperature (°C)",)          + tuple(f"{c['T_C']:.1f}"      for c in cases),
    ]
    for _sp in _all_sp:
        _cond.append(
            (f"{_sp} mass flow (kg/h)",) +
            tuple(f"{c['gas_flows_kgh'].get(_sp, 0.0):.3f}" for c in cases))
    _cond += [
        ("Liquid type",)               + tuple(c["liquid_type"]               for c in cases),
        ("Liquid vol. flow (m³/h)",)   + tuple(f"{c['q_lye']:.3f}"           for c in cases),
        ("ΔP correlation",)            + tuple(c.get("correlation", "—")      for c in cases),
        ("Void fraction model",)       + tuple(c.get("voidage_method", "—")   for c in cases),
        ("Number of segments",)        + tuple(str(len(c["segments"]))         for c in cases),
    ]
    _nt(_cond)
    doc.add_paragraph()

    # ── 3. Phase Thermodynamics ──────────────────────────────────────────────
    _h1("Phase Thermodynamics  (inlet conditions)")
    _ps = [c["props"] for c in cases]

    def _pfmt(p, key, scale=1.0, fmt=".4f"):
        v = p.get(key)
        return f"{v * scale:{fmt}}" if v is not None else "—"

    _thm = [
        ("Gas density ρ_g (kg/m³)",)      + tuple(_pfmt(p, "rho_g")                    for p in _ps),
        ("Gas mixture MW (g/mol)",)        + tuple(_pfmt(p, "MW_mix_gmol", fmt=".3f")   for p in _ps),
        ("Liquid density ρ_l (kg/m³)",)    + tuple(_pfmt(p, "rho_l", fmt=".2f")         for p in _ps),
        ("Liquid viscosity μ_l (mPa·s)",)  + tuple(_pfmt(p, "mu_l", 1e3)               for p in _ps),
        ("Gas viscosity μ_g (µPa·s)",)     + tuple(_pfmt(p, "mu_g", 1e6, ".2f")        for p in _ps),
        ("Surface tension σ (mN/m)",)      + tuple(_pfmt(p, "sigma", 1e3, ".3f")        for p in _ps),
        ("Mass quality x (%)",)            + tuple(_pfmt(p, "x_gas", 100)              for p in _ps),
        ("Void fraction α (%)",)           + tuple(_pfmt(p, "alpha", 100, ".2f")        for p in _ps),
    ]
    if any(p.get("P_sat_H2O_pa", 0) > 0 for p in _ps):
        _thm += [
            ("H₂O sat. pressure (bara)",) + tuple(
                f"{p.get('P_sat_H2O_pa',0)/1e5:.4f}" if p.get('P_sat_H2O_pa',0) > 0 else "—"
                for p in _ps),
            ("H₂O vapour flow (kg/h)",) + tuple(
                f"{p.get('m_vapor_h2o_kgh',0):.4f}" if p.get('P_sat_H2O_pa',0) > 0 else "—"
                for p in _ps),
        ]
    _nt(_thm)
    doc.add_paragraph()

    # ── 4. System Totals ─────────────────────────────────────────────────────
    _h1("System Totals")
    _max_ve = [max((r["V_m/V_e"] for r in c["grid_records"]), default=0.0) for c in cases]
    _tot = [
        ("Outlet pressure (bara)",)             + tuple(f"{c['outlet_pressure_bara']:.4f}" for c in cases),
        ("Total ΔP (kPa)",)                     + tuple(f"{c['total_dp_kpa']:.4f}"         for c in cases),
        ("Total ΔP (bar)",)                     + tuple(f"{c['total_dp_kpa']/100:.6f}"     for c in cases),
        ("  ↳ Frictional ΔP (kPa)",)            + tuple(f"{c['total_dp_fric_kpa']:.4f}"   for c in cases),
        ("  ↳ Gravitational ΔP (kPa)",)         + tuple(f"{c['total_dp_grav_kpa']:.4f}"   for c in cases),
        ("Pipe length (m)",)                    + tuple(f"{c['pipe_length_m']:.2f}"        for c in cases),
        ("Eff. length incl. fittings (m)",)     + tuple(f"{c['cumulative_distance']:.2f}" for c in cases),
        ("Worst V_m/V_e (–)",)                  + tuple(f"{v:.3f}"                         for v in _max_ve),
    ]
    if n >= 2:
        _dp_delta = cases[1]["total_dp_kpa"] - cases[0]["total_dp_kpa"]
        _tot.append(
            ("ΔP  B − A (kPa)", "—", f"{_dp_delta:+.4f}") + ("—",) * max(0, n - 2)
        )
    _nt(_tot)
    doc.add_paragraph()

    # ── 5. System Total ΔP — Header (C) + Branch (A / B) ─────────────────────
    # Only rendered when at least 3 cases (C = header, A and B = branches)
    if n >= 3:
        _h1("System Total ΔP — Header (C) + Branch")
        _dp_c    = cases[2]["total_dp_kpa"]   # Case C = header
        _dp_a    = cases[0]["total_dp_kpa"]   # Case A = branch A
        _dp_b    = cases[1]["total_dp_kpa"]   # Case B = branch B
        _total_a = _dp_c + _dp_a
        _total_b = _dp_c + _dp_b
        _p_in_c  = cases[2]["P_bara"]
        _p_out_a = _p_in_c - _total_a / 100.0
        _p_out_b = _p_in_c - _total_b / 100.0
        _worst   = max(_total_a, _total_b)
        _worst_lbl = f"{case_labels[2]} + {case_labels[0]}" if _total_a >= _total_b \
                     else f"{case_labels[2]} + {case_labels[1]}"
        _p_out_worst = _p_in_c - _worst / 100.0

        _sys_intro = doc.add_paragraph(
            f"The header ({case_labels[2]}) feeds both branch pipelines. "
            f"The governing (worst-case) path is the one with the higher combined ΔP — "
            f"it sets the minimum required inlet pressure."
        )
        if _sys_intro.runs:
            _sys_intro.runs[0].font.size = Pt(9)
        doc.add_paragraph()

        _sys_rows = [
            ("System inlet pressure (bara)",                  "—",
             "—", f"{_p_in_c:.4f}"),
            (f"Header ΔP — {case_labels[2]} (kPa)",          "—",
             "—", f"{_dp_c:.4f}"),
            (f"Branch ΔP — {case_labels[0]} (kPa)",          f"{_dp_a:.4f}",
             "—", "—"),
            (f"Branch ΔP — {case_labels[1]} (kPa)",          "—",
             f"{_dp_b:.4f}", "—"),
            (f"Total path  {case_labels[2]}+{case_labels[0]} (kPa)",  f"{_total_a:.4f}",
             "—", "—"),
            (f"Total path  {case_labels[2]}+{case_labels[1]} (kPa)",  "—",
             f"{_total_b:.4f}", "—"),
            (f"Worst-case path  ({_worst_lbl})",              f"{_worst:.4f}" if _total_a >= _total_b else "—",
             f"{_worst:.4f}" if _total_b > _total_a else "—", "—"),
            (f"System outlet pressure — path A (bara)",       f"{_p_out_a:.4f}",
             "—", "—"),
            (f"System outlet pressure — path B (bara)",       "—",
             f"{_p_out_b:.4f}", "—"),
            (f"Min. outlet pressure / worst case (bara)",
             f"{_p_out_worst:.4f}" if _total_a >= _total_b else "—",
             f"{_p_out_worst:.4f}" if _total_b > _total_a else "—", "—"),
        ]
        # Four-column table: Parameter | Case A | Case B | Case C
        _sys_headers = ["Parameter", case_labels[0], case_labels[1], case_labels[2]]
        _kv_n_table(doc, _sys_headers, _sys_rows, col_widths=(2.4, 1.3, 1.3, 1.3))
        doc.add_paragraph()

    # ── 6+. Segment Analysis (one section per case) ───────────────────────────
    _SC = ["Seg", "Pipe", "ID (mm)", "Type", "L (m)", "Regime",
           "V_m (m/s)", "V_e (m/s)", "V_m/V_e", "ΔP (kPa)", "P_out (bara)"]
    _SW = [0.28, 0.55, 0.45, 0.88, 0.42, 0.90, 0.55, 0.55, 0.48, 0.55, 0.56]

    def _seg_tbl(records):
        if not records:
            return
        tbl = doc.add_table(rows=len(records) + 1, cols=len(_SC))
        tbl.style = "Table Grid"
        _style_header(tbl.rows[0], font_size=8)
        for j, col in enumerate(_SC):
            tbl.rows[0].cells[j].text = col
        for i, rec in enumerate(records, start=1):
            row = tbl.rows[i]
            if i % 2 == 0:
                for cell in row.cells:
                    _shd(cell, _ALT_BG)
            for j, col in enumerate(_SC):
                row.cells[j].text = str(rec.get(col, ""))
                _cell_font(row.cells[j], size_pt=8)
        _set_col_widths(tbl, _SW)

    for c, lbl in zip(cases, case_labels):
        _h1(f"Segment Analysis — {lbl}")
        _seg_tbl(c["grid_records"])
        doc.add_paragraph()

    # ── Visualisations ────────────────────────────────────────────────────────
    _has_figs = (fig_cmp is not None or fig_bar is not None
                 or any(c.get("fig_sch") or c.get("fig_prof") for c in cases))
    if _has_figs:
        doc.add_page_break()
        _h1("Visualisations")

        if fig_cmp is not None:
            doc.add_heading("Pressure Profiles — A vs B", level=2)
            img = _fig_to_png(fig_cmp, width=900, height=400, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")
            doc.add_paragraph()

        if fig_bar is not None:
            doc.add_heading("ΔP by Segment — A vs B", level=2)
            img = _fig_to_png(fig_bar, width=900, height=340, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")
            doc.add_paragraph()

        for c, lbl in zip(cases, case_labels):
            _fs = c.get("fig_sch")
            _fp = c.get("fig_prof")
            if _fs is not None or _fp is not None:
                doc.add_heading(f"{lbl} — Pipeline", level=2)
            if _fs is not None:
                img = _fig_to_png(_fs, width=900, height=440, scale=2)
                if img:
                    doc.add_picture(BytesIO(img), width=Inches(6.2))
                else:
                    doc.add_paragraph("(chart rendering timed out)")
                doc.add_paragraph()
            if _fp is not None:
                img = _fig_to_png(_fp, width=900, height=320, scale=2)
                if img:
                    doc.add_picture(BytesIO(img), width=Inches(6.2))
                else:
                    doc.add_paragraph("(chart rendering timed out)")
                doc.add_paragraph()

    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    if sensitivity_data is not None:
        _sa = sensitivity_data.get("sa", [])
        _sb = sensitivity_data.get("sb", [])
        _fig_s = sensitivity_data.get("fig")

        doc.add_page_break()
        _h1("Method Sensitivity Analysis")

        _intro = doc.add_paragraph(
            "All 12 method combinations (6 ΔP correlations × 2 void-fraction models) "
            "were evaluated for Cases A and B to quantify uncertainty due to method "
            "selection.  Combinations that failed to converge are excluded."
        )
        if _intro.runs:
            _intro.runs[0].font.size = Pt(9)
        doc.add_paragraph()

        _CORR_S = {"Beggs-Brill": "BB", "Friedel": "Friedel",
                   "Lockhart_Martinelli": "L-M", "Muller_Steinhagen_Heck": "MSH",
                   "Chisholm": "Chisholm", "Kim_Mudawar": "Kim-M"}
        _VOID_S = {"Homogeneous": "Homo", "Rouhani-1 (slip)": "Rouhani-1"}

        _va = [r["total_dp_kpa"] for r in _sa if r["ok"]]
        _vb = [r["total_dp_kpa"] for r in _sb if r["ok"]]
        if _va and _vb:
            _a_sel = cases[0]["total_dp_kpa"]
            _b_sel = cases[1]["total_dp_kpa"]
            _overlap = min(_va) <= max(_vb) and min(_vb) <= max(_va)
            _sum = [
                ("Case A — minimum ΔP (kPa)",       f"{min(_va):.3f}",  "—"),
                ("Case A — selected method (kPa)",  f"{_a_sel:.3f}",    "—"),
                ("Case A — maximum ΔP (kPa)",       f"{max(_va):.3f}",  "—"),
                ("Case A — spread (kPa)",            f"{max(_va)-min(_va):.3f}", "—"),
                ("Case B — minimum ΔP (kPa)",       "—",               f"{min(_vb):.3f}"),
                ("Case B — selected method (kPa)",  "—",               f"{_b_sel:.3f}"),
                ("Case B — maximum ΔP (kPa)",       "—",               f"{max(_vb):.3f}"),
                ("Case B — spread (kPa)",            "—",               f"{max(_vb)-min(_vb):.3f}"),
                ("Ranges overlap?",
                 "Yes — ordering method-dependent" if _overlap else "No — unambiguous", ""),
            ]
            _kv3_table(doc, _sum)
            doc.add_paragraph()

        _det = []
        for _ra, _rb in zip(_sa, _sb):
            _c = _CORR_S.get(_ra["correlation"], _ra["correlation"])
            _v = _VOID_S.get(_ra["voidage"], _ra["voidage"])
            _det.append((
                f"{_c} / {_v}",
                f"{_ra['total_dp_kpa']:.3f}" if _ra["ok"] else f"FAIL: {_ra['error']}",
                f"{_rb['total_dp_kpa']:.3f}" if _rb["ok"] else f"FAIL: {_rb['error']}",
            ))
        _kv3_table(doc, _det)
        doc.add_paragraph()

        if _fig_s is not None:
            _img_s = _fig_to_png(_fig_s, width=900, height=480, scale=2)
            if _img_s:
                doc.add_picture(BytesIO(_img_s), width=Inches(6.2))
            else:
                doc.add_paragraph("(sensitivity chart rendering timed out)")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    _all_gas = sorted({sp for c in cases for sp in c["gas_flows_kgh"]})
    _all_liq = sorted({c["liquid_type"] for c in cases})
    note = doc.add_paragraph(
        f"Disclaimer: The correlations implemented were developed primarily for oil/gas "
        f"systems. Application to {' / '.join(_all_gas)} / {' / '.join(_all_liq)} duty "
        f"carries an estimated uncertainty of ±20–30 %. Use the sensitivity analysis "
        f"(if present above) to bracket the range across all available methods. "
        f"Treat as an engineering estimate and validate against plant data before "
        f"use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
