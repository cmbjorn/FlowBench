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

    # ── 1. Process Conditions ────────────────────────────────────────────────
    doc.add_heading("1. Process Conditions", level=1)
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

    # ── 2. Phase Thermodynamics ──────────────────────────────────────────────
    doc.add_heading("2. Phase Thermodynamics", level=1)
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

    # ── 3. Segment Analysis ──────────────────────────────────────────────────
    doc.add_heading("3. Segment Analysis", level=1)

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

    # ── 4. System Totals ─────────────────────────────────────────────────────
    doc.add_heading("4. System Totals", level=1)
    _kv_table(doc, [
        ("Total Pressure Drop ΔP",                f"{total_dp_kpa:.4f} kPa"),
        ("Total Pressure Drop ΔP",                f"{total_dp_kpa / 100:.6f} bar"),
        ("Outlet Pressure",                            f"{outlet_pressure_bara:.4f} bara"),
        ("Pipe Length (physical segments)",            f"{pipe_length_m:.2f} m"),
        ("Effective Length (incl. fittings)",          f"{cumulative_distance:.2f} m"),
    ])

    # ── 5. Visualisations ────────────────────────────────────────────────────
    if fig_sch is not None or fig_prof is not None:
        try:
            import plotly.io as pio

            doc.add_page_break()
            doc.add_heading("5. Visualisations", level=1)

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
    note = doc.add_paragraph(
        "Disclaimer: Results are based on the Beggs & Brill (1973) correlation applied to "
        "H₂/O₂/KOH systems. This correlation was originally developed for oil/gas "
        "systems; application to alkaline electrolysis duty should be treated as an engineering "
        "estimate. Validate against experimental data before use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
