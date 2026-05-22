# report_generator.py
"""
Generates a Word (.docx) calculation report from multiphase hydraulics calculation results.
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
_png_cache: dict = {}  # (id(fig), width, height, scale) → PNG bytes or None


def _fig_to_png(fig, width=900, height=400, scale=2):
    """Render a Plotly figure to PNG bytes with a hard timeout.
    Returns bytes on success, None if kaleido hangs or fails.
    Checks _png_cache first so prefetch_figures() avoids re-rendering.
    """
    import plotly.io as pio
    _key = (id(fig), width, height, scale)
    if _key in _png_cache:
        return _png_cache[_key]
    _ex = ThreadPoolExecutor(max_workers=1)
    try:
        _fut = _ex.submit(pio.to_image, fig,
                          format="png", width=width, height=height, scale=scale)
        result = _fut.result(timeout=_IMG_TIMEOUT)
    except (_FuturesTimeout, Exception):
        result = None
    finally:
        _ex.shutdown(wait=False)
    _png_cache[_key] = result
    return result


def prefetch_figures(specs):
    """Render a list of (fig, width, height, scale) tuples in parallel.

    Call this before generating reports so all kaleido renders happen
    concurrently; subsequent _fig_to_png() calls are instant cache hits.
    specs with None fig are silently skipped.
    """
    import plotly.io as pio
    from concurrent.futures import wait as _wait

    to_render = [
        (fig, w, h, s)
        for fig, w, h, s in specs
        if fig is not None and (id(fig), w, h, s) not in _png_cache
    ]
    if not to_render:
        return

    _ex = ThreadPoolExecutor(max_workers=min(len(to_render), 4))
    try:
        futures = {
            _ex.submit(pio.to_image, fig,
                       format="png", width=w, height=h, scale=s): (id(fig), w, h, s)
            for fig, w, h, s in to_render
        }
        done, _ = _wait(futures, timeout=_IMG_TIMEOUT)
        for fut in done:
            key = futures[fut]
            try:
                _png_cache[key] = fut.result()
            except Exception:
                _png_cache[key] = None
        for fut, key in futures.items():
            if key not in _png_cache:
                _png_cache[key] = None
    finally:
        _ex.shutdown(wait=False)


def clear_fig_cache():
    """Clear the PNG render cache (call between report sessions if needed)."""
    _png_cache.clear()


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


def _fig_caption(doc, text):
    """Add a small grey italic caption paragraph below a figure."""
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(6)
    if p.runs:
        p.runs[0].font.size      = Pt(8)
        p.runs[0].font.italic    = True
        p.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)


def _cell_font(cell, size_pt=9):
    for para in cell.paragraphs:
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(1)
        for run in para.runs:
            run.font.size = Pt(size_pt)


def _add_toc(doc):
    """Insert a Word TOC field (levels 1–2). Word updates it on open."""
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _el

    toc_h = doc.add_heading("Table of Contents", level=1)
    toc_h.runs[0].font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    para = doc.add_paragraph()
    run  = para.add_run()

    def _fc(type_):
        fc = _el("w:fldChar"); fc.set(_qn("w:fldCharType"), type_); return fc

    run._r.append(_fc("begin"))
    instr = _el("w:instrText")
    instr.set(_qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    run._r.append(instr)
    run._r.append(_fc("separate"))
    placeholder = _el("w:r")
    t = _el("w:t"); t.text = "[Right-click → Update Field to populate]"
    placeholder.append(t)
    para._p.append(placeholder)
    run._r.append(_fc("end"))

    doc.add_page_break()


def _add_footer_page_numbers(doc):
    """Add a centred PAGE / NUMPAGES field in the first section's footer."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    footer = doc.sections[0].footer
    footer.is_linked_to_previous = False
    para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Remove any existing runs, keep pPr (alignment)
    p = para._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)

    def _field_runs(instr):
        """Return three w:r elements: fldChar begin, instrText, fldChar end."""
        r1 = OxmlElement("w:r")
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin"); r1.append(fc)
        r2 = OxmlElement("w:r")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = f" {instr} "; r2.append(it)
        r3 = OxmlElement("w:r")
        fc2 = OxmlElement("w:fldChar"); fc2.set(qn("w:fldCharType"), "end"); r3.append(fc2)
        return r1, r2, r3

    for r in _field_runs("PAGE"):
        p.append(r)
    r_sep = OxmlElement("w:r")
    t_sep = OxmlElement("w:t"); t_sep.set(qn("xml:space"), "preserve"); t_sep.text = " / "
    r_sep.append(t_sep); p.append(r_sep)
    for r in _field_runs("NUMPAGES"):
        p.append(r)


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
    case_label="Case",
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
    h = doc.add_heading(f"Branch Line Hydraulic Calculation — {case_label}", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        f"Gas–Liquid Piping  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
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

    # ── 1. Purpose ───────────────────────────────────────────────────────────
    doc.add_heading("1. Purpose", level=1)
    _purpose = doc.add_paragraph(
        f"This calculation determines the two-phase pressure drop along the {case_label} "
        f"branch pipeline, which carries {_gas_label} and {liquid_type} from a "
        f"process unit to a gas–liquid separator. "
        f"The result — inlet pressure, outlet pressure, and total ΔP — is used to "
        f"size the pipe and, in combination with the collecting header calculation, "
        f"to establish the required equipment outlet pressure."
    )
    _purpose.paragraph_format.space_after = Pt(4)
    if _purpose.runs:
        _purpose.runs[0].font.size = Pt(9)
    doc.add_paragraph()

    # ── 2. Method ────────────────────────────────────────────────────────────
    doc.add_heading("2. Method", level=1)

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

    # ── 3. Process Conditions ────────────────────────────────────────────────
    doc.add_heading("3. Process Conditions", level=1)
    _cond_rows = [
        ("Case",               case_label),
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

    # ── 4. Phase Thermodynamics ──────────────────────────────────────────────
    doc.add_heading("4. Phase Thermodynamics", level=1)
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

    # ── 5. Segment Analysis ──────────────────────────────────────────────────
    doc.add_heading("5. Segment Analysis", level=1)

    # Subset of columns that fits A4 portrait (6.47" available between margins)
    _COLS   = ["Seg", "Pipe", "ID (mm)", "Type", "L (m)", "L_eq (m)", "Fittings", "Regime",
               "V_m (m/s)", "V_m/V_e", "ΔP (kPa)", "P_out (bara)"]
    _WIDTHS = [0.25,  0.50,   0.42,     0.80,   0.38,   0.42,      0.70,      0.85,
               0.48,         0.44,      0.50,       0.53]

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

    # ── 6. System Totals ─────────────────────────────────────────────────────
    doc.add_heading("6. System Totals", level=1)
    _kv_table(doc, [
        ("Case",                                       case_label),
        ("Inlet Pressure",                             f"{P_bara:.4f} bara"),
        ("Outlet Pressure",                            f"{outlet_pressure_bara:.4f} bara"),
        ("Total Pressure Drop ΔP",                    f"{total_dp_kpa:.4f} kPa"),
        ("Total Pressure Drop ΔP",                    f"{total_dp_kpa / 100:.6f} bar"),
        ("Pipe Length (physical segments)",            f"{pipe_length_m:.2f} m"),
        ("Effective Length (incl. fittings)",          f"{cumulative_distance:.2f} m"),
    ])

    # ── 7. Visualisations ────────────────────────────────────────────────────
    if fig_sch is not None or fig_prof is not None:
        doc.add_page_break()
        doc.add_heading("7. Visualisations", level=1)

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
        f"Engineering Note: The two-phase correlations used here were developed primarily "
        f"for oil/gas systems. Their application to this service ({_gas_str} / "
        f"{liquid_type}) carries an estimated uncertainty of ±20–30 %. Use the sensitivity "
        f"analysis (Compare tab) to bracket the ΔP range across all available methods. "
        f"Treat as a first-pass engineering estimate; validate against commissioning data "
        f"before use in safety-critical design."
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
    stack_dp=None,
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
    h = doc.add_heading(f"Branch Line Comparison — {label_a}  vs.  {label_b}", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        f"Gas–Liquid Piping  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)

    doc.add_paragraph()

    # ── 1. Purpose ───────────────────────────────────────────────────────────
    doc.add_heading("1. Purpose", level=1)
    _p = doc.add_paragraph(
        f"This report compares the two-phase pressure drop along two branch pipelines: "
        f"{label_a} and {label_b}. Both carry gas–liquid flow from an upstream process unit "
        f"to a separator. The comparison supports pipe sizing — selecting the smallest "
        f"bore that keeps ΔP within budget and velocity below the erosion threshold — "
        f"and identifies which line governs the required equipment outlet pressure. "
        f"The sensitivity analysis (if run) quantifies the uncertainty in ΔP due to "
        f"correlation choice across all 12 method combinations."
    )
    _p.paragraph_format.space_after = Pt(4)
    if _p.runs:
        _p.runs[0].font.size = Pt(9)
    doc.add_paragraph()

    # ── 2. Method ────────────────────────────────────────────────────────────
    doc.add_heading("2. Method", level=1)
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
        ("Where a sensitivity analysis was run, the final section shows total ΔP and flow "
         "regimes across all 12 method combinations (6 correlations × 2 void-fraction models). "
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
    _COLS   = ["Seg","Pipe","ID (mm)","Type","L (m)","L_eq (m)","Fittings","Regime",
               "V_m (m/s)","V_m/V_e","ΔP (kPa)","P_out (bara)"]
    _WIDTHS = [0.25, 0.50, 0.42, 0.80, 0.38, 0.42, 0.70, 0.85, 0.48, 0.44, 0.50, 0.53]

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

    # ── Stack ΔP ──────────────────────────────────────────────────────────────
    if stack_dp is not None:
        _gsh  = stack_dp.get("gsr_h2") or {}
        _gso  = stack_dp.get("gsr_o2") or {}
        _ph   = stack_dp.get("P_sep_h2")
        _po   = stack_dp.get("P_sep_o2")
        _la_s = stack_dp.get("label_a", label_a)
        _lb_s = stack_dp.get("label_b", label_b)
        _p_in_a  = _gsh.get("P_line_in", 0.0)
        _p_in_b  = _gso.get("P_line_in", 0.0)
        _dp_s    = _p_in_a - _p_in_b
        _dp_kpa  = _dp_s * 100.0
        _dp_mbar = _dp_kpa * 10.0

        doc.add_page_break()
        doc.add_heading("Generator Differential Pressure", level=1)

        doc.add_heading("Target Conditions", level=2)
        _kv_table(doc, [
            (f"{_la_s} system — separator target pressure (bara)", f"{_ph:.3f}" if _ph is not None else "—"),
            (f"{_lb_s} system — separator target pressure (bara)", f"{_po:.3f}" if _po is not None else "—"),
        ])
        doc.add_paragraph()

        doc.add_heading(f"{_la_s} System  (Branch → Header C → Separator)", level=2)
        _kv_table(doc, [
            (f"{_la_s} line inlet pressure (bara)",       f"{_gsh.get('P_line_in', 0):.4f}"),
            (f"{_la_s} line ΔP (kPa)",                    f"{_gsh.get('dp_line', 0):.3f}"),
            (f"{_la_s} outlet / Header C inlet (bara)",   f"{_gsh.get('P_line_out', 0):.4f}"),
            ("Header C + T-seg ΔP (kPa)",                 f"{_gsh.get('dp_hdr', 0):.3f}"),
            (f"{_la_s} system separator pressure (bara)", f"{_gsh.get('P_sep', 0):.4f}"),
        ])
        doc.add_paragraph()

        doc.add_heading(f"{_lb_s} System  (Branch → Header D → Separator)", level=2)
        _kv_table(doc, [
            (f"{_lb_s} line inlet pressure (bara)",       f"{_gso.get('P_line_in', 0):.4f}"),
            (f"{_lb_s} line ΔP (kPa)",                    f"{_gso.get('dp_line', 0):.3f}"),
            (f"{_lb_s} outlet / Header D inlet (bara)",   f"{_gso.get('P_line_out', 0):.4f}"),
            ("Header D + T-seg ΔP (kPa)",                 f"{_gso.get('dp_hdr', 0):.3f}"),
            (f"{_lb_s} system separator pressure (bara)", f"{_gso.get('P_sep', 0):.4f}"),
        ])
        doc.add_paragraph()

        doc.add_heading(f"Generator ΔP Result  (P_inlet_{_la_s} − P_inlet_{_lb_s})", level=2)
        _kv_table(doc, [
            (f"ΔP  {_la_s} − {_lb_s}  (bara)",  f"{_dp_s:.4f}"),
            (f"ΔP  {_la_s} − {_lb_s}  (kPa)",   f"{_dp_kpa:.2f}"),
            (f"ΔP  {_la_s} − {_lb_s}  (mbar)",  f"{_dp_mbar:.1f}"),
        ])
        doc.add_paragraph()

    # ── Engineering note ──────────────────────────────────────────────────────
    doc.add_paragraph()
    _sp_a = " / ".join(results_a["gas_flows_kgh"].keys())
    _sp_b = " / ".join(results_b["gas_flows_kgh"].keys())
    _sp_str = _sp_a if _sp_a == _sp_b else f"{_sp_a}  |  {_sp_b}"
    note = doc.add_paragraph(
        f"Engineering Note: The two-phase correlations used here were developed primarily "
        f"for oil/gas systems. Their application to this service ({_sp_str}) carries "
        f"an estimated uncertainty of ±20–30 %. Use the sensitivity analysis (if present) "
        f"to bracket the ΔP range across all available methods. Treat as a first-pass "
        f"engineering estimate; validate against commissioning data before use in "
        f"safety-critical design."
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
    stack_dp=None,
    dn_study_data=None,
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
    h = doc.add_heading("Gas–Liquid Piping Hydraulic Study — Combined Report", level=0)
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

    _add_footer_page_numbers(doc)
    _add_toc(doc)

    # ── Helpers ───────────────────────────────────────────────────────────────
    _sec = [0]
    def _h1(title):
        _sec[0] += 1
        doc.add_heading(f"{_sec[0]}. {title}", level=1)

    # Column widths fit within 6.47" text area
    if n >= 4:
        _col_w = (1.6, 1.2, 1.2, 1.2, 1.2)
    elif n >= 3:
        _col_w = (1.9, 1.5, 1.5, 1.5)
    else:
        _col_w = (2.2, 2.1, 2.1)
    _headers = ["Parameter"] + case_labels[:n]

    def _nt(rows):
        return _kv_n_table(doc, _headers, rows, col_widths=_col_w)

    def _body_para(text):
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(4)
        if p.runs:
            p.runs[0].font.size = Pt(9)
        return p

    # Identify header vs branch cases once up front
    _is_hdr = [not c.get("segments") for c in cases]

    # ════════════════════════════════════════════════════════════════════════
    # 1. PURPOSE
    # ════════════════════════════════════════════════════════════════════════
    _h1("Purpose")
    _lbl_branch = "  ·  ".join(case_labels[:2]) if n >= 2 else case_labels[0]
    _lbl_header = "  ·  ".join(case_labels[2:]) if n >= 3 else ""
    _p_lines = [
        f"This study sizes the individual branch pipelines ({_lbl_branch}) and "
        f"{'collecting headers (' + _lbl_header + ') ' if _lbl_header else ''}"
        f"for a gas–liquid piping system. Each branch carries a two-phase mixture "
        f"from a process unit to a collecting header, which conveys the combined "
        f"flow to the gas–liquid separator."
    ]
    if n >= 4:
        _p_lines.append(
            f"Two independent collecting systems are evaluated: "
            f"{case_labels[0]} branches feed {case_labels[2]}, and "
            f"{case_labels[1]} branches feed {case_labels[3]}. "
            f"The goal-seek function finds the required branch inlet pressures for each "
            f"system such that each separator arrives at its target operating pressure. "
            f"The difference between the two branch inlet pressures is the differential "
            f"pressure across the upstream process unit."
        )
    elif n >= 3:
        _p_lines.append(
            f"The goal-seek function finds the required branch-line inlet pressure "
            f"such that the separator arrives at the target operating pressure. "
            f"For a two-line system, the difference between the two branch inlet "
            f"pressures gives the differential pressure across the upstream process unit."
        )
    for _txt in _p_lines:
        _body_para(_txt)
    doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 2. KEY RESULTS
    # ════════════════════════════════════════════════════════════════════════
    _h1("Key Results")

    # Per-case summary row (inlet P, total ΔP, outlet/separator P)
    _kr_rows = [
        ("Inlet / tap inlet pressure (bara)",)
            + tuple(f"{c['P_bara']:.4f}" for c in cases),
        ("Total ΔP (kPa)",)
            + tuple(f"{c['total_dp_kpa']:.3f}" for c in cases),
        ("Outlet / separator pressure (bara)",)
            + tuple(f"{c['outlet_pressure_bara']:.4f}" for c in cases),
    ]
    _nt(_kr_rows)
    doc.add_paragraph()

    # Combined system totals (branch + header)
    if n >= 3:
        _systems = []
        if n >= 3 and not _is_hdr[2]:
            pass  # case 2 is not a header — skip
        elif n >= 3 and _is_hdr[2]:
            _dp_br_1  = cases[0]["total_dp_kpa"]
            _dp_hd_1  = cases[2]["total_dp_kpa"]
            _p_sep_1  = cases[2].get("P_separator_bara", cases[2]["outlet_pressure_bara"])
            _systems.append((f"{case_labels[0]}+{case_labels[2]}",
                              _dp_br_1, _dp_hd_1, _dp_br_1 + _dp_hd_1, _p_sep_1))
        if n >= 4 and _is_hdr[3]:
            _dp_br_2  = cases[1]["total_dp_kpa"]
            _dp_hd_2  = cases[3]["total_dp_kpa"]
            _p_sep_2  = cases[3].get("P_separator_bara", cases[3]["outlet_pressure_bara"])
            _systems.append((f"{case_labels[1]}+{case_labels[3]}",
                              _dp_br_2, _dp_hd_2, _dp_br_2 + _dp_hd_2, _p_sep_2))
        if _systems:
            doc.add_heading("Combined System ΔP", level=2)
            if len(_systems) == 1:
                _sc_hdrs = ["Parameter", _systems[0][0]]
                _sc_cw   = (3.5, 3.0)
                _sc_data = [
                    ("Branch ΔP (kPa)",                              f"{_systems[0][1]:.3f}"),
                    ("Header ΔP — worst arm + T-seg (kPa)",          f"{_systems[0][2]:.3f}"),
                    ("System total ΔP (kPa)",                        f"{_systems[0][3]:.3f}"),
                    ("Separator pressure (bara)",                     f"{_systems[0][4]:.4f}"),
                ]
            else:
                _sc_hdrs = ["Parameter"] + [s[0] for s in _systems]
                _sc_cw   = (2.47, 2.0, 2.0)
                _sc_data = [
                    ("Branch ΔP (kPa)",)
                        + tuple(f"{s[1]:.3f}" for s in _systems),
                    ("Header ΔP — worst arm + T-seg (kPa)",)
                        + tuple(f"{s[2]:.3f}" for s in _systems),
                    ("System total ΔP (kPa)",)
                        + tuple(f"{s[3]:.3f}" for s in _systems),
                    ("Separator pressure (bara)",)
                        + tuple(f"{s[4]:.4f}" for s in _systems),
                ]
            _kv_n_table(doc, _sc_hdrs, _sc_data, col_widths=_sc_cw)
            doc.add_paragraph()

    # Generator ΔP summary (branch A inlet − branch B inlet)
    if n >= 2 and not _is_hdr[0] and not _is_hdr[1]:
        _p_in_a   = cases[0]["P_bara"]
        _p_in_b   = cases[1]["P_bara"]
        _gen_dp   = _p_in_a - _p_in_b
        _gen_kpa  = _gen_dp * 100.0
        _gen_mbar = _gen_kpa * 10.0
        doc.add_heading("Generator Differential Pressure", level=2)
        _kv_table(doc, [
            (f"{case_labels[0]} branch inlet pressure (bara)",              f"{_p_in_a:.4f}"),
            (f"{case_labels[1]} branch inlet pressure (bara)",              f"{_p_in_b:.4f}"),
            (f"Generator ΔP  [{case_labels[0]} − {case_labels[1]}]  (bara)", f"{_gen_dp:.4f}"),
            (f"Generator ΔP  [{case_labels[0]} − {case_labels[1]}]  (kPa)",  f"{_gen_kpa:.2f}"),
            (f"Generator ΔP  [{case_labels[0]} − {case_labels[1]}]  (mbar)", f"{_gen_mbar:.1f}"),
        ], col_widths=(4.0, 2.47))
        doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 3. METHOD
    # ════════════════════════════════════════════════════════════════════════
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
        _body_para(_txt)
    doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 4. PROCESS CONDITIONS
    # ════════════════════════════════════════════════════════════════════════
    _h1("Process Conditions")
    _all_sp = list(dict.fromkeys(sp for c in cases for sp in c["gas_flows_kgh"]))
    _cond = [
        ("Inlet pressure (bara)",)   + tuple(f"{c['P_bara']:.2f}"  for c in cases),
        ("Temperature (°C)",)        + tuple(f"{c['T_C']:.1f}"     for c in cases),
    ]
    for _sp in _all_sp:
        _cond.append(
            (f"{_sp} mass flow (kg/h)",) +
            tuple(f"{c['gas_flows_kgh'].get(_sp, 0.0):.3f}" for c in cases))
    _cond += [
        ("Liquid type",)             + tuple(c["liquid_type"]              for c in cases),
        ("Liquid vol. flow (m³/h)",) + tuple(f"{c['q_lye']:.3f}"          for c in cases),
        ("ΔP correlation",)          + tuple(c.get("correlation", "—")     for c in cases),
        ("Void fraction model",)     + tuple(c.get("voidage_method", "—")  for c in cases),
        ("Segments / taps",)         + tuple(
            str(len(c["segments"])) if c.get("segments")
            else f"{c.get('n_left',0)}L + {c.get('n_right',0)}R taps"
            for c in cases),
    ]
    _nt(_cond)
    doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 5. BRANCH LINE RESULTS
    # ════════════════════════════════════════════════════════════════════════
    _branch_cases  = [c   for c, h in zip(cases, _is_hdr) if not h]
    _branch_labels = [lbl for lbl, h in zip(case_labels, _is_hdr) if not h]
    if _branch_cases:
        _h1("Branch Line Results")
        nb = len(_branch_cases)
        if nb >= 3:
            _bcw = (1.9, 1.5, 1.5, 1.5)[:nb + 1]
        elif nb == 2:
            _bcw = (2.5, 2.0, 2.0)
        else:
            _bcw = (3.0, 3.47)
        _bhdrs = ["Parameter"] + _branch_labels
        _max_ve = [
            max((r.get("V_m/V_e", 0) for r in c["grid_records"]), default=0.0)
            for c in _branch_cases
        ]
        _br_rows = [
            ("Inlet pressure (bara)",)
                + tuple(f"{c['P_bara']:.4f}"           for c in _branch_cases),
            ("Total ΔP (kPa)",)
                + tuple(f"{c['total_dp_kpa']:.4f}"     for c in _branch_cases),
            ("  ↳ Frictional ΔP (kPa)",)
                + tuple(f"{c['total_dp_fric_kpa']:.4f}" for c in _branch_cases),
            ("  ↳ Gravitational ΔP (kPa)",)
                + tuple(f"{c['total_dp_grav_kpa']:.4f}" for c in _branch_cases),
            ("Outlet pressure (bara)",)
                + tuple(f"{c['outlet_pressure_bara']:.4f}" for c in _branch_cases),
            ("Pipe length (m)",)
                + tuple(f"{c['pipe_length_m']:.2f}"    for c in _branch_cases),
            ("Worst V_m/V_e (–)",)
                + tuple(f"{v:.3f}"                      for v in _max_ve),
        ]
        if nb >= 2:
            _dp_delta = _branch_cases[1]["total_dp_kpa"] - _branch_cases[0]["total_dp_kpa"]
            _br_rows.append(
                (f"ΔP  {_branch_labels[1]} − {_branch_labels[0]}  (kPa)",
                 "—", f"{_dp_delta:+.4f}") + ("—",) * max(0, nb - 2)
            )
        _kv_n_table(doc, _bhdrs, _br_rows, col_widths=_bcw)
        doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 6. COMBINED SYSTEM ΔP — one sub-section per header case
    # ════════════════════════════════════════════════════════════════════════
    _hdr_triples = [
        (idx, c, lbl)
        for idx, (c, lbl, h) in enumerate(zip(cases, case_labels, _is_hdr)) if h
    ]
    if _hdr_triples:
        _h1("Combined System ΔP — Header + Branch")
        for _hidx, _hc, _hlbl in _hdr_triples:
            # Header at index 2 pairs with branch at 0; index 3 pairs with branch at 1
            _br_idx = _hidx - 2
            if 0 <= _br_idx < n and not _is_hdr[_br_idx]:
                _bc, _blbl = cases[_br_idx], case_labels[_br_idx]
            else:
                _bc, _blbl = cases[0], case_labels[0]

            _dp_hdr   = _hc["total_dp_kpa"]
            _dp_br    = _bc["total_dp_kpa"]
            _p_in_br  = _bc["P_bara"]
            _p_out_br = _bc["outlet_pressure_bara"]
            _p_sep    = _hc.get("P_separator_bara", _hc["outlet_pressure_bara"])

            doc.add_heading(f"{_blbl} → {_hlbl} → Separator", level=2)
            _sys_p = doc.add_paragraph(
                f"The {_hlbl} collecting header receives flow from {_blbl} branch lines "
                f"and delivers the combined flow to the separator via the T-segment. "
                f"Equal flow per tap is assumed; the governing (highest-ΔP) arm sets the "
                f"required tap inlet pressure. This is conservative when header ΔP ≪ branch ΔP."
            )
            if _sys_p.runs:
                _sys_p.runs[0].font.size = Pt(9)
            doc.add_paragraph()
            _kv_table(doc, [
                (f"Branch inlet pressure — {_blbl} (bara)",          f"{_p_in_br:.4f}"),
                (f"Branch ΔP — {_blbl} (kPa)",                       f"{_dp_br:.4f}"),
                ("Branch outlet / header tap inlet pressure (bara)",  f"{_p_out_br:.4f}"),
                (f"Header ΔP (worst arm + T-seg) — {_hlbl} (kPa)",   f"{_dp_hdr:.4f}"),
                (f"Total system ΔP — {_blbl} + {_hlbl} (kPa)",       f"{_dp_br + _dp_hdr:.4f}"),
                ("Separator connection pressure (bara)",               f"{_p_sep:.4f}"),
            ])
            doc.add_paragraph()

    # ════════════════════════════════════════════════════════════════════════
    # 7. VISUALISATIONS  (page break)
    # ════════════════════════════════════════════════════════════════════════
    _has_figs = (fig_cmp is not None or fig_bar is not None
                 or any(c.get("fig_sch") or c.get("fig_prof") for c in cases))
    if _has_figs:
        doc.add_page_break()
        _h1("Visualisations")

        if fig_cmp is not None:
            _la_v = case_labels[0] if case_labels else "A"
            _lb_v = case_labels[1] if len(case_labels) > 1 else "B"
            doc.add_heading(f"Pressure Profiles — {_la_v} vs {_lb_v}", level=2)
            img = _fig_to_png(fig_cmp, width=900, height=400, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")
            _fig_caption(doc,
                f"Figure: Absolute pressure (bara) along the pipeline for {_la_v} (solid) "
                f"and {_lb_v} (dashed). A steeper slope indicates higher local resistance.")

        if fig_bar is not None:
            _la_v = case_labels[0] if case_labels else "A"
            _lb_v = case_labels[1] if len(case_labels) > 1 else "B"
            doc.add_heading(f"ΔP by Segment — {_la_v} vs {_lb_v}", level=2)
            img = _fig_to_png(fig_bar, width=900, height=340, scale=2)
            if img:
                doc.add_picture(BytesIO(img), width=Inches(6.2))
            else:
                doc.add_paragraph("(chart rendering timed out)")
            _fig_caption(doc,
                f"Figure: Pressure drop (kPa) per segment for {_la_v} and {_lb_v}. "
                f"Tallest bars are the dominant loss elements.")

        for c, lbl, is_hdr in zip(cases, case_labels, _is_hdr):
            _fs = c.get("fig_sch")
            _fp = c.get("fig_prof")
            if _fs is not None or _fp is not None:
                doc.add_heading(f"{lbl} — {'Header Layout' if is_hdr else 'Pipeline'}", level=2)
            if _fs is not None:
                img = _fig_to_png(_fs, width=900, height=340 if is_hdr else 440, scale=2)
                if img:
                    doc.add_picture(BytesIO(img), width=Inches(6.2))
                else:
                    doc.add_paragraph("(chart rendering timed out)")
                if is_hdr:
                    _fig_caption(doc,
                        f"Figure: Header piping layout for {lbl}. "
                        f"Blue = left arm, orange = right arm. Triangular markers = tap risers "
                        f"with distance from T. Flow arrows point toward T-junction. "
                        f"Thicker dark pipe = T-segment to separator. "
                        f"Governing arm (⚠) sets the required tap inlet pressure.")
                else:
                    _fig_caption(doc,
                        f"Figure: Pipeline schematic for {lbl}, colour-coded by flow regime. "
                        f"V_m/V_e > 1.0 flags erosion risk (API RP 14E, C = 100).")
            if _fp is not None:
                img = _fig_to_png(_fp, width=900, height=320, scale=2)
                if img:
                    doc.add_picture(BytesIO(img), width=Inches(6.2))
                else:
                    doc.add_paragraph("(chart rendering timed out)")
                if is_hdr:
                    _fig_caption(doc,
                        f"Figure: Header pressure profile for {lbl}. Left arm (blue) and "
                        f"right arm (orange) pressure vs. distance from T-junction. "
                        f"X-axis reversed so flow runs left to right toward T at x = 0.")
                else:
                    _fig_caption(doc,
                        f"Figure: Pressure profile for {lbl}. Pressure (bara) vs. cumulative "
                        f"distance. Coloured bands show predicted flow regime per segment.")

    # ════════════════════════════════════════════════════════════════════════
    # 8. SENSITIVITY ANALYSIS  (page break, if provided)
    # ════════════════════════════════════════════════════════════════════════
    if sensitivity_data is not None:
        _sa = sensitivity_data.get("sa", [])
        _sb = sensitivity_data.get("sb", [])
        _fig_s = sensitivity_data.get("fig")

        doc.add_page_break()
        _h1("Method Sensitivity Analysis")

        _intro = doc.add_paragraph(
            "All 12 method combinations (6 ΔP correlations × 2 void-fraction models) "
            "were evaluated to quantify uncertainty due to correlation choice. "
            "Combinations that failed to converge are excluded."
        )
        if _intro.runs:
            _intro.runs[0].font.size = Pt(9)
        doc.add_paragraph()

        _CORR_S = {"Beggs-Brill": "BB", "Friedel": "Friedel",
                   "Lockhart_Martinelli": "L-M", "Muller_Steinhagen_Heck": "MSH",
                   "Chisholm": "Chisholm", "Kim_Mudawar": "Kim-M"}
        _VOID_S = {"Homogeneous": "Homo", "Rouhani-1 (slip)": "Rouhani-1"}

        _lbl_sa = case_labels[0] if len(case_labels) > 0 else "Case A"
        _lbl_sb = case_labels[1] if len(case_labels) > 1 else "Case B"

        _va = [r["total_dp_kpa"] for r in _sa if r["ok"]]
        _vb = [r["total_dp_kpa"] for r in _sb if r["ok"]]
        if _va and _vb:
            _a_sel  = cases[0]["total_dp_kpa"]
            _b_sel  = cases[1]["total_dp_kpa"]
            _overlap = min(_va) <= max(_vb) and min(_vb) <= max(_va)
            _kv3_table(doc, [
                (f"{_lbl_sa} — min ΔP (kPa)",    f"{min(_va):.3f}",  "—"),
                (f"{_lbl_sa} — selected (kPa)",  f"{_a_sel:.3f}",    "—"),
                (f"{_lbl_sa} — max ΔP (kPa)",    f"{max(_va):.3f}",  "—"),
                (f"{_lbl_sa} — spread (kPa)",     f"{max(_va)-min(_va):.3f}", "—"),
                (f"{_lbl_sb} — min ΔP (kPa)",    "—", f"{min(_vb):.3f}"),
                (f"{_lbl_sb} — selected (kPa)",  "—", f"{_b_sel:.3f}"),
                (f"{_lbl_sb} — max ΔP (kPa)",    "—", f"{max(_vb):.3f}"),
                (f"{_lbl_sb} — spread (kPa)",     "—", f"{max(_vb)-min(_vb):.3f}"),
                ("Ranges overlap?",
                 "Yes — ordering method-dependent" if _overlap else "No — unambiguous", ""),
            ], label_a=_lbl_sa, label_b=_lbl_sb)
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
        _kv3_table(doc, _det, label_a=_lbl_sa, label_b=_lbl_sb)
        doc.add_paragraph()

        if _fig_s is not None:
            _img_s = _fig_to_png(_fig_s, width=900, height=480, scale=2)
            if _img_s:
                doc.add_picture(BytesIO(_img_s), width=Inches(6.2))
            else:
                doc.add_paragraph("(sensitivity chart rendering timed out)")
            _fig_caption(doc,
                f"Figure: Total ΔP for all 12 method combinations. "
                f"Spread quantifies the correlation-choice uncertainty band. "
                f"Non-overlapping clusters give an unambiguous result.")

    # ════════════════════════════════════════════════════════════════════════
    # 9. GENERATOR ΔP DETAIL  (page break, if goal-seek data available)
    # ════════════════════════════════════════════════════════════════════════
    if stack_dp is not None:
        _gsh  = stack_dp.get("gsr_h2") or {}
        _gso  = stack_dp.get("gsr_o2") or {}
        _ph   = stack_dp.get("P_sep_h2")
        _po   = stack_dp.get("P_sep_o2")
        _la_s = stack_dp.get("label_a", case_labels[0] if case_labels else "Case A")
        _lb_s = stack_dp.get("label_b", case_labels[1] if len(case_labels) > 1 else "Case B")
        _p_in_a  = _gsh.get("P_line_in", 0.0)
        _p_in_b  = _gso.get("P_line_in", 0.0)
        _dp_s    = _p_in_a - _p_in_b
        _dp_kpa  = _dp_s * 100.0
        _dp_mbar = _dp_kpa * 10.0

        doc.add_page_break()
        _h1("Generator Differential Pressure — Detail")

        doc.add_heading("Target Conditions", level=2)
        _kv_table(doc, [
            (f"{_la_s} separator target pressure (bara)", f"{_ph:.3f}" if _ph is not None else "—"),
            (f"{_lb_s} separator target pressure (bara)", f"{_po:.3f}" if _po is not None else "—"),
        ])
        doc.add_paragraph()

        for _la_x, _gs_x in [(_la_s, _gsh), (_lb_s, _gso)]:
            doc.add_heading(f"{_la_x}  (Branch → Header → Separator)", level=2)
            _kv_table(doc, [
                (f"{_la_x} branch inlet pressure (bara)",    f"{_gs_x.get('P_line_in', 0):.4f}"),
                (f"{_la_x} branch ΔP (kPa)",                 f"{_gs_x.get('dp_line', 0):.3f}"),
                ("Branch outlet / header tap inlet (bara)",  f"{_gs_x.get('P_line_out', 0):.4f}"),
                ("Header + T-seg ΔP (kPa)",                  f"{_gs_x.get('dp_hdr', 0):.3f}"),
                ("Separator pressure (bara)",                 f"{_gs_x.get('P_sep', 0):.4f}"),
            ])
            doc.add_paragraph()

        doc.add_heading(f"Generator ΔP  ({_la_s} − {_lb_s})", level=2)
        _kv_table(doc, [
            (f"ΔP  {_la_s} − {_lb_s}  (bara)",  f"{_dp_s:.4f}"),
            (f"ΔP  {_la_s} − {_lb_s}  (kPa)",   f"{_dp_kpa:.2f}"),
            (f"ΔP  {_la_s} − {_lb_s}  (mbar)",  f"{_dp_mbar:.1f}"),
        ])
        doc.add_paragraph()

    # ── Engineering note (end of main body) ──────────────────────────────────
    _all_gas = sorted({sp for c in cases for sp in c["gas_flows_kgh"]})
    _all_liq = sorted({c["liquid_type"] for c in cases})
    note = doc.add_paragraph(
        f"Engineering Note: The two-phase correlations used here were developed primarily "
        f"for oil/gas systems. Their application to this service "
        f"({' / '.join(_all_gas)} / {' / '.join(_all_liq)}) carries an estimated "
        f"uncertainty of ±20–30 %. Use the sensitivity analysis (if present) to bracket "
        f"the ΔP range. Treat as a first-pass estimate; validate against commissioning "
        f"data before use in safety-critical design."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ════════════════════════════════════════════════════════════════════════
    # APPENDIX
    # ════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    _app_h = doc.add_heading("Appendix", level=0)
    _app_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── A. Phase Thermodynamics ───────────────────────────────────────────────
    doc.add_heading("A.  Phase Thermodynamics  (inlet conditions)", level=1)
    _ps = [c["props"] for c in cases]

    def _pfmt(p, key, scale=1.0, fmt=".4f"):
        v = p.get(key)
        return f"{v * scale:{fmt}}" if v is not None else "—"

    _thm = [
        ("Gas density ρ_g (kg/m³)",)     + tuple(_pfmt(p, "rho_g")                  for p in _ps),
        ("Gas mixture MW (g/mol)",)       + tuple(_pfmt(p, "MW_mix_gmol", fmt=".3f") for p in _ps),
        ("Liquid density ρ_l (kg/m³)",)   + tuple(_pfmt(p, "rho_l", fmt=".2f")       for p in _ps),
        ("Liquid viscosity μ_l (mPa·s)",) + tuple(_pfmt(p, "mu_l", 1e3)             for p in _ps),
        ("Gas viscosity μ_g (µPa·s)",)   + tuple(_pfmt(p, "mu_g", 1e6, ".2f")       for p in _ps),
        ("Surface tension σ (mN/m)",)    + tuple(_pfmt(p, "sigma", 1e3, ".3f")       for p in _ps),
        ("Mass quality x (%)",)          + tuple(_pfmt(p, "x_gas", 100)             for p in _ps),
        ("Void fraction α (%)",)         + tuple(_pfmt(p, "alpha", 100, ".2f")       for p in _ps),
    ]
    if any(p.get("P_sat_H2O_pa", 0) > 0 for p in _ps):
        _thm += [
            ("H₂O sat. pressure (bara)",) + tuple(
                f"{p.get('P_sat_H2O_pa', 0) / 1e5:.4f}"
                if p.get("P_sat_H2O_pa", 0) > 0 else "—" for p in _ps),
            ("H₂O vapour flow (kg/h)",) + tuple(
                f"{p.get('m_vapor_h2o_kgh', 0):.4f}"
                if p.get("P_sat_H2O_pa", 0) > 0 else "—" for p in _ps),
        ]
    _nt(_thm)
    doc.add_paragraph()

    # ── B. Branch Segment Analysis ────────────────────────────────────────────
    _branch_pairs = [(c, lbl) for c, lbl, h in zip(cases, case_labels, _is_hdr) if not h]
    if _branch_pairs:
        doc.add_heading("B.  Segment Analysis — Branch Lines", level=1)
        _SC = ["Seg", "Pipe", "ID (mm)", "Type", "L (m)", "L_eq (m)", "Fittings",
               "Regime", "V_m (m/s)", "V_m/V_e", "ΔP (kPa)", "P_out (bara)"]
        _SW = [0.25, 0.50, 0.42, 0.80, 0.38, 0.42, 0.70, 0.85, 0.48, 0.44, 0.50, 0.53]

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

        for _bc, _blbl in _branch_pairs:
            doc.add_heading(_blbl, level=2)
            _seg_tbl(_bc["grid_records"])
            doc.add_paragraph()

    # ── C. Header Configuration ───────────────────────────────────────────────
    _hdr_pairs = [(c, lbl) for c, lbl, h in zip(cases, case_labels, _is_hdr) if h]
    if _hdr_pairs:
        doc.add_heading("C.  Header Configuration", level=1)

        # Column headers for arm detail table
        _HC = ["Arm", "Seg", "From T (m)", "To T (m)", "L (m)", "Pipe", "ID (mm)",
               "Regime", "Q_gas (kg/h)", "Q_liq (m³/h)", "ΔP (kPa)",
               "P_in (bara)", "P_out (bara)"]
        _HW = [0.30, 0.28, 0.50, 0.50, 0.36, 0.48, 0.42, 0.72,
               0.58, 0.56, 0.48, 0.50, 0.50]

        def _hdr_detail_tbl(records):
            if not records:
                return
            tbl = doc.add_table(rows=len(records) + 1, cols=len(_HC))
            tbl.style = "Table Grid"
            _style_header(tbl.rows[0], font_size=8)
            for j, col in enumerate(_HC):
                tbl.rows[0].cells[j].text = col
            _col_map = {"Q_gas (kg/h)": "Q_gas_kgh", "Q_liq (m³/h)": "Q_liq_m3h"}
            for i, rec in enumerate(records, start=1):
                row = tbl.rows[i]
                if i % 2 == 0:
                    for cell in row.cells:
                        _shd(cell, _ALT_BG)
                seg_id = str(rec.get("Seg", ""))
                row.cells[0].text = (
                    "Left" if seg_id.startswith("L")
                    else "Right" if seg_id.startswith("R") else "T-seg")
                for j, col in enumerate(_HC[1:], start=1):
                    row.cells[j].text = str(rec.get(_col_map.get(col, col), ""))
                for cell in row.cells:
                    _cell_font(cell, size_pt=8)
            _set_col_widths(tbl, _HW)

        for _hc, _hlbl in _hdr_pairs:
            doc.add_heading(_hlbl, level=2)
            _hp  = _hc.get("header_pipe", {})
            _ts  = _hc.get("t_seg", {})
            _ltp = sorted(_hc.get("left_taps",  []), reverse=True)
            _rtp = sorted(_hc.get("right_taps", []))
            _cfg_rows = [
                ("Header pipe DN / PN",
                 f"{_hp.get('dn', '—')} / {_hp.get('pn', '—')}"),
                ("Header pipe material",
                 _hp.get("material", "—")),
            ]
            if _hp.get("lined"):
                _cfg_rows += [
                    ("Liner material",       _hp.get("liner_material", "—")),
                    ("Liner thickness (mm)", f"{_hp.get('liner_thickness_mm', 0):.1f}"),
                ]
            _cfg_rows += [
                ("Taps — left arm",
                 str(_hc.get("n_left", len(_ltp)))),
                ("Taps — right arm",
                 str(_hc.get("n_right", len(_rtp)))),
                ("Left tap distances from T (m)",
                 "  |  ".join(f"{p:.2f}" for p in _ltp) or "—"),
                ("Right tap distances from T (m)",
                 "  |  ".join(f"{p:.2f}" for p in _rtp) or "—"),
                ("Governing arm",
                 _hc.get("worst_arm", "—")),
                ("T-segment DN / PN",
                 f"{_ts.get('dn', '—')} / {_ts.get('pn', '—')}"),
                ("T-segment material",
                 _ts.get("material", "—")),
                ("T-segment length (m)",
                 f"{_ts.get('length', 0):.2f}"),
            ]
            _kv_table(doc, _cfg_rows, col_widths=(3.0, 3.47))
            doc.add_paragraph()
            if _hc.get("grid_records"):
                _p = doc.add_paragraph("Arm segment detail:")
                if _p.runs:
                    _p.runs[0].font.size = Pt(9)
                _hdr_detail_tbl(_hc["grid_records"])
                doc.add_paragraph()

    # ── D. DN Study ──────────────────────────────────────────────────────────
    if dn_study_data:
        _dns = dn_study_data
        _dn_p  = _dns["dn_primary"]
        _dn_a  = _dns["dn_alt"]
        _la_dn = _dns.get("label_a", case_labels[0] if case_labels else "A")
        _lb_dn = _dns.get("label_b", case_labels[1] if len(case_labels) > 1 else "B")
        _gp_h2 = _dns["gsr_h2_primary"]
        _gp_o2 = _dns["gsr_o2_primary"]
        _ga_h2 = _dns["gsr_h2_alt"]
        _ga_o2 = _dns["gsr_o2_alt"]
        _dp_p  = _dns["dp_gen_primary_mbar"]
        _dp_a  = _dns["dp_gen_alt_mbar"]
        _vd    = _dns["vel_data"]

        doc.add_heading("D.  DN Study — Branch Line Size Comparison", level=1)
        _kv_table(doc, [
            ("Primary branch DN",         _dn_p),
            ("Alternative branch DN",     _dn_a),
            ("Header size",               "Unchanged for both cases"),
            ("H₂ separator target (bara)", f"{_dns.get('p_sep_h2', 0):.3f}"),
            ("O₂ separator target (bara)", f"{_dns.get('p_sep_o2', 0):.3f}"),
        ])
        doc.add_paragraph()

        doc.add_heading("Generator ΔP", level=2)
        _delta_mbar = _dp_a - _dp_p
        _winner = _dn_p if abs(_dp_p) <= abs(_dp_a) else _dn_a
        _kv3_table(doc, [
            (f"{_la_dn} branch inlet pressure (bara)",
                f"{_gp_h2['P_line_in']:.4f}", f"{_ga_h2['P_line_in']:.4f}"),
            (f"{_lb_dn} branch inlet pressure (bara)",
                f"{_gp_o2['P_line_in']:.4f}", f"{_ga_o2['P_line_in']:.4f}"),
            ("Generator ΔP (mbar)", f"{_dp_p:.1f}", f"{_dp_a:.1f}"),
            ("Change vs primary (mbar)", "—", f"{_delta_mbar:+.1f}"),
            ("Lower |Generator ΔP|", _winner, _winner),
        ], label_a=_dn_p, label_b=_dn_a)
        doc.add_paragraph()

        doc.add_heading("Pressure Drop by Case", level=2)
        _cases_dp = [
            (f"{_la_dn} branch", _gp_h2["dp_line"], _ga_h2["dp_line"]),
            (f"{_lb_dn} branch", _gp_o2["dp_line"], _ga_o2["dp_line"]),
            (f"{_la_dn} header", _gp_h2["dp_hdr"],  _ga_h2["dp_hdr"]),
            (f"{_lb_dn} header", _gp_o2["dp_hdr"],  _ga_o2["dp_hdr"]),
        ]
        _dp_rows = []
        for _lbl, _dp_pv, _dp_av in _cases_dp:
            _pct = (_dp_av - _dp_pv) / _dp_pv * 100 if abs(_dp_pv) > 1e-9 else 0.0
            _dp_rows.append((_lbl, f"{_dp_pv:.3f}", f"{_dp_av:.3f}  ({_pct:+.1f} %)"))
        _kv3_table(doc, _dp_rows,
                   label_a=f"{_dn_p} ΔP (kPa)", label_b=f"{_dn_a} ΔP (kPa)")
        doc.add_paragraph()

        doc.add_heading("Inlet Velocity — First Segment (Estimated)", level=2)
        _ratio_a = _vd["vm_a_alt"] / _vd["ve_a"] if _vd["ve_a"] > 0 else 0.0
        _ratio_b = _vd["vm_b_alt"] / _vd["ve_b"] if _vd["ve_b"] > 0 else 0.0
        _kv3_table(doc, [
            ("Effective ID (mm)",
                f"{_vd['D_p_mm']:.1f}", f"{_vd['D_a_mm']:.1f}"),
            ("Velocity scale factor (ID ratio²)", "1.00×", f"{_vd['vel_scale']:.2f}×"),
            (f"{_la_dn} V_m inlet (m/s)",
                f"{_vd['vm_a_primary']:.3f}", f"{_vd['vm_a_alt']:.3f}"),
            (f"{_la_dn} V_m / V_e",
                f"{_vd['vm_a_primary']/_vd['ve_a']:.2f}" if _vd["ve_a"] > 0 else "—",
                f"{_ratio_a:.2f}"),
            (f"{_lb_dn} V_m inlet (m/s)",
                f"{_vd['vm_b_primary']:.3f}", f"{_vd['vm_b_alt']:.3f}"),
            (f"{_lb_dn} V_m / V_e",
                f"{_vd['vm_b_primary']/_vd['ve_b']:.2f}" if _vd["ve_b"] > 0 else "—",
                f"{_ratio_b:.2f}"),
        ], label_a=_dn_p, label_b=_dn_a)
        _fig_caption(doc,
            "Velocity estimated via ID-ratio scaling. "
            "Erosion velocity V_e from API RP 14E (C = 100), primary case values.")
        doc.add_paragraph()

        doc.add_heading("Recommendation", level=2)
        _vel_ok   = _ratio_a <= 1.0 and _ratio_b <= 1.0
        _alt_wins = abs(_dp_a) < abs(_dp_p)
        if _alt_wins and _vel_ok:
            _rec = (f"{_dn_a} gives lower Generator |ΔP| ({_dp_a:.1f} vs {_dp_p:.1f} mbar) "
                    f"with V_m/V_e within the erosion limit. {_dn_a} is preferred.")
        elif _alt_wins:
            _rec = (f"{_dn_a} gives lower Generator |ΔP| ({_dp_a:.1f} vs {_dp_p:.1f} mbar) "
                    f"but estimated V_m/V_e may exceed 1.0 — verify erosion before selecting {_dn_a}.")
        else:
            _rec = (f"{_dn_p} (primary) gives lower Generator |ΔP| "
                    f"({_dp_p:.1f} vs {_dp_a:.1f} mbar). {_dn_a} appears oversized.")
        _p_rec = doc.add_paragraph(_rec)
        if _p_rec.runs:
            _p_rec.runs[0].font.size = Pt(9)
        doc.add_paragraph()

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def generate_dn_study_report(
    dn_primary, dn_alt,
    label_a, label_b,
    gsr_h2_primary, gsr_o2_primary,
    gsr_h2_alt, gsr_o2_alt,
    dp_gen_primary_mbar, dp_gen_alt_mbar,
    vel_data,
    p_sep_h2, p_sep_o2,
):
    """Word report comparing two branch DN sizes across the full system."""
    doc = Document()

    sec = doc.sections[0]
    sec.page_width    = Inches(8.27)
    sec.page_height   = Inches(11.69)
    sec.left_margin   = Inches(0.9)
    sec.right_margin  = Inches(0.9)
    sec.top_margin    = Inches(0.9)
    sec.bottom_margin = Inches(0.9)

    h = doc.add_heading(f"Pipe Size Study — {dn_primary} vs {dn_alt}", level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        f"Branch Line DN Comparison  ·  {datetime.now().strftime('%d %B %Y  %H:%M')}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        sub.runs[0].font.size = Pt(10)
    doc.add_paragraph()

    # ── 1. Study basis ────────────────────────────────────────────────────────
    doc.add_heading("Study Basis", level=1)
    _kv_table(doc, [
        ("Primary branch DN",           dn_primary),
        ("Alternative branch DN",        dn_alt),
        ("Header size",                  "Unchanged for both cases"),
        ("H₂ separator target (bara)",   f"{p_sep_h2:.3f}"),
        ("O₂ separator target (bara)",   f"{p_sep_o2:.3f}"),
        ("Correlation / voidage",        "As set in primary case"),
    ])
    doc.add_paragraph()

    # ── 2. Generator ΔP comparison ───────────────────────────────────────────
    doc.add_heading("Generator Differential Pressure", level=1)
    _delta_mbar = dp_gen_alt_mbar - dp_gen_primary_mbar
    _winner = dn_primary if abs(dp_gen_primary_mbar) <= abs(dp_gen_alt_mbar) else dn_alt
    _kv3_table(doc, [
        (f"{label_a} branch inlet pressure (bara)",
            f"{gsr_h2_primary['P_line_in']:.4f}",
            f"{gsr_h2_alt['P_line_in']:.4f}"),
        (f"{label_b} branch inlet pressure (bara)",
            f"{gsr_o2_primary['P_line_in']:.4f}",
            f"{gsr_o2_alt['P_line_in']:.4f}"),
        ("Generator ΔP (mbar)",
            f"{dp_gen_primary_mbar:.1f}",
            f"{dp_gen_alt_mbar:.1f}"),
        ("Change vs primary (mbar)", "—", f"{_delta_mbar:+.1f}"),
        ("Lower |Generator ΔP|", _winner, _winner),
    ], label_a=dn_primary, label_b=dn_alt)
    doc.add_paragraph()

    # ── 3. Pressure drop by case ──────────────────────────────────────────────
    doc.add_heading("Pressure Drop Summary by Case", level=1)
    _cases_dp = [
        (f"{label_a} branch",  gsr_h2_primary["dp_line"], gsr_h2_alt["dp_line"]),
        (f"{label_b} branch",  gsr_o2_primary["dp_line"], gsr_o2_alt["dp_line"]),
        (f"{label_a} header",  gsr_h2_primary["dp_hdr"],  gsr_h2_alt["dp_hdr"]),
        (f"{label_b} header",  gsr_o2_primary["dp_hdr"],  gsr_o2_alt["dp_hdr"]),
    ]
    _rows_dp = []
    for _lbl, _dp_p, _dp_a in _cases_dp:
        _pct = (_dp_a - _dp_p) / _dp_p * 100 if abs(_dp_p) > 1e-9 else 0.0
        _rows_dp.append((
            _lbl,
            f"{_dp_p:.3f}",
            f"{_dp_a:.3f}  ({_pct:+.1f} %)",
        ))
    _kv3_table(doc, _rows_dp, label_a=f"{dn_primary} ΔP (kPa)", label_b=f"{dn_alt} ΔP (kPa)")
    doc.add_paragraph()

    # ── 4. Velocity estimate ──────────────────────────────────────────────────
    doc.add_heading("Inlet Velocity — First Segment (Estimated)", level=1)
    _vd = vel_data
    _ratio_a = _vd["vm_a_alt"] / _vd["ve_a"] if _vd["ve_a"] > 0 else 0.0
    _ratio_b = _vd["vm_b_alt"] / _vd["ve_b"] if _vd["ve_b"] > 0 else 0.0
    _kv3_table(doc, [
        ("Effective ID (mm)",
            f"{_vd['D_p_mm']:.1f}", f"{_vd['D_a_mm']:.1f}"),
        ("Velocity scale factor (ID ratio²)", "1.00×", f"{_vd['vel_scale']:.2f}×"),
        (f"{label_a} V_m inlet (m/s)",
            f"{_vd['vm_a_primary']:.3f}", f"{_vd['vm_a_alt']:.3f}"),
        (f"{label_a} V_m / V_e",
            f"{_vd['vm_a_primary'] / _vd['ve_a']:.2f}" if _vd['ve_a'] > 0 else "—",
            f"{_ratio_a:.2f}"),
        (f"{label_b} V_m inlet (m/s)",
            f"{_vd['vm_b_primary']:.3f}", f"{_vd['vm_b_alt']:.3f}"),
        (f"{label_b} V_m / V_e",
            f"{_vd['vm_b_primary'] / _vd['ve_b']:.2f}" if _vd['ve_b'] > 0 else "—",
            f"{_ratio_b:.2f}"),
    ], label_a=dn_primary, label_b=dn_alt)
    doc.add_paragraph()
    _fig_caption(doc,
        "Velocity estimated via ID-ratio scaling: V_m(alt) = V_m(primary) × (ID_primary/ID_alt)². "
        "Erosion velocity V_e from primary case (API RP 14E, C = 100). Not recalculated for alt DN.")

    # ── 5. Recommendation ─────────────────────────────────────────────────────
    doc.add_heading("Recommendation", level=1)
    _vel_ok = _ratio_a <= 1.0 and _ratio_b <= 1.0
    _alt_better = abs(dp_gen_alt_mbar) < abs(dp_gen_primary_mbar)
    if _alt_better and _vel_ok:
        _rec = (
            f"{dn_alt} gives a lower Generator |ΔP| ({dp_gen_alt_mbar:.1f} mbar vs "
            f"{dp_gen_primary_mbar:.1f} mbar) with V_m/V_e within the erosion limit "
            f"for both branch lines. {dn_alt} is preferred for this duty."
        )
    elif _alt_better and not _vel_ok:
        _rec = (
            f"{dn_alt} gives a lower Generator |ΔP| ({dp_gen_alt_mbar:.1f} mbar vs "
            f"{dp_gen_primary_mbar:.1f} mbar) but the estimated inlet velocity ratio "
            f"V_m/V_e exceeds 1.0 for one or both branch lines. Verify erosion "
            f"against API RP 14E before selecting {dn_alt}."
        )
    else:
        _rec = (
            f"{dn_primary} (primary) gives a lower Generator |ΔP| "
            f"({dp_gen_primary_mbar:.1f} mbar vs {dp_gen_alt_mbar:.1f} mbar). "
            f"{dn_alt} appears oversized for this duty; the additional pipe cross-section "
            f"does not meaningfully reduce the system differential pressure."
        )
    _p_rec = doc.add_paragraph(_rec)
    if _p_rec.runs:
        _p_rec.runs[0].font.size = Pt(10)
    doc.add_paragraph()

    note = doc.add_paragraph(
        "Engineering Note: Velocity is estimated via ID-ratio scaling and is not exact. "
        "For a rigorous erosion check on the alternative DN, run it as the primary case. "
        "Correlations carry ±20–30 % uncertainty for H₂/O₂ over KOH systems."
    )
    if note.runs:
        note.runs[0].font.size      = Pt(8)
        note.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
