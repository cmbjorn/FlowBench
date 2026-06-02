"""
Piping standards lookup tables.

Sources:
    ASME B36.10M / B36.19M — pipe schedules and bore dimensions
    Crane TP-410            — material roughness, fitting Le/D factors
    ASME B16.5-2017         — pressure-temperature class ratings
"""

# ── Pipe bore database (ANSI B36.10 / B36.19) ───────────────────────────────
# Inner diameters in metres.
#   PN20 / PN25  ≈  Schedule 40  (material-independent at the same schedule)
#   PN40         ≈  Schedule 80
PIPE_DATABASE: dict[str, dict[str, float]] = {
    "DN20":  {"PN20": 0.0209, "PN25": 0.0209, "PN40": 0.0189},
    "DN25":  {"PN20": 0.0266, "PN25": 0.0266, "PN40": 0.0243},
    "DN40":  {"PN20": 0.0409, "PN25": 0.0409, "PN40": 0.0381},
    "DN50":  {"PN20": 0.0525, "PN25": 0.0525, "PN40": 0.0493},
    "DN65":  {"PN20": 0.0627, "PN25": 0.0627, "PN40": 0.0590},
    "DN80":  {"PN20": 0.0779, "PN25": 0.0779, "PN40": 0.0737},
    "DN100": {"PN20": 0.1023, "PN25": 0.1023, "PN40": 0.0972},
    "DN150": {"PN20": 0.1541, "PN25": 0.1541, "PN40": 0.1463},
    "DN200": {"PN20": 0.2027, "PN25": 0.2027, "PN40": 0.1937},
    "DN250": {"PN20": 0.2545, "PN25": 0.2545, "PN40": 0.2429},
}

# ── Swagelok metric SS tubing (316/316L seamless, OD × wall → ID) ────────────
# Inner diameters in metres.  Wall thicknesses per Swagelok catalog MS-01-140.
# Sizes T18 and above are typical connector / inter-harp tubing.
TUBING_DATABASE: dict[str, dict[str, float]] = {
    "T6":  {"1.0 mm wall": 0.004},
    "T8":  {"1.0 mm wall": 0.006},
    "T10": {"1.0 mm wall": 0.008},
    "T12": {"1.0 mm wall": 0.010},
    "T14": {"1.5 mm wall": 0.011, "2.0 mm wall": 0.010},
    "T16": {"1.5 mm wall": 0.013, "2.0 mm wall": 0.012},
    "T18": {"1.5 mm wall": 0.015, "2.0 mm wall": 0.014},
    "T20": {"1.5 mm wall": 0.017, "2.0 mm wall": 0.016},
    "T25": {"1.5 mm wall": 0.022, "2.0 mm wall": 0.021},
    "T28": {"1.5 mm wall": 0.025, "2.0 mm wall": 0.024},
    "T32": {"2.0 mm wall": 0.028, "3.0 mm wall": 0.026},
    "T38": {"2.0 mm wall": 0.034, "3.0 mm wall": 0.032},
}

# Absolute roughness (m) for seamless drawn SS instrument tubing.
# Swagelok SS-T catalog: Ra ≤ 0.8 µm → ε ≈ 1.5 µm (approx. 10× smoother than pipe).
TUBING_ROUGHNESS: float = 1.5e-6

# ── Absolute roughness (m) — Crane TP-410 / ASHRAE ──────────────────────────
MATERIAL_ROUGHNESS: dict[str, float] = {
    "SS316L":            1.5e-5,
    "Duplex SS 2205":    1.5e-5,
    "Carbon Steel":      4.6e-5,
    "Hastelloy C-276":   1.5e-5,
    "Titanium Gr. 2":    1.5e-5,
}

# Absolute roughness (m) for fluoropolymer pipe liners.
LINER_ROUGHNESS: dict[str, float] = {
    "PTFE":  5.0e-8,
    "FEP":   5.0e-8,
    "PFA":   5.0e-8,
    "PVDF":  1.5e-7,
}

# ── Fitting equivalent-length factors (Le/D) — Crane TP-410 ─────────────────
FITTING_Le_over_D: dict[str, int] = {
    "90° Standard Elbow":                    30,
    "90° Long Radius Elbow (1.5D)":          16,
    "45° Elbow":                             16,
    "180° Return Bend":                      50,
    "Tee — Branch Flow":                     60,
    "Tee — Run Through":                     20,
    "Gate Valve — Fully Open":                8,
    "Globe Valve — Fully Open":             340,
    "Ball Valve — Fully Open":                3,
    "Butterfly Valve":                       45,
    "Swing Check Valve":                    100,
    "Lift Check Valve":                     600,
    "Concentric Reducer — Gradual (15°)":     5,
    "Concentric Reducer — Sudden":           26,
    "Eccentric Reducer — Gradual (15°)":      5,
    "Expansion — Gradual (15°)":             10,
    "Expansion — Sudden":                    30,
}

# ── NPS (inches) → DN (mm) — ASME B36.10M standard designations ─────────────
_NPS_TO_DN: dict[float, int] = {
    0.5: 15, 0.75: 20, 1.0: 25, 1.25: 32, 1.5: 40,
    2.0: 50, 2.5: 65, 3.0: 80, 4.0: 100, 6.0: 150,
    8.0: 200, 10.0: 250, 12.0: 300,
}


def nps_to_dn(nps_inch: float) -> int:
    """Convert NPS (inches) to DN (mm). Returns nearest known DN."""
    return _NPS_TO_DN.get(nps_inch, round(nps_inch * 25.0))


# ── ASME B16.5-2017 pressure-temperature class ratings (barg at ≤ 38 °C) ────
# Tables 2-1.1, 2-2.2, 2-2.3, 2-1.4, 2-1.9
_B165_RATINGS: dict[str, dict[int, float]] = {
    "1.1 — Carbon steel (A105 / A106)": {
        150: 19.6, 300: 51.1, 600: 102.1,
        900: 153.2, 1500: 255.3, 2500: 425.5,
    },
    "2.3 — Stainless 316 / 316L": {
        150: 15.1, 300: 39.3, 600: 78.6,
        900: 117.9, 1500: 196.5, 2500: 327.6,
    },
    "2.2 — Stainless 304 / 304L": {
        150: 15.1, 300: 39.3, 600: 78.6,
        900: 117.9, 1500: 196.5, 2500: 327.6,
    },
    "1.4 — Low-temp carbon steel (A350 LF2)": {
        150: 19.6, 300: 51.1, 600: 102.1,
        900: 153.2, 1500: 255.3, 2500: 425.5,
    },
    "1.9 — Duplex SS (A182 F51)": {
        150: 20.0, 300: 51.1, 600: 102.1,
        900: 153.2, 1500: 255.3, 2500: 425.5,
    },
}

MATERIAL_GROUPS: list[str] = list(_B165_RATINGS.keys())


def ansi_class_lookup(P_design_barg: float, material_group: str) -> dict:
    """
    Return the lowest ANSI B16.5 pressure class adequate for P_design_barg.

    Returns dict with:
        class_number (int), class_label (str), rated_barg (float), adequate (bool),
        all_classes (list of dicts for table display)
    """
    ratings = _B165_RATINGS.get(material_group, list(_B165_RATINGS.values())[0])
    selected_class = None
    selected_rating = None

    all_classes = []
    for cls, rated in sorted(ratings.items()):
        adequate = rated >= P_design_barg
        all_classes.append({
            "Class":        f"ANSI {cls}",
            "Rated (barg)": rated,
            "Adequate":     "✓" if adequate else "✗",
        })
        if adequate and selected_class is None:
            selected_class  = cls
            selected_rating = rated

    if selected_class is None:
        selected_class  = 2500
        selected_rating = ratings[2500]

    return {
        "class_number": selected_class,
        "class_label":  f"ANSI {selected_class}",
        "rated_barg":   selected_rating,
        "adequate":     selected_rating >= P_design_barg,
        "all_classes":  all_classes,
    }


# ── Minimum ASME B16.5 flange class for PSV inlet (Group 1.1 CS at 38 °C) ──
# Class 300 is the industry minimum per API 526 practice.
_CLASS_LIMITS_BARA: list[tuple[int, float]] = [
    (300,   51.1),
    (600,  102.1),
    (900,  153.2),
    (1500, 255.2),
    (2500, 425.6),
]


def min_flange_class(P_bara: float) -> int:
    """Minimum ASME B16.5 flange class for the given inlet pressure (bara)."""
    for cls, limit in _CLASS_LIMITS_BARA:
        if P_bara <= limit:
            return cls
    return 2500


def sum_le_fit(seg: dict, D_eff: float) -> float:
    """Total equivalent pipe length (m) contributed by fittings in one segment.

    Accepts both the current format (``fittings_list`` list-of-dicts) and the
    legacy format (``fittings`` string + ``fitting_count`` int).
    """
    fl = seg.get("fittings_list")
    if fl is not None:
        total = 0.0
        for fit in fl:
            t = fit.get("type", "")
            q = fit.get("qty", 0)
            if t in FITTING_Le_over_D and q > 0:
                total += FITTING_Le_over_D[t] * D_eff * q
        return total
    f = seg.get("fittings", "None")
    c = seg.get("fitting_count", 0)
    if f in FITTING_Le_over_D and c > 0:
        return FITTING_Le_over_D[f] * D_eff * c
    return 0.0
