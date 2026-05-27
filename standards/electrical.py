"""
Electrical standards lookup tables.

Sources:
    IEC 60034 — rotating electrical machines (motor frame catalogue)
"""

# Standard IEC motor nameplate power ratings (kW) in ascending order.
_IEC_MOTOR_KW: list[float] = [
    0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
    7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0, 55.0, 75.0, 90.0,
    110.0, 132.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
]


def next_motor_frame_kw(P_required_kw: float) -> float:
    """Return the next standard IEC motor nameplate rating above P_required_kw."""
    for p in _IEC_MOTOR_KW:
        if p >= P_required_kw:
            return p
    return P_required_kw  # above catalogue — return as-is
