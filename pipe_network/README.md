# `pipe_network` — Two-Phase Pipe-Network Solver

A generic Hardy-Cross solver for gas/liquid two-phase flow in looped pipe
networks, with built-in support for the *harp manifold* geometry used in
electrolysers, plate-heat exchangers, and similar process equipment.

---

## Contents

| File | Purpose |
|------|---------|
| `graph.py` | `Node`, `Edge`, `Network` dataclasses + graph algorithms |
| `topology.py` | `build_harp()`, `build_series_harp()` factory functions |
| `solver.py` | `solve_network()` — Hardy-Cross iteration |
| `diagnostics.py` | `check_starvation()`, `StarvationReport` |
| `__init__.py` | Public API re-exports |
| `_smoke_test.py` | Verification tests |

---

## The harp manifold problem

A harp consists of two parallel headers (A and B) connected by N identical
channels:

```
  Q_in
   │
   ▼
  A_0 ──hA_0──▶ A_1 ──hA_1──▶ A_2 ──hA_2──▶ A_3
   │             │              │              │
  ch_0          ch_1           ch_2           ch_3       N = 4 channels
   │             │              │              │
   ▼             ▼              ▼              ▼
  B_0 ──hB_0──▶ B_1 ──hB_1──▶ B_2 ──hB_2──▶ B_3
                                               │
                                               ▼
                                             Q_out
```

**Key features of this topology:**

- **Header A** (supply): flow enters at A_0 and is progressively tapped off
  into channels.  The far end (A_N) is *capped* — zero flow exits there.
- **Channels**: identical parallel tubes carrying gas/liquid mixture from A to B.
- **Header B** (collection): flow from channels accumulates.  The near end
  (B_0) is capped; all collected flow exits at B_N.
- **U-flow**: inlet and outlet are on opposite ends of the harp (same flow
  direction in both headers).

Two harps in series are joined by a single connector pipe:

```
 [Harp 1 — see above] ──connector──▶ [Harp 2 — identical]
```

The engineering question is: **does every channel receive adequate liquid
flow?**  In two-phase systems, hydraulic imbalance concentrates flow in some
channels while others are starved of liquid, leading to poor heat or mass
transfer, local overheating, and corrosion.

---

## Why Hardy-Cross applies here

A common misconception is that Hardy-Cross cannot solve harp manifolds because
harps are "open branched circuits".  That is only true for harps with *both
ends of both headers open* (uncapped).  The engineering harp has **one end of
each header capped**, which creates closed cells:

```
  A_i ──hA_i──▶ A_{i+1}
   │                │
  ch_i            ch_{i+1}      ← rectangular cell / closed loop
   │                │
  B_i ◀──hB_i── B_{i+1}
```

For a harp with N channels and capped headers:

```
  Independent loops  =  Edges − Nodes + 1
                     =  3N  −  2(N+1)  +  1
                     =  N − 1
```

Each of the N−1 rectangular cells is one independent loop.  Hardy-Cross
corrects the flow distribution in each cell until the pressure around every
rectangle sums to zero.

---

## The Hardy-Cross algorithm

### Background

Hardy-Cross (1936) is the classical iterative method for pipe networks.  It
exploits two physical laws:

1. **Continuity** (Kirchhoff's current law analogue): at every junction, the
   net mass flow is zero — what enters must leave.
2. **Energy conservation** (Kirchhoff's voltage law analogue): around any
   closed loop, the net pressure drop is zero.

### How it works

**Step 0 — Initialise**

Assign a flow to every edge that satisfies all node mass balances.  For a harp
with N channels the analytical uniform-distribution initial guess is:

```
  channel i        :  m_total / N
  header-A seg. i  :  m_total × (N − i) / N   (decreasing toward dead end)
  header-B seg. i  :  m_total × (i + 1) / N   (increasing toward outlet)
```

This satisfies every mass balance exactly, so only loop pressure balance needs
to be corrected.

**Step A — Evaluate pressure drops**

For each edge compute ΔP using the existing two-phase correlation engine
(`calculate_segment_pressure_drop`).  The computation scales the total inlet
fluid composition to the fraction carried by each edge:

```
  m_gas_edge  =  |m_edge| × x_node       (x_node = quality at upstream node)
  m_liq_edge  =  |m_edge| × (1 − x_node)
```

**Step B — Loop pressure residuals**

For each fundamental loop (one per co-tree edge, found by BFS spanning tree):

```
  F_ℓ  =  Σ  (orientation × ΔP_e)   for all edges e in loop ℓ
```

`F_ℓ = 0` at convergence.  Orientation is +1 if the edge's from→to direction
matches the loop traversal direction, −1 otherwise.

**Step C — Hardy-Cross flow correction**

For each loop the correction that most reduces `F_ℓ` is:

```
  Δm  =  − relax × F_ℓ / Σ(2|ΔP_e| / |m_e|)
```

The denominator `2|ΔP_e| / |m_e|` is the local hydraulic resistance
linearisation (exact for turbulent Darcy-Weisbach where ΔP ∝ m²; a useful
approximation for two-phase Beggs-Brill).  The factor `relax` (default 0.7)
damps oscillations.

`Δm` is added to every edge in the loop with its orientation sign, preserving
mass balance.

**Step D — Quality propagation**

After updating flows, propagate the gas mass quality through the network using
BFS from the inlet:

```
  x_node  =  Σ(|m_i| × x_upstream_i) / Σ|m_i|   (mass-weighted mixing)
```

For splits (one inlet, multiple outlets) each outlet gets the same quality as
the junction — the **homogeneous assumption**.  More sophisticated T-junction
phase-split models (Azzopardi, Taitel–Barnea) can be substituted here in a
future extension.

**Step E — Pressure update**

Recompute node pressures by traversing from the inlet:

```
  P_downstream  =  P_upstream − ΔP_edge
```

**Step F — Convergence check**

```
  residual  =  max(|F_ℓ|) / max(|ΔP_e|)   <   tol_rel   (default 1 × 10⁻⁴)
```

If the residual grows for three consecutive iterations the relaxation factor is
halved (floor 0.05) to damp oscillations.

---

## Starvation detection

After convergence, `check_starvation()` flags any channel where:

| Criterion | Default threshold |
|-----------|------------------|
| Liquid superficial velocity `V_sl < threshold` | 0.05 m/s |
| Gas mass quality `x > threshold` (essentially dry) | 0.95 |

The **maldistribution index** (MDI) summarises the non-uniformity:

```
  MDI  =  (m_max − m_min) / m_mean   for channel flows
```

| MDI | Interpretation |
|-----|---------------|
| ≈ 0 | Perfectly uniform distribution |
| > 0.5 | Significant non-uniformity |
| > 1.0 | Some channels near starvation or flow reversal |

---

## Quick start

```python
from pipe_network import (
    build_series_harp, solve_network,
    check_starvation, format_report,
)

# Two identical 10-channel harps in series (electrolyser example)
net, topo1, topo2 = build_series_harp(
    N=10,
    header_dn="DN50",   header_pn="PN40",  header_segment_length=0.15,   # m
    channel_dn="DN25",  channel_pn="PN40", channel_length=0.50,
    channel_angle_rad=0.0,                 # horizontal channels
    connector_dn="DN50", connector_pn="PN40", connector_length=0.30,
    P_inlet_pa=30e5,    # 30 bara
    T_C=60.0,           # °C
    x_inlet=0.05,       # 5 wt% gas at inlet
)

# Inlet flow conditions
result = solve_network(
    net,
    gas_flows_kgh={"H2": 8.0},           # kg/h hydrogen
    liquid_type="Water",
    liquid_flows_kgh={"Water": 2480.0},  # kg/h water
    inlet_node_id=topo1.inlet_node_id,
    outlet_node_id=topo2.outlet_node_id,
    m_total_kgs=(8.0 + 2480.0) / 3600.0,
)

print(f"Converged in {result.iterations} iterations, residual={result.residual:.2e}")

for topo, label in [(topo1, "Harp 1"), (topo2, "Harp 2")]:
    report = check_starvation(result, topo.channel_edge_ids)
    print(format_report(report, label))
```

---

## `build_harp()` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | int | Number of parallel channels (≥ 2) |
| `header_dn` / `header_pn` | str | Nominal diameter / pressure class for both headers |
| `header_segment_length` | float | Length (m) of each inter-channel header segment |
| `channel_dn` / `channel_pn` | str | Channel pipe size |
| `channel_length` | float | Channel physical length (m) |
| `channel_angle_rad` | float | Channel inclination: 0 = horizontal, +π/2 = vertical up |
| `header_fittings_le` | float | Extra equivalent length (m) of fittings on each header segment |
| `channel_fittings_le` | float | Extra equivalent length (m) of fittings on each channel |
| `correlation` | str | Two-phase correlation (`"Beggs-Brill"`, `"Friedel"`, …) |
| `voidage_method` | str | Void-fraction method (`"Homogeneous"`, `"Rouhani-1 (slip)"`) |
| `harp_id` | str | Prefix for all node/edge IDs (use different values for each harp) |
| `P_inlet_pa` | float | Initial pressure (Pa) assigned to all nodes |
| `T_C` | float | Temperature (°C) — isothermal assumption |
| `x_inlet` | float | Inlet mass quality (0 = pure liquid, 1 = pure gas) |

`build_series_harp()` accepts the same parameters plus `connector_dn`,
`connector_pn`, `connector_length`, and `connector_angle_rad` / `connector_fittings_le`.

---

## `solve_network()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gas_flows_kgh` | — | Dict of gas species flows at inlet (kg/h) |
| `liquid_type` | — | Liquid identifier (e.g. `"Water"`, `"KOH"`) |
| `liquid_flows_kgh` | `None` | Dict of liquid species flows (kg/h); preferred over `q_lye_m3h` |
| `q_lye_m3h` | `None` | Total liquid volumetric flow (m³/h); used if `liquid_flows_kgh` not given |
| `m_total_kgs` | — | Total mass flow into the network (kg/s) |
| `max_iter` | 150 | Maximum Hardy-Cross iterations |
| `tol_rel` | 1×10⁻⁴ | Convergence: max loop residual / max edge ΔP |
| `relax` | 0.7 | Under-relaxation factor (reduce to 0.3–0.4 for high-GVF cases) |

---

## Limitations and future extensions

| Limitation | Note |
|------------|------|
| Isothermal only | Temperature is fixed at `T_C`; no heat loss along pipes |
| Homogeneous phase split | Gas and liquid split at junctions in proportion to total flow; Azzopardi T-junction correlations would improve accuracy |
| Named-edge convention | Initialisation relies on `_ch`, `_hA`, `_hB` in edge IDs — a consequence of coupling topology and solver |
| U-flow only | `build_harp` generates inlet at A_0, outlet at B_N; Z-flow (same side) is not yet available |

---

## References

- **Hardy, C. (1936)** — "Analysis of flow in networks of conduits or conductors."  
  *University of Illinois Bulletin*, 34(22).
- **Beggs, H. D. & Brill, J. P. (1973)** — two-phase pressure-drop correlation used for each edge.
- **Acrivos, A. et al.** — manifold flow analysis; analytical treatments of U-type and Z-type header pressure distribution.
