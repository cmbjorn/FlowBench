"""
Global Gradient Algorithm (GGA) pipe-network solver for two-phase gas/liquid flow.

Algorithm
---------
Todini & Pilati (1988) — the method used by EPANET.  Solves for nodal pressures
P and pipe mass-flows m simultaneously via Newton-Raphson.

Advantages over Hardy-Cross
----------------------------
- Quadratic convergence (Hardy-Cross is linear).
- No loop detection required — works on any topology.
- Handles flow reversal naturally every iteration.
- Dynamic friction exponent: n=2 turbulent, n=1 laminar.
- Dynamic T-junction K-factors that switch with flow direction (§B constraint).

Sign convention  (§A constraint: dP = K·Q·|Q|)
-----------------------------------------------
  h_e  = dP_e * sign(m_e)    ← always opposes flow direction
  r_e  = |dP_e| / max(m_e², ε)  [Pa·s²/kg²]
  D_ee = n_e * |dP_e| / max(|m_e|, ε)   [Jacobian diagonal]

Boundary conditions
-------------------
  - Outlet node: pressure PINNED at P_outlet_pa (user-provided).
  - Inlet node: net external inflow = m_total_kgs (mass BC).
  - All other nodes: net external flow = 0.

The inlet pressure is a RESULT of the calculation (the pressure required at the
inlet to push m_total through the network against the downstream P_outlet).

Units
-----
SI throughout: Pa, m, kg/s, °C.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .graph import Edge, Network


@dataclass
class SolveResult:
    converged:    bool
    iterations:   int
    residual:     float
    net:          Network           # mutated in-place; same object as input
    edge_results: dict[str, dict]   # edge_id → last calculate_segment_pressure_drop dict
    warnings:     list[str] = field(default_factory=list)


# ── Constants ─────────────────────────────────────────────────────────────────

_M_MIN    = 1e-8    # kg/s — floor to avoid div-by-zero in Jacobian
_R_MIN    = 1e-12   # Pa·s²/kg² — floor for resistance
_D_MIN    = 1e-10   # floor for Jacobian diagonal
_RE_LAM   = 2300.0  # Reynolds number below which n_e = 1 (laminar)

# ── Incidence matrix construction ─────────────────────────────────────────────

def _build_incidence(net: Network) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build the node-edge incidence matrix B and return it with ordered id lists.

    B[e, n] = +1 if edge e leaves node n  (from_node)
    B[e, n] = -1 if edge e enters node n  (to_node)
    Shape: (N_edges, N_nodes)

    The mass-balance equation is: B^T @ m = b
    The energy equation per edge:  h_e = (B @ P)[e]  where h = r*m*|m|
    """
    node_ids = [n.node_id for n in net.all_nodes()]
    edge_ids = [e.edge_id for e in net.all_edges()]
    n_idx    = {nid: i for i, nid in enumerate(node_ids)}

    B = np.zeros((len(edge_ids), len(node_ids)), dtype=float)
    for ei, eid in enumerate(edge_ids):
        e = net.edge(eid)
        B[ei, n_idx[e.from_node]] = +1.0
        B[ei, n_idx[e.to_node]]   = -1.0

    return B, node_ids, edge_ids


# ── Property computation for one edge ─────────────────────────────────────────

def _compute_props_and_dp(
    e: Edge,
    net: Network,
    m_total_kgs: float,
    x_inlet: float,
    gas_flows_kgh: dict[str, float],
    liquid_type: str,
    q_lye_m3h: float | None,
    liquid_flows_kgh: dict[str, float] | None,
    custom_gas: dict | None,
    custom_liquid: dict | None,
    rho_l_cached: float,
) -> tuple[dict[str, Any], float]:
    """
    Compute two-phase properties and unsigned pressure drop for edge *e*.

    Returns (props_dict, dP_Pa_unsigned).
    The unsigned dP_Pa is always >= 0; the caller applies the sign convention.

    Also applies T-junction K-factor (direction-dependent) to the returned dP.
    """
    from multiphase_engine import (
        calculate_segment_pressure_drop,
        calculate_two_phase_properties,
    )

    x_node = net.node(e.from_node).x_gas
    m_abs  = max(abs(e.m_kgs), _M_MIN)

    m_gas_kgs = m_abs * x_node
    m_liq_kgs = m_abs * (1.0 - x_node)

    x_inlet_safe = max(x_inlet, 1e-12)
    gas_scale    = m_gas_kgs / (m_total_kgs * x_inlet_safe)
    local_gas    = {sp: v * gas_scale for sp, v in gas_flows_kgh.items()}

    local_liq: dict[str, float] | None = None
    local_q   = 0.0
    if liquid_flows_kgh:
        liq_total_inlet = max(sum(v / 3600.0 for v in liquid_flows_kgh.values()), 1e-12)
        liq_scale       = m_liq_kgs / liq_total_inlet
        local_liq       = {sp: v * liq_scale for sp, v in liquid_flows_kgh.items()}
    else:
        local_q = m_liq_kgs * 3600.0 / max(rho_l_cached, 1.0)

    P_bara = max(net.node(e.from_node).P_pa / 1e5, 0.01)
    T_C    = net.node(e.from_node).T_C

    props = calculate_two_phase_properties(
        P_bara, T_C,
        gas_flows_kgh=local_gas,
        liquid_type=liquid_type,
        q_lye_m3h=local_q if local_liq is None else 0.0,
        custom_gas=custom_gas,
        custom_liquid=custom_liquid,
        liquid_flows_kgh=local_liq,
    )

    res = calculate_segment_pressure_drop(
        props, e.D_inner, e.roughness, e.L_eff, e.angle_rad,
        correlation=e.correlation, voidage_method=e.voidage_method,
    )
    dP = float(res["dP_Pa"])

    # T-junction K-factor (§B): direction-dependent, applied to mass flux
    K_junc = e.junction_K_fwd if e.m_kgs >= 0 else e.junction_K_rev
    if K_junc > 0.0:
        A_e    = math.pi * e.D_inner ** 2 / 4.0
        G_e    = m_abs / max(A_e, 1e-12)                    # kg/(m²·s)
        rho_mix = props.get("rho_hom", props.get("rho_l", 1000.0))
        dP    += K_junc * G_e ** 2 / (2.0 * max(rho_mix, 1.0))

    # Write regime info to edge
    e.Vsg   = res["Vsg"]
    e.Vsl   = res["Vsl"]
    e.alpha = res["alpha"]
    e.regime = res["regime"]

    return props, max(dP, 0.0)


# ── Quality BFS ───────────────────────────────────────────────────────────────

def _quality_bfs(net: Network, inlet_node_id: str, x_inlet: float) -> None:
    """Propagate x_gas from inlet via mass-weighted mixing (homogeneous model)."""
    net.node(inlet_node_id).x_gas = x_inlet
    for nid in net.bfs_nodes(inlet_node_id):
        if nid == inlet_node_id:
            continue
        incoming: list[tuple[Edge, float]] = []
        for e in net.edges_to(nid):
            if e.m_kgs > 0:
                incoming.append((e, e.m_kgs))
        for e in net.edges_from(nid):
            if e.m_kgs < 0:
                incoming.append((e, -e.m_kgs))
        if not incoming:
            continue
        m_in = sum(f for _, f in incoming)
        if m_in < 1e-12:
            continue
        x_new = sum(
            f * net.node(e.to_node if e.m_kgs < 0 else e.from_node).x_gas
            for e, f in incoming
        ) / m_in
        net.node(nid).x_gas = min(1.0, max(0.0, x_new))


# ── Public API ────────────────────────────────────────────────────────────────

def solve_network(
    net: Network,
    *,
    gas_flows_kgh: dict[str, float],
    liquid_type: str,
    q_lye_m3h: float | None = None,
    liquid_flows_kgh: dict[str, float] | None = None,
    custom_gas: dict | None = None,
    custom_liquid: dict | None = None,
    inlet_node_id: str,
    outlet_node_id: str,
    m_total_kgs: float,
    P_outlet_pa: float | None = None,   # pin outlet pressure; default = 0.95 * P_inlet
    max_iter: int = 100,
    tol_rel: float = 1e-4,
    relax: float = 1.0,                 # GGA typically uses full Newton step (relax=1)
) -> SolveResult:
    """
    Solve for flow distribution in the pipe network *net* using the
    Global Gradient Algorithm (Newton-Raphson nodal analysis).

    Boundary conditions
    -------------------
    * Outlet node pressure is **pinned** at *P_outlet_pa* (or 0.95·P_inlet if
      not given).  This is the single pressure degree-of-freedom.
    * Inlet node receives net inflow *m_total_kgs*.
    * All other nodes have zero net external flow.

    The inlet pressure is a **result** — the pressure the network requires at
    the inlet to push *m_total_kgs* through to the pinned outlet.

    Parameters
    ----------
    relax
        Newton step size (0 < relax ≤ 1).  GGA converges quadratically at
        relax=1 for well-conditioned networks; reduce toward 0.5 for
        high-GVF or near-stall cases.
    """
    from multiphase_engine import calculate_two_phase_properties

    warnings: list[str] = []
    P_inlet = net.node(inlet_node_id).P_pa

    # ── Outlet pressure BC ────────────────────────────────────────────────────
    if P_outlet_pa is None:
        P_outlet_pa = 0.95 * P_inlet
    net.node(outlet_node_id).P_pa = P_outlet_pa

    # ── Cache inlet fluid properties (for rho_l) ─────────────────────────────
    x_inlet = net.node(inlet_node_id).x_gas
    _inlet_props = calculate_two_phase_properties(
        max(P_inlet / 1e5, 0.01), net.node(inlet_node_id).T_C,
        gas_flows_kgh=gas_flows_kgh,
        liquid_type=liquid_type,
        q_lye_m3h=q_lye_m3h or 0.0,
        custom_gas=custom_gas,
        custom_liquid=custom_liquid,
        liquid_flows_kgh=liquid_flows_kgh,
    )
    rho_l_cached = float(_inlet_props.get("rho_l", 1000.0))

    # ── Build incidence matrix ────────────────────────────────────────────────
    B, node_ids, edge_ids = _build_incidence(net)
    N_n = len(node_ids)
    N_e = len(edge_ids)
    n_idx = {nid: i for i, nid in enumerate(node_ids)}
    e_idx = {eid: i for i, eid in enumerate(edge_ids)}

    outlet_idx = n_idx[outlet_node_id]
    inlet_idx  = n_idx[inlet_node_id]

    # External flow vector b: (B^T @ m)[n] = b[n] at steady state,
    # where (B^T @ m)[n] = net pipe outflow from node n.
    # Inlet: pipes carry m_total away from it  → b[inlet]  = +m_total
    # Outlet: pipes bring m_total into it       → b[outlet] = -m_total
    b = np.zeros(N_n)
    b[inlet_idx]  = +m_total_kgs
    b[outlet_idx] = -m_total_kgs

    # ── Initialise flows ──────────────────────────────────────────────────────
    # Uniform initial guess: every edge carries m_total / N_e.
    # The mass balances are NOT satisfied initially; GGA corrects them.
    for e in net.all_edges():
        e.m_kgs = m_total_kgs / max(N_e, 1)

    m_vec = np.array([net.edge(eid).m_kgs for eid in edge_ids])
    P_vec = np.array([net.node(nid).P_pa  for nid in node_ids])

    # ── GGA iteration ─────────────────────────────────────────────────────────
    converged   = False
    residual    = float("inf")
    edge_results: dict[str, dict] = {}

    _fluid_kwargs = dict(
        m_total_kgs=m_total_kgs, x_inlet=x_inlet,
        gas_flows_kgh=gas_flows_kgh, liquid_type=liquid_type,
        q_lye_m3h=q_lye_m3h, liquid_flows_kgh=liquid_flows_kgh,
        custom_gas=custom_gas, custom_liquid=custom_liquid,
        rho_l_cached=rho_l_cached,
    )

    for iteration in range(1, max_iter + 1):

        # A. Write current flows/pressures into Network objects
        for i, eid in enumerate(edge_ids):
            net.edge(eid).m_kgs  = float(m_vec[i])
        for i, nid in enumerate(node_ids):
            net.node(nid).P_pa   = float(P_vec[i])

        # B. Quality BFS (homogeneous phase split)
        _quality_bfs(net, inlet_node_id, x_inlet)

        # C. Compute dP for every edge; build D (Jacobian diagonal) and h (signed losses)
        h_vec = np.zeros(N_e)
        D_vec = np.zeros(N_e)

        for i, eid in enumerate(edge_ids):
            e = net.edge(eid)
            props, dP_unsigned = _compute_props_and_dp(e, net, **_fluid_kwargs)
            edge_results[eid] = {"dP_Pa": dP_unsigned, "Vsg": e.Vsg, "Vsl": e.Vsl,
                                 "alpha": e.alpha, "regime": e.regime}

            # §A: signed head loss  h = dP * sign(m)
            h_vec[i] = dP_unsigned * (1.0 if m_vec[i] >= 0 else -1.0)
            e.dP_Pa  = h_vec[i]

            # Reynolds number for turbulent/laminar exponent
            A_e   = math.pi * e.D_inner ** 2 / 4.0
            rho_m = props.get("rho_hom", props.get("rho_l", 1000.0))
            mu_m  = props.get("mu_l", 1e-3)
            Re    = rho_m * abs(m_vec[i]) / max(A_e * mu_m, 1e-20)
            n_e   = 1.0 if Re < _RE_LAM else 2.0

            # Jacobian entry  D_ee = n * |dP| / max(|m|, eps)
            D_vec[i] = max(n_e * dP_unsigned / max(abs(m_vec[i]), _M_MIN), _D_MIN)

        # D. Assemble and solve the condensed nodal system
        # G = B^T * diag(1/D) * B   (N_nodes × N_nodes)
        inv_D = 1.0 / D_vec
        G = (B * inv_D[:, np.newaxis]).T @ B      # B^T @ diag(1/D) @ B

        # Energy residual per edge: r_e = h_e - (B @ P)[e]  = h - B·P
        BP    = B @ P_vec
        e_res = h_vec - BP                         # energy residual (should → 0)

        # Mass balance residual at each node
        m_res = B.T @ m_vec - b                    # should → 0

        # RHS = B^T @ diag(1/D) @ e_res  -  m_res
        rhs = (B * inv_D[:, np.newaxis]).T @ e_res - m_res

        # Pin outlet node: fix ΔP[outlet] = 0
        G[outlet_idx, :]  = 0.0
        G[:, outlet_idx]  = 0.0
        G[outlet_idx, outlet_idx] = 1.0
        rhs[outlet_idx]   = 0.0

        try:
            dP_nodes = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            warnings.append(f"Iteration {iteration}: singular matrix — reducing step.")
            dP_nodes = np.linalg.lstsq(G, rhs, rcond=None)[0]

        # E. Update pressures
        P_vec += relax * dP_nodes

        # F. Update flows:  Δm = diag(1/D) * (B @ P_new - h)
        dm_vec = inv_D * (B @ P_vec - h_vec)
        m_vec += relax * dm_vec

        # G. Clamp tiny flows to avoid complete stall
        m_vec = np.where(np.abs(m_vec) < _M_MIN / 10, 0.0, m_vec)

        # H. Convergence: relative pressure correction + mass balance error
        res_p   = np.max(np.abs(dP_nodes)) / max(P_inlet, 1.0)
        res_m   = np.max(np.abs(B.T @ m_vec - b)) / max(m_total_kgs, 1e-12)
        residual = max(res_p, res_m)

        if residual < tol_rel:
            converged = True
            break

    # Final write-back of converged state
    for i, eid in enumerate(edge_ids):
        net.edge(eid).m_kgs = float(m_vec[i])
    for i, nid in enumerate(node_ids):
        net.node(nid).P_pa  = float(P_vec[i])
    _quality_bfs(net, inlet_node_id, x_inlet)

    # Final edge properties pass
    for i, eid in enumerate(edge_ids):
        e = net.edge(eid)
        props, dP_u = _compute_props_and_dp(e, net, **_fluid_kwargs)
        e.dP_Pa = dP_u * (1.0 if m_vec[i] >= 0 else -1.0)
        edge_results[eid] = {
            "dP_Pa": dP_u, "dP_fric_Pa": dP_u,
            "dP_grav_Pa": 0.0, "dP_accel_Pa": 0.0,
            "Vsg": e.Vsg, "Vsl": e.Vsl, "alpha": e.alpha, "regime": e.regime,
        }

    if not converged:
        warnings.append(
            f"GGA did not converge after {max_iter} iterations "
            f"(residual={residual:.3e}, tol={tol_rel:.3e}). "
            "Results are the best approximation."
        )

    return SolveResult(
        converged=converged,
        iterations=iteration,
        residual=residual,
        net=net,
        edge_results=edge_results,
        warnings=warnings,
    )
