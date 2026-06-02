"""
General pipe-network solver UI.

Provides a grid-based network editor (Plotly + st.data_editor) where the user
can visually define any pipe topology, load templates (Z-manifold, U-manifold,
ring, …), specify fluid conditions, and solve using the Global Gradient Algorithm.

The harp manifold is just one of the available templates.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import plotly.graph_objects as go
import streamlit as st

import multiphase_engine as engine
from pipe_network import (
    SolveResult, StarvationReport,
    build_network_from_spec, check_all_edges, check_starvation,
    solve_network, validate_network,
    template_harp, template_ring,
)
from pipe_network.topology import build_harp, build_series_harp
from standards.piping import MATERIAL_ROUGHNESS, PIPE_DATABASE

# ── Constants ─────────────────────────────────────────────────────────────────

_DN_OPTIONS = list(PIPE_DATABASE.keys())
_PN_OPTIONS = ["PN20", "PN25", "PN40"]
_LIQUID_OPTIONS = list(engine.LIQUID_COOLPROP_ID.keys())
_GAS_CP = {
    "H₂": "H2", "CO₂": "CarbonDioxide", "Air": "Air",
    "N₂": "Nitrogen", "O₂": "O2", "CH₄": "Methane", "H₂S": "H2S",
}

_GRID_W, _GRID_H = 18, 10          # grid anchor dimensions
_NODE_COLORS = {
    "inlet":    "#16a34a",
    "outlet":   "#dc2626",
    "junction": "#2563EB",
    "pending":  "#f59e0b",
}

# ── Session-state keys ────────────────────────────────────────────────────────

def _k(name: str) -> str:
    return f"ns_{name}"


def _init_state() -> None:
    """Ensure required session-state keys exist."""
    if _k("nodes") not in st.session_state:
        st.session_state[_k("nodes")] = []
    if _k("edges") not in st.session_state:
        st.session_state[_k("edges")] = []
    if _k("mode") not in st.session_state:
        st.session_state[_k("mode")] = "node"
    if _k("pending_from") not in st.session_state:
        st.session_state[_k("pending_from")] = None


# ── Templates ─────────────────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    "— blank (custom) —":       "blank",
    "Z-manifold":               "harp_z",
    "U-manifold (same-side outlet)": "harp_u",
    "Two Z-manifolds in series": "series_z",
    "Two U-manifolds in series": "series_u",
    "Ring network":             "ring",
}


def _load_template(tpl_key: str, N: int, P_bara: float, T_C: float, x_gas: float,
                   hdr_D: float, ch_D: float, hdr_len: float, ch_len: float) -> None:
    """Populate session-state nodes/edges from a template."""
    if tpl_key == "blank":
        st.session_state[_k("nodes")] = []
        st.session_state[_k("edges")] = []
        return

    htype = "U" if "u" in tpl_key else "Z"

    if tpl_key in ("harp_z", "harp_u"):
        nodes, edges = template_harp(N, htype,
            header_D_m=hdr_D, channel_D_m=ch_D,
            header_seg_len=hdr_len, channel_len=ch_len,
            P_inlet_bara=P_bara, T_C=T_C, x_gas=x_gas)
        st.session_state[_k("nodes")] = nodes
        st.session_state[_k("edges")] = edges

    elif tpl_key in ("series_z", "series_u"):
        n1, e1 = template_harp(N, htype,
            header_D_m=hdr_D, channel_D_m=ch_D,
            header_seg_len=hdr_len, channel_len=ch_len,
            P_inlet_bara=P_bara, T_C=T_C, x_gas=x_gas)
        # Shift second harp to the right
        offset = N + 3
        n2, e2 = template_harp(N, htype,
            header_D_m=hdr_D, channel_D_m=ch_D,
            header_seg_len=hdr_len, channel_len=ch_len,
            P_inlet_bara=P_bara * 0.97, T_C=T_C, x_gas=x_gas)
        # Rename second harp nodes/edges to avoid clashes
        id_map = {}
        for nd in n2:
            new_id = f"H2_{nd['id']}"
            id_map[nd["id"]] = new_id
            nd["id"] = new_id
            nd["x"] = nd.get("x", 0) + offset
            nd["type"] = "junction"  # will re-assign inlet/outlet below
        for ed in e2:
            ed["id"] = f"H2_{ed['id']}"
            ed["from"] = id_map[ed["from"]]
            ed["to"]   = id_map[ed["to"]]
        # Connector from H1 outlet to H2 inlet
        h1_out = next(n["id"] for n in n1 if n["type"] == "outlet")
        h2_in  = next(n["id"] for n in n2 if n["id"].endswith("_A0"))
        n2_inlet = next(n for n in n2 if n["id"] == h2_in)
        n2_inlet["type"] = "junction"
        h2_out_id = next(n["id"] for n in n2 if n["id"].endswith(
            "_B0" if htype == "U" else f"_B{N}"))
        next(n for n in n2 if n["id"] == h2_out_id)["type"] = "outlet"
        # Make H1 outlet a junction (it feeds connector)
        next(n for n in n1 if n["id"] == h1_out)["type"] = "junction"
        connector = {
            "id": "connector", "from": h1_out, "to": h2_in,
            "D_inner_m": hdr_D, "roughness_m": 4.6e-5,
            "L_pipe_m": hdr_len * 2, "angle_deg": 0.0,
        }
        st.session_state[_k("nodes")] = n1 + n2
        st.session_state[_k("edges")] = e1 + e2 + [connector]

    elif tpl_key == "ring":
        nodes, edges = template_ring(N,
            D_m=hdr_D, seg_len=hdr_len,
            P_inlet_bara=P_bara, T_C=T_C, x_gas=x_gas)
        st.session_state[_k("nodes")] = nodes
        st.session_state[_k("edges")] = edges

    st.session_state[_k("pending_from")] = None


# ── Canvas helpers ────────────────────────────────────────────────────────────

def _build_canvas(nodes: list[dict], edges: list[dict],
                  mode: str, pending_from: str | None) -> go.Figure:
    """Build the interactive Plotly network canvas."""
    fig = go.Figure()
    node_by_id = {n["id"]: n for n in nodes}

    # Grid anchor points (faint dots)
    gx = [i for i in range(_GRID_W + 1) for _ in range(_GRID_H + 1)]
    gy = [j for _ in range(_GRID_W + 1) for j in range(_GRID_H + 1)]
    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode="markers",
        marker=dict(size=5, color="rgba(200,200,200,0.4)", symbol="circle"),
        hoverinfo="skip", showlegend=False,
        customdata=[[f"grid:{xi},{yi}" for xi, yi in zip(gx, gy)]],
    ))

    # Edges (lines)
    for ed in edges:
        fn = node_by_id.get(ed.get("from", ""), {})
        tn = node_by_id.get(ed.get("to", ""), {})
        if fn and tn:
            x0, y0 = fn.get("x", 0), fn.get("y", 0)
            x1, y1 = tn.get("x", 0), tn.get("y", 0)
            D_mm = ed.get("D_inner_m", 0.025) * 1000
            L_m  = ed.get("L_pipe_m", 1.0)
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#475569", width=3),
                hovertemplate=(
                    f"<b>{ed['id']}</b><br>"
                    f"{ed.get('from', '?')} → {ed.get('to', '?')}<br>"
                    f"ID = {D_mm:.1f} mm  L = {L_m:.3f} m"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

    # Nodes (circles)
    for nd in nodes:
        ntype   = nd.get("type", "junction")
        color   = _NODE_COLORS.get(ntype, "#64748b")
        if nd["id"] == pending_from:
            color = _NODE_COLORS["pending"]
        symbol  = {"inlet": "arrow-right", "outlet": "arrow-left"}.get(ntype, "circle")
        label   = nd.get("label", nd["id"])
        P_label = f"  {nd.get('P_bara', '?'):.1f} bara" if ntype in ("inlet", "outlet") else ""
        fig.add_trace(go.Scatter(
            x=[nd.get("x", 0)], y=[nd.get("y", 0)],
            mode="markers+text",
            marker=dict(size=16, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            text=[f"{label}{P_label}"],
            textposition="top center",
            textfont=dict(size=9, color=color),
            hovertemplate=(
                f"<b>{nd['id']}</b><br>"
                f"Type: {ntype}<br>"
                f"P = {nd.get('P_bara', '?')} bara<br>"
                f"x_gas = {nd.get('x_gas', 0):.4f}"
                "<extra></extra>"
            ),
            customdata=[[f"node:{nd['id']}"]],
            showlegend=False,
        ))

    # Mode hint annotation
    hints = {
        "node":   "Click grid → add node | Click node → remove",
        "pipe":   "Click node A then node B → add pipe",
        "delete": "Click node or near-edge → delete",
        "type":   "Click node → cycle type (junction → inlet → outlet → junction)",
    }
    fig.add_annotation(
        x=0, y=_GRID_H + 0.6,
        text=f"<i>Mode: <b>{mode}</b> — {hints.get(mode, '')}</i>",
        showarrow=False, xanchor="left",
        font=dict(size=10, color="#64748b"),
    )

    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(l=15, r=15, t=30, b=30),
        xaxis=dict(range=[-0.5, _GRID_W + 0.5], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, _GRID_H + 1.2], showgrid=False,
                   zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        clickmode="event+select",
    )
    return fig


def _handle_canvas_click(event: Any) -> bool:
    """Process a Plotly click event; return True if state was mutated."""
    if not event or not event.selection or not event.selection.points:
        return False

    pt       = event.selection.points[0]
    mode     = st.session_state[_k("mode")]
    nodes    = st.session_state[_k("nodes")]
    edges    = st.session_state[_k("edges")]
    node_ids = {n["id"] for n in nodes}

    # Snap clicked coordinates to nearest grid point
    x_raw = pt.get("x", 0)
    y_raw = pt.get("y", 0)
    x_snap = int(round(x_raw))
    y_snap = int(round(y_raw))
    x_snap = max(0, min(_GRID_W, x_snap))
    y_snap = max(0, min(_GRID_H, y_snap))

    # Identify if click hit an existing node
    clicked_node: str | None = None
    for nd in nodes:
        if nd.get("x") == x_snap and nd.get("y") == y_snap:
            clicked_node = nd["id"]
            break

    mutated = False

    if mode == "node":
        if clicked_node:
            # Remove the node (and its edges)
            st.session_state[_k("nodes")] = [n for n in nodes if n["id"] != clicked_node]
            st.session_state[_k("edges")] = [
                e for e in edges
                if e.get("from") != clicked_node and e.get("to") != clicked_node
            ]
        else:
            # Add a new junction node
            new_id = f"N{len(nodes)}"
            while new_id in node_ids:
                new_id = f"N{len(nodes) + len(new_id)}"
            nodes.append({
                "id": new_id, "type": "junction",
                "P_bara": 10.0, "T_C": 60.0, "x_gas": 0.0,
                "x": x_snap, "y": y_snap,
                "label": new_id,
            })
        mutated = True

    elif mode == "pipe":
        pending = st.session_state[_k("pending_from")]
        if pending is None:
            # First click: select source node
            if clicked_node:
                st.session_state[_k("pending_from")] = clicked_node
                mutated = True
        else:
            # Second click: create edge
            target = clicked_node
            if target is None:
                # Add a new node at clicked position and connect to it
                new_id = f"N{len(nodes)}"
                nodes.append({
                    "id": new_id, "type": "junction",
                    "P_bara": 10.0, "T_C": 60.0, "x_gas": 0.0,
                    "x": x_snap, "y": y_snap, "label": new_id,
                })
                target = new_id
            if target != pending:
                eid = f"p{len(edges)}"
                edges.append({
                    "id": eid, "from": pending, "to": target,
                    "D_inner_m": 0.025, "roughness_m": 4.6e-5,
                    "L_pipe_m": 1.0, "angle_deg": 0.0,
                    "junction_K_fwd": 0.0, "junction_K_rev": 0.0,
                })
            st.session_state[_k("pending_from")] = None
            mutated = True

    elif mode == "delete":
        if clicked_node:
            st.session_state[_k("nodes")] = [n for n in nodes if n["id"] != clicked_node]
            st.session_state[_k("edges")] = [
                e for e in edges
                if e.get("from") != clicked_node and e.get("to") != clicked_node
            ]
            mutated = True

    elif mode == "type":
        if clicked_node:
            cycle = {"junction": "inlet", "inlet": "outlet", "outlet": "junction"}
            nd = next(n for n in nodes if n["id"] == clicked_node)
            nd["type"] = cycle.get(nd.get("type", "junction"), "junction")
            mutated = True

    return mutated


# ── Results diagrams (adapted from network_tab.py) ────────────────────────────

def _make_results_diagram(solve_res: SolveResult, nodes: list[dict],
                           edges: list[dict], Vsl_thr: float,
                           single_phase: bool) -> go.Figure:
    """Plotly figure overlaying solve results on the network topology."""
    net = solve_res.net
    fig = go.Figure()
    node_by_id = {n["id"]: n for n in nodes}

    # Edges coloured by Vsl / flow direction
    for ed in edges:
        fn = node_by_id.get(ed.get("from", ""), {})
        tn = node_by_id.get(ed.get("to", ""), {})
        if not fn or not tn:
            continue
        x0, y0 = fn.get("x", 0), fn.get("y", 0)
        x1, y1 = tn.get("x", 0), tn.get("y", 0)
        try:
            e = net.edge(ed["id"])
        except KeyError:
            continue

        Vsl   = e.Vsl
        m_kgs = e.m_kgs
        if single_phase:
            mean_m = sum(abs(net.edge(ex).m_kgs) for ex in [ex.edge_id for ex in net.all_edges()]) / max(len(list(net.all_edges())), 1)
            frac = abs(m_kgs) / max(mean_m, 1e-12)
            color = "#22c55e" if frac >= 0.9 else ("#f59e0b" if frac >= 0.7 else "#ef4444")
        else:
            color = ("#22c55e" if Vsl >= Vsl_thr else
                     "#f59e0b" if Vsl >= 0.5 * Vsl_thr else "#ef4444")
        width = max(2, min(10, int(abs(m_kgs) * 3600 / 5)))
        arrow = "→" if m_kgs >= 0 else "←"
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=(
                f"<b>{ed['id']}</b>  {arrow}<br>"
                f"m = {m_kgs*1000:.2f} g/s<br>"
                f"Vsl = {Vsl:.4f} m/s<br>"
                f"Vsg = {e.Vsg:.4f} m/s<br>"
                f"Regime: {e.regime}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Nodes with pressure labels
    for nd in nodes:
        try:
            P = net.node(nd["id"]).P_pa / 1e5
        except KeyError:
            P = nd.get("P_bara", 0)
        ntype = nd.get("type", "junction")
        color = _NODE_COLORS.get(ntype, "#64748b")
        fig.add_trace(go.Scatter(
            x=[nd.get("x", 0)], y=[nd.get("y", 0)],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(color="white", width=2)),
            text=[f"{nd['id']}<br>{P:.3f}"],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate=f"<b>{nd['id']}</b><br>P = {P:.4f} bara<extra></extra>",
            showlegend=False,
        ))

    for label, color in [("OK", "#22c55e"), ("Low", "#f59e0b"), ("Starved/Low-flow", "#ef4444")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                  line=dict(color=color, width=3), name=label))

    fig.update_layout(
        template="plotly_white", height=360,
        margin=dict(l=15, r=15, t=30, b=30),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render_network_solver_tab() -> None:
    _init_state()

    st.markdown(
        "**General pipe-network solver** — define any topology on the grid, "
        "or load a template. The GGA (Global Gradient Algorithm / Newton-Raphson) "
        "solver handles any number of loops, flow reversal, and mixed "
        "laminar/turbulent flow. The harp manifold is one of the available templates."
    )

    col_ed, col_res = st.columns([11, 10], gap="large")

    # ═══════════════════════════════════════════════════════
    # LEFT COLUMN — editor
    # ═══════════════════════════════════════════════════════
    with col_ed:

        # ── Template loader ───────────────────────────────
        st.markdown("#### Template")
        tl1, tl2 = st.columns([3, 1])
        tpl_label = tl1.selectbox("", list(_TEMPLATES.keys()),
                                   key=_k("tpl_label"), label_visibility="hidden")
        tpl_key   = _TEMPLATES[tpl_label]

        with st.expander("Template parameters", expanded=(tpl_key != "blank")):
            tp1, tp2, tp3 = st.columns(3)
            N_tpl      = tp1.number_input("Channels", 2, 300, 6, key=_k("tpl_N"))
            P_tpl      = tp2.number_input("P_inlet (bara)", 0.1, 500.0, 10.0,
                                           step=0.5, key=_k("tpl_P"))
            T_tpl      = tp3.number_input("T (°C)", -50.0, 350.0, 60.0,
                                           step=5.0, key=_k("tpl_T"))
            td1, td2, td3, td4 = st.columns(4)
            hdr_D_tpl  = td1.number_input("Header ID (mm)", 5.0, 500.0, 50.0,
                                           step=1.0, key=_k("tpl_hdr_D")) / 1000.0
            ch_D_tpl   = td2.number_input("Channel ID (mm)", 1.0, 300.0, 25.0,
                                           step=0.5, key=_k("tpl_ch_D"))  / 1000.0
            hdr_len    = td3.number_input("Header seg (m)", 0.0, 100.0, 0.15,
                                           step=0.05, key=_k("tpl_hlen"))
            ch_len     = td4.number_input("Channel L (m)", 0.0, 100.0, 0.50,
                                           step=0.05, key=_k("tpl_clen"))
            x_tpl      = st.number_input("x_gas at inlet", 0.0, 1.0, 0.004,
                                          step=0.001, format="%.4f", key=_k("tpl_x"))

        if tl2.button("Load", use_container_width=True, type="secondary"):
            _load_template(tpl_key, int(N_tpl), P_tpl, T_tpl, x_tpl,
                           hdr_D_tpl, ch_D_tpl, hdr_len, ch_len)
            st.session_state.pop(_k("solve_cache"), None)
            st.rerun()

        st.divider()

        # ── Canvas mode selector ──────────────────────────
        st.markdown("#### Network canvas")
        mode_opts = ["node", "pipe", "delete", "type"]
        mode_lbls = ["✚ Node", "⟶ Pipe", "✕ Delete", "⊙ Set type"]
        _cur_mode = st.session_state[_k("mode")]
        _mode_idx = mode_opts.index(_cur_mode) if _cur_mode in mode_opts else 0
        mode_sel  = st.radio("", mode_lbls, index=_mode_idx,
                              horizontal=True, key=_k("mode_radio"),
                              label_visibility="hidden")
        st.session_state[_k("mode")] = mode_opts[mode_lbls.index(mode_sel)]

        # Canvas
        canvas_fig = _build_canvas(
            st.session_state[_k("nodes")],
            st.session_state[_k("edges")],
            st.session_state[_k("mode")],
            st.session_state[_k("pending_from")],
        )
        canvas_event = st.plotly_chart(
            canvas_fig, use_container_width=True,
            on_select="rerun", selection_mode=["points"],
            key=_k("canvas"),
        )
        if _handle_canvas_click(canvas_event):
            st.rerun()

        # ── Node / edge tables ────────────────────────────
        import pandas as pd

        with st.expander("Node table", expanded=False):
            node_cols = {
                "id":     st.column_config.TextColumn("ID"),
                "type":   st.column_config.SelectboxColumn("Type",
                              options=["junction", "inlet", "outlet"]),
                "P_bara": st.column_config.NumberColumn("P (bara)",
                              min_value=0.0, format="%.2f"),
                "T_C":    st.column_config.NumberColumn("T (°C)", format="%.0f"),
                "x_gas":  st.column_config.NumberColumn("x_gas",
                              min_value=0.0, max_value=1.0, format="%.4f"),
                "label":  st.column_config.TextColumn("Label"),
                "x":      st.column_config.NumberColumn("x (grid)", format="%.0f"),
                "y":      st.column_config.NumberColumn("y (grid)", format="%.0f"),
            }
            node_df = pd.DataFrame(st.session_state[_k("nodes")] or
                                   [{"id": "", "type": "junction", "P_bara": 10.0,
                                     "T_C": 60.0, "x_gas": 0.0, "label": "", "x": 0, "y": 0}])
            edited_nodes = st.data_editor(
                node_df, key=_k("node_tbl"),
                num_rows="dynamic", column_config=node_cols,
                hide_index=True, use_container_width=True,
            )
            if st.button("Apply node edits", key=_k("apply_nodes")):
                st.session_state[_k("nodes")] = edited_nodes.to_dict("records")
                st.session_state.pop(_k("solve_cache"), None)
                st.rerun()

        with st.expander("Edge table", expanded=False):
            edge_cols = {
                "id":             st.column_config.TextColumn("ID"),
                "from":           st.column_config.TextColumn("From"),
                "to":             st.column_config.TextColumn("To"),
                "D_inner_m":      st.column_config.NumberColumn("ID (m)", format="%.4f", min_value=0.0001),
                "L_pipe_m":       st.column_config.NumberColumn("L (m)", format="%.4f", min_value=0.0),
                "roughness_m":    st.column_config.NumberColumn("ε (m)", format="%.6f"),
                "angle_deg":      st.column_config.NumberColumn("Angle (°)", format="%.1f"),
                "junction_K_fwd": st.column_config.NumberColumn("K_fwd", format="%.3f", min_value=0.0),
                "junction_K_rev": st.column_config.NumberColumn("K_rev", format="%.3f", min_value=0.0),
            }
            edge_df = pd.DataFrame(st.session_state[_k("edges")] or
                                   [{"id": "", "from": "", "to": "",
                                     "D_inner_m": 0.025, "L_pipe_m": 1.0,
                                     "roughness_m": 4.6e-5, "angle_deg": 0.0,
                                     "junction_K_fwd": 0.0, "junction_K_rev": 0.0}])
            edited_edges = st.data_editor(
                edge_df, key=_k("edge_tbl"),
                num_rows="dynamic", column_config=edge_cols,
                hide_index=True, use_container_width=True,
            )
            if st.button("Apply edge edits", key=_k("apply_edges")):
                st.session_state[_k("edges")] = edited_edges.to_dict("records")
                st.session_state.pop(_k("solve_cache"), None)
                st.rerun()

        st.divider()

        # ── Fluid ─────────────────────────────────────────
        st.markdown("#### Fluid")
        single_phase = st.toggle("Single-phase liquid", key=_k("single_phase"),
                                  value=st.session_state.get(_k("single_phase"), False))

        fc1, fc2 = st.columns(2)
        liq_type = fc1.selectbox("Liquid", _LIQUID_OPTIONS, key=_k("liq_type"))
        liq_kgh  = fc2.number_input(f"{liq_type} (kg/h)", 0.0, 500_000.0,
                                     value=float(st.session_state.get(_k("liq_kgh"), 1000.0)),
                                     step=10.0, format="%.0f", key=_k("liq_kgh"))

        if not single_phase:
            gc1, gc2 = st.columns(2)
            gas_lbl  = gc1.selectbox("Gas", list(_GAS_CP.keys()), key=_k("gas_lbl"))
            gas_kgh  = gc2.number_input(f"{gas_lbl} (kg/h)", 0.0, 10000.0,
                                         value=float(st.session_state.get(_k("gas_kgh"), 5.0)),
                                         step=0.5, format="%.1f", key=_k("gas_kgh"))
        else:
            gas_lbl = "H₂"; gas_kgh = 0.0

        m_total  = (gas_kgh + liq_kgh) / 3600.0
        x_inlet  = gas_kgh / max(gas_kgh + liq_kgh, 1e-12)
        st.caption(f"Total: **{(gas_kgh+liq_kgh):.0f} kg/h**  |  "
                   f"x_gas = **{x_inlet:.4f}**  |  {m_total:.5f} kg/s")

        # Outlet pressure
        P_nodes = [n.get("P_bara", 10.0) for n in st.session_state[_k("nodes")]
                   if n.get("type") == "inlet"]
        P_inlet_default = P_nodes[0] if P_nodes else 10.0
        P_outlet = st.number_input(
            "Downstream pressure P_out (bara)",
            min_value=0.01, max_value=500.0,
            value=float(st.session_state.get(_k("P_outlet"), P_inlet_default * 0.95)),
            step=0.1, format="%.2f", key=_k("P_outlet"),
            help="Pressure pinned at the outlet node. "
                 "The inlet pressure is a result of the solve.",
        )

        # Starvation threshold
        Vsl_thr = st.number_input(
            "Min V_sl (m/s)", 0.0, 10.0,
            value=float(st.session_state.get(_k("Vsl_thr"), 0.05)),
            step=0.005, format="%.3f", key=_k("Vsl_thr"),
        )

        # Solver settings
        with st.expander("Solver settings", expanded=False):
            corr = st.selectbox("Correlation", engine.TWO_PHASE_CORRELATIONS,
                                 index=0, key=_k("corr"))
            void = st.selectbox("Void fraction", engine.VOIDAGE_METHODS,
                                 index=0, key=_k("void"))
            sc1, sc2 = st.columns(2)
            relax    = sc1.number_input("Relax", 0.1, 1.0, 1.0, step=0.1,
                                         format="%.2f", key=_k("relax"))
            max_iter = sc2.number_input("Max iters", 10, 200, 50, step=10,
                                         key=_k("max_iter"))

        solve_clicked = st.button("▶  Solve", use_container_width=True, type="primary")

    # ═══════════════════════════════════════════════════════
    # RIGHT COLUMN — results
    # ═══════════════════════════════════════════════════════

    # Build inputs hash for caching
    _inputs = dict(
        nodes=json.dumps(st.session_state[_k("nodes")], sort_keys=True, default=str),
        edges=json.dumps(st.session_state[_k("edges")], sort_keys=True, default=str),
        gas_kgh=gas_kgh, liq_kgh=liq_kgh, gas_lbl=gas_lbl, liq_type=liq_type,
        single_phase=single_phase, P_outlet=P_outlet, Vsl_thr=Vsl_thr,
        corr=corr, void=void, relax=relax, max_iter=max_iter,
    )
    _hash = hashlib.md5(json.dumps(_inputs, sort_keys=True).encode()).hexdigest()
    _cache_key = _k("solve_cache")
    _hash_key  = _k("solve_hash")

    if solve_clicked or (
        st.session_state.get(_hash_key) != _hash
        and _cache_key not in st.session_state
    ):
        if solve_clicked or _cache_key not in st.session_state:
            with col_res:
                with st.spinner("Validating and solving…"):
                    cur_nodes = st.session_state[_k("nodes")]
                    cur_edges = st.session_state[_k("edges")]

                    # Pre-validate
                    try:
                        net_tmp = build_network_from_spec(cur_nodes, cur_edges)
                        ok, msgs = validate_network(net_tmp)
                    except Exception as exc:
                        st.error(f"Network build error: {exc}")
                        st.session_state.pop(_cache_key, None)
                        ok = False; msgs = []

                    if not ok:
                        for msg in msgs:
                            if msg.startswith("ERROR"):
                                st.error(msg)
                            else:
                                st.warning(msg)
                        st.session_state.pop(_cache_key, None)
                    else:
                        for msg in msgs:
                            if not msg.startswith("ERROR"):
                                st.info(msg)
                        try:
                            # Re-build net (validate_network may not preserve state)
                            net = build_network_from_spec(cur_nodes, cur_edges,
                                defaults={"correlation": corr, "voidage_method": void})
                            inlet_id  = next(n.node_id for n in net.all_nodes() if n.is_inlet)
                            outlet_id = next(n.node_id for n in net.all_nodes() if n.is_outlet)

                            gas_flows   = {} if single_phase else {_GAS_CP[gas_lbl]: gas_kgh}
                            liq_flows   = {liq_type: liq_kgh}

                            res = solve_network(
                                net,
                                gas_flows_kgh=gas_flows,
                                liquid_type=liq_type,
                                liquid_flows_kgh=liq_flows,
                                inlet_node_id=inlet_id,
                                outlet_node_id=outlet_id,
                                m_total_kgs=m_total,
                                P_outlet_pa=P_outlet * 1e5,
                                max_iter=int(max_iter),
                                relax=float(relax),
                            )
                            rep = check_all_edges(res, Vsl_thr)

                            st.session_state[_cache_key] = (res, rep, cur_nodes, cur_edges)
                            st.session_state[_hash_key]  = _hash
                        except Exception as exc:
                            st.error(f"Solver error: {exc}")
                            st.session_state.pop(_cache_key, None)

    # ── Display results ───────────────────────────────────
    with col_res:
        if _cache_key not in st.session_state:
            st.info("Define a network and click **▶ Solve**.")
            return

        res, rep, rn, re = st.session_state[_cache_key]

        # Status metrics
        mc = st.columns(4)
        mc[0].metric("Solver", f"{'✅' if res.converged else '⚠️'} {res.iterations} iters")
        mc[1].metric("Residual", f"{res.residual:.2e}")
        P_in_result = res.net.node(
            next(n.node_id for n in res.net.all_nodes() if n.is_inlet)
        ).P_pa / 1e5
        mc[2].metric("P_inlet (result)", f"{P_in_result:.3f} bara")
        mc[3].metric("Low-flow edges", f"{rep.n_channels_starved}/{rep.n_channels_total}")

        if not res.converged:
            st.warning("Solver did not converge — results are approximate. "
                       "Try fewer channels, higher relaxation, or more iterations.")
        for w in res.warnings:
            if "did not converge" not in w.lower():
                st.caption(f"ℹ {w}")

        # Results diagram
        st.markdown("##### Network — solved")
        st.plotly_chart(
            _make_results_diagram(res, rn, re, Vsl_thr, single_phase),
            use_container_width=True, config={"displayModeBar": False},
        )

        # Edge detail table
        st.markdown("##### Edge results")
        import pandas as pd
        rows = []
        for e in res.net.all_edges():
            rows.append({
                "Edge": e.edge_id,
                "From → To": f"{e.from_node} → {e.to_node}",
                "m (g/s)": round(e.m_kgs * 1000, 2),
                "ΔP (Pa)": round(e.dP_Pa, 1),
                "V_sl (m/s)": round(e.Vsl, 4),
                "V_sg (m/s)": round(e.Vsg, 4),
                "α": round(e.alpha, 4),
                "Regime": e.regime,
                "Status": ("🟢" if e.Vsl >= Vsl_thr else
                            "🟡" if e.Vsl >= 0.5 * Vsl_thr else "🔴"),
            })
        if rows:
            df = pd.DataFrame(rows)
            def _style(row):
                v = row.get("V_sl (m/s)", 1)
                if v < Vsl_thr:
                    return ["background-color:#fee2e2"] * len(row)
                if v < 0.5 * Vsl_thr:
                    return ["background-color:#fef3c7"] * len(row)
                return [""] * len(row)
            st.dataframe(df.style.apply(_style, axis=1),
                         use_container_width=True, hide_index=True,
                         height=min(420, 52 + len(rows) * 36))

        # Method note
        with st.expander("ℹ Solver method", expanded=False):
            st.markdown(
                f"""
**Global Gradient Algorithm** (Todini & Pilati, 1988) — same method as EPANET.

Newton-Raphson nodal analysis: simultaneously solves for nodal pressures *P* and
pipe flows *m*.  Converges quadratically; no loop detection needed.

**Sign convention (§A):** `dP = r·m·|m|` — pressure drop always opposes flow
direction.  When a pipe reverses, its `dP` flips sign automatically.

**T-junction K-factors (§B):** each pipe carries `junction_K_fwd` (dividing T,
when `m ≥ 0`) or `junction_K_rev` (combining T, when `m < 0`).  These switch
every iteration, correctly modelling the **first few dozen taps** in a
U-manifold where the collection-header T-junctions experience reversed flow.

**Friction:** Churchill correlation (all Re, smooth and rough).

**Phase model:** homogeneous quality mixing at junctions.

*Convergence:* `max|ΔP|/P_inlet < {res.residual:.2e}` after {res.iterations} iterations.
                """
            )


# Back-compat alias used by network_tab.py
render_harp_network_tab = render_network_solver_tab
