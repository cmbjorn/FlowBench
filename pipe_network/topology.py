"""
Factory functions that construct pipe-network graphs for harp manifold
geometries.

A *harp manifold* consists of:
  - Header A  : supply header (one end capped), split into N segments
  - N channels: identical parallel tubes connecting header A to header B
  - Header B  : collection header (one end capped), split into N segments

Two standard configurations:

  **Z-manifold**  (``harp_type="Z"``)
    Inlet at A_0, outlet at B_N — *opposite ends*.
    Both headers flow in the same direction (→ →).
    The near-inlet channels get less flow than far-end channels.

    ::

        Q_in →
        A_0 ─── A_1 ─── A_2 ─── ... ─── A_N (capped)
         |       |       |                |
        ch_0   ch_1    ch_2            ch_{N-1}
         |       |       |                |
        B_0 ─── B_1 ─── B_2 ─── ... ─── B_N → Q_out

  **U-manifold**  (``harp_type="U"``) — also *reverse-return manifold*
    Inlet at A_0, outlet at B_0 — *same end*.
    Header A flows forward (→); header B flows in reverse (←).

    ::

        Q_in →
        A_0 ─── A_1 ─── A_2 ─── ... ─── A_N (capped)
         |       |       |                |
        ch_0   ch_1    ch_2            ch_{N-1}
         |       |       |                |
        B_0 ─── B_1 ─── B_2 ─── ... ─── B_N (capped)
         ↑
        Q_out  (same side as Q_in)

Topology for N channels (either type)
--------------------------------------
  Nodes : 2(N+1)   Edges : 3N   Independent loops : N − 1
"""
from __future__ import annotations

import math

from standards.piping import MATERIAL_ROUGHNESS, PIPE_DATABASE

from .graph import Edge, HarpTopology, Network, Node


def build_harp(
    N: int,
    *,
    header_dn: str = "",
    header_pn: str = "PN40",
    header_segment_length: float,
    channel_dn: str = "",
    channel_pn: str = "PN40",
    channel_length: float,
    channel_angle_rad: float = 0.0,
    header_material: str = "SS316L",
    channel_material: str = "SS316L",
    header_fittings_le: float = 0.0,
    channel_fittings_le: float = 0.0,
    correlation: str = "Beggs-Brill",
    voidage_method: str = "Homogeneous",
    harp_id: str = "H1",
    harp_type: str = "Z",
    P_inlet_pa: float = 30e5,
    T_C: float = 60.0,
    x_inlet: float = 0.05,
    # Optional custom geometry overrides (take precedence over DN/PN lookup)
    header_D_inner_m: float | None = None,
    header_roughness_m: float | None = None,
    channel_D_inner_m: float | None = None,
    channel_roughness_m: float | None = None,
) -> tuple[Network, HarpTopology]:
    """
    Build a single harp manifold with *N* channels.

    Parameters
    ----------
    N
        Number of parallel channels (2 – 300).
    header_dn / header_pn
        Nominal diameter / pressure class for both headers.
        Ignored when ``header_D_inner_m`` is provided.
    header_segment_length
        Physical length (m) of each inter-channel header pipe segment.
    channel_dn / channel_pn
        Diameter / pressure class for channel tubes.
        Ignored when ``channel_D_inner_m`` is provided.
    channel_length
        Physical length (m) of each channel.
    channel_angle_rad
        Inclination: 0 = horizontal, +π/2 = vertical up, −π/2 = vertical down.
    header_D_inner_m
        Custom inner diameter (m) for header pipes, overriding DN/PN lookup.
    header_roughness_m
        Custom absolute roughness (m) for header pipes.
        Defaults to ``MATERIAL_ROUGHNESS[header_material]`` when not set.
    channel_D_inner_m
        Custom inner diameter (m) for channel pipes, overriding DN/PN lookup.
    channel_roughness_m
        Custom absolute roughness (m) for channel pipes.
    harp_type
        ``"Z"`` — **Z-manifold**: inlet at A_0, outlet at B_N (opposite ends).
        Both headers flow in the same direction.

        ``"U"`` — **U-manifold** (reverse-return): inlet at A_0, outlet at B_0
        (same end).  The supply header A flows forward (→); the collection
        header B flows in reverse (←) back to the outlet.  Also known as
        *reverse-return manifold* or *U-tube manifold*.

    Returns
    -------
    net : Network
    topo : HarpTopology
    """
    if N < 2:
        raise ValueError("A harp requires at least 2 channels (N >= 2).")
    if N > 500:
        raise ValueError("Maximum 500 channels per harp.")
    if harp_type not in ("Z", "U"):
        raise ValueError(f"harp_type must be 'Z' or 'U', got {harp_type!r}.")

    # ── Pipe geometry ────────────────────────────────────────────────────────
    h_D   = header_D_inner_m  if header_D_inner_m  is not None else PIPE_DATABASE[header_dn][header_pn]
    c_D   = channel_D_inner_m if channel_D_inner_m is not None else PIPE_DATABASE[channel_dn][channel_pn]
    h_eps = header_roughness_m  if header_roughness_m  is not None else MATERIAL_ROUGHNESS[header_material]
    c_eps = channel_roughness_m if channel_roughness_m is not None else MATERIAL_ROUGHNESS[channel_material]

    net = Network()

    # ── Nodes ────────────────────────────────────────────────────────────────
    # Pressure guess: inlet pressure, identical everywhere at construction.
    a_node_ids: list[str] = []
    b_node_ids: list[str] = []

    for i in range(N + 1):
        aid = f"{harp_id}_A{i}"
        bid = f"{harp_id}_B{i}"
        a_node_ids.append(aid)
        b_node_ids.append(bid)
        net.add_node(Node(
            node_id=aid,
            P_pa=P_inlet_pa,
            x_gas=x_inlet,
            T_C=T_C,
            is_inlet=(i == 0),
            is_outlet=False,
        ))
        net.add_node(Node(
            node_id=bid,
            P_pa=P_inlet_pa,
            x_gas=x_inlet,
            T_C=T_C,
            is_inlet=False,
            is_outlet=(i == N),
        ))

    # ── Header-A edges  A_i → A_{i+1} ───────────────────────────────────────
    ha_edge_ids: list[str] = []
    for i in range(N):
        eid = f"{harp_id}_hA{i}"
        ha_edge_ids.append(eid)
        net.add_edge(Edge(
            edge_id=eid,
            from_node=a_node_ids[i],
            to_node=a_node_ids[i + 1],
            D_inner=h_D,
            roughness=h_eps,
            L_pipe=header_segment_length,
            L_fittings=header_fittings_le,
            angle_rad=0.0,
            m_kgs=0.0,
            correlation=correlation,
            voidage_method=voidage_method,
        ))

    # ── Channel edges  A_i → B_i ─────────────────────────────────────────────
    ch_edge_ids: list[str] = []
    for i in range(N):
        eid = f"{harp_id}_ch{i}"
        ch_edge_ids.append(eid)
        net.add_edge(Edge(
            edge_id=eid,
            from_node=a_node_ids[i],
            to_node=b_node_ids[i],
            D_inner=c_D,
            roughness=c_eps,
            L_pipe=channel_length,
            L_fittings=channel_fittings_le,
            angle_rad=channel_angle_rad,
            m_kgs=0.0,
            correlation=correlation,
            voidage_method=voidage_method,
            # T-junction K-factors (Idelchik §7-23): forward = dividing T,
            # reverse = combining T.  For U-manifold the collection-header Ts
            # may experience reversed flow, so K_rev applies there.
            junction_K_fwd=0.5,
            junction_K_rev=0.3,
        ))

    # ── Header-B edges ────────────────────────────────────────────────────────
    # Z-type: B_i → B_{i+1}  (outlet at B_N, same direction as header A)
    # U-type: B_{i+1} → B_i  (outlet at B_0, reverse direction)
    hb_edge_ids: list[str] = []
    for i in range(N):
        eid = f"{harp_id}_hB{i}"
        hb_edge_ids.append(eid)
        if harp_type == "Z":
            b_from, b_to = b_node_ids[i], b_node_ids[i + 1]
        else:  # U-type
            b_from, b_to = b_node_ids[i + 1], b_node_ids[i]
        net.add_edge(Edge(
            edge_id=eid,
            from_node=b_from,
            to_node=b_to,
            D_inner=h_D,
            roughness=h_eps,
            L_pipe=header_segment_length,
            L_fittings=header_fittings_le,
            angle_rad=0.0,
            m_kgs=0.0,
            correlation=correlation,
            voidage_method=voidage_method,
        ))

    outlet_node = b_node_ids[N] if harp_type == "Z" else b_node_ids[0]
    # Mark outlet on the correct node
    for nid in b_node_ids:
        net.node(nid).is_outlet = False
    net.node(outlet_node).is_outlet = True
    net._outlet_node_id = outlet_node

    topo = HarpTopology(
        harp_id=harp_id,
        harp_type=harp_type,
        inlet_node_id=a_node_ids[0],
        outlet_node_id=outlet_node,
        header_A_node_ids=a_node_ids,
        header_B_node_ids=b_node_ids,
        channel_edge_ids=ch_edge_ids,
        header_A_edge_ids=ha_edge_ids,
        header_B_edge_ids=hb_edge_ids,
    )
    return net, topo


def build_series_harp(
    N: int,
    *,
    # Harp geometry (identical for both harps in this builder)
    header_dn: str = "",
    header_pn: str = "PN40",
    header_segment_length: float,
    channel_dn: str = "",
    channel_pn: str = "PN40",
    channel_length: float,
    channel_angle_rad: float = 0.0,
    header_material: str = "SS316L",
    channel_material: str = "SS316L",
    header_fittings_le: float = 0.0,
    channel_fittings_le: float = 0.0,
    # Connector pipe between the two harps
    connector_dn: str = "",
    connector_pn: str = "PN40",
    connector_length: float,
    connector_angle_rad: float = 0.0,
    connector_material: str = "SS316L",
    connector_fittings_le: float = 0.0,
    # Shared parameters
    correlation: str = "Beggs-Brill",
    voidage_method: str = "Homogeneous",
    P_inlet_pa: float = 30e5,
    T_C: float = 60.0,
    x_inlet: float = 0.05,
    harp_type: str = "Z",
    # Optional custom geometry overrides
    header_D_inner_m: float | None = None,
    header_roughness_m: float | None = None,
    channel_D_inner_m: float | None = None,
    channel_roughness_m: float | None = None,
    connector_D_inner_m: float | None = None,
    connector_roughness_m: float | None = None,
) -> tuple[Network, HarpTopology, HarpTopology]:
    """
    Build two identical harps in series, joined by a single connector pipe.

    The outlet of harp 1 connects to the inlet of harp 2 regardless of the
    harp_type (Z or U).  Both harps use the same type.

    Returns
    -------
    net   : merged Network
    topo1 : HarpTopology for harp 1
    topo2 : HarpTopology for harp 2
    """
    _harp_kwargs = dict(
        header_dn=header_dn, header_pn=header_pn,
        header_segment_length=header_segment_length,
        channel_dn=channel_dn, channel_pn=channel_pn,
        channel_length=channel_length,
        channel_angle_rad=channel_angle_rad,
        header_material=header_material, channel_material=channel_material,
        header_fittings_le=header_fittings_le,
        channel_fittings_le=channel_fittings_le,
        correlation=correlation, voidage_method=voidage_method,
        P_inlet_pa=P_inlet_pa, T_C=T_C, x_inlet=x_inlet,
        harp_type=harp_type,
        header_D_inner_m=header_D_inner_m, header_roughness_m=header_roughness_m,
        channel_D_inner_m=channel_D_inner_m, channel_roughness_m=channel_roughness_m,
    )
    net1, topo1 = build_harp(N, harp_id="H1", **_harp_kwargs)
    net2, topo2 = build_harp(N, harp_id="H2", **_harp_kwargs)

    # Merge — keep only the lines that are actually needed after this refactor.
    # Clear boundary flags so the merged network has exactly one inlet/outlet.
    net1.node(topo1.outlet_node_id).is_outlet = False
    net2.node(topo2.inlet_node_id).is_inlet   = False

    net1.merge(net2)
    net = net1

    # Connector edge: H1 outlet → H2 inlet
    c_D   = connector_D_inner_m  if connector_D_inner_m  is not None else PIPE_DATABASE[connector_dn][connector_pn]
    c_eps = connector_roughness_m if connector_roughness_m is not None else MATERIAL_ROUGHNESS[connector_material]
    net.add_edge(Edge(
        edge_id="connector",
        from_node=topo1.outlet_node_id,
        to_node=topo2.inlet_node_id,
        D_inner=c_D,
        roughness=c_eps,
        L_pipe=connector_length,
        L_fittings=connector_fittings_le,
        angle_rad=connector_angle_rad,
        m_kgs=0.0,
        correlation=correlation,
        voidage_method=voidage_method,
    ))

    net.node(topo1.inlet_node_id).is_inlet   = True
    net.node(topo2.outlet_node_id).is_outlet = True
    net._inlet_node_id  = topo1.inlet_node_id
    net._outlet_node_id = topo2.outlet_node_id

    return net, topo1, topo2


# ============================================================================
# I-manifold (center-fed) and Double-inlet Z builders
# ============================================================================

def build_center_fed_harp(
    N: int,
    *,
    header_dn: str = "",
    header_pn: str = "PN40",
    header_segment_length: float,
    channel_dn: str = "",
    channel_pn: str = "PN40",
    channel_length: float,
    channel_angle_rad: float = 0.0,
    header_material: str = "SS316L",
    channel_material: str = "SS316L",
    header_fittings_le: float = 0.0,
    channel_fittings_le: float = 0.0,
    correlation: str = "Beggs-Brill",
    voidage_method: str = "Homogeneous",
    harp_id: str = "H1",
    P_inlet_pa: float = 30e5,
    T_C: float = 60.0,
    x_inlet: float = 0.05,
    header_D_inner_m: float | None = None,
    header_roughness_m: float | None = None,
    channel_D_inner_m: float | None = None,
    channel_roughness_m: float | None = None,
) -> tuple["Network", HarpTopology]:
    """
    **I-manifold (center-fed)**: inlet at the centre of header A, outlet at the
    centre of header B.  Both ends of each header are capped.

    Flow distributes symmetrically outward from the centre — dramatically more
    uniform than Z or U for large N.  Channels near the centre receive the most
    flow; the distribution is symmetric around the feed point.

    ::

        (capped)                                      (capped)
           A_0 ──── A_1 ──── A_k ──── A_{N} ... ──── A_N
                             │
                           ↑ INLET
                        (A_{N//2})

    The feed and collection pipes are modelled as very short (1 mm) pipes of the
    same diameter as the header, so they contribute negligible extra pressure drop.
    """
    if N < 2:
        raise ValueError("N >= 2 required.")
    if N > 500:
        raise ValueError("Maximum 500 channels.")

    h_D   = header_D_inner_m  if header_D_inner_m  is not None else PIPE_DATABASE[header_dn][header_pn]
    c_D   = channel_D_inner_m if channel_D_inner_m is not None else PIPE_DATABASE[channel_dn][channel_pn]
    h_eps = header_roughness_m  if header_roughness_m  is not None else MATERIAL_ROUGHNESS[header_material]
    c_eps = channel_roughness_m if channel_roughness_m is not None else MATERIAL_ROUGHNESS[channel_material]

    net = Network()
    center_a = N // 2   # feed point on header A
    center_b = N // 2   # collection point on header B

    # External inlet / outlet nodes
    inlet_id  = f"{harp_id}_IN"
    outlet_id = f"{harp_id}_OUT"
    net.add_node(Node(inlet_id,  P_pa=P_inlet_pa, x_gas=x_inlet, T_C=T_C,
                      is_inlet=True))
    net.add_node(Node(outlet_id, P_pa=P_inlet_pa * 0.98, x_gas=x_inlet, T_C=T_C,
                      is_outlet=True))

    # Header A and B nodes — both ends capped
    a_ids, b_ids = [], []
    for i in range(N + 1):
        aid = f"{harp_id}_A{i}"
        bid = f"{harp_id}_B{i}"
        a_ids.append(aid)
        b_ids.append(bid)
        net.add_node(Node(aid, P_pa=P_inlet_pa, x_gas=x_inlet, T_C=T_C))
        net.add_node(Node(bid, P_pa=P_inlet_pa * 0.98, x_gas=x_inlet, T_C=T_C))

    _ekw = dict(correlation=correlation, voidage_method=voidage_method)

    # Feed: INLET → A_{center_a} (very short stub)
    net.add_edge(Edge(f"{harp_id}_feed", inlet_id, a_ids[center_a],
                      D_inner=h_D, roughness=h_eps, L_pipe=0.001, L_fittings=0.0,
                      angle_rad=0.0, m_kgs=0.0, **_ekw))
    # Collection: B_{center_b} → OUTLET (very short stub)
    net.add_edge(Edge(f"{harp_id}_coll", b_ids[center_b], outlet_id,
                      D_inner=h_D, roughness=h_eps, L_pipe=0.001, L_fittings=0.0,
                      angle_rad=0.0, m_kgs=0.0, **_ekw))

    # Header A segments (A_i → A_{i+1}, both ends capped)
    ha_ids = []
    for i in range(N):
        eid = f"{harp_id}_hA{i}"
        ha_ids.append(eid)
        net.add_edge(Edge(eid, a_ids[i], a_ids[i + 1],
                          D_inner=h_D, roughness=h_eps,
                          L_pipe=header_segment_length, L_fittings=header_fittings_le,
                          angle_rad=0.0, m_kgs=0.0, **_ekw))

    # Channels (with T-junction K-factors)
    ch_ids = []
    for i in range(N):
        eid = f"{harp_id}_ch{i}"
        ch_ids.append(eid)
        net.add_edge(Edge(eid, a_ids[i], b_ids[i],
                          D_inner=c_D, roughness=c_eps,
                          L_pipe=channel_length, L_fittings=channel_fittings_le,
                          angle_rad=channel_angle_rad, m_kgs=0.0,
                          junction_K_fwd=0.5, junction_K_rev=0.3, **_ekw))

    # Header B segments (B_i → B_{i+1}, both ends capped)
    hb_ids = []
    for i in range(N):
        eid = f"{harp_id}_hB{i}"
        hb_ids.append(eid)
        net.add_edge(Edge(eid, b_ids[i], b_ids[i + 1],
                          D_inner=h_D, roughness=h_eps,
                          L_pipe=header_segment_length, L_fittings=header_fittings_le,
                          angle_rad=0.0, m_kgs=0.0, **_ekw))

    topo = HarpTopology(
        harp_id=harp_id,
        harp_type="I",
        inlet_node_id=inlet_id,
        outlet_node_id=outlet_id,
        header_A_node_ids=a_ids,
        header_B_node_ids=b_ids,
        channel_edge_ids=ch_ids,
        header_A_edge_ids=ha_ids,
        header_B_edge_ids=hb_ids,
    )
    return net, topo


def build_biinlet_harp(
    N: int,
    *,
    header_dn: str = "",
    header_pn: str = "PN40",
    header_segment_length: float,
    channel_dn: str = "",
    channel_pn: str = "PN40",
    channel_length: float,
    channel_angle_rad: float = 0.0,
    header_material: str = "SS316L",
    channel_material: str = "SS316L",
    header_fittings_le: float = 0.0,
    channel_fittings_le: float = 0.0,
    correlation: str = "Beggs-Brill",
    voidage_method: str = "Homogeneous",
    harp_id: str = "H1",
    P_inlet_pa: float = 30e5,
    T_C: float = 60.0,
    x_inlet: float = 0.05,
    header_D_inner_m: float | None = None,
    header_roughness_m: float | None = None,
    channel_D_inner_m: float | None = None,
    channel_roughness_m: float | None = None,
) -> tuple["Network", HarpTopology]:
    """
    **Double-inlet Z-manifold**: flow enters at **both** ends of the supply header
    simultaneously (A_0 and A_N).  The collection header exits at B_N as in a
    standard Z-manifold.

    A single external inlet node splits into two equal feed pipes — one to each
    end of header A.  The header pressure gradient is halved compared to a
    single-inlet Z, dramatically reducing maldistribution.  The distribution
    pattern is roughly parabolic, with channels near each end getting the most
    flow and the centre channel getting the least.

    ::

                      ↓ INLET
                     /       \\
        A_0 ─────────────────── A_N
         |                       |
        ch_0  ...            ch_{N-1}
         |                       |
        B_0 ─────────────────── B_N → OUTLET
    """
    if N < 2:
        raise ValueError("N >= 2 required.")
    if N > 500:
        raise ValueError("Maximum 500 channels.")

    h_D   = header_D_inner_m  if header_D_inner_m  is not None else PIPE_DATABASE[header_dn][header_pn]
    c_D   = channel_D_inner_m if channel_D_inner_m is not None else PIPE_DATABASE[channel_dn][channel_pn]
    h_eps = header_roughness_m  if header_roughness_m  is not None else MATERIAL_ROUGHNESS[header_material]
    c_eps = channel_roughness_m if channel_roughness_m is not None else MATERIAL_ROUGHNESS[channel_material]

    net = Network()

    # External inlet node → feeds both ends of header A
    inlet_id = f"{harp_id}_IN"
    net.add_node(Node(inlet_id, P_pa=P_inlet_pa, x_gas=x_inlet, T_C=T_C,
                      is_inlet=True))

    # Header nodes
    a_ids, b_ids = [], []
    for i in range(N + 1):
        aid = f"{harp_id}_A{i}"
        bid = f"{harp_id}_B{i}"
        a_ids.append(aid)
        b_ids.append(bid)
        net.add_node(Node(aid, P_pa=P_inlet_pa, x_gas=x_inlet, T_C=T_C))
        is_out = (i == N)
        net.add_node(Node(bid, P_pa=P_inlet_pa * 0.98, x_gas=x_inlet, T_C=T_C,
                          is_outlet=is_out))

    _ekw = dict(correlation=correlation, voidage_method=voidage_method)

    # Two feed stubs: INLET → A_0 and INLET → A_N
    net.add_edge(Edge(f"{harp_id}_feed0", inlet_id, a_ids[0],
                      D_inner=h_D, roughness=h_eps, L_pipe=0.001, L_fittings=0.0,
                      angle_rad=0.0, m_kgs=0.0, **_ekw))
    net.add_edge(Edge(f"{harp_id}_feedN", inlet_id, a_ids[N],
                      D_inner=h_D, roughness=h_eps, L_pipe=0.001, L_fittings=0.0,
                      angle_rad=0.0, m_kgs=0.0, **_ekw))

    # Header A (directed A_i → A_{i+1}; GGA allows reversed flow)
    ha_ids = []
    for i in range(N):
        eid = f"{harp_id}_hA{i}"
        ha_ids.append(eid)
        net.add_edge(Edge(eid, a_ids[i], a_ids[i + 1],
                          D_inner=h_D, roughness=h_eps,
                          L_pipe=header_segment_length, L_fittings=header_fittings_le,
                          angle_rad=0.0, m_kgs=0.0, **_ekw))

    # Channels
    ch_ids = []
    for i in range(N):
        eid = f"{harp_id}_ch{i}"
        ch_ids.append(eid)
        net.add_edge(Edge(eid, a_ids[i], b_ids[i],
                          D_inner=c_D, roughness=c_eps,
                          L_pipe=channel_length, L_fittings=channel_fittings_le,
                          angle_rad=channel_angle_rad, m_kgs=0.0,
                          junction_K_fwd=0.5, junction_K_rev=0.3, **_ekw))

    # Header B (Z-style: B_0 → B_N, outlet at B_N)
    hb_ids = []
    for i in range(N):
        eid = f"{harp_id}_hB{i}"
        hb_ids.append(eid)
        net.add_edge(Edge(eid, b_ids[i], b_ids[i + 1],
                          D_inner=h_D, roughness=h_eps,
                          L_pipe=header_segment_length, L_fittings=header_fittings_le,
                          angle_rad=0.0, m_kgs=0.0, **_ekw))

    topo = HarpTopology(
        harp_id=harp_id,
        harp_type="DI",
        inlet_node_id=inlet_id,
        outlet_node_id=b_ids[N],
        header_A_node_ids=a_ids,
        header_B_node_ids=b_ids,
        channel_edge_ids=ch_ids,
        header_A_edge_ids=ha_ids,
        header_B_edge_ids=hb_ids,
    )
    return net, topo


# ============================================================================
# Generic builder and validator
# ============================================================================

def build_network_from_spec(
    nodes: list[dict],
    edges: list[dict],
    defaults: dict | None = None,
) -> Network:
    """
    Build a ``Network`` from plain dicts — no harp assumptions.

    Node spec keys
    --------------
    id           str   required — unique node identifier
    type         str   "junction" | "inlet" | "outlet"  (default: "junction")
    P_bara       float initial pressure (bara)  (default: 10.0)
    T_C          float temperature °C           (default: 25.0)
    x_gas        float mass quality             (default: 0.0)

    Edge spec keys
    --------------
    id           str   required
    from         str   required — source node id
    to           str   required — target node id
    D_inner_m    float inner diameter (m)       required unless dn/pn given
    dn / pn      str   DN and PN — looked up in PIPE_DATABASE
    roughness_m  float absolute roughness (m)   (default: 4.6e-5 — SS316L)
    L_pipe_m     float pipe length (m)          (default: 1.0)
    L_fittings_m float fitting equiv. length    (default: 0.0)
    angle_deg    float inclination degrees      (default: 0)
    junction_K_fwd / junction_K_rev  float      (default: 0.0)
    correlation  str                            (default: "Beggs-Brill")
    voidage_method str                          (default: "Homogeneous")
    """
    defs = {
        "type": "junction", "P_bara": 10.0, "T_C": 25.0, "x_gas": 0.0,
        "roughness_m": 4.6e-5, "L_pipe_m": 1.0, "L_fittings_m": 0.0,
        "angle_deg": 0.0, "junction_K_fwd": 0.0, "junction_K_rev": 0.0,
        "correlation": "Beggs-Brill", "voidage_method": "Homogeneous",
    }
    if defaults:
        defs.update(defaults)

    net = Network()

    for nd in nodes:
        ntype = nd.get("type", defs["type"])
        node = Node(
            node_id   = str(nd["id"]),
            P_pa      = float(nd.get("P_bara", defs["P_bara"])) * 1e5,
            x_gas     = float(nd.get("x_gas",  defs["x_gas"])),
            T_C       = float(nd.get("T_C",    defs["T_C"])),
            is_inlet  = (ntype == "inlet"),
            is_outlet = (ntype == "outlet"),
        )
        net.add_node(node)

    for ed in edges:
        # Resolve diameter
        if "D_inner_m" in ed and ed["D_inner_m"] is not None:
            D = float(ed["D_inner_m"])
        elif "dn" in ed and "pn" in ed:
            D = PIPE_DATABASE[ed["dn"]][ed["pn"]]
        else:
            raise ValueError(f"Edge {ed['id']!r}: provide 'D_inner_m' or 'dn'+'pn'.")

        edge = Edge(
            edge_id        = str(ed["id"]),
            from_node      = str(ed["from"]),
            to_node        = str(ed["to"]),
            D_inner        = D,
            roughness      = float(ed.get("roughness_m",   defs["roughness_m"])),
            L_pipe         = float(ed.get("L_pipe_m",      defs["L_pipe_m"])),
            L_fittings     = float(ed.get("L_fittings_m",  defs["L_fittings_m"])),
            angle_rad      = float(ed.get("angle_deg",     defs["angle_deg"])) * math.pi / 180.0,
            m_kgs          = 0.0,
            correlation    = ed.get("correlation",    defs["correlation"]),
            voidage_method = ed.get("voidage_method", defs["voidage_method"]),
            junction_K_fwd = float(ed.get("junction_K_fwd", defs["junction_K_fwd"])),
            junction_K_rev = float(ed.get("junction_K_rev", defs["junction_K_rev"])),
        )
        net.add_edge(edge)

    return net


def validate_network(net: Network) -> tuple[bool, list[str]]:
    """
    Pre-solve validation.  Returns (ok, messages) where ok=False means at
    least one error was found (solver should not be called).  Warnings are
    included in messages but do not set ok=False.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    node_ids = {n.node_id for n in net.all_nodes()}
    inlets   = [n for n in net.all_nodes() if n.is_inlet]
    outlets  = [n for n in net.all_nodes() if n.is_outlet]

    # Node count
    if len(node_ids) < 2:
        errors.append("Network needs at least 2 nodes.")

    # Boundary conditions
    if len(inlets) == 0:
        errors.append("Mark exactly one node as inlet (→).")
    elif len(inlets) > 1:
        errors.append(f"Found {len(inlets)} inlet nodes — only one is supported.")

    if len(outlets) == 0:
        errors.append("Mark exactly one node as outlet (←).")
    elif len(outlets) > 1:
        errors.append(f"Found {len(outlets)} outlet nodes — only one is supported.")

    # Edge references
    for e in net.all_edges():
        if e.from_node not in node_ids:
            errors.append(f"Edge {e.edge_id!r} from_node {e.from_node!r} not found.")
        if e.to_node not in node_ids:
            errors.append(f"Edge {e.edge_id!r} to_node {e.to_node!r} not found.")
        if e.L_pipe + e.L_fittings == 0.0:
            warnings.append(f"Edge {e.edge_id!r} has zero total length.")

    # Connectivity: can we reach outlet from inlet?
    if inlets and outlets and not errors:
        inlet_id  = inlets[0].node_id
        outlet_id = outlets[0].node_id
        reachable = set(net.bfs_nodes(inlet_id))
        if outlet_id not in reachable:
            errors.append(
                f"Network is disconnected — outlet node {outlet_id!r} "
                f"is not reachable from inlet {inlet_id!r}."
            )

    # Loop check
    if inlets and not errors:
        _, cotree = net.spanning_tree_and_cotree(inlets[0].node_id)
        if not cotree:
            warnings.append(
                "No loops detected — this is a pure tree network. "
                "Flow is uniquely determined; the GGA will converge in one pass."
            )

    messages = [f"ERROR: {e}" for e in errors] + [f"Warning: {w}" for w in warnings]
    return (len(errors) == 0), messages


# ── Template functions ────────────────────────────────────────────────────────

def template_harp(
    N: int,
    harp_type: str = "Z",
    *,
    header_D_m: float = 0.050,
    channel_D_m: float = 0.025,
    header_seg_len: float = 0.15,
    channel_len: float = 0.50,
    channel_angle_deg: float = 0.0,
    roughness_m: float = 4.6e-5,
    P_inlet_bara: float = 10.0,
    T_C: float = 60.0,
    x_gas: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """
    Return (nodes_spec, edges_spec) for a single harp manifold.
    Suitable for passing directly to ``build_network_from_spec``.
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    # Header A nodes: A0 … AN
    for i in range(N + 1):
        nodes.append({
            "id": f"A{i}", "type": "inlet" if i == 0 else "junction",
            "P_bara": P_inlet_bara, "T_C": T_C, "x_gas": x_gas,
            "x": i, "y": 1,     # grid coordinates for diagram
        })
    # Header B nodes: B0 … BN
    for i in range(N + 1):
        outlet_node = (i == N) if harp_type == "Z" else (i == 0)
        nodes.append({
            "id": f"B{i}", "type": "outlet" if outlet_node else "junction",
            "P_bara": P_inlet_bara * 0.98, "T_C": T_C, "x_gas": x_gas,
            "x": i, "y": 0,
        })

    # Header A edges
    for i in range(N):
        edges.append({
            "id": f"hA{i}", "from": f"A{i}", "to": f"A{i+1}",
            "D_inner_m": header_D_m, "roughness_m": roughness_m,
            "L_pipe_m": header_seg_len, "angle_deg": 0.0,
        })

    # Channel edges (with T-junction K-factors)
    for i in range(N):
        edges.append({
            "id": f"ch{i}", "from": f"A{i}", "to": f"B{i}",
            "D_inner_m": channel_D_m, "roughness_m": roughness_m,
            "L_pipe_m": channel_len, "angle_deg": channel_angle_deg,
            "junction_K_fwd": 0.5, "junction_K_rev": 0.3,
        })

    # Header B edges
    for i in range(N):
        if harp_type == "Z":
            b_from, b_to = f"B{i}", f"B{i+1}"
        else:
            b_from, b_to = f"B{i+1}", f"B{i}"
        edges.append({
            "id": f"hB{i}", "from": b_from, "to": b_to,
            "D_inner_m": header_D_m, "roughness_m": roughness_m,
            "L_pipe_m": header_seg_len, "angle_deg": 0.0,
        })

    return nodes, edges


def template_ring(
    N: int,
    *,
    D_m: float = 0.050,
    seg_len: float = 1.0,
    roughness_m: float = 4.6e-5,
    P_inlet_bara: float = 10.0,
    T_C: float = 60.0,
    x_gas: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """
    Return (nodes_spec, edges_spec) for a closed ring of N segments.
    Node 0 is inlet, node 1 is outlet (adjacent to inlet).
    Verifies equal-split distribution for uniform geometry.
    """
    import math as _math
    nodes: list[dict] = []
    edges: list[dict] = []

    angle_step = 2 * _math.pi / N
    for i in range(N):
        a = i * angle_step
        nodes.append({
            "id": f"R{i}",
            "type": "inlet" if i == 0 else ("outlet" if i == 1 else "junction"),
            "P_bara": P_inlet_bara, "T_C": T_C, "x_gas": x_gas,
            "x": round(3 + 2.5 * _math.cos(a), 3),
            "y": round(1.5 + 1.5 * _math.sin(a), 3),
        })
    for i in range(N):
        j = (i + 1) % N
        edges.append({
            "id": f"re{i}", "from": f"R{i}", "to": f"R{j}",
            "D_inner_m": D_m, "roughness_m": roughness_m,
            "L_pipe_m": seg_len, "angle_deg": 0.0,
        })

    return nodes, edges
