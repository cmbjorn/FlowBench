"""
Node, Edge, and Network dataclasses plus graph-algorithm helpers.

All pressure values are in Pa; mass flows in kg/s; lengths in m.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Node:
    node_id:   str
    P_pa:      float       # absolute pressure — solver state variable
    x_gas:     float       # mass quality  (0 = all liquid, 1 = all gas)
    T_C:       float       # temperature °C  (isothermal MVP: constant through network)
    is_inlet:  bool = False
    is_outlet: bool = False


@dataclass
class Edge:
    edge_id:        str
    from_node:      str
    to_node:        str
    D_inner:        float          # m — from PIPE_DATABASE[dn][pn]
    roughness:      float          # m — from MATERIAL_ROUGHNESS
    L_pipe:         float          # m — physical pipe length
    L_fittings:     float          # m — equivalent length of fittings
    angle_rad:      float          # 0 = horizontal, +π/2 = vertical up
    m_kgs:          float          # PRIMARY UNKNOWN; +ve = from_node → to_node
    correlation:    str  = "Beggs-Brill"
    voidage_method: str  = "Homogeneous"
    # T-junction minor-loss K-factors (Idelchik §7-23 / Miller §6.3).
    # dP_junc = K * (|m|/A)² / (2·ρ_mix)  added to pipe friction each iteration.
    # The correct K is chosen each iteration based on sign(m_kgs) — §B constraint.
    junction_K_fwd: float = 0.0   # m_kgs >= 0: dividing T (flow into branch)
    junction_K_rev: float = 0.0   # m_kgs <  0: combining T (flow from branch)
    # Quantities written by the solver each iteration:
    dP_Pa:   float = 0.0
    Vsg:     float = 0.0
    Vsl:     float = 0.0
    alpha:   float = 0.0
    regime:  str   = ""

    @property
    def L_eff(self) -> float:
        return self.L_pipe + self.L_fittings


class Network:
    """
    Directed graph of Node and Edge objects representing a pipe network.

    Internally uses dict-of-lists adjacency for O(1) node/edge look-ups.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._out:   dict[str, list[str]] = {}   # node_id → [edge_ids leaving node]
        self._in:    dict[str, list[str]] = {}   # node_id → [edge_ids entering node]
        self._inlet_node_id:  str | None = None
        self._outlet_node_id: str | None = None

    # ── Construction ─────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._out[node.node_id] = []
        self._in[node.node_id]  = []
        if node.is_inlet:
            self._inlet_node_id = node.node_id
        if node.is_outlet:
            self._outlet_node_id = node.node_id

    def add_edge(self, edge: Edge) -> None:
        if edge.edge_id in self._edges:
            raise ValueError(f"Duplicate edge id: {edge.edge_id!r}")
        if edge.from_node not in self._nodes:
            raise KeyError(f"Unknown from_node: {edge.from_node!r}")
        if edge.to_node not in self._nodes:
            raise KeyError(f"Unknown to_node: {edge.to_node!r}")
        self._edges[edge.edge_id] = edge
        self._out[edge.from_node].append(edge.edge_id)
        self._in[edge.to_node].append(edge.edge_id)

    # ── Look-ups ─────────────────────────────────────────────────────────────

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def edge(self, edge_id: str) -> Edge:
        return self._edges[edge_id]

    def all_nodes(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def all_edges(self) -> Iterator[Edge]:
        return iter(self._edges.values())

    def edges_from(self, node_id: str) -> list[Edge]:
        return [self._edges[eid] for eid in self._out[node_id]]

    def edges_to(self, node_id: str) -> list[Edge]:
        return [self._edges[eid] for eid in self._in[node_id]]

    def edges_incident(self, node_id: str) -> list[Edge]:
        return self.edges_from(node_id) + self.edges_to(node_id)

    @property
    def inlet_node_id(self) -> str | None:
        return self._inlet_node_id

    @property
    def outlet_node_id(self) -> str | None:
        return self._outlet_node_id

    # ── Graph algorithms ─────────────────────────────────────────────────────

    def bfs_nodes(self, start_node_id: str) -> list[str]:
        """Return node ids in BFS order from *start_node_id* (undirected BFS)."""
        visited: set[str] = {start_node_id}
        order:   list[str] = [start_node_id]
        queue:   deque[str] = deque([start_node_id])
        while queue:
            nid = queue.popleft()
            for e in self.edges_incident(nid):
                nbr = e.to_node if e.from_node == nid else e.from_node
                if nbr not in visited:
                    visited.add(nbr)
                    order.append(nbr)
                    queue.append(nbr)
        return order

    def spanning_tree_and_cotree(
        self, start_node_id: str
    ) -> tuple[list[str], list[str]]:
        """
        Partition edges into a spanning tree (BFS) and co-tree.

        Returns (tree_edge_ids, cotree_edge_ids).
        Co-tree edges correspond 1-to-1 with independent loops.
        """
        visited:    set[str]  = {start_node_id}
        tree_eids:  list[str] = []
        cotree_eids: list[str] = []
        queue: deque[str] = deque([start_node_id])

        # Collect all edges incident on visited nodes; classify as tree or co-tree.
        while queue:
            nid = queue.popleft()
            for e in self.edges_incident(nid):
                nbr = e.to_node if e.from_node == nid else e.from_node
                if nbr not in visited:
                    visited.add(nbr)
                    tree_eids.append(e.edge_id)
                    queue.append(nbr)
                elif e.edge_id not in tree_eids and e.edge_id not in cotree_eids:
                    cotree_eids.append(e.edge_id)

        return tree_eids, cotree_eids

    def find_fundamental_loops(
        self, start_node_id: str | None = None
    ) -> list[list[tuple[str, int]]]:
        """
        Return one fundamental loop per co-tree edge.

        Each loop is a list of ``(edge_id, orientation)`` tuples where
        ``orientation = +1`` if the traversal direction matches
        ``edge.from_node → edge.to_node``, else ``-1``.

        Algorithm
        ---------
        1. Build a spanning tree T via BFS.
        2. For each co-tree edge (u, v):
           a. Find the unique tree-path from u to v using BFS on T.
           b. The fundamental loop is: co-tree edge (u→v, +1)
              followed by the tree-path from v back to u (reversed, signed).
        """
        if start_node_id is None:
            start_node_id = self._inlet_node_id
            if start_node_id is None:
                raise ValueError("No inlet node set and no start_node_id given.")

        tree_eids, cotree_eids = self.spanning_tree_and_cotree(start_node_id)
        tree_set  = set(tree_eids)

        # Build undirected adjacency restricted to tree edges only.
        # tree_adj[node] = [(neighbour, edge_id, orientation_sign)]
        tree_adj: dict[str, list[tuple[str, str, int]]] = {
            n: [] for n in self._nodes
        }
        for eid in tree_eids:
            e = self._edges[eid]
            tree_adj[e.from_node].append((e.to_node,   eid, +1))
            tree_adj[e.to_node  ].append((e.from_node, eid, -1))

        def tree_path(src: str, dst: str) -> list[tuple[str, int]]:
            """BFS path in spanning tree from src to dst; returns [(edge_id, sign)]."""
            if src == dst:
                return []
            parent: dict[str, tuple[str, str, int] | None] = {src: None}
            q: deque[str] = deque([src])
            while q:
                cur = q.popleft()
                for nbr, eid, sign in tree_adj[cur]:
                    if nbr not in parent:
                        parent[nbr] = (cur, eid, sign)
                        if nbr == dst:
                            # Reconstruct path
                            path: list[tuple[str, int]] = []
                            node = dst
                            while parent[node] is not None:
                                prev, eid_, sgn = parent[node]  # type: ignore[misc]
                                path.append((eid_, sgn))
                                node = prev
                            path.reverse()
                            return path
                        q.append(nbr)
            raise RuntimeError(
                f"No tree path between {src!r} and {dst!r} — graph may be disconnected."
            )

        loops: list[list[tuple[str, int]]] = []
        for ceid in cotree_eids:
            ce = self._edges[ceid]
            # Loop: co-tree edge (from → to, +1) then tree-path (to → from)
            loop: list[tuple[str, int]] = [(ceid, +1)]
            back_path = tree_path(ce.to_node, ce.from_node)
            loop.extend(back_path)
            loops.append(loop)

        return loops

    # ── Merging (used by build_series_harp) ──────────────────────────────────

    def merge(self, other: "Network") -> None:
        """Absorb all nodes and edges from *other* into this network in-place."""
        for n in other._nodes.values():
            self.add_node(n)
        for e in other._edges.values():
            self.add_edge(e)
        # Transfer inlet/outlet only if not already set
        if self._outlet_node_id is None and other._outlet_node_id:
            self._outlet_node_id = other._outlet_node_id


@dataclass
class HarpTopology:
    """Topology descriptor returned by build_harp()."""
    harp_id:           str
    harp_type:         str         # "Z" (opposite-end outlet) or "U" (same-end outlet)
    inlet_node_id:     str
    outlet_node_id:    str
    header_A_node_ids: list[str]   # A_0 … A_N ordered
    header_B_node_ids: list[str]   # B_0 … B_N ordered
    channel_edge_ids:  list[str]   # ch_0 … ch_{N-1} ordered
    header_A_edge_ids: list[str]   # hA_0 … hA_{N-1}
    header_B_edge_ids: list[str]   # hB_0 … hB_{N-1}
