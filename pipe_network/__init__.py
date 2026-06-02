"""
pipe_network — generic two-phase pipe-network solver.

Public API
----------
build_harp(N, ...)            Build a single harp manifold.
build_series_harp(N, ...)     Build two harps connected in series.
solve_network(net, ...)       Hardy-Cross solver.
check_starvation(result, ...) Post-solve liquid starvation diagnostics.
format_report(report, ...)    Text summary of a StarvationReport.

Data types
----------
Node, Edge, Network, HarpTopology
SolveResult, StarvationReport, ChannelResult
"""

from .diagnostics import (
    ChannelResult, StarvationReport,
    check_starvation, check_all_edges, format_report,
)
from .graph import Edge, HarpTopology, Network, Node
from .solver import SolveResult, solve_network
from .topology import (
    build_harp, build_series_harp,
    build_center_fed_harp, build_biinlet_harp,
    build_network_from_spec, validate_network,
    template_harp, template_ring,
)

__all__ = [
    # graph
    "Node", "Edge", "Network", "HarpTopology",
    # topology — harp builders
    "build_harp", "build_series_harp",
    # topology — generic
    "build_network_from_spec", "validate_network",
    "template_harp", "template_ring",
    # solver
    "solve_network", "SolveResult",
    # diagnostics
    "check_starvation", "check_all_edges",
    "StarvationReport", "ChannelResult", "format_report",
]
