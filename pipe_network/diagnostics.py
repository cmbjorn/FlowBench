"""
Post-solve diagnostics: starvation detection and result formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .solver import SolveResult


@dataclass
class ChannelResult:
    channel_edge_id: str
    channel_index:   int       # 0-based position in the harp
    m_kgs:   float             # solved mass flow (kg/s)
    Vsl:     float             # liquid superficial velocity (m/s)
    Vsg:     float             # gas superficial velocity (m/s)
    x_gas:   float             # mass quality at channel inlet node
    regime:  str               # flow-regime string from engine
    starved: bool
    reason:  str               # "" | "low_Vsl" | "high_x" | "low_Vsl+high_x"


@dataclass
class StarvationReport:
    converged:             bool
    n_channels_total:      int
    n_channels_starved:    int
    channel_results:       list[ChannelResult]
    maldistribution_index: float   # (m_max − m_min) / m_mean for channel flows
    mean_m_kgs:            float
    warnings:              list[str] = field(default_factory=list)


def check_all_edges(
    solve_result: "SolveResult",
    Vsl_threshold: float = 0.05,
    x_gas_threshold: float = 0.95,
) -> "StarvationReport":
    """
    Check every edge in the network for low liquid velocity / high quality.
    Useful for general networks where there is no dedicated channel list.
    """
    edge_ids = [e.edge_id for e in solve_result.net.all_edges()]
    return check_starvation(solve_result, edge_ids, Vsl_threshold, x_gas_threshold)


def check_starvation(
    solve_result: SolveResult,
    channel_edge_ids: list[str],
    Vsl_threshold: float = 0.05,
    x_gas_threshold: float = 0.95,
) -> StarvationReport:
    """
    Analyse per-channel results from a completed ``solve_network`` call.

    Parameters
    ----------
    solve_result
        Result returned by ``solve_network``.
    channel_edge_ids
        Ordered list of channel edge IDs (from ``HarpTopology.channel_edge_ids``).
    Vsl_threshold
        Channels with liquid superficial velocity below this value (m/s) are
        flagged as starved.  Default 0.05 m/s.
    x_gas_threshold
        Channels with mass quality above this value are flagged as starved
        (essentially dry gas).  Default 0.95.

    Returns
    -------
    StarvationReport
    """
    net      = solve_result.net
    warnings = list(solve_result.warnings)

    channel_results: list[ChannelResult] = []
    flows: list[float] = []

    for idx, eid in enumerate(channel_edge_ids):
        e = net.edge(eid)
        x_node = net.node(e.from_node).x_gas
        m      = abs(e.m_kgs)
        Vsl    = e.Vsl
        Vsg    = e.Vsg

        reasons: list[str] = []
        if Vsl < Vsl_threshold:
            reasons.append("low_Vsl")
        if x_node > x_gas_threshold:
            reasons.append("high_x")

        starved = bool(reasons)
        reason  = "+".join(reasons)

        channel_results.append(ChannelResult(
            channel_edge_id=eid,
            channel_index=idx,
            m_kgs=m,
            Vsl=Vsl,
            Vsg=Vsg,
            x_gas=x_node,
            regime=e.regime,
            starved=starved,
            reason=reason,
        ))
        flows.append(m)

    n_starved = sum(1 for r in channel_results if r.starved)

    # Maldistribution index: (m_max - m_min) / m_mean
    if flows:
        m_mean = sum(flows) / len(flows)
        mdi = (max(flows) - min(flows)) / max(m_mean, 1e-12)
    else:
        m_mean = 0.0
        mdi = 0.0

    if mdi > 1.0:
        warnings.append(
            f"Maldistribution index {mdi:.2f} > 1.0 — some channels near "
            "starvation or flow reversal."
        )
    elif mdi > 0.5:
        warnings.append(
            f"Maldistribution index {mdi:.2f} > 0.5 — significant flow "
            "non-uniformity detected."
        )

    return StarvationReport(
        converged=solve_result.converged,
        n_channels_total=len(channel_edge_ids),
        n_channels_starved=n_starved,
        channel_results=channel_results,
        maldistribution_index=mdi,
        mean_m_kgs=m_mean,
        warnings=warnings,
    )


def format_report(report: StarvationReport, label: str = "") -> str:
    """Return a compact text summary of a StarvationReport."""
    header = f"{'='*60}\n{label or 'Starvation Report'}\n{'='*60}"
    conv   = "converged" if report.converged else "DID NOT CONVERGE"
    lines  = [
        header,
        f"Solver: {conv}",
        f"Channels: {report.n_channels_starved}/{report.n_channels_total} starved",
        f"MDI:      {report.maldistribution_index:.4f}  "
        f"(mean flow = {report.mean_m_kgs*1000:.2f} g/s)",
        "",
        f"{'Ch':>3}  {'m [g/s]':>9}  {'Vsl [m/s]':>10}  {'Vsg [m/s]':>10}"
        f"  {'x_gas':>6}  {'Regime':<22}  Status",
    ]
    for r in report.channel_results:
        status = f"STARVED ({r.reason})" if r.starved else "ok"
        lines.append(
            f"{r.channel_index:>3}  {r.m_kgs*1000:>9.2f}  {r.Vsl:>10.4f}"
            f"  {r.Vsg:>10.4f}  {r.x_gas:>6.4f}  {r.regime:<22}  {status}"
        )
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  • {w}")
    return "\n".join(lines)
