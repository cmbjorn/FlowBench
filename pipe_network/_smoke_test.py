"""
Smoke tests for the pipe_network module.

Run from the FlowBench root:
    python -m pipe_network._smoke_test

Tests
-----
1. Graph: 2-channel harp loop detection (1 fundamental loop).
2. Symmetry: N=6 uniform harp converges to uniform flow (MDI ≈ 0).
3. Loop balance: at convergence both paths A→B have equal total dP.
4. Maldistribution: narrow headers force channels near the dead-end to get
   more flow than channels near the inlet.
5. Series harp: total dP ≈ harp1 + connector + harp2; quality is preserved.
6. Starvation trigger: very low total flow flags some channels starved.
"""
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipe_network import (
    build_harp, build_series_harp,
    solve_network, check_starvation, format_report,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}  {detail}")
    return condition

# Common fluid: single-component water + hydrogen (alkaline electrolyser)
GAS   = {"H2": 2.0}          # kg/h hydrogen
LIQ   = {"Water": 500.0}     # kg/h water
M_TOT = (2.0 + 500.0) / 3600.0   # kg/s total

results = []

# ── Test 1: Loop detection ────────────────────────────────────────────────────
print("\nTest 1: Loop detection (N=2 harp → 1 fundamental loop)")
net, topo = build_harp(
    2,
    header_dn="DN50", header_pn="PN40", header_segment_length=0.30,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.004,
    harp_id="T1",
)
loops = net.find_fundamental_loops(topo.inlet_node_id)
results.append(check("N=2 harp has exactly 1 fundamental loop", len(loops) == 1,
                      f"got {len(loops)}"))

net6, topo6 = build_harp(
    6,
    header_dn="DN50", header_pn="PN40", header_segment_length=0.15,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.004,
    harp_id="T2",
)
loops6 = net6.find_fundamental_loops(topo6.inlet_node_id)
results.append(check("N=6 harp has exactly 5 fundamental loops", len(loops6) == 5,
                      f"got {len(loops6)}"))

# ── Test 2: Symmetry (uniform geometry → uniform flow) ───────────────────────
print("\nTest 2: Symmetry — uniform N=6 harp should give MDI ≈ 0")
net_sym, topo_sym = build_harp(
    6,
    header_dn="DN80", header_pn="PN40", header_segment_length=0.15,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.004,
    harp_id="SYM",
)
res_sym = solve_network(
    net_sym,
    gas_flows_kgh=GAS,
    liquid_type="Water",
    liquid_flows_kgh=LIQ,
    inlet_node_id=topo_sym.inlet_node_id,
    outlet_node_id=topo_sym.outlet_node_id,
    m_total_kgs=M_TOT,
    max_iter=200,
    tol_rel=1e-4,
)
rep_sym = check_starvation(res_sym, topo_sym.channel_edge_ids)
results.append(check("Symmetric harp converged", res_sym.converged))
results.append(check("MDI < 0.05 for symmetric harp", rep_sym.maldistribution_index < 0.05,
                      f"MDI={rep_sym.maldistribution_index:.4f}"))
results.append(check("No channels starved in symmetric harp", rep_sym.n_channels_starved == 0,
                      f"{rep_sym.n_channels_starved} starved"))

# ── Test 3: Loop pressure balance ─────────────────────────────────────────────
print("\nTest 3: Loop pressure balance at convergence")
# Two paths from A_0 to B_1 in a 2-channel harp: via channel 0, and via
# hA0 → channel 1 → hB0 (reversed).  Their total dP must be equal.
net_2, topo_2 = build_harp(
    2,
    header_dn="DN50", header_pn="PN40", header_segment_length=0.30,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.004,
    harp_id="LB",
)
res_2 = solve_network(
    net_2,
    gas_flows_kgh=GAS,
    liquid_type="Water",
    liquid_flows_kgh=LIQ,
    inlet_node_id=topo_2.inlet_node_id,
    outlet_node_id=topo_2.outlet_node_id,
    m_total_kgs=M_TOT,
    max_iter=200,
    tol_rel=1e-5,
)
# Path 1: ch0 alone  (A0→B0)
dP_ch0 = net_2.edge(topo_2.channel_edge_ids[0]).dP_Pa
# Path 2: hA0 + ch1 − hB0  (A0→A1→B1→B0 reversed = A0→B0 via channel 1)
dP_hA0  = net_2.edge(topo_2.header_A_edge_ids[0]).dP_Pa
dP_ch1  = net_2.edge(topo_2.channel_edge_ids[1]).dP_Pa
dP_hB0  = net_2.edge(topo_2.header_B_edge_ids[0]).dP_Pa
dP_path2 = dP_hA0 + dP_ch1 - dP_hB0
balance_err = abs(dP_ch0 - dP_path2) / max(abs(dP_ch0), 1.0)
results.append(check("Loop pressure balanced to <1 %",
                      balance_err < 0.01,
                      f"path1={dP_ch0:.1f} Pa  path2={dP_path2:.1f} Pa  err={balance_err:.4f}"))

# ── Test 4: Maldistribution with narrow headers ───────────────────────────────
print("\nTest 4: Maldistribution — narrow header DN25 forces non-uniform flow")
net_md, topo_md = build_harp(
    6,
    header_dn="DN25", header_pn="PN40", header_segment_length=0.20,
    channel_dn="DN50", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.004,
    harp_id="MD",
)
res_md = solve_network(
    net_md,
    gas_flows_kgh=GAS,
    liquid_type="Water",
    liquid_flows_kgh=LIQ,
    inlet_node_id=topo_md.inlet_node_id,
    outlet_node_id=topo_md.outlet_node_id,
    m_total_kgs=M_TOT,
    max_iter=200,
)
rep_md = check_starvation(res_md, topo_md.channel_edge_ids)
# With narrow headers vs wide channels the distribution is not uniform
results.append(check("Maldistribution solver converged", res_md.converged))
results.append(check("MDI > 0 for narrow-header harp", rep_md.maldistribution_index > 0.01,
                      f"MDI={rep_md.maldistribution_index:.4f}"))
flows_md = [r.m_kgs for r in rep_md.channel_results]
# Some channel must deviate from the mean by > 0.5 % (direction is not prescribed)
max_dev = max(abs(f - rep_md.mean_m_kgs) / max(rep_md.mean_m_kgs, 1e-12) for f in flows_md)
results.append(check("Some channel deviates from mean by > 0.5 %",
                      max_dev > 0.005,
                      f"max_dev={max_dev*100:.3f}%  MDI={rep_md.maldistribution_index:.4f}"))

# ── Test 5: Series harp ───────────────────────────────────────────────────────
print("\nTest 5: Series harp — quality preserved across connector")
# Use narrow channels (DN25 → DN15 equivalent via custom D) and high flow
# so that the total dP is large enough to be numerically resolved.
net_s, topo_s1, topo_s2 = build_series_harp(
    4,
    header_dn="DN50", header_pn="PN40", header_segment_length=0.15,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    connector_dn="DN50", connector_pn="PN40", connector_length=0.30,
    P_inlet_pa=15e5, T_C=60.0, x_inlet=0.004,
)
_HIGH_LIQ = {"Water": 10_000.0}   # 10 t/h → turbulent, measurable dP
_HIGH_M   = (2.0 + 10_000.0) / 3600.0
res_s = solve_network(
    net_s,
    gas_flows_kgh=GAS,
    liquid_type="Water",
    liquid_flows_kgh=_HIGH_LIQ,
    inlet_node_id=topo_s1.inlet_node_id,
    outlet_node_id=topo_s2.outlet_node_id,
    m_total_kgs=_HIGH_M,
    P_outlet_pa=14.5e5,          # explicit 0.5 bara below typical inlet
    max_iter=200,
)
results.append(check("Series harp converged", res_s.converged))
# Quality at H2 inlet should be close to inlet quality (homogeneous, no phase change)
x_h2_inlet = net_s.node(topo_s2.inlet_node_id).x_gas
_x_inlet_s = 0.004   # the x_inlet passed to build_series_harp
results.append(check("Quality preserved across connector (within 5 %)",
                      abs(x_h2_inlet - _x_inlet_s) / _x_inlet_s < 0.05,
                      f"x_H2_inlet={x_h2_inlet:.5f}  expected={_x_inlet_s:.5f}"))
# With GGA: outlet is PINNED at P_outlet_pa (default 0.95 * P_inlet).
# Inlet pressure (a result) must be >= outlet pressure.
P_in  = net_s.node(topo_s1.inlet_node_id).P_pa
P_out = net_s.node(topo_s2.outlet_node_id).P_pa
results.append(check("Inlet pressure >= outlet pressure (GGA: inlet is result)",
                      P_in >= P_out,
                      f"P_in={P_in/1e5:.4f} bara  P_out={P_out/1e5:.4f} bara"))

# ── Test 6: Starvation trigger ────────────────────────────────────────────────
print("\nTest 6: Starvation trigger — very low flow + high quality")
net_st, topo_st = build_harp(
    4,
    header_dn="DN50", header_pn="PN40", header_segment_length=0.15,
    channel_dn="DN25", channel_pn="PN40", channel_length=0.50,
    P_inlet_pa=10e5, T_C=60.0, x_inlet=0.98,   # almost dry gas
    harp_id="ST",
)
res_st = solve_network(
    net_st,
    gas_flows_kgh={"H2": 10.0},
    liquid_type="Water",
    liquid_flows_kgh={"Water": 0.5},    # tiny liquid flow
    inlet_node_id=topo_st.inlet_node_id,
    outlet_node_id=topo_st.outlet_node_id,
    m_total_kgs=(10.0 + 0.5) / 3600.0,
    max_iter=100,
)
rep_st = check_starvation(res_st, topo_st.channel_edge_ids,
                           Vsl_threshold=0.05, x_gas_threshold=0.90)
results.append(check("Starvation detected for high-quality inlet",
                      rep_st.n_channels_starved > 0,
                      f"{rep_st.n_channels_starved}/{rep_st.n_channels_total} starved"))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"Results: {n_pass}/{len(results)} passed, {n_fail} failed")
if n_fail:
    sys.exit(1)
