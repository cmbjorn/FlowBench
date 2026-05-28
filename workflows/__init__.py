"""
Workflow orchestration layer for FlowBench.

Workflows combine engine calls into complete engineering calculation sequences.
No Streamlit imports — all workflows are callable from CLI, API, or tests.

Modules:
    pipeline_case — multi-segment pipeline hydraulics calculation
    pump_case     — centrifugal / PD pump sizing (operating point, power, NPSH, design pressure)
"""
