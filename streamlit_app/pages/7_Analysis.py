"""
6_Analysis.py — compare detected nodes/edges from the analyst against
ground-truth observation labels, with side-by-side graphs and metrics.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
_REPO_ROOT = _STREAMLIT_APP_DIR.parent
for _p in [str(_STREAMLIT_APP_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components.graph import render_cognitive_graph
from components.sidebar import render_sidebar_selector

# All possible directed edges in the model
from config.formulation_config import DIRECT_EDGES
ALL_EDGES = [f"{p}--{c}" for (p, c) in DIRECT_EDGES.keys()]

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis")

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────

NODE_ORDER = [
    "Triggers",
    "Memory",
    "Negative Appraisals",
    "Threat",
    "Maladaptive Strategies",
]

# ── Data extraction ────────────────────────────────────────────────────────

analysis   = patient_data.get("analysis", {})
obs_nodes  = patient_data.get("obs_nodes", {})
obs_edges  = patient_data.get("obs_edges", {})
patient_id = patient_data.get("patient_id", "?")

analysis_nodes = analysis.get("nodes", {}) if isinstance(analysis, dict) else {}
analysis_edges = analysis.get("edges", {}) if isinstance(analysis, dict) else {}

st.subheader(f"Patient {patient_id} - Analyst Output vs Ground Truth")

if not analysis:
    st.info("No analysis data available for this patient.")
    st.stop()


def _norm(key: str) -> str:
    """Normalise edge key to 'NodeA--NodeB' (no spaces around dashes)."""
    return key.replace(" -- ", "--").strip()


def _display(key: str) -> str:
    return key.replace("--", " — ")


obs_edges_norm = {_norm(k): int(v) for k, v in obs_edges.items()}

# ── Metrics computation ────────────────────────────────────────────────────

# Nodes
tp_nodes = fp_nodes = fn_nodes = tn_nodes = 0
for node in NODE_ORDER:
    det = analysis_nodes.get(node, {}).get("score", 0)
    gt  = obs_nodes.get(node, 0)
    if det == 1 and gt == 1: tp_nodes += 1
    elif det == 1 and gt == 0: fp_nodes += 1
    elif det == 0 and gt == 1: fn_nodes += 1
    else: tn_nodes += 1

total_nodes   = len(NODE_ORDER)
node_accuracy  = (tp_nodes + tn_nodes) / total_nodes if total_nodes else 0
node_precision = tp_nodes / (tp_nodes + fp_nodes) if (tp_nodes + fp_nodes) else 0
node_recall    = tp_nodes / (tp_nodes + fn_nodes) if (tp_nodes + fn_nodes) else 0

# Edges — iterate over all 18 model edges
tp_edges = fp_edges = fn_edges = tn_edges = 0
for ek in ALL_EDGES:
    det_score = 0
    for ak, av in analysis_edges.items():
        if _norm(ak) == ek:
            det_score = av.get("score", 0) if isinstance(av, dict) else 0
            break
    gt_score = obs_edges_norm.get(ek, 0)
    if det_score == 1 and gt_score == 1: tp_edges += 1
    elif det_score == 1 and gt_score == 0: fp_edges += 1
    elif det_score == 0 and gt_score == 1: fn_edges += 1
    else: tn_edges += 1

total_edges    = len(ALL_EDGES)
edge_accuracy  = (tp_edges + tn_edges) / total_edges if total_edges else 0
edge_precision = tp_edges / (tp_edges + fp_edges) if (tp_edges + fp_edges) else 0
edge_recall    = tp_edges / (tp_edges + fn_edges) if (tp_edges + fn_edges) else 0

# ── Metrics display ────────────────────────────────────────────────────────

st.markdown("#### Metrics")
m_node_col, m_edge_col = st.columns(2)

with m_node_col:
    st.markdown(
        "<div style='background:#EDF7BD;border-left:4px solid #85C79A;padding:6px 12px;border-radius:4px;margin-bottom:10px;'>"
        "<span style='font-weight:bold;font-size:15px;color:#281C59;'>Nodes</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{node_accuracy:.0%}")
    c2.metric("Precision", f"{node_precision:.0%}")
    c3.metric("Recall",    f"{node_recall:.0%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", tp_nodes)
    c2.metric("FP", fp_nodes)
    c3.metric("FN", fn_nodes)
    c4.metric("TN", tn_nodes)

with m_edge_col:
    st.markdown(
        "<div style='background:#e6f2f5;border-left:4px solid #4E8D9C;padding:6px 12px;border-radius:4px;margin-bottom:10px;'>"
        "<span style='font-weight:bold;font-size:15px;color:#281C59;'>Edges</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{edge_accuracy:.0%}")
    c2.metric("Precision", f"{edge_precision:.0%}")
    c3.metric("Recall",    f"{edge_recall:.0%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", tp_edges)
    c2.metric("FP", fp_edges)
    c3.metric("FN", fn_edges)
    c4.metric("TN", tn_edges)

# ── Side-by-side graphs ────────────────────────────────────────────────────

st.divider()
st.markdown("#### Cognitive Model Graphs")

analyst_nodes_binary = {
    node: data.get("score", 0)
    for node, data in analysis_nodes.items()
}
analyst_edges_binary = {}
for ak, av in analysis_edges.items():
    score = av.get("score", 0) if isinstance(av, dict) else 0
    analyst_edges_binary[ak] = float(score)

gt_edges_binary = {k: float(v) for k, v in obs_edges.items()}

graph_left, graph_right = st.columns(2)

with graph_left:
    st.markdown("**Analyst Detected**")
    fig_analyst = render_cognitive_graph(analyst_edges_binary, analyst_nodes_binary)
    st.plotly_chart(fig_analyst, use_container_width=True)

with graph_right:
    st.markdown("**Ground Truth**")
    fig_gt = render_cognitive_graph(gt_edges_binary, obs_nodes)
    st.plotly_chart(fig_gt, use_container_width=True)

# ── Nodes section ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Nodes")

if analysis_nodes:
    node_rows = []
    node_list = [n for n in NODE_ORDER if n in analysis_nodes] or list(analysis_nodes.keys())
    for node_name in node_list:
        node_data = analysis_nodes.get(node_name, {})
        detected  = node_data.get("score", 0)
        quote     = node_data.get("quote", "")
        gt_value  = obs_nodes.get(node_name, None)

        if gt_value is None:
            match_str = "N/A"
        elif detected == gt_value:
            match_str = "✅"
        else:
            match_str = "❌"

        explanation = node_data.get("explanation", "")
        node_rows.append({
            "Node":         node_name,
            "Detected":     detected,
            "Ground Truth": gt_value if gt_value is not None else "",
            "Match":        match_str,
            "Explanation":  explanation,
            "Quote":        quote,
        })

    n_rows = len(node_rows)
    st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True,
                 height=(n_rows + 1) * 35 + 3)
else:
    st.info("No node analysis data available.")

# ── Edges section ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Edges")

edge_rows = []
for nk in ALL_EDGES:
    edata          = analysis_edges.get(nk, analysis_edges.get(nk.replace("--", " -- "), {}))
    detected_score = edata.get("score", 0) if isinstance(edata, dict) else 0
    quote          = edata.get("quote", "") if isinstance(edata, dict) else ""
    explanation    = edata.get("explanation", "") if isinstance(edata, dict) else ""
    gt_val         = obs_edges_norm.get(nk, None)

    if gt_val is None:
        match_str = "N/A"
    elif detected_score == gt_val:
        match_str = "✅"
    else:
        match_str = "❌"

    edge_rows.append({
        "Edge":         _display(nk),
        "Detected":     detected_score,
        "Ground Truth": gt_val if gt_val is not None else "",
        "Match":        match_str,
        "Explanation":  explanation,
        "Quote":        (quote[:120] + "…") if len(quote) > 120 else quote,
    })

if edge_rows:
    n_rows = len(edge_rows)
    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True,
                 height=(n_rows + 1) * 35 + 3)
else:
    st.info("No edge data available.")
