"""
8_Memory.py — browse a patient's persistent memory knowledge graph.

Memory lives in data/memory/, independent of any single experiment run
(see simulation/memory/store.py) — so this page has its own patient
selector rather than using the experiment/session selector the other
pages share.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from components.memory_graph import render_memory_graph
from utils.loader import (
    get_memory_patients, load_memory_graph,
    get_memory_snapshots, load_memory_snapshot,
)

st.set_page_config(page_title="Memory", layout="wide")
st.title("Memory")
st.caption(
    "The interviewer's cross-session knowledge graph for one patient — built from "
    "specific things mentioned across sessions, not the coarse cognitive-model scores."
)

# ── Patient selector ─────────────────────────────────────────────────────────

patients = get_memory_patients()
if not patients:
    st.info("No patient memory found yet. Run a simulation with the `full_memory` pipeline first.")
    st.stop()

label_by_id = {p["memory_id"]: p["label"] for p in patients}
memory_id = st.sidebar.selectbox(
    "Patient",
    options=[p["memory_id"] for p in patients],
    format_func=lambda mid: label_by_id.get(mid, mid),
    key="memory_patient_select",
)

# ── Snapshot selector ─────────────────────────────────────────────────────────

snapshots = get_memory_snapshots(memory_id)
snapshot_options = ["Current"] + [f"As of session {s['session_idx']}" for s in snapshots]

selected_snapshot = st.sidebar.selectbox(
    "Point in time",
    options=snapshot_options,
    index=0,
    key="memory_snapshot_select",
)

if selected_snapshot == "Current":
    graph = load_memory_graph(memory_id)
else:
    session_idx = int(selected_snapshot.rsplit(" ", 1)[-1])
    snap = next(s for s in snapshots if s["session_idx"] == session_idx)
    graph = load_memory_snapshot(snap["path"])

nodes = graph.get("nodes", [])
edges = graph.get("edges", [])

# ── Summary + graph ────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric("Nodes", len(nodes))
col2.metric("Edges", len(edges))
col3.metric("Sessions covered", len({s for n in nodes for s in n.get("sessions_seen", [])}))

st.subheader("Graph")
if nodes:
    fig = render_memory_graph(graph)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("This patient's memory graph is empty at this point in time.")

# ── Nodes table, grouped by type ────────────────────────────────────────────────

st.divider()
st.subheader("Nodes")

NODE_TYPE_ORDER = [
    "Triggers", "Negative Appraisals", "Memory", "Threat", "Maladaptive Strategies",
]

if nodes:
    node_rows = []
    for n in sorted(nodes, key=lambda n: NODE_TYPE_ORDER.index(n["type"]) if n["type"] in NODE_TYPE_ORDER else 99):
        node_rows.append({
            "Type":     n["type"],
            "Label":    n["label"],
            "Sessions": ", ".join(str(s) for s in n.get("sessions_seen", [])),
            "Quotes":   " / ".join(n.get("quotes", [])),
            "id":       n["id"],
        })
    n_rows = len(node_rows)
    st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True,
                 height=min((n_rows + 1) * 35 + 3, 500))
else:
    st.info("No nodes yet.")

# ── Edges table ──────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Relationships")

if edges:
    id_to_label = {n["id"]: n["label"] for n in nodes}
    edge_rows = []
    for e in edges:
        edge_rows.append({
            "Source":   id_to_label.get(e["source"], e["source"]),
            "Relation": e["relation"],
            "Target":   id_to_label.get(e["target"], e["target"]),
            "Weight":   round(e.get("weight", 0), 2),
            "Sessions": ", ".join(str(s) for s in e.get("sessions_seen", [])),
        })
    n_rows = len(edge_rows)
    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True,
                 height=min((n_rows + 1) * 35 + 3, 500))
else:
    st.info("No relationships yet.")
