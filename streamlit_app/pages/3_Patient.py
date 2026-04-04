"""
3_Patient.py — demographics, self-report items, and aggregated EMA
cognitive model for the selected patient.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from components.graph import render_cognitive_graph
from components.sidebar import render_sidebar_selector

st.set_page_config(page_title="Patient Overview", layout="wide")
st.title("Patient Overview")

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

demo        = patient_data.get("demographics", {})
self_report = patient_data.get("self_report", {})
agg_edges   = patient_data.get("agg_edges", {})
patient_id  = patient_data.get("patient_id", "?")

# ── Demographics ───────────────────────────────────────────────────────────

st.subheader(f"Patient {patient_id}")
st.markdown("#### Demographics")

demo_cols = st.columns(6)
demo_fields = [
    ("PCL-5", demo.get("pcl5", "N/A")),
    ("Age", demo.get("age", "")),
    ("Gender", demo.get("gender", "")),
    ("Nationality", demo.get("nationality", "")),
    ("Profession", demo.get("profession", "")),
    ("Trauma Type", demo.get("trauma_type", "")),
]
for col, (label, value) in zip(demo_cols, demo_fields):
    col.metric(label, value)

# ── Self-report items ──────────────────────────────────────────────────────

st.divider()
st.markdown("#### Self-Report Items")

SELF_REPORT_SECTIONS = {
    "Triggers":               "trigger_items",
    "Memory":                 "memory_items_original",
    "Negative Appraisals":    "appraisals_items_original",
    "Threat":                 "threat_items_original",
    "Maladaptive Strategies": "strategies_items_original",
    "Avoidance":              "avoidance_items_original",
}

rows = []
for section_name, key in SELF_REPORT_SECTIONS.items():
    value = self_report.get(key, "")
    items = [v.strip() for v in str(value).split("|") if v.strip()] if value else []
    rows.append({"Component": section_name, "Items": "\n".join(items) if items else "—"})

df_sr = pd.DataFrame(rows)
st.dataframe(df_sr, use_container_width=True, hide_index=True, column_config={
    "Items": st.column_config.TextColumn(width="large"),
})

# ── Aggregated EMA ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Aggregated EMA - Cognitive Model")

graph_col, table_col = st.columns([2, 1])

with graph_col:
    if agg_edges:
        fig = render_cognitive_graph(agg_edges)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aggregated edge data available.")

with table_col:
    if agg_edges:
        st.markdown("**Edge weights**")
        edge_rows = [
            {"Edge": k.replace(" -- ", " → "), "Weight": round(v, 4)}
            for k, v in agg_edges.items()
        ]
        if edge_rows:
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True,
                         height=(len(edge_rows) + 1) * 35 + 3)
        else:
            st.info("All edge weights are zero.")
