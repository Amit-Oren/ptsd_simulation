"""
4_Vignette.py — persona vignette and persona validation summary
(attempts, feedback, pass/fail) for the selected patient.
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

st.set_page_config(page_title="Vignette", layout="wide")
st.title("Persona Vignette")

st.markdown("""
<style>
    .block-container { max-width: 95% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
</style>
""", unsafe_allow_html=True)

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

persona           = patient_data.get("persona", "")
val_summary       = patient_data.get("validation_summary", {})
persona_attempts  = patient_data.get("persona_attempts", [])
patient_id        = patient_data.get("patient_id", "?")
vignette_analysis = patient_data.get("vignette_analysis", {})

st.subheader(f"Patient {patient_id}")

# ── Persona Vignette ───────────────────────────────────────────────────────

if persona:
    st.markdown(
        f"""
        <div style="
            background-color:#f0f4f8;
            border-left:4px solid #3498db;
            padding:16px 20px;
            border-radius:6px;
            font-size:16px;
            line-height:1.6;
        ">{persona}</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No persona vignette available.")

# ── Validation Summary ─────────────────────────────────────────────────────

st.divider()
st.subheader("Validation Summary")

persona_val  = val_summary.get("persona", {})
passed_flag  = persona_val.get("ultimately_passed", None)
attempts     = persona_val.get("attempts", "N/A")
passed_count = persona_val.get("passed", "N/A")
failed_count = persona_val.get("failed", "N/A")

if passed_flag is True:
    status_label, status_color = "Passed", "#2ecc71"
elif passed_flag is False:
    status_label, status_color = "Failed", "#e74c3c"
else:
    status_label, status_color = "N/A", "#bdc3c7"

summary_col, metrics_col = st.columns([1, 3])

with summary_col:
    st.markdown(
        f"""
        <div style="
            border:2px solid {status_color};
            border-radius:8px;
            padding:16px;
            text-align:center;
        ">
            <span style="font-size:22px;font-weight:bold;color:{status_color};">
                {status_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metrics_col:
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Attempts", attempts)
    m2.metric("Passed",         passed_count)
    m3.metric("Failed",         failed_count)

# ── Attempt Details ────────────────────────────────────────────────────────

st.divider()
st.subheader("Attempt Details")

if persona_attempts:
    for i, att in enumerate(persona_attempts, start=1):
        passed   = att.get("passed", False)
        feedback = att.get("feedback") or ""
        output   = att.get("output", "")
        icon     = "✅" if passed else "❌"
        label    = "PASS" if passed else "FAIL"
        color    = "#2ecc71" if passed else "#e74c3c"

        with st.expander(f"{icon} Attempt {i} — {label}", expanded=(not passed)):
            if output:
                st.markdown("**Vignette:**")
                st.markdown(
                    f"""<div style="
                        background:#f9f9f9;
                        border-left:3px solid {color};
                        padding:14px 20px;
                        font-size:15px;
                        line-height:1.7;
                        border-radius:4px;
                        margin-bottom:12px;
                        width:100%;
                        box-sizing:border-box;
                    ">{output}</div>""",
                    unsafe_allow_html=True,
                )
            if feedback:
                st.markdown("**Validator feedback:**")
                st.markdown(
                    f"""<div style="
                        background:#fef9e7;
                        border-left:3px solid #f39c12;
                        padding:14px 20px;
                        font-size:14px;
                        line-height:1.6;
                        border-radius:4px;
                        width:100%;
                        box-sizing:border-box;
                        margin-bottom:24px;
                    ">{feedback}</div>""",
                    unsafe_allow_html=True,
                )
else:
    st.info("No validation attempts recorded.")

# ── Vignette Analysis ──────────────────────────────────────────────────────

st.divider()
st.subheader("Vignette Analysis")

if vignette_analysis:
    agg_edges = patient_data.get("agg_edges", {})

    # Normalise agg_edges keys to "A--B" format
    def _norm(k):
        return k.replace(" -- ", "--").strip()

    agg_norm = {_norm(k): v for k, v in agg_edges.items()}

    analyst_weights = {k: v.get("weight", 0.0) for k, v in vignette_analysis.items()}

    # ── Side-by-side graphs ────────────────────────────────────────────────
    st.markdown("#### Cognitive Model Graphs")
    graph_left, graph_right = st.columns(2)

    with graph_left:
        st.markdown("**Analyst Detected**")
        st.plotly_chart(render_cognitive_graph(analyst_weights), use_container_width=True)

    with graph_right:
        st.markdown("**Ground Truth (Aggregated EMA)**")
        st.plotly_chart(render_cognitive_graph(agg_norm), use_container_width=True)

    # ── Comparison table ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Edge Comparison")

    ALL_EDGES = list(vignette_analysis.keys())
    rows = []
    for ek in ALL_EDGES:
        detected = round(analyst_weights.get(ek, 0.0), 3)
        gt       = round(agg_norm.get(ek, 0.0), 4)
        det_bin  = 1 if detected >= 0.5 else 0
        gt_bin   = 1 if gt > 0 else 0
        match    = "✅" if det_bin == gt_bin else "❌"
        edata    = vignette_analysis.get(ek, {})
        rows.append({
            "Edge":        ek.replace("--", " → "),
            "Detected":    detected,
            "Ground Truth": gt,
            "Match":       match,
            "Explanation": edata.get("explanation", ""),
            "Quote":       edata.get("quote", ""),
        })

    n = len(rows)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=(n + 1) * 35 + 3)
else:
    st.info("No vignette analysis available. Run a pipeline that includes 'analyse_vignette'.")
