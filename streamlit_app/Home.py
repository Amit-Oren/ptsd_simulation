"""
Home.py — landing page for the PTSD Simulation Streamlit app.
"""

import sys
from pathlib import Path

# Add streamlit_app/ directory to sys.path so local modules are importable.
_STREAMLIT_APP_DIR = Path(__file__).parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st

from utils.loader import get_experiments, get_patients

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PTSD Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ─────────────────────────────────────────────────────────────────

st.title("PTSD Simulation")
st.markdown(
    """
    This dashboard explores **AI-driven simulations of PTSD patient conversations**.

    A language-model pipeline crafts a patient persona from demographic and
    self-report data, conducts a multi-turn therapy dialogue, and then analyses
    the conversation against a cognitive model of PTSD constructs (Triggers,
    Memory, Negative Appraisals, Threat, and Maladaptive Strategies).
    """
)

st.divider()

# ── Summary stats ──────────────────────────────────────────────────────────

experiments = get_experiments()
total_patients = sum(len(get_patients(e["path"])) for e in experiments)

col1, col2 = st.columns(2)
with col1:
    st.metric("Experiments", len(experiments))
with col2:
    st.metric("Total patients simulated", total_patients)

st.divider()

# ── Navigation guide ───────────────────────────────────────────────────────

st.subheader("Navigation")

st.markdown(
    """
    Use the sidebar to move between pages:

    | Page | Description |
    |------|-------------|
    | **Run Simulation** | Configure and launch a new simulation run. Monitor progress live. |
    | **Experiments** | Browse all past experiment runs and their patients. |
    | **Patient** | View demographics, PCL-5 score, persona vignette and validation results for a selected patient. |
    | **Cognitive Model** | Visualise the weighted PTSD network graph for a patient. |
    | **Conversation** | Replay the full client–bot dialogue. |
    | **Analysis** | Compare the model's detected nodes/edges against ground-truth observation labels. |
    """
)

# ── Recent experiments quick-view ─────────────────────────────────────────

if experiments:
    st.divider()
    st.subheader("Recent experiments")
    for exp in experiments[:5]:
        patients = get_patients(exp["path"])
        st.markdown(f"- **{exp['name']}** — {len(patients)} patient(s)")
else:
    st.info("No experiments found yet. Go to **Run Simulation** to create one.")
