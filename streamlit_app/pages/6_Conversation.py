"""
5_Conversation.py — replay the full client–bot dialogue for a patient.
"""

import sys
from pathlib import Path

# Add streamlit_app/ to sys.path
_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st

from components.chat import render_conversation
from components.sidebar import render_sidebar_selector

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Conversation", layout="wide")
st.title("Conversation")

# ── Sidebar selector ───────────────────────────────────────────────────────

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

# ── Sidebar extras ─────────────────────────────────────────────────────────

conversation     = patient_data.get("conversation", [])
moderation_log   = patient_data.get("moderation_log", [])
obs_nodes        = patient_data.get("obs_nodes", {})
patient_id       = patient_data.get("patient_id", "?")

# Count total turns and active nodes
num_turns    = len(conversation)
active_nodes = sum(1 for v in obs_nodes.values() if v == 1)

st.sidebar.divider()
st.sidebar.markdown("**Conversation stats**")
st.sidebar.metric("Total turns",   num_turns)
st.sidebar.metric("Active nodes",  active_nodes)

# ── Header info ────────────────────────────────────────────────────────────

st.subheader(f"Patient {patient_id}")

obs_datetime = patient_data.get("obs_datetime", "")

if obs_datetime:
    st.markdown(f"**Observation datetime:** {obs_datetime}")

st.divider()

# ── Conversation replay ────────────────────────────────────────────────────

render_conversation(conversation, moderation_log=moderation_log)
