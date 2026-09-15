"""
2_Experiments.py — browse all past experiment runs and their patients.
"""

import sys
from pathlib import Path

# Add streamlit_app/ to sys.path
_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from utils.loader import get_experiments, get_patients, load_patient, load_manifest

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Experiments", layout="wide")
st.title("Experiments")

# ── Helper: build patient summary table ───────────────────────────────────

def _build_patient_table(experiment_dir: Path) -> pd.DataFrame:
    """Return a DataFrame summarising each patient in *experiment_dir*."""
    rows = []
    for p in get_patients(experiment_dir):
        try:
            data = load_patient(p["path"])
        except Exception:
            continue

        demo   = data.get("demographics", {})
        val    = data.get("validation_summary", {})
        p_val  = val.get("persona", {})
        m_val  = val.get("message", {})

        rows.append(
            {
                "Patient ID":       data.get("patient_id", p["patient_id"]),
                "PCL-5":            demo.get("pcl5", ""),
                "Gender":           demo.get("gender", ""),
                "Trauma Type":      demo.get("trauma_type", ""),
                "Persona passed":   "Yes" if p_val.get("ultimately_passed") else "No",
                "Persona attempts": p_val.get("attempts", ""),
                "Message passed":   "Yes" if m_val.get("ultimately_passed") else "No",
                "_path":            str(p["path"]),
                "_patient_id":      p["patient_id"],
            }
        )

    return pd.DataFrame(rows)


# ── Main content ───────────────────────────────────────────────────────────

experiments = get_experiments()

if not experiments:
    st.info("No experiments found. Run a simulation first.")
    st.stop()

# Show note if a patient is already selected
selected_pid = st.session_state.get("patient_id")
if selected_pid:
    st.success(
        f"Patient **{selected_pid}** is currently selected — "
        "navigate to the **Patient** page to view details."
    )

st.markdown(f"Found **{len(experiments)}** experiment(s).")

# ── One expander per experiment ────────────────────────────────────────────

for exp in experiments:
    with st.expander(f"{exp['name']}  —  {exp['timestamp']}", expanded=False):

        # Load the run-level manifest (config is stored once per experiment, not per session)
        cfg = load_manifest(exp["path"])
        config_summary = {}
        models = {}
        if cfg:
            config_summary = {
                "Pipeline":    cfg.get("pipeline", ""),
                "Temperature": cfg.get("temperature", ""),
                "Turns":       cfg.get("num_turns", ""),
                "Seed":        cfg.get("seed", ""),
            }
            models = cfg.get("models", {})

        # Config summary columns
        meta_cols = st.columns(len(config_summary) or 1)
        for col, (k, v) in zip(meta_cols, config_summary.items()):
            col.metric(k, v)

        if models:
            with st.expander("Models", expanded=False):
                for role, model in models.items():
                    st.markdown(f"- **{role}**: `{model}`")

        st.markdown("---")

        # Patient table
        df = _build_patient_table(exp["path"])

        if df.empty:
            st.warning("No patient data found.")
            continue

        display_cols = [
            "Patient ID", "PCL-5", "Gender", "Trauma Type",
            "Persona passed", "Persona attempts", "Message passed",
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Select-patient buttons
        st.markdown("**Select a patient to view details:**")
        btn_cols = st.columns(min(len(df), 8))
        for col, (_, row) in zip(btn_cols, df.iterrows()):
            pid   = row["_patient_id"]
            ppath = row["_path"]
            if col.button(f"Patient {pid}", key=f"sel_{exp['name']}_{pid}"):
                patient_data = load_patient(Path(ppath))
                st.session_state["experiment_path"] = str(exp["path"])
                st.session_state["patient_id"]      = pid
                st.session_state["patient_data"]    = patient_data
                st.success(
                    f"Patient **{pid}** selected — navigate to the **Patient** page."
                )
