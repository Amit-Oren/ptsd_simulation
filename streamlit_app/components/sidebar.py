"""
sidebar.py — shared sidebar widget for selecting an experiment and patient.

Import and call render_sidebar_selector() at the top of any page that needs
a patient context.
"""

import sys
from pathlib import Path

# Ensure streamlit_app/ is importable
_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st

from utils.loader import get_experiments, get_patients, load_patient


def render_sidebar_selector() -> tuple:
    """
    Render experiment and patient selectors in the sidebar.

    Returns
    -------
    (experiment_dir, patient_data)
        experiment_dir  Path | None
        patient_data    dict | None

    Both are None when no valid selection exists.
    """
    experiments = get_experiments()

    if not experiments:
        st.sidebar.warning("No experiments found. Run a simulation first.")
        return None, None

    # ── Experiment selector ────────────────────────────────────────────────
    exp_names = [e["name"] for e in experiments]

    # Pre-select if session state carries a remembered path
    default_exp_index = 0
    remembered_path = st.session_state.get("experiment_path")
    if remembered_path:
        try:
            default_exp_index = exp_names.index(Path(remembered_path).name)
        except ValueError:
            pass

    selected_exp_name = st.sidebar.selectbox(
        "Experiment",
        options=exp_names,
        index=default_exp_index,
        key="sidebar_experiment_select",
    )
    experiment_dir = next(e["path"] for e in experiments if e["name"] == selected_exp_name)

    # ── Patient selector ───────────────────────────────────────────────────
    patients = get_patients(experiment_dir)

    if not patients:
        st.sidebar.warning("No patients found in this experiment.")
        return experiment_dir, None

    patient_ids = [p["patient_id"] for p in patients]

    default_pat_index = 0
    remembered_pid = st.session_state.get("patient_id")
    if remembered_pid and str(remembered_pid) in patient_ids:
        default_pat_index = patient_ids.index(str(remembered_pid))

    selected_patient_id = st.sidebar.selectbox(
        "Patient",
        options=patient_ids,
        index=default_pat_index,
        format_func=lambda pid: f"Patient {pid}",
        key="sidebar_patient_select",
    )

    patient_path = next(p["path"] for p in patients if p["patient_id"] == selected_patient_id)
    patient_data = load_patient(patient_path)

    # Persist selection in session state so other pages can see it
    st.session_state["experiment_path"] = str(experiment_dir)
    st.session_state["patient_id"] = selected_patient_id
    st.session_state["patient_data"] = patient_data

    return experiment_dir, patient_data
