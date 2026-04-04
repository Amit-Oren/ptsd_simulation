"""
1_Run_Simulation.py — configure and launch a PTSD simulation run.

Left column: configuration form.
Right column: live pipeline diagram + streaming log output.
"""

import sys
from pathlib import Path

# Add streamlit_app/ to sys.path for local imports
_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

# Add project root to sys.path so we can import simulation.pipelines
_PROJECT_ROOT = _STREAMLIT_APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from utils.loader import load_config
from utils.runner import find_experiment_dir, start_simulation, tail_log
from components.pipeline import parse_step_status, render_pipeline

# Import pipeline presets from the project source
from simulation.pipelines import PIPELINES

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Run Simulation", layout="wide")
st.title("Run Simulation")

# ── Load defaults from config/simulation_config.yaml ──────────────────────

config_yaml   = load_config()
sim_cfg       = config_yaml.get("simulation", {})
models_cfg    = config_yaml.get("models", {})

DEFAULT_TURNS       = int(sim_cfg.get("num_turns", 5))
DEFAULT_SEED        = int(sim_cfg.get("seed", 42))
DEFAULT_TEMP        = float(sim_cfg.get("temperature", 0.7))
DEFAULT_MAX_RETRIES = int(sim_cfg.get("max_validation_retries", 2))
DEFAULT_PIPELINE    = str(sim_cfg.get("pipeline", "full"))
DEFAULT_NUM_PAT     = int(sim_cfg.get("num_patients", 1))

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "claude-3-sonnet",
    "claude-3-opus",
    "qwen2.5:3b",
    "llama3.1:8b",
]

MODEL_ROLES = [
    "persona_crafter",
    "dialogue_state",
    "validator",
    "client",
    "bot",
    "analyst",
]

# ── Layout ─────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 2])

# ── Left column: config form ───────────────────────────────────────────────

with left_col:
    st.subheader("Configuration")

    pipeline_choice = st.selectbox(
        "Pipeline preset",
        options=list(PIPELINES.keys()),
        index=list(PIPELINES.keys()).index(DEFAULT_PIPELINE)
              if DEFAULT_PIPELINE in PIPELINES else 0,
    )

    # Patient selection mode
    patient_mode = st.radio(
        "Patient selection",
        options=["Number of patients", "Specific patient IDs"],
        horizontal=True,
    )

    if patient_mode == "Number of patients":
        num_patients = st.number_input(
            "Number of patients", min_value=1, max_value=100, value=DEFAULT_NUM_PAT
        )
        specific_ids_str = ""
    else:
        specific_ids_str = st.text_input(
            "Patient IDs (comma-separated)", value="1"
        )
        num_patients = None

    turns       = st.number_input("Turns per conversation", min_value=1, max_value=50,  value=DEFAULT_TURNS)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=DEFAULT_TEMP, step=0.05)
    seed        = st.number_input("Seed", min_value=0, max_value=99999, value=DEFAULT_SEED)
    max_retries = st.number_input("Max validation retries", min_value=0, max_value=10, value=DEFAULT_MAX_RETRIES)

    st.markdown("**Models**")
    selected_models: dict[str, str] = {}
    for role in MODEL_ROLES:
        default_model = models_cfg.get(role, "gpt-4o-mini")
        idx = AVAILABLE_MODELS.index(default_model) if default_model in AVAILABLE_MODELS else 0
        selected_models[role] = st.selectbox(
            role.replace("_", " ").title(),
            options=AVAILABLE_MODELS,
            index=idx,
            key=f"model_{role}",
        )

    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# ── Right column: pipeline diagram + live log ──────────────────────────────

with right_col:
    st.subheader("Pipeline")

    pipeline_steps = PIPELINES.get(pipeline_choice, [])

    # Show initial all-pending diagram
    pipeline_placeholder = st.empty()
    pipeline_placeholder.plotly_chart(
        render_pipeline(pipeline_steps),
        use_container_width=True,
        key="pipeline_init",
    )

    log_placeholder   = st.empty()
    status_placeholder = st.empty()

# ── Run logic ──────────────────────────────────────────────────────────────

if run_button:
    # Build config dict for runner
    patients_list = None
    if patient_mode == "Specific patient IDs" and specific_ids_str.strip():
        patients_list = [p.strip() for p in specific_ids_str.split(",") if p.strip()]

    run_config: dict = {
        "pipeline":    pipeline_choice,
        "turns":       turns,
        "seed":        seed,
        "temperature": temperature,
        "max_retries": max_retries,
        "models":      selected_models,
    }
    if patients_list:
        run_config["patients"] = patients_list
    else:
        run_config["num_patients"] = num_patients

    # Start subprocess
    with status_placeholder:
        st.info("Starting simulation...")

    proc, start_time = start_simulation(run_config)

    # Locate the experiment directory (created by main.py on startup)
    exp_dir = find_experiment_dir(start_time, timeout=20.0)

    if exp_dir is None:
        status_placeholder.error(
            "Could not find the experiment output directory. "
            "Check that main.py is running correctly."
        )
        proc.wait()
    else:
        status_placeholder.info(f"Experiment directory: `{exp_dir.name}`")

        log_lines: list[str] = []

        for line in tail_log(exp_dir / "simulation.log", proc):
            log_lines.append(line)

            # Update pipeline diagram
            step_status = parse_step_status(log_lines)
            pipeline_placeholder.plotly_chart(
                render_pipeline(pipeline_steps, step_status),
                use_container_width=True,
                key=f"pipeline_{len(log_lines)}",
            )

            # Show last 20 log lines
            log_placeholder.code("\n".join(log_lines[-20:]), language=None)

        proc.wait()

        if proc.returncode == 0:
            status_placeholder.success(
                f"Simulation complete! Results saved in `{exp_dir.name}`. "
                "Navigate to **Experiments** or **Patient** to explore results."
            )
            # Final pipeline — all done
            final_status = {s: "done" for s in pipeline_steps}
            pipeline_placeholder.plotly_chart(
                render_pipeline(pipeline_steps, final_status),
                use_container_width=True,
                key="pipeline_final",
            )
        else:
            status_placeholder.error(
                f"Simulation failed (exit code {proc.returncode}). "
                "Check the log above for details."
            )
