"""
loader.py — utilities for reading experiment outputs and config from disk.

All paths are resolved relative to PROJECT_ROOT so this module works
regardless of where Streamlit is launched from.
"""

import json
import re
from pathlib import Path

import yaml

# Navigate from streamlit_app/utils/ up two levels to the project root.
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_experiments() -> list[dict]:
    """
    Return a list of experiment directories found under data/output/,
    sorted newest first.

    Each item is a dict with keys:
        path      (Path)  — absolute path to the experiment directory
        name      (str)   — directory name, e.g. "experiment_20260331_191849"
        timestamp (str)   — human-readable timestamp extracted from the name
    """
    output_dir = PROJECT_ROOT / "data" / "output"
    if not output_dir.exists():
        return []

    experiments = []
    for d in sorted(output_dir.glob("experiment_*"), reverse=True):
        if d.is_dir():
            # Extract timestamp portion after "experiment_"
            timestamp = d.name.replace("experiment_", "").replace("_", " ", 1)
            experiments.append(
                {
                    "path": d,
                    "name": d.name,
                    "timestamp": timestamp,
                }
            )
    return experiments


def get_patients(experiment_dir: Path) -> list[dict]:
    """
    Return a list of patient result files inside *experiment_dir*.

    Each item is a dict with keys:
        path       (Path) — absolute path to the JSON file
        patient_id (str)  — e.g. "1"
    """
    patients = []
    for json_file in sorted(experiment_dir.glob("experiment_*.json")):
        if json_file.name.endswith("_summary.json"):
            continue
        match = re.search(r"experiment_(\w+)\.json$", json_file.name)
        patient_id = match.group(1) if match else json_file.stem
        patients.append({"path": json_file, "patient_id": patient_id})
    return patients


def load_patient(json_path: Path) -> dict:
    """Load and return the patient result JSON as a dict."""
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_log(experiment_dir: Path) -> list[str]:
    """
    Read simulation.log from *experiment_dir* and return its lines
    (trailing newlines stripped).
    """
    log_path = experiment_dir / "simulation.log"
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh.readlines()]


def load_config() -> dict:
    """
    Load config/simulation_config.yaml and return as a dict.
    Returns an empty dict if the file is missing.
    """
    config_path = PROJECT_ROOT / "config" / "simulation_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_context_files(experiment_dir: Path, patient_id: str) -> list[dict]:
    """
    Load all context JSON files for *patient_id* inside
    *experiment_dir*/context/.

    Each item is a dict with keys:
        filename (str)  — e.g. "Client_1_PersonaCrafter_call00.json"
        data     (dict) — parsed JSON content
    """
    context_dir = experiment_dir / "context"
    if not context_dir.exists():
        return []

    results = []
    pattern = f"Client_{patient_id}_*.json"
    for json_file in sorted(context_dir.glob(pattern)):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
        results.append({"filename": json_file.name, "data": data})
    return results
