"""
loader.py — utilities for reading experiment outputs and config from disk.

All paths are resolved relative to PROJECT_ROOT so this module works
regardless of where Streamlit is launched from.

Directory layout (see simulation/runner.py):
    data/output/experiment_<timestamp>/
        manifest.json                              # run-level config, once
        simulation.log
        patients/patient_<idx>/
            state.json
            memory_snapshot.json
            context/
                persona/<Agent>_callNN.json
                session_<idx>/<Agent>_callNN.json
            sessions/session_<idx>/
                full.json
                report.txt
                summary.json
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


def load_manifest(experiment_dir: Path) -> dict:
    """Load manifest.json (run-level config, written once per experiment)."""
    path = experiment_dir / "manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_patients(experiment_dir: Path) -> list[dict]:
    """
    Return one entry per (patient, session) — same granularity the UI has always
    shown, since each session's full.json carries everything (demographics,
    vignette, conversation, analysis) needed to render any page.

    Each item is a dict with keys:
        path       (Path) — absolute path to that session's full.json
        patient_id (str)  — e.g. "0_session0", kept in this format for
                             backward compatibility with the existing sidebar/
                             experiment-table display code.
    """
    patients_dir = experiment_dir / "patients"
    if not patients_dir.exists():
        return []

    results = []
    for patient_dir in sorted(patients_dir.glob("patient_*")):
        m = re.match(r"patient_(\d+)$", patient_dir.name)
        if not m or not patient_dir.is_dir():
            continue
        patient_idx = m.group(1)
        sessions_dir = patient_dir / "sessions"
        if not sessions_dir.exists():
            continue
        for session_dir in sorted(sessions_dir.glob("session_*")):
            sm = re.match(r"session_(\d+)$", session_dir.name)
            full_json = session_dir / "full.json"
            if not sm or not full_json.exists():
                continue
            session_idx = sm.group(1)
            results.append({
                "path": full_json,
                "patient_id": f"{patient_idx}_session{session_idx}",
            })
    return results


def load_patient(json_path: Path) -> dict:
    """Load and return a session's full.json as a dict."""
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


def _find_memory_id_owners() -> dict:
    """
    Best-effort reverse lookup: memory_id -> [{"experiment": name, "patient_idx": idx}, ...],
    by scanning every experiment's patient state.json for the memory_id it recorded.
    Used only to give opaque memory ids a human-readable label in the UI.
    """
    owners: dict[str, list[dict]] = {}
    output_dir = PROJECT_ROOT / "data" / "output"
    if not output_dir.exists():
        return owners

    for state_path in output_dir.glob("experiment_*/patients/patient_*/state.json"):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        mid = data.get("memory_id")
        if not mid:
            continue
        owners.setdefault(mid, []).append({
            "experiment": state_path.parents[2].name,
            "patient_idx": data.get("patient_idx"),
        })
    return owners


def get_memory_patients() -> list[dict]:
    """
    Return every patient that has a persistent memory graph under data/memory/
    (independent of any experiment run — memory outlives any single output dir).

    Each item is a dict with keys:
        memory_id (str)   — the random id memory is actually keyed by
        path      (Path)  — patient's memory directory
        label     (str)   — human-readable label, e.g. "Patient 0 (experiment_...)"
                             when a matching experiment/patient_idx was found,
                             otherwise just the raw memory_id
    """
    patients_dir = PROJECT_ROOT / "data" / "memory" / "patients"
    if not patients_dir.exists():
        return []

    owners = _find_memory_id_owners()

    results = []
    for patient_dir in sorted(patients_dir.glob("patient_*")):
        m = re.match(r"patient_(.+)$", patient_dir.name)
        if not m or not patient_dir.is_dir():
            continue
        memory_id = m.group(1)
        owned_by = owners.get(memory_id, [])
        if owned_by:
            label = " / ".join(f"Patient {o['patient_idx']} ({o['experiment']})" for o in owned_by)
        else:
            label = f"memory_id={memory_id}"
        results.append({"memory_id": memory_id, "path": patient_dir, "label": label})
    return results


def load_memory_graph(memory_id: str) -> dict:
    """Load the current (latest) memory graph for one patient."""
    path = PROJECT_ROOT / "data" / "memory" / "patients" / f"patient_{memory_id}" / "graph.json"
    if not path.exists():
        return {"nodes": [], "edges": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_memory_snapshots(memory_id: str) -> list[dict]:
    """
    Return every per-session snapshot of a patient's memory graph, oldest first.

    Each item is a dict with keys:
        session_idx (int)
        path        (Path)
    """
    snapshots_dir = (
        PROJECT_ROOT / "data" / "memory" / "patients" / f"patient_{memory_id}" / "snapshots"
    )
    if not snapshots_dir.exists():
        return []

    results = []
    for f in sorted(snapshots_dir.glob("session_*.json")):
        m = re.match(r"session_(\d+)\.json$", f.name)
        if not m:
            continue
        results.append({"session_idx": int(m.group(1)), "path": f})
    return sorted(results, key=lambda r: r["session_idx"])


def load_memory_snapshot(path: Path) -> dict:
    """Load one memory-graph snapshot (same shape as load_memory_graph)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_context_files(experiment_dir: Path, patient_id: str) -> list[dict]:
    """
    Load all context JSON files for one (patient, session) — *patient_id* is
    the "<idx>_session<idx>" string returned by get_patients().

    Each item is a dict with keys:
        filename (str)  — e.g. "Interviewer_call00.json"
        data     (dict) — parsed JSON content
    """
    m = re.match(r"(\d+)_session(\d+)$", patient_id)
    if not m:
        return []
    patient_idx, session_idx = m.group(1), m.group(2)

    context_dir = (
        experiment_dir / "patients" / f"patient_{patient_idx}"
        / "context" / f"session_{session_idx}"
    )
    if not context_dir.exists():
        return []

    results = []
    for json_file in sorted(context_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
        results.append({"filename": json_file.name, "data": data})
    return results
