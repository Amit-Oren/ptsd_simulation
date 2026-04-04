"""
runner.py — utilities for launching main.py as a subprocess and streaming
its log output back to the Streamlit UI.
"""

import subprocess
import time
from pathlib import Path
from typing import Generator, Optional

# Navigate from streamlit_app/utils/ up two levels to the project root.
PROJECT_ROOT = Path(__file__).parent.parent.parent


def build_args(config: dict) -> list[str]:
    """
    Convert a UI config dict into a list of CLI arguments for main.py.

    Recognised keys
    ---------------
    patients      list[int|str]  — explicit patient IDs (mutually exclusive with num_patients)
    num_patients  int            — number of patients to simulate
    turns         int            — number of conversation turns
    seed          int            — random seed
    temperature   float          — model temperature
    max_retries   int            — maximum validation retries
    pipeline      str            — pipeline preset name
    models        dict           — per-role model overrides, e.g. {"client": "gpt-4o-mini"}
    """
    args: list[str] = []

    # Patient selection — either explicit IDs or a count
    patients = config.get("patients")
    if patients:
        ids = ",".join(str(p) for p in patients)
        args += ["--patient_ids", ids]
    elif config.get("num_patients") is not None:
        args += ["--num_patients", str(config["num_patients"])]

    if config.get("turns") is not None:
        args += ["--num_turns", str(config["turns"])]

    if config.get("seed") is not None:
        args += ["--seed", str(config["seed"])]

    if config.get("temperature") is not None:
        args += ["--temperature", str(config["temperature"])]

    if config.get("max_retries") is not None:
        args += ["--max_retries", str(config["max_retries"])]

    if config.get("pipeline"):
        args += ["--pipeline", config["pipeline"]]

    # Per-role model overrides
    models: dict = config.get("models") or {}
    role_flags = {
        "persona_crafter": "--model_persona_crafter",
        "dialogue_state":  "--model_dialogue_state",
        "validator":       "--model_validator",
        "client":          "--model_client",
        "bot":             "--model_bot",
        "analyst":         "--model_analyst",
    }
    for role, flag in role_flags.items():
        if models.get(role):
            args += [flag, models[role]]

    return args


def start_simulation(config: dict) -> tuple[subprocess.Popen, float]:
    """
    Start main.py as a subprocess.

    Returns
    -------
    proc        subprocess.Popen  — the running process
    start_time  float             — time.time() just before launch
    """
    args = build_args(config)
    cmd = ["python", "main.py"] + args

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, start_time


def find_experiment_dir(start_time: float, timeout: float = 10.0) -> Optional[Path]:
    """
    Poll data/output/ until a directory named experiment_* appears whose
    modification time is >= *start_time*, or until *timeout* seconds elapse.

    Returns the Path on success, None on timeout.
    """
    output_dir = PROJECT_ROOT / "data" / "output"
    deadline = time.time() + timeout

    while time.time() < deadline:
        if output_dir.exists():
            for candidate in sorted(output_dir.glob("experiment_*"), reverse=True):
                if candidate.is_dir() and candidate.stat().st_mtime >= start_time:
                    return candidate
        time.sleep(0.5)

    return None


def tail_log(log_path: Path, proc: subprocess.Popen) -> Generator[str, None, None]:
    """
    Yield lines from *log_path* while *proc* is still running.

    The generator opens the file in follow mode (similar to `tail -f`) and
    keeps yielding new lines until the process exits.  After the process
    finishes it drains any remaining lines before returning.
    """
    # Wait briefly for the log file to be created
    deadline = time.time() + 15.0
    while not log_path.exists() and time.time() < deadline:
        time.sleep(0.3)

    if not log_path.exists():
        return

    with open(log_path, "r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
            else:
                # No new data — check whether the process is still alive
                if proc.poll() is not None:
                    # Process finished; drain any remaining lines
                    for remaining in fh.readlines():
                        remaining = remaining.rstrip("\n")
                        if remaining:
                            yield remaining
                    break
                time.sleep(0.2)
