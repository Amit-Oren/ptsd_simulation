import contextvars
from pathlib import Path

_experiment_dir_var: contextvars.ContextVar[Path] = contextvars.ContextVar(
    "experiment_dir", default=Path("data/output")
)
_context_subdir_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "context_subdir", default=""
)
_run_tokens_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "run_tokens", default={"input": 0, "output": 0}
)


def reset_run_tokens() -> None:
    _run_tokens_var.set({"input": 0, "output": 0})


def get_run_tokens() -> dict:
    tokens = _run_tokens_var.get()
    total = tokens["input"] + tokens["output"]
    return {"input": tokens["input"], "output": tokens["output"], "total": total}


def count_tokens(msg) -> None:
    usage = getattr(msg, "usage_metadata", None) or {}
    current = _run_tokens_var.get()
    _run_tokens_var.set({
        "input":  current["input"]  + usage.get("input_tokens", 0),
        "output": current["output"] + usage.get("output_tokens", 0),
    })


def set_experiment_dir(path: Path) -> None:
    _experiment_dir_var.set(path)


def set_context_subdir(subdir: str) -> None:
    _context_subdir_var.set(subdir)


def get_experiment_dir() -> Path:
    return _experiment_dir_var.get()


def get_context_subdir() -> str:
    return _context_subdir_var.get()
