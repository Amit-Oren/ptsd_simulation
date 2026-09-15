"""
Unit tests for simulation/run_context.py's ContextVar-backed run state.

Run: python -m pytest tests/ -x -q
"""
import contextvars
from pathlib import Path

from simulation.run_context import (
    set_experiment_dir, set_context_subdir, get_experiment_dir, get_context_subdir,
    reset_run_tokens, get_run_tokens, count_tokens,
)


class _FakeMsg:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


def test_set_and_get_experiment_dir():
    set_experiment_dir(Path("data/output/experiment_test"))
    assert get_experiment_dir() == Path("data/output/experiment_test")


def test_set_and_get_context_subdir():
    set_context_subdir("patients/patient_0/context/session_0")
    assert get_context_subdir() == "patients/patient_0/context/session_0"


def test_reset_run_tokens_starts_at_zero():
    reset_run_tokens()
    tokens = get_run_tokens()
    assert tokens == {"input": 0, "output": 0, "total": 0}


def test_count_tokens_accumulates():
    reset_run_tokens()
    count_tokens(_FakeMsg(input_tokens=10, output_tokens=5))
    count_tokens(_FakeMsg(input_tokens=3, output_tokens=2))
    tokens = get_run_tokens()
    assert tokens == {"input": 13, "output": 7, "total": 20}


def test_reset_run_tokens_clears_previous_accumulation():
    reset_run_tokens()
    count_tokens(_FakeMsg(input_tokens=100, output_tokens=100))
    reset_run_tokens()
    assert get_run_tokens() == {"input": 0, "output": 0, "total": 0}


def test_count_tokens_handles_missing_usage_metadata():
    class _NoUsage:
        pass
    reset_run_tokens()
    count_tokens(_NoUsage())  # must not raise
    assert get_run_tokens() == {"input": 0, "output": 0, "total": 0}


def test_count_tokens_handles_none():
    reset_run_tokens()
    count_tokens(None)  # must not raise (base_agent passes raw.get("raw"), which can be None)
    assert get_run_tokens() == {"input": 0, "output": 0, "total": 0}


def test_context_isolation_across_contextvars_contexts():
    """A separate context (simulating a concurrently-running task) must not see
    changes made in another context — this is the actual bug the ContextVar
    swap fixes vs. plain module globals."""
    reset_run_tokens()
    count_tokens(_FakeMsg(input_tokens=50, output_tokens=50))
    assert get_run_tokens()["total"] == 100

    def _in_separate_context():
        # A fresh copy of the current context, then mutated independently —
        # analogous to a concurrently-dispatched task's own execution context.
        reset_run_tokens()
        count_tokens(_FakeMsg(input_tokens=1, output_tokens=1))
        return get_run_tokens()["total"]

    ctx = contextvars.copy_context()
    isolated_total = ctx.run(_in_separate_context)

    assert isolated_total == 2
    # The outer context's own tokens must be untouched by the isolated run.
    assert get_run_tokens()["total"] == 100
