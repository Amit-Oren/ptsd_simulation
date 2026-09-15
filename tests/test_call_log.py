"""
Unit tests for simulation/call_log.py.

Run: python -m pytest tests/ -x -q
"""
import json

from simulation.run_context import set_experiment_dir, set_context_subdir, reset_run_tokens, get_run_tokens
from simulation.call_log import write_call_log, CallLogger


def test_write_call_log_writes_expected_file_and_content(tmp_path):
    set_experiment_dir(tmp_path)
    set_context_subdir("patients/patient_0/context/session_0")

    write_call_log(
        agent_name="Interviewer", role="Interviewer", call_index=2,
        system_prompt="be brief", input_msg="hello", response="hi there",
    )

    expected_path = tmp_path / "patients/patient_0/context/session_0" / "Interviewer_call02.json"
    assert expected_path.exists()

    data = json.loads(expected_path.read_text(encoding="utf-8"))
    assert data["meta"]["agent_name"] == "Interviewer"
    assert data["meta"]["call_index"] == 2
    assert data["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert data["output"] == "hi there"  # falls back to response when output=None


def test_write_call_log_uses_explicit_output_when_given(tmp_path):
    set_experiment_dir(tmp_path)
    set_context_subdir("")  # falls back to "<experiment_dir>/context"

    write_call_log(
        agent_name="Analyst", role="Analyst", call_index=0,
        system_prompt="score it", input_msg="conversation text", response="raw text",
        output={"score": 1},
    )

    expected_path = tmp_path / "context" / "Analyst_call00.json"
    assert expected_path.exists()
    data = json.loads(expected_path.read_text(encoding="utf-8"))
    assert data["output"] == {"score": 1}


class _FakeMsg:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


def test_call_logger_increments_its_own_call_index(tmp_path):
    set_experiment_dir(tmp_path)
    set_context_subdir("patients/patient_0/context/session_0")

    logger = CallLogger("Interviewer", "Interviewer", "be brief")
    logger.record("hi", "hello there")
    logger.record("how are you", "fine thanks")

    base = tmp_path / "patients/patient_0/context/session_0"
    assert (base / "Interviewer_call00.json").exists()
    assert (base / "Interviewer_call01.json").exists()


def test_call_logger_two_instances_have_independent_call_indices(tmp_path):
    """Per-instance state, not shared — two agents of the same role logging
    concurrently shouldn't clobber each other's call numbering."""
    set_experiment_dir(tmp_path)
    set_context_subdir("")

    logger_a = CallLogger("Client_0", "Patient", "p1")
    logger_b = CallLogger("Client_1", "Patient", "p2")
    logger_a.record("a1", "r1")
    logger_a.record("a2", "r2")
    logger_b.record("b1", "r1")

    assert logger_a._call_index == 2
    assert logger_b._call_index == 1


def test_call_logger_counts_tokens_only_when_msg_given(tmp_path):
    set_experiment_dir(tmp_path)
    set_context_subdir("")

    logger = CallLogger("Analyst", "Analyst", "score it")
    reset_run_tokens()

    logger.record("input with no message object", "output", output={"x": 1})  # msg=None
    assert get_run_tokens()["total"] == 0

    logger.record("input with a message", "output", msg=_FakeMsg(input_tokens=5, output_tokens=3))
    assert get_run_tokens()["total"] == 8
