"""
call_log.py — writes the audit-trail record of one agent call to disk.
"""
import json
from datetime import datetime

from simulation.run_context import get_experiment_dir, get_context_subdir, count_tokens


def write_call_log(agent_name: str, role: str, call_index: int, system_prompt: str,
                    input_msg: str, response: str,
                    output: dict | list | str | None = None) -> None:
    experiment_dir = get_experiment_dir()
    context_subdir = get_context_subdir()
    context_dir = experiment_dir / context_subdir if context_subdir else experiment_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{agent_name}_call{call_index:02d}.json"
    data = {
        "meta": {
            "agent_name": agent_name,
            "role":       role,
            "call_index": call_index,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": input_msg},
            {"role": "assistant", "content": response},
        ],
        "output": output if output is not None else response,
    }
    with open(context_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class CallLogger:
    """Owned by one agent instance (see agents/base_agent.py). Holds that
    agent's own call-index — per-instance, not shared, so no concurrency
    risk the way a module-level counter would have. record() is the single
    place a call's tokens get counted (only if msg is given — the four
    agents that call log_response directly today have no message object to
    count, so they pass none and stay uncounted, same as before) and its
    audit JSON gets written."""

    def __init__(self, agent_name: str, role: str, system_prompt: str):
        self.agent_name = agent_name
        self.role = role
        self.system_prompt = system_prompt
        self._call_index = 0

    def record(self, input_msg: str, response: str,
               output: dict | list | str | None = None, msg=None) -> None:
        if msg is not None:
            count_tokens(msg)
        write_call_log(self.agent_name, self.role, self._call_index, self.system_prompt,
                        input_msg, response, output)
        self._call_index += 1
