import json
import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_EXPERIMENT_DIR: Path = Path("data/output")
_CONTEXT_SUBDIR: str = ""


def set_experiment_dir(path: Path) -> None:
    """Set the experiment directory. Called once from main.py."""
    global _EXPERIMENT_DIR
    _EXPERIMENT_DIR = path


def set_context_subdir(subdir: str) -> None:
    """Set the context subfolder (e.g. 'session0', 'patient_1'). Called per session."""
    global _CONTEXT_SUBDIR
    _CONTEXT_SUBDIR = subdir


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, system_prompt: str, llm, tools=None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []
        self.messages = []
        self._call_index = 0
        self.agent = self.setup_agent()

    def setup_agent(self):
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SystemMessage(content=self.system_prompt)
        )

    def respond(self, message: str) -> str:
        """Generate a response to a message"""
        self.messages.append({"role": "user", "content": message})
        result = self.agent.invoke({"messages": self.messages})
        response = result["messages"][-1].content
        self.messages = result["messages"]
        self.log_response(message, response)
        return response

    def response_format(self):
        pass

    def reset_memory(self):
        """Clear conversation history"""
        self.messages = []

    def log_response(self, input_msg: str, response: str):
        self._write_finetune(input_msg, response)
        self._call_index += 1

    def _write_finetune(self, input_msg: str, response: str):
        finetune_dir = _EXPERIMENT_DIR / "context" / _CONTEXT_SUBDIR if _CONTEXT_SUBDIR else _EXPERIMENT_DIR / "context"
        finetune_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name}_call{self._call_index:02d}.json"
        data = {
            "meta": {
                "agent_name": self.name,
                "role":       self.role,
                "call_index": self._call_index,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "messages": [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": input_msg},
                {"role": "assistant", "content": response},
            ],
        }
        with open(finetune_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"<Agent name={self.name} role={self.role}>"
