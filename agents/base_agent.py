import logging
from abc import ABC
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from simulation.call_log import CallLogger

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Common base every agent class inherits from."""

    def __init__(self, name: str, role: str, system_prompt: str, llm, tools=None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []
        self.messages = []
        self._call_logger = CallLogger(name, role, system_prompt)
        self.agent = self.setup_agent()

    def setup_agent(self):
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SystemMessage(content=self.system_prompt)
        )

    def respond(self, message: str) -> str:
        """Send a message to the agent and get its response.
        The agent's message history is updated with the new message and the response."""
        self.messages.append({"role": "user", "content": message})
        result = self.agent.invoke({"messages": self.messages})
        last = result["messages"][-1]
        response = last.content
        self.messages = result["messages"]
        self._call_logger.record(message, response, msg=last)
        return response

    def reset_memory(self):
        """Clear the agent's message history, so the next call starts fresh."""
        self.messages = []

    def log_response(self, input_msg: str, response: str, output: dict | list | str | None = None):
        """Record one call to disk for auditing purposes."""
        self._call_logger.record(input_msg, response, output)

    def _invoke_structured(self, schema, system_prompt: str, user_prompt: str):
        """One-shot, schema-validated call used by agents that want a Pydantic
        object back instead of free text."""
        structured_llm = self.llm.with_structured_output(schema, include_raw=True, method="function_calling")
        raw = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        parsed = None
        raw_text = ""
        raw_msg = None
        if isinstance(raw, dict):
            raw_msg = raw.get("raw")
            parsed = raw.get("parsed")
            raw_text = getattr(raw_msg, "content", "") if raw_msg else ""
        elif hasattr(raw, "model_dump"):
            parsed = raw
        output = parsed.model_dump() if parsed is not None else {"parse_error": True}
        self._call_logger.record(user_prompt, raw_text or "(no raw content)", output=output, msg=raw_msg)
        return parsed

    def __repr__(self):
        """So printing an agent shows something readable instead of Python's
        default object repr."""
        return f"<Agent name={self.name} role={self.role}>"
