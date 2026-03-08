import json
import logging
from abc import ABC
from pathlib import Path
from datetime import datetime
from langchain.agents import create_agent

_LOG_DIR = Path("data/output")
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_shared_handler = None


def _get_shared_handler():
    global _shared_handler
    if _shared_handler is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _shared_handler = logging.FileHandler(
            _LOG_DIR / f"simulation_log_{_RUN_TIMESTAMP}.txt", encoding="utf-8"
        )
        _shared_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    return _shared_handler


class BaseAgent(ABC):
        def __init__(self, name: str, role: str, system_prompt: str, llm, tools=None):
            self.name = name
            self.role = role
            self.system_prompt = system_prompt
            self.llm = llm
            self.tools = tools or []
            self.messages = []
            self.logger = self._setup_logger()
            self.agent = self._setup_agent()

        def _setup_logger(self):
            logger = logging.getLogger(self.name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                logger.addHandler(_get_shared_handler())
            return logger

        def _setup_agent(self):
            return create_agent(
                model=self.llm,
                system_prompt=self.system_prompt,
                tools=self.tools,
                response_format=self.response_format()
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
            """Log interaction to file"""
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": self.name,
                "input": input_msg,
                "response": response
            }
            self.logger.info(json.dumps(log_entry))

        def __repr__(self):
            return f"<Agent name={self.name} role={self.role}>"