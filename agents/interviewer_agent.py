from .base_agent import BaseAgent
from config.prompts import INTERVIEWER_AGENT_PROMPT


class InterviewerAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, memory_context: str = ""):
        prompt = INTERVIEWER_AGENT_PROMPT.format(
            memory_context=memory_context
            or "(No prior sessions — this is the first time meeting this patient.)"
        )
        super().__init__(name, role, prompt, llm)
