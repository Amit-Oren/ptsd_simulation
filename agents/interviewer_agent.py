from .base_agent import BaseAgent
from config.prompts import INTERVIEWER_AGENT_PROMPT


class InterviewerAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, INTERVIEWER_AGENT_PROMPT, llm)
