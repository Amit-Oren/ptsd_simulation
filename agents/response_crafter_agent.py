from .base_agent import BaseAgent
from ..config.prompts import RESPONSE_CRAFTER_AGENT_PROMPT    

class ResponseCrafterAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, RESPONSE_CRAFTER_AGENT_PROMPT, llm)