from .base_agent import BaseAgent
from config.prompts import BOT_AGENT_PROMPT


class BotAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, BOT_AGENT_PROMPT, llm)
