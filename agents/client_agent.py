import numpy as np
from typing import Dict
from .base_agent import BaseAgent
from config.prompts import CLIENT_AGENT_PROMPT
from config.conversation_config import (EMOTIONS,
                                        INTENSITY_LEVELS,
                                        EXPRESSIVENESS_LEVELS,
                                        DISCLOSURE_LEVELS,
                                        POLITENESS_STYLES,
                                        ASSERTIVENESS_LEVELS)


class ClientAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, id, schedule, nodes, edges, conversation_pattern=None):
        self.id = id
        self.schedule = schedule
        self.nodes = nodes
        self.edges = edges
        self.conversation_pattern = conversation_pattern or self.sample_conversation_pattern()
        self.set_system_prompt()
        super().__init__(name, role, self.system_prompt, llm)


    def sample_conversation_pattern(self) -> Dict[str, str]:
        return {
            "emotion": np.random.choice(EMOTIONS),
            "intensity": np.random.choice(INTENSITY_LEVELS),
            "expressiveness": np.random.choice(EXPRESSIVENESS_LEVELS),
            "disclosure": np.random.choice(DISCLOSURE_LEVELS),
            "politeness": np.random.choice(POLITENESS_STYLES),
            "assertiveness": np.random.choice(ASSERTIVENESS_LEVELS),
        }
        
    def set_system_prompt(self):
        self.system_prompt = CLIENT_AGENT_PROMPT.format(
            id=self.id,
            nodes=self.nodes,
            edges=self.edges,
            schedule=self.schedule,
            **self.conversation_pattern
        )
        
