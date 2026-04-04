""" Agent responsible for generating the user's prompt message based on the current dialogue state, 
    which includes datetime, nodes, and edges. """

import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)
from data.input.input import get_session_components
from config.prompts import (DIALOGUE_STATE_AGENT_PROMPT, 
                            DIALOGUE_STATE_AGENT_RETRY_PROMPT)



class DialogueStateAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, patient_id: int, persona: str, obs_index: int = 0):
        prompt = DIALOGUE_STATE_AGENT_PROMPT.format(persona=persona)
        super().__init__(name, role, prompt, llm)
        self.patient_id = patient_id
        self._obs_index = obs_index
        self.datetime, self.nodes, self.edges = get_session_components(patient_id, self._obs_index)

    def _build_observation(self) -> str:
        nodes_str, edges_str = self._format_components(self.nodes, self.edges)
        return (
            f"Current datetime: {self.datetime}\n"
            f"Nodes:\n{nodes_str}\n"
            f"Edges:\n{edges_str}"
        )

    def create_message(self) -> str:
        """Generate a user prompt message for the current observation."""
        message = self.respond(self._build_observation())
        logger.info("[%s] initial message generated", self.name)
        return message

    def create_message_with_feedback(self, feedback: str) -> str:
        """Retry message generation with validator feedback"""
        retry_prompt = DIALOGUE_STATE_AGENT_RETRY_PROMPT.format(
            feedback=feedback,
            previous_message=self.messages[-1].content if self.messages else "(no previous message)"
        )
        message = self.respond(retry_prompt)
        logger.info("[%s] message revised after feedback", self.name)
        return message
        

    def next_observation(self):
        """Advance to the next datetime and update nodes and edges."""
        self._obs_index += 1
        self.datetime, self.nodes, self.edges = get_session_components(self.patient_id, self._obs_index)

    def _format_components(self, nodes: dict, edges: dict) -> tuple[str, str]:
        nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
        edges_str = "\n".join(f"  - {p} → {c}: {val:.4f}" for (p, c), val in edges.items())
        return nodes_str, edges_str
