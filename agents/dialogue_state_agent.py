from .base_agent import BaseAgent
from ..config.prompts import DIALOGUE_STATE_AGENT_PROMPT
from ..data.input import get_observation_components


class DialogueStateAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, patient_id: int):
        self.patient_id = patient_id
        self._obs_index = 0
        self.datetime, self.nodes, self.edges = get_observation_components(patient_id, self._obs_index)
        self.set_system_prompt()
        super().__init__(name, role, self.system_prompt, llm)

    
    def set_system_prompt(self):
        self.system_prompt = DIALOGUE_STATE_AGENT_PROMPT.format(
            id=self.id,
            nodes=self.nodes,
            edges=self.edges,
        )

    def next_observation(self):
        """Advance to the next datetime and update nodes and edges."""
        self._obs_index += 1
        self.datetime, self.nodes, self.edges = get_observation_components(self.patient_id, self._obs_index)
