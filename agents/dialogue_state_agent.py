from .base_agent import BaseAgent
from config.prompts import DIALOGUE_STATE_AGENT_PROMPT
from data.input import get_observation_components


def _format_components(nodes: dict, edges: dict) -> tuple[str, str]:
    nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
    edges_str = "\n".join(f"  - {p} → {c}: {val:.4f}" for (p, c), val in edges.items())
    return nodes_str, edges_str


class DialogueStateAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, patient_id: int, persona: str):
        prompt = DIALOGUE_STATE_AGENT_PROMPT.format(persona=persona)
        super().__init__(name, role, prompt, llm)
        self.patient_id = patient_id
        self._obs_index = 0
        self.datetime, self.nodes, self.edges = get_observation_components(patient_id, self._obs_index)

    def create_message(self) -> str:
        """Generate a user prompt message for the current observation."""
        nodes_str, edges_str = _format_components(self.nodes, self.edges)
        observation = (
            f"Current datetime: {self.datetime}\n"
            f"Nodes:\n{nodes_str}\n"
            f"Edges:\n{edges_str}"
        )
        return self.respond(observation)

    def next_observation(self):
        """Advance to the next datetime and update nodes and edges."""
        self._obs_index += 1
        self.datetime, self.nodes, self.edges = get_observation_components(self.patient_id, self._obs_index)
