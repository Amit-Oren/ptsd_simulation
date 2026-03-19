from .base_agent import BaseAgent
from config.prompts import PERSONA_CRAFTER_AGENT_PROMPT, PERSONA_CRAFTER_USER_PROMPT
from data.input import prepare_patient_components


def _format_components(nodes: dict, edges: dict) -> tuple[str, str]:
    nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
    edges_str = "\n".join(f"  - {p} → {c}: {val:.4f}" for (p, c), val in edges.items())
    return nodes_str, edges_str


class PersonaCrafterAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, patient_id: int):
        nodes, edges = prepare_patient_components(patient_id)
        nodes_str, edges_str = _format_components(nodes, edges)
        prompt = PERSONA_CRAFTER_AGENT_PROMPT.format(nodes=nodes_str, edges=edges_str)
        super().__init__(name, role, prompt, llm)

    def create_persona(self) -> str:
        """Generate and return the persona description vignette."""
        self.persona = self.respond(PERSONA_CRAFTER_USER_PROMPT)
        return self.persona
