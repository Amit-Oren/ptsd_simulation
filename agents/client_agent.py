from .base_agent import BaseAgent
from config.prompts import CLIENT_AGENT_PROMPT


def _format_components(nodes: dict, edges: dict) -> tuple[str, str]:
    nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
    edges_str = "\n".join(f"  - {p} → {c}: {val:.4f}" for (p, c), val in edges.items())
    return nodes_str, edges_str


class ClientAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, persona: str, nodes: dict, edges: dict, initial_message: str):
        nodes_str, edges_str = _format_components(nodes, edges)
        prompt = CLIENT_AGENT_PROMPT.format(persona=persona, nodes=nodes_str, edges=edges_str)
        super().__init__(name, role, prompt, llm)

        # Seed the conversation with the validated initial message as the first client turn
        self.initial_message = initial_message
        self.messages.append({"role": "assistant", "content": initial_message})
