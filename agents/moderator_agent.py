from .base_agent import BaseAgent
from config.prompts import MODERATOR_AGENT_PROMPT, MODERATOR_USER_PROMPT


class ModeratorAgent(BaseAgent):
    """Agent responsible for refining client messages to be more human-like."""
    def __init__(self, name: str, role: str, llm, persona: str):
        prompt = MODERATOR_AGENT_PROMPT.format(persona=persona)
        super().__init__(name, role, prompt, llm)

    def moderate_message(self, message: str, nodes: dict, edges: dict) -> str:
        """Refine the client's drafted message. Stateless — no history kept."""
        active_nodes = [n for n, v in nodes.items() if v]
        active_edges = [(p, c, v) for (p, c), v in edges.items() if v > 0]

        nodes_str = ", ".join(active_nodes) or "(none)"
        edges_str = "\n".join(f"  - {p} → {c}: {v:.4f}" for p, c, v in active_edges) or "  (none)"

        user_message = MODERATOR_USER_PROMPT.format(
            nodes=nodes_str,
            edges=edges_str,
            message=message,
        )
        self.messages = []
        refined = self.respond(user_message)
        self.messages = []
        return refined
