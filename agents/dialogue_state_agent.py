import logging
from .base_agent import BaseAgent
from config.prompts import DIALOGUE_STATE_AGENT_PROMPT, DIALOGUE_STATE_AGENT_RETRY_PROMPT
from simulation.graph import iter_edges

logger = logging.getLogger(__name__)


class DialogueStateAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, vignette: str, nodes: dict, edges: dict,
                 session_history: str = ""):
        prompt = DIALOGUE_STATE_AGENT_PROMPT.format(vignette=vignette, session_history=session_history)
        super().__init__(name, role, prompt, llm)
        self.nodes = nodes
        self.edges = edges

    def _build_observation(self) -> str:
        nodes_str, edges_str = self._format_components(self.nodes, self.edges)
        return f"Nodes:\n{nodes_str}\n\nEdges:\n{edges_str}"

    def create_message(self) -> str:
        message = self.respond(self._build_observation())
        logger.info("[%s] initial message generated", self.name)
        return message

    def create_message_with_feedback(self, feedback: str) -> str:
        last = self.messages[-1] if self.messages else None
        prev = getattr(last, "content", last.get("content", "") if isinstance(last, dict) else "") if last else ""
        retry_prompt = DIALOGUE_STATE_AGENT_RETRY_PROMPT.format(
            feedback=feedback,
            previous_message=prev or "(no previous message)",
        )
        message = self.respond(retry_prompt)
        logger.info("[%s] message revised after feedback", self.name)
        return message

    def _format_components(self, nodes: dict, edges: dict) -> tuple[str, str]:
        nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
        edges_str = "\n".join(f"  - {p} → {c}: {v:.4f}" for p, c, v in iter_edges(edges))
        return nodes_str, edges_str
