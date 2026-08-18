from .base_agent import BaseAgent
from config.prompts import CLIENT_AGENT_PROMPT
from simulation.graph import iter_edges


def _format_components(nodes: dict, edges: dict) -> tuple[str, str]:
    nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
    edges_str = "\n".join(f"  - {p} → {c}: {v:.4f}" for p, c, v in iter_edges(edges))
    return nodes_str, edges_str


def _format_self_report(self_report: dict) -> str:
    if not self_report:
        return "  (none)"
    lines = []
    for node, items in self_report.items():
        lines.append(f"  {node}:")
        for item in items:
            if isinstance(item, dict):
                lines.append(f"    • {item['key']}: {item['value']}")
            else:
                lines.append(f"    • {item}")
    return "\n".join(lines)


class ClientAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm, vignette: str, nodes: dict, edges: dict,
                 initial_message: str, session_history: str = "", self_report: dict = None):
        nodes_str, edges_str = _format_components(nodes, edges)
        self_report_str = _format_self_report(self_report or {})
        prompt = CLIENT_AGENT_PROMPT.format(
            vignette=vignette, nodes=nodes_str, edges=edges_str,
            session_history=session_history, self_report=self_report_str,
        )
        super().__init__(name, role, prompt, llm)

        # Seed the conversation with the validated initial message as the first client turn
        self.initial_message = initial_message
        self.messages.append({"role": "assistant", "content": initial_message})
