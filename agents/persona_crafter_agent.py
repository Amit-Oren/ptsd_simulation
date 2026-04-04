import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)
from data.input.input import get_aggregated_context, get_demographics
from config.prompts import (PERSONA_CRAFTER_AGENT_PROMPT,
                            PERSONA_CRAFTER_AGENT_PROMPT_CONTEXT,
                            PERSONA_CRAFTER_RETRY_PROMPT,
                            PERSONA_CRAFTER_USER_PROMPT)


class PersonaCrafterAgent(BaseAgent):
    """Agent responsible for crafting a persona vignette based on patient data."""
    def __init__(self, name: str, role: str, llm, patient_id: int,
                 use_demographics: bool = False, use_self_report: bool = False):
        context = get_aggregated_context(patient_id)

        required_edges, forbidden_edges = self._split_edges(context["edges"])

        if use_demographics or use_self_report:
            prompt = PERSONA_CRAFTER_AGENT_PROMPT_CONTEXT.format(
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
                demographics=self._format_demographics(get_demographics(patient_id)) if use_demographics else "",
                self_report=self._format_self_report(context["nodes"]) if use_self_report else "",
            )
        else:
            prompt = PERSONA_CRAFTER_AGENT_PROMPT.format(
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )

        super().__init__(name, role, prompt, llm)

    def create_persona(self) -> str:
        self.persona = self.respond(PERSONA_CRAFTER_USER_PROMPT)
        logger.info("[%s] vignette written", self.name)
        return self.persona

    def create_persona_with_feedback(self, feedback: str) -> str:
        retry_prompt = PERSONA_CRAFTER_RETRY_PROMPT.format(
            feedback=feedback,
            previous_persona=self.persona,
        )
        self.persona = self.respond(retry_prompt)
        logger.info("[%s] vignette revised after feedback", self.name)
        return self.persona

    def _split_edges(self, edges: dict) -> tuple[str, str]:
        required = [k for k, v in edges.items() if v.get("strength", 0) > 0]
        forbidden = [k for k, v in edges.items() if v.get("strength", 0) == 0]
        req_str = "\n".join(f"  - {k[0]} → {k[1]}" for k in required) or "  (none)"
        forb_str = "\n".join(f"  - {k[0]} → {k[1]}" for k in forbidden) or "  (none)"
        return req_str, forb_str

    def _format_demographics(self, demo: dict) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in demo.items())

    def _format_self_report(self, nodes: dict) -> str:
        lines = []
        for node, data in nodes.items():
            items = data.get("items", [])
            if items:
                lines.append(f"  {node}: {', '.join(items)}")
        return "\n".join(lines)
