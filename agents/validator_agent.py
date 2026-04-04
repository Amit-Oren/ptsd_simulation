"""Agent responsible for validating outputs (vignettes or dialogue messages)
against the patient's cognitive graph context."""

import logging
from typing import TypedDict, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from config.prompts import (VALIDATOR_VIGNETTE_PROMPT,
                             VALIDATOR_DIALOGUE_PROMPT)

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reasoning: str


class ValidatorContext(TypedDict):
    output_type: str   # "vignette" or "dialogue"
    edges: dict
    nodes: dict        # required for dialogue, empty for vignette
    persona: str       # required for dialogue, empty for vignette


def _validator_prompt(context: ValidatorContext) -> str:
    """Build the system prompt dynamically from the validation context."""
    output_type = context.get("output_type")
    edges = context.get("edges", {})

    if output_type == "vignette":
        active = set()
        inactive_candidates = set()
        for (p, c), val in edges.items():
            if val > 0:
                active.update([p, c])
            else:
                inactive_candidates.update([p, c])
        inactive = inactive_candidates - active

        required  = [(p, c) for (p, c), v in edges.items() if v > 0]
        forbidden = [(p, c) for (p, c), v in edges.items() if v == 0.0]

        return VALIDATOR_VIGNETTE_PROMPT.format(
            active_components  = "\n".join(f"  - {n}" for n in sorted(active))   or "  (none)",
            inactive_components= "\n".join(f"  - {n}" for n in sorted(inactive)) or "  (none)",
            required_edges     = "\n".join(f"  - {p} → {c}" for p, c in required)  or "  (none)",
            forbidden_edges    = "\n".join(f"  - {p} → {c}" for p, c in forbidden) or "  (none)",
        )

    if output_type == "dialogue":
        nodes = context.get("nodes", {})
        active_nodes   = [n for n, v in nodes.items() if v]
        inactive_nodes = [n for n, v in nodes.items() if not v]
        active_edges   = [(p, c, v) for (p, c), v in edges.items() if v > 0]

        return VALIDATOR_DIALOGUE_PROMPT.format(
            active_components  = "\n".join(f"  - {n}" for n in active_nodes)   or "  (none)",
            inactive_components= "\n".join(f"  - {n}" for n in inactive_nodes) or "  (none)",
            active_edges       = "\n".join(f"  - {p} → {c}: {v:.4f}" for p, c, v in active_edges) or "  (none)",
        )

    return ""  # placeholder for initial agent creation before context is set


class ValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        self._validation_context: ValidatorContext = ValidatorContext(
            output_type="", edges={}, nodes={}, persona=""
        )
        super().__init__(name, role, "", llm)

    def setup_agent(self): 
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SystemMessage(content=_validator_prompt(self._validation_context))
        )
        
    def _set_context(self, output_type: str, context: dict):
        self._validation_context = ValidatorContext(
            output_type=output_type,
            edges=context.get("edges", {}),
            nodes=context.get("nodes", {}),
            persona=context.get("persona", ""),
        )
        self.system_prompt = _validator_prompt(self._validation_context)
        self.agent = self.setup_agent()

    def respond(self, message: str) -> str:
        """Override respond to pass validation context at invoke time."""
        self.messages.append({"role": "user", "content": message})
        result = self.agent.invoke(
            {"messages": self.messages},
            config={"configurable": {"context": self._validation_context}},
        )
        response = result["messages"][-1].content
        self.messages = result["messages"]
        self.log_response(message, response)
        return response

    def build_user_message(self, output: str, output_type: str, context: dict) -> str:
        """Build the user message containing only the output to validate."""
        if output_type == "vignette":
            return f"Vignette:\n{output}"

        if output_type == "dialogue":
            return f"Persona:\n{context['persona']}\n\nMessage:\n{output}"

        raise ValueError(f"Unknown output_type: {output_type!r}")

    def validate_with_retry(self, initial_output: str, output_type: str,
                             context: dict, retry_fn, max_retries: int, label: str) -> tuple[str, list]:
        """
        Validation loop with feedback-driven retry.
        - context : dict with "edges", and "nodes" and "persona" (for dialogue)
        - retry_fn(feedback) -> str : generates a new output given the feedback
        Returns (final_output, attempts) where attempts is a list of
        {"output": str, "passed": bool, "feedback": str | None}.
        """
        output = initial_output
        attempts = []
        for attempt in range(max_retries):
            result = self.validate(output, output_type, context)
            attempts.append({"output": output, "passed": result["passed"], "feedback": result["feedback"]})
            if result["passed"]:
                logger.info("[%s] validation PASS on attempt %d", label, attempt + 1)
                return output, attempts
            logger.warning("[%s] validation FAIL (attempt %d): %s", label, attempt + 1, result["feedback"])
            output = retry_fn(result["feedback"])
        logger.warning("[%s] max retries reached — using last output", label)
        return output, attempts

    def validate(self, output: str, output_type: str, context: dict) -> dict:
        """
        output_type: "vignette" | "dialogue"
        Returns {"passed": bool, "feedback": str | None}
        """
        self._set_context(output_type, context)
        system_prompt = _validator_prompt(self._validation_context)
        user_message = self.build_user_message(output, output_type, context)

        structured_llm = self.llm.with_structured_output(ValidationResult)
        result: ValidationResult = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ])
        self.log_response(user_message, result.model_dump_json(indent=2))

        passed = result.verdict == "PASS"
        if passed:
            logger.info("[%s] verdict: PASS", self.name)
        else:
            logger.info("[%s] verdict: FAIL — %s", self.name, result.reasoning)
        return {"passed": passed, "feedback": None if passed else result.reasoning}
    
