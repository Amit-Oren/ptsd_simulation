"""Agent responsible for validating dialogue messages against the patient's cognitive graph."""

import logging
from typing import TypedDict, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import VALIDATOR_DIALOGUE_PROMPT
from simulation.graph import iter_edges

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reasoning: str


class ValidatorContext(TypedDict):
    edges:    dict
    nodes:    dict
    vignette: str


def _build_dialogue_prompt(context: ValidatorContext) -> str:
    nodes = context.get("nodes", {})
    edges = context.get("edges", {})
    active_nodes   = [n for n, v in nodes.items() if v]
    inactive_nodes = [n for n, v in nodes.items() if not v]
    return VALIDATOR_DIALOGUE_PROMPT.format(
        active_components  = "\n".join(f"  - {n}" for n in active_nodes)   or "  (none)",
        inactive_components= "\n".join(f"  - {n}" for n in inactive_nodes) or "  (none)",
        active_edges       = "\n".join(
            f"  - {p} → {c}: {v:.4f}" for p, c, v in iter_edges(edges) if v > 0
        ) or "  (none)",
    )


class ValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        self._validation_context: ValidatorContext = ValidatorContext(edges={}, nodes={}, vignette="")
        super().__init__(name, role, "", llm)

    def setup_agent(self):
        return None

    def validate(self, output: str, output_type: str, context: dict) -> dict:
        self._validation_context = ValidatorContext(
            edges=context.get("edges", {}),
            nodes=context.get("nodes", {}),
            vignette=context.get("vignette", ""),
        )
        system_prompt = _build_dialogue_prompt(self._validation_context)
        user_message  = f"Vignette:\n{self._validation_context['vignette']}\n\nMessage:\n{output}"

        structured_llm = self.llm.with_structured_output(ValidationResult)
        result: ValidationResult = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ])
        self.log_response(user_message, result.model_dump_json(indent=2))

        passed = result.verdict == "PASS"
        logger.info("[%s] verdict: %s%s", self.name, result.verdict,
                    f" — {result.reasoning}" if not passed else "")
        return {"passed": passed, "feedback": None if passed else result.reasoning}

    def validate_with_retry(self, initial_output: str, output_type: str,
                             context: dict, retry_fn, max_retries: int, label: str) -> tuple[str, list]:
        output   = initial_output
        attempts = []
        for attempt in range(max_retries + 1):
            result = self.validate(output, output_type, context)
            attempts.append({"output": output, "passed": result["passed"], "feedback": result["feedback"]})
            if result["passed"]:
                logger.info("[%s] validation PASS on attempt %d", label, attempt + 1)
                return output, attempts
            logger.warning("[%s] validation FAIL (attempt %d): %s", label, attempt + 1, result["feedback"])
            if attempt < max_retries:
                output = retry_fn(result["feedback"])
        logger.warning("[%s] max retries reached — using last output", label)
        return output, attempts
