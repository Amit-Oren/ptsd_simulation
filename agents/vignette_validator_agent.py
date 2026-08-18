import logging
from typing import Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import VALIDATOR_VIGNETTE_SYSTEM_PROMPT, VALIDATOR_VIGNETTE_USER_PROMPT
from config.formulation_config import EDGE_EXAMPLES
from simulation.graph import iter_edges

logger = logging.getLogger(__name__)


def _normalize_edge(edge: str) -> str:
    return edge.replace("->", "→").replace("→", " → ").replace("  →  ", " → ").strip()


class ValidationResult(BaseModel):
    verdict:                 Literal["PASS", "FAIL"]
    violation_edges:         list[str] = []
    violation_explanations:  list[str] = []


def _build_system_prompt(edges: dict) -> str:
    active: set[str] = set()
    required: list[tuple[str, str]] = []
    for p, c, val in iter_edges(edges):
        if val > 0:
            active.update([p, c])
            required.append((p, c))
    return VALIDATOR_VIGNETTE_SYSTEM_PROMPT.format(
        active_components="\n".join(f"  - {n}" for n in sorted(active)) or "  (none)",
        required_edges="\n".join(f"  - {p} → {c}" for p, c in required) or "  (none)",
    )


class VignetteValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, "", llm)

    def setup_agent(self):
        return None

    def validate(self, vignette: str, context: dict) -> dict:
        edges = context.get("edges", {})
        self.system_prompt = _build_system_prompt(edges)
        user_prompt = VALIDATOR_VIGNETTE_USER_PROMPT.format(vignette=vignette)

        result: ValidationResult = self._invoke_structured(ValidationResult, self.system_prompt, user_prompt)
        if result is None:
            logger.warning("[%s] vignette validation parsing failed — defaulting to FAIL", self.name)
            return {"passed": False, "violations": [], "feedback": "Validation could not be parsed. Please rewrite the vignette cleanly."}

        required_edges = {f"{p} → {c}" for p, c, v in iter_edges(edges) if v > 0}
        violations = [{"edge": e, "explanation": ex}
                      for e, ex in zip(result.violation_edges, result.violation_explanations)
                      if _normalize_edge(e) in required_edges]
        feedback = self._build_feedback(violations)
        passed = len(violations) == 0
        logger.info("[%s] %s", self.name, "PASS" if passed else f"FAIL — {len(violations)} violation(s)")
        return {"passed": passed, "violations": violations, "feedback": feedback}

    @staticmethod
    def _build_feedback(violations: list[dict]) -> str | None:
        if not violations:
            return None
        lines = ["REQUIRED EDGES MISSING:"]
        for v in violations:
            parts = v["edge"].split(" → ")
            src = parts[0].strip() if len(parts) == 2 else ""
            tgt = parts[1].strip() if len(parts) == 2 else ""
            lines.append(f"- {v['edge']}: {v['explanation']}")
            for ex in EDGE_EXAMPLES.get((src, tgt), []):
                lines.append(f'  e.g. "{ex}"')
        return "\n".join(lines)

    def validate_with_retry(self, initial_vignette: str, context: dict,
                            retry_fn, max_retries: int, label: str) -> tuple[str, list]:
        vignette = initial_vignette
        attempts = []
        for attempt in range(max_retries + 1):
            result = self.validate(vignette, context)
            attempts.append({
                "vignette":        vignette,
                "passed":          result["passed"],
                "violations":      result["violations"],
                "violation_count": len(result["violations"]),
                "feedback":        result["feedback"],
            })
            if result["passed"]:
                logger.info("[%s] PASS on attempt %d", label, attempt + 1)
                return vignette, attempts
            logger.warning("[%s] FAIL on attempt %d", label, attempt + 1)
            if attempt < max_retries:
                vignette = retry_fn(result["feedback"])
        logger.warning("[%s] max retries reached — using last vignette", label)
        return vignette, attempts
