import logging
from typing import Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import (
    PERSONA_VALIDATOR_DEMOGRAPHICS_SYSTEM_PROMPT,
    PERSONA_VALIDATOR_DEMOGRAPHICS_USER_PROMPT,
    PERSONA_VALIDATOR_SELFREPORT_SYSTEM_PROMPT,
    PERSONA_VALIDATOR_SELFREPORT_USER_PROMPT,
)
from config.formatting import fmt_demographics, fmt_self_report

logger = logging.getLogger(__name__)


class DemographicsValidationResult(BaseModel):
    verdict:              Literal["PASS", "FAIL"]
    issue_fields:         list[str] = []
    issue_explanations:   list[str] = []


class SelfReportValidationResult(BaseModel):
    verdict:              Literal["PASS", "FAIL"]
    issue_components:     list[str] = []
    issue_items:          list[str] = []
    issue_explanations:   list[str] = []


class PersonaValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, "", llm)

    def setup_agent(self):
        return None

    def validate_demographics(self, demographics: dict) -> dict:
        user_prompt = PERSONA_VALIDATOR_DEMOGRAPHICS_USER_PROMPT.format(
            demographics=fmt_demographics(demographics),
        )
        result: DemographicsValidationResult = self._invoke_structured(
            DemographicsValidationResult, PERSONA_VALIDATOR_DEMOGRAPHICS_SYSTEM_PROMPT, user_prompt
        )
        if result is None:
            logger.warning("[%s] demographics validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "issues": [], "problematic_fields": []}

        passed = result.verdict == "PASS"
        issues = [{"field": f, "explanation": e}
                  for f, e in zip(result.issue_fields, result.issue_explanations)]
        logger.info("[%s] demographics: %s", self.name, result.verdict)
        return {"passed": passed, "issues": issues, "problematic_fields": list(result.issue_fields)}

    def validate_self_report(self, demographics: dict, agg_edges: dict, self_report: dict) -> dict:
        from simulation.graph import iter_edges
        active   = "\n".join(f"  - {p} → {c} (strength: {w:.2f})"
                             for p, c, w in iter_edges(agg_edges) if w > 0)
        inactive = "\n".join(f"  - {p} → {c}"
                             for p, c, w in iter_edges(agg_edges) if w == 0)
        cognitive_model_str = (
            f"Active edges:\n{active or '  (none)'}\n\n"
            f"Inactive edges:\n{inactive or '  (none)'}"
        )
        user_prompt = PERSONA_VALIDATOR_SELFREPORT_USER_PROMPT.format(
            demographics=fmt_demographics(demographics),
            cognitive_model=cognitive_model_str,
            self_report=fmt_self_report(self_report),
        )
        result: SelfReportValidationResult = self._invoke_structured(
            SelfReportValidationResult, PERSONA_VALIDATOR_SELFREPORT_SYSTEM_PROMPT, user_prompt
        )
        if result is None:
            logger.warning("[%s] self-report validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "issues": [], "problematic_items": {}}

        passed = result.verdict == "PASS"
        issues = [{"component": c, "item": it, "explanation": e}
                  for c, it, e in zip(result.issue_components, result.issue_items, result.issue_explanations)]
        problematic_items: dict[str, list[str]] = {}
        for c, it in zip(result.issue_components, result.issue_items):
            if it.startswith(c + ":"):
                it = it[len(c) + 1:].strip()
            key = it.split(":")[0].strip()
            problematic_items.setdefault(c, []).append(key)
        logger.info("[%s] self-report: %s", self.name, result.verdict)
        return {"passed": passed, "issues": issues, "problematic_items": problematic_items}
