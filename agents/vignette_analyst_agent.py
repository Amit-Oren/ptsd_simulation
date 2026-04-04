import logging
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import VIGNETTE_ANALYST_PROMPT

logger = logging.getLogger(__name__)


class VignetteEdgeScore(BaseModel):
    weight: int   # 0–10; divided by 10 on output
    explanation: str = ""
    quote: str = ""


class VignetteEdges(BaseModel):
    Triggers__Maladaptive_Strategies: VignetteEdgeScore
    Triggers__Threat: VignetteEdgeScore
    Triggers__Memory: VignetteEdgeScore
    Triggers__Negative_Appraisals: VignetteEdgeScore
    Negative_Appraisals__Maladaptive_Strategies: VignetteEdgeScore
    Negative_Appraisals__Threat: VignetteEdgeScore
    Negative_Appraisals__Memory: VignetteEdgeScore
    Negative_Appraisals__Triggers: VignetteEdgeScore
    Memory__Maladaptive_Strategies: VignetteEdgeScore
    Memory__Threat: VignetteEdgeScore
    Memory__Negative_Appraisals: VignetteEdgeScore
    Memory__Triggers: VignetteEdgeScore
    Threat__Maladaptive_Strategies: VignetteEdgeScore
    Threat__Memory: VignetteEdgeScore
    Threat__Negative_Appraisals: VignetteEdgeScore
    Threat__Triggers: VignetteEdgeScore
    Maladaptive_Strategies__Memory: VignetteEdgeScore
    Maladaptive_Strategies__Negative_Appraisals: VignetteEdgeScore


class VignetteAnalysisResult(BaseModel):
    edges: VignetteEdges


def _format_result(result: VignetteAnalysisResult) -> dict:
    e = result.edges

    def _normalise(score: VignetteEdgeScore) -> dict:
        d = score.model_dump()
        d["weight"] = round(d["weight"] / 10, 1)
        return d

    return {
        "Triggers--Maladaptive Strategies":            _normalise(e.Triggers__Maladaptive_Strategies),
        "Triggers--Threat":                            _normalise(e.Triggers__Threat),
        "Triggers--Memory":                            _normalise(e.Triggers__Memory),
        "Triggers--Negative Appraisals":               _normalise(e.Triggers__Negative_Appraisals),
        "Negative Appraisals--Maladaptive Strategies": _normalise(e.Negative_Appraisals__Maladaptive_Strategies),
        "Negative Appraisals--Threat":                 _normalise(e.Negative_Appraisals__Threat),
        "Negative Appraisals--Memory":                 _normalise(e.Negative_Appraisals__Memory),
        "Negative Appraisals--Triggers":               _normalise(e.Negative_Appraisals__Triggers),
        "Memory--Maladaptive Strategies":              _normalise(e.Memory__Maladaptive_Strategies),
        "Memory--Threat":                              _normalise(e.Memory__Threat),
        "Memory--Negative Appraisals":                 _normalise(e.Memory__Negative_Appraisals),
        "Memory--Triggers":                            _normalise(e.Memory__Triggers),
        "Threat--Maladaptive Strategies":              _normalise(e.Threat__Maladaptive_Strategies),
        "Threat--Memory":                              _normalise(e.Threat__Memory),
        "Threat--Negative Appraisals":                 _normalise(e.Threat__Negative_Appraisals),
        "Threat--Triggers":                            _normalise(e.Threat__Triggers),
        "Maladaptive Strategies--Memory":              _normalise(e.Maladaptive_Strategies__Memory),
        "Maladaptive Strategies--Negative Appraisals": _normalise(e.Maladaptive_Strategies__Negative_Appraisals),
    }


class VignetteAnalystAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, VIGNETTE_ANALYST_PROMPT, llm)

    def _setup_agent(self):
        return None

    def respond(self, message: str) -> dict:
        structured_llm = self.llm.with_structured_output(VignetteAnalysisResult)
        result = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": message},
        ])
        self.log_response(message, result.model_dump_json(indent=2))
        formatted = _format_result(result)
        logger.info("[%s] vignette analysis complete — %d edges scored",
                    self.name, len(formatted))
        return formatted
