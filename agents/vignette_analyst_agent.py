import logging
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import VIGNETTE_ANALYST_PROMPT
from config.formulation_config import EDGE_MAP_TRIPLES

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
    Maladaptive_Strategies__Triggers: VignetteEdgeScore
    Maladaptive_Strategies__Threat: VignetteEdgeScore


class VignetteAnalysisResult(BaseModel):
    edges: VignetteEdges




def _format_result(result: VignetteAnalysisResult) -> dict:
    def _normalise(score: VignetteEdgeScore) -> dict:
        d = score.model_dump()
        d["weight"] = round(d["weight"] / 10, 1)
        return d

    edges: dict = {}
    for p, c, field in EDGE_MAP_TRIPLES:
        edges.setdefault(p, {})[c] = _normalise(getattr(result.edges, field))
    return edges


class VignetteAnalystAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, VIGNETTE_ANALYST_PROMPT, llm)

    def setup_agent(self):
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
