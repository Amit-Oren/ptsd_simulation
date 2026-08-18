import logging
from typing import Optional, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import CONVERSATION_ANALYST_PROMPT
from config.formulation_config import EDGE_MAP_TRIPLES

logger = logging.getLogger(__name__)


class NodeScore(BaseModel):
    score: Literal[0, 1]
    explanation: Optional[str] = None
    quote: str

class EdgeScore(BaseModel):
    score: Literal[0, 1]
    explanation: Optional[str] = None
    quote: str

class AnalysisNodes(BaseModel):
    Triggers: NodeScore
    Negative_Appraisals: NodeScore
    Memory: NodeScore
    Threat: NodeScore
    Maladaptive_Strategies: NodeScore

class AnalysisEdges(BaseModel):
    Triggers__Maladaptive_Strategies: EdgeScore
    Triggers__Threat: EdgeScore
    Triggers__Memory: EdgeScore
    Triggers__Negative_Appraisals: EdgeScore
    Negative_Appraisals__Maladaptive_Strategies: EdgeScore
    Negative_Appraisals__Threat: EdgeScore
    Negative_Appraisals__Memory: EdgeScore
    Negative_Appraisals__Triggers: EdgeScore
    Memory__Maladaptive_Strategies: EdgeScore
    Memory__Threat: EdgeScore
    Memory__Negative_Appraisals: EdgeScore
    Memory__Triggers: EdgeScore
    Threat__Maladaptive_Strategies: EdgeScore
    Threat__Memory: EdgeScore
    Threat__Negative_Appraisals: EdgeScore
    Threat__Triggers: EdgeScore
    Maladaptive_Strategies__Memory: EdgeScore
    Maladaptive_Strategies__Negative_Appraisals: EdgeScore
    Maladaptive_Strategies__Triggers: EdgeScore
    Maladaptive_Strategies__Threat: EdgeScore

class AnalysisResult(BaseModel):
    nodes: AnalysisNodes
    edges: AnalysisEdges
    summary: str




def _format_result(result: AnalysisResult) -> dict:
    nodes = {
        "Triggers":               result.nodes.Triggers.model_dump(),
        "Negative Appraisals":    result.nodes.Negative_Appraisals.model_dump(),
        "Memory":                 result.nodes.Memory.model_dump(),
        "Threat":                 result.nodes.Threat.model_dump(),
        "Maladaptive Strategies": result.nodes.Maladaptive_Strategies.model_dump(),
    }
    edges: dict = {}
    for p, c, field in EDGE_MAP_TRIPLES:
        edges.setdefault(p, {})[c] = getattr(result.edges, field).model_dump()
    return {"nodes": nodes, "edges": edges, "summary": result.summary}


class ConversationAnalystAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, CONVERSATION_ANALYST_PROMPT, llm)

    def setup_agent(self):
        return None

    def respond(self, message: str) -> dict:
        structured_llm = self.llm.with_structured_output(AnalysisResult)
        result = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message},
        ])
        self.log_response(message, result.model_dump_json(indent=2))
        formatted = _format_result(result)
        nodes_scored = sum(v["score"] for v in formatted["nodes"].values())
        edges_scored = sum(
            data["score"]
            for children in formatted["edges"].values()
            for data in children.values()
        )
        logger.info("[%s] analysis — %d/5 nodes, %d/20 edges active",
                    self.name, nodes_scored, edges_scored)
        return formatted
