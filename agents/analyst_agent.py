from typing import Optional, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import ANALYST_AGENT_PROMPT


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
    Memory__Maladaptive_Strategies: EdgeScore
    Memory__Threat: EdgeScore
    Threat__Maladaptive_Strategies: EdgeScore

class AnalysisResult(BaseModel):
    nodes: AnalysisNodes
    edges: AnalysisEdges


def _format_result(result: AnalysisResult) -> dict:
    nodes = {
        "Triggers":               result.nodes.Triggers.model_dump(),
        "Negative Appraisals":    result.nodes.Negative_Appraisals.model_dump(),
        "Memory":                 result.nodes.Memory.model_dump(),
        "Threat":                 result.nodes.Threat.model_dump(),
        "Maladaptive Strategies": result.nodes.Maladaptive_Strategies.model_dump(),
    }
    edges = {
        "Triggers--Maladaptive Strategies":            result.edges.Triggers__Maladaptive_Strategies.model_dump(),
        "Triggers--Threat":                            result.edges.Triggers__Threat.model_dump(),
        "Triggers--Memory":                            result.edges.Triggers__Memory.model_dump(),
        "Triggers--Negative Appraisals":               result.edges.Triggers__Negative_Appraisals.model_dump(),
        "Negative Appraisals--Maladaptive Strategies": result.edges.Negative_Appraisals__Maladaptive_Strategies.model_dump(),
        "Negative Appraisals--Threat":                 result.edges.Negative_Appraisals__Threat.model_dump(),
        "Negative Appraisals--Memory":                 result.edges.Negative_Appraisals__Memory.model_dump(),
        "Memory--Maladaptive Strategies":              result.edges.Memory__Maladaptive_Strategies.model_dump(),
        "Memory--Threat":                              result.edges.Memory__Threat.model_dump(),
        "Threat--Maladaptive Strategies":              result.edges.Threat__Maladaptive_Strategies.model_dump(),
    }
    return {"nodes": nodes, "edges": edges}


class AnalystAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, ANALYST_AGENT_PROMPT, llm)

    def _setup_agent(self):
        return None

    def respond(self, message: str) -> dict:
        structured_llm = self.llm.with_structured_output(AnalysisResult)
        result = structured_llm.invoke(message)
        self.log_response(message, str(result))
        return _format_result(result)
