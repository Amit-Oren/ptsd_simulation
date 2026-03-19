from pydantic import BaseModel
from .base_agent import BaseAgent
from config.prompts import VALIDATOR_AGENT_PROMPT


class ValidationResult(BaseModel):
    valid: bool
    reason: str


class ValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, VALIDATOR_AGENT_PROMPT, llm)

    def _setup_agent(self):
        return None

    def validate(self, persona: str, message: str, nodes: dict, edges: dict) -> ValidationResult:
        nodes_str = "\n".join(f"  - {node}: {val}" for node, val in nodes.items())
        edges_str = "\n".join(f"  - {p} → {c}: {val:.4f}" for (p, c), val in edges.items())

        input_text = (
            f"Persona:\n{persona}\n\n"
            f"Nodes:\n{nodes_str}\n\n"
            f"Edges:\n{edges_str}\n\n"
            f"Message:\n{message}"
        )

        structured_llm = self.llm.with_structured_output(ValidationResult)
        result = structured_llm.invoke([
            {"role": "system", "content": VALIDATOR_AGENT_PROMPT},
            {"role": "user", "content": input_text},
        ])
        self.log_response(input_text, str(result))
        return result
