"""
The narrator writes one scenario from the pool set

"""
from simulation.factory import build_llm
from scenarios.pools import describe_material

SYSTEM = """
Transform the given scenario specification into a short, natural account of
one plausible real-life event.

Write 3-4 short sentences in the second person ("you") and past tense.

Make the main event the center of the narrative. Use the other details only
when they fit naturally into the same situation. Prioritize a coherent,
plausible event over including every field.

Describe concrete actions and observable events. Do not describe feelings,
thoughts, intentions, memories, or psychological meaning.

Do not invent additional events or use vague actions to fill space. Do not
add decorative details that do not contribute to the account. If a detail
does not fit naturally, leave it out.

Use plain, ordinary language. No title, markdown, or commentary. Return only
the narrative.
"""

REWRITE = """That account has a problem:
{problems}

Rewrite it, fixing only the stated problem.
Preserve the original situation and all details that do not need to change.
Follow the original instructions."""


class Narrator:
    """Writes scenario prose. One call per scenario, plus one on a rejection."""

    def __init__(self, model: str = "gpt-5.4-mini", temperature: float = 0.9):
        self.model = model
        self.llm = build_llm(model, temperature=temperature)

    @staticmethod
    def _clean(content) -> str:
        return " ".join(str(content).strip().strip('"').split())

    def _prompt(self, material: dict) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{describe_material(material)}\n\nWrite the account."},
        ]

    def write(self, material: dict) -> str:
        return self._clean(self.llm.invoke(self._prompt(material)).content)

    def rewrite(self, material: dict, previous: str, problems: list[str]) -> str:
        """One repair attempt, with the judge's reasons.
        """
        messages = self._prompt(material) + [
            {"role": "assistant", "content": previous},
            {"role": "user", "content": REWRITE.format(
                problems="\n".join("- " + p for p in problems))},
        ]
        return self._clean(self.llm.invoke(messages).content)
