"""
The judge — decides whether a written scenario is usable.
"""
import re

from pydantic import BaseModel, Field

from simulation.factory import build_llm
from scenarios.pools import describe_material

# The FACTUAL check's own instructions name these words as a failure, and the
# model still let 13/117 through in one batch. Caught deterministically
# instead of trusted to a reading the model keeps getting wrong.
_CAUSAL_LINK = re.compile(r"\b(so|because|after that)\b", re.IGNORECASE)


def _causal_link(text: str) -> str:
    match = _CAUSAL_LINK.search(text)
    return match.group(0) if match else ""


class Check(BaseModel):
    passed: bool
    quote: str = Field(
        default="",
        description="The exact words at fault, quoted from the account. "
                    "Empty if passed - if you cannot quote them, it is not "
                    "a problem.")


class Verdict(BaseModel):
    factual: Check = Field(
        description="States only what the person saw, heard, smelled or did. "
                    "Adds no feeling, motive, meaning or bodily reaction that "
                    "was not in what it was given.")
    coherent: Check = Field(
        description="Time, place, people and action fit together, and nothing "
                    "contradicts anything else.")
    plausible: Check = Field(
        description="The people, place, activity and event fit together and "
                    "are understandable without inventing missing context - "
                    "a material-level problem, not a wording one.")
    specific: Check = Field(
        description="Every action is concrete and attributed to an "
                    "identifiable person - a wording-level problem, fixable "
                    "by rewriting without changing what happened.")
    natural: Check = Field(
        description="Reads as an account of something that happened, not as a "
                    "list of the details it was given.")

    @property
    def passed(self) -> bool:
        return bool(self.factual.passed and self.coherent.passed and self.plausible.passed
                    and self.specific.passed and self.natural.passed)

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.factual.passed:
            out.append(f"factual: {self.factual.quote}")
        if not self.coherent.passed:
            out.append(f"coherent: {self.coherent.quote}")
        if not self.plausible.passed:
            out.append(f"plausible: {self.plausible.quote}")
        if not self.specific.passed:
            out.append(f"specific: {self.specific.quote}")
        if not self.natural.passed:
            out.append(f"natural: {self.natural.quote}")
        return out


SYSTEM = """You check short factual accounts of everyday events.

You are shown the situation the account was written from, then the account.

FACTUAL
  Scan for anything the account asserts about an inner state: a feeling, a
  motive, a meaning, a bodily reaction. Then scan for words that smuggle a
  judgement in - only, still, even, finally, unusually, just, seemed, appeared,
  as if, longer than, pointedly - and ask of each whether it reports or
  evaluates.
  bad examples:
  - you felt on edge
  - your heart was going
  - he looked at you longer than he needed to

COHERENT
  "Starting point", "What happened", "What you did" and "How it was left"
  are the causal core - the account must represent all four, even if the
  wording differs from how they were given. Missing one of these is a
  failure, not an omission: a scenario built on a delay still needs its
  outcome, or there is no event left to measure.
  Every other line is optional context. Take each one in turn, find it in the
  account, and check nothing else contradicts it - a context detail simply
  left out is not a failure.

PLAUSIBLE  (a material problem - rewriting the words cannot fix this)
  Ask whether the people, place, activity and event fit together as one
  occasion a person could actually have been in - not whether it is common,
  only whether it hangs together. Flag:
  - a relationship and activity that read as randomly combined (a parent
    "doing work" for you at the beach, exercising while someone performs a
    service for you)
  - a place and an implied setting from the event that contradict each other
  - an account where no concrete event actually happens
  Being unusual is not a failure - an unusual situation can still be
  plausible. This check is about fit, not wording - if the fix would be
  rewriting a sentence, that is NATURAL or SPECIFIC's problem, not this one.

SPECIFIC  (a wording problem - rewriting without changing the material can fix this)
  Read for vague placeholders that name a category rather than an action:
  "changed the arrangement", "did something together", "handled it",
  "worked things out", "it was sorted out". Each should be a reader could
  picture, done by someone identifiable.
  Ask of each sentence: does it add information the one before it did not
  already give. Fail for padding or restatement with different words.

NATURAL
  Read each sentence and check it parses and is not a restatement of one
  before it. Plain prose and mild awkwardness are fine. Fail this only for
  sentences that do not work, or for an account that names each given detail in
  order like a form.

Quote the exact words at fault in each check you fail. If you cannot quote
them, it is not a failure. Most accounts should pass."""

USER = """The situation it was written from:
{material}

The account:
{text}"""


class Judge:
    """Validates scenario prose. One call per scenario, plus one after a repair."""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.llm = build_llm(model, temperature=0.0).with_structured_output(Verdict)

    def check(self, material: dict, text: str) -> Verdict:
        verdict = self.llm.invoke([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(
                material=describe_material(material), text=text)},
        ])
        link = _causal_link(text)
        if link and verdict.factual.passed:
            verdict.factual = Check(passed=False, quote=link)
        return verdict

    @staticmethod
    def passed(verdict: Verdict) -> bool:
        return verdict.passed
