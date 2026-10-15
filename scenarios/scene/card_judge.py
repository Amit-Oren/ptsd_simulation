"""
The card judge - checks a scene card before it is played.

Most of what has gone wrong in a played scene was already wrong on the card,
and checking a card costs one call where playing it costs about fourteen. So
the card is judged first, on the three things that decide whether a scene is
usable at all.

A failure on FIT means the ingredients can't work together - resample them.
A failure on LANDS or NEUTRAL MAIN means the director set them up badly - ask
it again with the problems as feedback.
"""
from typing import cast

from pydantic import BaseModel, Field

from simulation.factory import build_llm
from scenarios.scene.director import SceneCard, describe_card


class Check(BaseModel):
    passed: bool
    quote: str = Field(default="",
                       description="The exact words on the card at fault. Empty if passed.")


class CardVerdict(BaseModel):
    fit: Check
    lands: Check
    neutral_main: Check

    @property
    def passed(self) -> bool:
        return self.fit.passed and self.lands.passed and self.neutral_main.passed

    @property
    def problems(self) -> list[str]:
        checks = {"fit": self.fit, "lands": self.lands, "neutral main": self.neutral_main}
        return [f"{name}: {c.quote}" for name, c in checks.items() if not c.passed]


SYSTEM = """You check the setup of a short everyday scene before it is acted out.
You are shown the setting and the scene card. Three checks.

FIT
  Does everyone and everything belong in this place, at this time?
  - Is each relationship literally true? A pharmacist is not someone with
    authority over a customer; a nurse is not someone the patient works with.
  - Does the place make sense at this time? A bank is not open late at night.
  - Is anyone only reached by phone shown as physically present?
  - Is every object concrete - something a reader could picture? "A
    prescription bag" passes; "the item", "the document", "the handoff" fail.
  Unusual is not a failure - only what could not plausibly happen.

LANDS
  Does each secondary character's task happen to the main character - their
  path, their space, their plans, something said to or about them?
  It fails if a task happens beside them and would not reach them: someone
  blocking an elevator the main character never needs to use.
  Only secondary characters are checked here. Entities are not - a light
  flickering or a phone ringing does not need to happen to anyone.

NEUTRAL MAIN
  Does the main character stay plain?
  - The circumstance has no reason that was invented for it: "has to stay
    until the appointment is finished" passes, "has to stay because it is
    raining" fails. A circumstance on its own is never a failure - being
    somewhere for the first time, arriving late or carrying a lot all pass.
    Only a reason added to it fails.
  - The goal is an ordinary reason to be here and does not itself create an
    event.
  - The main character is not the one in charge here - not the teacher, the
    host, the director, or anyone the others are waiting on to lead.

For each check that fails, quote the exact words on the card at fault. If
you cannot quote them, it has not failed."""

USER = """Setting: {setting}

Scene card:
{card}"""


class CardJudge:
    """One call per card."""

    def __init__(self, model: str = "gpt-5.4"):
        self.model = model
        self.llm = build_llm(model, temperature=0.0).with_structured_output(CardVerdict)

    def check(self, setting: str, card: SceneCard) -> CardVerdict:
        return cast(CardVerdict, self.llm.invoke([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(setting=setting, card=describe_card(card))},
        ]))
