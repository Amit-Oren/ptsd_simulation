"""
The narrator - turns a played scene into a short factual account.
"""
import re

from simulation.factory import build_llm
from scenarios.scene.director import SceneCard, format_script

SYSTEM = """You turn the script of a short scene into a factual account of
what happened, for research use.

Write it to the main character, as "you", in the past tense. One short
paragraph, as long as the scene needs and no longer.

Keep:
  - the setting, briefly, once at the start, in natural words - never the
    label "surroundings". Don't repeat background details (a hum, a corridor,
    the rain) after that unless they change.
  - everything that happened, in order - every action and every event
  - what people said, in their own words, in quotes. Make it clear who spoke,
    the way a person telling the story would: "The clerk took the card and
    said, ..." - name the speaker once per sentence, never twice ("the clerk
    took the card and the clerk said"). For your own lines, just "you said"
    or let your words follow your action - not "and you said" every time.
  - who each person is: the first time someone appears, use their label from
    the cast list, written from your side - "their partner" becomes "your
    partner", never "their". "Your" is only for people marked as someone you know
    ("your partner", "your coworker"); everyone else is "the" or "a" ("the
    clerk", "a stranger in line"). After the first time, use a short form -
    never the full label again. Nobody has a name - refer to people only
    this way.

Leave out:
  - feelings, thoughts, intentions, and what anything meant
  - how something was said or done when it implies a feeling or intention:
    "dryly", "sarcastically", "kindly", "slips in", "nervously"
  - words that link events by cause: "so", "because", "after that"

Do not add anything that is not in the script. No title, no commentary -
return only the account."""

USER = """Setting: {setting}
"You" is the person called "the main character" in the script: {main}.
The other people:
{cast}

Script:
{script}"""

REWRITE = """Your account uses these words, which describe a feeling, an intention or
a cause: {words}. Rewrite it without them. Change nothing else."""

# The prompt already forbids these, and the narrator still let "slipped into"
# through on its first run. Checked in code instead of trusted to the prompt.
# Quoted speech is skipped - people are allowed to say "so" out loud.
_BANNED = re.compile(
    r"\b(so|because|after that|slip\w*|nervous\w*|awkward\w*|dryly|sarcastic\w*|"
    r"kindly|hesitat\w*|reluctant\w*|embarrass\w*|anxious\w*|sheepish\w*)\b",
    re.IGNORECASE)
_QUOTED = re.compile(r"[\"“][^\"”]*[\"”]")

# Relationships from the pool that are not someone the main character knows -
# the narrator writes "the clerk", not "your clerk", for these. Only a known /
# not-known flag reaches the narrator; given the category itself, it copied it
# into the account word for word.
_NOT_KNOWN = ("stranger", "service")


def _known(relationship: str) -> bool:
    low = relationship.lower()
    return not any(word in low for word in _NOT_KNOWN)


def banned_words(text: str) -> list[str]:
    outside_quotes = _QUOTED.sub("", text)
    return sorted({m.group(0).lower() for m in _BANNED.finditer(outside_quotes)})


class Narrator:
    """One call per scene, plus one rewrite if banned words get through."""

    def __init__(self, model: str = "gpt-5.4-mini", temperature: float = 0.3):
        self.model = model
        self.llm = build_llm(model, temperature=temperature)

    def narrate(self, setting: str, card: SceneCard, script: list[dict]) -> str:
        cast = "\n".join(
            f"- {c.role} ({'someone you know' if _known(c.relationship) else 'someone you do not know'})"
            for c in card.secondary)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(
                setting=setting, main=card.main.brief, cast=cast or "(none)",
                script=format_script(script))},
        ]
        text = self._clean(self.llm.invoke(messages).content)

        words = banned_words(text)
        if words:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": REWRITE.format(words=", ".join(words))}]
            text = self._clean(self.llm.invoke(messages).content)
        return text

    @staticmethod
    def _clean(content) -> str:
        return " ".join(str(content).split())
