"""
Plays a scene: the director cues each turn, the cued actor acts.

Enforced here rather than left to the director, which has broken each of these
at least once:
  - an entity acts at most once
  - every secondary character and entity acts at least once: anyone still
    waiting is forced in once the turns left barely cover them, leaving the
    last turn for a reaction where possible
  - no character acts more than twice in a row, and the main character never
    twice in a row - with nothing new to react to, they only echo themselves
  - a character whose task is silence doesn't speak, and the main character
    doesn't apologise - a line that breaks either is redone once with a
    correction, since both rules were in the prompts and both still broke
"""
import re

from scenarios.scene.actors import CharacterActor, EntityActor
from scenarios.scene.director import MAIN, Director, SceneCard

_SILENT = ("without saying anything", "without a word", "does not respond",
           "doesn't respond", "does not answer", "doesn't answer", "without answering",
           "says nothing", "stays silent", "in silence")
_SPEECH = re.compile(r"[\"“]")
_APOLOGY = re.compile(r"\b(sorry|apologi[sz]\w*)\b", re.IGNORECASE)


def _correction(who: str, line: str, silent: set[str]) -> str:
    """A nudge to redo `line` with, or "" if it breaks no rule."""
    if who in silent and _SPEECH.search(line):
        return "Your task is to say nothing - redo your turn with no speech at all."
    if who == MAIN and _APOLOGY.search(line):
        return "Don't apologise - redo your turn without saying sorry."
    return ""


def _resolve(name: str, actors: dict) -> str | None:
    """The director's pick, matched to an actor even if it's spelled loosely."""
    if name in actors:
        return name
    low = name.strip().lower()
    for actor_name in actors:
        if actor_name.lower() == low or low in actor_name.lower() or actor_name.lower() in low:
            return actor_name
    return None


def play_scene(director: Director, setting: str, card: SceneCard, turns: int = 6,
               model: str = "gpt-5.4-mini", temperature: float = 0.9) -> list[dict]:
    actors = {MAIN: CharacterActor.main(card.main, setting, model, temperature)}
    for c in card.secondary:
        actors[c.role] = CharacterActor.secondary(c, setting, model, temperature)
    for e in card.entities:
        actors[e.name] = EntityActor(e, setting, model, temperature)

    secondaries = [c.role for c in card.secondary]
    characters = [MAIN, *secondaries]
    entities = {e.name for e in card.entities}
    silent = {c.role for c in card.secondary if any(s in c.task.lower() for s in _SILENT)}
    acted: set[str] = set()
    script: list[dict] = []

    for turn in range(turns):
        turns_left = turns - turn
        pending = [n for n in (*secondaries, *entities) if n not in acted]

        if pending and len(pending) >= max(turns_left - 1, 1):
            who = pending[0]
            nudge = "" if who in entities else "now is the moment for your task"
        else:
            cue = director.next_turn(setting, card, script, turns_left)
            who, nudge = _resolve(cue.who, actors), cue.nudge
            if who is None or (who in entities and who in acted):
                who, nudge = MAIN, ""

            repeat_main = who == MAIN and script and script[-1]["who"] == MAIN
            repeat_any = len(script) >= 2 and script[-1]["who"] == script[-2]["who"] == who
            if repeat_main or repeat_any:
                who, nudge = next(c for c in characters if c != who), ""

        line = actors[who].act(script, nudge)
        correction = _correction(who, line, silent)
        if correction:
            line = actors[who].act(script, correction)
        acted.add(who)
        script.append({"who": who, "kind": "entity" if who in entities else "character",
                       "text": line, "nudge": nudge})

    return script
