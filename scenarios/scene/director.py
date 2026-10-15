"""
The director - turns a setting and a few sampled ingredients into a scene card,
then cues the scene turn by turn while it is played.

Nobody has a personal name. Left to choose, the model named almost every main
character "Maya", so characters are known by their role instead ("the
lecturer", "a classmate in the next seat"), and the main character is always
just "the main character" - the narrator turns that into "you".
"""
from typing import cast

from pydantic import BaseModel, Field

from simulation.factory import build_llm

MAIN = "the main character"


class MainCharacter(BaseModel):
    brief: str = Field(description="Who they are, in a short phrase, with no name.")
    goal: str = Field(description="The ordinary reason they are here. "
                                  "Neutral - it does not create any event.")
    circumstance: str = Field(description="The given circumstance, made concrete for "
                                          "this setting, stated plainly.")


class SecondaryCharacter(BaseModel):
    role: str = Field(description="How they are referred to in the scene: a short, natural "
                                  "label with no personal name, the way a story would "
                                  "describe them - 'the lecturer', 'a coworker in the "
                                  "waiting room'. Never the relationship category itself. "
                                  "Different for every character.")
    relationship: str = Field(description="Who they are to the main character - "
                                          "the given relationship.")
    task: str = Field(description="The one thing they do at some point in the scene - "
                                  "something others can see or hear.")


class Entity(BaseModel):
    name: str = Field(description="What it is, e.g. 'a student's phone'.")
    task: str = Field(description="What it does at some point in the scene, e.g. 'rings'.")


class SceneCard(BaseModel):
    surroundings: str = Field(description="The given surroundings, if they fit this place. "
                                          "Empty if they don't.")
    main: MainCharacter
    secondary: list[SecondaryCharacter]
    entities: list[Entity]


class Cue(BaseModel):
    who: str = Field(description="Who acts next, exactly as on the scene card: "
                                 "'the main character', a character's role, or an "
                                 "entity's name.")
    nudge: str = Field(default="",
                       description="Optional private direction for that character, e.g. "
                                   "'now is a good moment for your task'. Empty if none.")


SYSTEM = """You are the director of a short, ordinary scene from everyday life.
Your mession is to set the scene up.
Do no not write it.
Actors will improvise it from your setup.

You are given a setting, the main character's circumstance, one or more task
types each paired with a relationship, and possibly an entity type. Decide:

SURROUNDINGS
  Keep the given surroundings only if they fit this place. If they don't - a
  "crowded" bedroom, a car with "one way in or out" - leave them empty rather
  than bending the scene around them.

MAIN CHARACTER
  Who they are and their goal: the ordinary reason they are here ("attend the
  morning lecture", "pay a bill"). The main character never causes anything
  to happen - their goal is neutral. They are never the one in charge - not
  the teacher, the host, or anyone the others wait on to lead.
  Make the given circumstance concrete for this setting ("arrives late"
  becomes "arrives ten minutes after the lecture has started"). Keep it
  plain: no manner words ("slips in", "hurries nervously").
  Never give the circumstance a reason you were not given, in the
  circumstance or in the goal. "Has to stay until what they came for is
  finished" becomes "has to stay until the appointment is finished" - it does
  not become "has to stay because it is raining", and the goal does not
  become "wait for the rain to stop". If no reason was given, there is none.
  If it is "nothing unusual", keep it as that.

SECONDARY CHARACTERS
  Exactly one for each task type you were given - never fewer. Each has a
  role, the relationship to the main character it was paired with, and a
  task: the task type made concrete for this setting.
  The role is how a story would describe this person - "a coworker in the
  waiting room", "her partner", "the pharmacist" - not the relationship
  category you were given. Never "someone they know well" as a role.
  Someone who is only reached by phone or message is somewhere else: say so
  in the role ("a family member calling from home"). They never appear in
  the room - their task happens through the main character's phone.
  The relationship must be literally true. If it does not naturally belong in
  this place, keep the person's own reason for being here - they came along,
  they happen to be there - rather than giving them a job that bends the
  relationship. A store manager is not someone with authority over a
  customer; a nurse is not someone the patient works with.

ENTITIES
  If you were given an entity type, include exactly one entity of that type
  that fits the setting, with its task: what it does. Otherwise include none.
  An entity's task describes only what the entity does - it never mentions
  what any character does before or after it - and it is a separate event
  from every character's task: if a character calls someone, the entity
  cannot be that phone ringing.

Rules for every task:
  - Something others could see or hear.
  - Never a thought or a feeling.
  - Concrete: "makes a joke to the class about the late arrival", not
    "draws attention to her".
  - Describe what they do, never their exact words - no quotes or dialogue
    in a task. The actor chooses the words.
  - Ambiguous where possible: it could be read as friendly or as unkind, and
    the scene should not decide which.
  - It touches the main character's goal or situation. A blocked way blocks
    the path they need to take; someone who goes somewhere goes without
    them when they expected to go too. Set the scene up so the task lands on
    them, not beside them.
  - Done fully. No softening words: "for a moment", "briefly", "slightly",
    "a little".
  - If the task needs the main character to do something first - ask, wave,
    hand something over - make it the most ordinary version: a simple
    question, a greeting. Never a demand, a claim or an argument ("says they
    should go first").
  - Everyone and everything must plausibly be in this setting.
  - No personal names anywhere."""

USER = """Setting: {place}, {time}. Surroundings: {surroundings}

Main character's circumstance: {circumstance}

Task types, each with the relationship to the main character:
{tasks}

Entity type: {entity_type}"""


TURN_SYSTEM = """You are the director of a short scene that is being acted out
right now. You set it up; now you decide, turn by turn, who acts next.

You choose one character or one entity per turn. Your job is to make sure
the scene works in the turns available:
  - every secondary character's task and every entity's task happens before
    the scene ends - if turns are running out, cue what hasn't happened yet
  - cue each task at the moment it would naturally happen - a joke about a
    late arrival comes right as they arrive, not a few turns later
  - people react to what just happened - after a task or an entity, someone
    usually responds to it, often the main character
  - an entity acts at most once
  - avoid giving the same character two turns in a row
  - the scene does not need to resolve - never cue anyone to smooth over,
    explain or undo what happened

You may add a short private nudge for the character you pick, such as "now
is a good moment for your task" or "react to what just happened". Leave it
empty when the character doesn't need one. Never script their words."""

TURN_USER = """Setting: {setting}

The scene card:
{card}

The scene so far:
{script}

Turns left, including this one: {turns_left}

Who acts next?"""


def describe_card(card: SceneCard) -> str:
    m = card.main
    lines = [f"MAIN CHARACTER  {m.brief}. Goal: {m.goal}. Circumstance: {m.circumstance}"]
    for c in card.secondary:
        lines.append(f"CHARACTER       {c.role} ({c.relationship}). Task: {c.task}")
    for e in card.entities:
        lines.append(f"ENTITY          {e.name}. Task: {e.task}")
    return "\n".join(lines)


def format_script(script: list[dict]) -> str:
    if not script:
        return "(Nothing has happened yet.)"
    return "\n".join(f"{line['who']}: {line['text']}" for line in script)


def _card_problem(card: SceneCard, n_tasks: int, want_entity: bool) -> str:
    """What's wrong with a card, or "" if nothing. The director has dropped a
    task before even though the prompt says one character per task type, so
    the shape is checked here rather than trusted."""
    if len(card.secondary) != n_tasks:
        return f"{len(card.secondary)} secondary characters for {n_tasks} tasks"
    roles = [c.role.strip().lower() for c in card.secondary]
    if len(set(roles)) != len(roles) or MAIN in roles:
        return f"roles not distinct: {roles}"
    if len(card.entities) != (1 if want_entity else 0):
        return f"{len(card.entities)} entities, expected {1 if want_entity else 0}"
    return ""


class Director:
    """One call to set the scene up, then one call per turn to cue it."""

    def __init__(self, model: str = "gpt-5.4-mini", temperature: float = 0.9,
                 attempts: int = 3, cue_model: str | None = None):
        """`cue_model` runs the per-turn cueing - six calls a scene against one
        for the setup - so it can be a cheaper model. Defaults to `model`."""
        self.model = model
        self.cue_model = cue_model or model
        self.attempts = attempts
        self.llm = build_llm(model, temperature=temperature).with_structured_output(SceneCard)
        self.cue_llm = build_llm(self.cue_model, temperature=0.3).with_structured_output(Cue)

    def direct(self, place: str, time: str, surroundings: str, circumstance: str,
               tasks: list[tuple[str, str]], entity_type: str | None = None,
               feedback: list[str] | None = None) -> SceneCard:
        """`tasks` is a list of (task type, relationship) pairs - one secondary
        character each. `feedback` is the card judge's problems with an
        earlier card for the same ingredients. Retried until the card has the
        right shape."""
        user = USER.format(
            place=place, time=time, surroundings=surroundings,
            circumstance=circumstance,
            tasks="\n".join(f"- {t} ({r})" for t, r in tasks),
            entity_type=entity_type or "none")
        if feedback:
            user += ("\n\nAn earlier card for this setup was rejected for these "
                     "problems - do not repeat them:\n"
                     + "\n".join(f"- {p}" for p in feedback))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        problem = ""
        for _ in range(self.attempts):
            card = cast(SceneCard, self.llm.invoke(messages))
            problem = _card_problem(card, len(tasks), entity_type is not None)
            if not problem:
                return card
        raise ValueError(f"director could not produce a valid card: {problem}")

    def next_turn(self, setting: str, card: SceneCard, script: list[dict],
                  turns_left: int) -> Cue:
        return cast(Cue, self.cue_llm.invoke([
            {"role": "system", "content": TURN_SYSTEM},
            {"role": "user", "content": TURN_USER.format(
                setting=setting, card=describe_card(card),
                script=format_script(script), turns_left=turns_left)},
        ]))
