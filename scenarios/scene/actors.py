"""
The actors - each one plays its part of the scene card, one turn at a time.
"""
from simulation.factory import build_llm
from scenarios.scene.director import (
    MAIN, Entity, MainCharacter, SecondaryCharacter, format_script)

TURN_RULES = """On your turn, write what you do and anything you say out loud: one or
two sentences, present tense, third person, starting with "{label}". Put
speech in quotes. Only what someone else in the room could see or hear -
never thoughts or feelings.

Don't describe the surroundings - the room, the weather, background noise -
unless something in them changes. Everyone already knows them.

Nobody in this scene has a name. When you describe what someone does, refer
to them by who they are ("the lecturer", "the main character"). When you
speak, talk the way a real person would - "you", "I", "she", "they", "excuse
me" - never a name, and never a label like "the main character"."""

MAIN_SYSTEM = """You are the main character in a short scene acted out turn by turn:
{brief}.

Setting: {setting}
You are here to {goal}. Your situation: {circumstance}

Behave plainly and ordinarily - do what someone here for that reason would do,
and respond to what other people do. Do not start anything new yourself.
Never apologise, explain or justify yourself, and never say why you are doing
something - just act. Never say your goal or your situation out loud ("I'm
waiting for the visit", "I'm staying until it's done").
Speak only when someone speaks to you, or when the director tells you to.
Keep what you say short and flat. Don't ask questions unless the director
tells you to, never argue back, never give orders or make claims, and never
comment on what others do.
If someone asks you a direct question, answer it. Never repeat something you
have already said. If there is nothing new to respond to, do something small
and ordinary instead of speaking.
Keep a plain face - no grins, snorts, sighs, eye-rolls or other expressions.

{rules}"""

SECONDARY_SYSTEM = """You are {role} in a short scene acted out turn by turn.
To the main character, you are {relationship}.

Setting: {setting}

If you are only reached by phone or message, you are somewhere else - you
never appear in the room, and everything you say comes through the main
character's phone.

At some point in this scene, do this, once: {task}
Your task comes first. If it says you say nothing, you say nothing for the
whole scene.
Do it when the director tells you to, or when the moment is right. Once you
have done it, don't take it back, apologise for it or smooth it over - nothing
in this scene needs to be resolved. Never do it a second time, and never repeat
something you have already said. On other turns, behave naturally for who you
are and respond to what is happening.

{rules}"""

ENTITY_SYSTEM = """You describe something happening in a scene.

Setting: {setting}
In this scene, {name} does this: {task}

Write one sentence describing it happening right now, fitting what has
happened so far. Present tense. Only what could be seen or heard. No names."""


class CharacterActor:
    def __init__(self, system: str, name: str, model: str, temperature: float):
        self.name = name
        self.system = system
        self.llm = build_llm(model, temperature=temperature)

    @classmethod
    def main(cls, c: MainCharacter, setting: str, model: str, temperature: float):
        return cls(MAIN_SYSTEM.format(
            brief=c.brief, setting=setting, goal=c.goal, circumstance=c.circumstance,
            rules=TURN_RULES.format(label=MAIN.capitalize())), MAIN, model, temperature)

    @classmethod
    def secondary(cls, c: SecondaryCharacter, setting: str, model: str, temperature: float):
        return cls(SECONDARY_SYSTEM.format(
            role=c.role, relationship=c.relationship, setting=setting, task=c.task,
            rules=TURN_RULES.format(label=c.role[:1].upper() + c.role[1:])),
            c.role, model, temperature)

    def act(self, script: list[dict], nudge: str = "") -> str:
        prompt = f"The scene so far:\n{format_script(script)}\n\n"
        if nudge:
            prompt += f"Director: {nudge}\n\n"
        prompt += "Your turn."
        reply = self.llm.invoke([
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]).content
        return " ".join(str(reply).split())


class EntityActor:
    def __init__(self, entity: Entity, setting: str, model: str, temperature: float):
        self.name = entity.name
        self.system = ENTITY_SYSTEM.format(setting=setting, name=entity.name, task=entity.task)
        self.llm = build_llm(model, temperature=temperature)

    def act(self, script: list[dict], nudge: str = "") -> str:
        reply = self.llm.invoke([
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"The scene so far:\n{format_script(script)}"},
        ]).content
        return " ".join(str(reply).split())
