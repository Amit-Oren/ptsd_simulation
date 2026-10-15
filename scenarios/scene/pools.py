"""
What the director is given to build a scene from. A draft - edit freely.
"""
from scenarios.pools import WHERE, WHEN, ENVIRONMENT

# ── setting ───────────────────────────────────────────────────────────────────
# Reused as-is from the old pools.

PLACES = WHERE
TIMES = WHEN
SURROUNDINGS = ENVIRONMENT


# ── task types ────────────────────────────────────────────────────────────────

TASK_TYPES = [
    # attention
    "draws attention to the main character",
    "makes a joke that involves the main character",
    "comments on something about the main character",
    "watches the main character without saying anything",

    # information
    "gives information",
    "gives a vague answer when asked",
    "gives instructions",
    "gives information that contradicts something said earlier",
    "asks the main character a question",

    # plans and agreements
    "changes an arrangement",
    "cancels something",
    "does not do what was agreed",
    "asks for something unexpected",

    # conversation
    "interrupts",
    "disagrees",
    "criticises something",
    "apologises",
    "says it will be fine",
    "offers help",

    # responsiveness
    "does not respond to a request",
    "takes a long time to reply",
    "leaves partway through",

    # closeness
    "approaches the main character",
    "moves away from the main character",
    "picks up something that belongs to the main character",

    # mistakes
    "makes a mistake",
    "corrects a mistake",

    # physical
    "bumps into the main character",
    "blocks the way",
    "stands very close to the main character",

    # evaluation
    "judges the main character's work in front of others",
    "compares the main character to someone else",

    # pressure
    "asks the main character to decide something on the spot",
    "asks the main character to do something in front of others",

    # voice
    "raises their voice",
    "speaks sharply",

    # separation
    "leaves the main character waiting alone",
    "goes somewhere without the main character",
]


# ── relationships ─────────────────────────────────────────────────────────────


RELATIONSHIPS = [
    "someone they know well",
    "a family member",
    "a romantic partner",
    "someone they work or study with",
    "someone with authority over them",
    "someone providing them a service",
    "an acquaintance",
    "a stranger",
]


# ── circumstances ─────────────────────────────────────────────────────────────


CIRCUMSTANCES = [
    "arrives late",
    "is there for the first time",
    "is in a hurry",
    "is waiting for something",
    "is carrying a lot",
    "is alone",
    "has to stay until what they came for is finished",
    "has to rely on someone else to get it done",
    "is waiting for help",
    "is somewhere they know well",
    "nothing unusual",
]


# ── entity types ──────────────────────────────────────────────────────────────


ENTITY_TYPES = [
    "a phone",
    "an alarm",
    "a door",
    "the lights",
    "an announcement over a speaker",
    "the weather",
    "a vehicle",
    "a machine or appliance",
    "a screen or display",
    "an animal",
    "an object that falls or is dropped",
    "an elevator",
]
