"""
The authored pools a scenario is built from.

"""

# ── where ──────────────────────────────────────────────────────────────────
WHERE = [
    "home",
    "work",
    "a friend's house",
    "school",
    "parking lot",
    "restaurant",
    "bar",
    "club",
    "the street",
    "park",
    "shopping centre",
    "bus station",
    "train station",
    "airport",
    "hospital",
    "the beach",
    "subway",
    "car",
    "taxi",
    "sidewalk",
    "supermarket",
    "grocery store",
    "cafe",
    "playground",
    "theatre",
    "cinema",
    "museum",
    "hiking trail",
    "clinic",
    "gym",
    "pharmacy",
    "bank",
    "library",
    "church",
    "police station",
    "court",
    "construction site",
    "concert venue",
    "bedroom",
    "living room",
    "hallway",
    "staircase",
    "elevator",
    "rooftop",
    "neighborhood",
    "classroom",
    "laboratory",
    "university campus",
]


# ── when ──────────────────────────────────────────────────────────────────────

WHEN = [
   "morning",
   "early morning",
    "late morning",
    "afternoon",
    "evening",
    "night",
    "late night",
    "weekday",
    "weekend",
    "during the day",
    "day off",
    "holiday",
    "vacation",
]


# ── place categories ──────────────────────────────────────────────────────────
# Shared by who_pool() and activity_pool() below: a place belongs to at most
# one category, and that's what unlocks the role- or activity-specific values
# that only make sense there. A place with no category draws generic-only.

PLACE_CATEGORY = {
    "work": "workplace", "laboratory": "workplace", "construction site": "workplace",
    "bank": "retail", "pharmacy": "retail", "shopping centre": "retail",
    "supermarket": "retail", "grocery store": "retail",
    "school": "school", "classroom": "school", "university campus": "school",
    "library": "school", "gym": "school",
    "hospital": "medical", "clinic": "medical",
    "restaurant": "service", "bar": "service", "club": "service",
    "cafe": "service", "theatre": "service", "cinema": "service",
    "museum": "service", "concert venue": "service",
    "bus station": "transit", "train station": "transit",
    "airport": "transit", "subway": "transit",
    "police station": "civic", "court": "civic",
}


# ── who ───────────────────────────────────────────────────────────────────────
# Split in two so a role never lands somewhere it can't plausibly be - a
# bartender in a hospital, a waiter standing around a ward. WHO_GENERIC fits
# any place; WHO_BY_CATEGORY only gets drawn where PLACE_CATEGORY says the
# place belongs to that category. who_pool() is what generate.py actually
# samples from - WHO itself stays as the flat union, for anything that wants
# the full set (tests, pair sweeps across the whole pool).
#
# "anonymous online user" used to sit in the generic bucket and then turned up
# as someone physically present in a park at night - it's not a person who can
# be "with" you somewhere, only a remote one. Dropped rather than recategorized:
# nothing here currently represents a phone/text-only interaction.

WHO_GENERIC = [
    "alone",
    "partner",
    "friend",
    "family member",
    "child",
    "parent",
    "sibling",
    "stranger",
    "close friend",
    "acquaintance",
    "neighbor",
    "roommate",
    "passerby",
    "group of people",
    "crowd",
    "group of friends",
    "family",
]

WHO_BY_CATEGORY = {
    "workplace": ["colleague", "supervisor", "manager", "coworker", "employee",
                 "client", "customer", "work team"],
    "retail":    ["employee", "customer", "manager"],
    "school":    ["classmate", "teammate", "teacher", "student"],
    "medical":   ["doctor", "nurse", "therapist"],
    "service":   ["bartender", "waiter", "worker in public space"],
    "transit":   ["bus driver", "train conductor", "airport staff"],
    "civic":     ["police officer", "security guard"],
}

WHO = WHO_GENERIC + sorted({role for roles in WHO_BY_CATEGORY.values() for role in roles})


def who_pool(place: str) -> list[str]:
    """Who can plausibly be present at this place. Generic roles always
    qualify; role-specific ones only where PLACE_CATEGORY says they fit -
    places with no category draw from WHO_GENERIC alone."""
    category = PLACE_CATEGORY.get(place, "")
    return WHO_GENERIC + WHO_BY_CATEGORY.get(category, [])


# ── what you did ──────────────────────────────────────────────────────────────
# Same split as who: ACTIVITY_GENERIC fits anywhere, ACTIVITY_BY_CATEGORY only
# where the place's category matches - "shopping" only where there's something
# to shop for.

ACTIVITY_GENERIC = [
    "walking",
    "running",
    "texting",
    "driving",
    "eating",
    "drinking",
    "exercising",
    "relaxing",
    "socializing",
    "waiting",
    "sitting",
    "standing",
    "listening to music",
    "reading",
    "cleaning",
    "listening",
    "talking",
    "observing",
    "making a phone call",
    "requesting help",
    "asking for directions",
    "providing assistance",
    "asking questions",
    "answering questions",
    "searching for something",
    "pausing activity",
    "taking a break",
    "entering a place",
]

ACTIVITY_BY_CATEGORY = {
    "retail":    ["shopping"],
    "workplace": ["working", "participating in a meeting"],
    "school":    ["studying"],
    "service":   ["watching a movie", "attending an event", "cooking"],
}

ACTION = ACTIVITY_GENERIC + sorted({a for acts in ACTIVITY_BY_CATEGORY.values() for a in acts})


def activity_pool(place: str) -> list[str]:
    """What someone can plausibly be doing at this place - same mechanism as
    who_pool(), same reason: an unconstrained draw put 'shopping' in a bar."""
    category = PLACE_CATEGORY.get(place, "")
    return ACTIVITY_GENERIC + ACTIVITY_BY_CATEGORY.get(category, [])


# ── what someone else did ─────────────────────────────────────────────────────
# Observable behaviour by another person.


OTHER_ACTION = [
    "spoke normally",
    "spoke loudly",
    "asked a question",
    "gave instructions",
    "offered help",
    "gave information",
    "withheld information",
    "interrupted",
    "did not respond to a request",
    "took a long time to reply",
    "approached",
    "moved away",
    "left",
    "waited",
    "changed the arrangement",
    "made a mistake",
    "disagreed",
    "criticised something",
    "assessed how it was going",
    "asked for something unexpected",
    "did not do what had been agreed",
    "cancelled",
    "apologised",
    "said it would be fine",
    "watched without joining in",
    "called without warning",
    "did not reply",
]


# ── events ─────────────────────────────────────────────────────────────────────
# The anchor every scenario is built from - required, unlike everything else in
# POOLS. A flat gist ("an unexpected delay") gives the narrator a label with no
# shape; it has to invent a beginning, a middle and an end to make a scene out
# of it, and that invention is where incoherence crept in. Each template fixes
# that shape itself with four fields, in order: situation (what was already
# going on), event (the specific thing that happened - not a label like
# "a mistake" but the actual action), response (what the protagonist or
# someone else concretely said or did about it), outcome (how it was left -
# an open question counts as an outcome). Four camera-only facts, already one
# small causal sequence; place/who/activity/when/other_action/environment are
# context layered on afterward, only where they fit.
#
# Two rules hold while writing these, same as the narrator's own:
#   - no vague placeholders ("changed the arrangement", "did something
#     together") - name the actual action.
#   - no interpretation ("you were left out", "you had been expecting
#     things to go a certain way") - state what happened and let a reader
#     draw their own conclusion, the same restriction the narrator is held to.
#
# `compatible`, where present, overrides which values an optional context
# dimension may draw for this event - the event decides what context fits it,
# not the other way round (a generic place-category lookup can't know that an
# unexpected delay wants a place where you check in and wait, specifically).
# Per dimension:
#   dimension absent from the dict    falls back to the normal behaviour
#                                      (who/activity still place-filtered by
#                                      category if a place was drawn; when/
#                                      other_action/environment unrestricted)
#   dimension present as []           never drawn for this event - the
#                                      situation line doesn't need it, or
#                                      naming one independently would
#                                      contradict what the event already
#                                      implies
#   dimension present as [a, b, c]    only these values may be drawn
#
# This is a starting batch of 8, covering a spread (ordinary/benign through
# ambiguous) rather than a complete pool - see the note in generate.py's
# docstring about continuing it. Only unexpected_delay and
# mistake_and_correction carry a `compatible` entry so far, where a generic
# place/who draw had already produced implausible or non-diverse results;
# the rest still use the place-category fallback.

EVENT_TEMPLATES = {
    "unexpected_delay": {
        "situation": "You checked in for a scheduled appointment.",
        "event":     "You were told it was running behind, with no estimate of the wait.",
        "response":  "You asked for an update.",
        "outcome":   "They still could not say when it would start.",
        "compatible": {
            "place": ["clinic", "hospital", "bank", "pharmacy"],
            "who":   ["alone", "friend", "family member", "colleague"],
        },
    },
    "unanswered_message": {
        "situation": "You sent someone a text message and were waiting to hear back.",
        "event":     "An hour passed without a reply.",
        "response":  "You checked your phone again and reread the message you had sent.",
        "outcome":   "Still no reply came.",
    },
    "interruption": {
        "situation": "You were in the middle of telling someone something.",
        "event":     "A third person cut in before you finished.",
        "response":  "You stopped talking and let them speak instead.",
        "outcome":   "You did not go back to what you had been saying.",
    },
    "mistake_and_correction": {
        "situation": "You checked in for something you had booked in advance.",
        "event":     "You were given the wrong time slot.",
        "response":  "You showed them your confirmation.",
        "outcome":   "They checked the system and corrected it.",
        "compatible": {
            "place": ["gym", "clinic", "restaurant", "theatre"],
        },
    },
    "being_left_out": {
        "situation": "A few people nearby were deciding where to go for lunch.",
        "event":     "They agreed on a place and started gathering their things.",
        "response":  "You stayed where you were without saying anything.",
        "outcome":   "They finished and walked off without saying anything to you.",
    },
    "unclear_instructions": {
        "situation": "Someone was explaining what you needed to do.",
        "event":     "Partway through, they gave you a different set of steps than before.",
        "response":  "You asked them to go over it again from the start.",
        "outcome":   "The two versions still did not match.",
    },
    "easy_resolution": {
        "situation": "You had made plans to meet someone at a set time.",
        "event":     "They sent a message saying the time had to move earlier.",
        "response":  "You agreed to the new time.",
        "outcome":   "You both arrived within a few minutes of each other as planned.",
    },
    "unexplained_cancellation": {
        "situation": "You had arranged to meet someone at a set time.",
        "event":     "They sent a message an hour before saying it was off.",
        "response":  "You asked why.",
        "outcome":   "They did not say.",
    },
}

EVENT = [t["event"] for t in EVENT_TEMPLATES.values()]


# ── what the place was like ───────────────────────────────────────────────────

ENVIRONMENT = [
    "quiet",
    "people talking in the background",
    "loud",
    "traffic outside",
    "machinery running",
    "a sudden noise",
    "an announcement over a speaker",
    "bright",
    "dim",
    "daylight",
    "strip lighting",
    "crowded",
    "a few people about",
    "empty",
    "open",
    "enclosed",
    "one way in or out",
    "cold",
    "hot",
    "raining",
    "somewhere you had never been",
    "somewhere you knew well",
]



POOLS = {
    "place":        WHERE,
    "when":         WHEN,
    "who":          WHO,
    "activity":     ACTION,
    "other_action": OTHER_ACTION,
    "event":        EVENT,
    "environment":  ENVIRONMENT,
}


def pool_sizes() -> dict[str, int]:
    return {name: len(values) for name, values in POOLS.items()}


# The event template's four fields are always given - they're what make the
# scenario a scene rather than a label. Everything else is optional context,
# layered on only where it fits; a scenario carries a couple of these, not
# all five.
OPTIONAL_FIELDS = ("place", "when", "who", "activity", "other_action", "environment")

FIELD_LABELS = {
    "situation":    "Starting point",
    "event":        "What happened",
    "response":     "What you did",
    "outcome":      "How it was left",
    "place":        "Where",
    "when":         "When",
    "who":          "Who with",
    "activity":     "You were",
    "other_action": "They",
    "environment":  "The place was",
}


def describe_material(material: dict) -> str:
    """The material as a labelled list, one line per field actually given.

    Used by both the narrator and the judge, so what the judge checks against
    is exactly what the narrator was given - never more, never less.
    """
    return "\n".join(f"{FIELD_LABELS[k]}: {material[k]}"
                    for k in FIELD_LABELS if k in material)
