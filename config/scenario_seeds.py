"""
Seeds and dimensions for the scenario pool — the authored input to generation.

A scenario is one SEED (what happened) crossed with one combination of
DIMENSIONS (who was involved, where, when). The prose is written once, offline,
by scripts/generate_scenarios.py; this module holds only the small amount of
material a person has to author.

Dimensions are organised into three facets and deliberately carry no clinical
meaning. `relationship` and `place` are dimensions any work on social
reasoning, interpretation bias, or persona simulation would want — the PTSD
relevance emerges when a particular patient's cognitive graph meets a
particular combination, rather than being baked into the data. The same reason
scenarios carry no tags for which components they might engage.

Each seed declares which dimensions apply to it: `relationship` means nothing
for a power cut, and an alarm at work has no choice of place.

Eight seeds come from ATOMIC-2020 and carry its crowd annotations (see
data/atomic/seed_relations.json). The four sensory seeds have no ATOMIC
equivalent — ATOMIC is almost entirely interpersonal, while a large share of
the trigger pool is sensory — and are marked as such rather than left to look
like an oversight.
"""

# ── dimensions ────────────────────────────────────────────────────────────────

SOCIAL = {
    "relationship": [
        "a manager at work",
        "a colleague you know a little",
        "a colleague you have worked with for years",
        "a close friend",
        "a sibling",
        "a parent",
        "a neighbour",
        "someone you barely know",
    ],
    "others_present": [
        "nobody else was around",
        "one other person was there",
        "a few people were within earshot",
        "it was crowded",
    ],
}

PHYSICAL = {
    "place": [
        "at your home",
        "at their home",
        "at your workplace",
        "in the street",
        "on public transport",
        "in a shop",
        "in a park",
        "in a waiting area",
    ],
    "weather": [
        "heavy rain",
        "cold and clear",
        "warm",
        "overcast",
    ],
    "sound": [
        "quiet",
        "traffic outside",
        "a lot of noise",
    ],
}

TEMPORAL = {
    # Light is folded in rather than sampled separately: drawn independently,
    # the two contradict each other roughly half the time.
    "time_of_day": [
        "early morning, barely light yet",
        "the middle of the day",
        "late afternoon, the light going",
        "the evening, after dark",
        "late at night",
    ],
    "duration": [
        "a couple of minutes",
        "around twenty minutes",
        "most of an hour",
        "a few hours",
    ],
    # Phrased as clauses about the event rather than adjective phrases: as
    # adjectives they attached to whatever noun stood nearest, producing "a
    # list that had been arranged in advance".
    "expectedness": [
        "you had known it was coming",
        "you heard about it the same day",
        "it happened with no warning",
    ],
}

FACETS = {"social": SOCIAL, "physical": PHYSICAL, "temporal": TEMPORAL}
DIMENSIONS = {**SOCIAL, **PHYSICAL, **TEMPORAL}


# ── seeds ─────────────────────────────────────────────────────────────────────
# `gist` is what happened, in one line, from the patient's side. `uses` lists
# the dimensions that make sense for it, and `restrict` narrows a dimension to
# the values that are coherent with the event — a dog running up to you does
# not happen under harsh indoor lighting, and the person who calls you in to
# speak with them is not your neighbour. Without this the generator is handed
# combinations no amount of good prose can reconcile.

SEEDS: dict[str, dict] = {
    "called_in": {
        "gist": "someone with authority over you asks you to come and speak with "
                "them, and does not say what it is about",
        "uses": ["relationship", "others_present", "place", "time_of_day", "duration", "expectedness"],
        "restrict": {
            "relationship": ["a manager at work", "a colleague you have worked with for years"],
            "place": ["at your workplace", "in a waiting area"],
        },
        "atomic_event": "PersonX calls PersonY into the office",
    },
    "forgot_name": {
        "gist": "the person you are with cannot recall your name while introducing "
                "you to someone else, and has to be reminded of it",
        "uses": ["relationship", "others_present", "place", "sound",
                 "time_of_day"],
        "restrict": {
            "place": ["at your workplace", "in a shop", "in a waiting area", "in the street"],
        },
        "atomic_event": "PersonX forgets PersonY's name",
    },
    "offered_hand": {
        "gist": "someone takes on part of your work without being asked, and you "
                "find out afterwards",
        "uses": ["relationship", "others_present", "place", "time_of_day",
                 "expectedness"],
        "restrict": {
            "relationship": ["a manager at work", "a colleague you know a little",
                             "a colleague you have worked with for years"],
            "place": ["at your workplace"],
        },
        "atomic_event": "PersonX gives PersonY a hand",
    },
    "unannounced_visit": {
        "gist": "someone close to you arrives at your door without arranging it "
                "beforehand and stays a while",
        "uses": ["relationship", "weather", "time_of_day", "duration",
                 "expectedness"],
        "restrict": {
            "relationship": ["a close friend", "a sibling", "a parent", "a neighbour"],
            "duration": ["around twenty minutes", "most of an hour", "a few hours"],
        },
        "atomic_event": "PersonX pays PersonY a visit",
    },
    "texted_back": {
        "gist": "someone replies to your message several days later and answers "
                "only part of what you asked",
        "uses": ["relationship", "place", "time_of_day"],
        "atomic_event": "PersonX texts PersonY back",
    },
    "took_hours": {
        "gist": "someone takes over work that was yours, and you learn it from a "
                "list rather than from them",
        "uses": ["relationship", "place", "time_of_day", "expectedness"],
        "restrict": {
            "relationship": ["a manager at work", "a colleague you know a little",
                             "a colleague you have worked with for years"],
            "place": ["at your workplace"],
        },
        "atomic_event": "PersonX takes PersonY's hours",
    },
    "invited_over": {
        "gist": "the person described is having people round, and is the one who "
                "asks you whether you want to join them",
        "uses": ["relationship", "others_present", "weather", "sound", "time_of_day", "expectedness"],
        "restrict": {
            "relationship": ["a neighbour", "a close friend", "a colleague you know a little"],
        },
        "atomic_event": "PersonX invites all PersonY's friends over",
    },
    "opportunity": {
        "gist": "you are offered something you have not done before, and asked to "
                "decide by the end of the week",
        "uses": ["relationship", "others_present", "place", "time_of_day",
                 "expectedness"],
        "restrict": {
            "relationship": ["a manager at work", "a colleague you have worked with for years"],
            "place": ["at your workplace", "in a waiting area"],
        },
        "atomic_event": "PersonX gives PersonY opportunity",
    },

    # Written — no ATOMIC equivalent. These carry the sensory and situational
    # material that ATOMIC's interpersonal events do not reach.
    "fire_alarm": {
        "gist": "an alarm goes off where you are and everyone has to go outside "
                "and wait, with no one saying why",
        "uses": ["weather", "sound", "time_of_day", "duration"],
        "atomic_event": None,
    },
    "power_cut": {
        "gist": "the electricity goes off where you are and stays off for a while",
        "uses": ["others_present", "place", "time_of_day", "duration"],
        "restrict": {
            "place": ["at your home", "at your workplace", "in a shop", "in a waiting area"],
        },
        "atomic_event": None,
    },
    "dog_off_lead": {
        "gist": "a large dog with no lead comes straight up to you in public "
                "before its owner calls it back",
        "uses": ["others_present", "place", "weather", "time_of_day"],
        "restrict": {
            "place": ["in the street", "in a park"],
        },
        "atomic_event": None,
    },
    "detour": {
        "gist": "the route you normally take is closed for the week and you have "
                "to go a longer way round",
        "uses": ["weather", "time_of_day", "duration"],
        "restrict": {"duration": ["around twenty minutes", "most of an hour"]},
        "atomic_event": None,
    },
}


def dimensions_for(seed_id: str) -> dict[str, list[str]]:
    """The dimension pools that apply to one seed, narrowed by its restrictions."""
    seed = SEEDS[seed_id]
    restrict = seed.get("restrict", {})
    return {name: restrict.get(name, DIMENSIONS[name]) for name in seed["uses"]}


def facet_of(dimension: str) -> str:
    for facet, dims in FACETS.items():
        if dimension in dims:
            return facet
    raise KeyError(f"unknown dimension {dimension!r}")
