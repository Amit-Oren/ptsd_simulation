"""
Tests for the scenario pool.

These run over the generated pool itself, not over a template, so they check
the data that a simulation will actually use. The load-bearing one is
test_no_scenario_states_a_reaction: a scenario that names how the person felt
has already done the cognitive graph's work, and there is nothing left to
measure.

Skipped unless the pool has been generated:
    python -m scenarios.generate
"""
import random

import pytest

from config.scenarios import (
    POOL_PATH, all_scenarios, get_scenario, sample_scenario, format_for_prompt,
)
from scenarios.pools import POOLS, EVENT_TEMPLATES

pytestmark = pytest.mark.skipif(
    not POOL_PATH.exists(),
    reason="scenario pool not generated — run python -m scenarios.generate",
)

_TEMPLATE_FIELDS = ("situation", "event", "response", "outcome")


# ── the pool ──────────────────────────────────────────────────────────────────

def test_pool_is_not_empty():
    assert len(all_scenarios()) > 50


def test_every_scenario_has_the_expected_fields():
    for s in all_scenarios():
        for field in ("id", "event_key", "place", "text", "material", "verdict",
                      "narrator", "judge", "pool_version", "generated_at"):
            assert field in s, f"{s.get('id')} missing {field}"


def test_ids_are_unique():
    ids = [s["id"] for s in all_scenarios()]
    assert len(ids) == len(set(ids))


def test_every_event_is_represented():
    assert {s["event_key"] for s in all_scenarios()} == set(EVENT_TEMPLATES)


def test_every_scenario_carries_its_whole_event_template():
    """event/situation/detail/outcome must match the template named by
    event_key exactly - not a mix-and-match of different templates' fields."""
    for s in all_scenarios():
        template = EVENT_TEMPLATES[s["event_key"]]
        for field in _TEMPLATE_FIELDS:
            assert s["material"].get(field) == template[field], \
                f"{s['id']}: material.{field} does not match its event template"


def test_text_is_substantial_but_short():
    for s in all_scenarios():
        words = len(s["text"].split())
        assert 15 <= words <= 120, f"{s['id']}: {words} words"


def test_material_fields_come_from_the_declared_pools():
    for s in all_scenarios():
        for name, value in s["material"].items():
            if name in _TEMPLATE_FIELDS:
                continue  # part of the event template, not an independent pool
            assert name in POOLS, f"{s['id']}: undeclared field {name!r}"
            assert value in POOLS[name], f"{s['id']}: {name}={value!r} not in pool"


def test_every_kept_scenario_passed_every_check():
    for s in all_scenarios():
        v = s["verdict"]
        for check in ("factual", "coherent", "plausible", "specific", "natural"):
            assert v[check]["passed"], f"{s['id']} kept despite failing {check}"


# ── the camera-facts rule ─────────────────────────────────────────────────────

FEELING_WORDS = [
    "panic", "anxious", "anxiety", "afraid",
    "scared", "frightened", "upset", "angry", "ashamed", "guilty", "distress",
    "overwhelm", "triggered", "reminded you", "flashback", "intrusive",
    "uneasy", "uncomfortable", "nervous", "worried", "dread", "tense",
    "struggled", "unsettl",
]

# "felt"/"feeling"/"feels" alone is too broad - "the room felt cold" is a
# camera-observable fact about the room, not a reaction. Only the person
# feeling something is the violation.
YOU_FEEL = ["you felt", "you feel", "you are feeling", "you were feeling"]


def test_no_scenario_states_a_reaction():
    for s in all_scenarios():
        low = s["text"].lower()
        for word in FEELING_WORDS:
            assert word not in low, f"{s['id']} leaks a reaction: {word!r}"
        for phrase in YOU_FEEL:
            assert phrase not in low, f"{s['id']} leaks a reaction: {phrase!r}"


# ── lookup and sampling ───────────────────────────────────────────────────────

def test_get_scenario_round_trips():
    first = all_scenarios()[0]
    assert get_scenario(first["id"])["text"] == first["text"]


def test_get_scenario_rejects_unknown_id():
    with pytest.raises(KeyError):
        get_scenario("no_such_scenario")


def test_sample_is_reproducible():
    assert sample_scenario(random.Random(4))["id"] == sample_scenario(random.Random(4))["id"]


def test_sample_respects_event_exclusions():
    events = {s["event_key"] for s in all_scenarios()}
    if len(events) < 2:
        pytest.skip("need at least two events in the pool to test exclusion")
    keep = sorted(events)[0]
    s = sample_scenario(random.Random(0), exclude_events=events - {keep})
    assert s["event_key"] == keep


def test_sample_falls_back_when_every_event_is_excluded():
    events = {s["event_key"] for s in all_scenarios()}
    assert sample_scenario(random.Random(0), exclude_events=events)["event_key"] in events


def test_format_for_prompt_returns_the_text():
    s = sample_scenario(random.Random(0))
    assert format_for_prompt(s) == s["text"]
