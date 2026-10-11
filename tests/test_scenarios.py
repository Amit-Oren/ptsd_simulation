"""
Tests for the scenario pool.

These run over the generated pool itself, not over a template, so they check
the data that a simulation will actually use. The load-bearing one is
test_no_scenario_states_a_reaction: a scenario that names how the person felt
has already done the cognitive graph's work, and there is nothing left to
measure.

Skipped unless the pool has been generated:
    python scripts/generate_scenarios.py
"""
import random

import pytest

from config.scenario_seeds import SEEDS, DIMENSIONS, dimensions_for
from config.scenarios import (
    POOL_PATH, all_scenarios, get_scenario, sample_scenario,
    minimal_pairs, pair_groups, format_for_prompt,
)

pytestmark = pytest.mark.skipif(
    not POOL_PATH.exists(),
    reason="scenario pool not generated — run scripts/generate_scenarios.py",
)


# ── the pool ──────────────────────────────────────────────────────────────────

def test_pool_is_not_empty():
    assert len(all_scenarios()) > 50


def test_every_scenario_has_the_expected_fields():
    for s in all_scenarios():
        for field in ("id", "seed_id", "text", "dimensions", "pair_id",
                      "varied_dimension", "atomic_event", "generator",
                      "prompt_version", "generated_at"):
            assert field in s, f"{s.get('id')} missing {field}"


def test_ids_are_unique():
    ids = [s["id"] for s in all_scenarios()]
    assert len(ids) == len(set(ids))


def test_every_seed_is_represented():
    assert {s["seed_id"] for s in all_scenarios()} == set(SEEDS)


def test_text_is_substantial_but_short():
    for s in all_scenarios():
        words = len(s["text"].split())
        assert 15 <= words <= 120, f"{s['id']}: {words} words"


def test_dimensions_come_from_the_declared_pools():
    for s in all_scenarios():
        pools = dimensions_for(s["seed_id"])
        for name, value in s["dimensions"].items():
            assert name in pools, f"{s['id']}: undeclared dimension {name!r}"
            assert value in pools[name], f"{s['id']}: {name}={value!r} not in pool"


# ── the camera-facts rule ─────────────────────────────────────────────────────

FEELING_WORDS = [
    "felt", "feeling", "feels", "panic", "anxious", "anxiety", "afraid",
    "scared", "frightened", "upset", "angry", "ashamed", "guilty", "distress",
    "overwhelm", "triggered", "reminded you", "flashback", "intrusive",
    "uneasy", "uncomfortable", "nervous", "worried", "dread", "tense",
    "struggled", "unsettl",
]


def test_no_scenario_states_a_reaction():
    for s in all_scenarios():
        low = s["text"].lower()
        for word in FEELING_WORDS:
            assert word not in low, f"{s['id']} leaks a reaction: {word!r}"


def test_no_scenario_states_the_other_persons_intent():
    """The real reason behind an ambiguous act must stay inferable."""
    for s in all_scenarios():
        low = s["text"].lower()
        for intent in s["other_intent"]:
            phrase = intent.lower().strip(" .").removeprefix("to ")
            if len(phrase) < 8:        # too short to match meaningfully
                continue
            assert phrase not in low, f"{s['id']} states the intent: {intent!r}"


# ── ATOMIC annotations ────────────────────────────────────────────────────────

def test_atomic_seeds_carry_their_baselines():
    for s in all_scenarios():
        if s["atomic_event"] is None:
            continue
        assert s["relations"], f"{s['id']} has an ATOMIC event but no relations"
        assert s["norm_feel"], f"{s['id']} has no oReact baseline"


def test_written_seeds_declare_no_atomic_source():
    written = [s for s in all_scenarios() if s["atomic_event"] is None]
    assert written, "expected some seeds without an ATOMIC equivalent"
    for s in written:
        assert s["relations"] == {}
        assert s["other_intent"] == []


# ── minimal pairs ─────────────────────────────────────────────────────────────

def test_pair_groups_exist():
    assert pair_groups()


def test_pair_members_differ_only_in_the_varied_dimension():
    for pid in pair_groups():
        members = minimal_pairs(pid)
        assert len(members) >= 2
        varied = {m["varied_dimension"] for m in members}
        assert len(varied) == 1, f"{pid}: inconsistent varied_dimension"
        dim = varied.pop()
        others = [{k: v for k, v in m["dimensions"].items() if k != dim}
                  for m in members]
        assert all(o == others[0] for o in others), f"{pid}: other dimensions drifted"


def test_pair_members_take_different_values():
    for pid in pair_groups():
        members = minimal_pairs(pid)
        dim = members[0]["varied_dimension"]
        values = [m["dimensions"][dim] for m in members]
        assert len(values) == len(set(values)), f"{pid}: repeated value"


# ── lookup and sampling ───────────────────────────────────────────────────────

def test_get_scenario_round_trips():
    first = all_scenarios()[0]
    assert get_scenario(first["id"])["text"] == first["text"]


def test_get_scenario_rejects_unknown_id():
    with pytest.raises(KeyError):
        get_scenario("no_such_scenario")


def test_sample_is_reproducible():
    assert sample_scenario(random.Random(4))["id"] == sample_scenario(random.Random(4))["id"]


def test_sample_respects_seed_exclusions():
    keep = "detour"
    s = sample_scenario(random.Random(0), exclude_seeds=set(SEEDS) - {keep})
    assert s["seed_id"] == keep


def test_sample_falls_back_when_every_seed_is_excluded():
    assert sample_scenario(random.Random(0), exclude_seeds=set(SEEDS))["seed_id"] in SEEDS


def test_format_for_prompt_returns_the_text():
    s = sample_scenario(random.Random(0))
    assert format_for_prompt(s) == s["text"]


# ── seed configuration ────────────────────────────────────────────────────────

def test_every_seed_declares_known_dimensions():
    for seed_id, seed in SEEDS.items():
        for name in seed["uses"]:
            assert name in DIMENSIONS, f"{seed_id}: unknown dimension {name!r}"


def test_restrictions_only_narrow_declared_dimensions():
    for seed_id, seed in SEEDS.items():
        for name, values in seed.get("restrict", {}).items():
            assert name in seed["uses"], f"{seed_id}: restricts unused {name!r}"
            assert set(values) <= set(DIMENSIONS[name]), \
                f"{seed_id}.{name}: restriction is not a subset of the pool"
