"""
Unit tests for the spreading-activation mechanism.

Run: python -m pytest tests/test_activation.py -q
"""
import pytest
from simulation.activation import (
    ActivationParams, from_self_report, activate, next_unused,
    cluster_levels, hottest_item, spread, propagate, decay,
)


def _params(**kw) -> ActivationParams:
    return ActivationParams(**kw)


# ── from_self_report ──────────────────────────────────────────────────────────

def test_from_self_report_zeroes_every_item():
    sr = {"Memory": [{"key": "a", "value": "A"}, {"key": "b", "value": "B"}]}
    assert from_self_report(sr) == {"Memory": {"a": 0.0, "b": 0.0}}


def test_from_self_report_accepts_plain_strings():
    assert from_self_report({"Threat": ["x"]}) == {"Threat": {"x": 0.0}}


def test_from_self_report_preserves_sampled_order():
    sr = {"Memory": [{"key": "z"}, {"key": "a"}, {"key": "m"}]}
    assert list(from_self_report(sr)["Memory"]) == ["z", "a", "m"]


# ── activate ──────────────────────────────────────────────────────────────────

def test_activate_raises_level():
    A = {"Memory": {"a": 0.1}}
    activate(A, "Memory", "a", 0.5, 1.0)
    assert A["Memory"]["a"] == pytest.approx(0.6)


def test_activate_clamps_to_max():
    A = {"Memory": {"a": 0.8}}
    activate(A, "Memory", "a", 0.9, 1.0)
    assert A["Memory"]["a"] == 1.0


def test_activate_creates_missing_entries():
    A = {}
    activate(A, "Threat", "new", 0.4, 1.0)
    assert A == {"Threat": {"new": 0.4}}


# ── next_unused ───────────────────────────────────────────────────────────────

def test_next_unused_returns_first_cold_item_in_order():
    A = {"Memory": {"z": 0.0, "a": 0.0}}
    assert next_unused(A, "Memory", 0.5) == "z"   # sampled order, not alphabetical


def test_next_unused_skips_live_items():
    A = {"Memory": {"z": 0.9, "a": 0.1}}
    assert next_unused(A, "Memory", 0.5) == "a"


def test_next_unused_none_when_all_live():
    A = {"Memory": {"z": 0.9, "a": 0.7}}
    assert next_unused(A, "Memory", 0.5) is None


def test_next_unused_none_for_unknown_node():
    assert next_unused({}, "Memory", 0.5) is None


# ── cluster_levels ────────────────────────────────────────────────────────────

def test_cluster_levels_noisy_or_exceeds_max_but_stays_bounded():
    A = {"Triggers": {"a": 0.6, "b": 0.6}}
    level = cluster_levels(A)["Triggers"]
    assert level == pytest.approx(1 - 0.4 * 0.4)   # 0.84
    assert 0.6 < level < 1.0


def test_cluster_levels_single_item_is_that_item():
    assert cluster_levels({"Memory": {"a": 0.6}})["Memory"] == pytest.approx(0.6)


def test_cluster_levels_never_exceeds_one():
    A = {"Triggers": {k: 0.9 for k in "abcdef"}}
    assert cluster_levels(A)["Triggers"] <= 1.0


def test_cluster_levels_monotone_in_each_item():
    low  = cluster_levels({"T": {"a": 0.5, "b": 0.1}})["T"]
    high = cluster_levels({"T": {"a": 0.5, "b": 0.3}})["T"]
    assert high > low


def test_hottest_item_picks_most_active():
    assert hottest_item({"T": {"a": 0.2, "b": 0.7, "c": 0.4}}, "T") == "b"


def test_hottest_item_none_for_empty_node():
    assert hottest_item({"T": {}}, "T") is None


def test_cluster_levels_empty_node_is_zero():
    assert cluster_levels({"Memory": {}})["Memory"] == 0.0


# ── spread ─────────────────────────────────────────────────────────────────────

def test_spread_moves_item_level_times_weight_times_rate():
    A = {"Triggers": {"t": 0.0}, "Memory": {"m": 0.0}}
    p = _params(spread_rate=0.4)
    spread(A, {"Triggers": {"Memory": 0.5}}, p, wave={("Triggers", "t"): 1.0})
    assert A["Memory"]["m"] == pytest.approx(1.0 * 0.5 * 0.4)


def test_spread_ignores_zero_weight_edges():
    A = {"Triggers": {"t": 1.0}, "Memory": {"m": 0.0}}
    spread(A, {"Triggers": {"Memory": 0.0}}, _params(), wave={("Triggers", "t"): 1.0})
    assert A["Memory"]["m"] == 0.0


def test_spread_quiet_item_does_not_broadcast():
    A = {"Triggers": {"t": 0.01}, "Memory": {"m": 0.0}}
    spread(A, {"Triggers": {"Memory": 1.0}}, _params(spread_floor=0.05),
           wave={("Triggers", "t"): 0.01})
    assert A["Memory"]["m"] == 0.0


def test_spread_only_items_in_wave_broadcast():
    A = {"Triggers": {"a": 1.0, "b": 1.0}, "Memory": {"m": 0.0}}
    spread(A, {"Triggers": {"Memory": 1.0}}, _params(spread_rate=0.5),
           wave={("Triggers", "a"): 1.0})
    assert A["Memory"]["m"] == pytest.approx(0.5)   # only "a" sent, not "a" and "b"


def test_spread_carries_wave_energy_not_item_level():
    """A hot item passes on only what just reached it, not its whole level."""
    A = {"Triggers": {"t": 0.9}, "Memory": {"m": 0.0}}
    spread(A, {"Triggers": {"Memory": 1.0}}, _params(spread_rate=0.5),
           wave={("Triggers", "t"): 0.2})
    assert A["Memory"]["m"] == pytest.approx(0.2 * 1.0 * 0.5)


def test_spread_is_simultaneous_not_chained():
    """One hop must not let an item re-broadcast what it just received."""
    A = {"Triggers": {"t": 1.0}, "Memory": {"m": 0.0}, "Threat": {"x": 0.0}}
    edges = {"Triggers": {"Memory": 1.0}, "Memory": {"Threat": 1.0}}
    spread(A, edges, _params(spread_rate=0.5), wave={("Triggers", "t"): 1.0})
    assert A["Memory"]["m"] == pytest.approx(0.5)
    assert A["Threat"]["x"] == 0.0


def test_spread_skips_node_with_no_cold_items():
    A = {"Triggers": {"t": 1.0}, "Memory": {"m": 0.9}}
    spread(A, {"Triggers": {"Memory": 1.0}}, _params(surface_threshold=0.5),
           wave={("Triggers", "t"): 1.0})
    assert A["Memory"]["m"] == 0.9


def test_spread_records_item_level_provenance():
    A = {"Triggers": {"car_sounds": 0.6}, "Memory": {"m": 0.0}}
    got = spread(A, {"Triggers": {"Memory": 0.5}}, _params(spread_rate=0.4),
                 wave={("Triggers", "car_sounds"): 0.6})
    entry = got[("Memory", "m")]
    assert entry["from_node"] == "Triggers"
    assert entry["from_item"] == "car_sounds"
    assert entry["amount"] == pytest.approx(0.6 * 0.5 * 0.4)


# ── propagate ────────────────────────────────────────────────────────────

def _chain():
    A = {"Triggers": {"car_sounds": 0.6}, "Memory": {"the_impact": 0.0},
         "Threat": {"something_bad": 0.0}}
    edges = {"Triggers": {"Memory": 0.8}, "Memory": {"Threat": 0.7}}
    return A, edges


def test_propagate_cascade_weakens_each_hop():
    A, edges = _chain()
    propagate(A, edges, _params(spread_rate=0.4), "Triggers", "car_sounds", hops=3)
    assert A["Memory"]["the_impact"] > A["Threat"]["something_bad"] > 0.0


def test_propagate_origin_fires_only_once():
    """Regression: the origin must not re-broadcast on every hop."""
    A, edges = _chain()
    propagate(A, edges, _params(spread_rate=0.4), "Triggers", "car_sounds", hops=3)
    assert A["Memory"]["the_impact"] == pytest.approx(0.6 * 0.8 * 0.4)


def test_propagate_chains_item_to_item():
    A, edges = _chain()
    history = propagate(A, edges, _params(spread_rate=0.4),
                        "Triggers", "car_sounds", hops=3)
    hop1 = history[0][("Memory", "the_impact")]
    hop2 = history[1][("Threat", "something_bad")]
    assert hop1["from_item"] == "car_sounds"
    assert hop2["from_item"] == "the_impact"     # the item that received now sends


def test_propagate_stops_when_wave_dies():
    A = {"Triggers": {"t": 0.6}, "Memory": {"m": 0.0}}
    edges = {"Triggers": {"Memory": 0.8}}        # Memory has no outgoing edges
    history = propagate(A, edges, _params(), "Triggers", "t", hops=5)
    assert len(history) == 1


def test_propagate_respects_hop_limit():
    A = {n: {n.lower(): 0.0} for n in ("B", "C", "D")}
    A["A"] = {"a": 1.0}
    edges = {"A": {"B": 1.0}, "B": {"C": 1.0}, "C": {"D": 1.0}}
    propagate(A, edges, _params(spread_rate=0.9), "A", "a", hops=2)
    assert A["B"]["b"] > 0.0
    assert A["C"]["c"] > 0.0
    assert A["D"]["d"] == 0.0


def test_propagate_unknown_origin_does_nothing():
    A = {"Triggers": {"t": 1.0}, "Memory": {"m": 0.0}}
    propagate(A, {"Triggers": {"Memory": 1.0}}, _params(), "Threat", "nope", hops=2)
    assert A["Memory"]["m"] == 0.0


def test_propagate_two_way_edge_attenuates():
    """Triggers<->Memory must lose energy each pass, not pump."""
    A = {"Triggers": {"a": 0.6, "b": 0.0}, "Memory": {"m": 0.0, "n": 0.0}}
    edges = {"Triggers": {"Memory": 0.8}, "Memory": {"Triggers": 0.5}}
    history = propagate(A, edges, _params(spread_rate=0.4), "Triggers", "a", hops=6)
    arrivals = [sum(e["amount"] for e in h.values()) for h in history]
    assert arrivals == sorted(arrivals, reverse=True)
    assert A["Memory"]["m"] == pytest.approx(0.6 * 0.8 * 0.4)


# ── decay ─────────────────────────────────────────────────────────────────────

def test_decay_fades_every_item():
    A = {"Memory": {"a": 1.0, "b": 0.5}}
    decay(A, _params(decay_rate=0.7))
    assert A["Memory"]["a"] == pytest.approx(0.7)
    assert A["Memory"]["b"] == pytest.approx(0.35)


def test_decay_zeroes_items_below_quiet_level():
    A = {"Memory": {"a": 0.02}}
    decay(A, _params(decay_rate=0.7, quiet_level=0.02))
    assert A["Memory"]["a"] == 0.0


def test_decay_leaves_cold_items_cold():
    A = {"Memory": {"a": 0.0}}
    decay(A, _params())
    assert A["Memory"]["a"] == 0.0


def test_decay_makes_faded_item_surfaceable_again():
    A = {"Memory": {"a": 0.6}}
    p = _params(decay_rate=0.7, surface_threshold=0.5)
    assert next_unused(A, "Memory", p.surface_threshold) is None
    decay(A, p)                                    # 0.6 -> 0.42
    assert next_unused(A, "Memory", p.surface_threshold) == "a"
